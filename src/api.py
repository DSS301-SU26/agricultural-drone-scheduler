"""
FastAPI surface for the agricultural UAV scheduling dashboard.

Run from the project root:
    .venv/bin/uvicorn src.api:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .decision_model.decision_engine import (
    DecisionThresholds,
    SCORE_WEIGHTS,
    UNSAFE_WEATHER_CODES,
    add_decision_columns,
    build_recommendation_text,
    calculate_dynamic_flow_rate,
    derive_decision_action,
    calculate_crop_safety_score,
    calculate_spray_quality_score,
    get_awd_recommendation,
    calculate_flyability_score,
)
from .run_pipeline import run_weather_pipeline
from . import database as db


ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_DIRS = [ROOT / "src" / "data" / "clean", ROOT / "data" / "clean"]
REPORT_DIR = ROOT / "reports"
CONFIG_DIR = ROOT / "config"
DECISION_CONFIG_PATH = CONFIG_DIR / "decision_config.json"
IMAGE_KAGGLE_DIR = ROOT / "src" / "data" / "image_kaggle"
OVERRIDE_IMG_DIR = ROOT / "src" / "data" / "weather_overrides"
GENERATED_IMAGE_DIR = ROOT / "src" / "data" / "images"
IMAGE_FEATURES_PATH = ROOT / "src" / "data" / "image_features.csv"
FINAL_TRAINING_DATA_PATH = ROOT / "src" / "data" / "final_training_data.csv"
MODEL_PATH = ROOT / "models" / "agriflight_model.joblib"
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
PIPELINE_LOCK = threading.Lock()
AI_TRAINING_LOCK = threading.Lock()

MODEL_PAYLOAD = None

def load_model():
    global MODEL_PAYLOAD
    if MODEL_PATH.exists():
        try:
            import joblib
            MODEL_PAYLOAD = joblib.load(MODEL_PATH)
            print(f"Loaded drone decision model from {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load model from {MODEL_PATH}: {e}")

# Initial load
load_model()

THRESHOLD_BOUNDS = {
    "max_wind_speed": (1.0, 80.0),
    "max_wind_gust": (1.0, 120.0),
    "max_rain_probability": (0.0, 100.0),
    "max_rain_hourly": (0.0, 50.0),
    "return_to_charging_rain_probability": (0.0, 100.0),
    "max_cloud_cover": (0.0, 100.0),
    "min_visibility": (0.0, 50_000.0),
    "max_safe_temperature": (0.0, 60.0),
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "https://agricultural-drone-scheduler.vercel.app",
]


def cors_origins() -> list[str]:
    configured = os.getenv("FRONTEND_ORIGINS", "")
    extra_origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    return sorted(set(DEFAULT_CORS_ORIGINS + extra_origins))

app = FastAPI(
    title="Agricultural Drone Scheduler API",
    description="Dashboard API backed by the DSS301 weather pipeline and decision engine.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def latest_clean_dataset() -> Path:
    candidates = [
        path
        for directory in CLEAN_DATA_DIRS
        for path in directory.glob("weather_clean_*.csv")
    ]
    if not candidates:
        raise HTTPException(status_code=503, detail="No cleaned weather forecast is available.")
    return max(candidates, key=lambda path: path.name)


def default_decision_config() -> dict[str, Any]:
    return {
        "thresholds": asdict(DecisionThresholds()),
        "unsafe_weather_codes": sorted(UNSAFE_WEATHER_CODES),
        "score_weights": SCORE_WEIGHTS,
    }


def validate_decision_config(payload: dict[str, Any]) -> dict[str, Any]:
    defaults = default_decision_config()
    raw_thresholds = payload.get("thresholds", payload)
    if not isinstance(raw_thresholds, dict):
        raise HTTPException(status_code=422, detail="Decision thresholds must be an object.")

    thresholds = defaults["thresholds"].copy()
    for key, value in raw_thresholds.items():
        if key not in thresholds:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Threshold '{key}' must be a number.") from exc
        lower, upper = THRESHOLD_BOUNDS[key]
        if not lower <= number <= upper:
            raise HTTPException(
                status_code=422,
                detail=f"Threshold '{key}' must be between {lower:g} and {upper:g}.",
            )
        thresholds[key] = number

    if thresholds["return_to_charging_rain_probability"] < thresholds["max_rain_probability"]:
        raise HTTPException(
            status_code=422,
            detail="Return-to-charging rain probability must be greater than or equal to delay-flight rain probability.",
        )

    unsafe_weather_codes = payload.get("unsafe_weather_codes", defaults["unsafe_weather_codes"])
    if not isinstance(unsafe_weather_codes, list):
        raise HTTPException(status_code=422, detail="Unsafe weather codes must be a list.")
    try:
        unsafe_weather_codes = sorted({int(code) for code in unsafe_weather_codes})
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Unsafe weather codes must contain numbers only.") from exc

    return {
        "thresholds": thresholds,
        "unsafe_weather_codes": unsafe_weather_codes,
        "score_weights": defaults["score_weights"],
    }


def read_decision_config() -> dict[str, Any]:
    if not DECISION_CONFIG_PATH.exists():
        return default_decision_config()
    try:
        payload = json.loads(DECISION_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Decision config file is invalid JSON.") from exc
    return validate_decision_config(payload)


def write_decision_config(config: dict[str, Any]) -> dict[str, Any]:
    CONFIG_DIR.mkdir(exist_ok=True)
    DECISION_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return config


def decision_config_response(config: dict[str, Any] | None = None) -> dict[str, Any]:
    active_config = config or read_decision_config()
    source = "file" if DECISION_CONFIG_PATH.exists() and active_config != default_decision_config() else "default"
    return {
        **active_config,
        "updated_at": datetime.fromtimestamp(
            DECISION_CONFIG_PATH.stat().st_mtime,
            VIETNAM_TZ,
        ).isoformat(timespec="minutes") if DECISION_CONFIG_PATH.exists() else None,
        "source": source,
    }


def config_to_engine_args(config: dict[str, Any]) -> tuple[DecisionThresholds, set[int]]:
    return DecisionThresholds(**config["thresholds"]), set(config["unsafe_weather_codes"])


def load_forecast() -> tuple[pd.DataFrame, Path, dict[str, Any], DecisionThresholds]:
    source_path = latest_clean_dataset()
    config = read_decision_config()
    thresholds, unsafe_weather_codes = config_to_engine_args(config)
    df = pd.read_csv(source_path)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    return add_decision_columns(df, thresholds, unsafe_weather_codes), source_path, config, thresholds


def load_json_report(name: str) -> dict[str, Any]:
    path = REPORT_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": stat.st_size,
        "updated_at": datetime.fromtimestamp(stat.st_mtime, VIETNAM_TZ).isoformat(timespec="minutes"),
    }


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def image_category_counts() -> dict[str, int]:
    if not IMAGE_KAGGLE_DIR.exists():
        return {}
    return {
        directory.name: len(image_files(directory))
        for directory in sorted(IMAGE_KAGGLE_DIR.iterdir())
        if directory.is_dir()
    }


def read_csv_shape(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"rows": 0, "columns": 0, "image_feature_columns": 0}
    df = pd.read_csv(path, nrows=5)
    with path.open(encoding="utf-8") as handle:
        row_count = max(sum(1 for _ in handle) - 1, 0)
    return {
        "rows": row_count,
        "columns": len(df.columns),
        "image_feature_columns": len([col for col in df.columns if col.startswith("img_feature_")]),
    }


def load_model_metrics() -> list[dict[str, Any]]:
    path = REPORT_DIR / "model_metrics.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def label_counts(series: pd.Series) -> dict[str, int]:
    return {str(label): int(count) for label, count in series.value_counts().sort_index().items()}


def calculate_location_model_metrics(location: str | None) -> list[dict[str, Any]]:
    if not location:
        return load_model_metrics()

    df = load_training_rows(location)
    if df.empty:
        return []

    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import GroupShuffleSplit

    from .decision_model.train_decision_model import build_feature_matrix, model_candidates

    config = read_decision_config()
    thresholds, unsafe_weather_codes = config_to_engine_args(config)
    df = add_decision_columns(df, thresholds, unsafe_weather_codes)
    x, y, _ = build_feature_matrix(df)
    groups = df["timestamp"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]
    evaluation_context = {
        "scope": "location",
        "location": location,
        "metric_basis": "macro_f1",
        "split_strategy": "GroupShuffleSplit grouped by timestamp",
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_timestamps": int(train_groups.nunique()),
        "test_timestamps": int(test_groups.nunique()),
        "train_class_distribution": label_counts(y_train),
        "test_class_distribution": label_counts(y_test),
    }

    metrics = []
    for name, pipeline in model_candidates().items():
        try:
            fitted = clone(pipeline).fit(x_train, y_train)
            predictions = fitted.predict(x_test)
            correct_predictions = int((predictions == y_test).sum())
            metrics.append({
                "model": name,
                "accuracy": round(accuracy_score(y_test, predictions), 4),
                "macro_precision": round(
                    precision_score(y_test, predictions, average="macro", zero_division=0),
                    4,
                ),
                "macro_recall": round(
                    recall_score(y_test, predictions, average="macro", zero_division=0),
                    4,
                ),
                "macro_f1": round(f1_score(y_test, predictions, average="macro", zero_division=0), 4),
                "weighted_f1": round(f1_score(y_test, predictions, average="weighted", zero_division=0), 4),
                "correct_predictions": correct_predictions,
                "incorrect_predictions": int(len(y_test) - correct_predictions),
                **evaluation_context,
            })
        except ValueError:
            metrics.append({
                "model": name,
                "accuracy": 0,
                "macro_precision": 0,
                "macro_recall": 0,
                "macro_f1": 0,
                "weighted_f1": 0,
                "correct_predictions": 0,
                "incorrect_predictions": int(len(y_test)),
                **evaluation_context,
            })

    return sorted(metrics, key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True)


def image_category_for_row(row: pd.Series) -> str:
    timestamp = pd.to_datetime(row.get("timestamp"))
    rain = float(row.get("precipitation", 0))
    cloud_cover = float(row.get("cloud_cover", 0))
    if rain > 0:
        return "Rain"
    if cloud_cover > 50:
        return "Cloudy"
    if 5 <= timestamp.hour <= 7:
        return "Sunrise"
    return "Shine"


def timestamp_to_image_filename(timestamp: Any) -> str:
    safe_timestamp = str(timestamp).replace(":", "-").replace(" ", "_")
    return f"{safe_timestamp}.jpg"


def load_training_rows(location: str | None = None) -> pd.DataFrame:
    if not FINAL_TRAINING_DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(FINAL_TRAINING_DATA_PATH)
    if location:
        df = df[df["location_name"] == location].copy()
    return df


def sample_generated_images(rows: pd.DataFrame | None = None, limit: int = 8) -> list[dict[str, Any]]:
    if rows is None or rows.empty:
        samples = image_files(GENERATED_IMAGE_DIR)[:limit]
        sample_rows = None
    else:
        sample_rows = rows.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").head(limit)
        samples = [GENERATED_IMAGE_DIR / timestamp_to_image_filename(row["timestamp"]) for _, row in sample_rows.iterrows()]

    payload = []
    for index, path in enumerate(samples):
        if not path.exists():
            continue
        date_part, _, time_part = path.stem.partition("_")
        timestamp = f"{date_part} {time_part.replace('-', ':')}" if time_part else path.stem
        item = {
            "filename": path.name,
            "timestamp": timestamp,
            "url": f"/api/ai-training/image/{path.name}",
        }
        if sample_rows is not None:
            row = sample_rows.iloc[index]
            item.update({
                "location": str(row.get("location_name", "")),
                "category": image_category_for_row(row),
                "weather_description": str(row.get("weather_description", "")),
            })
        payload.append(item)
    return payload


def ai_training_status(location: str | None = None) -> dict[str, Any]:
    generated_images = image_files(GENERATED_IMAGE_DIR)
    training_rows = load_training_rows(location)
    all_training_shape = read_csv_shape(FINAL_TRAINING_DATA_PATH)
    feature_shape = read_csv_shape(IMAGE_FEATURES_PATH)
    training_summary = load_json_report("training_summary.json")
    metrics = calculate_location_model_metrics(location)
    best_metric = metrics[0] if metrics else {}
    model_evaluation = {
        "metric_basis": best_metric.get("metric_basis", "macro_f1"),
        "split_strategy": best_metric.get("split_strategy", training_summary.get("strategy")),
        "train_rows": best_metric.get("train_rows", training_summary.get("train_rows", 0)),
        "test_rows": best_metric.get("test_rows", training_summary.get("test_rows", 0)),
        "train_timestamps": best_metric.get("train_timestamps", training_summary.get("train_groups", 0)),
        "test_timestamps": best_metric.get("test_timestamps", training_summary.get("test_groups", 0)),
        "train_class_distribution": best_metric.get("train_class_distribution", {}),
        "test_class_distribution": best_metric.get("test_class_distribution", {}),
    }
    timestamps = training_rows["timestamp"].drop_duplicates().tolist() if not training_rows.empty else []
    generated_count = sum((GENERATED_IMAGE_DIR / timestamp_to_image_filename(timestamp)).exists() for timestamp in timestamps)
    category_counts = (
        training_rows.apply(image_category_for_row, axis=1).value_counts().sort_index().to_dict()
        if not training_rows.empty
        else image_category_counts()
    )
    return {
        "location": location,
        "scope": "location" if location else "all",
        "refreshed_at": datetime.now(VIETNAM_TZ).isoformat(timespec="seconds"),
        "image_categories": category_counts,
        "generated_image_count": generated_count if location else len(generated_images),
        "generated_image_samples": sample_generated_images(training_rows if location else None),
        "image_features": {
            **feature_shape,
            "file": file_metadata(IMAGE_FEATURES_PATH),
        },
        "training_dataset": {
            "rows": int(len(training_rows)) if location else all_training_shape["rows"],
            "columns": all_training_shape["columns"],
            "image_feature_columns": all_training_shape["image_feature_columns"],
            "file": file_metadata(FINAL_TRAINING_DATA_PATH),
        },
        "model": {
            "file": file_metadata(MODEL_PATH),
            "best_model": best_metric.get("model"),
            "macro_f1": best_metric.get("macro_f1"),
            "accuracy": best_metric.get("accuracy"),
        },
        "training_summary": training_summary,
        "model_evaluation": model_evaluation,
        "metrics": metrics,
    }


def run_ai_step(step_name: str, runner, location: str | None = None) -> dict[str, Any]:
    if not AI_TRAINING_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="AI training pipeline is already running.")
    started_at = datetime.now(VIETNAM_TZ)
    output = StringIO()
    try:
        with redirect_stdout(output):
            result = runner()
        finished_at = datetime.now(VIETNAM_TZ)
        return {
            "status": "ok",
            "step": step_name,
            "result": result,
            "started_at": started_at.isoformat(timespec="minutes"),
            "finished_at": finished_at.isoformat(timespec="minutes"),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 2),
            "log": output.getvalue(),
            "ai_training": ai_training_status(location),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(exc),
                "step": step_name,
                "started_at": started_at.isoformat(timespec="minutes"),
                "log": output.getvalue(),
            },
        ) from exc
    finally:
        AI_TRAINING_LOCK.release()


def parse_reference_time(at: str | None) -> datetime:
    if not at:
        return datetime.now(VIETNAM_TZ).replace(tzinfo=None)
    try:
        return datetime.fromisoformat(at).replace(tzinfo=None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Parameter 'at' must be ISO-8601.") from exc


def pick_operational_slot(df: pd.DataFrame, reference_time: datetime) -> pd.Series:
    upcoming = df[df["timestamp_dt"] >= reference_time]
    if not upcoming.empty:
        return upcoming.sort_values("timestamp_dt").iloc[0]
    return df.sort_values("timestamp_dt").iloc[-1]


def as_number(value: Any, digits: int = 1) -> float:
    return round(float(value), digits)


def serialize_slot(row: pd.Series, thresholds: DecisionThresholds) -> dict[str, Any]:
    action = str(row["decision_action"])
    return {
        "timestamp": row["timestamp_dt"].isoformat(timespec="minutes"),
        "time": row["timestamp_dt"].strftime("%H:%M"),
        "end_time": (row["timestamp_dt"] + timedelta(hours=1)).strftime("%H:%M"),
        "temperature": as_number(row["temperature_2m"]),
        "humidity": as_number(row["relative_humidity_2m"], 0),
        "rain_probability": as_number(row["precipitation_probability"], 0),
        "precipitation": as_number(row["precipitation"]),
        "cloud_cover": as_number(row["cloud_cover"], 0),
        "visibility": as_number(row["visibility"], 0),
        "wind_speed": as_number(row["wind_speed_10m"]),
        "wind_gust": as_number(row["wind_gusts_10m"]),
        "weather_code": int(row["weather_code"]),
        "weather_description": str(row.get("weather_description", "")),
        "flyability_score": as_number(float(row["flyability_score"]) * 100, 0),
        "risk_level": str(row["risk_level"]),
        "crop_condition": str(row["crop_condition"]),
        "dynamic_flow_rate_pct": as_number(row["dynamic_flow_rate_pct"]),
        "decision_action": action,
        "schedule_eligible": action == "TAKE_OFF",
        "recommendation_text": build_recommendation_text(row, action, thresholds),
        "evapotranspiration": as_number(row.get("evapotranspiration", 0.0), 2),
        "soil_moisture": as_number(row.get("soil_moisture_0_to_7cm", 0.0), 2),
    }


def recommended_slots(
    df: pd.DataFrame,
    reference_time: datetime,
    thresholds: DecisionThresholds,
) -> list[dict[str, Any]]:
    upcoming = df[df["timestamp_dt"] >= reference_time].copy()
    safe = upcoming[upcoming["decision_action"] == "TAKE_OFF"].copy()
    candidates = safe if not safe.empty else upcoming
    if candidates.empty:
        candidates = df.copy()
    candidates = candidates.sort_values(
        ["flyability_score", "timestamp_dt"],
        ascending=[False, True],
    ).head(3)
    return [serialize_slot(row, thresholds) for _, row in candidates.iterrows()]


def dashboard_kpis() -> list[dict[str, Any]]:
    backtest = load_json_report("backtesting_summary.json")
    training = load_json_report("training_summary.json")
    return [
        {
            "key": "risk_reduction",
            "label": "Rủi ro vận hành giảm",
            "value": backtest.get("risk_reduction_pct", 0),
            "suffix": "%",
            "note": f"{backtest.get('baseline_risky_operations', 0)} xuống {backtest.get('dss_risky_operations', 0)} ca rủi ro",
            "tone": "green",
        },
        {
            "key": "waste_reduction",
            "label": "Lãng phí nước / thuốc giảm",
            "value": backtest.get("waste_reduction_pct", 0),
            "suffix": "%",
            "note": f"Tiết kiệm {backtest.get('baseline_waste_liters', 0) - backtest.get('dss_waste_liters', 0):,.0f} lít mô phỏng",
            "tone": "blue",
        },
        {
            "key": "evaluated_rows",
            "label": "Mẫu dữ liệu đánh giá",
            "value": training.get("input_rows", 0),
            "suffix": "",
            "note": f"{training.get('unique_timestamps', 0)} khung giờ duy nhất",
            "tone": "orange",
        },
    ]


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "agricultural-drone-scheduler"}


@app.get("/api/ai-training/status")
def get_ai_training_status(location: str | None = None) -> dict[str, Any]:
    return ai_training_status(location)


@app.get("/api/ai-training/metrics")
def get_ai_training_metrics(location: str | None = None) -> dict[str, Any]:
    return {
        "metrics": calculate_location_model_metrics(location),
        "training_summary": load_json_report("training_summary.json"),
        "classification_report": (REPORT_DIR / "classification_report.txt").read_text(encoding="utf-8")
        if (REPORT_DIR / "classification_report.txt").exists()
        else "",
    }


@app.get("/api/ai-training/image/{filename}")
def get_ai_training_image(filename: str) -> FileResponse:
    image_path = (GENERATED_IMAGE_DIR / filename).resolve()
    if image_path.parent != GENERATED_IMAGE_DIR.resolve() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=404, detail="Image not found.")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(image_path)


# Image simulation and feature extraction are removed from the MVP.


@app.post("/api/ai-training/train")
def train_ai_model(location: str | None = None) -> dict[str, Any]:
    def runner() -> dict[str, Any]:
        from .data_pipeline.merge_data import main as merge_data_main
        from .decision_model.train_decision_model import train_models

        merge_data_main()
        res = train_models(FINAL_TRAINING_DATA_PATH)
        load_model()
        return res

    return run_ai_step("train_model", runner, location)


@app.get("/api/decision-config")
def get_decision_config() -> dict[str, Any]:
    return decision_config_response()


@app.put("/api/decision-config")
def update_decision_config(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    config = validate_decision_config(payload)
    return decision_config_response(write_decision_config(config))


@app.post("/api/decision-config/reset")
def reset_decision_config() -> dict[str, Any]:
    if DECISION_CONFIG_PATH.exists():
        DECISION_CONFIG_PATH.unlink()
    return decision_config_response(default_decision_config())


@app.post("/api/pipeline/run")
def run_pipeline_endpoint(days: int = 3, skip_upload: bool = False) -> dict[str, Any]:
    if not PIPELINE_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Weather pipeline is already running.")

    started_at = datetime.now(VIETNAM_TZ)
    output = StringIO()
    try:
        with redirect_stdout(output):
            result = run_weather_pipeline(days=days, skip_upload=skip_upload)

        finished_at = datetime.now(VIETNAM_TZ)
        return {
            **result,
            "started_at": started_at.isoformat(timespec="minutes"),
            "finished_at": finished_at.isoformat(timespec="minutes"),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 2),
            "log": output.getvalue(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(exc),
                "started_at": started_at.isoformat(timespec="minutes"),
                "log": output.getvalue(),
            },
        ) from exc
    finally:
        PIPELINE_LOCK.release()


@app.get("/api/locations")
def locations() -> list[dict[str, Any]]:
    df, _, _, _ = load_forecast()
    location_rows = (
        df[["location_name", "latitude", "longitude"]]
        .drop_duplicates()
        .sort_values("location_name")
    )
    return [
        {
            "id": row["location_name"],
            "name": row["location_name"],
            "latitude": as_number(row["latitude"], 4),
            "longitude": as_number(row["longitude"], 4),
        }
        for _, row in location_rows.iterrows()
    ]


@app.get("/api/dashboard")
def dashboard(
    location: str = "Dong Thap",
    at: str | None = None,
) -> dict[str, Any]:
    df, source_path, config, thresholds = load_forecast()
    location_df = df[df["location_name"] == location].copy()
    if location_df.empty:
        available = sorted(df["location_name"].unique().tolist())
        raise HTTPException(
            status_code=404,
            detail=f"Unknown location '{location}'. Available locations: {available}",
        )

    reference_time = parse_reference_time(at)
    current = pick_operational_slot(location_df, reference_time)
    selected_date = current["timestamp_dt"].date()
    daily_df = location_df[location_df["timestamp_dt"].dt.date == selected_date].copy()
    daily_df = daily_df.sort_values("timestamp_dt")
    current_payload = serialize_slot(current, thresholds)

    # ── Auto-save to Supabase (fire-and-forget) ──
    try:
        _auto_save_to_db(daily_df, thresholds)
    except Exception:
        pass  # Non-blocking: DB save failure must not break dashboard

    import math
    def clean_nans(obj):
        if isinstance(obj, dict):
            return {k: clean_nans(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nans(v) for v in obj]
        elif isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    return clean_nans({
        "source": {
            "dataset": source_path.name,
            "updated_at": datetime.fromtimestamp(
                source_path.stat().st_mtime,
                VIETNAM_TZ,
            ).isoformat(timespec="minutes"),
            "reference_time": reference_time.isoformat(timespec="minutes"),
        },
        "decision_config": decision_config_response(config),
        "location": {
            "id": location,
            "name": location,
            "latitude": as_number(current["latitude"], 4),
            "longitude": as_number(current["longitude"], 4),
        },
        "current": current_payload,
        "forecast": [serialize_slot(row, thresholds) for _, row in daily_df.iterrows()],
        "recommended_slots": recommended_slots(location_df, reference_time, thresholds),
        "has_safe_slot": bool(
            (
                location_df[location_df["timestamp_dt"] >= reference_time]["decision_action"]
                == "TAKE_OFF"
            ).any()
        ),
        "timeline_tiles": [
            serialize_slot(row, thresholds)
            for _, row in daily_df.iterrows()
        ],
        "kpis": dashboard_kpis(),
        "backtesting_note": load_json_report("backtesting_summary.json").get(
            "interpretation",
            "",
        ),
    })


# ── Auto-save helper ──────────────────────────────────────────

def _auto_save_to_db(daily_df: pd.DataFrame, thresholds: DecisionThresholds) -> None:
    """Persist the current forecast slice to Supabase (best-effort)."""
    flight_rows = []
    weather_rows = []
    for _, row in daily_df.iterrows():
        ts = str(row["timestamp_dt"].isoformat())
        loc = str(row["location_name"])
        flight_rows.append({
            "location_name": loc,
            "flight_timestamp": ts,
            "decision_action": str(row["decision_action"]),
            "risk_level": str(row["risk_level"]),
            "flyability_score": float(row["flyability_score"]),
            "dynamic_flow_rate_pct": float(row["dynamic_flow_rate_pct"]),
            "crop_condition": str(row["crop_condition"]),
            "recommendation_text": build_recommendation_text(row, str(row["decision_action"]), thresholds),
            "weather_source": "api",
        })
        weather_rows.append({
            "location_name": loc,
            "timestamp": ts,
            "temperature_2m": float(row.get("temperature_2m", 0)),
            "relative_humidity_2m": float(row.get("relative_humidity_2m", 0)),
            "precipitation_probability": float(row.get("precipitation_probability", 0)),
            "precipitation": float(row.get("precipitation", 0)),
            "cloud_cover": float(row.get("cloud_cover", 0)),
            "visibility": float(row.get("visibility", 0)),
            "wind_speed_10m": float(row.get("wind_speed_10m", 0)),
            "wind_gusts_10m": float(row.get("wind_gusts_10m", 0)),
            "weather_code": int(row.get("weather_code", 0)),
            "weather_description": str(row.get("weather_description", "")),
            "flyability_score": float(row["flyability_score"]),
            "decision_action": str(row["decision_action"]),
            "risk_level": str(row["risk_level"]),
            "source": "WeatherAPI",
        })
    if flight_rows:
        db.save_flight_logs_batch(flight_rows)
    if weather_rows:
        db.save_analyzed_weather_batch(weather_rows)


# ══════════════════════════════════════════════════════════════
# 3-LAYER DECISION ENGINE AND CHAT ENDPOINTS
# ══════════════════════════════════════════════════════════════

RISK_WEIGHTS = {
    "TAKE_OFF": 0,
    "DELAY_FLIGHT": 1,
    "LOCK_SPRAY": 2,
    "RETURN_TO_CHARGING": 3
}

def get_risk_level_from_action(action: str) -> str:
    if action in {"LOCK_SPRAY", "RETURN_TO_CHARGING"}:
        return "HIGH"
    if action == "DELAY_FLIGHT":
        return "MEDIUM"
    return "LOW"

def generate_xai_explanation(row: pd.Series, thresholds: DecisionThresholds) -> str:
    reasons = []
    wind = float(row.get("wind_speed_10m", 0))
    gust = float(row.get("wind_gusts_10m", 0))
    rain = float(row.get("precipitation", 0))
    rain_prob = float(row.get("precipitation_probability", 0))
    temp = float(row.get("temperature_2m", 0))
    weather_code = int(row.get("weather_code", 0))
    
    if rain > thresholds.max_rain_hourly:
        reasons.append(f"lượng mưa lớn ({rain:.1f} mm/h vượt quá ngưỡng an toàn {thresholds.max_rain_hourly:.1f} mm/h)")
    if rain_prob > thresholds.max_rain_probability:
        reasons.append(f"xác suất mưa cao ({rain_prob:.0f}% vượt quá ngưỡng an toàn {thresholds.max_rain_probability:.0f}%)")
    if weather_code in UNSAFE_WEATHER_CODES:
        reasons.append(f"mã thời tiết nguy hiểm ({weather_code})")
    if wind > thresholds.max_wind_speed:
        reasons.append(f"tốc độ gió cao ({wind:.1f} km/h vượt quá ngưỡng an toàn {thresholds.max_wind_speed:.1f} km/h)")
    if gust > thresholds.max_wind_gust:
        reasons.append(f"gió giật mạnh ({gust:.1f} km/h vượt quá ngưỡng an toàn {thresholds.max_wind_gust:.1f} km/h)")
    if temp > thresholds.max_safe_temperature:
        reasons.append(f"nhiệt độ quá nóng ({temp:.1f}°C vượt quá ngưỡng an toàn {thresholds.max_safe_temperature:.1f}°C)")
        
    if reasons:
        return "Hạn chế bay do: " + ", ".join(reasons) + "."
    return "Thời tiết hoàn hảo, không có cảnh báo an toàn nào bị vi phạm."


def compile_decision_factors(
    row: pd.Series,
    thresholds: DecisionThresholds,
    drone_profile: dict[str, Any] | None = None,
    crop_stage: dict[str, Any] | None = None,
    pesticide: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    factors = []

    wind = float(row.get("wind_speed_10m", 0))
    gust = float(row.get("wind_gusts_10m", 0))
    rain = float(row.get("precipitation", 0))
    rain_prob = float(row.get("precipitation_probability", 0))
    temp = float(row.get("temperature_2m", 0))
    humidity = float(row.get("relative_humidity_2m", 0))
    visibility = float(row.get("visibility", 10000))
    cloud_cover = float(row.get("cloud_cover", 0))
    hour = int(row.get("hour", 12))

    max_w = float(drone_profile.get("max_wind_resistance_kph", thresholds.max_wind_speed)) if drone_profile else thresholds.max_wind_speed
    max_g = float(drone_profile.get("max_gust_resistance_kph", thresholds.max_wind_gust)) if drone_profile else thresholds.max_wind_gust

    # 1. wind_speed
    if wind > max_w:
        factors.append({"factor": "drone_wind_limit", "verdict": "STOP", "message": f"Tốc độ gió {wind:.1f} km/h vượt giới hạn {max_w:.1f} km/h"})
    elif wind >= 18.0:
        factors.append({"factor": "wind_speed", "verdict": "WARN", "message": f"Tốc độ gió ngang {wind:.1f} km/h tăng nguy cơ tán xạ thuốc"})
    else:
        factors.append({"factor": "wind_speed", "verdict": "ALLOW", "message": f"Gió nhẹ {wind:.1f} km/h, an toàn"})

    # 2. wind_gust
    if gust > max_g:
        factors.append({"factor": "drone_gust_limit", "verdict": "STOP", "message": f"Gió giật {gust:.1f} km/h vượt giới hạn {max_g:.1f} km/h"})
    elif gust >= 25.0:
        factors.append({"factor": "wind_gust", "verdict": "WARN", "message": f"Gió giật mạnh {gust:.1f} km/h có rủi ro đổ ngã lúa"})
    else:
        factors.append({"factor": "wind_gust", "verdict": "ALLOW", "message": f"Gió giật {gust:.1f} km/h trong ngưỡng an toàn"})

    # 3. temperature
    if temp >= 35.0:
        factors.append({"factor": "temperature", "verdict": "STOP", "message": f"Nhiệt độ cực đoan {temp:.1f}°C gây sốc nhiệt và bốc hơi thuốc"})
    elif temp >= 32.0:
        factors.append({"factor": "temperature", "verdict": "WARN", "message": f"Thời tiết nóng {temp:.1f}°C, đề xuất tăng flow rate"})
    else:
        factors.append({"factor": "temperature", "verdict": "ALLOW", "message": f"Nhiệt độ mát mẻ {temp:.1f}°C"})

    # 4. humidity
    if humidity < 45.0:
        factors.append({"factor": "humidity", "verdict": "STOP", "message": f"Độ ẩm quá thấp {humidity:.1f}% gây co hạt sương"})
    elif humidity > 90.0:
        factors.append({"factor": "humidity", "verdict": "WARN", "message": f"Độ ẩm quá cao {humidity:.1f}% gây pha loãng thuốc trên lá"})
    else:
        factors.append({"factor": "humidity", "verdict": "ALLOW", "message": f"Độ ẩm {humidity:.1f}% thích hợp"})

    # 5. rain
    if rain > thresholds.max_rain_hourly:
        factors.append({"factor": "rain", "verdict": "STOP", "message": f"Lượng mưa lớn {rain:.1f} mm/h gây rửa trôi hoàn toàn"})
    elif rain_prob > thresholds.max_rain_probability:
        factors.append({"factor": "rain", "verdict": "STOP", "message": f"Xác suất mưa cao {rain_prob:.0f}%"})
    else:
        factors.append({"factor": "rain", "verdict": "ALLOW", "message": "Không có mưa dự báo"})

    # 6. visibility
    if visibility < thresholds.min_visibility:
        factors.append({"factor": "visibility", "verdict": "STOP", "message": f"Tầm nhìn {visibility:.0f} m dưới ngưỡng VLOS"})
    else:
        factors.append({"factor": "visibility", "verdict": "ALLOW", "message": f"Tầm nhìn tốt {visibility:.0f} m"})

    # 7. cloud_cover
    if cloud_cover >= 80.0:
        factors.append({"factor": "cloud_cover", "verdict": "ALLOW", "message": f"Trời râm mát ({cloud_cover:.0f}%), tốt cho thuốc sinh học"})
    else:
        factors.append({"factor": "cloud_cover", "verdict": "ALLOW", "message": f"Mây che phủ {cloud_cover:.0f}%"})

    # 8. Giai đoạn sinh trưởng cấm bay giờ thụ phấn
    if crop_stage:
        ban_start = crop_stage.get("hard_ban_start_hour")
        ban_end = crop_stage.get("hard_ban_end_hour")
        if ban_start is not None and ban_end is not None:
            if ban_start <= hour <= ban_end:
                factors.append({"factor": "stage_time_ban", "verdict": "STOP", "message": f"Giờ thụ phấn {hour}h cấm cất cánh"})
            else:
                factors.append({"factor": "stage_time_ban", "verdict": "ALLOW", "message": "Khung giờ thụ phấn an toàn"})

    # 9. Pesticide UV timing
    if pesticide and pesticide.get("uv_sensitivity"):
        if temp >= 32.0 and cloud_cover < 50.0:
            factors.append({"factor": "pesticide_uv_timing", "verdict": "STOP", "message": f"Thuốc sinh học nhạy cảm UV dưới nắng gắt"})
        else:
            factors.append({"factor": "pesticide_uv_timing", "verdict": "ALLOW", "message": "UV trong ngưỡng an toàn"})

    return factors
def run_3_layer_decision_engine(
    df: pd.DataFrame,
    thresholds: DecisionThresholds,
    unsafe_weather_codes: set[int] | frozenset[int],
    farm_size_ha: float = 10.0,
    distance_to_field_km: float = 1.0,
    drone_profile: dict[str, Any] | None = None,
    crop_stage: dict[str, Any] | None = None,
    pesticide: dict[str, Any] | None = None,
    current_water_level: float = -12.0,
) -> list[dict[str, Any]]:
    global MODEL_PAYLOAD
    if MODEL_PAYLOAD is None:
        load_model()
        
    has_model = (MODEL_PAYLOAD is not None)
    if has_model:
        feature_cols = MODEL_PAYLOAD["feature_columns"]
        champion = MODEL_PAYLOAD["champion"]
        
        df_ml = df.copy()
        # Populate drone parameters
        df_ml["max_wind_resistance_kph"] = float(drone_profile.get("max_wind_resistance_kph", 28.8)) if drone_profile else 28.8
        df_ml["max_gust_resistance_kph"] = float(drone_profile.get("max_gust_resistance_kph", 35.0)) if drone_profile else 35.0
        
        # Populate pesticide parameters
        df_ml["uv_sensitivity"] = float(pesticide.get("uv_sensitivity", 0.5)) if pesticide else 0.5
        df_ml["rain_washout_hours"] = float(pesticide.get("rain_washout_hours", 2.0)) if pesticide else 2.0
        
        # Populate crop stage one-hot
        stage_code = crop_stage.get("stage_code", "TILLERING") if crop_stage else "TILLERING"
        for stg in ["SEEDLING", "TILLERING", "BOOTING", "GRAIN_FILLING"]:
            df_ml[f"crop_stage_{stg}"] = 1 if stage_code == stg else 0
            
        # Also ensure timestamp hour and month exist if needed by the model
        if "hour" in feature_cols and "hour" not in df_ml.columns:
            df_ml["hour"] = df_ml["timestamp_dt"].dt.hour
        if "month" in feature_cols and "month" not in df_ml.columns:
            df_ml["month"] = df_ml["timestamp_dt"].dt.month
            
        # Build derived weather features using the shared engineering module
        from app.features.engineering import build_features
        if "timestamp" in df_ml.columns:
            df_ml = build_features(df_ml, ts_col="timestamp")
        else:
            # Fallback if timestamp is missing, though we know timestamp_dt exists
            df_ml["timestamp"] = df_ml["timestamp_dt"]
            df_ml = build_features(df_ml, ts_col="timestamp")

        # Ensure any other missing columns from feature_cols are filled with 0
        for col in feature_cols:
            if col not in df_ml.columns:
                df_ml[col] = 0

        X = df_ml[feature_cols].copy()
        
        champ_preds_idx = champion.predict(X)
        champ_probs = champion.predict_proba(X)
        
        challenger = MODEL_PAYLOAD.get("challenger")
        if challenger:
            chall_preds_idx = challenger.predict(X)
            chall_probs = challenger.predict_proba(X)
        else:
            chall_preds_idx = champ_preds_idx
            chall_probs = champ_probs
            
        label_map = {v: k for k, v in MODEL_PAYLOAD.get("label_to_idx", {}).items()}
        if not label_map:
            label_map = {0: "FLY", 1: "DELAY", 2: "LOCK_SPRAY", 3: "NO_FLY"}
            
        _ML_TO_UNIFIED = {
            "TAKE_OFF": "FLY", 
            "DELAY_FLIGHT": "DELAY", 
            "LOCK_SPRAY": "LOCK_SPRAY", 
            "RETURN_TO_CHARGING": "NO_FLY",
            "FLY": "FLY", "DELAY": "DELAY", "NO_FLY": "NO_FLY"
        }
            
        champ_preds = [_ML_TO_UNIFIED.get(label_map.get(idx, "FLY"), "FLY") for idx in champ_preds_idx]
        chall_preds = [_ML_TO_UNIFIED.get(label_map.get(idx, "FLY"), "FLY") for idx in chall_preds_idx]
    else:
        champ_preds = ["FLY"] * len(df)
        champ_probs = [[1.0, 0.0, 0.0, 0.0]] * len(df)
        chall_preds = champ_preds
        chall_probs = champ_probs
        
    slots = []
    n = len(df)
    
    tank_cap = float(drone_profile.get("tank_capacity_liters", 30)) if drone_profile else 30
    tank_usable_l = tank_cap * 0.9

    for i in range(n):
        row = df.iloc[i]
        
        # 1. Rule Engine evaluation
        rule_action = derive_decision_action(row, thresholds, unsafe_weather_codes, drone_profile, crop_stage, pesticide)
        flyability_score = calculate_flyability_score(row, thresholds, unsafe_weather_codes, drone_profile)
        crop_impact_score = calculate_crop_safety_score(row, crop_stage, pesticide)
        
        flow_rate_l_ha = calculate_dynamic_flow_rate(row, thresholds, crop_stage)
        spray_quality_score = calculate_spray_quality_score(row, pesticide, flow_rate_l_ha)
        
        # 2. AI validation
        p_champ = float(champ_probs[i][0]) # Probability of FLY class
        ai_pred = champ_preds[i]
        
        p_chall = float(chall_probs[i][0])
        chall_ai_pred = chall_preds[i]
        
        # Determine the most conservative AI prediction
        RISK_LEVELS = {"FLY": 0, "DELAY": 1, "LOCK_SPRAY": 2, "NO_FLY": 3}
        worst_ai_pred = ai_pred if RISK_LEVELS.get(ai_pred, 0) >= RISK_LEVELS.get(chall_ai_pred, 0) else chall_ai_pred
        
        # Calibration: Force AI predictions to align with updated humidity threshold
        humidity = float(row.get("relative_humidity_2m", 0))
        if humidity <= 90.0 and rule_action == "FLY" and worst_ai_pred == "DELAY":
            worst_ai_pred = "FLY"
            ai_pred = "FLY"
            chall_ai_pred = "FLY"
            
        # Calibration: If AI predicts NO_FLY but physical rules say it's just LOCK_SPRAY (e.g. drizzle),
        # we downgrade the AI to LOCK_SPRAY so it doesn't ban flight unnecessarily.
        if worst_ai_pred == "NO_FLY" and rule_action == "LOCK_SPRAY":
            wind = float(row.get("wind_speed_10m", 0))
            gust = float(row.get("wind_gusts_10m", 0))
            rain = float(row.get("precipitation", 0))
            max_w = float(drone_profile.get("max_wind_resistance_kph", 28.8)) if drone_profile else 28.8
            max_g = float(drone_profile.get("max_gust_resistance_kph", 35.0)) if drone_profile else 35.0
            
            if wind <= max_w and gust <= max_g and rain <= thresholds.max_rain_hourly:
                worst_ai_pred = "LOCK_SPRAY"
        
        # If AI is more conservative than rules, we downgrade
        final_decision = rule_action
        if RISK_LEVELS.get(worst_ai_pred, 0) > RISK_LEVELS.get(rule_action, 0):
            final_decision = worst_ai_pred
            
            # Align flyability score with the AI's conservative decision
            if final_decision == "NO_FLY":
                flyability_score = min(flyability_score, 0.39)
            elif final_decision == "DELAY":
                flyability_score = min(flyability_score, 0.69)
            elif final_decision == "LOCK_SPRAY":
                flyability_score = min(flyability_score, 0.49)
            
        was_conflict = (ai_pred != chall_ai_pred) or (abs(p_champ - p_chall) > 0.20)
        is_safe_to_fly = (final_decision == "FLY")
        
        # 3. Flight Estimates (BRD matching)
        if is_safe_to_fly:
            total_liters = round(flow_rate_l_ha * farm_size_ha, 2)
            import math
            estimated_trips = math.ceil(total_liters / tank_usable_l)
            
            stage_code = crop_stage.get("stage_code") if crop_stage else ""
            if stage_code == "SEEDLING":
                planned_speed_mps = 6.5
                swath_width_m = 6.0
            elif stage_code in ["BOOTING", "GRAIN_FILLING"]:
                planned_speed_mps = 4.5
                swath_width_m = 5.0
            else:
                planned_speed_mps = 5.5
                swath_width_m = 5.5
                
            overlap_factor = 1.1
            estimated_flight_time_min = ((farm_size_ha * 10000) / (planned_speed_mps * swath_width_m * 60)) * overlap_factor
            estimated_total_time_min = estimated_flight_time_min + (estimated_trips - 1) * 3.0 + 5.0  # turnaround 3 min, setup 5 min
            
            battery_cycles = estimated_trips + math.ceil(distance_to_field_km * 2 * 0.1)
        else:
            total_liters = 0.0
            estimated_trips = 0
            planned_speed_mps = 0.0
            swath_width_m = 0.0
            estimated_flight_time_min = 0.0
            estimated_total_time_min = 0.0
            battery_cycles = 0

        # AWD Recommendation
        start_time = row["timestamp_dt"]
        end_time = start_time + timedelta(hours=24)
        future_df = df[(df["timestamp_dt"] > start_time) & (df["timestamp_dt"] <= end_time)]
        future_precip = float(future_df["precipitation"].sum()) if not future_df.empty else 0.0
        awd_thresh = float(crop_stage.get("awd_threshold_cm", -15.0)) if crop_stage else -15.0
        awd_rec = get_awd_recommendation(current_water_level, awd_thresh, future_precip)

        slots.append({
            "row": row,
            "champion_score": p_champ,
            "challenger_score": p_chall,
            "was_conflict": was_conflict,
            "flyability_score": flyability_score,
            "is_safe_to_fly": is_safe_to_fly,
            "champion_pred": ai_pred,
            "champion_conf": p_champ,
            "challenger_pred": chall_ai_pred,
            "challenger_conf": p_chall,
            "final_decision": final_decision,
            "risk_level": "LOW" if is_safe_to_fly else "HIGH",
            "xai_alert": build_recommendation_text(row, final_decision, thresholds, drone_profile, crop_stage, pesticide),
            "crop_impact_score": crop_impact_score,
            "spray_quality_score": spray_quality_score,
            "awd_recommendation": awd_rec,
            "factors": compile_decision_factors(row, thresholds, drone_profile, crop_stage, pesticide),
            "resource_regressor": {
                "flow_rate_l_ha": flow_rate_l_ha,
                "total_liters": total_liters,
                "target_area_ha": farm_size_ha,
                "tank_usable_l": tank_usable_l,
                "swath_width_m": swath_width_m,
                "planned_speed_mps": planned_speed_mps,
                "sorties": estimated_trips,
                "estimated_trips": estimated_trips,
                "estimated_flight_time_min": round(estimated_flight_time_min, 1),
                "estimated_total_time_min": round(estimated_total_time_min, 1),
                "battery_cycles": battery_cycles,
            }
        })
        
    return slots

def _reconcile_and_save_decisions_log(
    slots: list[dict[str, Any]], 
    location: str, 
    farm_size_ha: float, 
    thresholds: DecisionThresholds,
    distance_to_field_km: float = 1.0,
    drone_profile: dict[str, Any] | None = None,
    crop_stage: dict[str, Any] | None = None,
    pesticide: dict[str, Any] | None = None,
    current_water_level: float = -12.0,
) -> list[dict[str, Any]]:
    try:
        client = db.get_client()
    except Exception as e:
        print(f"Failed to get db client: {e}")
        return slots

    timestamps = [s["row"]["timestamp_dt"].isoformat() for s in slots]
    if not timestamps:
        return slots

    use_singular_schema = True
    try:
        res = client.table("flight_decision_log").select("*").in_("slot_timestamp", timestamps).eq("location_name", location).execute()
        existing_rows = res.data or []
    except Exception as e:
        use_singular_schema = False
        try:
            res = client.table("flight_decisions_log").select("*").in_("timestamp", timestamps).execute()
            existing_rows = res.data or []
        except Exception as ex:
            existing_rows = []

    existing_map = {}
    for r in existing_rows:
        try:
            if use_singular_schema:
                db_ts = pd.to_datetime(r["slot_timestamp"]).replace(tzinfo=None).isoformat()
                existing_map[db_ts] = r
            else:
                snapshot = r.get("weather_snapshot", {})
                loc_name = snapshot.get("location_name")
                if loc_name == location:
                    db_ts = pd.to_datetime(r["timestamp"]).replace(tzinfo=None).isoformat()
                    existing_map[db_ts] = r
        except Exception as e:
            pass

    to_insert_rows = []
    TRAVEL_COST_FACTOR = 0.1
    tank_cap = float(drone_profile.get("tank_capacity_liters", 30)) if drone_profile else 30
    tank_usable_l = tank_cap * 0.9

    for idx, s in enumerate(slots):
        row = s["row"]
        ts_iso = pd.to_datetime(row["timestamp_dt"]).replace(tzinfo=None).isoformat()

        db_row = existing_map.get(ts_iso)
        if db_row:
            s["id"] = db_row["log_id"] if use_singular_schema else db_row["id"]
            s["was_human_overridden"] = db_row.get("is_user_overridden" if use_singular_schema else "was_human_overridden", False)
            if s["was_human_overridden"]:
                s["original_ai_decision"] = s["final_decision"]
                s["final_decision"] = db_row["system_decision" if use_singular_schema else "final_decision"]
                if use_singular_schema:
                    s["user_notes"] = db_row.get("override_reason") or ""
                else:
                    s["user_notes"] = db_row.get("weather_snapshot", {}).get("user_override_notes", "")
                override_decision = s["final_decision"]
                s["is_safe_to_fly"] = (override_decision == "FLY")
                
                if s["is_safe_to_fly"]:
                    flow_rate_l_ha = calculate_dynamic_flow_rate(row, thresholds, crop_stage)
                    total_liters = round(flow_rate_l_ha * farm_size_ha, 2)
                    import math
                    estimated_trips = math.ceil(total_liters / tank_usable_l)
                    
                    stage_code = crop_stage.get("stage_code") if crop_stage else ""
                    if stage_code == "SEEDLING":
                        planned_speed_mps = 6.5
                        swath_width_m = 6.0
                    elif stage_code in ["BOOTING", "GRAIN_FILLING"]:
                        planned_speed_mps = 4.5
                        swath_width_m = 5.0
                    else:
                        planned_speed_mps = 5.5
                        swath_width_m = 5.5
                        
                    overlap_factor = 1.1
                    estimated_flight_time_min = ((farm_size_ha * 10000) / (planned_speed_mps * swath_width_m * 60)) * overlap_factor
                    estimated_total_time_min = estimated_flight_time_min + (estimated_trips - 1) * 3.0 + 5.0
                    
                    battery_cycles = estimated_trips + math.ceil(distance_to_field_km * 2 * TRAVEL_COST_FACTOR)
                    
                    s["resource_regressor"] = {
                        "flow_rate_l_ha": flow_rate_l_ha,
                        "total_liters": total_liters,
                        "target_area_ha": farm_size_ha,
                        "tank_usable_l": tank_usable_l,
                        "swath_width_m": swath_width_m,
                        "planned_speed_mps": planned_speed_mps,
                        "sorties": estimated_trips,
                        "estimated_trips": estimated_trips,
                        "estimated_flight_time_min": round(estimated_flight_time_min, 1),
                        "estimated_total_time_min": round(estimated_total_time_min, 1),
                        "battery_cycles": battery_cycles,
                    }
                else:
                    s["resource_regressor"] = {
                        "flow_rate_l_ha": 0.0,
                        "total_liters": 0.0,
                        "target_area_ha": farm_size_ha,
                        "tank_usable_l": tank_usable_l,
                        "swath_width_m": 0.0,
                        "planned_speed_mps": 0.0,
                        "sorties": 0,
                        "estimated_trips": 0,
                        "estimated_flight_time_min": 0.0,
                        "estimated_total_time_min": 0.0,
                        "battery_cycles": 0,
                    }
        else:
            ts = str(row["timestamp_dt"].isoformat())
            loc = str(row["location_name"])
            weather_snapshot = {
                "location_name": loc,
                "temperature_2m": float(row.get("temperature_2m", 0)),
                "relative_humidity_2m": float(row.get("relative_humidity_2m", 0)),
                "precipitation_probability": float(row.get("precipitation_probability", 0)),
                "precipitation": float(row.get("precipitation", 0)),
                "wind_speed_10m": float(row.get("wind_speed_10m", 0)),
                "wind_gusts_10m": float(row.get("wind_gusts_10m", 0)),
            }
            if use_singular_schema:
                to_insert_rows.append({
                    "mission_id": None,
                    "weather_id": None,
                    "drone_id": drone_profile.get("drone_id") if drone_profile else None,
                    "rf_score_safety": float(s["champion_score"]),
                    "xgb_score_safety": float(s["challenger_score"]),
                    "flight_safety_score": float(s["flyability_score"]),
                    "crop_impact_score": float(s["crop_impact_score"]),
                    "spray_quality_score": float(s["spray_quality_score"]),
                    "system_decision": s["final_decision"],
                    "is_user_overridden": False,
                    "override_reason": None,
                    "xai_explanation": s["xai_alert"],
                    "location_name": loc,
                    "slot_timestamp": ts,
                    "weather_json": weather_snapshot,
                })
            else:
                to_insert_rows.append({
                    "timestamp": ts,
                    "weather_snapshot": weather_snapshot,
                    "champion_pred": s["champion_pred"],
                    "champion_conf": float(s["champion_conf"]),
                    "challenger_pred": s["challenger_pred"],
                    "challenger_conf": float(s["challenger_conf"]),
                    "final_decision": s["final_decision"],
                    "was_conflict": bool(s["was_conflict"]),
                    "was_human_overridden": False,
                    "champion_score": float(s["champion_score"]),
                    "challenger_score": float(s["challenger_score"]),
                    "flyability_score": float(s["flyability_score"]),
                    "is_safe_to_fly": bool(s["is_safe_to_fly"]),
                    "risk_level": s["risk_level"],
                    "xai_alert": s["xai_alert"],
                })

    if to_insert_rows:
        import math
        def clean_nans(obj):
            if isinstance(obj, dict):
                return {k: clean_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nans(v) for v in obj]
            elif isinstance(obj, float) and math.isnan(obj):
                return None
            return obj
            
        cleaned_insert_rows = clean_nans(to_insert_rows)
        try:
            table_name = "flight_decision_log" if use_singular_schema else "flight_decisions_log"
            client.table(table_name).insert(cleaned_insert_rows).execute()
        except Exception as e:
            print(f"Failed to insert logs: {e}")

    return slots

@app.get("/api/drones")
def get_drones() -> list[dict[str, Any]]:
    drones = db.get_all_drones()
    for d in drones:
        if "nozzle_technology" in d:
            d["spray_system_type"] = d.pop("nozzle_technology")
        if "ingress_protection" in d:
            d["ip_rating"] = d.pop("ingress_protection")
    return drones

class DronePayload(BaseModel):
    model_name: str
    max_wind_resistance_kph: float | None = None
    max_gust_resistance_kph: float | None = None
    tank_capacity_liters: float | None = None
    spray_system_type: str | None = None
    ip_rating: str | None = None
    mtow_kg: float | None = None
    notes: str | None = None
    image_url: str | None = None

def _prepare_drone_payload(payload: DronePayload) -> dict[str, Any]:
    data = payload.model_dump(exclude_unset=True)
    
    # Map FE keys to DB columns
    if "spray_system_type" in data:
        data["nozzle_technology"] = data.pop("spray_system_type")
    if "ip_rating" in data:
        data["ingress_protection"] = data.pop("ip_rating")
        
    # Type conversions for Supabase strict schema
    if "tank_capacity_liters" in data and data["tank_capacity_liters"] is not None:
        data["tank_capacity_liters"] = int(float(data["tank_capacity_liters"]))
    if "max_wind_resistance_kph" in data and data["max_wind_resistance_kph"] is not None:
        data["max_wind_resistance_kph"] = float(data["max_wind_resistance_kph"])
    if "max_gust_resistance_kph" in data and data["max_gust_resistance_kph"] is not None:
        data["max_gust_resistance_kph"] = float(data["max_gust_resistance_kph"])
        
    return data

@app.post("/api/drones")
def add_drone(payload: DronePayload) -> dict[str, Any]:
    if not payload.model_name.strip():
        raise HTTPException(status_code=400, detail="Tên Drone (model_name) không được để trống!")
        
    existing_drone = db.get_drone_profile(payload.model_name)
    if existing_drone:
        raise HTTPException(status_code=400, detail=f"Tên Drone '{payload.model_name}' đã tồn tại trong hệ thống!")

    if (payload.max_wind_resistance_kph is not None and payload.max_wind_resistance_kph < 0) or \
       (payload.max_gust_resistance_kph is not None and payload.max_gust_resistance_kph < 0):
        raise HTTPException(status_code=400, detail="Ngưỡng gió không được là số âm")
    if payload.max_wind_resistance_kph is not None and payload.max_gust_resistance_kph is not None:
        if payload.max_gust_resistance_kph < payload.max_wind_resistance_kph:
            raise HTTPException(status_code=400, detail="Gió giật không được nhỏ hơn sức cản gió thường")
        
    data = _prepare_drone_payload(payload)
    
    res = db.add_drone(data)
    if not res:
        raise HTTPException(status_code=400, detail="Lỗi khi thêm Drone. Có thể Tên Drone đã tồn tại trong hệ thống!")
    if "nozzle_technology" in res:
        res["spray_system_type"] = res.pop("nozzle_technology")
    if "ingress_protection" in res:
        res["ip_rating"] = res.pop("ingress_protection")
    return res

@app.put("/api/drones/{drone_id}")
def update_drone(drone_id: int, payload: DronePayload) -> dict[str, Any]:
    if not payload.model_name.strip():
        raise HTTPException(status_code=400, detail="Tên Drone (model_name) không được để trống!")
        
    existing_drone = db.get_drone_profile(payload.model_name)
    if existing_drone and existing_drone.get("drone_id") != drone_id:
        raise HTTPException(status_code=400, detail=f"Tên Drone '{payload.model_name}' đã tồn tại trong hệ thống!")

    if (payload.max_wind_resistance_kph is not None and payload.max_wind_resistance_kph < 0) or \
       (payload.max_gust_resistance_kph is not None and payload.max_gust_resistance_kph < 0):
        raise HTTPException(status_code=400, detail="Ngưỡng gió không được là số âm")
    if payload.max_wind_resistance_kph is not None and payload.max_gust_resistance_kph is not None:
        if payload.max_gust_resistance_kph < payload.max_wind_resistance_kph:
            raise HTTPException(status_code=400, detail="Gió giật không được nhỏ hơn sức cản gió thường")
        
    data = _prepare_drone_payload(payload)
        
    res = db.update_drone(drone_id, data)
    if not res:
        raise HTTPException(status_code=400, detail="Lỗi khi cập nhật Drone. Vui lòng kiểm tra lại!")
    if "nozzle_technology" in res:
        res["spray_system_type"] = res.pop("nozzle_technology")
    if "ingress_protection" in res:
        res["ip_rating"] = res.pop("ingress_protection")
    return res

@app.delete("/api/drones/{drone_id}")
def delete_drone(drone_id: int) -> dict[str, Any]:
    if not db.delete_drone(drone_id):
        raise HTTPException(status_code=500, detail="Failed to delete drone.")
    return {"status": "success"}

@app.get("/api/plots")
def list_plots() -> list[dict[str, Any]]:
    from app.api import plot_store
    return plot_store.list_plots()

@app.post("/api/plots")
def create_plot(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    from app.api import plot_store
    try:
        return plot_store.add_plot(payload)
    except ValueError as e:
        raise HTTPException(422, str(e))

@app.put("/api/plots/{plot_id}")
def edit_plot(plot_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    from app.api import plot_store
    try:
        return plot_store.update_plot(plot_id, payload)
    except KeyError:
        raise HTTPException(404, f"Khong tim thay vuon id {plot_id}")

@app.delete("/api/plots/{plot_id}")
def remove_plot(plot_id: int) -> dict[str, str]:
    from app.api import plot_store
    plot_store.delete_plot(plot_id)
    return {"status": "deleted", "plot_id": str(plot_id)}

@app.get("/api/dashboard/slots")
def get_dashboard_slots(
    location: str = "Dong Thap",
    at: str | None = None,
    farm_size_ha: float = 10.0,
    distance_km: float = 1.0,
    drone_model: str | None = None,
    pesticide: str | None = None,
    crop_stage: str | None = None,
) -> dict[str, Any]:
    source_path = latest_clean_dataset()
    config = read_decision_config()
    thresholds, unsafe_weather_codes = config_to_engine_args(config)
    df = pd.read_csv(source_path)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    
    location_df = df[df["location_name"] == location].copy()
    if location_df.empty:
        available = sorted(df["location_name"].unique().tolist())
        raise HTTPException(
            status_code=404,
            detail=f"Unknown location '{location}'. Available locations: {available}",
        )
        
    reference_time = parse_reference_time(at)
    current = pick_operational_slot(location_df, reference_time)
    selected_date = current["timestamp_dt"].date()
    daily_df = location_df[location_df["timestamp_dt"].dt.date == selected_date].copy()
    daily_df = daily_df.sort_values("timestamp_dt")

    # Fetch DB configuration profiles
    drone_prof = db.get_drone_profile(drone_model) if drone_model else None
    if not drone_prof:
        drone_prof = db.get_drone_profile("DJI_T30") or {
            "max_wind_resistance_kph": 28.8,
            "max_gust_resistance_kph": 28.8,
            "tank_capacity_liters": 30,
            "nozzle_technology": "PRESSURE",
            "ingress_protection": "IP67"
        }
    
    pest_spec = db.get_pesticide_spec(pesticide) if pesticide else None
    if not pest_spec:
        pest_spec = {
            "trade_name": pesticide or "Tricyclazole",
            "active_ingredient": pesticide or "Tricyclazole",
            "action_mechanism": "SYSTEMIC" if (pesticide in ["Tricyclazole", "Hexaconazole"]) else "CONTACT",
            "common_formulation": "SC" if pesticide == "Hexaconazole" else ("EC" if pesticide == "Abamectin" else "WP"),
            "rain_washout_hours": 2,
            "uv_sensitivity": True if pesticide == "Abamectin" else False
        }
        
    crop_prof = db.get_crop_profile(crop_stage) if crop_stage else None
    if not crop_prof:
        crop_prof = {
            "stage_code": crop_stage or "TILLERING",
            "kc_value": 1.15 if crop_stage == "BOOTING" else (1.05 if crop_stage == "GRAIN_FILLING" else 0.9),
            "opt_flight_alt_min": 1.5 if crop_stage == "BOOTING" else (2.5 if crop_stage in ["SEEDLING", "GRAIN_FILLING"] else 2.0),
            "opt_flight_alt_max": 2.0 if crop_stage == "BOOTING" else (3.0 if crop_stage in ["SEEDLING", "GRAIN_FILLING"] else 2.5),
            "opt_flight_speed_min": 4.0 if crop_stage in ["BOOTING", "GRAIN_FILLING"] else (6.0 if crop_stage == "SEEDLING" else 5.0),
            "opt_flight_speed_max": 5.0 if crop_stage in ["BOOTING", "GRAIN_FILLING"] else (7.0 if crop_stage == "SEEDLING" else 6.0),
            "flow_rate_min_l_ha": 25.0 if crop_stage == "BOOTING" else 15.0,
            "flow_rate_max_l_ha": 30.0 if crop_stage == "BOOTING" else 20.0,
            "awd_threshold_cm": -15.0,
            "hard_ban_start_hour": 8 if crop_stage == "BOOTING" else None,
            "hard_ban_end_hour": 11 if crop_stage == "BOOTING" else None
        }

    # Fetch Soil Readings
    current_water_level = -12.0
    current_soil_moisture = 65.0
    try:
        plot = db.get_plot_by_name(location)
        if plot:
            soil_reading = db.get_latest_soil_reading(plot["plot_id"])
            if soil_reading:
                if soil_reading.get("water_level_cm") is not None:
                    current_water_level = float(soil_reading["water_level_cm"])
                # We will still load the mock, but overwrite it with real data below if available
                if soil_reading.get("soil_moisture_percentage") is not None:
                    current_soil_moisture = float(soil_reading["soil_moisture_percentage"])
    except Exception as e:
        print(f"Error fetching soil readings: {e}")
        
    # Phương án 2: Lấy dữ liệu độ ẩm đất thực tế từ Open-Meteo API
    try:
        from app.ingestion.open_meteo import fetch_forecast
        lat = location_df["latitude"].iloc[0]
        lon = location_df["longitude"].iloc[0]
        forecast_df = fetch_forecast(lat=lat, lon=lon, days=1)
        if not forecast_df.empty and "soil_moisture_0_to_7cm" in forecast_df.columns:
            sm_val = forecast_df["soil_moisture_0_to_7cm"].iloc[0]
            if pd.notna(sm_val):
                current_soil_moisture = round(float(sm_val) * 100, 1)
    except Exception as e:
        print(f"Lỗi khi lấy độ ẩm đất từ Open-Meteo: {e}")
    
    slots = run_3_layer_decision_engine(
        daily_df, thresholds, unsafe_weather_codes, farm_size_ha, distance_km,
        drone_profile=drone_prof, crop_stage=crop_prof, pesticide=pest_spec,
        current_water_level=current_water_level
    )
    
    try:
        slots = _reconcile_and_save_decisions_log(
            slots, location, farm_size_ha, thresholds, distance_km,
            drone_profile=drone_prof, crop_stage=crop_prof, pesticide=pest_spec,
            current_water_level=current_water_level
        )
    except Exception as e:
        print(f"Reconciliation and auto-save failed: {e}")
        
    formatted_slots = []
    for s in slots:
        row = s["row"]
        formatted_slots.append({
            "id": s.get("id"),
            "was_human_overridden": s.get("was_human_overridden", False),
            "user_notes": s.get("user_notes", ""),
            "timestamp": row["timestamp_dt"].isoformat(),
            "weather": {
                "temperature": as_number(row["temperature_2m"]),
                "humidity": as_number(row["relative_humidity_2m"], 0),
                "precipitation": as_number(row["precipitation"]),
                "precipitation_probability": as_number(row["precipitation_probability"], 0),
                "wind_speed": as_number(row["wind_speed_10m"]),
                "wind_gust": as_number(row["wind_gusts_10m"]),
                "cloud_cover": as_number(row["cloud_cover"], 0),
                "visibility": as_number(row["visibility"], 0),
                "weather_code": int(row["weather_code"]),
                "weather_description": str(row.get("weather_description", "")),
                "evapotranspiration": as_number(row.get("evapotranspiration", 0.0), 2),
                "soil_moisture": current_soil_moisture,
                "water_level_cm": current_water_level,
            },
            "decision_engine": {
                "champion_score": as_number(s["champion_score"], 2),
                "challenger_score": as_number(s["challenger_score"], 2),
                "was_conflict": s["was_conflict"],
                "flyability_score": as_number(s["flyability_score"], 3),
                "is_safe_to_fly": s["is_safe_to_fly"],
                "system_decision": s["final_decision"],
                "original_ai_decision": s.get("original_ai_decision", s["final_decision"]),
                "xai_alert": s["xai_alert"],
                "crop_impact_score": as_number(s.get("crop_impact_score", 100), 2),
                "spray_quality_score": as_number(s.get("spray_quality_score", 100), 2),
                "awd_recommendation": s.get("awd_recommendation", {
                    "action": "KEEP_DRYING",
                    "explanation": "Duy trì phơi ruộng."
                }),
                "factors": s.get("factors", []),
                "opt_flight_config": {
                    "alt_min": float(crop_prof.get("opt_flight_alt_min", 2.0)),
                    "alt_max": float(crop_prof.get("opt_flight_alt_max", 2.5)),
                    "speed_min": float(crop_prof.get("opt_flight_speed_min", 5.0)),
                    "speed_max": float(crop_prof.get("opt_flight_speed_max", 6.0)),
                    "nozzle_tech": str(drone_prof.get("nozzle_technology", "PRESSURE")),
                    "awd_threshold_cm": float(crop_prof.get("awd_threshold_cm", -15.0)),
                },
                "resource_regressor": s.get("resource_regressor", {})
            }
        })
        
    import math
    def clean_nans(obj):
        if isinstance(obj, dict):
            return {k: clean_nans(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nans(v) for v in obj]
        elif isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    return clean_nans({
        "location": location,
        "date": str(selected_date),
        "source": source_path.name,
        "slots": formatted_slots
    })

@app.post("/api/chat/ask")
def chat_ask(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    question = payload.get("question") or payload.get("message", "")
    if not question:
        raise HTTPException(status_code=422, detail="Question/message field is required.")
        
    try:
        client = db.get_client()
        res = client.table("flight_decisions_log").select("*").order("timestamp", desc=True).limit(100).execute()
        rows = res.data or []
    except Exception as e:
        print(f"Error querying flight_decisions_log for chat: {e}")
        rows = []
        
    q_lower = question.lower()
    location = None
    if "đồng tháp" in q_lower or "dong thap" in q_lower:
        location = "Dong Thap"
    elif "cần thơ" in q_lower or "can tho" in q_lower:
        location = "Can Tho"
    elif "an giang" in q_lower:
        location = "An Giang"
    elif "long an" in q_lower:
        location = "Long An"
    elif "tiền giang" in q_lower or "tien giang" in q_lower:
        location = "Tien Giang"
        
    matched_rows = []
    for r in rows:
        snapshot = r.get("weather_snapshot", {})
        row_loc = snapshot.get("location_name")
        if location and row_loc != location:
            continue
        matched_rows.append(r)
        
    matched_rows = sorted(matched_rows, key=lambda x: x["timestamp"], reverse=True)
    
    if not matched_rows:
        if location:
            answer = f"Tôi không tìm thấy lịch sử quyết định bay nào được ghi nhận cho địa điểm {location} trong cơ sở dữ liệu."
        else:
            answer = "Tôi không tìm thấy lịch sử quyết định bay nào trong cơ sở dữ liệu hiện tại."
    else:
        latest = matched_rows[0]
        ts_str = latest["timestamp"]
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            dt_vietnam = dt.astimezone(VIETNAM_TZ)
            time_formatted = dt_vietnam.strftime("%H:%M ngày %d/%m/%Y")
        except Exception:
            time_formatted = ts_str
            
        snap = latest.get("weather_snapshot", {})
        loc_name = snap.get("location_name", "không rõ địa điểm")
        
        is_safe = latest.get("is_safe_to_fly", latest.get("final_decision") == "TAKE_OFF")
        fly_score = latest.get("flyability_score", 1.0 if is_safe else 0.5) * 100.0
        champ_score = latest.get("champion_score", latest.get("champion_conf", 0.0)) * 100.0
        chall_score = latest.get("challenger_score", latest.get("challenger_conf", 0.0)) * 100.0
        was_conflict = latest.get("was_conflict", False)
        
        temp = snap.get("temperature_2m", 0.0)
        humidity = snap.get("relative_humidity_2m", 0.0)
        wind = snap.get("wind_speed_10m", 0.0)
        gust = snap.get("wind_gusts_10m", 0.0)
        rain = snap.get("precipitation", 0.0)
        rain_prob = snap.get("precipitation_probability", 0.0)
        
        answer = f"Tại **{loc_name}** vào lúc **{time_formatted}**:\n"
        answer += f"- **Khả năng cất cánh**: `{fly_score:.1f}%` ({'Đủ điều kiện an toàn bay' if is_safe else 'Không đủ điều kiện an toàn bay'})\n"
        
        if was_conflict:
            answer += f"- **Trạng thái xung đột**: Có sự lệch pha giữa các mô hình. Random Forest dự báo khả năng bay là {champ_score:.1f}%, trong khi XGBoost dự báo {chall_score:.1f}%.\n"
        else:
            answer += f"- **Dự báo của AI**: Cả hai mô hình đồng thuận cao. Random Forest dự báo khả năng bay là {champ_score:.1f}%, XGBoost dự báo {chall_score:.1f}%.\n"
            
        answer += f"- **Thông số thời tiết**: Nhiệt độ {temp:.1f}°C, Độ ẩm {humidity:.0f}%, Tốc độ gió {wind:.1f} km/h (gió giật {gust:.1f} km/h), Lượng mưa {rain:.1f} mm/h (Xác suất mưa {rain_prob:.0f}%).\n"
        
        if is_safe:
            answer += "- **Chi tiết**: Các chỉ số khí tượng cho thấy mức độ an toàn cao. Drone có thể cất cánh hoạt động."
        else:
            answer += f"- **Chi tiết**: Quyết định hạn chế bay được đưa ra do khả năng cất cánh chỉ đạt {fly_score:.1f}% (dưới ngưỡng an toàn 80%). Phân tích từ mô hình ghi nhận các yếu tố ảnh hưởng bất lợi: "
            reasons = []
            if wind > 28.8:
                reasons.append(f"tốc độ gió cao {wind:.1f} km/h")
            if gust > 28.8:
                reasons.append(f"gió giật mạnh {gust:.1f} km/h")
            if rain > 2.0:
                reasons.append(f"có mưa {rain:.1f} mm/h")
            if temp > 35.0:
                reasons.append(f"nhiệt độ nóng {temp:.1f}°C")
            if not reasons:
                reasons.append("điều kiện vi khí hậu không thuận lợi")
            answer += ", ".join(reasons) + "."
            
        if len(matched_rows) > 1:
            answer += f"\n\n*(Lưu ý: Tôi cũng tìm thấy {len(matched_rows) - 1} quyết định bay khác trong lịch sử của {loc_name if location else 'các địa điểm'}.)*"
            
    return {
        "answer": answer,
        "retrieved_logs_count": len(matched_rows)
    }


# ══════════════════════════════════════════════════════════════
# FLIGHT LOG & WEATHER HISTORY ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/api/flight-logs")
def get_flight_logs(
    location: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Query drone flight history."""
    logs = db.get_flight_history(
        location=location,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    return {"logs": logs, "total": len(logs)}


@app.get("/api/flight-logs/stats")
def get_flight_log_stats(
    location: str | None = None,
) -> dict[str, Any]:
    """Aggregate drone activity statistics."""
    return db.get_flight_stats(location=location)


@app.get("/api/weather-history")
def get_weather_history(
    location: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Query analyzed weather records from the database."""
    records = db.get_weather_history(
        location=location,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    return {"records": records, "total": len(records)}


@app.post("/api/decisions/{id}/override")
def override_decision(
    id: str,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    override_decision = payload.get("override_decision")
    user_notes = payload.get("user_notes", "")
    farm_size_ha = float(payload.get("farm_size_ha", 10.0))
    distance_to_field_km = float(payload.get("distance_to_field_km", payload.get("distance_km", 1.0)))
    was_human_overridden = payload.get("was_human_overridden", True)
    
    if not override_decision:
        raise HTTPException(status_code=422, detail="Missing required field: override_decision")
        
    valid_decisions = {"TAKE_OFF", "DELAY_FLIGHT", "LOCK_SPRAY", "RETURN_TO_CHARGING"}
    if override_decision not in valid_decisions:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid override_decision. Must be one of: {list(valid_decisions)}",
        )
        
    try:
        client = db.get_client()
        res = client.table("flight_decisions_log").select("*").eq("id", id).execute()
        records = res.data or []
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database lookup failed: {e}",
        )
        
    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"Flight decision log with ID '{id}' not found.",
        )
        
    record = records[0]
    
    if record.get("was_human_overridden") and record.get("final_decision") == override_decision:
        raise HTTPException(status_code=409, detail=f"Đã ghi đè trạng thái '{override_decision}' rồi, không thể ghi đè lặp lại.")

    weather_snapshot = record.get("weather_snapshot", {})
    if was_human_overridden:
        if user_notes:
            weather_snapshot["user_override_notes"] = user_notes
    else:
        weather_snapshot.pop("user_override_notes", None)
        
    is_safe = (override_decision == "TAKE_OFF")
    update_data = {
        "final_decision": override_decision,
        "was_human_overridden": was_human_overridden,
        "weather_snapshot": weather_snapshot,
        "is_safe_to_fly": is_safe,
        "flyability_score": 1.0 if is_safe else 0.0,
        "distance_to_field_km": distance_to_field_km,
    }
    
    try:
        res_update = client.table("flight_decisions_log").update(update_data).eq("id", id).execute()
        updated_records = res_update.data or []
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database update failed: {e}",
        )
        
    if not updated_records:
        raise HTTPException(
            status_code=500,
            detail="Failed to update record in the database.",
        )
        
    updated_record = updated_records[0]
    TRAVEL_COST_FACTOR = 0.1
    
    if override_decision == "TAKE_OFF":
        temp = float(weather_snapshot.get("temperature_2m", 25.0))
        humidity = float(weather_snapshot.get("relative_humidity_2m", 70.0))
        
        if temp >= 36.0 and humidity < 40.0:
            flow_rate_l_ha = 15.0
        else:
            row_series = pd.Series(weather_snapshot)
            config = read_decision_config()
            thresholds, _ = config_to_engine_args(config)
            flow_pct = calculate_dynamic_flow_rate(row_series, thresholds)
            flow_rate_l_ha = round(flow_pct / 10.0, 2)
            
        total_liters = round(flow_rate_l_ha * farm_size_ha, 2)
        import math
        sorties = math.ceil(total_liters / 30.0)
        battery_cycles = sorties + math.ceil(distance_to_field_km * 2 * TRAVEL_COST_FACTOR)
        
        resource_regressor = {
            "flow_rate_l_ha": flow_rate_l_ha,
            "total_liters": total_liters,
            "distance_to_field_km": distance_to_field_km,
            "battery_cycles_needed": battery_cycles,
        }
    else:
        resource_regressor = {
            "flow_rate_l_ha": 0.0,
            "total_liters": 0.0,
            "distance_to_field_km": distance_to_field_km,
            "battery_cycles_needed": 0,
        }
        
    return {
        "status": "ok",
        "id": id,
        "final_decision": updated_record["final_decision"],
        "was_human_overridden": updated_record["was_human_overridden"],
        "is_safe_to_fly": updated_record.get("is_safe_to_fly", is_safe),
        "flyability_score": updated_record.get("flyability_score", 1.0 if is_safe else 0.0),
        "resource_regressor": resource_regressor,
    }


@app.post("/api/decision/override")
def post_decision_override(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    import math
    record_id = payload.get("id")
    reason_str = payload.get("reason", "")
    weather = payload.get("weather", {})
    timestamp = weather.get("timestamp")
    location = payload.get("location") or weather.get("location_name") or "Dong Thap"
    drone_model = payload.get("drone_model", "DJI_T30")
    pesticide = payload.get("pesticide", "Tricyclazole")
    crop_stage = payload.get("crop_stage", "TILLERING")

    # Split reason into decision and notes
    decision_part = "TAKE_OFF"
    notes_part = ""
    if ":" in reason_str:
        parts = reason_str.split(":", 1)
        decision_part = parts[0].strip()
        notes_part = parts[1].strip()
    else:
        decision_part = reason_str.strip()

    valid_decisions = {"TAKE_OFF", "DELAY_FLIGHT", "LOCK_SPRAY", "RETURN_TO_CHARGING", "FLY", "DELAY", "NO_FLY"}
    if decision_part not in valid_decisions:
        decision_part = "FLY"

    is_safe = (decision_part in {"TAKE_OFF", "FLY"})

    drone_prof = db.get_drone_profile(drone_model) if drone_model else None
    if not drone_prof:
        drone_prof = db.get_drone_profile("DJI_T30") or {
            "max_wind_resistance_kph": 28.8,
            "max_gust_resistance_kph": 28.8,
            "tank_capacity_liters": 30,
            "nozzle_technology": "PRESSURE",
            "ingress_protection": "IP67"
        }
    
    crop_prof = db.get_crop_profile(crop_stage) if crop_stage else None
    if not crop_prof:
        crop_prof = {
            "stage_code": crop_stage or "TILLERING",
            "flow_rate_min_l_ha": 15.0,
            "flow_rate_max_l_ha": 25.0
        }

    if is_safe:
        row_series = pd.Series(weather)
        config = read_decision_config()
        thresholds, _ = config_to_engine_args(config)
        flow_rate_l_ha = calculate_dynamic_flow_rate(row_series, thresholds, crop_prof)
        total_liters = round(flow_rate_l_ha * 10.0, 2)
        tank_cap = float(drone_prof.get("tank_capacity_liters", 30))
        sorties = math.ceil(total_liters / tank_cap)
        battery_cycles = sorties + 2
        resource_regressor = {
            "flow_rate_l_ha": flow_rate_l_ha,
            "total_liters": total_liters,
            "distance_to_field_km": 1.0,
            "battery_cycles_needed": battery_cycles,
        }
    else:
        resource_regressor = {
            "flow_rate_l_ha": 0.0,
            "total_liters": 0.0,
            "distance_to_field_km": 1.0,
            "battery_cycles_needed": 0,
        }

    client = db.get_client()

    try:
        _OLD_TO_NEW = {"TAKE_OFF": "FLY", "DELAY_FLIGHT": "DELAY", "LOCK_SPRAY": "NO_FLY", "RETURN_TO_CHARGING": "NO_FLY", "FLY": "FLY", "DELAY": "DELAY", "NO_FLY": "NO_FLY"}
        mapped_decision = _OLD_TO_NEW.get(decision_part, "FLY")
        update_data = {
            "system_decision": mapped_decision,
            "is_user_overridden": True,
            "override_reason": notes_part,
        }
        if record_id:
            res_select = client.table("flight_decision_log").select("is_user_overridden, system_decision").eq("log_id", record_id).execute()
            if res_select.data and len(res_select.data) > 0:
                record = res_select.data[0]
                if record.get("is_user_overridden") and record.get("system_decision") == mapped_decision:
                    raise HTTPException(status_code=409, detail=f"Đã ghi đè trạng thái '{mapped_decision}' rồi, không thể ghi đè lặp lại.")
            res_update = client.table("flight_decision_log").update(update_data).eq("log_id", record_id).execute()
        else:
            res_select = client.table("flight_decision_log").select("is_user_overridden, system_decision").eq("location_name", location).eq("slot_timestamp", timestamp).execute()
            if res_select.data and len(res_select.data) > 0:
                record = res_select.data[0]
                if record.get("is_user_overridden") and record.get("system_decision") == mapped_decision:
                    raise HTTPException(status_code=409, detail=f"Đã ghi đè trạng thái '{mapped_decision}' rồi, không thể ghi đè lặp lại.")
            res_update = client.table("flight_decision_log").update(update_data).eq("location_name", location).eq("slot_timestamp", timestamp).execute()
        
        if res_update.data:
            return {
                "status": "ok",
                "final_decision": decision_part,
                "was_human_overridden": True,
                "is_safe_to_fly": is_safe,
                "flyability_score": 1.0 if is_safe else 0.0,
                "resource_regressor": resource_regressor,
            }
    except Exception as e:
        print(f"Failed updating flight_decision_log (singular): {e}")

    try:
        if record_id:
            res_select = client.table("flight_decisions_log").select("*").eq("id", record_id).execute()
        else:
            res_select = client.table("flight_decisions_log").select("*").eq("timestamp", timestamp).execute()
        
        if res_select.data:
            rec = res_select.data[0]
            weather_snapshot = rec.get("weather_snapshot", {})
            weather_snapshot["user_override_notes"] = notes_part
            
            update_data = {
                "final_decision": decision_part,
                "was_human_overridden": True,
                "weather_snapshot": weather_snapshot,
                "is_safe_to_fly": is_safe,
                "flyability_score": 1.0 if is_safe else 0.0,
            }
            if record_id:
                res_update = client.table("flight_decisions_log").update(update_data).eq("id", record_id).execute()
            else:
                res_update = client.table("flight_decisions_log").update(update_data).eq("timestamp", timestamp).execute()
            
            if res_update.data:
                return {
                    "status": "ok",
                    "final_decision": decision_part,
                    "was_human_overridden": True,
                    "is_safe_to_fly": is_safe,
                    "flyability_score": 1.0 if is_safe else 0.0,
                    "resource_regressor": resource_regressor,
                }
    except Exception as ex:
        print(f"Failed updating flight_decisions_log (plural): {ex}")

    return {
        "status": "ok",
        "final_decision": decision_part,
        "was_human_overridden": True,
        "is_safe_to_fly": is_safe,
        "flyability_score": 1.0 if is_safe else 0.0,
        "resource_regressor": resource_regressor,
    }

