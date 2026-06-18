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

from .decision_model.decision_engine import (
    DecisionThresholds,
    SCORE_WEIGHTS,
    UNSAFE_WEATHER_CODES,
    add_decision_columns,
    build_recommendation_text,
)
from .run_pipeline import run_weather_pipeline
from . import database as db
from .weather_override import (
    OVERRIDE_IMAGE_DIR,
    available_conditions,
    classify_weather_image,
    apply_override_to_weather,
    recalculate_decision,
    save_override_image,
    WEATHER_CONDITIONS,
)


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
MODEL_PATH = ROOT / "models" / "drone_decision_model.joblib"
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
PIPELINE_LOCK = threading.Lock()
AI_TRAINING_LOCK = threading.Lock()

THRESHOLD_BOUNDS = {
    "max_wind_speed": (1.0, 80.0),
    "max_wind_gust": (1.0, 120.0),
    "max_rain_probability": (0.0, 100.0),
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


@app.post("/api/ai-training/simulate-images")
def simulate_ai_training_images(location: str | None = None) -> dict[str, Any]:
    def runner() -> None:
        from .data_pipeline.simulate_images import main as simulate_images_main

        return simulate_images_main()

    return run_ai_step("simulate_images", runner, location)


@app.post("/api/ai-training/extract-features")
def extract_ai_training_features(location: str | None = None) -> dict[str, Any]:
    def runner() -> None:
        from .data_pipeline.extract_features import main as extract_features_main

        return extract_features_main()

    return run_ai_step("extract_features", runner, location)


@app.post("/api/ai-training/train")
def train_ai_model(location: str | None = None) -> dict[str, Any]:
    def runner() -> dict[str, Any]:
        from .data_pipeline.merge_data import main as merge_data_main
        from .decision_model.train_decision_model import train_models

        merge_data_main()
        return train_models(FINAL_TRAINING_DATA_PATH)

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

    return {
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
    }


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
# WEATHER OVERRIDE ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/api/weather-override/conditions")
def get_override_conditions() -> list[dict[str, str]]:
    """List all selectable weather conditions for the override form."""
    return available_conditions()


@app.post("/api/weather-override/analyze")
async def analyze_weather_image(
    image: UploadFile = File(...),
    location: str = Form(...),
    timestamp: str = Form(...),
) -> dict[str, Any]:
    """
    Step 1 of override flow: upload image → AI classifies → return suggestion.

    The frontend should show the AI suggestion and let the user confirm or change.
    """
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="File phai la hinh anh (jpg/png).")

    contents = await image.read()
    if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=422, detail="Hinh anh qua lon (toi da 10MB).")

    ext = Path(image.filename or "photo.jpg").suffix.lower() or ".jpg"
    filename, full_path = save_override_image(contents, ext)

    # AI classification
    ai_result = classify_weather_image(str(full_path))

    # Upload to Supabase Storage
    try:
        public_url = db.upload_image_to_storage(full_path, filename)
    except Exception as exc:
        print(f"Loi upload ảnh len Supabase Storage: {exc}")
        public_url = f"/api/weather-override/image/{filename}"

    # Load the original API weather for this slot
    original_weather = _get_original_weather_for_slot(location, timestamp)

    return {
        "override_image": {
            "filename": filename,
            "url": public_url,
        },
        "ai_suggestion": {
            "condition": ai_result["suggested_condition"],
            "confidence": ai_result["confidence"],
            "all_scores": ai_result.get("all_scores", {}),
            "message": ai_result.get("message"),
        },
        "available_conditions": available_conditions(),
        "original_weather": original_weather,
        "location": location,
        "timestamp": timestamp,
    }


@app.post("/api/weather-override/confirm")
def confirm_weather_override(
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    """
    Step 2 of override flow: user confirms/changes the condition → system
    recalculates and saves everything to DB.

    Expected payload::

        {
            "location": "Dong Thap",
            "timestamp": "2026-06-16T10:00",
            "image_filename": "override_20260616_..._abc123.jpg",
            "ai_suggested_condition": "rainy",
            "ai_confidence": 0.87,
            "user_final_condition": "cloudy",   // user changed it
            "user_notes": "Troi co may nhung chua mua"  // optional
        }
    """
    location = payload.get("location")
    timestamp = payload.get("timestamp")
    image_filename = payload.get("image_filename", "")
    image_url = payload.get("image_url", "")
    ai_suggested = payload.get("ai_suggested_condition")
    ai_confidence = float(payload.get("ai_confidence", 0))
    user_condition = payload.get("user_final_condition")
    user_notes = payload.get("user_notes", "")

    if not location or not timestamp or not user_condition:
        raise HTTPException(
            status_code=422,
            detail="Thieu truong bat buoc: location, timestamp, user_final_condition.",
        )
    if user_condition not in WEATHER_CONDITIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Dieu kien khong hop le. Chon tu: {list(WEATHER_CONDITIONS.keys())}",
        )

    # Get original weather from API data
    original = _get_original_weather_for_slot(location, timestamp)

    # Calculate PREVIOUS decision (from API data)
    prev_decision = recalculate_decision(original)

    # Apply override and recalculate
    overridden = apply_override_to_weather(original, user_condition)
    new_decision = recalculate_decision(overridden)

    user_accepted_ai = (
        ai_suggested is not None and user_condition == ai_suggested
    )

    condition_info = WEATHER_CONDITIONS[user_condition]

    # Save override to DB
    override_record = db.create_weather_override({
        "location_name": location,
        "override_timestamp": timestamp,
        "image_filename": image_filename,
        "image_url": image_url or (f"/api/weather-override/image/{image_filename}" if image_filename else None),
        "ai_suggested_condition": ai_suggested,
        "ai_confidence": ai_confidence,
        "user_final_condition": user_condition,
        "user_accepted_ai": user_accepted_ai,
        "user_notes": user_notes or None,
        "original_api_weather_code": int(original.get("weather_code", 0)),
        "original_api_description": str(original.get("weather_description", "")),
        "override_weather_code": condition_info["weather_code"],
        "override_description": condition_info["weather_description"],
        "previous_decision": prev_decision["decision_action"],
        "previous_flyability": prev_decision["flyability_score"],
        "new_decision": new_decision["decision_action"],
        "new_flyability": new_decision["flyability_score"],
    })

    # Also save the updated flight log with override reference
    override_id = override_record.get("id")
    db.save_flight_log({
        "location_name": location,
        "flight_timestamp": timestamp,
        "decision_action": new_decision["decision_action"],
        "risk_level": new_decision["risk_level"],
        "flyability_score": new_decision["flyability_score"],
        "dynamic_flow_rate_pct": new_decision["dynamic_flow_rate_pct"],
        "crop_condition": new_decision["crop_condition"],
        "recommendation_text": new_decision["recommendation_text"],
        "weather_source": "user_override",
        "override_id": override_id,
    })

    decision_changed = prev_decision["decision_action"] != new_decision["decision_action"]

    return {
        "status": "ok",
        "override_id": override_id,
        "user_accepted_ai": user_accepted_ai,
        "decision_changed": decision_changed,
        "previous": {
            "weather_code": int(original.get("weather_code", 0)),
            "weather_description": str(original.get("weather_description", "")),
            "decision_action": prev_decision["decision_action"],
            "flyability_score": prev_decision["flyability_score"],
            "risk_level": prev_decision["risk_level"],
        },
        "new": {
            "weather_code": condition_info["weather_code"],
            "weather_description": condition_info["weather_description"],
            "decision_action": new_decision["decision_action"],
            "flyability_score": new_decision["flyability_score"],
            "risk_level": new_decision["risk_level"],
            "recommendation_text": new_decision["recommendation_text"],
        },
    }


def _get_original_weather_for_slot(
    location: str, timestamp: str,
) -> dict[str, Any]:
    """Look up the API weather data for a specific location + timestamp."""
    try:
        df, _, config, thresholds = load_forecast()
        loc_df = df[df["location_name"] == location].copy()
        if loc_df.empty:
            return _default_weather_params(location, timestamp)

        ref = parse_reference_time(timestamp)
        slot = pick_operational_slot(loc_df, ref)
        return {
            "location_name": location,
            "timestamp": timestamp,
            "temperature_2m": float(slot.get("temperature_2m", 30)),
            "relative_humidity_2m": float(slot.get("relative_humidity_2m", 70)),
            "precipitation_probability": float(slot.get("precipitation_probability", 0)),
            "precipitation": float(slot.get("precipitation", 0)),
            "cloud_cover": float(slot.get("cloud_cover", 50)),
            "visibility": float(slot.get("visibility", 10000)),
            "wind_speed_10m": float(slot.get("wind_speed_10m", 10)),
            "wind_gusts_10m": float(slot.get("wind_gusts_10m", 15)),
            "weather_code": int(slot.get("weather_code", 1000)),
            "weather_description": str(slot.get("weather_description", "")),
        }
    except Exception:
        return _default_weather_params(location, timestamp)


def _default_weather_params(location: str, timestamp: str) -> dict[str, Any]:
    """Fallback weather parameters when no forecast is available."""
    return {
        "location_name": location,
        "timestamp": timestamp,
        "temperature_2m": 30.0,
        "relative_humidity_2m": 70.0,
        "precipitation_probability": 0.0,
        "precipitation": 0.0,
        "cloud_cover": 50.0,
        "visibility": 10000.0,
        "wind_speed_10m": 10.0,
        "wind_gusts_10m": 15.0,
        "weather_code": 1000,
        "weather_description": "Clear",
    }


@app.get("/api/weather-overrides")
def list_weather_overrides(
    location: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List all weather overrides, newest first."""
    overrides = db.get_overrides(location=location, limit=limit)
    return {"overrides": overrides, "total": len(overrides)}


@app.get("/api/weather-override/image/{filename}")
def get_override_image(filename: str) -> FileResponse:
    """Serve a user-uploaded weather override image."""
    image_path = (OVERRIDE_IMG_DIR / filename).resolve()
    if (
        image_path.parent != OVERRIDE_IMG_DIR.resolve()
        or image_path.suffix.lower() not in IMAGE_SUFFIXES
    ):
        raise HTTPException(status_code=404, detail="Image not found.")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(image_path)


@app.get("/api/weather-override/stats")
def get_override_stats() -> dict[str, Any]:
    """AI suggestion accuracy statistics."""
    return db.get_override_accuracy_stats()


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

