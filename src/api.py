"""
FastAPI surface for the agricultural UAV scheduling dashboard.

Run from the project root:
    .venv/bin/uvicorn src.api:app --reload --port 8000
"""
from __future__ import annotations

import json
import threading
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .decision_model.decision_engine import (
    DecisionThresholds,
    SCORE_WEIGHTS,
    UNSAFE_WEATHER_CODES,
    add_decision_columns,
    build_recommendation_text,
)
from .run_pipeline import run_weather_pipeline


ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_DIRS = [ROOT / "src" / "data" / "clean", ROOT / "data" / "clean"]
REPORT_DIR = ROOT / "reports"
CONFIG_DIR = ROOT / "config"
DECISION_CONFIG_PATH = CONFIG_DIR / "decision_config.json"
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
PIPELINE_LOCK = threading.Lock()

THRESHOLD_BOUNDS = {
    "max_wind_speed": (1.0, 80.0),
    "max_wind_gust": (1.0, 120.0),
    "max_rain_probability": (0.0, 100.0),
    "return_to_charging_rain_probability": (0.0, 100.0),
    "max_cloud_cover": (0.0, 100.0),
    "min_visibility": (0.0, 50_000.0),
    "max_safe_temperature": (0.0, 60.0),
}

app = FastAPI(
    title="Agricultural Drone Scheduler API",
    description="Dashboard API backed by the DSS301 weather pipeline and decision engine.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
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
    return {
        **active_config,
        "updated_at": datetime.fromtimestamp(
            DECISION_CONFIG_PATH.stat().st_mtime,
            VIETNAM_TZ,
        ).isoformat(timespec="minutes") if DECISION_CONFIG_PATH.exists() else None,
        "source": "file" if DECISION_CONFIG_PATH.exists() else "default",
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
