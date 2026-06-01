"""
Demo a real-time-like DSS recommendation for presentation.

Examples:
    .venv/bin/python -m src.decision_model.live_demo --location "Can Tho"
    .venv/bin/python -m src.decision_model.live_demo --location "Can Tho" --scenario lock_spray
    .venv/bin/python -m src.decision_model.live_demo --location "Can Tho" --scenario return_to_charging
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from .decision_engine import (
    add_decision_columns,
    build_recommendation_text,
    calculate_dynamic_flow_rate,
    derive_decision_action,
)
from .train_decision_model import DEFAULT_DATASET, MODEL_DIR


SCENARIOS = {
    "take_off": {
        "temperature_2m": 30.0,
        "relative_humidity_2m": 72.0,
        "precipitation_probability": 10.0,
        "precipitation": 0.0,
        "wind_speed_10m": 8.0,
        "wind_gusts_10m": 14.0,
        "weather_code": 1,
        "cloud_cover": 35.0,
        "visibility": 20_000.0,
    },
    "delay_flight": {
        "temperature_2m": 37.0,
        "relative_humidity_2m": 58.0,
        "precipitation_probability": 45.0,
        "precipitation": 0.0,
        "wind_speed_10m": 11.0,
        "wind_gusts_10m": 17.0,
        "weather_code": 2,
        "cloud_cover": 60.0,
        "visibility": 18_000.0,
    },
    "lock_spray": {
        "temperature_2m": 32.0,
        "relative_humidity_2m": 68.0,
        "precipitation_probability": 20.0,
        "precipitation": 0.0,
        "wind_speed_10m": 26.0,
        "wind_gusts_10m": 36.0,
        "weather_code": 2,
        "cloud_cover": 45.0,
        "visibility": 15_000.0,
    },
    "return_to_charging": {
        "temperature_2m": 29.0,
        "relative_humidity_2m": 86.0,
        "precipitation_probability": 85.0,
        "precipitation": 2.4,
        "wind_speed_10m": 12.0,
        "wind_gusts_10m": 19.0,
        "weather_code": 63,
        "cloud_cover": 95.0,
        "visibility": 8_000.0,
    },
}


def pick_current_slot(df: pd.DataFrame, location: str | None) -> pd.Series:
    if location:
        df = df[df["location_name"] == location]
    if df.empty:
        raise ValueError("Khong tim thay location trong dataset.")

    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).replace(tzinfo=None)
    slots = df.copy()
    slots["timestamp_dt"] = pd.to_datetime(slots["timestamp"])
    future_slots = slots[slots["timestamp_dt"] >= now]
    if future_slots.empty:
        future_slots = slots

    return future_slots.assign(
        time_distance=(future_slots["timestamp_dt"] - now).abs()
    ).sort_values("time_distance").iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--location", default=None)
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default=None,
        help="Optional demo override to force a clear decision scenario.",
    )
    args = parser.parse_args()

    payload = joblib.load(MODEL_DIR / "drone_decision_model.joblib")
    run_time = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).replace(tzinfo=None)
    df = add_decision_columns(pd.read_csv(args.dataset))
    slot = pick_current_slot(df, args.location).copy()

    scenario_name = args.scenario or "dataset_nearest_time"
    if args.scenario:
        slot["timestamp"] = run_time.strftime("%Y-%m-%d %H:%M:%S")
        slot["hour"] = run_time.hour
        slot["dayofweek"] = run_time.weekday()
        slot["month"] = run_time.month
        for key, value in SCENARIOS[args.scenario].items():
            slot[key] = value

    slot["decision_action"] = derive_decision_action(slot)
    slot["dynamic_flow_rate_pct"] = calculate_dynamic_flow_rate(slot)

    feature_cols = payload["feature_columns"]
    model_action = payload["pipeline"].predict(slot[feature_cols].to_frame().T)[0]
    final_action = slot["decision_action"] if args.scenario else model_action
    safety_override = bool(args.scenario and model_action != final_action)

    print("\n=== DSS301 UAV Decision Demo ===")
    print(f"Location        : {slot['location_name']}")
    print(f"Scenario        : {scenario_name}")
    print(f"Run time        : {run_time:%Y-%m-%d %H:%M:%S}")
    print(f"Decision time   : {slot['timestamp']}")
    print(f"Temperature     : {float(slot['temperature_2m']):.1f} C")
    print(f"Wind / Gust     : {float(slot['wind_speed_10m']):.1f} / {float(slot['wind_gusts_10m']):.1f} km/h")
    print(f"Rain probability: {float(slot['precipitation_probability']):.0f}%")
    print(f"Precipitation   : {float(slot['precipitation']):.1f} mm")
    print(f"Flow-rate       : {float(slot['dynamic_flow_rate_pct']):.1f}%")
    print(f"DSS decision    : {final_action}")
    if args.scenario:
        print(f"ML suggestion   : {model_action} (reference only: original image embedding retained)")
        print(f"Safety override : {'APPLIED' if safety_override else 'not needed'}")
    print(f"Recommendation  : {build_recommendation_text(slot, final_action)}")


if __name__ == "__main__":
    main()
