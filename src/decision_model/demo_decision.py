"""
Run a small DSS demo from the trained model.

Usage:
    .venv/bin/python -m src.decision_model.demo_decision --location "Dong Thap"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .decision_engine import add_decision_columns, build_recommendation_text, recommend_best_slot
from .train_decision_model import DEFAULT_DATASET, MODEL_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--location", default=None)
    args = parser.parse_args()

    payload = joblib.load(MODEL_DIR / "drone_decision_model.joblib")
    df = add_decision_columns(pd.read_csv(args.dataset))
    if args.location:
        df = df[df["location_name"] == args.location]

    slot = recommend_best_slot(df)
    if slot is None:
        print("Khong co khung gio TAKE_OFF an toan trong dataset hien tai.")
        return

    feature_cols = payload["feature_columns"]
    action = payload["pipeline"].predict(slot[feature_cols].to_frame().T)[0]
    print(f"Best slot       : {slot['location_name']} - {slot['timestamp']}")
    print(f"Model           : {payload['model_name']}")
    print(f"Predicted action: {action}")
    print(f"Recommendation : {build_recommendation_text(slot, action)}")


if __name__ == "__main__":
    main()

