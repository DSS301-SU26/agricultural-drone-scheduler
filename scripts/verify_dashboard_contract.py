"""Verify the dashboard API against the cleaned forecast and decision engine."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import (  # noqa: E402
    config_to_engine_args,
    dashboard,
    latest_clean_dataset,
    locations,
    read_decision_config,
    serialize_slot,
)
from src.decision_model.decision_engine import add_decision_columns  # noqa: E402


def verify_dashboard_contract() -> None:
    source = latest_clean_dataset()
    thresholds, unsafe_weather_codes = config_to_engine_args(read_decision_config())
    enriched = add_decision_columns(pd.read_csv(source), thresholds, unsafe_weather_codes)
    checked_rows = 0

    for location in locations():
        location_df = enriched[enriched["location_name"] == location["name"]].copy()
        location_df["timestamp_dt"] = pd.to_datetime(location_df["timestamp"])
        location_df = location_df.sort_values("timestamp_dt")

        for _, daily_df in location_df.groupby(location_df["timestamp_dt"].dt.date):
            first_slot = daily_df.iloc[0]
            payload = dashboard(
                location=location["name"],
                at=first_slot["timestamp_dt"].isoformat(),
            )
            expected = [serialize_slot(row, thresholds) for _, row in daily_df.iterrows()]
            actual = payload["timeline_tiles"]
            assert actual == expected, (
                f"Timeline mismatch for {location['name']} "
                f"on {first_slot['timestamp_dt'].date()}"
            )
            assert all(
                slot["schedule_eligible"] == (slot["decision_action"] == "TAKE_OFF")
                for slot in actual
            ), f"Eligibility mismatch for {location['name']}"
            checked_rows += len(actual)

    report = json.loads((ROOT / "reports" / "backtesting_summary.json").read_text())
    hcm_df = enriched[enriched["location_name"] == "Ho Chi Minh"].copy()
    hcm_safe_slot = hcm_df[hcm_df["decision_action"] == "TAKE_OFF"].iloc[0]
    hcm_rain_slot = hcm_df[hcm_df["decision_action"] == "RETURN_TO_CHARGING"].iloc[0]
    hcm_safe = dashboard(location="Ho Chi Minh", at=pd.to_datetime(hcm_safe_slot["timestamp"]).isoformat())
    hcm_rain = dashboard(location="Ho Chi Minh", at=pd.to_datetime(hcm_rain_slot["timestamp"]).isoformat())

    assert hcm_safe["current"]["decision_action"] == "TAKE_OFF"
    assert hcm_rain["current"]["decision_action"] == "RETURN_TO_CHARGING"
    assert hcm_safe["kpis"][0]["value"] == report["risk_reduction_pct"]
    assert hcm_safe["kpis"][1]["value"] == report["waste_reduction_pct"]

    print("Dashboard contract verification passed")
    print(f"- Dataset: {source.name}")
    print(f"- Locations: {len(locations())}")
    print(f"- Forecast rows checked: {checked_rows}")
    print(f"- Actions: {dict(Counter(enriched['decision_action']))}")
    print(f"- Scenario TAKE_OFF: Ho Chi Minh, {hcm_safe_slot['timestamp']}")
    print(f"- Scenario RETURN_TO_CHARGING: Ho Chi Minh, {hcm_rain_slot['timestamp']}")


if __name__ == "__main__":
    verify_dashboard_contract()
