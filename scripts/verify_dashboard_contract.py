"""Verify the dashboard API against the cleaned forecast and decision engine."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import dashboard, latest_clean_dataset, locations, serialize_slot  # noqa: E402
from src.decision_model.decision_engine import add_decision_columns  # noqa: E402


def verify_dashboard_contract() -> None:
    source = latest_clean_dataset()
    enriched = add_decision_columns(pd.read_csv(source))
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
            expected = [serialize_slot(row) for _, row in daily_df.iterrows()]
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
    hcm_safe = dashboard(location="Ho Chi Minh", at="2026-06-02T05:30:00")
    hcm_rain = dashboard(location="Ho Chi Minh", at="2026-06-02T11:00:00")

    assert hcm_safe["current"]["decision_action"] == "TAKE_OFF"
    assert hcm_rain["current"]["decision_action"] == "RETURN_TO_CHARGING"
    assert hcm_safe["kpis"][0]["value"] == report["risk_reduction_pct"]
    assert hcm_safe["kpis"][1]["value"] == report["waste_reduction_pct"]

    print("Dashboard contract verification passed")
    print(f"- Dataset: {source.name}")
    print(f"- Locations: {len(locations())}")
    print(f"- Forecast rows checked: {checked_rows}")
    print(f"- Actions: {dict(Counter(enriched['decision_action']))}")
    print("- Scenario TAKE_OFF: Ho Chi Minh, 2026-06-02 06:00")
    print("- Scenario RETURN_TO_CHARGING: Ho Chi Minh, 2026-06-02 11:00")


if __name__ == "__main__":
    verify_dashboard_contract()
