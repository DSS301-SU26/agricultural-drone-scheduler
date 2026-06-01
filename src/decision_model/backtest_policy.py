"""
Backtest the DSS policy against a fixed-schedule baseline.

Baseline scenario: operator always schedules the UAV at 12:00.
DSS scenario: system selects the safest TAKE_OFF slot per location/date.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .decision_engine import (
    AGRICULTURAL_LOCATIONS,
    BASE_SPRAY_LITERS,
    add_decision_columns,
    build_recommendation_text,
    calculate_dynamic_flow_rate,
)
from .train_decision_model import DEFAULT_DATASET, REPORT_DIR


def waste_risk_liters(row: pd.Series) -> float:
    """Estimate liters exposed to waste if an operator flies this slot."""
    waste_rate = 0.0
    if float(row.get("temperature_2m", 0)) >= 35:
        waste_rate += 0.40
    if float(row.get("wind_speed_10m", 0)) > 15:
        waste_rate += 0.20
    if float(row.get("precipitation_probability", 0)) > 30:
        waste_rate += 0.25
    if float(row.get("precipitation", 0)) > 0:
        waste_rate += 0.35
    return round(BASE_SPRAY_LITERS * min(waste_rate, 1.0), 2)


def select_dss_slot(group: pd.DataFrame) -> pd.Series:
    safe = group[group["decision_action"] == "TAKE_OFF"].copy()
    candidates = safe if not safe.empty else group.copy()
    return candidates.sort_values(
        [
            "flyability_score",
            "wind_gusts_10m",
            "precipitation_probability",
            "temperature_2m",
        ],
        ascending=[False, True, True, True],
    ).iloc[0]


def run_backtest(dataset_path: Path) -> dict[str, object]:
    df = pd.read_csv(dataset_path).drop_duplicates(
        subset=["location_name", "timestamp"],
        keep="last",
    )
    df = df[df["location_name"].isin(AGRICULTURAL_LOCATIONS)].copy()
    df = add_decision_columns(df)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)

    rows = []
    for (location_name, date), group in df.groupby(["location_name", "date"]):
        noon_rows = group[pd.to_datetime(group["timestamp"]).dt.hour == 12]
        baseline = noon_rows.iloc[0] if not noon_rows.empty else group.iloc[len(group) // 2]
        dss = select_dss_slot(group)

        baseline_waste = waste_risk_liters(baseline)
        dss_waste = (
            waste_risk_liters(dss)
            if dss["decision_action"] == "TAKE_OFF"
            else 0.0
        )

        rows.append(
            {
                "location_name": location_name,
                "date": date,
                "baseline_timestamp": baseline["timestamp"],
                "baseline_action": baseline["decision_action"],
                "baseline_waste_liters": baseline_waste,
                "dss_timestamp": dss["timestamp"],
                "dss_action": dss["decision_action"],
                "dss_waste_liters": round(dss_waste, 2),
                "avoided_risky_operation": int(
                    baseline["decision_action"] != "TAKE_OFF"
                    and dss["decision_action"] == "TAKE_OFF"
                ),
                "recommendation_text": build_recommendation_text(dss, dss["decision_action"]),
            }
        )

    result_df = pd.DataFrame(rows)
    baseline_risks = int((result_df["baseline_action"] != "TAKE_OFF").sum())
    dss_risks = int((result_df["dss_action"] != "TAKE_OFF").sum())
    baseline_waste = float(result_df["baseline_waste_liters"].sum())
    dss_waste = float(result_df["dss_waste_liters"].sum())

    risk_reduction_pct = (
        ((baseline_risks - dss_risks) / baseline_risks) * 100 if baseline_risks else 0.0
    )
    waste_reduction_pct = (
        ((baseline_waste - dss_waste) / baseline_waste) * 100 if baseline_waste else 0.0
    )

    summary = {
        "evaluated_location_days": int(len(result_df)),
        "baseline_risky_operations": baseline_risks,
        "dss_risky_operations": dss_risks,
        "risk_reduction_pct": round(risk_reduction_pct, 2),
        "baseline_waste_liters": round(baseline_waste, 2),
        "dss_waste_liters": round(dss_waste, 2),
        "waste_reduction_pct": round(waste_reduction_pct, 2),
        "interpretation": (
            "Simulation upper bound only: fixed-noon baseline versus best available "
            "DSS slot. Do not present as measured field performance."
        ),
    }

    REPORT_DIR.mkdir(exist_ok=True)
    result_df.to_csv(REPORT_DIR / "backtesting_daily_results.csv", index=False)
    pd.DataFrame([summary]).to_csv(REPORT_DIR / "backtesting_summary.csv", index=False)
    (REPORT_DIR / "backtesting_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    print(json.dumps(run_backtest(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
