"""
Chan doan ghi flight_decision_log (P1 #5).
Chay: .venv/bin/python -m app.db.diag_log
In ra loi THAT neu upsert that bai (thay vi nuot loi nhu ban best-effort).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    load_dotenv(ROOT / ".env")
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("[ERROR] Thieu SUPABASE_URL/SUPABASE_KEY trong .env")
        return
    sb = create_client(url, key)

    row = {
        "location_name": "DIAG_TEST",
        "slot_timestamp": "2026-07-03T09:00:00+07:00",
        "rf_score_safety": 90.0, "xgb_score_safety": 88.0, "flight_safety_score": 89.0,
        "crop_impact_score": 100.0, "spray_quality_score": 95.0,
        "system_decision": "FLY", "is_user_overridden": False,
        "override_reason": None, "xai_explanation": "diag test",
        "weather_json": {"temperature_2m": 30, "wind_speed_10m": 8},
    }

    print("1) Thu UPSERT (on_conflict location_name,slot_timestamp)...")
    try:
        sb.table("flight_decision_log").upsert(
            [row], on_conflict="location_name,slot_timestamp").execute()
        print("   -> UPSERT OK. Log dashboard se hoat dong.")
    except Exception as e:
        print(f"   -> UPSERT LOI: {e}")
        print("   (Thuong do: chua chay migration_002 / thieu unique index / thieu cot)")

    print("2) Dem so dong hien co:")
    try:
        r = sb.table("flight_decision_log").select("*", count="exact").execute()
        print(f"   -> {r.count} dong")
    except Exception as e:
        print(f"   -> LOI dem: {e}")

    print("3) Don dep dong DIAG_TEST...")
    try:
        sb.table("flight_decision_log").delete().eq("location_name", "DIAG_TEST").execute()
        print("   -> da xoa dong test")
    except Exception as e:
        print(f"   -> LOI xoa: {e}")


if __name__ == "__main__":
    main()
