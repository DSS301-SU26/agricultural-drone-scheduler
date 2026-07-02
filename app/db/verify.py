"""
Verify schema + seed tren Supabase (HM1).
Chay: .venv/bin/python -m app.db.verify
Doc SUPABASE_URL / SUPABASE_KEY tu .env.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parents[2]

TABLES = [
    "u_profiles", "crop_profile", "m_plots", "drone_profiles", "pesticide_specs",
    "weather_hourly", "soil_readings", "spray_mission_plans", "flight_decision_log",
]
SEED_TABLES = {
    "drone_profiles": "model_name",
    "pesticide_specs": "active_ingredient",
    "crop_profile": "stage_code",
}


def main() -> int:
    load_dotenv(ROOT / ".env")
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("[ERROR] Thieu SUPABASE_URL/SUPABASE_KEY trong .env")
        return 1

    client = create_client(url, key)
    print("=== Kiem tra 9 bang ton tai ===")
    all_ok = True
    for tbl in TABLES:
        try:
            resp = client.table(tbl).select("*", count="exact").limit(1).execute()
            print(f"  OK  {tbl:22s} (rows={resp.count})")
        except Exception as e:
            all_ok = False
            print(f"  FAIL {tbl:22s} -> {e}")

    print("\n=== Kiem tra du lieu seed ===")
    for tbl, col in SEED_TABLES.items():
        try:
            resp = client.table(tbl).select(col).execute()
            vals = [r[col] for r in resp.data]
            print(f"  {tbl}: {vals}")
        except Exception as e:
            all_ok = False
            print(f"  FAIL seed {tbl} -> {e}")

    print("\n" + ("ALL GOOD ✅" if all_ok else "CO LOI ❌"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
