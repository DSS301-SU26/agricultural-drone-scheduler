"""
load_to_supabase.py - Upload clean data len Supabase
"""
import os, pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError("Thieu SUPABASE_URL hoac SUPABASE_KEY trong .env")
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def prepare_rows(df):
    col_map = {
        "location_name": "location_name",
        "latitude": "latitude",
        "longitude": "longitude",
        "timestamp": "timestamp",
        "wind_speed_10m": "wind_speed",
        "wind_gusts_10m": "wind_gusts",
        "precipitation": "precipitation",
        "precipitation_probability": "precipitation_probability",
        "cloud_cover": "cloud_cover",
        "visibility": "visibility",
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "weather_code": "weather_code",
        "weather_description": "weather_description",
        "flyability_score": "flyability_score",
        "fly_label": "fly_label",
        "risk_level": "risk_level",
    }
    available = {k: v for k, v in col_map.items() if k in df.columns}
    subset = df[list(available.keys())].rename(columns=available).copy()
    subset["timestamp"] = subset["timestamp"].astype(str)
    if "risk_level" in subset.columns:
        subset["risk_level"] = subset["risk_level"].astype(str)
        
    # Replace NaN with None so it becomes valid JSON 'null'
    subset = subset.astype(object).where(pd.notnull(subset), None)
    
    return subset.to_dict(orient="records")

def load_to_supabase(filepath, table="raw_weather_data", batch_size=500):
    print(f"\n  Connecting Supabase...")
    client = get_supabase_client()
    print(f"  OK: {SUPABASE_URL[:40]}...")

    df = pd.read_csv(filepath, encoding="utf-8-sig", parse_dates=["timestamp"])
    print(f"  Doc: {len(df)} ban ghi")

    rows = prepare_rows(df)
    success = 0
    total = len(rows)

    for i in range(0, total, batch_size):
        batch = rows[i:i+batch_size]
        try:
            client.table(table).upsert(batch, on_conflict="location_name,timestamp").execute()
            success += len(batch)
            print(f"  Batch {i//batch_size+1}: {success}/{total} ban ghi")
        except Exception as e:
            print(f"  [ERROR] Batch {i//batch_size+1}: {e}")

    print(f"  DONE: {success}/{total} ban ghi da len Supabase\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    load_to_supabase(args.input)
