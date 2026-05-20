"""
fetch_historical.py - Lay data lich su 60 ngay tu Open-Meteo
Dung de train ML model (Random Forest, Decision Tree)

Open-Meteo Historical API: mien phi, khong can key
Docs: https://open-meteo.com/en/docs/historical-weather-api

Chay: python3 fetch_historical.py
      python3 fetch_historical.py --start 2026-01-01 --end 2026-05-19
"""
import requests
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timedelta

FARM_LOCATIONS = [
    {"name": "Dong Thap",   "lat": 10.4939, "lon": 105.6882},
    {"name": "Long An",     "lat": 10.5360, "lon": 106.4052},
    {"name": "Tien Giang",  "lat": 10.3598, "lon": 106.3567},
    {"name": "An Giang",    "lat": 10.5216, "lon": 105.1259},
    {"name": "Can Tho",     "lat": 10.0341, "lon": 105.7878},
    {"name": "Ho Chi Minh", "lat": 10.7769, "lon": 106.7009},
    {"name": "Ha Noi",      "lat": 21.0285, "lon": 105.8542},
]

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_FIELDS = ",".join([
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation_probability",
    "precipitation",
    "cloud_cover",
    "visibility",
    "wind_speed_10m",
    "wind_gusts_10m",
    "weather_code",
])


def fetch_one(location: dict, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude":   location["lat"],
        "longitude":  location["lon"],
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     HOURLY_FIELDS,
        "timezone":   "Asia/Bangkok",
    }
    try:
        resp = requests.get(HISTORICAL_URL, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] {location['name']}: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(resp.json()["hourly"])
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df["timestamp"]           = pd.to_datetime(df["timestamp"])
    df["location_name"]       = location["name"]
    df["latitude"]            = location["lat"]
    df["longitude"]           = location["lon"]
    df["source"]              = "Open-Meteo-Historical"
    df["weather_description"] = ""
    return df


def fetch_all(start_date: str, end_date: str) -> pd.DataFrame:
    all_dfs = []
    print(f"\n{'='*52}")
    print(f"  Open-Meteo Historical Fetcher")
    print(f"  Tu {start_date} den {end_date} | {len(FARM_LOCATIONS)} dia diem")
    print(f"{'='*52}")

    for loc in FARM_LOCATIONS:
        print(f"\n-> {loc['name']}")
        df = fetch_one(loc, start_date, end_date)
        if not df.empty:
            all_dfs.append(df)
            print(f"  OK: {len(df)} ban ghi")
        else:
            print(f"  FAIL")

    if not all_dfs:
        print("[WARN] Khong co du lieu!")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  TONG: {len(combined)} ban ghi tu {len(all_dfs)} dia diem")
    return combined


def save(df: pd.DataFrame, output_dir: str = "data/raw") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{output_dir}/historical_60days.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"  Saved: {filename}")
    return filename


if __name__ == "__main__":
    # Mac dinh: 60 ngay truoc hom nay
    default_end   = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(description="Fetch historical weather data")
    parser.add_argument("--start", default=default_start, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=default_end,   help="End date YYYY-MM-DD")
    args = parser.parse_args()

    df = fetch_all(args.start, args.end)
    if not df.empty:
        save(df)
        print(f"\n  San sang de clean va train model!")
