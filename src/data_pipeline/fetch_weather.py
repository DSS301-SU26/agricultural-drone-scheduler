"""
fetch_weather.py - Thu thập dữ liệu thời tiết từ Open-Meteo API
"""
import requests
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path

FARM_LOCATIONS = [
    {"name": "Dong Thap",  "latitude": 10.4939, "longitude": 105.6882},
    {"name": "Long An",    "latitude": 10.5360, "longitude": 106.4052},
    {"name": "Tien Giang", "latitude": 10.3598, "longitude": 106.3567},
    {"name": "An Giang",   "latitude": 10.5216, "longitude": 105.1259},
    {"name": "Can Tho",    "latitude": 10.0341, "longitude": 105.7878},
]

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_FIELDS = [
    "temperature_2m", "relative_humidity_2m",
    "precipitation_probability", "precipitation",
    "cloud_cover", "visibility",
    "wind_speed_10m", "wind_gusts_10m", "weather_code",
]

WMO_DESCRIPTIONS = {
    0: "Troi quang", 1: "Chu yeu quang", 2: "Nhieu may mot phan",
    3: "Nhieu may", 45: "Suong mu", 48: "Suong mu co bang",
    51: "Mua phun nhe", 53: "Mua phun vua", 55: "Mua phun day",
    61: "Mua nhe", 63: "Mua vua", 65: "Mua lon",
    80: "Mua rao nhe", 81: "Mua rao vua", 82: "Mua rao manh",
    95: "Dong", 99: "Dong kem mua da",
}

def fetch_one_location(location, forecast_days=7):
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "hourly": ",".join(HOURLY_FIELDS),
        "forecast_days": forecast_days,
        "timezone": "Asia/Bangkok",
    }
    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] {location['name']}: {e}")
        return pd.DataFrame()

    raw = response.json()
    hourly = raw.get("hourly", {})
    if not hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.insert(0, "location_name", location["name"])
    df.insert(1, "latitude",      location["latitude"])
    df.insert(2, "longitude",     location["longitude"])
    df["weather_description"] = df["weather_code"].map(WMO_DESCRIPTIONS).fillna("Khong xac dinh")
    return df

def fetch_all_locations(locations, forecast_days=7):
    all_dfs = []
    print(f"\n{'='*50}")
    print(f"  Open-Meteo Fetcher — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  {forecast_days} ngay | {len(locations)} dia diem")
    print(f"{'='*50}")
    for loc in locations:
        print(f"\n-> {loc['name']} ({loc['latitude']}, {loc['longitude']})")
        df = fetch_one_location(loc, forecast_days)
        if not df.empty:
            all_dfs.append(df)
            print(f"  OK: {len(df)} ban ghi")
        else:
            print(f"  FAIL: khong lay duoc du lieu")
    if not all_dfs:
        print("\n[WARN] Khong co du lieu nao!")
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  TONG: {len(combined)} ban ghi tu {len(all_dfs)} dia diem\n")
    return combined

def save_raw(df, output_dir="data/raw"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{output_dir}/weather_raw_{ts}.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"  Saved: {filename}")
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    df = fetch_all_locations(FARM_LOCATIONS, forecast_days=args.days)
    if args.save and not df.empty:
        save_raw(df)
