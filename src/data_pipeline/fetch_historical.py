"""
fetch_historical.py - Lay data lich su 60 ngay tu Open-Meteo
Dung de train ML model (Random Forest, Decision Tree)

Open-Meteo Historical API: mien phi, khong can key
Docs: https://open-meteo.com/en/docs/historical-weather-api

Chay: python3 fetch_historical.py
      python3 fetch_historical.py --start 2026-01-01 --end 2026-05-19
"""
import time
import requests
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timedelta

def get_date_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    chunks = []
    current_dt = start_dt
    while current_dt <= end_dt:
        year = current_dt.year
        year_end = pd.to_datetime(f"{year}-12-31")
        chunk_end = min(year_end, end_dt)
        chunks.append((current_dt.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current_dt = chunk_end + pd.Timedelta(days=1)
    return chunks

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

WMO_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

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
    chunks = get_date_chunks(start_date, end_date)
    chunk_dfs = []
    
    for chunk_start, chunk_end in chunks:
        params = {
            "latitude":   location["lat"],
            "longitude":  location["lon"],
            "start_date": chunk_start,
            "end_date":   chunk_end,
            "hourly":     HOURLY_FIELDS,
            "timezone":   "Asia/Bangkok",
        }
        
        # Retry with backoff
        success = False
        df_chunk = None
        for attempt in range(3):
            try:
                resp = requests.get(HISTORICAL_URL, params=params, timeout=30)
                resp.raise_for_status()
                df_chunk = pd.DataFrame(resp.json()["hourly"])
                success = True
                break
            except Exception as e:
                wait_time = (attempt + 1) * 2
                print(f"  [Attempt {attempt+1} Failed] {location['name']} ({chunk_start} to {chunk_end}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
        if not success or df_chunk is None or df_chunk.empty:
            print(f"  [ERROR] {location['name']}: Failed to fetch chunk {chunk_start} to {chunk_end}")
            return pd.DataFrame()
            
        chunk_dfs.append(df_chunk)
        time.sleep(0.5)

    df = pd.concat(chunk_dfs, ignore_index=True)
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df["timestamp"]           = pd.to_datetime(df["timestamp"])
    df["location_name"]       = location["name"]
    df["latitude"]            = location["lat"]
    df["longitude"]           = location["lon"]
    df["source"]              = "Open-Meteo-Historical"
    df["weather_description"] = df["weather_code"].map(WMO_CODE_MAP).fillna("Unknown")
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


def save(df: pd.DataFrame, filepath: str) -> str:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {path}")
    return str(path)


if __name__ == "__main__":
    # Mac dinh: 60 ngay truoc hom nay
    default_end   = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(description="Fetch historical weather data")
    parser.add_argument("--start", default=default_start, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=default_end,   help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="data/raw/historical_60days.csv", help="Output file path")
    args = parser.parse_args()

    df = fetch_all(args.start, args.end)
    if not df.empty:
        save(df, args.output)
        print(f"\n  San sang de clean va train model!")
