import os
import requests
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

API_KEY  = os.getenv("WEATHERAPI_KEY")
BASE_URL = "http://api.weatherapi.com/v1/forecast.json"

FARM_LOCATIONS = [
    {"name": "Dong Thap",   "latitude": 10.4939, "longitude": 105.6882},
    {"name": "Long An",     "latitude": 10.5360, "longitude": 106.4052},
    {"name": "Tien Giang",  "latitude": 10.3598, "longitude": 106.3567},
    {"name": "An Giang",    "latitude": 10.5216, "longitude": 105.1259},
    {"name": "Can Tho",     "latitude": 10.0341, "longitude": 105.7878},
    {"name": "Ho Chi Minh", "latitude": 10.7769, "longitude": 106.7009},
    {"name": "Ha Noi",      "latitude": 21.0285, "longitude": 105.8542},
]

UNSAFE_CONDITION_CODES = {
    1087, 1273, 1276, 1279, 1282,
    1192, 1195, 1201, 1243, 1246,
    1135, 1147,
}


def fetch_one_location(location: dict, days: int = 3) -> pd.DataFrame:
    global API_KEY
    if not API_KEY:
        load_dotenv(Path(__file__).parent.parent.parent / ".env")
        API_KEY = os.getenv("WEATHERAPI_KEY")
    if not API_KEY:
        raise EnvironmentError(
            "Thieu WEATHERAPI_KEY trong .env\n"
            "  -> Dang ky tai weatherapi.com roi them key vao .env"
        )

    params = {
        "key":    API_KEY,
        "q":      f"{location['latitude']},{location['longitude']}",
        "days":   min(days, 3),
        "aqi":    "no",
        "alerts": "no",
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"  [HTTP ERROR] {location['name']}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  [ERROR] {location['name']}: {e}")
        return pd.DataFrame()

    data = resp.json()
    rows = []

    for day in data.get("forecast", {}).get("forecastday", []):
        for hour in day.get("hour", []):
            code = hour.get("condition", {}).get("code", 0)
            rows.append({
                "location_name":             location["name"],
                "latitude":                  location["latitude"],
                "longitude":                 location["longitude"],
                "timestamp":                 pd.to_datetime(hour["time"]),
                "source":                    "WeatherAPI",
                "temperature_2m":            hour.get("temp_c"),
                "relative_humidity_2m":      hour.get("humidity"),
                "precipitation_probability": hour.get("chance_of_rain"),
                "precipitation":             hour.get("precip_mm"),
                "cloud_cover":               hour.get("cloud"),
                "visibility":                hour.get("vis_km", 0) * 1000,
                "wind_speed_10m":            hour.get("wind_kph"),
                "wind_gusts_10m":            hour.get("gust_kph"),
                "weather_code":              code,
                "weather_description":       hour.get("condition", {}).get("text", ""),
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def fetch_all_locations(locations: list, forecast_days: int = 3) -> pd.DataFrame:
    all_dfs = []
    print(f"\n{'='*52}")
    print(f"  WeatherAPI Fetcher — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  {min(forecast_days,3)} ngay | {len(locations)} dia diem")
    print(f"{'='*52}")

    for loc in locations:
        print(f"\n-> {loc['name']} ({loc['latitude']}, {loc['longitude']})")
        df = fetch_one_location(loc, days=forecast_days)
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


def save_raw(df: pd.DataFrame, output_dir: str = "data/raw") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{output_dir}/weather_raw_{ts}.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"  Saved: {filename}")
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    df = fetch_all_locations(FARM_LOCATIONS, forecast_days=args.days)
    if args.save and not df.empty:
        save_raw(df)
