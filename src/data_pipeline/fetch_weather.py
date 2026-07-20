import os
import time
import requests
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

FARM_LOCATIONS = [
    {"name": "Dong Thap",   "latitude": 10.4939, "longitude": 105.6882},
    {"name": "Long An",     "latitude": 10.5360, "longitude": 106.4052},
    {"name": "Tien Giang",  "latitude": 10.3598, "longitude": 106.3567},
    {"name": "An Giang",    "latitude": 10.5216, "longitude": 105.1259},
    {"name": "Can Tho",     "latitude": 10.0341, "longitude": 105.7878},
    {"name": "Ho Chi Minh", "latitude": 10.7769, "longitude": 106.7009},
    {"name": "Ha Noi",      "latitude": 21.0285, "longitude": 105.8542},
]

def fetch_one_location(location: dict, days: int = 3) -> pd.DataFrame:
    """
    Fetch weather forecast from Open-Meteo with 5s timeout, 3 retries, exponential backoff,
    linear interpolation, and fallback to OpenWeatherMap or generated data.
    """
    base_url = "http://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,precipitation_probability,cloud_cover,visibility,wind_speed_10m,wind_gusts_10m,weather_code,evapotranspiration",
        "timezone": "Asia/Ho_Chi_Minh",
        "forecast_days": min(days, 7)
    }

    # Circuit Breaker with 3 Retries and Exponential Backoff
    success = False
    data = None
    for attempt in range(3):
        try:
            resp = requests.get(base_url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            success = True
            break
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"  [Attempt {attempt+1} Failed] {location['name']}: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    # Fallback Mechanism
    if not success or not data:
        print(f"  [Fallback Activated] Fetching fallback data for {location['name']}...")
        # Check if we can use OpenWeatherMap if key exists, otherwise generate mock data
        owm_key = os.getenv("OPENWEATHERMAP_KEY")
        if owm_key:
            try:
                # Basic OWM call
                owm_url = "https://api.openweathermap.org/data/2.5/forecast"
                owm_params = {
                    "lat": location["latitude"],
                    "lon": location["longitude"],
                    "appid": owm_key,
                    "units": "metric"
                }
                resp = requests.get(owm_url, params=owm_params, timeout=5)
                resp.raise_for_status()
                owm_data = resp.json()
                
                # Convert OWM to Open-Meteo structure
                rows = []
                for item in owm_data.get("list", []):
                    dt = pd.to_datetime(item.get("dt"), unit='s')
                    main = item.get("main", {})
                    wind = item.get("wind", {})
                    rain = item.get("rain", {}).get("3h", 0.0) / 3.0 # convert 3h to hourly approx
                    clouds = item.get("clouds", {}).get("all", 0.0)
                    weather = item.get("weather", [{}])[0]
                    rows.append({
                        "location_name": location["name"],
                        "latitude": location["latitude"],
                        "longitude": location["longitude"],
                        "timestamp": dt,
                        "source": "OpenWeatherMap",
                        "temperature_2m": main.get("temp"),
                        "relative_humidity_2m": main.get("humidity"),
                        "precipitation_probability": item.get("pop", 0.0) * 100,
                        "precipitation": rain,
                        "cloud_cover": clouds,
                        "visibility": item.get("visibility", 10000.0),
                        "wind_speed_10m": wind.get("speed", 0.0) * 3.6, # m/s to km/h
                        "wind_gusts_10m": wind.get("gust", wind.get("speed", 0.0)) * 3.6,
                        "weather_code": weather.get("id", 800),
                        "weather_description": weather.get("description", ""),
                        "evapotranspiration": 0.1 # fallback
                    })
                return pd.DataFrame(rows)
            except Exception as e:
                print(f"  [OWM Fallback Failed] {e}. Falling back to simulated weather data.")

        # Fallback to simulated data generator
        print(f"  [Mock Fallback] Generating simulated forecast for {location['name']}...")
        dates = pd.date_range(start=datetime.now().replace(minute=0, second=0, microsecond=0), periods=days*24, freq='h')
        rows = []
        for dt in dates:
            # Generate deterministic weather patterns based on hour of day
            hour = dt.hour
            temp = 25.0 + 7.0 * (1.0 - abs(hour - 14) / 12.0) # peak at 14:00
            humidity = 90.0 - 30.0 * (1.0 - abs(hour - 14) / 12.0)
            wind = 5.0 + 8.0 * (1.0 - abs(hour - 15) / 12.0)
            rows.append({
                "location_name": location["name"],
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "timestamp": dt,
                "source": "Simulated",
                "temperature_2m": temp,
                "relative_humidity_2m": humidity,
                "precipitation_probability": 10.0 if hour < 12 else 45.0,
                "precipitation": 0.0 if hour < 15 else 0.5,
                "cloud_cover": 20.0 + 5.0 * hour,
                "visibility": 10000.0,
                "wind_speed_10m": wind,
                "wind_gusts_10m": wind * 1.3,
                "weather_code": 1000 if hour < 12 else 1087,
                "weather_description": "Clear" if hour < 12 else "Thundery",
                "evapotranspiration": 0.05 + 0.02 * (hour / 24.0)
            })
        return pd.DataFrame(rows)

    # Process Open-Meteo response
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    rows = []
    for i, t in enumerate(times):
        rows.append({
            "location_name": location["name"],
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "timestamp": pd.to_datetime(t),
            "source": "Open-Meteo",
            "temperature_2m": hourly.get("temperature_2m", [])[i],
            "relative_humidity_2m": hourly.get("relative_humidity_2m", [])[i],
            "precipitation": hourly.get("precipitation", [])[i],
            "precipitation_probability": hourly.get("precipitation_probability", [])[i],
            "cloud_cover": hourly.get("cloud_cover", [])[i],
            "visibility": hourly.get("visibility", [])[i],
            "wind_speed_10m": hourly.get("wind_speed_10m", [])[i],
            "wind_gusts_10m": hourly.get("wind_gusts_10m", [])[i],
            "weather_code": hourly.get("weather_code", [])[i],
            "weather_description": f"Code {hourly.get('weather_code', [])[i]}",
            "evapotranspiration": hourly.get("evapotranspiration", [])[i]
        })

    df = pd.DataFrame(rows)
    # Data Preprocessor: Linear Interpolation for any NaN/Null values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    return df

def fetch_all_locations(locations: list, forecast_days: int = 3) -> pd.DataFrame:
    all_dfs = []
    print(f"\n{'='*52}")
    print(f"  Open-Meteo Fetcher — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  {min(forecast_days,7)} ngay | {len(locations)} dia diem")
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
