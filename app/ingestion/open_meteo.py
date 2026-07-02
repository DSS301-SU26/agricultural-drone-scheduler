"""
Thu thap thoi tiet tu Open-Meteo (HM2 - Layer 1).

- Lich su (5 nam) tu archive-api de train  -> fetch_historical()
- Du bao (live, theo GPS plot) de phuc vu   -> fetch_forecast()

Cot output = chuan bang weather_hourly. So voi code cu, BO SUNG:
et0_fao_evapotranspiration, wind_direction_10m, soil_moisture_0_to_7cm (cho AWD).

parse_hourly() tach rieng de test offline (khong can mang).
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Cac truong hourly yeu cau. Mot so chi co o forecast (visibility, precip probability)
# hoac chi co o archive; parse_hourly xu ly thieu cot an toan.
HOURLY_FIELDS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "precipitation_probability", "cloud_cover", "visibility",
    "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m",
    "weather_code", "et0_fao_evapotranspiration", "soil_moisture_0_to_7cm",
]

# Cot chuan weather_hourly + soil (soil_moisture giu de dua sang soil_readings)
WEATHER_COLUMNS = [
    "location_name", "latitude", "longitude", "timestamp",
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "precipitation_probability", "cloud_cover", "visibility",
    "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m",
    "weather_code", "et0_fao_evapotranspiration", "soil_moisture_0_to_7cm",
    "source",
]


def parse_hourly(payload: dict[str, Any], location: dict, source: str) -> pd.DataFrame:
    """Chuyen JSON Open-Meteo -> DataFrame chuan weather_hourly. An toan voi cot thieu."""
    hourly = payload.get("hourly") or {}
    if "time" not in hourly:
        return pd.DataFrame(columns=WEATHER_COLUMNS)

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Dam bao du cot HOURLY_FIELDS
    for f in HOURLY_FIELDS:
        if f not in df.columns:
            df[f] = pd.NA

    # precipitation_probability khong co o archive -> proxy tu luong mua
    if df["precipitation_probability"].isna().all():
        precip = pd.to_numeric(df["precipitation"], errors="coerce").fillna(0)
        df["precipitation_probability"] = (precip > 0).astype(int) * 80 + (precip <= 0).astype(int) * 10

    # visibility co the thieu o archive -> mac dinh 10km (an toan VLOS)
    df["visibility"] = pd.to_numeric(df["visibility"], errors="coerce").fillna(10000.0)

    df["location_name"] = location["name"]
    df["latitude"] = location["lat"]
    df["longitude"] = location["lon"]
    df["source"] = source
    return df.reindex(columns=WEATHER_COLUMNS)


def _year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    chunks, cur = [], s
    while cur <= e:
        year_end = min(pd.to_datetime(f"{cur.year}-12-31"), e)
        chunks.append((cur.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d")))
        cur = year_end + pd.Timedelta(days=1)
    return chunks


def fetch_historical(locations: list[dict], start: str, end: str,
                     pause: float = 0.5) -> pd.DataFrame:
    """Crawl lich su tung nam mot cho danh sach dia diem. Yeu cau mang (chay tren may that)."""
    import requests  # import cuc bo de test offline khong can requests

    frames: list[pd.DataFrame] = []
    for loc in locations:
        print(f"-> {loc['name']}")
        for c_start, c_end in _year_chunks(start, end):
            params = {
                "latitude": loc["lat"], "longitude": loc["lon"],
                "start_date": c_start, "end_date": c_end,
                "hourly": ",".join(HOURLY_FIELDS), "timezone": "Asia/Bangkok",
            }
            for attempt in range(3):
                try:
                    r = requests.get(ARCHIVE_URL, params=params, timeout=60)
                    r.raise_for_status()
                    frames.append(parse_hourly(r.json(), loc, "open-meteo-archive"))
                    break
                except Exception as e:
                    print(f"   [retry {attempt+1}] {loc['name']} {c_start}: {e}")
                    time.sleep(2 * (attempt + 1))
            time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=WEATHER_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def fetch_forecast(lat: float, lon: float, name: str = "plot",
                   days: int = 3) -> pd.DataFrame:
    """Du bao live theo GPS (Pha B - serving). Yeu cau mang."""
    import requests

    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(HOURLY_FIELDS),
        "forecast_days": min(max(days, 1), 16), "timezone": "Asia/Bangkok",
    }
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    return parse_hourly(r.json(), {"name": name, "lat": lat, "lon": lon}, "open-meteo-forecast")
