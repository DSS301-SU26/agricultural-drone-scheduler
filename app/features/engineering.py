"""
Feature engineering THUAN (Layer 2 - phan tach ro voi rules/labeling).

KHONG sinh nhan o day. Chi bien doi dac trung dau vao cho model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Cac dac trung khi tuong dua vao model (co trong weather_hourly + dan xuat)
WEATHER_FEATURES: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "cloud_cover",
    "visibility",
    "wind_speed_10m",
    "wind_gusts_10m",
    "weather_code",
    "et0_fao_evapotranspiration",
    "hour",
    "dayofweek",
    "month",
    "vpd",
    "delta_t",
    "gust_wind_gap",
]


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cac dai luong nong hoc dan xuat: VPD, Delta T, chenh gio giat."""
    out = df.copy()
    temp = out["temperature_2m"].astype(float)
    rh = out["relative_humidity_2m"].astype(float).clip(0, 100)

    # Ap suat hoi bao hoa (kPa) - Tetens; VPD = es * (1 - RH/100)
    es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
    out["vpd"] = (es * (1.0 - rh / 100.0)).round(3)

    # Delta T (xap xi) = nhiet bau kho - nhiet bau uot; proxy tu temp & rh
    out["delta_t"] = ((temp * (0.45 + 0.006 * temp)) * (1 - rh / 100.0)).round(3)

    gust = out.get("wind_gusts_10m", 0.0)
    wind = out.get("wind_speed_10m", 0.0)
    out["gust_wind_gap"] = (pd.to_numeric(gust, errors="coerce").fillna(0)
                            - pd.to_numeric(wind, errors="coerce").fillna(0)).round(2)
    return out


def build_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Pipeline day du: time + derived. Tra ve df da co du cac cot WEATHER_FEATURES."""
    out = add_time_features(df, ts_col)
    out = add_derived_features(out)
    for col in WEATHER_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Lay dung cac cot feature theo thu tu on dinh."""
    return df.reindex(columns=WEATHER_FEATURES).astype(float)
