"""
Sinh du lieu MO PHONG khi tuong ĐBSCL + nhan co NHIEU (HM4).

Muc dich: co bo du lieu de train RF/XGB NGAY ca khi chua crawl 5 nam that.
Diem then chot chong loi "ML hoc lai luat":
  - Nhan (label) sinh tu mot "latent risk" LIEN TUC + nhieu Gaussian + ca OVERRIDE
    cua con nguoi -> anh xa feature->label la ngau nhien, model phai hoc XAC SUAT.
  - Rules cung (drone/stage) van la lop rao chan rieng o HM5, khong nam trong nhan nay.

Sau nay chi can thay ham nay bang du lieu that tu weather_hourly (cung schema cot).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labeling import attach_noisy_labels

PROVINCES = {
    "An Giang": 10.52, "Can Tho": 10.03, "Dong Thap": 10.49,
    "Long An": 10.54, "Tien Giang": 10.36, "Kien Giang": 10.01,
}

# WMO code theo tinh trang mua
_WMO_CLEAR = [0, 1, 2, 3]
_WMO_RAIN = [61, 63, 65, 80, 81, 95]


def _season_factor(month: np.ndarray) -> np.ndarray:
    """Mua mua ĐBSCL ~ thang 5-11 (mua nhieu, am cao). Tra ve 0..1 (1=dinh mua mua)."""
    # dinh vao thang 8
    return 0.5 * (1 + np.cos((month - 8) / 12 * 2 * np.pi)) * 0 + \
        np.clip(np.sin((month - 4) / 12 * np.pi), 0, 1)


def simulate(n: int = 40_000, start: str = "2021-01-01", end: str = "2025-12-31",
             seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Chon ngau nhien timestamp trong khoang, gio ban ngay 5..18h (gio bay thuc te)
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    day_offset = rng.integers(0, days + 1, size=n)
    hour = rng.integers(5, 19, size=n)
    ts = pd.Timestamp(start) + pd.to_timedelta(day_offset, unit="D") + pd.to_timedelta(hour, unit="h")
    month = ts.month.to_numpy()

    loc = rng.choice(list(PROVINCES.keys()), size=n)

    wet = _season_factor(month)  # 0..1

    # --- Sinh khi tuong theo mua ---
    temperature = rng.normal(31 - 3 * wet, 2.5, n).clip(20, 40)                  # mua mua mat hon
    humidity = rng.normal(70 + 15 * wet, 8, n).clip(35, 100)
    # gio: nhieu gio chuong mua kho (gio mua dong bac); base 8-14 km/h
    wind = np.abs(rng.normal(10 + 4 * (1 - wet), 5, n)).clip(0, 45)
    gust = (wind + np.abs(rng.normal(6, 4, n))).clip(0, 60)
    rain_prob = (wet * 100 * rng.uniform(0.4, 1.0, n)).clip(0, 100)
    precipitation = np.where(rng.uniform(0, 1, n) < wet * 0.5,
                             np.abs(rng.normal(3 * wet, 3, n)), 0.0).clip(0, 40)
    cloud = (rain_prob * 0.7 + rng.normal(25, 15, n)).clip(0, 100)
    visibility = np.where(rng.uniform(0, 1, n) < 0.05,
                          rng.uniform(300, 1500, n), rng.uniform(5000, 20000, n))
    et0 = (rng.normal(4.5 + 1.5 * (1 - wet), 1.0, n)).clip(1, 9)
    is_rain = (precipitation > 1) | (rain_prob > 70)
    weather_code = np.where(is_rain, rng.choice(_WMO_RAIN, n), rng.choice(_WMO_CLEAR, n))

    df = pd.DataFrame({
        "location_name": loc,
        "timestamp": ts,
        "temperature_2m": temperature.round(1),
        "relative_humidity_2m": humidity.round(0),
        "precipitation": precipitation.round(2),
        "precipitation_probability": rain_prob.round(0),
        "cloud_cover": cloud.round(0),
        "visibility": visibility.round(0),
        "wind_speed_10m": wind.round(1),
        "wind_gusts_10m": gust.round(1),
        "weather_code": weather_code.astype(int),
        "et0_fao_evapotranspiration": et0.round(2),
    })
    df = attach_noisy_labels(df, seed=seed)
    return df
