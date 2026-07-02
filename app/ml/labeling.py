"""
Gan nhan FLY/DELAY/NO_FLY co NHIEU (dung chung cho ca du lieu MO PHONG lan THAT).

Ly do dung chung: du lieu thoi tiet that KHONG co san nhan (khong ai ghi "hom do co
bay khong"). Ta gan nhan tu latent risk lien tuc + nhieu Gaussian + ca override cua
con nguoi -> anh xa feature->label ngau nhien, model hoc XAC SUAT thay vi thuoc luat.
Rules cung (drone/stage) van la lop rao chan RIENG o HM5, khong nam trong nhan nay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def attach_noisy_labels(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Them cot `system_decision` (FLY/DELAY/NO_FLY) va `is_user_overridden`.
    Ap dung cho bat ky DataFrame co cac cot weather_hourly chuan."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(default).to_numpy()
        return np.full(n, default)

    wind = col("wind_speed_10m")
    gust = col("wind_gusts_10m")
    rain = col("precipitation")
    rain_p = col("precipitation_probability")
    temp = col("temperature_2m", 30.0)
    vis = col("visibility", 10000.0)

    # Diem rui ro lien tuc (cang cao cang nguy hiem)
    risk = (
        0.045 * wind + 0.030 * gust + 0.25 * rain + 0.010 * rain_p
        + 0.05 * np.clip(temp - 33, 0, None)
        + 1.2 * (vis < 1000)
        + rng.normal(0, 0.25, n)              # nhieu bat dinh
    )
    p_unsafe = 1 / (1 + np.exp(-(risk - 1.6)))

    u = rng.uniform(0, 1, n)
    label = np.where(p_unsafe > 0.66, "NO_FLY",
             np.where(p_unsafe > 0.40, "DELAY", "FLY"))
    # Vung xam: mot so DELAY bi nhieu day len/xuong
    flip = (p_unsafe > 0.33) & (p_unsafe < 0.72) & (u < 0.12)
    label = np.where(flip & (label == "DELAY"), rng.choice(["FLY", "NO_FLY"], n), label)

    # OVERRIDE cua con nguoi: ~8% ca DELAY bi ep thanh FLY (dap dich khan cap)
    is_override = (label == "DELAY") & (rng.uniform(0, 1, n) < 0.08)
    label = np.where(is_override, "FLY", label)

    out["system_decision"] = label
    out["is_user_overridden"] = is_override
    return out
