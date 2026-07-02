"""
Sinh du lieu cam bien IoT thuc dia (soil_readings) - HM2.

Chua co cam bien that -> mo phong muc nuoc AWD + do am + do man cho demo/serving.
Khi co IoT that chi can thay ham nay bang doc tu thiet bi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_soil_series(plot_id: int, days: int = 14, seed: int = 0,
                         start_water_cm: float = 3.0) -> pd.DataFrame:
    """Chuoi muc nuoc AWD theo ngay: giam dan do boc hoi, thinh thoang bom bu."""
    rng = np.random.default_rng(seed + plot_id)
    ts = pd.date_range(pd.Timestamp.now().normalize() - pd.Timedelta(days=days - 1),
                       periods=days, freq="D")
    water, level = [], start_water_cm
    for _ in range(days):
        level -= rng.uniform(1.0, 3.0)              # boc hoi/tham
        if level <= -15:                            # cham nguong -> bom bu
            level = rng.uniform(3.0, 5.0)
        water.append(round(level, 1))
    return pd.DataFrame({
        "plot_id": plot_id,
        "timestamp": ts,
        "soil_moisture_percentage": np.clip(rng.normal(65, 8, days), 20, 100).round(1),
        "water_level_cm": water,
        "salinity_ec": np.clip(rng.normal(0.8, 0.3, days), 0.1, 4.0).round(2),
    })


def latest_water_level(plot_id: int, seed: int = 0) -> float:
    """Muc nuoc moi nhat (cm) cho 1 plot - dung cho quyet dinh AWD."""
    return float(simulate_soil_series(plot_id, days=14, seed=seed)["water_level_cm"].iloc[-1])
