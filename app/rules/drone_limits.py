"""
Lop 1 - Rao chan co hoc (Hard Rules) theo tung ho so drone (BRD §8 Lop 1).

Neu gio TB / gio giat vuot gioi han vat ly cua may -> NO_FLY khoa cung
(is_hard=True), nguoi dung KHONG duoc override.
"""
from __future__ import annotations

from .context import DroneProfile
from .types import FactorResult, Verdict


def evaluate_drone_limits(
    wind_speed_kph: float,
    wind_gust_kph: float,
    drone: DroneProfile,
) -> list[FactorResult]:
    """Tra ve cac FactorResult hard-limit cho gio va gio giat."""
    results: list[FactorResult] = []

    if wind_speed_kph > drone.max_wind_resistance_kph:
        results.append(FactorResult(
            factor="drone_wind_limit",
            verdict=Verdict.STOP,
            value=round(wind_speed_kph, 1),
            message=(
                f"Gio TB {wind_speed_kph:.1f} km/h vuot gioi han vat ly "
                f"{drone.max_wind_resistance_kph:.1f} km/h cua {drone.model_name}. "
                "Khoa cung dong co (NO_FLY)."
            ),
            is_hard=True,
        ))
    else:
        results.append(FactorResult(
            factor="drone_wind_limit",
            verdict=Verdict.ALLOW,
            value=round(wind_speed_kph, 1),
            message=(
                f"Gio TB {wind_speed_kph:.1f} km/h trong gioi han "
                f"{drone.max_wind_resistance_kph:.1f} km/h cua {drone.model_name}."
            ),
            is_hard=True,
        ))

    if wind_gust_kph > drone.max_gust_resistance_kph:
        results.append(FactorResult(
            factor="drone_gust_limit",
            verdict=Verdict.STOP,
            value=round(wind_gust_kph, 1),
            message=(
                f"Gio giat {wind_gust_kph:.1f} km/h vuot nguong {drone.max_gust_resistance_kph:.1f} "
                f"km/h cua {drone.model_name} - nguy co lat may. NO_FLY."
            ),
            is_hard=True,
        ))

    return results
