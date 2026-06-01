"""
DSS decision rules used to label training data and explain model outputs.

The machine-learning model learns this transparent decision policy from
historical/simulated data, while these helpers keep the demo explainable for
DSS301 reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


HARDWARE_DAMAGE_COST_USD = 2_000
BASE_SPRAY_LITERS = 100.0
AGRICULTURAL_LOCATIONS = {
    "An Giang",
    "Can Tho",
    "Dong Thap",
    "Long An",
    "Tien Giang",
}


@dataclass(frozen=True)
class DecisionThresholds:
    max_wind_speed: float = 20.0
    max_wind_gust: float = 28.0
    max_rain_probability: float = 30.0
    max_cloud_cover: float = 80.0
    min_visibility: float = 1_000.0
    max_safe_temperature: float = 35.0


THRESHOLDS = DecisionThresholds()
# The merged dataset can contain both Open-Meteo WMO codes and WeatherAPI
# condition codes. Keep both lists so model labels match the source data.
UNSAFE_WMO_CODES = {45, 48, 55, 63, 65, 71, 80, 81, 82, 95, 99}
UNSAFE_WEATHERAPI_CODES = {
    1087,
    1135,
    1147,
    1192,
    1195,
    1201,
    1243,
    1246,
    1273,
    1276,
    1279,
    1282,
}
UNSAFE_WEATHER_CODES = UNSAFE_WMO_CODES | UNSAFE_WEATHERAPI_CODES

WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation_probability",
    "precipitation",
    "cloud_cover",
    "visibility",
    "wind_speed_10m",
    "wind_gusts_10m",
    "weather_code",
    "hour",
    "dayofweek",
    "month",
]


def image_feature_columns(columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col.startswith("img_feature_")]


def infer_crop_condition(row: pd.Series) -> str:
    """Proxy crop condition for demo when images are represented as embeddings."""
    temp = float(row.get("temperature_2m", 0))
    humidity = float(row.get("relative_humidity_2m", 100))
    rain_prob = float(row.get("precipitation_probability", 100))
    precipitation = float(row.get("precipitation", 0))

    if precipitation > 0 or rain_prob >= 65:
        return "HEALTHY"
    if temp >= 35 and humidity <= 60:
        return "DRY_SOIL"
    if temp >= 32 and humidity <= 70:
        return "WATER_STRESS"
    return "HEALTHY"


def calculate_dynamic_flow_rate(row: pd.Series) -> float:
    """
    Estimate spray/irrigation flow-rate percentage.

    The rate is intentionally simple and explainable: dry conditions increase
    flow, while wind, rain probability, and high heat reduce waste-prone output.
    """
    condition = row.get("crop_condition") or infer_crop_condition(row)
    flow = 100.0

    if condition == "DRY_SOIL":
        flow += 15.0
    elif condition == "WATER_STRESS":
        flow += 8.0

    if float(row.get("temperature_2m", 0)) >= 35:
        flow -= 12.0
    if float(row.get("wind_speed_10m", 0)) > 15:
        flow -= 10.0
    if float(row.get("precipitation_probability", 0)) > 30:
        flow -= 15.0

    return round(min(120.0, max(0.0, flow)), 1)


def derive_decision_action(row: pd.Series) -> str:
    wind = float(row.get("wind_speed_10m", 0))
    gust = float(row.get("wind_gusts_10m", 0))
    rain = float(row.get("precipitation", 0))
    rain_prob = float(row.get("precipitation_probability", 0))
    weather_code = int(row.get("weather_code", 0))
    temp = float(row.get("temperature_2m", 0))

    if rain > 0 or rain_prob >= 70 or weather_code in UNSAFE_WEATHER_CODES:
        return "RETURN_TO_CHARGING"
    if gust > THRESHOLDS.max_wind_gust or wind > THRESHOLDS.max_wind_speed:
        return "LOCK_SPRAY"
    if rain_prob > THRESHOLDS.max_rain_probability or temp > THRESHOLDS.max_safe_temperature:
        return "DELAY_FLIGHT"
    return "TAKE_OFF"


def add_decision_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["crop_condition"] = enriched.apply(infer_crop_condition, axis=1)
    enriched["decision_action"] = enriched.apply(derive_decision_action, axis=1)
    enriched["dynamic_flow_rate_pct"] = enriched.apply(calculate_dynamic_flow_rate, axis=1)
    enriched["estimated_damage_cost_usd"] = (
        enriched["decision_action"].isin(["LOCK_SPRAY", "RETURN_TO_CHARGING"]).astype(int)
        * HARDWARE_DAMAGE_COST_USD
    )
    return enriched


def build_recommendation_text(row: pd.Series, action: str | None = None) -> str:
    action = action or str(row.get("decision_action", "TAKE_OFF"))
    wind = float(row.get("wind_speed_10m", 0))
    gust = float(row.get("wind_gusts_10m", 0))
    rain_prob = float(row.get("precipitation_probability", 0))
    temp = float(row.get("temperature_2m", 0))
    flow = float(row.get("dynamic_flow_rate_pct", calculate_dynamic_flow_rate(row)))

    if action == "TAKE_OFF":
        return (
            f"TAKE_OFF: Dieu kien bay chap nhan duoc. Gio {wind:.1f} km/h, "
            f"gio giat {gust:.1f} km/h, xac suat mua {rain_prob:.0f}%. "
            f"De xuat flow-rate {flow:.1f}%."
        )
    if action == "LOCK_SPRAY":
        return (
            f"LOCK_SPRAY: Khoa lenh phun vi gio/gio giat vuot nguong an toan "
            f"({wind:.1f}/{gust:.1f} km/h). Tranh pesticide drift va mat on dinh UAV."
        )
    if action == "RETURN_TO_CHARGING":
        return (
            f"RETURN_TO_CHARGING: Thoi tiet mua/nguy hiem, xac suat mua {rain_prob:.0f}%. "
            "Dua drone ve tram sac de bao ve thiet bi."
        )
    return (
        f"DELAY_FLIGHT: Tam hoan bay do nhiet do {temp:.1f}C hoac rui ro mua "
        f"{rain_prob:.0f}%. Kiem tra lai khung gio ke tiep."
    )


def recommend_best_slot(df: pd.DataFrame, location_name: str | None = None) -> pd.Series | None:
    candidate_df = add_decision_columns(df)
    if location_name:
        candidate_df = candidate_df[candidate_df["location_name"] == location_name]

    safe_slots = candidate_df[candidate_df["decision_action"] == "TAKE_OFF"].copy()
    if safe_slots.empty:
        return None

    sort_cols = [
        "flyability_score",
        "wind_gusts_10m",
        "precipitation_probability",
        "temperature_2m",
    ]
    ascending = [False, True, True, True]
    return safe_slots.sort_values(sort_cols, ascending=ascending).iloc[0]
