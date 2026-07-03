"""
DSS decision rules used to label training data and explain model outputs.
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
    max_wind_speed: float = 28.8  # 8 m/s in km/h
    max_wind_gust: float = 28.8   # 8 m/s in km/h
    max_rain_probability: float = 50.0  # Precipitation Probability > 50%
    max_rain_hourly: float = 2.0        # Precipitation > 2 mm/h
    return_to_charging_rain_probability: float = 50.0
    max_cloud_cover: float = 80.0
    min_visibility: float = 1_000.0
    max_safe_temperature: float = 35.0


THRESHOLDS = DecisionThresholds()
UNSAFE_WMO_CODES = {45, 48, 55, 63, 65, 71, 80, 81, 82, 95, 99}
UNSAFE_WEATHERAPI_CODES = {
    1087, 1135, 1147, 1192, 1195, 1201, 1243, 1246, 1273, 1276, 1279, 1282,
}
UNSAFE_WEATHER_CODES = UNSAFE_WMO_CODES | UNSAFE_WEATHERAPI_CODES

SCORE_WEIGHTS = {
    "wind": 0.20,
    "gust": 0.15,
    "rain": 0.15,
    "rain_prob": 0.08,
    "cloud": 0.05,
    "visibility": 0.03,
    "weather": 0.04,
    "temperature": 0.30,
}

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
    "evapotranspiration",
    "soil_moisture_0_to_7cm"
]

RISK_WEIGHTS = {
    "TAKE_OFF": 0,
    "DELAY_FLIGHT": 1,
    "LOCK_SPRAY": 2,
    "RETURN_TO_CHARGING": 3
}


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


def calculate_dynamic_flow_rate(
    row: pd.Series,
    thresholds: DecisionThresholds = THRESHOLDS,
    crop_stage: dict[str, Any] | None = None,
) -> float:
    """
    Estimate spray/irrigation flow-rate in L/ha based on weather and crop stage.
    If crop_stage is provided, we use its min/max flow rate.
    """
    if crop_stage:
        min_f = float(crop_stage.get("flow_rate_min_l_ha", 15.0))
        max_f = float(crop_stage.get("flow_rate_max_l_ha", 25.0))
    else:
        min_f = 15.0
        max_f = 25.0

    temp = float(row.get("temperature_2m", 0))
    humidity = float(row.get("relative_humidity_2m", 100))
    
    # If weather is extremely dry/hot, use maximum flow rate
    if temp >= 35.0 and humidity < 50.0:
        return round(max_f, 1)
    
    # Scale based on humidity
    if humidity < 60.0:
        flow = min_f + (max_f - min_f) * 0.75
    elif humidity > 85.0:
        flow = min_f
    else:
        flow = (min_f + max_f) / 2.0
        
    return round(flow, 1)


def calculate_crop_safety_score(
    row: pd.Series,
    crop_stage: dict[str, Any] | None = None,
    pesticide: dict[str, Any] | None = None,
) -> float:
    """
    Calculate Crop Safety Score (0-100). Higher is safer (minimal negative crop impact).
    Factors in temperature stress, UV sensitivity of bio-pesticides, cold stress and pollination bans.
    """
    score = 100.0
    temp = float(row.get("temperature_2m", 0))
    humidity = float(row.get("relative_humidity_2m", 100))
    cloud_cover = float(row.get("cloud_cover", 0))
    wind = float(row.get("wind_speed_10m", 0))
    gust = float(row.get("wind_gusts_10m", 0))
    hour = int(row.get("hour", 12))

    stage_code = crop_stage.get("stage_code") if crop_stage else "TILLERING"
    uv_sens = pesticide.get("uv_sensitivity", False) if pesticide else False
    mechanism = pesticide.get("action_mechanism", "SYSTEMIC") if pesticide else "SYSTEMIC"

    # 1. Heat stress
    if temp >= 35.0:
        score -= 25.0
    # 2. UV Degradation risk for bio-pesticides
    if temp >= 32.0 and uv_sens and cloud_cover < 50.0:
        score -= 30.0
    # 3. Cold stress for systemic pesticides
    if temp < 20.0 and mechanism == "SYSTEMIC":
        score -= 20.0
    # 4. Wind spray drift risk at critical stages
    if wind >= 18.0 and stage_code in ["TILLERING", "BOOTING"]:
        score -= 15.0
    # 5. Lodging risk due to wind gust during grain filling
    if gust >= 25.0 and stage_code == "GRAIN_FILLING":
        score -= 25.0
    # 6. Pollination hard ban hours
    if crop_stage:
        ban_start = crop_stage.get("hard_ban_start_hour")
        ban_end = crop_stage.get("hard_ban_end_hour")
        if ban_start is not None and ban_end is not None:
            if ban_start <= hour <= ban_end:
                score = 0.0

    return max(0.0, min(100.0, score))


def calculate_spray_quality_score(
    row: pd.Series,
    pesticide: dict[str, Any] | None = None,
    flow_rate_l_ha: float = 15.0,
) -> float:
    """
    Calculate Spray Quality Score (0-100). Higher is better.
    Factors in wind drift, droplet evaporation under heat/dry, rain washout and formulation issues.
    """
    score = 100.0
    temp = float(row.get("temperature_2m", 0))
    humidity = float(row.get("relative_humidity_2m", 100))
    wind = float(row.get("wind_speed_10m", 0))
    rain = float(row.get("precipitation", 0))
    rain_prob = float(row.get("precipitation_probability", 0))

    mechanism = pesticide.get("action_mechanism", "SYSTEMIC") if pesticide else "SYSTEMIC"
    formulation = pesticide.get("common_formulation", "EC") if pesticide else "EC"
    washout_hours = pesticide.get("rain_washout_hours", 2) if pesticide else 2

    # 1. Rain Washout Risk (Contact pesticides are heavily washed out)
    if (rain > 0.0 or rain_prob >= 50.0) and mechanism == "CONTACT":
        score -= 50.0
        if rain >= 2.0:
            score -= 30.0

    # 2. Droplet Evaporation Risk (High Temp & Low Humidity)
    if temp >= 35.0 and humidity < 55.0:
        score -= 30.0

    # 3. Dilution risk by excessive dew/high humidity
    if humidity > 90.0:
        score -= 15.0

    # 4. Wind Drift Risk
    if wind >= 15.0:
        score -= 20.0
        if wind >= 25.0:
            score -= 20.0

    # 5. Low flow rate risk for WP/SC formulations (clogging)
    if formulation in ["WP", "SC"] and flow_rate_l_ha < 15.0:
        score -= 15.0

    return max(0.0, min(100.0, score))


def get_awd_recommendation(
    current_water_level: float,
    awd_threshold: float,
    future_precipitation_24h: float,
) -> dict[str, Any]:
    """Calculate AWD Irrigation Recommendation based on water level and forecast rain."""
    if current_water_level > awd_threshold:
        return {
            "action": "KEEP_DRYING",
            "explanation": f"Mực nước ngầm ({current_water_level:.1f} cm) đang ở mức an toàn (trên ngưỡng {awd_threshold:.1f} cm). Tiếp tục phơi ruộng tiết kiệm nước."
        }
    
    if future_precipitation_24h >= 20.0:
        return {
            "action": "DELAY_PUMP",
            "explanation": f"Nước đã tụt xuống {current_water_level:.1f} cm (dưới ngưỡng {awd_threshold:.1f} cm), nhưng dự báo có mưa lớn ({future_precipitation_24h:.1f} mm) trong 24h tới. Hoãn bơm nước để tận dụng nước mưa."
        }
    else:
        return {
            "action": "START_PUMP",
            "explanation": f"Nước đã tụt xuống {current_water_level:.1f} cm (dưới ngưỡng {awd_threshold:.1f} cm) và trời ít mưa ({future_precipitation_24h:.1f} mm). Cần khởi động trạm bơm ngay."
        }


def calculate_flyability_score(
    row: pd.Series,
    thresholds: DecisionThresholds = THRESHOLDS,
    unsafe_weather_codes: set[int] | frozenset[int] = UNSAFE_WEATHER_CODES,
    drone_profile: dict[str, Any] | None = None,
) -> float:
    """Score the safety conditions, factored dynamically by drone limits if available."""
    max_w = float(drone_profile.get("max_wind_resistance_kph", thresholds.max_wind_speed)) if drone_profile else thresholds.max_wind_speed
    max_g = float(drone_profile.get("max_gust_resistance_kph", thresholds.max_wind_gust)) if drone_profile else thresholds.max_wind_gust

    checks = {
        "wind": float(row.get("wind_speed_10m", 0)) <= max_w,
        "gust": float(row.get("wind_gusts_10m", 0)) <= max_g,
        "rain": float(row.get("precipitation", 0)) <= thresholds.max_rain_hourly,
        "rain_prob": float(row.get("precipitation_probability", 0)) <= thresholds.max_rain_probability,
        "cloud": float(row.get("cloud_cover", 0)) <= thresholds.max_cloud_cover,
        "visibility": float(row.get("visibility", thresholds.min_visibility)) >= thresholds.min_visibility,
        "weather": int(row.get("weather_code", 0)) not in unsafe_weather_codes,
        "temperature": float(row.get("temperature_2m", 0)) <= thresholds.max_safe_temperature,
    }
    return round(sum(float(checks[name]) * weight for name, weight in SCORE_WEIGHTS.items()), 4)


def derive_risk_level(
    row: pd.Series,
    action: str | None = None,
    thresholds: DecisionThresholds = THRESHOLDS,
    unsafe_weather_codes: set[int] | frozenset[int] = UNSAFE_WEATHER_CODES,
    drone_profile: dict[str, Any] | None = None,
    crop_stage: dict[str, Any] | None = None,
) -> str:
    action = action or str(row.get("decision_action", derive_decision_action(row, thresholds, unsafe_weather_codes, drone_profile, crop_stage)))
    if action in {"LOCK_SPRAY", "RETURN_TO_CHARGING"}:
        return "HIGH"
    if action == "DELAY_FLIGHT":
        return "MEDIUM"

    score = float(row.get("flyability_score", calculate_flyability_score(row, thresholds, unsafe_weather_codes, drone_profile)))
    if score < 0.4:
        return "HIGH"
    if score < 0.7:
        return "MEDIUM"
    return "LOW"


def derive_decision_action(
    row: pd.Series,
    thresholds: DecisionThresholds = THRESHOLDS,
    unsafe_weather_codes: set[int] | frozenset[int] = UNSAFE_WEATHER_CODES,
    drone_profile: dict[str, Any] | None = None,
    crop_stage: dict[str, Any] | None = None,
) -> str:
    wind = float(row.get("wind_speed_10m", 0))
    gust = float(row.get("wind_gusts_10m", 0))
    rain = float(row.get("precipitation", 0))
    rain_prob = float(row.get("precipitation_probability", 0))
    weather_code = int(row.get("weather_code", 0))
    temp = float(row.get("temperature_2m", 0))
    hour = int(row.get("hour", 12))

    max_w = float(drone_profile.get("max_wind_resistance_kph", thresholds.max_wind_speed)) if drone_profile else thresholds.max_wind_speed
    max_g = float(drone_profile.get("max_gust_resistance_kph", thresholds.max_wind_gust)) if drone_profile else thresholds.max_wind_gust

    # 1. Hard Ban Hours (Pollination time)
    if crop_stage:
        ban_start = crop_stage.get("hard_ban_start_hour")
        ban_end = crop_stage.get("hard_ban_end_hour")
        if ban_start is not None and ban_end is not None:
            if ban_start <= hour <= ban_end:
                return "LOCK_SPRAY"

    # 2. Hard limits: Rain or dangerous weather
    if rain > thresholds.max_rain_hourly or rain_prob > thresholds.max_rain_probability or weather_code in unsafe_weather_codes:
        return "RETURN_TO_CHARGING"
    
    # 3. Wind resistance exceedance
    if gust > max_g or wind > max_w:
        return "LOCK_SPRAY"
        
    # 4. Moderate temperature issues
    if temp > thresholds.max_safe_temperature:
        return "DELAY_FLIGHT"
        
    return "TAKE_OFF"


def add_decision_columns(
    df: pd.DataFrame,
    thresholds: DecisionThresholds = THRESHOLDS,
    unsafe_weather_codes: set[int] | frozenset[int] = UNSAFE_WEATHER_CODES,
    drone_profile: dict[str, Any] | None = None,
    crop_stage: dict[str, Any] | None = None,
    pesticide: dict[str, Any] | None = None,
) -> pd.DataFrame:
    enriched = df.copy()
    enriched["crop_condition"] = enriched.apply(infer_crop_condition, axis=1)
    enriched["flyability_score"] = enriched.apply(
        lambda row: calculate_flyability_score(row, thresholds, unsafe_weather_codes, drone_profile),
        axis=1,
    )
    enriched["decision_action"] = enriched.apply(
        lambda row: derive_decision_action(row, thresholds, unsafe_weather_codes, drone_profile, crop_stage),
        axis=1,
    )
    # Apply bootstrap rule
    enriched = apply_bootstrap_rule(enriched)
    
    enriched["risk_level"] = enriched.apply(
        lambda row: derive_risk_level(row, str(row["decision_action"]), thresholds, unsafe_weather_codes, drone_profile, crop_stage),
        axis=1,
    )
    enriched["dynamic_flow_rate_pct"] = enriched.apply(
        lambda row: calculate_dynamic_flow_rate(row, thresholds, crop_stage),
        axis=1,
    )
    enriched["estimated_damage_cost_usd"] = (
        enriched["decision_action"].isin(["LOCK_SPRAY", "RETURN_TO_CHARGING"]).astype(int)
        * HARDWARE_DAMAGE_COST_USD
    )
    return enriched


def apply_bootstrap_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bootstrap Rule: Minimum of 2 consecutive GREEN (TAKE_OFF) slots.
    Any single isolated TAKE_OFF slot is demoted to DELAY_FLIGHT.
    """
    df = df.copy()
    actions = df["decision_action"].tolist()
    n = len(actions)
    for i in range(n):
        if actions[i] == "TAKE_OFF":
            left_ok = (i > 0 and actions[i-1] == "TAKE_OFF")
            right_ok = (i < n - 1 and actions[i+1] == "TAKE_OFF")
            if not (left_ok or right_ok):
                actions[i] = "DELAY_FLIGHT"
    df["decision_action"] = actions
    return df


def build_recommendation_text(
    row: pd.Series,
    action: str | None = None,
    thresholds: DecisionThresholds = THRESHOLDS,
    drone_profile: dict[str, Any] | None = None,
    crop_stage: dict[str, Any] | None = None,
) -> str:
    action = action or str(row.get("decision_action", "TAKE_OFF"))
    wind = float(row.get("wind_speed_10m", 0))
    gust = float(row.get("wind_gusts_10m", 0))
    rain_prob = float(row.get("precipitation_probability", 0))
    temp = float(row.get("temperature_2m", 0))
    flow = float(row.get("dynamic_flow_rate_pct", calculate_dynamic_flow_rate(row, thresholds, crop_stage)))

    max_w = float(drone_profile.get("max_wind_resistance_kph", thresholds.max_wind_speed)) if drone_profile else thresholds.max_wind_speed
    max_g = float(drone_profile.get("max_gust_resistance_kph", thresholds.max_wind_gust)) if drone_profile else thresholds.max_wind_gust

    if action == "TAKE_OFF":
        return (
            f"TAKE_OFF: Điều kiện bay đạt yêu cầu an toàn. Gió {wind:.1f} km/h, "
            f"gió giật {gust:.1f} km/h, xác suất mưa {rain_prob:.0f}%. "
            f"Đề xuất lưu lượng phun {flow:.1f} L/ha."
        )
    if action == "LOCK_SPRAY":
        hour = int(row.get("hour", 12))
        if crop_stage:
            ban_start = crop_stage.get("hard_ban_start_hour")
            ban_end = crop_stage.get("hard_ban_end_hour")
            if ban_start is not None and ban_end is not None and ban_start <= hour <= ban_end:
                return f"LOCK_SPRAY: Khóa bay do thời gian {hour}h00 thuộc giờ thụ phấn lúa ({ban_start}h00-{ban_end}h00), tránh downwash làm rụng phấn hoa gây lép hạt."
        
        return (
            f"LOCK_SPRAY: Khóa lệnh phun vì gió/gió giật vượt ngưỡng an toàn thiết bị "
            f"({wind:.1f}/{gust:.1f} km/h vượt giới hạn {max_w:.1f}/{max_g:.1f} km/h). Tránh gió giạt thuốc."
        )
    if action == "RETURN_TO_CHARGING":
        return (
            f"RETURN_TO_CHARGING: Thời tiết mưa hoặc nguy hiểm, xác suất mưa {rain_prob:.0f}%. "
            "Đưa drone về trạm sạc khẩn cấp để tránh hỏng thiết bị điện tử."
        )
    return (
        f"DELAY_FLIGHT: Tạm hoãn bay do nhiệt độ cao {temp:.1f}°C vượt quá giới hạn an toàn {thresholds.max_safe_temperature}°C "
        f"hoặc khả năng mưa {rain_prob:.0f}%. Đợi khung giờ mát mẻ hơn."
    )


def recommend_best_slot(
    df: pd.DataFrame,
    location_name: str | None = None,
    drone_profile: dict[str, Any] | None = None,
    crop_stage: dict[str, Any] | None = None,
    pesticide: dict[str, Any] | None = None,
) -> pd.Series | None:
    candidate_df = add_decision_columns(df, thresholds=THRESHOLDS, drone_profile=drone_profile, crop_stage=crop_stage, pesticide=pesticide)
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


