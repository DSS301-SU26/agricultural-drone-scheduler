"""
DSS decision rules used to label training data and explain model outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

import pandas as pd


HARDWARE_DAMAGE_COST_USD = 2_000
BASE_SPRAY_LITERS = 100.0
AGRICULTURAL_LOCATIONS = {
    "An Giang",
    "Can Tho",
    "Dong Thap",
    "Long An",
    "Tien Giang",
    "Ho Chi Minh",
    "Ha Noi",
}


@dataclass(frozen=True)
class DecisionThresholds:
    max_wind_speed: float = 28.8  # Default baseline (T30)
    max_wind_gust: float = 28.8   # Default baseline (T30)
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

MODEL_FEATURES = [
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
    "soil_moisture_0_to_7cm",
    # New joined features:
    "max_wind_resistance_kph",
    "max_gust_resistance_kph",
    "uv_sensitivity",
    "rain_washout_hours",
    # Crop stage features
    "crop_stage_SEEDLING",
    "crop_stage_TILLERING",
    "crop_stage_BOOTING",
    "crop_stage_GRAIN_FILLING",
]

RISK_WEIGHTS = {
    "FLY": 0,
    "DELAY": 1,
    "LOCK_SPRAY": 2,
    "NO_FLY": 3
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
    """
    if crop_stage:
        min_f = float(crop_stage.get("flow_rate_min_l_ha", 15.0))
        max_f = float(crop_stage.get("flow_rate_max_l_ha", 25.0))
    else:
        min_f = float(row.get("flow_rate_min_l_ha", 15.0))
        max_f = float(row.get("flow_rate_max_l_ha", 25.0))

    temp = float(row.get("temperature_2m", 0))
    humidity = float(row.get("relative_humidity_2m", 100))
    
    if temp >= 35.0 and humidity < 50.0:
        return round(max_f, 1)
    
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

    stage_code = crop_stage.get("stage_code") if crop_stage else str(row.get("crop_stage", "TILLERING"))
    uv_sens = pesticide.get("uv_sensitivity", False) if pesticide else bool(row.get("uv_sensitivity", False))
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
    else:
        ban_start = 8 if stage_code in ["BOOTING", "GRAIN_FILLING"] else None
        ban_end = 11 if stage_code in ["BOOTING", "GRAIN_FILLING"] else None
        
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
    washout_hours = pesticide.get("rain_washout_hours", 2) if pesticide else float(row.get("rain_washout_hours", 2))

    # 1. Rain Washout Risk
    if (rain > 0.0 or rain_prob >= 50.0) and mechanism == "CONTACT":
        score -= 50.0
        if rain >= 2.0:
            score -= 30.0
    elif (rain > 0.0 or rain_prob >= 50.0) and washout_hours > 0:
        score -= 30.0

    # 2. Droplet Evaporation Risk
    if temp >= 35.0 and humidity < 55.0:
        score -= 30.0

    # 3. Dilution risk
    if humidity > 90.0:
        score -= 15.0

    # 4. Wind Drift Risk
    if wind >= 15.0:
        score -= 20.0
        if wind >= 25.0:
            score -= 20.0

    return max(0.0, min(100.0, score))


def get_awd_recommendation(
    current_water_level: float,
    awd_threshold: float,
    future_precipitation_24h: float,
) -> dict[str, Any]:
    if current_water_level > awd_threshold:
        return {
            "action": "KEEP_DRYING",
            "explanation": f"Mực nước ngầm ({current_water_level:.1f} cm) đang ở mức an toàn. Tiếp tục phơi ruộng tiết kiệm nước."
        }
    if future_precipitation_24h >= 20.0:
        return {
            "action": "DELAY_PUMP",
            "explanation": f"Nước đã tụt xuống {current_water_level:.1f} cm, dự báo có mưa lớn ({future_precipitation_24h:.1f} mm). Hoãn bơm nước để tận dụng nước mưa."
        }
    return {
        "action": "START_PUMP",
        "explanation": f"Nước đã tụt xuống {current_water_level:.1f} cm và trời ít mưa ({future_precipitation_24h:.1f} mm). Khởi động trạm bơm."
    }


def calculate_flyability_score(
    row: pd.Series,
    thresholds: DecisionThresholds = THRESHOLDS,
    unsafe_weather_codes: set[int] | frozenset[int] = UNSAFE_WEATHER_CODES,
    drone_profile: dict[str, Any] | None = None,
) -> float:
    if drone_profile:
        max_w = float(drone_profile.get("max_wind_resistance_kph", thresholds.max_wind_speed))
        max_g = float(drone_profile.get("max_gust_resistance_kph", thresholds.max_wind_gust))
    else:
        max_w = float(row.get("max_wind_resistance_kph", thresholds.max_wind_speed))
        max_g = float(row.get("max_gust_resistance_kph", thresholds.max_wind_gust))

    v_wind = float(row.get("wind_speed_10m", 0))
    v_gust = float(row.get("wind_gusts_10m", 0))
    v_rain = float(row.get("precipitation", 0))
    v_rain_prob = float(row.get("precipitation_probability", 0))
    v_cloud = float(row.get("cloud_cover", 0))
    v_vis = float(row.get("visibility", thresholds.min_visibility))
    v_weather = int(row.get("weather_code", 0))
    v_temp = float(row.get("temperature_2m", 0))

    def get_factor(val, threshold, tolerance, is_max=True, optimal=0.0):
        if is_max:
            if val <= optimal: return 1.0
            if val >= threshold + tolerance: return 0.0
            if val <= threshold:
                if threshold == optimal: return 1.0
                return 1.0 - 0.2 * ((val - optimal) / (threshold - optimal))
            return 0.8 - 0.8 * ((val - threshold) / tolerance)
        else:
            if val >= optimal: return 1.0
            if val <= threshold - tolerance: return 0.0
            if val >= threshold:
                if optimal == threshold: return 1.0
                return 0.8 + 0.2 * ((val - threshold) / (optimal - threshold))
            return 0.8 * ((val - (threshold - tolerance)) / tolerance)

    factors = {
        "wind": get_factor(v_wind, max_w, 10.0, optimal=5.0),
        "gust": get_factor(v_gust, max_g, 15.0, optimal=10.0),
        "rain": get_factor(v_rain, thresholds.max_rain_hourly, 5.0, optimal=0.0),
        "rain_prob": get_factor(v_rain_prob, thresholds.max_rain_probability, 30.0, optimal=10.0),
        "cloud": get_factor(v_cloud, thresholds.max_cloud_cover, 40.0, optimal=20.0),
        "visibility": get_factor(v_vis, thresholds.min_visibility, thresholds.min_visibility, is_max=False, optimal=10000.0),
        "weather": 0.0 if v_weather in unsafe_weather_codes else 1.0,
        "temperature": get_factor(v_temp, thresholds.max_safe_temperature, 5.0, optimal=25.0),
    }

    base_score = sum(factors[name] * weight for name, weight in SCORE_WEIGHTS.items())
    
    # Critical failure multiplier: Thay vì chốt cứng 40%, nhân với hệ số suy giảm tuyến tính
    critical_min = min(factors["wind"], factors["gust"], factors["rain"], factors["rain_prob"], factors["weather"])
    
    if critical_min < 1.0:
        # Nếu critical_min = 0 (vượt giới hạn an toàn rất xa), hệ số phạt là 0.20
        # Nếu critical_min = 0.9 (vừa chạm ngưỡng), hệ số phạt là 0.92
        penalty_multiplier = 0.20 + 0.80 * critical_min
        base_score = base_score * penalty_multiplier

    # Hard cap the flyability score if physical safety thresholds are breached
    # If any threshold is breached, the rule engine will output NO_FLY or DELAY.
    # We must ensure the numerical score mathematically aligns with that decision.
    is_no_fly = False
    is_delay = False
    
    if v_wind > max_w or v_gust > max_g or v_rain > thresholds.max_rain_hourly or v_rain_prob > 90 or v_weather in unsafe_weather_codes:
        is_no_fly = True
    elif v_temp > thresholds.max_safe_temperature or v_vis < thresholds.min_visibility or v_rain_prob > thresholds.max_rain_probability:
        is_delay = True
        
    if is_no_fly:
        base_score = min(base_score, 0.39)
    elif is_delay:
        base_score = min(base_score, 0.69)
        
    return round(base_score, 4)


def derive_risk_level(
    row: pd.Series,
    action: str | None = None,
    thresholds: DecisionThresholds = THRESHOLDS,
    unsafe_weather_codes: set[int] | frozenset[int] = UNSAFE_WEATHER_CODES,
    drone_profile: dict[str, Any] | None = None,
    crop_stage: dict[str, Any] | None = None,
    pesticide: dict[str, Any] | None = None,
) -> str:
    action = action or str(row.get("decision_action", derive_decision_action(row, thresholds, unsafe_weather_codes, drone_profile, crop_stage, pesticide)))
    if action in {"LOCK_SPRAY", "NO_FLY"}:
        return "HIGH"
    if action == "DELAY":
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
    pesticide: dict[str, Any] | None = None,
) -> str:
    wind = float(row.get("wind_speed_10m", 0))
    gust = float(row.get("wind_gusts_10m", 0))
    rain = float(row.get("precipitation", 0))
    rain_prob = float(row.get("precipitation_probability", 0))
    weather_code = int(row.get("weather_code", 0))
    temp = float(row.get("temperature_2m", 0))
    visibility = float(row.get("visibility", thresholds.min_visibility))
    hour = int(row.get("hour", 0))
    
    if crop_stage:
        stage_code = crop_stage.get("stage_code")
        ban_start = crop_stage.get("hard_ban_start_hour")
        ban_end = crop_stage.get("hard_ban_end_hour")
    else:
        stage_code = str(row.get("crop_stage", ""))
        ban_start = 8 if stage_code in ["BOOTING", "GRAIN_FILLING"] else None
        ban_end = 11 if stage_code in ["BOOTING", "GRAIN_FILLING"] else None

    if drone_profile:
        max_w = float(drone_profile.get("max_wind_resistance_kph", thresholds.max_wind_speed))
        max_g = float(drone_profile.get("max_gust_resistance_kph", thresholds.max_wind_gust))
    else:
        max_w = float(row.get("max_wind_resistance_kph", thresholds.max_wind_speed))
        max_g = float(row.get("max_gust_resistance_kph", thresholds.max_wind_gust))

    if pesticide:
        uv_sens = pesticide.get("uv_sensitivity", False)
        rain_washout = float(pesticide.get("rain_washout_hours", 0))
    else:
        uv_sens = bool(row.get("uv_sensitivity", False))
        rain_washout = float(row.get("rain_washout_hours", 0))

    # 1. HARD LIMITS (NO_FLY)
    if wind > max_w or gust > max_g or rain > thresholds.max_rain_hourly or weather_code in unsafe_weather_codes:
        return "NO_FLY"

    # 2. LOCK SPRAY
    if ban_start is not None and ban_end is not None:
        if ban_start <= hour <= ban_end:
            return "LOCK_SPRAY"

    if uv_sens and temp >= 32.0:
        return "LOCK_SPRAY"
    
    if rain_prob > thresholds.max_rain_probability and rain_washout > 0:
        return "LOCK_SPRAY"
        
    if weather_code in [51, 53, 58, 59, 61, 62]:
        return "LOCK_SPRAY"

    # 3. DELAY
    if temp > thresholds.max_safe_temperature or visibility < thresholds.min_visibility:
        return "DELAY"

    # 4. FLY
    return "FLY"


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
        lambda row: derive_decision_action(row, thresholds, unsafe_weather_codes, drone_profile, crop_stage, pesticide),
        axis=1,
    )
    enriched = apply_bootstrap_rule(enriched)
    
    enriched["risk_level"] = enriched.apply(
        lambda row: derive_risk_level(row, str(row["decision_action"]), thresholds, unsafe_weather_codes, drone_profile, crop_stage, pesticide),
        axis=1,
    )
    enriched["dynamic_flow_rate_pct"] = enriched.apply(
        lambda row: calculate_dynamic_flow_rate(row, thresholds, crop_stage),
        axis=1,
    )
    enriched["estimated_damage_cost_usd"] = (
        enriched["decision_action"].isin(["LOCK_SPRAY", "NO_FLY"]).astype(int)
        * HARDWARE_DAMAGE_COST_USD
    )
    return enriched


def apply_bootstrap_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bootstrap Rule: Minimum of 2 consecutive GREEN (FLY) slots.
    Any single isolated FLY slot is demoted to DELAY.
    """
    df = df.copy()
    actions = df["decision_action"].tolist()
    n = len(actions)
    for i in range(n):
        if actions[i] == "FLY":
            left_ok = (i > 0 and actions[i-1] == "FLY")
            right_ok = (i < n - 1 and actions[i+1] == "FLY")
            if not (left_ok or right_ok):
                actions[i] = "DELAY"
    df["decision_action"] = actions
    return df


def build_recommendation_text(
    row: pd.Series,
    action: str,
    thresholds: DecisionThresholds,
    drone_profile: dict[str, Any] | None = None,
    crop_stage: dict[str, Any] | None = None,
    pesticide: dict[str, Any] | None = None,
    flyability_score: float | None = None,
) -> str:
    """Build a natural language explanation for the decision."""
    wind = float(row.get("wind_speed_10m", 0))
    gust = float(row.get("wind_gusts_10m", 0))
    rain_prob = float(row.get("precipitation_probability", 0))
    rain = float(row.get("precipitation", 0))
    temp = float(row.get("temperature_2m", 0))
    humidity = float(row.get("relative_humidity_2m", 0))
    visibility = float(row.get("visibility", 10000))
    hour = int(row.get("hour", 12))
    flow = float(row.get("dynamic_flow_rate_pct", calculate_dynamic_flow_rate(row, thresholds, crop_stage)))

    # --- Lấy thông tin ngữ cảnh ---
    drone_name = drone_profile.get("model_name", "drone") if drone_profile else "drone"
    drone_max_w = float(drone_profile.get("max_wind_resistance_kph", thresholds.max_wind_speed)) if drone_profile else thresholds.max_wind_speed
    drone_max_g = float(drone_profile.get("max_gust_resistance_kph", thresholds.max_wind_gust)) if drone_profile else thresholds.max_wind_gust

    pest_name = pesticide.get("active_ingredient", pesticide.get("trade_name", "thuốc")) if pesticide else "thuốc"
    pest_trade = pesticide.get("trade_name", "") if pesticide else ""
    pest_form = pesticide.get("common_formulation", "") if pesticide else ""
    pest_washout = int(pesticide.get("rain_washout_hours", 0)) if pesticide else 0
    pest_uv = pesticide.get("uv_sensitivity", False) if pesticide else False
    pest_label = pest_name + (f" ({pest_trade})" if pest_trade and pest_trade != pest_name else "")

    STAGE_NAMES = {"SEEDLING": "Mạ", "TILLERING": "Đẻ nhánh", "BOOTING": "Làm đòng-Trổ", "GRAIN_FILLING": "Chín"}
    if crop_stage:
        stage_code = crop_stage.get("stage_code", "")
        ban_start = crop_stage.get("hard_ban_start_hour")
        ban_end = crop_stage.get("hard_ban_end_hour")
    else:
        stage_code = str(row.get("crop_stage", ""))
        ban_start = 8 if stage_code in ["BOOTING", "GRAIN_FILLING"] else None
        ban_end = 11 if stage_code in ["BOOTING", "GRAIN_FILLING"] else None
    stage_label = STAGE_NAMES.get(stage_code, stage_code)

    # === FLY ===
    if action == "FLY":
        parts = [f"Điều kiện bay an toàn — Gió {wind:.1f} km/h (ngưỡng {drone_name}: {drone_max_w:.0f} km/h), nhiệt độ {temp:.1f}°C, xác suất mưa chỉ {rain_prob:.0f}%."]
        if stage_code == "SEEDLING":
            parts.append(f"Lúa giai đoạn {stage_label}: nên bay cao 2.5-3m và tốc độ nhanh để tránh downwash dập mạ non.")
        elif stage_code in ["BOOTING", "GRAIN_FILLING"]:
            parts.append(f"Lúa giai đoạn {stage_label}: nên bay thấp 1.5-2m để thuốc xuyên được tán lá dày.")
        elif stage_code == "TILLERING":
            parts.append(f"Lúa giai đoạn {stage_label}: tán bắt đầu khép, bay ở độ cao 2-2.5m là phù hợp.")
        if pest_label and pest_label != "thuốc":
            parts.append(f"Thuốc {pest_label}" + (f" (dạng {pest_form})" if pest_form else "") + f" — đề xuất lưu lượng xả {flow:.1f} L/ha.")
        else:
            parts.append(f"Đề xuất lưu lượng xả {flow:.1f} L/ha.")
        return " ".join(parts)

    # === LOCK_SPRAY ===
    if action == "LOCK_SPRAY":
        # Lý do 1: Giờ thụ phấn
        if ban_start is not None and ban_end is not None and ban_start <= hour <= ban_end:
            return (
                f"Khóa lệnh phun — Khung giờ {hour}h00 nằm trong giờ thụ phấn lúa "
                f"({ban_start}h00-{ban_end}h00, giai đoạn {stage_label}). "
                "Cánh quạt drone tạo luồng gió downwash mạnh có thể làm rụng phấn hoa, gây lép hạt và giảm năng suất nghiêm trọng. "
                f"→ Đề xuất: Chờ bay sau {ban_end + 1}h00 hoặc trước {ban_start}h00."
            )
        # Lý do 2: UV nhạy cảm
        if pest_uv and temp >= 32.0:
            return (
                f"Khóa lệnh phun — Nhiệt độ hiện tại {temp:.1f}°C quá cao khi dùng thuốc {pest_label}"
                + (f" (dạng {pest_form})" if pest_form else "")
                + " là loại nhạy cảm với tia UV. "
                "Dưới nắng gắt và nhiệt độ cao, hoạt chất sẽ bị phân hủy quang hóa nhanh chóng, làm giảm hiệu lực thuốc và lãng phí chi phí. "
                "→ Đề xuất: Phun vào sáng sớm (5h-7h) hoặc chiều muộn (16h-18h) khi trời mát hơn."
            )
        # Lý do 3: Mưa phùn nhẹ (Drizzle)
        weather_code = int(row.get("weather_code", 0))
        if weather_code in [51, 53, 58, 59, 61, 62]:
            return (
                f"Khóa lệnh phun — Đang có mưa phùn hoặc mưa nhỏ (mã thời tiết {weather_code}). "
                "Drone vẫn có thể bay an toàn, nhưng phun lúc này thuốc sẽ bị loãng hoặc rửa trôi, gây lãng phí và giảm hiệu lực. "
                "→ Đề xuất: Theo dõi timeline bên dưới để đợi trời tạnh."
            )
        
        # Lý do 4: Mưa rửa trôi (Dự báo mưa)
        washout_note = ""
        if pest_washout > 0:
            washout_note = (
                f" Thuốc {pest_label}" + (f" (dạng {pest_form})" if pest_form else "")
                + f" cần tối thiểu {pest_washout} giờ khô ráo sau khi phun mới bám dính hiệu quả trên lá."
            )
        return (
            f"Khóa lệnh phun — Xác suất mưa cao ({rain_prob:.0f}%), dự báo có mưa trong vòng 2-4 giờ tới."
            + washout_note
            + " Phun lúc này thuốc sẽ bị rửa trôi, gây lãng phí và ô nhiễm nguồn nước."
            + " → Đề xuất: Chờ đợi khung giờ trời tạnh ổn định trong timeline bên dưới."
        )

    # === NO_FLY ===
    if action == "NO_FLY":
        reasons = []
        if wind > drone_max_w:
            reasons.append(f"gió {wind:.1f} km/h vượt giới hạn chịu đựng của {drone_name} ({drone_max_w:.0f} km/h)")
        if gust > drone_max_g:
            reasons.append(f"gió giật {gust:.1f} km/h vượt ngưỡng an toàn ({drone_max_g:.0f} km/h)")
        if rain > thresholds.max_rain_hourly:
            reasons.append(f"lượng mưa {rain:.1f} mm/h quá lớn")
        if rain_prob > 90:
            reasons.append(f"xác suất mưa rất cao ({rain_prob:.0f}%)")
        
        weather_code = int(row.get("weather_code", 0))
        if weather_code in [95, 96, 99]:
            reasons.append(f"có dông lốc, sấm sét nguy hiểm (mã thời tiết {weather_code})")
        elif weather_code in [65, 67, 82, 86]:
            reasons.append(f"mưa rất to, mưa rào mạnh (mã thời tiết {weather_code})")
        elif weather_code in [55, 63, 66, 73, 75, 77, 81]:
            reasons.append(f"mưa nặng hạt, thời tiết xấu (mã thời tiết {weather_code})")
        elif not reasons:
            borderline = []
            if gust > drone_max_g * 0.7:
                borderline.append(f"gió giật {gust:.1f} km/h (ngưỡng {drone_max_g:.0f} km/h)")
            elif wind > drone_max_w * 0.7:
                borderline.append(f"gió thổi đều {wind:.1f} km/h (ngưỡng {drone_max_w:.0f} km/h)")
            if temp > thresholds.max_safe_temperature * 0.85:
                borderline.append(f"nhiệt độ cao {temp:.1f}°C")
            if rain_prob >= 40:
                borderline.append(f"xác suất mưa {rain_prob:.0f}%")
            if humidity > 85:
                borderline.append(f"độ ẩm cao {humidity:.0f}%")
            
            # Luôn hiện thông số cơ bản nếu điểm quá thấp
            base_stats = f"Hiện tại: Gió giật {gust:.1f} km/h, Nhiệt độ {temp:.1f}°C, Mưa {rain_prob:.0f}%."
            if borderline:
                reason_text = "sự kết hợp của các yếu tố bất lợi: " + ", ".join(borderline)
            else:
                reason_text = "điều kiện tổng hợp vượt ngưỡng an toàn của mô hình AI"
            
            score_str = f" (điểm an toàn: {flyability_score*100:.0f}%)" if flyability_score is not None else ""
            reasons.append(f"{reason_text}{score_str}. {base_stats}")

        reason_text = ", ".join(reasons)
        return (
            f"Cấm cất cánh — Phát hiện {reason_text}. "
            f"Nếu cố bay trong điều kiện này, {drone_name} có nguy cơ mất kiểm soát, hư hỏng phần cứng hoặc rơi rớt. "
            "→ Đề xuất: Tuyệt đối không bay. Theo dõi timeline để tìm khung giờ an toàn hơn."
        )

    # === DELAY ===
    reasons = []
    if temp > thresholds.max_safe_temperature:
        reasons.append(f"nhiệt độ {temp:.1f}°C quá cao (ngưỡng an toàn: {thresholds.max_safe_temperature:.0f}°C), thuốc dễ bốc hơi nhanh và giảm hiệu lực")
    if visibility < thresholds.min_visibility:
        reasons.append(f"tầm nhìn chỉ {visibility:.0f}m (yêu cầu tối thiểu {thresholds.min_visibility:.0f}m để giữ tầm quan sát trực tiếp VLOS)")
    if humidity < 45:
        reasons.append(f"độ ẩm {humidity:.0f}% quá thấp, giọt phun sẽ bị co lại và bay hơi trước khi chạm lá")
    if not reasons:
        reasons.append("một số chỉ số thời tiết chưa đạt ngưỡng lý tưởng")

    reason_text = "; ".join(reasons)
    return (
        f"Tạm hoãn chuyến bay — Phát hiện {reason_text}. "
        "Vẫn có thể bay nhưng hiệu quả phun thuốc sẽ giảm đáng kể. "
        "→ Đề xuất: Kiểm tra lại khung giờ kế tiếp trên timeline."
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

    safe_slots = candidate_df[candidate_df["decision_action"] == "FLY"].copy()
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
