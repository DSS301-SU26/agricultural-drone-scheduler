"""
Ma tran 13 tac nhan ra quyet dinh DUOC PHEP BAY / PHAI DUNG BAY (BRD §3.3).

`evaluate_flight_rules` gop:
  - Tac nhan khi tuong (nhiet, am, gio, gio giat, mua, tam nhin, may, ma WMO)
  - Rao chan co hoc theo drone (drone_limits)
  - Khung gio cam theo giai doan lua (growth_stage)
  - Rui ro thoi diem & rua troi theo thuoc (pesticide)
-> tra ve RuleEvaluation (FLY/DELAY/NO_FLY) kem chi tiet tung tac nhan cho XAI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context import CropStage, DroneProfile, PesticideSpec
from .drone_limits import evaluate_drone_limits
from .growth_stage import evaluate_stage_time_ban
from .mission_factors import MissionConfig, evaluate_mission_factors
from .pesticide import evaluate_pesticide_timing, evaluate_rain_washout
from .thresholds import DEFAULT_THRESHOLDS, WeatherThresholds
from .types import Decision, FactorResult, RuleEvaluation, Verdict, combine_verdicts


@dataclass
class RuleInput:
    """Goi input cho 1 khung gio danh gia."""
    weather: dict[str, Any]          # 1 dong weather_hourly
    hour: int
    drone: DroneProfile
    pesticide: PesticideSpec | None = None
    crop_stage: CropStage | None = None
    rain_prob_washout_window_pct: float | None = None  # xs mua cao nhat trong cua so rao la
    mission_config: MissionConfig | None = None        # pH/tro luc/dang thuoc/bec/tan la
    thresholds: WeatherThresholds = DEFAULT_THRESHOLDS


def _f(weather: dict[str, Any], key: str, default: float = 0.0) -> float:
    val = weather.get(key)
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


# --- Cac tac nhan khi tuong ------------------------------------------------

def _temperature(temp: float, rh: float, t: WeatherThresholds) -> FactorResult:
    if temp >= t.temp_stop:
        v, msg = Verdict.STOP_SPRAY, f"Nhiệt độ {temp:.1f}°C vượt ngưỡng {t.temp_stop:.0f}°C: Sốc nhiệt làm cháy lá và bốc hơi thuốc nhanh."
    elif temp > t.temp_allow_max:
        if temp <= t.temp_soft_max and rh >= t.rh_allow_min:
            v, msg = Verdict.WARN, f"Nhiệt độ {temp:.1f}°C hơi cao nhưng độ ẩm {rh:.0f}% đủ bù đắp."
        else:
            v, msg = Verdict.WARN, f"Nhiệt độ {temp:.1f}°C vượt {t.temp_allow_max:.0f}°C: Rủi ro bốc hơi giọt thuốc."
    elif temp < t.temp_allow_min:
        v, msg = Verdict.WARN, f"Nhiệt độ {temp:.1f}°C quá thấp: Cây hấp thụ thuốc lưu dẫn kém."
    else:
        v, msg = Verdict.ALLOW, f"Nhiệt độ {temp:.1f}°C lý tưởng."
    return FactorResult("temperature", v, round(temp, 1), msg)


def _humidity(rh: float, temp: float, wind_kph: float, t: WeatherThresholds) -> FactorResult:
    if rh > t.rh_high_stop:
        v, msg = Verdict.STOP_SPRAY, f"Độ ẩm {rh:.0f}% > {t.rh_high_stop:.0f}%: Sương mù đặc làm loãng thuốc, dễ ủ bệnh nấm."
    elif rh < t.rh_low_stop and (temp > t.temp_allow_max or wind_kph > t.wind_warn_kph):
        v, msg = Verdict.STOP_SPRAY, f"Độ ẩm {rh:.0f}% < {t.rh_low_stop:.0f}% (Quá khô): Giọt thuốc co lại và bay hơi trước khi chạm lá."
    elif rh < t.rh_allow_min or rh > t.rh_allow_max:
        v, msg = Verdict.WARN, f"Độ ẩm {rh:.0f}% nằm ngoài ngưỡng tối ưu."
    else:
        v, msg = Verdict.ALLOW, f"Độ ẩm {rh:.0f}% lý tưởng."
    return FactorResult("humidity", v, round(rh, 0), msg)


def _wind_speed(wind_kph: float, cfg: MissionConfig | None, t: WeatherThresholds) -> FactorResult:
    if wind_kph > t.wind_stop_kph:
        v, msg = Verdict.STOP_SPRAY, f"Gió {wind_kph:.1f} km/h vượt ngưỡng: Gây phát tán hóa chất mạnh sang các mương, ruộng lân cận."
    elif wind_kph > t.wind_warn_kph:
        mode = (cfg.nozzle_mode if cfg else None) or "COARSE"
        if mode in {"COARSE", "MEDIUM"}:
            v, msg = Verdict.ALLOW, f"Gió {wind_kph:.1f} km/h (Mức trung bình) nhưng béc phun {mode} đủ lớn để đảm bảo an toàn."
        else:
            v, msg = Verdict.WARN, f"Gió {wind_kph:.1f} km/h: Cần dùng béc phun hạt to để tránh trôi thuốc."
    else:
        v, msg = Verdict.ALLOW, f"Gió {wind_kph:.1f} km/h lý tưởng."
    return FactorResult("wind_speed", v, round(wind_kph, 1), msg)


def _gust(gust_kph: float, wind_kph: float, t: WeatherThresholds) -> FactorResult:
    # Chenh lech gio giat lon -> canh bao ha do cao (khong hard o day; hard nam o drone_limits)
    if gust_kph > t.gust_warn_kph:
        v, msg = Verdict.WARN, f"Gió giật {gust_kph:.1f} km/h: Rủi ro mất ổn định đường bay RTK, cần hạ độ cao."
    else:
        v, msg = Verdict.ALLOW, f"Gió giật {gust_kph:.1f} km/h trong ngưỡng ổn định."
    return FactorResult("wind_gust", v, round(gust_kph, 1), msg)


def _rain(precip_mm: float, rain_prob: float, weather_code: int, t: WeatherThresholds) -> FactorResult:
    if precip_mm > t.rain_hourly_stop_mm:
        v, msg = Verdict.STOP, f"Mưa lớn {precip_mm:.1f} mm/h: Chập mạch thiết bị drone và rửa trôi toàn bộ thuốc."
    elif weather_code in t.unsafe_wmo_codes:
        v, msg = Verdict.STOP, f"Cảnh báo mã thời tiết ({weather_code}): Bão / Mưa dông / Sương mù dày."
    elif rain_prob > t.rain_prob_stop_pct:
        v, msg = Verdict.STOP_SPRAY, f"Xác suất mưa {rain_prob:.0f}% quá cao: Nguy cơ rửa trôi thuốc ngay sau khi phun."
    elif rain_prob > t.rain_prob_warn_pct:
        v, msg = Verdict.WARN, f"Xác suất mưa {rain_prob:.0f}%: Cần theo dõi sát sao, chuẩn bị hoãn chuyến."
    else:
        v, msg = Verdict.ALLOW, f"Trời tạnh ráo, xác suất mưa thấp ({rain_prob:.0f}%)."
    return FactorResult("rain", v, round(precip_mm, 1), msg)


def _visibility(vis_m: float | None, t: WeatherThresholds) -> FactorResult:
    if vis_m is not None and vis_m < t.visibility_stop_m:
        return FactorResult("visibility", Verdict.STOP, round(vis_m, 0),
                            f"Tầm nhìn {vis_m:.0f}m < {t.visibility_stop_m:.0f}m: Mất tầm nhìn an toàn (VLOS).")
    return FactorResult("visibility", Verdict.ALLOW, round(vis_m, 0) if vis_m is not None else None,
                        "Tầm nhìn quang đãng, đảm bảo VLOS.")


def _cloud_cover(cloud_pct: float, pesticide: PesticideSpec | None, t: WeatherThresholds) -> FactorResult:
    # May 80-100% = khung vang cho thuoc sinh hoc nhay UV
    if pesticide and pesticide.uv_sensitivity and cloud_pct >= t.cloud_bio_golden_min:
        return FactorResult("cloud_cover", Verdict.ALLOW, round(cloud_pct, 0),
                            f"Mây che phủ {cloud_pct:.0f}%: Thời điểm vàng cho thuốc sinh học {pesticide.active_ingredient}.")
    return FactorResult("cloud_cover", Verdict.ALLOW, round(cloud_pct, 0), f"Mây che phủ {cloud_pct:.0f}%.")


# --- Ham gop chinh ----------------------------------------------------------

def evaluate_flight_rules(inp: RuleInput) -> RuleEvaluation:
    """Danh gia toan bo tac nhan -> 1 RuleEvaluation."""
    w, t = inp.weather, inp.thresholds
    temp = _f(w, "temperature_2m")
    rh = _f(w, "relative_humidity_2m", 100.0)
    wind = _f(w, "wind_speed_10m")
    gust = _f(w, "wind_gusts_10m")
    precip = _f(w, "precipitation")
    rain_prob = _f(w, "precipitation_probability")
    cloud = _f(w, "cloud_cover")
    vis = _f(w, "visibility", t.visibility_stop_m)
    wcode = int(_f(w, "weather_code"))

    factors: list[FactorResult] = [
        _temperature(temp, rh, t),
        _humidity(rh, temp, wind, t),
        _wind_speed(wind, inp.mission_config, t),
        _gust(gust, wind, t),
        _rain(precip, rain_prob, wcode, t),
        _visibility(vis, t),
        _cloud_cover(cloud, inp.pesticide, t),
    ]

    # Rao chan co hoc theo drone (hard)
    factors.extend(evaluate_drone_limits(wind, gust, inp.drone))

    # Khung gio cam theo giai doan lua
    if inp.crop_stage is not None:
        ban = evaluate_stage_time_ban(inp.hour, inp.crop_stage)
        if ban is not None:
            factors.append(ban)

    # Rui ro thoi diem & rua troi theo thuoc
    if inp.pesticide is not None:
        timing = evaluate_pesticide_timing(inp.hour, temp, cloud, inp.pesticide, t)
        if timing is not None:
            factors.append(timing)
        washout_prob = inp.rain_prob_washout_window_pct
        if washout_prob is None:
            washout_prob = rain_prob
        washout = evaluate_rain_washout(washout_prob, inp.pesticide, t)
        if washout is not None:
            factors.append(washout)

    # Cac tac nhan cau hinh nhiem vu (huong gio, pH, tro luc, dang thuoc, bec, tan la)
    factors.extend(evaluate_mission_factors(
        inp.weather, inp.pesticide, inp.crop_stage, inp.mission_config, t))

    decision = combine_verdicts(factors)
    return RuleEvaluation(decision=decision, factors=factors)
