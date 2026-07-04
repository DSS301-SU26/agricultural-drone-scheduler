"""
Nguong dinh luong cho ma tran 13 tac nhan (BRD §3.3).

Tat ca gia tri gio deu quy ve km/h de dong bo voi du lieu Open-Meteo
(1 m/s = 3.6 km/h). Cac nguong nay tunable qua decision-config sau nay.
"""
from __future__ import annotations

from dataclasses import dataclass

MS_TO_KPH = 3.6


@dataclass(frozen=True)
class WeatherThresholds:
    # --- Nhiet do (°C) ---
    temp_allow_min: float = 20.0
    temp_allow_max: float = 32.0
    temp_soft_max: float = 34.0          # canh bao mem den 34 neu RH>=55 & gio nhe
    temp_stop: float = 35.0              # >=35 -> STOP
    temp_abamectin_sun_stop: float = 32.0  # >32 khi Abamectin nang truc tiep -> STOP

    # --- Do am tuong doi (%) ---
    rh_allow_min: float = 55.0
    rh_allow_max: float = 90.0
    rh_low_stop: float = 45.0            # <45 kem nhiet cao/gio -> STOP
    rh_high_stop: float = 95.0           # >95 co suong/mua phun -> STOP

    # --- Gio trung binh (km/h) --- (3 m/s=10.8 ; 5 m/s=18.0)
    wind_ideal_kph: float = 10.8         # <3 m/s ly tuong
    wind_warn_kph: float = 10.8          # 3-5 m/s: chi bay khi giot tho
    wind_stop_kph: float = 18.0          # >5 m/s -> STOP (tan xa)

    # --- Gio giat (km/h) --- canh bao chenh lech gio giat
    gust_warn_kph: float = 25.2          # ~7 m/s

    # --- Mua ---
    rain_hourly_stop_mm: float = 2.0     # >2 mm/h -> STOP (rua troi/RTB)
    rain_prob_warn_pct: float = 50.0
    rain_prob_stop_pct: float = 60.0     # >60% -> STOP khoi tao nhiem vu

    # --- Tam nhin (m) ---
    visibility_stop_m: float = 1000.0    # <1000m -> STOP (VLOS)

    # --- May che phu (%) ---
    cloud_bio_golden_min: float = 80.0   # 80-100% = khung vang cho thuoc sinh hoc

    # --- Khung gio ---
    uv_ban_start_hour: int = 10          # 10-15h nang gat -> STOP cho thuoc nhay UV
    uv_ban_end_hour: int = 15

    # --- Ma WMO nguy hiem (suong mu day, mua/dong) ---
    unsafe_wmo_codes: frozenset[int] = frozenset(
        {45, 48, 55, 63, 65, 71, 80, 81, 82, 95, 96, 99}
    )


DEFAULT_THRESHOLDS = WeatherThresholds()


@dataclass(frozen=True)
class AWDThresholds:
    """Nguong cho chien luoc tuoi ngap kho xen ke (BRD §3.4)."""
    safe_water_level_cm: float = -15.0   # tut duoi -15cm -> can bom bu
    refill_target_cm: float = 5.0        # bom bu len +5cm
    rain_24h_hold_mm: float = 20.0       # mua du bao >20mm/24h -> hoan bom
    et0_extreme_mm: float = 6.0          # ET0 >6 mm/ngay -> canh bao han


DEFAULT_AWD_THRESHOLDS = AWDThresholds()
