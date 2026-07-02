"""
Ngu canh nghiep vu cho tang rules: ho so drone, thuoc, giai doan lua.

Cac dataclass nay phan anh dung 3 bang seed (drone_profiles / pesticide_specs /
crop_profile). Kem theo registry mac dinh de rules chay duoc NGAY ca khi chua co DB
(phuc vu unit test va giai doan chua ket noi Supabase).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DroneProfile:
    model_name: str
    max_wind_resistance_kph: float
    max_gust_resistance_kph: float
    tank_capacity_liters: int
    nozzle_technology: str      # 'PRESSURE' | 'CENTRIFUGAL'
    ingress_protection: str


@dataclass(frozen=True)
class PesticideSpec:
    trade_name: str
    active_ingredient: str
    action_mechanism: str       # 'SYSTEMIC' | 'CONTACT'
    common_formulation: str     # WP/EC/SC/SG
    rain_washout_hours: int
    uv_sensitivity: bool


@dataclass(frozen=True)
class CropStage:
    stage_code: str             # SEEDLING/TILLERING/BOOTING/GRAIN_FILLING
    stage_name: str
    kc_value: float
    opt_flight_alt_min: float
    opt_flight_alt_max: float
    opt_flight_speed_min: float
    opt_flight_speed_max: float
    flow_rate_min_l_ha: float
    flow_rate_max_l_ha: float
    awd_threshold_cm: float
    hard_ban_start_hour: int | None
    hard_ban_end_hour: int | None


# --- Registry mac dinh (giong app/db/seed.sql) ------------------------------

DRONES: dict[str, DroneProfile] = {
    "DJI_T30": DroneProfile("DJI_T30", 28.8, 36.0, 30, "PRESSURE", "IP67"),
    "DJI_T50": DroneProfile("DJI_T50", 21.6, 30.0, 40, "CENTRIFUGAL", "IPX6K"),
    "XAG_P100_PRO": DroneProfile("XAG_P100_PRO", 36.0, 46.0, 50, "CENTRIFUGAL", "IPX6K"),
}

PESTICIDES: dict[str, PesticideSpec] = {
    "Tricyclazole": PesticideSpec("Beam", "Tricyclazole", "SYSTEMIC", "WP", 4, False),
    "Abamectin": PesticideSpec("Agri-Mek", "Abamectin", "CONTACT", "EC", 2, True),
    "Hexaconazole": PesticideSpec("Anvil", "Hexaconazole", "SYSTEMIC", "SC", 3, False),
}

CROP_STAGES: dict[str, CropStage] = {
    "SEEDLING": CropStage("SEEDLING", "Ma", 1.05, 2.5, 3.0, 6.0, 7.0, 10.0, 15.0, -15.0, None, None),
    "TILLERING": CropStage("TILLERING", "De nhanh", 1.10, 2.0, 2.5, 5.0, 6.0, 15.0, 20.0, -15.0, None, None),
    "BOOTING": CropStage("BOOTING", "Lam dong-Tro", 1.20, 1.5, 2.0, 4.0, 5.0, 25.0, 30.0, -5.0, 8, 11),
    "GRAIN_FILLING": CropStage("GRAIN_FILLING", "Chin", 0.95, 2.5, 3.5, 5.0, 6.0, 15.0, 20.0, -15.0, None, None),
}


def get_drone(model_name: str) -> DroneProfile:
    return DRONES.get(model_name, DRONES["DJI_T30"])


def get_pesticide(active_ingredient: str | None) -> PesticideSpec | None:
    if not active_ingredient:
        return None
    return PESTICIDES.get(active_ingredient)


def get_crop_stage(stage_code: str | None) -> CropStage | None:
    if not stage_code:
        return None
    return CROP_STAGES.get(stage_code)
