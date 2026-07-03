"""
Cac tac nhan con lai cua ma tran 13 tac nhan §3.3 lien quan CAU HINH NHIEM VU
(khong phai thoi tiet thuan): huong gio, dang thuoc, pH nuoc, chat tro luc,
loai bec, mat do tan la.

Nhieu tac nhan la INPUT nguoi dung khai bao. Neu khong khai bao -> dung mac dinh
an toan (suy ra tu giai doan lua / gio) -> tra ALLOW/WARN, hiem khi STOP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context import CropStage, PesticideSpec
from .thresholds import DEFAULT_THRESHOLDS, WeatherThresholds
from .types import FactorResult, Verdict


@dataclass(frozen=True)
class MissionConfig:
    water_ph: float | None = None            # pH nuoc pha
    adjuvant_used: bool = False              # co dung chat tro luc
    adjuvant_compatible: bool = True         # da kiem tra tuong thich nhan
    formulation: str | None = None           # WP/SC/EC/SG (mac dinh lay tu thuoc)
    nozzle_mode: str | None = None           # FINE/MEDIUM/COARSE (mac dinh suy tu gio)
    canopy_density: str | None = None        # SPARSE/MEDIUM/DENSE (mac dinh suy tu giai doan)


# Suy mat do tan tu giai doan sinh truong
_STAGE_CANOPY = {
    "SEEDLING": "SPARSE", "TILLERING": "MEDIUM",
    "BOOTING": "DENSE", "GRAIN_FILLING": "DENSE",
}


def _wind_direction(weather: dict[str, Any], t: WeatherThresholds) -> FactorResult:
    wind = float(weather.get("wind_speed_10m", 0) or 0)
    wdir = weather.get("wind_direction_10m")
    if wind > t.wind_stop_kph and wdir is not None:
        return FactorResult("wind_direction", Verdict.WARN, round(float(wdir), 0),
                            f"Gio {wind:.1f} km/h huong {float(wdir):.0f}° - lap duong bay vuong goc, "
                            "tranh thoi ve ao/kenh/khu dan cu.")
    return FactorResult("wind_direction", Verdict.ALLOW, round(float(wdir), 0) if wdir is not None else None,
                        "Huong gio on dinh, drift trong tam kiem soat.")


def _formulation(cfg: MissionConfig, pesticide: PesticideSpec | None) -> FactorResult:
    form = cfg.formulation or (pesticide.common_formulation if pesticide else None)
    if form in {"WP", "SC", "SG"}:
        return FactorResult("formulation", Verdict.WARN, form,
                            f"Dang {form}: hoa tan truoc, loc luoi & khuay lien tuc tranh nghet bec (nuoc UAV it).")
    if form == "EC":
        return FactorResult("formulation", Verdict.WARN, form,
                            "Dang EC: dam bao nhu hoa on dinh, khong de tach lop trong binh nho.")
    return FactorResult("formulation", Verdict.ALLOW, form, "Dang thuoc phu hop phun UAV.")


def _water_ph(cfg: MissionConfig) -> FactorResult:
    if cfg.water_ph is None:
        return FactorResult("water_ph", Verdict.ALLOW, None, "Chua do pH - khuyen nghi pha pH 5.5-6.5.")
    ph = cfg.water_ph
    if ph > 7.5 or ph < 5.0:
        return FactorResult("water_ph", Verdict.STOP, ph,
                            f"pH {ph:.1f} ngoai an toan (5-7.5) - thuoc de thuy phan. Chinh pH truoc khi phun.")
    if not (5.5 <= ph <= 6.5):
        return FactorResult("water_ph", Verdict.WARN, ph, f"pH {ph:.1f} lech vung toi uu 5.5-6.5.")
    return FactorResult("water_ph", Verdict.ALLOW, ph, f"pH {ph:.1f} toi uu.")


def _adjuvant(cfg: MissionConfig) -> FactorResult:
    if cfg.adjuvant_used and not cfg.adjuvant_compatible:
        return FactorResult("adjuvant", Verdict.STOP, "incompatible",
                            "Chat tro luc chua ro tuong thich - nguy co pha nhu/chay la. Jar-test truoc.")
    if cfg.adjuvant_used:
        return FactorResult("adjuvant", Verdict.ALLOW, "compatible", "Chat tro luc tuong thich nhan.")
    return FactorResult("adjuvant", Verdict.ALLOW, "none", "Khong dung chat tro luc.")


def _nozzle(cfg: MissionConfig, weather: dict[str, Any], t: WeatherThresholds) -> FactorResult:
    wind = float(weather.get("wind_speed_10m", 0) or 0)
    mode = cfg.nozzle_mode or ("COARSE" if wind > t.wind_warn_kph else "MEDIUM")
    if wind > t.wind_warn_kph and mode == "FINE":
        return FactorResult("nozzle", Verdict.STOP, mode,
                            f"Gio {wind:.1f} km/h ma dung giot sieu min - tan xa manh. Chuyen giot trung binh-tho.")
    return FactorResult("nozzle", Verdict.ALLOW, mode, f"Cau hinh giot '{mode}' phu hop gio {wind:.1f} km/h.")


def _canopy(cfg: MissionConfig, crop_stage: CropStage | None, weather: dict[str, Any]) -> FactorResult:
    canopy = cfg.canopy_density or (_STAGE_CANOPY.get(crop_stage.stage_code) if crop_stage else "MEDIUM")
    wind = float(weather.get("wind_speed_10m", 0) or 0)
    if canopy == "DENSE" and wind > 18.0:
        return FactorResult("canopy_density", Verdict.WARN, canopy,
                            "Tan day + gio manh - ha thap/cham lai & tang luu luong nuoc de thuoc xuong tang duoi.")
    return FactorResult("canopy_density", Verdict.ALLOW, canopy, f"Mat do tan '{canopy}' - cau hinh theo giai doan.")


def evaluate_mission_factors(
    weather: dict[str, Any],
    pesticide: PesticideSpec | None,
    crop_stage: CropStage | None,
    cfg: MissionConfig | None = None,
    thresholds: WeatherThresholds = DEFAULT_THRESHOLDS,
) -> list[FactorResult]:
    cfg = cfg or MissionConfig()
    return [
        _wind_direction(weather, thresholds),
        _formulation(cfg, pesticide),
        _water_ph(cfg),
        _adjuvant(cfg),
        _nozzle(cfg, weather, thresholds),
        _canopy(cfg, crop_stage, weather),
    ]
