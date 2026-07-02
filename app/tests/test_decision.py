"""Smoke test HM5 (decision orchestrator). Yeu cau da train model."""
from __future__ import annotations

import pytest

from app.decision import apply_override, decide
from app.ml.scores import Predictor
from app.rules.context import get_crop_stage, get_drone, get_pesticide

PRED = Predictor()


def _w(**over):
    base = dict(
        location_name="An Giang", timestamp="2025-03-15 09:00:00",
        temperature_2m=30, relative_humidity_2m=70, precipitation=0,
        precipitation_probability=10, cloud_cover=30, visibility=12000,
        wind_speed_10m=8, wind_gusts_10m=13, weather_code=1,
        et0_fao_evapotranspiration=4.5,
    )
    base.update(over)
    return base


def test_ideal_fly():
    r = decide(_w(), get_drone("DJI_T30"), PRED, hour=17,
               pesticide=get_pesticide("Tricyclazole"), crop_stage=get_crop_stage("TILLERING"))
    assert r.decision == "FLY"
    assert r.flight_config is not None and r.spray_config is not None


def test_storm_nofly():
    r = decide(_w(precipitation=6, precipitation_probability=90, weather_code=65,
                  wind_speed_10m=20, wind_gusts_10m=30, visibility=2000),
               get_drone("DJI_T30"), PRED, hour=14)
    assert r.decision == "NO_FLY"


def test_drone_hard_lock_not_overridable():
    # 25 km/h vuot gioi han T50 (21.6) -> khoa cung
    r = decide(_w(wind_speed_10m=25.0), get_drone("DJI_T50"), PRED, hour=15)
    assert r.decision == "NO_FLY" and r.locked and not r.overridable
    with pytest.raises(PermissionError):
        apply_override(r, "co gang ep bay")


def test_booting_timeban_locked():
    r = decide(_w(), get_drone("DJI_T30"), PRED, hour=9, crop_stage=get_crop_stage("BOOTING"))
    assert r.decision == "NO_FLY" and r.locked


def test_soft_delay_overridable():
    # Tao tinh huong DELAY mem (Abamectin trua nang, khong bi khoa cung)
    r = decide(_w(temperature_2m=34, cloud_cover=20, wind_speed_10m=9),
               get_drone("XAG_P100_PRO"), PRED, hour=12, pesticide=get_pesticide("Abamectin"))
    if r.decision in ("DELAY", "NO_FLY") and not r.locked:
        r2 = apply_override(r, "Dap dich ray nau khan cap")
        assert r2.decision == "FLY" and r2.is_user_overridden
        assert r2.override_reason


def test_override_requires_reason():
    r = decide(_w(temperature_2m=34, cloud_cover=20), get_drone("XAG_P100_PRO"),
               PRED, hour=12, pesticide=get_pesticide("Abamectin"))
    if not r.locked and r.decision != "FLY":
        with pytest.raises(ValueError):
            apply_override(r, "   ")


def test_awd_included_when_soil_given():
    r = decide(_w(), get_drone("DJI_T30"), PRED, hour=17,
               crop_stage=get_crop_stage("TILLERING"),
               soil_water_level_cm=-16.0, rain_24h_forecast_mm=30.0)
    assert r.awd is not None and r.awd["action"] == "HOLD_PUMP"
