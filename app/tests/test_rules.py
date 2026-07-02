"""Smoke test cho tang rules (HM3). Chay: .venv/bin/python -m pytest app/tests/test_rules.py"""
from __future__ import annotations

from app.rules import (
    AWDAction,
    Decision,
    RuleInput,
    Verdict,
    evaluate_awd,
    evaluate_flight_rules,
    get_crop_stage,
    get_drone,
    get_pesticide,
)


def _weather(**over):
    base = dict(
        temperature_2m=28.0, relative_humidity_2m=70.0,
        wind_speed_10m=8.0, wind_gusts_10m=14.0,
        precipitation=0.0, precipitation_probability=10.0,
        cloud_cover=40.0, visibility=10000.0, weather_code=1,
    )
    base.update(over)
    return base


def test_ideal_conditions_fly():
    inp = RuleInput(weather=_weather(), hour=17, drone=get_drone("DJI_T30"),
                    pesticide=get_pesticide("Tricyclazole"),
                    crop_stage=get_crop_stage("TILLERING"))
    result = evaluate_flight_rules(inp)
    assert result.decision is Decision.FLY
    assert not result.blocking


def test_high_wind_stops_spray():
    inp = RuleInput(weather=_weather(wind_speed_10m=20.0), hour=17, drone=get_drone("DJI_T30"))
    result = evaluate_flight_rules(inp)
    assert result.decision is Decision.NO_FLY
    assert any(f.factor == "wind_speed" for f in result.blocking)


def test_drone_hard_limit_t50_stricter():
    # 25 km/h: T50 (limit 21.6) khoa cung, T30 (28.8) thi khong
    w = _weather(wind_speed_10m=25.0)
    r50 = evaluate_flight_rules(RuleInput(weather=w, hour=17, drone=get_drone("DJI_T50")))
    assert any(f.factor == "drone_wind_limit" and f.is_hard for f in r50.hard_blocking)
    r30 = evaluate_flight_rules(RuleInput(weather=w, hour=17, drone=get_drone("DJI_T30")))
    assert not any(f.factor == "drone_wind_limit" for f in r30.hard_blocking)


def test_booting_time_ban():
    inp = RuleInput(weather=_weather(), hour=9, drone=get_drone("DJI_T30"),
                    crop_stage=get_crop_stage("BOOTING"))
    result = evaluate_flight_rules(inp)
    assert result.decision is Decision.NO_FLY
    assert any(f.factor == "stage_time_ban" and f.is_hard for f in result.hard_blocking)


def test_abamectin_uv_midday_delay_or_stop():
    inp = RuleInput(weather=_weather(temperature_2m=34.0, cloud_cover=20.0), hour=12,
                    drone=get_drone("DJI_T30"), pesticide=get_pesticide("Abamectin"))
    result = evaluate_flight_rules(inp)
    assert result.decision in (Decision.DELAY, Decision.NO_FLY)


def test_rain_washout_high_prob_stops():
    inp = RuleInput(weather=_weather(), hour=17, drone=get_drone("DJI_T30"),
                    pesticide=get_pesticide("Tricyclazole"),
                    rain_prob_washout_window_pct=80.0)
    result = evaluate_flight_rules(inp)
    assert any(f.factor == "pesticide_rain_washout" for f in result.blocking)


def test_awd_hold_when_rain_coming():
    rec = evaluate_awd(water_level_cm=-16.0, et0_mm_day=4.0, rain_24h_forecast_mm=30.0)
    assert rec.action is AWDAction.HOLD_PUMP


def test_awd_start_pump_when_dry():
    rec = evaluate_awd(water_level_cm=-16.0, et0_mm_day=6.5, rain_24h_forecast_mm=0.0)
    assert rec.action is AWDAction.START_PUMP


def test_awd_early_warning_high_et0():
    rec = evaluate_awd(water_level_cm=-8.0, et0_mm_day=7.0, rain_24h_forecast_mm=0.0)
    assert rec.action is AWDAction.EARLY_WARNING
