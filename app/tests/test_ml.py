"""Smoke test HM4 (ML engine). Yeu cau da train model (models/agriflight_model.joblib)."""
from __future__ import annotations

import pandas as pd

from app.ml.scores import Predictor, crop_impact_score, spray_quality_score
from app.rules.context import get_crop_stage, get_pesticide


def _row(**over):
    base = dict(
        location_name="An Giang", timestamp="2025-03-15 09:00:00",
        temperature_2m=30, relative_humidity_2m=70, precipitation=0,
        precipitation_probability=10, cloud_cover=30, visibility=12000,
        wind_speed_10m=8, wind_gusts_10m=13, weather_code=1,
        et0_fao_evapotranspiration=4.5,
    )
    base.update(over)
    return base


def test_safety_orders_ideal_above_storm():
    pred = Predictor()
    ideal = _row()
    storm = _row(temperature_2m=28, precipitation=6, precipitation_probability=85,
                 weather_code=65, visibility=3000, wind_speed_10m=20, wind_gusts_10m=30)
    df = pd.DataFrame([ideal, storm])
    safety, rf, xgb = pred.flight_safety(df)
    assert safety[0] > 80 and safety[1] < 20
    assert 0 <= safety[0] <= 100 and 0 <= safety[1] <= 100


def test_crop_score_penalizes_heat_at_booting():
    stage = get_crop_stage("BOOTING")
    cool = crop_impact_score(_row(temperature_2m=30), stage)
    hot = crop_impact_score(_row(temperature_2m=37), stage)
    assert cool > hot


def test_spray_score_penalizes_wind():
    pest = get_pesticide("Tricyclazole")
    calm = spray_quality_score(_row(wind_speed_10m=6), pest)
    windy = spray_quality_score(_row(wind_speed_10m=25), pest)
    assert calm > windy
