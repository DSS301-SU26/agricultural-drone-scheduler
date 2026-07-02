"""
3 diem rui ro cua DSS (BRD §2.2, log: flight_safety/crop_impact/spray_quality).

Quy uoc: CA 3 diem thang 0-100, CANG CAO CANG TOT (an toan/chat luong cao).
  - flight_safety_score: tu ML (P(FLY) consensus RF+XGB, nguyen tac bao thu).
  - crop_impact_score:   tu luat (it ton hai sinh ly lua cang cao).
  - spray_quality_score: tu luat (bam dinh tot, it tan xa cang cao).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..features.engineering import build_features, feature_matrix
from ..rules.context import CropStage, PesticideSpec

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "agriflight_model.joblib"


@dataclass
class Scores:
    flight_safety_score: float
    crop_impact_score: float
    spray_quality_score: float
    rf_score_safety: float
    xgb_score_safety: float
    was_conflict: bool


class Predictor:
    """Load model 1 lan, du doan P(FLY) cho tung khung gio."""

    def __init__(self, model_path: Path = MODEL_PATH):
        payload = joblib.load(model_path)
        self.champion = payload["champion"]
        self.challenger = payload["challenger"]
        self.features = payload["feature_columns"]
        self.labels = payload["labels"]
        self._fly_idx = payload["label_to_idx"]["FLY"]

    def _p_fly(self, model, X: pd.DataFrame) -> np.ndarray:
        proba = model.predict_proba(X)
        return proba[:, self._fly_idx]

    def flight_safety(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tra ve (safety_consensus, rf, xgb) x100. Bao thu: xung dot lon -> lay min."""
        feat = build_features(df)
        X = feature_matrix(feat)
        p_rf = self._p_fly(self.champion, X)
        p_ch = self._p_fly(self.challenger, X)
        delta = np.abs(p_rf - p_ch)
        # Nguyen tac bao thu: xung dot >0.2 -> lay diem an toan thap hon (min)
        consensus = np.where(delta > 0.20, np.minimum(p_rf, p_ch), (p_rf + p_ch) / 2)
        return consensus * 100, p_rf * 100, p_ch * 100


def crop_impact_score(row: dict, stage: CropStage | None) -> float:
    """It ton hai sinh ly lua -> diem cao. Phat theo soc nhiet, do nga (gust), downwash."""
    temp = float(row.get("temperature_2m", 30))
    gust = float(row.get("wind_gusts_10m", 0))
    penalty = 0.0
    penalty += max(0.0, temp - 35) * 8          # soc nhiet, thui phan hoa
    if stage is not None:
        if stage.stage_code == "BOOTING":
            penalty += max(0.0, temp - 33) * 6   # tro bong cuc nhay nhiet
        if stage.stage_code == "GRAIN_FILLING":
            penalty += max(0.0, gust - 25) * 2.0  # do nga/rung hat
        if stage.stage_code == "SEEDLING":
            penalty += max(0.0, gust - 20) * 1.5  # downwash dap ma
    return round(float(np.clip(100 - penalty, 0, 100)), 1)


def spray_quality_score(row: dict, pesticide: PesticideSpec | None) -> float:
    """Bam dinh tot, it tan xa -> diem cao. Phat theo gio (drift), boc hoi (temp+RH thap)."""
    wind = float(row.get("wind_speed_10m", 0))
    temp = float(row.get("temperature_2m", 30))
    rh = float(row.get("relative_humidity_2m", 70))
    cloud = float(row.get("cloud_cover", 50))
    penalty = 0.0
    penalty += max(0.0, wind - 10.8) * 4.0        # tan xa khi gio >3 m/s
    penalty += max(0.0, temp - 33) * 3.0          # boc hoi giot ULV
    penalty += max(0.0, 50 - rh) * 1.2            # RH thap -> giot co lai
    # Thuoc nhay UV: may che phu cao la loi the (bao toan hoat chat)
    if pesticide is not None and pesticide.uv_sensitivity:
        penalty -= max(0.0, cloud - 80) * 0.3
    return round(float(np.clip(100 - penalty, 0, 100)), 1)
