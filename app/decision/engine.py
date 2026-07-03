"""
Decision Orchestrator (HM5) - Layer 3.

Rap tat ca:
  - Rules (HM3): rao chan cung + danh gia 13 tac nhan.
  - ML (HM4): 3 diem flight_safety / crop_impact / spray_quality.
-> Quyet dinh cuoi FLY / DELAY / NO_FLY (BRD §8 luong 3 lop) + XAI + override.

Luong quyet dinh 3 lop:
  Lop 1 - Hard rules (drone limit, cam gio theo giai doan, mua/tam nhin nguy hiem):
          STOP cung -> NO_FLY, KHOA (khong cho override).
  Lop 2 - Soft scoring (ML safety + canh bao nong hoc/thuoc): -> DELAY hoac NO_FLY mem.
  Lop 3 - Override: cac lenh DELAY/NO_FLY mem cho phep con nguoi ep bay kem ly do.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

from ..ml.scores import Predictor, crop_impact_score, spray_quality_score
from ..rules.awd import AWDRecommendation, evaluate_awd
from ..rules.context import CropStage, DroneProfile, PesticideSpec
from ..rules.factors import RuleInput, evaluate_flight_rules
from ..rules.growth_stage import recommend_flight_config
from ..rules.pesticide import recommend_nozzle_and_water
from ..rules.types import Decision, RuleEvaluation

# Nguong ML safety -> quyet dinh mem
SAFETY_FLY_MIN = 70.0     # >=70 -> du dieu kien FLY
SAFETY_NOFLY_MAX = 40.0   # <40  -> NO_FLY mem

_SEVERITY = {Decision.FLY: 0, Decision.DELAY: 1, Decision.NO_FLY: 2}


@dataclass
class DecisionResult:
    decision: str                       # FLY / DELAY / NO_FLY
    locked: bool                        # True = hard NO_FLY, khong cho override
    overridable: bool
    flight_safety_score: float
    crop_impact_score: float
    spray_quality_score: float
    rf_score_safety: float
    xgb_score_safety: float
    was_conflict: bool
    blocking_factors: list[str] = field(default_factory=list)
    warning_factors: list[str] = field(default_factory=list)
    all_factors: list[dict[str, Any]] = field(default_factory=list)
    flight_config: dict[str, Any] | None = None
    spray_config: dict[str, Any] | None = None
    awd: dict[str, Any] | None = None
    xai_explanation: str = ""
    is_user_overridden: bool = False
    override_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ml_decision_from_safety(safety: float) -> Decision:
    if safety >= SAFETY_FLY_MIN:
        return Decision.FLY
    if safety < SAFETY_NOFLY_MAX:
        return Decision.NO_FLY
    return Decision.DELAY


def _build_xai(rule_eval: RuleEvaluation, safety: float, crop: float, spray: float,
               decision: Decision) -> str:
    parts = [f"Quyet dinh: {decision.value}.",
             f"Diem: An toan bay {safety:.0f}/100, Tac dong lua {crop:.0f}/100, "
             f"Chat luong phun {spray:.0f}/100."]
    if rule_eval.hard_blocking:
        parts.append("KHOA CUNG do: " + "; ".join(f.message for f in rule_eval.hard_blocking))
    else:
        blk = [f for f in rule_eval.blocking]
        if blk:
            parts.append("Yeu to phai dung: " + "; ".join(f.message for f in blk))
        warns = rule_eval.warnings
        if warns:
            parts.append("Canh bao: " + "; ".join(f.message for f in warns))
    if decision is Decision.FLY:
        parts.append("Dieu kien dat nguong an toan - cho phep cat canh.")
    return " ".join(parts)


def decide(
    weather: dict[str, Any],
    drone: DroneProfile,
    predictor: Predictor,
    hour: int | None = None,
    pesticide: PesticideSpec | None = None,
    crop_stage: CropStage | None = None,
    rain_prob_washout_window_pct: float | None = None,
    soil_water_level_cm: float | None = None,
    rain_24h_forecast_mm: float = 0.0,
) -> DecisionResult:
    """Ra quyet dinh cho 1 khung gio."""
    if hour is None:
        hour = int(pd.to_datetime(weather.get("timestamp")).hour) if weather.get("timestamp") else 12

    # --- Lop rules ---
    rule_eval = evaluate_flight_rules(RuleInput(
        weather=weather, hour=hour, drone=drone, pesticide=pesticide,
        crop_stage=crop_stage, rain_prob_washout_window_pct=rain_prob_washout_window_pct,
    ))

    # --- Lop ML: 3 diem ---
    df = pd.DataFrame([weather])
    safety_arr, rf_arr, xgb_arr = predictor.flight_safety(df)
    safety = float(safety_arr[0]); rf = float(rf_arr[0]); xgb = float(xgb_arr[0])
    was_conflict = abs(rf - xgb) > 20.0
    crop = crop_impact_score(weather, crop_stage)
    spray = spray_quality_score(weather, pesticide)

    ml_decision = _ml_decision_from_safety(safety)

    # --- Tong hop: lay muc nghiem trong cao nhat giua rule & ML ---
    final = max([rule_eval.decision, ml_decision], key=lambda d: _SEVERITY[d])

    hard_locked = bool(rule_eval.hard_blocking)
    if hard_locked:
        final = Decision.NO_FLY

    # --- Cau hinh bay & phun (chi khi khong bi khoa) ---
    flight_config = spray_config = None
    if final is not Decision.NO_FLY and crop_stage is not None:
        fc = recommend_flight_config(crop_stage)
        flight_config = asdict(fc)
    if final is not Decision.NO_FLY:
        spray_config = recommend_nozzle_and_water(
            pesticide, float(weather.get("wind_speed_10m", 0)))

    # --- AWD (doc lap voi quyet dinh bay) ---
    awd_dict = None
    if soil_water_level_cm is not None:
        awd_rec: AWDRecommendation = evaluate_awd(
            water_level_cm=soil_water_level_cm,
            et0_mm_day=float(weather.get("et0_fao_evapotranspiration", 4.0)),
            rain_24h_forecast_mm=rain_24h_forecast_mm,
            awd_threshold_cm=crop_stage.awd_threshold_cm if crop_stage else None,
        )
        awd_dict = {"action": awd_rec.action.value, "message": awd_rec.message,
                    "target_level_cm": awd_rec.target_level_cm}

    result = DecisionResult(
        decision=final.value,
        locked=hard_locked,
        overridable=(final is not Decision.FLY) and not hard_locked,
        flight_safety_score=round(safety, 1),
        crop_impact_score=round(crop, 1),
        spray_quality_score=round(spray, 1),
        rf_score_safety=round(rf, 1),
        xgb_score_safety=round(xgb, 1),
        was_conflict=was_conflict,
        blocking_factors=[f.factor for f in rule_eval.blocking],
        warning_factors=[f.factor for f in rule_eval.warnings],
        all_factors=[{"factor": f.factor, "verdict": f.verdict.value,
                      "value": f.value, "message": f.message, "is_hard": f.is_hard}
                     for f in rule_eval.factors],
        flight_config=flight_config,
        spray_config=spray_config,
        awd=awd_dict,
        xai_explanation=_build_xai(rule_eval, safety, crop, spray, final),
    )
    return result


def apply_override(result: DecisionResult, reason: str) -> DecisionResult:
    """Lop 3 - con nguoi ep bay. Chi cho phep khi KHONG bi khoa cung.
    Bat buoc co ly do (BRD §8)."""
    if result.locked:
        raise PermissionError("Khong the override: lenh NO_FLY khoa cung (rao chan co hoc).")
    if not reason or not reason.strip():
        raise ValueError("Override bat buoc phai co ly do.")
    result.decision = Decision.FLY.value
    result.is_user_overridden = True
    result.override_reason = reason.strip()
    result.xai_explanation += f" | OVERRIDE boi nguoi dung: {reason.strip()}"
    return result
