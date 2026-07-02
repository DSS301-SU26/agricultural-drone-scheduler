"""
Chien luoc tuoi ngap kho xen ke - AWD (BRD §3.4 / §6).

Chuyen tu tuoi bi dong sang du bao chu dong dua tren ET0 + du bao mua 24h + muc nuoc.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .thresholds import DEFAULT_AWD_THRESHOLDS, AWDThresholds


class AWDAction(str, Enum):
    HOLD_PUMP = "HOLD_PUMP"        # hoan bom, tan dung nuoc mua sap toi
    START_PUMP = "START_PUMP"      # bom bu nuoc ngay
    EARLY_WARNING = "EARLY_WARNING"  # canh bao som chuan bi bom (ET0 cao)
    OK = "OK"                      # muc nuoc an toan, khong can hanh dong


@dataclass(frozen=True)
class AWDRecommendation:
    action: AWDAction
    message: str
    target_level_cm: float | None = None


def evaluate_awd(
    water_level_cm: float,
    et0_mm_day: float,
    rain_24h_forecast_mm: float,
    awd_threshold_cm: float | None = None,
    thresholds: AWDThresholds = DEFAULT_AWD_THRESHOLDS,
) -> AWDRecommendation:
    """
    - Muc nuoc tut duoi nguong an toan:
        + Neu du bao mua lon 24h toi  -> HOLD (tan dung nuoc troi).
        + Nguoc lai                    -> START_PUMP bom bu len +5cm.
    - Muc nuoc con an toan nhung ET0 cuc doan -> EARLY_WARNING chuan bi bom.
    - Con lai -> OK.
    """
    safe_level = awd_threshold_cm if awd_threshold_cm is not None else thresholds.safe_water_level_cm

    if water_level_cm <= safe_level:
        if rain_24h_forecast_mm > thresholds.rain_24h_hold_mm:
            return AWDRecommendation(
                AWDAction.HOLD_PUMP,
                f"Muc nuoc {water_level_cm:.1f}cm cham nguong {safe_level:.0f}cm nhung du bao mua "
                f"{rain_24h_forecast_mm:.0f}mm/24h - hoan bom, tan dung nuoc mua tiet kiem chi phi.",
            )
        return AWDRecommendation(
            AWDAction.START_PUMP,
            f"Muc nuoc {water_level_cm:.1f}cm tut duoi nguong an toan {safe_level:.0f}cm, troi kho "
            f"(mua {rain_24h_forecast_mm:.0f}mm) - BOM BU nuoc len +{thresholds.refill_target_cm:.0f}cm.",
            target_level_cm=thresholds.refill_target_cm,
        )

    if et0_mm_day > thresholds.et0_extreme_mm and rain_24h_forecast_mm < thresholds.rain_24h_hold_mm:
        return AWDRecommendation(
            AWDAction.EARLY_WARNING,
            f"Muc nuoc {water_level_cm:.1f}cm con an toan nhung ET0 {et0_mm_day:.1f}mm/ngay cuc doan, "
            "it mua - canh bao som chuan bi tram bom.",
        )

    return AWDRecommendation(
        AWDAction.OK,
        f"Muc nuoc {water_level_cm:.1f}cm an toan, ET0 {et0_mm_day:.1f}mm/ngay - chua can bom.",
    )
