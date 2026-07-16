"""
Luat theo giai doan sinh truong lua (BRD §3.2 / §5.1).

- Cung cap cau hinh bay (do cao / toc do / luu luong) theo giai doan.
- Rao chan CUNG theo khung gio: BOOTING cam bay 08-11h (lua phoi mau thu phan).
"""
from __future__ import annotations

from dataclasses import dataclass

from .context import CropStage
from .types import FactorResult, Verdict


@dataclass(frozen=True)
class FlightConfig:
    """Cau hinh bay khuyen nghi theo giai doan."""
    altitude_min_m: float
    altitude_max_m: float
    speed_min_ms: float
    speed_max_ms: float
    flow_rate_min_l_ha: float
    flow_rate_max_l_ha: float


def recommend_flight_config(stage: CropStage) -> FlightConfig:
    return FlightConfig(
        altitude_min_m=stage.opt_flight_alt_min,
        altitude_max_m=stage.opt_flight_alt_max,
        speed_min_ms=stage.opt_flight_speed_min,
        speed_max_ms=stage.opt_flight_speed_max,
        flow_rate_min_l_ha=stage.flow_rate_min_l_ha,
        flow_rate_max_l_ha=stage.flow_rate_max_l_ha,
    )


def evaluate_stage_time_ban(hour: int, stage: CropStage) -> FactorResult | None:
    """Kiem tra khung gio cam bay cung theo giai doan.
    Tra None neu giai doan khong co lenh cam gio."""
    if stage.hard_ban_start_hour is None or stage.hard_ban_end_hour is None:
        return None

    if stage.hard_ban_start_hour <= hour < stage.hard_ban_end_hour:
        return FactorResult(
            factor="stage_time_ban",
            verdict=Verdict.STOP_SPRAY,
            value=hour,
            message=(
                f"Lúa đang ở giai đoạn {stage.stage_name}: BẮT BUỘC KHÓA PHUN từ {stage.hard_ban_start_hour:02d}h đến "
                f"{stage.hard_ban_end_hour:02d}h. Đây là lúc lúa phơi màu thụ phấn, sức gió từ cánh quạt (downwash) "
                "sẽ thổi bay phấn hoa gây lép hạt diện rộng."
            ),
            is_hard=False,
        )
    return FactorResult(
        factor="stage_time_ban",
        verdict=Verdict.ALLOW,
        value=hour,
        message=f"Khung giờ an toàn, không vi phạm thời gian thụ phấn của giai đoạn {stage.stage_name}.",
        is_hard=False,
    )
