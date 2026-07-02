"""Pydantic models cho request/response API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DecisionRequest(BaseModel):
    latitude: float
    longitude: float
    drone_model: str = "DJI_T30"
    pesticide: str | None = None          # active_ingredient
    crop_stage: str | None = None         # SEEDLING/TILLERING/BOOTING/GRAIN_FILLING
    farm_size_ha: float = 10.0
    days: int = 2
    plot_id: int | None = None            # de mo phong soil / ghi log


class OverrideRequest(BaseModel):
    reason: str = Field(..., min_length=3)
    weather: dict[str, Any]
    drone_model: str = "DJI_T30"
    pesticide: str | None = None
    crop_stage: str | None = None
    hour: int | None = None
    plot_id: int | None = None
    mission_id: int | None = None


class SlotResponse(BaseModel):
    timestamp: str
    hour: int
    decision: str
    locked: bool
    overridable: bool
    flight_safety_score: float
    crop_impact_score: float
    spray_quality_score: float
    rf_score_safety: float
    xgb_score_safety: float
    was_conflict: bool
    blocking_factors: list[str]
    warning_factors: list[str]
    flight_config: dict[str, Any] | None
    spray_config: dict[str, Any] | None
    xai_explanation: str
    weather: dict[str, Any]


class DecisionResponse(BaseModel):
    source: str                            # forecast | simulated_fallback
    location: dict[str, float]
    drone_model: str
    slots: list[SlotResponse]
    best_slot: SlotResponse | None
    awd: dict[str, Any] | None
