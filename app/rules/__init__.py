"""Tang rules (Layer 3) - ma tran 13 tac nhan + drone/stage/pesticide/AWD."""
from .awd import AWDAction, AWDRecommendation, evaluate_awd
from .context import (
    CROP_STAGES,
    DRONES,
    PESTICIDES,
    CropStage,
    DroneProfile,
    PesticideSpec,
    get_crop_stage,
    get_drone,
    get_pesticide,
)
from .factors import RuleInput, evaluate_flight_rules
from .growth_stage import FlightConfig, recommend_flight_config
from .pesticide import recommend_nozzle_and_water
from .types import Decision, FactorResult, RuleEvaluation, Verdict

__all__ = [
    "AWDAction", "AWDRecommendation", "evaluate_awd",
    "CROP_STAGES", "DRONES", "PESTICIDES",
    "CropStage", "DroneProfile", "PesticideSpec",
    "get_crop_stage", "get_drone", "get_pesticide",
    "RuleInput", "evaluate_flight_rules",
    "FlightConfig", "recommend_flight_config",
    "recommend_nozzle_and_water",
    "Decision", "FactorResult", "RuleEvaluation", "Verdict",
]
