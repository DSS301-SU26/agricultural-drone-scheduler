"""Tang ML (Layer 2) - RF Champion + XGB Challenger, 3 diem rui ro."""
from .scores import Predictor, Scores, crop_impact_score, spray_quality_score
from .simulator import simulate

__all__ = ["Predictor", "Scores", "crop_impact_score", "spray_quality_score", "simulate"]
