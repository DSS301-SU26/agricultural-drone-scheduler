"""Tang Decision (Layer 3) - orchestrator gop rules + ML -> FLY/DELAY/NO_FLY."""
from .engine import DecisionResult, apply_override, decide

__all__ = ["DecisionResult", "apply_override", "decide"]
