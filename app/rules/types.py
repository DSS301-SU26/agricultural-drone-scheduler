"""
Kieu du lieu dung chung cho tang rules (Layer 3).

Tach biet ro rang voi ML (Layer 2): rules o day la HARD/SOFT constraint dua tren
ma tran 13 tac nhan (BRD §3.3), khong phai output cua model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    """Ket qua danh gia mot tac nhan don le."""
    ALLOW = "ALLOW"   # nam trong vung DUOC PHEP BAY
    WARN = "WARN"     # canh bao mem -> gop thanh DELAY
    STOP = "STOP"     # roi vao cot PHAI DUNG BAY -> NO_FLY


class Decision(str, Enum):
    """Quyet dinh cuoi cung cua DSS (BRD taxonomy)."""
    FLY = "FLY"
    DELAY = "DELAY"
    NO_FLY = "NO_FLY"


# Muc do nghiem trong de gop verdict -> decision
_SEVERITY = {Verdict.ALLOW: 0, Verdict.WARN: 1, Verdict.STOP: 2}
_VERDICT_TO_DECISION = {
    Verdict.ALLOW: Decision.FLY,
    Verdict.WARN: Decision.DELAY,
    Verdict.STOP: Decision.NO_FLY,
}


@dataclass(frozen=True)
class FactorResult:
    """Ket qua danh gia mot trong 13 tac nhan."""
    factor: str            # ten tac nhan (vd "wind_speed")
    verdict: Verdict
    value: float | str | None
    message: str           # giai thich ngan (phuc vu XAI)
    is_hard: bool = False   # True neu la rao chan co hoc (Layer 1 hard rule)


@dataclass
class RuleEvaluation:
    """Tong hop ket qua toan bo tac nhan -> 1 quyet dinh."""
    decision: Decision
    factors: list[FactorResult] = field(default_factory=list)

    @property
    def blocking(self) -> list[FactorResult]:
        """Cac tac nhan khien phai DUNG BAY (STOP)."""
        return [f for f in self.factors if f.verdict is Verdict.STOP]

    @property
    def warnings(self) -> list[FactorResult]:
        return [f for f in self.factors if f.verdict is Verdict.WARN]

    @property
    def hard_blocking(self) -> list[FactorResult]:
        """Cac STOP thuoc rao chan co hoc (khoa cung, khong cho override)."""
        return [f for f in self.factors if f.verdict is Verdict.STOP and f.is_hard]


def combine_verdicts(factors: list[FactorResult]) -> Decision:
    """Quy tac tong hop BRD: bat ky tac nhan STOP -> NO_FLY;
    con lai co WARN -> DELAY; tat ca ALLOW -> FLY."""
    if not factors:
        return Decision.FLY
    worst = max(factors, key=lambda f: _SEVERITY[f.verdict]).verdict
    return _VERDICT_TO_DECISION[worst]
