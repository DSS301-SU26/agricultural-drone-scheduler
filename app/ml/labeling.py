"""
Gan nhan FLY/DELAY/NO_FLY co NHIEU (dung chung cho ca du lieu MO PHONG lan THAT).

Ly do dung chung: du lieu thoi tiet that KHONG co san nhan (khong ai ghi "hom do co
bay khong"). Ta gan nhan tu latent risk lien tuc + nhieu Gaussian + ca override cua
con nguoi -> anh xa feature->label ngau nhien, model hoc XAC SUAT thay vi thuoc luat.
Rules cung (drone/stage) van la lop rao chan RIENG o HM5, khong nam trong nhan nay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision_model.decision_engine import calculate_flyability_score

def attach_noisy_labels(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Them cot `system_decision` (FLY/DELAY/NO_FLY) va `is_user_overridden`.
    Ap dung cho bat ky DataFrame co cac cot weather_hourly chuan."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    # Tinh diem an toan vat ly
    base_scores = out.apply(calculate_flyability_score, axis=1).to_numpy()

    # Them nhieu nho vao diem de mo hinh hoc xac suat, tranh bi overfit vao luat cung
    scores = base_scores + rng.normal(0, 0.05, n)

    # Gan nhan dua tren diem an toan
    label = np.where(scores < 0.40, "NO_FLY",
             np.where(scores >= 0.70, "FLY", "DELAY"))
             
    # Vung xam: mot so DELAY bi day len/xuong
    flip = (scores > 0.35) & (scores < 0.75) & (rng.uniform(0, 1, n) < 0.05)
    label = np.where(flip & (label == "DELAY"), rng.choice(["FLY", "NO_FLY"], n), label)

    # OVERRIDE cua con nguoi: ~5% cac truong hop khong bay duoc ep thanh FLY (dap dich khan cap)
    is_override = (label != "FLY") & (rng.uniform(0, 1, n) < 0.05)
    label = np.where(is_override, "FLY", label)

    out["system_decision"] = label
    out["is_user_overridden"] = is_override
    return out
