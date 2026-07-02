"""
Dung bo du lieu HUAN LUYEN tu thoi tiet THAT (HM2 -> HM4).

Crawl lich su Open-Meteo cho 6 tinh ĐBSCL -> gan nhan co nhieu -> luu CSV.
Sau do train bang: .venv/bin/python -m app.ml.train --csv data/training_real.csv

Chay (can mang, tren may that):
    .venv/bin/python -m app.ingestion.build_dataset --start 2021-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..ml.labeling import attach_noisy_labels
from .locations import DELTA_LOCATIONS
from .open_meteo import fetch_historical

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "training_real.csv"


def build(start: str, end: str, out: Path = DEFAULT_OUT, seed: int = 42) -> pd.DataFrame:
    df = fetch_historical(DELTA_LOCATIONS, start, end)
    if df.empty:
        raise RuntimeError("Khong crawl duoc du lieu. Kiem tra mang.")

    df = df.drop_duplicates(subset=["location_name", "timestamp"], keep="last")
    df = df.sort_values(["location_name", "timestamp"]).reset_index(drop=True)

    labeled = attach_noisy_labels(df, seed=seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out, index=False)
    print(f"\nDa luu {len(labeled)} dong -> {out}")
    print("Phan bo nhan:", labeled["system_decision"].value_counts().to_dict())
    print("So ca override:", int(labeled["is_user_overridden"].sum()))
    return labeled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    build(args.start, args.end, args.out)


if __name__ == "__main__":
    main()
