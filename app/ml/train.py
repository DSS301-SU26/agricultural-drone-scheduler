"""
Train RF (Champion) + XGBoost (Challenger) - HM4.

Chay: .venv/bin/python -m app.ml.train            (dung du lieu mo phong)
      .venv/bin/python -m app.ml.train --csv data.csv   (dung du lieu that/crawl)

Neu chua cai xgboost -> tu fallback sang HistGradientBoostingClassifier de pipeline
chay duoc; production nen cai xgboost de dung dung Challenger theo BRD.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit

from ..features.engineering import WEATHER_FEATURES, build_features, feature_matrix
from .simulator import simulate

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "agriflight_model.joblib"

LABELS = ["FLY", "DELAY", "NO_FLY"]
LABEL_TO_IDX = {c: i for i, c in enumerate(LABELS)}
IDX_TO_LABEL = {i: c for c, i in LABEL_TO_IDX.items()}


def _make_challenger():
    """XGBoost neu co, khong thi HistGradientBoosting (fallback)."""
    try:
        from xgboost import XGBClassifier
        return ("xgboost", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=-1, eval_metric="mlogloss",
        ))
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return ("hist_gradient_boosting_fallback", HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.1, max_iter=300, random_state=42,
        ))


def _champion():
    return RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )


def train(df: pd.DataFrame) -> dict:
    feat = build_features(df)
    X = feature_matrix(feat)
    y = feat["system_decision"].map(LABEL_TO_IDX)

    # Tach theo NGAY de tranh ro ri thoi gian (grouped split)
    groups = pd.to_datetime(feat["timestamp"]).dt.date.astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(splitter.split(X, y, groups))
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    champ = _champion()
    chall_name, chall = _make_challenger()

    metrics = {}
    for name, model in [("random_forest", champ), (chall_name, chall)]:
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        metrics[name] = {
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
            "weighted_f1": round(f1_score(y_te, pred, average="weighted"), 4),
        }

    # Fit lai tren toan bo du lieu de deploy
    champ.fit(X, y)
    chall.fit(X, y)

    # Feature importance (RF - phuc vu XAI)
    importance = sorted(
        zip(WEATHER_FEATURES, champ.feature_importances_.tolist()),
        key=lambda t: t[1], reverse=True,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "champion": champ,
        "challenger": chall,
        "challenger_name": chall_name,
        "feature_columns": WEATHER_FEATURES,
        "labels": LABELS,
        "label_to_idx": LABEL_TO_IDX,
    }, MODEL_PATH)

    report = {
        "rows": int(len(df)),
        "class_distribution": feat["system_decision"].value_counts().to_dict(),
        "champion": "random_forest",
        "challenger": chall_name,
        "metrics": metrics,
        "top_features": importance[:8],
        "model_path": str(MODEL_PATH),
    }
    (REPORT_DIR / "hm4_training_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "hm4_classification_report.txt").write_text(
        classification_report(
            y_te.map(IDX_TO_LABEL),
            pd.Series(champ.predict(X_te)).map(IDX_TO_LABEL),
        ), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None, help="CSV du lieu that (weather_hourly cols + system_decision)")
    ap.add_argument("--n", type=int, default=40_000, help="So dong mo phong neu khong co --csv")
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        print(f"Dung du lieu that: {args.csv} ({len(df)} dong)")
    else:
        df = simulate(n=args.n)
        print(f"Dung du lieu MO PHONG: {len(df)} dong")

    report = train(df)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
