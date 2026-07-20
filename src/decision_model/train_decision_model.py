"""
Train and compare DSS decision models for agricultural UAV scheduling.

Usage:
    .venv/bin/python -m src.decision_model.train_decision_model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .decision_engine import (
    AGRICULTURAL_LOCATIONS,
    MODEL_FEATURES,
    RISK_WEIGHTS,
    add_decision_columns,
    build_recommendation_text,
    recommend_best_slot,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "src" / "data" / "final_training_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
LEGACY_BEST_SLOT_REPORT = REPORT_DIR / "best_slot.json"
TRAINING_SNAPSHOT_BEST_SLOT_REPORT = REPORT_DIR / "training_snapshot_best_slot.json"

LABEL_MAPPING = {
    "FLY": 0,
    "DELAY": 1,
    "LOCK_SPRAY": 2,
    "NO_FLY": 3
}
INV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


def build_dataset_time_range(df: pd.DataFrame) -> dict[str, str]:
    timestamps = pd.to_datetime(df["timestamp"])
    return {
        "min_timestamp": timestamps.min().strftime("%Y-%m-%d %H:%M:%S"),
        "max_timestamp": timestamps.max().strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_cols = [col for col in MODEL_FEATURES if col in df.columns]
    return df[feature_cols], df["decision_action"].map(LABEL_MAPPING), feature_cols


def model_candidates() -> dict[str, Pipeline]:
    tree_preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                slice(0, None),
            )
        ]
    )

    return {
        "random_forest": Pipeline(
            steps=[
                ("preprocess", tree_preprocess),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=10,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocess", tree_preprocess),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        n_jobs=-1,
                        eval_metric="mlogloss"
                    ),
                ),
            ]
        ),
    }


def train_models(dataset_path: Path) -> dict[str, object]:
    raw_df = pd.read_csv(dataset_path)
    df = raw_df.drop_duplicates(
        subset=["location_name", "timestamp"],
        keep="last",
    ).copy()
    df = add_decision_columns(df)
    x, y, feature_cols = build_feature_matrix(df)
    dataset_time_range = build_dataset_time_range(df)

    groups = df["timestamp"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if set(y_train) != set(y):
        # Fallback to normal stratified split if grouped split misses some classes
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

    split_summary = {
        "dataset_scope": "training_snapshot_not_live_forecast",
        "dataset_time_range": dataset_time_range,
        "strategy": "GroupShuffleSplit grouped by timestamp",
        "input_rows": int(len(raw_df)),
        "model_rows_after_dedup": int(len(df)),
        "duplicate_location_timestamp_rows_removed": int(len(raw_df) - len(df)),
        "unique_timestamps": int(groups.nunique()),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_unique_timestamps": int(groups.iloc[train_idx].nunique()),
        "test_unique_timestamps": int(groups.iloc[test_idx].nunique()),
    }

    metrics = []
    fitted_models = {}
    for name, pipeline in model_candidates().items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics.append(
            {
                "model": name,
                "accuracy": round(accuracy_score(y_test, predictions), 4),
                "macro_precision": round(
                    precision_score(y_test, predictions, average="macro", zero_division=0),
                    4,
                ),
                "macro_recall": round(
                    recall_score(y_test, predictions, average="macro", zero_division=0),
                    4,
                ),
                "macro_f1": round(f1_score(y_test, predictions, average="macro"), 4),
                "weighted_f1": round(f1_score(y_test, predictions, average="weighted"), 4),
            }
        )
        fitted_models[name] = pipeline

    metrics_df = pd.DataFrame(metrics).sort_values(
        ["macro_f1", "accuracy"], ascending=False
    )
    
    # Random Forest is Champion, XGBoost is Challenger
    champion_model = fitted_models["random_forest"]
    challenger_model = fitted_models["xgboost"]
    
    deployment_champion = clone(champion_model).fit(x, y)
    deployment_challenger = clone(challenger_model).fit(x, y)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    model_payload = {
        "champion": deployment_champion,
        "challenger": deployment_challenger,
        "feature_columns": feature_cols,
        "classes": sorted(y.unique().tolist()),
        "label_mapping": LABEL_MAPPING,
        "inv_label_mapping": INV_LABEL_MAPPING,
        "evaluation_split": split_summary,
    }
    joblib.dump(model_payload, MODEL_DIR / "agriflight_model.joblib")

    metrics_df.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    
    # Save classification report for champion
    best_predictions = champion_model.predict(x_test)
    y_test_str = y_test.map(INV_LABEL_MAPPING)
    best_predictions_str = pd.Series(best_predictions).map(INV_LABEL_MAPPING)
    
    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test_str, best_predictions_str),
        encoding="utf-8",
    )
    (REPORT_DIR / "training_summary.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    demo_df = df.copy()
    demo_df["model_action_idx"] = deployment_champion.predict(x)
    demo_df["model_action"] = demo_df["model_action_idx"].map(INV_LABEL_MAPPING)
    demo_df["recommendation_text"] = demo_df.apply(
        lambda row: build_recommendation_text(row, row["model_action"]),
        axis=1,
    )
    demo_cols = [
        "location_name",
        "timestamp",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation_probability",
        "wind_speed_10m",
        "wind_gusts_10m",
        "flyability_score",
        "crop_condition",
        "dynamic_flow_rate_pct",
        "decision_action",
        "model_action",
        "recommendation_text",
    ]
    demo_df[demo_cols].to_csv(REPORT_DIR / "recommendation_demo.csv", index=False)

    training_snapshot_best_slot = recommend_best_slot(
        df[df["location_name"].isin(AGRICULTURAL_LOCATIONS)]
    )
    training_snapshot_best_slot_payload = None
    if training_snapshot_best_slot is not None:
        training_snapshot_best_slot_payload = {
            "scope": "best_slot_inside_training_snapshot_not_live_forecast",
            "dataset_time_range": dataset_time_range,
            "location_name": training_snapshot_best_slot["location_name"],
            "timestamp": training_snapshot_best_slot["timestamp"],
            "flyability_score": float(training_snapshot_best_slot["flyability_score"]),
            "recommendation_text": build_recommendation_text(
                training_snapshot_best_slot,
                "FLY",
            ),
        }
        TRAINING_SNAPSHOT_BEST_SLOT_REPORT.write_text(
            json.dumps(
                training_snapshot_best_slot_payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    LEGACY_BEST_SLOT_REPORT.unlink(missing_ok=True)

    return {
        "dataset_scope": "training_snapshot_not_live_forecast",
        "dataset_time_range": dataset_time_range,
        "dataset_rows": len(df),
        "split_summary": split_summary,
        "feature_count": len(feature_cols),
        "class_distribution": y.value_counts().to_dict(),
        "metrics": metrics_df.to_dict(orient="records"),
        "best_model": "random_forest",
        "training_snapshot_best_slot": training_snapshot_best_slot_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    result = train_models(args.dataset)
    output_json = json.dumps(result, ensure_ascii=False, indent=2)
    with open(MODEL_DIR / "evaluation_log.txt", "w", encoding="utf-8") as f:
        f.write(output_json)

if __name__ == "__main__":
    main()

