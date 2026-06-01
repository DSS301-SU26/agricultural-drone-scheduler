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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .decision_engine import (
    AGRICULTURAL_LOCATIONS,
    WEATHER_FEATURES,
    add_decision_columns,
    build_recommendation_text,
    image_feature_columns,
    recommend_best_slot,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "src" / "data" / "final_training_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    img_cols = image_feature_columns(df.columns)
    feature_cols = [col for col in WEATHER_FEATURES if col in df.columns] + img_cols
    return df[feature_cols], df["decision_action"], feature_cols


def model_candidates() -> dict[str, Pipeline]:
    numeric_preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                slice(0, None),
            )
        ]
    )

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
        "baseline_majority": Pipeline(
            steps=[
                ("preprocess", tree_preprocess),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("preprocess", tree_preprocess),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=6,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", tree_preprocess),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=10,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", numeric_preprocess),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=42,
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

    # One simulated image is reused by all locations for the same timestamp.
    # Grouping by timestamp prevents the same image context from appearing in
    # both train and test sets and makes the evaluation more honest.
    groups = df["timestamp"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if set(y_train) != set(y):
        raise ValueError(
            "Grouped train split does not contain every decision class. "
            "Collect more timestamps or adjust the split seed."
        )

    split_summary = {
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
    best_name = str(metrics_df.iloc[0]["model"])
    evaluation_model = fitted_models[best_name]
    best_predictions = evaluation_model.predict(x_test)
    deployment_model = clone(evaluation_model).fit(x, y)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    model_payload = {
        "model_name": best_name,
        "pipeline": deployment_model,
        "feature_columns": feature_cols,
        "classes": sorted(y.unique().tolist()),
        "evaluation_split": split_summary,
    }
    joblib.dump(model_payload, MODEL_DIR / "drone_decision_model.joblib")

    metrics_df.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, best_predictions),
        encoding="utf-8",
    )
    (REPORT_DIR / "training_summary.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    demo_df = df.copy()
    demo_df["model_action"] = deployment_model.predict(x)
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

    best_slot = recommend_best_slot(
        df[df["location_name"].isin(AGRICULTURAL_LOCATIONS)]
    )
    best_slot_payload = None
    if best_slot is not None:
        best_slot_payload = {
            "location_name": best_slot["location_name"],
            "timestamp": best_slot["timestamp"],
            "flyability_score": float(best_slot["flyability_score"]),
            "recommendation_text": build_recommendation_text(best_slot, "TAKE_OFF"),
        }
        (REPORT_DIR / "best_slot.json").write_text(
            json.dumps(best_slot_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "dataset_rows": len(df),
        "split_summary": split_summary,
        "feature_count": len(feature_cols),
        "class_distribution": y.value_counts().to_dict(),
        "metrics": metrics_df.to_dict(orient="records"),
        "best_model": best_name,
        "best_slot": best_slot_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    result = train_models(args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
