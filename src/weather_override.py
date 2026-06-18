"""
weather_override.py - AI-assisted weather override with human-in-the-loop.

Flow:
    1. User uploads a real weather photo
    2. MobileNetV2 extracts features → compares with category centroids
    3. AI suggests a weather condition + confidence
    4. User confirms OR overrides the suggestion
    5. System recalculates flyability / decision based on the final condition
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .decision_model.decision_engine import (
    DecisionThresholds,
    THRESHOLDS,
    UNSAFE_WEATHER_CODES,
    calculate_flyability_score,
    derive_decision_action,
    derive_risk_level,
    calculate_dynamic_flow_rate,
    infer_crop_condition,
    build_recommendation_text,
)

ROOT = Path(__file__).resolve().parents[1]
OVERRIDE_IMAGE_DIR = ROOT / "src" / "data" / "weather_overrides"
CENTROIDS_PATH = ROOT / "src" / "data" / "category_centroids.npy"


# ── Weather condition → parameter mapping ─────────────────────

WEATHER_CONDITIONS = {
    "sunny": {
        "label": "Trời nắng",
        "weather_code": 1000,
        "weather_description": "Sunny",
        "precipitation": 0.0,
        "precipitation_probability": 0.0,
        "cloud_cover": 10.0,
    },
    "cloudy": {
        "label": "Trời nhiều mây",
        "weather_code": 1006,
        "weather_description": "Cloudy",
        "precipitation": 0.0,
        "precipitation_probability": 10.0,
        "cloud_cover": 75.0,
    },
    "rainy": {
        "label": "Trời mưa",
        "weather_code": 1183,
        "weather_description": "Light rain",
        "precipitation": 2.0,
        "precipitation_probability": 80.0,
        "cloud_cover": 90.0,
    },
    "stormy": {
        "label": "Giông bão",
        "weather_code": 1087,
        "weather_description": "Thundery outbreaks in nearby",
        "precipitation": 5.0,
        "precipitation_probability": 95.0,
        "cloud_cover": 100.0,
        "wind_speed_10m": 35.0,
        "wind_gusts_10m": 50.0,
    },
    "foggy": {
        "label": "Sương mù",
        "weather_code": 1135,
        "weather_description": "Fog",
        "precipitation": 0.0,
        "precipitation_probability": 20.0,
        "cloud_cover": 100.0,
        "visibility": 500.0,
    },
}

# Map category names from Kaggle dataset to our condition keys
_CATEGORY_TO_CONDITION = {
    "Shine": "sunny",
    "Sunrise": "sunny",
    "Cloudy": "cloudy",
    "Rain": "rainy",
}


def available_conditions() -> list[dict[str, str]]:
    """Return the list of selectable weather conditions for the frontend."""
    return [
        {"key": key, "label": info["label"], "description": info["weather_description"]}
        for key, info in WEATHER_CONDITIONS.items()
    ]


# ── AI classification ─────────────────────────────────────────

def _load_mobilenet():
    """Lazy-load MobileNetV2 for feature extraction."""
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")


def _extract_features_from_file(image_path: str | Path, model=None):
    """Extract a 1280-dim feature vector from an image file."""
    import cv2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array

    if model is None:
        model = _load_mobilenet()

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Khong doc duoc anh: {image_path}")

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0).flatten()
    return features


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def classify_weather_image(image_path: str | Path) -> dict[str, Any]:
    """
    Classify a weather image using MobileNetV2 + centroid comparison.

    Returns:
        {
            "suggested_condition": "rainy",
            "confidence": 0.87,
            "all_scores": {"sunny": 0.45, "cloudy": 0.62, "rainy": 0.87, ...}
        }
    """
    if not CENTROIDS_PATH.exists():
        return {
            "suggested_condition": None,
            "confidence": 0.0,
            "all_scores": {},
            "message": "Chua co file centroids. Hay chay build_category_centroids.py truoc.",
        }

    try:
        model = _load_mobilenet()
        features = _extract_features_from_file(image_path, model)

        centroids = np.load(str(CENTROIDS_PATH), allow_pickle=True).item()

        scores: dict[str, float] = {}
        for category_name, centroid_vector in centroids.items():
            condition_key = _CATEGORY_TO_CONDITION.get(category_name, category_name)
            similarity = _cosine_similarity(features, centroid_vector)
            # Keep the highest score if multiple categories map to the same condition
            if condition_key not in scores or similarity > scores[condition_key]:
                scores[condition_key] = round(similarity, 4)

        best_condition = max(scores, key=scores.get) if scores else None
        best_score = scores.get(best_condition, 0.0) if best_condition else 0.0

        return {
            "suggested_condition": best_condition,
            "confidence": best_score,
            "all_scores": scores,
        }

    except Exception as exc:
        return {
            "suggested_condition": None,
            "confidence": 0.0,
            "all_scores": {},
            "message": f"Loi khi phan loai anh: {exc}",
        }


# ── Override application ──────────────────────────────────────

def apply_override_to_weather(
    original_row: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    """
    Merge user-chosen weather condition into the original weather parameters.

    Fields not specified in the condition map are kept from *original_row*.
    """
    if condition not in WEATHER_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}. Choose from {list(WEATHER_CONDITIONS)}")

    overrides = WEATHER_CONDITIONS[condition]
    merged = dict(original_row)

    for key in ("weather_code", "weather_description", "precipitation",
                "precipitation_probability", "cloud_cover"):
        if key in overrides:
            merged[key] = overrides[key]

    # Only override wind / visibility when the condition specifies them (stormy, foggy)
    if "wind_speed_10m" in overrides:
        merged["wind_speed_10m"] = overrides["wind_speed_10m"]
    if "wind_gusts_10m" in overrides:
        merged["wind_gusts_10m"] = overrides["wind_gusts_10m"]
    if "visibility" in overrides:
        merged["visibility"] = overrides["visibility"]

    return merged


def recalculate_decision(
    weather_params: dict[str, Any],
    thresholds: DecisionThresholds = THRESHOLDS,
    unsafe_weather_codes: set[int] | frozenset[int] = UNSAFE_WEATHER_CODES,
) -> dict[str, Any]:
    """
    Run the decision engine on (possibly overridden) weather parameters.

    Returns a dict with the new flyability_score, decision_action, risk_level, etc.
    """
    row = pd.Series(weather_params)
    crop = infer_crop_condition(row)
    score = calculate_flyability_score(row, thresholds, unsafe_weather_codes)
    action = derive_decision_action(row, thresholds, unsafe_weather_codes)
    risk = derive_risk_level(row, action, thresholds, unsafe_weather_codes)
    flow = calculate_dynamic_flow_rate(row, thresholds)
    text = build_recommendation_text(row, action, thresholds)

    return {
        "crop_condition": crop,
        "flyability_score": score,
        "decision_action": action,
        "risk_level": risk,
        "dynamic_flow_rate_pct": flow,
        "recommendation_text": text,
    }


def save_override_image(image_bytes: bytes, extension: str = ".jpg") -> tuple[str, Path]:
    """
    Save uploaded image to disk and return (filename, full_path).
    """
    OVERRIDE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"override_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{extension}"
    full_path = OVERRIDE_IMAGE_DIR / filename
    full_path.write_bytes(image_bytes)
    return filename, full_path
