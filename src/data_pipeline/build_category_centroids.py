"""
build_category_centroids.py - Pre-compute average feature vectors for each
weather image category (Rain, Cloudy, Shine, Sunrise).

Run once (or whenever the Kaggle image set changes):
    python -m src.data_pipeline.build_category_centroids

The output file ``category_centroids.npy`` is used at runtime by
``weather_override.py`` to classify user-uploaded photos.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

BASE_DIR = Path(__file__).resolve().parents[2]
KAGGLE_DIR = BASE_DIR / "src" / "data" / "image_kaggle"
OUTPUT_PATH = BASE_DIR / "src" / "data" / "category_centroids.npy"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# Maximum images per category to process (keeps build time reasonable)
MAX_IMAGES_PER_CATEGORY = 100


def main():
    if not KAGGLE_DIR.exists():
        print(f"LOI: Khong tim thay thu muc Kaggle tai {KAGGLE_DIR}")
        sys.exit(1)

    categories = sorted(
        d.name for d in KAGGLE_DIR.iterdir()
        if d.is_dir() and any(f.suffix.lower() in IMAGE_SUFFIXES for f in d.iterdir())
    )
    if not categories:
        print("LOI: Khong tim thay category nao co anh.")
        sys.exit(1)

    print(f"1. Tim thay {len(categories)} categories: {categories}")
    print("2. Dang tai MobileNetV2...")
    model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

    centroids: dict[str, np.ndarray] = {}

    for cat in categories:
        cat_dir = KAGGLE_DIR / cat
        images = sorted(
            p for p in cat_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )[:MAX_IMAGES_PER_CATEGORY]

        print(f"\n-> Category '{cat}': {len(images)} anh")
        feature_list: list[np.ndarray] = []

        for idx, img_path in enumerate(images, 1):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                arr = np.expand_dims(img_to_array(img), axis=0)
                arr = preprocess_input(arr)
                features = model.predict(arr, verbose=0).flatten()
                feature_list.append(features)

                if idx % 25 == 0:
                    print(f"   ... {idx}/{len(images)} anh")
            except Exception as exc:
                print(f"   [SKIP] {img_path.name}: {exc}")

        if feature_list:
            centroids[cat] = np.mean(feature_list, axis=0)
            print(f"   Centroid '{cat}': shape {centroids[cat].shape}")
        else:
            print(f"   [WARN] Khong extract duoc feature nao cho '{cat}'")

    np.save(str(OUTPUT_PATH), centroids)
    print(f"\n--- HOAN TAT ---")
    print(f"Da luu {len(centroids)} centroids vao: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
