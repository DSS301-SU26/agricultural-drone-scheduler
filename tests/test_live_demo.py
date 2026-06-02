import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.decision_model.live_demo import add_reference_features, latest_clean_dataset


class LatestCleanDatasetTest(unittest.TestCase):
    def test_returns_newest_timestamped_clean_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            clean_dir = Path(tmp_dir)
            older = clean_dir / "weather_clean_20260601_1200.csv"
            newer = clean_dir / "weather_clean_20260602_0900.csv"
            older.touch()
            newer.touch()

            self.assertEqual(latest_clean_dataset(clean_dir), newer)


class AddReferenceFeaturesTest(unittest.TestCase):
    def test_fills_missing_model_features_from_training_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reference_dataset = Path(tmp_dir) / "training.csv"
            pd.DataFrame(
                {
                    "img_feature_0": [1.0, 3.0],
                    "img_feature_1": [2.0, 4.0],
                }
            ).to_csv(reference_dataset, index=False)

            live_df = pd.DataFrame({"temperature_2m": [30.0, 31.0]})
            enriched_df, fallback_cols = add_reference_features(
                live_df,
                ["temperature_2m", "img_feature_0", "img_feature_1"],
                reference_dataset,
            )

            self.assertEqual(fallback_cols, ["img_feature_0", "img_feature_1"])
            self.assertEqual(enriched_df["img_feature_0"].tolist(), [2.0, 2.0])
            self.assertEqual(enriched_df["img_feature_1"].tolist(), [3.0, 3.0])


if __name__ == "__main__":
    unittest.main()
