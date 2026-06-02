import unittest

import pandas as pd

from src.decision_model.train_decision_model import build_dataset_time_range


class BuildDatasetTimeRangeTest(unittest.TestCase):
    def test_returns_snapshot_boundaries(self):
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2026-05-30 17:00:00",
                    "2026-05-25 06:00:00",
                ]
            }
        )

        self.assertEqual(
            build_dataset_time_range(df),
            {
                "min_timestamp": "2026-05-25 06:00:00",
                "max_timestamp": "2026-05-30 17:00:00",
            },
        )


if __name__ == "__main__":
    unittest.main()
