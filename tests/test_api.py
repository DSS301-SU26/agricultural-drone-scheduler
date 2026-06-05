import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.api import (
    dashboard,
    get_ai_training_status,
    health,
    locations,
    reset_decision_config,
    run_pipeline_endpoint,
    train_ai_model,
    update_decision_config,
)


class DashboardApiTest(unittest.TestCase):
    def test_health(self):
        self.assertEqual(health()["status"], "ok")

    def test_locations_are_loaded_from_clean_forecast(self):
        location_names = {location["name"] for location in locations()}
        self.assertIn("Dong Thap", location_names)
        self.assertIn("Can Tho", location_names)

    def test_dashboard_uses_real_pipeline_columns(self):
        payload = dashboard(location="Dong Thap", at="2026-06-01T10:00:00")
        self.assertEqual(payload["location"]["name"], "Dong Thap")
        self.assertTrue(payload["source"]["dataset"].startswith("weather_clean_"))
        self.assertIn("decision_config", payload)
        self.assertIn(payload["current"]["decision_action"], {
            "TAKE_OFF",
            "DELAY_FLIGHT",
            "LOCK_SPRAY",
            "RETURN_TO_CHARGING",
        })
        self.assertEqual(len(payload["forecast"]), 12)
        self.assertEqual(len(payload["timeline_tiles"]), 12)
        self.assertEqual(len(payload["kpis"]), 3)

    def test_dashboard_uses_dynamic_decision_config(self):
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "decision_config.json"
            with (
                patch("src.api.CONFIG_DIR", Path(temp_dir)),
                patch("src.api.DECISION_CONFIG_PATH", config_path),
            ):
                baseline = dashboard(location="Dong Thap", at="2026-06-05T06:00:00")
                self.assertEqual(baseline["current"]["decision_action"], "TAKE_OFF")

                update_decision_config({"thresholds": {"max_wind_speed": 5}})
                restricted = dashboard(location="Dong Thap", at="2026-06-05T06:00:00")

                self.assertEqual(restricted["decision_config"]["source"], "file")
                self.assertEqual(restricted["current"]["decision_action"], "LOCK_SPRAY")

                reset_decision_config()
                restored = dashboard(location="Dong Thap", at="2026-06-05T06:00:00")
                self.assertEqual(restored["current"]["decision_action"], "TAKE_OFF")

    def test_ai_training_status_reads_existing_artifacts(self):
        payload = get_ai_training_status("Dong Thap")
        self.assertEqual(payload["scope"], "location")
        self.assertEqual(payload["location"], "Dong Thap")
        self.assertGreater(payload["generated_image_count"], 0)
        self.assertGreater(payload["image_features"]["image_feature_columns"], 0)
        self.assertGreater(payload["training_dataset"]["rows"], 0)
        self.assertGreaterEqual(len(payload["metrics"]), 1)
        self.assertTrue(all(metric["scope"] == "location" for metric in payload["metrics"]))
        self.assertEqual(payload["model_evaluation"]["metric_basis"], "macro_f1")
        self.assertGreater(payload["model_evaluation"]["test_rows"], 0)
        self.assertGreater(payload["metrics"][0]["test_rows"], 0)
        self.assertIn("correct_predictions", payload["metrics"][0])
        self.assertIn("test_class_distribution", payload["metrics"][0])
        self.assertTrue(all(sample["location"] == "Dong Thap" for sample in payload["generated_image_samples"]))

    def test_ai_training_metrics_change_by_location(self):
        dong_thap = get_ai_training_status("Dong Thap")
        can_tho = get_ai_training_status("Can Tho")

        self.assertNotEqual(
            [metric["macro_f1"] for metric in dong_thap["metrics"]],
            [metric["macro_f1"] for metric in can_tho["metrics"]],
        )

    def test_ai_training_train_endpoint_reports_status(self):
        fake_result = {
            "best_model": "decision_tree",
            "feature_count": 42,
            "metrics": [],
        }
        with (
            patch("src.data_pipeline.merge_data.main", return_value=None),
            patch("src.decision_model.train_decision_model.train_models", return_value=fake_result),
        ):
            payload = train_ai_model("Dong Thap")

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["step"], "train_model")
        self.assertEqual(payload["result"]["best_model"], "decision_tree")
        self.assertIn("ai_training", payload)
        self.assertEqual(payload["ai_training"]["location"], "Dong Thap")

    def test_pipeline_endpoint_reports_steps_without_fetching_weather(self):
        fake_result = {
            "status": "ok",
            "steps": [{"name": "fetch_weather", "status": "done", "rows": 10}],
            "raw_path": "data/raw/weather_raw_20260604_1200.csv",
            "clean_path": "data/clean/weather_clean_20260604_1200.csv",
            "rows": 10,
            "uploaded": True,
            "upload_error": None,
        }
        with patch("src.api.run_weather_pipeline", return_value=fake_result):
            payload = run_pipeline_endpoint(days=3)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["clean_path"], fake_result["clean_path"])
        self.assertEqual(payload["steps"][0]["name"], "fetch_weather")
        self.assertIn("duration_seconds", payload)


if __name__ == "__main__":
    unittest.main()
