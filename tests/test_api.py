import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.api import dashboard, health, locations, reset_decision_config, run_pipeline_endpoint, update_decision_config


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
