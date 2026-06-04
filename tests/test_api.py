import unittest
from unittest.mock import patch

from src.api import dashboard, health, locations, run_pipeline_endpoint


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
        self.assertIn(payload["current"]["decision_action"], {
            "TAKE_OFF",
            "DELAY_FLIGHT",
            "LOCK_SPRAY",
            "RETURN_TO_CHARGING",
        })
        self.assertEqual(len(payload["forecast"]), 12)
        self.assertEqual(len(payload["timeline_tiles"]), 12)
        self.assertEqual(len(payload["kpis"]), 3)

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
