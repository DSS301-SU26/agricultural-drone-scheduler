import unittest

from src.api import dashboard, health, locations


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


if __name__ == "__main__":
    unittest.main()
