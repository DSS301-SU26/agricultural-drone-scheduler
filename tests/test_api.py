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
    get_dashboard_slots,
    chat_ask,
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
            from src.api import ROOT
            fixed_dataset = ROOT / "src" / "data" / "clean" / "weather_clean_20260602_1537.csv"
            with (
                patch("src.api.CONFIG_DIR", Path(temp_dir)),
                patch("src.api.DECISION_CONFIG_PATH", config_path),
                patch("src.api.latest_clean_dataset", return_value=fixed_dataset),
            ):
                baseline = dashboard(location="Dong Thap", at="2026-06-04T06:00:00")
                self.assertEqual(baseline["current"]["decision_action"], "DELAY_FLIGHT")

                update_decision_config({"thresholds": {"max_wind_speed": 5}})
                restricted = dashboard(location="Dong Thap", at="2026-06-04T06:00:00")

                self.assertEqual(restricted["decision_config"]["source"], "file")
                self.assertEqual(restricted["current"]["decision_action"], "LOCK_SPRAY")

                reset_decision_config()
                restored = dashboard(location="Dong Thap", at="2026-06-04T06:00:00")
                self.assertEqual(restored["current"]["decision_action"], "DELAY_FLIGHT")

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

    def test_get_dashboard_slots_returns_correct_json(self):
        payload = get_dashboard_slots(location="Dong Thap", at="2026-06-04T06:00:00", farm_size_ha=10.0)
        self.assertEqual(payload["location"], "Dong Thap")
        self.assertIn("date", payload)
        self.assertGreater(len(payload["slots"]), 0)
        
        slot = payload["slots"][0]
        self.assertIn("timestamp", slot)
        self.assertIn("weather", slot)
        self.assertIn("decision_engine", slot)
        
        weather = slot["weather"]
        self.assertIn("temperature", weather)
        self.assertIn("humidity", weather)
        self.assertIn("precipitation", weather)
        self.assertIn("precipitation_probability", weather)
        self.assertIn("wind_speed", weather)
        self.assertIn("wind_gust", weather)
        self.assertIn("cloud_cover", weather)
        self.assertIn("visibility", weather)
        self.assertIn("weather_code", weather)
        self.assertIn("weather_description", weather)
        self.assertIn("evapotranspiration", weather)
        self.assertIn("soil_moisture", weather)
        
        decision_engine = slot["decision_engine"]
        self.assertIn("champion_prediction", decision_engine)
        self.assertIn("champion_confidence", decision_engine)
        self.assertIn("challenger_prediction", decision_engine)
        self.assertIn("challenger_confidence", decision_engine)
        self.assertIn("was_conflict", decision_engine)
        self.assertIn("final_decision", decision_engine)
        self.assertIn("risk_level", decision_engine)
        self.assertIn("xai_alert", decision_engine)
        
        resource_regressor = decision_engine["resource_regressor"]
        self.assertIn("flow_rate_l_ha", resource_regressor)
        self.assertIn("total_liters", resource_regressor)
        self.assertIn("sorties", resource_regressor)
        self.assertIn("battery_cycles", resource_regressor)
        self.assertEqual(resource_regressor["battery_cycles"], resource_regressor["sorties"])

    def test_chat_ask_returns_vietnamese_answer(self):
        fake_log = [{
            "timestamp": "2026-06-25T10:00:00+07:00",
            "weather_snapshot": {
                "location_name": "Dong Thap",
                "temperature_2m": 30.0,
                "relative_humidity_2m": 72.0,
                "precipitation_probability": 10.0,
                "precipitation": 0.0,
                "wind_speed_10m": 8.0,
                "wind_gusts_10m": 14.0,
                "weather_code": 1,
            },
            "champion_pred": "TAKE_OFF",
            "champion_conf": 0.95,
            "challenger_pred": "TAKE_OFF",
            "challenger_conf": 0.92,
            "final_decision": "TAKE_OFF",
            "was_conflict": False,
            "was_human_overridden": False
        }]
        
        with patch("src.database.get_client") as mock_client_getter:
            mock_client = mock_client_getter.return_value
            mock_client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = fake_log
            
            payload = chat_ask({"question": "Thời tiết ở Đồng Tháp thế nào?"})
            self.assertIn("answer", payload)
            self.assertIn("Dong Thap", payload["answer"])
            self.assertIn("TAKE_OFF", payload["answer"])
            self.assertGreater(payload["retrieved_logs_count"], 0)


if __name__ == "__main__":
    unittest.main()
