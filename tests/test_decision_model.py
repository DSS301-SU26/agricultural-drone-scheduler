import unittest

import pandas as pd

from src.decision_model.decision_engine import (
    calculate_flyability_score,
    derive_decision_action,
    derive_risk_level,
)


class DeriveDecisionActionTest(unittest.TestCase):
    def make_row(self, **overrides):
        values = {
            "weather_code": 1000,
            "precipitation": 0.0,
            "precipitation_probability": 10.0,
            "wind_speed_10m": 8.0,
            "wind_gusts_10m": 14.0,
            "temperature_2m": 30.0,
        }
        values.update(overrides)
        return pd.Series(values)

    def test_take_off_in_safe_weather(self):
        self.assertEqual(derive_decision_action(self.make_row()), "TAKE_OFF")

    def test_delay_flight_for_extreme_heat(self):
        row = self.make_row(temperature_2m=37.0)
        self.assertEqual(derive_decision_action(row), "DELAY_FLIGHT")

    def test_extreme_heat_reduces_score_and_risk_matches_delay(self):
        row = self.make_row(temperature_2m=37.0)
        row["flyability_score"] = calculate_flyability_score(row)
        action = derive_decision_action(row)

        self.assertLess(row["flyability_score"], 1.0)
        self.assertEqual(action, "DELAY_FLIGHT")
        self.assertEqual(derive_risk_level(row, action), "MEDIUM")

    def test_lock_spray_for_strong_wind(self):
        row = self.make_row(wind_speed_10m=26.0, wind_gusts_10m=36.0)
        self.assertEqual(derive_decision_action(row), "LOCK_SPRAY")

    def test_return_to_charging_for_weatherapi_thunder_code(self):
        row = self.make_row(weather_code=1087)
        self.assertEqual(derive_decision_action(row), "RETURN_TO_CHARGING")

    def test_return_to_charging_for_wmo_thunder_code(self):
        row = self.make_row(weather_code=95)
        self.assertEqual(derive_decision_action(row), "RETURN_TO_CHARGING")


if __name__ == "__main__":
    unittest.main()
