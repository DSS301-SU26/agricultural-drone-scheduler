"""Smoke test HM6 (API). Chay inline hoac pytest. Yeu cau da train model."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_reference_endpoints():
    assert client.get("/api/health").json() == {"status": "ok"}
    assert len(client.get("/api/drones").json()) == 3
    assert len(client.get("/api/pesticides").json()) == 3
    assert len(client.get("/api/crop-stages").json()) == 4
    assert len(client.get("/api/locations").json()) == 6


def test_ml_metrics():
    m = client.get("/api/ml/metrics").json()
    assert m["champion"] == "random_forest"
    assert "metrics" in m


def test_decision_returns_slots():
    r = client.post("/api/decision", json={
        "latitude": 10.52, "longitude": 105.13, "drone_model": "DJI_T30",
        "pesticide": "Tricyclazole", "crop_stage": "TILLERING", "days": 2, "plot_id": 1,
    })
    assert r.status_code == 200
    j = r.json()
    assert j["source"] in ("forecast", "simulated_fallback")
    assert len(j["slots"]) > 0
    s = j["slots"][0]
    for k in ("decision", "flight_safety_score", "crop_impact_score", "spray_quality_score", "xai_explanation"):
        assert k in s


def test_invalid_drone_400():
    r = client.post("/api/decision", json={"latitude": 10.5, "longitude": 105.1, "drone_model": "NO_SUCH"})
    assert r.status_code == 400


def test_override_hard_lock_returns_423():
    r = client.post("/api/decision/override", json={
        "reason": "co ep bay", "drone_model": "DJI_T50", "hour": 15,
        "weather": {"timestamp": "2025-03-15 15:00:00", "temperature_2m": 30,
                    "relative_humidity_2m": 70, "precipitation": 0, "precipitation_probability": 10,
                    "cloud_cover": 30, "visibility": 12000, "wind_speed_10m": 25,
                    "wind_gusts_10m": 30, "weather_code": 1, "et0_fao_evapotranspiration": 4.5},
    })
    assert r.status_code == 423


def test_override_soft_success():
    r = client.post("/api/decision/override", json={
        "reason": "Dap dich ray nau khan cap", "drone_model": "XAG_P100_PRO",
        "pesticide": "Abamectin", "hour": 12,
        "weather": {"timestamp": "2025-03-15 12:00:00", "temperature_2m": 34,
                    "relative_humidity_2m": 60, "precipitation": 0, "precipitation_probability": 10,
                    "cloud_cover": 20, "visibility": 12000, "wind_speed_10m": 9,
                    "wind_gusts_10m": 14, "weather_code": 1, "et0_fao_evapotranspiration": 5.5},
    })
    assert r.status_code == 200
    j = r.json()
    assert j["decision"] == "FLY" and j["is_user_overridden"] is True
