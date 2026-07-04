"""
AgriFlight DSS API (HM6) - boc decide() thanh REST.

Chay: .venv/bin/uvicorn app.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ..decision import apply_override, decide
from ..ingestion.locations import DELTA_LOCATIONS
from ..ingestion.soil import latest_water_level
from ..rules.context import CROP_STAGES, DRONES, PESTICIDES, get_crop_stage, get_drone, get_pesticide, DroneProfile
from . import drone_store, plot_store
from .compat import router as compat_router
from .decision_log import log_override, recent_logs
from .deps import get_predictor, get_supabase
from .schemas import DecisionRequest, DecisionResponse, OverrideRequest, DroneCreate

app = FastAPI(title="AgriFlight DSS API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
# Router tuong thich giao dien FE cu (giu UI chuan khong doi)
app.include_router(compat_router)

DAYTIME = range(6, 18)  # gio bay thuc te 06-17h


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/drones")
def list_drones() -> list[dict[str, Any]]:
    sb = get_supabase()
    if sb is not None:
        try:
            res = sb.table("drone_profiles").select("*").execute()
            drones = []
            for row in res.data:
                drones.append({
                    "drone_id": row.get("drone_id"),
                    "model_name": row.get("model_name"),
                    "max_wind_resistance_kph": row.get("max_wind_resistance_kph"),
                    "max_gust_resistance_kph": row.get("max_gust_resistance_kph"),
                    "tank_capacity_liters": row.get("tank_capacity_liters"),
                    "spray_system_type": row.get("nozzle_technology"),
                    "ip_rating": row.get("ingress_protection"),
                    "image_url": row.get("image_url"),
                    "notes": row.get("notes")
                })
            return drones
        except Exception as e:
            print(f"Error fetching drones from DB: {e}")
    drones_fe = []
    for d in DRONES.values():
        d_dict = asdict(d)
        d_dict["spray_system_type"] = d_dict.pop("nozzle_technology")
        d_dict["ip_rating"] = d_dict.pop("ingress_protection")
        drones_fe.append(d_dict)
    return drones_fe


@app.post("/api/drones")
def add_drone(drone: DroneCreate):
    sb = get_supabase()
    db_data = {
        "model_name": drone.model_name,
        "max_wind_resistance_kph": drone.max_wind_resistance_kph,
        "max_gust_resistance_kph": drone.max_gust_resistance_kph,
        "tank_capacity_liters": int(drone.tank_capacity_liters),
        "nozzle_technology": drone.spray_system_type,
        "ingress_protection": drone.ip_rating,
        "image_url": drone.image_url,
        "notes": drone.notes
    }
    if sb is not None:
        try:
            res = sb.table("drone_profiles").insert(db_data).execute()
            if res.data:
                db_data["drone_id"] = res.data[0].get("drone_id")
        except Exception as e:
            print(f"Error inserting drone to DB: {e}")

    DRONES[drone.model_name] = DroneProfile(**db_data)
    
    fe_data = dict(db_data)
    fe_data["spray_system_type"] = fe_data.pop("nozzle_technology")
    fe_data["ip_rating"] = fe_data.pop("ingress_protection")
    return {"status": "ok", "drone": fe_data}


@app.put("/api/drones/{drone_id}")
def update_drone(drone_id: int, drone: DroneCreate):
    sb = get_supabase()
    if sb is None:
        raise HTTPException(503, "Supabase connection not configured")
    
    data = {
        "model_name": drone.model_name,
        "max_wind_resistance_kph": drone.max_wind_resistance_kph,
        "max_gust_resistance_kph": drone.max_gust_resistance_kph,
        "tank_capacity_liters": int(drone.tank_capacity_liters),
        "nozzle_technology": drone.spray_system_type,
        "ingress_protection": drone.ip_rating,
        "image_url": drone.image_url,
        "notes": drone.notes
    }
    try:
        resp = sb.table("drone_profiles").update(data).eq("drone_id", drone_id).execute()
        if not resp.data:
            raise HTTPException(404, "Drone not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error updating drone: {str(e)}")

@app.delete("/api/drones/{drone_id}")
def delete_drone(drone_id: int):
    sb = get_supabase()
    if sb is None:
        raise HTTPException(503, "Supabase connection not configured")
    try:
        sb.table("drone_profiles").delete().eq("drone_id", drone_id).execute()
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, f"Error deleting drone: {str(e)}")


@app.get("/api/pesticides")
def list_pesticides() -> list[dict[str, Any]]:
    return [asdict(p) for p in PESTICIDES.values()]


@app.get("/api/crop-stages")
def list_crop_stages() -> list[dict[str, Any]]:
    return [asdict(s) for s in CROP_STAGES.values()]


@app.get("/api/plots")
def list_plots() -> list[dict[str, Any]]:
    return plot_store.list_plots()


@app.post("/api/plots")
def create_plot(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return plot_store.add_plot(payload)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.put("/api/plots/{plot_id}")
def edit_plot(plot_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return plot_store.update_plot(plot_id, payload)
    except KeyError:
        raise HTTPException(404, f"Khong tim thay vuon id {plot_id}")


@app.delete("/api/plots/{plot_id}")
def remove_plot(plot_id: int) -> dict[str, str]:
    plot_store.delete_plot(plot_id)
    return {"status": "deleted", "plot_id": str(plot_id)}


@app.get("/api/decision-log")
def decision_log(limit: int = 100, location: str | None = None) -> list[dict[str, Any]]:
    """Nhat ky quyet dinh ("hop den") - doc tu file local, luon hoat dong."""
    return recent_logs(limit=limit, location=location)


@app.get("/api/ml/metrics")
def ml_metrics() -> dict[str, Any]:
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "reports" / "hm4_training_summary.json"
    if not p.exists():
        raise HTTPException(404, "Chua co bao cao train. Chay app.ml.train truoc.")
    return json.loads(p.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# Decision core
# ----------------------------------------------------------------------------
def _fetch_slots(lat: float, lon: float, days: int) -> tuple[pd.DataFrame, str]:
    """Lay du bao live; neu loi mang -> fallback mo phong de API van chay."""
    try:
        from ..ingestion.open_meteo import fetch_forecast
        df = fetch_forecast(lat, lon, days=days)
        if not df.empty:
            return df, "forecast"
    except Exception:
        pass
    from ..ml.simulator import simulate
    df = simulate(n=days * 14).sort_values("timestamp").reset_index(drop=True)
    return df, "simulated_fallback"


def _washout_window_prob(df: pd.DataFrame, idx: int, hours: int) -> float:
    """Xac suat mua cao nhat trong `hours` gio ke tiep (cho luat rua troi)."""
    window = df["precipitation_probability"].iloc[idx: idx + max(hours, 1)]
    return float(pd.to_numeric(window, errors="coerce").fillna(0).max())


def _build_slots(df: pd.DataFrame, req: DecisionRequest) -> list[dict[str, Any]]:
    predictor = get_predictor()
    drone = drone_store.resolve(req.drone_model)
    pesticide = get_pesticide(req.pesticide)
    stage = get_crop_stage(req.crop_stage)
    water_level = latest_water_level(req.plot_id or 1)
    washout_hours = pesticide.rain_washout_hours if pesticide else 0

    slots: list[dict[str, Any]] = []
    df = df.reset_index(drop=True)
    for idx, row in df.iterrows():
        ts = pd.to_datetime(row["timestamp"])
        if ts.hour not in DAYTIME:
            continue
        weather = row.to_dict()
        result = decide(
            weather=weather, drone=drone, predictor=predictor, hour=int(ts.hour),
            pesticide=pesticide, crop_stage=stage,
            rain_prob_washout_window_pct=_washout_window_prob(df, idx, washout_hours),
            soil_water_level_cm=water_level,
            rain_24h_forecast_mm=float(pd.to_numeric(
                df["precipitation"].iloc[idx: idx + 24], errors="coerce").fillna(0).sum()),
        )
        d = result.to_dict()
        d.update({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "hour": int(ts.hour),
                  "weather": {k: (None if pd.isna(v) else v) for k, v in weather.items()}})
        slots.append(d)
    return slots


@app.post("/api/decision", response_model=DecisionResponse)
def get_decision(req: DecisionRequest) -> dict[str, Any]:
    if req.drone_model not in DRONES:
        raise HTTPException(400, f"Drone khong hop le. Chon: {list(DRONES)}")
    df, source = _fetch_slots(req.latitude, req.longitude, req.days)
    slots = _build_slots(df, req)
    if not slots:
        raise HTTPException(502, "Khong co khung gio bay nao.")
    fly = [s for s in slots if s["decision"] == "FLY"]
    best = max(fly, key=lambda s: s["flight_safety_score"]) if fly else None
    return {
        "source": source,
        "location": {"latitude": req.latitude, "longitude": req.longitude},
        "drone_model": req.drone_model,
        "slots": slots,
        "best_slot": best,
        "awd": slots[0].get("awd"),
    }


@app.post("/api/decision/override")
def override_decision(req: OverrideRequest) -> dict[str, Any]:
    try:
        predictor = get_predictor()
        
        # Robust fallback for frontend sending sanitized keys or missing timestamp
        w = req.weather
        if "timestamp" not in w:
            import datetime
            w["timestamp"] = datetime.datetime.now().isoformat()
        if "temperature_2m" not in w and "temperature" in w:
            w["temperature_2m"] = w["temperature"]
        if "relative_humidity_2m" not in w and "humidity" in w:
            w["relative_humidity_2m"] = w["humidity"]
        if "wind_speed_10m" not in w and "wind_speed" in w:
            w["wind_speed_10m"] = w["wind_speed"]
        if "wind_gusts_10m" not in w and "wind_gust" in w:
            w["wind_gusts_10m"] = w["wind_gust"]
        for key in ["precipitation", "precipitation_probability", "cloud_cover", "visibility", "weather_code", "et0_fao_evapotranspiration", "soil_moisture_0_to_7cm", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m"]:
            if key not in w:
                w[key] = 0.0

        result = decide(
            weather=w, drone=get_drone(req.drone_model), predictor=predictor,
            hour=req.hour, pesticide=get_pesticide(req.pesticide),
            crop_stage=get_crop_stage(req.crop_stage),
        )
        try:
            result = apply_override(result, req.reason)
        except PermissionError as e:
            raise HTTPException(423, str(e))     # 423 Locked
        except ValueError as e:
            raise HTTPException(400, str(e))

        log_override(result.to_dict(), req.weather, req.location)
        return result.to_dict()
    except Exception as e:
        import traceback
        raise HTTPException(500, traceback.format_exc())
