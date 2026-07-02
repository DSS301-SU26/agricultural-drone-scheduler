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
from ..rules.context import CROP_STAGES, DRONES, PESTICIDES, get_crop_stage, get_drone, get_pesticide
from .compat import router as compat_router
from .deps import get_predictor, get_supabase
from .schemas import DecisionRequest, DecisionResponse, OverrideRequest

app = FastAPI(title="AgriFlight DSS API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
# Router tuong thich giao dien FE cu (giu UI chuan khong doi)
app.include_router(compat_router)

DAYTIME = range(5, 19)  # gio bay thuc te 05-18h


# ----------------------------------------------------------------------------
# Reference data
# ----------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/drones")
def list_drones() -> list[dict[str, Any]]:
    return [asdict(d) for d in DRONES.values()]


@app.get("/api/pesticides")
def list_pesticides() -> list[dict[str, Any]]:
    return [asdict(p) for p in PESTICIDES.values()]


@app.get("/api/crop-stages")
def list_crop_stages() -> list[dict[str, Any]]:
    return [asdict(s) for s in CROP_STAGES.values()]


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
    drone = get_drone(req.drone_model)
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
    predictor = get_predictor()
    result = decide(
        weather=req.weather, drone=get_drone(req.drone_model), predictor=predictor,
        hour=req.hour, pesticide=get_pesticide(req.pesticide),
        crop_stage=get_crop_stage(req.crop_stage),
    )
    try:
        result = apply_override(result, req.reason)
    except PermissionError as e:
        raise HTTPException(423, str(e))     # 423 Locked
    except ValueError as e:
        raise HTTPException(400, str(e))

    _log_decision(result.to_dict(), req)
    return result.to_dict()


def _log_decision(result: dict[str, Any], req: OverrideRequest) -> None:
    """Ghi flight_decision_log (best-effort; khong chan API neu DB loi)."""
    sb = get_supabase()
    if sb is None:
        return
    try:
        sb.table("flight_decision_log").insert({
            "mission_id": req.mission_id,
            "rf_score_safety": result["rf_score_safety"],
            "xgb_score_safety": result["xgb_score_safety"],
            "flight_safety_score": result["flight_safety_score"],
            "crop_impact_score": result["crop_impact_score"],
            "spray_quality_score": result["spray_quality_score"],
            "system_decision": result["decision"],
            "is_user_overridden": result["is_user_overridden"],
            "override_reason": result["override_reason"],
            "xai_explanation": result["xai_explanation"],
        }).execute()
    except Exception:
        pass
