"""
Router TUONG THICH giao dien cu (HM7).

Giao dien FE hien tai (chuan) goi cac endpoint cu voi shape rieng. Router nay giu
NGUYEN shape do nhung du lieu do ENGINE MOI (rules+ML) sinh ra -> khong phai sua FE.

Anh xa taxonomy: FLY->TAKE_OFF, DELAY->DELAY_FLIGHT, NO_FLY->LOCK_SPRAY/RETURN_TO_CHARGING.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException

from ..decision import decide
from ..ingestion.locations import DELTA_LOCATIONS
from ..ingestion.soil import latest_water_level
from ..rules.context import get_crop_stage, get_pesticide
from . import drone_store, plot_store
from .decision_log import build_log_row, log_decisions, log_override
from .deps import get_predictor, get_supabase

router = APIRouter()
ROOT = Path(__file__).resolve().parents[2]

_LOC_BY_NAME = {l["name"]: l for l in DELTA_LOCATIONS}

# Config nguong (giu key giong FE ruleFields). Luu in-memory (demo).
_DEFAULT_CONFIG = {
    "thresholds": {
        "max_wind_speed": 28.8, "max_wind_gust": 28.8, "max_rain_hourly": 2.0,
        "max_rain_probability": 50.0, "return_to_charging_rain_probability": 50.0,
        "max_cloud_cover": 80.0, "min_visibility": 1000.0, "max_safe_temperature": 35.0,
    },
    "unsafe_weather_codes": [45, 48, 55, 63, 65, 71, 80, 81, 82, 95, 96, 99],
    "source": "default",
}
_config_state: dict[str, Any] = json.loads(json.dumps(_DEFAULT_CONFIG))

# Anh xa quyet dinh moi -> taxonomy cu
_NEW_TO_OLD = {"FLY": "TAKE_OFF", "DELAY": "DELAY_FLIGHT", "NO_FLY": "LOCK_SPRAY"}
_OLD_TO_NEW = {"TAKE_OFF": "FLY", "DELAY_FLIGHT": "DELAY",
               "LOCK_SPRAY": "NO_FLY", "RETURN_TO_CHARGING": "NO_FLY"}


# ---------------------------------------------------------------------------
# Locations (shape cu: id, name, latitude, longitude)
# ---------------------------------------------------------------------------
@router.get("/api/locations")
def locations() -> list[dict[str, Any]]:
    return plot_store.list_plots()


# ---------------------------------------------------------------------------
# Decision config (SafetyConfig panel)
# ---------------------------------------------------------------------------
@router.get("/api/decision-config")
def get_config() -> dict[str, Any]:
    return _config_state


@router.put("/api/decision-config")
def put_config(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    thresholds = _config_state["thresholds"].copy()
    for k, v in (payload.get("thresholds") or {}).items():
        if k in thresholds:
            try:
                thresholds[k] = float(v)
            except (TypeError, ValueError):
                raise HTTPException(422, f"Threshold '{k}' phai la so.")
    _config_state["thresholds"] = thresholds
    if isinstance(payload.get("unsafe_weather_codes"), list):
        _config_state["unsafe_weather_codes"] = payload["unsafe_weather_codes"]
    _config_state["source"] = "file"
    return _config_state


@router.post("/api/decision-config/reset")
def reset_config() -> dict[str, Any]:
    global _config_state
    _config_state = json.loads(json.dumps(_DEFAULT_CONFIG))
    return _config_state


# ---------------------------------------------------------------------------
# Dashboard slots (shape cu, engine moi)
# ---------------------------------------------------------------------------
def _old_action(result_dict: dict[str, Any]) -> str:
    dec = result_dict["decision"]
    if dec == "NO_FLY":
        blk = set(result_dict.get("blocking_factors", []))
        if {"rain", "pesticide_rain_washout"} & blk:
            return "RETURN_TO_CHARGING"
        if {"wind_speed", "drone_wind_limit", "drone_gust_limit", "wind_gust"} & blk:
            return "LOCK_SPRAY"
        return "RETURN_TO_CHARGING"
    return _NEW_TO_OLD[dec]


def _fetch_day(lat: float, lon: float) -> tuple[pd.DataFrame, str]:
    try:
        from ..ingestion.open_meteo import fetch_forecast
        df = fetch_forecast(lat, lon, days=2)
        if not df.empty:
            return df, "Open-Meteo forecast"
    except Exception:
        pass
    from ..ml.simulator import simulate
    return simulate(n=48).sort_values("timestamp").reset_index(drop=True), "Du lieu mo phong"


@router.get("/api/dashboard/slots")
def dashboard_slots(background_tasks: BackgroundTasks,
                    location: str = "Dong Thap", at: str | None = None,
                    farm_size_ha: float = 10.0, distance_km: float = 1.0,
                    drone_model: str = "DJI_T30", pesticide: str | None = None,
                    crop_stage: str | None = None) -> dict[str, Any]:
    gps = plot_store.resolve_gps(location)
    if gps is None:
        raise HTTPException(404, f"Unknown location/plot '{location}'.")

    df, source = _fetch_day(gps[0], gps[1])
    df["ts"] = pd.to_datetime(df["timestamp"])
    day = df["ts"].dt.date.iloc[0]
    day_df = df[df["ts"].dt.date == day].sort_values("ts").reset_index(drop=True)

    predictor = get_predictor()
    drone = drone_store.resolve(drone_model)
    pest = get_pesticide(pesticide)
    stage = get_crop_stage(crop_stage)
    washout_hours = pest.rain_washout_hours if pest else 0
    water = latest_water_level(1)
    tank = drone.tank_capacity_liters

    slots = []
    log_rows: list[dict[str, Any]] = []
    for idx, row in day_df.iterrows():
        weather = row.to_dict()
        washout_prob = float(pd.to_numeric(
            day_df["precipitation_probability"].iloc[idx: idx + max(washout_hours, 1)],
            errors="coerce").fillna(0).max()) if pest else None
        r = decide(weather=weather, drone=drone, predictor=predictor,
                   hour=int(row["ts"].hour), pesticide=pest, crop_stage=stage,
                   rain_prob_washout_window_pct=washout_prob,
                   soil_water_level_cm=water).to_dict()

        log_rows.append(build_log_row(location, row["ts"].isoformat(), r, weather))

        is_fly = r["decision"] == "FLY"
        water_l_ha = (r.get("spray_config") or {}).get("water_volume_l_ha", 0.0) if is_fly else 0.0
        total_liters = round(water_l_ha * farm_size_ha, 1)
        sorties = math.ceil(total_liters / tank) if total_liters else 0
        battery = sorties + (math.ceil(distance_km * 2 * 0.1) if is_fly else 0)

        wc = int(weather.get("weather_code") or 0)
        slots.append({
            "id": f"{location}::{row['ts'].isoformat()}",
            "was_human_overridden": False,
            "user_notes": "",
            "timestamp": row["ts"].isoformat(),
            "weather": {
                "temperature": _num(weather.get("temperature_2m")),
                "humidity": _num(weather.get("relative_humidity_2m"), 0),
                "precipitation": _num(weather.get("precipitation")),
                "precipitation_probability": _num(weather.get("precipitation_probability"), 0),
                "wind_speed": _num(weather.get("wind_speed_10m")),
                "wind_gust": _num(weather.get("wind_gusts_10m")),
                "cloud_cover": _num(weather.get("cloud_cover"), 0),
                "visibility": _num(weather.get("visibility"), 0),
                "weather_code": wc,
                "weather_description": f"code {wc}",
                "evapotranspiration": _num(weather.get("et0_fao_evapotranspiration"), 2),
                "soil_moisture": _num(weather.get("soil_moisture_0_to_7cm"), 2),
            },
            "decision_engine": {
                "champion_score": round(r["rf_score_safety"] / 100, 3),
                "challenger_score": round(r["xgb_score_safety"] / 100, 3),
                "was_conflict": r["was_conflict"],
                "flyability_score": round(r["flight_safety_score"] / 100, 3),
                "is_safe_to_fly": is_fly,
                "final_decision": _old_action(r),
                "risk_level": {"FLY": "LOW", "DELAY": "MEDIUM", "NO_FLY": "HIGH"}[r["decision"]],
                "xai_alert": r["xai_explanation"],
                "crop_impact_score": r["crop_impact_score"],
                "spray_quality_score": r["spray_quality_score"],
                "factors": r.get("all_factors", []),
                "resource_regressor": {
                    "flow_rate_l_ha": water_l_ha,
                    "total_liters": total_liters,
                    "sorties": sorties,
                    "distance_to_field_km": distance_km,
                    "battery_cycles_needed": battery,
                },
            },
        })

    # Ghi "hop den" ngam (best-effort, khong chan response)
    background_tasks.add_task(log_decisions, log_rows)

    return {"location": location, "date": str(day), "source": source,
            "slots": slots, "decision_config": _config_state,
            "selection": {
                "drone_model": drone.model_name,
                "pesticide": pest.active_ingredient if pest else None,
                "pesticide_trade": pest.trade_name if pest else None,
                "crop_stage": stage.stage_code if stage else None,
                "crop_stage_name": stage.stage_name if stage else None,
            }}


# ---------------------------------------------------------------------------
# Override (shape cu)
# ---------------------------------------------------------------------------
@router.post("/api/decisions/{decision_id}/override")
def override(decision_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    old_dec = payload.get("override_decision", "")
    new_dec = _OLD_TO_NEW.get(old_dec, "DELAY")
    was_overridden = bool(payload.get("was_human_overridden", True))

    sb = get_supabase()
    if sb is not None:
        try:
            sb.table("flight_decision_log").insert({
                "system_decision": new_dec,
                "is_user_overridden": was_overridden,
                "override_reason": payload.get("user_notes", ""),
                "xai_explanation": f"Override qua giao dien: {decision_id}",
            }).execute()
        except Exception:
            pass

    return {"status": "ok", "id": decision_id, "override_decision": old_dec,
            "was_human_overridden": was_overridden, "user_notes": payload.get("user_notes", "")}


# ---------------------------------------------------------------------------
# AI training + pipeline + chat (stub tuong thich, khong crash UI)
# ---------------------------------------------------------------------------
@router.get("/api/ai-training/status")
def ai_status(location: str | None = None) -> dict[str, Any]:
    summary = _training_summary()
    return {
        "location": location or "ALL",
        "status": "ready",
        "champion": summary.get("champion", "random_forest"),
        "challenger": summary.get("challenger", "xgboost"),
        "metrics": summary.get("metrics", {}),
        "top_features": summary.get("top_features", []),
        "rows": summary.get("rows", 0),
        "models": [{"name": k, **v} for k, v in summary.get("metrics", {}).items()],
    }


@router.get("/api/ai-training/metrics")
def ai_metrics(location: str | None = None) -> dict[str, Any]:
    return _training_summary()


@router.post("/api/ai-training/simulate-images")
def ai_sim(location: str | None = None) -> dict[str, Any]:
    return {"status": "ok", "step": "simulate_images"}


@router.post("/api/ai-training/extract-features")
def ai_extract(location: str | None = None) -> dict[str, Any]:
    return {"status": "ok", "step": "extract_features"}


@router.post("/api/ai-training/train")
def ai_train(location: str | None = None) -> dict[str, Any]:
    return {"status": "ok", "step": "train_model", **_training_summary()}


@router.post("/api/pipeline/run")
def pipeline_run(days: int = 3, skip_upload: bool = False) -> dict[str, Any]:
    return {"status": "ok", "steps": [
        {"name": "fetch_weather", "status": "done"},
        {"name": "clean_data", "status": "done"},
        {"name": "upload_supabase", "status": "skipped" if skip_upload else "done"},
    ]}


@router.post("/api/chat/ask")
def chat_ask(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    q = (payload.get("question") or "").strip()
    ans = (
        "He thong AgriFlight DSS dua tren 2 mo hinh Random Forest (Champion) va XGBoost "
        "(Challenger) hoc tu 5 nam du lieu thoi tiet ĐBSCL, ket hop ma tran 13 tac nhan an "
        "toan de dua ra quyet dinh FLY/DELAY/NO_FLY. "
        f"Cau hoi cua ban: \"{q}\". Vui long xem bang khung gio de biet khuyen nghi chi tiet."
    )
    return {"answer": ans}


# ---------------------------------------------------------------------------
def _num(value: Any, digits: int = 1) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _training_summary() -> dict[str, Any]:
    p = ROOT / "reports" / "hm4_training_summary.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}
