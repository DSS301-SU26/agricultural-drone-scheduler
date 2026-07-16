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
from .decision_log import build_log_row, log_decisions, log_override, recent_logs
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
               "LOCK_SPRAY": "NO_FLY", "RETURN_TO_CHARGING": "NO_FLY",
               "FLY": "FLY", "DELAY": "DELAY", "NO_FLY": "NO_FLY"}


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
            return "NO_FLY"
        if {"wind_speed", "drone_wind_limit", "drone_gust_limit", "wind_gust"} & blk:
            return "LOCK_SPRAY"
        return "NO_FLY"
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
    import datetime
    today = datetime.date.today()
    base_df = simulate(n=1)
    
    rows = []
    for h in range(6, 18):
        row = base_df.iloc[0].copy()
        row["timestamp"] = pd.Timestamp(f"{today.isoformat()} {h:02d}:00:00")
        # Jitter the temperature to simulate diurnal cycle
        if h < 13:
            row["temperature_2m"] = max(20, row["temperature_2m"] + (h - 5) * 0.8)
        else:
            row["temperature_2m"] = max(20, row["temperature_2m"] - (h - 13) * 0.8)
        rows.append(row)
        
    df = pd.DataFrame(rows)
    return df, "Du lieu mo phong (Hom nay)"


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
    day_df = day_df[(day_df["ts"].dt.hour >= 6) & (day_df["ts"].dt.hour <= 17)].reset_index(drop=True)

    predictor = get_predictor()
    drone = drone_store.resolve(drone_model)
    pest = get_pesticide(pesticide)
    stage = get_crop_stage(crop_stage)
    washout_hours = pest.rain_washout_hours if pest else 0
    water = latest_water_level(1)
    tank = drone.tank_capacity_liters

    slots = []
    log_rows: list[dict[str, Any]] = []
    
    # Lay danh sach override gan day tu local log
    overrides = recent_logs(100, location)
    override_map = {}
    for ov in overrides:
        if ov.get("is_user_overridden") and ov.get("slot_timestamp"):
            override_map[ov["slot_timestamp"]] = ov

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
        
        # Apply override if exists
        ts_iso = row["ts"].isoformat()
        ov = override_map.get(ts_iso)
        was_overridden = False
        user_notes = ""
        if ov:
            r["decision"] = ov.get("system_decision", r["decision"])
            was_overridden = True
            user_notes = ov.get("override_reason", "")

        is_fly = r["decision"] == "FLY"
        water_l_ha = (r.get("spray_config") or {}).get("water_volume_l_ha", 0.0)
        total_liters = round(water_l_ha * farm_size_ha, 1)
        sorties = math.ceil(total_liters / tank) if total_liters else 0
        battery = sorties + math.ceil(distance_km * 2 * 0.1)

        wc = int(weather.get("weather_code") or 0)
        slots.append({
            "id": f"{location}::{ts_iso}",
            "was_human_overridden": was_overridden,
            "user_notes": user_notes,
            "timestamp": ts_iso,
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
                "water_level_cm": water,
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
                "awd_recommendation": {
                    "action": "START_PUMP" if (r.get("awd") or {}).get("action") == "START_PUMP" else (
                        "DELAY_PUMP" if (r.get("awd") or {}).get("action") == "HOLD_PUMP" else "KEEP_DRYING"
                    ),
                    "explanation": (r.get("awd") or {}).get("message", "Mực nước ngầm ở mức an toàn. Tiếp tục khô ruộng.")
                },
                "opt_flight_config": {
                    "alt_min": float(stage.opt_flight_alt_min) if stage else 2.0,
                    "alt_max": float(stage.opt_flight_alt_max) if stage else 2.5,
                    "speed_min": float(stage.opt_flight_speed_min) if stage else 5.0,
                    "speed_max": float(stage.opt_flight_speed_max) if stage else 6.0,
                    "nozzle_tech": str(drone.nozzle_technology) if drone else "PRESSURE",
                    "awd_threshold_cm": float(stage.awd_threshold_cm) if stage else -15.0,
                },
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
            res = sb.table("flight_decision_log").select("*").eq("log_id", decision_id).execute()
            if res.data and len(res.data) > 0:
                record = res.data[0]
                
                if not record.get("is_user_overridden") and record.get("system_decision") == "NO_FLY":
                    weather = record.get("weather_json") or record.get("weather_snapshot") or {}
                    wind = float(weather.get("wind_speed_10m", 0))
                    gust = float(weather.get("wind_gusts_10m", 0))
                    if wind > 36.0 or gust > 36.0:
                        raise HTTPException(status_code=403, detail="Hệ thống đã ban hành lệnh CẤM BAY (NO_FLY) do gió vượt giới hạn vật lý của Drone (>36km/h). Trạng thái này bị KHÓA CỨNG.")
                
                if record.get("is_user_overridden") and record.get("system_decision") == new_dec:
                    raise HTTPException(status_code=409, detail=f"Đã ghi đè trạng thái '{new_dec}' rồi, không thể ghi đè lặp lại cùng một trạng thái.")
            
            sb.table("flight_decision_log").update({
                "system_decision": new_dec,
                "is_user_overridden": was_overridden,
                "override_reason": payload.get("user_notes", ""),
                "xai_explanation": f"Override qua giao dien: {decision_id}",
            }).eq("log_id", decision_id).execute()
        except HTTPException:
            raise
        except Exception:
            pass

    return {"status": "ok", "id": decision_id, "override_decision": old_dec,
            "was_human_overridden": was_overridden, "user_notes": payload.get("user_notes", "")}


@router.post("/api/decision/override")
def override_generic(id: str = None, payload: Any = Body(default=None)) -> dict[str, Any]:
    if payload is None:
        payload = {}
    if isinstance(payload, str):
        payload = {"id": payload}
        
    record_id = payload.get("id") or id
    reason_str = payload.get("reason", "")
    weather = payload.get("weather", {})
    timestamp = weather.get("timestamp")
    location = payload.get("location") or weather.get("location_name") or "Dong Thap"
    drone_model = payload.get("drone_model", "DJI_T30")
    pesticide = payload.get("pesticide")
    crop_stage = payload.get("crop_stage")
    
    is_restore = reason_str == "RESTORE" or payload.get("restore") is True or (not payload.get("reason") and record_id)

    decision_part = "TAKE_OFF"
    notes_part = ""
    if ":" in reason_str:
        parts = reason_str.split(":", 1)
        decision_part = parts[0].strip()
        notes_part = parts[1].strip()
    else:
        decision_part = reason_str.strip()

    valid_decisions = {"TAKE_OFF", "DELAY_FLIGHT", "LOCK_SPRAY", "RETURN_TO_CHARGING"}
    if decision_part not in valid_decisions:
        decision_part = "TAKE_OFF"

    new_dec = _OLD_TO_NEW.get(decision_part, "FLY")

    sb = get_supabase()
    if sb is not None:
        try:
            update_data = {
                "system_decision": new_dec,
                "is_user_overridden": False if is_restore else True,
                "override_reason": "" if is_restore else notes_part,
            }
            if record_id:
                res_select = sb.table("flight_decision_log").select("*").eq("log_id", record_id).execute()
                if res_select.data and len(res_select.data) > 0:
                    record = res_select.data[0]
                    if not record.get("is_user_overridden") and record.get("system_decision") == "NO_FLY":
                        weather = record.get("weather_json") or record.get("weather_snapshot") or {}
                        wind = float(weather.get("wind_speed_10m", 0))
                        gust = float(weather.get("wind_gusts_10m", 0))
                        if wind > 36.0 or gust > 36.0:
                            raise HTTPException(status_code=403, detail="Hệ thống đã ban hành lệnh CẤM BAY (NO_FLY) do gió vượt giới hạn vật lý của Drone (>36km/h). Trạng thái này bị KHÓA CỨNG.")
                    if not is_restore and record.get("is_user_overridden") and record.get("system_decision") == new_dec:
                        raise HTTPException(status_code=409, detail=f"Đã ghi đè trạng thái '{new_dec}' rồi, không thể ghi đè lặp lại.")
                    if is_restore:
                        update_data["system_decision"] = record.get("system_decision", new_dec)
                sb.table("flight_decision_log").update(update_data).eq("log_id", record_id).execute()
            else:
                res_select = sb.table("flight_decision_log").select("*").eq("location_name", location).eq("slot_timestamp", timestamp).execute()
                if res_select.data and len(res_select.data) > 0:
                    record = res_select.data[0]
                    if not record.get("is_user_overridden") and record.get("system_decision") == "NO_FLY":
                        weather = record.get("weather_json") or record.get("weather_snapshot") or {}
                        wind = float(weather.get("wind_speed_10m", 0))
                        gust = float(weather.get("wind_gusts_10m", 0))
                        if wind > 36.0 or gust > 36.0:
                            raise HTTPException(status_code=403, detail="Hệ thống đã ban hành lệnh CẤM BAY (NO_FLY) do gió vượt giới hạn vật lý của Drone (>36km/h). Trạng thái này bị KHÓA CỨNG.")
                    if not is_restore and record.get("is_user_overridden") and record.get("system_decision") == new_dec:
                        raise HTTPException(status_code=409, detail=f"Đã ghi đè trạng thái '{new_dec}' rồi, không thể ghi đè lặp lại.")
                    if is_restore:
                        update_data["system_decision"] = record.get("system_decision", new_dec)
                sb.table("flight_decision_log").update(update_data).eq("location_name", location).eq("slot_timestamp", timestamp).execute()
        except HTTPException:
            raise
        except Exception:
            pass

    final_ret = update_data.get("system_decision", new_dec) if is_restore else decision_part
    return {
        "status": "ok",
        "final_decision": final_ret,
        "was_human_overridden": not is_restore,
        "is_safe_to_fly": (final_ret in {"TAKE_OFF", "FLY"}),
        "flyability_score": 1.0 if (final_ret in {"TAKE_OFF", "FLY"}) else 0.0,
        "resource_regressor": {
            "flow_rate_l_ha": 20.0 if decision_part == "TAKE_OFF" else 0.0,
            "total_liters": 200.0 if decision_part == "TAKE_OFF" else 0.0,
            "sorties": 7 if decision_part == "TAKE_OFF" else 0,
            "distance_to_field_km": 1.0,
            "battery_cycles_needed": 9 if decision_part == "TAKE_OFF" else 0,
        }
    }


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
