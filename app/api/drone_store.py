"""
Kho drone cho API (P0) - CRUD ho tro giao dien Quan ly Ham doi Drone.

- 3 drone MAC DINH (T30/T50/XAG) tu registry tinh -> tra ve KHONG co drone_id
  (FE hieu la mac dinh -> khong cho sua/xoa).
- Drone TU THEM -> co drone_id, luu Supabase (best-effort) + cache in-memory.

Anh xa ten truong FE <-> DB:
  FE spray_system_type  <-> DB/engine nozzle_technology
  FE ip_rating          <-> DB/engine ingress_protection
"""
from __future__ import annotations

from typing import Any

from ..rules.context import DRONES, DroneProfile
from .deps import get_supabase

_customs: dict[int, dict[str, Any]] = {}
_next_local_id = 1000          # id tam khi khong co Supabase
_loaded = False


def _default_rows() -> list[dict[str, Any]]:
    rows = []
    for d in DRONES.values():
        rows.append({
            "drone_id": None,                     # None => FE coi la mac dinh
            "model_name": d.model_name,
            "max_wind_resistance_kph": d.max_wind_resistance_kph,
            "max_gust_resistance_kph": d.max_gust_resistance_kph,
            "tank_capacity_liters": d.tank_capacity_liters,
            "spray_system_type": d.nozzle_technology,
            "ip_rating": d.ingress_protection,
            "image_url": None,
            "notes": "Drone mac dinh he thong",
        })
    return rows


def _row_from_db(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "drone_id": r.get("drone_id"),
        "model_name": r.get("model_name"),
        "max_wind_resistance_kph": r.get("max_wind_resistance_kph"),
        "max_gust_resistance_kph": r.get("max_gust_resistance_kph"),
        "tank_capacity_liters": r.get("tank_capacity_liters"),
        "spray_system_type": r.get("nozzle_technology"),
        "ip_rating": r.get("ingress_protection"),
        "image_url": r.get("image_url"),
        "notes": r.get("notes"),
    }


def _db_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Chuyen payload FE -> cot bang drone_profiles."""
    return {
        "model_name": payload.get("model_name"),
        "max_wind_resistance_kph": float(payload.get("max_wind_resistance_kph", 28.8)),
        "max_gust_resistance_kph": float(payload.get("max_gust_resistance_kph", 36.0)),
        "tank_capacity_liters": int(float(payload.get("tank_capacity_liters", 30))),
        "nozzle_technology": payload.get("spray_system_type", "CENTRIFUGAL"),
        "ingress_protection": payload.get("ip_rating", "IP67"),
        "notes": payload.get("notes", ""),
    }


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    sb = get_supabase()
    if sb is None:
        return
    try:
        rows = sb.table("drone_profiles").select("*").execute().data or []
        for r in rows:
            if r.get("model_name") in DRONES:      # bo qua 3 default da seed
                continue
            did = r.get("drone_id")
            if did is not None:
                _customs[int(did)] = _row_from_db(r)
    except Exception:
        pass


def list_drones() -> list[dict[str, Any]]:
    _ensure_loaded()
    return _default_rows() + list(_customs.values())


def add_drone(payload: dict[str, Any]) -> dict[str, Any]:
    global _next_local_id
    _ensure_loaded()
    if not payload.get("model_name", "").strip():
        raise ValueError("model_name bat buoc")

    record = {
        "model_name": payload["model_name"],
        "max_wind_resistance_kph": float(payload.get("max_wind_resistance_kph", 28.8)),
        "max_gust_resistance_kph": float(payload.get("max_gust_resistance_kph", 36.0)),
        "tank_capacity_liters": float(payload.get("tank_capacity_liters", 30)),
        "spray_system_type": payload.get("spray_system_type", "CENTRIFUGAL"),
        "ip_rating": payload.get("ip_rating", "IP67"),
        "image_url": payload.get("image_url") or None,
        "notes": payload.get("notes", ""),
    }

    drone_id = None
    sb = get_supabase()
    if sb is not None:
        try:
            res = sb.table("drone_profiles").insert(_db_payload(payload)).execute()
            if res.data:
                drone_id = res.data[0].get("drone_id")
        except Exception:
            pass
    if drone_id is None:
        drone_id = _next_local_id
        _next_local_id += 1

    record["drone_id"] = int(drone_id)
    _customs[int(drone_id)] = record
    return record


def update_drone(drone_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_loaded()
    if drone_id not in _customs:
        raise KeyError(drone_id)
    rec = _customs[drone_id]
    for k in ("model_name", "spray_system_type", "ip_rating", "image_url", "notes"):
        if k in payload:
            rec[k] = payload[k]
    for k in ("max_wind_resistance_kph", "max_gust_resistance_kph", "tank_capacity_liters"):
        if k in payload and payload[k] is not None:
            rec[k] = float(payload[k])

    sb = get_supabase()
    if sb is not None:
        try:
            sb.table("drone_profiles").update(_db_payload({**rec, **payload})).eq("drone_id", drone_id).execute()
        except Exception:
            pass
    return rec


def delete_drone(drone_id: int) -> None:
    _ensure_loaded()
    _customs.pop(drone_id, None)
    sb = get_supabase()
    if sb is not None:
        try:
            sb.table("drone_profiles").delete().eq("drone_id", drone_id).execute()
        except Exception:
            pass


def resolve(model_name: str) -> DroneProfile:
    """Tra ve DroneProfile cho engine (uu tien custom, roi default, fallback T30)."""
    _ensure_loaded()
    for rec in _customs.values():
        if rec["model_name"] == model_name:
            return DroneProfile(
                model_name=rec["model_name"],
                max_wind_resistance_kph=float(rec["max_wind_resistance_kph"]),
                max_gust_resistance_kph=float(rec["max_gust_resistance_kph"]),
                tank_capacity_liters=int(float(rec["tank_capacity_liters"])),
                nozzle_technology=rec["spray_system_type"],
                ingress_protection=rec["ip_rating"],
            )
    return DRONES.get(model_name, DRONES["DJI_T30"])
