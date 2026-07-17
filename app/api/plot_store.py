"""
Kho Vuon/Thua ruong (m_plots) - P1 Quan ly Vuon.

- 6 tinh ĐBSCL = vuon MAC DINH (plot_id=None -> khong xoa duoc).
- Vuon TU THEM: plot_id, GPS, dien tich, giai doan lua. Luu Supabase best-effort + cache.

`id`/`name` = ten vuon (FE dropdown dung loc.id/loc.name + lam tham so `location`).
"""
from __future__ import annotations

from typing import Any

from ..ingestion.locations import DELTA_LOCATIONS
from .deps import get_supabase

_customs: dict[int, dict[str, Any]] = {}
_next_local_id = 2000
_loaded = False


def _nearest_location(lat: float, lon: float) -> str:
    """Tim tinh DBSCL gan nhat theo toa do GPS (Euclidean don gian)."""
    best, best_dist = DELTA_LOCATIONS[0]["name"], float("inf")
    for loc in DELTA_LOCATIONS:
        d = (lat - loc["lat"]) ** 2 + (lon - loc["lon"]) ** 2
        if d < best_dist:
            best, best_dist = loc["name"], d
    return best


def _default_plots() -> list[dict[str, Any]]:
    return [{
        "plot_id": None,
        "id": l["name"], "name": l["name"], "plot_name": l["name"],
        "latitude": l["lat"], "longitude": l["lon"],
        "area_hectares": None, "current_crop_stage": None,
        "current_pesticide": None,
        "location_name": l["name"],
        "is_default": True,
    } for l in DELTA_LOCATIONS]


def _custom_to_out(rec: dict[str, Any]) -> dict[str, Any]:
    lat = float(rec.get("latitude", 10.0))
    lon = float(rec.get("longitude", 105.0))
    return {
        "plot_id": rec["plot_id"],
        "id": rec["plot_name"], "name": rec["plot_name"], "plot_name": rec["plot_name"],
        "latitude": lat, "longitude": lon,
        "area_hectares": rec.get("area_hectares"),
        "current_crop_stage": rec.get("current_crop_stage"),
        "current_pesticide": rec.get("current_pesticide"),
        "location_name": _nearest_location(lat, lon),
        "is_default": False,
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
        rows = sb.table("m_plots").select("*").execute().data or []
        for r in rows:
            pid = r.get("plot_id")
            if pid is not None:
                _customs[int(pid)] = r
    except Exception:
        pass


def list_plots() -> list[dict[str, Any]]:
    _ensure_loaded()
    return [_custom_to_out(r) for r in _customs.values()]


def resolve_gps(name: str) -> tuple[float, float] | None:
    _ensure_loaded()
    for l in DELTA_LOCATIONS:
        if l["name"] == name:
            return l["lat"], l["lon"]
    for r in _customs.values():
        if r["plot_name"] == name:
            return float(r["latitude"]), float(r["longitude"])
    return None


def add_plot(payload: dict[str, Any]) -> dict[str, Any]:
    global _next_local_id
    _ensure_loaded()
    name = (payload.get("plot_name") or payload.get("name") or "").strip()
    if not name:
        raise ValueError("plot_name bat buoc")
    rec = {
        "plot_name": name,
        "latitude": float(payload.get("latitude", 10.0)),
        "longitude": float(payload.get("longitude", 105.0)),
        "area_hectares": float(payload["area_hectares"]) if payload.get("area_hectares") else None,
        "current_crop_stage": payload.get("current_crop_stage"),
    }
    plot_id = None
    sb = get_supabase()
    if sb is not None:
        try:
            res = sb.table("m_plots").insert(rec).execute()
            if res.data:
                plot_id = res.data[0].get("plot_id")
        except Exception:
            pass
    if plot_id is None:
        plot_id = _next_local_id
        _next_local_id += 1
    rec["plot_id"] = int(plot_id)
    _customs[int(plot_id)] = rec
    return _custom_to_out(rec)

def _find_custom_plot_id(plot_id: str | int) -> int | None:
    if isinstance(plot_id, int) or (isinstance(plot_id, str) and plot_id.isdigit()):
        pid = int(plot_id)
        if pid in _customs:
            return pid
    for pid, p in _customs.items():
        if p["plot_name"] == str(plot_id) or str(pid) == str(plot_id):
            return pid
    return None


def update_plot(plot_id: str | int, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_loaded()
    pid = _find_custom_plot_id(plot_id)
    if pid is None:
        lat, lon = 10.0, 105.0
        for l in DELTA_LOCATIONS:
            if l["name"] == str(plot_id) or str(plot_id).startswith("plot_"):
                lat, lon = l["lat"], l["lon"]
                break
        
        new_payload = {
            "plot_name": str(plot_id) if not str(plot_id).startswith("plot_") else "Ruong " + str(plot_id),
            "latitude": lat,
            "longitude": lon,
            **payload
        }
        return add_plot(new_payload)

    rec = _customs[pid]
    for k in ("plot_name", "current_crop_stage", "current_pesticide"):
        if k in payload:
            rec[k] = payload[k]
    for k in ("latitude", "longitude", "area_hectares"):
        if k in payload and payload[k] is not None:
            rec[k] = float(payload[k])
    sb = get_supabase()
    if sb is not None:
        try:
            sb.table("m_plots").update(rec).eq("plot_id", pid).execute()
        except Exception:
            pass
    return _custom_to_out(rec)


def delete_plot(plot_id: str | int) -> None:
    _ensure_loaded()
    pid = _find_custom_plot_id(plot_id)
    if pid is None:
        return
    _customs.pop(pid, None)
    sb = get_supabase()
    if sb is not None:
        try:
            sb.table("m_plots").delete().eq("plot_id", pid).execute()
        except Exception:
            pass
