"""
Ghi nhat ky quyet dinh - "hop den" BRD §3.8 (P1 #5).

Ghi song song 2 noi:
  1. FILE LOCAL (reports/decision_log.json) - LUON hoat dong, khong can mang.
     -> Dam bao demo va trang Safety Logs chay du khi Supabase khong ket noi.
  2. Supabase flight_decision_log - best-effort (co mang thi ghi, khong thi bo qua).

Chong trung theo (location_name, slot_timestamp).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .deps import get_supabase

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "reports" / "decision_log.json"
_MAX_LOCAL = 2000
_lock = threading.Lock()

_WEATHER_KEYS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "precipitation_probability", "cloud_cover", "visibility",
    "wind_speed_10m", "wind_gusts_10m", "weather_code",
    "et0_fao_evapotranspiration",
]


def build_log_row(location: str | None, slot_timestamp: str | None,
                  result: dict[str, Any], weather: dict[str, Any]) -> dict[str, Any]:
    def _num(v):
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None
    return {
        "location_name": location,
        "slot_timestamp": slot_timestamp,
        "rf_score_safety": _num(result.get("rf_score_safety")),
        "xgb_score_safety": _num(result.get("xgb_score_safety")),
        "flight_safety_score": _num(result.get("flight_safety_score")),
        "crop_impact_score": _num(result.get("crop_impact_score")),
        "spray_quality_score": _num(result.get("spray_quality_score")),
        "system_decision": result.get("decision"),
        "is_user_overridden": bool(result.get("is_user_overridden", False)),
        "override_reason": result.get("override_reason"),
        "xai_explanation": result.get("xai_explanation"),
        "weather_json": {k: weather.get(k) for k in _WEATHER_KEYS if k in weather},
    }


# --- Local file store -------------------------------------------------------
def _load_local() -> dict[str, Any]:
    if not LOG_PATH.exists():
        return {}
    try:
        return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_local(store: dict[str, Any]) -> None:
    if len(store) > _MAX_LOCAL:                       # gioi han kich thuoc
        keys = sorted(store, key=lambda k: str(store[k].get("slot_timestamp") or ""))
        for k in keys[: len(store) - _MAX_LOCAL]:
            store.pop(k, None)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")


def _write_local(rows: list[dict[str, Any]], override: bool = False) -> None:
    with _lock:
        store = _load_local()
        for i, r in enumerate(rows):
            if override:
                key = f"OVR|{r.get('location_name')}|{r.get('slot_timestamp')}|{len(store)+i}"
            else:
                key = f"{r.get('location_name')}|{r.get('slot_timestamp')}"
            store[key] = r
        _save_local(store)


# --- Public API -------------------------------------------------------------
def log_decisions(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    _write_local(rows)                                # LUON ghi local
    sb = get_supabase()
    if sb is None:
        return
    try:                                              # best-effort Supabase
        sb.table("flight_decision_log").upsert(
            rows, on_conflict="location_name,slot_timestamp").execute()
    except Exception:
        pass


def log_override(result: dict[str, Any], weather: dict[str, Any],
                 location: str | None = None) -> None:
    row = build_log_row(location, weather.get("timestamp"), result, weather)
    _write_local([row], override=True)
    sb = get_supabase()
    if sb is None:
        return
    try:
        sb.table("flight_decision_log").upsert(
            row, on_conflict="location_name,slot_timestamp").execute()
    except Exception:
        pass


def recent_logs(limit: int = 100, location: str | None = None) -> list[dict[str, Any]]:
    """Doc log gan nhat tu file local (cho trang Safety Logs)."""
    rows = list(_load_local().values())
    if location:
        rows = [r for r in rows if r.get("location_name") == location]
    rows.sort(key=lambda r: str(r.get("slot_timestamp") or ""), reverse=True)
    return rows[:limit]
