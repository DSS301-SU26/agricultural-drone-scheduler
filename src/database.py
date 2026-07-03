"""
database.py - CRUD operations for Supabase tables.

Tables managed:
    - drone_flight_logs      : Drone activity history
    - analyzed_weather_data  : Analyzed weather records
    - weather_overrides      : User-uploaded image overrides
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

_client = None


def get_client():
    """Return a cached Supabase client (created once per process)."""
    global _client
    if _client is not None:
        return _client
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError("Thieu SUPABASE_URL hoac SUPABASE_KEY trong .env")
    from supabase import create_client
    _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ── drone_flight_logs ──────────────────────────────────────────

def save_flight_log(data: dict[str, Any]) -> dict[str, Any]:
    """Upsert a single drone flight log row."""
    client = get_client()
    result = (
        client.table("drone_flight_logs")
        .upsert(data, on_conflict="location_name,flight_timestamp")
        .execute()
    )
    return result.data[0] if result.data else {}


def save_flight_logs_batch(rows: list[dict[str, Any]], batch_size: int = 200) -> int:
    """Upsert drone flight logs in batches. Returns count of saved rows."""
    client = get_client()
    saved = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        try:
            client.table("drone_flight_logs").upsert(
                batch, on_conflict="location_name,flight_timestamp"
            ).execute()
            saved += len(batch)
        except Exception as exc:
            print(f"  [ERROR] drone_flight_logs batch {i // batch_size + 1}: {exc}")
    return saved


def get_flight_history(
    location: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Query drone flight history with optional filters."""
    client = get_client()
    query = client.table("drone_flight_logs").select("*")
    if location:
        query = query.eq("location_name", location)
    if start_date:
        query = query.gte("flight_timestamp", start_date)
    if end_date:
        query = query.lte("flight_timestamp", end_date)
    query = query.order("flight_timestamp", desc=True).limit(limit)
    return query.execute().data or []


def get_flight_stats(location: str | None = None) -> dict[str, Any]:
    """Aggregate statistics for drone flight logs."""
    rows = get_flight_history(location=location, limit=10_000)
    if not rows:
        return {"total": 0, "by_action": {}, "by_risk": {}, "by_source": {}}

    by_action: dict[str, int] = {}
    by_risk: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for row in rows:
        action = row.get("decision_action", "UNKNOWN")
        by_action[action] = by_action.get(action, 0) + 1
        risk = row.get("risk_level", "UNKNOWN")
        by_risk[risk] = by_risk.get(risk, 0) + 1
        source = row.get("weather_source", "api")
        by_source[source] = by_source.get(source, 0) + 1

    return {
        "total": len(rows),
        "by_action": by_action,
        "by_risk": by_risk,
        "by_source": by_source,
    }


# ── analyzed_weather_data ─────────────────────────────────────

def save_analyzed_weather(data: dict[str, Any]) -> dict[str, Any]:
    """Upsert a single analyzed weather row."""
    client = get_client()
    result = (
        client.table("analyzed_weather_data")
        .upsert(data, on_conflict="location_name,timestamp")
        .execute()
    )
    return result.data[0] if result.data else {}


def save_analyzed_weather_batch(rows: list[dict[str, Any]], batch_size: int = 200) -> int:
    """Upsert analyzed weather records in batches. Returns count of saved rows."""
    client = get_client()
    saved = 0
    import math

    def clean_nan(val):
        if isinstance(val, float) and math.isnan(val):
            return None
        return val

    # Clean rows to prevent JSON NaN errors
    cleaned_rows = [
        {k: clean_nan(v) for k, v in row.items()}
        for row in rows
    ]

    for i in range(0, len(cleaned_rows), batch_size):
        batch = cleaned_rows[i : i + batch_size]
        try:
            client.table("analyzed_weather_data").upsert(
                batch, on_conflict="location_name,timestamp"
            ).execute()
            saved += len(batch)
        except Exception as exc:
            print(f"  [ERROR] analyzed_weather_data batch {i // batch_size + 1}: {exc}")
    return saved


def get_weather_history(
    location: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Query analyzed weather history with optional filters."""
    client = get_client()
    query = client.table("analyzed_weather_data").select("*")
    if location:
        query = query.eq("location_name", location)
    if start_date:
        query = query.gte("timestamp", start_date)
    if end_date:
        query = query.lte("timestamp", end_date)
    query = query.order("timestamp", desc=True).limit(limit)
    return query.execute().data or []


# ── weather_overrides ─────────────────────────────────────────

def create_weather_override(data: dict[str, Any]) -> dict[str, Any]:
    """Insert a new weather override record."""
    client = get_client()
    result = client.table("weather_overrides").insert(data).execute()
    return result.data[0] if result.data else {}


def get_overrides(
    location: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch weather overrides, newest first."""
    client = get_client()
    query = client.table("weather_overrides").select("*")
    if location:
        query = query.eq("location_name", location)
    query = query.order("created_at", desc=True).limit(limit)
    return query.execute().data or []


def get_override_accuracy_stats() -> dict[str, Any]:
    """Calculate how often the AI suggestion matched the user's final choice."""
    client = get_client()
    rows = (
        client.table("weather_overrides")
        .select("ai_suggested_condition,user_final_condition,user_accepted_ai")
        .execute()
        .data
        or []
    )
    if not rows:
        return {"total": 0, "ai_accepted": 0, "ai_rejected": 0, "accuracy_pct": 0.0}

    total = len(rows)
    accepted = sum(1 for r in rows if r.get("user_accepted_ai"))
    return {
        "total": total,
        "ai_accepted": accepted,
        "ai_rejected": total - accepted,
        "accuracy_pct": round(accepted / total * 100, 1) if total else 0.0,
    }


# ── Storage ───────────────────────────────────────────────────

def upload_image_to_storage(file_path: Path | str, filename: str) -> str:
    """
    Upload an image file to Supabase 'weather_overrides' bucket and return its public URL.
    """
    client = get_client()
    bucket_name = "weather_overrides"
    
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Determine content type based on extension
    ext = Path(filename).suffix.lower()
    content_type = "image/png" if ext == ".png" else "image/jpeg"

    # Upload file
    client.storage.from_(bucket_name).upload(
        file=file_bytes,
        path=filename,
        file_options={"content-type": content_type}
    )

    return client.storage.from_(bucket_name).get_public_url(filename)


def save_decision_log(data: dict[str, Any]) -> dict[str, Any]:
    """Insert a decision log row into flight_decision_log (singular or plural fallback)."""
    client = get_client()
    try:
        result = (
            client.table("flight_decision_log")
            .insert(data)
            .execute()
        )
        return result.data[0] if result.data else {}
    except Exception as e:
        print(f"Error inserting into flight_decision_log, trying plural fallback: {e}")
        try:
            result = (
                client.table("flight_decisions_log")
                .insert(data)
                .execute()
            )
            return result.data[0] if result.data else {}
        except Exception as ex:
            print(f"Plural fallback also failed: {ex}")
            return {}


# ── New Schema Helper Queries ───────────────────────────────────

def get_drone_profile(model_name: str) -> dict[str, Any] | None:
    """Fetch drone profile by model name."""
    try:
        client = get_client()
        result = client.table("drone_profiles").select("*").eq("model_name", model_name).execute()
        if result.data:
            return result.data[0]
    except Exception as e:
        print(f"Error fetching drone profile '{model_name}': {e}")
    return None

def get_all_drones() -> list[dict[str, Any]]:
    """Fetch all drone profiles."""
    try:
        client = get_client()
        result = client.table("drone_profiles").select("*").execute()
        return result.data
    except Exception as e:
        print(f"Error fetching all drones: {e}")
    return []


def get_crop_profile(stage_code: str) -> dict[str, Any] | None:
    """Fetch crop profile by crop stage code (e.g. SEEDLING, TILLERING, BOOTING, GRAIN_FILLING)."""
    try:
        client = get_client()
        result = client.table("crop_profile").select("*").eq("stage_code", stage_code).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Error fetching crop profile '{stage_code}': {e}")
        return None


def get_pesticide_spec(trade_name: str) -> dict[str, Any] | None:
    """Fetch pesticide spec by trade name or active ingredient."""
    try:
        client = get_client()
        # Try trade name first
        result = client.table("pesticide_specs").select("*").eq("trade_name", trade_name).execute()
        if result.data:
            return result.data[0]
        # Try active ingredient if not found
        result = client.table("pesticide_specs").select("*").eq("active_ingredient", trade_name).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Error fetching pesticide spec '{trade_name}': {e}")
        return None


def get_plot_by_name(plot_name: str) -> dict[str, Any] | None:
    """Fetch plot by plot name (location)."""
    try:
        client = get_client()
        result = client.table("m_plots").select("*").eq("plot_name", plot_name).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Error fetching plot '{plot_name}': {e}")
        return None


def get_latest_soil_reading(plot_id: int) -> dict[str, Any] | None:
    """Fetch the latest soil reading for a plot."""
    try:
        client = get_client()
        result = (
            client.table("soil_readings")
            .select("*")
            .eq("plot_id", plot_id)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Error fetching latest soil reading for plot {plot_id}: {e}")
        return None


