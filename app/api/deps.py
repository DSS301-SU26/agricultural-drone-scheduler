"""
Phu thuoc dung chung cho API: Predictor (load 1 lan) + Supabase client (tuy chon).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from ..ml.scores import Predictor

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Load model 1 lan cho toan bo vong doi app."""
    return Predictor()


@lru_cache(maxsize=1)
def get_supabase():
    """Tra ve supabase client neu co .env, nguoc lai None (API van chay khong can DB)."""
    try:
        from dotenv import load_dotenv
        from supabase import create_client
        load_dotenv(ROOT / ".env")
        url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None
