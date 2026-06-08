"""
main.py - Bam Run de chay toan bo pipeline
Chay o thu muc goc: agricultural-drone-scheduler/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api import app
from data_pipeline.fetch_weather import fetch_all_locations, save_raw, FARM_LOCATIONS
from data_pipeline.clean_data import run_pipeline as clean_pipeline
from data_pipeline.load_to_supabase import load_to_supabase

if __name__ == "__main__":
    print("[1/3] Lay du lieu WeatherAPI...")
    df = fetch_all_locations(FARM_LOCATIONS, forecast_days=3)
    raw_path = save_raw(df, output_dir="data/raw")

    print("[2/3] Lam sach du lieu...")
    clean_pipeline(raw_path, save=True)

    print("[3/3] Upload len Supabase...")
    clean_files = sorted(Path("data/clean").glob("weather_clean_*.csv"))
    load_to_supabase(str(clean_files[-1]))

    print("HOAN THANH!")
# //test
