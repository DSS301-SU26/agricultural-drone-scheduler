"""
run_pipeline.py - Chay toan bo pipeline: fetch -> clean -> upload
"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline.fetch_weather import fetch_all_locations, save_raw, FARM_LOCATIONS
from data_pipeline.clean_data import run_pipeline as clean_pipeline

ROOT = Path(__file__).resolve().parents[1]


def run_weather_pipeline(days=3, skip_upload=False):
    previous_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        steps = []

        print("\n" + "#"*52)
        print("  DSS301 - Agricultural Drone Scheduler Pipeline")
        print("#"*52)

        print("\n[1/3] Thu thap du lieu tu WeatherAPI")
        df = fetch_all_locations(FARM_LOCATIONS, forecast_days=days)
        if df.empty:
            raise RuntimeError("Khong lay duoc du lieu. Kiem tra internet hoac WEATHERAPI_KEY.")
        raw_path = save_raw(df)
        steps.append({"name": "fetch_weather", "status": "done", "rows": len(df), "output": raw_path})

        print("\n[2/3] Lam sach du lieu...")
        clean_df = clean_pipeline(raw_path, save=True)

        clean_files = sorted(Path("data/clean").glob("weather_clean_*.csv"))
        if not clean_files:
            raise RuntimeError("Pipeline khong tao duoc file clean.")
        clean_path = str(clean_files[-1])
        steps.append({"name": "clean_data", "status": "done", "rows": len(clean_df), "output": clean_path})

        uploaded = False
        upload_error = None
        if not skip_upload:
            print("\n[3/3] Upload len Supabase...")
            try:
                from data_pipeline.load_to_supabase import load_to_supabase
                load_to_supabase(clean_path)
                uploaded = True
                steps.append({"name": "upload_supabase", "status": "done", "output": "raw_weather_data"})
            except Exception as e:
                upload_error = str(e)
                print(f"[WARN] Upload that bai: {e}")
                steps.append({"name": "upload_supabase", "status": "warning", "error": upload_error})
        else:
            print("\n[3/3] Bo qua upload (--skip-upload)")
            steps.append({"name": "upload_supabase", "status": "skipped"})

        print("\n" + "#"*52)
        print(f"  HOAN THANH!")
        print(f"  Raw  : {raw_path}")
        print(f"  Clean: {clean_path}")
        print(f"  Rows : {len(clean_df)} ban ghi san sang cho model")
        print("#"*52 + "\n")

        return {
            "status": "ok",
            "steps": steps,
            "raw_path": raw_path,
            "clean_path": clean_path,
            "rows": len(clean_df),
            "uploaded": uploaded,
            "upload_error": upload_error,
        }
    finally:
        os.chdir(previous_cwd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",        type=int, default=3)
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    run_weather_pipeline(days=args.days, skip_upload=args.skip_upload)

if __name__ == "__main__":
    main()
