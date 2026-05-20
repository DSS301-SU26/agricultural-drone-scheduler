"""
run_pipeline.py - Chay toan bo pipeline: fetch -> clean -> upload
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline.fetch_weather import fetch_all_locations, save_raw, FARM_LOCATIONS
from data_pipeline.clean_data import run_pipeline as clean_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",        type=int, default=7)
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    print("\n" + "#"*52)
    print("  DSS301 - Agricultural Drone Scheduler Pipeline")
    print("#"*52)

    # Buoc 1: Fetch
    print("\n[1/3] Thu thap du lieu tu WeatherAPI")
    df = fetch_all_locations(FARM_LOCATIONS, forecast_days=args.days)
    if df.empty:
        print("FAIL: Khong lay duoc du lieu. Kiem tra internet.")
        sys.exit(1)
    raw_path = save_raw(df)

    # Buoc 2: Clean
    print("\n[2/3] Lam sach du lieu...")
    clean_df = clean_pipeline(raw_path, save=True)

    clean_files = sorted(Path("data/clean").glob("weather_clean_*.csv"))
    clean_path = str(clean_files[-1])

    # Buoc 3: Upload
    if not args.skip_upload:
        print("\n[3/3] Upload len Supabase...")
        try:
            from data_pipeline.load_to_supabase import load_to_supabase
            load_to_supabase(clean_path)
        except Exception as e:
            print(f"[WARN] Upload that bai: {e}")
    else:
        print("\n[3/3] Bo qua upload (--skip-upload)")

    print("\n" + "#"*52)
    print(f"  HOAN THANH!")
    print(f"  Raw  : {raw_path}")
    print(f"  Clean: {clean_path}")
    print(f"  Rows : {len(clean_df)} ban ghi san sang cho model")
    print("#"*52 + "\n")

if __name__ == "__main__":
    main()
