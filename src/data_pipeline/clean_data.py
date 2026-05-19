"""
clean_data.py - Lam sach du lieu va tao features cho drone scheduling
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

SAFETY_THRESHOLDS = {
    "wind_speed_10m": 20.0,
    "wind_gusts_10m": 28.0,
    "precipitation": 0.0,
    "precipitation_probability": 30.0,
    "cloud_cover": 80.0,
    "visibility": 1000.0,
}
UNSAFE_WEATHER_CODES = {45, 48, 55, 63, 65, 71, 80, 81, 82, 95, 99}
FLY_HOUR_START = 6
FLY_HOUR_END   = 18

def run_pipeline(raw_filepath, save=True):
    print(f"\n{'='*50}")
    print(f"  Cleaning Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}\n")

    df = pd.read_csv(raw_filepath, parse_dates=["timestamp"], encoding="utf-8-sig")
    print(f"  Doc: {len(df)} ban ghi")

    # Duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["location_name", "timestamp"])
    if before - len(df): print(f"  Loai {before - len(df)} duplicates")

    # Missing values
    df = df.sort_values(["location_name", "timestamp"])
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df.groupby("location_name")[col].transform(lambda x: x.ffill().bfill())
        df[col] = df[col].fillna(df[col].median())
    print(f"  Missing values: da xu ly")

    # Gio bay hop le
    df["hour"] = df["timestamp"].dt.hour
    df = df[(df["hour"] >= FLY_HOUR_START) & (df["hour"] < FLY_HOUR_END)].copy()
    print(f"  Loc gio bay (6h-18h): con {len(df)} ban ghi")

    # Feature engineering
    df["date"]      = df["timestamp"].dt.date
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"]     = df["timestamp"].dt.month

    conds = {
        "ok_wind":      df["wind_speed_10m"]            <= SAFETY_THRESHOLDS["wind_speed_10m"],
        "ok_gust":      df["wind_gusts_10m"]            <= SAFETY_THRESHOLDS["wind_gusts_10m"],
        "ok_rain":      df["precipitation"]             == SAFETY_THRESHOLDS["precipitation"],
        "ok_rain_prob": df["precipitation_probability"] <= SAFETY_THRESHOLDS["precipitation_probability"],
        "ok_cloud":     df["cloud_cover"]               <= SAFETY_THRESHOLDS["cloud_cover"],
        "ok_vis":       df["visibility"]                >= SAFETY_THRESHOLDS["visibility"],
        "ok_wmo":       ~df["weather_code"].isin(UNSAFE_WEATHER_CODES),
    }
    weights = {"ok_wind":0.30,"ok_gust":0.20,"ok_rain":0.20,
               "ok_rain_prob":0.10,"ok_cloud":0.10,"ok_vis":0.05,"ok_wmo":0.05}

    for k, v in conds.items():
        df[k] = v

    df["flyability_score"] = sum(df[c].astype(float)*w for c,w in weights.items()).round(4)
    df["fly_label"]  = df[["ok_wind","ok_gust","ok_rain","ok_wmo"]].all(axis=1).map({True:"FLY",False:"NO_FLY"})
    df["risk_level"] = pd.cut(df["flyability_score"],bins=[0,0.4,0.7,1.0],
                              labels=["HIGH","MEDIUM","LOW"],include_lowest=True).astype(str)

    df.drop(columns=list(conds.keys()), inplace=True)

    fly_pct = (df["fly_label"]=="FLY").mean()*100
    print(f"  FLY: {fly_pct:.1f}% | NO_FLY: {100-fly_pct:.1f}%")

    if save:
        Path("data/clean").mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out = f"data/clean/weather_clean_{ts}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  Saved: {out}")

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.input, save=not args.no_save)
