import pandas as pd
import os
import glob
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLEAN_DIR = os.path.join(BASE_DIR, "src", "data", "clean")
FINAL_CSV = os.path.join(BASE_DIR, "src", "data", "final_training_data.csv")

# Synthetic data pools to simulate joined DB tables
DRONES = [
    {"drone_model": "DJI Agras T30", "max_wind_resistance_kph": 28.8, "max_gust_resistance_kph": 28.8, "tank_capacity_liters": 30},
    {"drone_model": "DJI Agras T40", "max_wind_resistance_kph": 21.6, "max_gust_resistance_kph": 21.6, "tank_capacity_liters": 40},
    {"drone_model": "XAG P100 Pro", "max_wind_resistance_kph": 36.0, "max_gust_resistance_kph": 36.0, "tank_capacity_liters": 50},
]

PESTICIDES = [
    {"pesticide_name": "Tricyclazole", "uv_sensitivity": False, "rain_washout_hours": 4},
    {"pesticide_name": "Abamectin", "uv_sensitivity": True, "rain_washout_hours": 2},
    {"pesticide_name": "Hexaconazole", "uv_sensitivity": False, "rain_washout_hours": 2},
]

PLOTS = [
    {"crop_stage": "SEEDLING", "area_hectares": 2.5},
    {"crop_stage": "TILLERING", "area_hectares": 5.0},
    {"crop_stage": "BOOTING", "area_hectares": 1.5},
    {"crop_stage": "GRAIN_FILLING", "area_hectares": 10.0},
]

def main():
    print("1. Đang gộp các file dữ liệu thời tiết API và giả lập JOIN (Drone, Pesticide, Plot)...")
    csv_files = glob.glob(os.path.join(CLEAN_DIR, "*.csv"))
    if not csv_files:
        print("LỖI: Không tìm thấy file thời tiết clean nào.")
        return

    # Đọc và nối tất cả các file thời tiết (nếu có nhiều file)
    df_weather_list = []
    for file in csv_files:
        df_weather_list.append(pd.read_csv(file))
    df_weather = pd.concat(df_weather_list, ignore_index=True)
    before_dedup = len(df_weather)
    df_weather = df_weather.drop_duplicates(
        subset=["location_name", "timestamp"],
        keep="last",
    )
    df_weather = df_weather.sort_values(["location_name", "timestamp"]).reset_index(drop=True)
    print(
        f"   -> Loại {before_dedup - len(df_weather)} dòng forecast trùng "
        "(location_name, timestamp)."
    )

    # Simulate JOIN by assigning random profiles to each row
    np.random.seed(42) # For reproducibility
    n_rows = len(df_weather)
    
    # 1. Join Drone Table
    drone_indices = np.random.choice(len(DRONES), size=n_rows)
    drone_df = pd.DataFrame([DRONES[i] for i in drone_indices])
    
    # 2. Join Pesticide Table
    pesticide_indices = np.random.choice(len(PESTICIDES), size=n_rows)
    pesticide_df = pd.DataFrame([PESTICIDES[i] for i in pesticide_indices])
    
    # 3. Join Plot Table
    plot_indices = np.random.choice(len(PLOTS), size=n_rows)
    plot_df = pd.DataFrame([PLOTS[i] for i in plot_indices])

    # Merge everything
    df_final = pd.concat([df_weather, drone_df, pesticide_df, plot_df], axis=1)

    # One-hot encode crop_stage for ML training
    df_final = pd.get_dummies(df_final, columns=['crop_stage'], prefix='crop_stage', dtype=int)
    
    # We still need the original crop_stage for the decision engine rule (Mating Hours)
    # Re-assign from the plot_df before we save
    df_final["crop_stage"] = plot_df["crop_stage"]

    # Save to Final
    df_final.to_csv(FINAL_CSV, index=False)
    print(f"\n--- HOÀN TẤT ---")
    print(f"Đã tạo thành công bộ dataset FULL (Weather + Drone + Pesticide) tại: {FINAL_CSV}")
    print(f"Tổng số dòng dữ liệu sẵn sàng train AI: {len(df_final)}")

if __name__ == "__main__":
    main()
