import pandas as pd
import os
import glob


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLEAN_DIR = os.path.join(BASE_DIR, "src", "data", "clean")
FINAL_CSV = os.path.join(BASE_DIR, "src", "data", "final_training_data.csv")

def main():
    print("1. Đang gộp các file dữ liệu thời tiết API...")
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
    # Sắp xếp lại theo địa điểm và thời gian để dễ nhìn
    df_weather = df_weather.sort_values(["location_name", "timestamp"])
    print(
        f"   -> Loại {before_dedup - len(df_weather)} dòng forecast trùng "
        "(location_name, timestamp)."
    )

    # Lưu ra file Final
    df_weather.to_csv(FINAL_CSV, index=False)
    print(f"\n--- HOÀN TẤT ---")
    print(f"Đã tạo thành công bộ dataset thời tiết tại: {FINAL_CSV}")
    print(f"Tổng số dòng dữ liệu sẵn sàng train AI: {len(df_weather)}")

if __name__ == "__main__":
    main()

