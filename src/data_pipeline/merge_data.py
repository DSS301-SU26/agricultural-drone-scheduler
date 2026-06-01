import pandas as pd
import os
import glob


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLEAN_DIR = os.path.join(BASE_DIR, "src", "data", "clean")
FEATURES_CSV = os.path.join(BASE_DIR, "src", "data", "image_features.csv")
# Đây là file "trùm cuối" chúng ta sẽ dùng để train model
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
    print(
        f"   -> Loại {before_dedup - len(df_weather)} dòng forecast trùng "
        "(location_name, timestamp)."
    )

    # Tạo cột 'match_id' để ghép (vì lúc lưu ảnh ta đã đổi dấu : thành dấu -)
    df_weather['match_id'] = df_weather['timestamp'].astype(str).str.replace(":", "-").str.replace(" ", "_")

    print("2. Đang đọc dữ liệu đặc trưng hình ảnh...")
    if not os.path.exists(FEATURES_CSV):
        print(f"LỖI: Không tìm thấy file {FEATURES_CSV}")
        return
    df_features = pd.read_csv(FEATURES_CSV)

    print(f"3. Bắt đầu ghép {len(df_weather)} dòng thời tiết với {len(df_features)} dòng ảnh...")

    # Nối 2 bảng dựa trên mốc thời gian
    df_final = pd.merge(df_weather, df_features, left_on='match_id', right_on='timestamp', how='inner')

    # Dọn dẹp các cột thừa sau khi nối
    if 'timestamp_y' in df_final.columns:
        df_final = df_final.drop(columns=['timestamp_y', 'match_id'])
        df_final = df_final.rename(columns={'timestamp_x': 'timestamp'})
    else:
        df_final = df_final.drop(columns=['match_id'], errors='ignore')

    # 4. Lưu ra file Final
    df_final.to_csv(FINAL_CSV, index=False)
    print(f"\n--- XUẤT SẮC! HOÀN TẤT PHASE 2 ---")
    print(f"Đã tạo thành công bộ dataset TRÙM CUỐI tại: {FINAL_CSV}")
    print(f"Tổng số dòng dữ liệu sẵn sàng train AI: {len(df_final)}")

if __name__ == "__main__":
    main()
