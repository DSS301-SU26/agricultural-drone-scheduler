import pandas as pd
import os
import random
import shutil
import glob


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLEAN_DIR = os.path.join(BASE_DIR, "src", "data", "clean")
KAGGLE_SOURCE_DIR = os.path.join(BASE_DIR, "src", "data", "image_kaggle")
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "src", "data", "images")

def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    csv_files = glob.glob(os.path.join(CLEAN_DIR, "*.csv"))

    if not csv_files:
        print(f"LỖI: Không tìm thấy file CSV nào trong {CLEAN_DIR}")
        return

    print(f"1. Tìm thấy {len(csv_files)} file dữ liệu thời tiết. Bắt đầu xử lý gộp...")
    success_count = 0

    for file_path in csv_files:
        print(f"\n- Đang đọc file: {os.path.basename(file_path)}")
        df = pd.read_csv(file_path)

        if 'timestamp' not in df.columns:
            print(f"  -> Bỏ qua file này vì không có cột 'timestamp'.")
            continue

        # In ra số dòng của file CSV để kiểm tra xem file có bị trống không
        print(f"  -> File này có {len(df)} dòng dữ liệu.")

        for index, row in df.iterrows():
            raw_timestamp = str(row['timestamp'])
            safe_timestamp = raw_timestamp.replace(":", "-").replace(" ", "_")

        # LẤY RA GIỜ HIỆN TẠI (từ 0 đến 23)
        # Chuyển chuỗi thời gian thành đối tượng datetime để trích xuất giờ
            hour = pd.to_datetime(raw_timestamp).hour

            cloud_cover = row.get('cloud_cover', 0)
            rain = row.get('precipitation', 0)

            # SỬA LỖI Ở ĐÂY: Dùng chữ thường cho khớp với 100% hình ảnh bạn gửi
            if rain > 0:
                category = "Rain"
            elif cloud_cover > 50:
                category = "Cloudy"
            elif 5 <= hour <= 7:
                # Nếu từ 5h đến 7h sáng thì lấy ảnh bình minh
                category = "Sunrise"
            else:
                # Các giờ còn lại trong ngày (trời quang) thì lấy ảnh nắng
                category = "Shine"

            category_path = os.path.join(KAGGLE_SOURCE_DIR, category)

            # Bắt đầu đi tìm và copy ảnh
            if os.path.exists(category_path):
                available_images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if len(available_images) > 0:
                    random_img = random.choice(available_images)
                    src_img_path = os.path.join(category_path, random_img)
                    dest_img_path = os.path.join(OUTPUT_IMG_DIR, f"{safe_timestamp}.jpg")

                    shutil.copy(src_img_path, dest_img_path)
                    success_count += 1

                    # In ra để bạn thấy nó đang copy file nào
                    print(f"     + Đã copy: {category}/{random_img} -> {safe_timestamp}.jpg")
                else:
                    print(f"     ! Thư mục '{category}' không có tấm ảnh nào bên trong.")
            else:
                print(f"     ! LỖI: Không tìm thấy đường dẫn thư mục {category_path}")

    print(f"\n--- HOÀN TẤT ---")
    print(f"Đã giả lập thành công {success_count} bức ảnh vào thư mục {OUTPUT_IMG_DIR}!")

if __name__ == "__main__":
    main()