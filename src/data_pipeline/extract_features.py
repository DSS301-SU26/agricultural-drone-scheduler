import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_FOLDER = os.path.join(BASE_DIR, "src", "data", "images")
OUTPUT_CSV = os.path.join(BASE_DIR, "src", "data", "image_features.csv")

def main():
    print("1. Đang tải mô hình AI xử lý ảnh (MobileNetV2)...")
    # Tải mô hình đã được train sẵn bởi Google
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

    extracted_data = []

    if not os.path.exists(IMAGE_FOLDER):
        print(f"LỖI: Không tìm thấy thư mục ảnh tại {IMAGE_FOLDER}")
        return

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"2. Bắt đầu dùng AI để 'đọc' {len(image_files)} bức ảnh. Quá trình này có thể mất vài phút...")

    # Duyệt qua từng tấm ảnh
    for count, img_name in enumerate(image_files, 1):
        img_path = os.path.join(IMAGE_FOLDER, img_name)

        try:
            # Đọc và resize ảnh về chuẩn 224x224 cho AI
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Tiền xử lý
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Trích xuất đặc trưng (Biến ảnh thành 1280 con số)
            features = model.predict(img_array, verbose=0).flatten()

            # Lấy tên file làm timestamp (VD: 2026-05-25_15-00)
            timestamp = img_name.split('.jpg')[0]

            row_data = {'timestamp': timestamp}
            for i, val in enumerate(features):
                row_data[f'img_feature_{i}'] = val

            extracted_data.append(row_data)

            # In tiến độ cho mỗi 50 ảnh để bạn đỡ sốt ruột
            if count % 50 == 0:
                print(f"   ... Đã xử lý xong {count}/{len(image_files)} ảnh")

        except Exception as e:
            print(f" - Lỗi khi xử lý {img_name}: {e}")


    # LƯU KẾT QUẢ
    print("\n3. Đang lưu kết quả ra file CSV...")
    df_features = pd.DataFrame(extracted_data)
    df_features.to_csv(OUTPUT_CSV, index=False)

    print(f"--- HOÀN TẤT ---")
    print(f"Đã lưu thành công đặc trưng của {len(df_features)} ảnh vào file: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()