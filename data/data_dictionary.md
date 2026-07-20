# Data Dictionary
## DSS301 — Agricultural Drone Flight Scheduler
**Đề tài:** DSS cho lập lịch bay UAV nông nghiệp theo điều kiện thời tiết (Chuyên canh cây lúa)  
**Nguồn dữ liệu:** Dữ liệu thời tiết lịch sử (WeatherAPI), Cấu hình Drone, Loại thuốc BVTV, Chu kỳ sinh trưởng cây trồng  
**Cập nhật lần cuối:** 2026-07-17  
**Người phụ trách:** Nhóm DSS301

---

## 1. Tổng quan Dataset

| Hạng mục | Thông tin |
|---|---|
| Nguồn | WeatherAPI Historical + Cấu hình nghiệp vụ canh tác |
| Phạm vi địa lý | 7 địa điểm: 5 tỉnh/thành ĐBSCL và 2 địa điểm đối chiếu (Hà Nội, TP.HCM) |
| Phạm vi thời gian | 6 năm (2020 – 2025) |
| Múi giờ | UTC+7 |
| Khung giờ lọc | 06:00 – 17:59 (giờ bay hợp lệ ban ngày) |
| Số bản ghi | ~185,304 bản ghi (sau khi gộp drone và cây trồng) |
| Số lượng cột (features) | 36 cột |
| Định dạng lưu | CSV (UTF-8 BOM) |

---

## 2. Mô tả các cột — Final Training Dataset (`data/final_training_data.csv`)

Tập dữ liệu dùng để huấn luyện mô hình (Final Dataset) là kết quả của quá trình gộp (merge) dữ liệu thời tiết lịch sử đã làm sạch cùng với dữ liệu cấu hình loại Drone, Thuốc bảo vệ thực vật, và Giai đoạn phát triển của cây trồng.

### 2.1 Thông tin thời gian và không gian (Spatio-Temporal Features)

| Tên cột | Kiểu dữ liệu | Đơn vị | Mô tả |
|---|---|---|---|
| `location_name` | string | — | Tên địa điểm nông nghiệp (VD: Dong Thap, An Giang) |
| `latitude` | float | độ | Vĩ độ địa điểm |
| `longitude` | float | độ | Kinh độ địa điểm |
| `timestamp` | datetime | — | Thời điểm thời tiết ghi nhận (UTC+7) |
| `source` | string | — | Nguồn cung cấp dữ liệu thời tiết (Open-Meteo/WeatherAPI) |
| `date` | date | — | Ngày (tách từ timestamp) |
| `hour` | int | giờ | Giờ trong ngày (6–17) |
| `dayofweek` | int | ngày | Thứ trong tuần (0=Thứ 2, 6=CN) |
| `month` | int | tháng | Tháng trong năm (1-12) |

### 2.2 Dữ liệu Thời tiết (Weather Features)

| Tên cột | Kiểu dữ liệu | Đơn vị | Mô tả |
|---|---|---|---|
| `temperature_2m` | float | °C | Nhiệt độ không khí tại độ cao 2m |
| `relative_humidity_2m` | float | % | Độ ẩm tương đối tại độ cao 2m |
| `precipitation_probability` | float | % | Xác suất có mưa trong giờ đó |
| `precipitation` | float | mm | Lượng mưa thực tế trong giờ |
| `cloud_cover` | int | % | Tỷ lệ bầu trời bị mây che phủ |
| `visibility` | float | m | Tầm nhìn xa |
| `wind_speed_10m` | float | km/h | Tốc độ gió trung bình tại độ cao 10m |
| `wind_gusts_10m` | float | km/h | Tốc độ gió giật tại độ cao 10m |
| `weather_code` | int | Code | Mã điều kiện thời tiết |
| `weather_description` | string | — | Mô tả điều kiện thời tiết tương ứng |

### 2.3 Cấu hình Drone và Canh tác (Domain/Operational Features)

| Tên cột | Kiểu dữ liệu | Đơn vị | Mô tả |
|---|---|---|---|
| `drone_model` | string | — | Loại Drone sử dụng (VD: DJI_Agras_T40, XAG_P100) |
| `max_wind_resistance_kph` | float | km/h | Sức cản gió tối đa của drone |
| `max_gust_resistance_kph` | float | km/h | Sức cản gió giật tối đa của drone |
| `tank_capacity_liters` | float | lít | Dung tích bình chứa thuốc của drone |
| `pesticide_name` | string | — | Tên loại thuốc bảo vệ thực vật sử dụng |
| `uv_sensitivity` | string | — | Độ nhạy của thuốc với tia UV (HIGH, MEDIUM, LOW) |
| `rain_washout_hours` | float | giờ | Số giờ cần thiết sau phun để thuốc không bị mưa rửa trôi |
| `area_hectares` | float | ha | Diện tích khu vực canh tác cần phun thuốc |
| `crop_stage` | string | — | Giai đoạn phát triển của Lúa (SEEDLING, TILLERING, BOOTING, GRAIN_FILLING) |
| `crop_stage_SEEDLING` | int/bool | — | Biến One-hot encode (Giai đoạn mạ non) |
| `crop_stage_TILLERING` | int/bool | — | Biến One-hot encode (Giai đoạn đẻ nhánh) |
| `crop_stage_BOOTING` | int/bool | — | Biến One-hot encode (Giai đoạn làm đòng) |
| `crop_stage_GRAIN_FILLING`| int/bool | — | Biến One-hot encode (Giai đoạn lúa chín/vào hạt) |

### 2.4 Chỉ số đánh giá và Nhãn quyết định (Targets & Scores)

| Tên cột | Kiểu dữ liệu | Mô tả |
|---|---|---|
| `flyability_score` | float | Điểm khả năng bay (0.0–1.0), tính toán theo trọng số thời tiết |
| `risk_level` | string | Mức độ rủi ro thời tiết (LOW, MEDIUM, HIGH) |
| `fly_label` | string | Nhãn bay thô theo thời tiết (FLY / NO_FLY) |
| `decision_action` | string | **NHÃN DỰ ĐOÁN CUỐI CÙNG** (Action). VD: `PROCEED`, `DELAY_WIND`, `DELAY_RAIN`, `REDUCE_PAYLOAD`, `CHANGE_PESTICIDE`... |

---

## 3. Cấu trúc và Quy tắc Nhãn Quyết Định (`decision_action`)

Nhãn `decision_action` được gán dựa trên cây quyết định chuyên gia (Rules-based) từ việc đánh giá các vi phạm giới hạn bay. Các hành động cụ thể bao gồm:

- `PROCEED`: Điều kiện an toàn tuyệt đối, tiến hành bay bình thường.
- `DELAY_WIND`: Tạm hoãn do tốc độ gió trung bình vượt quá giới hạn của dòng Drone.
- `DELAY_GUST`: Tạm hoãn do gió giật vượt quá sức chịu đựng của Drone.
- `DELAY_RAIN`: Tạm hoãn do phát hiện có mưa.
- `REDUCE_PAYLOAD`: Khuyến nghị giảm tải trọng thuốc để đối phó điều kiện gió/nhiệt độ/khí áp khó khăn.
- `CHANGE_PESTICIDE`: (Ví dụ) Khuyến nghị đổi thời gian hoặc loại thuốc do UV cao phân hủy thuốc nhạy cảm sáng.

*(Note: Quy tắc này thay thế cho nhãn `FLY`/`NO_FLY` đơn giản ban đầu để hỗ trợ quyết định chi tiết hơn).*

---

## 4. Ngưỡng an toàn bay drone nông nghiệp hiện tại

Các ngưỡng không còn cố định cho một hệ thống mà **phụ thuộc động** vào dòng Drone đang vận hành.

| Chỉ số thời tiết | Ngưỡng an toàn (So sánh) | Căn cứ |
|---|---|---|
| Tốc độ gió | `<= drone.max_wind_resistance_kph` | Giới hạn kỹ thuật của từng loại Drone (Ví dụ: T40 chịu được gió mạnh hơn T20P) |
| Gió giật | `<= drone.max_gust_resistance_kph` | Nguy cơ lật hoặc mất quỹ đạo đột ngột |
| Lượng mưa | `= 0 mm` | Mưa làm trôi thuốc, hỏng cảm biến (Ngoại trừ dòng chống nước chuẩn cao) |
| Xác suất mưa | `<= 30%` | Ngưỡng an toàn lập kế hoạch trước |

---

## 5. Địa điểm thu thập dữ liệu (Trạm quan trắc ảo)

| Tên địa điểm | Tỉnh | Vĩ độ | Kinh độ | Đặc điểm canh tác |
|---|---|---|---|---|
| Dong Thap | Đồng Tháp | 10.4939 | 105.6882 | Chuyên canh Lúa nước |
| Long An | Long An | 10.5360 | 106.4052 | Chuyên canh Lúa nước |
| Tien Giang | Tiền Giang | 10.3598 | 106.3567 | Chuyên canh Lúa nước |
| An Giang | An Giang | 10.5216 | 105.1259 | Chuyên canh Lúa nước |
| Can Tho | Cần Thơ | 10.0341 | 105.7878 | Chuyên canh Lúa nước |
| Ho Chi Minh | TP. Hồ Chí Minh | 10.7769 | 106.7009 | Địa điểm đối chiếu |
| Ha Noi | Hà Nội | 21.0285 | 105.8542 | Địa điểm đối chiếu |

---

## 6. Quy trình làm sạch dữ liệu và Gộp đặc trưng (Data Pipeline)

| Bước | Script tương ứng | Mô tả |
|---|---|---|
| 1. Kéo dữ liệu thời tiết | `fetch_historical.py` | Sử dụng Open-Meteo Historical API lấy dữ liệu 6 năm cho 7 tọa độ |
| 2. Làm sạch thời tiết | `clean_data.py` | Lọc khung giờ 6:00 - 17:00, loại bỏ duplicate, xử lý missing values, tính toán `fly_label` và `flyability_score` |
| 3. Tạo tính năng nghiệp vụ | `merge_data.py` | Sinh ngẫu nhiên theo phân phối phân cảnh các loại Drone, Thuốc BVTV, diện tích nông trại và Giai đoạn sinh trưởng cây trồng |
| 4. Khớp Rule Engine | `merge_data.py` | Chạy Rule-based gán nhãn `decision_action` cuối cùng dựa trên sự kết hợp phức tạp của thời tiết và cấu hình Drone/Cây trồng. Lưu thành file CSV cuối (`final_training_data.csv`) |

---

## 7. Ghi chú kỹ thuật

- File final training data với kích thước lớn (~185K dòng) **không nên commit trực tiếp** lên GitHub nếu dung lượng vượt hạn mức (thêm vào `.gitignore` hoặc zip lại).
- Để tái tạo toàn bộ Data Pipeline từ đầu, chạy script shell / python tuần tự trong thư mục `src/data_pipeline/`.
- File data dictionary này đã được dùng cho Package nộp cuối kỳ.
