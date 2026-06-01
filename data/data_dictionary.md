# Data Dictionary
## DSS301 — Agricultural Drone Flight Scheduler
**Đề tài:** DSS cho lập lịch bay UAV nông nghiệp theo thời tiết  
**Nguồn dữ liệu:** WeatherAPI Forecast API (https://www.weatherapi.com)  
**Cập nhật lần cuối:** 2026-06-01  
**Người phụ trách:** Data Engineer

---

## 1. Tổng quan Dataset

| Hạng mục | Thông tin |
|---|---|
| Nguồn | WeatherAPI Forecast API — yêu cầu API key |
| Tần suất cập nhật | Mỗi giờ (hourly forecast) |
| Phạm vi địa lý | 7 địa điểm: 5 tỉnh/thành ĐBSCL và 2 địa điểm đối chiếu |
| Phạm vi thời gian | 3 ngày dự báo |
| Múi giờ | UTC+7 |
| Khung giờ lọc | 06:00 – 17:59 (giờ bay hợp lệ) |
| Số bản ghi (raw) | 504 bản ghi / lần chạy (7 địa điểm × 3 ngày × 24 giờ) |
| Số bản ghi (clean) | 252 bản ghi / lần chạy (7 địa điểm × 3 ngày × 12 giờ) |
| Định dạng lưu | CSV (UTF-8 BOM) |

---

## 2. Mô tả các cột — Raw Dataset (`data/raw/`)

### 2.1 Metadata cột

| Tên cột | Kiểu dữ liệu | Đơn vị | Mô tả | Ví dụ |
|---|---|---|---|---|
| `location_name` | string | — | Tên địa điểm nông nghiệp | `Dong Thap` |
| `latitude` | float | độ | Vĩ độ địa điểm | `10.4939` |
| `longitude` | float | độ | Kinh độ địa điểm | `105.6882` |
| `timestamp` | datetime | — | Thời điểm dự báo (UTC+7) | `2026-06-01 06:00:00` |
| `source` | string | — | Nguồn cung cấp dữ liệu | `WeatherAPI` |
| `temperature_2m` | float | °C | Nhiệt độ không khí tại độ cao 2m | `31.5` |
| `relative_humidity_2m` | float | % | Độ ẩm tương đối tại độ cao 2m | `78.0` |
| `precipitation_probability` | int | % | Xác suất có mưa trong giờ đó | `40` |
| `precipitation` | float | mm | Lượng mưa thực tế dự báo | `0.0` |
| `cloud_cover` | int | % | Tỷ lệ bầu trời bị mây che phủ | `65` |
| `visibility` | float | m | Tầm nhìn xa | `24140.0` |
| `wind_speed_10m` | float | km/h | Tốc độ gió trung bình tại độ cao 10m | `14.3` |
| `wind_gusts_10m` | float | km/h | Tốc độ gió giật tại độ cao 10m | `22.1` |
| `weather_code` | int | WeatherAPI code | Mã điều kiện thời tiết của WeatherAPI | `1003` |
| `weather_description` | string | — | Mô tả điều kiện thời tiết từ WeatherAPI | `Partly Cloudy` |

### 2.2 Bảng mã thời tiết WeatherAPI

Các mã dưới đây xuất hiện trong clean dataset ngày 2026-06-01. Nhãn bay cuối cùng còn phụ thuộc các ngưỡng gió, mưa, mây và tầm nhìn tại mục 3.

| Mã WeatherAPI | Mô tả | Ghi chú |
|---|---|---|
| 1000 | Sunny | Điều kiện quang đãng |
| 1003 | Partly Cloudy | Có mây một phần |
| 1006 | Cloudy | Nhiều mây |
| 1009 | Overcast | Trời âm u |
| 1030 | Mist | Sương mỏng |
| 1063 | Patchy rain nearby | Có mưa rải rác gần khu vực |
| 1087 | Thundery outbreaks in nearby | Không an toàn: có dông gần khu vực |
| 1150, 1153 | Patchy light drizzle / Light drizzle | Mưa phùn nhẹ |
| 1180, 1183 | Patchy light rain / Light rain | Mưa nhẹ |
| 1240 | Light rain shower | Mưa rào nhẹ |

Các mã được cấu hình là không an toàn: `1087`, `1135`, `1147`, `1192`, `1195`, `1201`, `1243`, `1246`, `1273`, `1276`, `1279`, `1282`.

---

## 3. Mô tả các cột — Clean Dataset (`data/clean/`)

Clean dataset bao gồm toàn bộ cột raw + các cột được tạo thêm qua Feature Engineering.

### 3.1 Cột thêm mới (Feature Engineering)

| Tên cột | Kiểu dữ liệu | Mô tả | Cách tính | Ví dụ |
|---|---|---|---|---|
| `date` | date | Ngày (tách từ timestamp) | `timestamp.dt.date` | `2026-06-01` |
| `hour` | int | Giờ trong ngày | `timestamp.dt.hour` | `8` |
| `dayofweek` | int | Thứ trong tuần (0=Thứ 2, 6=CN) | `timestamp.dt.dayofweek` | `1` |
| `month` | int | Tháng | `timestamp.dt.month` | `5` |
| `flyability_score` | float | Điểm khả năng bay (0.0–1.0) | Tổng có trọng số 7 tiêu chí | `0.85` |
| `fly_label` | string | Nhãn quyết định bay | Dựa trên 4 tiêu chí bắt buộc | `FLY` / `NO_FLY` |
| `risk_level` | string | Mức độ rủi ro | Dựa trên flyability_score | `LOW` / `MEDIUM` / `HIGH` |

### 3.2 Cách tính `flyability_score`

Điểm được tính theo công thức có trọng số:

| Tiêu chí | Điều kiện đạt | Trọng số |
|---|---|---|
| Tốc độ gió (`ok_wind`) | wind_speed_10m ≤ 20 km/h | 30% |
| Gió giật (`ok_gust`) | wind_gusts_10m ≤ 28 km/h | 20% |
| Không mưa (`ok_rain`) | precipitation = 0 mm | 20% |
| Xác suất mưa (`ok_rain_prob`) | precipitation_probability ≤ 30% | 10% |
| Độ che phủ mây (`ok_cloud`) | cloud_cover ≤ 80% | 10% |
| Tầm nhìn (`ok_vis`) | visibility ≥ 1000 m | 5% |
| Mã thời tiết (`ok_weather`) | weather_code không nằm trong danh sách nguy hiểm | 5% |

### 3.3 Quy tắc phân loại `fly_label`

```
FLY    → Đạt TẤT CẢ 4 tiêu chí bắt buộc:
           ok_wind AND ok_gust AND ok_rain AND ok_weather

NO_FLY → Vi phạm ÍT NHẤT 1 trong 4 tiêu chí bắt buộc
```

### 3.4 Quy tắc phân loại `risk_level`

| risk_level | Khoảng flyability_score | Ý nghĩa |
|---|---|---|
| `LOW` | 0.70 – 1.00 | Điều kiện tốt, an toàn bay |
| `MEDIUM` | 0.40 – 0.69 | Cần cân nhắc, theo dõi thêm |
| `HIGH` | 0.00 – 0.39 | Không nên bay, rủi ro cao |

---

## 4. Ngưỡng an toàn bay drone nông nghiệp

> Các ngưỡng dưới đây là cấu hình nghiệp vụ hiện tại của hệ thống hỗ trợ quyết định.

| Chỉ số | Ngưỡng an toàn | Căn cứ |
|---|---|---|
| Tốc độ gió | ≤ 20 km/h (Cấp 4 Beaufort) | Drone nông nghiệp mất kiểm soát trên cấp 4 |
| Gió giật | ≤ 28 km/h | Gió giật đột ngột gây mất ổn định |
| Lượng mưa | = 0 mm | Mưa làm hỏng thiết bị phun, ảnh hưởng GPS |
| Xác suất mưa | ≤ 30% | Ngưỡng an toàn lập kế hoạch trước |
| Độ che phủ mây | ≤ 80% | Ảnh hưởng GPS và tầm nhìn của pilot |
| Tầm nhìn | ≥ 1000 m | Yêu cầu tối thiểu để quan sát drone |

---

## 5. Địa điểm thu thập dữ liệu

| Tên địa điểm | Tỉnh | Vĩ độ | Kinh độ | Đặc điểm canh tác |
|---|---|---|---|---|
| Dong Thap | Đồng Tháp | 10.4939 | 105.6882 | Lúa, sen, xoài |
| Long An | Long An | 10.5360 | 106.4052 | Lúa, thanh long |
| Tien Giang | Tiền Giang | 10.3598 | 106.3567 | Lúa, sầu riêng, dứa |
| An Giang | An Giang | 10.5216 | 105.1259 | Lúa, cá tra, rau màu |
| Can Tho | Cần Thơ | 10.0341 | 105.7878 | Lúa, cây ăn trái |
| Ho Chi Minh | TP. Hồ Chí Minh | 10.7769 | 106.7009 | Địa điểm đối chiếu |
| Ha Noi | Hà Nội | 21.0285 | 105.8542 | Địa điểm đối chiếu |

---

## 6. Quy trình làm sạch dữ liệu (Data Cleaning)

| Bước | Kỹ thuật | Mô tả |
|---|---|---|
| 1. Loại duplicates | `drop_duplicates` | Theo cặp (location_name, timestamp) |
| 2. Xử lý missing | Forward fill → Backward fill → Median | Trong từng nhóm địa điểm |
| 3. Lọc giờ bay | Filter hour 6–17 | Chỉ giữ khung giờ có thể bay |
| 4. Feature Engineering | Tính toán features mới | Xem mục 3 |

---

## 7. Thống kê mô tả (EDA Summary)

> Dữ liệu forecast được thu thập ngày 2026-06-01 cho khoảng 2026-06-01 đến 2026-06-03.

| Chỉ số | Min | Max | Mean | Std |
|---|---|---|---|---|
| wind_speed_10m (km/h) | 1.10 | 30.20 | 13.65 | 6.56 |
| wind_gusts_10m (km/h) | 1.70 | 44.70 | 19.04 | 9.29 |
| precipitation (mm) | 0.00 | 2.22 | 0.35 | 0.56 |
| precipitation_probability (%) | 0 | 91 | 42.31 | 33.42 |
| cloud_cover (%) | 12 | 100 | 68.94 | 21.05 |
| visibility (m) | 2000 | 10000 | 9420.63 | 1804.61 |
| temperature_2m (°C) | 24.10 | 38.90 | 28.81 | 3.28 |
| flyability_score | 0.10 | 1.00 | 0.70 | 0.25 |

**Phân phối nhãn:**
- FLY: 70 bản ghi (27.8%)
- NO_FLY: 182 bản ghi (72.2%)

**Insight chính:**
1. 27.8% khung giờ đạt nhãn `FLY`; 72.2% khung giờ mang nhãn `NO_FLY`.
2. Khung giờ có tỷ lệ `FLY` cao nhất là **06:00–07:59**.
3. Tầm nhìn không phải yếu tố hạn chế chính trong lần chạy này: tối thiểu 2,000 m.
4. Hai địa điểm Hà Nội và TP. Hồ Chí Minh được giữ lại để đối chiếu với nhóm địa điểm ĐBSCL.

---

## 8. Ghi chú kỹ thuật

- File raw **không được commit** lên GitHub (đã thêm vào `.gitignore`)
- File clean **không được commit** lên GitHub (đã thêm vào `.gitignore`)
- Dữ liệu được lưu trữ trên **Supabase** (bảng `raw_weather_data`)
- Để tái tạo dataset: chạy `.venv/bin/python src/run_pipeline.py --days 3`
- Encoding: UTF-8 BOM (`utf-8-sig`) để Excel đọc được tiếng Việt
