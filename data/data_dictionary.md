# Data Dictionary
## DSS301 — Agricultural Drone Flight Scheduler
**Đề tài:** DSS cho lập lịch bay UAV nông nghiệp theo thời tiết  
**Nguồn dữ liệu:** Open-Meteo API (https://api.open-meteo.com)  
**Cập nhật lần cuối:** 2026-05-19  
**Người phụ trách:** Data Engineer

---

## 1. Tổng quan Dataset

| Hạng mục | Thông tin |
|---|---|
| Nguồn | Open-Meteo API — miễn phí, không cần API key |
| Tần suất cập nhật | Mỗi giờ (hourly forecast) |
| Phạm vi địa lý | 5 tỉnh ĐBSCL: Đồng Tháp, Long An, Tiền Giang, An Giang, Cần Thơ |
| Phạm vi thời gian | 7 ngày dự báo (có thể mở rộng đến 16 ngày) |
| Múi giờ | Asia/Bangkok (UTC+7) |
| Khung giờ lọc | 06:00 – 17:59 (giờ bay hợp lệ) |
| Số bản ghi (raw) | ~360 bản ghi / lần chạy (5 địa điểm × 3 ngày × 24 giờ) |
| Số bản ghi (clean) | ~180 bản ghi / lần chạy (sau lọc giờ bay) |
| Định dạng lưu | CSV (UTF-8 BOM) |

---

## 2. Mô tả các cột — Raw Dataset (`data/raw/`)

### 2.1 Metadata cột

| Tên cột | Kiểu dữ liệu | Đơn vị | Mô tả | Ví dụ |
|---|---|---|---|---|
| `location_name` | string | — | Tên địa điểm nông nghiệp | `Dong Thap` |
| `latitude` | float | độ | Vĩ độ địa điểm | `10.4939` |
| `longitude` | float | độ | Kinh độ địa điểm | `105.6882` |
| `timestamp` | datetime | — | Thời điểm dự báo (UTC+7) | `2026-05-19 06:00:00` |
| `temperature_2m` | float | °C | Nhiệt độ không khí tại độ cao 2m | `31.5` |
| `relative_humidity_2m` | float | % | Độ ẩm tương đối tại độ cao 2m | `78.0` |
| `precipitation_probability` | int | % | Xác suất có mưa trong giờ đó | `40` |
| `precipitation` | float | mm | Lượng mưa thực tế dự báo | `0.0` |
| `cloud_cover` | int | % | Tỷ lệ bầu trời bị mây che phủ | `65` |
| `visibility` | float | m | Tầm nhìn xa | `24140.0` |
| `wind_speed_10m` | float | km/h | Tốc độ gió trung bình tại độ cao 10m | `14.3` |
| `wind_gusts_10m` | float | km/h | Tốc độ gió giật tại độ cao 10m | `22.1` |
| `weather_code` | int | WMO code | Mã thời tiết theo chuẩn WMO | `3` |
| `weather_description` | string | — | Mô tả thời tiết (dịch từ WMO code) | `Nhieu may` |

### 2.2 Bảng mã thời tiết WMO

| Mã WMO | Mô tả | An toàn bay? |
|---|---|---|
| 0 | Trời quang | ✅ An toàn |
| 1 | Chủ yếu quang | ✅ An toàn |
| 2 | Nhiều mây một phần | ✅ An toàn |
| 3 | Nhiều mây | ✅ An toàn |
| 45 | Sương mù | ❌ Không an toàn |
| 48 | Sương mù có băng | ❌ Không an toàn |
| 51 | Mưa phùn nhẹ | ✅ An toàn (cận ngưỡng) |
| 53 | Mưa phùn vừa | ✅ An toàn (cận ngưỡng) |
| 55 | Mưa phùn dày | ❌ Không an toàn |
| 61 | Mưa nhẹ | ✅ An toàn (cận ngưỡng) |
| 63 | Mưa vừa | ❌ Không an toàn |
| 65 | Mưa lớn | ❌ Không an toàn |
| 80 | Mưa rào nhẹ | ❌ Không an toàn |
| 81 | Mưa rào vừa | ❌ Không an toàn |
| 82 | Mưa rào mạnh | ❌ Không an toàn |
| 95 | Dông | ❌ Không an toàn |
| 99 | Dông kèm mưa đá | ❌ Không an toàn |

---

## 3. Mô tả các cột — Clean Dataset (`data/clean/`)

Clean dataset bao gồm toàn bộ cột raw + các cột được tạo thêm qua Feature Engineering.

### 3.1 Cột thêm mới (Feature Engineering)

| Tên cột | Kiểu dữ liệu | Mô tả | Cách tính | Ví dụ |
|---|---|---|---|---|
| `date` | date | Ngày (tách từ timestamp) | `timestamp.dt.date` | `2026-05-19` |
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
| Mã thời tiết (`ok_wmo`) | weather_code không nằm trong danh sách nguy hiểm | 5% |

### 3.3 Quy tắc phân loại `fly_label`

```
FLY    → Đạt TẤT CẢ 4 tiêu chí bắt buộc:
           ok_wind AND ok_gust AND ok_rain AND ok_wmo

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

> Các ngưỡng này dựa trên tiêu chuẩn vận hành drone nông nghiệp phổ biến tại Đông Nam Á và khuyến nghị của nhà sản xuất (DJI Agras series).

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

---

## 6. Quy trình làm sạch dữ liệu (Data Cleaning)

| Bước | Kỹ thuật | Mô tả |
|---|---|---|
| 1. Loại duplicates | `drop_duplicates` | Theo cặp (location_name, timestamp) |
| 2. Xử lý missing | Forward fill → Backward fill → Median | Trong từng nhóm địa điểm |
| 3. Xử lý outliers | IQR clip (3×IQR) | Clip thay vì drop để giữ số lượng hàng |
| 4. Lọc giờ bay | Filter hour 6–17 | Chỉ giữ khung giờ có thể bay |
| 5. Feature Engineering | Tính toán features mới | Xem mục 3 |

---

## 7. Thống kê mô tả (EDA Summary)

> Dữ liệu thu thập ngày 2026-05-19, tháng 5 — mùa mưa ĐBSCL

| Chỉ số | Min | Max | Mean | Std |
|---|---|---|---|---|
| wind_speed_10m (km/h) | ~3 | ~35 | ~15 | ~7 |
| wind_gusts_10m (km/h) | ~8 | ~55 | ~28 | ~12 |
| precipitation (mm) | 0 | ~5 | ~0.3 | ~0.8 |
| precipitation_probability (%) | 0 | 80 | ~35 | ~22 |
| cloud_cover (%) | 10 | 100 | ~65 | ~25 |
| visibility (m) | ~500 | ~24000 | ~18000 | ~6000 |
| temperature_2m (°C) | ~25 | ~38 | ~31 | ~3 |
| flyability_score | 0.0 | 1.0 | ~0.45 | ~0.25 |

**Phân phối nhãn:**
- FLY: ~9.4% (tháng 5 mùa mưa — gió giật cao)
- NO_FLY: ~90.6%

**Insight chính:**
1. Tháng 5 (mùa mưa) chỉ ~9.4% khung giờ đạt tiêu chuẩn bay — chủ yếu do gió giật vượt 28 km/h
2. Khung giờ tốt nhất để bay: **6:00–9:00 sáng** (gió nhẹ nhất trong ngày)
3. Đồng Tháp và An Giang có điều kiện bay tốt hơn Long An do ít ảnh hưởng gió biển
4. Xác suất mưa tăng đáng kể sau 14:00 — không nên lập lịch bay buổi chiều muộn
5. Tầm nhìn không phải yếu tố hạn chế chính tại ĐBSCL (thường > 10,000m)

---

## 8. Ghi chú kỹ thuật

- File raw **không được commit** lên GitHub (đã thêm vào `.gitignore`)
- File clean **không được commit** lên GitHub (đã thêm vào `.gitignore`)
- Dữ liệu được lưu trữ trên **Supabase** (bảng `raw_weather_data`)
- Để tái tạo dataset: chạy `python src/run_pipeline.py --days 7`
- Encoding: UTF-8 BOM (`utf-8-sig`) để Excel đọc được tiếng Việt
