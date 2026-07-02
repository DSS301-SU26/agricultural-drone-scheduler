# Dự án: Hệ thống Hỗ trợ Ra quyết định Điều phối Drone Nông nghiệp (Agricultural Drone Scheduler DSS)
**Môn học:** Decision Support Systems for Business Intelligence (DSS301)
**Học kỳ:** Summer 2026
**Trạng thái tài liệu:** Báo cáo Tuần 5

## Contents
1. Giới thiệu
2. Quyết định loại mô hình
3. Chuẩn bị dữ liệu
4. Tiền xử lý dữ liệu
5. Mô hình chính và mô hình so sánh
   5.1. Mô hình chính: Random Forest
   5.2. Mô hình so sánh: Logistic Regression & Decision Tree
6. Training trên dataset UAV
7. Log hyperparameter
8. Prediction trên tập test

---

## 1. Giới thiệu
Trong giai đoạn này, nhóm thực hiện xây dựng mô hình học máy (Machine Learning) hỗ trợ ra quyết định (DSS) cho bài toán lập lịch bay và vận hành Drone nông nghiệp dựa trên dữ liệu khí tượng (thời tiết) và tình trạng mùa màng. Mục tiêu của mô hình là dự đoán quyết định hành động `decision_action` nhằm đánh giá xem thiết bị có nên cất cánh hay không. Đây là bài toán phân loại nhiều nhãn (Multi-class Classification), phù hợp với hướng tiếp cận Machine Learning Classification hơn là Regression.

Dữ liệu sử dụng gồm các thuộc tính liên quan đến thời tiết (nhiệt độ, độ ẩm, tốc độ gió, sức gió giật, lượng mưa, xác suất mưa, tầm nhìn), thời gian và các đặc trưng hình ảnh đồng ruộng. Trên cơ sở đó, nhóm tiến hành tiền xử lý dữ liệu, xây dựng mô hình chính, mô hình so sánh, và đánh giá kết quả trên tập kiểm tra.

## 2. Quyết định loại mô hình
Sau khi phân tích bản chất của biến mục tiêu `decision_action` với 4 trạng thái phân loại riêng biệt (`TAKE_OFF`, `DELAY_FLIGHT`, `LOCK_SPRAY`, `RETURN_TO_CHARGING`), nhóm quyết định lựa chọn **ML Classification** làm phương pháp chính. Lý do là `decision_action` là biến phân loại, không phải biến số liên tục. Vì vậy, regression không phù hợp với bài toán này.

Dù hệ thống có xây dựng tập luật (rule-based) để gán nhãn dữ liệu chuẩn ban đầu, nhưng việc đào tạo mô hình học máy sẽ giúp hệ thống phân tích và dự đoán (inference) với tốc độ cao trên quy mô dữ liệu lớn, cũng như tự học được các tương tác đa chiều phức tạp giữa môi trường và thời tiết.

## 3. Chuẩn bị dữ liệu
Dữ liệu được thu thập và tổng hợp thông qua quy trình tự động lấy thông tin từ WeatherAPI và lưu trữ tại `final_training_data.csv`. Trong quá trình chuẩn bị, dữ liệu trùng lặp (các bản ghi có `location_name` và `timestamp` giống nhau) được làm sạch và chỉ giữ lại bản ghi mới nhất.

Biến mục tiêu `decision_action` gồm 4 lớp:
* `TAKE_OFF`: Điều kiện lý tưởng.
* `DELAY_FLIGHT`: Hoãn do rủi ro thời tiết (mưa, nắng nóng).
* `LOCK_SPRAY`: Khóa lệnh bay do gió giật vượt ngưỡng.
* `RETURN_TO_CHARGING`: Bắt buộc thu hồi do có mưa.

Các đặc trưng đầu vào chủ yếu là các biến số học (Numeric variables): `temperature_2m`, `relative_humidity_2m`, `precipitation_probability`, `precipitation`, `cloud_cover`, `visibility`, `wind_speed_10m`, `wind_gusts_10m`, `weather_code`, `hour`, `dayofweek`, `month` (cùng các biến đặc trưng hình ảnh nếu có).

Dữ liệu được chia Train/Test theo tỷ lệ 75/25. Để bài toán phản ánh đúng thực tế dự báo thời tiết mới, nhóm áp dụng kỹ thuật **GroupShuffleSplit** nhóm theo thuộc tính `timestamp` để ngăn chặn hiện tượng rò rỉ dữ liệu (data leakage) giữa các địa điểm trong cùng một khung giờ.

## 4. Tiền xử lý dữ liệu
Do dữ liệu đầu vào là biến số và có thể có giá trị thiếu, nhóm sử dụng pipeline tiền xử lý để bảo đảm quá trình train và test được nhất quán. Cụ thể:
* **Với mô hình Tree-based (Random Forest, Decision Tree):** Sử dụng `SimpleImputer(strategy="median")` để điền các giá trị bị thiếu bằng giá trị trung vị.
* **Với mô hình tuyến tính (Logistic Regression):** Dữ liệu được bổ sung thêm bước chuẩn hóa thang đo bằng `StandardScaler()` giúp thuật toán hội tụ tốt hơn.

Việc sử dụng `ColumnTransformer` và `Pipeline` giúp toàn bộ quá trình tiền xử lý và huấn luyện được đóng gói rõ ràng, tránh lỗi rò rỉ dữ liệu từ tập Test sang Train.

## 5. Mô hình chính và mô hình so sánh
Nhóm triển khai các mô hình để so sánh hiệu quả:

### 5.1. Mô hình chính: Random Forest
Random Forest được chọn làm mô hình chính vì đây là một thuật toán ensemble cực kỳ mạnh mẽ, xử lý rất tốt các tương tác phi tuyến tính của dữ liệu thời tiết (ví dụ: nhiệt độ cao nhưng độ ẩm lại thấp). Mô hình ít bị ảnh hưởng bởi giá trị ngoại lai (outliers) và xử lý khá ổn định các class mất cân bằng.
Hyperparameter sử dụng:
* `n_estimators = 250`
* `max_depth = 10`
* `class_weight = "balanced_subsample"`

### 5.2. Mô hình so sánh: Logistic Regression & Decision Tree
Hai thuật toán được sử dụng để đối chiếu:
* **Logistic Regression:** Phù hợp làm baseline cho mô hình tuyến tính, có tốc độ xử lý siêu nhanh nhưng có thể giới hạn khả năng học nếu dữ liệu phi tuyến tính. (`max_iter = 2000`, `class_weight = "balanced"`).
* **Decision Tree:** Mô hình cây quyết định đơn giản, độ phức tạp thấp, làm cơ sở so sánh hiệu năng cho thuật toán Random Forest. (`max_depth = 6`).

Các mô hình được huấn luyện trên cùng một tập train và đánh giá trên cùng tập test để đảm bảo tính công bằng.

## 6. Training trên dataset UAV
Sau khi dữ liệu được làm sạch và chia train/test (dùng GroupShuffleSplit), nhóm khởi chạy pipeline tự động. Hệ thống sẽ lặp qua danh sách các mô hình (`model_candidates`) và gọi hàm `fit()` trên `X_train`.

Quá trình training cho thấy:
* Random Forest và Decision Tree thường xuyên bám sát thực tế các luật giới hạn phi tuyến tính do chúng chia cắt dữ liệu theo các ngưỡng điều kiện khá tốt.
* Mô hình nào cho ra chỉ số **Macro F1** tốt nhất sẽ được lưu thành deployment model (`drone_decision_model.joblib`) để phục vụ phía server.

## 7. Log hyperparameter
Nhóm đã ghi lại các hyperparameter chính được định nghĩa trong mã nguồn nhằm phục vụ việc báo cáo và tái lập kết quả huấn luyện:

| Model | Hyperparameters |
|---|---|
| **Random Forest** | `n_estimators=250`, `max_depth=10`, `min_samples_leaf=4`, `class_weight="balanced_subsample"`, `random_state=42` |
| **Logistic Regression** | `max_iter=2000`, `class_weight="balanced"`, `random_state=42` |
| **Decision Tree** | `max_depth=6`, `min_samples_leaf=8`, `class_weight="balanced"`, `random_state=42` |

Việc log hyperparameter giúp đảm bảo mô hình có thể kiểm tra lại và làm nền tảng cho việc tinh chỉnh (fine-tuning) sau này.

## 8. Prediction trên tập test
Cả ba mô hình được sử dụng để dự đoán trên `X_test`. Do tập dữ liệu này bao gồm những khung thời gian mới hoàn toàn (nhờ GroupShuffleSplit), mô hình được đánh giá khắt khe nhất dựa trên:
* **Accuracy** (Độ chính xác tổng thể).
* **Macro Precision / Macro Recall / Macro F1** (Rất quan trọng, để đảm bảo hệ thống không dự đoán thiên lệch về nhãn quá đa số như `TAKE_OFF` mà bỏ quên các nhãn nguy hiểm như `LOCK_SPRAY` hay `RETURN_TO_CHARGING`).

Kết quả cuối cùng và Classification Report được tự động sinh ra và ghi đè vào file `reports/classification_report.txt` cùng `reports/model_metrics.csv`. Căn cứ vào đó, nhóm có thể khẳng định mô hình chính (Random Forest) cho khả năng ổn định và phát hiện lỗi do điều kiện thời tiết khắc nghiệt một cách an toàn nhất.
