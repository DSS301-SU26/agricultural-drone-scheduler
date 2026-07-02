# BẢN MÔ TẢ MODEL PROTOTYPE - HỆ THỐNG DSS
**Môn học:** DSS301 - Hệ thống hỗ trợ ra quyết định (Decision Support System)
**Chủ đề:** Hỗ trợ ra quyết định vận hành đội bay nông nghiệp (Agricultural Drone Scheduler)
**Dự án:** Hệ thống UAV/Drone Ecosystem

---

## 1. Tổng quan Mô hình (Model Overview)
Mô hình hỗ trợ ra quyết định (DSS Model) trong dự án này được thiết kế để tự động hóa quá trình đánh giá và lập lịch cất cánh cho các máy bay không người lái (UAV) trong nông nghiệp (đặc biệt tại khu vực Đồng bằng sông Cửu Long như An Giang, Cần Thơ, Đồng Tháp, v.v.).

Mục tiêu chính của mô hình là tối ưu hóa cửa sổ bay (flyability), tăng hiệu quả phun tưới, đồng thời bảo vệ các thiết bị phần cứng đắt tiền khỏi rủi ro thời tiết. Hệ thống là sự kết hợp (hybrid) giữa:
1. **Rule-based Decision Engine:** Hệ luật chuyên gia diễn giải các ngưỡng thời tiết an toàn nhằm gán nhãn quyết định rõ ràng, có tính giải thích cao (Explainable).
2. **Machine Learning Model:** Các mô hình học máy (như Random Forest, Decision Tree) học hỏi từ dữ liệu lịch sử và giả lập để dự báo trước quyết định một cách tự động, sẵn sàng scale với lượng dữ liệu lớn.

## 2. Dữ liệu Đầu vào (Model Inputs & Features)
Dữ liệu đầu vào thu thập từ các trạm thời tiết (API) và dữ liệu đồng ruộng, bao gồm:
*   **Khí tượng (Meteorology):**
    *   `temperature_2m` (Nhiệt độ): Đánh giá rủi ro sốc nhiệt cho thiết bị và mức độ bay hơi của dung dịch phun.
    *   `relative_humidity_2m` (Độ ẩm): Góp phần chẩn đoán tình trạng đất.
    *   `precipitation` & `precipitation_probability` (Lượng mưa và Xác suất mưa): Ước lượng rủi ro rơi hoặc hỏng mạch điện.
    *   `wind_speed_10m` & `wind_gusts_10m` (Tốc độ gió và gió giật): Ảnh hưởng đến quỹ đạo bay và nguy cơ trôi dạt thuốc (pesticide drift).
    *   `visibility` & `cloud_cover` (Tầm nhìn & mây): Đảm bảo điều kiện bay VLOS/BVLOS.
    *   `weather_code`: Mã điều kiện thời tiết theo WMO hoặc WeatherAPI.
*   **Thời gian (Time Context):** `hour`, `dayofweek`, `month` giúp model học được tính chu kỳ của môi trường.
*   **Nông học (Agronomy):** `crop_condition` (HEALTHY, DRY_SOIL, WATER_STRESS) - có thể được trích xuất từ camera quang phổ hoặc được suy diễn từ điều kiện nhiệt độ/độ ẩm.

## 3. Hệ luật và Động cơ Quyết định (Decision Engine & Thresholds)
Mô hình ra quyết định hoạt động dựa trên các ngưỡng sinh tử (safety thresholds) được định nghĩa sẵn trong hệ thống (`DecisionThresholds`):

### 3.1. Các Ngưỡng An Toàn (Thresholds)
*   **Gió tối đa:** $20.0 \text{ km/h}$ | **Gió giật tối đa:** $28.0 \text{ km/h}$
*   **Nhiệt độ an toàn:** $\leq 35^{\circ}C$
*   **Xác suất mưa an toàn:** $\leq 30\%$ (Quay về trạm sạc khẩn cấp nếu $> 70\%$)
*   **Tầm nhìn:** $\geq 1000m$ | **Độ phủ mây:** $\leq 80\%$

### 3.2. Flyability Score (Chỉ số khả năng bay)
Được tính toán có trọng số nhằm xếp hạng mức độ thuận lợi của các khung giờ bay.
*   Nhiệt độ ($30\%$), Gió ($20\%$), Gió giật ($15\%$), Mưa ($15\%$), Xác suất mưa ($8\%$), Mây ($5\%$), Mã thời tiết ($4\%$), Tầm nhìn ($3\%$).

### 3.3. Dynamic Flow Rate (Tính toán lưu lượng xả linh hoạt)
*   Tỷ lệ chuẩn (Baseline): $100\%$
*   Nếu cây thiếu nước/đất khô (`DRY_SOIL`, `WATER_STRESS`): Tăng thêm $8\% - 15\%$ lưu lượng.
*   Nếu nhiệt độ $\geq 35^{\circ}C$, gió $> 15 \text{ km/h}$, hoặc sắp mưa: Giảm $10\% - 15\%$ lưu lượng để tránh bốc hơi và rửa trôi lãng phí.

## 4. Đầu ra của Mô hình (Model Outputs)
Từ các đầu vào trên, hệ thống xuất ra các biến quyết định hỗ trợ operator (người điều khiển):

1.  **Decision Action (Hành động vận hành):**
    *   🟢 `TAKE_OFF`: Điều kiện lý tưởng, tiến hành nhiệm vụ.
    *   🟡 `DELAY_FLIGHT`: Hoãn bay, chờ đánh giá lại do nhiệt độ cao hoặc có khả năng mưa.
    *   🟠 `LOCK_SPRAY`: Khóa lệnh cất cánh do gió giật vượt ngưỡng, đe dọa trực tiếp tới an toàn thiết bị.
    *   🔴 `RETURN_TO_CHARGING`: Bắt buộc thu hồi Drone về trạm ngay lập tức do có mưa hoặc thời tiết cực đoan.
2.  **Risk Level:** Tổng hợp rủi ro thành `LOW`, `MEDIUM`, `HIGH`.
3.  **Recommendation Text:** Đoạn text giải thích trực quan bằng tiếng Việt cho người dùng cuối (Explainable output). Vd: *"TAKE_OFF: Điều kiện bay chấp nhận được. Gió 12.0 km/h..."*
4.  **Estimated Damage Cost:** Ước tính chi phí thiệt hại (VD: \$2,000 USD) nếu bỏ qua cảnh báo và cố tình bay trong khung giờ cấm.

## 5. Machine Learning Pipeline
*   **Tiền xử lý (Preprocessing):** `SimpleImputer` xử lý missing values (median) và `StandardScaler` (dùng cho Linear Models) để chuẩn hóa dải giá trị.
*   **Mô hình huấn luyện (Algorithms Evaluated):** Decision Tree Classifier, Random Forest Classifier, Logistic Regression. Mô hình tree-based được ưa chuộng do phản ánh tốt các tập luật if-else phi tuyến tính của thời tiết.
*   **Cơ chế Validation:** Sử dụng `GroupShuffleSplit` gom nhóm theo `timestamp` (thời gian). Kỹ thuật này giúp mô hình không bị thiên lệch (data leakage) khi dữ liệu thời tiết ở các location gần nhau thường xuyên trùng lặp về mặt ngữ cảnh mây mưa tại cùng một thời điểm.

## 6. Tổng kết & Đánh giá (RBL Reflection)
*   **Giá trị hỗ trợ quyết định (DSS Value):** Mô hình biến đổi dữ liệu thô (raw weather data) thành một hành động cụ thể (Actionable Insights). Hệ thống không chỉ đưa ra cảnh báo "Không an toàn" mà còn đề xuất "Khung giờ bay tốt nhất" (Best Slot Recommendation) cho người nông dân.
*   **Trade-off (Sự đánh đổi):** Mô hình lựa chọn cách tiếp cận kết hợp (Hybrid) giữa Rules và ML. Rules đảm bảo 100% tuân thủ quy định an toàn cứng (không đâm rơi drone vì gió to), trong khi ML giúp tự động phân loại, mở rộng và phát hiện quy luật phi tuyến tiềm ẩn.
*   **Future Work:** Tích hợp dữ liệu hình ảnh viễn thám thực tế (Multispectral Images) từ camera gắn trên UAV thay vì chỉ ước tính trạng thái đất dựa trên nhiệt độ và độ ẩm không khí.
