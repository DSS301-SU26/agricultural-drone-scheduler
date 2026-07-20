<div align="center">
  <h1>🚁 AGRI-FLIGHT SCHEDULER</h1>
  <h3>Hệ Thống Hỗ Trợ Ra Quyết Định (DSS) Lập Lịch Bay UAV Nông Nghiệp</h3>
  <p><i>Môn học: DSS301 - Chuyên canh cây Lúa nước</i></p>
</div>

---

##  1. TỔNG QUAN DỰ ÁN (EXECUTIVE SUMMARY)
**Agri-Flight Scheduler** là một Hệ thống Hỗ trợ Ra quyết định (Decision Support System - DSS) được thiết kế chuyên biệt cho công tác phun thuốc/rải phân bằng Drone (UAV) trên các cánh đồng lúa nước. 

Hệ thống kết hợp dữ liệu thời tiết thời gian thực và mô hình Học máy (Machine Learning) để sinh ra **Flyability Score (Điểm khả năng bay)**. Từ đó, tự động đưa ra các lời khuyên chi tiết về lịch bay, độ cao bay an toàn và lưu lượng xả phù hợp theo từng giai đoạn sinh trưởng của cây lúa.

---

##  2. YÊU CẦU HỆ THỐNG & CÀI ĐẶT
Dự án được phát triển trên môi trường **Python 3.10+**. Để hệ thống hoạt động ổn định nhất, khuyến nghị sử dụng môi trường ảo (Virtual Environment).

**Bước 1: Tạo và kích hoạt môi trường ảo**
```bash
# Trên Windows:
python -m venv .venv
.venv\Scripts\activate

# Trên macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
```

**Bước 2: Cài đặt các thư viện phụ thuộc (Dependencies)**
```bash
pip install -r requirements.txt
```

---

##  3. HƯỚNG DẪN VẬN HÀNH HỆ THỐNG (PIPELINE)

Hệ thống được thiết kế dạng Pipeline tự động hóa từ khâu lấy dữ liệu đến khi đưa ra quyết định trên Dashboard. Chạy lần lượt các lệnh sau:

###  Phase 1: Thu thập và Xử lý dữ liệu (Data Pipeline)
Thực thi luồng công việc tải dữ liệu thời tiết mới nhất qua API, làm sạch (Data Cleaning), chuẩn hóa và nội suy các đặc trưng (Feature Engineering).
```bash
python src/run_pipeline.py
```
*  **Đầu vào:** Gọi API thời tiết / Đọc file raw `data/raw/historical_6years.csv`.
*  **Đầu ra:** Dữ liệu chuẩn hóa được lưu tại `data/clean/` sẵn sàng cho việc huấn luyện.

###  Phase 2: Huấn luyện AI (Model Training)
Huấn luyện mô hình học máy (XGBoost / Random Forest) để tìm ra quy luật giữa thời tiết, giai đoạn lúa và quyết định bay.
```bash
python -m src.decision_model.train_decision_model
```
*  **Đầu ra 1:** Sinh ra tệp siêu dữ liệu mô hình `models/agriflight_model.joblib`.
*  **Đầu ra 2:** Tự động tạo Báo cáo đánh giá hiệu suất mô hình tại `models/evaluation_report.md` (Đạt độ chính xác > 95%).

###  Phase 3: Khởi động Dashboard (User Interface)
Khởi chạy Giao diện tương tác trực quan (Web-based) cho người dùng cuối (Nông dân/Người điều khiển Drone).
```bash
streamlit run src/api.py
```
*  **Truy cập:** Mở trình duyệt và truy cập `http://localhost:8501`.
*  **Tính năng chính:** Xem dự báo thời tiết theo trạm, xem điểm Flyability Score, nhận cảnh báo gió giật, và đọc lời khuyên vận hành Drone chi tiết.

---

## 📂 4. CẤU TRÚC THƯ MỤC NỘP BÀI (SUBMISSION PACKAGE)
Dưới đây là sơ đồ cấu trúc của bộ hồ sơ nộp bài cuối kỳ:

```text
📦 agricultural-drone-scheduler
 ┣ 📂 data/                    # (1) Chứa toàn bộ dữ liệu dự án
 ┃ ┣ 📜 historical_6years.csv    # - Raw dataset (Dữ liệu thô)
 ┃ ┣ 📜 weather_clean_*.csv      # - Clean dataset (Dữ liệu đã làm sạch)
 ┃ ┗ 📜 data_dictionary_final.docx # - Data dictionary (Từ điển dữ liệu)
 ┃
 ┣ 📂 models/                  # (2) Chứa mô hình AI và đánh giá
 ┃ ┣ 📜 agriflight_model.joblib  # - Trained model (Mô hình đã huấn luyện)
 ┃ ┗ 📜 evaluation_report.md     # - Evaluation log (Báo cáo đánh giá metrics)
 ┃
 ┣ 📂 src/                     # (3) Chứa toàn bộ Mã nguồn (Source Code)
 ┃ ┣ 📂 decision_model/          # - Code xử lý và Train AI (train_decision_model.py)
 ┃ ┣ 📂 data_pipeline/           # - Code ETL xử lý dữ liệu
 ┃ ┗ 📜 api.py                   # - Source code Frontend/Dashboard (Streamlit)
 ┃
 ┣ 📜 requirements.txt         # Danh sách thư viện Python
 ┗ 📜 README.md                # Tài liệu hướng dẫn sử dụng (File này)
```

---

## 5. ĐỘI NGŨ PHÁT TRIỂN (TEAM MEMBERS)

Dự án được nghiên cứu và phát triển bởi Nhóm sinh viên:

| STT 	| Họ và Tên Sinh Viên 	| Mã Sinh Viên 	| SĐT 		| Liên hệ (Email) 	   |
|:---:	|:---------------------:|:-------------:|:-------------:|:------------------------:|
| **1** | Ngô Sỹ Bảo Duy 	| SE192371 	| 0939643769 	| Bdbdh3208@gmail.com      |
| **2** | Đinh Thế Quyền 	| SE192359 	| 0908845984 	| quyenthedinh@gmail.com   |
| **3** | Nghê Tài Phát 	| SE192584 	| 0922393339 	| phat280405@gmail.com     |
| **4** | Phan Tất Thành 	| SE192631 	| 0944995611 	| tatthanh070705@gmail.com |
| **5** | Lương Đức Duy 	| SE196575 	| 0906415480 	| ducduy.luong22@gmail.com |

