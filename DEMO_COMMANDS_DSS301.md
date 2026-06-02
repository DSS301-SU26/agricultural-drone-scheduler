# DSS301 Demo Commands - UAV Decision Support System

File nay dung de Phat mo trong IntelliJ va chay demo Data Scientist cho de tai:

**DSS cho lap lich bay Drone/UAV theo thoi tiet**

---

## Quy trinh trinh bay

He thong co 2 pha rieng:

```text
OFFLINE - Xay dung model:
Du lieu lich su + anh mo phong -> clean -> merge -> train va so sanh model
-> chon model tot nhat -> backtesting voi baseline bay co dinh luc 12:00.

ONLINE - Van hanh gan thoi gian hien tai:
WeatherAPI forecast moi nhat -> clean -> upload Supabase
-> model da train + safety policy -> DSS recommendation cho operator.
```

Luu y quan trong:

```text
src/data/final_training_data.csv la snapshot train/backtest, khong phai forecast live.
Timestamp cu trong snapshot train la binh thuong.

data/clean/weather_clean_*.csv la forecast moi nhat dung cho live demo.
Sau khi chay main.py, live_demo tu dong chon file clean moi nhat trong thu muc nay.
```

---

## Pipeline day du tu dau den model AI

Day la chuoi lenh de demo tung buoc ro rang thay vi gom tat ca vao `main.py`.
Chay tung block theo thu tu:

```bash
cd /Users/macprocuaphat/Desktop/DSS301/DSS301_Code/BE/agricultural-drone-scheduler
```

```bash
.venv/bin/python -m src.data_pipeline.fetch_weather --days 3 --save
RAW_FILE="$(ls -t data/raw/weather_raw_*.csv | head -n 1)"
echo "$RAW_FILE"
```

```bash
.venv/bin/python -m src.data_pipeline.clean_data --input "$RAW_FILE"
CLEAN_FILE="$(ls -t data/clean/weather_clean_*.csv | head -n 1)"
echo "$CLEAN_FILE"
```

```bash
cp "$CLEAN_FILE" src/data/clean/
```

```bash
.venv/bin/python -m src.data_pipeline.simulate_images
```

```bash
.venv/bin/python -m src.data_pipeline.extract_features
```

```bash
.venv/bin/python -m src.data_pipeline.merge_data
```

```bash
.venv/bin/python -m src.decision_model.train_decision_model
```

```bash
.venv/bin/python -m src.decision_model.backtest_policy
```

```bash
.venv/bin/python -m src.data_pipeline.load_to_supabase --input "$CLEAN_FILE"
```

```bash
.venv/bin/python -m src.decision_model.live_demo --location "Can Tho"
```

Y nghia tung buoc:

```text
fetch_weather       -> lay forecast WeatherAPI moi nhat vao data/raw
clean_data          -> xu ly missing, loc gio bay va tao flyability_score
cp                  -> dua snapshot clean moi vao thu muc dataset train
simulate_images     -> chon anh weather-scene mo phong theo timestamp
extract_features    -> MobileNetV2 bien moi anh thanh 1280 img_feature
merge_data          -> ghep weather + image feature thanh final_training_data.csv
train_decision_model-> so sanh model va luu model tot nhat
backtest_policy     -> so sanh DSS voi baseline bay co dinh luc 12:00
load_to_supabase    -> upload forecast clean moi nhat
live_demo           -> khuyen nghi van hanh gan thoi gian hien tai
```

Luu y: `extract_features` co the mat vai phut vi MobileNetV2 phai doc tung anh.

---

## 1. Mo Terminal trong IntelliJ

Trong IntelliJ:

1. Mo project `agricultural-drone-scheduler`
2. Bam tab **Terminal** o duoi man hinh
3. Dam bao dang dung thu muc goc:

```bash
pwd
```

Ket qua nen la:

```text
/Users/macprocuaphat/Desktop/DSS301/DSS301_Code/BE/agricultural-drone-scheduler
```

Neu chua dung thu muc, chay:

```bash
cd /Users/macprocuaphat/Desktop/DSS301/DSS301_Code/BE/agricultural-drone-scheduler
```

---

## 2. Train model truoc khi demo

Chay lenh nay de train va so sanh cac model tren snapshot lich su/mo phong:

```bash
.venv/bin/python -m src.decision_model.train_decision_model
```

Lenh nay se tao/cap nhat cac file:

```text
models/drone_decision_model.joblib
reports/model_metrics.csv
reports/classification_report.txt
reports/training_summary.json
reports/recommendation_demo.csv
reports/training_snapshot_best_slot.json
```

Noi voi giang vien:

```text
Em train va so sanh Decision Tree, Random Forest, Logistic Regression voi baseline.
Sau khi loai du lieu trung va chia train/test theo timestamp de tranh leakage,
em danh gia accuracy, precision, recall va F1-score roi chon model tot nhat.
Dong training_snapshot_best_slot chi la vi du tot nhat ben trong snapshot train,
khong phai khuyen nghi bay live. Khuyen nghi live nam o buoc tiep theo.
```

---

## 3. Cap nhat forecast va demo theo du lieu gan thoi diem hien tai

Lay forecast 3 ngay moi nhat, clean va upload Supabase:

```bash
.venv/bin/python main.py
```

Sau do chay live demo. Script se tu doc file forecast clean moi nhat:

```bash
.venv/bin/python -m src.decision_model.live_demo --location "Can Tho"
```

Co the doi `"Can Tho"` thanh cac dia diem sau:

```text
"An Giang"
"Can Tho"
"Dong Thap"
"Ho Chi Minh"
"Long An"
"Tien Giang"
```

Vi du:

```bash
.venv/bin/python -m src.decision_model.live_demo --location "An Giang"
```

```bash
.venv/bin/python -m src.decision_model.live_demo --location "Long An"
```

```bash
.venv/bin/python -m src.decision_model.live_demo --location "Tien Giang"
```

Luu y:

```text
Output co dong Data source de kiem tra script dang doc forecast live nao.
Neu hien tai ngoai gio bay 06:00-17:00, script co the lay slot forecast tiep theo.
Neu muon demo dung gio hien tai, dung cac scenario o muc 4.
Neu chua co anh ruong moi, model dung image embedding tham chieu tu snapshot train
va output se ghi ro: Image embedding : reference snapshot fallback.
```

---

## 4. Demo du 4 tinh huong ra quyet dinh

Day la cac lenh quan trong nhat de thuyet trinh.

### 4.1. Tinh huong 1 - Duoc bay

```bash
.venv/bin/python -m src.decision_model.live_demo --location "Can Tho" --scenario take_off
```

Ket qua mong doi:

```text
DSS decision    : TAKE_OFF
```

Y nghia:

```text
Thoi tiet tot, gio va mua nam trong nguong an toan, drone co the cat canh.
```

---

### 4.2. Tinh huong 2 - Hoan bay

```bash
.venv/bin/python -m src.decision_model.live_demo --location "Can Tho" --scenario delay_flight
```

Ket qua mong doi:

```text
DSS decision    : DELAY_FLIGHT
```

Y nghia:

```text
Nhiet do cao hoac xac suat mua tang, he thong khuyen nghi tam hoan bay va kiem tra khung gio tiep theo.
```

---

### 4.3. Tinh huong 3 - Khoa lenh phun thuoc

```bash
.venv/bin/python -m src.decision_model.live_demo --location "Can Tho" --scenario lock_spray
```

Ket qua mong doi:

```text
DSS decision    : LOCK_SPRAY
```

Y nghia:

```text
Gio hoac gio giat vuot nguong an toan, he thong khoa lenh phun de tranh pesticide drift va giam rui ro mat on dinh UAV.
```

---

### 4.4. Tinh huong 4 - Quay ve tram sac

```bash
.venv/bin/python -m src.decision_model.live_demo --location "Can Tho" --scenario return_to_charging
```

Ket qua mong doi:

```text
DSS decision    : RETURN_TO_CHARGING
```

Y nghia:

```text
Mua hoac thoi tiet nguy hiem, he thong yeu cau drone quay ve tram sac de bao ve thiet bi.
```

---

## 5. Chay backtesting KPI

Lenh nay so sanh:

- Baseline: operator luon bay luc 12:00
- DSS Activated: he thong chon khung gio an toan nhat trong ngay

```bash
.venv/bin/python -m src.decision_model.backtest_policy
```

Lenh nay se tao/cap nhat:

```text
reports/backtesting_summary.csv
reports/backtesting_daily_results.csv
reports/backtesting_summary.json
```

Noi voi giang vien:

```text
Nhom dung backtesting de so sanh baseline schedule voi DSS schedule.
Muc tieu la chung minh DSS co the giam risky operations va giam lang phi nuoc/thuoc trong du lieu mo phong.
```

Luu y khi thuyet trinh:

```text
Day la simulation upper bound trong pham vi mon hoc, khong phai so lieu trien khai thuc te ngoai dong.
Baseline la lich bay co dinh luc 12:00; DSS chon khung gio tot nhat dang co.
```

---

## 6. Lenh demo nen chay khi thuyet trinh

Chay theo thu tu nay:

```bash
cd /Users/macprocuaphat/Desktop/DSS301/DSS301_Code/BE/agricultural-drone-scheduler
```

```bash
.venv/bin/python -m src.decision_model.train_decision_model
```

```bash
.venv/bin/python main.py
```

```bash
.venv/bin/python -m src.decision_model.live_demo --location "Can Tho"
```

```bash
.venv/bin/python -m src.decision_model.live_demo --location "Can Tho" --scenario take_off
```

```bash
.venv/bin/python -m src.decision_model.live_demo --location "Can Tho" --scenario delay_flight
```

```bash
.venv/bin/python -m src.decision_model.live_demo --location "Can Tho" --scenario lock_spray
```

```bash
.venv/bin/python -m src.decision_model.live_demo --location "Can Tho" --scenario return_to_charging
```

```bash
.venv/bin/python -m src.decision_model.backtest_policy
```

---

## 7. Cach giai thich nhanh vai tro cua Phat

Neu giang vien hoi Phat lam gi, noi:

```text
Em phu trach Data Scientist. Sau khi Data Engineer thu thap va clean du lieu thoi tiet + anh, em xay dung Decision Model bang scikit-learn.
Em tao nhan quyet dinh gom TAKE_OFF, DELAY_FLIGHT, LOCK_SPRAY va RETURN_TO_CHARGING.
Sau do em train va so sanh Decision Tree, Random Forest, Logistic Regression bang accuracy, precision, recall va F1-score.
Cuoi cung em xuat recommendation de dashboard hien thi quyet dinh cho operator.
He thong la hybrid DSS: neu ML goi y xung dot voi rule an toan cung, rule an toan se override.
Anh hien tai la weather-scene simulation dung MobileNetV2 embedding, chua phai anh ruong gan nhan thuc te.
```

---

## 8. Neu bi loi thuong gap

### Loi: khong tim thay model

Neu gap loi lien quan den:

```text
models/drone_decision_model.joblib
```

Chay lai:

```bash
.venv/bin/python -m src.decision_model.train_decision_model
```

### Loi: sai thu muc

Neu lenh khong chay, kiem tra:

```bash
pwd
```

Neu khong phai project folder, chay:

```bash
cd /Users/macprocuaphat/Desktop/DSS301/DSS301_Code/BE/agricultural-drone-scheduler
```

### Loi: Python khong co thu vien

Dung dung Python cua `.venv`:

```bash
.venv/bin/python -m src.decision_model.train_decision_model
```

Khong nen chay:

```bash
python -m src.decision_model.train_decision_model
```
