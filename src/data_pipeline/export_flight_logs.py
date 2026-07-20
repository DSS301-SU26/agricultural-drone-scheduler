import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "reports" / "decision_log.json"
OUT_PATH = ROOT / "data" / "retrain_dataset.csv"

def export_logs():
    if not LOG_PATH.exists():
        print(f"File {LOG_PATH} không tồn tại.")
        return

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for key, val in data.items():
        # Lấy thông tin cơ bản
        row = {
            "location_name": val.get("location_name"),
            "timestamp": val.get("slot_timestamp"),
            "system_decision": val.get("system_decision"),
            "is_user_overridden": val.get("is_user_overridden"),
            "feedback_status": val.get("feedback_status") # Only present for FB entries
        }
        
        # Lấy thời tiết (làm phẳng từ dictionary weather_json)
        weather = val.get("weather_json", {})
        for w_k, w_v in weather.items():
            row[w_k] = w_v
            
        rows.append(row)

    if not rows:
        print("Không có bản ghi nào để export.")
        return

    df = pd.DataFrame(rows)
    # Lọc những record có đủ dữ liệu thời tiết quan trọng (ví dụ: nhiệt độ, lượng mưa)
    if "temperature_2m" in df.columns:
        df = df.dropna(subset=["temperature_2m", "system_decision"])
        
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Đã export {len(df)} bản ghi ra file: {OUT_PATH}")

if __name__ == "__main__":
    export_logs()
