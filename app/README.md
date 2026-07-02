# AgriFlight DSS — Backend (rebuild)

Backend mới, xây lại theo BRD (bản 1) + tài liệu "Thiết kế CSDL cho Hệ thống DSS Drone".
Đặt song song với `src/` cũ; `src/` sẽ được archive dần khi các hạng mục hoàn tất.

## Kiến trúc 3 lớp (BRD §2)

```
Layer 1 Ingestion → Layer 2 ML Engine → Layer 3 Decision (FLY/DELAY/NO_FLY)
```

## Cấu trúc thư mục

| Thư mục | Vai trò |
|---|---|
| `db/` | `schema.sql` (9 bảng) + `seed.sql` (3 drone, 3 thuốc, 4 giai đoạn lúa) + client Supabase |
| `ingestion/` | Crawl Open-Meteo (lịch sử 5 năm + dự báo live) theo GPS plot, cảm biến soil |
| `features/` | Feature engineering thuần (KHÔNG sinh nhãn — tránh lỗi "ML học lại luật") |
| `rules/` | Ma trận 13 tác nhân (BRD §3.3) + drone limit + growth stage + pesticide + AWD |
| `ml/` | Train RF (Champion) + XGBoost (Challenger), 3 điểm safety/crop/spray |
| `decision/` | Orchestrator gộp luật + ML → FLY/DELAY/NO_FLY + XAI + override |
| `api/` | FastAPI routers |
| `tests/` | Unit + integration |

## Áp database (Supabase)

Chạy trên Supabase SQL Editor (hoặc `psql`), theo thứ tự:

```sql
\i app/db/schema.sql   -- tạo 9 bảng + index
\i app/db/seed.sql     -- nạp drone/pesticide/crop_profile
```

## 9 bảng (schema dung hòa)

`u_profiles` · `crop_profile` · `m_plots` · `drone_profiles` · `pesticide_specs`
· `weather_hourly` · `soil_readings` · `spray_mission_plans` · `flight_decision_log`

Quyết định quyết định (decision taxonomy): **FLY / DELAY / NO_FLY** (thay taxonomy cũ
TAKE_OFF/LOCK_SPRAY/... của `src/`).

## Trạng thái hạng mục

- [x] HM0 — Dựng khung thư mục
- [x] HM1 — `schema.sql` + `seed.sql`
- [x] HM2 — Ingestion (Open-Meteo lịch sử+live theo GPS, +ET0/soil/wind_dir, gán nhãn dùng chung, soil IoT) — parse test offline OK; crawl thật chạy trên máy user
- [x] HM3 — Rules (ma trận 13 tác nhân) — 9/9 test pass
- [x] HM4 — ML Engine (RF/XGB + 3 điểm) — ĐÃ train 262k dòng THẬT 5 năm: RF acc 0.80, XGB acc 0.80
- [x] HM5 — Decision orchestrator (rules+ML → FLY/DELAY/NO_FLY + XAI + override)
- [x] HM6 — API (FastAPI: reference + /decision live Open-Meteo + override + ml/metrics) — 6/6 test pass
- [x] HM7 — FE nối API mới qua router tương thích (app/api/compat.py) — UI chuẩn GIỮ NGUYÊN, đã chạy thật + screenshot OK
