-- =============================================================================
-- AgriFlight DSS - Physical Database Schema (PostgreSQL / Supabase)
-- =============================================================================
-- Schema "dung hoa" (superset) tu 2 tai lieu:
--   - BRD (ban 1): 8 bang u_profiles..flight_decision_log
--   - Thiet ke CSDL: them crop_profile, mo rong spray_mission_plan,
--                    soil them salinity_ec, log dung 3 diem safety/crop/spray
-- Tong: 9 bang. Chay tren Supabase SQL Editor hoac psql.
-- =============================================================================

SET TIME ZONE 'Asia/Ho_Chi_Minh';

-- -----------------------------------------------------------------------------
-- 1. u_profiles - Nguoi dung (nong dan / phi cong)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS u_profiles (
    user_id       SERIAL PRIMARY KEY,
    full_name     VARCHAR(100) NOT NULL,
    phone_number  VARCHAR(15) UNIQUE NOT NULL,
    user_role     VARCHAR(20) NOT NULL CHECK (user_role IN ('FARMER', 'PILOT')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- 2. crop_profile - Cau hinh 4 giai doan sinh truong lua
--    (nguon: file Thiet ke CSDL bang 4 + BRD §3.2)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS crop_profile (
    crop_stage_id     SERIAL PRIMARY KEY,
    stage_code        VARCHAR(20) UNIQUE NOT NULL
                      CHECK (stage_code IN ('SEEDLING','TILLERING','BOOTING','GRAIN_FILLING')),
    stage_name        VARCHAR(50) NOT NULL,           -- Ma / De nhanh / Lam dong / Chin
    day_from          INT,                            -- ngay bat dau (vd 0)
    day_to            INT,                            -- ngay ket thuc (vd 20)
    kc_value          REAL,                           -- he so cay trong (ETc = kc * ET0)
    opt_flight_alt_min REAL,                          -- tran bay toi uu min (m tren ngon lua)
    opt_flight_alt_max REAL,                          -- tran bay toi uu max (m)
    opt_flight_speed_min REAL,                        -- toc do quet min (m/s)
    opt_flight_speed_max REAL,                        -- toc do quet max (m/s)
    flow_rate_min_l_ha REAL,                          -- luu luong nuoc min (L/ha)
    flow_rate_max_l_ha REAL,                          -- luu luong nuoc max (L/ha)
    awd_threshold_cm   REAL DEFAULT -15.0,            -- nguong tut nuoc AWD (cm)
    hard_ban_start_hour INT,                          -- gio cam bay cung (NULL neu khong)
    hard_ban_end_hour   INT,                          -- vd BOOTING: 8..11
    notes              TEXT
);

-- -----------------------------------------------------------------------------
-- 3. m_plots - Ho so thua ruong (GPS phuc vu query Open-Meteo)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS m_plots (
    plot_id            SERIAL PRIMARY KEY,
    user_id            INT REFERENCES u_profiles(user_id) ON DELETE CASCADE,
    plot_name          VARCHAR(100) NOT NULL,
    area_hectares      REAL NOT NULL CHECK (area_hectares > 0),
    latitude           DOUBLE PRECISION NOT NULL,
    longitude          DOUBLE PRECISION NOT NULL,
    map_image_2d_url   VARCHAR(255),
    current_crop_stage VARCHAR(20)
                       REFERENCES crop_profile(stage_code) ON DELETE RESTRICT,
    current_pesticide  VARCHAR(100)
                       REFERENCES pesticide_specs(active_ingredient) ON DELETE RESTRICT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- 4. drone_profiles - Thong so dong luc hoc phan cung
--    (nguon: BRD §3.3 + file CSDL bang 3). Chot 3 may: T30 / T50 / XAG P100 Pro
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drone_profiles (
    drone_id                SERIAL PRIMARY KEY,
    model_name              VARCHAR(50) UNIQUE NOT NULL,
    max_wind_resistance_kph REAL NOT NULL CHECK (max_wind_resistance_kph > 0),
    max_gust_resistance_kph REAL CHECK (max_gust_resistance_kph > 0),
    tank_capacity_liters    INT NOT NULL CHECK (tank_capacity_liters > 0),
    nozzle_technology       VARCHAR(30) NOT NULL
                            CHECK (nozzle_technology IN ('PRESSURE','CENTRIFUGAL')),
    ingress_protection      VARCHAR(10) NOT NULL,       -- IP67, IPX6K
    mtow_kg                 REAL,                       -- trong luong cat canh toi da
    notes                   TEXT
);

-- -----------------------------------------------------------------------------
-- 5. pesticide_specs - Ho so sinh hoa hoat chat BVTV
--    (nguon: BRD §3.4 + ma tran hoat chat-dang thuoc §3.3 ban 1)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pesticide_specs (
    pesticide_id       SERIAL PRIMARY KEY,
    trade_name         VARCHAR(100) NOT NULL,          -- Beam, Agri-Mek, Anvil...
    active_ingredient  VARCHAR(100) UNIQUE NOT NULL,   -- Tricyclazole/Abamectin/Hexaconazole
    action_mechanism   VARCHAR(30) NOT NULL
                       CHECK (action_mechanism IN ('SYSTEMIC','CONTACT')),
    common_formulation VARCHAR(20),                    -- WP, EC, SC, SG...
    rain_washout_hours INT CHECK (rain_washout_hours >= 0),  -- gio rao la toi thieu
    uv_sensitivity     BOOLEAN NOT NULL DEFAULT FALSE, -- nhay UV (Abamectin=TRUE)
    water_solubility_mg_l REAL,                        -- do tan (mg/L)
    vapor_pressure_mpa  REAL,                          -- ap suat hoi (mPa)
    notes              TEXT
);

-- -----------------------------------------------------------------------------
-- 6. weather_hourly - Nhat ky khi tuong Open-Meteo (theo plot + gio)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weather_hourly (
    weather_id                 BIGSERIAL PRIMARY KEY,
    plot_id                    INT REFERENCES m_plots(plot_id) ON DELETE CASCADE,
    timestamp                  TIMESTAMPTZ NOT NULL,
    temperature_2m             REAL NOT NULL,
    relative_humidity_2m       REAL NOT NULL CHECK (relative_humidity_2m BETWEEN 0 AND 100),
    wind_speed_10m             REAL NOT NULL CHECK (wind_speed_10m >= 0),
    wind_gusts_10m             REAL NOT NULL CHECK (wind_gusts_10m >= 0),
    wind_direction_10m         REAL CHECK (wind_direction_10m BETWEEN 0 AND 360),
    precipitation              REAL NOT NULL CHECK (precipitation >= 0),         -- mm tich luy
    precipitation_probability  REAL NOT NULL CHECK (precipitation_probability BETWEEN 0 AND 100),
    cloud_cover                INT NOT NULL CHECK (cloud_cover BETWEEN 0 AND 100),
    visibility                 REAL CHECK (visibility >= 0),                     -- m
    weather_code               INT,                                             -- WMO
    et0_fao_evapotranspiration REAL CHECK (et0_fao_evapotranspiration >= 0),     -- mm/ngay
    source                     VARCHAR(30) DEFAULT 'open-meteo',
    UNIQUE (plot_id, timestamp)
);

-- -----------------------------------------------------------------------------
-- 7. soil_readings - Cam bien IoT thuc dia (AWD)  [them salinity_ec tu file CSDL]
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS soil_readings (
    reading_id       BIGSERIAL PRIMARY KEY,
    plot_id          INT REFERENCES m_plots(plot_id) ON DELETE CASCADE,
    timestamp        TIMESTAMPTZ NOT NULL,
    soil_moisture_percentage REAL NOT NULL CHECK (soil_moisture_percentage BETWEEN 0 AND 100),
    water_level_cm   REAL NOT NULL,                    -- muc nuoc so voi mat ruong (cm, am = tut)
    salinity_ec      REAL,                             -- do dan dien (xam nhap man)
    UNIQUE (plot_id, timestamp)
);

-- -----------------------------------------------------------------------------
-- 8. spray_mission_plans - Ke hoach bay/phun  [mo rong theo file CSDL bang 5]
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS spray_mission_plans (
    mission_id           BIGSERIAL PRIMARY KEY,
    plot_id              INT REFERENCES m_plots(plot_id) ON DELETE CASCADE,
    drone_id             INT REFERENCES drone_profiles(drone_id) ON DELETE RESTRICT,
    pesticide_id         INT REFERENCES pesticide_specs(pesticide_id) ON DELETE RESTRICT,
    scheduled_start_time TIMESTAMPTZ NOT NULL,
    target_dosage_l_per_ha REAL CHECK (target_dosage_l_per_ha > 0),
    droplet_size_um      INT,                          -- kich thuoc giot thiet lap (µm)
    formulation_type     VARCHAR(20),                  -- SC/EC/WP/SG (danh gia tac nghen bec)
    water_ph             REAL,                          -- pH nuoc pha (khuyen nghi 5.5-6.5)
    adjuvant_used        BOOLEAN NOT NULL DEFAULT FALSE,
    nozzle_mode          VARCHAR(30),                  -- min / trung binh / tho / ly tam
    canopy_density       VARCHAR(20)                   -- thua / trung binh / day
                         CHECK (canopy_density IS NULL OR canopy_density IN ('SPARSE','MEDIUM','DENSE')),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- 9. flight_decision_log - "Hop den" + Override (luu ca 5 diem)
--    Dung hoa: BRD (rf/xgb) + file CSDL (flight_safety/crop_impact/spray_quality)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS flight_decision_log (
    log_id              BIGSERIAL PRIMARY KEY,
    mission_id          BIGINT REFERENCES spray_mission_plans(mission_id) ON DELETE CASCADE,
    weather_id          BIGINT REFERENCES weather_hourly(weather_id) ON DELETE RESTRICT,
    drone_id            INT REFERENCES drone_profiles(drone_id) ON DELETE RESTRICT,
    rf_score_safety     REAL CHECK (rf_score_safety BETWEEN 0 AND 100),   -- Champion (RF)
    xgb_score_safety    REAL CHECK (xgb_score_safety BETWEEN 0 AND 100),  -- Challenger (XGB)
    flight_safety_score REAL CHECK (flight_safety_score BETWEEN 0 AND 100), -- consensus
    crop_impact_score   REAL CHECK (crop_impact_score BETWEEN 0 AND 100),
    spray_quality_score REAL CHECK (spray_quality_score BETWEEN 0 AND 100),
    system_decision     VARCHAR(20) NOT NULL
                        CHECK (system_decision IN ('FLY','DELAY','NO_FLY')),
    is_user_overridden  BOOLEAN NOT NULL DEFAULT FALSE,
    override_reason     TEXT,
    xai_explanation     TEXT,                           -- giai thich quyet dinh cho nong dan
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- Indexes toi uu truy van chuoi thoi gian
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_weather_plot_ts   ON weather_hourly(plot_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_soil_plot_ts       ON soil_readings(plot_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_decision_mission    ON flight_decision_log(mission_id);
CREATE INDEX IF NOT EXISTS idx_decision_overridden ON flight_decision_log(is_user_overridden) WHERE is_user_overridden;

-- ============ SEED ============

-- =============================================================================
-- AgriFlight DSS - Seed data (drone / pesticide / crop stages)
-- Chay SAU schema.sql. Cac gia tri nguong lay tu BRD §3.2-3.4 va file Thiet ke CSDL.
-- Cac cot nguong deu tunable qua module decision-config sau nay.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- crop_profile: 4 giai doan sinh truong lua (BRD §3.2)
-- -----------------------------------------------------------------------------
INSERT INTO crop_profile
    (stage_code, stage_name, day_from, day_to, kc_value,
     opt_flight_alt_min, opt_flight_alt_max, opt_flight_speed_min, opt_flight_speed_max,
     flow_rate_min_l_ha, flow_rate_max_l_ha, awd_threshold_cm,
     hard_ban_start_hour, hard_ban_end_hour, notes)
VALUES
    ('SEEDLING',     'Ma',            0,  20, 1.05, 2.5, 3.0, 6.0, 7.0, 10.0, 15.0, -15.0, NULL, NULL,
     'Tan thua, mat nuoc lo. Bay cao & nhanh de tranh downwash dap ma.'),
    ('TILLERING',    'De nhanh',      20, 40, 1.10, 2.0, 2.5, 5.0, 6.0, 15.0, 20.0, -15.0, NULL, NULL,
     'Bat dau khep tan, nhu cau nuoc tang. Bung phat dao on so ky.'),
    ('BOOTING',      'Lam dong-Tro',  40, 70, 1.20, 1.5, 2.0, 4.0, 5.0, 25.0, 30.0,  -5.0, 8, 11,
     'Tan day. CAM BAY 08-11h (lua phoi mau thu phan). Giu nuoc ngap 5cm.'),
    ('GRAIN_FILLING','Chin',          70, 100, 0.95, 2.5, 3.5, 5.0, 6.0, 15.0, 20.0, -15.0, NULL, NULL,
     'Bong nang, canh bao do nga (lodging) va rung hat co hoc. Bay cao han che gio de bong.')
ON CONFLICT (stage_code) DO NOTHING;

-- -----------------------------------------------------------------------------
-- drone_profiles: chot 3 may = DJI T30 / DJI T50 / XAG P100 Pro
--   max_wind_resistance_kph: nguong cam bay CUNG cua DSS theo tung may
--     - T30: 28.8 km/h (baseline DSS, sach HD DJI cho toi ~28 km/h)
--     - T50: 21.6 km/h (danh dinh 6 m/s; DSS giam bien do vi tai nang)
--     - XAG P100 Pro: 36 km/h (10 m/s, bien an toan cao)
--   max_gust_resistance_kph: nguong gio giat (uoc tinh ~1.25x, tunable)
-- -----------------------------------------------------------------------------
INSERT INTO drone_profiles
    (model_name, max_wind_resistance_kph, max_gust_resistance_kph,
     tank_capacity_liters, nozzle_technology, ingress_protection, mtow_kg, notes)
VALUES
    ('DJI_T30',      28.8, 36.0, 30, 'PRESSURE',    'IP67',  78.0,
     'Baseline ho nong dan nho. Voi ap luc, khang nuoc IP67 toan than.'),
    ('DJI_T50',      21.6, 30.0, 40, 'CENTRIFUGAL', 'IPX6K', 103.0,
     'Canh dong lon/HTX. Canh quat kep dong truc, voi ly tam kep 16-24 L/phut.'),
    ('XAG_P100_PRO', 36.0, 46.0, 50, 'CENTRIFUGAL', 'IPX6K', 110.0,
     'Uu tien vung ven bien nhieu gio. Voi RevoSpray chinh kich thuoc giot linh hoat.')
ON CONFLICT (model_name) DO NOTHING;

-- -----------------------------------------------------------------------------
-- pesticide_specs: 3 hoat chat pho bien ĐBSCL (BRD §3.4 + ma tran §3.3)
-- -----------------------------------------------------------------------------
INSERT INTO pesticide_specs
    (trade_name, active_ingredient, action_mechanism, common_formulation,
     rain_washout_hours, uv_sensitivity, water_solubility_mg_l, vapor_pressure_mpa, notes)
VALUES
    ('Beam',    'Tricyclazole', 'SYSTEMIC', 'WP', 4, FALSE, 596.0,  0.027,
     'Tri dao on. Luu dan, it bay hoi. Can hoa tan/loc can, la kho & khong mua 2-4h sau phun.'),
    ('Agri-Mek','Abamectin',    'CONTACT',  'EC', 2, TRUE,  0.020,  0.0037,
     'Tru sau cuon la sinh hoc. Nhay UV, cam phun 10-15h khi nang nong, RH thap.'),
    ('Anvil',   'Hexaconazole', 'SYSTEMIC', 'SC', 3, FALSE, 18.0,   0.018,
     'Tri kho van goc be la. Uu tien giot trung binh-tho, bay thap, luu luong nuoc cao.')
ON CONFLICT (active_ingredient) DO NOTHING;
