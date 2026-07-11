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
