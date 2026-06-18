-- ============================================
-- DSS301 Agricultural Drone Scheduler
-- Database Migration: Create Tables
-- Run on Supabase SQL Editor
-- ============================================

-- 1. Bảng raw_weather_data (giống DB cũ)
CREATE TABLE IF NOT EXISTS raw_weather_data (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    location_name TEXT NOT NULL,
    latitude      DOUBLE PRECISION,
    longitude     DOUBLE PRECISION,
    timestamp     TIMESTAMPTZ NOT NULL,
    wind_speed    DOUBLE PRECISION,
    wind_gusts    DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    precipitation_probability DOUBLE PRECISION,
    cloud_cover   DOUBLE PRECISION,
    visibility    DOUBLE PRECISION,
    temperature   DOUBLE PRECISION,
    humidity      DOUBLE PRECISION,
    weather_code  INTEGER,
    weather_description TEXT,
    flyability_score DOUBLE PRECISION,
    fly_label     TEXT,
    risk_level    TEXT,
    UNIQUE (location_name, timestamp)
);

-- 2. Bảng drone_flight_logs
CREATE TABLE IF NOT EXISTS drone_flight_logs (
    id                    UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    location_name         TEXT NOT NULL,
    flight_timestamp      TIMESTAMPTZ NOT NULL,
    decision_action       TEXT NOT NULL,
    risk_level            TEXT,
    flyability_score      DOUBLE PRECISION,
    dynamic_flow_rate_pct DOUBLE PRECISION,
    crop_condition        TEXT,
    recommendation_text   TEXT,
    weather_source        TEXT DEFAULT 'api',
    override_id           UUID,
    created_at            TIMESTAMPTZ DEFAULT now(),
    UNIQUE (location_name, flight_timestamp)
);

-- 3. Bảng analyzed_weather_data
CREATE TABLE IF NOT EXISTS analyzed_weather_data (
    id                        UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    location_name             TEXT NOT NULL,
    timestamp                 TIMESTAMPTZ NOT NULL,
    temperature_2m            DOUBLE PRECISION,
    relative_humidity_2m      DOUBLE PRECISION,
    precipitation_probability DOUBLE PRECISION,
    precipitation             DOUBLE PRECISION,
    cloud_cover               DOUBLE PRECISION,
    visibility                DOUBLE PRECISION,
    wind_speed_10m            DOUBLE PRECISION,
    wind_gusts_10m            DOUBLE PRECISION,
    weather_code              INTEGER,
    weather_description       TEXT,
    flyability_score          DOUBLE PRECISION,
    decision_action           TEXT,
    risk_level                TEXT,
    source                    TEXT DEFAULT 'WeatherAPI',
    created_at                TIMESTAMPTZ DEFAULT now(),
    UNIQUE (location_name, timestamp)
);

-- 4. Bảng weather_overrides
CREATE TABLE IF NOT EXISTS weather_overrides (
    id                          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    location_name               TEXT NOT NULL,
    override_timestamp          TIMESTAMPTZ NOT NULL,
    image_filename              TEXT,
    image_url                   TEXT,
    ai_suggested_condition      TEXT,
    ai_confidence               DOUBLE PRECISION,
    user_final_condition        TEXT NOT NULL,
    user_accepted_ai            BOOLEAN DEFAULT false,
    user_notes                  TEXT,
    original_api_weather_code   INTEGER,
    original_api_description    TEXT,
    override_weather_code       INTEGER,
    override_description        TEXT,
    previous_decision           TEXT,
    previous_flyability         DOUBLE PRECISION,
    new_decision                TEXT,
    new_flyability              DOUBLE PRECISION,
    created_at                  TIMESTAMPTZ DEFAULT now()
);

-- 5. Foreign Key
ALTER TABLE drone_flight_logs
    ADD CONSTRAINT fk_override
    FOREIGN KEY (override_id) REFERENCES weather_overrides(id)
    ON DELETE SET NULL;

-- 6. Indexes
CREATE INDEX idx_flight_logs_location ON drone_flight_logs(location_name);
CREATE INDEX idx_flight_logs_timestamp ON drone_flight_logs(flight_timestamp);
CREATE INDEX idx_analyzed_weather_location ON analyzed_weather_data(location_name);
CREATE INDEX idx_analyzed_weather_timestamp ON analyzed_weather_data(timestamp);
CREATE INDEX idx_overrides_location ON weather_overrides(location_name);
CREATE INDEX idx_overrides_timestamp ON weather_overrides(override_timestamp);
