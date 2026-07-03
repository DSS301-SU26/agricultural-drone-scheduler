-- =============================================================================
-- Migration 002: mo rong flight_decision_log de lam "hop den" tu mo ta (P1 #5)
-- Chay tren Supabase SQL Editor SAU setup_all.sql.
-- =============================================================================

ALTER TABLE flight_decision_log
    ADD COLUMN IF NOT EXISTS location_name  TEXT,
    ADD COLUMN IF NOT EXISTS slot_timestamp TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS weather_json   JSONB;

-- Chong trung: 1 dong / (location, khung gio) -> FE refresh chi UPDATE, khong nhan doi.
-- (NULL location/timestamp -> NULL la distinct nen cac dong override rieng le van cho phep.)
-- LUU Y: KHONG dung partial index (WHERE ...) vi Postgres ON CONFLICT khong dung duoc
-- partial index qua PostgREST upsert. Dung unique index THUONG.
DROP INDEX IF EXISTS uq_decision_log_loc_slot;
CREATE UNIQUE INDEX IF NOT EXISTS uq_decision_log_loc_slot
    ON flight_decision_log(location_name, slot_timestamp);

-- Truy van nhat ky theo thoi gian
CREATE INDEX IF NOT EXISTS idx_decision_log_created ON flight_decision_log(created_at DESC);
