-- ============================================
-- AgriFlight DSS MVP
-- Database Migration: Add Probabilistic ML and Distance columns to flight_decisions_log
-- Run on Supabase SQL Editor
-- ============================================

-- 1. Add new columns to flight_decisions_log
ALTER TABLE flight_decisions_log 
ADD COLUMN IF NOT EXISTS champion_score DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS challenger_score DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS flyability_score DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS distance_to_field_km DOUBLE PRECISION DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS is_safe_to_fly BOOLEAN DEFAULT false;

-- 2. Add comments to deprecate old columns
COMMENT ON COLUMN flight_decisions_log.champion_pred IS 'Deprecated: Use champion_score instead';
COMMENT ON COLUMN flight_decisions_log.challenger_pred IS 'Deprecated: Use challenger_score instead';
COMMENT ON COLUMN flight_decisions_log.final_decision IS 'Deprecated: Use is_safe_to_fly instead';
