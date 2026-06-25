-- ============================================
-- AgriFlight DSS MVP
-- Database Migration: Create flight_decisions_log Table
-- Run on Supabase SQL Editor
-- ============================================

CREATE TABLE IF NOT EXISTS flight_decisions_log (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    weather_snapshot JSONB NOT NULL,
    champion_pred TEXT NOT NULL,
    champion_conf DOUBLE PRECISION NOT NULL,
    challenger_pred TEXT NOT NULL,
    challenger_conf DOUBLE PRECISION NOT NULL,
    final_decision TEXT NOT NULL,
    was_conflict BOOLEAN DEFAULT false,
    was_human_overridden BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Index for querying timestamp
CREATE INDEX IF NOT EXISTS idx_flight_decisions_log_timestamp ON flight_decisions_log(timestamp);
