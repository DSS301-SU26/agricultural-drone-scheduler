-- Migration 003: Add current_pesticide to m_plots
-- Connect a fixed pesticide to each field (plot) similar to current_crop_stage.

SET TIME ZONE 'Asia/Ho_Chi_Minh';

ALTER TABLE public.m_plots 
ADD COLUMN current_pesticide VARCHAR(100) REFERENCES public.pesticide_specs(active_ingredient) ON DELETE RESTRICT;
