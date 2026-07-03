-- Xóa ràng buộc cũ
ALTER TABLE public.flight_decision_log 
DROP CONSTRAINT IF EXISTS flight_decision_log_system_decision_check;

-- Thêm ràng buộc mới cho phép LOCK_SPRAY
ALTER TABLE public.flight_decision_log
ADD CONSTRAINT flight_decision_log_system_decision_check 
CHECK (system_decision::text = ANY (ARRAY['FLY'::character varying, 'DELAY'::character varying, 'LOCK_SPRAY'::character varying, 'NO_FLY'::character varying]::text[]));
