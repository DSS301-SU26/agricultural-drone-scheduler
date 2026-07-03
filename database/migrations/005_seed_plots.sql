-- 1. Insert Nông dân / Phi công
INSERT INTO public.u_profiles (user_id, full_name, phone_number, user_role)
VALUES 
(1, 'Nguyen Van A', '0901234567', 'FARMER'),
(2, 'Tran Van B', '0912345678', 'PILOT')
ON CONFLICT (phone_number) DO NOTHING;

-- 2. Insert Mảnh vườn mẫu (Chỉ lấy ở 5 tỉnh có dữ liệu thời tiết: An Giang, Can Tho, Dong Thap, Long An, Tien Giang)
INSERT INTO public.m_plots (plot_id, user_id, plot_name, area_hectares, latitude, longitude, current_crop_stage)
VALUES 
-- Nông dân 1 (Nguyen Van A)
(1, 1, 'Vuon Lua Tien Giang 1', 2.5, 10.25, 106.33, 'SEEDLING'),
(2, 1, 'Vuon Lua Tien Giang 2', 4.0, 10.35, 106.20, 'TILLERING'),
(3, 1, 'Vuon Lua Dong Thap 1', 5.0, 10.45, 105.63, 'TILLERING'),
(4, 1, 'Vuon Lua Dong Thap 2', 3.0, 10.50, 105.65, 'BOOTING'),
(5, 1, 'Vuon Lua An Giang 1', 10.0, 10.38, 105.42, 'BOOTING'),
(6, 1, 'Vuon Lua An Giang 2', 8.5, 10.40, 105.35, 'GRAIN_FILLING'),

-- Nông dân 2 (Tran Van B - Ngoài vai trò phi công còn làm chủ vườn)
(7, 2, 'Vuon Lua Can Tho 1', 1.5, 10.03, 105.78, 'GRAIN_FILLING'),
(8, 2, 'Vuon Lua Can Tho 2', 2.2, 10.10, 105.70, 'SEEDLING'),
(9, 2, 'Vuon Lua Long An 1', 6.0, 10.53, 106.38, 'TILLERING'),
(10, 2, 'Vuon Lua Long An 2', 12.0, 10.60, 106.25, 'BOOTING'),
(11, 2, 'Vuon Mau Dong Thap', 15.0, 10.42, 105.68, 'SEEDLING'),
(12, 2, 'Ruong Thu Nghiem An Giang', 2.8, 10.39, 105.45, 'GRAIN_FILLING'),
(13, 2, 'Nong Trai Can Tho', 5.5, 10.05, 105.75, 'BOOTING'),
(14, 2, 'Vuon Rau Cu Chi - Ho Chi Minh', 4.5, 10.95, 106.50, 'TILLERING'),
(15, 2, 'Nong Trai Gia Lam - Ha Noi', 3.5, 21.02, 105.90, 'GRAIN_FILLING')
ON CONFLICT (plot_id) DO NOTHING;

-- Chú ý cập nhật lại Sequence nếu cần thiết
SELECT setval('u_profiles_user_id_seq', (SELECT MAX(user_id) FROM u_profiles));
SELECT setval('m_plots_plot_id_seq', (SELECT MAX(plot_id) FROM m_plots));
