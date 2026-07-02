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
