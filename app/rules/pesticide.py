"""
Luat sinh hoa thuoc BVTV (BRD §3.3 ma tran hoat chat + §3.4).

- Tricyclazole (SYSTEMIC, WP): can la kho, khong mua 2-4h sau phun.
- Abamectin (CONTACT, EC, nhay UV): cam 10-15h khi nang nong; uu tien chieu mat
  hoac may che phu 80-100%.
- Hexaconazole (SC): giot trung binh-tho, bay thap, luu luong nuoc cao.
"""
from __future__ import annotations

from .context import PesticideSpec
from .thresholds import DEFAULT_THRESHOLDS, WeatherThresholds
from .types import FactorResult, Verdict


def evaluate_pesticide_timing(
    hour: int,
    temperature_c: float,
    cloud_cover_pct: float,
    pesticide: PesticideSpec | None,
    thresholds: WeatherThresholds = DEFAULT_THRESHOLDS,
) -> FactorResult | None:
    """Danh gia rui ro thoi diem phun theo dac tinh thuoc.
    Tra None neu khong chon thuoc."""
    if pesticide is None:
        return None

    in_uv_window = thresholds.uv_ban_start_hour <= hour < thresholds.uv_ban_end_hour
    cloudy_cover = cloud_cover_pct >= thresholds.cloud_bio_golden_min

    # Thuoc nhay UV (Abamectin): cam khung gio nang gat neu troi khong du may
    if pesticide.uv_sensitivity and in_uv_window and not cloudy_cover:
        if temperature_c > thresholds.temp_abamectin_sun_stop:
            return FactorResult(
                factor="pesticide_uv_timing",
                verdict=Verdict.STOP_SPRAY,
                value=hour,
                message=(
                    f"Hoạt chất {pesticide.active_ingredient} nhạy cảm UV: Cấm phun lúc {hour}h do nắng gắt "
                    f"(Nhiệt độ {temperature_c:.1f}°C, Mây che {cloud_cover_pct:.0f}%). Hãy dời sang chiều mát."
                ),
                is_hard=False,
            )
        return FactorResult(
            factor="pesticide_uv_timing",
            verdict=Verdict.WARN,
            value=hour,
            message=(
                f"Hoạt chất {pesticide.active_ingredient} nhạy UV: Đang trong khung giờ nắng gắt "
                f"({thresholds.uv_ban_start_hour}-{thresholds.uv_ban_end_hour}h). Nên dời sang chiều để bảo toàn thuốc."
            ),
            is_hard=False,
        )

    return FactorResult(
        factor="pesticide_uv_timing",
        verdict=Verdict.ALLOW,
        value=hour,
        message=f"Thoi diem phun phu hop voi {pesticide.active_ingredient}.",
        is_hard=False,
    )


def evaluate_rain_washout(
    rain_prob_next_hours_pct: float,
    pesticide: PesticideSpec | None,
    thresholds: WeatherThresholds = DEFAULT_THRESHOLDS,
) -> FactorResult | None:
    """Canh bao rua troi: thuoc can rao la nhung du bao co mua trong khoang washout.
    `rain_prob_next_hours_pct` = xac suat mua cao nhat trong cua so rain_washout_hours toi."""
    if pesticide is None:
        return None

    if rain_prob_next_hours_pct > thresholds.rain_prob_stop_pct:
        return FactorResult(
            factor="pesticide_rain_washout",
            verdict=Verdict.STOP_SPRAY,
            value=round(rain_prob_next_hours_pct, 0),
            message=(
                f"Thuốc {pesticide.active_ingredient} cần {pesticide.rain_washout_hours} giờ để ráo lá, "
                f"nhưng xác suất mưa lên tới {rain_prob_next_hours_pct:.0f}%: Nguy cơ cao bị mưa rửa trôi hoàn toàn. BẮT BUỘC HOÃN PHUN."
            ),
            is_hard=False,
        )
    if rain_prob_next_hours_pct > thresholds.rain_prob_warn_pct:
        return FactorResult(
            factor="pesticide_rain_washout",
            verdict=Verdict.WARN,
            value=round(rain_prob_next_hours_pct, 0),
            message=(
                f"Xác suất mưa {rain_prob_next_hours_pct:.0f}% trong cửa sổ ráo lá "
                f"{pesticide.rain_washout_hours} giờ: Cần theo dõi sát thời tiết."
            ),
            is_hard=False,
        )
    return FactorResult(
        factor="pesticide_rain_washout",
        verdict=Verdict.ALLOW,
        value=round(rain_prob_next_hours_pct, 0),
        message=f"Trời khô ráo trong cửa sổ {pesticide.rain_washout_hours} giờ, thuốc bám tốt.",
        is_hard=False,
    )


def recommend_nozzle_and_water(
    pesticide: PesticideSpec | None,
    wind_speed_kph: float,
    canopy_density: str | None = None,
) -> dict[str, str | float]:
    """Goi y cau hinh voi/giot va luu luong nuoc theo hoat chat + gio + tan la.
    (BRD §3.3: Hexaconazole -> giot tho, nuoc cao; gio manh -> tang kich thuoc giot)."""
    # Gio manh -> giot tho de chong tan xa (RevoSpray / ly tam giam toc)
    if wind_speed_kph > 18.0:
        droplet = "COARSE"        # >250 µm
    elif wind_speed_kph > 10.8:
        droplet = "MEDIUM_COARSE"
    else:
        droplet = "MEDIUM"

    water_l_ha = 20.0
    if pesticide and pesticide.active_ingredient == "Hexaconazole":
        droplet = "COARSE" if droplet == "MEDIUM" else droplet
        water_l_ha = 28.0         # can nhieu nuoc de xuong be/goc
    if canopy_density == "DENSE":
        water_l_ha = max(water_l_ha, 28.0)

    return {"droplet_class": droplet, "water_volume_l_ha": water_l_ha}
