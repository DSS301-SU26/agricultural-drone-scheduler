"""Tang Ingestion (Layer 1) - Open-Meteo (lich su + du bao) + soil IoT."""
from .locations import DELTA_LOCATIONS
from .open_meteo import fetch_forecast, fetch_historical, parse_hourly
from .soil import latest_water_level, simulate_soil_series

__all__ = [
    "DELTA_LOCATIONS",
    "fetch_forecast", "fetch_historical", "parse_hourly",
    "latest_water_level", "simulate_soil_series",
]
