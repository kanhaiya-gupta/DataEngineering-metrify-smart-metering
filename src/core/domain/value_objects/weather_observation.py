"""
Weather Observation Value Object
Represents a single weather observation
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class WeatherObservation:
    """A single weather observation"""
    timestamp: datetime
    temperature_celsius: float
    humidity_percent: float
    pressure_hpa: float
    wind_speed_ms: float
    wind_direction_degrees: float
    cloud_cover_percent: float
    visibility_km: float
    uv_index: Optional[float] = None
    precipitation_mm: Optional[float] = None
    data_quality_score: float = 1.0
    
    def __post_init__(self):
        """Validate weather observation data"""
        if not (-50 <= self.temperature_celsius <= 60):
            raise ValueError("Temperature must be between -50°C and 60°C")
        if not (0 <= self.humidity_percent <= 100):
            raise ValueError("Humidity must be between 0% and 100%")
        if not (950 <= self.pressure_hpa <= 1050):
            raise ValueError("Pressure must be between 950hPa and 1050hPa")
        if not (0 <= self.wind_speed_ms <= 100):
            raise ValueError("Wind speed must be between 0m/s and 100m/s")
        if not (0 <= self.wind_direction_degrees <= 360):
            raise ValueError("Wind direction must be between 0° and 360°")
        if not (0 <= self.cloud_cover_percent <= 100):
            raise ValueError("Cloud cover must be between 0% and 100%")
        if not (0 <= self.visibility_km <= 50):
            raise ValueError("Visibility must be between 0km and 50km")
        if self.uv_index is not None and not (0 <= self.uv_index <= 15):
            raise ValueError("UV index must be between 0 and 15")
        if self.precipitation_mm is not None and self.precipitation_mm < 0:
            raise ValueError("Precipitation cannot be negative")
        if not (0 <= self.data_quality_score <= 1):
            raise ValueError("Data quality score must be between 0 and 1")
