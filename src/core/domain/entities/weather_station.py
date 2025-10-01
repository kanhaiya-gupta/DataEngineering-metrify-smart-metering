"""
Weather Station Domain Entity
Represents a weather station for environmental data collection
"""

from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from ..value_objects.location import Location
from ..enums.weather_station_status import WeatherStationStatus
from ..events.weather_events import WeatherDataRecordedEvent, WeatherAnomalyDetectedEvent


class WeatherStationType(Enum):
    """Types of weather stations"""
    AUTOMATIC = "automatic"      # Automated weather station
    MANUAL = "manual"           # Manual observation station
    MOBILE = "mobile"           # Mobile weather station
    SATELLITE = "satellite"     # Satellite-based data


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


@dataclass
class WeatherStation:
    """
    Weather Station Domain Entity
    
    Represents a weather station that collects environmental data
    for energy demand correlation analysis
    """
    
    # Identity
    station_id: str
    name: str
    station_type: WeatherStationType
    
    # Location
    location: Location
    
    # Contact information
    operator: str
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    
    # Status and metadata
    status: WeatherStationStatus = WeatherStationStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Performance metrics
    total_observations: int = 0
    average_quality_score: float = 1.0
    last_observation_at: Optional[datetime] = None
    
    # Events
    _domain_events: List = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """Validate entity after initialization"""
        if not self.station_id:
            raise ValueError("Station ID is required")
        if not self.name:
            raise ValueError("Station name is required")
        if not self.location:
            raise ValueError("Location is required")
        if not self.operator:
            raise ValueError("Operator is required")
    
    @property
    def domain_events(self) -> List:
        """Get domain events"""
        return self._domain_events.copy()
    
    def clear_domain_events(self) -> None:
        """Clear domain events after they've been processed"""
        self._domain_events.clear()
    
    def add_domain_event(self, event) -> None:
        """Add a domain event"""
        self._domain_events.append(event)
    
    def record_observation(self, observation: WeatherObservation) -> None:
        """Record a new weather observation"""
        if self.status != WeatherStationStatus.ACTIVE:
            raise ValueError("Cannot record observations for inactive station")
        
        # Update observation statistics
        self.total_observations += 1
        self.last_observation_at = observation.timestamp
        self.updated_at = datetime.utcnow()
        
        # Update quality metrics
        self._update_quality_metrics(observation)
        
        # Check for anomalies
        self._check_for_anomalies(observation)
        
        # Add domain event
        self.add_domain_event(
            WeatherDataRecordedEvent(
                station_id=self.station_id,
                observation=observation,
                recorded_at=self.updated_at
            )
        )
    
    def _update_quality_metrics(self, observation: WeatherObservation) -> None:
        """Update quality metrics based on new observation"""
        if self.total_observations == 1:
            self.average_quality_score = observation.data_quality_score
        else:
            # Weighted average giving more weight to recent observations
            weight = min(0.1, 1.0 / self.total_observations)
            self.average_quality_score = (
                (1 - weight) * self.average_quality_score + 
                weight * observation.data_quality_score
            )
    
    def _check_for_anomalies(self, observation: WeatherObservation) -> None:
        """Check for anomalies in the weather observation"""
        anomalies = []
        
        # Check for temperature anomalies
        if observation.temperature_celsius < -30 or observation.temperature_celsius > 50:
            anomalies.append(f"Temperature anomaly: {observation.temperature_celsius}°C")
        
        # Check for humidity anomalies
        if observation.humidity_percent > 95:
            anomalies.append(f"High humidity: {observation.humidity_percent}%")
        
        # Check for pressure anomalies
        if observation.pressure_hpa < 980 or observation.pressure_hpa > 1030:
            anomalies.append(f"Pressure anomaly: {observation.pressure_hpa}hPa")
        
        # Check for wind speed anomalies
        if observation.wind_speed_ms > 30:
            anomalies.append(f"High wind speed: {observation.wind_speed_ms}m/s")
        
        # If anomalies detected, raise event
        if anomalies:
            self.add_domain_event(
                WeatherAnomalyDetectedEvent(
                    station_id=self.station_id,
                    anomaly_type="weather_anomaly",
                    anomaly_description="; ".join(anomalies),
                    detected_at=datetime.utcnow(),
                    severity="high" if len(anomalies) > 2 else "medium",
                    observation_data=observation.__dict__
                )
            )
    
    def deactivate_station(self, reason: str) -> None:
        """Deactivate the weather station"""
        if self.status == WeatherStationStatus.INACTIVE:
            raise ValueError("Station is already inactive")
        
        self.status = WeatherStationStatus.INACTIVE
        self.updated_at = datetime.utcnow()
    
    def reactivate_station(self) -> None:
        """Reactivate the weather station"""
        if self.status != WeatherStationStatus.INACTIVE:
            raise ValueError("Only inactive stations can be reactivated")
        
        self.status = WeatherStationStatus.ACTIVE
        self.updated_at = datetime.utcnow()
    
    def calculate_energy_demand_factor(self, observation: WeatherObservation) -> float:
        """Calculate energy demand factor based on weather conditions"""
        factor = 1.0
        
        # Temperature effect (heating/cooling demand)
        temp = observation.temperature_celsius
        if temp < 15:  # Heating demand
            factor += (15 - temp) * 0.02
        elif temp > 25:  # Cooling demand
            factor += (temp - 25) * 0.03
        
        # Humidity effect
        if observation.humidity_percent > 80:
            factor += 0.1  # Higher humidity increases energy demand
        
        # Wind effect (wind chill/heat index)
        if observation.wind_speed_ms > 10:
            if temp < 10:
                factor += 0.05  # Wind chill increases heating demand
            elif temp > 25:
                factor += 0.03  # Wind can reduce cooling demand
        
        # Cloud cover effect (affects solar generation)
        if observation.cloud_cover_percent > 70:
            factor += 0.05  # Less solar generation, more grid demand
        
        return max(0.5, min(2.0, factor))  # Clamp between 0.5 and 2.0
    
    def get_weather_category(self, observation: WeatherObservation) -> str:
        """Get weather category based on observation"""
        temp = observation.temperature_celsius
        
        if temp < 0:
            return "freezing"
        elif temp < 10:
            return "cold"
        elif temp < 20:
            return "cool"
        elif temp < 30:
            return "warm"
        else:
            return "hot"
    
    def is_operational(self) -> bool:
        """Check if the station is operational"""
        return (self.status == WeatherStationStatus.ACTIVE and 
                self.last_observation_at is not None)
    
    def requires_attention(self) -> bool:
        """Check if the station requires attention"""
        if not self.is_operational():
            return True
        
        # Check if no observations in the last 2 hours
        if self.last_observation_at:
            time_since_last = datetime.utcnow() - self.last_observation_at
            if time_since_last.total_seconds() > 7200:  # 2 hours
                return True
        
        return False
    
    def __str__(self) -> str:
        """String representation"""
        return f"WeatherStation(id={self.station_id}, name={self.name}, status={self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"WeatherStation("
            f"id={self.station_id}, "
            f"name={self.name}, "
            f"type={self.station_type.value}, "
            f"status={self.status.value}, "
            f"observations={self.total_observations}"
            f")"
        )
