"""
Weather Station Domain Events
Events that occur in the weather station domain
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from ..value_objects.weather_observation import WeatherObservation


class EventType(Enum):
    """Types of domain events"""
    WEATHER_DATA_RECORDED = "weather_data_recorded"
    WEATHER_ANOMALY_DETECTED = "weather_anomaly_detected"
    WEATHER_STATION_ACTIVATED = "weather_station_activated"
    WEATHER_STATION_DEACTIVATED = "weather_station_deactivated"
    WEATHER_STATION_MAINTENANCE = "weather_station_maintenance"
    WEATHER_ALERT = "weather_alert"


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events"""
    
    event_id: str
    event_type: EventType
    occurred_at: datetime
    aggregate_id: str
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at.isoformat(),
            "aggregate_id": self.aggregate_id,
            "version": self.version
        }


@dataclass(frozen=True)
class WeatherDataRecordedEvent(DomainEvent):
    """Event raised when weather data is recorded"""
    
    station_id: str
    observation: WeatherObservation
    recorded_at: datetime
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.WEATHER_DATA_RECORDED)
        object.__setattr__(self, 'aggregate_id', self.station_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "station_id": self.station_id,
            "observation": {
                "timestamp": self.observation.timestamp.isoformat(),
                "temperature_celsius": self.observation.temperature_celsius,
                "humidity_percent": self.observation.humidity_percent,
                "pressure_hpa": self.observation.pressure_hpa,
                "wind_speed_ms": self.observation.wind_speed_ms,
                "wind_direction_degrees": self.observation.wind_direction_degrees,
                "cloud_cover_percent": self.observation.cloud_cover_percent,
                "visibility_km": self.observation.visibility_km,
                "uv_index": self.observation.uv_index,
                "precipitation_mm": self.observation.precipitation_mm,
                "data_quality_score": self.observation.data_quality_score
            },
            "recorded_at": self.recorded_at.isoformat()
        })
        return base_dict


@dataclass(frozen=True)
class WeatherAnomalyDetectedEvent(DomainEvent):
    """Event raised when a weather anomaly is detected"""
    
    station_id: str
    anomaly_type: str
    anomaly_description: str
    detected_at: datetime
    severity: str
    observation_data: Dict[str, Any]
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.WEATHER_ANOMALY_DETECTED)
        object.__setattr__(self, 'aggregate_id', self.station_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "station_id": self.station_id,
            "anomaly_type": self.anomaly_type,
            "anomaly_description": self.anomaly_description,
            "detected_at": self.detected_at.isoformat(),
            "severity": self.severity,
            "observation_data": self.observation_data
        })
        return base_dict


@dataclass(frozen=True)
class WeatherStationActivatedEvent(DomainEvent):
    """Event raised when a weather station is activated"""
    
    station_id: str
    station_name: str
    activated_at: datetime
    activated_by: Optional[str] = None
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.WEATHER_STATION_ACTIVATED)
        object.__setattr__(self, 'aggregate_id', self.station_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "station_id": self.station_id,
            "station_name": self.station_name,
            "activated_at": self.activated_at.isoformat(),
            "activated_by": self.activated_by
        })
        return base_dict


@dataclass(frozen=True)
class WeatherStationDeactivatedEvent(DomainEvent):
    """Event raised when a weather station is deactivated"""
    
    station_id: str
    station_name: str
    deactivated_at: datetime
    reason: str
    deactivated_by: Optional[str] = None
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.WEATHER_STATION_DEACTIVATED)
        object.__setattr__(self, 'aggregate_id', self.station_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "station_id": self.station_id,
            "station_name": self.station_name,
            "deactivated_at": self.deactivated_at.isoformat(),
            "reason": self.reason,
            "deactivated_by": self.deactivated_by
        })
        return base_dict


@dataclass(frozen=True)
class WeatherStationMaintenanceEvent(DomainEvent):
    """Event raised when maintenance is performed on a weather station"""
    
    station_id: str
    station_name: str
    maintenance_type: str
    performed_at: datetime
    performed_by: Optional[str] = None
    maintenance_notes: Optional[str] = None
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.WEATHER_STATION_MAINTENANCE)
        object.__setattr__(self, 'aggregate_id', self.station_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "station_id": self.station_id,
            "station_name": self.station_name,
            "maintenance_type": self.maintenance_type,
            "performed_at": self.performed_at.isoformat(),
            "performed_by": self.performed_by,
            "maintenance_notes": self.maintenance_notes
        })
        return base_dict


@dataclass(frozen=True)
class WeatherAlertEvent(DomainEvent):
    """Event raised when a weather alert is triggered"""
    
    station_id: str
    alert_type: str
    alert_message: str
    triggered_at: datetime
    severity: str
    weather_conditions: Dict[str, Any]
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.WEATHER_ALERT)
        object.__setattr__(self, 'aggregate_id', self.station_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "station_id": self.station_id,
            "alert_type": self.alert_type,
            "alert_message": self.alert_message,
            "triggered_at": self.triggered_at.isoformat(),
            "severity": self.severity,
            "weather_conditions": self.weather_conditions
        })
        return base_dict
