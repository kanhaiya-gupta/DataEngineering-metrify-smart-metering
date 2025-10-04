"""
Weather Station Database Model
SQLAlchemy model for weather station data persistence
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, JSON, Boolean, Enum as SQLEnum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from ....core.domain.enums.weather_station_status import WeatherStationStatus
from ..base import Base


class WeatherStationModel(Base):
    """
    SQLAlchemy model for Weather Station entity
    
    Maps the domain entity to database table structure
    """
    
    __tablename__ = "weather_stations"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    station_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Basic information
    name = Column(String(255), nullable=False)
    station_type = Column(String(100), nullable=False)
    
    # Location information
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    address = Column(Text, nullable=False)
    
    # Contact information
    operator = Column(String(255), nullable=False)
    contact_email = Column(String(255), nullable=True)
    contact_phone = Column(String(50), nullable=True)
    
    # Status and metadata
    status = Column(SQLEnum(WeatherStationStatus), nullable=False, default=WeatherStationStatus.ACTIVE)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Performance metrics
    total_observations = Column(Integer, nullable=False, default=0)
    average_quality_score = Column(Float, nullable=False, default=1.0)
    last_observation_at = Column(DateTime, nullable=True)
    
    # Version for optimistic locking
    version = Column(Integer, nullable=False, default=0)
    
    # Relationships
    observations = relationship("WeatherObservationModel", back_populates="station", cascade="all, delete-orphan")
    events = relationship("WeatherEventModel", back_populates="station", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<WeatherStationModel(id={self.station_id}, name={self.name}, status={self.status.value})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "station_id": self.station_id,
            "name": self.name,
            "station_type": self.station_type,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "address": self.address,
            "operator": self.operator,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_observations": self.total_observations,
            "average_quality_score": self.average_quality_score,
            "last_observation_at": self.last_observation_at.isoformat() if self.last_observation_at else None,
            "version": self.version
        }


class WeatherObservationModel(Base):
    """
    SQLAlchemy model for Weather Observation entity
    
    Stores individual weather observations with timestamps
    """
    
    __tablename__ = "weather_observations"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    station_id = Column(String(255), ForeignKey("weather_stations.station_id"), nullable=False, index=True)
    
    # Observation data
    timestamp = Column(DateTime, nullable=False, index=True)
    temperature_celsius = Column(Float, nullable=False)
    humidity_percent = Column(Float, nullable=False)
    pressure_hpa = Column(Float, nullable=False)
    wind_speed_ms = Column(Float, nullable=False)
    wind_direction_degrees = Column(Float, nullable=False)
    cloud_cover_percent = Column(Float, nullable=False)
    visibility_km = Column(Float, nullable=False)
    uv_index = Column(Float, nullable=True)
    precipitation_mm = Column(Float, nullable=True)
    
    # Quality metrics
    data_quality_score = Column(Float, nullable=False, default=1.0)
    is_anomaly = Column(Boolean, nullable=False, default=False)
    anomaly_type = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    station = relationship("WeatherStationModel", back_populates="observations")
    
    def __repr__(self):
        return f"<WeatherObservationModel(id={self.id}, station_id={self.station_id}, timestamp={self.timestamp})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "station_id": self.station_id,
            "timestamp": self.timestamp.isoformat(),
            "temperature_celsius": self.temperature_celsius,
            "humidity_percent": self.humidity_percent,
            "pressure_hpa": self.pressure_hpa,
            "wind_speed_ms": self.wind_speed_ms,
            "wind_direction_degrees": self.wind_direction_degrees,
            "cloud_cover_percent": self.cloud_cover_percent,
            "visibility_km": self.visibility_km,
            "uv_index": self.uv_index,
            "precipitation_mm": self.precipitation_mm,
            "data_quality_score": self.data_quality_score,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
            "created_at": self.created_at.isoformat()
        }


class WeatherEventModel(Base):
    """
    SQLAlchemy model for Weather Domain Events
    
    Stores domain events for audit and event sourcing
    """
    
    __tablename__ = "weather_events"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    station_id = Column(String(255), ForeignKey("weather_stations.station_id"), nullable=False, index=True)
    
    # Event data
    event_type = Column(String(100), nullable=False, index=True)
    event_data = Column(JSON, nullable=False)
    occurred_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Event sourcing
    aggregate_version = Column(Integer, nullable=False)
    event_version = Column(Integer, nullable=False, default=1)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    station = relationship("WeatherStationModel", back_populates="events")
    
    def __repr__(self):
        return f"<WeatherEventModel(id={self.id}, station_id={self.station_id}, event_type={self.event_type})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "station_id": self.station_id,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "occurred_at": self.occurred_at.isoformat(),
            "aggregate_version": self.aggregate_version,
            "event_version": self.event_version,
            "created_at": self.created_at.isoformat()
        }
