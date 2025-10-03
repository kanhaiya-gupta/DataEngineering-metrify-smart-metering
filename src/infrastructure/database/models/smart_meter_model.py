"""
Smart Meter Database Model
SQLAlchemy model for smart meter data persistence
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from ....core.domain.enums.meter_status import MeterStatus
from ....core.domain.enums.quality_tier import QualityTier

Base = declarative_base()


class SmartMeterModel(Base):
    """
    SQLAlchemy model for Smart Meter entity
    
    Maps the domain entity to database table structure
    """
    
    __tablename__ = "smart_meters"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meter_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Location information
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    address = Column(Text, nullable=False)
    
    # Meter specifications
    manufacturer = Column(String(255), nullable=False)
    model = Column(String(255), nullable=False)
    installation_date = Column(DateTime, nullable=False)
    
    # Status and quality
    status = Column(SQLEnum(MeterStatus), nullable=False, default=MeterStatus.ACTIVE)
    quality_tier = Column(SQLEnum(QualityTier), nullable=False, default=QualityTier.UNKNOWN)
    
    # Timestamps
    installed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_reading_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Technical details
    firmware_version = Column(String(50), nullable=False, default="1.0.0")
    meter_metadata = Column(JSON, nullable=True)
    
    # Version for optimistic locking
    version = Column(Integer, nullable=False, default=0)
    
    # Relationships
    readings = relationship("MeterReadingModel", back_populates="meter", cascade="all, delete-orphan")
    events = relationship("MeterEventModel", back_populates="meter", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<SmartMeterModel(id={self.meter_id}, status={self.status.value})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "meter_id": self.meter_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "address": self.address,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "installation_date": self.installation_date.isoformat(),
            "status": self.status.value,
            "quality_tier": self.quality_tier.value,
            "installed_at": self.installed_at.isoformat(),
            "last_reading_at": self.last_reading_at.isoformat() if self.last_reading_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "firmware_version": self.firmware_version,
            "metadata": self.metadata,
            "version": self.version
        }


class MeterReadingModel(Base):
    """
    SQLAlchemy model for Meter Reading entity
    
    Stores individual meter readings with timestamps
    """
    
    __tablename__ = "meter_readings"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meter_id = Column(String(255), nullable=False, index=True)
    
    # Reading data
    timestamp = Column(DateTime, nullable=False, index=True)
    voltage = Column(Float, nullable=False)
    current = Column(Float, nullable=False)
    power_factor = Column(Float, nullable=False)
    frequency = Column(Float, nullable=False)
    active_power = Column(Float, nullable=False)
    reactive_power = Column(Float, nullable=False)
    apparent_power = Column(Float, nullable=False)
    
    # Quality metrics
    data_quality_score = Column(Float, nullable=False, default=1.0)
    is_anomaly = Column(Boolean, nullable=False, default=False)
    anomaly_type = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    meter = relationship("SmartMeterModel", back_populates="readings")
    
    def __repr__(self):
        return f"<MeterReadingModel(id={self.id}, meter_id={self.meter_id}, timestamp={self.timestamp})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "meter_id": self.meter_id,
            "timestamp": self.timestamp.isoformat(),
            "voltage": self.voltage,
            "current": self.current,
            "power_factor": self.power_factor,
            "frequency": self.frequency,
            "active_power": self.active_power,
            "reactive_power": self.reactive_power,
            "apparent_power": self.apparent_power,
            "data_quality_score": self.data_quality_score,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
            "created_at": self.created_at.isoformat()
        }


class MeterEventModel(Base):
    """
    SQLAlchemy model for Meter Domain Events
    
    Stores domain events for audit and event sourcing
    """
    
    __tablename__ = "meter_events"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meter_id = Column(String(255), nullable=False, index=True)
    
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
    meter = relationship("SmartMeterModel", back_populates="events")
    
    def __repr__(self):
        return f"<MeterEventModel(id={self.id}, meter_id={self.meter_id}, event_type={self.event_type})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "meter_id": self.meter_id,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "occurred_at": self.occurred_at.isoformat(),
            "aggregate_version": self.aggregate_version,
            "event_version": self.event_version,
            "created_at": self.created_at.isoformat()
        }
