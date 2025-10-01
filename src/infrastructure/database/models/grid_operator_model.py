"""
Grid Operator Database Model
SQLAlchemy model for grid operator data persistence
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from ....core.domain.enums.grid_operator_status import GridOperatorStatus

Base = declarative_base()


class GridOperatorModel(Base):
    """
    SQLAlchemy model for Grid Operator entity
    
    Maps the domain entity to database table structure
    """
    
    __tablename__ = "grid_operators"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    operator_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Basic information
    name = Column(String(255), nullable=False)
    operator_type = Column(String(100), nullable=False)
    
    # Location information
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    address = Column(Text, nullable=False)
    
    # Contact information
    contact_email = Column(String(255), nullable=True)
    contact_phone = Column(String(50), nullable=True)
    website = Column(String(255), nullable=True)
    
    # Status and metadata
    status = Column(SQLEnum(GridOperatorStatus), nullable=False, default=GridOperatorStatus.ACTIVE)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Technical details
    grid_capacity_mw = Column(Float, nullable=True)
    voltage_level_kv = Column(Float, nullable=True)
    coverage_area_km2 = Column(Float, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Version for optimistic locking
    version = Column(Integer, nullable=False, default=0)
    
    # Relationships
    grid_statuses = relationship("GridStatusModel", back_populates="operator", cascade="all, delete-orphan")
    events = relationship("GridEventModel", back_populates="operator", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<GridOperatorModel(id={self.operator_id}, name={self.name}, status={self.status.value})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "operator_id": self.operator_id,
            "name": self.name,
            "operator_type": self.operator_type,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "address": self.address,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "website": self.website,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "grid_capacity_mw": self.grid_capacity_mw,
            "voltage_level_kv": self.voltage_level_kv,
            "coverage_area_km2": self.coverage_area_km2,
            "metadata": self.metadata,
            "version": self.version
        }


class GridStatusModel(Base):
    """
    SQLAlchemy model for Grid Status entity
    
    Stores grid status readings with timestamps
    """
    
    __tablename__ = "grid_statuses"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    operator_id = Column(String(255), nullable=False, index=True)
    
    # Status data
    timestamp = Column(DateTime, nullable=False, index=True)
    voltage_level = Column(Float, nullable=False)
    frequency = Column(Float, nullable=False)
    load_percentage = Column(Float, nullable=False)
    stability_score = Column(Float, nullable=False)
    power_quality_score = Column(Float, nullable=False)
    
    # Grid metrics
    total_generation_mw = Column(Float, nullable=True)
    total_consumption_mw = Column(Float, nullable=True)
    grid_frequency_hz = Column(Float, nullable=True)
    voltage_deviation_percent = Column(Float, nullable=True)
    
    # Quality metrics
    data_quality_score = Column(Float, nullable=False, default=1.0)
    is_anomaly = Column(Boolean, nullable=False, default=False)
    anomaly_type = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    operator = relationship("GridOperatorModel", back_populates="grid_statuses")
    
    def __repr__(self):
        return f"<GridStatusModel(id={self.id}, operator_id={self.operator_id}, timestamp={self.timestamp})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "operator_id": self.operator_id,
            "timestamp": self.timestamp.isoformat(),
            "voltage_level": self.voltage_level,
            "frequency": self.frequency,
            "load_percentage": self.load_percentage,
            "stability_score": self.stability_score,
            "power_quality_score": self.power_quality_score,
            "total_generation_mw": self.total_generation_mw,
            "total_consumption_mw": self.total_consumption_mw,
            "grid_frequency_hz": self.grid_frequency_hz,
            "voltage_deviation_percent": self.voltage_deviation_percent,
            "data_quality_score": self.data_quality_score,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
            "created_at": self.created_at.isoformat()
        }


class GridEventModel(Base):
    """
    SQLAlchemy model for Grid Domain Events
    
    Stores domain events for audit and event sourcing
    """
    
    __tablename__ = "grid_events"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    operator_id = Column(String(255), nullable=False, index=True)
    
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
    operator = relationship("GridOperatorModel", back_populates="events")
    
    def __repr__(self):
        return f"<GridEventModel(id={self.id}, operator_id={self.operator_id}, event_type={self.event_type})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "operator_id": self.operator_id,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "occurred_at": self.occurred_at.isoformat(),
            "aggregate_version": self.aggregate_version,
            "event_version": self.event_version,
            "created_at": self.created_at.isoformat()
        }
