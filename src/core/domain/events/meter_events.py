"""
Smart Meter Domain Events
Events that occur in the smart meter domain
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from ..value_objects.location import Location
from ..enums.meter_status import MeterStatus


class EventType(Enum):
    """Types of domain events"""
    METER_REGISTERED = "meter_registered"
    METER_STATUS_CHANGED = "meter_status_changed"
    METER_READING_RECORDED = "meter_reading_recorded"
    METER_MAINTENANCE_SCHEDULED = "meter_maintenance_scheduled"
    METER_MAINTENANCE_COMPLETED = "meter_maintenance_completed"
    METER_ANOMALY_DETECTED = "meter_anomaly_detected"
    METER_CALIBRATION_DUE = "meter_calibration_due"


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events"""
    
    event_id: str
    event_type: EventType
    occurred_at: datetime
    aggregate_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at.isoformat(),
            "aggregate_id": self.aggregate_id,
            "version": getattr(self, 'version', 1)
        }


@dataclass(frozen=True)
class MeterRegisteredEvent(DomainEvent):
    """Event raised when a smart meter is registered"""
    
    meter_id: str
    location: Location
    registered_at: datetime
    version: int = 1
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.METER_REGISTERED)
        object.__setattr__(self, 'aggregate_id', self.meter_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "meter_id": self.meter_id,
            "location": {
                "latitude": self.location.latitude,
                "longitude": self.location.longitude,
                "address": self.location.address,
                "city": self.location.city,
                "postal_code": self.location.postal_code,
                "country": self.location.country,
                "grid_zone": self.location.grid_zone
            },
            "registered_at": self.registered_at.isoformat()
        })
        return base_dict


@dataclass(frozen=True)
class MeterStatusChangedEvent(DomainEvent):
    """Event raised when a smart meter's status changes"""
    
    meter_id: str
    old_status: MeterStatus
    new_status: MeterStatus
    reason: str
    changed_at: datetime
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.METER_STATUS_CHANGED)
        object.__setattr__(self, 'aggregate_id', self.meter_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "meter_id": self.meter_id,
            "old_status": self.old_status.value,
            "new_status": self.new_status.value,
            "reason": self.reason,
            "changed_at": self.changed_at.isoformat()
        })
        return base_dict


@dataclass(frozen=True)
class MeterReadingRecordedEvent(DomainEvent):
    """Event raised when a new meter reading is recorded"""
    
    meter_id: str
    reading_timestamp: datetime
    consumption_kwh: float
    voltage: float
    current: float
    power_factor: float
    frequency: float
    data_quality_score: float
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.METER_READING_RECORDED)
        object.__setattr__(self, 'aggregate_id', self.meter_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "meter_id": self.meter_id,
            "reading_timestamp": self.reading_timestamp.isoformat(),
            "consumption_kwh": self.consumption_kwh,
            "voltage": self.voltage,
            "current": self.current,
            "power_factor": self.power_factor,
            "frequency": self.frequency,
            "data_quality_score": self.data_quality_score
        })
        return base_dict


@dataclass(frozen=True)
class MeterMaintenanceScheduledEvent(DomainEvent):
    """Event raised when maintenance is scheduled for a meter"""
    
    meter_id: str
    scheduled_at: datetime
    reason: str
    priority: str
    estimated_duration_hours: Optional[int] = None
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.METER_MAINTENANCE_SCHEDULED)
        object.__setattr__(self, 'aggregate_id', self.meter_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "meter_id": self.meter_id,
            "scheduled_at": self.scheduled_at.isoformat(),
            "reason": self.reason,
            "priority": self.priority,
            "estimated_duration_hours": self.estimated_duration_hours
        })
        return base_dict


@dataclass(frozen=True)
class MeterMaintenanceCompletedEvent(DomainEvent):
    """Event raised when maintenance is completed on a meter"""
    
    meter_id: str
    completed_at: datetime
    maintenance_notes: str
    technician_id: Optional[str] = None
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.METER_MAINTENANCE_COMPLETED)
        object.__setattr__(self, 'aggregate_id', self.meter_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "meter_id": self.meter_id,
            "completed_at": self.completed_at.isoformat(),
            "maintenance_notes": self.maintenance_notes,
            "technician_id": self.technician_id
        })
        return base_dict


@dataclass(frozen=True)
class MeterAnomalyDetectedEvent(DomainEvent):
    """Event raised when an anomaly is detected in meter readings"""
    
    meter_id: str
    anomaly_type: str
    anomaly_description: str
    detected_at: datetime
    severity: str
    reading_data: Dict[str, Any]
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.METER_ANOMALY_DETECTED)
        object.__setattr__(self, 'aggregate_id', self.meter_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "meter_id": self.meter_id,
            "anomaly_type": self.anomaly_type,
            "anomaly_description": self.anomaly_description,
            "detected_at": self.detected_at.isoformat(),
            "severity": self.severity,
            "reading_data": self.reading_data
        })
        return base_dict


@dataclass(frozen=True)
class MeterCalibrationDueEvent(DomainEvent):
    """Event raised when a meter is due for calibration"""
    
    meter_id: str
    calibration_due_date: datetime
    days_until_due: int
    last_calibration_date: Optional[datetime] = None
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.METER_CALIBRATION_DUE)
        object.__setattr__(self, 'aggregate_id', self.meter_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "meter_id": self.meter_id,
            "calibration_due_date": self.calibration_due_date.isoformat(),
            "days_until_due": self.days_until_due,
            "last_calibration_date": self.last_calibration_date.isoformat() if self.last_calibration_date else None
        })
        return base_dict
