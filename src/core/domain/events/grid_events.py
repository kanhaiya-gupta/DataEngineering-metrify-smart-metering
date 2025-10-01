"""
Grid Operator Domain Events
Events that occur in the grid operator domain
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from ..entities.grid_operator import GridStatus


class EventType(Enum):
    """Types of domain events"""
    GRID_STATUS_UPDATED = "grid_status_updated"
    GRID_ANOMALY_DETECTED = "grid_anomaly_detected"
    GRID_OPERATOR_ACTIVATED = "grid_operator_activated"
    GRID_OPERATOR_DEACTIVATED = "grid_operator_deactivated"
    GRID_CAPACITY_CRITICAL = "grid_capacity_critical"
    GRID_STABILITY_ALERT = "grid_stability_alert"


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
class GridStatusUpdatedEvent(DomainEvent):
    """Event raised when grid status is updated"""
    
    operator_id: str
    old_status: Optional[GridStatus]
    new_status: GridStatus
    updated_at: datetime
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.GRID_STATUS_UPDATED)
        object.__setattr__(self, 'aggregate_id', self.operator_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "operator_id": self.operator_id,
            "old_status": self.old_status.__dict__ if self.old_status else None,
            "new_status": self.new_status.__dict__,
            "updated_at": self.updated_at.isoformat()
        })
        return base_dict


@dataclass(frozen=True)
class GridAnomalyDetectedEvent(DomainEvent):
    """Event raised when an anomaly is detected in grid data"""
    
    operator_id: str
    anomaly_type: str
    anomaly_description: str
    detected_at: datetime
    severity: str
    status_data: Dict[str, Any]
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.GRID_ANOMALY_DETECTED)
        object.__setattr__(self, 'aggregate_id', self.operator_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "operator_id": self.operator_id,
            "anomaly_type": self.anomaly_type,
            "anomaly_description": self.anomaly_description,
            "detected_at": self.detected_at.isoformat(),
            "severity": self.severity,
            "status_data": self.status_data
        })
        return base_dict


@dataclass(frozen=True)
class GridOperatorActivatedEvent(DomainEvent):
    """Event raised when a grid operator is activated"""
    
    operator_id: str
    operator_name: str
    activated_at: datetime
    activated_by: Optional[str] = None
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.GRID_OPERATOR_ACTIVATED)
        object.__setattr__(self, 'aggregate_id', self.operator_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "activated_at": self.activated_at.isoformat(),
            "activated_by": self.activated_by
        })
        return base_dict


@dataclass(frozen=True)
class GridOperatorDeactivatedEvent(DomainEvent):
    """Event raised when a grid operator is deactivated"""
    
    operator_id: str
    operator_name: str
    deactivated_at: datetime
    reason: str
    deactivated_by: Optional[str] = None
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.GRID_OPERATOR_DEACTIVATED)
        object.__setattr__(self, 'aggregate_id', self.operator_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "deactivated_at": self.deactivated_at.isoformat(),
            "reason": self.reason,
            "deactivated_by": self.deactivated_by
        })
        return base_dict


@dataclass(frozen=True)
class GridCapacityCriticalEvent(DomainEvent):
    """Event raised when grid capacity reaches critical levels"""
    
    operator_id: str
    utilization_rate: float
    available_capacity_mw: float
    total_capacity_mw: float
    detected_at: datetime
    threshold_exceeded: float
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.GRID_CAPACITY_CRITICAL)
        object.__setattr__(self, 'aggregate_id', self.operator_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "operator_id": self.operator_id,
            "utilization_rate": self.utilization_rate,
            "available_capacity_mw": self.available_capacity_mw,
            "total_capacity_mw": self.total_capacity_mw,
            "detected_at": self.detected_at.isoformat(),
            "threshold_exceeded": self.threshold_exceeded
        })
        return base_dict


@dataclass(frozen=True)
class GridStabilityAlertEvent(DomainEvent):
    """Event raised when grid stability falls below acceptable levels"""
    
    operator_id: str
    stability_score: float
    frequency_hz: float
    voltage_kv: float
    detected_at: datetime
    alert_level: str
    
    def __post_init__(self):
        """Set event properties after initialization"""
        object.__setattr__(self, 'event_type', EventType.GRID_STABILITY_ALERT)
        object.__setattr__(self, 'aggregate_id', self.operator_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        base_dict = super().to_dict()
        base_dict.update({
            "operator_id": self.operator_id,
            "stability_score": self.stability_score,
            "frequency_hz": self.frequency_hz,
            "voltage_kv": self.voltage_kv,
            "detected_at": self.detected_at.isoformat(),
            "alert_level": self.alert_level
        })
        return base_dict
