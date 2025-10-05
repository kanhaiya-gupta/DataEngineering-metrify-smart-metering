"""
Smart Meter Domain Entity
Represents a smart meter device in the energy grid
"""

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

from ..value_objects.meter_id import MeterId
from ..value_objects.location import Location
from ..value_objects.meter_specifications import MeterSpecifications
from ..enums.meter_status import MeterStatus
from ..enums.quality_tier import QualityTier
from ..events.meter_events import MeterRegisteredEvent, MeterStatusChangedEvent


class MeterType(Enum):
    """Types of smart meters"""
    ELECTRICITY = "electricity"
    GAS = "gas"
    WATER = "water"
    HEAT = "heat"


@dataclass
class MeterReading:
    """A single reading from a smart meter"""
    timestamp: datetime
    consumption_kwh: float
    voltage: float
    current: float
    power_factor: float
    frequency: float
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    data_quality_score: float = 1.0
    
    def __post_init__(self):
        """Validate reading data after initialization"""
        if self.consumption_kwh < 0:
            raise ValueError("Consumption cannot be negative")
        if not (200 <= self.voltage <= 250):
            raise ValueError("Voltage must be between 200V and 250V")
        if not (0 <= self.current <= 100):
            raise ValueError("Current must be between 0A and 100A")
        if not (0 <= self.power_factor <= 1):
            raise ValueError("Power factor must be between 0 and 1")
        if not (49.5 <= self.frequency <= 50.5):
            raise ValueError("Frequency must be between 49.5Hz and 50.5Hz")


@dataclass
class SmartMeter:
    """
    Smart Meter Domain Entity
    
    Represents a smart meter device with all its business logic and rules.
    This is the core entity that encapsulates all smart meter-related business logic.
    """
    
    # Identity
    meter_id: MeterId
    
    # Location and specifications
    location: Location
    specifications: MeterSpecifications
    
    # Status and metadata
    status: MeterStatus = MeterStatus.ACTIVE
    meter_type: MeterType = MeterType.ELECTRICITY
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_reading_at: Optional[datetime] = None
    
    # Performance metrics
    total_readings: int = 0
    average_quality_score: float = 1.0
    quality_tier: QualityTier = QualityTier.HIGH
    
    # Maintenance
    last_maintenance_at: Optional[datetime] = None
    maintenance_due_at: Optional[datetime] = None
    
    # Events
    _domain_events: List = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """Validate entity after initialization"""
        if not self.meter_id:
            raise ValueError("Meter ID is required")
        if not self.location:
            raise ValueError("Location is required")
        if not self.specifications:
            raise ValueError("Meter specifications are required")
    
    @property
    def domain_events(self) -> List:
        """Get domain events"""
        return self._domain_events.copy()
    
    def clear_domain_events(self) -> None:
        """Clear domain events after they've been processed"""
        self._domain_events.clear()
    
    def get_uncommitted_events(self) -> List:
        """Get uncommitted domain events (alias for domain_events)"""
        return self._domain_events.copy()
    
    def clear_uncommitted_events(self) -> None:
        """Clear uncommitted domain events (alias for clear_domain_events)"""
        self._domain_events.clear()
    
    def add_domain_event(self, event) -> None:
        """Add a domain event"""
        self._domain_events.append(event)
    
    def register_meter(self) -> None:
        """Register a new smart meter"""
        if self.status != MeterStatus.PENDING:
            raise ValueError("Only pending meters can be registered")
        
        self.status = MeterStatus.ACTIVE
        self.updated_at = datetime.utcnow()
        
        # Add domain event
        self.add_domain_event(
            MeterRegisteredEvent(
                meter_id=self.meter_id.value,
                location=self.location,
                registered_at=self.updated_at
            )
        )
    
    def deactivate_meter(self, reason: str) -> None:
        """Deactivate the smart meter"""
        if self.status == MeterStatus.DEACTIVATED:
            raise ValueError("Meter is already deactivated")
        
        old_status = self.status
        self.status = MeterStatus.DEACTIVATED
        self.updated_at = datetime.utcnow()
        
        # Add domain event
        self.add_domain_event(
            MeterStatusChangedEvent(
                meter_id=self.meter_id.value,
                old_status=old_status,
                new_status=self.status,
                reason=reason,
                changed_at=self.updated_at
            )
        )
    
    def record_reading(self, reading: MeterReading) -> None:
        """Record a new meter reading"""
        if self.status != MeterStatus.ACTIVE:
            raise ValueError("Cannot record reading for inactive meter")
        
        # Update reading statistics
        self.total_readings += 1
        self.last_reading_at = reading.timestamp
        self.updated_at = datetime.utcnow()
        
        # Update quality metrics
        self._update_quality_metrics(reading)
        
        # Check for anomalies
        self._check_for_anomalies(reading)
    
    def _update_quality_metrics(self, reading: MeterReading) -> None:
        """Update quality metrics based on new reading"""
        # Calculate running average of quality scores
        if self.total_readings == 1:
            self.average_quality_score = reading.data_quality_score
        else:
            # Weighted average giving more weight to recent readings
            weight = min(0.1, 1.0 / self.total_readings)
            self.average_quality_score = (
                (1 - weight) * self.average_quality_score + 
                weight * reading.data_quality_score
            )
        
        # Update quality tier based on average score
        if self.average_quality_score >= 0.9:
            self.quality_tier = QualityTier.HIGH
        elif self.average_quality_score >= 0.7:
            self.quality_tier = QualityTier.MEDIUM
        else:
            self.quality_tier = QualityTier.LOW
    
    def _check_for_anomalies(self, reading: MeterReading) -> None:
        """Check for anomalies in the reading"""
        anomalies = []
        
        # Check for voltage anomalies
        if not (220 <= reading.voltage <= 240):
            anomalies.append(f"Voltage anomaly: {reading.voltage}V")
        
        # Check for current anomalies
        if reading.current > 80:  # High current threshold
            anomalies.append(f"High current: {reading.current}A")
        
        # Check for power factor anomalies
        if reading.power_factor < 0.8:
            anomalies.append(f"Low power factor: {reading.power_factor}")
        
        # Check for frequency anomalies
        if not (49.8 <= reading.frequency <= 50.2):
            anomalies.append(f"Frequency anomaly: {reading.frequency}Hz")
        
        # If multiple anomalies, consider maintenance
        if len(anomalies) >= 3:
            self._schedule_maintenance("Multiple anomalies detected")
    
    def _schedule_maintenance(self, reason: str) -> None:
        """Schedule maintenance for the meter"""
        # Schedule maintenance for next week
        self.maintenance_due_at = datetime.utcnow().replace(
            hour=9, minute=0, second=0, microsecond=0
        )
        # Add 7 days
        from datetime import timedelta
        self.maintenance_due_at += timedelta(days=7)
    
    def perform_maintenance(self, maintenance_notes: str) -> None:
        """Perform maintenance on the meter"""
        self.last_maintenance_at = datetime.utcnow()
        self.maintenance_due_at = None
        self.updated_at = datetime.utcnow()
        
        # Reset quality metrics after maintenance
        self.average_quality_score = 1.0
        self.quality_tier = QualityTier.HIGH
    
    def is_maintenance_due(self) -> bool:
        """Check if maintenance is due"""
        if not self.maintenance_due_at:
            return False
        return datetime.utcnow() >= self.maintenance_due_at
    
    def get_performance_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        score = 0.0
        
        # Quality score (40% weight)
        score += self.average_quality_score * 40
        
        # Uptime score (30% weight)
        if self.status == MeterStatus.ACTIVE:
            score += 30
        elif self.status == MeterStatus.MAINTENANCE:
            score += 20
        else:
            score += 0
        
        # Reading frequency score (20% weight)
        if self.total_readings > 0:
            # Calculate readings per day (assuming 24 readings per day is optimal)
            days_since_creation = (datetime.utcnow() - self.created_at).days
            if days_since_creation > 0:
                readings_per_day = self.total_readings / days_since_creation
                frequency_score = min(1.0, readings_per_day / 24)
                score += frequency_score * 20
        
        # Maintenance score (10% weight)
        if not self.is_maintenance_due():
            score += 10
        elif self.last_maintenance_at:
            days_since_maintenance = (datetime.utcnow() - self.last_maintenance_at).days
            if days_since_maintenance < 30:  # Recent maintenance
                score += 10
            else:
                score += 5  # Partial score for older maintenance
        
        return min(100.0, max(0.0, score))
    
    def get_health_status(self) -> str:
        """Get human-readable health status"""
        performance_score = self.get_performance_score()
        
        if performance_score >= 90:
            return "Excellent"
        elif performance_score >= 75:
            return "Good"
        elif performance_score >= 60:
            return "Fair"
        elif performance_score >= 40:
            return "Poor"
        else:
            return "Critical"
    
    def __str__(self) -> str:
        """String representation of the smart meter"""
        return f"SmartMeter(id={self.meter_id.value}, status={self.status.value}, location={self.location})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"SmartMeter("
            f"id={self.meter_id.value}, "
            f"status={self.status.value}, "
            f"type={self.meter_type.value}, "
            f"location={self.location}, "
            f"readings={self.total_readings}, "
            f"quality={self.average_quality_score:.2f}"
            f")"
        )
