"""
Grid Operator Domain Entity
Represents a grid operator in the energy system
"""

from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from ..value_objects.location import Location
from ..value_objects.grid_status import GridStatus
from ..enums.grid_operator_status import GridOperatorStatus


class GridOperatorType(Enum):
    """Types of grid operators"""
    TRANSMISSION = "transmission"  # High voltage transmission
    DISTRIBUTION = "distribution"  # Medium/low voltage distribution
    BALANCING = "balancing"        # System balancing authority


@dataclass
class GridOperator:
    """
    Grid Operator Domain Entity
    
    Represents a grid operator responsible for managing part of the electrical grid
    """
    
    # Identity
    operator_id: str
    name: str
    operator_type: GridOperatorType
    
    # Location and coverage
    headquarters: Location
    coverage_regions: List[str]
    
    # Contact information
    contact_email: str
    contact_phone: Optional[str] = None
    website: Optional[str] = None
    
    # API configuration
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    
    # Status and metadata
    status: GridOperatorStatus = GridOperatorStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Current grid status
    current_status: Optional[GridStatus] = None
    last_status_update: Optional[datetime] = None
    
    # Performance metrics
    uptime_percentage: float = 100.0
    average_response_time_ms: float = 0.0
    data_quality_score: float = 1.0
    
    # Events
    _domain_events: List = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """Validate entity after initialization"""
        if not self.operator_id:
            raise ValueError("Operator ID is required")
        if not self.name:
            raise ValueError("Operator name is required")
        if not self.headquarters:
            raise ValueError("Headquarters location is required")
        if not self.contact_email:
            raise ValueError("Contact email is required")
    
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
    
    def update_grid_status(self, status: GridStatus) -> None:
        """Update the current grid status"""
        if self.status != GridOperatorStatus.ACTIVE:
            raise ValueError("Cannot update status for inactive operator")
        
        old_status = self.current_status
        self.current_status = status
        self.last_status_update = status.timestamp
        self.updated_at = datetime.utcnow()
        
        # Update performance metrics
        self._update_performance_metrics(status)
        
        # Check for anomalies
        self._check_for_anomalies(status)
        
        # Add domain event
        self.add_domain_event(
            GridStatusUpdatedEvent(
                operator_id=self.operator_id,
                old_status=old_status,
                new_status=status,
                updated_at=self.updated_at
            )
        )
    
    def _update_performance_metrics(self, status: GridStatus) -> None:
        """Update performance metrics based on new status"""
        # Calculate utilization rate
        if status.total_capacity_mw > 0:
            utilization_rate = (status.total_capacity_mw - status.available_capacity_mw) / status.total_capacity_mw
        else:
            utilization_rate = 0.0
        
        # Update data quality score based on grid stability
        self.data_quality_score = status.grid_stability_score
        
        # Update uptime (simplified - in reality this would be more complex)
        if status.grid_stability_score >= 0.8:
            self.uptime_percentage = min(100.0, self.uptime_percentage + 0.1)
        else:
            self.uptime_percentage = max(0.0, self.uptime_percentage - 0.5)
    
    def _check_for_anomalies(self, status: GridStatus) -> None:
        """Check for anomalies in the grid status"""
        anomalies = []
        
        # Check for frequency anomalies
        if not (49.8 <= status.frequency_hz <= 50.2):
            anomalies.append(f"Frequency anomaly: {status.frequency_hz}Hz")
        
        # Check for voltage anomalies
        if not (380 <= status.voltage_kv <= 420):
            anomalies.append(f"Voltage anomaly: {status.voltage_kv}kV")
        
        # Check for capacity anomalies
        if status.available_capacity_mw > status.total_capacity_mw:
            anomalies.append("Available capacity exceeds total capacity")
        
        # Check for stability anomalies
        if status.grid_stability_score < 0.7:
            anomalies.append(f"Low grid stability: {status.grid_stability_score}")
        
        # If anomalies detected, raise event
        if anomalies:
            self.add_domain_event(
                GridAnomalyDetectedEvent(
                    operator_id=self.operator_id,
                    anomaly_type="grid_anomaly",
                    anomaly_description="; ".join(anomalies),
                    detected_at=datetime.utcnow(),
                    severity="high" if len(anomalies) > 2 else "medium",
                    status_data=status.__dict__
                )
            )
    
    def deactivate_operator(self, reason: str) -> None:
        """Deactivate the grid operator"""
        if self.status == GridOperatorStatus.INACTIVE:
            raise ValueError("Operator is already inactive")
        
        old_status = self.status
        self.status = GridOperatorStatus.INACTIVE
        self.updated_at = datetime.utcnow()
        
        # Clear current status
        self.current_status = None
        self.last_status_update = None
    
    def reactivate_operator(self) -> None:
        """Reactivate the grid operator"""
        if self.status != GridOperatorStatus.INACTIVE:
            raise ValueError("Only inactive operators can be reactivated")
        
        self.status = GridOperatorStatus.ACTIVE
        self.updated_at = datetime.utcnow()
    
    def get_utilization_rate(self) -> float:
        """Get current grid utilization rate"""
        if not self.current_status or self.current_status.total_capacity_mw == 0:
            return 0.0
        
        return (self.current_status.total_capacity_mw - self.current_status.available_capacity_mw) / self.current_status.total_capacity_mw
    
    def get_capacity_status(self) -> str:
        """Get human-readable capacity status"""
        if not self.current_status:
            return "Unknown"
        
        utilization_rate = self.get_utilization_rate()
        
        if utilization_rate < 0.1:
            return "Very Low"
        elif utilization_rate < 0.3:
            return "Low"
        elif utilization_rate < 0.7:
            return "Normal"
        elif utilization_rate < 0.9:
            return "High"
        else:
            return "Critical"
    
    def get_stability_status(self) -> str:
        """Get human-readable stability status"""
        if not self.current_status:
            return "Unknown"
        
        stability_score = self.current_status.grid_stability_score
        
        if stability_score >= 0.9:
            return "Excellent"
        elif stability_score >= 0.8:
            return "Good"
        elif stability_score >= 0.7:
            return "Fair"
        elif stability_score >= 0.5:
            return "Poor"
        else:
            return "Critical"
    
    def is_operational(self) -> bool:
        """Check if the operator is operational"""
        return (self.status == GridOperatorStatus.ACTIVE and 
                self.current_status is not None and
                self.last_status_update is not None)
    
    def requires_attention(self) -> bool:
        """Check if the operator requires attention"""
        if not self.is_operational():
            return True
        
        if not self.current_status:
            return True
        
        # Check for critical conditions
        if (self.current_status.grid_stability_score < 0.7 or
            not (49.8 <= self.current_status.frequency_hz <= 50.2) or
            not (380 <= self.current_status.voltage_kv <= 420)):
            return True
        
        return False
    
    def __str__(self) -> str:
        """String representation"""
        return f"GridOperator(id={self.operator_id}, name={self.name}, status={self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"GridOperator("
            f"id={self.operator_id}, "
            f"name={self.name}, "
            f"type={self.operator_type.value}, "
            f"status={self.status.value}, "
            f"uptime={self.uptime_percentage:.1f}%"
            f")"
        )
