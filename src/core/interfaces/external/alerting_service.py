"""
Alerting Service Interface
Abstract interface for alerting operations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    METER_ANOMALY = "meter_anomaly"
    GRID_ANOMALY = "grid_anomaly"
    DATA_QUALITY = "data_quality"
    SYSTEM_ERROR = "system_error"
    MAINTENANCE_DUE = "maintenance_due"
    CALIBRATION_DUE = "calibration_due"
    CAPACITY_CRITICAL = "capacity_critical"
    STABILITY_ALERT = "stability_alert"


class IAlertingService(ABC):
    """
    Abstract interface for alerting service
    
    Defines the contract for alerting operations.
    This interface allows for different implementations (email, SMS, Slack, etc.)
    while maintaining the same interface.
    """
    
    @abstractmethod
    async def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: AlertSeverity,
        data: Optional[Dict[str, Any]] = None,
        recipients: Optional[List[str]] = None,
        operator_id: Optional[str] = None,
        meter_id: Optional[str] = None
    ) -> str:
        """
        Send an alert
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity level
            data: Additional alert data
            recipients: List of recipient email addresses
            operator_id: Associated grid operator ID
            meter_id: Associated meter ID
            
        Returns:
            Alert ID for tracking
        """
        pass
    
    @abstractmethod
    async def send_meter_alert(
        self,
        meter_id: str,
        alert_type: AlertType,
        message: str,
        severity: AlertSeverity,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a meter-specific alert
        
        Args:
            meter_id: Meter ID
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity level
            data: Additional alert data
            
        Returns:
            Alert ID for tracking
        """
        pass
    
    @abstractmethod
    async def send_grid_alert(
        self,
        operator_id: str,
        alert_type: AlertType,
        message: str,
        severity: AlertSeverity,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a grid operator-specific alert
        
        Args:
            operator_id: Grid operator ID
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity level
            data: Additional alert data
            
        Returns:
            Alert ID for tracking
        """
        pass
    
    @abstractmethod
    async def get_alert_history(
        self,
        since: Optional[datetime] = None,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
        meter_id: Optional[str] = None,
        operator_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get alert history with optional filters
        
        Args:
            since: Start datetime for history
            alert_type: Filter by alert type
            severity: Filter by severity level
            meter_id: Filter by meter ID
            operator_id: Filter by operator ID
            
        Returns:
            List of historical alerts
        """
        pass
    
    @abstractmethod
    async def get_alert_statistics(
        self,
        since: datetime,
        group_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get alert statistics
        
        Args:
            since: Start datetime for statistics
            group_by: Optional grouping field (type, severity, etc.)
            
        Returns:
            Alert statistics dictionary
        """
        pass
    
    @abstractmethod
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, notes: Optional[str] = None) -> None:
        """
        Acknowledge an alert
        
        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User who acknowledged the alert
            notes: Optional acknowledgment notes
        """
        pass
    
    @abstractmethod
    async def resolve_alert(self, alert_id: str, resolved_by: str, resolution_notes: Optional[str] = None) -> None:
        """
        Resolve an alert
        
        Args:
            alert_id: Alert ID to resolve
            resolved_by: User who resolved the alert
            resolution_notes: Optional resolution notes
        """
        pass
    
    @abstractmethod
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get all active (unresolved) alerts
        
        Returns:
            List of active alerts
        """
        pass
    
    @abstractmethod
    async def configure_alert_rules(self, rules: List[Dict[str, Any]]) -> None:
        """
        Configure alert rules
        
        Args:
            rules: List of alert rule configurations
        """
        pass
    
    @abstractmethod
    async def test_alert_channel(self, channel: str, test_message: str) -> bool:
        """
        Test an alert channel
        
        Args:
            channel: Alert channel to test (email, sms, slack, etc.)
            test_message: Test message to send
            
        Returns:
            True if test was successful, False otherwise
        """
        pass
