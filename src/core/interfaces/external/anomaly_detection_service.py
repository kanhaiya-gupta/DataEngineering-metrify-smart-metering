"""
Anomaly Detection Service Interface
Abstract interface for anomaly detection operations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime

from ...domain.entities.smart_meter import SmartMeter, MeterReading


class IAnomalyDetectionService(ABC):
    """
    Abstract interface for anomaly detection service
    
    Defines the contract for anomaly detection operations.
    This interface allows for different implementations (ML-based, statistical, etc.)
    while maintaining the same interface.
    """
    
    @abstractmethod
    async def detect_anomalies(self, reading: MeterReading, meter: SmartMeter) -> List[Dict[str, Any]]:
        """
        Detect anomalies in a meter reading
        
        Args:
            reading: Meter reading to analyze
            meter: Smart meter that generated the reading
            
        Returns:
            List of detected anomalies, each containing:
            - type: str - Type of anomaly
            - description: str - Human-readable description
            - severity: str - Severity level (low, medium, high, critical)
            - confidence: float - Confidence score (0.0-1.0)
            - data: Dict - Additional anomaly data
        """
        pass
    
    @abstractmethod
    async def detect_voltage_anomalies(self, reading: MeterReading, meter: SmartMeter) -> List[Dict[str, Any]]:
        """
        Detect voltage-related anomalies
        
        Args:
            reading: Meter reading to analyze
            meter: Smart meter that generated the reading
            
        Returns:
            List of voltage anomalies
        """
        pass
    
    @abstractmethod
    async def detect_current_anomalies(self, reading: MeterReading, meter: SmartMeter) -> List[Dict[str, Any]]:
        """
        Detect current-related anomalies
        
        Args:
            reading: Meter reading to analyze
            meter: Smart meter that generated the reading
            
        Returns:
            List of current anomalies
        """
        pass
    
    @abstractmethod
    async def detect_power_factor_anomalies(self, reading: MeterReading, meter: SmartMeter) -> List[Dict[str, Any]]:
        """
        Detect power factor anomalies
        
        Args:
            reading: Meter reading to analyze
            meter: Smart meter that generated the reading
            
        Returns:
            List of power factor anomalies
        """
        pass
    
    @abstractmethod
    async def detect_frequency_anomalies(self, reading: MeterReading, meter: SmartMeter) -> List[Dict[str, Any]]:
        """
        Detect frequency anomalies
        
        Args:
            reading: Meter reading to analyze
            meter: Smart meter that generated the reading
            
        Returns:
            List of frequency anomalies
        """
        pass
    
    @abstractmethod
    async def detect_consumption_anomalies(self, reading: MeterReading, meter: SmartMeter) -> List[Dict[str, Any]]:
        """
        Detect consumption pattern anomalies
        
        Args:
            reading: Meter reading to analyze
            meter: Smart meter that generated the reading
            
        Returns:
            List of consumption anomalies
        """
        pass
    
    @abstractmethod
    async def get_anomaly_history(self, meter_id: str, since: datetime) -> List[Dict[str, Any]]:
        """
        Get anomaly history for a meter
        
        Args:
            meter_id: Meter ID
            since: Start datetime for history
            
        Returns:
            List of historical anomalies
        """
        pass
    
    @abstractmethod
    async def get_anomaly_statistics(self, meter_id: str, days: int) -> Dict[str, Any]:
        """
        Get anomaly statistics for a meter
        
        Args:
            meter_id: Meter ID
            days: Number of days to analyze
            
        Returns:
            Anomaly statistics dictionary
        """
        pass
    
    @abstractmethod
    async def update_anomaly_thresholds(self, thresholds: Dict[str, Any]) -> None:
        """
        Update anomaly detection thresholds
        
        Args:
            thresholds: New threshold configuration
        """
        pass
    
    @abstractmethod
    async def train_anomaly_model(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Train the anomaly detection model
        
        Args:
            training_data: Training data for the model
        """
        pass
