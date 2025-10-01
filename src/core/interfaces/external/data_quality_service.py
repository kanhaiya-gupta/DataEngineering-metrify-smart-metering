"""
Data Quality Service Interface
Abstract interface for data quality operations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from datetime import datetime

from ...domain.entities.smart_meter import MeterReading


class IDataQualityService(ABC):
    """
    Abstract interface for data quality service
    
    Defines the contract for data quality operations.
    This interface allows for different implementations (ML-based, rule-based, etc.)
    while maintaining the same interface.
    """
    
    @abstractmethod
    async def calculate_quality_score(self, reading: MeterReading) -> float:
        """
        Calculate data quality score for a meter reading
        
        Args:
            reading: Meter reading to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    async def validate_reading(self, reading: MeterReading) -> Dict[str, Any]:
        """
        Validate a meter reading against quality rules
        
        Args:
            reading: Meter reading to validate
            
        Returns:
            Validation result dictionary with:
            - is_valid: bool
            - violations: List[str]
            - quality_score: float
            - recommendations: List[str]
        """
        pass
    
    @abstractmethod
    async def get_quality_metrics(self, meter_id: str, since: datetime) -> Dict[str, Any]:
        """
        Get quality metrics for a specific meter
        
        Args:
            meter_id: Meter ID
            since: Start datetime for metrics
            
        Returns:
            Quality metrics dictionary
        """
        pass
    
    @abstractmethod
    async def get_system_quality_summary(self) -> Dict[str, Any]:
        """
        Get overall system quality summary
        
        Returns:
            System quality summary dictionary
        """
        pass
    
    @abstractmethod
    async def update_quality_rules(self, rules: Dict[str, Any]) -> None:
        """
        Update data quality rules
        
        Args:
            rules: New quality rules configuration
        """
        pass
    
    @abstractmethod
    async def get_quality_trends(self, meter_id: str, days: int) -> Dict[str, Any]:
        """
        Get quality trends for a meter over time
        
        Args:
            meter_id: Meter ID
            days: Number of days to analyze
            
        Returns:
            Quality trends dictionary
        """
        pass
