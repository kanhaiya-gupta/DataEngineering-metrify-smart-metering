"""
Smart Meter Repository Interface
Abstract interface for smart meter data access
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

from ...domain.entities.smart_meter import SmartMeter
from ...domain.value_objects.meter_id import MeterId
from ...domain.enums.meter_status import MeterStatus
from ...domain.enums.quality_tier import QualityTier


class ISmartMeterRepository(ABC):
    """
    Abstract interface for smart meter repository
    
    Defines the contract for smart meter data access operations.
    This interface allows for different implementations (database, cache, etc.)
    while maintaining the same interface.
    """
    
    @abstractmethod
    async def get_by_id(self, meter_id: MeterId) -> Optional[SmartMeter]:
        """
        Get a smart meter by ID
        
        Args:
            meter_id: Meter ID
            
        Returns:
            Smart meter if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def save(self, meter: SmartMeter) -> None:
        """
        Save a smart meter
        
        Args:
            meter: Smart meter to save
        """
        pass
    
    @abstractmethod
    async def delete(self, meter_id: MeterId) -> None:
        """
        Delete a smart meter
        
        Args:
            meter_id: Meter ID to delete
        """
        pass
    
    @abstractmethod
    async def get_all(self) -> List[SmartMeter]:
        """
        Get all smart meters
        
        Returns:
            List of all smart meters
        """
        pass
    
    @abstractmethod
    async def get_by_status(self, status: MeterStatus) -> List[SmartMeter]:
        """
        Get smart meters by status
        
        Args:
            status: Meter status
            
        Returns:
            List of smart meters with the specified status
        """
        pass
    
    @abstractmethod
    async def get_by_quality_tier(self, quality_tier: QualityTier) -> List[SmartMeter]:
        """
        Get smart meters by quality tier
        
        Args:
            quality_tier: Quality tier
            
        Returns:
            List of smart meters with the specified quality tier
        """
        pass
    
    @abstractmethod
    async def get_by_location(self, latitude: float, longitude: float, radius_km: float) -> List[SmartMeter]:
        """
        Get smart meters within a radius of a location
        
        Args:
            latitude: Latitude of center point
            longitude: Longitude of center point
            radius_km: Radius in kilometers
            
        Returns:
            List of smart meters within the radius
        """
        pass
    
    @abstractmethod
    async def get_by_manufacturer(self, manufacturer: str) -> List[SmartMeter]:
        """
        Get smart meters by manufacturer
        
        Args:
            manufacturer: Manufacturer name
            
        Returns:
            List of smart meters from the specified manufacturer
        """
        pass
    
    @abstractmethod
    async def get_requiring_maintenance(self) -> List[SmartMeter]:
        """
        Get smart meters that require maintenance
        
        Returns:
            List of smart meters requiring maintenance
        """
        pass
    
    @abstractmethod
    async def get_with_anomalies(self, since: Optional[datetime] = None) -> List[SmartMeter]:
        """
        Get smart meters with recent anomalies
        
        Args:
            since: Optional datetime to filter anomalies since
            
        Returns:
            List of smart meters with anomalies
        """
        pass
    
    @abstractmethod
    async def get_by_performance_score_range(self, min_score: float, max_score: float) -> List[SmartMeter]:
        """
        Get smart meters by performance score range
        
        Args:
            min_score: Minimum performance score
            max_score: Maximum performance score
            
        Returns:
            List of smart meters within the performance score range
        """
        pass
    
    @abstractmethod
    async def count_by_status(self, status: MeterStatus) -> int:
        """
        Count smart meters by status
        
        Args:
            status: Meter status
            
        Returns:
            Number of smart meters with the specified status
        """
        pass
    
    @abstractmethod
    async def count_by_quality_tier(self, quality_tier: QualityTier) -> int:
        """
        Count smart meters by quality tier
        
        Args:
            quality_tier: Quality tier
            
        Returns:
            Number of smart meters with the specified quality tier
        """
        pass
    
    @abstractmethod
    async def get_average_quality_score(self) -> float:
        """
        Get average quality score across all smart meters
        
        Returns:
            Average quality score
        """
        pass
    
    @abstractmethod
    async def get_performance_statistics(self) -> dict:
        """
        Get performance statistics for all smart meters
        
        Returns:
            Dictionary with performance statistics
        """
        pass
