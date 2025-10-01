"""
Weather Station Repository Interface
Abstract interface for weather station data access
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

from ...domain.entities.weather_station import WeatherStation, WeatherObservation
from ...domain.enums.weather_station_status import WeatherStationStatus


class IWeatherStationRepository(ABC):
    """
    Abstract interface for weather station repository
    
    Defines the contract for weather station data access operations.
    This interface allows for different implementations (database, cache, etc.)
    while maintaining the same interface.
    """
    
    @abstractmethod
    async def get_by_id(self, station_id: str) -> Optional[WeatherStation]:
        """
        Get a weather station by ID
        
        Args:
            station_id: Station ID
            
        Returns:
            Weather station if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def save(self, station: WeatherStation) -> None:
        """
        Save a weather station
        
        Args:
            station: Weather station to save
        """
        pass
    
    @abstractmethod
    async def delete(self, station_id: str) -> None:
        """
        Delete a weather station
        
        Args:
            station_id: Station ID to delete
        """
        pass
    
    @abstractmethod
    async def get_all(self) -> List[WeatherStation]:
        """
        Get all weather stations
        
        Returns:
            List of all weather stations
        """
        pass
    
    @abstractmethod
    async def get_by_status(self, status: WeatherStationStatus) -> List[WeatherStation]:
        """
        Get weather stations by status
        
        Args:
            status: Station status
            
        Returns:
            List of weather stations with the specified status
        """
        pass
    
    @abstractmethod
    async def get_by_operator(self, operator: str) -> List[WeatherStation]:
        """
        Get weather stations by operator
        
        Args:
            operator: Operator name
            
        Returns:
            List of weather stations operated by the specified operator
        """
        pass
    
    @abstractmethod
    async def get_by_location(self, latitude: float, longitude: float, radius_km: float) -> List[WeatherStation]:
        """
        Get weather stations within a radius of a location
        
        Args:
            latitude: Latitude of center point
            longitude: Longitude of center point
            radius_km: Radius in kilometers
            
        Returns:
            List of weather stations within the radius
        """
        pass
    
    @abstractmethod
    async def get_operational(self) -> List[WeatherStation]:
        """
        Get operational weather stations
        
        Returns:
            List of operational weather stations
        """
        pass
    
    @abstractmethod
    async def get_requiring_attention(self) -> List[WeatherStation]:
        """
        Get weather stations that require attention
        
        Returns:
            List of weather stations requiring attention
        """
        pass
    
    @abstractmethod
    async def get_observations_by_date_range(
        self, 
        station_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[WeatherObservation]:
        """
        Get weather observations for a station within a date range
        
        Args:
            station_id: Station ID
            start_date: Start date for observations
            end_date: End date for observations
            
        Returns:
            List of weather observations within the date range
        """
        pass
    
    @abstractmethod
    async def get_recent_observations(self, station_id: str, hours: int = 24) -> List[WeatherObservation]:
        """
        Get recent weather observations for a station
        
        Args:
            station_id: Station ID
            hours: Number of hours to look back
            
        Returns:
            List of recent weather observations
        """
        pass
    
    @abstractmethod
    async def get_observations_by_quality_threshold(
        self, 
        min_quality: float, 
        since: Optional[datetime] = None
    ) -> List[WeatherObservation]:
        """
        Get weather observations above a quality threshold
        
        Args:
            min_quality: Minimum quality score
            since: Optional datetime to filter observations since
            
        Returns:
            List of high-quality weather observations
        """
        pass
    
    @abstractmethod
    async def count_by_status(self, status: WeatherStationStatus) -> int:
        """
        Count weather stations by status
        
        Args:
            status: Station status
            
        Returns:
            Number of weather stations with the specified status
        """
        pass
    
    @abstractmethod
    async def get_average_quality_score(self) -> float:
        """
        Get average quality score across all weather stations
        
        Returns:
            Average quality score
        """
        pass
    
    @abstractmethod
    async def get_performance_statistics(self) -> dict:
        """
        Get performance statistics for all weather stations
        
        Returns:
            Dictionary with performance statistics
        """
        pass
