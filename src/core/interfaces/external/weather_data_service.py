"""
Weather Data Service Interface
Abstract interface for weather data operations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...domain.entities.weather_station import WeatherObservation


class IWeatherDataService(ABC):
    """
    Abstract interface for weather data service
    
    Defines the contract for weather data operations.
    This interface allows for different implementations (API-based, file-based, etc.)
    while maintaining the same interface.
    """
    
    @abstractmethod
    async def calculate_quality_score(self, observation: WeatherObservation) -> float:
        """
        Calculate data quality score for a weather observation
        
        Args:
            observation: Weather observation to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    async def validate_observation(self, observation: WeatherObservation) -> Dict[str, Any]:
        """
        Validate a weather observation against quality rules
        
        Args:
            observation: Weather observation to validate
            
        Returns:
            Validation result dictionary with:
            - is_valid: bool
            - violations: List[str]
            - quality_score: float
            - recommendations: List[str]
        """
        pass
    
    @abstractmethod
    async def fetch_weather_data(self, station_id: str, start_date: datetime, end_date: datetime) -> List[WeatherObservation]:
        """
        Fetch weather data from external source
        
        Args:
            station_id: Weather station ID
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            List of weather observations
        """
        pass
    
    @abstractmethod
    async def get_weather_forecast(self, station_id: str, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """
        Get weather forecast for a station
        
        Args:
            station_id: Weather station ID
            hours_ahead: Number of hours to forecast ahead
            
        Returns:
            List of forecast data points
        """
        pass
    
    @abstractmethod
    async def get_weather_alerts(self, station_id: str) -> List[Dict[str, Any]]:
        """
        Get weather alerts for a station
        
        Args:
            station_id: Weather station ID
            
        Returns:
            List of weather alerts
        """
        pass
    
    @abstractmethod
    async def calculate_energy_demand_factor(self, observation: WeatherObservation) -> float:
        """
        Calculate energy demand factor based on weather conditions
        
        Args:
            observation: Weather observation
            
        Returns:
            Energy demand factor (0.5 to 2.0)
        """
        pass
    
    @abstractmethod
    async def get_weather_trends(self, station_id: str, days: int) -> Dict[str, Any]:
        """
        Get weather trends for a station
        
        Args:
            station_id: Weather station ID
            days: Number of days to analyze
            
        Returns:
            Weather trends dictionary
        """
        pass
    
    @abstractmethod
    async def get_climate_normals(self, station_id: str) -> Dict[str, Any]:
        """
        Get climate normals for a station
        
        Args:
            station_id: Weather station ID
            
        Returns:
            Climate normals dictionary
        """
        pass
    
    @abstractmethod
    async def detect_weather_anomalies(self, observation: WeatherObservation) -> List[Dict[str, Any]]:
        """
        Detect weather anomalies in an observation
        
        Args:
            observation: Weather observation to analyze
            
        Returns:
            List of detected anomalies
        """
        pass
    
    @abstractmethod
    async def get_weather_correlation_analysis(self, station_id: str, days: int) -> Dict[str, Any]:
        """
        Get weather correlation analysis for energy demand
        
        Args:
            station_id: Weather station ID
            days: Number of days to analyze
            
        Returns:
            Correlation analysis dictionary
        """
        pass
