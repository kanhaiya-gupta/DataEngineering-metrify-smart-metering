"""
Weather Service
Business logic for weather station operations
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..domain.entities.weather_station import WeatherStation, WeatherObservation
from ..domain.enums.weather_station_status import WeatherStationStatus
from ..interfaces.repositories.weather_station_repository import IWeatherStationRepository
from ..interfaces.external.weather_data_service import IWeatherDataService
from ..interfaces.external.alerting_service import IAlertingService
from ..exceptions.domain_exceptions import WeatherStationNotFoundError, InvalidWeatherOperationError


@dataclass
class WeatherStationRegistrationRequest:
    """Request to register a new weather station"""
    station_id: str
    name: str
    station_type: str
    location: 'Location'
    operator: str
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None


@dataclass
class WeatherObservationRequest:
    """Request to record a weather observation"""
    station_id: str
    timestamp: datetime
    temperature_celsius: float
    humidity_percent: float
    pressure_hpa: float
    wind_speed_ms: float
    wind_direction_degrees: float
    cloud_cover_percent: float
    visibility_km: float
    uv_index: Optional[float] = None
    precipitation_mm: Optional[float] = None


class WeatherService:
    """
    Weather Service
    
    Contains all business logic related to weather station operations.
    This service orchestrates domain entities and coordinates with external services.
    """
    
    def __init__(
        self,
        weather_repository: IWeatherStationRepository,
        weather_data_service: IWeatherDataService,
        alerting_service: IAlertingService
    ):
        self.weather_repository = weather_repository
        self.weather_data_service = weather_data_service
        self.alerting_service = alerting_service
    
    async def register_station(self, request: WeatherStationRegistrationRequest) -> WeatherStation:
        """
        Register a new weather station
        
        Args:
            request: Weather station registration request
            
        Returns:
            Registered weather station
            
        Raises:
            InvalidWeatherOperationError: If station ID already exists
        """
        # Check if station already exists
        existing_station = await self.weather_repository.get_by_id(request.station_id)
        if existing_station:
            raise InvalidWeatherOperationError(f"Weather station {request.station_id} already exists")
        
        # Create new station
        station = WeatherStation(
            station_id=request.station_id,
            name=request.name,
            station_type=request.station_type,
            location=request.location,
            operator=request.operator,
            contact_email=request.contact_email,
            contact_phone=request.contact_phone
        )
        
        # Save to repository
        await self.weather_repository.save(station)
        
        return station
    
    async def get_station(self, station_id: str) -> WeatherStation:
        """
        Get a weather station by ID
        
        Args:
            station_id: Station ID
            
        Returns:
            Weather station
            
        Raises:
            WeatherStationNotFoundError: If station doesn't exist
        """
        station = await self.weather_repository.get_by_id(station_id)
        if not station:
            raise WeatherStationNotFoundError(f"Weather station {station_id} not found")
        
        return station
    
    async def record_observation(self, request: WeatherObservationRequest) -> WeatherStation:
        """
        Record a new weather observation
        
        Args:
            request: Weather observation request
            
        Returns:
            Updated weather station
            
        Raises:
            WeatherStationNotFoundError: If station doesn't exist
            InvalidWeatherOperationError: If station cannot record observations
        """
        # Get the station
        station = await self.get_station(request.station_id)
        
        # Create weather observation
        observation = WeatherObservation(
            timestamp=request.timestamp,
            temperature_celsius=request.temperature_celsius,
            humidity_percent=request.humidity_percent,
            pressure_hpa=request.pressure_hpa,
            wind_speed_ms=request.wind_speed_ms,
            wind_direction_degrees=request.wind_direction_degrees,
            cloud_cover_percent=request.cloud_cover_percent,
            visibility_km=request.visibility_km,
            uv_index=request.uv_index,
            precipitation_mm=request.precipitation_mm
        )
        
        # Calculate data quality score
        quality_score = await self.weather_data_service.calculate_quality_score(observation)
        observation.data_quality_score = quality_score
        
        # Record the observation
        station.record_observation(observation)
        
        # Save updated station
        await self.weather_repository.save(station)
        
        return station
    
    async def deactivate_station(self, station_id: str, reason: str) -> WeatherStation:
        """
        Deactivate a weather station
        
        Args:
            station_id: Station ID
            reason: Reason for deactivation
            
        Returns:
            Deactivated weather station
        """
        station = await self.get_station(station_id)
        station.deactivate_station(reason)
        
        await self.weather_repository.save(station)
        return station
    
    async def reactivate_station(self, station_id: str) -> WeatherStation:
        """
        Reactivate a weather station
        
        Args:
            station_id: Station ID
            
        Returns:
            Reactivated weather station
        """
        station = await self.get_station(station_id)
        station.reactivate_station()
        
        await self.weather_repository.save(station)
        return station
    
    async def get_stations_by_status(self, status: WeatherStationStatus) -> List[WeatherStation]:
        """
        Get all stations with a specific status
        
        Args:
            status: Station status
            
        Returns:
            List of stations with the specified status
        """
        return await self.weather_repository.get_by_status(status)
    
    async def get_stations_requiring_attention(self) -> List[WeatherStation]:
        """
        Get all stations that require attention
        
        Returns:
            List of stations requiring attention
        """
        all_stations = await self.weather_repository.get_all()
        return [station for station in all_stations if station.requires_attention()]
    
    async def get_operational_stations(self) -> List[WeatherStation]:
        """
        Get all operational stations
        
        Returns:
            List of operational stations
        """
        all_stations = await self.weather_repository.get_all()
        return [station for station in all_stations if station.is_operational()]
    
    async def get_station_performance_summary(self, station_id: str) -> Dict[str, Any]:
        """
        Get performance summary for a station
        
        Args:
            station_id: Station ID
            
        Returns:
            Performance summary dictionary
        """
        station = await self.get_station(station_id)
        
        return {
            "station_id": station.station_id,
            "name": station.name,
            "status": station.status.value,
            "total_observations": station.total_observations,
            "average_quality_score": station.average_quality_score,
            "is_operational": station.is_operational(),
            "requires_attention": station.requires_attention(),
            "last_observation_at": station.last_observation_at.isoformat() if station.last_observation_at else None,
            "created_at": station.created_at.isoformat(),
            "updated_at": station.updated_at.isoformat()
        }
    
    async def get_weather_impact_analysis(self, station_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Get weather impact analysis for energy demand
        
        Args:
            station_id: Station ID
            days: Number of days to analyze
            
        Returns:
            Weather impact analysis dictionary
        """
        station = await self.get_station(station_id)
        
        # Get recent observations
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        observations = await self.weather_repository.get_observations_by_date_range(
            station_id, start_date, end_date
        )
        
        if not observations:
            return {
                "station_id": station_id,
                "analysis_period_days": days,
                "total_observations": 0,
                "average_energy_demand_factor": 1.0,
                "weather_categories": {},
                "temperature_impact": {},
                "humidity_impact": {},
                "wind_impact": {}
            }
        
        # Calculate energy demand factors
        demand_factors = [station.calculate_energy_demand_factor(obs) for obs in observations]
        avg_demand_factor = sum(demand_factors) / len(demand_factors)
        
        # Analyze weather categories
        weather_categories = {}
        for obs in observations:
            category = station.get_weather_category(obs)
            weather_categories[category] = weather_categories.get(category, 0) + 1
        
        # Analyze temperature impact
        temperatures = [obs.temperature_celsius for obs in observations]
        temp_impact = {
            "min_temperature": min(temperatures),
            "max_temperature": max(temperatures),
            "avg_temperature": sum(temperatures) / len(temperatures),
            "heating_demand_days": len([t for t in temperatures if t < 15]),
            "cooling_demand_days": len([t for t in temperatures if t > 25])
        }
        
        # Analyze humidity impact
        humidities = [obs.humidity_percent for obs in observations]
        humidity_impact = {
            "min_humidity": min(humidities),
            "max_humidity": max(humidities),
            "avg_humidity": sum(humidities) / len(humidities),
            "high_humidity_days": len([h for h in humidities if h > 80])
        }
        
        # Analyze wind impact
        wind_speeds = [obs.wind_speed_ms for obs in observations]
        wind_impact = {
            "min_wind_speed": min(wind_speeds),
            "max_wind_speed": max(wind_speeds),
            "avg_wind_speed": sum(wind_speeds) / len(wind_speeds),
            "high_wind_days": len([w for w in wind_speeds if w > 10])
        }
        
        return {
            "station_id": station_id,
            "analysis_period_days": days,
            "total_observations": len(observations),
            "average_energy_demand_factor": avg_demand_factor,
            "weather_categories": weather_categories,
            "temperature_impact": temp_impact,
            "humidity_impact": humidity_impact,
            "wind_impact": wind_impact
        }
    
    async def get_system_weather_summary(self) -> Dict[str, Any]:
        """
        Get overall weather system summary
        
        Returns:
            Weather system summary dictionary
        """
        all_stations = await self.weather_repository.get_all()
        
        if not all_stations:
            return {
                "total_stations": 0,
                "operational_stations": 0,
                "average_quality_score": 0.0,
                "stations_requiring_attention": 0,
                "total_observations": 0
            }
        
        operational_stations = [s for s in all_stations if s.is_operational()]
        stations_requiring_attention = [s for s in all_stations if s.requires_attention()]
        
        average_quality_score = sum(s.average_quality_score for s in all_stations) / len(all_stations)
        total_observations = sum(s.total_observations for s in all_stations)
        
        return {
            "total_stations": len(all_stations),
            "operational_stations": len(operational_stations),
            "average_quality_score": average_quality_score,
            "stations_requiring_attention": len(stations_requiring_attention),
            "total_observations": total_observations,
            "status_distribution": {
                status.value: len([s for s in all_stations if s.status == status])
                for status in WeatherStationStatus
            }
        }
