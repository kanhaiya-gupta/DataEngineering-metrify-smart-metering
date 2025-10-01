"""
Weather Data Service Implementation
Concrete implementation of IWeatherDataService using external APIs
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from ....core.domain.entities.weather_station import WeatherObservation
from ....core.interfaces.external.weather_data_service import IWeatherDataService
from ....core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class WeatherDataService(IWeatherDataService):
    """
    Weather Data Service Implementation
    
    Provides weather data operations by integrating with external
    weather APIs and data sources.
    """
    
    def __init__(
        self,
        api_base_url: str = "https://api.weather-service.com",
        api_key: str = "",
        timeout: int = 30
    ):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._stations_cache = {}
        self._last_cache_update = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'Metrify-SmartMetering/1.0'
            }
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def calculate_quality_score(self, observation: WeatherObservation) -> float:
        """
        Calculate data quality score for a weather observation
        
        Args:
            observation: Weather observation to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            quality_factors = []
            
            # Temperature validity check
            if -50 <= observation.temperature_celsius <= 60:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)
            
            # Humidity validity check
            if 0 <= observation.humidity_percent <= 100:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)
            
            # Pressure validity check
            if 950 <= observation.pressure_hpa <= 1050:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)
            
            # Wind speed validity check
            if 0 <= observation.wind_speed_ms <= 100:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)
            
            # Wind direction validity check
            if 0 <= observation.wind_direction_degrees <= 360:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)
            
            # Cloud cover validity check
            if 0 <= observation.cloud_cover_percent <= 100:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)
            
            # Visibility validity check
            if 0 <= observation.visibility_km <= 50:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)
            
            # Calculate average quality score
            quality_score = sum(quality_factors) / len(quality_factors)
            
            logger.debug(f"Calculated weather quality score: {quality_score}")
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculating weather quality score: {str(e)}")
            return 0.5  # Default moderate score
    
    async def validate_observation(self, observation: WeatherObservation) -> Dict[str, Any]:
        """
        Validate a weather observation against quality rules
        
        Args:
            observation: Weather observation to validate
            
        Returns:
            Validation result dictionary
        """
        try:
            violations = []
            recommendations = []
            
            # Temperature validation
            if not (-50 <= observation.temperature_celsius <= 60):
                violations.append(f"Temperature {observation.temperature_celsius}°C is outside valid range (-50°C to 60°C)")
            
            # Humidity validation
            if not (0 <= observation.humidity_percent <= 100):
                violations.append(f"Humidity {observation.humidity_percent}% is outside valid range (0% to 100%)")
            
            # Pressure validation
            if not (950 <= observation.pressure_hpa <= 1050):
                violations.append(f"Pressure {observation.pressure_hpa}hPa is outside valid range (950hPa to 1050hPa)")
            
            # Wind speed validation
            if not (0 <= observation.wind_speed_ms <= 100):
                violations.append(f"Wind speed {observation.wind_speed_ms}m/s is outside valid range (0m/s to 100m/s)")
            
            # Wind direction validation
            if not (0 <= observation.wind_direction_degrees <= 360):
                violations.append(f"Wind direction {observation.wind_direction_degrees}° is outside valid range (0° to 360°)")
            
            # Cloud cover validation
            if not (0 <= observation.cloud_cover_percent <= 100):
                violations.append(f"Cloud cover {observation.cloud_cover_percent}% is outside valid range (0% to 100%)")
            
            # Visibility validation
            if not (0 <= observation.visibility_km <= 50):
                violations.append(f"Visibility {observation.visibility_km}km is outside valid range (0km to 50km)")
            
            # Generate recommendations
            if violations:
                recommendations.append("Check sensor calibration")
                recommendations.append("Verify data collection process")
            
            quality_score = await self.calculate_quality_score(observation)
            
            return {
                'is_valid': len(violations) == 0,
                'violations': violations,
                'quality_score': quality_score,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error validating weather observation: {str(e)}")
            raise InfrastructureError(f"Failed to validate observation: {str(e)}", service="weather_api")
    
    async def fetch_weather_data(
        self, 
        station_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[WeatherObservation]:
        """
        Fetch weather data from external source
        
        Args:
            station_id: Weather station ID
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            List of weather observations
        """
        try:
            session = await self._get_session()
            url = f"{self.api_base_url}/stations/{station_id}/observations"
            
            params = {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'limit': 1000
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [self._parse_weather_observation(item) for item in data.get('observations', [])]
                else:
                    logger.error(f"Failed to fetch weather data: {response.status}")
                    return []
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching weather data for station {station_id}")
            raise InfrastructureError(f"Timeout fetching weather data", service="weather_api")
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            raise InfrastructureError(f"Failed to fetch weather data: {str(e)}", service="weather_api")
    
    async def get_weather_forecast(self, station_id: str, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """
        Get weather forecast for a station
        
        Args:
            station_id: Weather station ID
            hours_ahead: Number of hours to forecast ahead
            
        Returns:
            List of forecast data points
        """
        try:
            session = await self._get_session()
            url = f"{self.api_base_url}/stations/{station_id}/forecast"
            
            params = {
                'hours_ahead': hours_ahead
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('forecast', [])
                else:
                    logger.error(f"Failed to fetch weather forecast: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching weather forecast: {str(e)}")
            raise InfrastructureError(f"Failed to fetch weather forecast: {str(e)}", service="weather_api")
    
    async def get_weather_alerts(self, station_id: str) -> List[Dict[str, Any]]:
        """
        Get weather alerts for a station
        
        Args:
            station_id: Weather station ID
            
        Returns:
            List of weather alerts
        """
        try:
            session = await self._get_session()
            url = f"{self.api_base_url}/stations/{station_id}/alerts"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('alerts', [])
                else:
                    logger.error(f"Failed to fetch weather alerts: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching weather alerts: {str(e)}")
            raise InfrastructureError(f"Failed to fetch weather alerts: {str(e)}", service="weather_api")
    
    async def calculate_energy_demand_factor(self, observation: WeatherObservation) -> float:
        """
        Calculate energy demand factor based on weather conditions
        
        Args:
            observation: Weather observation
            
        Returns:
            Energy demand factor (0.5 to 2.0)
        """
        try:
            factor = 1.0
            
            # Temperature effect (heating/cooling demand)
            temp = observation.temperature_celsius
            if temp < 15:  # Heating demand
                factor += (15 - temp) * 0.02
            elif temp > 25:  # Cooling demand
                factor += (temp - 25) * 0.03
            
            # Humidity effect
            if observation.humidity_percent > 80:
                factor += 0.1  # Higher humidity increases energy demand
            
            # Wind effect (wind chill/heat index)
            if observation.wind_speed_ms > 10:
                if temp < 10:
                    factor += 0.05  # Wind chill increases heating demand
                elif temp > 25:
                    factor += 0.03  # Wind can reduce cooling demand
            
            # Cloud cover effect (affects solar generation)
            if observation.cloud_cover_percent > 70:
                factor += 0.05  # Less solar generation, more grid demand
            
            # Clamp between 0.5 and 2.0
            return max(0.5, min(2.0, factor))
            
        except Exception as e:
            logger.error(f"Error calculating energy demand factor: {str(e)}")
            return 1.0  # Default neutral factor
    
    async def get_weather_trends(self, station_id: str, days: int) -> Dict[str, Any]:
        """
        Get weather trends for a station
        
        Args:
            station_id: Weather station ID
            days: Number of days to analyze
            
        Returns:
            Weather trends dictionary
        """
        try:
            session = await self._get_session()
            url = f"{self.api_base_url}/stations/{station_id}/trends"
            
            params = {
                'days': days
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch weather trends: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching weather trends: {str(e)}")
            raise InfrastructureError(f"Failed to fetch weather trends: {str(e)}", service="weather_api")
    
    async def get_climate_normals(self, station_id: str) -> Dict[str, Any]:
        """
        Get climate normals for a station
        
        Args:
            station_id: Weather station ID
            
        Returns:
            Climate normals dictionary
        """
        try:
            session = await self._get_session()
            url = f"{self.api_base_url}/stations/{station_id}/normals"
            
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch climate normals: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching climate normals: {str(e)}")
            raise InfrastructureError(f"Failed to fetch climate normals: {str(e)}", service="weather_api")
    
    async def detect_weather_anomalies(self, observation: WeatherObservation) -> List[Dict[str, Any]]:
        """
        Detect weather anomalies in an observation
        
        Args:
            observation: Weather observation to analyze
            
        Returns:
            List of detected anomalies
        """
        try:
            anomalies = []
            
            # Temperature anomalies
            if observation.temperature_celsius < -30 or observation.temperature_celsius > 50:
                anomalies.append({
                    'type': 'temperature_anomaly',
                    'description': f'Extreme temperature: {observation.temperature_celsius}°C',
                    'severity': 'high',
                    'confidence': 0.9
                })
            
            # Humidity anomalies
            if observation.humidity_percent > 95:
                anomalies.append({
                    'type': 'humidity_anomaly',
                    'description': f'Very high humidity: {observation.humidity_percent}%',
                    'severity': 'medium',
                    'confidence': 0.8
                })
            
            # Pressure anomalies
            if observation.pressure_hpa < 980 or observation.pressure_hpa > 1030:
                anomalies.append({
                    'type': 'pressure_anomaly',
                    'description': f'Pressure anomaly: {observation.pressure_hpa}hPa',
                    'severity': 'high',
                    'confidence': 0.9
                })
            
            # Wind speed anomalies
            if observation.wind_speed_ms > 30:
                anomalies.append({
                    'type': 'wind_anomaly',
                    'description': f'High wind speed: {observation.wind_speed_ms}m/s',
                    'severity': 'high',
                    'confidence': 0.9
                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting weather anomalies: {str(e)}")
            return []
    
    async def get_weather_correlation_analysis(self, station_id: str, days: int) -> Dict[str, Any]:
        """
        Get weather correlation analysis for energy demand
        
        Args:
            station_id: Weather station ID
            days: Number of days to analyze
            
        Returns:
            Correlation analysis dictionary
        """
        try:
            session = await self._get_session()
            url = f"{self.api_base_url}/stations/{station_id}/correlation"
            
            params = {
                'days': days
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch correlation analysis: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching correlation analysis: {str(e)}")
            raise InfrastructureError(f"Failed to fetch correlation analysis: {str(e)}", service="weather_api")
    
    def _parse_weather_observation(self, data: Dict[str, Any]) -> WeatherObservation:
        """Parse API response into WeatherObservation object"""
        try:
            return WeatherObservation(
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat())),
                temperature_celsius=data.get('temperature_celsius', 0.0),
                humidity_percent=data.get('humidity_percent', 0.0),
                pressure_hpa=data.get('pressure_hpa', 0.0),
                wind_speed_ms=data.get('wind_speed_ms', 0.0),
                wind_direction_degrees=data.get('wind_direction_degrees', 0.0),
                cloud_cover_percent=data.get('cloud_cover_percent', 0.0),
                visibility_km=data.get('visibility_km', 0.0),
                uv_index=data.get('uv_index'),
                precipitation_mm=data.get('precipitation_mm'),
                data_quality_score=data.get('data_quality_score', 1.0)
            )
        except Exception as e:
            logger.error(f"Error parsing weather observation: {str(e)}")
            # Return a default observation
            return WeatherObservation(
                timestamp=datetime.utcnow(),
                temperature_celsius=20.0,
                humidity_percent=50.0,
                pressure_hpa=1013.25,
                wind_speed_ms=0.0,
                wind_direction_degrees=0.0,
                cloud_cover_percent=0.0,
                visibility_km=10.0
            )
