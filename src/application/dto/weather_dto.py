"""
Weather DTOs
Data Transfer Objects for weather station related API communication
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class WeatherStationStatusDTO(str, Enum):
    """Weather station status enumeration for DTOs"""
    ACTIVE = "ACTIVE"
    MAINTENANCE = "MAINTENANCE"
    CALIBRATION = "CALIBRATION"
    ERROR = "ERROR"
    OFFLINE = "OFFLINE"
    MALFUNCTION = "MALFUNCTION"
    SENSOR_ERROR = "SENSOR_ERROR"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


class WeatherStationTypeDTO(str, Enum):
    """Weather station type enumeration for DTOs"""
    AUTOMATIC = "AUTOMATIC"
    MANUAL = "MANUAL"
    MOBILE = "MOBILE"
    SATELLITE = "SATELLITE"
    RADAR = "RADAR"
    SONDE = "SONDE"


class LocationDTO(BaseModel):
    """Location data transfer object"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    address: str = Field(..., min_length=1, description="Physical address")
    
    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "address": "123 Main St, New York, NY 10001"
            }
        }


class WeatherStationCreateDTO(BaseModel):
    """DTO for creating a new weather station"""
    station_id: str = Field(..., min_length=1, description="Unique station identifier")
    name: str = Field(..., min_length=1, description="Station name")
    station_type: WeatherStationTypeDTO = Field(..., description="Type of weather station")
    location: LocationDTO = Field(..., description="Station location")
    operator: str = Field(..., min_length=1, description="Station operator")
    contact_email: Optional[str] = Field(default=None, description="Contact email")
    contact_phone: Optional[str] = Field(default=None, description="Contact phone")
    status: WeatherStationStatusDTO = Field(default=WeatherStationStatusDTO.ACTIVE, description="Station status")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "station_id": "WS001",
                "name": "Central Park Weather Station",
                "station_type": "AUTOMATIC",
                "location": {
                    "latitude": 40.7829,
                    "longitude": -73.9654,
                    "address": "Central Park, New York, NY 10024"
                },
                "operator": "NYC Weather Service",
                "contact_email": "weather@nyc.gov",
                "contact_phone": "+1-555-0123",
                "status": "ACTIVE",
                "metadata": {"zone": "manhattan", "priority": "high"}
            }
        }


class WeatherStationUpdateDTO(BaseModel):
    """DTO for updating an existing weather station"""
    name: Optional[str] = Field(default=None, min_length=1, description="Updated name")
    location: Optional[LocationDTO] = Field(default=None, description="Updated location")
    operator: Optional[str] = Field(default=None, min_length=1, description="Updated operator")
    contact_email: Optional[str] = Field(default=None, description="Updated contact email")
    contact_phone: Optional[str] = Field(default=None, description="Updated contact phone")
    status: Optional[WeatherStationStatusDTO] = Field(default=None, description="Updated status")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "MAINTENANCE",
                "metadata": {"last_maintenance": "2024-01-20T14:00:00Z"}
            }
        }


class WeatherStationResponseDTO(BaseModel):
    """DTO for weather station response"""
    station_id: str = Field(..., description="Station identifier")
    name: str = Field(..., description="Station name")
    station_type: WeatherStationTypeDTO = Field(..., description="Type of weather station")
    location: LocationDTO = Field(..., description="Station location")
    operator: str = Field(..., description="Station operator")
    contact_email: Optional[str] = Field(default=None, description="Contact email")
    contact_phone: Optional[str] = Field(default=None, description="Contact phone")
    status: WeatherStationStatusDTO = Field(..., description="Station status")
    total_observations: int = Field(..., ge=0, description="Total number of observations")
    average_quality_score: float = Field(..., ge=0, le=1, description="Average quality score")
    last_observation_at: Optional[datetime] = Field(default=None, description="Last observation timestamp")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    version: int = Field(..., description="Entity version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "station_id": "WS001",
                "name": "Central Park Weather Station",
                "station_type": "AUTOMATIC",
                "location": {
                    "latitude": 40.7829,
                    "longitude": -73.9654,
                    "address": "Central Park, New York, NY 10024"
                },
                "operator": "NYC Weather Service",
                "contact_email": "weather@nyc.gov",
                "contact_phone": "+1-555-0123",
                "status": "ACTIVE",
                "total_observations": 50000,
                "average_quality_score": 0.95,
                "last_observation_at": "2024-01-20T15:45:00Z",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-20T15:45:00Z",
                "metadata": {"zone": "manhattan", "priority": "high"},
                "version": 1
            }
        }


class WeatherObservationCreateDTO(BaseModel):
    """DTO for creating a new weather observation"""
    station_id: str = Field(..., min_length=1, description="Station identifier")
    timestamp: datetime = Field(..., description="Observation timestamp")
    temperature_celsius: float = Field(..., ge=-50, le=60, description="Temperature in Celsius")
    humidity_percent: float = Field(..., ge=0, le=100, description="Humidity percentage")
    pressure_hpa: float = Field(..., ge=950, le=1050, description="Pressure in hPa")
    wind_speed_ms: float = Field(..., ge=0, le=100, description="Wind speed in m/s")
    wind_direction_degrees: float = Field(..., ge=0, le=360, description="Wind direction in degrees")
    cloud_cover_percent: float = Field(..., ge=0, le=100, description="Cloud cover percentage")
    visibility_km: float = Field(..., ge=0, le=50, description="Visibility in km")
    uv_index: Optional[float] = Field(default=None, ge=0, le=15, description="UV index")
    precipitation_mm: Optional[float] = Field(default=None, ge=0, description="Precipitation in mm")
    
    class Config:
        json_schema_extra = {
            "example": {
                "station_id": "WS001",
                "timestamp": "2024-01-20T15:45:00Z",
                "temperature_celsius": 22.5,
                "humidity_percent": 65.0,
                "pressure_hpa": 1013.25,
                "wind_speed_ms": 5.2,
                "wind_direction_degrees": 180.0,
                "cloud_cover_percent": 30.0,
                "visibility_km": 10.0,
                "uv_index": 6.5,
                "precipitation_mm": 0.0
            }
        }


class WeatherObservationResponseDTO(BaseModel):
    """DTO for weather observation response"""
    id: str = Field(..., description="Observation identifier")
    station_id: str = Field(..., description="Station identifier")
    timestamp: datetime = Field(..., description="Observation timestamp")
    temperature_celsius: float = Field(..., description="Temperature in Celsius")
    humidity_percent: float = Field(..., description="Humidity percentage")
    pressure_hpa: float = Field(..., description="Pressure in hPa")
    wind_speed_ms: float = Field(..., description="Wind speed in m/s")
    wind_direction_degrees: float = Field(..., description="Wind direction in degrees")
    cloud_cover_percent: float = Field(..., description="Cloud cover percentage")
    visibility_km: float = Field(..., description="Visibility in km")
    uv_index: Optional[float] = Field(default=None, description="UV index")
    precipitation_mm: Optional[float] = Field(default=None, description="Precipitation in mm")
    data_quality_score: float = Field(..., ge=0, le=1, description="Data quality score")
    is_anomaly: bool = Field(..., description="Whether observation is an anomaly")
    anomaly_type: Optional[str] = Field(default=None, description="Type of anomaly")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "obs_123",
                "station_id": "WS001",
                "timestamp": "2024-01-20T15:45:00Z",
                "temperature_celsius": 22.5,
                "humidity_percent": 65.0,
                "pressure_hpa": 1013.25,
                "wind_speed_ms": 5.2,
                "wind_direction_degrees": 180.0,
                "cloud_cover_percent": 30.0,
                "visibility_km": 10.0,
                "uv_index": 6.5,
                "precipitation_mm": 0.0,
                "data_quality_score": 0.95,
                "is_anomaly": False,
                "anomaly_type": None,
                "created_at": "2024-01-20T15:45:00Z"
            }
        }


class WeatherObservationBatchDTO(BaseModel):
    """DTO for batch weather observation ingestion"""
    station_id: str = Field(..., min_length=1, description="Station identifier")
    observations: List[WeatherObservationCreateDTO] = Field(..., min_items=1, description="List of observations")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Batch metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "station_id": "WS001",
                "observations": [
                    {
                        "station_id": "WS001",
                        "timestamp": "2024-01-20T15:45:00Z",
                        "temperature_celsius": 22.5,
                        "humidity_percent": 65.0,
                        "pressure_hpa": 1013.25,
                        "wind_speed_ms": 5.2,
                        "wind_direction_degrees": 180.0,
                        "cloud_cover_percent": 30.0,
                        "visibility_km": 10.0,
                        "uv_index": 6.5,
                        "precipitation_mm": 0.0
                    }
                ],
                "metadata": {"batch_id": "batch_001", "source": "api"}
            }
        }


class WeatherStationListResponseDTO(BaseModel):
    """DTO for weather station list response"""
    stations: List[WeatherStationResponseDTO] = Field(..., description="List of weather stations")
    total_count: int = Field(..., description="Total number of stations")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")
    
    class Config:
        json_schema_extra = {
            "example": {
                "stations": [],
                "total_count": 100,
                "page": 1,
                "page_size": 20,
                "has_next": True,
                "has_previous": False
            }
        }


class WeatherObservationListResponseDTO(BaseModel):
    """DTO for weather observation list response"""
    observations: List[WeatherObservationResponseDTO] = Field(..., description="List of weather observations")
    total_count: int = Field(..., description="Total number of observations")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")
    
    class Config:
        json_schema_extra = {
            "example": {
                "observations": [],
                "total_count": 10000,
                "page": 1,
                "page_size": 100,
                "has_next": True,
                "has_previous": False
            }
        }


class WeatherStationAnalyticsDTO(BaseModel):
    """DTO for weather station analytics"""
    station_id: str = Field(..., description="Station identifier")
    analysis_period: str = Field(..., description="Analysis period")
    total_observations: int = Field(..., description="Total number of observations")
    avg_temperature: float = Field(..., description="Average temperature")
    avg_humidity: float = Field(..., description="Average humidity")
    avg_pressure: float = Field(..., description="Average pressure")
    avg_wind_speed: float = Field(..., description="Average wind speed")
    avg_cloud_cover: float = Field(..., description="Average cloud cover")
    avg_visibility: float = Field(..., description="Average visibility")
    avg_quality_score: float = Field(..., description="Average quality score")
    anomaly_count: int = Field(..., description="Number of anomalies detected")
    anomaly_rate: float = Field(..., description="Anomaly rate percentage")
    max_temperature: float = Field(..., description="Maximum temperature")
    min_temperature: float = Field(..., description="Minimum temperature")
    max_wind_speed: float = Field(..., description="Maximum wind speed")
    total_precipitation: float = Field(..., description="Total precipitation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "station_id": "WS001",
                "analysis_period": "24h",
                "total_observations": 1440,
                "avg_temperature": 22.5,
                "avg_humidity": 65.0,
                "avg_pressure": 1013.25,
                "avg_wind_speed": 5.2,
                "avg_cloud_cover": 30.0,
                "avg_visibility": 10.0,
                "avg_quality_score": 0.95,
                "anomaly_count": 8,
                "anomaly_rate": 0.56,
                "max_temperature": 28.5,
                "min_temperature": 18.2,
                "max_wind_speed": 15.8,
                "total_precipitation": 2.5
            }
        }


class WeatherStationSearchDTO(BaseModel):
    """DTO for weather station search parameters"""
    station_id: Optional[str] = Field(default=None, description="Filter by station ID")
    name: Optional[str] = Field(default=None, description="Filter by name")
    station_type: Optional[WeatherStationTypeDTO] = Field(default=None, description="Filter by station type")
    operator: Optional[str] = Field(default=None, description="Filter by operator")
    status: Optional[WeatherStationStatusDTO] = Field(default=None, description="Filter by status")
    location_radius: Optional[float] = Field(default=None, ge=0, description="Search radius in km")
    latitude: Optional[float] = Field(default=None, ge=-90, le=90, description="Center latitude")
    longitude: Optional[float] = Field(default=None, ge=-180, le=180, description="Center longitude")
    min_quality_score: Optional[float] = Field(default=None, ge=0, le=1, description="Minimum quality score")
    max_quality_score: Optional[float] = Field(default=None, ge=0, le=1, description="Maximum quality score")
    created_after: Optional[datetime] = Field(default=None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(default=None, description="Filter by creation date")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Page size")
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$", description="Sort order")
    
    class Config:
        json_schema_extra = {
            "example": {
                "station_type": "AUTOMATIC",
                "status": "ACTIVE",
                "location_radius": 25.0,
                "latitude": 40.7128,
                "longitude": -74.0060,
                "min_quality_score": 0.8,
                "page": 1,
                "page_size": 20,
                "sort_by": "created_at",
                "sort_order": "desc"
            }
        }


class WeatherForecastDTO(BaseModel):
    """DTO for weather forecast"""
    station_id: str = Field(..., description="Station identifier")
    forecast_hours: int = Field(..., description="Number of forecast hours")
    forecasts: List[Dict[str, Any]] = Field(..., description="List of forecast data")
    confidence: float = Field(..., ge=0, le=1, description="Forecast confidence")
    generated_at: datetime = Field(..., description="Forecast generation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "station_id": "WS001",
                "forecast_hours": 24,
                "forecasts": [
                    {
                        "timestamp": "2024-01-21T00:00:00Z",
                        "temperature_celsius": 20.0,
                        "humidity_percent": 70.0,
                        "pressure_hpa": 1015.0,
                        "wind_speed_ms": 4.5,
                        "wind_direction_degrees": 200.0,
                        "cloud_cover_percent": 40.0,
                        "precipitation_mm": 0.0,
                        "confidence": 0.85
                    }
                ],
                "confidence": 0.85,
                "generated_at": "2024-01-20T15:45:00Z"
            }
        }
