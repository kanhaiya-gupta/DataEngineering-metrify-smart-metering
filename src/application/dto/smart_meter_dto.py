"""
Smart Meter DTOs
Data Transfer Objects for smart meter related API communication
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class MeterStatusDTO(str, Enum):
    """Meter status enumeration for DTOs"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    FAULTY = "FAULTY"
    MAINTENANCE = "MAINTENANCE"


class QualityTierDTO(str, Enum):
    """Quality tier enumeration for DTOs"""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    UNKNOWN = "UNKNOWN"


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


class MeterSpecificationsDTO(BaseModel):
    """Meter specifications data transfer object"""
    manufacturer: str = Field(..., min_length=1, description="Meter manufacturer")
    model: str = Field(..., min_length=1, description="Meter model")
    firmware_version: str = Field(default="1.0.0", description="Firmware version")
    installation_date: datetime = Field(..., description="Installation date")
    
    class Config:
        json_schema_extra = {
            "example": {
                "manufacturer": "Siemens",
                "model": "7KT1546",
                "firmware_version": "2.1.3",
                "installation_date": "2024-01-15T10:30:00Z"
            }
        }


class SmartMeterCreateDTO(BaseModel):
    """DTO for creating a new smart meter"""
    meter_id: str = Field(..., min_length=1, description="Unique meter identifier")
    location: LocationDTO = Field(..., description="Meter location")
    specifications: MeterSpecificationsDTO = Field(..., description="Meter specifications")
    status: MeterStatusDTO = Field(default=MeterStatusDTO.ACTIVE, description="Meter status")
    quality_tier: QualityTierDTO = Field(default=QualityTierDTO.UNKNOWN, description="Quality tier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "meter_id": "SM001",
                "location": {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "address": "123 Main St, New York, NY 10001"
                },
                "specifications": {
                    "manufacturer": "Siemens",
                    "model": "7KT1546",
                    "firmware_version": "2.1.3",
                    "installation_date": "2024-01-15T10:30:00Z"
                },
                "status": "ACTIVE",
                "quality_tier": "UNKNOWN",
                "metadata": {"zone": "downtown", "priority": "high"}
            }
        }


class SmartMeterUpdateDTO(BaseModel):
    """DTO for updating an existing smart meter"""
    location: Optional[LocationDTO] = Field(default=None, description="Updated location")
    status: Optional[MeterStatusDTO] = Field(default=None, description="Updated status")
    quality_tier: Optional[QualityTierDTO] = Field(default=None, description="Updated quality tier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "MAINTENANCE",
                "quality_tier": "GOOD",
                "metadata": {"last_maintenance": "2024-01-20T14:00:00Z"}
            }
        }


class SmartMeterResponseDTO(BaseModel):
    """DTO for smart meter response"""
    meter_id: str = Field(..., description="Meter identifier")
    location: LocationDTO = Field(..., description="Meter location")
    specifications: MeterSpecificationsDTO = Field(..., description="Meter specifications")
    status: MeterStatusDTO = Field(..., description="Meter status")
    quality_tier: QualityTierDTO = Field(..., description="Quality tier")
    installed_at: datetime = Field(..., description="Installation timestamp")
    last_reading_at: Optional[datetime] = Field(default=None, description="Last reading timestamp")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    version: int = Field(..., description="Entity version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "meter_id": "SM001",
                "location": {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "address": "123 Main St, New York, NY 10001"
                },
                "specifications": {
                    "manufacturer": "Siemens",
                    "model": "7KT1546",
                    "firmware_version": "2.1.3",
                    "installation_date": "2024-01-15T10:30:00Z"
                },
                "status": "ACTIVE",
                "quality_tier": "GOOD",
                "installed_at": "2024-01-15T10:30:00Z",
                "last_reading_at": "2024-01-20T15:45:00Z",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-20T15:45:00Z",
                "metadata": {"zone": "downtown", "priority": "high"},
                "version": 1
            }
        }


class MeterReadingCreateDTO(BaseModel):
    """DTO for creating a new meter reading"""
    meter_id: str = Field(..., min_length=1, description="Meter identifier")
    timestamp: datetime = Field(..., description="Reading timestamp")
    voltage: float = Field(..., ge=0, le=1000, description="Voltage reading")
    current: float = Field(..., ge=0, le=1000, description="Current reading")
    power_factor: float = Field(..., ge=0, le=1, description="Power factor")
    frequency: float = Field(..., ge=45, le=55, description="Frequency reading")
    active_power: float = Field(..., ge=0, description="Active power")
    reactive_power: float = Field(..., ge=0, description="Reactive power")
    apparent_power: float = Field(..., ge=0, description="Apparent power")
    
    class Config:
        json_schema_extra = {
            "example": {
                "meter_id": "SM001",
                "timestamp": "2024-01-20T15:45:00Z",
                "voltage": 230.5,
                "current": 15.2,
                "power_factor": 0.95,
                "frequency": 50.0,
                "active_power": 3500.0,
                "reactive_power": 1200.0,
                "apparent_power": 3684.2
            }
        }


class MeterReadingResponseDTO(BaseModel):
    """DTO for meter reading response"""
    id: str = Field(..., description="Reading identifier")
    meter_id: str = Field(..., description="Meter identifier")
    timestamp: datetime = Field(..., description="Reading timestamp")
    voltage: float = Field(..., description="Voltage reading")
    current: float = Field(..., description="Current reading")
    power_factor: float = Field(..., description="Power factor")
    frequency: float = Field(..., description="Frequency reading")
    active_power: float = Field(..., description="Active power")
    reactive_power: float = Field(..., description="Reactive power")
    apparent_power: float = Field(..., description="Apparent power")
    data_quality_score: float = Field(..., ge=0, le=1, description="Data quality score")
    is_anomaly: bool = Field(..., description="Whether reading is an anomaly")
    anomaly_type: Optional[str] = Field(default=None, description="Type of anomaly")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "reading_123",
                "meter_id": "SM001",
                "timestamp": "2024-01-20T15:45:00Z",
                "voltage": 230.5,
                "current": 15.2,
                "power_factor": 0.95,
                "frequency": 50.0,
                "active_power": 3500.0,
                "reactive_power": 1200.0,
                "apparent_power": 3684.2,
                "data_quality_score": 0.95,
                "is_anomaly": False,
                "anomaly_type": None,
                "created_at": "2024-01-20T15:45:00Z"
            }
        }


class MeterReadingBatchDTO(BaseModel):
    """DTO for batch meter reading ingestion"""
    meter_id: str = Field(..., min_length=1, description="Meter identifier")
    readings: List[MeterReadingCreateDTO] = Field(..., min_items=1, description="List of readings")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Batch metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "meter_id": "SM001",
                "readings": [
                    {
                        "meter_id": "SM001",
                        "timestamp": "2024-01-20T15:45:00Z",
                        "voltage": 230.5,
                        "current": 15.2,
                        "power_factor": 0.95,
                        "frequency": 50.0,
                        "active_power": 3500.0,
                        "reactive_power": 1200.0,
                        "apparent_power": 3684.2
                    }
                ],
                "metadata": {"batch_id": "batch_001", "source": "api"}
            }
        }


class SmartMeterListResponseDTO(BaseModel):
    """DTO for smart meter list response"""
    meters: List[SmartMeterResponseDTO] = Field(..., description="List of smart meters")
    total_count: int = Field(..., description="Total number of meters")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")
    
    class Config:
        json_schema_extra = {
            "example": {
                "meters": [],
                "total_count": 100,
                "page": 1,
                "page_size": 20,
                "has_next": True,
                "has_previous": False
            }
        }


class MeterReadingListResponseDTO(BaseModel):
    """DTO for meter reading list response"""
    readings: List[MeterReadingResponseDTO] = Field(..., description="List of meter readings")
    total_count: int = Field(..., description="Total number of readings")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")
    
    class Config:
        json_schema_extra = {
            "example": {
                "readings": [],
                "total_count": 1000,
                "page": 1,
                "page_size": 50,
                "has_next": True,
                "has_previous": False
            }
        }


class SmartMeterAnalyticsDTO(BaseModel):
    """DTO for smart meter analytics"""
    meter_id: str = Field(..., description="Meter identifier")
    analysis_period: str = Field(..., description="Analysis period")
    total_readings: int = Field(..., description="Total number of readings")
    avg_voltage: float = Field(..., description="Average voltage")
    avg_current: float = Field(..., description="Average current")
    avg_power_factor: float = Field(..., description="Average power factor")
    avg_frequency: float = Field(..., description="Average frequency")
    total_energy_consumed: float = Field(..., description="Total energy consumed")
    avg_quality_score: float = Field(..., description="Average quality score")
    anomaly_count: int = Field(..., description="Number of anomalies detected")
    anomaly_rate: float = Field(..., description="Anomaly rate percentage")
    peak_consumption: float = Field(..., description="Peak consumption")
    peak_consumption_time: Optional[datetime] = Field(default=None, description="Peak consumption time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "meter_id": "SM001",
                "analysis_period": "24h",
                "total_readings": 1440,
                "avg_voltage": 230.2,
                "avg_current": 15.1,
                "avg_power_factor": 0.94,
                "avg_frequency": 50.0,
                "total_energy_consumed": 50400.0,
                "avg_quality_score": 0.95,
                "anomaly_count": 5,
                "anomaly_rate": 0.35,
                "peak_consumption": 4500.0,
                "peak_consumption_time": "2024-01-20T18:30:00Z"
            }
        }


class SmartMeterSearchDTO(BaseModel):
    """DTO for smart meter search parameters"""
    meter_id: Optional[str] = Field(default=None, description="Filter by meter ID")
    status: Optional[MeterStatusDTO] = Field(default=None, description="Filter by status")
    quality_tier: Optional[QualityTierDTO] = Field(default=None, description="Filter by quality tier")
    manufacturer: Optional[str] = Field(default=None, description="Filter by manufacturer")
    model: Optional[str] = Field(default=None, description="Filter by model")
    location_radius: Optional[float] = Field(default=None, ge=0, description="Search radius in km")
    latitude: Optional[float] = Field(default=None, ge=-90, le=90, description="Center latitude")
    longitude: Optional[float] = Field(default=None, ge=-180, le=180, description="Center longitude")
    installed_after: Optional[datetime] = Field(default=None, description="Filter by installation date")
    installed_before: Optional[datetime] = Field(default=None, description="Filter by installation date")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Page size")
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$", description="Sort order")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ACTIVE",
                "quality_tier": "GOOD",
                "manufacturer": "Siemens",
                "location_radius": 10.0,
                "latitude": 40.7128,
                "longitude": -74.0060,
                "page": 1,
                "page_size": 20,
                "sort_by": "created_at",
                "sort_order": "desc"
            }
        }
