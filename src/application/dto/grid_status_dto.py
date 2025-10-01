"""
Grid Status DTOs
Data Transfer Objects for grid operator related API communication
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class GridOperatorStatusDTO(str, Enum):
    """Grid operator status enumeration for DTOs"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    MAINTENANCE = "MAINTENANCE"


class GridOperatorTypeDTO(str, Enum):
    """Grid operator type enumeration for DTOs"""
    TRANSMISSION = "TRANSMISSION"
    DISTRIBUTION = "DISTRIBUTION"
    GENERATION = "GENERATION"
    RETAIL = "RETAIL"


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


class GridOperatorCreateDTO(BaseModel):
    """DTO for creating a new grid operator"""
    operator_id: str = Field(..., min_length=1, description="Unique operator identifier")
    name: str = Field(..., min_length=1, description="Operator name")
    operator_type: GridOperatorTypeDTO = Field(..., description="Type of grid operator")
    location: LocationDTO = Field(..., description="Operator location")
    contact_email: Optional[str] = Field(default=None, description="Contact email")
    contact_phone: Optional[str] = Field(default=None, description="Contact phone")
    website: Optional[str] = Field(default=None, description="Website URL")
    status: GridOperatorStatusDTO = Field(default=GridOperatorStatusDTO.ACTIVE, description="Operator status")
    grid_capacity_mw: Optional[float] = Field(default=None, ge=0, description="Grid capacity in MW")
    voltage_level_kv: Optional[float] = Field(default=None, ge=0, description="Voltage level in kV")
    coverage_area_km2: Optional[float] = Field(default=None, ge=0, description="Coverage area in km²")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operator_id": "GO001",
                "name": "New York Grid Operator",
                "operator_type": "TRANSMISSION",
                "location": {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "address": "123 Main St, New York, NY 10001"
                },
                "contact_email": "contact@nygrid.com",
                "contact_phone": "+1-555-0123",
                "website": "https://nygrid.com",
                "status": "ACTIVE",
                "grid_capacity_mw": 5000.0,
                "voltage_level_kv": 345.0,
                "coverage_area_km2": 1000.0,
                "metadata": {"region": "northeast", "priority": "high"}
            }
        }


class GridOperatorUpdateDTO(BaseModel):
    """DTO for updating an existing grid operator"""
    name: Optional[str] = Field(default=None, min_length=1, description="Updated name")
    location: Optional[LocationDTO] = Field(default=None, description="Updated location")
    contact_email: Optional[str] = Field(default=None, description="Updated contact email")
    contact_phone: Optional[str] = Field(default=None, description="Updated contact phone")
    website: Optional[str] = Field(default=None, description="Updated website URL")
    status: Optional[GridOperatorStatusDTO] = Field(default=None, description="Updated status")
    grid_capacity_mw: Optional[float] = Field(default=None, ge=0, description="Updated grid capacity")
    voltage_level_kv: Optional[float] = Field(default=None, ge=0, description="Updated voltage level")
    coverage_area_km2: Optional[float] = Field(default=None, ge=0, description="Updated coverage area")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "MAINTENANCE",
                "grid_capacity_mw": 5500.0,
                "metadata": {"last_maintenance": "2024-01-20T14:00:00Z"}
            }
        }


class GridOperatorResponseDTO(BaseModel):
    """DTO for grid operator response"""
    operator_id: str = Field(..., description="Operator identifier")
    name: str = Field(..., description="Operator name")
    operator_type: GridOperatorTypeDTO = Field(..., description="Type of grid operator")
    location: LocationDTO = Field(..., description="Operator location")
    contact_email: Optional[str] = Field(default=None, description="Contact email")
    contact_phone: Optional[str] = Field(default=None, description="Contact phone")
    website: Optional[str] = Field(default=None, description="Website URL")
    status: GridOperatorStatusDTO = Field(..., description="Operator status")
    grid_capacity_mw: Optional[float] = Field(default=None, description="Grid capacity in MW")
    voltage_level_kv: Optional[float] = Field(default=None, description="Voltage level in kV")
    coverage_area_km2: Optional[float] = Field(default=None, description="Coverage area in km²")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    version: int = Field(..., description="Entity version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operator_id": "GO001",
                "name": "New York Grid Operator",
                "operator_type": "TRANSMISSION",
                "location": {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "address": "123 Main St, New York, NY 10001"
                },
                "contact_email": "contact@nygrid.com",
                "contact_phone": "+1-555-0123",
                "website": "https://nygrid.com",
                "status": "ACTIVE",
                "grid_capacity_mw": 5000.0,
                "voltage_level_kv": 345.0,
                "coverage_area_km2": 1000.0,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-20T15:45:00Z",
                "metadata": {"region": "northeast", "priority": "high"},
                "version": 1
            }
        }


class GridStatusCreateDTO(BaseModel):
    """DTO for creating a new grid status"""
    operator_id: str = Field(..., min_length=1, description="Operator identifier")
    timestamp: datetime = Field(..., description="Status timestamp")
    voltage_level: float = Field(..., ge=0, le=1000, description="Voltage level")
    frequency: float = Field(..., ge=45, le=55, description="Frequency")
    load_percentage: float = Field(..., ge=0, le=100, description="Load percentage")
    stability_score: float = Field(..., ge=0, le=1, description="Stability score")
    power_quality_score: float = Field(..., ge=0, le=1, description="Power quality score")
    total_generation_mw: Optional[float] = Field(default=None, ge=0, description="Total generation in MW")
    total_consumption_mw: Optional[float] = Field(default=None, ge=0, description="Total consumption in MW")
    grid_frequency_hz: Optional[float] = Field(default=None, ge=45, le=55, description="Grid frequency in Hz")
    voltage_deviation_percent: Optional[float] = Field(default=None, ge=-10, le=10, description="Voltage deviation percentage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operator_id": "GO001",
                "timestamp": "2024-01-20T15:45:00Z",
                "voltage_level": 230.5,
                "frequency": 50.0,
                "load_percentage": 75.5,
                "stability_score": 0.92,
                "power_quality_score": 0.88,
                "total_generation_mw": 4500.0,
                "total_consumption_mw": 4200.0,
                "grid_frequency_hz": 50.0,
                "voltage_deviation_percent": 0.2
            }
        }


class GridStatusResponseDTO(BaseModel):
    """DTO for grid status response"""
    id: str = Field(..., description="Status identifier")
    operator_id: str = Field(..., description="Operator identifier")
    timestamp: datetime = Field(..., description="Status timestamp")
    voltage_level: float = Field(..., description="Voltage level")
    frequency: float = Field(..., description="Frequency")
    load_percentage: float = Field(..., description="Load percentage")
    stability_score: float = Field(..., description="Stability score")
    power_quality_score: float = Field(..., description="Power quality score")
    total_generation_mw: Optional[float] = Field(default=None, description="Total generation in MW")
    total_consumption_mw: Optional[float] = Field(default=None, description="Total consumption in MW")
    grid_frequency_hz: Optional[float] = Field(default=None, description="Grid frequency in Hz")
    voltage_deviation_percent: Optional[float] = Field(default=None, description="Voltage deviation percentage")
    data_quality_score: float = Field(..., ge=0, le=1, description="Data quality score")
    is_anomaly: bool = Field(..., description="Whether status is an anomaly")
    anomaly_type: Optional[str] = Field(default=None, description="Type of anomaly")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "status_123",
                "operator_id": "GO001",
                "timestamp": "2024-01-20T15:45:00Z",
                "voltage_level": 230.5,
                "frequency": 50.0,
                "load_percentage": 75.5,
                "stability_score": 0.92,
                "power_quality_score": 0.88,
                "total_generation_mw": 4500.0,
                "total_consumption_mw": 4200.0,
                "grid_frequency_hz": 50.0,
                "voltage_deviation_percent": 0.2,
                "data_quality_score": 0.95,
                "is_anomaly": False,
                "anomaly_type": None,
                "created_at": "2024-01-20T15:45:00Z"
            }
        }


class GridStatusBatchDTO(BaseModel):
    """DTO for batch grid status ingestion"""
    operator_id: str = Field(..., min_length=1, description="Operator identifier")
    statuses: List[GridStatusCreateDTO] = Field(..., min_items=1, description="List of status records")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Batch metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operator_id": "GO001",
                "statuses": [
                    {
                        "operator_id": "GO001",
                        "timestamp": "2024-01-20T15:45:00Z",
                        "voltage_level": 230.5,
                        "frequency": 50.0,
                        "load_percentage": 75.5,
                        "stability_score": 0.92,
                        "power_quality_score": 0.88,
                        "total_generation_mw": 4500.0,
                        "total_consumption_mw": 4200.0,
                        "grid_frequency_hz": 50.0,
                        "voltage_deviation_percent": 0.2
                    }
                ],
                "metadata": {"batch_id": "batch_001", "source": "api"}
            }
        }


class GridOperatorListResponseDTO(BaseModel):
    """DTO for grid operator list response"""
    operators: List[GridOperatorResponseDTO] = Field(..., description="List of grid operators")
    total_count: int = Field(..., description="Total number of operators")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operators": [],
                "total_count": 50,
                "page": 1,
                "page_size": 20,
                "has_next": True,
                "has_previous": False
            }
        }


class GridStatusListResponseDTO(BaseModel):
    """DTO for grid status list response"""
    statuses: List[GridStatusResponseDTO] = Field(..., description="List of grid status records")
    total_count: int = Field(..., description="Total number of status records")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")
    
    class Config:
        json_schema_extra = {
            "example": {
                "statuses": [],
                "total_count": 1000,
                "page": 1,
                "page_size": 50,
                "has_next": True,
                "has_previous": False
            }
        }


class GridOperatorAnalyticsDTO(BaseModel):
    """DTO for grid operator analytics"""
    operator_id: str = Field(..., description="Operator identifier")
    analysis_period: str = Field(..., description="Analysis period")
    total_statuses: int = Field(..., description="Total number of status records")
    avg_voltage_level: float = Field(..., description="Average voltage level")
    avg_frequency: float = Field(..., description="Average frequency")
    avg_load_percentage: float = Field(..., description="Average load percentage")
    avg_stability_score: float = Field(..., description="Average stability score")
    avg_power_quality_score: float = Field(..., description="Average power quality score")
    avg_quality_score: float = Field(..., description="Average data quality score")
    anomaly_count: int = Field(..., description="Number of anomalies detected")
    anomaly_rate: float = Field(..., description="Anomaly rate percentage")
    peak_load: float = Field(..., description="Peak load percentage")
    peak_load_time: Optional[datetime] = Field(default=None, description="Peak load time")
    min_stability_score: float = Field(..., description="Minimum stability score")
    min_stability_time: Optional[datetime] = Field(default=None, description="Minimum stability time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operator_id": "GO001",
                "analysis_period": "24h",
                "total_statuses": 288,
                "avg_voltage_level": 230.2,
                "avg_frequency": 50.0,
                "avg_load_percentage": 75.5,
                "avg_stability_score": 0.92,
                "avg_power_quality_score": 0.88,
                "avg_quality_score": 0.95,
                "anomaly_count": 3,
                "anomaly_rate": 1.04,
                "peak_load": 95.2,
                "peak_load_time": "2024-01-20T18:30:00Z",
                "min_stability_score": 0.85,
                "min_stability_time": "2024-01-20T12:15:00Z"
            }
        }


class GridOperatorSearchDTO(BaseModel):
    """DTO for grid operator search parameters"""
    operator_id: Optional[str] = Field(default=None, description="Filter by operator ID")
    name: Optional[str] = Field(default=None, description="Filter by name")
    operator_type: Optional[GridOperatorTypeDTO] = Field(default=None, description="Filter by operator type")
    status: Optional[GridOperatorStatusDTO] = Field(default=None, description="Filter by status")
    location_radius: Optional[float] = Field(default=None, ge=0, description="Search radius in km")
    latitude: Optional[float] = Field(default=None, ge=-90, le=90, description="Center latitude")
    longitude: Optional[float] = Field(default=None, ge=-180, le=180, description="Center longitude")
    min_capacity_mw: Optional[float] = Field(default=None, ge=0, description="Minimum grid capacity")
    max_capacity_mw: Optional[float] = Field(default=None, ge=0, description="Maximum grid capacity")
    min_voltage_kv: Optional[float] = Field(default=None, ge=0, description="Minimum voltage level")
    max_voltage_kv: Optional[float] = Field(default=None, ge=0, description="Maximum voltage level")
    created_after: Optional[datetime] = Field(default=None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(default=None, description="Filter by creation date")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Page size")
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$", description="Sort order")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operator_type": "TRANSMISSION",
                "status": "ACTIVE",
                "location_radius": 50.0,
                "latitude": 40.7128,
                "longitude": -74.0060,
                "min_capacity_mw": 1000.0,
                "page": 1,
                "page_size": 20,
                "sort_by": "created_at",
                "sort_order": "desc"
            }
        }
