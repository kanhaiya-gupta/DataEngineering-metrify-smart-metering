"""
Common API Schemas
Shared schemas used across multiple API endpoints
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints"""
    
    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(50, ge=1, le=1000, description="Number of items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: str = Field("asc", regex="^(asc|desc)$", description="Sort order")


class PaginationResponse(BaseModel):
    """Pagination response metadata"""
    
    total_count: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")
    total_pages: int = Field(..., description="Total number of pages")


class ErrorResponse(BaseModel):
    """Standard error response schema"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class SuccessResponse(BaseModel):
    """Standard success response schema"""
    
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    components: Dict[str, str] = Field(..., description="Component health status")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")


class LocationSchema(BaseModel):
    """Location schema for API requests/responses"""
    
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    address: str = Field(..., min_length=1, max_length=500, description="Physical address")


class TimeRangeParams(BaseModel):
    """Time range parameters for filtering"""
    
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")


class SearchParams(BaseModel):
    """Generic search parameters"""
    
    query: Optional[str] = Field(None, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")


class BatchOperationResponse(BaseModel):
    """Response schema for batch operations"""
    
    total_processed: int = Field(..., description="Total items processed")
    successful: int = Field(..., description="Successfully processed items")
    failed: int = Field(..., description="Failed items")
    errors: Optional[Dict[str, str]] = Field(None, description="Error details for failed items")
    processing_time: float = Field(..., description="Processing time in seconds")


class ValidationErrorDetail(BaseModel):
    """Validation error detail schema"""
    
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Any = Field(..., description="Invalid value")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response schema"""
    
    validation_errors: list[ValidationErrorDetail] = Field(..., description="Validation error details")
