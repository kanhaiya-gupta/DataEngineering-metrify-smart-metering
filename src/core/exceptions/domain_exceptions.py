"""
Domain Exceptions
Custom exceptions for the smart metering domain
"""

from typing import Optional, Dict, Any


class DomainException(Exception):
    """Base exception for domain-related errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class MeterNotFoundError(DomainException):
    """Raised when a smart meter is not found"""
    
    def __init__(self, message: str, meter_id: Optional[str] = None):
        super().__init__(message, error_code="METER_NOT_FOUND")
        self.meter_id = meter_id


class InvalidMeterOperationError(DomainException):
    """Raised when an invalid operation is attempted on a smart meter"""
    
    def __init__(self, message: str, operation: Optional[str] = None, meter_id: Optional[str] = None):
        super().__init__(message, error_code="INVALID_METER_OPERATION")
        self.operation = operation
        self.meter_id = meter_id


class MeterAlreadyExistsError(DomainException):
    """Raised when trying to create a meter that already exists"""
    
    def __init__(self, message: str, meter_id: Optional[str] = None):
        super().__init__(message, error_code="METER_ALREADY_EXISTS")
        self.meter_id = meter_id


class InvalidMeterReadingError(DomainException):
    """Raised when a meter reading is invalid"""
    
    def __init__(self, message: str, reading_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="INVALID_METER_READING")
        self.reading_data = reading_data


class MeterInactiveError(DomainException):
    """Raised when trying to perform operations on an inactive meter"""
    
    def __init__(self, message: str, meter_id: Optional[str] = None, current_status: Optional[str] = None):
        super().__init__(message, error_code="METER_INACTIVE")
        self.meter_id = meter_id
        self.current_status = current_status


class GridOperatorNotFoundError(DomainException):
    """Raised when a grid operator is not found"""
    
    def __init__(self, message: str, operator_id: Optional[str] = None):
        super().__init__(message, error_code="GRID_OPERATOR_NOT_FOUND")
        self.operator_id = operator_id


class InvalidGridOperationError(DomainException):
    """Raised when an invalid operation is attempted on a grid operator"""
    
    def __init__(self, message: str, operation: Optional[str] = None, operator_id: Optional[str] = None):
        super().__init__(message, error_code="INVALID_GRID_OPERATION")
        self.operation = operation
        self.operator_id = operator_id


class GridOperatorAlreadyExistsError(DomainException):
    """Raised when trying to create a grid operator that already exists"""
    
    def __init__(self, message: str, operator_id: Optional[str] = None):
        super().__init__(message, error_code="GRID_OPERATOR_ALREADY_EXISTS")
        self.operator_id = operator_id


class InvalidGridStatusError(DomainException):
    """Raised when grid status data is invalid"""
    
    def __init__(self, message: str, status_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="INVALID_GRID_STATUS")
        self.status_data = status_data


class GridOperatorInactiveError(DomainException):
    """Raised when trying to perform operations on an inactive grid operator"""
    
    def __init__(self, message: str, operator_id: Optional[str] = None, current_status: Optional[str] = None):
        super().__init__(message, error_code="GRID_OPERATOR_INACTIVE")
        self.operator_id = operator_id
        self.current_status = current_status


class DataQualityError(DomainException):
    """Raised when data quality issues are detected"""
    
    def __init__(self, message: str, quality_score: Optional[float] = None, violations: Optional[list] = None):
        super().__init__(message, error_code="DATA_QUALITY_ERROR")
        self.quality_score = quality_score
        self.violations = violations or []


class AnomalyDetectionError(DomainException):
    """Raised when anomaly detection fails"""
    
    def __init__(self, message: str, meter_id: Optional[str] = None, anomaly_type: Optional[str] = None):
        super().__init__(message, error_code="ANOMALY_DETECTION_ERROR")
        self.meter_id = meter_id
        self.anomaly_type = anomaly_type


class ValidationError(DomainException):
    """Raised when validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(message, error_code="VALIDATION_ERROR")
        self.field = field
        self.value = value


class BusinessRuleViolationError(DomainException):
    """Raised when a business rule is violated"""
    
    def __init__(self, message: str, rule: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="BUSINESS_RULE_VIOLATION")
        self.rule = rule
        self.context = context or {}


class ConcurrencyError(DomainException):
    """Raised when a concurrency conflict occurs"""
    
    def __init__(self, message: str, entity_type: Optional[str] = None, entity_id: Optional[str] = None):
        super().__init__(message, error_code="CONCURRENCY_ERROR")
        self.entity_type = entity_type
        self.entity_id = entity_id


class WeatherStationNotFoundError(DomainException):
    """Raised when a weather station is not found"""
    
    def __init__(self, message: str, station_id: Optional[str] = None):
        super().__init__(message, error_code="WEATHER_STATION_NOT_FOUND")
        self.station_id = station_id


class InvalidWeatherOperationError(DomainException):
    """Raised when an invalid operation is attempted on a weather station"""
    
    def __init__(self, message: str, operation: Optional[str] = None, station_id: Optional[str] = None):
        super().__init__(message, error_code="INVALID_WEATHER_OPERATION")
        self.operation = operation
        self.station_id = station_id


class WeatherStationAlreadyExistsError(DomainException):
    """Raised when trying to create a weather station that already exists"""
    
    def __init__(self, message: str, station_id: Optional[str] = None):
        super().__init__(message, error_code="WEATHER_STATION_ALREADY_EXISTS")
        self.station_id = station_id


class InvalidWeatherObservationError(DomainException):
    """Raised when a weather observation is invalid"""
    
    def __init__(self, message: str, observation_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="INVALID_WEATHER_OBSERVATION")
        self.observation_data = observation_data


class WeatherStationInactiveError(DomainException):
    """Raised when trying to perform operations on an inactive weather station"""
    
    def __init__(self, message: str, station_id: Optional[str] = None, current_status: Optional[str] = None):
        super().__init__(message, error_code="WEATHER_STATION_INACTIVE")
        self.station_id = station_id
        self.current_status = current_status


class InfrastructureError(DomainException):
    """Raised when infrastructure-related errors occur"""
    
    def __init__(self, message: str, service: Optional[str] = None, operation: Optional[str] = None):
        super().__init__(message, error_code="INFRASTRUCTURE_ERROR")
        self.service = service
        self.operation = operation
