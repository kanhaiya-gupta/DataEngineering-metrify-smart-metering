"""
Application Layer Module
Contains use cases, DTOs, and handlers for the application layer
"""

from .use_cases.ingest_smart_meter_data import IngestSmartMeterDataUseCase
from .use_cases.process_grid_status import ProcessGridStatusUseCase
from .use_cases.analyze_weather_impact import AnalyzeWeatherImpactUseCase
from .use_cases.detect_anomalies import DetectAnomaliesUseCase

from .dto.smart_meter_dto import (
    SmartMeterCreateDTO,
    SmartMeterUpdateDTO,
    SmartMeterResponseDTO,
    MeterReadingCreateDTO,
    MeterReadingResponseDTO,
    MeterReadingBatchDTO,
    SmartMeterListResponseDTO,
    MeterReadingListResponseDTO,
    SmartMeterAnalyticsDTO,
    SmartMeterSearchDTO
)

from .dto.grid_status_dto import (
    GridOperatorCreateDTO,
    GridOperatorUpdateDTO,
    GridOperatorResponseDTO,
    GridStatusCreateDTO,
    GridStatusResponseDTO,
    GridStatusBatchDTO,
    GridOperatorListResponseDTO,
    GridStatusListResponseDTO,
    GridOperatorAnalyticsDTO,
    GridOperatorSearchDTO
)

from .dto.weather_dto import (
    WeatherStationCreateDTO,
    WeatherStationUpdateDTO,
    WeatherStationResponseDTO,
    WeatherObservationCreateDTO,
    WeatherObservationResponseDTO,
    WeatherObservationBatchDTO,
    WeatherStationListResponseDTO,
    WeatherObservationListResponseDTO,
    WeatherStationAnalyticsDTO,
    WeatherStationSearchDTO,
    WeatherForecastDTO
)

from .handlers.event_handlers.meter_event_handlers import MeterEventHandlers
from .handlers.command_handlers.smart_meter_command_handlers import SmartMeterCommandHandlers

__all__ = [
    # Use Cases
    "IngestSmartMeterDataUseCase",
    "ProcessGridStatusUseCase",
    "AnalyzeWeatherImpactUseCase",
    "DetectAnomaliesUseCase",
    
    # Smart Meter DTOs
    "SmartMeterCreateDTO",
    "SmartMeterUpdateDTO",
    "SmartMeterResponseDTO",
    "MeterReadingCreateDTO",
    "MeterReadingResponseDTO",
    "MeterReadingBatchDTO",
    "SmartMeterListResponseDTO",
    "MeterReadingListResponseDTO",
    "SmartMeterAnalyticsDTO",
    "SmartMeterSearchDTO",
    
    # Grid Status DTOs
    "GridOperatorCreateDTO",
    "GridOperatorUpdateDTO",
    "GridOperatorResponseDTO",
    "GridStatusCreateDTO",
    "GridStatusResponseDTO",
    "GridStatusBatchDTO",
    "GridOperatorListResponseDTO",
    "GridStatusListResponseDTO",
    "GridOperatorAnalyticsDTO",
    "GridOperatorSearchDTO",
    
    # Weather DTOs
    "WeatherStationCreateDTO",
    "WeatherStationUpdateDTO",
    "WeatherStationResponseDTO",
    "WeatherObservationCreateDTO",
    "WeatherObservationResponseDTO",
    "WeatherObservationBatchDTO",
    "WeatherStationListResponseDTO",
    "WeatherObservationListResponseDTO",
    "WeatherStationAnalyticsDTO",
    "WeatherStationSearchDTO",
    "WeatherForecastDTO",
    
    # Handlers
    "MeterEventHandlers",
    "SmartMeterCommandHandlers"
]
