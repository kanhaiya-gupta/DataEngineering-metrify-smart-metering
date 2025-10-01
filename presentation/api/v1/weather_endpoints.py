"""
Weather Station API Endpoints
REST API endpoints for weather station operations
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from fastapi.responses import JSONResponse

from src.application.dto.weather_dto import (
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
from src.application.use_cases.analyze_weather_impact import AnalyzeWeatherImpactUseCase
from src.core.config.config_loader import get_database_config, get_kafka_config, get_s3_config
from src.infrastructure.database.repositories.weather_station_repository import WeatherStationRepository
from src.infrastructure.external.apis.data_quality_service import DataQualityService
from src.infrastructure.external.apis.anomaly_detection_service import AnomalyDetectionService
from src.infrastructure.external.apis.alerting_service import AlertingService
from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
from src.infrastructure.external.s3.s3_client import S3Client
from src.infrastructure.external.snowflake.query_executor import SnowflakeQueryExecutor
from src.core.exceptions.domain_exceptions import WeatherStationNotFoundError, InvalidWeatherOperationError

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection
def get_weather_station_repository():
    """Get weather station repository instance"""
    db_config = get_database_config()
    return WeatherStationRepository(db_config)

def get_data_quality_service():
    """Get data quality service instance"""
    return DataQualityService()

def get_anomaly_detection_service():
    """Get anomaly detection service instance"""
    return AnomalyDetectionService()

def get_alerting_service():
    """Get alerting service instance"""
    return AlertingService()

def get_kafka_producer():
    """Get Kafka producer instance"""
    kafka_config = get_kafka_config()
    return KafkaProducer(kafka_config)

def get_s3_client():
    """Get S3 client instance"""
    s3_config = get_s3_config()
    return S3Client(s3_config)

def get_snowflake_query_executor():
    """Get Snowflake query executor instance"""
    from ...core.config.config_loader import get_snowflake_config
    snowflake_config = get_snowflake_config()
    return SnowflakeQueryExecutor(snowflake_config)


@router.post("/stations", response_model=WeatherStationResponseDTO, status_code=status.HTTP_201_CREATED)
async def create_weather_station(
    station_data: WeatherStationCreateDTO,
    station_repo: WeatherStationRepository = Depends(get_weather_station_repository)
):
    """Create a new weather station"""
    try:
        # Convert DTO to domain entity
        from ...core.domain.entities.weather_station import WeatherStation
        from ...core.domain.value_objects.location import Location
        from ...core.domain.enums.weather_station_status import WeatherStationStatus
        from ...core.domain.enums.weather_station_type import WeatherStationType
        
        location = Location(
            latitude=station_data.location.latitude,
            longitude=station_data.location.longitude,
            address=station_data.location.address
        )
        
        station = WeatherStation(
            station_id=station_data.station_id,
            name=station_data.name,
            station_type=WeatherStationType(station_data.station_type.value),
            location=location,
            operator=station_data.operator,
            contact_email=station_data.contact_email,
            contact_phone=station_data.contact_phone,
            status=WeatherStationStatus(station_data.status.value),
            metadata=station_data.metadata
        )
        
        # Save station
        saved_station = await station_repo.save(station)
        
        # Convert to response DTO
        response = WeatherStationResponseDTO(
            station_id=saved_station.station_id,
            name=saved_station.name,
            station_type=saved_station.station_type.value,
            location=station_data.location,
            operator=saved_station.operator,
            contact_email=saved_station.contact_email,
            contact_phone=saved_station.contact_phone,
            status=saved_station.status.value,
            total_observations=0,  # Will be calculated
            average_quality_score=1.0,  # Will be calculated
            last_observation_at=None,  # Will be calculated
            created_at=saved_station.created_at,
            updated_at=saved_station.updated_at,
            metadata=saved_station.metadata,
            version=saved_station.version
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating weather station: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create weather station"
        )


@router.get("/stations/{station_id}", response_model=WeatherStationResponseDTO)
async def get_weather_station(
    station_id: str = Path(..., description="Weather station ID"),
    station_repo: WeatherStationRepository = Depends(get_weather_station_repository)
):
    """Get weather station by ID"""
    try:
        station = await station_repo.get_by_id(station_id)
        if not station:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Weather station {station_id} not found"
            )
        
        # Get additional statistics
        total_observations = await station_repo.get_observation_count(station_id)
        avg_quality_score = await station_repo.get_average_quality_score(station_id)
        last_observation = await station_repo.get_last_observation(station_id)
        
        # Convert to response DTO
        response = WeatherStationResponseDTO(
            station_id=station.station_id,
            name=station.name,
            station_type=station.station_type.value,
            location={
                "latitude": station.location.latitude,
                "longitude": station.location.longitude,
                "address": station.location.address
            },
            operator=station.operator,
            contact_email=station.contact_email,
            contact_phone=station.contact_phone,
            status=station.status.value,
            total_observations=total_observations,
            average_quality_score=avg_quality_score,
            last_observation_at=last_observation.timestamp if last_observation else None,
            created_at=station.created_at,
            updated_at=station.updated_at,
            metadata=station.metadata,
            version=station.version
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting weather station: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get weather station"
        )


@router.get("/stations", response_model=WeatherStationListResponseDTO)
async def list_weather_stations(
    search_params: WeatherStationSearchDTO = Depends(),
    station_repo: WeatherStationRepository = Depends(get_weather_station_repository)
):
    """List weather stations with filtering and pagination"""
    try:
        # Convert search params to repository filters
        filters = {
            "station_id": search_params.station_id,
            "name": search_params.name,
            "station_type": search_params.station_type.value if search_params.station_type else None,
            "operator": search_params.operator,
            "status": search_params.status.value if search_params.status else None,
            "page": search_params.page,
            "page_size": search_params.page_size,
            "sort_by": search_params.sort_by,
            "sort_order": search_params.sort_order
        }
        
        # Get stations from repository
        stations, total_count = await station_repo.get_stations_with_filters(filters)
        
        # Convert to response DTOs
        station_responses = []
        for station in stations:
            # Get additional statistics for each station
            total_observations = await station_repo.get_observation_count(station.station_id)
            avg_quality_score = await station_repo.get_average_quality_score(station.station_id)
            last_observation = await station_repo.get_last_observation(station.station_id)
            
            station_responses.append(WeatherStationResponseDTO(
                station_id=station.station_id,
                name=station.name,
                station_type=station.station_type.value,
                location={
                    "latitude": station.location.latitude,
                    "longitude": station.location.longitude,
                    "address": station.location.address
                },
                operator=station.operator,
                contact_email=station.contact_email,
                contact_phone=station.contact_phone,
                status=station.status.value,
                total_observations=total_observations,
                average_quality_score=avg_quality_score,
                last_observation_at=last_observation.timestamp if last_observation else None,
                created_at=station.created_at,
                updated_at=station.updated_at,
                metadata=station.metadata,
                version=station.version
            ))
        
        # Calculate pagination info
        has_next = (search_params.page * search_params.page_size) < total_count
        has_previous = search_params.page > 1
        
        response = WeatherStationListResponseDTO(
            stations=station_responses,
            total_count=total_count,
            page=search_params.page,
            page_size=search_params.page_size,
            has_next=has_next,
            has_previous=has_previous
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error listing weather stations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list weather stations"
        )


@router.post("/stations/{station_id}/observations", response_model=List[WeatherObservationResponseDTO], status_code=status.HTTP_201_CREATED)
async def create_weather_observation(
    station_id: str = Path(..., description="Weather station ID"),
    observation_data: WeatherObservationCreateDTO = ...,
    station_repo: WeatherStationRepository = Depends(get_weather_station_repository)
):
    """Create a new weather observation"""
    try:
        # Check if station exists
        station = await station_repo.get_by_id(station_id)
        if not station:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Weather station {station_id} not found"
            )
        
        # Convert DTO to domain entity
        from ...core.domain.entities.weather_station import WeatherObservation
        
        observation = WeatherObservation(
            station_id=station_id,
            timestamp=observation_data.timestamp,
            temperature_celsius=observation_data.temperature_celsius,
            humidity_percent=observation_data.humidity_percent,
            pressure_hpa=observation_data.pressure_hpa,
            wind_speed_ms=observation_data.wind_speed_ms,
            wind_direction_degrees=observation_data.wind_direction_degrees,
            cloud_cover_percent=observation_data.cloud_cover_percent,
            visibility_km=observation_data.visibility_km,
            uv_index=observation_data.uv_index,
            precipitation_mm=observation_data.precipitation_mm
        )
        
        # Save observation
        saved_observation = await station_repo.add_observation(observation)
        
        # Convert to response DTO
        response = WeatherObservationResponseDTO(
            id=str(saved_observation.id),
            station_id=saved_observation.station_id,
            timestamp=saved_observation.timestamp,
            temperature_celsius=saved_observation.temperature_celsius,
            humidity_percent=saved_observation.humidity_percent,
            pressure_hpa=saved_observation.pressure_hpa,
            wind_speed_ms=saved_observation.wind_speed_ms,
            wind_direction_degrees=saved_observation.wind_direction_degrees,
            cloud_cover_percent=saved_observation.cloud_cover_percent,
            visibility_km=saved_observation.visibility_km,
            uv_index=saved_observation.uv_index,
            precipitation_mm=saved_observation.precipitation_mm,
            data_quality_score=getattr(saved_observation, 'data_quality_score', 1.0),
            is_anomaly=getattr(saved_observation, 'is_anomaly', False),
            anomaly_type=getattr(saved_observation, 'anomaly_type', None),
            created_at=saved_observation.created_at
        )
        
        return [response]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating weather observation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create weather observation"
        )


@router.post("/stations/{station_id}/observations/batch", response_model=List[WeatherObservationResponseDTO], status_code=status.HTTP_201_CREATED)
async def create_weather_observations_batch(
    station_id: str = Path(..., description="Weather station ID"),
    batch_data: WeatherObservationBatchDTO = ...,
    station_repo: WeatherStationRepository = Depends(get_weather_station_repository)
):
    """Create multiple weather observations in batch"""
    try:
        # Check if station exists
        station = await station_repo.get_by_id(station_id)
        if not station:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Weather station {station_id} not found"
            )
        
        # Convert observations to domain entities
        from ...core.domain.entities.weather_station import WeatherObservation
        
        observations = []
        for obs_data in batch_data.observations:
            observation = WeatherObservation(
                station_id=station_id,
                timestamp=obs_data.timestamp,
                temperature_celsius=obs_data.temperature_celsius,
                humidity_percent=obs_data.humidity_percent,
                pressure_hpa=obs_data.pressure_hpa,
                wind_speed_ms=obs_data.wind_speed_ms,
                wind_direction_degrees=obs_data.wind_direction_degrees,
                cloud_cover_percent=obs_data.cloud_cover_percent,
                visibility_km=obs_data.visibility_km,
                uv_index=obs_data.uv_index,
                precipitation_mm=obs_data.precipitation_mm
            )
            observations.append(observation)
        
        # Save observations in batch
        saved_observations = await station_repo.add_observations_batch(observations)
        
        # Convert to response DTOs
        responses = []
        for saved_observation in saved_observations:
            responses.append(WeatherObservationResponseDTO(
                id=str(saved_observation.id),
                station_id=saved_observation.station_id,
                timestamp=saved_observation.timestamp,
                temperature_celsius=saved_observation.temperature_celsius,
                humidity_percent=saved_observation.humidity_percent,
                pressure_hpa=saved_observation.pressure_hpa,
                wind_speed_ms=saved_observation.wind_speed_ms,
                wind_direction_degrees=saved_observation.wind_direction_degrees,
                cloud_cover_percent=saved_observation.cloud_cover_percent,
                visibility_km=saved_observation.visibility_km,
                uv_index=saved_observation.uv_index,
                precipitation_mm=saved_observation.precipitation_mm,
                data_quality_score=getattr(saved_observation, 'data_quality_score', 1.0),
                is_anomaly=getattr(saved_observation, 'is_anomaly', False),
                anomaly_type=getattr(saved_observation, 'anomaly_type', None),
                created_at=saved_observation.created_at
            ))
        
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating batch weather observations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create batch weather observations"
        )


@router.get("/stations/{station_id}/observations", response_model=WeatherObservationListResponseDTO)
async def get_weather_observations(
    station_id: str = Path(..., description="Weather station ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Page size"),
    start_time: Optional[str] = Query(None, description="Start time filter"),
    end_time: Optional[str] = Query(None, description="End time filter"),
    station_repo: WeatherStationRepository = Depends(get_weather_station_repository)
):
    """Get weather observations with pagination and time filtering"""
    try:
        # Check if station exists
        station = await station_repo.get_by_id(station_id)
        if not station:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Weather station {station_id} not found"
            )
        
        # Parse time filters
        from datetime import datetime
        start_dt = None
        end_dt = None
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        # Get observations from repository
        observations, total_count = await station_repo.get_observations_with_filters(
            station_id=station_id,
            start_time=start_dt,
            end_time=end_dt,
            page=page,
            page_size=page_size
        )
        
        # Convert to response DTOs
        observation_responses = []
        for observation in observations:
            observation_responses.append(WeatherObservationResponseDTO(
                id=str(observation.id),
                station_id=observation.station_id,
                timestamp=observation.timestamp,
                temperature_celsius=observation.temperature_celsius,
                humidity_percent=observation.humidity_percent,
                pressure_hpa=observation.pressure_hpa,
                wind_speed_ms=observation.wind_speed_ms,
                wind_direction_degrees=observation.wind_direction_degrees,
                cloud_cover_percent=observation.cloud_cover_percent,
                visibility_km=observation.visibility_km,
                uv_index=observation.uv_index,
                precipitation_mm=observation.precipitation_mm,
                data_quality_score=getattr(observation, 'data_quality_score', 1.0),
                is_anomaly=getattr(observation, 'is_anomaly', False),
                anomaly_type=getattr(observation, 'anomaly_type', None),
                created_at=observation.created_at
            ))
        
        # Calculate pagination info
        has_next = (page * page_size) < total_count
        has_previous = page > 1
        
        response = WeatherObservationListResponseDTO(
            observations=observation_responses,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=has_next,
            has_previous=has_previous
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting weather observations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get weather observations"
        )


@router.get("/stations/{station_id}/analytics", response_model=WeatherStationAnalyticsDTO)
async def get_weather_station_analytics(
    station_id: str = Path(..., description="Weather station ID"),
    period: str = Query("24h", description="Analysis period"),
    station_repo: WeatherStationRepository = Depends(get_weather_station_repository)
):
    """Get weather station analytics"""
    try:
        # Check if station exists
        station = await station_repo.get_by_id(station_id)
        if not station:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Weather station {station_id} not found"
            )
        
        # Parse period
        from datetime import datetime, timedelta
        if period == "24h":
            hours = 24
        elif period == "7d":
            hours = 168
        elif period == "30d":
            hours = 720
        else:
            hours = 24
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get observations for analysis
        observations, _ = await station_repo.get_observations_with_filters(
            station_id=station_id,
            start_time=start_time,
            end_time=end_time,
            page=1,
            page_size=10000  # Large page size for analytics
        )
        
        if not observations:
            return WeatherStationAnalyticsDTO(
                station_id=station_id,
                analysis_period=period,
                total_observations=0,
                avg_temperature=0.0,
                avg_humidity=0.0,
                avg_pressure=0.0,
                avg_wind_speed=0.0,
                avg_cloud_cover=0.0,
                avg_visibility=0.0,
                avg_quality_score=0.0,
                anomaly_count=0,
                anomaly_rate=0.0,
                max_temperature=0.0,
                min_temperature=0.0,
                max_wind_speed=0.0,
                total_precipitation=0.0
            )
        
        # Calculate analytics
        total_observations = len(observations)
        avg_temperature = sum(o.temperature_celsius for o in observations) / total_observations
        avg_humidity = sum(o.humidity_percent for o in observations) / total_observations
        avg_pressure = sum(o.pressure_hpa for o in observations) / total_observations
        avg_wind_speed = sum(o.wind_speed_ms for o in observations) / total_observations
        avg_cloud_cover = sum(o.cloud_cover_percent for o in observations) / total_observations
        avg_visibility = sum(o.visibility_km for o in observations) / total_observations
        avg_quality_score = sum(getattr(o, 'data_quality_score', 1.0) for o in observations) / total_observations
        anomaly_count = sum(1 for o in observations if getattr(o, 'is_anomaly', False))
        anomaly_rate = (anomaly_count / total_observations) * 100 if total_observations > 0 else 0.0
        
        # Find extremes
        max_temperature = max(o.temperature_celsius for o in observations)
        min_temperature = min(o.temperature_celsius for o in observations)
        max_wind_speed = max(o.wind_speed_ms for o in observations)
        total_precipitation = sum(o.precipitation_mm or 0 for o in observations)
        
        response = WeatherStationAnalyticsDTO(
            station_id=station_id,
            analysis_period=period,
            total_observations=total_observations,
            avg_temperature=round(avg_temperature, 2),
            avg_humidity=round(avg_humidity, 2),
            avg_pressure=round(avg_pressure, 2),
            avg_wind_speed=round(avg_wind_speed, 2),
            avg_cloud_cover=round(avg_cloud_cover, 2),
            avg_visibility=round(avg_visibility, 2),
            avg_quality_score=round(avg_quality_score, 3),
            anomaly_count=anomaly_count,
            anomaly_rate=round(anomaly_rate, 2),
            max_temperature=round(max_temperature, 2),
            min_temperature=round(min_temperature, 2),
            max_wind_speed=round(max_wind_speed, 2),
            total_precipitation=round(total_precipitation, 2)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting weather station analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get weather station analytics"
        )


@router.get("/stations/{station_id}/forecast", response_model=WeatherForecastDTO)
async def get_weather_forecast(
    station_id: str = Path(..., description="Weather station ID"),
    hours: int = Query(24, ge=1, le=168, description="Forecast hours"),
    station_repo: WeatherStationRepository = Depends(get_weather_station_repository),
    data_quality_service: DataQualityService = Depends(get_data_quality_service),
    anomaly_detection_service: AnomalyDetectionService = Depends(get_anomaly_detection_service),
    alerting_service: AlertingService = Depends(get_alerting_service),
    kafka_producer: KafkaProducer = Depends(get_kafka_producer),
    s3_client: S3Client = Depends(get_s3_client),
    snowflake_query_executor: SnowflakeQueryExecutor = Depends(get_snowflake_query_executor)
):
    """Get weather forecast for a station"""
    try:
        # Check if station exists
        station = await station_repo.get_by_id(station_id)
        if not station:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Weather station {station_id} not found"
            )
        
        # Create use case for weather impact analysis
        analyze_use_case = AnalyzeWeatherImpactUseCase(
            weather_station_repository=station_repo,
            smart_meter_repository=None,  # Not needed for forecast
            grid_operator_repository=None,  # Not needed for forecast
            data_quality_service=data_quality_service,
            anomaly_detection_service=anomaly_detection_service,
            alerting_service=alerting_service,
            kafka_producer=kafka_producer,
            s3_client=s3_client,
            snowflake_query_executor=snowflake_query_executor
        )
        
        # Generate weather forecast
        result = await analyze_use_case.execute(
            station_id=station_id,
            analysis_period_hours=hours
        )
        
        # Extract forecast data from result
        forecast_results = result.get("energy_forecast", {})
        forecasts = forecast_results.get("predictions", [])
        
        # Convert to forecast DTO
        response = WeatherForecastDTO(
            station_id=station_id,
            forecast_hours=hours,
            forecasts=forecasts,
            confidence=forecast_results.get("confidence", 0.0),
            generated_at=forecast_results.get("generated_at", "2024-01-20T15:45:00Z")
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting weather forecast: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get weather forecast"
        )
