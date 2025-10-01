"""
Smart Meter API Endpoints
REST API endpoints for smart meter operations
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from fastapi.responses import JSONResponse

from src.application.dto.smart_meter_dto import (
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
from src.application.use_cases.ingest_smart_meter_data import IngestSmartMeterDataUseCase
from src.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
from src.core.config.config_loader import get_database_config, get_kafka_config, get_s3_config
from src.infrastructure.database.repositories.smart_meter_repository import SmartMeterRepository
from src.infrastructure.external.apis.data_quality_service import DataQualityService
from src.infrastructure.external.apis.anomaly_detection_service import AnomalyDetectionService
from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
from src.infrastructure.external.s3.s3_client import S3Client
from src.infrastructure.external.monitoring.monitoring_service import MonitoringService
from src.core.exceptions.domain_exceptions import MeterNotFoundException, InvalidMeterOperationError

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection
def get_smart_meter_repository():
    """Get smart meter repository instance"""
    db_config = get_database_config()
    return SmartMeterRepository(db_config)

def get_data_quality_service():
    """Get data quality service instance"""
    return DataQualityService()

def get_anomaly_detection_service():
    """Get anomaly detection service instance"""
    return AnomalyDetectionService()

def get_kafka_producer():
    """Get Kafka producer instance"""
    kafka_config = get_kafka_config()
    return KafkaProducer(kafka_config)

def get_s3_client():
    """Get S3 client instance"""
    s3_config = get_s3_config()
    return S3Client(s3_config)

def get_monitoring_service():
    """Get monitoring service instance"""
    # This would be injected from the main app
    return None


@router.post("/", response_model=SmartMeterResponseDTO, status_code=status.HTTP_201_CREATED)
async def create_smart_meter(
    meter_data: SmartMeterCreateDTO,
    meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository)
):
    """Create a new smart meter"""
    try:
        # Convert DTO to domain entity
        from ...core.domain.entities.smart_meter import SmartMeter
        from ...core.domain.value_objects.meter_id import MeterId
        from ...core.domain.value_objects.location import Location
        from ...core.domain.value_objects.meter_specifications import MeterSpecifications
        from ...core.domain.enums.meter_status import MeterStatus
        from ...core.domain.enums.quality_tier import QualityTier
        
        location = Location(
            latitude=meter_data.location.latitude,
            longitude=meter_data.location.longitude,
            address=meter_data.location.address
        )
        
        specifications = MeterSpecifications(
            manufacturer=meter_data.specifications.manufacturer,
            model=meter_data.specifications.model,
            firmware_version=meter_data.specifications.firmware_version,
            installation_date=meter_data.specifications.installation_date
        )
        
        meter = SmartMeter(
            meter_id=MeterId(meter_data.meter_id),
            location=location,
            specifications=specifications,
            status=MeterStatus(meter_data.status.value),
            quality_tier=QualityTier(meter_data.quality_tier.value),
            metadata=meter_data.metadata
        )
        
        # Save meter
        saved_meter = await meter_repo.save(meter)
        
        # Convert to response DTO
        response = SmartMeterResponseDTO(
            meter_id=saved_meter.meter_id.value,
            location=meter_data.location,
            specifications=meter_data.specifications,
            status=saved_meter.status.value,
            quality_tier=saved_meter.quality_tier.value,
            installed_at=saved_meter.installed_at,
            last_reading_at=saved_meter.last_reading_at,
            created_at=saved_meter.created_at,
            updated_at=saved_meter.updated_at,
            metadata=saved_meter.metadata,
            version=saved_meter.version
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating smart meter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create smart meter"
        )


@router.get("/{meter_id}", response_model=SmartMeterResponseDTO)
async def get_smart_meter(
    meter_id: str = Path(..., description="Smart meter ID"),
    meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository)
):
    """Get smart meter by ID"""
    try:
        meter = await meter_repo.get_by_id(meter_id)
        if not meter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Smart meter {meter_id} not found"
            )
        
        # Convert to response DTO
        response = SmartMeterResponseDTO(
            meter_id=meter.meter_id.value,
            location={
                "latitude": meter.location.latitude,
                "longitude": meter.location.longitude,
                "address": meter.location.address
            },
            specifications={
                "manufacturer": meter.specifications.manufacturer,
                "model": meter.specifications.model,
                "firmware_version": meter.specifications.firmware_version,
                "installation_date": meter.specifications.installation_date
            },
            status=meter.status.value,
            quality_tier=meter.quality_tier.value,
            installed_at=meter.installed_at,
            last_reading_at=meter.last_reading_at,
            created_at=meter.created_at,
            updated_at=meter.updated_at,
            metadata=meter.metadata,
            version=meter.version
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting smart meter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get smart meter"
        )


@router.get("/", response_model=SmartMeterListResponseDTO)
async def list_smart_meters(
    search_params: SmartMeterSearchDTO = Depends(),
    meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository)
):
    """List smart meters with filtering and pagination"""
    try:
        # Convert search params to repository filters
        filters = {
            "meter_id": search_params.meter_id,
            "status": search_params.status.value if search_params.status else None,
            "quality_tier": search_params.quality_tier.value if search_params.quality_tier else None,
            "manufacturer": search_params.manufacturer,
            "model": search_params.model,
            "page": search_params.page,
            "page_size": search_params.page_size,
            "sort_by": search_params.sort_by,
            "sort_order": search_params.sort_order
        }
        
        # Get meters from repository
        meters, total_count = await meter_repo.get_meters_with_filters(filters)
        
        # Convert to response DTOs
        meter_responses = []
        for meter in meters:
            meter_responses.append(SmartMeterResponseDTO(
                meter_id=meter.meter_id.value,
                location={
                    "latitude": meter.location.latitude,
                    "longitude": meter.location.longitude,
                    "address": meter.location.address
                },
                specifications={
                    "manufacturer": meter.specifications.manufacturer,
                    "model": meter.specifications.model,
                    "firmware_version": meter.specifications.firmware_version,
                    "installation_date": meter.specifications.installation_date
                },
                status=meter.status.value,
                quality_tier=meter.quality_tier.value,
                installed_at=meter.installed_at,
                last_reading_at=meter.last_reading_at,
                created_at=meter.created_at,
                updated_at=meter.updated_at,
                metadata=meter.metadata,
                version=meter.version
            ))
        
        # Calculate pagination info
        has_next = (search_params.page * search_params.page_size) < total_count
        has_previous = search_params.page > 1
        
        response = SmartMeterListResponseDTO(
            meters=meter_responses,
            total_count=total_count,
            page=search_params.page,
            page_size=search_params.page_size,
            has_next=has_next,
            has_previous=has_previous
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error listing smart meters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list smart meters"
        )


@router.put("/{meter_id}", response_model=SmartMeterResponseDTO)
async def update_smart_meter(
    meter_id: str = Path(..., description="Smart meter ID"),
    update_data: SmartMeterUpdateDTO = ...,
    meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository)
):
    """Update smart meter"""
    try:
        # Get existing meter
        meter = await meter_repo.get_by_id(meter_id)
        if not meter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Smart meter {meter_id} not found"
            )
        
        # Update meter fields
        if update_data.location:
            meter.location = Location(
                latitude=update_data.location.latitude,
                longitude=update_data.location.longitude,
                address=update_data.location.address
            )
        
        if update_data.status:
            meter.update_status(update_data.status.value)
        
        if update_data.quality_tier:
            meter.update_quality_tier(update_data.quality_tier.value)
        
        if update_data.metadata:
            meter.metadata = update_data.metadata
        
        # Save updated meter
        updated_meter = await meter_repo.update(meter)
        
        # Convert to response DTO
        response = SmartMeterResponseDTO(
            meter_id=updated_meter.meter_id.value,
            location={
                "latitude": updated_meter.location.latitude,
                "longitude": updated_meter.location.longitude,
                "address": updated_meter.location.address
            },
            specifications={
                "manufacturer": updated_meter.specifications.manufacturer,
                "model": updated_meter.specifications.model,
                "firmware_version": updated_meter.specifications.firmware_version,
                "installation_date": updated_meter.specifications.installation_date
            },
            status=updated_meter.status.value,
            quality_tier=updated_meter.quality_tier.value,
            installed_at=updated_meter.installed_at,
            last_reading_at=updated_meter.last_reading_at,
            created_at=updated_meter.created_at,
            updated_at=updated_meter.updated_at,
            metadata=updated_meter.metadata,
            version=updated_meter.version
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating smart meter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update smart meter"
        )


@router.delete("/{meter_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_smart_meter(
    meter_id: str = Path(..., description="Smart meter ID"),
    meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository)
):
    """Delete smart meter"""
    try:
        # Check if meter exists
        meter = await meter_repo.get_by_id(meter_id)
        if not meter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Smart meter {meter_id} not found"
            )
        
        # Delete meter
        await meter_repo.delete(meter_id)
        
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting smart meter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete smart meter"
        )


@router.post("/{meter_id}/readings", response_model=List[MeterReadingResponseDTO], status_code=status.HTTP_201_CREATED)
async def create_meter_reading(
    meter_id: str = Path(..., description="Smart meter ID"),
    reading_data: MeterReadingCreateDTO = ...,
    meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository)
):
    """Create a new meter reading"""
    try:
        # Check if meter exists
        meter = await meter_repo.get_by_id(meter_id)
        if not meter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Smart meter {meter_id} not found"
            )
        
        # Convert DTO to domain entity
        from ...core.domain.entities.smart_meter import MeterReading
        from ...core.domain.value_objects.meter_id import MeterId
        
        reading = MeterReading(
            meter_id=MeterId(meter_id),
            timestamp=reading_data.timestamp,
            voltage=reading_data.voltage,
            current=reading_data.current,
            power_factor=reading_data.power_factor,
            frequency=reading_data.frequency,
            active_power=reading_data.active_power,
            reactive_power=reading_data.reactive_power,
            apparent_power=reading_data.apparent_power
        )
        
        # Save reading
        saved_reading = await meter_repo.add_reading(reading)
        
        # Convert to response DTO
        response = MeterReadingResponseDTO(
            id=str(saved_reading.id),
            meter_id=saved_reading.meter_id.value,
            timestamp=saved_reading.timestamp,
            voltage=saved_reading.voltage,
            current=saved_reading.current,
            power_factor=saved_reading.power_factor,
            frequency=saved_reading.frequency,
            active_power=saved_reading.active_power,
            reactive_power=saved_reading.reactive_power,
            apparent_power=saved_reading.apparent_power,
            data_quality_score=getattr(saved_reading, 'data_quality_score', 1.0),
            is_anomaly=getattr(saved_reading, 'is_anomaly', False),
            anomaly_type=getattr(saved_reading, 'anomaly_type', None),
            created_at=saved_reading.created_at
        )
        
        return [response]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating meter reading: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create meter reading"
        )


@router.post("/{meter_id}/readings/batch", response_model=List[MeterReadingResponseDTO], status_code=status.HTTP_201_CREATED)
async def create_meter_readings_batch(
    meter_id: str = Path(..., description="Smart meter ID"),
    batch_data: MeterReadingBatchDTO = ...,
    meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository),
    data_quality_service: DataQualityService = Depends(get_data_quality_service),
    anomaly_detection_service: AnomalyDetectionService = Depends(get_anomaly_detection_service),
    kafka_producer: KafkaProducer = Depends(get_kafka_producer),
    s3_client: S3Client = Depends(get_s3_client)
):
    """Create multiple meter readings in batch"""
    try:
        # Check if meter exists
        meter = await meter_repo.get_by_id(meter_id)
        if not meter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Smart meter {meter_id} not found"
            )
        
        # Create use case
        ingest_use_case = IngestSmartMeterDataUseCase(
            smart_meter_repository=meter_repo,
            data_quality_service=data_quality_service,
            anomaly_detection_service=anomaly_detection_service,
            kafka_producer=kafka_producer,
            s3_client=s3_client
        )
        
        # Convert readings to format expected by use case
        readings_data = []
        for reading in batch_data.readings:
            readings_data.append({
                'timestamp': reading.timestamp.isoformat(),
                'voltage': reading.voltage,
                'current': reading.current,
                'power_factor': reading.power_factor,
                'frequency': reading.frequency,
                'active_power': reading.active_power,
                'reactive_power': reading.reactive_power,
                'apparent_power': reading.apparent_power
            })
        
        # Execute use case
        result = await ingest_use_case.execute(
            meter_id=meter_id,
            readings_data=readings_data,
            metadata=batch_data.metadata
        )
        
        # Return success response
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "Batch readings created successfully",
                "readings_processed": result.get("readings_processed", 0),
                "quality_score": result.get("quality_score", 0.0),
                "anomalies_detected": result.get("anomalies_detected", 0)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating batch meter readings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create batch meter readings"
        )


@router.get("/{meter_id}/readings", response_model=MeterReadingListResponseDTO)
async def get_meter_readings(
    meter_id: str = Path(..., description="Smart meter ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Page size"),
    start_time: Optional[str] = Query(None, description="Start time filter"),
    end_time: Optional[str] = Query(None, description="End time filter"),
    meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository)
):
    """Get meter readings with pagination and time filtering"""
    try:
        # Check if meter exists
        meter = await meter_repo.get_by_id(meter_id)
        if not meter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Smart meter {meter_id} not found"
            )
        
        # Parse time filters
        from datetime import datetime
        start_dt = None
        end_dt = None
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        # Get readings from repository
        readings, total_count = await meter_repo.get_readings_with_filters(
            meter_id=meter_id,
            start_time=start_dt,
            end_time=end_dt,
            page=page,
            page_size=page_size
        )
        
        # Convert to response DTOs
        reading_responses = []
        for reading in readings:
            reading_responses.append(MeterReadingResponseDTO(
                id=str(reading.id),
                meter_id=reading.meter_id.value,
                timestamp=reading.timestamp,
                voltage=reading.voltage,
                current=reading.current,
                power_factor=reading.power_factor,
                frequency=reading.frequency,
                active_power=reading.active_power,
                reactive_power=reading.reactive_power,
                apparent_power=reading.apparent_power,
                data_quality_score=getattr(reading, 'data_quality_score', 1.0),
                is_anomaly=getattr(reading, 'is_anomaly', False),
                anomaly_type=getattr(reading, 'anomaly_type', None),
                created_at=reading.created_at
            ))
        
        # Calculate pagination info
        has_next = (page * page_size) < total_count
        has_previous = page > 1
        
        response = MeterReadingListResponseDTO(
            readings=reading_responses,
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
        logger.error(f"Error getting meter readings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get meter readings"
        )


@router.get("/{meter_id}/analytics", response_model=SmartMeterAnalyticsDTO)
async def get_meter_analytics(
    meter_id: str = Path(..., description="Smart meter ID"),
    period: str = Query("24h", description="Analysis period"),
    meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository)
):
    """Get smart meter analytics"""
    try:
        # Check if meter exists
        meter = await meter_repo.get_by_id(meter_id)
        if not meter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Smart meter {meter_id} not found"
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
        
        # Get readings for analysis
        readings, _ = await meter_repo.get_readings_with_filters(
            meter_id=meter_id,
            start_time=start_time,
            end_time=end_time,
            page=1,
            page_size=10000  # Large page size for analytics
        )
        
        if not readings:
            return SmartMeterAnalyticsDTO(
                meter_id=meter_id,
                analysis_period=period,
                total_readings=0,
                avg_voltage=0.0,
                avg_current=0.0,
                avg_power_factor=0.0,
                avg_frequency=0.0,
                total_energy_consumed=0.0,
                avg_quality_score=0.0,
                anomaly_count=0,
                anomaly_rate=0.0,
                peak_consumption=0.0,
                peak_consumption_time=None
            )
        
        # Calculate analytics
        total_readings = len(readings)
        avg_voltage = sum(r.voltage for r in readings) / total_readings
        avg_current = sum(r.current for r in readings) / total_readings
        avg_power_factor = sum(r.power_factor for r in readings) / total_readings
        avg_frequency = sum(r.frequency for r in readings) / total_readings
        total_energy_consumed = sum(r.active_power for r in readings) / 1000  # Convert to kWh
        avg_quality_score = sum(getattr(r, 'data_quality_score', 1.0) for r in readings) / total_readings
        anomaly_count = sum(1 for r in readings if getattr(r, 'is_anomaly', False))
        anomaly_rate = (anomaly_count / total_readings) * 100 if total_readings > 0 else 0.0
        
        # Find peak consumption
        peak_reading = max(readings, key=lambda r: r.active_power)
        peak_consumption = peak_reading.active_power
        peak_consumption_time = peak_reading.timestamp
        
        response = SmartMeterAnalyticsDTO(
            meter_id=meter_id,
            analysis_period=period,
            total_readings=total_readings,
            avg_voltage=round(avg_voltage, 2),
            avg_current=round(avg_current, 2),
            avg_power_factor=round(avg_power_factor, 3),
            avg_frequency=round(avg_frequency, 2),
            total_energy_consumed=round(total_energy_consumed, 2),
            avg_quality_score=round(avg_quality_score, 3),
            anomaly_count=anomaly_count,
            anomaly_rate=round(anomaly_rate, 2),
            peak_consumption=round(peak_consumption, 2),
            peak_consumption_time=peak_consumption_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting meter analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get meter analytics"
        )
