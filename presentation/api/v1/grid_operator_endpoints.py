"""
Grid Operator API Endpoints
REST API endpoints for grid operator operations
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from fastapi.responses import JSONResponse

from src.application.dto.grid_status_dto import (
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
from src.application.use_cases.process_grid_status import ProcessGridStatusUseCase
from src.infrastructure.database.config import get_database_config
from src.core.config.config_loader import get_kafka_config, get_s3_config
from src.infrastructure.database.repositories.grid_operator_repository import GridOperatorRepository
from src.infrastructure.external.apis.data_quality_service import DataQualityService
from src.infrastructure.external.apis.anomaly_detection_service import AnomalyDetectionService
from src.infrastructure.external.apis.alerting_service import AlertingService
from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
from src.infrastructure.external.s3.s3_client import S3Client
from src.core.exceptions.domain_exceptions import GridOperatorNotFoundError, InvalidGridOperationError

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection
def get_grid_operator_repository():
    """Get grid operator repository instance"""
    db_config = get_database_config()
    session = db_config.session_factory()
    return GridOperatorRepository(session)

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


@router.post("/", response_model=GridOperatorResponseDTO, status_code=status.HTTP_201_CREATED)
async def create_grid_operator(
    operator_data: GridOperatorCreateDTO,
    operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository)
):
    """Create a new grid operator"""
    try:
        # Convert DTO to domain entity
        from ...core.domain.entities.grid_operator import GridOperator
        from ...core.domain.value_objects.location import Location
        from ...core.domain.enums.grid_operator_status import GridOperatorStatus
        from ...core.domain.enums.grid_operator_type import GridOperatorType
        
        location = Location(
            latitude=operator_data.location.latitude,
            longitude=operator_data.location.longitude,
            address=operator_data.location.address
        )
        
        operator = GridOperator(
            operator_id=operator_data.operator_id,
            name=operator_data.name,
            operator_type=GridOperatorType(operator_data.operator_type.value),
            headquarters=location,
            coverage_regions=[],  # Default empty list, can be populated later
            contact_email=operator_data.contact_email,
            contact_phone=operator_data.contact_phone,
            website=operator_data.website,
            status=GridOperatorStatus(operator_data.status.value)
        )
        
        # Save operator
        saved_operator = await operator_repo.save(operator)
        
        # Convert to response DTO
        response = GridOperatorResponseDTO(
            operator_id=saved_operator.operator_id,
            name=saved_operator.name,
            operator_type=saved_operator.operator_type.value,
            location=operator_data.location,
            contact_email=saved_operator.contact_email,
            contact_phone=saved_operator.contact_phone,
            website=saved_operator.website,
            status=saved_operator.status.value,
            grid_capacity_mw=None,  # Not available in entity
            voltage_level_kv=None,  # Not available in entity
            coverage_area_km2=None,  # Not available in entity
            created_at=saved_operator.created_at,
            updated_at=saved_operator.updated_at,
            metadata=None,  # Not available in entity
            version=1  # Default version
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating grid operator: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create grid operator"
        )


@router.get("/{operator_id}", response_model=GridOperatorResponseDTO)
async def get_grid_operator(
    operator_id: str = Path(..., description="Grid operator ID"),
    operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository)
):
    """Get grid operator by ID"""
    try:
        operator = await operator_repo.get_by_id(operator_id)
        if not operator:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Grid operator {operator_id} not found"
            )
        
        # Convert to response DTO
        response = GridOperatorResponseDTO(
            operator_id=operator.operator_id,
            name=operator.name,
            operator_type=operator.operator_type.value,
            location={
                "latitude": operator.headquarters.latitude,
                "longitude": operator.headquarters.longitude,
                "address": operator.headquarters.address
            },
            contact_email=operator.contact_email,
            contact_phone=operator.contact_phone,
            website=operator.website,
            status=operator.status.value,
            grid_capacity_mw=None,  # Not available in entity
            voltage_level_kv=None,  # Not available in entity
            coverage_area_km2=None,  # Not available in entity
            created_at=operator.created_at,
            updated_at=operator.updated_at,
            metadata=None,  # Not available in entity
            version=operator.version
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting grid operator: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get grid operator"
        )


@router.get("/", response_model=GridOperatorListResponseDTO)
async def list_grid_operators(
    search_params: GridOperatorSearchDTO = Depends(),
    operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository)
):
    """List grid operators with filtering and pagination"""
    try:
        # Convert search params to repository filters
        filters = {
            "operator_id": search_params.operator_id,
            "name": search_params.name,
            "operator_type": search_params.operator_type.value if search_params.operator_type else None,
            "status": search_params.status.value if search_params.status else None,
            "page": search_params.page,
            "page_size": search_params.page_size,
            "sort_by": search_params.sort_by,
            "sort_order": search_params.sort_order
        }
        
        # Get operators from repository
        operators, total_count = await operator_repo.get_operators_with_filters(filters)
        
        # Convert to response DTOs
        operator_responses = []
        for operator in operators:
            operator_responses.append(GridOperatorResponseDTO(
                operator_id=operator.operator_id,
                name=operator.name,
                operator_type=operator.operator_type.value,
                location={
                    "latitude": operator.headquarters.latitude,
                    "longitude": operator.headquarters.longitude,
                    "address": operator.headquarters.address
                },
                contact_email=operator.contact_email,
                contact_phone=operator.contact_phone,
                website=operator.website,
                status=operator.status.value,
                grid_capacity_mw=None,  # Not available in entity
                voltage_level_kv=None,  # Not available in entity
                coverage_area_km2=None,  # Not available in entity
                created_at=operator.created_at,
                updated_at=operator.updated_at,
                metadata=None,  # Not available in entity
                version=1  # Default version
            ))
        
        # Calculate pagination info
        has_next = (search_params.page * search_params.page_size) < total_count
        has_previous = search_params.page > 1
        
        response = GridOperatorListResponseDTO(
            operators=operator_responses,
            total_count=total_count,
            page=search_params.page,
            page_size=search_params.page_size,
            has_next=has_next,
            has_previous=has_previous
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error listing grid operators: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list grid operators"
        )


@router.put("/{operator_id}", response_model=GridOperatorResponseDTO)
async def update_grid_operator(
    operator_id: str = Path(..., description="Grid operator ID"),
    update_data: GridOperatorUpdateDTO = ...,
    operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository)
):
    """Update grid operator"""
    try:
        # Get existing operator
        operator = await operator_repo.get_by_id(operator_id)
        if not operator:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Grid operator {operator_id} not found"
            )
        
        # Update operator fields
        if update_data.name:
            operator.name = update_data.name
        
        if update_data.location:
            operator.location = Location(
                latitude=update_data.location.latitude,
                longitude=update_data.location.longitude,
                address=update_data.location.address
            )
        
        if update_data.contact_email:
            operator.contact_email = update_data.contact_email
        
        if update_data.contact_phone:
            operator.contact_phone = update_data.contact_phone
        
        if update_data.website:
            operator.website = update_data.website
        
        if update_data.status:
            operator.status = GridOperatorStatus(update_data.status.value)
        
        # Note: grid_capacity_mw, voltage_level_kv, coverage_area_km2, and metadata
        # are not available in the GridOperator entity, so we skip updating them
        
        # Save updated operator
        updated_operator = await operator_repo.update(operator)
        
        # Convert to response DTO
        response = GridOperatorResponseDTO(
            operator_id=updated_operator.operator_id,
            name=updated_operator.name,
            operator_type=updated_operator.operator_type.value,
            location={
                "latitude": updated_operator.headquarters.latitude,
                "longitude": updated_operator.headquarters.longitude,
                "address": updated_operator.headquarters.address
            },
            contact_email=updated_operator.contact_email,
            contact_phone=updated_operator.contact_phone,
            website=updated_operator.website,
            status=updated_operator.status.value,
            grid_capacity_mw=None,  # Not available in entity
            voltage_level_kv=None,  # Not available in entity
            coverage_area_km2=None,  # Not available in entity
            created_at=updated_operator.created_at,
            updated_at=updated_operator.updated_at,
            metadata=None,  # Not available in entity
            version=1  # Default version
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating grid operator: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update grid operator"
        )


@router.delete("/{operator_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_grid_operator(
    operator_id: str = Path(..., description="Grid operator ID"),
    operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository)
):
    """Delete grid operator"""
    try:
        # Check if operator exists
        operator = await operator_repo.get_by_id(operator_id)
        if not operator:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Grid operator {operator_id} not found"
            )
        
        # Delete operator
        await operator_repo.delete(operator_id)
        
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting grid operator: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete grid operator"
        )


@router.post("/{operator_id}/status", response_model=List[GridStatusResponseDTO], status_code=status.HTTP_201_CREATED)
async def create_grid_status(
    operator_id: str = Path(..., description="Grid operator ID"),
    status_data: GridStatusCreateDTO = ...,
    operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository)
):
    """Create a new grid status record"""
    try:
        # Check if operator exists
        operator = await operator_repo.get_by_id(operator_id)
        if not operator:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Grid operator {operator_id} not found"
            )
        
        # Convert DTO to domain entity
        from ...core.domain.entities.grid_operator import GridStatus
        
        status = GridStatus(
            operator_id=operator_id,
            timestamp=status_data.timestamp,
            voltage_level=status_data.voltage_level,
            frequency=status_data.frequency,
            load_percentage=status_data.load_percentage,
            stability_score=status_data.stability_score,
            power_quality_score=status_data.power_quality_score,
            total_generation_mw=status_data.total_generation_mw,
            total_consumption_mw=status_data.total_consumption_mw,
            grid_frequency_hz=status_data.grid_frequency_hz,
            voltage_deviation_percent=status_data.voltage_deviation_percent
        )
        
        # Save status
        saved_status = await operator_repo.add_status(status)
        
        # Convert to response DTO
        response = GridStatusResponseDTO(
            id=str(saved_status.id),
            operator_id=saved_status.operator_id,
            timestamp=saved_status.timestamp,
            voltage_level=saved_status.voltage_level,
            frequency=saved_status.frequency,
            load_percentage=saved_status.load_percentage,
            stability_score=saved_status.stability_score,
            power_quality_score=saved_status.power_quality_score,
            total_generation_mw=saved_status.total_generation_mw,
            total_consumption_mw=saved_status.total_consumption_mw,
            grid_frequency_hz=saved_status.grid_frequency_hz,
            voltage_deviation_percent=saved_status.voltage_deviation_percent,
            data_quality_score=getattr(saved_status, 'data_quality_score', 1.0),
            is_anomaly=getattr(saved_status, 'is_anomaly', False),
            anomaly_type=getattr(saved_status, 'anomaly_type', None),
            created_at=saved_status.created_at
        )
        
        return [response]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating grid status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create grid status"
        )


@router.post("/{operator_id}/status/batch", response_model=List[GridStatusResponseDTO], status_code=status.HTTP_201_CREATED)
async def create_grid_status_batch(
    operator_id: str = Path(..., description="Grid operator ID"),
    batch_data: GridStatusBatchDTO = ...,
    operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository),
    data_quality_service: DataQualityService = Depends(get_data_quality_service),
    anomaly_detection_service: AnomalyDetectionService = Depends(get_anomaly_detection_service),
    alerting_service: AlertingService = Depends(get_alerting_service),
    kafka_producer: KafkaProducer = Depends(get_kafka_producer),
    s3_client: S3Client = Depends(get_s3_client)
):
    """Create multiple grid status records in batch"""
    try:
        # Check if operator exists
        operator = await operator_repo.get_by_id(operator_id)
        if not operator:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Grid operator {operator_id} not found"
            )
        
        # Create use case
        process_use_case = ProcessGridStatusUseCase(
            grid_operator_repository=operator_repo,
            data_quality_service=data_quality_service,
            anomaly_detection_service=anomaly_detection_service,
            alerting_service=alerting_service,
            kafka_producer=kafka_producer,
            s3_client=s3_client
        )
        
        # Convert statuses to format expected by use case
        statuses_data = []
        for status in batch_data.statuses:
            statuses_data.append({
                'timestamp': status.timestamp.isoformat(),
                'voltage_level': status.voltage_level,
                'frequency': status.frequency,
                'load_percentage': status.load_percentage,
                'stability_score': status.stability_score,
                'power_quality_score': status.power_quality_score,
                'total_generation_mw': status.total_generation_mw,
                'total_consumption_mw': status.total_consumption_mw,
                'grid_frequency_hz': status.grid_frequency_hz,
                'voltage_deviation_percent': status.voltage_deviation_percent
            })
        
        # Execute use case
        result = await process_use_case.execute(
            operator_id=operator_id,
            status_data=statuses_data,
            metadata=batch_data.metadata
        )
        
        # Return success response
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "Batch status records created successfully",
                "statuses_processed": result.get("statuses_processed", 0),
                "quality_score": result.get("quality_score", 0.0),
                "anomalies_detected": result.get("anomalies_detected", 0),
                "stability_score": result.get("stability_score", 0.0),
                "alerts_sent": result.get("alerts_sent", 0)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating batch grid status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create batch grid status"
        )


@router.get("/{operator_id}/status", response_model=GridStatusListResponseDTO)
async def get_grid_status(
    operator_id: str = Path(..., description="Grid operator ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Page size"),
    start_time: Optional[str] = Query(None, description="Start time filter"),
    end_time: Optional[str] = Query(None, description="End time filter"),
    operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository)
):
    """Get grid status records with pagination and time filtering"""
    try:
        # Check if operator exists
        operator = await operator_repo.get_by_id(operator_id)
        if not operator:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Grid operator {operator_id} not found"
            )
        
        # Parse time filters
        from datetime import datetime
        start_dt = None
        end_dt = None
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        # Get statuses from repository
        statuses, total_count = await operator_repo.get_statuses_with_filters(
            operator_id=operator_id,
            start_time=start_dt,
            end_time=end_dt,
            page=page,
            page_size=page_size
        )
        
        # Convert to response DTOs
        status_responses = []
        for status in statuses:
            status_responses.append(GridStatusResponseDTO(
                id=str(status.id),
                operator_id=status.operator_id,
                timestamp=status.timestamp,
                voltage_level=status.voltage_level,
                frequency=status.frequency,
                load_percentage=status.load_percentage,
                stability_score=status.stability_score,
                power_quality_score=status.power_quality_score,
                total_generation_mw=status.total_generation_mw,
                total_consumption_mw=status.total_consumption_mw,
                grid_frequency_hz=status.grid_frequency_hz,
                voltage_deviation_percent=status.voltage_deviation_percent,
                data_quality_score=getattr(status, 'data_quality_score', 1.0),
                is_anomaly=getattr(status, 'is_anomaly', False),
                anomaly_type=getattr(status, 'anomaly_type', None),
                created_at=status.created_at
            ))
        
        # Calculate pagination info
        has_next = (page * page_size) < total_count
        has_previous = page > 1
        
        response = GridStatusListResponseDTO(
            statuses=status_responses,
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
        logger.error(f"Error getting grid status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get grid status"
        )


@router.get("/{operator_id}/analytics", response_model=GridOperatorAnalyticsDTO)
async def get_grid_operator_analytics(
    operator_id: str = Path(..., description="Grid operator ID"),
    period: str = Query("24h", description="Analysis period"),
    operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository)
):
    """Get grid operator analytics"""
    try:
        # Check if operator exists
        operator = await operator_repo.get_by_id(operator_id)
        if not operator:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Grid operator {operator_id} not found"
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
        
        # Get statuses for analysis
        statuses, _ = await operator_repo.get_statuses_with_filters(
            operator_id=operator_id,
            start_time=start_time,
            end_time=end_time,
            page=1,
            page_size=10000  # Large page size for analytics
        )
        
        if not statuses:
            return GridOperatorAnalyticsDTO(
                operator_id=operator_id,
                analysis_period=period,
                total_statuses=0,
                avg_voltage_level=0.0,
                avg_frequency=0.0,
                avg_load_percentage=0.0,
                avg_stability_score=0.0,
                avg_power_quality_score=0.0,
                avg_quality_score=0.0,
                anomaly_count=0,
                anomaly_rate=0.0,
                peak_load=0.0,
                peak_load_time=None,
                min_stability_score=0.0,
                min_stability_time=None
            )
        
        # Calculate analytics
        total_statuses = len(statuses)
        avg_voltage_level = sum(s.voltage_level for s in statuses) / total_statuses
        avg_frequency = sum(s.frequency for s in statuses) / total_statuses
        avg_load_percentage = sum(s.load_percentage for s in statuses) / total_statuses
        avg_stability_score = sum(s.stability_score for s in statuses) / total_statuses
        avg_power_quality_score = sum(s.power_quality_score for s in statuses) / total_statuses
        avg_quality_score = sum(getattr(s, 'data_quality_score', 1.0) for s in statuses) / total_statuses
        anomaly_count = sum(1 for s in statuses if getattr(s, 'is_anomaly', False))
        anomaly_rate = (anomaly_count / total_statuses) * 100 if total_statuses > 0 else 0.0
        
        # Find peak load
        peak_status = max(statuses, key=lambda s: s.load_percentage)
        peak_load = peak_status.load_percentage
        peak_load_time = peak_status.timestamp
        
        # Find minimum stability
        min_stability_status = min(statuses, key=lambda s: s.stability_score)
        min_stability_score = min_stability_status.stability_score
        min_stability_time = min_stability_status.timestamp
        
        response = GridOperatorAnalyticsDTO(
            operator_id=operator_id,
            analysis_period=period,
            total_statuses=total_statuses,
            avg_voltage_level=round(avg_voltage_level, 2),
            avg_frequency=round(avg_frequency, 2),
            avg_load_percentage=round(avg_load_percentage, 2),
            avg_stability_score=round(avg_stability_score, 3),
            avg_power_quality_score=round(avg_power_quality_score, 3),
            avg_quality_score=round(avg_quality_score, 3),
            anomaly_count=anomaly_count,
            anomaly_rate=round(anomaly_rate, 2),
            peak_load=round(peak_load, 2),
            peak_load_time=peak_load_time,
            min_stability_score=round(min_stability_score, 3),
            min_stability_time=min_stability_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting grid operator analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get grid operator analytics"
        )
