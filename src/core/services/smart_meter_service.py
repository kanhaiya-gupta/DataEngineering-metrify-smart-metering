"""
Smart Meter Service
Business logic for smart meter operations
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..domain.entities.smart_meter import SmartMeter, MeterReading
from ..domain.value_objects.meter_id import MeterId
from ..domain.value_objects.location import Location
from ..domain.value_objects.meter_specifications import MeterSpecifications
from ..domain.enums.meter_status import MeterStatus
from ..domain.enums.quality_tier import QualityTier
from ..interfaces.repositories.smart_meter_repository import ISmartMeterRepository
from ..interfaces.external.data_quality_service import IDataQualityService
from ..interfaces.external.anomaly_detection_service import IAnomalyDetectionService
from ..exceptions.domain_exceptions import MeterNotFoundError, InvalidMeterOperationError


@dataclass
class MeterRegistrationRequest:
    """Request to register a new smart meter"""
    meter_id: str
    location: Location
    specifications: MeterSpecifications
    initial_status: MeterStatus = MeterStatus.PENDING


@dataclass
class MeterReadingRequest:
    """Request to record a meter reading"""
    meter_id: str
    timestamp: datetime
    consumption_kwh: float
    voltage: float
    current: float
    power_factor: float
    frequency: float
    temperature: Optional[float] = None
    humidity: Optional[float] = None


@dataclass
class MeterMaintenanceRequest:
    """Request to schedule maintenance for a meter"""
    meter_id: str
    reason: str
    priority: str
    scheduled_at: Optional[datetime] = None
    estimated_duration_hours: Optional[int] = None


class SmartMeterService:
    """
    Smart Meter Service
    
    Contains all business logic related to smart meter operations.
    This service orchestrates domain entities and coordinates with external services.
    """
    
    def __init__(
        self,
        meter_repository: ISmartMeterRepository,
        data_quality_service: IDataQualityService,
        anomaly_detection_service: IAnomalyDetectionService
    ):
        self.meter_repository = meter_repository
        self.data_quality_service = data_quality_service
        self.anomaly_detection_service = anomaly_detection_service
    
    async def register_meter(self, request: MeterRegistrationRequest) -> SmartMeter:
        """
        Register a new smart meter
        
        Args:
            request: Meter registration request
            
        Returns:
            Registered smart meter
            
        Raises:
            InvalidMeterOperationError: If meter ID already exists
        """
        # Check if meter already exists
        existing_meter = await self.meter_repository.get_by_id(MeterId(request.meter_id))
        if existing_meter:
            raise InvalidMeterOperationError(f"Meter {request.meter_id} already exists")
        
        # Create new meter
        meter = SmartMeter(
            meter_id=MeterId(request.meter_id),
            location=request.location,
            specifications=request.specifications,
            status=request.initial_status
        )
        
        # Register the meter (this will add domain events)
        meter.register_meter()
        
        # Save to repository
        await self.meter_repository.save(meter)
        
        return meter
    
    async def get_meter(self, meter_id: str) -> SmartMeter:
        """
        Get a smart meter by ID
        
        Args:
            meter_id: Meter ID
            
        Returns:
            Smart meter
            
        Raises:
            MeterNotFoundError: If meter doesn't exist
        """
        meter = await self.meter_repository.get_by_id(MeterId(meter_id))
        if not meter:
            raise MeterNotFoundError(f"Meter {meter_id} not found")
        
        return meter
    
    async def record_reading(self, request: MeterReadingRequest) -> SmartMeter:
        """
        Record a new meter reading
        
        Args:
            request: Meter reading request
            
        Returns:
            Updated smart meter
            
        Raises:
            MeterNotFoundError: If meter doesn't exist
            InvalidMeterOperationError: If meter cannot record readings
        """
        # Get the meter
        meter = await self.get_meter(request.meter_id)
        
        # Create meter reading
        reading = MeterReading(
            timestamp=request.timestamp,
            consumption_kwh=request.consumption_kwh,
            voltage=request.voltage,
            current=request.current,
            power_factor=request.power_factor,
            frequency=request.frequency,
            temperature=request.temperature,
            humidity=request.humidity
        )
        
        # Calculate data quality score
        quality_score = await self.data_quality_service.calculate_quality_score(reading)
        reading.data_quality_score = quality_score
        
        # Check for anomalies
        anomalies = await self.anomaly_detection_service.detect_anomalies(reading, meter)
        if anomalies:
            # Handle anomalies (this could trigger alerts, maintenance, etc.)
            await self._handle_anomalies(meter, anomalies)
        
        # Record the reading
        meter.record_reading(reading)
        
        # Save updated meter
        await self.meter_repository.save(meter)
        
        return meter
    
    async def deactivate_meter(self, meter_id: str, reason: str) -> SmartMeter:
        """
        Deactivate a smart meter
        
        Args:
            meter_id: Meter ID
            reason: Reason for deactivation
            
        Returns:
            Deactivated smart meter
        """
        meter = await self.get_meter(meter_id)
        meter.deactivate_meter(reason)
        
        await self.meter_repository.save(meter)
        return meter
    
    async def schedule_maintenance(self, request: MeterMaintenanceRequest) -> SmartMeter:
        """
        Schedule maintenance for a meter
        
        Args:
            request: Maintenance request
            
        Returns:
            Updated smart meter
        """
        meter = await self.get_meter(request.meter_id)
        
        # Schedule maintenance
        meter._schedule_maintenance(request.reason)
        
        await self.meter_repository.save(meter)
        return meter
    
    async def perform_maintenance(self, meter_id: str, maintenance_notes: str) -> SmartMeter:
        """
        Perform maintenance on a meter
        
        Args:
            meter_id: Meter ID
            maintenance_notes: Notes about the maintenance performed
            
        Returns:
            Updated smart meter
        """
        meter = await self.get_meter(meter_id)
        meter.perform_maintenance(maintenance_notes)
        
        await self.meter_repository.save(meter)
        return meter
    
    async def get_meters_by_status(self, status: MeterStatus) -> List[SmartMeter]:
        """
        Get all meters with a specific status
        
        Args:
            status: Meter status
            
        Returns:
            List of meters with the specified status
        """
        return await self.meter_repository.get_by_status(status)
    
    async def get_meters_requiring_maintenance(self) -> List[SmartMeter]:
        """
        Get all meters that require maintenance
        
        Returns:
            List of meters requiring maintenance
        """
        all_meters = await self.meter_repository.get_all()
        return [meter for meter in all_meters if meter.is_maintenance_due()]
    
    async def get_meters_by_quality_tier(self, quality_tier: QualityTier) -> List[SmartMeter]:
        """
        Get all meters with a specific quality tier
        
        Args:
            quality_tier: Quality tier
            
        Returns:
            List of meters with the specified quality tier
        """
        all_meters = await self.meter_repository.get_all()
        return [meter for meter in all_meters if meter.quality_tier == quality_tier]
    
    async def get_meter_performance_summary(self, meter_id: str) -> Dict[str, Any]:
        """
        Get performance summary for a meter
        
        Args:
            meter_id: Meter ID
            
        Returns:
            Performance summary dictionary
        """
        meter = await self.get_meter(meter_id)
        
        return {
            "meter_id": meter.meter_id.value,
            "status": meter.status.value,
            "total_readings": meter.total_readings,
            "average_quality_score": meter.average_quality_score,
            "quality_tier": meter.quality_tier.value,
            "performance_score": meter.get_performance_score(),
            "health_status": meter.get_health_status(),
            "is_maintenance_due": meter.is_maintenance_due(),
            "last_reading_at": meter.last_reading_at.isoformat() if meter.last_reading_at else None,
            "created_at": meter.created_at.isoformat(),
            "updated_at": meter.updated_at.isoformat()
        }
    
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """
        Get overall system health summary
        
        Returns:
            System health summary dictionary
        """
        all_meters = await self.meter_repository.get_all()
        
        if not all_meters:
            return {
                "total_meters": 0,
                "active_meters": 0,
                "average_quality_score": 0.0,
                "meters_requiring_attention": 0,
                "maintenance_due_count": 0
            }
        
        active_meters = [m for m in all_meters if m.status == MeterStatus.ACTIVE]
        meters_requiring_attention = [m for m in all_meters if m.requires_attention()]
        maintenance_due = [m for m in all_meters if m.is_maintenance_due()]
        
        average_quality_score = sum(m.average_quality_score for m in all_meters) / len(all_meters)
        
        return {
            "total_meters": len(all_meters),
            "active_meters": len(active_meters),
            "average_quality_score": average_quality_score,
            "meters_requiring_attention": len(meters_requiring_attention),
            "maintenance_due_count": len(maintenance_due),
            "quality_distribution": {
                tier.value: len([m for m in all_meters if m.quality_tier == tier])
                for tier in QualityTier
            },
            "status_distribution": {
                status.value: len([m for m in all_meters if m.status == status])
                for status in MeterStatus
            }
        }
    
    async def _handle_anomalies(self, meter: SmartMeter, anomalies: List[Dict[str, Any]]) -> None:
        """
        Handle detected anomalies
        
        Args:
            meter: Smart meter with anomalies
            anomalies: List of detected anomalies
        """
        # Log anomalies
        for anomaly in anomalies:
            print(f"Anomaly detected in meter {meter.meter_id.value}: {anomaly}")
        
        # If multiple anomalies, consider scheduling maintenance
        if len(anomalies) >= 3:
            meter._schedule_maintenance("Multiple anomalies detected")
        
        # Update meter status if necessary
        if len(anomalies) >= 5:
            meter.status = MeterStatus.ERROR
