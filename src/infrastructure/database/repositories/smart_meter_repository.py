"""
Smart Meter Repository Implementation
Concrete implementation of ISmartMeterRepository using SQLAlchemy
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from sqlalchemy.exc import IntegrityError

from ....core.domain.entities.smart_meter import SmartMeter, MeterReading
from ....core.domain.value_objects.meter_id import MeterId
from ....core.domain.value_objects.location import Location
from ....core.domain.value_objects.meter_specifications import MeterSpecifications
from ....core.domain.enums.meter_status import MeterStatus
from ....core.domain.enums.quality_tier import QualityTier
from ....core.interfaces.repositories.smart_meter_repository import ISmartMeterRepository
from ....core.exceptions.domain_exceptions import MeterNotFoundError, DataQualityError
from ..models.smart_meter_model import SmartMeterModel, MeterReadingModel, MeterEventModel


class SmartMeterRepository(ISmartMeterRepository):
    """
    SQLAlchemy implementation of Smart Meter Repository
    
    Provides concrete implementation of data access operations
    for smart meter entities using PostgreSQL database.
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    async def get_by_id(self, meter_id: MeterId) -> Optional[SmartMeter]:
        """Get smart meter by ID"""
        model = self.db_session.query(SmartMeterModel).filter(
            SmartMeterModel.meter_id == meter_id.value
        ).first()
        
        if not model:
            return None
        
        return self._model_to_entity(model)
    
    async def save(self, meter: SmartMeter) -> None:
        """Save smart meter entity"""
        try:
            # Check if meter exists
            existing_model = self.db_session.query(SmartMeterModel).filter(
                SmartMeterModel.meter_id == meter.meter_id.value
            ).first()
            
            if existing_model:
                # Update existing meter
                self._update_model_from_entity(existing_model, meter)
                model = existing_model
            else:
                # Create new meter
                model = self._entity_to_model(meter)
                self.db_session.add(model)
            
            # Save domain events
            await self._save_domain_events(meter, model)
            
            self.db_session.commit()
            
        except IntegrityError as e:
            self.db_session.rollback()
            raise DataQualityError(f"Failed to save meter {meter.meter_id.value}: {str(e)}")
        except Exception as e:
            self.db_session.rollback()
            raise DataQualityError(f"Unexpected error saving meter {meter.meter_id.value}: {str(e)}")
    
    async def delete(self, meter_id: MeterId) -> None:
        """Delete smart meter"""
        model = self.db_session.query(SmartMeterModel).filter(
            SmartMeterModel.meter_id == meter_id.value
        ).first()
        
        if not model:
            raise MeterNotFoundError(f"Meter {meter_id.value} not found")
        
        self.db_session.delete(model)
        self.db_session.commit()
    
    async def get_all(self) -> List[SmartMeter]:
        """Get all smart meters"""
        models = self.db_session.query(SmartMeterModel).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_by_status(self, status: MeterStatus) -> List[SmartMeter]:
        """Get meters by status"""
        models = self.db_session.query(SmartMeterModel).filter(
            SmartMeterModel.status == status
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_by_quality_tier(self, quality_tier: QualityTier) -> List[SmartMeter]:
        """Get meters by quality tier"""
        models = self.db_session.query(SmartMeterModel).filter(
            SmartMeterModel.quality_tier == quality_tier
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_by_location(self, latitude: float, longitude: float, radius_km: float) -> List[SmartMeter]:
        """Get meters within radius of location"""
        # Simple bounding box approximation (not precise for large distances)
        lat_delta = radius_km / 111.0  # Approximate km per degree latitude
        lng_delta = radius_km / (111.0 * abs(latitude))  # Approximate km per degree longitude
        
        models = self.db_session.query(SmartMeterModel).filter(
            and_(
                SmartMeterModel.latitude >= latitude - lat_delta,
                SmartMeterModel.latitude <= latitude + lat_delta,
                SmartMeterModel.longitude >= longitude - lng_delta,
                SmartMeterModel.longitude <= longitude + lng_delta
            )
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_requiring_maintenance(self) -> List[SmartMeter]:
        """Get meters requiring maintenance"""
        # Get meters with low quality scores or old last reading
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        models = self.db_session.query(SmartMeterModel).filter(
            or_(
                SmartMeterModel.quality_tier == QualityTier.POOR,
                SmartMeterModel.last_reading_at < cutoff_date,
                SmartMeterModel.status == MeterStatus.MAINTENANCE
            )
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_readings_by_date_range(
        self, 
        meter_id: MeterId, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[MeterReading]:
        """Get meter readings within date range"""
        models = self.db_session.query(MeterReadingModel).filter(
            and_(
                MeterReadingModel.meter_id == meter_id.value,
                MeterReadingModel.timestamp >= start_date,
                MeterReadingModel.timestamp <= end_date
            )
        ).order_by(asc(MeterReadingModel.timestamp)).all()
        
        return [self._reading_model_to_entity(model) for model in models]
    
    async def get_recent_readings(self, meter_id: MeterId, hours: int = 24) -> List[MeterReading]:
        """Get recent meter readings"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        models = self.db_session.query(MeterReadingModel).filter(
            and_(
                MeterReadingModel.meter_id == meter_id.value,
                MeterReadingModel.timestamp >= cutoff_time
            )
        ).order_by(desc(MeterReadingModel.timestamp)).all()
        
        return [self._reading_model_to_entity(model) for model in models]
    
    async def get_readings_by_quality_threshold(
        self, 
        min_quality: float, 
        since: Optional[datetime] = None
    ) -> List[MeterReading]:
        """Get readings above quality threshold"""
        query = self.db_session.query(MeterReadingModel).filter(
            MeterReadingModel.data_quality_score >= min_quality
        )
        
        if since:
            query = query.filter(MeterReadingModel.timestamp >= since)
        
        models = query.order_by(desc(MeterReadingModel.timestamp)).all()
        return [self._reading_model_to_entity(model) for model in models]
    
    async def count_by_status(self, status: MeterStatus) -> int:
        """Count meters by status"""
        return self.db_session.query(SmartMeterModel).filter(
            SmartMeterModel.status == status
        ).count()
    
    async def get_average_quality_score(self) -> float:
        """Get average quality score across all meters"""
        result = self.db_session.query(SmartMeterModel.quality_tier).all()
        if not result:
            return 0.0
        
        # Convert quality tiers to numeric scores
        tier_scores = {
            QualityTier.EXCELLENT: 1.0,
            QualityTier.GOOD: 0.8,
            QualityTier.FAIR: 0.6,
            QualityTier.POOR: 0.4,
            QualityTier.UNKNOWN: 0.0
        }
        
        scores = [tier_scores.get(tier[0], 0.0) for tier in result]
        return sum(scores) / len(scores) if scores else 0.0
    
    async def get_performance_statistics(self) -> dict:
        """Get performance statistics for all meters"""
        total_meters = self.db_session.query(SmartMeterModel).count()
        
        if total_meters == 0:
            return {
                "total_meters": 0,
                "active_meters": 0,
                "average_quality_score": 0.0,
                "meters_requiring_maintenance": 0,
                "total_readings": 0
            }
        
        active_meters = self.db_session.query(SmartMeterModel).filter(
            SmartMeterModel.status == MeterStatus.ACTIVE
        ).count()
        
        maintenance_meters = self.db_session.query(SmartMeterModel).filter(
            SmartMeterModel.status == MeterStatus.MAINTENANCE
        ).count()
        
        total_readings = self.db_session.query(MeterReadingModel).count()
        
        avg_quality = await self.get_average_quality_score()
        
        return {
            "total_meters": total_meters,
            "active_meters": active_meters,
            "maintenance_meters": maintenance_meters,
            "average_quality_score": avg_quality,
            "meters_requiring_maintenance": maintenance_meters,
            "total_readings": total_readings
        }
    
    def _model_to_entity(self, model: SmartMeterModel) -> SmartMeter:
        """Convert database model to domain entity"""
        meter_id = MeterId(model.meter_id)
        location = Location(
            latitude=model.latitude,
            longitude=model.longitude,
            address=model.address
        )
        specifications = MeterSpecifications(
            manufacturer=model.manufacturer,
            model=model.model,
            installation_date=model.installation_date.isoformat()
        )
        
        return SmartMeter(
            meter_id=meter_id,
            location=location,
            specifications=specifications,
            status=model.status,
            quality_tier=model.quality_tier,
            installed_at=model.installed_at,
            last_reading_at=model.last_reading_at,
            firmware_version=model.firmware_version,
            metadata=model.metadata or {},
            version=model.version
        )
    
    def _entity_to_model(self, meter: SmartMeter) -> SmartMeterModel:
        """Convert domain entity to database model"""
        return SmartMeterModel(
            meter_id=meter.meter_id.value,
            latitude=meter.location.latitude,
            longitude=meter.location.longitude,
            address=meter.location.address,
            manufacturer=meter.specifications.manufacturer,
            model=meter.specifications.model,
            installation_date=datetime.fromisoformat(meter.specifications.installation_date),
            status=meter.status,
            quality_tier=meter.quality_tier,
            installed_at=meter.installed_at,
            last_reading_at=meter.last_reading_at,
            firmware_version=meter.firmware_version,
            metadata=meter.metadata,
            version=meter.version
        )
    
    def _update_model_from_entity(self, model: SmartMeterModel, meter: SmartMeter) -> None:
        """Update model from entity"""
        model.latitude = meter.location.latitude
        model.longitude = meter.location.longitude
        model.address = meter.location.address
        model.manufacturer = meter.specifications.manufacturer
        model.model = meter.specifications.model
        model.installation_date = datetime.fromisoformat(meter.specifications.installation_date)
        model.status = meter.status
        model.quality_tier = meter.quality_tier
        model.installed_at = meter.installed_at
        model.last_reading_at = meter.last_reading_at
        model.firmware_version = meter.firmware_version
        model.metadata = meter.metadata
        model.version = meter.version
        model.updated_at = datetime.utcnow()
    
    def _reading_model_to_entity(self, model: MeterReadingModel) -> MeterReading:
        """Convert reading model to domain entity"""
        return MeterReading(
            timestamp=model.timestamp,
            voltage=model.voltage,
            current=model.current,
            power_factor=model.power_factor,
            frequency=model.frequency,
            active_power=model.active_power,
            reactive_power=model.reactive_power,
            apparent_power=model.apparent_power,
            data_quality_score=model.data_quality_score,
            is_anomaly=model.is_anomaly,
            anomaly_type=model.anomaly_type
        )
    
    async def _save_domain_events(self, meter: SmartMeter, model: SmartMeterModel) -> None:
        """Save domain events to database"""
        for event in meter.get_uncommitted_events():
            event_model = MeterEventModel(
                meter_id=meter.meter_id.value,
                event_type=event.__class__.__name__,
                event_data=event.to_dict(),
                occurred_at=event.occurred_at,
                aggregate_version=meter.version,
                event_version=1
            )
            self.db_session.add(event_model)
        
        # Clear uncommitted events
        meter.clear_uncommitted_events()
