"""
Smart Meter Repository Implementation
Concrete implementation of ISmartMeterRepository using SQLAlchemy
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.exc import IntegrityError

from ....core.domain.entities.smart_meter import SmartMeter, MeterReading
from ....core.domain.value_objects.meter_id import MeterId
from ....core.domain.value_objects.location import Location
from ....core.domain.value_objects.meter_specifications import MeterSpecifications
from ....core.domain.enums.meter_status import MeterStatus
from ....core.domain.enums.quality_tier import QualityTier
from ....core.interfaces.repositories.smart_meter_repository import ISmartMeterRepository
from ....core.exceptions.domain_exceptions import MeterNotFoundError, DataQualityError
from ..models.smart_meter_model import SmartMeterModel, SmartMeterReadingModel, MeterEventModel


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
        models = self.db_session.query(SmartMeterReadingModel).filter(
            and_(
                SmartMeterReadingModel.meter_id == meter_id.value,
                SmartMeterReadingModel.timestamp >= start_date,
                SmartMeterReadingModel.timestamp <= end_date
            )
        ).order_by(asc(SmartMeterReadingModel.timestamp)).all()
        
        return [self._reading_model_to_entity(model) for model in models]
    
    async def get_recent_readings(self, meter_id: MeterId, hours: int = 24) -> List[MeterReading]:
        """Get recent meter readings"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        models = self.db_session.query(SmartMeterReadingModel).filter(
            and_(
                SmartMeterReadingModel.meter_id == meter_id.value,
                SmartMeterReadingModel.timestamp >= cutoff_time
            )
        ).order_by(desc(SmartMeterReadingModel.timestamp)).all()
        
        return [self._reading_model_to_entity(model) for model in models]
    
    async def get_readings_by_quality_threshold(
        self, 
        min_quality: float, 
        since: Optional[datetime] = None
    ) -> List[MeterReading]:
        """Get readings above quality threshold"""
        query = self.db_session.query(SmartMeterReadingModel).filter(
            SmartMeterReadingModel.data_quality_score >= min_quality
        )
        
        if since:
            query = query.filter(SmartMeterReadingModel.timestamp >= since)
        
        models = query.order_by(desc(SmartMeterReadingModel.timestamp)).all()
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
        
        total_readings = self.db_session.query(SmartMeterReadingModel).count()
        
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
            installed_at=getattr(model, 'installed_at', None),
            last_reading_at=model.last_reading_at,
            firmware_version=getattr(model, 'firmware_version', None),
            metadata=getattr(model, 'metadata', {}),
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
            installation_date=datetime.fromisoformat(meter.specifications.installation_date) if meter.specifications.installation_date else None,
            status=meter.status,
            quality_tier=meter.quality_tier,
            installed_at=getattr(meter, 'installed_at', None),
            last_reading_at=meter.last_reading_at,
            firmware_version=getattr(meter, 'firmware_version', None),
            metadata=getattr(meter, 'metadata', {}),
        )
    
    def _update_model_from_entity(self, model: SmartMeterModel, meter: SmartMeter) -> None:
        """Update model from entity"""
        model.latitude = meter.location.latitude
        model.longitude = meter.location.longitude
        model.address = meter.location.address
        model.manufacturer = meter.specifications.manufacturer
        model.model = meter.specifications.model
        model.installation_date = datetime.fromisoformat(meter.specifications.installation_date) if meter.specifications.installation_date else None
        model.status = meter.status
        model.quality_tier = meter.quality_tier
        model.installed_at = getattr(meter, 'installed_at', None)
        model.last_reading_at = meter.last_reading_at
        model.firmware_version = getattr(meter, 'firmware_version', None)
        model.metadata = getattr(meter, 'metadata', {})
        model.updated_at = datetime.utcnow()
    
    def _reading_model_to_entity(self, model: SmartMeterReadingModel) -> MeterReading:
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
                aggregate_version=1,
                event_version=1
            )
            self.db_session.add(event_model)
        
        # Clear uncommitted events
        meter.clear_uncommitted_events()
    
    async def get_by_manufacturer(self, manufacturer: str) -> List[SmartMeter]:
        """Get smart meters by manufacturer"""
        try:
            models = self.db_session.query(SmartMeterModel).filter(
                SmartMeterModel.manufacturer == manufacturer
            ).all()
            return [self._model_to_entity(model) for model in models]
        except Exception as e:
            raise DataQualityError(f"Failed to get meters by manufacturer: {str(e)}")
    
    async def get_with_anomalies(self, since: Optional[datetime] = None) -> List[SmartMeter]:
        """Get smart meters with anomalies since given date"""
        try:
            query = self.db_session.query(SmartMeterModel).join(SmartMeterReadingModel).filter(
                SmartMeterReadingModel.is_anomaly == True
            )
            if since:
                query = query.filter(SmartMeterReadingModel.timestamp >= since)
            models = query.distinct().all()
            return [self._model_to_entity(model) for model in models]
        except Exception as e:
            raise DataQualityError(f"Failed to get meters with anomalies: {str(e)}")
    
    async def get_by_performance_score_range(self, min_score: float, max_score: float) -> List[SmartMeter]:
        """Get smart meters by performance score range (using quality_tier as proxy)"""
        try:
            # Map score range to quality tiers
            # 0.0-0.3 = POOR, 0.3-0.7 = FAIR, 0.7-1.0 = GOOD/EXCELLENT
            quality_tiers = []
            if min_score <= 0.3:
                quality_tiers.append(QualityTier.POOR)
            if min_score <= 0.7 and max_score >= 0.3:
                quality_tiers.append(QualityTier.FAIR)
            if max_score >= 0.7:
                quality_tiers.extend([QualityTier.GOOD, QualityTier.EXCELLENT])
            
            if not quality_tiers:
                return []
                
            models = self.db_session.query(SmartMeterModel).filter(
                SmartMeterModel.quality_tier.in_(quality_tiers)
            ).all()
            return [self._model_to_entity(model) for model in models]
        except Exception as e:
            raise DataQualityError(f"Failed to get meters by performance score: {str(e)}")
    
    async def count_by_quality_tier(self, quality_tier: QualityTier) -> int:
        """Count smart meters by quality tier"""
        try:
            count = self.db_session.query(SmartMeterModel).filter(
                SmartMeterModel.quality_tier == quality_tier.value
            ).count()
            return count
        except Exception as e:
            raise DataQualityError(f"Failed to count meters by quality tier: {str(e)}")
    
    async def get_meters_with_filters(self, filters: dict) -> tuple[List[SmartMeter], int]:
        """Get smart meters with filters and return (meters, total_count)"""
        try:
            query = self.db_session.query(SmartMeterModel)
            
            # Apply filters
            if filters.get('status'):
                query = query.filter(SmartMeterModel.status == filters['status'])
            if filters.get('quality_tier'):
                query = query.filter(SmartMeterModel.quality_tier == filters['quality_tier'])
            if filters.get('manufacturer'):
                query = query.filter(SmartMeterModel.manufacturer == filters['manufacturer'])
            
            # Get total count before pagination
            total_count = query.count()
            
            # Apply pagination
            if filters.get('offset'):
                query = query.offset(filters['offset'])
            if filters.get('limit'):
                query = query.limit(filters['limit'])
            
            models = query.all()
            meters = [self._model_to_entity(model) for model in models]
            return meters, total_count
        except Exception as e:
            raise DataQualityError(f"Failed to get meters with filters: {str(e)}")
    
    async def get_total_count(self) -> int:
        """Get total count of smart meters"""
        try:
            return self.db_session.query(SmartMeterModel).count()
        except Exception as e:
            raise DataQualityError(f"Failed to get total count: {str(e)}")
    
    async def get_active_count(self) -> int:
        """Get count of active smart meters"""
        try:
            return self.db_session.query(SmartMeterModel).filter(
                SmartMeterModel.status == MeterStatus.ACTIVE
            ).count()
        except Exception as e:
            raise DataQualityError(f"Failed to get active count: {str(e)}")
    
    async def get_readings_count_in_period(self, start_time: datetime, end_time: datetime) -> int:
        """Get count of readings in a time period"""
        try:
            return self.db_session.query(SmartMeterReadingModel).filter(
                SmartMeterReadingModel.timestamp >= start_time,
                SmartMeterReadingModel.timestamp <= end_time
            ).count()
        except Exception as e:
            raise DataQualityError(f"Failed to get readings count: {str(e)}")
    
    async def get_average_quality_score(self) -> float:
        """Get average quality score of smart meters"""
        try:
            result = self.db_session.query(
                func.avg(SmartMeterReadingModel.data_quality_score)
            ).scalar()
            return float(result) if result is not None else 0.0
        except Exception as e:
            raise DataQualityError(f"Failed to get average quality score: {str(e)}")
    
    async def get_anomaly_rate(self) -> float:
        """Get anomaly rate for smart meters"""
        try:
            total_readings = self.db_session.query(SmartMeterReadingModel).count()
            if total_readings == 0:
                return 0.0
            
            anomaly_count = self.db_session.query(SmartMeterReadingModel).filter(
                SmartMeterReadingModel.is_anomaly == True
            ).count()
            
            return (anomaly_count / total_readings) * 100
        except Exception as e:
            raise DataQualityError(f"Failed to get anomaly rate: {str(e)}")
    
    async def get_data_quality_metrics(self, start_time: datetime, end_time: datetime) -> dict:
        """Get data quality metrics for a time period"""
        try:
            # Get total records in period
            total_records = self.db_session.query(SmartMeterReadingModel).filter(
                SmartMeterReadingModel.timestamp >= start_time,
                SmartMeterReadingModel.timestamp <= end_time
            ).count()
            
            # Get quality issues
            missing_data = self.db_session.query(SmartMeterReadingModel).filter(
                SmartMeterReadingModel.timestamp >= start_time,
                SmartMeterReadingModel.timestamp <= end_time,
                SmartMeterReadingModel.active_power.is_(None)
            ).count()
            
            invalid_data = self.db_session.query(SmartMeterReadingModel).filter(
                SmartMeterReadingModel.timestamp >= start_time,
                SmartMeterReadingModel.timestamp <= end_time,
                SmartMeterReadingModel.active_power < 0
            ).count()
            
            outliers = self.db_session.query(SmartMeterReadingModel).filter(
                SmartMeterReadingModel.timestamp >= start_time,
                SmartMeterReadingModel.timestamp <= end_time,
                SmartMeterReadingModel.is_anomaly == True
            ).count()
            
            # Calculate quality score
            quality_issues = missing_data + invalid_data + outliers
            quality_score = max(0, (total_records - quality_issues) / max(total_records, 1)) * 100
            
            return {
                'total_records': total_records,
                'avg_quality_score': quality_score,
                'quality_trend': 0.0,  # Placeholder
                'quality_issues': quality_issues,
                'missing_data': missing_data,
                'invalid_data': invalid_data,
                'outliers': outliers
            }
        except Exception as e:
            raise DataQualityError(f"Failed to get data quality metrics: {str(e)}")
    
    async def get_daily_stats(self, start_time: datetime, end_time: datetime) -> dict:
        """Get daily statistics for smart meters"""
        try:
            total_readings = self.db_session.query(SmartMeterReadingModel).filter(
                SmartMeterReadingModel.timestamp >= start_time,
                SmartMeterReadingModel.timestamp <= end_time
            ).count()
            
            avg_quality_score = self.db_session.query(
                func.avg(SmartMeterReadingModel.data_quality_score)
            ).scalar() or 0.0
            
            anomaly_count = self.db_session.query(SmartMeterReadingModel).filter(
                SmartMeterReadingModel.timestamp >= start_time,
                SmartMeterReadingModel.timestamp <= end_time,
                SmartMeterReadingModel.is_anomaly == True
            ).count()
            
            return {
                'total_readings': total_readings,
                'avg_quality_score': float(avg_quality_score),
                'anomaly_count': anomaly_count
            }
        except Exception as e:
            raise DataQualityError(f"Failed to get daily stats: {str(e)}")