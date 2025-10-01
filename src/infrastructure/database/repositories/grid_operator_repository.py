"""
Grid Operator Repository Implementation
Concrete implementation of IGridOperatorRepository using SQLAlchemy
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from sqlalchemy.exc import IntegrityError

from ....core.domain.entities.grid_operator import GridOperator, GridStatus
from ....core.domain.value_objects.location import Location
from ....core.domain.enums.grid_operator_status import GridOperatorStatus
from ....core.interfaces.repositories.grid_operator_repository import IGridOperatorRepository
from ....core.exceptions.domain_exceptions import GridOperatorNotFoundError, DataQualityError
from ..models.grid_operator_model import GridOperatorModel, GridStatusModel, GridEventModel


class GridOperatorRepository(IGridOperatorRepository):
    """
    SQLAlchemy implementation of Grid Operator Repository
    
    Provides concrete implementation of data access operations
    for grid operator entities using PostgreSQL database.
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    async def get_by_id(self, operator_id: str) -> Optional[GridOperator]:
        """Get grid operator by ID"""
        model = self.db_session.query(GridOperatorModel).filter(
            GridOperatorModel.operator_id == operator_id
        ).first()
        
        if not model:
            return None
        
        return self._model_to_entity(model)
    
    async def save(self, operator: GridOperator) -> None:
        """Save grid operator entity"""
        try:
            # Check if operator exists
            existing_model = self.db_session.query(GridOperatorModel).filter(
                GridOperatorModel.operator_id == operator.operator_id
            ).first()
            
            if existing_model:
                # Update existing operator
                self._update_model_from_entity(existing_model, operator)
                model = existing_model
            else:
                # Create new operator
                model = self._entity_to_model(operator)
                self.db_session.add(model)
            
            # Save domain events
            await self._save_domain_events(operator, model)
            
            self.db_session.commit()
            
        except IntegrityError as e:
            self.db_session.rollback()
            raise DataQualityError(f"Failed to save operator {operator.operator_id}: {str(e)}")
        except Exception as e:
            self.db_session.rollback()
            raise DataQualityError(f"Unexpected error saving operator {operator.operator_id}: {str(e)}")
    
    async def delete(self, operator_id: str) -> None:
        """Delete grid operator"""
        model = self.db_session.query(GridOperatorModel).filter(
            GridOperatorModel.operator_id == operator_id
        ).first()
        
        if not model:
            raise GridOperatorNotFoundError(f"Grid operator {operator_id} not found")
        
        self.db_session.delete(model)
        self.db_session.commit()
    
    async def get_all(self) -> List[GridOperator]:
        """Get all grid operators"""
        models = self.db_session.query(GridOperatorModel).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_by_status(self, status: GridOperatorStatus) -> List[GridOperator]:
        """Get operators by status"""
        models = self.db_session.query(GridOperatorModel).filter(
            GridOperatorModel.status == status
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_by_operator_type(self, operator_type: str) -> List[GridOperator]:
        """Get operators by type"""
        models = self.db_session.query(GridOperatorModel).filter(
            GridOperatorModel.operator_type == operator_type
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_by_location(self, latitude: float, longitude: float, radius_km: float) -> List[GridOperator]:
        """Get operators within radius of location"""
        # Simple bounding box approximation
        lat_delta = radius_km / 111.0
        lng_delta = radius_km / (111.0 * abs(latitude))
        
        models = self.db_session.query(GridOperatorModel).filter(
            and_(
                GridOperatorModel.latitude >= latitude - lat_delta,
                GridOperatorModel.latitude <= latitude + lat_delta,
                GridOperatorModel.longitude >= longitude - lng_delta,
                GridOperatorModel.longitude <= longitude + lng_delta
            )
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_active_operators(self) -> List[GridOperator]:
        """Get active grid operators"""
        models = self.db_session.query(GridOperatorModel).filter(
            GridOperatorModel.status == GridOperatorStatus.ACTIVE
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_operators_requiring_attention(self) -> List[GridOperator]:
        """Get operators requiring attention"""
        # Get operators with low stability scores or old status updates
        cutoff_date = datetime.utcnow() - timedelta(hours=24)
        
        models = self.db_session.query(GridOperatorModel).filter(
            or_(
                GridOperatorModel.status == GridOperatorStatus.MAINTENANCE,
                GridOperatorModel.updated_at < cutoff_date
            )
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_status_by_date_range(
        self, 
        operator_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[GridStatus]:
        """Get grid status within date range"""
        models = self.db_session.query(GridStatusModel).filter(
            and_(
                GridStatusModel.operator_id == operator_id,
                GridStatusModel.timestamp >= start_date,
                GridStatusModel.timestamp <= end_date
            )
        ).order_by(asc(GridStatusModel.timestamp)).all()
        
        return [self._status_model_to_entity(model) for model in models]
    
    async def get_recent_status(self, operator_id: str, hours: int = 24) -> List[GridStatus]:
        """Get recent grid status"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        models = self.db_session.query(GridStatusModel).filter(
            and_(
                GridStatusModel.operator_id == operator_id,
                GridStatusModel.timestamp >= cutoff_time
            )
        ).order_by(desc(GridStatusModel.timestamp)).all()
        
        return [self._status_model_to_entity(model) for model in models]
    
    async def get_status_by_quality_threshold(
        self, 
        min_quality: float, 
        since: Optional[datetime] = None
    ) -> List[GridStatus]:
        """Get status above quality threshold"""
        query = self.db_session.query(GridStatusModel).filter(
            GridStatusModel.data_quality_score >= min_quality
        )
        
        if since:
            query = query.filter(GridStatusModel.timestamp >= since)
        
        models = query.order_by(desc(GridStatusModel.timestamp)).all()
        return [self._status_model_to_entity(model) for model in models]
    
    async def count_by_status(self, status: GridOperatorStatus) -> int:
        """Count operators by status"""
        return self.db_session.query(GridOperatorModel).filter(
            GridOperatorModel.status == status
        ).count()
    
    async def get_average_stability_score(self) -> float:
        """Get average stability score across all operators"""
        result = self.db_session.query(GridStatusModel.stability_score).all()
        if not result:
            return 0.0
        
        scores = [row[0] for row in result if row[0] is not None]
        return sum(scores) / len(scores) if scores else 0.0
    
    async def get_performance_statistics(self) -> dict:
        """Get performance statistics for all operators"""
        total_operators = self.db_session.query(GridOperatorModel).count()
        
        if total_operators == 0:
            return {
                "total_operators": 0,
                "active_operators": 0,
                "average_stability_score": 0.0,
                "operators_requiring_attention": 0,
                "total_status_updates": 0
            }
        
        active_operators = self.db_session.query(GridOperatorModel).filter(
            GridOperatorModel.status == GridOperatorStatus.ACTIVE
        ).count()
        
        maintenance_operators = self.db_session.query(GridOperatorModel).filter(
            GridOperatorModel.status == GridOperatorStatus.MAINTENANCE
        ).count()
        
        total_status_updates = self.db_session.query(GridStatusModel).count()
        
        avg_stability = await self.get_average_stability_score()
        
        return {
            "total_operators": total_operators,
            "active_operators": active_operators,
            "maintenance_operators": maintenance_operators,
            "average_stability_score": avg_stability,
            "operators_requiring_attention": maintenance_operators,
            "total_status_updates": total_status_updates
        }
    
    def _model_to_entity(self, model: GridOperatorModel) -> GridOperator:
        """Convert database model to domain entity"""
        location = Location(
            latitude=model.latitude,
            longitude=model.longitude,
            address=model.address
        )
        
        return GridOperator(
            operator_id=model.operator_id,
            name=model.name,
            operator_type=model.operator_type,
            location=location,
            contact_email=model.contact_email,
            contact_phone=model.contact_phone,
            website=model.website,
            status=model.status,
            grid_capacity_mw=model.grid_capacity_mw,
            voltage_level_kv=model.voltage_level_kv,
            coverage_area_km2=model.coverage_area_km2,
            metadata=model.metadata or {},
            version=model.version
        )
    
    def _entity_to_model(self, operator: GridOperator) -> GridOperatorModel:
        """Convert domain entity to database model"""
        return GridOperatorModel(
            operator_id=operator.operator_id,
            name=operator.name,
            operator_type=operator.operator_type,
            latitude=operator.location.latitude,
            longitude=operator.location.longitude,
            address=operator.location.address,
            contact_email=operator.contact_email,
            contact_phone=operator.contact_phone,
            website=operator.website,
            status=operator.status,
            grid_capacity_mw=operator.grid_capacity_mw,
            voltage_level_kv=operator.voltage_level_kv,
            coverage_area_km2=operator.coverage_area_km2,
            metadata=operator.metadata,
            version=operator.version
        )
    
    def _update_model_from_entity(self, model: GridOperatorModel, operator: GridOperator) -> None:
        """Update model from entity"""
        model.name = operator.name
        model.operator_type = operator.operator_type
        model.latitude = operator.location.latitude
        model.longitude = operator.location.longitude
        model.address = operator.location.address
        model.contact_email = operator.contact_email
        model.contact_phone = operator.contact_phone
        model.website = operator.website
        model.status = operator.status
        model.grid_capacity_mw = operator.grid_capacity_mw
        model.voltage_level_kv = operator.voltage_level_kv
        model.coverage_area_km2 = operator.coverage_area_km2
        model.metadata = operator.metadata
        model.version = operator.version
        model.updated_at = datetime.utcnow()
    
    def _status_model_to_entity(self, model: GridStatusModel) -> GridStatus:
        """Convert status model to domain entity"""
        return GridStatus(
            operator_id=model.operator_id,
            timestamp=model.timestamp,
            voltage_level=model.voltage_level,
            frequency=model.frequency,
            load_percentage=model.load_percentage,
            stability_score=model.stability_score,
            power_quality_score=model.power_quality_score,
            total_generation_mw=model.total_generation_mw,
            total_consumption_mw=model.total_consumption_mw,
            grid_frequency_hz=model.grid_frequency_hz,
            voltage_deviation_percent=model.voltage_deviation_percent,
            data_quality_score=model.data_quality_score,
            is_anomaly=model.is_anomaly,
            anomaly_type=model.anomaly_type
        )
    
    async def _save_domain_events(self, operator: GridOperator, model: GridOperatorModel) -> None:
        """Save domain events to database"""
        for event in operator.get_uncommitted_events():
            event_model = GridEventModel(
                operator_id=operator.operator_id,
                event_type=event.__class__.__name__,
                event_data=event.to_dict(),
                occurred_at=event.occurred_at,
                aggregate_version=operator.version,
                event_version=1
            )
            self.db_session.add(event_model)
        
        # Clear uncommitted events
        operator.clear_uncommitted_events()
