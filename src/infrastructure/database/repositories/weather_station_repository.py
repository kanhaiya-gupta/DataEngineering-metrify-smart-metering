"""
Weather Station Repository Implementation
Concrete implementation of IWeatherStationRepository using SQLAlchemy
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from sqlalchemy.exc import IntegrityError

from ....core.domain.entities.weather_station import WeatherStation, WeatherObservation
from ....core.domain.enums.weather_station_status import WeatherStationStatus
from ....core.interfaces.repositories.weather_station_repository import IWeatherStationRepository
from ....core.exceptions.domain_exceptions import WeatherStationNotFoundError, DataQualityError
from ..models.weather_station_model import WeatherStationModel, WeatherObservationModel, WeatherEventModel


class WeatherStationRepository(IWeatherStationRepository):
    """
    SQLAlchemy implementation of Weather Station Repository
    
    Provides concrete implementation of data access operations
    for weather station entities using PostgreSQL database.
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    async def get_by_id(self, station_id: str) -> Optional[WeatherStation]:
        """Get weather station by ID"""
        model = self.db_session.query(WeatherStationModel).filter(
            WeatherStationModel.station_id == station_id
        ).first()
        
        if not model:
            return None
        
        return self._model_to_entity(model)
    
    async def save(self, station: WeatherStation) -> None:
        """Save weather station entity"""
        try:
            # Check if station exists
            existing_model = self.db_session.query(WeatherStationModel).filter(
                WeatherStationModel.station_id == station.station_id
            ).first()
            
            if existing_model:
                # Update existing station
                self._update_model_from_entity(existing_model, station)
                model = existing_model
            else:
                # Create new station
                model = self._entity_to_model(station)
                self.db_session.add(model)
            
            # Save domain events
            await self._save_domain_events(station, model)
            
            self.db_session.commit()
            
        except IntegrityError as e:
            self.db_session.rollback()
            raise DataQualityError(f"Failed to save station {station.station_id}: {str(e)}")
        except Exception as e:
            self.db_session.rollback()
            raise DataQualityError(f"Unexpected error saving station {station.station_id}: {str(e)}")
    
    async def delete(self, station_id: str) -> None:
        """Delete weather station"""
        model = self.db_session.query(WeatherStationModel).filter(
            WeatherStationModel.station_id == station_id
        ).first()
        
        if not model:
            raise WeatherStationNotFoundError(f"Weather station {station_id} not found")
        
        self.db_session.delete(model)
        self.db_session.commit()
    
    async def get_all(self) -> List[WeatherStation]:
        """Get all weather stations"""
        models = self.db_session.query(WeatherStationModel).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_by_status(self, status: WeatherStationStatus) -> List[WeatherStation]:
        """Get stations by status"""
        models = self.db_session.query(WeatherStationModel).filter(
            WeatherStationModel.status == status
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_by_operator(self, operator: str) -> List[WeatherStation]:
        """Get stations by operator"""
        models = self.db_session.query(WeatherStationModel).filter(
            WeatherStationModel.operator == operator
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_by_location(self, latitude: float, longitude: float, radius_km: float) -> List[WeatherStation]:
        """Get stations within radius of location"""
        # Simple bounding box approximation
        lat_delta = radius_km / 111.0
        lng_delta = radius_km / (111.0 * abs(latitude))
        
        models = self.db_session.query(WeatherStationModel).filter(
            and_(
                WeatherStationModel.latitude >= latitude - lat_delta,
                WeatherStationModel.latitude <= latitude + lat_delta,
                WeatherStationModel.longitude >= longitude - lng_delta,
                WeatherStationModel.longitude <= longitude + lng_delta
            )
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_operational(self) -> List[WeatherStation]:
        """Get operational weather stations"""
        models = self.db_session.query(WeatherStationModel).filter(
            WeatherStationModel.status == WeatherStationStatus.ACTIVE
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_requiring_attention(self) -> List[WeatherStation]:
        """Get stations requiring attention"""
        # Get stations with low quality scores or old observations
        cutoff_date = datetime.utcnow() - timedelta(hours=2)
        
        models = self.db_session.query(WeatherStationModel).filter(
            or_(
                WeatherStationModel.status != WeatherStationStatus.ACTIVE,
                WeatherStationModel.last_observation_at < cutoff_date,
                WeatherStationModel.average_quality_score < 0.5
            )
        ).all()
        return [self._model_to_entity(model) for model in models]
    
    async def get_observations_by_date_range(
        self, 
        station_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[WeatherObservation]:
        """Get weather observations within date range"""
        models = self.db_session.query(WeatherObservationModel).filter(
            and_(
                WeatherObservationModel.station_id == station_id,
                WeatherObservationModel.timestamp >= start_date,
                WeatherObservationModel.timestamp <= end_date
            )
        ).order_by(asc(WeatherObservationModel.timestamp)).all()
        
        return [self._observation_model_to_entity(model) for model in models]
    
    async def get_recent_observations(self, station_id: str, hours: int = 24) -> List[WeatherObservation]:
        """Get recent weather observations"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        models = self.db_session.query(WeatherObservationModel).filter(
            and_(
                WeatherObservationModel.station_id == station_id,
                WeatherObservationModel.timestamp >= cutoff_time
            )
        ).order_by(desc(WeatherObservationModel.timestamp)).all()
        
        return [self._observation_model_to_entity(model) for model in models]
    
    async def get_observations_by_quality_threshold(
        self, 
        min_quality: float, 
        since: Optional[datetime] = None
    ) -> List[WeatherObservation]:
        """Get observations above quality threshold"""
        query = self.db_session.query(WeatherObservationModel).filter(
            WeatherObservationModel.data_quality_score >= min_quality
        )
        
        if since:
            query = query.filter(WeatherObservationModel.timestamp >= since)
        
        models = query.order_by(desc(WeatherObservationModel.timestamp)).all()
        return [self._observation_model_to_entity(model) for model in models]
    
    async def count_by_status(self, status: WeatherStationStatus) -> int:
        """Count stations by status"""
        return self.db_session.query(WeatherStationModel).filter(
            WeatherStationModel.status == status
        ).count()
    
    async def get_average_quality_score(self) -> float:
        """Get average quality score across all stations"""
        result = self.db_session.query(WeatherStationModel.average_quality_score).all()
        if not result:
            return 0.0
        
        scores = [row[0] for row in result if row[0] is not None]
        return sum(scores) / len(scores) if scores else 0.0
    
    async def get_performance_statistics(self) -> dict:
        """Get performance statistics for all stations"""
        total_stations = self.db_session.query(WeatherStationModel).count()
        
        if total_stations == 0:
            return {
                "total_stations": 0,
                "operational_stations": 0,
                "average_quality_score": 0.0,
                "stations_requiring_attention": 0,
                "total_observations": 0
            }
        
        operational_stations = self.db_session.query(WeatherStationModel).filter(
            WeatherStationModel.status == WeatherStationStatus.ACTIVE
        ).count()
        
        attention_stations = len(await self.get_requiring_attention())
        
        total_observations = self.db_session.query(WeatherObservationModel).count()
        
        avg_quality = await self.get_average_quality_score()
        
        return {
            "total_stations": total_stations,
            "operational_stations": operational_stations,
            "average_quality_score": avg_quality,
            "stations_requiring_attention": attention_stations,
            "total_observations": total_observations
        }
    
    def _model_to_entity(self, model: WeatherStationModel) -> WeatherStation:
        """Convert database model to domain entity"""
        from ....core.domain.value_objects.location import Location
        
        location = Location(
            latitude=model.latitude,
            longitude=model.longitude,
            address=model.address
        )
        
        return WeatherStation(
            station_id=model.station_id,
            name=model.name,
            station_type=model.station_type,
            location=location,
            operator=model.operator,
            contact_email=model.contact_email,
            contact_phone=model.contact_phone,
            status=model.status,
            created_at=model.created_at,
            updated_at=model.updated_at,
            total_observations=model.total_observations,
            average_quality_score=model.average_quality_score,
            last_observation_at=model.last_observation_at,
            version=model.version
        )
    
    def _entity_to_model(self, station: WeatherStation) -> WeatherStationModel:
        """Convert domain entity to database model"""
        return WeatherStationModel(
            station_id=station.station_id,
            name=station.name,
            station_type=station.station_type,
            latitude=station.location.latitude,
            longitude=station.location.longitude,
            address=station.location.address,
            operator=station.operator,
            contact_email=station.contact_email,
            contact_phone=station.contact_phone,
            status=station.status,
            total_observations=station.total_observations,
            average_quality_score=station.average_quality_score,
            last_observation_at=station.last_observation_at,
            version=station.version
        )
    
    def _update_model_from_entity(self, model: WeatherStationModel, station: WeatherStation) -> None:
        """Update model from entity"""
        model.name = station.name
        model.station_type = station.station_type
        model.latitude = station.location.latitude
        model.longitude = station.location.longitude
        model.address = station.location.address
        model.operator = station.operator
        model.contact_email = station.contact_email
        model.contact_phone = station.contact_phone
        model.status = station.status
        model.total_observations = station.total_observations
        model.average_quality_score = station.average_quality_score
        model.last_observation_at = station.last_observation_at
        model.version = station.version
        model.updated_at = datetime.utcnow()
    
    def _observation_model_to_entity(self, model: WeatherObservationModel) -> WeatherObservation:
        """Convert observation model to domain entity"""
        return WeatherObservation(
            timestamp=model.timestamp,
            temperature_celsius=model.temperature_celsius,
            humidity_percent=model.humidity_percent,
            pressure_hpa=model.pressure_hpa,
            wind_speed_ms=model.wind_speed_ms,
            wind_direction_degrees=model.wind_direction_degrees,
            cloud_cover_percent=model.cloud_cover_percent,
            visibility_km=model.visibility_km,
            uv_index=model.uv_index,
            precipitation_mm=model.precipitation_mm,
            data_quality_score=model.data_quality_score
        )
    
    async def _save_domain_events(self, station: WeatherStation, model: WeatherStationModel) -> None:
        """Save domain events to database"""
        for event in station.get_uncommitted_events():
            event_model = WeatherEventModel(
                station_id=station.station_id,
                event_type=event.__class__.__name__,
                event_data=event.to_dict(),
                occurred_at=event.occurred_at,
                aggregate_version=station.version,
                event_version=1
            )
            self.db_session.add(event_model)
        
        # Clear uncommitted events
        station.clear_uncommitted_events()
    
    async def get_stations_with_filters(self, filters: dict) -> tuple[List[WeatherStation], int]:
        """Get weather stations with filters and return (stations, total_count)"""
        try:
            query = self.db_session.query(WeatherStationModel)
            
            # Apply filters
            if filters.get('status'):
                query = query.filter(WeatherStationModel.status == filters['status'])
            if filters.get('operator'):
                query = query.filter(WeatherStationModel.operator == filters['operator'])
            if filters.get('station_type'):
                query = query.filter(WeatherStationModel.station_type == filters['station_type'])
            
            # Get total count before pagination
            total_count = query.count()
            
            # Apply pagination
            if filters.get('offset'):
                query = query.offset(filters['offset'])
            if filters.get('limit'):
                query = query.limit(filters['limit'])
            
            models = query.all()
            stations = [self._model_to_entity(model) for model in models]
            return stations, total_count
        except Exception as e:
            raise DataQualityError(f"Failed to get stations with filters: {str(e)}")