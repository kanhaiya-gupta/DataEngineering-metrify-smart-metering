"""
Read Model Synchronization
Read model synchronization for CQRS pattern implementation
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import threading
import json

from ..sourcing.event_store import EventStore, Event, EventType

logger = logging.getLogger(__name__)

class SyncStatus(Enum):
    """Synchronization status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class ProjectionType(Enum):
    """Projection type"""
    AGGREGATE = "aggregate"
    CROSS_AGGREGATE = "cross_aggregate"
    TEMPORAL = "temporal"
    STATISTICAL = "statistical"

@dataclass
class ReadModelProjection:
    """Read model projection definition"""
    projection_id: str
    projection_type: ProjectionType
    event_types: List[EventType]
    projection_function: Callable[[Event, Dict[str, Any]], Dict[str, Any]]
    read_model_store: Dict[str, Any]
    last_processed_event_id: Optional[str] = None
    last_processed_timestamp: Optional[datetime] = None
    created_at: datetime = None

@dataclass
class SyncResult:
    """Synchronization result"""
    projection_id: str
    status: SyncStatus
    events_processed: int
    events_failed: int
    start_time: datetime
    end_time: Optional[datetime]
    error_message: Optional[str] = None
    progress_percentage: float = 0.0

class ReadModelSync:
    """
    Read model synchronization for CQRS pattern
    """
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.projections: Dict[str, ReadModelProjection] = {}
        self.active_syncs: Dict[str, SyncResult] = {}
        self.sync_lock = threading.RLock()
        
        logger.info("ReadModelSync initialized")
    
    def register_projection(self,
                          projection_id: str,
                          projection_type: ProjectionType,
                          event_types: List[EventType],
                          projection_function: Callable[[Event, Dict[str, Any]], Dict[str, Any]],
                          read_model_store: Dict[str, Any]) -> bool:
        """Register a read model projection"""
        try:
            with self.sync_lock:
                projection = ReadModelProjection(
                    projection_id=projection_id,
                    projection_type=projection_type,
                    event_types=event_types,
                    projection_function=projection_function,
                    read_model_store=read_model_store,
                    created_at=datetime.now()
                )
                
                self.projections[projection_id] = projection
                logger.info(f"Registered projection: {projection_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register projection {projection_id}: {str(e)}")
            return False
    
    def unregister_projection(self, projection_id: str) -> bool:
        """Unregister a read model projection"""
        try:
            with self.sync_lock:
                if projection_id in self.projections:
                    del self.projections[projection_id]
                    logger.info(f"Unregistered projection: {projection_id}")
                    return True
                else:
                    logger.warning(f"Projection {projection_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to unregister projection {projection_id}: {str(e)}")
            return False
    
    def sync_projection(self, projection_id: str, from_timestamp: datetime = None) -> str:
        """Start synchronizing a projection"""
        try:
            if projection_id not in self.projections:
                raise ValueError(f"Projection {projection_id} not found")
            
            projection = self.projections[projection_id]
            
            # Create sync result
            sync_result = SyncResult(
                projection_id=projection_id,
                status=SyncStatus.PENDING,
                events_processed=0,
                events_failed=0,
                start_time=datetime.now()
            )
            
            with self.sync_lock:
                self.active_syncs[projection_id] = sync_result
            
            # Start sync in background thread
            thread = threading.Thread(
                target=self._run_sync,
                args=(projection_id, from_timestamp),
                daemon=True
            )
            thread.start()
            
            logger.info(f"Started sync for projection {projection_id}")
            return projection_id
            
        except Exception as e:
            logger.error(f"Failed to start sync for projection {projection_id}: {str(e)}")
            raise
    
    def _run_sync(self, projection_id: str, from_timestamp: datetime = None):
        """Run projection sync in background thread"""
        try:
            projection = self.projections[projection_id]
            
            with self.sync_lock:
                if projection_id not in self.active_syncs:
                    return
                
                self.active_syncs[projection_id].status = SyncStatus.RUNNING
            
            # Get events to process
            events = self._get_events_for_sync(projection, from_timestamp)
            total_events = len(events)
            
            if total_events == 0:
                self._complete_sync(projection_id, "No events to process")
                return
            
            # Process events
            processed = 0
            failed = 0
            
            for event in events:
                try:
                    # Check if sync was stopped
                    with self.sync_lock:
                        if projection_id not in self.active_syncs:
                            return
                        
                        if self.active_syncs[projection_id].status == SyncStatus.PAUSED:
                            # Wait until resumed
                            while (projection_id in self.active_syncs and 
                                   self.active_syncs[projection_id].status == SyncStatus.PAUSED):
                                threading.Event().wait(0.1)
                            
                            if projection_id not in self.active_syncs:
                                return
                    
                    # Apply projection
                    projection.projection_function(event, projection.read_model_store)
                    processed += 1
                    
                    # Update projection state
                    projection.last_processed_event_id = event.event_id
                    projection.last_processed_timestamp = event.timestamp
                    
                except Exception as e:
                    failed += 1
                    logger.error(f"Error processing event {event.event_id} for projection {projection_id}: {str(e)}")
                
                # Update progress
                with self.sync_lock:
                    if projection_id in self.active_syncs:
                        progress = (processed + failed) / total_events * 100
                        self.active_syncs[projection_id].events_processed = processed
                        self.active_syncs[projection_id].events_failed = failed
                        self.active_syncs[projection_id].progress_percentage = progress
            
            # Complete sync
            self._complete_sync(projection_id, None)
            
        except Exception as e:
            logger.error(f"Sync failed for projection {projection_id}: {str(e)}")
            self._fail_sync(projection_id, str(e))
    
    def _get_events_for_sync(self, projection: ReadModelProjection, from_timestamp: datetime = None) -> List[Event]:
        """Get events to process for sync"""
        try:
            events = []
            
            # Determine start timestamp
            start_timestamp = from_timestamp
            if not start_timestamp and projection.last_processed_timestamp:
                start_timestamp = projection.last_processed_timestamp
            
            # Get events by type
            for event_type in projection.event_types:
                type_events = self.event_store.get_events_by_type(
                    event_type=event_type,
                    from_timestamp=start_timestamp,
                    limit=10000  # Large limit for sync
                )
                events.extend(type_events)
            
            # Sort by timestamp
            events.sort(key=lambda e: e.timestamp)
            
            # Filter out already processed events
            if projection.last_processed_event_id:
                filtered_events = []
                for event in events:
                    if event.event_id == projection.last_processed_event_id:
                        # Start from next event
                        continue
                    filtered_events.append(event)
                events = filtered_events
            
            logger.debug(f"Retrieved {len(events)} events for projection {projection.projection_id}")
            return events
            
        except Exception as e:
            logger.error(f"Failed to get events for sync: {str(e)}")
            return []
    
    def _complete_sync(self, projection_id: str, message: str = None):
        """Complete a sync"""
        try:
            with self.sync_lock:
                if projection_id in self.active_syncs:
                    self.active_syncs[projection_id].status = SyncStatus.COMPLETED
                    self.active_syncs[projection_id].end_time = datetime.now()
                    if message:
                        self.active_syncs[projection_id].error_message = message
                    
                    logger.info(f"Sync completed for projection {projection_id}: {self.active_syncs[projection_id].events_processed} events processed")
                    
        except Exception as e:
            logger.error(f"Failed to complete sync {projection_id}: {str(e)}")
    
    def _fail_sync(self, projection_id: str, error_message: str):
        """Fail a sync"""
        try:
            with self.sync_lock:
                if projection_id in self.active_syncs:
                    self.active_syncs[projection_id].status = SyncStatus.FAILED
                    self.active_syncs[projection_id].end_time = datetime.now()
                    self.active_syncs[projection_id].error_message = error_message
                    
                    logger.error(f"Sync failed for projection {projection_id}: {error_message}")
                    
        except Exception as e:
            logger.error(f"Failed to fail sync {projection_id}: {str(e)}")
    
    def pause_sync(self, projection_id: str) -> bool:
        """Pause an active sync"""
        try:
            with self.sync_lock:
                if projection_id in self.active_syncs:
                    if self.active_syncs[projection_id].status == SyncStatus.RUNNING:
                        self.active_syncs[projection_id].status = SyncStatus.PAUSED
                        logger.info(f"Paused sync for projection {projection_id}")
                        return True
                    else:
                        logger.warning(f"Cannot pause sync {projection_id} with status {self.active_syncs[projection_id].status}")
                        return False
                else:
                    logger.warning(f"Sync {projection_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to pause sync {projection_id}: {str(e)}")
            return False
    
    def resume_sync(self, projection_id: str) -> bool:
        """Resume a paused sync"""
        try:
            with self.sync_lock:
                if projection_id in self.active_syncs:
                    if self.active_syncs[projection_id].status == SyncStatus.PAUSED:
                        self.active_syncs[projection_id].status = SyncStatus.RUNNING
                        logger.info(f"Resumed sync for projection {projection_id}")
                        return True
                    else:
                        logger.warning(f"Cannot resume sync {projection_id} with status {self.active_syncs[projection_id].status}")
                        return False
                else:
                    logger.warning(f"Sync {projection_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to resume sync {projection_id}: {str(e)}")
            return False
    
    def stop_sync(self, projection_id: str) -> bool:
        """Stop an active sync"""
        try:
            with self.sync_lock:
                if projection_id in self.active_syncs:
                    del self.active_syncs[projection_id]
                    logger.info(f"Stopped sync for projection {projection_id}")
                    return True
                else:
                    logger.warning(f"Sync {projection_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to stop sync {projection_id}: {str(e)}")
            return False
    
    def get_sync_status(self, projection_id: str) -> Optional[SyncResult]:
        """Get sync status"""
        try:
            with self.sync_lock:
                return self.active_syncs.get(projection_id)
                
        except Exception as e:
            logger.error(f"Failed to get sync status for {projection_id}: {str(e)}")
            return None
    
    def get_all_syncs(self) -> List[SyncResult]:
        """Get all active syncs"""
        try:
            with self.sync_lock:
                return list(self.active_syncs.values())
                
        except Exception as e:
            logger.error(f"Failed to get all syncs: {str(e)}")
            return []
    
    def get_projection_statistics(self) -> Dict[str, Any]:
        """Get projection statistics"""
        try:
            with self.sync_lock:
                stats = {
                    "total_projections": len(self.projections),
                    "active_syncs": len(self.active_syncs),
                    "projections": {}
                }
                
                for projection_id, projection in self.projections.items():
                    stats["projections"][projection_id] = {
                        "type": projection.projection_type.value,
                        "event_types": [et.value for et in projection.event_types],
                        "last_processed_event_id": projection.last_processed_event_id,
                        "last_processed_timestamp": projection.last_processed_timestamp.isoformat() if projection.last_processed_timestamp else None,
                        "created_at": projection.created_at.isoformat()
                    }
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get projection statistics: {str(e)}")
            return {}
    
    def reset_projection(self, projection_id: str) -> bool:
        """Reset a projection to start from beginning"""
        try:
            if projection_id not in self.projections:
                return False
            
            projection = self.projections[projection_id]
            projection.last_processed_event_id = None
            projection.last_processed_timestamp = None
            
            logger.info(f"Reset projection {projection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset projection {projection_id}: {str(e)}")
            return False
    
    def cleanup_completed_syncs(self, older_than_hours: int = 24) -> int:
        """Clean up completed syncs older than specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            cleaned_count = 0
            
            with self.sync_lock:
                syncs_to_remove = []
                for projection_id, result in self.active_syncs.items():
                    if (result.status in [SyncStatus.COMPLETED, SyncStatus.FAILED] and
                        result.end_time and result.end_time < cutoff_time):
                        syncs_to_remove.append(projection_id)
                
                for projection_id in syncs_to_remove:
                    del self.active_syncs[projection_id]
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} completed syncs")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup completed syncs: {str(e)}")
            return 0

# Example projection functions
def smart_meter_projection(event: Event, read_model_store: Dict[str, Any]) -> None:
    """Projection function for smart meter events"""
    try:
        if event.event_name == "SmartMeterCreated":
            meter_data = {
                "meter_id": event.data.get("meter_id"),
                "location": event.data.get("location"),
                "meter_type": event.data.get("meter_type"),
                "installation_date": event.data.get("installation_date"),
                "status": "active",
                "created_at": event.timestamp.isoformat(),
                "updated_at": event.timestamp.isoformat()
            }
            
            if "meters" not in read_model_store:
                read_model_store["meters"] = {}
            
            read_model_store["meters"][meter_data["meter_id"]] = meter_data
            
        elif event.event_name == "SmartMeterUpdated":
            meter_id = event.data.get("meter_id")
            updates = event.data.get("updates", {})
            
            if "meters" in read_model_store and meter_id in read_model_store["meters"]:
                read_model_store["meters"][meter_id].update(updates)
                read_model_store["meters"][meter_id]["updated_at"] = event.timestamp.isoformat()
                
        elif event.event_name == "MeterReadingRecorded":
            meter_id = event.data.get("meter_id")
            reading_data = {
                "meter_id": meter_id,
                "value": event.data.get("reading_value"),
                "timestamp": event.data.get("reading_timestamp"),
                "quality_score": event.data.get("quality_score", 1.0),
                "recorded_at": event.timestamp.isoformat()
            }
            
            if "readings" not in read_model_store:
                read_model_store["readings"] = {}
            
            if meter_id not in read_model_store["readings"]:
                read_model_store["readings"][meter_id] = []
            
            read_model_store["readings"][meter_id].append(reading_data)
            
        elif event.event_name == "SmartMeterDeactivated":
            meter_id = event.data.get("meter_id")
            
            if "meters" in read_model_store and meter_id in read_model_store["meters"]:
                read_model_store["meters"][meter_id]["status"] = "inactive"
                read_model_store["meters"][meter_id]["deactivated_at"] = event.data.get("deactivated_at")
                read_model_store["meters"][meter_id]["updated_at"] = event.timestamp.isoformat()
                
    except Exception as e:
        logger.error(f"Failed to apply smart meter projection: {str(e)}")

def grid_operator_projection(event: Event, read_model_store: Dict[str, Any]) -> None:
    """Projection function for grid operator events"""
    try:
        if event.event_name == "GridOperatorCreated":
            operator_data = {
                "operator_id": event.data.get("operator_id"),
                "operator_name": event.data.get("operator_name"),
                "region": event.data.get("region"),
                "contact_info": event.data.get("contact_info"),
                "created_at": event.timestamp.isoformat(),
                "updated_at": event.timestamp.isoformat()
            }
            
            if "operators" not in read_model_store:
                read_model_store["operators"] = {}
            
            read_model_store["operators"][operator_data["operator_id"]] = operator_data
            
        elif event.event_name == "GridOperatorUpdated":
            operator_id = event.data.get("operator_id")
            updates = event.data.get("updates", {})
            
            if "operators" in read_model_store and operator_id in read_model_store["operators"]:
                read_model_store["operators"][operator_id].update(updates)
                read_model_store["operators"][operator_id]["updated_at"] = event.timestamp.isoformat()
                
        elif event.event_name == "GridOperatorDataReceived":
            operator_id = event.data.get("operator_id")
            data_item = {
                "operator_id": operator_id,
                "data_type": event.data.get("data_type"),
                "data": event.data.get("data"),
                "timestamp": event.data.get("timestamp"),
                "received_at": event.timestamp.isoformat()
            }
            
            if "operator_data" not in read_model_store:
                read_model_store["operator_data"] = {}
            
            if operator_id not in read_model_store["operator_data"]:
                read_model_store["operator_data"][operator_id] = []
            
            read_model_store["operator_data"][operator_id].append(data_item)
            
    except Exception as e:
        logger.error(f"Failed to apply grid operator projection: {str(e)}")
