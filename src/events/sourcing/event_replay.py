"""
Event Replay
Event replay capabilities for state reconstruction and debugging
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import time

from .event_store import EventStore, Event, EventType, EventStatus

logger = logging.getLogger(__name__)

class ReplayStatus(Enum):
    """Replay status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class ReplayConfig:
    """Replay configuration"""
    from_timestamp: Optional[datetime] = None
    to_timestamp: Optional[datetime] = None
    from_version: int = 0
    to_version: Optional[int] = None
    event_types: List[EventType] = None
    aggregate_ids: List[str] = None
    batch_size: int = 1000
    delay_between_batches: float = 0.0
    stop_on_error: bool = True

@dataclass
class ReplayResult:
    """Replay result"""
    replay_id: str
    status: ReplayStatus
    events_processed: int
    events_failed: int
    start_time: datetime
    end_time: Optional[datetime]
    error_message: Optional[str] = None
    progress_percentage: float = 0.0

class EventReplay:
    """
    Event replay capabilities for state reconstruction and debugging
    """
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.active_replays = {}
        self.replay_lock = threading.RLock()
        
        logger.info("EventReplay initialized")
    
    def start_replay(self,
                    config: ReplayConfig,
                    event_handler: Callable[[Event], bool],
                    replay_id: str = None) -> str:
        """Start an event replay"""
        try:
            if replay_id is None:
                replay_id = f"replay_{int(time.time())}"
            
            with self.replay_lock:
                if replay_id in self.active_replays:
                    raise ValueError(f"Replay {replay_id} is already active")
                
                # Create replay result
                result = ReplayResult(
                    replay_id=replay_id,
                    status=ReplayStatus.PENDING,
                    events_processed=0,
                    events_failed=0,
                    start_time=datetime.now()
                )
                
                self.active_replays[replay_id] = result
                
                # Start replay in background thread
                thread = threading.Thread(
                    target=self._run_replay,
                    args=(replay_id, config, event_handler),
                    daemon=True
                )
                thread.start()
                
                logger.info(f"Started replay {replay_id}")
                return replay_id
                
        except Exception as e:
            logger.error(f"Failed to start replay: {str(e)}")
            raise
    
    def _run_replay(self,
                   replay_id: str,
                   config: ReplayConfig,
                   event_handler: Callable[[Event], bool]):
        """Run the replay in background thread"""
        try:
            with self.replay_lock:
                if replay_id not in self.active_replays:
                    return
                
                self.active_replays[replay_id].status = ReplayStatus.RUNNING
            
            # Get events to replay
            events = self._get_events_for_replay(config)
            total_events = len(events)
            
            if total_events == 0:
                self._complete_replay(replay_id, "No events to replay")
                return
            
            # Process events in batches
            processed = 0
            failed = 0
            
            for i in range(0, total_events, config.batch_size):
                batch = events[i:i + config.batch_size]
                
                # Check if replay was stopped
                with self.replay_lock:
                    if replay_id not in self.active_replays:
                        return
                    
                    if self.active_replays[replay_id].status == ReplayStatus.PAUSED:
                        # Wait until resumed
                        while (replay_id in self.active_replays and 
                               self.active_replays[replay_id].status == ReplayStatus.PAUSED):
                            time.sleep(0.1)
                        
                        if replay_id not in self.active_replays:
                            return
                
                # Process batch
                for event in batch:
                    try:
                        success = event_handler(event)
                        if success:
                            processed += 1
                        else:
                            failed += 1
                            
                        if config.stop_on_error and not success:
                            self._fail_replay(replay_id, f"Event handler returned false for event {event.event_id}")
                            return
                            
                    except Exception as e:
                        failed += 1
                        logger.error(f"Error processing event {event.event_id}: {str(e)}")
                        
                        if config.stop_on_error:
                            self._fail_replay(replay_id, f"Error processing event {event.event_id}: {str(e)}")
                            return
                
                # Update progress
                with self.replay_lock:
                    if replay_id in self.active_replays:
                        progress = (processed + failed) / total_events * 100
                        self.active_replays[replay_id].events_processed = processed
                        self.active_replays[replay_id].events_failed = failed
                        self.active_replays[replay_id].progress_percentage = progress
                
                # Delay between batches
                if config.delay_between_batches > 0:
                    time.sleep(config.delay_between_batches)
            
            # Complete replay
            self._complete_replay(replay_id, None)
            
        except Exception as e:
            logger.error(f"Replay {replay_id} failed: {str(e)}")
            self._fail_replay(replay_id, str(e))
    
    def _get_events_for_replay(self, config: ReplayConfig) -> List[Event]:
        """Get events to replay based on configuration"""
        try:
            events = []
            
            # If specific aggregate IDs are provided
            if config.aggregate_ids:
                for aggregate_id in config.aggregate_ids:
                    aggregate_events = self.event_store.get_events(
                        aggregate_id=aggregate_id,
                        from_version=config.from_version,
                        to_version=config.to_version
                    )
                    events.extend(aggregate_events)
            else:
                # Get events by type and time range
                for event_type in (config.event_types or [EventType.DOMAIN_EVENT]):
                    type_events = self.event_store.get_events_by_type(
                        event_type=event_type,
                        from_timestamp=config.from_timestamp,
                        to_timestamp=config.to_timestamp,
                        limit=10000  # Large limit for replay
                    )
                    events.extend(type_events)
            
            # Sort by timestamp
            events.sort(key=lambda e: e.timestamp)
            
            # Apply version filters if needed
            if config.from_version > 0 or config.to_version is not None:
                filtered_events = []
                for event in events:
                    if config.from_version > 0 and event.version < config.from_version:
                        continue
                    if config.to_version is not None and event.version > config.to_version:
                        continue
                    filtered_events.append(event)
                events = filtered_events
            
            logger.debug(f"Retrieved {len(events)} events for replay")
            return events
            
        except Exception as e:
            logger.error(f"Failed to get events for replay: {str(e)}")
            return []
    
    def _complete_replay(self, replay_id: str, message: str = None):
        """Complete a replay"""
        try:
            with self.replay_lock:
                if replay_id in self.active_replays:
                    self.active_replays[replay_id].status = ReplayStatus.COMPLETED
                    self.active_replays[replay_id].end_time = datetime.now()
                    if message:
                        self.active_replays[replay_id].error_message = message
                    
                    logger.info(f"Replay {replay_id} completed: {self.active_replays[replay_id].events_processed} events processed")
                    
        except Exception as e:
            logger.error(f"Failed to complete replay {replay_id}: {str(e)}")
    
    def _fail_replay(self, replay_id: str, error_message: str):
        """Fail a replay"""
        try:
            with self.replay_lock:
                if replay_id in self.active_replays:
                    self.active_replays[replay_id].status = ReplayStatus.FAILED
                    self.active_replays[replay_id].end_time = datetime.now()
                    self.active_replays[replay_id].error_message = error_message
                    
                    logger.error(f"Replay {replay_id} failed: {error_message}")
                    
        except Exception as e:
            logger.error(f"Failed to fail replay {replay_id}: {str(e)}")
    
    def pause_replay(self, replay_id: str) -> bool:
        """Pause an active replay"""
        try:
            with self.replay_lock:
                if replay_id in self.active_replays:
                    if self.active_replays[replay_id].status == ReplayStatus.RUNNING:
                        self.active_replays[replay_id].status = ReplayStatus.PAUSED
                        logger.info(f"Paused replay {replay_id}")
                        return True
                    else:
                        logger.warning(f"Cannot pause replay {replay_id} with status {self.active_replays[replay_id].status}")
                        return False
                else:
                    logger.warning(f"Replay {replay_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to pause replay {replay_id}: {str(e)}")
            return False
    
    def resume_replay(self, replay_id: str) -> bool:
        """Resume a paused replay"""
        try:
            with self.replay_lock:
                if replay_id in self.active_replays:
                    if self.active_replays[replay_id].status == ReplayStatus.PAUSED:
                        self.active_replays[replay_id].status = ReplayStatus.RUNNING
                        logger.info(f"Resumed replay {replay_id}")
                        return True
                    else:
                        logger.warning(f"Cannot resume replay {replay_id} with status {self.active_replays[replay_id].status}")
                        return False
                else:
                    logger.warning(f"Replay {replay_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to resume replay {replay_id}: {str(e)}")
            return False
    
    def stop_replay(self, replay_id: str) -> bool:
        """Stop an active replay"""
        try:
            with self.replay_lock:
                if replay_id in self.active_replays:
                    del self.active_replays[replay_id]
                    logger.info(f"Stopped replay {replay_id}")
                    return True
                else:
                    logger.warning(f"Replay {replay_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to stop replay {replay_id}: {str(e)}")
            return False
    
    def get_replay_status(self, replay_id: str) -> Optional[ReplayResult]:
        """Get replay status"""
        try:
            with self.replay_lock:
                return self.active_replays.get(replay_id)
                
        except Exception as e:
            logger.error(f"Failed to get replay status for {replay_id}: {str(e)}")
            return None
    
    def get_all_replays(self) -> List[ReplayResult]:
        """Get all active replays"""
        try:
            with self.replay_lock:
                return list(self.active_replays.values())
                
        except Exception as e:
            logger.error(f"Failed to get all replays: {str(e)}")
            return []
    
    def replay_aggregate_events(self,
                              aggregate_id: str,
                              event_handler: Callable[[Event], bool],
                              from_version: int = 0,
                              to_version: int = None) -> ReplayResult:
        """Replay events for a specific aggregate"""
        try:
            config = ReplayConfig(
                aggregate_ids=[aggregate_id],
                from_version=from_version,
                to_version=to_version,
                batch_size=100
            )
            
            replay_id = self.start_replay(config, event_handler)
            
            # Wait for completion
            while True:
                status = self.get_replay_status(replay_id)
                if status is None or status.status in [ReplayStatus.COMPLETED, ReplayStatus.FAILED]:
                    break
                time.sleep(0.1)
            
            return status or ReplayResult(
                replay_id=replay_id,
                status=ReplayStatus.FAILED,
                events_processed=0,
                events_failed=0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message="Replay not found"
            )
            
        except Exception as e:
            logger.error(f"Failed to replay aggregate events for {aggregate_id}: {str(e)}")
            return ReplayResult(
                replay_id="",
                status=ReplayStatus.FAILED,
                events_processed=0,
                events_failed=0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=str(e)
            )
    
    def replay_events_by_time_range(self,
                                  from_timestamp: datetime,
                                  to_timestamp: datetime,
                                  event_handler: Callable[[Event], bool],
                                  event_types: List[EventType] = None) -> ReplayResult:
        """Replay events within a time range"""
        try:
            config = ReplayConfig(
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
                event_types=event_types or [EventType.DOMAIN_EVENT],
                batch_size=1000
            )
            
            replay_id = self.start_replay(config, event_handler)
            
            # Wait for completion
            while True:
                status = self.get_replay_status(replay_id)
                if status is None or status.status in [ReplayStatus.COMPLETED, ReplayStatus.FAILED]:
                    break
                time.sleep(0.1)
            
            return status or ReplayResult(
                replay_id=replay_id,
                status=ReplayStatus.FAILED,
                events_processed=0,
                events_failed=0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message="Replay not found"
            )
            
        except Exception as e:
            logger.error(f"Failed to replay events by time range: {str(e)}")
            return ReplayResult(
                replay_id="",
                status=ReplayStatus.FAILED,
                events_processed=0,
                events_failed=0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=str(e)
            )
    
    def create_state_snapshot(self,
                            aggregate_id: str,
                            snapshot_handler: Callable[[List[Event], Any], Any],
                            initial_state: Any = None) -> Any:
        """Create a state snapshot by replaying events"""
        try:
            events = self.event_store.get_events(aggregate_id)
            state = initial_state
            
            for event in events:
                state = snapshot_handler(events, state)
            
            logger.info(f"Created state snapshot for aggregate {aggregate_id}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to create state snapshot for aggregate {aggregate_id}: {str(e)}")
            return initial_state
    
    def cleanup_completed_replays(self, older_than_hours: int = 24) -> int:
        """Clean up completed replays older than specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            cleaned_count = 0
            
            with self.replay_lock:
                replays_to_remove = []
                for replay_id, result in self.active_replays.items():
                    if (result.status in [ReplayStatus.COMPLETED, ReplayStatus.FAILED] and
                        result.end_time and result.end_time < cutoff_time):
                        replays_to_remove.append(replay_id)
                
                for replay_id in replays_to_remove:
                    del self.active_replays[replay_id]
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} completed replays")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup completed replays: {str(e)}")
            return 0
