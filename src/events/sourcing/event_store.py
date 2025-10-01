"""
Event Store
Persistent storage for events with versioning and replay capabilities
"""

import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event types"""
    DOMAIN_EVENT = "domain_event"
    INTEGRATION_EVENT = "integration_event"
    SYSTEM_EVENT = "system_event"
    AUDIT_EVENT = "audit_event"

class EventStatus(Enum):
    """Event status"""
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"

@dataclass
class Event:
    """Represents an event in the event store"""
    event_id: str
    aggregate_id: str
    event_type: EventType
    event_name: str
    version: int
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    status: EventStatus = EventStatus.PENDING
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None

@dataclass
class EventStream:
    """Represents a stream of events for an aggregate"""
    aggregate_id: str
    events: List[Event]
    version: int
    created_at: datetime
    updated_at: datetime

class EventStore:
    """
    Persistent event store with versioning and replay capabilities
    """
    
    def __init__(self, db_path: str = "events.db"):
        self.db_path = db_path
        self.lock = threading.RLock()
        self._initialize_database()
        
        logger.info(f"EventStore initialized with database: {db_path}")
    
    def _initialize_database(self):
        """Initialize the event store database"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        event_id TEXT PRIMARY KEY,
                        aggregate_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        event_name TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        data TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        status TEXT NOT NULL,
                        correlation_id TEXT,
                        causation_id TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_aggregate_id 
                    ON events(aggregate_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON events(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_event_type 
                    ON events(event_type)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_correlation_id 
                    ON events(correlation_id)
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize event store database: {str(e)}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def append_events(self,
                     aggregate_id: str,
                     events: List[Event],
                     expected_version: int = None) -> bool:
        """Append events to an aggregate's stream"""
        try:
            with self.lock:
                with self._get_connection() as conn:
                    # Check expected version
                    if expected_version is not None:
                        current_version = self._get_aggregate_version(conn, aggregate_id)
                        if current_version != expected_version:
                            raise ValueError(
                                f"Version mismatch: expected {expected_version}, got {current_version}"
                            )
                    
                    # Insert events
                    for i, event in enumerate(events):
                        event.version = expected_version + i + 1 if expected_version is not None else i + 1
                        event.timestamp = datetime.now()
                        
                        conn.execute("""
                            INSERT INTO events (
                                event_id, aggregate_id, event_type, event_name,
                                version, data, metadata, timestamp, status,
                                correlation_id, causation_id, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            event.event_id,
                            event.aggregate_id,
                            event.event_type.value,
                            event.event_name,
                            event.version,
                            json.dumps(event.data),
                            json.dumps(event.metadata),
                            event.timestamp.isoformat(),
                            event.status.value,
                            event.correlation_id,
                            event.causation_id,
                            datetime.now().isoformat()
                        ))
                    
                    conn.commit()
                    logger.info(f"Appended {len(events)} events to aggregate {aggregate_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to append events to aggregate {aggregate_id}: {str(e)}")
            return False
    
    def get_events(self,
                  aggregate_id: str,
                  from_version: int = 0,
                  to_version: int = None) -> List[Event]:
        """Get events for an aggregate"""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT * FROM events 
                    WHERE aggregate_id = ? AND version > ?
                """
                params = [aggregate_id, from_version]
                
                if to_version is not None:
                    query += " AND version <= ?"
                    params.append(to_version)
                
                query += " ORDER BY version ASC"
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    event = Event(
                        event_id=row['event_id'],
                        aggregate_id=row['aggregate_id'],
                        event_type=EventType(row['event_type']),
                        event_name=row['event_name'],
                        version=row['version'],
                        data=json.loads(row['data']),
                        metadata=json.loads(row['metadata']),
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        status=EventStatus(row['status']),
                        correlation_id=row['correlation_id'],
                        causation_id=row['causation_id']
                    )
                    events.append(event)
                
                logger.debug(f"Retrieved {len(events)} events for aggregate {aggregate_id}")
                return events
                
        except Exception as e:
            logger.error(f"Failed to get events for aggregate {aggregate_id}: {str(e)}")
            return []
    
    def get_events_by_type(self,
                          event_type: EventType,
                          from_timestamp: datetime = None,
                          to_timestamp: datetime = None,
                          limit: int = 1000) -> List[Event]:
        """Get events by type within a time range"""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT * FROM events 
                    WHERE event_type = ?
                """
                params = [event_type.value]
                
                if from_timestamp:
                    query += " AND timestamp >= ?"
                    params.append(from_timestamp.isoformat())
                
                if to_timestamp:
                    query += " AND timestamp <= ?"
                    params.append(to_timestamp.isoformat())
                
                query += " ORDER BY timestamp ASC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    event = Event(
                        event_id=row['event_id'],
                        aggregate_id=row['aggregate_id'],
                        event_type=EventType(row['event_type']),
                        event_name=row['event_name'],
                        version=row['version'],
                        data=json.loads(row['data']),
                        metadata=json.loads(row['metadata']),
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        status=EventStatus(row['status']),
                        correlation_id=row['correlation_id'],
                        causation_id=row['causation_id']
                    )
                    events.append(event)
                
                logger.debug(f"Retrieved {len(events)} events of type {event_type.value}")
                return events
                
        except Exception as e:
            logger.error(f"Failed to get events by type {event_type.value}: {str(e)}")
            return []
    
    def get_events_by_correlation_id(self, correlation_id: str) -> List[Event]:
        """Get events by correlation ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM events 
                    WHERE correlation_id = ?
                    ORDER BY timestamp ASC
                """, (correlation_id,))
                
                rows = cursor.fetchall()
                events = []
                for row in rows:
                    event = Event(
                        event_id=row['event_id'],
                        aggregate_id=row['aggregate_id'],
                        event_type=EventType(row['event_type']),
                        event_name=row['event_name'],
                        version=row['version'],
                        data=json.loads(row['data']),
                        metadata=json.loads(row['metadata']),
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        status=EventStatus(row['status']),
                        correlation_id=row['correlation_id'],
                        causation_id=row['causation_id']
                    )
                    events.append(event)
                
                logger.debug(f"Retrieved {len(events)} events for correlation {correlation_id}")
                return events
                
        except Exception as e:
            logger.error(f"Failed to get events by correlation ID {correlation_id}: {str(e)}")
            return []
    
    def _get_aggregate_version(self, conn, aggregate_id: str) -> int:
        """Get current version of an aggregate"""
        try:
            cursor = conn.execute("""
                SELECT MAX(version) as max_version 
                FROM events 
                WHERE aggregate_id = ?
            """, (aggregate_id,))
            
            row = cursor.fetchone()
            return row['max_version'] if row['max_version'] is not None else 0
            
        except Exception as e:
            logger.error(f"Failed to get aggregate version for {aggregate_id}: {str(e)}")
            return 0
    
    def get_aggregate_version(self, aggregate_id: str) -> int:
        """Get current version of an aggregate"""
        try:
            with self._get_connection() as conn:
                return self._get_aggregate_version(conn, aggregate_id)
                
        except Exception as e:
            logger.error(f"Failed to get aggregate version for {aggregate_id}: {str(e)}")
            return 0
    
    def update_event_status(self, event_id: str, status: EventStatus) -> bool:
        """Update event status"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE events 
                    SET status = ? 
                    WHERE event_id = ?
                """, (status.value, event_id))
                
                conn.commit()
                logger.debug(f"Updated event {event_id} status to {status.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update event {event_id} status: {str(e)}")
            return False
    
    def delete_events(self,
                     aggregate_id: str,
                     before_version: int = None,
                     before_timestamp: datetime = None) -> int:
        """Delete events (for cleanup/archiving)"""
        try:
            with self._get_connection() as conn:
                query = "DELETE FROM events WHERE aggregate_id = ?"
                params = [aggregate_id]
                
                if before_version is not None:
                    query += " AND version < ?"
                    params.append(before_version)
                
                if before_timestamp is not None:
                    query += " AND timestamp < ?"
                    params.append(before_timestamp.isoformat())
                
                cursor = conn.execute(query, params)
                deleted_count = cursor.rowcount
                
                conn.commit()
                logger.info(f"Deleted {deleted_count} events for aggregate {aggregate_id}")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to delete events for aggregate {aggregate_id}: {str(e)}")
            return 0
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event store statistics"""
        try:
            with self._get_connection() as conn:
                stats = {}
                
                # Total events
                cursor = conn.execute("SELECT COUNT(*) as total FROM events")
                stats['total_events'] = cursor.fetchone()['total']
                
                # Events by type
                cursor = conn.execute("""
                    SELECT event_type, COUNT(*) as count 
                    FROM events 
                    GROUP BY event_type
                """)
                stats['events_by_type'] = {row['event_type']: row['count'] for row in cursor.fetchall()}
                
                # Events by status
                cursor = conn.execute("""
                    SELECT status, COUNT(*) as count 
                    FROM events 
                    GROUP BY status
                """)
                stats['events_by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}
                
                # Unique aggregates
                cursor = conn.execute("SELECT COUNT(DISTINCT aggregate_id) as count FROM events")
                stats['unique_aggregates'] = cursor.fetchone()['count']
                
                # Date range
                cursor = conn.execute("""
                    SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date 
                    FROM events
                """)
                row = cursor.fetchone()
                stats['date_range'] = {
                    'min_date': row['min_date'],
                    'max_date': row['max_date']
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get event statistics: {str(e)}")
            return {}
    
    def create_event(self,
                    aggregate_id: str,
                    event_type: EventType,
                    event_name: str,
                    data: Dict[str, Any],
                    metadata: Dict[str, Any] = None,
                    correlation_id: str = None,
                    causation_id: str = None) -> Event:
        """Create a new event"""
        try:
            event = Event(
                event_id=str(uuid.uuid4()),
                aggregate_id=aggregate_id,
                event_type=event_type,
                event_name=event_name,
                version=0,  # Will be set when appended
                data=data,
                metadata=metadata or {},
                timestamp=datetime.now(),
                correlation_id=correlation_id,
                causation_id=causation_id
            )
            
            logger.debug(f"Created event {event.event_id} for aggregate {aggregate_id}")
            return event
            
        except Exception as e:
            logger.error(f"Failed to create event: {str(e)}")
            raise
    
    def get_event_stream(self, aggregate_id: str) -> EventStream:
        """Get complete event stream for an aggregate"""
        try:
            events = self.get_events(aggregate_id)
            version = self.get_aggregate_version(aggregate_id)
            
            created_at = events[0].timestamp if events else datetime.now()
            updated_at = events[-1].timestamp if events else datetime.now()
            
            return EventStream(
                aggregate_id=aggregate_id,
                events=events,
                version=version,
                created_at=created_at,
                updated_at=updated_at
            )
            
        except Exception as e:
            logger.error(f"Failed to get event stream for aggregate {aggregate_id}: {str(e)}")
            return EventStream(
                aggregate_id=aggregate_id,
                events=[],
                version=0,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
    
    def search_events(self,
                     query: str,
                     event_types: List[EventType] = None,
                     from_timestamp: datetime = None,
                     to_timestamp: datetime = None,
                     limit: int = 100) -> List[Event]:
        """Search events by content"""
        try:
            with self._get_connection() as conn:
                # Build search query
                where_conditions = []
                params = []
                
                # Text search in data and metadata
                where_conditions.append("(data LIKE ? OR metadata LIKE ?)")
                search_term = f"%{query}%"
                params.extend([search_term, search_term])
                
                # Event type filter
                if event_types:
                    placeholders = ','.join(['?' for _ in event_types])
                    where_conditions.append(f"event_type IN ({placeholders})")
                    params.extend([et.value for et in event_types])
                
                # Time range filter
                if from_timestamp:
                    where_conditions.append("timestamp >= ?")
                    params.append(from_timestamp.isoformat())
                
                if to_timestamp:
                    where_conditions.append("timestamp <= ?")
                    params.append(to_timestamp.isoformat())
                
                # Build final query
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                query_sql = f"""
                    SELECT * FROM events 
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor = conn.execute(query_sql, params)
                rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    event = Event(
                        event_id=row['event_id'],
                        aggregate_id=row['aggregate_id'],
                        event_type=EventType(row['event_type']),
                        event_name=row['event_name'],
                        version=row['version'],
                        data=json.loads(row['data']),
                        metadata=json.loads(row['metadata']),
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        status=EventStatus(row['status']),
                        correlation_id=row['correlation_id'],
                        causation_id=row['causation_id']
                    )
                    events.append(event)
                
                logger.debug(f"Found {len(events)} events matching search query")
                return events
                
        except Exception as e:
            logger.error(f"Failed to search events: {str(e)}")
            return []
