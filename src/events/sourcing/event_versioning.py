"""
Event Versioning
Event versioning and migration support for schema evolution
"""

import logging
import json
from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import hashlib
import threading

from .event_store import Event, EventType

logger = logging.getLogger(__name__)

class VersioningStrategy(Enum):
    """Versioning strategies"""
    FORWARD_COMPATIBLE = "forward_compatible"
    BACKWARD_COMPATIBLE = "backward_compatible"
    BREAKING_CHANGE = "breaking_change"
    MIGRATION_REQUIRED = "migration_required"

class MigrationStatus(Enum):
    """Migration status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class EventSchema:
    """Event schema definition"""
    event_name: str
    version: int
    schema: Dict[str, Any]
    migration_script: Optional[str] = None
    versioning_strategy: VersioningStrategy = VersioningStrategy.FORWARD_COMPATIBLE
    deprecated: bool = False
    created_at: datetime = None

@dataclass
class Migration:
    """Event migration definition"""
    migration_id: str
    from_version: int
    to_version: int
    event_name: str
    migration_function: Callable[[Dict[str, Any]], Dict[str, Any]]
    rollback_function: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    status: MigrationStatus = MigrationStatus.PENDING
    created_at: datetime = None

@dataclass
class VersionedEvent:
    """Versioned event with schema information"""
    event: Event
    schema_version: int
    is_migrated: bool = False
    migration_applied: Optional[str] = None

class EventVersioning:
    """
    Event versioning and migration support for schema evolution
    """
    
    def __init__(self):
        self.event_schemas = {}
        self.migrations = {}
        self.version_lock = threading.RLock()
        
        logger.info("EventVersioning initialized")
    
    def register_event_schema(self,
                            event_name: str,
                            version: int,
                            schema: Dict[str, Any],
                            versioning_strategy: VersioningStrategy = VersioningStrategy.FORWARD_COMPATIBLE,
                            migration_script: str = None) -> bool:
        """Register an event schema"""
        try:
            with self.version_lock:
                schema_key = f"{event_name}_v{version}"
                
                event_schema = EventSchema(
                    event_name=event_name,
                    version=version,
                    schema=schema,
                    migration_script=migration_script,
                    versioning_strategy=versioning_strategy,
                    created_at=datetime.now()
                )
                
                self.event_schemas[schema_key] = event_schema
                
                logger.info(f"Registered schema for {event_name} version {version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register event schema for {event_name} v{version}: {str(e)}")
            return False
    
    def register_migration(self,
                         event_name: str,
                         from_version: int,
                         to_version: int,
                         migration_function: Callable[[Dict[str, Any]], Dict[str, Any]],
                         rollback_function: Callable[[Dict[str, Any]], Dict[str, Any]] = None) -> str:
        """Register a migration between event versions"""
        try:
            migration_id = f"{event_name}_v{from_version}_to_v{to_version}"
            
            migration = Migration(
                migration_id=migration_id,
                from_version=from_version,
                to_version=to_version,
                event_name=event_name,
                migration_function=migration_function,
                rollback_function=rollback_function,
                created_at=datetime.now()
            )
            
            with self.version_lock:
                self.migrations[migration_id] = migration
            
            logger.info(f"Registered migration {migration_id}")
            return migration_id
            
        except Exception as e:
            logger.error(f"Failed to register migration for {event_name}: {str(e)}")
            return ""
    
    def get_latest_schema_version(self, event_name: str) -> int:
        """Get the latest schema version for an event"""
        try:
            with self.version_lock:
                max_version = 0
                for schema_key, schema in self.event_schemas.items():
                    if schema.event_name == event_name and schema.version > max_version:
                        max_version = schema.version
                
                return max_version
                
        except Exception as e:
            logger.error(f"Failed to get latest schema version for {event_name}: {str(e)}")
            return 0
    
    def get_schema(self, event_name: str, version: int) -> Optional[EventSchema]:
        """Get event schema for specific version"""
        try:
            schema_key = f"{event_name}_v{version}"
            return self.event_schemas.get(schema_key)
            
        except Exception as e:
            logger.error(f"Failed to get schema for {event_name} v{version}: {str(e)}")
            return None
    
    def validate_event(self, event: Event) -> bool:
        """Validate event against its schema"""
        try:
            schema = self.get_schema(event.event_name, event.version)
            if not schema:
                logger.warning(f"No schema found for {event.event_name} v{event.version}")
                return True  # Allow if no schema registered
            
            return self._validate_against_schema(event.data, schema.schema)
            
        except Exception as e:
            logger.error(f"Failed to validate event {event.event_id}: {str(e)}")
            return False
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate data against JSON schema"""
        try:
            # Simple validation - in real implementation, use jsonschema library
            required_fields = schema.get("required", [])
            
            # Check required fields
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Check field types
            properties = schema.get("properties", {})
            for field, field_schema in properties.items():
                if field in data:
                    expected_type = field_schema.get("type")
                    if expected_type and not self._check_field_type(data[field], expected_type):
                        logger.warning(f"Invalid type for field {field}: expected {expected_type}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate against schema: {str(e)}")
            return False
    
    def _check_field_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        try:
            type_mapping = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict
            }
            
            expected_python_type = type_mapping.get(expected_type)
            if expected_python_type:
                return isinstance(value, expected_python_type)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check field type: {str(e)}")
            return False
    
    def migrate_event(self, event: Event, target_version: int) -> Optional[Event]:
        """Migrate event to target version"""
        try:
            if event.version == target_version:
                return event
            
            # Find migration path
            migration_path = self._find_migration_path(event.event_name, event.version, target_version)
            if not migration_path:
                logger.warning(f"No migration path found for {event.event_name} from v{event.version} to v{target_version}")
                return None
            
            # Apply migrations
            current_data = event.data.copy()
            current_version = event.version
            
            for migration_id in migration_path:
                migration = self.migrations.get(migration_id)
                if not migration:
                    logger.error(f"Migration {migration_id} not found")
                    return None
                
                try:
                    current_data = migration.migration_function(current_data)
                    current_version = migration.to_version
                    logger.debug(f"Applied migration {migration_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to apply migration {migration_id}: {str(e)}")
                    return None
            
            # Create migrated event
            migrated_event = Event(
                event_id=event.event_id,
                aggregate_id=event.aggregate_id,
                event_type=event.event_type,
                event_name=event.event_name,
                version=target_version,
                data=current_data,
                metadata=event.metadata.copy(),
                timestamp=event.timestamp,
                status=event.status,
                correlation_id=event.correlation_id,
                causation_id=event.causation_id
            )
            
            # Add migration metadata
            migrated_event.metadata["migrated_from_version"] = event.version
            migrated_event.metadata["migration_path"] = migration_path
            migrated_event.metadata["migrated_at"] = datetime.now().isoformat()
            
            logger.info(f"Migrated event {event.event_id} from v{event.version} to v{target_version}")
            return migrated_event
            
        except Exception as e:
            logger.error(f"Failed to migrate event {event.event_id}: {str(e)}")
            return None
    
    def _find_migration_path(self, event_name: str, from_version: int, to_version: int) -> List[str]:
        """Find migration path between versions"""
        try:
            if from_version == to_version:
                return []
            
            # Simple path finding - in real implementation, use graph algorithms
            migration_path = []
            current_version = from_version
            
            while current_version != to_version:
                # Find direct migration
                migration_id = f"{event_name}_v{current_version}_to_v{to_version}"
                if migration_id in self.migrations:
                    migration_path.append(migration_id)
                    break
                
                # Find next version migration
                next_migration = None
                for migration_id, migration in self.migrations.items():
                    if (migration.event_name == event_name and 
                        migration.from_version == current_version and
                        migration.to_version > current_version):
                        if not next_migration or migration.to_version < next_migration.to_version:
                            next_migration = migration
                
                if not next_migration:
                    logger.warning(f"No migration found from {event_name} v{current_version}")
                    return []
                
                migration_path.append(next_migration.migration_id)
                current_version = next_migration.to_version
                
                # Prevent infinite loops
                if len(migration_path) > 10:
                    logger.error(f"Migration path too long for {event_name}")
                    return []
            
            return migration_path
            
        except Exception as e:
            logger.error(f"Failed to find migration path: {str(e)}")
            return []
    
    def get_versioned_event(self, event: Event) -> VersionedEvent:
        """Get versioned event with schema information"""
        try:
            schema = self.get_schema(event.event_name, event.version)
            latest_version = self.get_latest_schema_version(event.event_name)
            
            is_migrated = event.version < latest_version
            migration_applied = None
            
            if is_migrated and "migration_path" in event.metadata:
                migration_applied = event.metadata["migration_path"]
            
            return VersionedEvent(
                event=event,
                schema_version=event.version,
                is_migrated=is_migrated,
                migration_applied=migration_applied
            )
            
        except Exception as e:
            logger.error(f"Failed to get versioned event: {str(e)}")
            return VersionedEvent(
                event=event,
                schema_version=event.version,
                is_migrated=False
            )
    
    def get_schema_compatibility(self, event_name: str, from_version: int, to_version: int) -> VersioningStrategy:
        """Get compatibility between schema versions"""
        try:
            from_schema = self.get_schema(event_name, from_version)
            to_schema = self.get_schema(event_name, to_version)
            
            if not from_schema or not to_schema:
                return VersioningStrategy.BREAKING_CHANGE
            
            # Check if migration is required
            migration_id = f"{event_name}_v{from_version}_to_v{to_version}"
            if migration_id in self.migrations:
                return VersioningStrategy.MIGRATION_REQUIRED
            
            # Simple compatibility check - in real implementation, use schema comparison
            from_required = set(from_schema.schema.get("required", []))
            to_required = set(to_schema.schema.get("required", []))
            
            # If new required fields added, it's a breaking change
            if to_required - from_required:
                return VersioningStrategy.BREAKING_CHANGE
            
            # If fields removed, it's a breaking change
            if from_required - to_required:
                return VersioningStrategy.BREAKING_CHANGE
            
            # If no breaking changes, it's forward compatible
            return VersioningStrategy.FORWARD_COMPATIBLE
            
        except Exception as e:
            logger.error(f"Failed to get schema compatibility: {str(e)}")
            return VersioningStrategy.BREAKING_CHANGE
    
    def deprecate_schema(self, event_name: str, version: int) -> bool:
        """Deprecate an event schema version"""
        try:
            schema_key = f"{event_name}_v{version}"
            
            with self.version_lock:
                if schema_key in self.event_schemas:
                    self.event_schemas[schema_key].deprecated = True
                    logger.info(f"Deprecated schema {schema_key}")
                    return True
                else:
                    logger.warning(f"Schema {schema_key} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to deprecate schema {event_name} v{version}: {str(e)}")
            return False
    
    def get_schema_statistics(self) -> Dict[str, Any]:
        """Get schema versioning statistics"""
        try:
            with self.version_lock:
                stats = {
                    "total_schemas": len(self.event_schemas),
                    "total_migrations": len(self.migrations),
                    "event_types": {},
                    "deprecated_schemas": 0
                }
                
                # Count by event type
                for schema in self.event_schemas.values():
                    event_name = schema.event_name
                    if event_name not in stats["event_types"]:
                        stats["event_types"][event_name] = {
                            "versions": [],
                            "latest_version": 0,
                            "deprecated_versions": 0
                        }
                    
                    stats["event_types"][event_name]["versions"].append(schema.version)
                    stats["event_types"][event_name]["latest_version"] = max(
                        stats["event_types"][event_name]["latest_version"],
                        schema.version
                    )
                    
                    if schema.deprecated:
                        stats["event_types"][event_name]["deprecated_versions"] += 1
                        stats["deprecated_schemas"] += 1
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get schema statistics: {str(e)}")
            return {}
    
    def create_schema_hash(self, event_name: str, version: int) -> str:
        """Create hash for schema version"""
        try:
            schema = self.get_schema(event_name, version)
            if not schema:
                return ""
            
            # Create hash from schema content
            schema_content = json.dumps(schema.schema, sort_keys=True)
            return hashlib.sha256(schema_content.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"Failed to create schema hash: {str(e)}")
            return ""
    
    def validate_migration_integrity(self, migration_id: str) -> bool:
        """Validate migration integrity"""
        try:
            migration = self.migrations.get(migration_id)
            if not migration:
                return False
            
            # Test migration with sample data
            sample_data = {"test": "data"}
            
            try:
                migrated_data = migration.migration_function(sample_data)
                
                # Test rollback if available
                if migration.rollback_function:
                    rollback_data = migration.rollback_function(migrated_data)
                
                return True
                
            except Exception as e:
                logger.error(f"Migration {migration_id} integrity check failed: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to validate migration integrity: {str(e)}")
            return False
