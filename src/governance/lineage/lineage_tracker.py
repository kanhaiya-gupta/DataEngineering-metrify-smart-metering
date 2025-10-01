"""
Lineage Tracker
Tracks data lineage across the entire data pipeline
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import pandas as pd

logger = logging.getLogger(__name__)

class LineageEventType(Enum):
    """Types of lineage events"""
    DATA_INGESTION = "data_ingestion"
    DATA_TRANSFORMATION = "data_transformation"
    DATA_STORAGE = "data_storage"
    DATA_EXPORT = "data_export"
    DATA_DELETION = "data_deletion"

@dataclass
class LineageEvent:
    """Represents a lineage event"""
    event_id: str
    event_type: LineageEventType
    timestamp: datetime
    source_entity: str
    target_entity: str
    process_name: str
    metadata: Dict[str, Any]
    user_id: Optional[str] = None

@dataclass
class DataEntity:
    """Represents a data entity in the lineage"""
    entity_id: str
    entity_type: str
    name: str
    location: str
    schema: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    owner: Optional[str] = None
    tags: List[str] = None

class LineageTracker:
    """
    Tracks data lineage across the entire data pipeline
    """
    
    def __init__(self, atlas_integration=None):
        self.atlas_integration = atlas_integration
        self.lineage_events = []
        self.data_entities = {}
        self.lineage_graph = {}
        
        logger.info("LineageTracker initialized")
    
    def register_data_entity(self, 
                           entity_id: str,
                           entity_type: str,
                           name: str,
                           location: str,
                           schema: Dict[str, Any],
                           owner: Optional[str] = None,
                           tags: List[str] = None) -> bool:
        """Register a new data entity"""
        try:
            entity = DataEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                name=name,
                location=location,
                schema=schema,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                owner=owner,
                tags=tags or []
            )
            
            self.data_entities[entity_id] = entity
            
            # Register with Atlas if available
            if self.atlas_integration:
                atlas_guid = self.atlas_integration.create_dataset_entity(
                    name=name,
                    qualified_name=entity_id,
                    description=f"Data entity: {name}",
                    schema=schema,
                    location=location
                )
                if atlas_guid:
                    logger.info(f"Entity {entity_id} registered with Atlas GUID: {atlas_guid}")
            
            logger.info(f"Data entity registered: {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register data entity {entity_id}: {str(e)}")
            return False
    
    def track_lineage_event(self,
                          event_type: LineageEventType,
                          source_entity: str,
                          target_entity: str,
                          process_name: str,
                          metadata: Dict[str, Any],
                          user_id: Optional[str] = None) -> str:
        """Track a lineage event"""
        try:
            event_id = f"{event_type.value}_{source_entity}_{target_entity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            event = LineageEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                source_entity=source_entity,
                target_entity=target_entity,
                process_name=process_name,
                metadata=metadata,
                user_id=user_id
            )
            
            self.lineage_events.append(event)
            
            # Update lineage graph
            if source_entity not in self.lineage_graph:
                self.lineage_graph[source_entity] = []
            
            self.lineage_graph[source_entity].append({
                "target": target_entity,
                "process": process_name,
                "event_type": event_type.value,
                "timestamp": event.timestamp.isoformat()
            })
            
            # Track with Atlas if available
            if self.atlas_integration:
                # This would require getting GUIDs from Atlas
                # For now, we'll just log the event
                logger.debug(f"Lineage event tracked with Atlas: {event_id}")
            
            logger.info(f"Lineage event tracked: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to track lineage event: {str(e)}")
            return ""
    
    def track_smart_meter_ingestion(self,
                                  meter_id: str,
                                  kafka_topic: str,
                                  s3_location: str,
                                  user_id: Optional[str] = None) -> str:
        """Track smart meter data ingestion lineage"""
        try:
            # Register source (Kafka topic)
            source_entity = f"kafka_topic_{kafka_topic}"
            if source_entity not in self.data_entities:
                self.register_data_entity(
                    entity_id=source_entity,
                    entity_type="kafka_topic",
                    name=f"Smart Meter Data - {kafka_topic}",
                    location=f"kafka://{kafka_topic}",
                    schema={"type": "kafka_topic", "topic": kafka_topic},
                    tags=["smart_meter", "ingestion", "kafka"]
                )
            
            # Register target (S3 location)
            target_entity = f"s3_location_{s3_location.replace('/', '_')}"
            if target_entity not in self.data_entities:
                self.register_data_entity(
                    entity_id=target_entity,
                    entity_type="s3_dataset",
                    name=f"Smart Meter Data - {meter_id}",
                    location=s3_location,
                    schema={
                        "fields": [
                            {"name": "meter_id", "type": "string"},
                            {"name": "timestamp", "type": "timestamp"},
                            {"name": "consumption_kwh", "type": "double"},
                            {"name": "voltage", "type": "double"}
                        ]
                    },
                    tags=["smart_meter", "storage", "s3"]
                )
            
            # Track lineage event
            event_id = self.track_lineage_event(
                event_type=LineageEventType.DATA_INGESTION,
                source_entity=source_entity,
                target_entity=target_entity,
                process_name="smart_meter_ingestion",
                metadata={
                    "meter_id": meter_id,
                    "kafka_topic": kafka_topic,
                    "s3_location": s3_location,
                    "record_count": 0  # Would be populated from actual processing
                },
                user_id=user_id
            )
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to track smart meter ingestion lineage: {str(e)}")
            return ""
    
    def track_data_transformation(self,
                                source_entity: str,
                                target_entity: str,
                                transformation_name: str,
                                transformation_config: Dict[str, Any],
                                user_id: Optional[str] = None) -> str:
        """Track data transformation lineage"""
        try:
            event_id = self.track_lineage_event(
                event_type=LineageEventType.DATA_TRANSFORMATION,
                source_entity=source_entity,
                target_entity=target_entity,
                process_name=transformation_name,
                metadata={
                    "transformation_config": transformation_config,
                    "transformation_type": "dbt_model",  # or "spark_job", "python_script", etc.
                    "execution_time": 0  # Would be populated from actual execution
                },
                user_id=user_id
            )
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to track data transformation lineage: {str(e)}")
            return ""
    
    def track_ml_model_lineage(self,
                             training_data_entity: str,
                             model_entity: str,
                             model_name: str,
                             model_version: str,
                             training_config: Dict[str, Any],
                             user_id: Optional[str] = None) -> str:
        """Track ML model lineage"""
        try:
            # Register model entity if not exists
            if model_entity not in self.data_entities:
                self.register_data_entity(
                    entity_id=model_entity,
                    entity_type="ml_model",
                    name=f"ML Model - {model_name}",
                    location=f"mlflow://models/{model_name}/{model_version}",
                    schema={
                        "model_name": model_name,
                        "model_version": model_version,
                        "model_type": "tensorflow",
                        "framework": "tensorflow"
                    },
                    tags=["ml_model", "tensorflow", "mlflow"]
                )
            
            event_id = self.track_lineage_event(
                event_type=LineageEventType.DATA_TRANSFORMATION,
                source_entity=training_data_entity,
                target_entity=model_entity,
                process_name="ml_model_training",
                metadata={
                    "model_name": model_name,
                    "model_version": model_version,
                    "training_config": training_config,
                    "model_type": "tensorflow"
                },
                user_id=user_id
            )
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to track ML model lineage: {str(e)}")
            return ""
    
    def get_entity_lineage(self, entity_id: str, max_depth: int = 3) -> Dict[str, Any]:
        """Get lineage for a specific entity"""
        try:
            lineage = {
                "entity_id": entity_id,
                "upstream": [],
                "downstream": [],
                "max_depth": max_depth,
                "generated_at": datetime.now().isoformat()
            }
            
            # Find upstream entities
            upstream = self._find_upstream_entities(entity_id, max_depth)
            lineage["upstream"] = upstream
            
            # Find downstream entities
            downstream = self._find_downstream_entities(entity_id, max_depth)
            lineage["downstream"] = downstream
            
            logger.info(f"Retrieved lineage for entity {entity_id}: {len(upstream)} upstream, {len(downstream)} downstream")
            return lineage
            
        except Exception as e:
            logger.error(f"Failed to get entity lineage: {str(e)}")
            return {"entity_id": entity_id, "error": str(e)}
    
    def _find_upstream_entities(self, entity_id: str, max_depth: int, current_depth: int = 0) -> List[Dict[str, Any]]:
        """Find upstream entities recursively"""
        if current_depth >= max_depth:
            return []
        
        upstream = []
        for source, targets in self.lineage_graph.items():
            for target_info in targets:
                if target_info["target"] == entity_id:
                    upstream_entity = {
                        "entity_id": source,
                        "process": target_info["process"],
                        "event_type": target_info["event_type"],
                        "timestamp": target_info["timestamp"],
                        "depth": current_depth + 1
                    }
                    upstream.append(upstream_entity)
                    
                    # Recursively find upstream of this entity
                    upstream.extend(self._find_upstream_entities(source, max_depth, current_depth + 1))
        
        return upstream
    
    def _find_downstream_entities(self, entity_id: str, max_depth: int, current_depth: int = 0) -> List[Dict[str, Any]]:
        """Find downstream entities recursively"""
        if current_depth >= max_depth:
            return []
        
        downstream = []
        if entity_id in self.lineage_graph:
            for target_info in self.lineage_graph[entity_id]:
                downstream_entity = {
                    "entity_id": target_info["target"],
                    "process": target_info["process"],
                    "event_type": target_info["event_type"],
                    "timestamp": target_info["timestamp"],
                    "depth": current_depth + 1
                }
                downstream.append(downstream_entity)
                
                # Recursively find downstream of this entity
                downstream.extend(self._find_downstream_entities(target_info["target"], max_depth, current_depth + 1))
        
        return downstream
    
    def get_lineage_statistics(self) -> Dict[str, Any]:
        """Get lineage tracking statistics"""
        try:
            total_entities = len(self.data_entities)
            total_events = len(self.lineage_events)
            
            # Count events by type
            event_counts = {}
            for event in self.lineage_events:
                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Count entities by type
            entity_counts = {}
            for entity in self.data_entities.values():
                entity_type = entity.entity_type
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            return {
                "total_entities": total_entities,
                "total_events": total_events,
                "event_counts_by_type": event_counts,
                "entity_counts_by_type": entity_counts,
                "lineage_graph_size": len(self.lineage_graph),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get lineage statistics: {str(e)}")
            return {"error": str(e)}
    
    def export_lineage_data(self, format: str = "json") -> str:
        """Export lineage data in specified format"""
        try:
            export_data = {
                "entities": [
                    {
                        "entity_id": entity.entity_id,
                        "entity_type": entity.entity_type,
                        "name": entity.name,
                        "location": entity.location,
                        "schema": entity.schema,
                        "created_at": entity.created_at.isoformat(),
                        "updated_at": entity.updated_at.isoformat(),
                        "owner": entity.owner,
                        "tags": entity.tags
                    }
                    for entity in self.data_entities.values()
                ],
                "events": [
                    {
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "timestamp": event.timestamp.isoformat(),
                        "source_entity": event.source_entity,
                        "target_entity": event.target_entity,
                        "process_name": event.process_name,
                        "metadata": event.metadata,
                        "user_id": event.user_id
                    }
                    for event in self.lineage_events
                ],
                "lineage_graph": self.lineage_graph,
                "exported_at": datetime.now().isoformat()
            }
            
            if format.lower() == "json":
                return json.dumps(export_data, indent=2)
            elif format.lower() == "csv":
                # Convert to CSV format (simplified)
                df_entities = pd.DataFrame(export_data["entities"])
                df_events = pd.DataFrame(export_data["events"])
                return f"Entities:\n{df_entities.to_csv()}\n\nEvents:\n{df_events.to_csv()}"
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export lineage data: {str(e)}")
            return f"Export failed: {str(e)}"
    
    def get_impact_analysis(self, entity_id: str) -> Dict[str, Any]:
        """Analyze impact of changes to an entity"""
        try:
            # Get all downstream entities
            downstream = self._find_downstream_entities(entity_id, max_depth=10)
            
            # Analyze impact
            impact_analysis = {
                "entity_id": entity_id,
                "total_impacted_entities": len(downstream),
                "impacted_entities": downstream,
                "risk_level": "low",
                "recommendations": []
            }
            
            # Determine risk level based on number of downstream entities
            if len(downstream) > 10:
                impact_analysis["risk_level"] = "high"
                impact_analysis["recommendations"].append("High impact change - notify all downstream consumers")
            elif len(downstream) > 5:
                impact_analysis["risk_level"] = "medium"
                impact_analysis["recommendations"].append("Medium impact change - notify key downstream consumers")
            else:
                impact_analysis["risk_level"] = "low"
                impact_analysis["recommendations"].append("Low impact change - proceed with caution")
            
            # Add general recommendations
            impact_analysis["recommendations"].extend([
                "Test changes in development environment first",
                "Monitor downstream systems after deployment",
                "Have rollback plan ready"
            ])
            
            logger.info(f"Impact analysis completed for entity {entity_id}: {impact_analysis['risk_level']} risk")
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Failed to get impact analysis: {str(e)}")
            return {"entity_id": entity_id, "error": str(e)}
