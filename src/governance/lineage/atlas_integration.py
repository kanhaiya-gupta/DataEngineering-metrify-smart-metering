"""
Apache Atlas Integration
Provides integration with Apache Atlas for metadata management and data lineage
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class AtlasConfig:
    """Configuration for Apache Atlas integration"""
    base_url: str
    username: str
    password: str
    timeout: int = 30
    verify_ssl: bool = True

@dataclass
class EntityReference:
    """Reference to an Atlas entity"""
    guid: str
    type_name: str
    qualified_name: str
    display_name: str

class AtlasIntegration:
    """
    Apache Atlas integration for metadata management and data lineage
    """
    
    def __init__(self, config: AtlasConfig):
        self.config = config
        self.session = requests.Session()
        self.session.auth = (config.username, config.password)
        self.session.verify = config.verify_ssl
        self.session.timeout = config.timeout
        
        # Atlas API endpoints
        self.base_url = config.base_url.rstrip('/')
        self.api_v2 = f"{self.base_url}/api/atlas/v2"
        self.api_v1 = f"{self.base_url}/api/atlas/v1"
        
        logger.info(f"Atlas integration initialized with base URL: {self.base_url}")
    
    def authenticate(self) -> bool:
        """Authenticate with Apache Atlas"""
        try:
            response = self.session.get(f"{self.api_v2}/types/typedefs")
            response.raise_for_status()
            logger.info("Successfully authenticated with Apache Atlas")
            return True
        except Exception as e:
            logger.error(f"Atlas authentication failed: {str(e)}")
            return False
    
    def create_entity(self, entity_data: Dict[str, Any]) -> Optional[str]:
        """Create an entity in Atlas"""
        try:
            response = self.session.post(
                f"{self.api_v2}/entity",
                json=entity_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            if 'guidAssignments' in result and result['guidAssignments']:
                guid = list(result['guidAssignments'].values())[0]
                logger.info(f"Entity created successfully with GUID: {guid}")
                return guid
            else:
                logger.warning("Entity creation response did not contain GUID")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create entity: {str(e)}")
            return None
    
    def get_entity(self, guid: str) -> Optional[Dict[str, Any]]:
        """Get entity by GUID"""
        try:
            response = self.session.get(f"{self.api_v2}/entity/guid/{guid}")
            response.raise_for_status()
            
            entity = response.json()
            logger.debug(f"Retrieved entity: {guid}")
            return entity
            
        except Exception as e:
            logger.error(f"Failed to get entity {guid}: {str(e)}")
            return None
    
    def search_entities(self, 
                       query: str,
                       type_name: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Search for entities"""
        try:
            search_params = {
                "query": query,
                "limit": limit
            }
            
            if type_name:
                search_params["typeName"] = type_name
            
            response = self.session.get(
                f"{self.api_v2}/search/basic",
                params=search_params
            )
            response.raise_for_status()
            
            result = response.json()
            entities = result.get('entities', [])
            logger.info(f"Found {len(entities)} entities for query: {query}")
            return entities
            
        except Exception as e:
            logger.error(f"Entity search failed: {str(e)}")
            return []
    
    def create_lineage(self, 
                      source_guid: str, 
                      target_guid: str,
                      process_guid: str,
                      lineage_type: str = "data_flow") -> bool:
        """Create lineage relationship between entities"""
        try:
            lineage_data = {
                "typeName": "Process",
                "attributes": {
                    "qualifiedName": f"{lineage_type}_{source_guid}_{target_guid}",
                    "name": f"Lineage: {lineage_type}",
                    "description": f"Data lineage from {source_guid} to {target_guid}",
                    "inputs": [{"guid": source_guid}],
                    "outputs": [{"guid": target_guid}]
                }
            }
            
            response = self.session.post(
                f"{self.api_v2}/entity",
                json=lineage_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            logger.info(f"Lineage created between {source_guid} and {target_guid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create lineage: {str(e)}")
            return False
    
    def get_lineage(self, guid: str, direction: str = "both") -> Dict[str, Any]:
        """Get lineage information for an entity"""
        try:
            params = {"direction": direction}
            response = self.session.get(
                f"{self.api_v2}/lineage/{guid}",
                params=params
            )
            response.raise_for_status()
            
            lineage = response.json()
            logger.debug(f"Retrieved lineage for entity: {guid}")
            return lineage
            
        except Exception as e:
            logger.error(f"Failed to get lineage for {guid}: {str(e)}")
            return {}
    
    def create_dataset_entity(self, 
                            name: str,
                            qualified_name: str,
                            description: str,
                            schema: Dict[str, Any],
                            location: str) -> Optional[str]:
        """Create a dataset entity in Atlas"""
        try:
            entity_data = {
                "typeName": "DataSet",
                "attributes": {
                    "qualifiedName": qualified_name,
                    "name": name,
                    "description": description,
                    "schema": json.dumps(schema),
                    "location": location,
                    "createTime": datetime.now().isoformat(),
                    "updateTime": datetime.now().isoformat()
                }
            }
            
            return self.create_entity(entity_data)
            
        except Exception as e:
            logger.error(f"Failed to create dataset entity: {str(e)}")
            return None
    
    def create_process_entity(self,
                            name: str,
                            qualified_name: str,
                            description: str,
                            process_type: str,
                            inputs: List[str],
                            outputs: List[str]) -> Optional[str]:
        """Create a process entity in Atlas"""
        try:
            entity_data = {
                "typeName": "Process",
                "attributes": {
                    "qualifiedName": qualified_name,
                    "name": name,
                    "description": description,
                    "processType": process_type,
                    "inputs": [{"guid": guid} for guid in inputs],
                    "outputs": [{"guid": guid} for guid in outputs],
                    "createTime": datetime.now().isoformat(),
                    "updateTime": datetime.now().isoformat()
                }
            }
            
            return self.create_entity(entity_data)
            
        except Exception as e:
            logger.error(f"Failed to create process entity: {str(e)}")
            return None
    
    def track_smart_meter_data_lineage(self, 
                                     meter_id: str,
                                     reading_data: Dict[str, Any]) -> Optional[str]:
        """Track lineage for smart meter data"""
        try:
            # Create dataset entity for smart meter reading
            dataset_guid = self.create_dataset_entity(
                name=f"smart_meter_{meter_id}_readings",
                qualified_name=f"smart_meter.readings.{meter_id}",
                description=f"Smart meter readings for meter {meter_id}",
                schema={
                    "fields": [
                        {"name": "meter_id", "type": "string"},
                        {"name": "timestamp", "type": "timestamp"},
                        {"name": "consumption_kwh", "type": "double"},
                        {"name": "voltage", "type": "double"},
                        {"name": "quality_score", "type": "double"}
                    ]
                },
                location=f"s3://metrify-data/smart_meters/{meter_id}/"
            )
            
            if dataset_guid:
                logger.info(f"Smart meter data lineage tracked for meter {meter_id}")
                return dataset_guid
            else:
                logger.warning(f"Failed to track lineage for meter {meter_id}")
                return None
                
        except Exception as e:
            logger.error(f"Smart meter lineage tracking failed: {str(e)}")
            return None
    
    def track_kafka_processing_lineage(self,
                                     topic: str,
                                     consumer_group: str,
                                     output_location: str) -> Optional[str]:
        """Track lineage for Kafka data processing"""
        try:
            # Create process entity for Kafka processing
            process_guid = self.create_process_entity(
                name=f"kafka_processing_{topic}",
                qualified_name=f"kafka.processing.{topic}.{consumer_group}",
                description=f"Kafka data processing for topic {topic}",
                process_type="stream_processing",
                inputs=[],  # Will be linked to source datasets
                outputs=[]  # Will be linked to output datasets
            )
            
            if process_guid:
                logger.info(f"Kafka processing lineage tracked for topic {topic}")
                return process_guid
            else:
                logger.warning(f"Failed to track lineage for Kafka topic {topic}")
                return None
                
        except Exception as e:
            logger.error(f"Kafka lineage tracking failed: {str(e)}")
            return None
    
    def get_data_catalog(self, 
                        entity_type: Optional[str] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Get data catalog entries"""
        try:
            if entity_type:
                entities = self.search_entities("*", type_name=entity_type, limit=limit)
            else:
                entities = self.search_entities("*", limit=limit)
            
            catalog = []
            for entity in entities:
                catalog_entry = {
                    "guid": entity.get("guid"),
                    "type_name": entity.get("typeName"),
                    "qualified_name": entity.get("attributes", {}).get("qualifiedName"),
                    "display_name": entity.get("attributes", {}).get("name"),
                    "description": entity.get("attributes", {}).get("description"),
                    "create_time": entity.get("attributes", {}).get("createTime"),
                    "update_time": entity.get("attributes", {}).get("updateTime")
                }
                catalog.append(catalog_entry)
            
            logger.info(f"Retrieved {len(catalog)} catalog entries")
            return catalog
            
        except Exception as e:
            logger.error(f"Failed to get data catalog: {str(e)}")
            return []
    
    def export_lineage_graph(self, 
                           root_guid: str,
                           max_depth: int = 3) -> Dict[str, Any]:
        """Export lineage as a graph structure"""
        try:
            lineage = self.get_lineage(root_guid, direction="both")
            
            graph = {
                "nodes": [],
                "edges": [],
                "root_guid": root_guid,
                "exported_at": datetime.now().isoformat()
            }
            
            # Process nodes
            if "guidEntityMap" in lineage:
                for guid, entity in lineage["guidEntityMap"].items():
                    node = {
                        "id": guid,
                        "type": entity.get("typeName"),
                        "name": entity.get("attributes", {}).get("name"),
                        "qualified_name": entity.get("attributes", {}).get("qualifiedName")
                    }
                    graph["nodes"].append(node)
            
            # Process edges
            if "relations" in lineage:
                for relation in lineage["relations"]:
                    edge = {
                        "source": relation.get("fromEntityId"),
                        "target": relation.get("toEntityId"),
                        "type": relation.get("relationshipType")
                    }
                    graph["edges"].append(edge)
            
            logger.info(f"Exported lineage graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
            return graph
            
        except Exception as e:
            logger.error(f"Failed to export lineage graph: {str(e)}")
            return {"nodes": [], "edges": [], "error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get Atlas integration health status"""
        try:
            # Test connection
            response = self.session.get(f"{self.api_v2}/types/typedefs", timeout=5)
            is_healthy = response.status_code == 200
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "base_url": self.base_url,
                "last_check": datetime.now().isoformat(),
                "response_time_ms": response.elapsed.total_seconds() * 1000 if is_healthy else None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "base_url": self.base_url,
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }
