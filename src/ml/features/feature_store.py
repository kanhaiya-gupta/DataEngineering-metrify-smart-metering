"""
Feature Store

This module implements a feature store for managing and serving ML features
with support for feature versioning, validation, and real-time serving.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import json
import pickle
from dataclasses import dataclass
import hashlib

import redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import joblib

logger = logging.getLogger(__name__)


@dataclass
class FeatureStoreConfig:
    """Configuration for feature store"""
    # Database configuration
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "feature_store"
    db_user: str = "feature_user"
    db_password: str = "feature_password"
    
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Feature store settings
    cache_ttl: int = 3600  # seconds
    max_features_per_entity: int = 1000
    enable_versioning: bool = True
    enable_validation: bool = True
    enable_caching: bool = True


class FeatureStore:
    """
    Feature store for managing and serving ML features
    
    Features:
    - Feature registration and versioning
    - Real-time feature serving
    - Feature validation and monitoring
    - Caching and performance optimization
    - Feature lineage tracking
    """
    
    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.db_engine = None
        self.redis_client = None
        self.feature_registry = {}
        
        self._setup_database()
        self._setup_redis()
        self._create_tables()
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            db_url = f"postgresql://{self.config.db_user}:{self.config.db_password}@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
            self.db_engine = create_engine(db_url)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to setup database: {str(e)}")
            raise
    
    def _setup_redis(self):
        """Setup Redis connection for caching"""
        if not self.config.enable_caching:
            return
        
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to setup Redis: {str(e)}")
            self.redis_client = None
    
    def _create_tables(self):
        """Create feature store tables"""
        try:
            with self.db_engine.connect() as conn:
                # Features table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS features (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        version VARCHAR(50) NOT NULL,
                        entity_type VARCHAR(100) NOT NULL,
                        feature_type VARCHAR(50) NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE,
                        UNIQUE(name, version)
                    )
                """))
                
                # Feature values table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS feature_values (
                        id SERIAL PRIMARY KEY,
                        feature_id INTEGER REFERENCES features(id),
                        entity_id VARCHAR(255) NOT NULL,
                        value JSONB NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX(entity_id, timestamp),
                        INDEX(feature_id, entity_id)
                    )
                """))
                
                # Feature lineage table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS feature_lineage (
                        id SERIAL PRIMARY KEY,
                        feature_id INTEGER REFERENCES features(id),
                        source_table VARCHAR(255) NOT NULL,
                        source_column VARCHAR(255) NOT NULL,
                        transformation TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.commit()
                logger.info("Feature store tables created successfully")
                
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise
    
    def register_feature(self, 
                        name: str,
                        entity_type: str,
                        feature_type: str,
                        description: str = "",
                        version: Optional[str] = None) -> int:
        """Register a new feature"""
        try:
            if version is None:
                version = "1.0.0"
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO features (name, version, entity_type, feature_type, description)
                    VALUES (:name, :version, :entity_type, :feature_type, :description)
                    RETURNING id
                """), {
                    "name": name,
                    "version": version,
                    "entity_type": entity_type,
                    "feature_type": feature_type,
                    "description": description
                })
                
                feature_id = result.fetchone()[0]
                conn.commit()
                
                logger.info(f"Registered feature: {name} v{version} with ID: {feature_id}")
                return feature_id
                
        except Exception as e:
            logger.error(f"Failed to register feature: {str(e)}")
            raise
    
    def get_feature(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get feature information"""
        try:
            with self.db_engine.connect() as conn:
                if version:
                    query = "SELECT * FROM features WHERE name = :name AND version = :version"
                    params = {"name": name, "version": version}
                else:
                    query = "SELECT * FROM features WHERE name = :name AND is_active = TRUE ORDER BY created_at DESC LIMIT 1"
                    params = {"name": name}
                
                result = conn.execute(text(query), params)
                row = result.fetchone()
                
                if row:
                    return {
                        "id": row[0],
                        "name": row[1],
                        "version": row[2],
                        "entity_type": row[3],
                        "feature_type": row[4],
                        "description": row[5],
                        "created_at": row[6],
                        "updated_at": row[7],
                        "is_active": row[8]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get feature: {str(e)}")
            return None
    
    def store_feature_values(self, 
                           feature_id: int,
                           entity_id: str,
                           value: Any,
                           timestamp: Optional[datetime] = None) -> bool:
        """Store feature values for an entity"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Serialize value
            if isinstance(value, (dict, list)):
                value_json = json.dumps(value)
            else:
                value_json = json.dumps({"value": value})
            
            with self.db_engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO feature_values (feature_id, entity_id, value, timestamp)
                    VALUES (:feature_id, :entity_id, :value, :timestamp)
                """), {
                    "feature_id": feature_id,
                    "entity_id": entity_id,
                    "value": value_json,
                    "timestamp": timestamp
                })
                
                conn.commit()
                
                # Cache the value if Redis is available
                if self.redis_client:
                    cache_key = f"feature:{feature_id}:{entity_id}"
                    self.redis_client.setex(
                        cache_key, 
                        self.config.cache_ttl, 
                        value_json
                    )
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store feature values: {str(e)}")
            return False
    
    def get_feature_values(self, 
                          feature_name: str,
                          entity_id: str,
                          version: Optional[str] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get feature values for an entity"""
        try:
            # Get feature ID
            feature = self.get_feature(feature_name, version)
            if not feature:
                return []
            
            feature_id = feature["id"]
            
            # Check cache first
            if self.redis_client:
                cache_key = f"feature:{feature_id}:{entity_id}"
                cached_value = self.redis_client.get(cache_key)
                if cached_value:
                    return [{"value": json.loads(cached_value), "timestamp": datetime.now()}]
            
            # Query database
            with self.db_engine.connect() as conn:
                query = """
                    SELECT value, timestamp FROM feature_values 
                    WHERE feature_id = :feature_id AND entity_id = :entity_id
                """
                params = {"feature_id": feature_id, "entity_id": entity_id}
                
                if start_time:
                    query += " AND timestamp >= :start_time"
                    params["start_time"] = start_time
                
                if end_time:
                    query += " AND timestamp <= :end_time"
                    params["end_time"] = end_time
                
                query += " ORDER BY timestamp DESC"
                
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                
                values = []
                for row in rows:
                    try:
                        value_data = json.loads(row[0])
                        values.append({
                            "value": value_data,
                            "timestamp": row[1]
                        })
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode feature value for {entity_id}")
                
                return values
                
        except Exception as e:
            logger.error(f"Failed to get feature values: {str(e)}")
            return []
    
    def get_latest_feature_values(self, 
                                 feature_names: List[str],
                                 entity_id: str,
                                 version: Optional[str] = None) -> Dict[str, Any]:
        """Get latest values for multiple features"""
        try:
            values = {}
            
            for feature_name in feature_names:
                feature_values = self.get_feature_values(
                    feature_name, entity_id, version
                )
                if feature_values:
                    values[feature_name] = feature_values[0]["value"]
                else:
                    values[feature_name] = None
            
            return values
            
        except Exception as e:
            logger.error(f"Failed to get latest feature values: {str(e)}")
            return {}
    
    def batch_store_features(self, 
                           features_data: List[Dict[str, Any]]) -> bool:
        """Store multiple feature values in batch"""
        try:
            with self.db_engine.connect() as conn:
                for data in features_data:
                    feature_id = data["feature_id"]
                    entity_id = data["entity_id"]
                    value = data["value"]
                    timestamp = data.get("timestamp", datetime.now())
                    
                    # Serialize value
                    if isinstance(value, (dict, list)):
                        value_json = json.dumps(value)
                    else:
                        value_json = json.dumps({"value": value})
                    
                    conn.execute(text("""
                        INSERT INTO feature_values (feature_id, entity_id, value, timestamp)
                        VALUES (:feature_id, :entity_id, :value, :timestamp)
                    """), {
                        "feature_id": feature_id,
                        "entity_id": entity_id,
                        "value": value_json,
                        "timestamp": timestamp
                    })
                
                conn.commit()
                logger.info(f"Stored {len(features_data)} feature values in batch")
                return True
                
        except Exception as e:
            logger.error(f"Failed to batch store features: {str(e)}")
            return False
    
    def get_feature_lineage(self, feature_id: int) -> List[Dict[str, Any]]:
        """Get feature lineage information"""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT source_table, source_column, transformation, created_at
                    FROM feature_lineage
                    WHERE feature_id = :feature_id
                    ORDER BY created_at DESC
                """), {"feature_id": feature_id})
                
                rows = result.fetchall()
                
                lineage = []
                for row in rows:
                    lineage.append({
                        "source_table": row[0],
                        "source_column": row[1],
                        "transformation": row[2],
                        "created_at": row[3]
                    })
                
                return lineage
                
        except Exception as e:
            logger.error(f"Failed to get feature lineage: {str(e)}")
            return []
    
    def add_feature_lineage(self, 
                          feature_id: int,
                          source_table: str,
                          source_column: str,
                          transformation: str = "") -> bool:
        """Add lineage information for a feature"""
        try:
            with self.db_engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO feature_lineage (feature_id, source_table, source_column, transformation)
                    VALUES (:feature_id, :source_table, :source_column, :transformation)
                """), {
                    "feature_id": feature_id,
                    "source_table": source_table,
                    "source_column": source_column,
                    "transformation": transformation
                })
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to add feature lineage: {str(e)}")
            return False
    
    def list_features(self, 
                     entity_type: Optional[str] = None,
                     feature_type: Optional[str] = None,
                     active_only: bool = True) -> List[Dict[str, Any]]:
        """List all features with optional filtering"""
        try:
            with self.db_engine.connect() as conn:
                query = "SELECT * FROM features WHERE 1=1"
                params = {}
                
                if entity_type:
                    query += " AND entity_type = :entity_type"
                    params["entity_type"] = entity_type
                
                if feature_type:
                    query += " AND feature_type = :feature_type"
                    params["feature_type"] = feature_type
                
                if active_only:
                    query += " AND is_active = TRUE"
                
                query += " ORDER BY name, created_at DESC"
                
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                
                features = []
                for row in rows:
                    features.append({
                        "id": row[0],
                        "name": row[1],
                        "version": row[2],
                        "entity_type": row[3],
                        "feature_type": row[4],
                        "description": row[5],
                        "created_at": row[6],
                        "updated_at": row[7],
                        "is_active": row[8]
                    })
                
                return features
                
        except Exception as e:
            logger.error(f"Failed to list features: {str(e)}")
            return []
    
    def deactivate_feature(self, feature_id: int) -> bool:
        """Deactivate a feature"""
        try:
            with self.db_engine.connect() as conn:
                conn.execute(text("""
                    UPDATE features 
                    SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                    WHERE id = :feature_id
                """), {"feature_id": feature_id})
                
                conn.commit()
                logger.info(f"Deactivated feature ID: {feature_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to deactivate feature: {str(e)}")
            return False
    
    def get_feature_statistics(self, feature_id: int) -> Dict[str, Any]:
        """Get statistics for a feature"""
        try:
            with self.db_engine.connect() as conn:
                # Get basic counts
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_values,
                        COUNT(DISTINCT entity_id) as unique_entities,
                        MIN(timestamp) as earliest_value,
                        MAX(timestamp) as latest_value
                    FROM feature_values
                    WHERE feature_id = :feature_id
                """), {"feature_id": feature_id})
                
                row = result.fetchone()
                
                stats = {
                    "total_values": row[0],
                    "unique_entities": row[1],
                    "earliest_value": row[2],
                    "latest_value": row[3]
                }
                
                # Get value distribution for numeric features
                feature = self.get_feature_by_id(feature_id)
                if feature and feature["feature_type"] == "numeric":
                    result = conn.execute(text("""
                        SELECT 
                            AVG((value->>'value')::float) as mean_value,
                            MIN((value->>'value')::float) as min_value,
                            MAX((value->>'value')::float) as max_value,
                            STDDEV((value->>'value')::float) as std_value
                        FROM feature_values
                        WHERE feature_id = :feature_id
                        AND value->>'value' ~ '^[0-9]+\.?[0-9]*$'
                    """), {"feature_id": feature_id})
                    
                    numeric_row = result.fetchone()
                    if numeric_row:
                        stats.update({
                            "mean_value": float(numeric_row[0]) if numeric_row[0] else None,
                            "min_value": float(numeric_row[1]) if numeric_row[1] else None,
                            "max_value": float(numeric_row[2]) if numeric_row[2] else None,
                            "std_value": float(numeric_row[3]) if numeric_row[3] else None
                        })
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get feature statistics: {str(e)}")
            return {}
    
    def get_feature_by_id(self, feature_id: int) -> Optional[Dict[str, Any]]:
        """Get feature by ID"""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT * FROM features WHERE id = :feature_id
                """), {"feature_id": feature_id})
                
                row = result.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "name": row[1],
                        "version": row[2],
                        "entity_type": row[3],
                        "feature_type": row[4],
                        "description": row[5],
                        "created_at": row[6],
                        "updated_at": row[7],
                        "is_active": row[8]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get feature by ID: {str(e)}")
            return None
    
    def clear_cache(self, feature_id: Optional[int] = None, entity_id: Optional[str] = None):
        """Clear feature cache"""
        if not self.redis_client:
            return
        
        try:
            if feature_id and entity_id:
                cache_key = f"feature:{feature_id}:{entity_id}"
                self.redis_client.delete(cache_key)
            elif feature_id:
                pattern = f"feature:{feature_id}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                pattern = "feature:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            
            logger.info("Feature cache cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
    
    def get_store_health(self) -> Dict[str, Any]:
        """Get feature store health status"""
        health = {
            "timestamp": datetime.now().isoformat(),
            "database_connected": False,
            "redis_connected": False,
            "total_features": 0,
            "total_values": 0,
            "status": "healthy"
        }
        
        try:
            # Check database
            with self.db_engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM features"))
                health["total_features"] = result.fetchone()[0]
                health["database_connected"] = True
                
                result = conn.execute(text("SELECT COUNT(*) FROM feature_values"))
                health["total_values"] = result.fetchone()[0]
            
            # Check Redis
            if self.redis_client:
                self.redis_client.ping()
                health["redis_connected"] = True
            
            # Determine overall status
            if not health["database_connected"]:
                health["status"] = "unhealthy"
            elif not health["redis_connected"] and self.config.enable_caching:
                health["status"] = "degraded"
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health
