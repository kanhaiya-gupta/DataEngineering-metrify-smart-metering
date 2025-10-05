"""
Data Source Factory
Factory pattern for creating data sources based on configuration
Supports multiple data source types: CSV, JSON, Parquet, API, Database, Kafka, S3, Snowflake
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from .csv_source import CSVDataSource
from .json_source import JSONDataSource
from .parquet_source import ParquetDataSource
from .api_source import APIDataSource
from .database_source import DatabaseDataSource
from .kafka_source import KafkaDataSource
from .s3_source import S3DataSource
from .snowflake_source import SnowflakeDataSource

logger = logging.getLogger(__name__)


class DataSourceFactory:
    """
    Factory for creating data sources based on configuration
    """
    
    _source_types = {
        'csv': CSVDataSource,
        'json': JSONDataSource,
        'parquet': ParquetDataSource,
        'api': APIDataSource,
        'database': DatabaseDataSource,
        'kafka': KafkaDataSource,
        's3': S3DataSource,
        'snowflake': SnowflakeDataSource
    }
    
    @classmethod
    def create_source(
        self, 
        source_type: str, 
        spark: SparkSession, 
        config: Dict[str, Any],
        schema: Optional[StructType] = None
    ) -> 'BaseDataSource':
        """
        Create a data source instance based on type
        
        Args:
            source_type: Type of data source (csv, json, parquet, api, database, kafka, s3, snowflake)
            spark: Spark session
            config: Configuration for the data source
            schema: Optional schema for the data
            
        Returns:
            Data source instance
            
        Raises:
            ValueError: If source type is not supported
        """
        if source_type not in self._source_types:
            supported_types = ', '.join(self._source_types.keys())
            raise ValueError(f"Unsupported source type: {source_type}. Supported types: {supported_types}")
        
        source_class = self._source_types[source_type]
        logger.info(f"Creating {source_type} data source with config: {list(config.keys())}")
        
        return source_class(spark, config, schema)
    
    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported data source types"""
        return list(cls._source_types.keys())
    
    @classmethod
    def register_source_type(cls, source_type: str, source_class: type):
        """
        Register a new data source type
        
        Args:
            source_type: Name of the source type
            source_class: Class implementing the data source
        """
        if not issubclass(source_class, BaseDataSource):
            raise ValueError(f"Source class must inherit from BaseDataSource")
        
        cls._source_types[source_type] = source_class
        logger.info(f"Registered new data source type: {source_type}")


class BaseDataSource(ABC):
    """
    Abstract base class for all data sources
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any], schema: Optional[StructType] = None):
        self.spark = spark
        self.config = config
        self.schema = schema
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def extract(self, source_path: str) -> DataFrame:
        """
        Extract data from the source
        
        Args:
            source_path: Path or identifier for the data source
            
        Returns:
            DataFrame containing the extracted data
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate the configuration for this data source
        
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the data source
        
        Returns:
            Dictionary containing metadata
        """
        return {
            'source_type': self.__class__.__name__,
            'config_keys': list(self.config.keys()),
            'has_schema': self.schema is not None
        }


class DataSourceManager:
    """
    Manager for handling multiple data sources and environment-specific configurations
    """
    
    def __init__(self, spark: SparkSession, data_source_config: Dict[str, Any], environment: str = "development"):
        self.spark = spark
        self.config = data_source_config
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
        # Get environment-specific overrides
        self.env_config = self._get_environment_config()
        
        # Determine primary source
        self.primary_source = self.env_config.get('primary_source', self.config.get('primary_source', 'csv'))
        
        self.logger.info(f"Data source manager initialized for environment: {environment}, primary source: {self.primary_source}")
    
    def _get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration overrides"""
        env_config = self.config.get('environments', {}).get(self.environment, {})
        
        # Merge with base config
        merged_config = self.config.copy()
        if env_config:
            merged_config.update(env_config)
            # Deep merge data_sources if present
            if 'data_sources' in env_config:
                merged_config['data_sources'] = {
                    **self.config.get('data_sources', {}),
                    **env_config['data_sources']
                }
        
        return merged_config
    
    def create_primary_source(self, schema: Optional[StructType] = None) -> BaseDataSource:
        """
        Create the primary data source for the current environment
        
        Args:
            schema: Optional schema for the data
            
        Returns:
            Primary data source instance
        """
        source_config = self.env_config.get('data_sources', {}).get(self.primary_source, {})
        
        if not source_config:
            raise ValueError(f"No configuration found for primary source: {self.primary_source}")
        
        return DataSourceFactory.create_source(
            self.primary_source,
            self.spark,
            source_config,
            schema
        )
    
    def create_source(self, source_type: str, schema: Optional[StructType] = None) -> BaseDataSource:
        """
        Create a specific data source type
        
        Args:
            source_type: Type of data source to create
            schema: Optional schema for the data
            
        Returns:
            Data source instance
        """
        source_config = self.env_config.get('data_sources', {}).get(source_type, {})
        
        if not source_config:
            raise ValueError(f"No configuration found for source type: {source_type}")
        
        return DataSourceFactory.create_source(
            source_type,
            self.spark,
            source_config,
            schema
        )
    
    def get_available_sources(self) -> list:
        """Get list of available data source types"""
        return list(self.env_config.get('data_sources', {}).keys())
    
    def switch_primary_source(self, new_source: str):
        """
        Switch the primary data source
        
        Args:
            new_source: New primary source type
        """
        if new_source not in self.get_available_sources():
            raise ValueError(f"Source type {new_source} not available. Available: {self.get_available_sources()}")
        
        self.primary_source = new_source
        self.logger.info(f"Switched primary source to: {new_source}")
    
    def get_source_config(self, source_type: str) -> Dict[str, Any]:
        """Get configuration for a specific source type"""
        return self.env_config.get('data_sources', {}).get(source_type, {})
