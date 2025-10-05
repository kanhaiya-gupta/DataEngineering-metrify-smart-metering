"""
Data Sources Package
Contains implementations for various data source types
"""

from .data_source_factory import DataSourceFactory, BaseDataSource, DataSourceManager
from .csv_source import CSVDataSource
from .api_source import APIDataSource
from .kafka_source import KafkaDataSource

# Placeholder imports for future implementations
try:
    from .json_source import JSONDataSource
except ImportError:
    JSONDataSource = None

try:
    from .parquet_source import ParquetDataSource
except ImportError:
    ParquetDataSource = None

try:
    from .database_source import DatabaseDataSource
except ImportError:
    DatabaseDataSource = None

try:
    from .s3_source import S3DataSource
except ImportError:
    S3DataSource = None

try:
    from .snowflake_source import SnowflakeDataSource
except ImportError:
    SnowflakeDataSource = None

__all__ = [
    'DataSourceFactory',
    'BaseDataSource', 
    'DataSourceManager',
    'CSVDataSource',
    'APIDataSource',
    'KafkaDataSource',
    'JSONDataSource',
    'ParquetDataSource',
    'DatabaseDataSource',
    'S3DataSource',
    'SnowflakeDataSource'
]
