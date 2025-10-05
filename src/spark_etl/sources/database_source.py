"""
Database Data Source Implementation
Handles extraction from various database systems
"""

import logging
from typing import Dict, Any, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from .data_source_factory import BaseDataSource

logger = logging.getLogger(__name__)


class DatabaseDataSource(BaseDataSource):
    """Database data source implementation"""
    
    def extract(self, source_path: str) -> DataFrame:
        """Extract data from database tables"""
        try:
            # Determine table name
            if source_path in self.config:
                table_name = self.config[source_path]
            else:
                table_name = source_path
            
            self.logger.info(f"Extracting database data from table: {table_name}")
            
            # Build JDBC connection string
            connection = self.config.get('connection', {})
            db_type = connection.get('type', 'postgresql')
            host = connection.get('host', 'localhost')
            port = connection.get('port', '5432')
            database = connection.get('database', 'metrify')
            username = connection.get('username', 'postgres')
            password = connection.get('password', '')
            
            # Create JDBC URL
            if db_type == 'postgresql':
                jdbc_url = f"jdbc:postgresql://{host}:{port}/{database}"
            elif db_type == 'mysql':
                jdbc_url = f"jdbc:mysql://{host}:{port}/{database}"
            elif db_type == 'oracle':
                jdbc_url = f"jdbc:oracle:thin:@{host}:{port}:{database}"
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            # Read from database
            df = self.spark.read \
                .format("jdbc") \
                .option("url", jdbc_url) \
                .option("dbtable", table_name) \
                .option("user", username) \
                .option("password", password) \
                .load()
            
            record_count = df.count()
            self.logger.info(f"Successfully extracted {record_count} records from database table: {table_name}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Database extraction failed: {str(e)}")
            raise
    
    def validate_config(self) -> bool:
        """Validate database data source configuration"""
        try:
            connection = self.config.get('connection', {})
            required_fields = ['type', 'host', 'port', 'database', 'username', 'password']
            
            for field in required_fields:
                if not connection.get(field):
                    self.logger.error(f"Missing database connection field: {field}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Database configuration validation failed: {str(e)}")
            return False
