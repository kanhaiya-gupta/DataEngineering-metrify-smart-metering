"""
Snowflake Data Source Implementation
Handles extraction from Snowflake tables
"""

import logging
from typing import Dict, Any, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from .data_source_factory import BaseDataSource

logger = logging.getLogger(__name__)


class SnowflakeDataSource(BaseDataSource):
    """Snowflake data source implementation"""
    
    def extract(self, source_path: str) -> DataFrame:
        """Extract data from Snowflake tables"""
        try:
            # Determine table name
            if source_path in self.config:
                table_name = self.config[source_path]
            else:
                table_name = source_path
            
            self.logger.info(f"Extracting Snowflake data from table: {table_name}")
            
            # Read from Snowflake
            df = self.spark.read \
                .format("net.snowflake.spark.snowflake") \
                .option("sfURL", f"{self.config.get('account', '')}.snowflakecomputing.com") \
                .option("sfUser", self.config.get('username', '')) \
                .option("sfPassword", self.config.get('password', '')) \
                .option("sfDatabase", self.config.get('database', '')) \
                .option("sfSchema", self.config.get('schema', 'PUBLIC')) \
                .option("sfWarehouse", self.config.get('warehouse', '')) \
                .option("dbtable", table_name) \
                .load()
            
            record_count = df.count()
            self.logger.info(f"Successfully extracted {record_count} records from Snowflake table: {table_name}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Snowflake extraction failed: {str(e)}")
            raise
    
    def validate_config(self) -> bool:
        """Validate Snowflake data source configuration"""
        try:
            required_fields = ['account', 'username', 'password', 'database', 'warehouse']
            for field in required_fields:
                if not self.config.get(field):
                    self.logger.error(f"Missing Snowflake configuration field: {field}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Snowflake configuration validation failed: {str(e)}")
            return False
