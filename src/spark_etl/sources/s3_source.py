"""
S3 Data Source Implementation
Handles extraction from S3 buckets
"""

import logging
from typing import Dict, Any, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from .data_source_factory import BaseDataSource

logger = logging.getLogger(__name__)


class S3DataSource(BaseDataSource):
    """S3 data source implementation"""
    
    def extract(self, source_path: str) -> DataFrame:
        """Extract data from S3"""
        try:
            # Determine S3 path
            if source_path in self.config:
                s3_path = self.config[source_path]
            else:
                s3_path = source_path
            
            self.logger.info(f"Extracting S3 data from: {s3_path}")
            
            # Configure S3 access
            bucket = self.config.get('readings_bucket', 'metrify-raw-data')
            prefix = self.config.get('readings_prefix', '')
            region = self.config.get('region', 'eu-central-1')
            
            # Build S3 path
            full_s3_path = f"s3a://{bucket}/{prefix}{s3_path}"
            
            # Read from S3 (assuming Parquet format)
            df = self.spark.read.parquet(full_s3_path)
            
            record_count = df.count()
            self.logger.info(f"Successfully extracted {record_count} records from S3: {full_s3_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"S3 extraction failed: {str(e)}")
            raise
    
    def validate_config(self) -> bool:
        """Validate S3 data source configuration"""
        try:
            required_fields = ['readings_bucket', 'region']
            for field in required_fields:
                if not self.config.get(field):
                    self.logger.error(f"Missing S3 configuration field: {field}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"S3 configuration validation failed: {str(e)}")
            return False
