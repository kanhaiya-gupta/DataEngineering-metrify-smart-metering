"""
JSON Data Source Implementation
Handles extraction from JSON files
"""

import logging
from typing import Dict, Any, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from .data_source_factory import BaseDataSource

logger = logging.getLogger(__name__)


class JSONDataSource(BaseDataSource):
    """JSON file data source implementation"""
    
    def extract(self, source_path: str) -> DataFrame:
        """Extract data from JSON files"""
        try:
            self.logger.info(f"Extracting JSON data from: {source_path}")
            
            # Read JSON files
            df = self.spark.read \
                .option("multiline", "true") \
                .option("mode", "PERMISSIVE") \
                .json(f"{source_path}*.json")
            
            record_count = df.count()
            self.logger.info(f"Successfully extracted {record_count} records from JSON source")
            
            return df
            
        except Exception as e:
            self.logger.error(f"JSON extraction failed: {str(e)}")
            raise
    
    def validate_config(self) -> bool:
        """Validate JSON data source configuration"""
        return True
