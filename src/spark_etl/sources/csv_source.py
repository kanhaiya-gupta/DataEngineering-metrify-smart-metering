"""
CSV Data Source Implementation
Handles extraction from CSV files with configurable options
"""

import logging
from typing import Dict, Any, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from .data_source_factory import BaseDataSource

logger = logging.getLogger(__name__)


class CSVDataSource(BaseDataSource):
    """
    CSV file data source implementation
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any], schema: Optional[StructType] = None):
        super().__init__(spark, config, schema)
        self.csv_settings = config.get('csv_settings', {})
    
    def extract(self, source_path: str) -> DataFrame:
        """
        Extract data from CSV files
        
        Args:
            source_path: Path to CSV file or directory
            
        Returns:
            DataFrame containing the CSV data
        """
        try:
            self.logger.info(f"Extracting CSV data from: {source_path}")
            
            # Build Spark read options from configuration
            read_options = {
                'header': str(self.csv_settings.get('has_header', True)).lower(),
                'inferSchema': 'true',
                'mode': 'PERMISSIVE',
                'timestampFormat': 'yyyy-MM-dd\'T\'HH:mm:ss.SSSSSS'
            }
            
            # Add CSV-specific options
            if 'delimiter' in self.csv_settings:
                read_options['delimiter'] = self.csv_settings['delimiter']
            if 'encoding' in self.csv_settings:
                read_options['encoding'] = self.csv_settings['encoding']
            if 'quote' in self.csv_settings:
                read_options['quote'] = self.csv_settings['quote']
            if 'escape' in self.csv_settings:
                read_options['escape'] = self.csv_settings['escape']
            if 'nullValue' in self.csv_settings:
                read_options['nullValue'] = self.csv_settings['nullValue']
            
            # Start building the DataFrame reader
            reader = self.spark.read
            
            # Apply all options
            for key, value in read_options.items():
                reader = reader.option(key, value)
            
            # Read the CSV file(s)
            if source_path.endswith('*.csv'):
                # Pattern matching for multiple files
                df = reader.csv(source_path)
            else:
                # Single file or directory
                df = reader.csv(f"{source_path}*.csv")
            
            # Apply schema if provided
            if self.schema:
                df = df.select(*[col.name for col in self.schema.fields])
            
            record_count = df.count()
            self.logger.info(f"Successfully extracted {record_count} records from CSV source")
            
            return df
            
        except Exception as e:
            self.logger.error(f"CSV extraction failed: {str(e)}")
            raise
    
    def validate_config(self) -> bool:
        """
        Validate CSV data source configuration
        
        Returns:
            True if configuration is valid
        """
        try:
            # Check required CSV settings
            required_settings = ['delimiter', 'has_header', 'encoding']
            for setting in required_settings:
                if setting not in self.csv_settings:
                    self.logger.warning(f"Missing CSV setting: {setting}")
            
            # Validate delimiter
            delimiter = self.csv_settings.get('delimiter', ',')
            if len(delimiter) != 1:
                self.logger.error(f"Invalid delimiter: {delimiter}. Must be a single character.")
                return False
            
            # Validate encoding
            encoding = self.csv_settings.get('encoding', 'utf-8')
            valid_encodings = ['utf-8', 'utf-16', 'ascii', 'latin-1', 'cp1252']
            if encoding.lower() not in valid_encodings:
                self.logger.warning(f"Uncommon encoding: {encoding}. Common encodings: {valid_encodings}")
            
            self.logger.info("CSV configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"CSV configuration validation failed: {str(e)}")
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get CSV-specific metadata"""
        metadata = super().get_metadata()
        metadata.update({
            'csv_settings': self.csv_settings,
            'delimiter': self.csv_settings.get('delimiter', ','),
            'has_header': self.csv_settings.get('has_header', True),
            'encoding': self.csv_settings.get('encoding', 'utf-8')
        })
        return metadata
