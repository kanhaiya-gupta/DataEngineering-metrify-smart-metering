"""
Kafka Data Source Implementation
Handles extraction from Kafka topics with configurable consumer settings
"""

import logging
from typing import Dict, Any, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.functions import col, current_timestamp, lit

from .data_source_factory import BaseDataSource

logger = logging.getLogger(__name__)


class KafkaDataSource(BaseDataSource):
    """
    Kafka data source implementation
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any], schema: Optional[StructType] = None):
        super().__init__(spark, config, schema)
        self.bootstrap_servers = config.get('bootstrap_servers', 'localhost:9092')
        self.consumer_group = config.get('consumer_group', 'spark-etl-consumer')
        self.security_protocol = config.get('security_protocol', 'PLAINTEXT')
        self.sasl_mechanism = config.get('sasl_mechanism', 'PLAIN')
    
    def extract(self, source_path: str) -> DataFrame:
        """
        Extract data from Kafka topic
        
        Args:
            source_path: Kafka topic name or topic name from config
            
        Returns:
            DataFrame containing the Kafka data
        """
        try:
            # Determine the actual topic name
            if source_path in self.config:
                topic_name = self.config[source_path]
            else:
                topic_name = source_path
            
            self.logger.info(f"Extracting Kafka data from topic: {topic_name}")
            
            # Build Kafka read options
            kafka_options = {
                'kafka.bootstrap.servers': self.bootstrap_servers,
                'subscribe': topic_name,
                'startingOffsets': 'earliest',  # or 'latest' for real-time
                'failOnDataLoss': 'false'
            }
            
            # Add security options if needed
            if self.security_protocol != 'PLAINTEXT':
                kafka_options['kafka.security.protocol'] = self.security_protocol
                if self.sasl_mechanism:
                    kafka_options['kafka.sasl.mechanism'] = self.sasl_mechanism
            
            # Read from Kafka
            df = self.spark.read \
                .format('kafka') \
                .options(**kafka_options) \
                .load()
            
            # Parse the value column (assuming JSON format)
            from pyspark.sql.functions import from_json, col as spark_col
            
            # Define schema for parsing JSON values
            if self.schema:
                value_schema = self.schema
            else:
                # Default schema for common fields
                from pyspark.sql.types import StringType, TimestampType, DoubleType
                value_schema = StructType([
                    StructField("id", StringType(), True),
                    StructField("timestamp", TimestampType(), True),
                    StructField("value", DoubleType(), True)
                ])
            
            # Parse JSON values
            parsed_df = df.select(
                spark_col('key').cast('string').alias('kafka_key'),
                spark_col('value').cast('string').alias('kafka_value'),
                spark_col('topic').alias('kafka_topic'),
                spark_col('partition').alias('kafka_partition'),
                spark_col('offset').alias('kafka_offset'),
                spark_col('timestamp').alias('kafka_timestamp'),
                from_json(spark_col('value').cast('string'), value_schema).alias('parsed_data')
            )
            
            # Flatten the parsed data
            if self.schema:
                # Select all fields from parsed data
                data_columns = [f"parsed_data.{field.name}" for field in self.schema.fields]
                final_df = parsed_df.select(
                    'kafka_key', 'kafka_topic', 'kafka_partition', 'kafka_offset', 'kafka_timestamp',
                    *data_columns
                )
            else:
                # Use parsed data as-is
                final_df = parsed_df.select(
                    'kafka_key', 'kafka_topic', 'kafka_partition', 'kafka_offset', 'kafka_timestamp',
                    'parsed_data.*'
                )
            
            # Add extraction metadata
            final_df = final_df.withColumn("extraction_timestamp", current_timestamp())
            
            record_count = final_df.count()
            self.logger.info(f"Successfully extracted {record_count} records from Kafka topic: {topic_name}")
            
            return final_df
            
        except Exception as e:
            self.logger.error(f"Kafka extraction failed: {str(e)}")
            raise
    
    def extract_streaming(self, source_path: str, checkpoint_location: str) -> DataFrame:
        """
        Extract data from Kafka topic in streaming mode
        
        Args:
            source_path: Kafka topic name
            checkpoint_location: Location for Spark checkpointing
            
        Returns:
            Streaming DataFrame
        """
        try:
            # Determine the actual topic name
            if source_path in self.config:
                topic_name = self.config[source_path]
            else:
                topic_name = source_path
            
            self.logger.info(f"Starting streaming extraction from Kafka topic: {topic_name}")
            
            # Build Kafka streaming options
            kafka_options = {
                'kafka.bootstrap.servers': self.bootstrap_servers,
                'subscribe': topic_name,
                'startingOffsets': 'latest',  # Start from latest for streaming
                'failOnDataLoss': 'false'
            }
            
            # Add security options if needed
            if self.security_protocol != 'PLAINTEXT':
                kafka_options['kafka.security.protocol'] = self.security_protocol
                if self.sasl_mechanism:
                    kafka_options['kafka.sasl.mechanism'] = self.sasl_mechanism
            
            # Read streaming data from Kafka
            streaming_df = self.spark.readStream \
                .format('kafka') \
                .options(**kafka_options) \
                .load()
            
            # Parse the value column (assuming JSON format)
            from pyspark.sql.functions import from_json
            
            # Define schema for parsing JSON values
            if self.schema:
                value_schema = self.schema
            else:
                # Default schema
                from pyspark.sql.types import StringType, TimestampType, DoubleType
                value_schema = StructType([
                    StructField("id", StringType(), True),
                    StructField("timestamp", TimestampType(), True),
                    StructField("value", DoubleType(), True)
                ])
            
            # Parse JSON values
            parsed_streaming_df = streaming_df.select(
                col('key').cast('string').alias('kafka_key'),
                col('value').cast('string').alias('kafka_value'),
                col('topic').alias('kafka_topic'),
                col('partition').alias('kafka_partition'),
                col('offset').alias('kafka_offset'),
                col('timestamp').alias('kafka_timestamp'),
                from_json(col('value').cast('string'), value_schema).alias('parsed_data')
            )
            
            # Flatten the parsed data
            if self.schema:
                data_columns = [f"parsed_data.{field.name}" for field in self.schema.fields]
                final_streaming_df = parsed_streaming_df.select(
                    'kafka_key', 'kafka_topic', 'kafka_partition', 'kafka_offset', 'kafka_timestamp',
                    *data_columns
                )
            else:
                final_streaming_df = parsed_streaming_df.select(
                    'kafka_key', 'kafka_topic', 'kafka_partition', 'kafka_offset', 'kafka_timestamp',
                    'parsed_data.*'
                )
            
            # Add extraction metadata
            final_streaming_df = final_streaming_df.withColumn("extraction_timestamp", current_timestamp())
            
            self.logger.info(f"Streaming extraction setup complete for Kafka topic: {topic_name}")
            
            return final_streaming_df
            
        except Exception as e:
            self.logger.error(f"Kafka streaming extraction failed: {str(e)}")
            raise
    
    def validate_config(self) -> bool:
        """
        Validate Kafka data source configuration
        
        Returns:
            True if configuration is valid
        """
        try:
            # Check required configuration
            if not self.bootstrap_servers:
                self.logger.error("bootstrap_servers is required for Kafka data source")
                return False
            
            if not self.consumer_group:
                self.logger.error("consumer_group is required for Kafka data source")
                return False
            
            # Validate security protocol
            valid_protocols = ['PLAINTEXT', 'SSL', 'SASL_PLAINTEXT', 'SASL_SSL']
            if self.security_protocol not in valid_protocols:
                self.logger.error(f"Invalid security protocol: {self.security_protocol}. Valid: {valid_protocols}")
                return False
            
            # Validate SASL mechanism if using SASL
            if 'SASL' in self.security_protocol:
                valid_mechanisms = ['PLAIN', 'SCRAM-SHA-256', 'SCRAM-SHA-512', 'GSSAPI']
                if self.sasl_mechanism not in valid_mechanisms:
                    self.logger.error(f"Invalid SASL mechanism: {self.sasl_mechanism}. Valid: {valid_mechanisms}")
                    return False
            
            self.logger.info("Kafka configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Kafka configuration validation failed: {str(e)}")
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get Kafka-specific metadata"""
        metadata = super().get_metadata()
        metadata.update({
            'bootstrap_servers': self.bootstrap_servers,
            'consumer_group': self.consumer_group,
            'security_protocol': self.security_protocol,
            'sasl_mechanism': self.sasl_mechanism
        })
        return metadata
