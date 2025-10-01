"""
Smart Meter Data Ingestion Pipeline
Handles real-time and batch ingestion of smart meter readings
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from kafka import KafkaProducer, KafkaConsumer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, when, isnan, isnull
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
import boto3
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SmartMeterReading:
    """Data class for smart meter readings"""
    meter_id: str
    timestamp: datetime
    consumption_kwh: float
    voltage: float
    current: float
    power_factor: float
    frequency: float
    temperature: Optional[float] = None
    humidity: Optional[float] = None

class SmartMeterIngestionPipeline:
    """Main class for smart meter data ingestion"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the ingestion pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.spark = self._init_spark_session()
        self.kafka_producer = self._init_kafka_producer()
        self.s3_client = boto3.client('s3')
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _init_spark_session(self) -> SparkSession:
        """Initialize Spark session for data processing"""
        return SparkSession.builder \
            .appName("SmartMeterIngestion") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
    
    def _init_kafka_producer(self) -> KafkaProducer:
        """Initialize Kafka producer for real-time data streaming"""
        kafka_config = self.config['kafka']
        return KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            security_protocol=kafka_config.get('security_protocol', 'PLAINTEXT'),
            sasl_mechanism=kafka_config.get('sasl_mechanism'),
            sasl_plain_username=kafka_config.get('sasl_username'),
            sasl_plain_password=kafka_config.get('sasl_password')
        )
    
    def get_smart_meter_schema(self) -> StructType:
        """Define the schema for smart meter readings"""
        return StructType([
            StructField("meter_id", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("consumption_kwh", DoubleType(), False),
            StructField("voltage", DoubleType(), False),
            StructField("current", DoubleType(), False),
            StructField("power_factor", DoubleType(), True),
            StructField("frequency", DoubleType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("humidity", DoubleType(), True),
            StructField("data_quality_score", DoubleType(), True)
        ])
    
    def validate_reading(self, reading: SmartMeterReading) -> bool:
        """Validate smart meter reading against quality rules"""
        quality_rules = self.config['data_quality']['smart_meter_readings']['validations']
        
        # Check consumption range
        if not (quality_rules['consumption_kwh']['min_value'] <= 
                reading.consumption_kwh <= 
                quality_rules['consumption_kwh']['max_value']):
            logger.warning(f"Invalid consumption value: {reading.consumption_kwh}")
            return False
        
        # Check voltage range
        if not (quality_rules['voltage']['min_value'] <= 
                reading.voltage <= 
                quality_rules['voltage']['max_value']):
            logger.warning(f"Invalid voltage value: {reading.voltage}")
            return False
        
        # Check current range
        if not (quality_rules['current']['min_value'] <= 
                reading.current <= 
                quality_rules['current']['max_value']):
            logger.warning(f"Invalid current value: {reading.current}")
            return False
        
        return True
    
    def calculate_data_quality_score(self, reading: SmartMeterReading) -> float:
        """Calculate data quality score for a reading"""
        score = 1.0
        
        # Check for missing values
        if reading.temperature is None:
            score -= 0.1
        if reading.humidity is None:
            score -= 0.1
        
        # Check for reasonable values
        if reading.power_factor < 0.8 or reading.power_factor > 1.0:
            score -= 0.2
        
        if reading.frequency < 49.5 or reading.frequency > 50.5:
            score -= 0.3
        
        return max(0.0, score)
    
    def process_real_time_stream(self, topic: str = "smart-meter-readings"):
        """Process real-time smart meter data stream"""
        logger.info(f"Starting real-time stream processing for topic: {topic}")
        
        # Create Kafka consumer
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='smart-meter-processor'
        )
        
        try:
            for message in consumer:
                reading_data = message.value
                reading = SmartMeterReading(**reading_data)
                
                # Validate and process reading
                if self.validate_reading(reading):
                    reading.data_quality_score = self.calculate_data_quality_score(reading)
                    
                    # Store in S3 for batch processing
                    self._store_reading_s3(reading)
                    
                    # Send to analytics topic
                    self._send_to_analytics(reading)
                    
                    logger.info(f"Processed reading for meter {reading.meter_id}")
                else:
                    logger.warning(f"Invalid reading rejected for meter {reading.meter_id}")
                    
        except KeyboardInterrupt:
            logger.info("Stopping real-time stream processing")
        finally:
            consumer.close()
    
    def process_batch_data(self, date: str, s3_path: str):
        """Process batch smart meter data from S3"""
        logger.info(f"Processing batch data for date: {date}")
        
        # Read data from S3
        df = self.spark.read.json(s3_path)
        
        # Apply schema and transformations
        df = df.select(
            col("meter_id"),
            to_timestamp(col("timestamp")).alias("timestamp"),
            col("consumption_kwh").cast("double"),
            col("voltage").cast("double"),
            col("current").cast("double"),
            col("power_factor").cast("double"),
            col("frequency").cast("double"),
            col("temperature").cast("double"),
            col("humidity").cast("double")
        )
        
        # Data quality transformations
        df = df.withColumn(
            "data_quality_score",
            when(
                (col("consumption_kwh") >= 0) & 
                (col("consumption_kwh") <= 1000) &
                (col("voltage") >= 200) & 
                (col("voltage") <= 250) &
                (col("current") >= 0) & 
                (col("current") <= 100),
                1.0
            ).otherwise(0.5)
        )
        
        # Filter out invalid readings
        df_clean = df.filter(col("data_quality_score") >= 0.5)
        
        # Write to Snowflake
        self._write_to_snowflake(df_clean, f"smart_meter_readings_{date}")
        
        logger.info(f"Processed {df_clean.count()} valid readings for date: {date}")
    
    def _store_reading_s3(self, reading: SmartMeterReading):
        """Store individual reading to S3 for batch processing"""
        s3_key = f"raw/smart_meter_readings/{reading.timestamp.strftime('%Y/%m/%d')}/{reading.meter_id}_{reading.timestamp.isoformat()}.json"
        
        reading_dict = {
            "meter_id": reading.meter_id,
            "timestamp": reading.timestamp.isoformat(),
            "consumption_kwh": reading.consumption_kwh,
            "voltage": reading.voltage,
            "current": reading.current,
            "power_factor": reading.power_factor,
            "frequency": reading.frequency,
            "temperature": reading.temperature,
            "humidity": reading.humidity,
            "data_quality_score": reading.data_quality_score
        }
        
        self.s3_client.put_object(
            Bucket=self.config['aws']['s3_bucket'],
            Key=s3_key,
            Body=json.dumps(reading_dict),
            ContentType='application/json'
        )
    
    def _send_to_analytics(self, reading: SmartMeterReading):
        """Send processed reading to analytics topic"""
        analytics_data = {
            "meter_id": reading.meter_id,
            "timestamp": reading.timestamp.isoformat(),
            "consumption_kwh": reading.consumption_kwh,
            "data_quality_score": reading.data_quality_score,
            "processing_timestamp": datetime.utcnow().isoformat()
        }
        
        self.kafka_producer.send(
            'smart-meter-analytics',
            value=analytics_data
        )
    
    def _write_to_snowflake(self, df, table_name: str):
        """Write DataFrame to Snowflake"""
        df.write \
            .format("snowflake") \
            .option("sfURL", self.config['databases']['snowflake']['account']) \
            .option("sfUser", self.config['databases']['snowflake']['user']) \
            .option("sfPassword", self.config['databases']['snowflake']['password']) \
            .option("sfDatabase", self.config['databases']['snowflake']['database']) \
            .option("sfSchema", self.config['databases']['snowflake']['schema']) \
            .option("sfWarehouse", self.config['databases']['snowflake']['warehouse']) \
            .option("dbtable", table_name) \
            .mode("append") \
            .save()

def main():
    """Main function to run the ingestion pipeline"""
    pipeline = SmartMeterIngestionPipeline()
    
    # Run real-time processing
    pipeline.process_real_time_stream()

if __name__ == "__main__":
    main()
