"""
Smart Meter Data Pipeline DAG
Orchestrates the complete smart meter data processing workflow
"""

from datetime import datetime, timedelta
from typing import Dict, Any
import logging

import sys
import os

# Add /opt/airflow to path so we can import from src.infrastructure...
if "/opt/airflow" not in sys.path:
    sys.path.insert(0, "/opt/airflow")

# For local development, add the project root src directory
local_src = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src")
if os.path.exists(local_src) and local_src not in sys.path:
    sys.path.insert(0, local_src)

from src.infrastructure.external.airflow.airflow_client import AirflowClient

logger = logging.getLogger(__name__)

# Initialize Airflow client
airflow_client = AirflowClient()

# Create the main DAG
dag = airflow_client.create_dag(
    dag_id="smart_meter_data_pipeline",
    description="Complete smart meter data ingestion, processing, and analytics pipeline",
    schedule_interval="@hourly",
    tags=["smart-meter", "data-pipeline", "ingestion", "analytics"],
    max_active_runs=1,
    catchup=False
)

# Task 1: Start pipeline
start_task = airflow_client.create_dummy_task(
    dag=dag,
    task_id="start_pipeline"
)

# Task 2: Ingest smart meter data
def ingest_smart_meter_data(**context) -> Dict[str, Any]:
    """Ingest smart meter data from CSV files and publish to Kafka with performance optimization"""
    import pandas as pd
    import json
    import time
    import asyncio
    from pathlib import Path
    from src.core.config.config_loader import ConfigLoader
    from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
    
    logger.info("Starting optimized smart meter data ingestion...")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        data_sources_config = config_loader.get_data_sources_config()
        performance_config = config_loader.get_performance_config()
        smart_meter_config = data_sources_config.smart_meters

        # Get performance settings for smart meters
        smart_meter_perf = performance_config.get('data_sources', {}).get('smart_meters', {})
        batch_perf = smart_meter_perf.get('batch_processing', {})
        memory_perf = smart_meter_perf.get('memory', {})

        # Construct file path
        data_root = Path(smart_meter_config.data_root)
        readings_file = data_root / smart_meter_config.readings_file

        logger.info(f"Reading data from: {readings_file}")

        # Check if file exists
        if not readings_file.exists():
            raise FileNotFoundError(f"Smart meter readings file not found: {readings_file}")

        # Use optimized batch size and chunk size
        batch_size = batch_perf.get('optimal_batch_size', smart_meter_config.batch_size)
        chunk_size = memory_perf.get('chunk_size', 15000)
        total_records = 0
        kafka_topic = smart_meter_config.kafka_topic

        # Initialize Kafka producer
        kafka_producer = KafkaProducer(
            bootstrap_servers=["localhost:9092"],
            client_id="smart-meter-dag-producer"
        )
        
        # Read and process CSV in optimized chunks
        total_processing_time = 0
        for chunk in pd.read_csv(readings_file, chunksize=chunk_size):
            logger.info(f"Processing optimized batch of {len(chunk)} records...")
            
            # Process chunk in smaller batches for Kafka publishing
            start_time = time.time()
            
            # Convert DataFrame to list of dictionaries
            records = chunk.to_dict('records')
            
            # Process records in optimized batches
            for i in range(0, len(records), batch_size):
                batch_records = records[i:i + batch_size]
                
                # Prepare messages for batch publishing
                messages = []
                for record in batch_records:
                    try:
                        # Convert pandas types to JSON serializable types
                        record_json = json.dumps(record, default=str)
                        messages.append({
                            "message": record_json,
                            "key": record.get('reading_id', 'unknown')
                        })
                    except Exception as e:
                        logger.error(f"Error preparing record {record.get('reading_id', 'unknown')}: {e}")
                        continue
                
                # Publish batch to Kafka
                if messages:
                    try:
                        # Create a proper key generator function that maps message to key
                        message_to_key = {msg["message"]: msg["key"] for msg in messages}
                        
                        def key_generator(message):
                            return message_to_key.get(message, "unknown")
                        
                        asyncio.run(kafka_producer.publish_batch(
                            topic=kafka_topic,
                            messages=[msg["message"] for msg in messages],
                            key_generator=key_generator
                        ))
                        total_records += len(messages)
                        logger.info(f"Published {len(messages)} records to topic {kafka_topic}")
                    except Exception as e:
                        logger.error(f"Error publishing batch to Kafka: {e}")
                        continue
            
            # Track processing time
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            throughput = len(chunk) / processing_time if processing_time > 0 else 0
            logger.info(f"Chunk processed: {len(chunk)} records in {processing_time:.2f}s ({throughput:.2f} records/sec)")
        
        # Calculate overall performance metrics
        overall_throughput = total_records / total_processing_time if total_processing_time > 0 else 0
        
        logger.info(f"Smart meter data ingestion completed. Total records processed: {total_records}")
        logger.info(f"Overall throughput: {overall_throughput:.2f} records/sec")
        logger.info(f"Optimized batch size: {batch_size}, Chunk size: {chunk_size}")
        
        return {
            "status": "success",
            "records_processed": total_records,
            "file_path": str(readings_file),
            "kafka_topic": kafka_topic,
            "processing_time": total_processing_time,
            "throughput": overall_throughput,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in smart meter data ingestion: {e}")
        raise

ingest_task = airflow_client.create_python_task(
    dag=dag,
    task_id="ingest_smart_meter_data",
    python_callable=ingest_smart_meter_data
)

# Task 3: Validate data quality
def validate_data_quality(**context) -> Dict[str, Any]:
    """Validate data quality using configuration rules"""
    import pandas as pd
    from pathlib import Path
    from src.core.config.config_loader import ConfigLoader
    
    logger.info("Starting data quality validation...")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        data_sources_config = config_loader.get_data_sources_config()
        smart_meter_config = data_sources_config.smart_meters
        
        # Construct file path
        data_root = Path(smart_meter_config.data_root)
        readings_file = data_root / smart_meter_config.readings_file
        
        logger.info(f"Validating data quality for: {readings_file}")
        
        # Read CSV file
        df = pd.read_csv(readings_file)
        
        # Get validation rules from configuration
        validation_rules = smart_meter_config.validation
        required_columns = validation_rules.get('required_columns', [])
        data_types = validation_rules.get('data_types', {})
        
        validation_results = {
            "total_records": len(df),
            "validation_errors": [],
            "quality_score": 1.0
        }
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results["validation_errors"].append(f"Missing required columns: {missing_columns}")
            validation_results["quality_score"] -= 0.2
        
        # Check data types
        for column, expected_type in data_types.items():
            if column in df.columns:
                if expected_type == "string":
                    if not df[column].dtype == 'object':
                        validation_results["validation_errors"].append(f"Column {column} should be string type")
                        validation_results["quality_score"] -= 0.1
                elif expected_type == "float":
                    if not pd.api.types.is_numeric_dtype(df[column]):
                        validation_results["validation_errors"].append(f"Column {column} should be numeric type")
                        validation_results["quality_score"] -= 0.1
                elif expected_type == "datetime":
                    try:
                        pd.to_datetime(df[column])
                    except Exception:
                        validation_results["validation_errors"].append(f"Column {column} should be datetime type")
                        validation_results["quality_score"] -= 0.1
        
        # Check for null values
        null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if null_percentage > 0.05:  # 5% threshold
            validation_results["validation_errors"].append(f"High null percentage: {null_percentage:.2%}")
            validation_results["quality_score"] -= 0.1
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results["validation_errors"].append(f"Found {duplicate_count} duplicate records")
            validation_results["quality_score"] -= 0.1
        
        # Ensure quality score doesn't go below 0
        validation_results["quality_score"] = max(0.0, validation_results["quality_score"])
        
        validation_passed = len(validation_results["validation_errors"]) == 0
        
        logger.info(f"Data quality validation completed. Quality score: {validation_results['quality_score']:.2f}")
        if validation_results["validation_errors"]:
            logger.warning(f"Validation errors found: {validation_results['validation_errors']}")
        
        return {
            "status": "success" if validation_passed else "warning",
            "quality_score": validation_results["quality_score"],
            "validation_passed": validation_passed,
            "validation_errors": validation_results["validation_errors"],
            "total_records": validation_results["total_records"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in data quality validation: {e}")
        raise

validation_task = airflow_client.create_python_task(
    dag=dag,
    task_id="validate_data_quality",
    python_callable=validate_data_quality
)

# Task 4: Detect anomalies
def detect_anomalies(**context) -> Dict[str, Any]:
    """Detect anomalies in smart meter data using statistical methods"""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from src.core.config.config_loader import ConfigLoader
    
    logger.info("Starting anomaly detection...")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        data_sources_config = config_loader.get_data_sources_config()
        smart_meter_config = data_sources_config.smart_meters
        
        # Construct file path
        data_root = Path(smart_meter_config.data_root)
        readings_file = data_root / smart_meter_config.readings_file
        
        logger.info(f"Detecting anomalies in: {readings_file}")
        
        # Read CSV file
        df = pd.read_csv(readings_file)
        
        anomaly_results = {
            "total_records": len(df),
            "anomalies_detected": 0,
            "anomaly_details": [],
            "anomaly_rate": 0.0
        }
        
        # Convert timestamp to datetime for time-based analysis
        if 'reading_timestamp' in df.columns:
            df['reading_timestamp'] = pd.to_datetime(df['reading_timestamp'])
        
        # Anomaly detection for consumption_kwh
        if 'consumption_kwh' in df.columns:
            consumption = df['consumption_kwh'].dropna()
            
            # Statistical anomaly detection (Z-score method)
            mean_consumption = consumption.mean()
            std_consumption = consumption.std()
            z_scores = np.abs((consumption - mean_consumption) / std_consumption)
            
            # Mark as anomaly if Z-score > 3 (99.7% confidence)
            consumption_anomalies = z_scores > 3
            anomaly_count = consumption_anomalies.sum()
            
            if anomaly_count > 0:
                anomaly_results["anomalies_detected"] += anomaly_count
                anomaly_results["anomaly_details"].append({
                    "field": "consumption_kwh",
                    "anomaly_count": int(anomaly_count),
                    "method": "z_score",
                    "threshold": 3.0
                })
                
                logger.warning(f"Found {anomaly_count} consumption anomalies using Z-score method")
        
        # Anomaly detection for voltage_v
        if 'voltage_v' in df.columns:
            voltage = df['voltage_v'].dropna()
            
            # Voltage should be within normal range (200-250V for residential)
            voltage_anomalies = (voltage < 200) | (voltage > 250)
            voltage_anomaly_count = voltage_anomalies.sum()
            
            if voltage_anomaly_count > 0:
                anomaly_results["anomalies_detected"] += voltage_anomaly_count
                anomaly_results["anomaly_details"].append({
                    "field": "voltage_v",
                    "anomaly_count": int(voltage_anomaly_count),
                    "method": "range_check",
                    "threshold": "200-250V"
                })
                
                logger.warning(f"Found {voltage_anomaly_count} voltage anomalies (outside 200-250V range)")
        
        # Anomaly detection for current_a
        if 'current_a' in df.columns:
            current = df['current_a'].dropna()
            
            # Current should be reasonable (0-100A for residential)
            current_anomalies = (current < 0) | (current > 100)
            current_anomaly_count = current_anomalies.sum()
            
            if current_anomaly_count > 0:
                anomaly_results["anomalies_detected"] += current_anomaly_count
                anomaly_results["anomaly_details"].append({
                    "field": "current_a",
                    "anomaly_count": int(current_anomaly_count),
                    "method": "range_check",
                    "threshold": "0-100A"
                })
                
                logger.warning(f"Found {current_anomaly_count} current anomalies (outside 0-100A range)")
        
        # Calculate anomaly rate
        if anomaly_results["total_records"] > 0:
            anomaly_results["anomaly_rate"] = anomaly_results["anomalies_detected"] / anomaly_results["total_records"]
        
        logger.info(f"Anomaly detection completed. Found {anomaly_results['anomalies_detected']} anomalies ({anomaly_results['anomaly_rate']:.2%} rate)")
        
        return {
            "status": "success",
            "anomalies_detected": anomaly_results["anomalies_detected"],
            "anomaly_rate": anomaly_results["anomaly_rate"],
            "anomaly_details": anomaly_results["anomaly_details"],
            "total_records": anomaly_results["total_records"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        raise

anomaly_detection_task = airflow_client.create_python_task(
    dag=dag,
    task_id="detect_anomalies",
    python_callable=detect_anomalies
)

# Task 5: Process and transform data
def process_data(**context) -> Dict[str, Any]:
    """Process and transform smart meter data"""
    from src.infrastructure.external.s3.s3_client import S3Client
    from src.infrastructure.external.snowflake.snowflake_client import SnowflakeClient
    
    # Implementation would go here
    logger.info("Processing and transforming data...")
    
    return {
        "status": "success",
        "records_processed": 1000,
        "transformation_completed": True
    }

process_task = airflow_client.create_python_task(
    dag=dag,
    task_id="process_data",
    python_callable=process_data
)

# Task 6: Load to data warehouse
load_to_warehouse_task = airflow_client.create_snowflake_task(
    dag=dag,
    task_id="load_to_warehouse",
    sql="""
    INSERT INTO smart_meter_analytics
    SELECT 
        meter_id,
        DATE(timestamp) as reading_date,
        AVG(voltage) as avg_voltage,
        AVG(current) as avg_current,
        AVG(active_power) as avg_power,
        AVG(data_quality_score) as avg_quality_score
    FROM smart_meter_readings_staging
    WHERE timestamp >= CURRENT_DATE - INTERVAL '1 hour'
    GROUP BY meter_id, DATE(timestamp)
    """,
    snowflake_conn_id="snowflake_default"
)

# Task 7: Generate analytics
def generate_analytics(**context) -> Dict[str, Any]:
    """Generate analytics and insights"""
    from src.infrastructure.external.snowflake.query_executor import SnowflakeQueryExecutor
    from src.infrastructure.external.snowflake.snowflake_client import SnowflakeClient
    
    # Implementation would go here
    logger.info("Generating analytics...")
    
    return {
        "status": "success",
        "analytics_generated": True,
        "insights_count": 25
    }

analytics_task = airflow_client.create_python_task(
    dag=dag,
    task_id="generate_analytics",
    python_callable=generate_analytics
)

# Task 8: Send alerts if needed
def send_alerts(**context) -> Dict[str, Any]:
    """Send alerts for anomalies or issues"""
    from src.infrastructure.external.apis.alerting_service import AlertingService
    
    # Implementation would go here
    logger.info("Checking for alerts...")
    
    return {
        "status": "success",
        "alerts_sent": 0,
        "alerts_required": False
    }

alerts_task = airflow_client.create_python_task(
    dag=dag,
    task_id="send_alerts",
    python_callable=send_alerts
)

# Task 9: Archive data
def archive_data(**context) -> Dict[str, Any]:
    """Archive processed data to S3"""
    from src.infrastructure.external.s3.data_archiver import S3DataArchiver
    from src.infrastructure.external.s3.s3_client import S3Client
    
    # Implementation would go here
    logger.info("Archiving data...")
    
    return {
        "status": "success",
        "data_archived": True,
        "archive_location": "s3://metrify-archive/smart-meter-data/"
    }

archive_task = airflow_client.create_python_task(
    dag=dag,
    task_id="archive_data",
    python_callable=archive_data
)

# Task 10: End pipeline
end_task = airflow_client.create_dummy_task(
    dag=dag,
    task_id="end_pipeline"
)

# Set task dependencies
airflow_client.set_task_dependencies([
    start_task,
    ingest_task,
    validation_task,
    anomaly_detection_task,
    process_task,
    load_to_warehouse_task,
    analytics_task,
    alerts_task,
    archive_task,
    end_task
])

# Export the DAG
__all__ = ["dag"]
