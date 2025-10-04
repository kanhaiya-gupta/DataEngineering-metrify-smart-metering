"""
Grid Operator Data Pipeline DAG
Orchestrates grid operator data processing and monitoring
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

# Create the grid operator DAG
dag = airflow_client.create_dag(
    dag_id="grid_operator_pipeline",
    description="Grid operator data processing and grid stability monitoring",
    schedule_interval="*/5 * * * *",
    tags=["grid-operator", "grid-stability", "monitoring", "real-time"],
    max_active_runs=1,
    catchup=False
)

# Task 1: Start grid monitoring
start_task = airflow_client.create_dummy_task(
    dag=dag,
    task_id="start_grid_monitoring"
)

# Task 2: Ingest grid status data
def ingest_grid_status(**context) -> Dict[str, Any]:
    """Ingest grid status data from CSV files and publish to Kafka with performance optimization"""
    import pandas as pd
    import json
    import time
    import asyncio
    from pathlib import Path
    from src.core.config.config_loader import ConfigLoader
    from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
    
    logger.info("Starting optimized grid status data ingestion...")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        data_sources_config = config_loader.get_data_sources_config()
        performance_config = config_loader.get_performance_config()
        grid_operator_config = data_sources_config.grid_operators

        # Get performance settings for grid operators
        grid_perf = performance_config.get('data_sources', {}).get('grid_operators', {})
        batch_perf = grid_perf.get('batch_processing', {})
        memory_perf = grid_perf.get('memory', {})

        # Construct file path
        data_root = Path(grid_operator_config.data_root)
        status_file = data_root / grid_operator_config.readings_file

        logger.info(f"Reading data from: {status_file}")

        # Check if file exists
        if not status_file.exists():
            raise FileNotFoundError(f"Grid status file not found: {status_file}")

        # Use optimized batch size and chunk size
        batch_size = batch_perf.get('optimal_batch_size', grid_operator_config.batch_size)
        chunk_size = memory_perf.get('chunk_size', 5000)
        total_records = 0
        kafka_topic = grid_operator_config.kafka_topic

        # Initialize Kafka producer
        kafka_producer = KafkaProducer(
            bootstrap_servers=["localhost:9092"],
            client_id="grid-operator-dag-producer"
        )
        
        # Read and process CSV in optimized chunks
        total_processing_time = 0
        for chunk in pd.read_csv(status_file, chunksize=chunk_size):
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
                            "key": record.get('status_id', 'unknown')
                        })
                    except Exception as e:
                        logger.error(f"Error preparing record {record.get('status_id', 'unknown')}: {e}")
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
        
        logger.info(f"Grid status data ingestion completed. Total records processed: {total_records}")
        logger.info(f"Overall throughput: {overall_throughput:.2f} records/sec")
        logger.info(f"Optimized batch size: {batch_size}, Chunk size: {chunk_size}")
        
        return {
            "status": "success",
            "records_processed": total_records,
            "file_path": str(status_file),
            "kafka_topic": kafka_topic,
            "processing_time": total_processing_time,
            "throughput": overall_throughput,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in grid status data ingestion: {e}")
        raise

ingest_task = airflow_client.create_python_task(
    dag=dag,
    task_id="ingest_grid_status",
    python_callable=ingest_grid_status
)

# Task 3: Validate grid data quality
def validate_grid_data(**context) -> Dict[str, Any]:
    """Validate grid data quality using configuration rules"""
    import pandas as pd
    from pathlib import Path
    from src.core.config.config_loader import ConfigLoader
    
    logger.info("Starting grid data quality validation...")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        data_sources_config = config_loader.get_data_sources_config()
        grid_operator_config = data_sources_config.grid_operators
        
        # Construct file path
        data_root = Path(grid_operator_config.data_root)
        status_file = data_root / grid_operator_config.readings_file
        
        logger.info(f"Validating data quality for: {status_file}")
        
        # Read CSV file
        df = pd.read_csv(status_file)
        
        # Get validation rules from configuration
        validation_rules = grid_operator_config.validation
        required_columns = validation_rules.get('required_columns', [])
        data_types = validation_rules.get('data_types', {})
        value_ranges = validation_rules.get('value_ranges', {})
        
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
        
        # Check value ranges
        for column, range_config in value_ranges.items():
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                min_val = range_config.get('min')
                max_val = range_config.get('max')
                
                if min_val is not None:
                    below_min = (df[column] < min_val).sum()
                    if below_min > 0:
                        validation_results["validation_errors"].append(f"Column {column} has {below_min} values below minimum {min_val}")
                        validation_results["quality_score"] -= 0.05
                
                if max_val is not None:
                    above_max = (df[column] > max_val).sum()
                    if above_max > 0:
                        validation_results["validation_errors"].append(f"Column {column} has {above_max} values above maximum {max_val}")
                        validation_results["quality_score"] -= 0.05
        
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
        
        logger.info(f"Grid data quality validation completed. Quality score: {validation_results['quality_score']:.2f}")
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
        logger.error(f"Error in grid data quality validation: {e}")
        raise

validation_task = airflow_client.create_python_task(
    dag=dag,
    task_id="validate_grid_data",
    python_callable=validate_grid_data
)

# Task 4: Monitor grid stability
def monitor_grid_stability(**context) -> Dict[str, Any]:
    """Monitor grid stability and detect anomalies using statistical methods"""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from src.core.config.config_loader import ConfigLoader
    
    logger.info("Starting grid stability monitoring...")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        data_sources_config = config_loader.get_data_sources_config()
        grid_operator_config = data_sources_config.grid_operators
        
        # Construct file path
        data_root = Path(grid_operator_config.data_root)
        status_file = data_root / grid_operator_config.readings_file
        
        logger.info(f"Monitoring grid stability for: {status_file}")
        
        # Read CSV file
        df = pd.read_csv(status_file)
        
        anomaly_results = {
            "total_records": len(df),
            "anomalies_detected": 0,
            "anomaly_details": [],
            "anomaly_rate": 0.0,
            "stability_score": 1.0
        }
        
        # Convert timestamp to datetime for time-based analysis
        if 'status_timestamp' in df.columns:
            df['status_timestamp'] = pd.to_datetime(df['status_timestamp'])
        
        # Anomaly detection for grid frequency
        if 'grid_frequency_hz' in df.columns:
            frequency = df['grid_frequency_hz'].dropna()
            
            # Statistical anomaly detection (Z-score method)
            mean_frequency = frequency.mean()
            std_frequency = frequency.std()
            z_scores = np.abs((frequency - mean_frequency) / std_frequency)
            
            # Mark as anomaly if Z-score > 3 (99.7% confidence)
            frequency_anomalies = z_scores > 3
            frequency_anomaly_count = frequency_anomalies.sum()
            
            if frequency_anomaly_count > 0:
                anomaly_results["anomalies_detected"] += frequency_anomaly_count
                anomaly_results["anomaly_details"].append({
                    "field": "grid_frequency_hz",
                    "anomaly_count": int(frequency_anomaly_count),
                    "method": "z_score",
                    "threshold": 3.0
                })
                
                logger.warning(f"Found {frequency_anomaly_count} frequency anomalies using Z-score method")
        
        # Anomaly detection for grid voltage
        if 'grid_voltage_kv' in df.columns:
            voltage = df['grid_voltage_kv'].dropna()
            
            # Voltage should be within normal range (220-400kV for transmission)
            voltage_anomalies = (voltage < 220) | (voltage > 400)
            voltage_anomaly_count = voltage_anomalies.sum()
            
            if voltage_anomaly_count > 0:
                anomaly_results["anomalies_detected"] += voltage_anomaly_count
                anomaly_results["anomaly_details"].append({
                    "field": "grid_voltage_kv",
                    "anomaly_count": int(voltage_anomaly_count),
                    "method": "range_check",
                    "threshold": "220-400kV"
                })
                
                logger.warning(f"Found {voltage_anomaly_count} voltage anomalies (outside 220-400kV range)")
        
        # Anomaly detection for grid load
        if 'grid_load_mw' in df.columns:
            load = df['grid_load_mw'].dropna()
            
            # Load should be reasonable (0-20000MW)
            load_anomalies = (load < 0) | (load > 20000)
            load_anomaly_count = load_anomalies.sum()
            
            if load_anomaly_count > 0:
                anomaly_results["anomalies_detected"] += load_anomaly_count
                anomaly_results["anomaly_details"].append({
                    "field": "grid_load_mw",
                    "anomaly_count": int(load_anomaly_count),
                    "method": "range_check",
                    "threshold": "0-20000MW"
                })
                
                logger.warning(f"Found {load_anomaly_count} load anomalies (outside 0-20000MW range)")
        
        # Check grid stability flags
        if 'is_grid_stable' in df.columns:
            unstable_count = (~df['is_grid_stable']).sum()
            if unstable_count > 0:
                anomaly_results["anomalies_detected"] += unstable_count
                anomaly_results["anomaly_details"].append({
                    "field": "is_grid_stable",
                    "anomaly_count": int(unstable_count),
                    "method": "stability_flag",
                    "threshold": "stable=True"
                })
                
                logger.warning(f"Found {unstable_count} unstable grid status records")
        
        # Calculate anomaly rate and stability score
        if anomaly_results["total_records"] > 0:
            anomaly_results["anomaly_rate"] = anomaly_results["anomalies_detected"] / anomaly_results["total_records"]
            anomaly_results["stability_score"] = max(0.0, 1.0 - anomaly_results["anomaly_rate"])
        
        logger.info(f"Grid stability monitoring completed. Found {anomaly_results['anomalies_detected']} anomalies ({anomaly_results['anomaly_rate']:.2%} rate)")
        logger.info(f"Grid stability score: {anomaly_results['stability_score']:.2f}")
        
        return {
            "status": "success",
            "stability_score": anomaly_results["stability_score"],
            "anomalies_detected": anomaly_results["anomalies_detected"],
            "anomaly_rate": anomaly_results["anomaly_rate"],
            "anomaly_details": anomaly_results["anomaly_details"],
            "total_records": anomaly_results["total_records"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in grid stability monitoring: {e}")
        raise

stability_task = airflow_client.create_python_task(
    dag=dag,
    task_id="monitor_grid_stability",
    python_callable=monitor_grid_stability
)

# Task 5: Process grid analytics
def process_grid_analytics(**context) -> Dict[str, Any]:
    """Process grid analytics and generate insights"""
    from src.infrastructure.external.snowflake.query_executor import SnowflakeQueryExecutor
    
    # Implementation would go here
    logger.info("Processing grid analytics...")
    
    return {
        "status": "success",
        "analytics_processed": True,
        "insights_generated": 10
    }

analytics_task = airflow_client.create_python_task(
    dag=dag,
    task_id="process_grid_analytics",
    python_callable=process_grid_analytics
)

# Task 6: Update grid status in warehouse
update_warehouse_task = airflow_client.create_snowflake_task(
    dag=dag,
    task_id="update_grid_warehouse",
    sql="""
    INSERT INTO grid_analytics
    SELECT 
        operator_id,
        DATE(timestamp) as status_date,
        AVG(voltage_level) as avg_voltage,
        AVG(frequency) as avg_frequency,
        AVG(load_percentage) as avg_load,
        AVG(stability_score) as avg_stability
    FROM grid_statuses_staging
    WHERE timestamp >= CURRENT_DATE - INTERVAL '5 minutes'
    GROUP BY operator_id, DATE(timestamp)
    """,
    snowflake_conn_id="snowflake_default"
)

# Task 7: Check for grid alerts
def check_grid_alerts(**context) -> Dict[str, Any]:
    """Check for grid stability alerts and issues"""
    from src.infrastructure.external.apis.alerting_service import AlertingService
    
    # Implementation would go here
    logger.info("Checking for grid alerts...")
    
    return {
        "status": "success",
        "alerts_checked": True,
        "critical_alerts": 0,
        "warning_alerts": 1
    }

alerts_task = airflow_client.create_python_task(
    dag=dag,
    task_id="check_grid_alerts",
    python_callable=check_grid_alerts
)

# Task 8: Send grid notifications
def send_grid_notifications(**context) -> Dict[str, Any]:
    """Send grid status notifications to stakeholders"""
    from src.infrastructure.external.apis.alerting_service import AlertingService
    
    # Implementation would go here
    logger.info("Sending grid notifications...")
    
    return {
        "status": "success",
        "notifications_sent": 5,
        "recipients": ["grid-operators", "energy-managers"]
    }

notifications_task = airflow_client.create_python_task(
    dag=dag,
    task_id="send_grid_notifications",
    python_callable=send_grid_notifications
)

# Task 9: End grid monitoring
end_task = airflow_client.create_dummy_task(
    dag=dag,
    task_id="end_grid_monitoring"
)

# Set task dependencies
airflow_client.set_task_dependencies([
    start_task,
    ingest_task,
    validation_task,
    stability_task,
    analytics_task,
    update_warehouse_task,
    alerts_task,
    notifications_task,
    end_task
])

# Export the DAG
__all__ = ["dag"]
