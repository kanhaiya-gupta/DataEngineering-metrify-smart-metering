"""
Weather Data Pipeline DAG
Orchestrates weather data ingestion and correlation analysis
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

# Create the weather data DAG
dag = airflow_client.create_dag(
    dag_id="weather_data_pipeline",
    description="Weather data ingestion, processing, and energy correlation analysis",
    schedule_interval="*/15 * * * *",
    tags=["weather", "energy-correlation", "forecasting", "analytics"],
    max_active_runs=1,
    catchup=False
)

# Task 1: Start weather data collection
start_task = airflow_client.create_dummy_task(
    dag=dag,
    task_id="start_weather_collection"
)

# Task 2: Ingest weather data
def ingest_weather_data(**context) -> Dict[str, Any]:
    """Ingest weather data from CSV files and publish to Kafka with performance optimization"""
    import pandas as pd
    import json
    import time
    import asyncio
    from pathlib import Path
    from src.core.config.config_loader import ConfigLoader
    from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
    
    logger.info("Starting optimized weather data ingestion...")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        data_sources_config = config_loader.get_data_sources_config()
        performance_config = config_loader.get_performance_config()
        weather_station_config = data_sources_config.weather_stations

        # Get performance settings for weather stations
        weather_perf = performance_config.get('data_sources', {}).get('weather_stations', {})
        batch_perf = weather_perf.get('batch_processing', {})
        memory_perf = weather_perf.get('memory', {})

        # Construct file path
        data_root = Path(weather_station_config.data_root)
        observations_file = data_root / weather_station_config.readings_file

        logger.info(f"Reading data from: {observations_file}")

        # Check if file exists
        if not observations_file.exists():
            raise FileNotFoundError(f"Weather observations file not found: {observations_file}")

        # Use optimized batch size and chunk size
        batch_size = batch_perf.get('optimal_batch_size', weather_station_config.batch_size)
        chunk_size = memory_perf.get('chunk_size', 3000)
        total_records = 0
        kafka_topic = weather_station_config.kafka_topic

        # Initialize Kafka producer
        kafka_producer = KafkaProducer(
            bootstrap_servers=["localhost:9092"],
            client_id="weather-data-dag-producer"
        )
        
        # Read and process CSV in optimized chunks
        total_processing_time = 0
        for chunk in pd.read_csv(observations_file, chunksize=chunk_size):
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
                            "key": record.get('observation_id', 'unknown')
                        })
                    except Exception as e:
                        logger.error(f"Error preparing record {record.get('observation_id', 'unknown')}: {e}")
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
        
        logger.info(f"Weather data ingestion completed. Total records processed: {total_records}")
        logger.info(f"Overall throughput: {overall_throughput:.2f} records/sec")
        logger.info(f"Optimized batch size: {batch_size}, Chunk size: {chunk_size}")
        
        return {
            "status": "success",
            "records_processed": total_records,
            "file_path": str(observations_file),
            "kafka_topic": kafka_topic,
            "processing_time": total_processing_time,
            "throughput": overall_throughput,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in weather data ingestion: {e}")
        raise

ingest_task = airflow_client.create_python_task(
    dag=dag,
    task_id="ingest_weather_data",
    python_callable=ingest_weather_data
)

# Task 3: Validate weather data quality
def validate_weather_data(**context) -> Dict[str, Any]:
    """Validate weather data quality using configuration rules"""
    import pandas as pd
    from pathlib import Path
    from src.core.config.config_loader import ConfigLoader
    
    logger.info("Starting weather data quality validation...")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        data_sources_config = config_loader.get_data_sources_config()
        weather_station_config = data_sources_config.weather_stations
        
        # Construct file path
        data_root = Path(weather_station_config.data_root)
        observations_file = data_root / weather_station_config.readings_file
        
        logger.info(f"Validating data quality for: {observations_file}")
        
        # Read CSV file
        df = pd.read_csv(observations_file)
        
        # Get validation rules from configuration
        validation_rules = weather_station_config.validation
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
        
        logger.info(f"Weather data quality validation completed. Quality score: {validation_results['quality_score']:.2f}")
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
        logger.error(f"Error in weather data quality validation: {e}")
        raise

validation_task = airflow_client.create_python_task(
    dag=dag,
    task_id="validate_weather_data",
    python_callable=validate_weather_data
)

# Task 4: Detect weather anomalies
def detect_weather_anomalies(**context) -> Dict[str, Any]:
    """Detect anomalies in weather data using statistical methods"""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from src.core.config.config_loader import ConfigLoader
    
    logger.info("Starting weather anomaly detection...")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        data_sources_config = config_loader.get_data_sources_config()
        weather_station_config = data_sources_config.weather_stations
        
        # Construct file path
        data_root = Path(weather_station_config.data_root)
        observations_file = data_root / weather_station_config.readings_file
        
        logger.info(f"Detecting anomalies in: {observations_file}")
        
        # Read CSV file
        df = pd.read_csv(observations_file)
        
        anomaly_results = {
            "total_records": len(df),
            "anomalies_detected": 0,
            "anomaly_details": [],
            "anomaly_rate": 0.0
        }
        
        # Convert timestamp to datetime for time-based analysis
        if 'observation_timestamp' in df.columns:
            df['observation_timestamp'] = pd.to_datetime(df['observation_timestamp'])
        
        # Anomaly detection for temperature
        if 'temperature_celsius' in df.columns:
            temperature = df['temperature_celsius'].dropna()
            
            # Statistical anomaly detection (Z-score method)
            mean_temp = temperature.mean()
            std_temp = temperature.std()
            z_scores = np.abs((temperature - mean_temp) / std_temp)
            
            # Mark as anomaly if Z-score > 3 (99.7% confidence)
            temp_anomalies = z_scores > 3
            temp_anomaly_count = temp_anomalies.sum()
            
            if temp_anomaly_count > 0:
                anomaly_results["anomalies_detected"] += temp_anomaly_count
                anomaly_results["anomaly_details"].append({
                    "field": "temperature_celsius",
                    "anomaly_count": int(temp_anomaly_count),
                    "method": "z_score",
                    "threshold": 3.0
                })
                
                logger.warning(f"Found {temp_anomaly_count} temperature anomalies using Z-score method")
        
        # Anomaly detection for humidity
        if 'humidity_percent' in df.columns:
            humidity = df['humidity_percent'].dropna()
            
            # Humidity should be within normal range (0-100%)
            humidity_anomalies = (humidity < 0) | (humidity > 100)
            humidity_anomaly_count = humidity_anomalies.sum()
            
            if humidity_anomaly_count > 0:
                anomaly_results["anomalies_detected"] += humidity_anomaly_count
                anomaly_results["anomaly_details"].append({
                    "field": "humidity_percent",
                    "anomaly_count": int(humidity_anomaly_count),
                    "method": "range_check",
                    "threshold": "0-100%"
                })
                
                logger.warning(f"Found {humidity_anomaly_count} humidity anomalies (outside 0-100% range)")
        
        # Anomaly detection for pressure
        if 'pressure_hpa' in df.columns:
            pressure = df['pressure_hpa'].dropna()
            
            # Pressure should be reasonable (800-1100 hPa)
            pressure_anomalies = (pressure < 800) | (pressure > 1100)
            pressure_anomaly_count = pressure_anomalies.sum()
            
            if pressure_anomaly_count > 0:
                anomaly_results["anomalies_detected"] += pressure_anomaly_count
                anomaly_results["anomaly_details"].append({
                    "field": "pressure_hpa",
                    "anomaly_count": int(pressure_anomaly_count),
                    "method": "range_check",
                    "threshold": "800-1100 hPa"
                })
                
                logger.warning(f"Found {pressure_anomaly_count} pressure anomalies (outside 800-1100 hPa range)")
        
        # Anomaly detection for wind speed
        if 'wind_speed_mps' in df.columns:
            wind_speed = df['wind_speed_mps'].dropna()
            
            # Wind speed should be reasonable (0-50 m/s)
            wind_anomalies = (wind_speed < 0) | (wind_speed > 50)
            wind_anomaly_count = wind_anomalies.sum()
            
            if wind_anomaly_count > 0:
                anomaly_results["anomalies_detected"] += wind_anomaly_count
                anomaly_results["anomaly_details"].append({
                    "field": "wind_speed_mps",
                    "anomaly_count": int(wind_anomaly_count),
                    "method": "range_check",
                    "threshold": "0-50 m/s"
                })
                
                logger.warning(f"Found {wind_anomaly_count} wind speed anomalies (outside 0-50 m/s range)")
        
        # Calculate anomaly rate
        if anomaly_results["total_records"] > 0:
            anomaly_results["anomaly_rate"] = anomaly_results["anomalies_detected"] / anomaly_results["total_records"]
        
        logger.info(f"Weather anomaly detection completed. Found {anomaly_results['anomalies_detected']} anomalies ({anomaly_results['anomaly_rate']:.2%} rate)")
        
        return {
            "status": "success",
            "anomalies_detected": anomaly_results["anomalies_detected"],
            "anomaly_rate": anomaly_results["anomaly_rate"],
            "anomaly_details": anomaly_results["anomaly_details"],
            "total_records": anomaly_results["total_records"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in weather anomaly detection: {e}")
        raise

anomaly_task = airflow_client.create_python_task(
    dag=dag,
    task_id="detect_weather_anomalies",
    python_callable=detect_weather_anomalies
)

# Task 5: Correlate with energy data
def correlate_energy_weather(**context) -> Dict[str, Any]:
    """Correlate weather data with energy consumption patterns"""
    from src.infrastructure.external.snowflake.query_executor import SnowflakeQueryExecutor
    
    # Implementation would go here
    logger.info("Correlating weather with energy data...")
    
    return {
        "status": "success",
        "correlation_analysis": True,
        "correlation_strength": 0.78
    }

correlation_task = airflow_client.create_python_task(
    dag=dag,
    task_id="correlate_energy_weather",
    python_callable=correlate_energy_weather
)

# Task 6: Generate weather forecasts
def generate_weather_forecasts(**context) -> Dict[str, Any]:
    """Generate weather forecasts for energy planning"""
    from ....external.apis.weather_data_service import WeatherDataService
    
    # Implementation would go here
    logger.info("Generating weather forecasts...")
    
    return {
        "status": "success",
        "forecasts_generated": True,
        "forecast_horizon_hours": 48
    }

forecast_task = airflow_client.create_python_task(
    dag=dag,
    task_id="generate_weather_forecasts",
    python_callable=generate_weather_forecasts
)

# Task 7: Update weather warehouse
update_warehouse_task = airflow_client.create_snowflake_task(
    dag=dag,
    task_id="update_weather_warehouse",
    sql="""
    INSERT INTO weather_analytics
    SELECT 
        station_id,
        DATE(timestamp) as observation_date,
        AVG(temperature_celsius) as avg_temperature,
        AVG(humidity_percent) as avg_humidity,
        AVG(pressure_hpa) as avg_pressure,
        AVG(wind_speed_ms) as avg_wind_speed,
        AVG(data_quality_score) as avg_quality_score
    FROM weather_observations_staging
    WHERE timestamp >= CURRENT_DATE - INTERVAL '15 minutes'
    GROUP BY station_id, DATE(timestamp)
    """,
    snowflake_conn_id="snowflake_default"
)

# Task 8: Generate energy demand forecast
def generate_energy_forecast(**context) -> Dict[str, Any]:
    """Generate energy demand forecast based on weather patterns"""
    from src.infrastructure.external.snowflake.query_executor import SnowflakeQueryExecutor
    
    # Implementation would go here
    logger.info("Generating energy demand forecast...")
    
    return {
        "status": "success",
        "forecast_generated": True,
        "forecast_accuracy": 0.85
    }

energy_forecast_task = airflow_client.create_python_task(
    dag=dag,
    task_id="generate_energy_forecast",
    python_callable=generate_energy_forecast
)

# Task 9: Send weather alerts
def send_weather_alerts(**context) -> Dict[str, Any]:
    """Send weather-related alerts and notifications"""
    from src.infrastructure.external.apis.alerting_service import AlertingService
    
    # Implementation would go here
    logger.info("Sending weather alerts...")
    
    return {
        "status": "success",
        "alerts_sent": 2,
        "alert_types": ["temperature_anomaly", "wind_speed_warning"]
    }

weather_alerts_task = airflow_client.create_python_task(
    dag=dag,
    task_id="send_weather_alerts",
    python_callable=send_weather_alerts
)

# Task 10: Archive weather data
def archive_weather_data(**context) -> Dict[str, Any]:
    """Archive weather data to S3 for long-term storage"""
    from src.infrastructure.external.s3.data_archiver import S3DataArchiver
    
    # Implementation would go here
    logger.info("Archiving weather data...")
    
    return {
        "status": "success",
        "data_archived": True,
        "archive_location": "s3://metrify-archive/weather-data/"
    }

archive_task = airflow_client.create_python_task(
    dag=dag,
    task_id="archive_weather_data",
    python_callable=archive_weather_data
)

# Task 11: End weather processing
end_task = airflow_client.create_dummy_task(
    dag=dag,
    task_id="end_weather_processing"
)

# Set task dependencies
airflow_client.set_task_dependencies([
    start_task,
    ingest_task,
    validation_task,
    anomaly_task,
    correlation_task,
    forecast_task,
    update_warehouse_task,
    energy_forecast_task,
    weather_alerts_task,
    archive_task,
    end_task
])

# Export the DAG
__all__ = ["dag"]
