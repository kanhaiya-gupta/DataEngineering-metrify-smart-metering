"""
Airflow DAG for Smart Meter Data Pipeline
Orchestrates the complete data processing workflow
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
import logging

# Default arguments
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Create DAG
dag = DAG(
    'smart_meter_pipeline',
    default_args=default_args,
    description='Smart Meter Data Processing Pipeline',
    schedule_interval='@hourly',
    max_active_runs=1,
    tags=['smart-metering', 'energy', 'data-pipeline']
)

def extract_smart_meter_data(**context):
    """Extract smart meter data from various sources"""
    from ingestion.smart_meter_ingestion import SmartMeterIngestionPipeline
    
    pipeline = SmartMeterIngestionPipeline()
    
    # Process batch data for the previous hour
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    
    s3_path = f"s3a://metrify-smart-metering-data/raw/smart_meter_readings/{date_str}/"
    
    try:
        pipeline.process_batch_data(date_str, s3_path)
        logging.info(f"Successfully extracted smart meter data for {date_str}")
        return f"Extraction completed for {date_str}"
    except Exception as e:
        logging.error(f"Error extracting smart meter data: {e}")
        raise

def extract_grid_operator_data(**context):
    """Extract grid operator data"""
    from ingestion.grid_operator_ingestion import GridOperatorIngestionPipeline
    
    pipeline = GridOperatorIngestionPipeline()
    
    try:
        pipeline.process_all_operators()
        logging.info("Successfully extracted grid operator data")
        return "Grid operator data extraction completed"
    except Exception as e:
        logging.error(f"Error extracting grid operator data: {e}")
        raise

def extract_weather_data(**context):
    """Extract weather data"""
    from ingestion.weather_data_ingestion import WeatherDataIngestionPipeline
    
    pipeline = WeatherDataIngestionPipeline()
    
    try:
        pipeline.process_all_cities()
        logging.info("Successfully extracted weather data")
        return "Weather data extraction completed"
    except Exception as e:
        logging.error(f"Error extracting weather data: {e}")
        raise

def validate_data_quality(**context):
    """Validate data quality across all sources"""
    from data_quality.quality_checks import DataQualityValidator
    
    validator = DataQualityValidator()
    
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    
    try:
        # Run quality checks
        smart_meter_quality = validator.validate_smart_meter_data(date_str)
        grid_quality = validator.validate_grid_operator_data(date_str)
        weather_quality = validator.validate_weather_data(date_str)
        
        # Check if all quality checks pass
        if all([smart_meter_quality, grid_quality, weather_quality]):
            logging.info("All data quality checks passed")
            return "Data quality validation completed successfully"
        else:
            raise Exception("Data quality validation failed")
            
    except Exception as e:
        logging.error(f"Data quality validation error: {e}")
        raise

def trigger_dbt_run(**context):
    """Trigger dbt transformations"""
    from dbt_runner import DbtRunner
    
    runner = DbtRunner()
    
    try:
        # Run dbt models
        result = runner.run_models(
            models=['smart_meter_analytics', 'grid_analytics', 'weather_analytics'],
            full_refresh=False
        )
        
        if result.success:
            logging.info("dbt transformations completed successfully")
            return "dbt run completed"
        else:
            raise Exception("dbt run failed")
            
    except Exception as e:
        logging.error(f"dbt run error: {e}")
        raise

def update_data_lineage(**context):
    """Update data lineage and metadata"""
    from metadata.lineage_tracker import LineageTracker
    
    tracker = LineageTracker()
    
    try:
        execution_date = context['execution_date']
        date_str = execution_date.strftime('%Y-%m-%d')
        
        # Update lineage for processed data
        tracker.update_lineage(
            source_tables=['raw.smart_meter_readings', 'raw.grid_status', 'raw.weather_data'],
            target_tables=['analytics.smart_meter_analytics', 'analytics.grid_analytics', 'analytics.weather_analytics'],
            processing_date=date_str
        )
        
        logging.info("Data lineage updated successfully")
        return "Data lineage update completed"
        
    except Exception as e:
        logging.error(f"Data lineage update error: {e}")
        raise

# Define tasks
extract_smart_meter_task = PythonOperator(
    task_id='extract_smart_meter_data',
    python_callable=extract_smart_meter_data,
    dag=dag
)

extract_grid_operator_task = PythonOperator(
    task_id='extract_grid_operator_data',
    python_callable=extract_grid_operator_data,
    dag=dag
)

extract_weather_task = PythonOperator(
    task_id='extract_weather_data',
    python_callable=extract_weather_data,
    dag=dag
)

validate_quality_task = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag
)

run_dbt_task = PythonOperator(
    task_id='run_dbt_transformations',
    python_callable=trigger_dbt_run,
    dag=dag
)

update_lineage_task = PythonOperator(
    task_id='update_data_lineage',
    python_callable=update_data_lineage,
    dag=dag
)

# Create data quality monitoring task
data_quality_monitoring = BashOperator(
    task_id='data_quality_monitoring',
    bash_command='python monitoring/data_quality_monitor.py --date {{ ds }}',
    dag=dag
)

# Create performance monitoring task
performance_monitoring = BashOperator(
    task_id='performance_monitoring',
    bash_command='python monitoring/performance_monitor.py --date {{ ds }}',
    dag=dag
)

# Define task dependencies
[extract_smart_meter_task, extract_grid_operator_task, extract_weather_task] >> validate_quality_task
validate_quality_task >> run_dbt_task
run_dbt_task >> [update_lineage_task, data_quality_monitoring, performance_monitoring]
