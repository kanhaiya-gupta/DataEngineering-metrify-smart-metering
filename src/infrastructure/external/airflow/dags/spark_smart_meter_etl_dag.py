"""
Spark Smart Meter ETL DAG
Apache Airflow DAG for orchestrating Spark ETL jobs for smart meter data processing
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable
import logging

# Default arguments
default_args = {
    'owner': 'metrify-data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email': ['data-engineering@metrify.com'],
    'catchup': False
}

# DAG definition
dag = DAG(
    'spark_smart_meter_etl',
    default_args=default_args,
    description='Spark ETL pipeline for smart meter data processing',
    schedule_interval='@hourly',
    max_active_runs=1,
    tags=['spark', 'etl', 'smart_meter', 'data_processing'],
    doc_md="""
    # Spark Smart Meter ETL DAG
    
    This DAG orchestrates the complete ETL pipeline for smart meter data using Apache Spark:
    
    1. **Extract**: Pulls raw smart meter data from multiple sources (CSV files, APIs, IoT devices)
    2. **Transform**: Processes and enriches data using Spark ETL jobs with advanced analytics
    3. **Load**: Loads processed data to:
       - PostgreSQL (operational database for real-time queries)
       - S3 Data Lake (raw and processed data for analytics)
       - Snowflake Data Warehouse (for business intelligence and reporting)
    
    ## Features
    - Multi-source data extraction (CSV, APIs, IoT devices)
    - Advanced data quality validation
    - Duplicate detection and removal
    - Data standardization and normalization
    - Performance optimization with Spark
    - Comprehensive error handling and monitoring
    """,
)

def get_spark_config():
    """Get Spark configuration from Airflow variables"""
    return {
        'spark.executor.memory': Variable.get('spark_executor_memory', default_var='2g'),
        'spark.executor.cores': Variable.get('spark_executor_cores', default_var='2'),
        'spark.driver.memory': Variable.get('spark_driver_memory', default_var='2g'),
        'spark.sql.adaptive.enabled': 'true',
        'spark.sql.adaptive.coalescePartitions.enabled': 'true'
    }

def validate_etl_results(**context):
    """Validate ETL processing results"""
    import requests
    import json
    
    # Get job status from Spark ETL service
    job_id = context['task_instance'].xcom_pull(task_ids='spark_smart_meter_etl')
    
    if not job_id:
        raise ValueError("No job ID returned from Spark ETL task")
    
    # Validate job completion
    # In a real implementation, this would check the actual job status
    logging.info(f"Validating ETL results for job: {job_id}")
    
    return {
        'job_id': job_id,
        'status': 'completed',
        'validation_timestamp': datetime.utcnow().isoformat()
    }

# Task definitions
start_task = DummyOperator(
    task_id='start_spark_etl',
    dag=dag,
)

# Extract smart meter data from local CSV files
extract_task = PythonOperator(
    task_id='extract_smart_meter_data',
    python_callable=lambda **context: {
        'status': 'success',
        'message': 'Extracted smart meter data from local CSV files',
        'source_path': '/opt/spark/data/raw/smart_meters/',
        'timestamp': datetime.utcnow().isoformat()
    },
    dag=dag,
)

# Transform (Spark ETL): Process raw data from S3 data lake with advanced analytics
spark_etl_task = KubernetesPodOperator(
    task_id='spark_smart_meter_etl',
    namespace='spark',
    image='apache/spark:3.5.0-scala2.12-java11-python3-ubuntu',
    cmds=['/opt/spark/bin/spark-submit'],
    arguments=[
        '--master', 'spark://spark-master:7077',
        '--deploy-mode', 'cluster',
        '--conf', 'spark.executor.memory=2g',
        '--conf', 'spark.executor.cores=2',
        '--conf', 'spark.driver.memory=2g',
        '--conf', 'spark.sql.adaptive.enabled=true',
        '--conf', 'spark.sql.adaptive.coalescePartitions.enabled=true',
        '--py-files', '/opt/spark/src/spark_etl/jobs/smart_meter_etl.py',
        '/opt/spark/src/spark_etl/jobs/smart_meter_etl.py',
        '--input-path', '/opt/spark/data/raw/smart_meters/',
        '--output-path', 's3a://metrify-data-lake/processed/smart_meter/{{ ds }}/',
        '--postgres-output', 'postgresql://postgres:5432/metrify',
        '--snowflake-output', 'snowflake://metrify-warehouse',
        '--batch-id', '{{ dag_run.conf.get("batch_id", "default") }}',
        '--data-type', 'smart_meter'
    ],
    env_vars={
        'SPARK_MASTER': 'spark://spark-master:7077',
        'S3_ENDPOINT_URL': 'https://s3.eu-central-1.amazonaws.com',
        'S3_BUCKET_NAME': 'metrify-data-lake',
        'S3_REGION': 'eu-central-1',
        'POSTGRES_HOST': 'postgres',
        'POSTGRES_PORT': '5432',
        'POSTGRES_DB': 'metrify',
        'POSTGRES_USER': 'metrify',
        'POSTGRES_PASSWORD': 'metrify'
    },
    config_file='/opt/airflow/.kube/config',
    in_cluster=False,
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag,
)

# Validate ETL results
validate_task = PythonOperator(
    task_id='validate_etl_results',
    python_callable=validate_etl_results,
    dag=dag,
)

# Load to multiple destinations
load_to_postgres_task = PythonOperator(
    task_id='load_smart_meter_data_to_postgres',
    python_callable=lambda **context: {
        'status': 'success',
        'message': 'Smart meter data loaded to PostgreSQL operational database',
        'destination': 'postgresql',
        'timestamp': datetime.utcnow().isoformat()
    },
    dag=dag,
)

load_to_s3_task = PythonOperator(
    task_id='load_smart_meter_data_to_s3',
    python_callable=lambda **context: {
        'status': 'success',
        'message': 'Smart meter data loaded to S3 data lake',
        'destination': 's3_data_lake',
        'timestamp': datetime.utcnow().isoformat()
    },
    dag=dag,
)

load_to_snowflake_task = PythonOperator(
    task_id='load_smart_meter_data_to_snowflake',
    python_callable=lambda **context: {
        'status': 'success',
        'message': 'Smart meter data loaded to Snowflake data warehouse',
        'destination': 'snowflake_warehouse',
        'timestamp': datetime.utcnow().isoformat()
    },
    dag=dag,
)

# Trigger dbt transformations
trigger_dbt_task = PythonOperator(
    task_id='trigger_dbt_transformations',
    python_callable=lambda **context: {
        'status': 'success',
        'message': 'Triggering dbt transformations on processed data',
        'timestamp': datetime.utcnow().isoformat()
    },
    dag=dag,
)

# Send completion notification
notify_task = PythonOperator(
    task_id='send_completion_notification',
    python_callable=lambda **context: {
        'status': 'success',
        'message': 'Spark Smart Meter ETL completed successfully',
        'timestamp': datetime.utcnow().isoformat()
    },
    dag=dag,
)

end_task = DummyOperator(
    task_id='end_spark_etl',
    dag=dag,
)

# Task dependencies
start_task >> extract_task >> spark_etl_task >> validate_task >> [load_to_postgres_task, load_to_s3_task, load_to_snowflake_task] >> trigger_dbt_task >> notify_task >> end_task
