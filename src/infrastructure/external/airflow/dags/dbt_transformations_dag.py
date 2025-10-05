"""
dbt Transformations DAG
Runs dbt transformations on Spark ETL processed data
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
import logging

# Default arguments for the DAG
default_args = {
    'owner': 'metrify',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email': ['data-engineering@metrify.com'],
    'catchup': False
}

# DAG definition
dag = DAG(
    'dbt_transformations',
    default_args=default_args,
    description='dbt transformations on Spark ETL processed data',
    schedule_interval='@hourly',
    max_active_runs=1,
    tags=['dbt', 'transformations', 'business_logic'],
    doc_md="""
    # dbt Transformations DAG
    
    This DAG runs dbt transformations on data processed by Spark ETL:
    
    1. **Staging Models**: Clean and standardize Spark ETL processed data
    2. **Marts Models**: Apply business logic and create dimensional models
    3. **Metrics Models**: Generate aggregated metrics and KPIs
    
    ## Data Flow
    - **Input**: Spark ETL processed data in Snowflake (PROCESSED schema)
    - **Output**: Business-ready data marts and metrics in Snowflake (STAGING, MARTS, METRICS schemas)
    
    ## Features
    - Data quality validation and testing
    - Business logic transformations
    - Dimensional modeling
    - Metrics calculation
    - Automated testing and documentation
    """,
)

def validate_dbt_results(**context):
    """Validate dbt transformation results"""
    logger = logging.getLogger(__name__)
    
    # Get dbt run results from XCom
    dbt_results = context['task_instance'].xcom_pull(task_ids='run_dbt_models')
    
    if not dbt_results:
        raise ValueError("No dbt results returned")
    
    logger.info(f"dbt transformation completed successfully")
    
    return {
        'status': 'completed',
        'models_processed': dbt_results.get('models_processed', 0),
        'tests_passed': dbt_results.get('tests_passed', 0),
        'validation_timestamp': datetime.utcnow().isoformat()
    }

# Task definitions
start_task = DummyOperator(
    task_id='start_dbt_transformations',
    dag=dag,
)

# Run dbt staging models
run_dbt_staging = DockerOperator(
    task_id='run_dbt_staging_models',
    image='metrify-dbt:latest',
    container_name='dbt-staging-{{ ds_nodash }}',
    api_version='auto',
    auto_remove=True,
    command=[
        'dbt', 'run', 
        '--models', 'tag:staging',
        '--vars', '{"run_date": "{{ ds }}"}'
    ],
    environment={
        'SNOWFLAKE_ACCOUNT': '{{ var.value.snowflake_account }}',
        'SNOWFLAKE_USER': '{{ var.value.snowflake_user }}',
        'SNOWFLAKE_PASSWORD': '{{ var.value.snowflake_password }}',
        'SNOWFLAKE_WAREHOUSE': '{{ var.value.snowflake_warehouse }}',
        'SNOWFLAKE_DATABASE': '{{ var.value.snowflake_database }}',
        'SNOWFLAKE_SCHEMA': 'PROCESSED',
        'DBT_ENVIRONMENT': 'production'
    },
    volumes=['/opt/airflow/dbt:/dbt'],
    working_dir='/dbt',
    dag=dag,
)

# Run dbt marts models
run_dbt_marts = DockerOperator(
    task_id='run_dbt_marts_models',
    image='metrify-dbt:latest',
    container_name='dbt-marts-{{ ds_nodash }}',
    api_version='auto',
    auto_remove=True,
    command=[
        'dbt', 'run', 
        '--models', 'tag:marts',
        '--vars', '{"run_date": "{{ ds }}"}'
    ],
    environment={
        'SNOWFLAKE_ACCOUNT': '{{ var.value.snowflake_account }}',
        'SNOWFLAKE_USER': '{{ var.value.snowflake_user }}',
        'SNOWFLAKE_PASSWORD': '{{ var.value.snowflake_password }}',
        'SNOWFLAKE_WAREHOUSE': '{{ var.value.snowflake_warehouse }}',
        'SNOWFLAKE_DATABASE': '{{ var.value.snowflake_database }}',
        'SNOWFLAKE_SCHEMA': 'STAGING',
        'DBT_ENVIRONMENT': 'production'
    },
    volumes=['/opt/airflow/dbt:/dbt'],
    working_dir='/dbt',
    dag=dag,
)

# Run dbt metrics models
run_dbt_metrics = DockerOperator(
    task_id='run_dbt_metrics_models',
    image='metrify-dbt:latest',
    container_name='dbt-metrics-{{ ds_nodash }}',
    api_version='auto',
    auto_remove=True,
    command=[
        'dbt', 'run', 
        '--models', 'tag:metrics',
        '--vars', '{"run_date": "{{ ds }}"}'
    ],
    environment={
        'SNOWFLAKE_ACCOUNT': '{{ var.value.snowflake_account }}',
        'SNOWFLAKE_USER': '{{ var.value.snowflake_user }}',
        'SNOWFLAKE_PASSWORD': '{{ var.value.snowflake_password }}',
        'SNOWFLAKE_WAREHOUSE': '{{ var.value.snowflake_warehouse }}',
        'SNOWFLAKE_DATABASE': '{{ var.value.snowflake_database }}',
        'SNOWFLAKE_SCHEMA': 'MARTS',
        'DBT_ENVIRONMENT': 'production'
    },
    volumes=['/opt/airflow/dbt:/dbt'],
    working_dir='/dbt',
    dag=dag,
)

# Run dbt tests
run_dbt_tests = DockerOperator(
    task_id='run_dbt_tests',
    image='metrify-dbt:latest',
    container_name='dbt-tests-{{ ds_nodash }}',
    api_version='auto',
    auto_remove=True,
    command=[
        'dbt', 'test',
        '--vars', '{"run_date": "{{ ds }}"}'
    ],
    environment={
        'SNOWFLAKE_ACCOUNT': '{{ var.value.snowflake_account }}',
        'SNOWFLAKE_USER': '{{ var.value.snowflake_user }}',
        'SNOWFLAKE_PASSWORD': '{{ var.value.snowflake_password }}',
        'SNOWFLAKE_WAREHOUSE': '{{ var.value.snowflake_warehouse }}',
        'SNOWFLAKE_DATABASE': '{{ var.value.snowflake_database }}',
        'DBT_ENVIRONMENT': 'production'
    },
    volumes=['/opt/airflow/dbt:/dbt'],
    working_dir='/dbt',
    dag=dag,
)

# Validate results
validate_task = PythonOperator(
    task_id='validate_dbt_results',
    python_callable=validate_dbt_results,
    dag=dag,
)

# Send completion notification
notify_task = PythonOperator(
    task_id='send_dbt_completion_notification',
    python_callable=lambda **context: {
        'status': 'success',
        'message': 'dbt transformations completed successfully',
        'timestamp': datetime.utcnow().isoformat()
    },
    dag=dag,
)

end_task = DummyOperator(
    task_id='end_dbt_transformations',
    dag=dag,
)

# Task dependencies
start_task >> run_dbt_staging >> run_dbt_marts >> run_dbt_metrics >> run_dbt_tests >> validate_task >> notify_task >> end_task
