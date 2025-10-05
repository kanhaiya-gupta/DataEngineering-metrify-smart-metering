"""
Apache Airflow Client Implementation
Handles DAG management and workflow orchestration
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import aiohttp
import asyncio

try:
    from airflow import DAG
    from airflow.providers.standard.operators.python import PythonOperator
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.providers.standard.operators.empty import EmptyOperator as DummyOperator
    from airflow.providers.standard.sensors.filesystem import FileSensor
    from airflow.models import Variable
    from airflow.sdk import TaskGroup
    from airflow.sdk.bases.hook import BaseHook
    from airflow.exceptions import AirflowException
    from datetime import datetime, timedelta
    
    # Optional provider imports (may not be installed)
    try:
        from airflow.sensors.sql import SqlSensor
    except ImportError:
        SqlSensor = None
    
    try:
        from airflow.providers.postgres.operators.postgres import PostgresOperator
    except ImportError:
        PostgresOperator = None
    
    try:
        from airflow.providers.amazon.aws.operators.s3 import S3FileTransformOperator
    except ImportError:
        S3FileTransformOperator = None
    
    try:
        from airflow.providers.snowflake.operators.snowflake import SnowflakeSqlApiOperator as SnowflakeOperator
    except ImportError:
        SnowflakeOperator = None
    
    try:
        from airflow.providers.apache.kafka.operators.produce import ProduceToTopicOperator
    except ImportError:
        ProduceToTopicOperator = None
    
    try:
        from airflow.providers.apache.kafka.sensors.kafka import KafkaTopicSensor
    except ImportError:
        KafkaTopicSensor = None
    
    AIRFLOW_AVAILABLE = True
except ImportError:
    DAG = None
    PythonOperator = None
    BashOperator = None
    DummyOperator = None
    FileSensor = None
    SqlSensor = None
    PostgresOperator = None
    S3FileTransformOperator = None
    SnowflakeOperator = None
    ProduceToTopicOperator = None
    KafkaTopicSensor = None
    Variable = None
    TaskGroup = None
    BaseHook = None
    AirflowException = None
    AIRFLOW_AVAILABLE = False

from ....core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class AirflowClient:
    """
    Apache Airflow Client for workflow orchestration
    
    Handles DAG creation, task management, and workflow execution
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:3200",
        username: str = "admin",
        password: str = "admin",
        default_args: Optional[Dict[str, Any]] = None,
        max_active_runs: int = 1,
        catchup: bool = False
    ):
        if not AIRFLOW_AVAILABLE:
            raise InfrastructureError("Apache Airflow not installed", service="airflow")
        
        # REST API configuration
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self._session = None
        
        # DAG creation configuration
        self.default_args = default_args or {
            'owner': 'metrify-data-team',
            'depends_on_past': False,
            'start_date': datetime(2024, 1, 1),
            'email_on_failure': True,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        }
        self.max_active_runs = max_active_runs
        self.catchup = catchup
        self._dags = {}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for Airflow REST API"""
        if self._session is None or self._session.closed:
            auth = aiohttp.BasicAuth(self.username, self.password)
            self._session = aiohttp.ClientSession(
                auth=auth,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session
    
    async def _close_session(self) -> None:
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup session"""
        await self._close_session()
    
    async def connect(self) -> None:
        """Connect to Airflow REST API"""
        try:
            # Test connection by getting session
            session = await self._get_session()
            # Test with a simple health check
            health_url = f"{self.base_url}/health"
            async with session.get(health_url) as response:
                if response.status == 200:
                    logger.info(f"Connected to Airflow at {self.base_url}")
                else:
                    logger.warning(f"Airflow health check returned status {response.status}")
        except Exception as e:
            logger.error(f"Failed to connect to Airflow: {str(e)}")
            raise InfrastructureError(f"Failed to connect to Airflow: {str(e)}", service="airflow")
    
    async def disconnect(self) -> None:
        """Disconnect from Airflow REST API"""
        try:
            await self._close_session()
            logger.info("Disconnected from Airflow")
        except Exception as e:
            logger.error(f"Error disconnecting from Airflow: {e}")
    
    def create_dag(
        self,
        dag_id: str,
        description: str,
        schedule_interval: str = "@daily",
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> DAG:
        """
        Create a new DAG
        
        Args:
            dag_id: Unique DAG identifier
            description: DAG description
            schedule_interval: Schedule interval (cron expression or preset)
            tags: Optional tags for the DAG
            **kwargs: Additional DAG parameters
            
        Returns:
            Created DAG object
        """
        try:
            # Remove duplicate parameters from kwargs to avoid conflicts
            dag_kwargs = kwargs.copy()
            duplicate_params = ['max_active_runs', 'catchup', 'schedule_interval', 'default_args', 'tags']
            for param in duplicate_params:
                if param in dag_kwargs:
                    del dag_kwargs[param]
            
            dag = DAG(
                dag_id=dag_id,
                description=description,
                default_args=self.default_args,
                schedule=schedule_interval,  # Changed from schedule_interval to schedule
                max_active_runs=self.max_active_runs,
                catchup=self.catchup,
                tags=tags or [],
                **dag_kwargs
            )
            
            self._dags[dag_id] = dag
            logger.info(f"Created DAG: {dag_id}")
            return dag
            
        except Exception as e:
            logger.error(f"Error creating DAG {dag_id}: {str(e)}")
            raise InfrastructureError(f"Failed to create DAG: {str(e)}", service="airflow")
    
    def create_python_task(
        self,
        dag: DAG,
        task_id: str,
        python_callable: callable,
        op_args: Optional[List] = None,
        op_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> PythonOperator:
        """
        Create a Python task
        
        Args:
            dag: DAG object
            task_id: Task identifier
            python_callable: Python function to execute
            op_args: Positional arguments for the function
            op_kwargs: Keyword arguments for the function
            **kwargs: Additional task parameters
            
        Returns:
            Created PythonOperator
        """
        try:
            task = PythonOperator(
                task_id=task_id,
                python_callable=python_callable,
                op_args=op_args or [],
                op_kwargs=op_kwargs or {},
                dag=dag,
                **kwargs
            )
            
            logger.debug(f"Created Python task: {task_id}")
            return task
            
        except Exception as e:
            logger.error(f"Error creating Python task {task_id}: {str(e)}")
            raise InfrastructureError(f"Failed to create Python task: {str(e)}", service="airflow")
    
    def create_bash_task(
        self,
        dag: DAG,
        task_id: str,
        bash_command: str,
        **kwargs
    ) -> BashOperator:
        """
        Create a Bash task
        
        Args:
            dag: DAG object
            task_id: Task identifier
            bash_command: Bash command to execute
            **kwargs: Additional task parameters
            
        Returns:
            Created BashOperator
        """
        try:
            task = BashOperator(
                task_id=task_id,
                bash_command=bash_command,
                dag=dag,
                **kwargs
            )
            
            logger.debug(f"Created Bash task: {task_id}")
            return task
            
        except Exception as e:
            logger.error(f"Error creating Bash task {task_id}: {str(e)}")
            raise InfrastructureError(f"Failed to create Bash task: {str(e)}", service="airflow")
    
    def create_postgres_task(
        self,
        dag: DAG,
        task_id: str,
        sql: str,
        postgres_conn_id: str = "postgres_default",
        **kwargs
    ) -> PostgresOperator:
        """
        Create a PostgreSQL task
        
        Args:
            dag: DAG object
            task_id: Task identifier
            sql: SQL query to execute
            postgres_conn_id: PostgreSQL connection ID
            **kwargs: Additional task parameters
            
        Returns:
            Created PostgresOperator
        """
        try:
            if PostgresOperator is None:
                raise InfrastructureError("PostgresOperator not available. Please install apache-airflow-providers-postgres", service="airflow")
            
            task = PostgresOperator(
                task_id=task_id,
                sql=sql,
                postgres_conn_id=postgres_conn_id,
                dag=dag,
                **kwargs
            )
            
            logger.debug(f"Created PostgreSQL task: {task_id}")
            return task
            
        except Exception as e:
            logger.error(f"Error creating PostgreSQL task {task_id}: {str(e)}")
            raise InfrastructureError(f"Failed to create PostgreSQL task: {str(e)}", service="airflow")
    
    def create_s3_transform_task(
        self,
        dag: DAG,
        task_id: str,
        source_s3_key: str,
        dest_s3_key: str,
        transform_script: str,
        **kwargs
    ) -> S3FileTransformOperator:
        """
        Create an S3 file transformation task
        
        Args:
            dag: DAG object
            task_id: Task identifier
            source_s3_key: Source S3 key
            dest_s3_key: Destination S3 key
            transform_script: Transformation script path
            **kwargs: Additional task parameters
            
        Returns:
            Created S3FileTransformOperator
        """
        try:
            if S3FileTransformOperator is None:
                raise InfrastructureError("S3FileTransformOperator not available. Please install apache-airflow-providers-amazon", service="airflow")
            
            task = S3FileTransformOperator(
                task_id=task_id,
                source_s3_key=source_s3_key,
                dest_s3_key=dest_s3_key,
                transform_script=transform_script,
                dag=dag,
                **kwargs
            )
            
            logger.debug(f"Created S3 transform task: {task_id}")
            return task
            
        except Exception as e:
            logger.error(f"Error creating S3 transform task {task_id}: {str(e)}")
            raise InfrastructureError(f"Failed to create S3 transform task: {str(e)}", service="airflow")
    
    def create_snowflake_task(
        self,
        dag: DAG,
        task_id: str,
        sql: str,
        snowflake_conn_id: str = "snowflake_default",
        **kwargs
    ) -> SnowflakeOperator:
        """
        Create a Snowflake task
        
        Args:
            dag: DAG object
            task_id: Task identifier
            sql: SQL query to execute
            snowflake_conn_id: Snowflake connection ID
            **kwargs: Additional task parameters
            
        Returns:
            Created SnowflakeOperator
        """
        try:
            if SnowflakeOperator is None:
                raise InfrastructureError("SnowflakeOperator not available. Please install apache-airflow-providers-snowflake", service="airflow")
            
            task = SnowflakeOperator(
                task_id=task_id,
                sql=sql,
                snowflake_conn_id=snowflake_conn_id,
                dag=dag,
                **kwargs
            )
            
            logger.debug(f"Created Snowflake task: {task_id}")
            return task
            
        except Exception as e:
            logger.error(f"Error creating Snowflake task {task_id}: {str(e)}")
            raise InfrastructureError(f"Failed to create Snowflake task: {str(e)}", service="airflow")
    
    def create_kafka_produce_task(
        self,
        dag: DAG,
        task_id: str,
        topic: str,
        message: str,
        kafka_conn_id: str = "kafka_default",
        **kwargs
    ) -> ProduceToTopicOperator:
        """
        Create a Kafka produce task
        
        Args:
            dag: DAG object
            task_id: Task identifier
            topic: Kafka topic
            message: Message to produce
            kafka_conn_id: Kafka connection ID
            **kwargs: Additional task parameters
            
        Returns:
            Created ProduceToTopicOperator
        """
        try:
            if ProduceToTopicOperator is None:
                raise InfrastructureError("ProduceToTopicOperator not available. Please install apache-airflow-providers-apache-kafka", service="airflow")
            
            task = ProduceToTopicOperator(
                task_id=task_id,
                topic=topic,
                message=message,
                kafka_conn_id=kafka_conn_id,
                dag=dag,
                **kwargs
            )
            
            logger.debug(f"Created Kafka produce task: {task_id}")
            return task
            
        except Exception as e:
            logger.error(f"Error creating Kafka produce task {task_id}: {str(e)}")
            raise InfrastructureError(f"Failed to create Kafka produce task: {str(e)}", service="airflow")
    
    def create_file_sensor(
        self,
        dag: DAG,
        task_id: str,
        filepath: str,
        **kwargs
    ) -> FileSensor:
        """
        Create a file sensor task
        
        Args:
            dag: DAG object
            task_id: Task identifier
            filepath: File path to monitor
            **kwargs: Additional task parameters
            
        Returns:
            Created FileSensor
        """
        try:
            task = FileSensor(
                task_id=task_id,
                filepath=filepath,
                dag=dag,
                **kwargs
            )
            
            logger.debug(f"Created file sensor task: {task_id}")
            return task
            
        except Exception as e:
            logger.error(f"Error creating file sensor task {task_id}: {str(e)}")
            raise InfrastructureError(f"Failed to create file sensor task: {str(e)}", service="airflow")
    
    def create_sql_sensor(
        self,
        dag: DAG,
        task_id: str,
        sql: str,
        conn_id: str = "postgres_default",
        **kwargs
    ) -> SqlSensor:
        """
        Create a SQL sensor task
        
        Args:
            dag: DAG object
            task_id: Task identifier
            sql: SQL query to execute
            conn_id: Database connection ID
            **kwargs: Additional task parameters
            
        Returns:
            Created SqlSensor
        """
        try:
            task = SqlSensor(
                task_id=task_id,
                sql=sql,
                conn_id=conn_id,
                dag=dag,
                **kwargs
            )
            
            logger.debug(f"Created SQL sensor task: {task_id}")
            return task
            
        except Exception as e:
            logger.error(f"Error creating SQL sensor task {task_id}: {str(e)}")
            raise InfrastructureError(f"Failed to create SQL sensor task: {str(e)}", service="airflow")
    
    def create_dummy_task(
        self,
        dag: DAG,
        task_id: str,
        **kwargs
    ) -> DummyOperator:
        """
        Create a dummy task
        
        Args:
            dag: DAG object
            task_id: Task identifier
            **kwargs: Additional task parameters
            
        Returns:
            Created DummyOperator
        """
        try:
            task = DummyOperator(
                task_id=task_id,
                dag=dag,
                **kwargs
            )
            
            logger.debug(f"Created dummy task: {task_id}")
            return task
            
        except Exception as e:
            logger.error(f"Error creating dummy task {task_id}: {str(e)}")
            raise InfrastructureError(f"Failed to create dummy task: {str(e)}", service="airflow")
    
    def set_task_dependencies(self, tasks: List[Any]) -> None:
        """
        Set task dependencies in sequence
        
        Args:
            tasks: List of tasks in dependency order
        """
        try:
            for i in range(len(tasks) - 1):
                tasks[i] >> tasks[i + 1]
            
            logger.debug(f"Set dependencies for {len(tasks)} tasks")
            
        except Exception as e:
            logger.error(f"Error setting task dependencies: {str(e)}")
            raise InfrastructureError(f"Failed to set task dependencies: {str(e)}", service="airflow")
    
    def get_dag(self, dag_id: str) -> Optional[DAG]:
        """Get a DAG by ID"""
        return self._dags.get(dag_id)
    
    def list_dags(self) -> List[str]:
        """List all DAG IDs"""
        return list(self._dags.keys())
    
    def get_dag_status(self, dag_id: str) -> Dict[str, Any]:
        """Get DAG status information"""
        dag = self.get_dag(dag_id)
        if not dag:
            return {"error": f"DAG {dag_id} not found"}
        
        return {
            "dag_id": dag_id,
            "description": dag.description,
            "schedule_interval": dag.schedule_interval,
            "max_active_runs": dag.max_active_runs,
            "catchup": dag.catchup,
            "tags": dag.tags,
            "task_count": len(dag.tasks)
        }
    
    async def health_check(self) -> bool:
        """Check if Airflow is healthy and accessible"""
        try:
            if not AIRFLOW_AVAILABLE:
                return False
            
            session = await self._get_session()
            
            # Check Airflow health endpoint
            health_url = f"{self.base_url}/health"
            async with session.get(health_url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.debug(f"Airflow health check successful: {health_data}")
                    return True
                else:
                    logger.warning(f"Airflow health check failed with status {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Airflow health check failed: {str(e)}")
            return False
    
    async def trigger_dag(self, dag_id: str, conf: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Trigger a DAG run"""
        try:
            session = await self._get_session()
            
            # Trigger DAG via REST API
            trigger_url = f"{self.base_url}/api/v1/dags/{dag_id}/dagRuns"
            payload = {
                "conf": conf or {},
                "dag_run_id": f"manual__{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            }
            
            logger.info(f"Triggering DAG {dag_id} via REST API")
            
            async with session.post(trigger_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully triggered DAG {dag_id}: {result}")
                    return {
                        "run_id": result.get("dag_run_id"),
                        "dag_id": dag_id,
                        "state": result.get("state", "queued"),
                        "conf": conf or {}
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to trigger DAG {dag_id}: {response.status} - {error_text}")
                    raise InfrastructureError(f"Failed to trigger DAG: {response.status} - {error_text}", service="airflow")
            
        except Exception as e:
            logger.error(f"Failed to trigger DAG {dag_id}: {str(e)}")
            raise InfrastructureError(f"Failed to trigger DAG: {str(e)}", service="airflow")
    
    async def get_dag_run_status(self, run_id: str) -> Dict[str, Any]:
        """Get status of a DAG run"""
        try:
            session = await self._get_session()
            
            # Get DAG run status via REST API
            # Note: This requires the DAG ID, but we only have run_id
            # We'll need to search for the DAG run first
            search_url = f"{self.base_url}/api/v1/dagRuns"
            params = {"dag_run_id": run_id}
            
            logger.debug(f"Getting status for DAG run {run_id}")
            
            async with session.get(search_url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("dag_runs"):
                        dag_run = result["dag_runs"][0]
                        return {
                            "run_id": dag_run.get("dag_run_id"),
                            "dag_id": dag_run.get("dag_id"),
                            "state": dag_run.get("state"),
                            "start_date": dag_run.get("start_date"),
                            "end_date": dag_run.get("end_date"),
                            "duration": dag_run.get("duration")
                        }
                    else:
                        raise InfrastructureError(f"DAG run {run_id} not found", service="airflow")
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get DAG run status {run_id}: {response.status} - {error_text}")
                    raise InfrastructureError(f"Failed to get DAG run status: {response.status} - {error_text}", service="airflow")
            
        except Exception as e:
            logger.error(f"Failed to get DAG run status {run_id}: {str(e)}")
            raise InfrastructureError(f"Failed to get DAG run status: {str(e)}", service="airflow")