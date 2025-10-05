"""
Spark Airflow Integration
Integration service for submitting Spark ETL jobs via Airflow
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import requests
from dataclasses import dataclass

from src.infrastructure.external.airflow.airflow_client import AirflowClient

logger = logging.getLogger(__name__)


@dataclass
class SparkJobConfig:
    """Configuration for Spark ETL job submission"""
    data_type: str
    input_path: str
    output_path: str
    batch_id: Optional[str] = None
    spark_config: Optional[Dict[str, str]] = None
    format: str = "delta"
    executor_memory: str = "2g"
    executor_cores: str = "2"
    driver_memory: str = "2g"


class SparkAirflowIntegration:
    """Service for integrating Spark ETL jobs with Airflow orchestration"""
    
    def __init__(self, airflow_client: AirflowClient):
        self.airflow_client = airflow_client
        self.spark_dag_mapping = {
            "smart_meter": "spark_smart_meter_etl",
            "grid_operator": "spark_grid_operator_etl",
            "weather_station": "spark_weather_station_etl"
        }
    
    async def submit_spark_etl_job(self, job_config: SparkJobConfig) -> Dict[str, Any]:
        """
        Submit a Spark ETL job via Airflow
        
        Args:
            job_config: Spark job configuration
            
        Returns:
            Job submission result
        """
        try:
            logger.info(f"Submitting Spark ETL job for {job_config.data_type}")
            
            # Validate data type
            if job_config.data_type not in self.spark_dag_mapping:
                raise ValueError(f"Invalid data_type: {job_config.data_type}. Must be one of: {list(self.spark_dag_mapping.keys())}")
            
            dag_id = self.spark_dag_mapping[job_config.data_type]
            
            # Prepare DAG run configuration
            dag_run_conf = {
                "batch_id": job_config.batch_id or f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "input_path": job_config.input_path,
                "output_path": job_config.output_path,
                "format": job_config.format,
                "spark_config": job_config.spark_config or {},
                "executor_memory": job_config.executor_memory,
                "executor_cores": job_config.executor_cores,
                "driver_memory": job_config.driver_memory,
                "processing_mode": "spark_etl",
                "submitted_at": datetime.utcnow().isoformat()
            }
            
            # Trigger Airflow DAG
            dag_run = await self.airflow_client.trigger_dag(
                dag_id=dag_id,
                conf=dag_run_conf
            )
            
            logger.info(f"Spark ETL job submitted successfully. DAG: {dag_id}, Run ID: {dag_run.run_id}")
            
            return {
                "status": "submitted",
                "dag_id": dag_id,
                "run_id": dag_run.run_id,
                "data_type": job_config.data_type,
                "batch_id": dag_run_conf["batch_id"],
                "submitted_at": dag_run_conf["submitted_at"],
                "message": f"Spark ETL job submitted successfully. Check Airflow UI for progress."
            }
            
        except Exception as e:
            logger.error(f"Failed to submit Spark ETL job: {str(e)}")
            raise
    
    async def get_spark_job_status(self, dag_id: str, run_id: str) -> Dict[str, Any]:
        """
        Get status of a Spark ETL job
        
        Args:
            dag_id: Airflow DAG ID
            run_id: Airflow DAG run ID
            
        Returns:
            Job status information
        """
        try:
            logger.info(f"Getting Spark ETL job status. DAG: {dag_id}, Run: {run_id}")
            
            # Get DAG run status from Airflow
            dag_run = await self.airflow_client.get_dag_run(dag_id, run_id)
            
            # Get task instances for the DAG run
            task_instances = await self.airflow_client.get_task_instances(dag_id, run_id)
            
            # Find the Spark ETL task
            spark_task = None
            for task in task_instances:
                if "spark" in task.task_id.lower() and "etl" in task.task_id.lower():
                    spark_task = task
                    break
            
            # Compile status information
            status_info = {
                "dag_id": dag_id,
                "run_id": run_id,
                "dag_run_state": dag_run.state,
                "start_date": dag_run.start_date.isoformat() if dag_run.start_date else None,
                "end_date": dag_run.end_date.isoformat() if dag_run.end_date else None,
                "duration": None,
                "spark_task_status": None,
                "spark_task_logs": None,
                "overall_progress": 0.0
            }
            
            # Calculate duration
            if dag_run.start_date and dag_run.end_date:
                duration = dag_run.end_date - dag_run.start_date
                status_info["duration"] = str(duration)
            elif dag_run.start_date:
                duration = datetime.utcnow() - dag_run.start_date
                status_info["duration"] = str(duration)
            
            # Get Spark task details
            if spark_task:
                status_info["spark_task_status"] = spark_task.state
                status_info["spark_task_logs"] = await self.airflow_client.get_task_logs(
                    dag_id, run_id, spark_task.task_id
                )
                
                # Calculate progress based on task states
                total_tasks = len(task_instances)
                completed_tasks = len([t for t in task_instances if t.state in ["success", "failed", "skipped"]])
                status_info["overall_progress"] = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get Spark ETL job status: {str(e)}")
            raise
    
    async def cancel_spark_job(self, dag_id: str, run_id: str) -> Dict[str, Any]:
        """
        Cancel a running Spark ETL job
        
        Args:
            dag_id: Airflow DAG ID
            run_id: Airflow DAG run ID
            
        Returns:
            Cancellation result
        """
        try:
            logger.info(f"Cancelling Spark ETL job. DAG: {dag_id}, Run: {run_id}")
            
            # Cancel the DAG run
            result = await self.airflow_client.cancel_dag_run(dag_id, run_id)
            
            return {
                "status": "cancelled",
                "dag_id": dag_id,
                "run_id": run_id,
                "cancelled_at": datetime.utcnow().isoformat(),
                "message": "Spark ETL job cancelled successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to cancel Spark ETL job: {str(e)}")
            raise
    
    async def list_spark_jobs(self, data_type: Optional[str] = None, status_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        List Spark ETL jobs
        
        Args:
            data_type: Filter by data type
            status_filter: Filter by status
            
        Returns:
            List of Spark ETL jobs
        """
        try:
            logger.info(f"Listing Spark ETL jobs. Data type: {data_type}, Status: {status_filter}")
            
            jobs = []
            
            # Get DAGs to check
            dags_to_check = []
            if data_type:
                if data_type in self.spark_dag_mapping:
                    dags_to_check = [self.spark_dag_mapping[data_type]]
                else:
                    raise ValueError(f"Invalid data_type: {data_type}")
            else:
                dags_to_check = list(self.spark_dag_mapping.values())
            
            # Get DAG runs for each Spark ETL DAG
            for dag_id in dags_to_check:
                try:
                    dag_runs = await self.airflow_client.get_dag_runs(dag_id, limit=10)
                    
                    for dag_run in dag_runs:
                        # Apply status filter if specified
                        if status_filter and dag_run.state != status_filter:
                            continue
                        
                        job_info = {
                            "dag_id": dag_id,
                            "run_id": dag_run.run_id,
                            "data_type": self._get_data_type_from_dag_id(dag_id),
                            "status": dag_run.state,
                            "start_date": dag_run.start_date.isoformat() if dag_run.start_date else None,
                            "end_date": dag_run.end_date.isoformat() if dag_run.end_date else None,
                            "conf": dag_run.conf
                        }
                        
                        jobs.append(job_info)
                        
                except Exception as e:
                    logger.warning(f"Failed to get DAG runs for {dag_id}: {str(e)}")
                    continue
            
            # Sort by start date (most recent first)
            jobs.sort(key=lambda x: x["start_date"] or "", reverse=True)
            
            return {
                "jobs": jobs,
                "total_count": len(jobs),
                "data_type_filter": data_type,
                "status_filter": status_filter,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to list Spark ETL jobs: {str(e)}")
            raise
    
    def _get_data_type_from_dag_id(self, dag_id: str) -> str:
        """Get data type from DAG ID"""
        for data_type, dag_id_mapping in self.spark_dag_mapping.items():
            if dag_id == dag_id_mapping:
                return data_type
        return "unknown"
    
    async def get_spark_cluster_status(self) -> Dict[str, Any]:
        """
        Get Spark cluster status
        
        Returns:
            Spark cluster status information
        """
        try:
            logger.info("Getting Spark cluster status")
            
            # In a real implementation, this would connect to the Spark master
            # For now, we'll return a mock status
            cluster_status = {
                "master_url": "spark://spark-master:7077",
                "status": "running",
                "workers": 3,
                "total_cores": 6,
                "total_memory": "12g",
                "active_applications": 0,
                "completed_applications": 0,
                "failed_applications": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return cluster_status
            
        except Exception as e:
            logger.error(f"Failed to get Spark cluster status: {str(e)}")
            raise


# Global integration instance
spark_airflow_integration = None


def get_spark_airflow_integration() -> SparkAirflowIntegration:
    """Get global Spark Airflow integration instance"""
    global spark_airflow_integration
    if spark_airflow_integration is None:
        airflow_client = AirflowClient()
        spark_airflow_integration = SparkAirflowIntegration(airflow_client)
    return spark_airflow_integration
