"""
Spark ETL Service
Service for running Spark ETL jobs from the main API
"""

import logging
import subprocess
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from src.infrastructure.external.airflow.spark_airflow_integration import get_spark_airflow_integration, SparkJobConfig

logger = logging.getLogger(__name__)


class SparkETLService:
    """Service for managing Spark ETL jobs"""
    
    def __init__(self):
        self.active_jobs = {}
        self.job_history = []
        self.airflow_integration = get_spark_airflow_integration()
    
    async def submit_job(
        self,
        data_type: str,
        source_path: str,
        target_path: str,
        output_format: str = "delta",
        batch_id: Optional[str] = None,
        spark_config: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Submit a Spark ETL job via Airflow
        
        Args:
            data_type: Type of data to process
            source_path: Source data path
            target_path: Target data path
            output_format: Output format (delta, parquet)
            batch_id: Optional batch ID
            spark_config: Optional Spark configuration
        
        Returns:
            Job submission result
        """
        try:
            # Create Spark job configuration
            job_config = SparkJobConfig(
                data_type=data_type,
                input_path=source_path,
                output_path=target_path,
                batch_id=batch_id,
                spark_config=spark_config,
                format=output_format
            )
            
            # Submit job via Airflow integration
            result = await self.airflow_integration.submit_spark_etl_job(job_config)
            
            # Generate unique job ID for tracking
            job_id = f"spark_etl_{data_type}_{uuid.uuid4().hex[:8]}"
            
            # Create job record
            job_record = {
                "job_id": job_id,
                "dag_id": result["dag_id"],
                "run_id": result["run_id"],
                "data_type": data_type,
                "source_path": source_path,
                "target_path": target_path,
                "output_format": output_format,
                "batch_id": result["batch_id"],
                "status": "submitted",
                "submitted_at": result["submitted_at"],
                "progress": 0.0,
                "records_processed": 0,
                "records_total": 0,
                "error_message": None
            }
            
            # Store job record
            self.active_jobs[job_id] = job_record
            self.job_history.append(job_record)
            
            logger.info(f"Spark ETL job submitted via Airflow: {job_id}, DAG: {result['dag_id']}, Run: {result['run_id']}")
            
            return {
                "job_id": job_id,
                "dag_id": result["dag_id"],
                "run_id": result["run_id"],
                "status": "submitted",
                "message": result["message"],
                "submitted_at": result["submitted_at"]
            }
            
        except Exception as e:
            logger.error(f"Failed to submit Spark ETL job: {str(e)}")
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a Spark ETL job
        
        Args:
            job_id: Job ID to check
        
        Returns:
            Job status information
        """
        try:
            if job_id not in self.active_jobs:
                # Check job history
                for job in self.job_history:
                    if job["job_id"] == job_id:
                        return job
                
                return {
                    "job_id": job_id,
                    "status": "not_found",
                    "message": "Job not found"
                }
            
            job_record = self.active_jobs[job_id]
            
            # Get real-time status from Airflow if we have DAG info
            if "dag_id" in job_record and "run_id" in job_record:
                try:
                    airflow_status = await self.airflow_integration.get_spark_job_status(
                        job_record["dag_id"], job_record["run_id"]
                    )
                    
                    # Update job record with real-time status
                    job_record.update({
                        "status": airflow_status["dag_run_state"],
                        "progress": airflow_status["overall_progress"],
                        "spark_task_status": airflow_status["spark_task_status"],
                        "duration": airflow_status["duration"],
                        "last_updated": datetime.utcnow().isoformat()
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to get Airflow status for job {job_id}: {str(e)}")
            
            return job_record
            
        except Exception as e:
            logger.error(f"Failed to get job status: {str(e)}")
            raise
    
    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running Spark ETL job
        
        Args:
            job_id: Job ID to cancel
        
        Returns:
            Cancellation result
        """
        try:
            if job_id not in self.active_jobs:
                return {
                    "job_id": job_id,
                    "status": "not_found",
                    "message": "Job not found"
                }
            
            job_record = self.active_jobs[job_id]
            
            if job_record["status"] in ["completed", "failed", "cancelled"]:
                return {
                    "job_id": job_id,
                    "status": "already_finished",
                    "message": f"Job is already {job_record['status']}"
                }
            
            # Cancel job via Airflow if we have DAG info
            if "dag_id" in job_record and "run_id" in job_record:
                try:
                    cancel_result = await self.airflow_integration.cancel_spark_job(
                        job_record["dag_id"], job_record["run_id"]
                    )
                    
                    # Update job status
                    job_record["status"] = "cancelled"
                    job_record["cancelled_at"] = cancel_result["cancelled_at"]
                    
                    logger.info(f"Spark ETL job cancelled via Airflow: {job_id}")
                    
                    return {
                        "job_id": job_id,
                        "status": "cancelled",
                        "message": cancel_result["message"],
                        "cancelled_at": cancel_result["cancelled_at"]
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to cancel job via Airflow: {str(e)}")
                    # Fall back to local cancellation
                    job_record["status"] = "cancelled"
                    job_record["cancelled_at"] = datetime.utcnow().isoformat()
                    
                    return {
                        "job_id": job_id,
                        "status": "cancelled",
                        "message": "Job cancelled locally (Airflow cancellation failed)",
                        "cancelled_at": job_record["cancelled_at"]
                    }
            else:
                # Local cancellation for jobs without Airflow integration
                job_record["status"] = "cancelled"
                job_record["cancelled_at"] = datetime.utcnow().isoformat()
                
                logger.info(f"Spark ETL job cancelled locally: {job_id}")
                
                return {
                    "job_id": job_id,
                    "status": "cancelled",
                    "message": "Job cancelled successfully",
                    "cancelled_at": job_record["cancelled_at"]
                }
            
        except Exception as e:
            logger.error(f"Failed to cancel job: {str(e)}")
            raise
    
    async def list_jobs(self, status_filter: Optional[str] = None, data_type: Optional[str] = None) -> Dict[str, Any]:
        """
        List Spark ETL jobs
        
        Args:
            status_filter: Optional status filter
            data_type: Optional data type filter
        
        Returns:
            List of jobs
        """
        try:
            # Get jobs from Airflow integration for real-time status
            airflow_jobs = await self.airflow_integration.list_spark_jobs(data_type, status_filter)
            
            # Combine with local job records
            local_jobs = list(self.active_jobs.values())
            
            # Merge and deduplicate jobs
            all_jobs = []
            airflow_job_ids = set()
            
            # Add Airflow jobs
            for job in airflow_jobs["jobs"]:
                airflow_job_ids.add(job["run_id"])
                all_jobs.append({
                    "source": "airflow",
                    "dag_id": job["dag_id"],
                    "run_id": job["run_id"],
                    "data_type": job["data_type"],
                    "status": job["status"],
                    "start_date": job["start_date"],
                    "end_date": job["end_date"],
                    "conf": job["conf"]
                })
            
            # Add local jobs not in Airflow
            for job in local_jobs:
                if job.get("run_id") not in airflow_job_ids:
                    all_jobs.append({
                        "source": "local",
                        "job_id": job["job_id"],
                        "data_type": job["data_type"],
                        "status": job["status"],
                        "submitted_at": job["submitted_at"],
                        "progress": job["progress"]
                    })
            
            # Apply filters
            if status_filter:
                all_jobs = [job for job in all_jobs if job["status"] == status_filter]
            
            if data_type:
                all_jobs = [job for job in all_jobs if job["data_type"] == data_type]
            
            return {
                "jobs": all_jobs,
                "total_count": len(all_jobs),
                "status_filter": status_filter,
                "data_type_filter": data_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {str(e)}")
            raise
    
    async def _simulate_job_processing(self, job_id: str):
        """
        Simulate job processing (for demonstration)
        In real implementation, this would monitor actual Spark job
        """
        try:
            if job_id not in self.active_jobs:
                return
            
            job_record = self.active_jobs[job_id]
            
            # Simulate processing stages
            stages = [
                {"progress": 25, "status": "extracting", "message": "Extracting data from source"},
                {"progress": 50, "status": "validating", "message": "Validating data quality"},
                {"progress": 75, "status": "transforming", "message": "Transforming data"},
                {"progress": 100, "status": "completed", "message": "Job completed successfully"}
            ]
            
            for stage in stages:
                await asyncio.sleep(2)  # Simulate processing time
                
                if job_id in self.active_jobs:
                    job_record["progress"] = stage["progress"]
                    job_record["status"] = stage["status"]
                    job_record["message"] = stage["message"]
                    
                    if stage["status"] == "completed":
                        job_record["completed_at"] = datetime.utcnow().isoformat()
                        job_record["records_processed"] = 10000  # Simulated
                        job_record["records_total"] = 10000
                        break
            
        except Exception as e:
            logger.error(f"Error in job simulation: {str(e)}")
            if job_id in self.active_jobs:
                self.active_jobs[job_id]["status"] = "failed"
                self.active_jobs[job_id]["error_message"] = str(e)


# Global service instance
spark_etl_service = SparkETLService()
