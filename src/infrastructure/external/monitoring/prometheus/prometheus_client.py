"""
Prometheus Client Implementation
Handles metrics collection and export for Prometheus
"""

import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
from enum import Enum

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, push_to_gateway
    )
    from prometheus_client.core import REGISTRY
except ImportError:
    Counter = None
    Histogram = None
    Gauge = None
    Summary = None
    Info = None
    CollectorRegistry = None
    generate_latest = None
    CONTENT_TYPE_LATEST = None
    start_http_server = None
    push_to_gateway = None
    REGISTRY = None

from src.core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Prometheus metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


class PrometheusClient:
    """
    Prometheus Client for metrics collection and export
    
    Handles all types of Prometheus metrics including counters,
    gauges, histograms, and summaries for comprehensive monitoring.
    """
    
    def __init__(
        self,
        namespace: str = "metrify",
        subsystem: str = "smart_metering",
        pushgateway_url: Optional[str] = None,
        job_name: str = "metrify-smart-metering"
    ):
        if Counter is None:
            raise InfrastructureError("Prometheus client not installed", service="prometheus")
        
        self.namespace = namespace
        self.subsystem = subsystem
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.registry = CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._http_server_started = False
        self._metrics_initialized = False
    
    def _initialize_core_metrics(self) -> None:
        """Initialize core system metrics"""
        if self._metrics_initialized:
            return
            
        try:
            # System metrics
            self._metrics["system_info"] = Info(
                "system_info",
                "System information",
                registry=self.registry
            )
            
            # Smart meter metrics
            self._metrics["meter_readings_total"] = Counter(
                "meter_readings_total",
                "Total number of meter readings processed",
                ["meter_id", "status"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["meter_quality_score"] = Gauge(
                "meter_quality_score",
                "Meter data quality score",
                ["meter_id"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["meter_anomalies_total"] = Counter(
                "meter_anomalies_total",
                "Total number of meter anomalies detected",
                ["meter_id", "anomaly_type"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            # Grid operator metrics
            self._metrics["grid_status_updates_total"] = Counter(
                "grid_status_updates_total",
                "Total number of grid status updates",
                ["operator_id", "status"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["grid_stability_score"] = Gauge(
                "grid_stability_score",
                "Grid stability score",
                ["operator_id"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["grid_load_percentage"] = Gauge(
                "grid_load_percentage",
                "Grid load percentage",
                ["operator_id"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            # Weather station metrics
            self._metrics["weather_observations_total"] = Counter(
                "weather_observations_total",
                "Total number of weather observations",
                ["station_id", "status"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["weather_temperature"] = Gauge(
                "weather_temperature",
                "Weather temperature in Celsius",
                ["station_id"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["weather_humidity"] = Gauge(
                "weather_humidity",
                "Weather humidity percentage",
                ["station_id"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            # Data quality metrics
            self._metrics["data_quality_score"] = Gauge(
                "data_quality_score",
                "Data quality score",
                ["entity_type", "entity_id"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["data_quality_issues_total"] = Counter(
                "data_quality_issues_total",
                "Total number of data quality issues",
                ["entity_type", "entity_id", "issue_type"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            # Performance metrics
            self._metrics["operation_duration_seconds"] = Histogram(
                "operation_duration_seconds",
                "Operation duration in seconds",
                ["operation", "status"],
                buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["operation_requests_total"] = Counter(
                "operation_requests_total",
                "Total number of operation requests",
                ["operation", "status"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            # Kafka metrics
            self._metrics["kafka_messages_produced_total"] = Counter(
                "kafka_messages_produced_total",
                "Total number of Kafka messages produced",
                ["topic", "status"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["kafka_messages_consumed_total"] = Counter(
                "kafka_messages_consumed_total",
                "Total number of Kafka messages consumed",
                ["topic", "consumer_group"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["kafka_consumer_lag"] = Gauge(
                "kafka_consumer_lag",
                "Kafka consumer lag",
                ["topic", "consumer_group"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            # Airflow metrics
            self._metrics["airflow_dag_runs_total"] = Counter(
                "airflow_dag_runs_total",
                "Total number of Airflow DAG runs",
                ["dag_id", "status"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["airflow_task_runs_total"] = Counter(
                "airflow_task_runs_total",
                "Total number of Airflow task runs",
                ["dag_id", "task_id", "status"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["airflow_dag_duration_seconds"] = Histogram(
                "airflow_dag_duration_seconds",
                "Airflow DAG duration in seconds",
                ["dag_id"],
                buckets=[60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            # Database metrics
            self._metrics["database_connections_active"] = Gauge(
                "database_connections_active",
                "Number of active database connections",
                ["database"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["database_queries_total"] = Counter(
                "database_queries_total",
                "Total number of database queries",
                ["database", "operation", "status"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["database_query_duration_seconds"] = Histogram(
                "database_query_duration_seconds",
                "Database query duration in seconds",
                ["database", "operation"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            # S3 metrics
            self._metrics["s3_operations_total"] = Counter(
                "s3_operations_total",
                "Total number of S3 operations",
                ["operation", "bucket", "status"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["s3_operation_duration_seconds"] = Histogram(
                "s3_operation_duration_seconds",
                "S3 operation duration in seconds",
                ["operation", "bucket"],
                buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            # Snowflake metrics
            self._metrics["snowflake_queries_total"] = Counter(
                "snowflake_queries_total",
                "Total number of Snowflake queries",
                ["warehouse", "status"],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics["snowflake_query_duration_seconds"] = Histogram(
                "snowflake_query_duration_seconds",
                "Snowflake query duration in seconds",
                ["warehouse"],
                buckets=[1, 5, 10, 30, 60, 300, 600, 1800, 3600],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry
            )
            
            self._metrics_initialized = True
            logger.info("Core Prometheus metrics initialized")
            
        except Exception as e:
            logger.error(f"Error initializing core metrics: {str(e)}")
            raise InfrastructureError(f"Failed to initialize core metrics: {str(e)}", service="prometheus")
    
    async def initialize(self) -> None:
        """Initialize the Prometheus client"""
        self._initialize_core_metrics()
        logger.info("Prometheus client initialized")
    
    def start_http_server(self, port: int = 8000, addr: str = "0.0.0.0") -> None:
        """Start HTTP server for metrics endpoint"""
        try:
            if not self._http_server_started:
                start_http_server(port, addr, registry=self.registry)
                self._http_server_started = True
                logger.info(f"Prometheus metrics server started on {addr}:{port}")
        except Exception as e:
            logger.error(f"Error starting HTTP server: {str(e)}")
            raise InfrastructureError(f"Failed to start HTTP server: {str(e)}", service="prometheus")
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            raise InfrastructureError(f"Failed to get metrics: {str(e)}", service="prometheus")
    
    def push_metrics(self) -> None:
        """Push metrics to Pushgateway"""
        if not self.pushgateway_url:
            logger.warning("Pushgateway URL not configured, skipping push")
            return
        
        try:
            push_to_gateway(
                self.pushgateway_url,
                job=self.job_name,
                registry=self.registry
            )
            logger.debug("Metrics pushed to Pushgateway")
        except Exception as e:
            logger.error(f"Error pushing metrics: {str(e)}")
            raise InfrastructureError(f"Failed to push metrics: {str(e)}", service="prometheus")
    
    # Smart Meter Metrics
    def increment_meter_readings(self, meter_id: str, status: str = "success") -> None:
        """Increment meter readings counter"""
        try:
            self._metrics["meter_readings_total"].labels(
                meter_id=meter_id,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Error incrementing meter readings: {str(e)}")
    
    def set_meter_quality_score(self, meter_id: str, score: float) -> None:
        """Set meter quality score"""
        try:
            self._metrics["meter_quality_score"].labels(meter_id=meter_id).set(score)
        except Exception as e:
            logger.error(f"Error setting meter quality score: {str(e)}")
    
    def increment_meter_anomalies(self, meter_id: str, anomaly_type: str) -> None:
        """Increment meter anomalies counter"""
        try:
            self._metrics["meter_anomalies_total"].labels(
                meter_id=meter_id,
                anomaly_type=anomaly_type
            ).inc()
        except Exception as e:
            logger.error(f"Error incrementing meter anomalies: {str(e)}")
    
    # Grid Operator Metrics
    def increment_grid_status_updates(self, operator_id: str, status: str = "success") -> None:
        """Increment grid status updates counter"""
        try:
            self._metrics["grid_status_updates_total"].labels(
                operator_id=operator_id,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Error incrementing grid status updates: {str(e)}")
    
    def set_grid_stability_score(self, operator_id: str, score: float) -> None:
        """Set grid stability score"""
        try:
            self._metrics["grid_stability_score"].labels(operator_id=operator_id).set(score)
        except Exception as e:
            logger.error(f"Error setting grid stability score: {str(e)}")
    
    def set_grid_load_percentage(self, operator_id: str, percentage: float) -> None:
        """Set grid load percentage"""
        try:
            self._metrics["grid_load_percentage"].labels(operator_id=operator_id).set(percentage)
        except Exception as e:
            logger.error(f"Error setting grid load percentage: {str(e)}")
    
    # Weather Station Metrics
    def increment_weather_observations(self, station_id: str, status: str = "success") -> None:
        """Increment weather observations counter"""
        try:
            self._metrics["weather_observations_total"].labels(
                station_id=station_id,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Error incrementing weather observations: {str(e)}")
    
    def set_weather_temperature(self, station_id: str, temperature: float) -> None:
        """Set weather temperature"""
        try:
            self._metrics["weather_temperature"].labels(station_id=station_id).set(temperature)
        except Exception as e:
            logger.error(f"Error setting weather temperature: {str(e)}")
    
    def set_weather_humidity(self, station_id: str, humidity: float) -> None:
        """Set weather humidity"""
        try:
            self._metrics["weather_humidity"].labels(station_id=station_id).set(humidity)
        except Exception as e:
            logger.error(f"Error setting weather humidity: {str(e)}")
    
    # Data Quality Metrics
    def set_data_quality_score(self, entity_type: str, entity_id: str, score: float) -> None:
        """Set data quality score"""
        try:
            self._metrics["data_quality_score"].labels(
                entity_type=entity_type,
                entity_id=entity_id
            ).set(score)
        except Exception as e:
            logger.error(f"Error setting data quality score: {str(e)}")
    
    def increment_data_quality_issues(self, entity_type: str, entity_id: str, issue_type: str) -> None:
        """Increment data quality issues counter"""
        try:
            self._metrics["data_quality_issues_total"].labels(
                entity_type=entity_type,
                entity_id=entity_id,
                issue_type=issue_type
            ).inc()
        except Exception as e:
            logger.error(f"Error incrementing data quality issues: {str(e)}")
    
    # Performance Metrics
    def record_operation_duration(self, operation: str, duration: float, status: str = "success") -> None:
        """Record operation duration"""
        try:
            self._metrics["operation_duration_seconds"].labels(
                operation=operation,
                status=status
            ).observe(duration)
        except Exception as e:
            logger.error(f"Error recording operation duration: {str(e)}")
    
    def increment_operation_requests(self, operation: str, status: str = "success") -> None:
        """Increment operation requests counter"""
        try:
            self._metrics["operation_requests_total"].labels(
                operation=operation,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Error incrementing operation requests: {str(e)}")
    
    # Kafka Metrics
    def increment_kafka_messages_produced(self, topic: str, status: str = "success") -> None:
        """Increment Kafka messages produced counter"""
        try:
            self._metrics["kafka_messages_produced_total"].labels(
                topic=topic,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Error incrementing Kafka messages produced: {str(e)}")
    
    def increment_kafka_messages_consumed(self, topic: str, consumer_group: str) -> None:
        """Increment Kafka messages consumed counter"""
        try:
            self._metrics["kafka_messages_consumed_total"].labels(
                topic=topic,
                consumer_group=consumer_group
            ).inc()
        except Exception as e:
            logger.error(f"Error incrementing Kafka messages consumed: {str(e)}")
    
    def set_kafka_consumer_lag(self, topic: str, consumer_group: str, lag: int) -> None:
        """Set Kafka consumer lag"""
        try:
            self._metrics["kafka_consumer_lag"].labels(
                topic=topic,
                consumer_group=consumer_group
            ).set(lag)
        except Exception as e:
            logger.error(f"Error setting Kafka consumer lag: {str(e)}")
    
    # Airflow Metrics
    def increment_airflow_dag_runs(self, dag_id: str, status: str) -> None:
        """Increment Airflow DAG runs counter"""
        try:
            self._metrics["airflow_dag_runs_total"].labels(
                dag_id=dag_id,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Error incrementing Airflow DAG runs: {str(e)}")
    
    def increment_airflow_task_runs(self, dag_id: str, task_id: str, status: str) -> None:
        """Increment Airflow task runs counter"""
        try:
            self._metrics["airflow_task_runs_total"].labels(
                dag_id=dag_id,
                task_id=task_id,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Error incrementing Airflow task runs: {str(e)}")
    
    def record_airflow_dag_duration(self, dag_id: str, duration: float) -> None:
        """Record Airflow DAG duration"""
        try:
            self._metrics["airflow_dag_duration_seconds"].labels(dag_id=dag_id).observe(duration)
        except Exception as e:
            logger.error(f"Error recording Airflow DAG duration: {str(e)}")
    
    # Database Metrics
    def set_database_connections_active(self, database: str, count: int) -> None:
        """Set active database connections"""
        try:
            self._metrics["database_connections_active"].labels(database=database).set(count)
        except Exception as e:
            logger.error(f"Error setting database connections: {str(e)}")
    
    def increment_database_queries(self, database: str, operation: str, status: str = "success") -> None:
        """Increment database queries counter"""
        try:
            self._metrics["database_queries_total"].labels(
                database=database,
                operation=operation,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Error incrementing database queries: {str(e)}")
    
    def record_database_query_duration(self, database: str, operation: str, duration: float) -> None:
        """Record database query duration"""
        try:
            self._metrics["database_query_duration_seconds"].labels(
                database=database,
                operation=operation
            ).observe(duration)
        except Exception as e:
            logger.error(f"Error recording database query duration: {str(e)}")
    
    # S3 Metrics
    def increment_s3_operations(self, operation: str, bucket: str, status: str = "success") -> None:
        """Increment S3 operations counter"""
        try:
            self._metrics["s3_operations_total"].labels(
                operation=operation,
                bucket=bucket,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Error incrementing S3 operations: {str(e)}")
    
    def record_s3_operation_duration(self, operation: str, bucket: str, duration: float) -> None:
        """Record S3 operation duration"""
        try:
            self._metrics["s3_operation_duration_seconds"].labels(
                operation=operation,
                bucket=bucket
            ).observe(duration)
        except Exception as e:
            logger.error(f"Error recording S3 operation duration: {str(e)}")
    
    # Snowflake Metrics
    def increment_snowflake_queries(self, warehouse: str, status: str = "success") -> None:
        """Increment Snowflake queries counter"""
        try:
            self._metrics["snowflake_queries_total"].labels(
                warehouse=warehouse,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Error incrementing Snowflake queries: {str(e)}")
    
    def record_snowflake_query_duration(self, warehouse: str, duration: float) -> None:
        """Record Snowflake query duration"""
        try:
            self._metrics["snowflake_query_duration_seconds"].labels(warehouse=warehouse).observe(duration)
        except Exception as e:
            logger.error(f"Error recording Snowflake query duration: {str(e)}")
    
    def set_system_info(self, version: str, environment: str, region: str) -> None:
        """Set system information"""
        try:
            self._metrics["system_info"].info({
                "version": version,
                "environment": environment,
                "region": region,
                "start_time": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Error setting system info: {str(e)}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "namespace": self.namespace,
            "subsystem": self.subsystem,
            "job_name": self.job_name,
            "pushgateway_url": self.pushgateway_url,
            "http_server_started": self._http_server_started,
            "metrics_count": len(self._metrics)
        }
