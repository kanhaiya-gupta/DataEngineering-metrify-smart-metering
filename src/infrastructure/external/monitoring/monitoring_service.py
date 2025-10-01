"""
Comprehensive Monitoring Service
Orchestrates all monitoring components for complete observability
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from .prometheus.prometheus_client import PrometheusClient
from .grafana.grafana_client import GrafanaClient
from .jaeger.jaeger_client import JaegerClient
from .datadog.datadog_client import DataDogClient
from ...core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class MonitoringService:
    """
    Comprehensive Monitoring Service
    
    Orchestrates all monitoring components including Prometheus,
    Grafana, Jaeger, and DataDog for complete observability.
    """
    
    def __init__(
        self,
        prometheus_client: PrometheusClient,
        grafana_client: GrafanaClient,
        jaeger_client: JaegerClient,
        datadog_client: DataDogClient
    ):
        self.prometheus = prometheus_client
        self.grafana = grafana_client
        self.jaeger = jaeger_client
        self.datadog = datadog_client
        
        self._initialized = False
        self._monitoring_tasks: List[asyncio.Task] = []
    
    async def initialize(self) -> None:
        """Initialize all monitoring components"""
        try:
            logger.info("Initializing monitoring service...")
            
            # Initialize all clients
            await self.prometheus.initialize()
            await self.grafana.initialize()
            await self.jaeger.initialize()
            await self.datadog.initialize()
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            self._initialized = True
            logger.info("Monitoring service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring service: {str(e)}")
            raise InfrastructureError(f"Failed to initialize monitoring service: {str(e)}", service="monitoring")
    
    async def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks"""
        try:
            # Start Prometheus HTTP server
            self.prometheus.start_http_server(port=8000)
            
            # Start metrics collection task
            metrics_task = asyncio.create_task(self._metrics_collection_loop())
            self._monitoring_tasks.append(metrics_task)
            
            # Start health check task
            health_task = asyncio.create_task(self._health_check_loop())
            self._monitoring_tasks.append(health_task)
            
            logger.info("Monitoring tasks started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring tasks: {str(e)}")
            raise InfrastructureError(f"Failed to start monitoring tasks: {str(e)}", service="monitoring")
    
    async def _metrics_collection_loop(self) -> None:
        """Background task for collecting and pushing metrics"""
        while True:
            try:
                # Push metrics to Prometheus Pushgateway
                self.prometheus.push_metrics()
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _health_check_loop(self) -> None:
        """Background task for health checks"""
        while True:
            try:
                # Perform health checks
                await self._perform_health_checks()
                
                # Wait before next health check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(120)  # Wait longer on error
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all components"""
        try:
            # Check Prometheus
            prometheus_health = await self._check_prometheus_health()
            
            # Check Grafana
            grafana_health = await self._check_grafana_health()
            
            # Check Jaeger
            jaeger_health = await self._check_jaeger_health()
            
            # Check DataDog
            datadog_health = await self._check_datadog_health()
            
            # Record health metrics
            self.prometheus.set_system_info(
                version="1.0.0",
                environment="production",
                region="eu-central-1"
            )
            
            # Send health status to DataDog
            if prometheus_health and grafana_health and jaeger_health and datadog_health:
                await self.datadog.send_event(
                    title="Monitoring Service Health Check",
                    text="All monitoring components are healthy",
                    alert_type="success"
                )
            else:
                await self.datadog.send_event(
                    title="Monitoring Service Health Check",
                    text="Some monitoring components are unhealthy",
                    alert_type="warning"
                )
            
        except Exception as e:
            logger.error(f"Error performing health checks: {str(e)}")
    
    async def _check_prometheus_health(self) -> bool:
        """Check Prometheus health"""
        try:
            metrics = self.prometheus.get_metrics()
            return len(metrics) > 0
        except Exception as e:
            logger.error(f"Prometheus health check failed: {str(e)}")
            return False
    
    async def _check_grafana_health(self) -> bool:
        """Check Grafana health"""
        try:
            info = await self.grafana.get_client_info()
            return info.get('initialized', False)
        except Exception as e:
            logger.error(f"Grafana health check failed: {str(e)}")
            return False
    
    async def _check_jaeger_health(self) -> bool:
        """Check Jaeger health"""
        try:
            info = self.jaeger.get_client_info()
            return info.get('initialized', False)
        except Exception as e:
            logger.error(f"Jaeger health check failed: {str(e)}")
            return False
    
    async def _check_datadog_health(self) -> bool:
        """Check DataDog health"""
        try:
            metrics = await self.datadog.get_metrics()
            return metrics.get('initialized', False)
        except Exception as e:
            logger.error(f"DataDog health check failed: {str(e)}")
            return False
    
    async def setup_monitoring_dashboards(self) -> Dict[str, Any]:
        """Setup all monitoring dashboards in Grafana"""
        try:
            # Create Prometheus data source
            prometheus_datasource = await self.grafana.create_prometheus_data_source(
                name="Prometheus",
                url="http://prometheus:9090"
            )
            
            # Setup all dashboards
            dashboards = await self.grafana.setup_monitoring_dashboards(
                prometheus_datasource['id']
            )
            
            logger.info("All monitoring dashboards setup completed")
            return dashboards
            
        except Exception as e:
            logger.error(f"Error setting up monitoring dashboards: {str(e)}")
            raise InfrastructureError(f"Failed to setup monitoring dashboards: {str(e)}", service="monitoring")
    
    # Smart Meter Monitoring
    async def track_smart_meter_metrics(
        self,
        meter_id: str,
        reading_count: int,
        quality_score: float,
        anomaly_count: int
    ) -> None:
        """Track smart meter metrics across all monitoring systems"""
        try:
            # Prometheus metrics
            self.prometheus.increment_meter_readings(meter_id, "success")
            self.prometheus.set_meter_quality_score(meter_id, quality_score)
            if anomaly_count > 0:
                self.prometheus.increment_meter_anomalies(meter_id, "detected")
            
            # DataDog metrics
            await self.datadog.track_meter_metrics(meter_id, reading_count, quality_score, anomaly_count)
            
        except Exception as e:
            logger.error(f"Error tracking smart meter metrics: {str(e)}")
    
    # Grid Operator Monitoring
    async def track_grid_operator_metrics(
        self,
        operator_id: str,
        stability_score: float,
        load_percentage: float,
        anomaly_count: int
    ) -> None:
        """Track grid operator metrics across all monitoring systems"""
        try:
            # Prometheus metrics
            self.prometheus.increment_grid_status_updates(operator_id, "success")
            self.prometheus.set_grid_stability_score(operator_id, stability_score)
            self.prometheus.set_grid_load_percentage(operator_id, load_percentage)
            
            # DataDog metrics
            await self.datadog.track_grid_metrics(operator_id, stability_score, load_percentage, anomaly_count)
            
        except Exception as e:
            logger.error(f"Error tracking grid operator metrics: {str(e)}")
    
    # Weather Station Monitoring
    async def track_weather_station_metrics(
        self,
        station_id: str,
        observation_count: int,
        quality_score: float,
        anomaly_count: int
    ) -> None:
        """Track weather station metrics across all monitoring systems"""
        try:
            # Prometheus metrics
            self.prometheus.increment_weather_observations(station_id, "success")
            self.prometheus.set_data_quality_score("weather_station", station_id, quality_score)
            
            # DataDog metrics
            await self.datadog.track_weather_metrics(station_id, observation_count, quality_score, anomaly_count)
            
        except Exception as e:
            logger.error(f"Error tracking weather station metrics: {str(e)}")
    
    # Performance Monitoring
    async def track_operation_performance(
        self,
        operation: str,
        duration: float,
        success: bool,
        error_count: int = 0
    ) -> None:
        """Track operation performance across all monitoring systems"""
        try:
            # Prometheus metrics
            self.prometheus.record_operation_duration(operation, duration, "success" if success else "error")
            self.prometheus.increment_operation_requests(operation, "success" if success else "error")
            
            # DataDog metrics
            await self.datadog.track_performance_metrics(operation, duration, success, error_count)
            
        except Exception as e:
            logger.error(f"Error tracking operation performance: {str(e)}")
    
    # Kafka Monitoring
    async def track_kafka_metrics(
        self,
        topic: str,
        messages_produced: int,
        messages_consumed: int,
        consumer_lag: int
    ) -> None:
        """Track Kafka metrics across all monitoring systems"""
        try:
            # Prometheus metrics
            for _ in range(messages_produced):
                self.prometheus.increment_kafka_messages_produced(topic, "success")
            
            for _ in range(messages_consumed):
                self.prometheus.increment_kafka_messages_consumed(topic, "default")
            
            self.prometheus.set_kafka_consumer_lag(topic, "default", consumer_lag)
            
        except Exception as e:
            logger.error(f"Error tracking Kafka metrics: {str(e)}")
    
    # Airflow Monitoring
    async def track_airflow_metrics(
        self,
        dag_id: str,
        task_id: str,
        status: str,
        duration: float
    ) -> None:
        """Track Airflow metrics across all monitoring systems"""
        try:
            # Prometheus metrics
            self.prometheus.increment_airflow_dag_runs(dag_id, status)
            self.prometheus.increment_airflow_task_runs(dag_id, task_id, status)
            self.prometheus.record_airflow_dag_duration(dag_id, duration)
            
        except Exception as e:
            logger.error(f"Error tracking Airflow metrics: {str(e)}")
    
    # Database Monitoring
    async def track_database_metrics(
        self,
        database: str,
        operation: str,
        duration: float,
        success: bool
    ) -> None:
        """Track database metrics across all monitoring systems"""
        try:
            # Prometheus metrics
            self.prometheus.increment_database_queries(database, operation, "success" if success else "error")
            self.prometheus.record_database_query_duration(database, operation, duration)
            
        except Exception as e:
            logger.error(f"Error tracking database metrics: {str(e)}")
    
    # S3 Monitoring
    async def track_s3_metrics(
        self,
        operation: str,
        bucket: str,
        duration: float,
        success: bool
    ) -> None:
        """Track S3 metrics across all monitoring systems"""
        try:
            # Prometheus metrics
            self.prometheus.increment_s3_operations(operation, bucket, "success" if success else "error")
            self.prometheus.record_s3_operation_duration(operation, bucket, duration)
            
        except Exception as e:
            logger.error(f"Error tracking S3 metrics: {str(e)}")
    
    # Snowflake Monitoring
    async def track_snowflake_metrics(
        self,
        warehouse: str,
        duration: float,
        success: bool
    ) -> None:
        """Track Snowflake metrics across all monitoring systems"""
        try:
            # Prometheus metrics
            self.prometheus.increment_snowflake_queries(warehouse, "success" if success else "error")
            self.prometheus.record_snowflake_query_duration(warehouse, duration)
            
        except Exception as e:
            logger.error(f"Error tracking Snowflake metrics: {str(e)}")
    
    # Data Quality Monitoring
    async def track_data_quality_metrics(
        self,
        entity_type: str,
        entity_id: str,
        quality_score: float,
        total_records: int,
        valid_records: int
    ) -> None:
        """Track data quality metrics across all monitoring systems"""
        try:
            # Prometheus metrics
            self.prometheus.set_data_quality_score(entity_type, entity_id, quality_score)
            
            # DataDog metrics
            await self.datadog.track_data_quality_metrics(
                entity_type, entity_id, quality_score, total_records, valid_records
            )
            
        except Exception as e:
            logger.error(f"Error tracking data quality metrics: {str(e)}")
    
    # System Monitoring
    async def track_system_metrics(
        self,
        total_meters: int,
        active_meters: int,
        total_operators: int,
        active_operators: int,
        total_stations: int,
        active_stations: int
    ) -> None:
        """Track system metrics across all monitoring systems"""
        try:
            # DataDog metrics
            await self.datadog.track_system_metrics(
                total_meters, active_meters, total_operators,
                active_operators, total_stations, active_stations
            )
            
        except Exception as e:
            logger.error(f"Error tracking system metrics: {str(e)}")
    
    # Alerting
    async def send_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send alerts across all monitoring systems"""
        try:
            # DataDog alert
            await self.datadog.send_alert(
                alert_title=f"{alert_type}: {message}",
                alert_message=message,
                severity=severity,
                entity_id=entity_id,
                entity_type=entity_type
            )
            
            # Grafana event (if supported)
            await self.grafana.send_event(
                title=alert_type,
                text=message,
                alert_type=severity,
                tags=[f"entity_id:{entity_id}", f"entity_type:{entity_type}"] if entity_id else None
            )
            
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")
    
    # Tracing
    def trace_operation(
        self,
        operation_name: str,
        component: str,
        entity_id: Optional[str] = None,
        parent_span: Optional[Any] = None
    ) -> Any:
        """Start tracing an operation"""
        try:
            tags = {
                'component': component,
                'service': 'metrify-smart-metering'
            }
            
            if entity_id:
                tags['entity_id'] = entity_id
            
            return self.jaeger.start_span(operation_name, parent_span, tags)
            
        except Exception as e:
            logger.error(f"Error starting trace: {str(e)}")
            return None
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        try:
            return {
                "initialized": self._initialized,
                "prometheus": {
                    "initialized": True,
                    "metrics_endpoint": "http://localhost:8000/metrics"
                },
                "grafana": await self.grafana.get_client_info(),
                "jaeger": self.jaeger.get_client_info(),
                "datadog": await self.datadog.get_metrics(),
                "monitoring_tasks": len(self._monitoring_tasks)
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring status: {str(e)}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown monitoring service"""
        try:
            # Cancel monitoring tasks
            for task in self._monitoring_tasks:
                task.cancel()
            
            # Close Jaeger client
            self.jaeger.close()
            
            logger.info("Monitoring service shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during monitoring service shutdown: {str(e)}")
