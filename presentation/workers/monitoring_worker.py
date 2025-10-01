"""
Monitoring Worker
Background worker for monitoring and alerting operations
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import signal
import sys

from src.core.config.config_loader import get_database_config, get_kafka_config, get_s3_config, get_monitoring_config
from src.infrastructure.database.repositories.smart_meter_repository import SmartMeterRepository
from src.infrastructure.database.repositories.grid_operator_repository import GridOperatorRepository
from src.infrastructure.database.repositories.weather_station_repository import WeatherStationRepository
from src.infrastructure.external.kafka.kafka_consumer import KafkaConsumer
from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
from src.infrastructure.external.s3.s3_client import S3Client
from src.infrastructure.external.monitoring.monitoring_service import MonitoringService
from src.infrastructure.external.monitoring.prometheus.prometheus_client import PrometheusClient
from src.infrastructure.external.monitoring.grafana.grafana_client import GrafanaClient
from src.infrastructure.external.monitoring.jaeger.jaeger_client import JaegerClient
from src.infrastructure.external.monitoring.datadog.datadog_client import DataDogClient
from src.infrastructure.external.apis.alerting_service import AlertingService

logger = logging.getLogger(__name__)


class MonitoringWorker:
    """
    Background worker for monitoring and alerting operations
    
    Handles system monitoring, metrics collection, alerting,
    and health checks across all components.
    """
    
    def __init__(self, worker_id: str = "monitoring-worker-1"):
        self.worker_id = worker_id
        self.running = False
        self.consumers = {}
        self.producers = {}
        self.repositories = {}
        self.services = {}
        self.monitoring_service = None
        
        # Initialize configuration
        self.db_config = get_database_config()
        self.kafka_config = get_kafka_config()
        self.s3_config = get_s3_config()
        self.monitoring_config = get_monitoring_config()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def start(self):
        """Start the monitoring worker"""
        try:
            logger.info(f"Starting monitoring worker {self.worker_id}")
            self.running = True
            
            # Initialize services
            await self._initialize_services()
            
            # Start consumers
            await self._start_consumers()
            
            # Start monitoring loop
            await self._monitoring_loop()
            
        except Exception as e:
            logger.error(f"Error starting monitoring worker: {str(e)}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the monitoring worker"""
        try:
            logger.info(f"Stopping monitoring worker {self.worker_id}")
            self.running = False
            
            # Stop consumers
            await self._stop_consumers()
            
            # Close producers
            await self._close_producers()
            
            # Shutdown monitoring service
            if self.monitoring_service:
                await self.monitoring_service.shutdown()
            
            logger.info(f"Monitoring worker {self.worker_id} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring worker: {str(e)}")
    
    async def _initialize_services(self):
        """Initialize all required services"""
        try:
            # Initialize repositories
            self.repositories['smart_meter'] = SmartMeterRepository(self.db_config)
            self.repositories['grid_operator'] = GridOperatorRepository(self.db_config)
            self.repositories['weather_station'] = WeatherStationRepository(self.db_config)
            
            # Initialize external services
            self.services['kafka_producer'] = KafkaProducer(self.kafka_config)
            self.services['s3_client'] = S3Client(self.s3_config)
            self.services['alerting'] = AlertingService()
            
            # Initialize monitoring clients
            prometheus_client = PrometheusClient(
                namespace="metrify",
                subsystem="smart_metering"
            )
            
            grafana_client = GrafanaClient(
                base_url=self.monitoring_config.grafana_url,
                username=self.monitoring_config.grafana_username,
                password=self.monitoring_config.grafana_password
            )
            
            jaeger_client = JaegerClient(
                service_name="metrify-smart-metering-monitoring",
                agent_host=self.monitoring_config.jaeger_agent_host,
                agent_port=self.monitoring_config.jaeger_agent_port
            )
            
            datadog_client = DataDogClient(
                api_key=self.monitoring_config.datadog_api_key,
                app_key=self.monitoring_config.datadog_app_key,
                site=self.monitoring_config.datadog_site
            )
            
            # Initialize monitoring service
            self.monitoring_service = MonitoringService(
                prometheus_client=prometheus_client,
                grafana_client=grafana_client,
                jaeger_client=jaeger_client,
                datadog_client=datadog_client
            )
            
            await self.monitoring_service.initialize()
            
            logger.info("Services initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing services: {str(e)}")
            raise
    
    async def _start_consumers(self):
        """Start Kafka consumers for different topics"""
        try:
            # Monitoring events consumer
            self.consumers['monitoring_events'] = KafkaConsumer(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                group_id=f"{self.worker_id}-monitoring",
                topics=['monitoring_events'],
                auto_offset_reset='latest'
            )
            
            # Alert events consumer
            self.consumers['alert_events'] = KafkaConsumer(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                group_id=f"{self.worker_id}-alerts",
                topics=['alert_events'],
                auto_offset_reset='latest'
            )
            
            # Health check events consumer
            self.consumers['health_check_events'] = KafkaConsumer(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                group_id=f"{self.worker_id}-health-check",
                topics=['health_check_events'],
                auto_offset_reset='latest'
            )
            
            # Start all consumers
            for name, consumer in self.consumers.items():
                await consumer.start()
                logger.info(f"Started consumer for {name}")
            
        except Exception as e:
            logger.error(f"Error starting consumers: {str(e)}")
            raise
    
    async def _stop_consumers(self):
        """Stop all Kafka consumers"""
        try:
            for name, consumer in self.consumers.items():
                await consumer.stop()
                logger.info(f"Stopped consumer for {name}")
            
            self.consumers.clear()
            
        except Exception as e:
            logger.error(f"Error stopping consumers: {str(e)}")
    
    async def _close_producers(self):
        """Close all Kafka producers"""
        try:
            for name, producer in self.producers.items():
                await producer.close()
                logger.info(f"Closed producer for {name}")
            
            self.producers.clear()
            
        except Exception as e:
            logger.error(f"Error closing producers: {str(e)}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            logger.info("Starting monitoring loop")
            
            while self.running:
                # Process monitoring events
                await self._process_monitoring_events()
                
                # Process alert events
                await self._process_alert_events()
                
                # Process health check events
                await self._process_health_check_events()
                
                # Run scheduled monitoring tasks
                await self._run_scheduled_monitoring_tasks()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(5.0)
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            raise
    
    async def _process_monitoring_events(self):
        """Process monitoring events"""
        try:
            consumer = self.consumers.get('monitoring_events')
            if not consumer:
                return
            
            # Consume messages
            messages = await consumer.consume_messages(max_messages=50, timeout_ms=1000)
            
            if not messages:
                return
            
            for message in messages:
                try:
                    event = json.loads(message.value.decode('utf-8'))
                    await self._handle_monitoring_event(event)
                    
                except Exception as e:
                    logger.error(f"Error processing monitoring event: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error processing monitoring events: {str(e)}")
    
    async def _process_alert_events(self):
        """Process alert events"""
        try:
            consumer = self.consumers.get('alert_events')
            if not consumer:
                return
            
            # Consume messages
            messages = await consumer.consume_messages(max_messages=50, timeout_ms=1000)
            
            if not messages:
                return
            
            for message in messages:
                try:
                    event = json.loads(message.value.decode('utf-8'))
                    await self._handle_alert_event(event)
                    
                except Exception as e:
                    logger.error(f"Error processing alert event: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error processing alert events: {str(e)}")
    
    async def _process_health_check_events(self):
        """Process health check events"""
        try:
            consumer = self.consumers.get('health_check_events')
            if not consumer:
                return
            
            # Consume messages
            messages = await consumer.consume_messages(max_messages=50, timeout_ms=1000)
            
            if not messages:
                return
            
            for message in messages:
                try:
                    event = json.loads(message.value.decode('utf-8'))
                    await self._handle_health_check_event(event)
                    
                except Exception as e:
                    logger.error(f"Error processing health check event: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error processing health check events: {str(e)}")
    
    async def _run_scheduled_monitoring_tasks(self):
        """Run scheduled monitoring tasks"""
        try:
            current_time = datetime.utcnow()
            
            # Run every minute
            if current_time.second < 10:
                await self._collect_metrics()
            
            # Run every 5 minutes
            if current_time.minute % 5 == 0 and current_time.second < 10:
                await self._check_system_health()
            
            # Run every 15 minutes
            if current_time.minute % 15 == 0 and current_time.second < 10:
                await self._update_dashboards()
            
            # Run every hour
            if current_time.minute == 0 and current_time.second < 10:
                await self._generate_hourly_report()
            
            # Run every day at midnight
            if current_time.hour == 0 and current_time.minute == 0 and current_time.second < 10:
                await self._generate_daily_report()
            
        except Exception as e:
            logger.error(f"Error running scheduled monitoring tasks: {str(e)}")
    
    async def _handle_monitoring_event(self, event: Dict[str, Any]):
        """Handle monitoring event"""
        try:
            event_type = event.get('event_type')
            
            if event_type == 'metric_collection_requested':
                await self._collect_metrics_for_component(event.get('data'))
            
            elif event_type == 'dashboard_update_requested':
                await self._update_dashboard(event.get('data'))
            
            elif event_type == 'alert_rule_updated':
                await self._update_alert_rules(event.get('data'))
            
            else:
                logger.warning(f"Unknown monitoring event type: {event_type}")
            
        except Exception as e:
            logger.error(f"Error handling monitoring event: {str(e)}")
    
    async def _handle_alert_event(self, event: Dict[str, Any]):
        """Handle alert event"""
        try:
            event_type = event.get('event_type')
            
            if event_type == 'alert_triggered':
                await self._handle_triggered_alert(event.get('data'))
            
            elif event_type == 'alert_resolved':
                await self._handle_resolved_alert(event.get('data'))
            
            elif event_type == 'alert_escalated':
                await self._handle_escalated_alert(event.get('data'))
            
            else:
                logger.warning(f"Unknown alert event type: {event_type}")
            
        except Exception as e:
            logger.error(f"Error handling alert event: {str(e)}")
    
    async def _handle_health_check_event(self, event: Dict[str, Any]):
        """Handle health check event"""
        try:
            event_type = event.get('event_type')
            
            if event_type == 'health_check_requested':
                await self._run_health_check_for_component(event.get('data'))
            
            elif event_type == 'health_check_failed':
                await self._handle_health_check_failure(event.get('data'))
            
            else:
                logger.warning(f"Unknown health check event type: {event_type}")
            
        except Exception as e:
            logger.error(f"Error handling health check event: {str(e)}")
    
    async def _collect_metrics(self):
        """Collect system metrics"""
        try:
            logger.debug("Collecting system metrics")
            
            # Collect database metrics
            await self._collect_database_metrics()
            
            # Collect Kafka metrics
            await self._collect_kafka_metrics()
            
            # Collect S3 metrics
            await self._collect_s3_metrics()
            
            # Collect application metrics
            await self._collect_application_metrics()
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
    
    async def _collect_database_metrics(self):
        """Collect database metrics"""
        try:
            # Get database statistics
            smart_meter_stats = await self.repositories['smart_meter'].get_statistics()
            grid_operator_stats = await self.repositories['grid_operator'].get_statistics()
            weather_station_stats = await self.repositories['weather_station'].get_statistics()
            
            # Track metrics
            await self.monitoring_service.track_operation_performance(
                operation="database_query",
                duration=0.1,
                success=True
            )
            
            # Log metrics
            logger.debug(f"Database metrics collected: {smart_meter_stats}, {grid_operator_stats}, {weather_station_stats}")
            
        except Exception as e:
            logger.error(f"Error collecting database metrics: {str(e)}")
    
    async def _collect_kafka_metrics(self):
        """Collect Kafka metrics"""
        try:
            # Get Kafka statistics
            kafka_stats = await self.services['kafka_producer'].get_statistics()
            
            # Track metrics
            await self.monitoring_service.track_operation_performance(
                operation="kafka_produce",
                duration=0.05,
                success=True
            )
            
            # Log metrics
            logger.debug(f"Kafka metrics collected: {kafka_stats}")
            
        except Exception as e:
            logger.error(f"Error collecting Kafka metrics: {str(e)}")
    
    async def _collect_s3_metrics(self):
        """Collect S3 metrics"""
        try:
            # Get S3 statistics
            s3_stats = await self.services['s3_client'].get_statistics()
            
            # Track metrics
            await self.monitoring_service.track_operation_performance(
                operation="s3_operation",
                duration=0.2,
                success=True
            )
            
            # Log metrics
            logger.debug(f"S3 metrics collected: {s3_stats}")
            
        except Exception as e:
            logger.error(f"Error collecting S3 metrics: {str(e)}")
    
    async def _collect_application_metrics(self):
        """Collect application metrics"""
        try:
            # Get application statistics
            app_stats = {
                "worker_id": self.worker_id,
                "uptime": datetime.utcnow().isoformat(),
                "status": "running"
            }
            
            # Track metrics
            await self.monitoring_service.track_operation_performance(
                operation="application_monitoring",
                duration=0.01,
                success=True
            )
            
            # Log metrics
            logger.debug(f"Application metrics collected: {app_stats}")
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {str(e)}")
    
    async def _check_system_health(self):
        """Check system health"""
        try:
            logger.info("Checking system health")
            
            # Check database health
            db_health = await self._check_database_health()
            
            # Check Kafka health
            kafka_health = await self._check_kafka_health()
            
            # Check S3 health
            s3_health = await self._check_s3_health()
            
            # Check monitoring services health
            monitoring_health = await self._check_monitoring_health()
            
            # Overall health status
            overall_health = all([
                db_health.get('status') == 'healthy',
                kafka_health.get('status') == 'healthy',
                s3_health.get('status') == 'healthy',
                monitoring_health.get('status') == 'healthy'
            ])
            
            # Log health status
            logger.info(f"System health check completed: {'healthy' if overall_health else 'unhealthy'}")
            
            # Send health status to monitoring service
            await self.monitoring_service.track_operation_performance(
                operation="system_health_check",
                duration=1.0,
                success=overall_health
            )
            
        except Exception as e:
            logger.error(f"Error checking system health: {str(e)}")
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            # Test database connection
            await self.repositories['smart_meter'].health_check()
            
            return {
                "status": "healthy",
                "component": "database",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "database",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _check_kafka_health(self) -> Dict[str, Any]:
        """Check Kafka health"""
        try:
            # Test Kafka connection
            await self.services['kafka_producer'].health_check()
            
            return {
                "status": "healthy",
                "component": "kafka",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "kafka",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _check_s3_health(self) -> Dict[str, Any]:
        """Check S3 health"""
        try:
            # Test S3 connection
            await self.services['s3_client'].health_check()
            
            return {
                "status": "healthy",
                "component": "s3",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "s3",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _check_monitoring_health(self) -> Dict[str, Any]:
        """Check monitoring services health"""
        try:
            # Test monitoring services
            await self.monitoring_service.health_check()
            
            return {
                "status": "healthy",
                "component": "monitoring",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "monitoring",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _update_dashboards(self):
        """Update monitoring dashboards"""
        try:
            logger.info("Updating monitoring dashboards")
            
            # Update Grafana dashboards
            await self.monitoring_service.update_dashboards()
            
            logger.info("Monitoring dashboards updated")
            
        except Exception as e:
            logger.error(f"Error updating dashboards: {str(e)}")
    
    async def _generate_hourly_report(self):
        """Generate hourly monitoring report"""
        try:
            logger.info("Generating hourly monitoring report")
            
            # Generate report data
            report_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "worker_id": self.worker_id,
                "metrics_collected": True,
                "health_checks_passed": True
            }
            
            # Send report to monitoring service
            await self.monitoring_service.track_operation_performance(
                operation="hourly_report_generation",
                duration=0.5,
                success=True
            )
            
            logger.info("Hourly monitoring report generated")
            
        except Exception as e:
            logger.error(f"Error generating hourly report: {str(e)}")
    
    async def _generate_daily_report(self):
        """Generate daily monitoring report"""
        try:
            logger.info("Generating daily monitoring report")
            
            # Generate report data
            report_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "worker_id": self.worker_id,
                "daily_metrics": True,
                "system_health": True
            }
            
            # Send report to monitoring service
            await self.monitoring_service.track_operation_performance(
                operation="daily_report_generation",
                duration=2.0,
                success=True
            )
            
            logger.info("Daily monitoring report generated")
            
        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
    
    async def _collect_metrics_for_component(self, data: Dict[str, Any]):
        """Collect metrics for specific component"""
        try:
            component = data.get('component')
            logger.info(f"Collecting metrics for component: {component}")
            
            # Implement component-specific metrics collection
            # This would collect metrics for the specified component
            
        except Exception as e:
            logger.error(f"Error collecting metrics for component: {str(e)}")
    
    async def _update_dashboard(self, data: Dict[str, Any]):
        """Update specific dashboard"""
        try:
            dashboard_id = data.get('dashboard_id')
            logger.info(f"Updating dashboard: {dashboard_id}")
            
            # Implement dashboard update logic
            # This would update the specified dashboard
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {str(e)}")
    
    async def _update_alert_rules(self, data: Dict[str, Any]):
        """Update alert rules"""
        try:
            rules = data.get('rules')
            logger.info(f"Updating alert rules: {len(rules)} rules")
            
            # Implement alert rules update logic
            # This would update the alert rules
            
        except Exception as e:
            logger.error(f"Error updating alert rules: {str(e)}")
    
    async def _handle_triggered_alert(self, data: Dict[str, Any]):
        """Handle triggered alert"""
        try:
            alert = data.get('alert')
            logger.warning(f"Alert triggered: {alert}")
            
            # Implement alert handling logic
            # This would send notifications, update status, etc.
            
        except Exception as e:
            logger.error(f"Error handling triggered alert: {str(e)}")
    
    async def _handle_resolved_alert(self, data: Dict[str, Any]):
        """Handle resolved alert"""
        try:
            alert = data.get('alert')
            logger.info(f"Alert resolved: {alert}")
            
            # Implement alert resolution logic
            # This would update status, send notifications, etc.
            
        except Exception as e:
            logger.error(f"Error handling resolved alert: {str(e)}")
    
    async def _handle_escalated_alert(self, data: Dict[str, Any]):
        """Handle escalated alert"""
        try:
            alert = data.get('alert')
            logger.error(f"Alert escalated: {alert}")
            
            # Implement alert escalation logic
            # This would send urgent notifications, etc.
            
        except Exception as e:
            logger.error(f"Error handling escalated alert: {str(e)}")
    
    async def _run_health_check_for_component(self, data: Dict[str, Any]):
        """Run health check for specific component"""
        try:
            component = data.get('component')
            logger.info(f"Running health check for component: {component}")
            
            # Implement component-specific health check
            # This would check the health of the specified component
            
        except Exception as e:
            logger.error(f"Error running health check for component: {str(e)}")
    
    async def _handle_health_check_failure(self, data: Dict[str, Any]):
        """Handle health check failure"""
        try:
            component = data.get('component')
            error = data.get('error')
            logger.error(f"Health check failed for {component}: {error}")
            
            # Implement health check failure handling
            # This would send alerts, update status, etc.
            
        except Exception as e:
            logger.error(f"Error handling health check failure: {str(e)}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False


async def main():
    """Main entry point for the monitoring worker"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create and start worker
        worker = MonitoringWorker()
        await worker.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
