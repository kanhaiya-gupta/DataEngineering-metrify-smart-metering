"""
Processing Worker
Background worker for data processing operations
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import signal
import sys

from src.core.config.config_loader import get_database_config, get_kafka_config, get_s3_config, get_snowflake_config
from src.infrastructure.database.repositories.smart_meter_repository import SmartMeterRepository
from src.infrastructure.database.repositories.grid_operator_repository import GridOperatorRepository
from src.infrastructure.database.repositories.weather_station_repository import WeatherStationRepository
from src.infrastructure.external.kafka.kafka_consumer import KafkaConsumer
from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
from src.infrastructure.external.s3.s3_client import S3Client
from src.infrastructure.external.snowflake.snowflake_client import SnowflakeClient
from src.infrastructure.external.snowflake.query_executor import SnowflakeQueryExecutor
from src.infrastructure.external.apis.data_quality_service import DataQualityService
from src.infrastructure.external.apis.anomaly_detection_service import AnomalyDetectionService
from src.infrastructure.external.apis.alerting_service import AlertingService
from src.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
from src.application.use_cases.analyze_weather_impact import AnalyzeWeatherImpactUseCase

logger = logging.getLogger(__name__)


class ProcessingWorker:
    """
    Background worker for data processing operations
    
    Handles batch processing, data transformation, and analytics
    operations on ingested data.
    """
    
    def __init__(self, worker_id: str = "processing-worker-1"):
        self.worker_id = worker_id
        self.running = False
        self.consumers = {}
        self.producers = {}
        self.repositories = {}
        self.services = {}
        self.use_cases = {}
        
        # Initialize configuration
        self.db_config = get_database_config()
        self.kafka_config = get_kafka_config()
        self.s3_config = get_s3_config()
        self.snowflake_config = get_snowflake_config()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def start(self):
        """Start the processing worker"""
        try:
            logger.info(f"Starting processing worker {self.worker_id}")
            self.running = True
            
            # Initialize services
            await self._initialize_services()
            
            # Start consumers
            await self._start_consumers()
            
            # Start processing loop
            await self._processing_loop()
            
        except Exception as e:
            logger.error(f"Error starting processing worker: {str(e)}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the processing worker"""
        try:
            logger.info(f"Stopping processing worker {self.worker_id}")
            self.running = False
            
            # Stop consumers
            await self._stop_consumers()
            
            # Close producers
            await self._close_producers()
            
            logger.info(f"Processing worker {self.worker_id} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping processing worker: {str(e)}")
    
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
            self.services['snowflake_client'] = SnowflakeClient(self.snowflake_config)
            self.services['snowflake_query_executor'] = SnowflakeQueryExecutor(self.snowflake_config)
            self.services['data_quality'] = DataQualityService()
            self.services['anomaly_detection'] = AnomalyDetectionService()
            self.services['alerting'] = AlertingService()
            
            # Initialize use cases
            self.use_cases['detect_anomalies'] = DetectAnomaliesUseCase(
                smart_meter_repository=self.repositories['smart_meter'],
                grid_operator_repository=self.repositories['grid_operator'],
                weather_station_repository=self.repositories['weather_station'],
                data_quality_service=self.services['data_quality'],
                anomaly_detection_service=self.services['anomaly_detection'],
                alerting_service=self.services['alerting'],
                kafka_producer=self.services['kafka_producer'],
                s3_client=self.services['s3_client']
            )
            
            self.use_cases['analyze_weather_impact'] = AnalyzeWeatherImpactUseCase(
                weather_station_repository=self.repositories['weather_station'],
                smart_meter_repository=self.repositories['smart_meter'],
                grid_operator_repository=self.repositories['grid_operator'],
                data_quality_service=self.services['data_quality'],
                anomaly_detection_service=self.services['anomaly_detection'],
                alerting_service=self.services['alerting'],
                kafka_producer=self.services['kafka_producer'],
                s3_client=self.services['s3_client'],
                snowflake_query_executor=self.services['snowflake_query_executor']
            )
            
            logger.info("Services initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing services: {str(e)}")
            raise
    
    async def _start_consumers(self):
        """Start Kafka consumers for different topics"""
        try:
            # Data quality events consumer
            self.consumers['data_quality_events'] = KafkaConsumer(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                group_id=f"{self.worker_id}-data-quality",
                topics=['data_quality_events'],
                auto_offset_reset='latest'
            )
            
            # Anomaly detection events consumer
            self.consumers['anomaly_detection_events'] = KafkaConsumer(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                group_id=f"{self.worker_id}-anomaly-detection",
                topics=['anomaly_detection_events'],
                auto_offset_reset='latest'
            )
            
            # Weather impact analysis events consumer
            self.consumers['weather_impact_events'] = KafkaConsumer(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                group_id=f"{self.worker_id}-weather-impact",
                topics=['weather_impact_events'],
                auto_offset_reset='latest'
            )
            
            # Batch processing events consumer
            self.consumers['batch_processing_events'] = KafkaConsumer(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                group_id=f"{self.worker_id}-batch-processing",
                topics=['batch_processing_events'],
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
    
    async def _processing_loop(self):
        """Main processing loop"""
        try:
            logger.info("Starting processing loop")
            
            while self.running:
                # Process data quality events
                await self._process_data_quality_events()
                
                # Process anomaly detection events
                await self._process_anomaly_detection_events()
                
                # Process weather impact analysis events
                await self._process_weather_impact_events()
                
                # Process batch processing events
                await self._process_batch_processing_events()
                
                # Run scheduled processing tasks
                await self._run_scheduled_tasks()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(1.0)
            
        except Exception as e:
            logger.error(f"Error in processing loop: {str(e)}")
            raise
    
    async def _process_data_quality_events(self):
        """Process data quality events"""
        try:
            consumer = self.consumers.get('data_quality_events')
            if not consumer:
                return
            
            # Consume messages
            messages = await consumer.consume_messages(max_messages=50, timeout_ms=1000)
            
            if not messages:
                return
            
            for message in messages:
                try:
                    event = json.loads(message.value.decode('utf-8'))
                    await self._handle_data_quality_event(event)
                    
                except Exception as e:
                    logger.error(f"Error processing data quality event: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error processing data quality events: {str(e)}")
    
    async def _process_anomaly_detection_events(self):
        """Process anomaly detection events"""
        try:
            consumer = self.consumers.get('anomaly_detection_events')
            if not consumer:
                return
            
            # Consume messages
            messages = await consumer.consume_messages(max_messages=50, timeout_ms=1000)
            
            if not messages:
                return
            
            for message in messages:
                try:
                    event = json.loads(message.value.decode('utf-8'))
                    await self._handle_anomaly_detection_event(event)
                    
                except Exception as e:
                    logger.error(f"Error processing anomaly detection event: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error processing anomaly detection events: {str(e)}")
    
    async def _process_weather_impact_events(self):
        """Process weather impact analysis events"""
        try:
            consumer = self.consumers.get('weather_impact_events')
            if not consumer:
                return
            
            # Consume messages
            messages = await consumer.consume_messages(max_messages=50, timeout_ms=1000)
            
            if not messages:
                return
            
            for message in messages:
                try:
                    event = json.loads(message.value.decode('utf-8'))
                    await self._handle_weather_impact_event(event)
                    
                except Exception as e:
                    logger.error(f"Error processing weather impact event: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error processing weather impact events: {str(e)}")
    
    async def _process_batch_processing_events(self):
        """Process batch processing events"""
        try:
            consumer = self.consumers.get('batch_processing_events')
            if not consumer:
                return
            
            # Consume messages
            messages = await consumer.consume_messages(max_messages=50, timeout_ms=1000)
            
            if not messages:
                return
            
            for message in messages:
                try:
                    event = json.loads(message.value.decode('utf-8'))
                    await self._handle_batch_processing_event(event)
                    
                except Exception as e:
                    logger.error(f"Error processing batch processing event: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error processing batch processing events: {str(e)}")
    
    async def _run_scheduled_tasks(self):
        """Run scheduled processing tasks"""
        try:
            # Run every 5 minutes
            current_time = datetime.utcnow()
            if current_time.minute % 5 == 0 and current_time.second < 10:
                await self._run_hourly_analytics()
            
            # Run every hour
            if current_time.minute == 0 and current_time.second < 10:
                await self._run_daily_analytics()
            
            # Run every day at midnight
            if current_time.hour == 0 and current_time.minute == 0 and current_time.second < 10:
                await self._run_daily_data_quality_check()
            
        except Exception as e:
            logger.error(f"Error running scheduled tasks: {str(e)}")
    
    async def _handle_data_quality_event(self, event: Dict[str, Any]):
        """Handle data quality event"""
        try:
            event_type = event.get('event_type')
            
            if event_type == 'quality_check_requested':
                # Trigger data quality check
                await self._run_data_quality_check(event.get('data'))
            
            elif event_type == 'quality_issue_detected':
                # Handle quality issue
                await self._handle_quality_issue(event.get('data'))
            
            else:
                logger.warning(f"Unknown data quality event type: {event_type}")
            
        except Exception as e:
            logger.error(f"Error handling data quality event: {str(e)}")
    
    async def _handle_anomaly_detection_event(self, event: Dict[str, Any]):
        """Handle anomaly detection event"""
        try:
            event_type = event.get('event_type')
            
            if event_type == 'anomaly_detection_requested':
                # Trigger anomaly detection
                await self._run_anomaly_detection(event.get('data'))
            
            elif event_type == 'anomaly_detected':
                # Handle detected anomaly
                await self._handle_detected_anomaly(event.get('data'))
            
            else:
                logger.warning(f"Unknown anomaly detection event type: {event_type}")
            
        except Exception as e:
            logger.error(f"Error handling anomaly detection event: {str(e)}")
    
    async def _handle_weather_impact_event(self, event: Dict[str, Any]):
        """Handle weather impact analysis event"""
        try:
            event_type = event.get('event_type')
            
            if event_type == 'weather_impact_analysis_requested':
                # Trigger weather impact analysis
                await self._run_weather_impact_analysis(event.get('data'))
            
            elif event_type == 'weather_impact_analyzed':
                # Handle weather impact analysis result
                await self._handle_weather_impact_result(event.get('data'))
            
            else:
                logger.warning(f"Unknown weather impact event type: {event_type}")
            
        except Exception as e:
            logger.error(f"Error handling weather impact event: {str(e)}")
    
    async def _handle_batch_processing_event(self, event: Dict[str, Any]):
        """Handle batch processing event"""
        try:
            event_type = event.get('event_type')
            
            if event_type == 'batch_processing_requested':
                # Trigger batch processing
                await self._run_batch_processing(event.get('data'))
            
            elif event_type == 'batch_processing_completed':
                # Handle batch processing completion
                await self._handle_batch_processing_completion(event.get('data'))
            
            else:
                logger.warning(f"Unknown batch processing event type: {event_type}")
            
        except Exception as e:
            logger.error(f"Error handling batch processing event: {str(e)}")
    
    async def _run_data_quality_check(self, data: Dict[str, Any]):
        """Run data quality check"""
        try:
            logger.info("Running data quality check")
            
            # Implement data quality check logic
            # This would use the data quality service to check data quality
            
            logger.info("Data quality check completed")
            
        except Exception as e:
            logger.error(f"Error running data quality check: {str(e)}")
    
    async def _handle_quality_issue(self, data: Dict[str, Any]):
        """Handle quality issue"""
        try:
            logger.info(f"Handling quality issue: {data}")
            
            # Implement quality issue handling logic
            # This would send alerts, update data, etc.
            
        except Exception as e:
            logger.error(f"Error handling quality issue: {str(e)}")
    
    async def _run_anomaly_detection(self, data: Dict[str, Any]):
        """Run anomaly detection"""
        try:
            logger.info("Running anomaly detection")
            
            # Use the anomaly detection use case
            result = await self.use_cases['detect_anomalies'].execute(
                data_source=data.get('data_source'),
                start_time=data.get('start_time'),
                end_time=data.get('end_time'),
                sensitivity=data.get('sensitivity', 0.5)
            )
            
            logger.info(f"Anomaly detection completed: {result}")
            
        except Exception as e:
            logger.error(f"Error running anomaly detection: {str(e)}")
    
    async def _handle_detected_anomaly(self, data: Dict[str, Any]):
        """Handle detected anomaly"""
        try:
            logger.info(f"Handling detected anomaly: {data}")
            
            # Implement anomaly handling logic
            # This would send alerts, update data, etc.
            
        except Exception as e:
            logger.error(f"Error handling detected anomaly: {str(e)}")
    
    async def _run_weather_impact_analysis(self, data: Dict[str, Any]):
        """Run weather impact analysis"""
        try:
            logger.info("Running weather impact analysis")
            
            # Use the weather impact analysis use case
            result = await self.use_cases['analyze_weather_impact'].execute(
                station_id=data.get('station_id'),
                analysis_period_hours=data.get('analysis_period_hours', 24)
            )
            
            logger.info(f"Weather impact analysis completed: {result}")
            
        except Exception as e:
            logger.error(f"Error running weather impact analysis: {str(e)}")
    
    async def _handle_weather_impact_result(self, data: Dict[str, Any]):
        """Handle weather impact analysis result"""
        try:
            logger.info(f"Handling weather impact result: {data}")
            
            # Implement weather impact result handling logic
            # This would update forecasts, send alerts, etc.
            
        except Exception as e:
            logger.error(f"Error handling weather impact result: {str(e)}")
    
    async def _run_batch_processing(self, data: Dict[str, Any]):
        """Run batch processing"""
        try:
            logger.info("Running batch processing")
            
            # Implement batch processing logic
            # This would process large batches of data
            
            logger.info("Batch processing completed")
            
        except Exception as e:
            logger.error(f"Error running batch processing: {str(e)}")
    
    async def _handle_batch_processing_completion(self, data: Dict[str, Any]):
        """Handle batch processing completion"""
        try:
            logger.info(f"Handling batch processing completion: {data}")
            
            # Implement batch processing completion handling logic
            # This would update status, send notifications, etc.
            
        except Exception as e:
            logger.error(f"Error handling batch processing completion: {str(e)}")
    
    async def _run_hourly_analytics(self):
        """Run hourly analytics"""
        try:
            logger.info("Running hourly analytics")
            
            # Implement hourly analytics logic
            # This would calculate hourly metrics, trends, etc.
            
            logger.info("Hourly analytics completed")
            
        except Exception as e:
            logger.error(f"Error running hourly analytics: {str(e)}")
    
    async def _run_daily_analytics(self):
        """Run daily analytics"""
        try:
            logger.info("Running daily analytics")
            
            # Implement daily analytics logic
            # This would calculate daily metrics, reports, etc.
            
            logger.info("Daily analytics completed")
            
        except Exception as e:
            logger.error(f"Error running daily analytics: {str(e)}")
    
    async def _run_daily_data_quality_check(self):
        """Run daily data quality check"""
        try:
            logger.info("Running daily data quality check")
            
            # Implement daily data quality check logic
            # This would check data quality across all sources
            
            logger.info("Daily data quality check completed")
            
        except Exception as e:
            logger.error(f"Error running daily data quality check: {str(e)}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False


async def main():
    """Main entry point for the processing worker"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create and start worker
        worker = ProcessingWorker()
        await worker.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
