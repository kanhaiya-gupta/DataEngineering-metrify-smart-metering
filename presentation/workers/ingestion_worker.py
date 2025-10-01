"""
Ingestion Worker
Background worker for data ingestion operations
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import signal
import sys

from src.core.config.config_loader import get_database_config, get_kafka_config, get_s3_config
from src.infrastructure.database.repositories.smart_meter_repository import SmartMeterRepository
from src.infrastructure.database.repositories.grid_operator_repository import GridOperatorRepository
from src.infrastructure.database.repositories.weather_station_repository import WeatherStationRepository
from src.infrastructure.external.kafka.kafka_consumer import KafkaConsumer
from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
from src.infrastructure.external.s3.s3_client import S3Client
from src.infrastructure.external.apis.data_quality_service import DataQualityService
from src.infrastructure.external.apis.anomaly_detection_service import AnomalyDetectionService
from src.infrastructure.external.apis.alerting_service import AlertingService
from src.application.use_cases.ingest_smart_meter_data import IngestSmartMeterDataUseCase
from src.application.use_cases.process_grid_status import ProcessGridStatusUseCase
from src.application.use_cases.analyze_weather_impact import AnalyzeWeatherImpactUseCase

logger = logging.getLogger(__name__)


class IngestionWorker:
    """
    Background worker for data ingestion operations
    
    Handles real-time data ingestion from Kafka topics and processes
    them through the appropriate use cases.
    """
    
    def __init__(self, worker_id: str = "ingestion-worker-1"):
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
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def start(self):
        """Start the ingestion worker"""
        try:
            logger.info(f"Starting ingestion worker {self.worker_id}")
            self.running = True
            
            # Initialize services
            await self._initialize_services()
            
            # Start consumers
            await self._start_consumers()
            
            # Start processing loop
            await self._processing_loop()
            
        except Exception as e:
            logger.error(f"Error starting ingestion worker: {str(e)}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the ingestion worker"""
        try:
            logger.info(f"Stopping ingestion worker {self.worker_id}")
            self.running = False
            
            # Stop consumers
            await self._stop_consumers()
            
            # Close producers
            await self._close_producers()
            
            logger.info(f"Ingestion worker {self.worker_id} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping ingestion worker: {str(e)}")
    
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
            self.services['data_quality'] = DataQualityService()
            self.services['anomaly_detection'] = AnomalyDetectionService()
            self.services['alerting'] = AlertingService()
            
            # Initialize use cases
            self.use_cases['ingest_smart_meter'] = IngestSmartMeterDataUseCase(
                smart_meter_repository=self.repositories['smart_meter'],
                data_quality_service=self.services['data_quality'],
                anomaly_detection_service=self.services['anomaly_detection'],
                kafka_producer=self.services['kafka_producer'],
                s3_client=self.services['s3_client']
            )
            
            self.use_cases['process_grid_status'] = ProcessGridStatusUseCase(
                grid_operator_repository=self.repositories['grid_operator'],
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
                snowflake_query_executor=None
            )
            
            logger.info("Services initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing services: {str(e)}")
            raise
    
    async def _start_consumers(self):
        """Start Kafka consumers for different topics"""
        try:
            # Smart meter data consumer
            self.consumers['smart_meter_data'] = KafkaConsumer(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                group_id=f"{self.worker_id}-smart-meter",
                topics=['smart_meter_data'],
                auto_offset_reset='latest'
            )
            
            # Grid operator data consumer
            self.consumers['grid_operator_data'] = KafkaConsumer(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                group_id=f"{self.worker_id}-grid-operator",
                topics=['grid_operator_data'],
                auto_offset_reset='latest'
            )
            
            # Weather data consumer
            self.consumers['weather_data'] = KafkaConsumer(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                group_id=f"{self.worker_id}-weather",
                topics=['weather_data'],
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
                # Process smart meter data
                await self._process_smart_meter_data()
                
                # Process grid operator data
                await self._process_grid_operator_data()
                
                # Process weather data
                await self._process_weather_data()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in processing loop: {str(e)}")
            raise
    
    async def _process_smart_meter_data(self):
        """Process smart meter data from Kafka"""
        try:
            consumer = self.consumers.get('smart_meter_data')
            if not consumer:
                return
            
            # Consume messages
            messages = await consumer.consume_messages(max_messages=100, timeout_ms=1000)
            
            if not messages:
                return
            
            # Group messages by meter ID
            meter_data = {}
            for message in messages:
                try:
                    data = json.loads(message.value.decode('utf-8'))
                    meter_id = data.get('meter_id')
                    
                    if meter_id not in meter_data:
                        meter_data[meter_id] = []
                    
                    meter_data[meter_id].append(data)
                    
                except Exception as e:
                    logger.error(f"Error processing smart meter message: {str(e)}")
                    continue
            
            # Process each meter's data
            for meter_id, readings in meter_data.items():
                try:
                    result = await self.use_cases['ingest_smart_meter'].execute(
                        meter_id=meter_id,
                        readings_data=readings,
                        metadata={"source": "kafka", "worker": self.worker_id}
                    )
                    
                    logger.info(f"Processed {len(readings)} readings for meter {meter_id}")
                    logger.debug(f"Result: {result}")
                    
                except Exception as e:
                    logger.error(f"Error processing smart meter data for {meter_id}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error processing smart meter data: {str(e)}")
    
    async def _process_grid_operator_data(self):
        """Process grid operator data from Kafka"""
        try:
            consumer = self.consumers.get('grid_operator_data')
            if not consumer:
                return
            
            # Consume messages
            messages = await consumer.consume_messages(max_messages=100, timeout_ms=1000)
            
            if not messages:
                return
            
            # Group messages by operator ID
            operator_data = {}
            for message in messages:
                try:
                    data = json.loads(message.value.decode('utf-8'))
                    operator_id = data.get('operator_id')
                    
                    if operator_id not in operator_data:
                        operator_data[operator_id] = []
                    
                    operator_data[operator_id].append(data)
                    
                except Exception as e:
                    logger.error(f"Error processing grid operator message: {str(e)}")
                    continue
            
            # Process each operator's data
            for operator_id, statuses in operator_data.items():
                try:
                    result = await self.use_cases['process_grid_status'].execute(
                        operator_id=operator_id,
                        status_data=statuses,
                        metadata={"source": "kafka", "worker": self.worker_id}
                    )
                    
                    logger.info(f"Processed {len(statuses)} statuses for operator {operator_id}")
                    logger.debug(f"Result: {result}")
                    
                except Exception as e:
                    logger.error(f"Error processing grid operator data for {operator_id}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error processing grid operator data: {str(e)}")
    
    async def _process_weather_data(self):
        """Process weather data from Kafka"""
        try:
            consumer = self.consumers.get('weather_data')
            if not consumer:
                return
            
            # Consume messages
            messages = await consumer.consume_messages(max_messages=100, timeout_ms=1000)
            
            if not messages:
                return
            
            # Group messages by station ID
            station_data = {}
            for message in messages:
                try:
                    data = json.loads(message.value.decode('utf-8'))
                    station_id = data.get('station_id')
                    
                    if station_id not in station_data:
                        station_data[station_id] = []
                    
                    station_data[station_id].append(data)
                    
                except Exception as e:
                    logger.error(f"Error processing weather message: {str(e)}")
                    continue
            
            # Process each station's data
            for station_id, observations in station_data.items():
                try:
                    result = await self.use_cases['analyze_weather_impact'].execute(
                        station_id=station_id,
                        analysis_period_hours=24
                    )
                    
                    logger.info(f"Processed {len(observations)} observations for station {station_id}")
                    logger.debug(f"Result: {result}")
                    
                except Exception as e:
                    logger.error(f"Error processing weather data for {station_id}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error processing weather data: {str(e)}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False


async def main():
    """Main entry point for the ingestion worker"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create and start worker
        worker = IngestionWorker()
        await worker.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
