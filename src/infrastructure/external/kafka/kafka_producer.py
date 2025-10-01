"""
Kafka Producer Implementation
Handles message publishing to Kafka topics
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from kafka import KafkaProducer as PyKafkaProducer
from kafka.errors import KafkaError
import logging

from ....core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class KafkaProducer:
    """
    Kafka Producer for publishing messages to topics
    
    Handles message serialization, error handling, and retry logic
    """
    
    def __init__(
        self,
        bootstrap_servers: List[str],
        client_id: str = "metrify-producer",
        acks: str = "all",
        retries: int = 3,
        retry_backoff_ms: int = 100,
        request_timeout_ms: int = 30000,
        max_block_ms: int = 10000,
        compression_type: str = "gzip"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self.acks = acks
        self.retries = retries
        self.retry_backoff_ms = retry_backoff_ms
        self.request_timeout_ms = request_timeout_ms
        self.max_block_ms = max_block_ms
        self.compression_type = compression_type
        
        self._producer: Optional[PyKafkaProducer] = None
        self._is_connected = False
    
    async def connect(self) -> None:
        """Connect to Kafka cluster"""
        try:
            self._producer = PyKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                client_id=self.client_id,
                acks=self.acks,
                retries=self.retries,
                retry_backoff_ms=self.retry_backoff_ms,
                request_timeout_ms=self.request_timeout_ms,
                max_block_ms=self.max_block_ms,
                compression_type=self.compression_type,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            self._is_connected = True
            logger.info(f"Connected to Kafka cluster: {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            raise InfrastructureError(f"Failed to connect to Kafka: {str(e)}", service="kafka")
    
    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster"""
        if self._producer:
            self._producer.close()
            self._producer = None
            self._is_connected = False
            logger.info("Disconnected from Kafka cluster")
    
    async def publish_message(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
        partition: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Publish a message to a Kafka topic
        
        Args:
            topic: Kafka topic name
            message: Message payload
            key: Optional message key for partitioning
            partition: Optional partition number
            timestamp: Optional message timestamp
        """
        if not self._is_connected or not self._producer:
            await self.connect()
        
        try:
            # Add metadata to message
            enriched_message = {
                "payload": message,
                "metadata": {
                    "timestamp": (timestamp or datetime.utcnow()).isoformat(),
                    "producer_id": self.client_id,
                    "version": "1.0"
                }
            }
            
            # Publish message
            future = self._producer.send(
                topic=topic,
                value=enriched_message,
                key=key,
                partition=partition,
                timestamp_ms=int((timestamp or datetime.utcnow()).timestamp() * 1000)
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            
            logger.debug(
                f"Message published to topic {topic}, "
                f"partition {record_metadata.partition}, "
                f"offset {record_metadata.offset}"
            )
            
        except KafkaError as e:
            logger.error(f"Failed to publish message to topic {topic}: {str(e)}")
            raise InfrastructureError(f"Failed to publish message: {str(e)}", service="kafka")
        except Exception as e:
            logger.error(f"Unexpected error publishing message: {str(e)}")
            raise InfrastructureError(f"Unexpected error: {str(e)}", service="kafka")
    
    async def publish_batch(
        self,
        topic: str,
        messages: List[Dict[str, Any]],
        key_generator: Optional[callable] = None
    ) -> None:
        """
        Publish multiple messages to a Kafka topic
        
        Args:
            topic: Kafka topic name
            messages: List of message payloads
            key_generator: Optional function to generate keys for messages
        """
        if not self._is_connected or not self._producer:
            await self.connect()
        
        try:
            futures = []
            
            for i, message in enumerate(messages):
                key = key_generator(message) if key_generator else None
                
                enriched_message = {
                    "payload": message,
                    "metadata": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "producer_id": self.client_id,
                        "version": "1.0",
                        "batch_index": i
                    }
                }
                
                future = self._producer.send(
                    topic=topic,
                    value=enriched_message,
                    key=key
                )
                futures.append(future)
            
            # Wait for all messages to be sent
            for future in futures:
                record_metadata = future.get(timeout=10)
                logger.debug(
                    f"Batch message published to topic {topic}, "
                    f"partition {record_metadata.partition}, "
                    f"offset {record_metadata.offset}"
                )
            
            logger.info(f"Published {len(messages)} messages to topic {topic}")
            
        except KafkaError as e:
            logger.error(f"Failed to publish batch to topic {topic}: {str(e)}")
            raise InfrastructureError(f"Failed to publish batch: {str(e)}", service="kafka")
        except Exception as e:
            logger.error(f"Unexpected error publishing batch: {str(e)}")
            raise InfrastructureError(f"Unexpected error: {str(e)}", service="kafka")
    
    async def publish_meter_reading(self, meter_id: str, reading: Dict[str, Any]) -> None:
        """Publish smart meter reading to Kafka"""
        await self.publish_message(
            topic="smart_meter_readings",
            message={
                "meter_id": meter_id,
                "reading": reading
            },
            key=meter_id
        )
    
    async def publish_grid_status(self, operator_id: str, status: Dict[str, Any]) -> None:
        """Publish grid status to Kafka"""
        await self.publish_message(
            topic="grid_status_updates",
            message={
                "operator_id": operator_id,
                "status": status
            },
            key=operator_id
        )
    
    async def publish_weather_observation(self, station_id: str, observation: Dict[str, Any]) -> None:
        """Publish weather observation to Kafka"""
        await self.publish_message(
            topic="weather_observations",
            message={
                "station_id": station_id,
                "observation": observation
            },
            key=station_id
        )
    
    async def publish_domain_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Publish domain event to Kafka"""
        await self.publish_message(
            topic="domain_events",
            message={
                "event_type": event_type,
                "event_data": event_data
            },
            key=event_type
        )
    
    async def publish_alert(self, alert_type: str, alert_data: Dict[str, Any]) -> None:
        """Publish alert to Kafka"""
        await self.publish_message(
            topic="alerts",
            message={
                "alert_type": alert_type,
                "alert_data": alert_data
            },
            key=alert_type
        )
    
    def is_connected(self) -> bool:
        """Check if producer is connected"""
        return self._is_connected and self._producer is not None
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics"""
        if not self._producer:
            return {"connected": False}
        
        try:
            metrics = self._producer.metrics()
            return {
                "connected": self._is_connected,
                "metrics": metrics
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return {"connected": self._is_connected, "error": str(e)}
