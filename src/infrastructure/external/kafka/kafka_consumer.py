"""
Kafka Consumer Implementation
Handles message consumption from Kafka topics
"""

import json
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from kafka import KafkaConsumer as PyKafkaConsumer
from kafka.errors import KafkaError
import logging

from ....core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class KafkaConsumer:
    """
    Kafka Consumer for consuming messages from topics
    
    Handles message deserialization, error handling, and processing
    """
    
    def __init__(
        self,
        bootstrap_servers: List[str],
        group_id: str = "metrify-consumer",
        client_id: str = "metrify-consumer",
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = True,
        auto_commit_interval_ms: int = 1000,
        session_timeout_ms: int = 30000,
        max_poll_records: int = 500,
        max_poll_interval_ms: int = 300000,
        request_timeout_ms: int = 30000
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.client_id = client_id
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        self.auto_commit_interval_ms = auto_commit_interval_ms
        self.session_timeout_ms = session_timeout_ms
        self.max_poll_records = max_poll_records
        self.max_poll_interval_ms = max_poll_interval_ms
        self.request_timeout_ms = request_timeout_ms
        
        self._consumer: Optional[PyKafkaConsumer] = None
        self._is_connected = False
        self._running = False
    
    async def connect(self, topics: List[str]) -> None:
        """
        Connect to Kafka cluster and subscribe to topics
        
        Args:
            topics: List of topics to subscribe to
        """
        try:
            self._consumer = PyKafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                client_id=self.client_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=self.enable_auto_commit,
                auto_commit_interval_ms=self.auto_commit_interval_ms,
                session_timeout_ms=self.session_timeout_ms,
                max_poll_records=self.max_poll_records,
                max_poll_interval_ms=self.max_poll_interval_ms,
                request_timeout_ms=self.request_timeout_ms,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None
            )
            self._is_connected = True
            logger.info(f"Connected to Kafka cluster: {self.bootstrap_servers}, topics: {topics}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            raise InfrastructureError(f"Failed to connect to Kafka: {str(e)}", service="kafka")
    
    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster"""
        self._running = False
        if self._consumer:
            self._consumer.close()
            self._consumer = None
            self._is_connected = False
            logger.info("Disconnected from Kafka cluster")
    
    async def consume_messages(
        self,
        message_handler: Callable[[str, Dict[str, Any], Dict[str, Any]], None],
        timeout_ms: int = 1000
    ) -> None:
        """
        Consume messages from subscribed topics
        
        Args:
            message_handler: Function to handle received messages
            timeout_ms: Timeout for polling messages
        """
        if not self._is_connected or not self._consumer:
            raise InfrastructureError("Consumer not connected to Kafka", service="kafka")
        
        self._running = True
        logger.info("Starting message consumption")
        
        try:
            while self._running:
                try:
                    # Poll for messages
                    message_batch = self._consumer.poll(timeout_ms=timeout_ms)
                    
                    for topic_partition, messages in message_batch.items():
                        for message in messages:
                            try:
                                # Extract message data
                                topic = topic_partition.topic
                                key = message.key
                                value = message.value
                                offset = message.offset
                                timestamp = message.timestamp
                                
                                # Log message details
                                logger.debug(
                                    f"Received message from topic {topic}, "
                                    f"partition {topic_partition.partition}, "
                                    f"offset {offset}"
                                )
                                
                                # Handle message
                                await self._handle_message(
                                    topic, key, value, message_handler
                                )
                                
                            except Exception as e:
                                logger.error(f"Error processing message: {str(e)}")
                                # Continue processing other messages
                                continue
                    
                    # Commit offsets if auto-commit is disabled
                    if not self.enable_auto_commit:
                        self._consumer.commit()
                        
                except KafkaError as e:
                    logger.error(f"Kafka error during consumption: {str(e)}")
                    await asyncio.sleep(1)  # Brief pause before retry
                except Exception as e:
                    logger.error(f"Unexpected error during consumption: {str(e)}")
                    await asyncio.sleep(1)  # Brief pause before retry
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping consumption")
        finally:
            self._running = False
            logger.info("Stopped message consumption")
    
    async def _handle_message(
        self,
        topic: str,
        key: Optional[str],
        value: Dict[str, Any],
        message_handler: Callable[[str, Dict[str, Any], Dict[str, Any]], None]
    ) -> None:
        """Handle individual message"""
        try:
            # Extract payload and metadata
            payload = value.get("payload", {})
            metadata = value.get("metadata", {})
            
            # Call message handler
            if asyncio.iscoroutinefunction(message_handler):
                await message_handler(topic, payload, metadata)
            else:
                message_handler(topic, payload, metadata)
                
        except Exception as e:
            logger.error(f"Error in message handler: {str(e)}")
            raise
    
    async def consume_smart_meter_readings(
        self,
        handler: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Consume smart meter readings from Kafka"""
        await self.connect(["smart_meter_readings"])
        
        async def message_handler(topic: str, payload: Dict[str, Any], metadata: Dict[str, Any]):
            if topic == "smart_meter_readings":
                meter_id = payload.get("meter_id")
                reading = payload.get("reading", {})
                if meter_id and reading:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(meter_id, reading)
                    else:
                        handler(meter_id, reading)
        
        await self.consume_messages(message_handler)
    
    async def consume_grid_status_updates(
        self,
        handler: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Consume grid status updates from Kafka"""
        await self.connect(["grid_status_updates"])
        
        async def message_handler(topic: str, payload: Dict[str, Any], metadata: Dict[str, Any]):
            if topic == "grid_status_updates":
                operator_id = payload.get("operator_id")
                status = payload.get("status", {})
                if operator_id and status:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(operator_id, status)
                    else:
                        handler(operator_id, status)
        
        await self.consume_messages(message_handler)
    
    async def consume_weather_observations(
        self,
        handler: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Consume weather observations from Kafka"""
        await self.connect(["weather_observations"])
        
        async def message_handler(topic: str, payload: Dict[str, Any], metadata: Dict[str, Any]):
            if topic == "weather_observations":
                station_id = payload.get("station_id")
                observation = payload.get("observation", {})
                if station_id and observation:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(station_id, observation)
                    else:
                        handler(station_id, observation)
        
        await self.consume_messages(message_handler)
    
    async def consume_domain_events(
        self,
        handler: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Consume domain events from Kafka"""
        await self.connect(["domain_events"])
        
        async def message_handler(topic: str, payload: Dict[str, Any], metadata: Dict[str, Any]):
            if topic == "domain_events":
                event_type = payload.get("event_type")
                event_data = payload.get("event_data", {})
                if event_type and event_data:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_type, event_data)
                    else:
                        handler(event_type, event_data)
        
        await self.consume_messages(message_handler)
    
    async def consume_alerts(
        self,
        handler: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Consume alerts from Kafka"""
        await self.connect(["alerts"])
        
        async def message_handler(topic: str, payload: Dict[str, Any], metadata: Dict[str, Any]):
            if topic == "alerts":
                alert_type = payload.get("alert_type")
                alert_data = payload.get("alert_data", {})
                if alert_type and alert_data:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert_type, alert_data)
                    else:
                        handler(alert_type, alert_data)
        
        await self.consume_messages(message_handler)
    
    def is_connected(self) -> bool:
        """Check if consumer is connected"""
        return self._is_connected and self._consumer is not None
    
    def is_running(self) -> bool:
        """Check if consumer is running"""
        return self._running
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics"""
        if not self._consumer:
            return {"connected": False}
        
        try:
            metrics = self._consumer.metrics()
            return {
                "connected": self._is_connected,
                "running": self._running,
                "metrics": metrics
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return {"connected": self._is_connected, "running": self._running, "error": str(e)}
