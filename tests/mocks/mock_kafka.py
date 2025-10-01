"""
Kafka Mock Services

Mock implementations of Kafka producers, consumers, and admin clients for testing.
"""

from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime


class MockKafkaProducer:
    """Mock Kafka producer for testing."""
    
    def __init__(self, **config):
        self.config = config
        self.messages_sent = []
        self.is_connected = True
    
    async def send(self, topic: str, value: Any, key: Optional[str] = None, 
                   partition: Optional[int] = None, timestamp_ms: Optional[int] = None) -> None:
        """Mock send message to Kafka topic."""
        message = {
            "topic": topic,
            "value": value,
            "key": key,
            "partition": partition,
            "timestamp_ms": timestamp_ms or int(datetime.utcnow().timestamp() * 1000)
        }
        self.messages_sent.append(message)
        
        # Simulate async operation
        await asyncio.sleep(0.001)
    
    async def send_batch(self, messages: List[Dict[str, Any]]) -> None:
        """Mock send batch of messages."""
        for message in messages:
            await self.send(
                topic=message["topic"],
                value=message["value"],
                key=message.get("key"),
                partition=message.get("partition")
            )
    
    async def flush(self) -> None:
        """Mock flush messages."""
        await asyncio.sleep(0.001)
    
    async def close(self) -> None:
        """Mock close producer."""
        self.is_connected = False
        await asyncio.sleep(0.001)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Mock get producer metrics."""
        return {
            "messages_sent": len(self.messages_sent),
            "is_connected": self.is_connected,
            "topics": list(set(msg["topic"] for msg in self.messages_sent))
        }


class MockKafkaConsumer:
    """Mock Kafka consumer for testing."""
    
    def __init__(self, **config):
        self.config = config
        self.subscribed_topics = []
        self.messages = []
        self.is_connected = True
        self.offset = 0
    
    async def subscribe(self, topics: List[str]) -> None:
        """Mock subscribe to topics."""
        self.subscribed_topics.extend(topics)
        await asyncio.sleep(0.001)
    
    async def poll(self, timeout_ms: int = 1000) -> Dict[str, Any]:
        """Mock poll messages from subscribed topics."""
        if not self.messages:
            await asyncio.sleep(timeout_ms / 1000)
            return {}
        
        # Return next message
        if self.offset < len(self.messages):
            message = self.messages[self.offset]
            self.offset += 1
            return {message["topic"]: [message]}
        
        return {}
    
    async def commit(self) -> None:
        """Mock commit offsets."""
        await asyncio.sleep(0.001)
    
    async def close(self) -> None:
        """Mock close consumer."""
        self.is_connected = False
        await asyncio.sleep(0.001)
    
    def add_message(self, topic: str, value: Any, key: Optional[str] = None) -> None:
        """Add a message to the consumer's message queue."""
        message = {
            "topic": topic,
            "value": value,
            "key": key,
            "offset": len(self.messages),
            "partition": 0,
            "timestamp": int(datetime.utcnow().timestamp() * 1000)
        }
        self.messages.append(message)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Mock get consumer metrics."""
        return {
            "subscribed_topics": self.subscribed_topics,
            "messages_consumed": self.offset,
            "is_connected": self.is_connected,
            "total_messages": len(self.messages)
        }


class MockKafkaAdminClient:
    """Mock Kafka admin client for testing."""
    
    def __init__(self, **config):
        self.config = config
        self.topics = {}
        self.is_connected = True
    
    async def create_topics(self, topics: List[Dict[str, Any]]) -> None:
        """Mock create topics."""
        for topic_config in topics:
            topic_name = topic_config["name"]
            self.topics[topic_name] = {
                "name": topic_name,
                "partitions": topic_config.get("num_partitions", 1),
                "replication_factor": topic_config.get("replication_factor", 1),
                "config": topic_config.get("config", {})
            }
        await asyncio.sleep(0.001)
    
    async def delete_topics(self, topics: List[str]) -> None:
        """Mock delete topics."""
        for topic_name in topics:
            if topic_name in self.topics:
                del self.topics[topic_name]
        await asyncio.sleep(0.001)
    
    async def list_topics(self) -> List[str]:
        """Mock list topics."""
        await asyncio.sleep(0.001)
        return list(self.topics.keys())
    
    async def describe_topics(self, topics: List[str]) -> Dict[str, Any]:
        """Mock describe topics."""
        await asyncio.sleep(0.001)
        result = {}
        for topic_name in topics:
            if topic_name in self.topics:
                result[topic_name] = self.topics[topic_name]
        return result
    
    async def close(self) -> None:
        """Mock close admin client."""
        self.is_connected = False
        await asyncio.sleep(0.001)


class MockKafkaCluster:
    """Mock Kafka cluster for integration testing."""
    
    def __init__(self):
        self.brokers = {}
        self.topics = {}
        self.consumers = []
        self.producers = []
    
    async def start(self) -> None:
        """Mock start cluster."""
        await asyncio.sleep(0.1)
    
    async def stop(self) -> None:
        """Mock stop cluster."""
        # Close all consumers and producers
        for consumer in self.consumers:
            await consumer.close()
        for producer in self.producers:
            await producer.close()
        await asyncio.sleep(0.1)
    
    def create_producer(self, **config) -> MockKafkaProducer:
        """Create a mock producer."""
        producer = MockKafkaProducer(**config)
        self.producers.append(producer)
        return producer
    
    def create_consumer(self, **config) -> MockKafkaConsumer:
        """Create a mock consumer."""
        consumer = MockKafkaConsumer(**config)
        self.consumers.append(consumer)
        return consumer
    
    def create_admin_client(self, **config) -> MockKafkaAdminClient:
        """Create a mock admin client."""
        return MockKafkaAdminClient(**config)
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information."""
        return {
            "brokers": len(self.brokers),
            "topics": len(self.topics),
            "consumers": len(self.consumers),
            "producers": len(self.producers)
        }


# Factory functions for easy testing
def create_mock_producer(**config) -> MockKafkaProducer:
    """Create a mock Kafka producer."""
    return MockKafkaProducer(**config)


def create_mock_consumer(**config) -> MockKafkaConsumer:
    """Create a mock Kafka consumer."""
    return MockKafkaConsumer(**config)


def create_mock_admin_client(**config) -> MockKafkaAdminClient:
    """Create a mock Kafka admin client."""
    return MockKafkaAdminClient(**config)


def create_mock_cluster() -> MockKafkaCluster:
    """Create a mock Kafka cluster."""
    return MockKafkaCluster()
