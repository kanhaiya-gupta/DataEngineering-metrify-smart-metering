"""
Integration tests for Kafka producer and consumer
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
from src.infrastructure.external.kafka.kafka_consumer import KafkaConsumer
from src.infrastructure.external.kafka.message_serializer import MessageSerializer


@pytest.mark.integration
@pytest.mark.kafka
class TestKafkaIntegration:
    """Integration tests for Kafka components"""
    
    @pytest.fixture
    def kafka_config(self):
        """Kafka configuration for testing"""
        return {
            "bootstrap_servers": ["localhost:9092"],
            "security_protocol": "PLAINTEXT",
            "sasl_mechanism": None,
            "sasl_username": None,
            "sasl_password": None
        }
    
    @pytest.fixture
    def producer_config(self, kafka_config):
        """Producer configuration"""
        return {
            **kafka_config,
            "acks": "all",
            "retries": 3,
            "batch_size": 16384,
            "linger_ms": 10,
            "compression_type": "snappy"
        }
    
    @pytest.fixture
    def consumer_config(self, kafka_config):
        """Consumer configuration"""
        return {
            **kafka_config,
            "group_id": "test-group",
            "auto_offset_reset": "earliest",
            "enable_auto_commit": True,
            "auto_commit_interval_ms": 1000
        }
    
    @pytest.fixture
    def message_serializer(self):
        """Message serializer instance"""
        return MessageSerializer()
    
    @pytest.fixture
    def sample_message(self):
        """Sample message for testing"""
        return {
            "meter_id": "SM001",
            "timestamp": datetime.utcnow().isoformat(),
            "energy_consumed_kwh": 1.5,
            "power_factor": 0.95,
            "voltage_v": 230.0,
            "current_a": 6.5,
            "frequency_hz": 50.0,
            "temperature_c": 25.0,
            "quality_score": 0.95,
            "anomaly_detected": False
        }
    
    @pytest.mark.asyncio
    async def test_producer_send_message_success(self, producer_config, message_serializer, sample_message):
        """Test successful message sending"""
        # Mock Kafka producer
        with patch('kafka.KafkaProducer') as mock_producer_class:
            mock_producer = Mock()
            mock_producer.send.return_value = Mock()
            mock_producer.flush.return_value = None
            mock_producer.close.return_value = None
            mock_producer_class.return_value = mock_producer
            
            # Create producer instance
            producer = KafkaProducer(producer_config, message_serializer)
            
            # Act
            await producer.send_message("smart-meter-data", "SM001", sample_message)
            
            # Assert
            mock_producer.send.assert_called_once()
            mock_producer.flush.assert_called_once()
            mock_producer.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_producer_send_message_with_partition(self, producer_config, message_serializer, sample_message):
        """Test message sending with specific partition"""
        # Mock Kafka producer
        with patch('kafka.KafkaProducer') as mock_producer_class:
            mock_producer = Mock()
            mock_producer.send.return_value = Mock()
            mock_producer.flush.return_value = None
            mock_producer.close.return_value = None
            mock_producer_class.return_value = mock_producer
            
            # Create producer instance
            producer = KafkaProducer(producer_config, message_serializer)
            
            # Act
            await producer.send_message("smart-meter-data", "SM001", sample_message, partition=0)
            
            # Assert
            mock_producer.send.assert_called_once()
            call_args = mock_producer.send.call_args
            assert call_args[1]["partition"] == 0
    
    @pytest.mark.asyncio
    async def test_producer_send_batch_messages_success(self, producer_config, message_serializer):
        """Test successful batch message sending"""
        # Mock Kafka producer
        with patch('kafka.KafkaProducer') as mock_producer_class:
            mock_producer = Mock()
            mock_producer.send.return_value = Mock()
            mock_producer.flush.return_value = None
            mock_producer.close.return_value = None
            mock_producer_class.return_value = mock_producer
            
            # Create producer instance
            producer = KafkaProducer(producer_config, message_serializer)
            
            # Prepare batch messages
            messages = [
                {"meter_id": "SM001", "energy_consumed_kwh": 1.5},
                {"meter_id": "SM002", "energy_consumed_kwh": 2.0},
                {"meter_id": "SM003", "energy_consumed_kwh": 1.8}
            ]
            
            # Act
            await producer.send_batch_messages("smart-meter-data", messages)
            
            # Assert
            assert mock_producer.send.call_count == 3
            mock_producer.flush.assert_called_once()
            mock_producer.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_consumer_consume_messages_success(self, consumer_config, message_serializer):
        """Test successful message consumption"""
        # Mock Kafka consumer
        with patch('kafka.KafkaConsumer') as mock_consumer_class:
            mock_consumer = Mock()
            mock_consumer.__iter__.return_value = [
                Mock(
                    topic="smart-meter-data",
                    partition=0,
                    offset=1,
                    key=b"SM001",
                    value=b'{"meter_id": "SM001", "energy_consumed_kwh": 1.5}',
                    timestamp=1234567890
                ),
                Mock(
                    topic="smart-meter-data",
                    partition=0,
                    offset=2,
                    key=b"SM002",
                    value=b'{"meter_id": "SM002", "energy_consumed_kwh": 2.0}',
                    timestamp=1234567891
                )
            ]
            mock_consumer.commit.return_value = None
            mock_consumer.close.return_value = None
            mock_consumer_class.return_value = mock_consumer
            
            # Create consumer instance
            consumer = KafkaConsumer(consumer_config, message_serializer)
            
            # Act
            messages = []
            async for message in consumer.consume_messages("smart-meter-data"):
                messages.append(message)
                if len(messages) >= 2:
                    break
            
            # Assert
            assert len(messages) == 2
            assert messages[0]["meter_id"] == "SM001"
            assert messages[1]["meter_id"] == "SM002"
            mock_consumer.commit.assert_called()
            mock_consumer.close.assert_called()
    
    @pytest.mark.asyncio
    async def test_consumer_consume_messages_with_timeout(self, consumer_config, message_serializer):
        """Test message consumption with timeout"""
        # Mock Kafka consumer
        with patch('kafka.KafkaConsumer') as mock_consumer_class:
            mock_consumer = Mock()
            mock_consumer.__iter__.return_value = []
            mock_consumer.commit.return_value = None
            mock_consumer.close.return_value = None
            mock_consumer_class.return_value = mock_consumer
            
            # Create consumer instance
            consumer = KafkaConsumer(consumer_config, message_serializer)
            
            # Act
            messages = []
            try:
                async for message in consumer.consume_messages("smart-meter-data", timeout_ms=1000):
                    messages.append(message)
            except asyncio.TimeoutError:
                pass
            
            # Assert
            assert len(messages) == 0
            mock_consumer.close.assert_called()
    
    @pytest.mark.asyncio
    async def test_message_serialization(self, message_serializer, sample_message):
        """Test message serialization and deserialization"""
        # Act
        serialized = message_serializer.serialize(sample_message)
        deserialized = message_serializer.deserialize(serialized)
        
        # Assert
        assert isinstance(serialized, bytes)
        assert deserialized == sample_message
    
    @pytest.mark.asyncio
    async def test_message_serialization_with_compression(self, message_serializer, sample_message):
        """Test message serialization with compression"""
        # Act
        serialized = message_serializer.serialize(sample_message, compress=True)
        deserialized = message_serializer.deserialize(serialized, decompress=True)
        
        # Assert
        assert isinstance(serialized, bytes)
        assert deserialized == sample_message
        assert len(serialized) < len(json.dumps(sample_message).encode())
    
    @pytest.mark.asyncio
    async def test_producer_error_handling(self, producer_config, message_serializer, sample_message):
        """Test producer error handling"""
        # Mock Kafka producer with error
        with patch('kafka.KafkaProducer') as mock_producer_class:
            mock_producer = Mock()
            mock_producer.send.side_effect = Exception("Kafka error")
            mock_producer.close.return_value = None
            mock_producer_class.return_value = mock_producer
            
            # Create producer instance
            producer = KafkaProducer(producer_config, message_serializer)
            
            # Act & Assert
            with pytest.raises(Exception, match="Kafka error"):
                await producer.send_message("smart-meter-data", "SM001", sample_message)
            
            mock_producer.close.assert_called()
    
    @pytest.mark.asyncio
    async def test_consumer_error_handling(self, consumer_config, message_serializer):
        """Test consumer error handling"""
        # Mock Kafka consumer with error
        with patch('kafka.KafkaConsumer') as mock_consumer_class:
            mock_consumer = Mock()
            mock_consumer.__iter__.side_effect = Exception("Kafka error")
            mock_consumer.close.return_value = None
            mock_consumer_class.return_value = mock_consumer
            
            # Create consumer instance
            consumer = KafkaConsumer(consumer_config, message_serializer)
            
            # Act & Assert
            with pytest.raises(Exception, match="Kafka error"):
                async for message in consumer.consume_messages("smart-meter-data"):
                    pass
            
            mock_consumer.close.assert_called()
    
    @pytest.mark.asyncio
    async def test_producer_connection_retry(self, producer_config, message_serializer, sample_message):
        """Test producer connection retry mechanism"""
        # Mock Kafka producer with connection retry
        with patch('kafka.KafkaProducer') as mock_producer_class:
            # First call fails, second call succeeds
            mock_producer_class.side_effect = [
                Exception("Connection failed"),
                Mock()
            ]
            
            # Create producer instance
            producer = KafkaProducer(producer_config, message_serializer)
            
            # Act
            await producer.send_message("smart-meter-data", "SM001", sample_message)
            
            # Assert
            assert mock_producer_class.call_count == 2
    
    @pytest.mark.asyncio
    async def test_consumer_group_rebalancing(self, consumer_config, message_serializer):
        """Test consumer group rebalancing"""
        # Mock Kafka consumer with rebalancing
        with patch('kafka.KafkaConsumer') as mock_consumer_class:
            mock_consumer = Mock()
            mock_consumer.__iter__.return_value = []
            mock_consumer.commit.return_value = None
            mock_consumer.close.return_value = None
            mock_consumer_class.return_value = mock_consumer
            
            # Create consumer instance
            consumer = KafkaConsumer(consumer_config, message_serializer)
            
            # Act
            messages = []
            async for message in consumer.consume_messages("smart-meter-data", timeout_ms=1000):
                messages.append(message)
            
            # Assert
            assert len(messages) == 0
            mock_consumer.close.assert_called()
    
    @pytest.mark.asyncio
    async def test_message_ordering_guarantee(self, producer_config, message_serializer):
        """Test message ordering guarantee"""
        # Mock Kafka producer
        with patch('kafka.KafkaProducer') as mock_producer_class:
            mock_producer = Mock()
            mock_producer.send.return_value = Mock()
            mock_producer.flush.return_value = None
            mock_producer.close.return_value = None
            mock_producer_class.return_value = mock_producer
            
            # Create producer instance
            producer = KafkaProducer(producer_config, message_serializer)
            
            # Send messages with same key to ensure ordering
            messages = [
                {"meter_id": "SM001", "sequence": 1},
                {"meter_id": "SM001", "sequence": 2},
                {"meter_id": "SM001", "sequence": 3}
            ]
            
            # Act
            for message in messages:
                await producer.send_message("smart-meter-data", "SM001", message)
            
            # Assert
            assert mock_producer.send.call_count == 3
            # All messages should have the same key for ordering
            for call in mock_producer.send.call_args_list:
                assert call[1]["key"] == "SM001"
    
    @pytest.mark.asyncio
    async def test_consumer_offset_management(self, consumer_config, message_serializer):
        """Test consumer offset management"""
        # Mock Kafka consumer
        with patch('kafka.KafkaConsumer') as mock_consumer_class:
            mock_consumer = Mock()
            mock_consumer.__iter__.return_value = [
                Mock(
                    topic="smart-meter-data",
                    partition=0,
                    offset=1,
                    key=b"SM001",
                    value=b'{"meter_id": "SM001", "energy_consumed_kwh": 1.5}',
                    timestamp=1234567890
                )
            ]
            mock_consumer.commit.return_value = None
            mock_consumer.close.return_value = None
            mock_consumer_class.return_value = mock_consumer
            
            # Create consumer instance
            consumer = KafkaConsumer(consumer_config, message_serializer)
            
            # Act
            messages = []
            async for message in consumer.consume_messages("smart-meter-data"):
                messages.append(message)
                break  # Process one message and commit
            
            # Assert
            assert len(messages) == 1
            mock_consumer.commit.assert_called()
            mock_consumer.close.assert_called()
