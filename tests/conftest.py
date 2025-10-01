"""
Pytest Configuration and Shared Fixtures
Global test configuration and reusable fixtures for all test modules
"""

import pytest
import asyncio
import os
import tempfile
from typing import Generator, AsyncGenerator, Dict, Any
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta

# Test configuration
pytest_plugins = ["pytest_asyncio"]

# Test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Test database URL
TEST_DATABASE_URL = "sqlite:///./test_metrify.db"

# Test Kafka configuration
TEST_KAFKA_CONFIG = {
    "bootstrap_servers": ["localhost:9092"],
    "security_protocol": "PLAINTEXT",
    "consumer_group": "test-group"
}

# Test S3 configuration
TEST_S3_CONFIG = {
    "region": "us-east-1",
    "bucket_name": "test-metrify-bucket",
    "access_key_id": "test-access-key",
    "secret_access_key": "test-secret-key"
}


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_data_dir():
    """Provide test data directory path."""
    return TEST_DATA_DIR


@pytest.fixture
def sample_smart_meter_data():
    """Sample smart meter data for testing."""
    return {
        "meter_id": "SM001",
        "location": {
            "latitude": 52.5200,
            "longitude": 13.4050,
            "address": "Berlin, Germany"
        },
        "specifications": {
            "manufacturer": "Siemens",
            "model": "SGM-1000",
            "firmware_version": "1.2.3",
            "installation_date": "2023-01-15"
        },
        "status": "ACTIVE",
        "quality_tier": "EXCELLENT",
        "metadata": {
            "customer_id": "CUST001",
            "installation_type": "residential"
        }
    }


@pytest.fixture
def sample_meter_reading_data():
    """Sample meter reading data for testing."""
    return {
        "meter_id": "SM001",
        "timestamp": datetime.utcnow(),
        "energy_consumed_kwh": 1.5,
        "power_factor": 0.95,
        "voltage_v": 230.0,
        "current_a": 6.5,
        "frequency_hz": 50.0,
        "temperature_c": 25.0,
        "quality_score": 0.95,
        "anomaly_detected": False
    }


@pytest.fixture
def sample_grid_operator_data():
    """Sample grid operator data for testing."""
    return {
        "operator_id": "TENNET",
        "name": "TenneT TSO B.V.",
        "country": "Netherlands",
        "status": "ACTIVE",
        "contact_info": {
            "email": "info@tennet.eu",
            "phone": "+31 26 373 1000"
        }
    }


@pytest.fixture
def sample_grid_status_data():
    """Sample grid status data for testing."""
    return {
        "operator_id": "TENNET",
        "timestamp": datetime.utcnow(),
        "frequency_hz": 50.02,
        "voltage_kv": 380.0,
        "load_mw": 15000.0,
        "generation_mw": 12000.0,
        "stability_score": 0.88,
        "alerts": []
    }


@pytest.fixture
def sample_weather_station_data():
    """Sample weather station data for testing."""
    return {
        "station_id": "WS001",
        "name": "Berlin Weather Station",
        "location": {
            "latitude": 52.5200,
            "longitude": 13.4050,
            "address": "Berlin, Germany"
        },
        "station_type": "AUTOMATIC",
        "status": "ACTIVE",
        "elevation_m": 34.0
    }


@pytest.fixture
def sample_weather_observation_data():
    """Sample weather observation data for testing."""
    return {
        "station_id": "WS001",
        "timestamp": datetime.utcnow(),
        "temperature_c": 15.5,
        "humidity_percent": 65.0,
        "pressure_hpa": 1013.25,
        "wind_speed_ms": 3.2,
        "wind_direction_deg": 180.0,
        "precipitation_mm": 0.0,
        "cloud_cover_percent": 30.0,
        "visibility_km": 10.0
    }


@pytest.fixture
def mock_database_session():
    """Mock database session for testing."""
    session = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.close = Mock()
    session.query = Mock()
    session.execute = Mock()
    return session


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for testing."""
    producer = AsyncMock()
    producer.send = AsyncMock()
    producer.flush = AsyncMock()
    producer.close = AsyncMock()
    return producer


@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer for testing."""
    consumer = AsyncMock()
    consumer.subscribe = AsyncMock()
    consumer.poll = AsyncMock()
    consumer.commit = AsyncMock()
    consumer.close = AsyncMock()
    return consumer


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing."""
    client = Mock()
    client.upload_file = Mock()
    client.download_file = Mock()
    client.list_objects_v2 = Mock()
    client.delete_object = Mock()
    client.generate_presigned_url = Mock()
    return client


@pytest.fixture
def mock_snowflake_client():
    """Mock Snowflake client for testing."""
    client = Mock()
    client.execute_query = AsyncMock()
    client.upload_data = AsyncMock()
    client.download_data = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_prometheus_client():
    """Mock Prometheus client for testing."""
    client = Mock()
    client.counter = Mock()
    client.histogram = Mock()
    client.gauge = Mock()
    client.summary = Mock()
    return client


@pytest.fixture
def mock_grafana_client():
    """Mock Grafana client for testing."""
    client = Mock()
    client.create_dashboard = AsyncMock()
    client.update_dashboard = AsyncMock()
    client.delete_dashboard = AsyncMock()
    client.get_dashboard = AsyncMock()
    return client


@pytest.fixture
def mock_jaeger_client():
    """Mock Jaeger client for testing."""
    client = Mock()
    client.start_span = Mock()
    client.finish_span = Mock()
    client.inject_context = Mock()
    client.extract_context = Mock()
    return client


@pytest.fixture
def mock_datadog_client():
    """Mock DataDog client for testing."""
    client = Mock()
    client.increment = Mock()
    client.gauge = Mock()
    client.histogram = Mock()
    client.event = Mock()
    return client


@pytest.fixture
def mock_airflow_client():
    """Mock Airflow client for testing."""
    client = Mock()
    client.trigger_dag = AsyncMock()
    client.get_dag_run = AsyncMock()
    client.get_task_instance = AsyncMock()
    client.pause_dag = AsyncMock()
    client.unpause_dag = AsyncMock()
    return client


@pytest.fixture
def mock_alerting_service():
    """Mock alerting service for testing."""
    service = Mock()
    service.send_alert = AsyncMock()
    service.send_notification = AsyncMock()
    service.create_alert_rule = AsyncMock()
    service.update_alert_rule = AsyncMock()
    return service


@pytest.fixture
def mock_data_quality_service():
    """Mock data quality service for testing."""
    service = Mock()
    service.validate_data = AsyncMock()
    service.calculate_quality_score = AsyncMock()
    service.detect_anomalies = AsyncMock()
    return service


@pytest.fixture
def mock_anomaly_detection_service():
    """Mock anomaly detection service for testing."""
    service = Mock()
    service.detect_anomalies = AsyncMock()
    service.train_model = AsyncMock()
    service.predict = AsyncMock()
    return service


@pytest.fixture
def mock_grid_data_service():
    """Mock grid data service for testing."""
    service = Mock()
    service.get_grid_status = AsyncMock()
    service.get_operator_info = AsyncMock()
    service.subscribe_to_updates = AsyncMock()
    return service


@pytest.fixture
def mock_weather_data_service():
    """Mock weather data service for testing."""
    service = Mock()
    service.get_current_weather = AsyncMock()
    service.get_weather_forecast = AsyncMock()
    service.get_historical_weather = AsyncMock()
    return service


@pytest.fixture
def temp_database():
    """Temporary database for testing."""
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_file.close()
    
    yield db_file.name
    
    # Cleanup
    if os.path.exists(db_file.name):
        os.unlink(db_file.name)


@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        "database": {
            "url": TEST_DATABASE_URL,
            "echo": False
        },
        "kafka": TEST_KAFKA_CONFIG,
        "s3": TEST_S3_CONFIG,
        "monitoring": {
            "prometheus": {
                "enabled": True,
                "port": 9090
            },
            "grafana": {
                "enabled": True,
                "url": "http://localhost:3000"
            },
            "jaeger": {
                "enabled": True,
                "agent_host": "localhost"
            }
        }
    }


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing."""
    client = Mock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    client.patch = AsyncMock()
    return client


@pytest.fixture
def sample_api_response():
    """Sample API response for testing."""
    return {
        "status": "success",
        "data": {
            "id": "test-id",
            "message": "Test response"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def sample_error_response():
    """Sample error response for testing."""
    return {
        "status": "error",
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Invalid input data"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def performance_test_data():
    """Performance test data generator."""
    def generate_data(count: int = 1000):
        return [
            {
                "meter_id": f"SM{i:06d}",
                "timestamp": datetime.utcnow() - timedelta(minutes=i),
                "energy_consumed_kwh": 1.0 + (i % 10),
                "power_factor": 0.9 + (i % 10) * 0.01,
                "voltage_v": 220.0 + (i % 20),
                "current_a": 5.0 + (i % 15),
                "frequency_hz": 50.0 + (i % 2) * 0.1,
                "temperature_c": 20.0 + (i % 30),
                "quality_score": 0.8 + (i % 20) * 0.01,
                "anomaly_detected": i % 100 == 0
            }
            for i in range(count)
        ]
    return generate_data


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "database: mark test as requiring database"
    )
    config.addinivalue_line(
        "markers", "kafka: mark test as requiring Kafka"
    )
    config.addinivalue_line(
        "markers", "s3: mark test as requiring S3"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker for tests that take longer than 1 second
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
