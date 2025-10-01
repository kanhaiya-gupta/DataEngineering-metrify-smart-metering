"""
Pytest Configuration and Shared Fixtures
Global test configuration and reusable fixtures for all test modules
"""

import pytest
import asyncio
import os
import tempfile
import pandas as pd
import numpy as np
from typing import Generator, AsyncGenerator, Dict, Any, List
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta

# Test configuration
pytest_plugins = ["pytest_asyncio"]

# Test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Test database URL
TEST_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/metrify_test"

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


# =============================================================================
# ML/AI Integration Fixtures (Phase 1)
# =============================================================================

@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client for ML model testing."""
    client = Mock()
    client.create_experiment = Mock(return_value="exp_123")
    client.start_run = Mock()
    client.log_metric = Mock()
    client.log_param = Mock()
    client.log_model = Mock()
    client.end_run = Mock()
    client.get_experiment = Mock(return_value={"experiment_id": "exp_123"})
    client.search_runs = Mock(return_value=[])
    return client


@pytest.fixture
def mock_tensorflow_model():
    """Mock TensorFlow model for testing."""
    model = Mock()
    model.predict = Mock(return_value=np.array([[0.8, 0.2]]))
    model.fit = Mock()
    model.evaluate = Mock(return_value=[0.1, 0.95])
    model.save = Mock()
    model.load_weights = Mock()
    model.compile = Mock()
    return model


@pytest.fixture
def mock_feature_store():
    """Mock feature store for testing."""
    store = Mock()
    store.get_features = AsyncMock(return_value=pd.DataFrame())
    store.create_feature_view = AsyncMock()
    store.get_historical_features = AsyncMock(return_value=pd.DataFrame())
    store.get_online_features = AsyncMock(return_value=pd.DataFrame())
    return store


@pytest.fixture
def sample_ml_training_data():
    """Sample ML training data for testing."""
    return pd.DataFrame({
        'meter_id': ['SM001', 'SM002', 'SM003', 'SM004', 'SM005'],
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='H'),
        'energy_consumed': [100.5, 150.2, 200.8, 175.3, 225.7],
        'temperature': [22.5, 23.1, 21.8, 24.2, 22.9],
        'humidity': [65.2, 68.1, 62.5, 70.3, 66.8],
        'anomaly_label': [0, 0, 1, 0, 0]
    })


@pytest.fixture
def sample_model_metrics():
    """Sample ML model metrics for testing."""
    return {
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.88,
        "f1_score": 0.90,
        "auc_roc": 0.94,
        "confusion_matrix": [[85, 5], [3, 7]]
    }


# =============================================================================
# Advanced Analytics Fixtures (Phase 2)
# =============================================================================

@pytest.fixture
def mock_analytics_engine():
    """Mock analytics engine for testing."""
    engine = Mock()
    engine.calculate_forecast = AsyncMock(return_value=pd.DataFrame())
    engine.detect_anomalies = AsyncMock(return_value=pd.DataFrame())
    engine.analyze_trends = AsyncMock(return_value=pd.DataFrame())
    engine.calculate_quality_metrics = AsyncMock(return_value={})
    return engine


@pytest.fixture
def sample_forecast_data():
    """Sample forecast data for testing."""
    dates = pd.date_range('2024-01-01', periods=24, freq='H')
    return pd.DataFrame({
        'timestamp': dates,
        'actual': 100 + np.random.randn(24) * 10,
        'forecast': 100 + np.random.randn(24) * 8,
        'confidence_lower': 90 + np.random.randn(24) * 5,
        'confidence_upper': 110 + np.random.randn(24) * 5
    })


@pytest.fixture
def sample_anomaly_data():
    """Sample anomaly detection data for testing."""
    normal_data = np.random.normal(100, 10, 100)
    anomaly_data = np.array([200, 300, 50, 400])  # Clear anomalies
    return np.concatenate([normal_data, anomaly_data])


@pytest.fixture
def sample_quality_metrics():
    """Sample data quality metrics for testing."""
    return {
        "completeness": 0.95,
        "accuracy": 0.92,
        "consistency": 0.88,
        "timeliness": 0.90,
        "validity": 0.94,
        "overall_score": 0.92
    }


# =============================================================================
# Event-Driven Architecture Fixtures (Phase 3)
# =============================================================================

@pytest.fixture
def mock_event_store():
    """Mock event store for event sourcing tests."""
    store = Mock()
    store.append_events = Mock()
    store.get_events_for_aggregate = Mock(return_value=[])
    store.get_current_version = Mock(return_value=0)
    store.get_all_events = Mock(return_value=[])
    return store


@pytest.fixture
def mock_command_handler():
    """Mock command handler for CQRS tests."""
    handler = Mock()
    handler.handle = Mock(return_value={"status": "success"})
    return handler


@pytest.fixture
def mock_query_handler():
    """Mock query handler for CQRS tests."""
    handler = Mock()
    handler.handle = Mock(return_value={"data": []})
    return handler


@pytest.fixture
def sample_domain_event():
    """Sample domain event for testing."""
    return {
        "event_id": "evt_123",
        "event_type": "SmartMeterCreated",
        "aggregate_id": "SM001",
        "timestamp": datetime.utcnow(),
        "version": 1,
        "data": {
            "meter_id": "SM001",
            "location": "Berlin",
            "status": "ACTIVE"
        }
    }


# =============================================================================
# Performance Optimization Fixtures (Phase 3)
# =============================================================================

@pytest.fixture
def mock_cache_client():
    """Mock cache client for performance testing."""
    client = Mock()
    client.get = Mock(return_value=None)
    client.set = Mock(return_value=True)
    client.delete = Mock(return_value=True)
    client.clear = Mock()
    client.keys = Mock(return_value=[])
    return client


@pytest.fixture
def mock_query_optimizer():
    """Mock query optimizer for testing."""
    optimizer = Mock()
    optimizer.optimize_query = Mock(return_value="SELECT * FROM optimized_table")
    optimizer.analyze_query_performance = Mock(return_value={})
    optimizer.recommend_indexes = Mock(return_value=[])
    return optimizer


@pytest.fixture
def mock_index_optimizer():
    """Mock index optimizer for testing."""
    optimizer = Mock()
    optimizer.analyze_index_usage = Mock(return_value={})
    optimizer.generate_recommendations = Mock(return_value=[])
    optimizer.create_index = Mock(return_value=True)
    optimizer.drop_index = Mock(return_value=True)
    return optimizer


@pytest.fixture
def mock_stream_joiner():
    """Mock stream joiner for testing."""
    joiner = Mock()
    joiner.add_left_record = Mock(return_value=[])
    joiner.add_right_record = Mock(return_value=[])
    joiner.get_metrics = Mock(return_value={})
    return joiner


@pytest.fixture
def mock_real_time_analytics():
    """Mock real-time analytics for testing."""
    analytics = Mock()
    analytics.add_record = Mock(return_value=[])
    analytics.calculate_windowed_aggregation = Mock(return_value=[])
    analytics.detect_anomalies = Mock(return_value=[])
    analytics.analyze_trends = Mock(return_value=[])
    return analytics


# =============================================================================
# Multi-Cloud Infrastructure Fixtures (Phase 3)
# =============================================================================

@pytest.fixture
def mock_aws_client():
    """Mock AWS client for testing."""
    client = Mock()
    client.upload_file = Mock()
    client.download_file = Mock()
    client.list_objects_v2 = Mock(return_value={'Contents': []})
    client.create_bucket = Mock()
    client.delete_bucket = Mock()
    return client


@pytest.fixture
def mock_azure_client():
    """Mock Azure client for testing."""
    client = Mock()
    client.upload_blob = Mock()
    client.download_blob = Mock()
    client.list_blobs = Mock(return_value=[])
    client.create_container = Mock()
    client.delete_container = Mock()
    return client


@pytest.fixture
def mock_gcp_client():
    """Mock GCP client for testing."""
    client = Mock()
    client.upload_file = Mock()
    client.download_file = Mock()
    client.list_objects = Mock(return_value=[])
    client.create_bucket = Mock()
    client.delete_bucket = Mock()
    return client


# =============================================================================
# Data Governance Fixtures (Phase 2)
# =============================================================================

@pytest.fixture
def mock_data_catalog():
    """Mock data catalog for governance testing."""
    catalog = Mock()
    catalog.register_dataset = Mock()
    catalog.get_dataset = Mock(return_value={})
    catalog.search_datasets = Mock(return_value=[])
    catalog.update_lineage = Mock()
    return catalog


@pytest.fixture
def sample_data_lineage():
    """Sample data lineage for testing."""
    return {
        "source": "smart_meter_raw",
        "transformations": ["data_cleaning", "feature_engineering"],
        "destination": "analytics_warehouse",
        "dependencies": ["weather_data", "grid_data"],
        "last_updated": datetime.utcnow()
    }


# =============================================================================
# Custom Assertions
# =============================================================================

def assert_performance_requirement(actual_time: float, max_time: float, operation: str):
    """Assert that performance requirement is met."""
    assert actual_time <= max_time, f"{operation} took {actual_time:.3f}s, expected <= {max_time:.3f}s"


def assert_data_quality(data: pd.DataFrame, required_columns: List[str], null_threshold: float = 0.1):
    """Assert data quality requirements."""
    # Check required columns
    missing_columns = set(required_columns) - set(data.columns)
    assert not missing_columns, f"Missing required columns: {missing_columns}"
    
    # Check null ratio
    null_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
    assert null_ratio <= null_threshold, f"Null ratio {null_ratio:.3f} exceeds threshold {null_threshold}"


def assert_ml_model_performance(accuracy: float, min_accuracy: float = 0.8):
    """Assert ML model performance requirements."""
    assert accuracy >= min_accuracy, f"Model accuracy {accuracy:.3f} below minimum {min_accuracy}"


def assert_event_consistency(events: List[Dict], expected_count: int):
    """Assert event sourcing consistency."""
    assert len(events) == expected_count, f"Expected {expected_count} events, got {len(events)}"
    for event in events:
        assert "event_id" in event, "Event missing event_id"
        assert "event_type" in event, "Event missing event_type"
        assert "timestamp" in event, "Event missing timestamp"


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
