"""
Unit tests for API endpoints
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from presentation.api.main import app
from presentation.api.v1.smart_meter_endpoints import router as smart_meter_router
from presentation.api.v1.grid_operator_endpoints import router as grid_operator_router
from presentation.api.v1.weather_endpoints import router as weather_router
from presentation.api.v1.analytics_endpoints import router as analytics_router


@pytest.fixture
def client():
    """Test client for API endpoints"""
    return TestClient(app)


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for API endpoints"""
    return {
        "meter_repository": Mock(),
        "grid_repository": Mock(),
        "weather_repository": Mock(),
        "quality_service": Mock(),
        "anomaly_service": Mock(),
        "kafka_producer": Mock(),
        "s3_client": Mock()
    }


class TestSmartMeterEndpoints:
    """Test cases for smart meter API endpoints"""
    
    def test_create_smart_meter_success(self, client, mock_dependencies):
        """Test successful smart meter creation"""
        # Arrange
        meter_data = {
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
            "metadata": {"customer_id": "CUST001"}
        }
        
        # Mock repository response
        mock_meter = Mock()
        mock_meter.meter_id.value = "SM001"
        mock_dependencies["meter_repository"].find_by_id.return_value = None
        mock_dependencies["meter_repository"].save.return_value = mock_meter
        
        # Act
        with patch.dict("presentation.api.main.dependencies", mock_dependencies):
            response = client.post("/api/v1/smart-meters/", json=meter_data)
        
        # Assert
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["meter_id"] == "SM001"
        assert response_data["status"] == "ACTIVE"
    
    def test_create_smart_meter_validation_error(self, client, mock_dependencies):
        """Test smart meter creation with validation error"""
        # Arrange
        invalid_meter_data = {
            "meter_id": "",  # Invalid empty ID
            "location": {
                "latitude": 200.0,  # Invalid latitude
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
            "metadata": {}
        }
        
        # Act
        with patch.dict("presentation.api.main.dependencies", mock_dependencies):
            response = client.post("/api/v1/smart-meters/", json=invalid_meter_data)
        
        # Assert
        assert response.status_code == 422  # Validation error
    
    def test_get_smart_meter_success(self, client, mock_dependencies):
        """Test successful smart meter retrieval"""
        # Arrange
        meter_id = "SM001"
        mock_meter = Mock()
        mock_meter.meter_id.value = meter_id
        mock_meter.status = "ACTIVE"
        mock_meter.quality_tier = "EXCELLENT"
        mock_dependencies["meter_repository"].find_by_id.return_value = mock_meter
        
        # Act
        with patch.dict("presentation.api.main.dependencies", mock_dependencies):
            response = client.get(f"/api/v1/smart-meters/{meter_id}")
        
        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["meter_id"] == meter_id
        assert response_data["status"] == "ACTIVE"
    
    def test_get_smart_meter_not_found(self, client, mock_dependencies):
        """Test smart meter retrieval when not found"""
        # Arrange
        meter_id = "NONEXISTENT"
        mock_dependencies["meter_repository"].find_by_id.return_value = None
        
        # Act
        with patch.dict("presentation.api.main.dependencies", mock_dependencies):
            response = client.get(f"/api/v1/smart-meters/{meter_id}")
        
        # Assert
        assert response.status_code == 404
        response_data = response.json()
        assert "not found" in response_data["detail"].lower()
    
    def test_add_meter_reading_success(self, client, mock_dependencies):
        """Test successful meter reading addition"""
        # Arrange
        meter_id = "SM001"
        reading_data = {
            "meter_id": meter_id,
            "timestamp": "2023-01-15T10:00:00Z",
            "energy_consumed_kwh": 1.5,
            "power_factor": 0.95,
            "voltage_v": 230.0,
            "current_a": 6.5,
            "frequency_hz": 50.0,
            "temperature_c": 25.0,
            "quality_score": 0.95,
            "anomaly_detected": False
        }
        
        mock_meter = Mock()
        mock_meter.meter_id.value = meter_id
        mock_dependencies["meter_repository"].find_by_id.return_value = mock_meter
        mock_dependencies["quality_service"].validate_reading.return_value = True
        mock_dependencies["quality_service"].calculate_quality_score.return_value = 0.95
        
        # Act
        with patch.dict("presentation.api.main.dependencies", mock_dependencies):
            response = client.post(f"/api/v1/smart-meters/{meter_id}/readings/", json=reading_data)
        
        # Assert
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["meter_id"] == meter_id
        assert response_data["energy_consumed_kwh"] == 1.5
    
    def test_batch_ingest_readings_success(self, client, mock_dependencies):
        """Test successful batch reading ingestion"""
        # Arrange
        batch_data = {
            "readings": [
                {
                    "meter_id": "SM001",
                    "timestamp": "2023-01-15T10:00:00Z",
                    "energy_consumed_kwh": 1.5,
                    "power_factor": 0.95,
                    "voltage_v": 230.0,
                    "current_a": 6.5,
                    "frequency_hz": 50.0,
                    "temperature_c": 25.0,
                    "quality_score": 0.95,
                    "anomaly_detected": False
                },
                {
                    "meter_id": "SM001",
                    "timestamp": "2023-01-15T11:00:00Z",
                    "energy_consumed_kwh": 2.0,
                    "power_factor": 0.96,
                    "voltage_v": 231.0,
                    "current_a": 7.0,
                    "frequency_hz": 50.1,
                    "temperature_c": 26.0,
                    "quality_score": 0.96,
                    "anomaly_detected": False
                }
            ]
        }
        
        mock_meter = Mock()
        mock_meter.meter_id.value = "SM001"
        mock_dependencies["meter_repository"].find_by_id.return_value = mock_meter
        mock_dependencies["quality_service"].validate_reading.return_value = True
        mock_dependencies["quality_service"].calculate_quality_score.return_value = 0.95
        
        # Act
        with patch.dict("presentation.api.main.dependencies", mock_dependencies):
            response = client.post("/api/v1/smart-meters/batch/readings/", json=batch_data)
        
        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["total_processed"] == 2
        assert response_data["successful"] == 2
        assert response_data["failed"] == 0
    
    def test_get_meter_analytics_success(self, client, mock_dependencies):
        """Test successful meter analytics retrieval"""
        # Arrange
        meter_id = "SM001"
        mock_meter = Mock()
        mock_meter.meter_id.value = meter_id
        mock_meter.calculate_average_consumption.return_value = 2.5
        mock_meter.get_readings_in_time_range.return_value = []
        mock_dependencies["meter_repository"].find_by_id.return_value = mock_meter
        
        # Act
        with patch.dict("presentation.api.main.dependencies", mock_dependencies):
            response = client.get(f"/api/v1/smart-meters/{meter_id}/analytics/")
        
        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert "average_consumption" in response_data
        assert "consumption_trend" in response_data
        assert "anomaly_count" in response_data


class TestGridOperatorEndpoints:
    """Test cases for grid operator API endpoints"""
    
    def test_create_grid_operator_success(self, client, mock_dependencies):
        """Test successful grid operator creation"""
        # Arrange
        operator_data = {
            "operator_id": "TENNET",
            "name": "TenneT TSO B.V.",
            "country": "Netherlands",
            "status": "ACTIVE",
            "contact_info": {
                "email": "info@tennet.eu",
                "phone": "+31 26 373 1000"
            }
        }
        
        mock_operator = Mock()
        mock_operator.operator_id = "TENNET"
        mock_dependencies["grid_repository"].find_by_id.return_value = None
        mock_dependencies["grid_repository"].save.return_value = mock_operator
        
        # Act
        with patch.dict("presentation.api.main.dependencies", mock_dependencies):
            response = client.post("/api/v1/grid-operators/", json=operator_data)
        
        # Assert
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["operator_id"] == "TENNET"
        assert response_data["status"] == "ACTIVE"
    
    def test_get_grid_status_success(self, client, mock_dependencies):
        """Test successful grid status retrieval"""
        # Arrange
        operator_id = "TENNET"
        mock_status = Mock()
        mock_status.operator_id = operator_id
        mock_status.frequency_hz = 50.02
        mock_status.voltage_kv = 380.0
        mock_dependencies["grid_repository"].get_latest_status.return_value = mock_status
        
        # Act
        with patch.dict("presentation.api.main.dependencies", mock_dependencies):
            response = client.get(f"/api/v1/grid-operators/{operator_id}/status/")
        
        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["operator_id"] == operator_id
        assert response_data["frequency_hz"] == 50.02


class TestWeatherEndpoints:
    """Test cases for weather API endpoints"""
    
    def test_create_weather_station_success(self, client, mock_dependencies):
        """Test successful weather station creation"""
        # Arrange
        station_data = {
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
        
        mock_station = Mock()
        mock_station.station_id = "WS001"
        mock_dependencies["weather_repository"].find_by_id.return_value = None
        mock_dependencies["weather_repository"].save.return_value = mock_station
        
        # Act
        with patch.dict("presentation.api.main.dependencies", mock_dependencies):
            response = client.post("/api/v1/weather-stations/", json=station_data)
        
        # Assert
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["station_id"] == "WS001"
        assert response_data["status"] == "ACTIVE"
    
    def test_add_weather_observation_success(self, client, mock_dependencies):
        """Test successful weather observation addition"""
        # Arrange
        station_id = "WS001"
        observation_data = {
            "station_id": station_id,
            "timestamp": "2023-01-15T10:00:00Z",
            "temperature_c": 15.5,
            "humidity_percent": 65.0,
            "pressure_hpa": 1013.25,
            "wind_speed_ms": 3.2,
            "wind_direction_deg": 180.0,
            "precipitation_mm": 0.0,
            "cloud_cover_percent": 30.0,
            "visibility_km": 10.0
        }
        
        mock_station = Mock()
        mock_station.station_id = station_id
        mock_dependencies["weather_repository"].find_by_id.return_value = mock_station
        
        # Act
        with patch.dict("presentation.api.main.dependencies", mock_dependencies):
            response = client.post(f"/api/v1/weather-stations/{station_id}/observations/", json=observation_data)
        
        # Assert
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["station_id"] == station_id
        assert response_data["temperature_c"] == 15.5


class TestAnalyticsEndpoints:
    """Test cases for analytics API endpoints"""
    
    def test_get_consumption_analytics_success(self, client, mock_dependencies):
        """Test successful consumption analytics retrieval"""
        # Arrange
        mock_analytics = {
            "total_consumption": 1000.0,
            "average_consumption": 50.0,
            "consumption_by_meter": {
                "SM001": 500.0,
                "SM002": 500.0
            }
        }
        
        # Mock analytics service
        mock_analytics_service = Mock()
        mock_analytics_service.get_consumption_analytics.return_value = mock_analytics
        
        # Act
        with patch.dict("src.presentation.api.main.dependencies", {"analytics_service": mock_analytics_service}):
            response = client.get("/api/v1/analytics/consumption/")
        
        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["total_consumption"] == 1000.0
        assert response_data["average_consumption"] == 50.0
    
    def test_get_quality_analytics_success(self, client, mock_dependencies):
        """Test successful quality analytics retrieval"""
        # Arrange
        mock_analytics = {
            "overall_quality_score": 0.95,
            "quality_by_meter": {
                "SM001": 0.96,
                "SM002": 0.94
            },
            "anomaly_summary": {
                "total_anomalies": 5,
                "anomaly_rate": 0.02
            }
        }
        
        # Mock analytics service
        mock_analytics_service = Mock()
        mock_analytics_service.get_quality_analytics.return_value = mock_analytics
        
        # Act
        with patch.dict("src.presentation.api.main.dependencies", {"analytics_service": mock_analytics_service}):
            response = client.get("/api/v1/analytics/quality/")
        
        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["overall_quality_score"] == 0.95
        assert response_data["anomaly_summary"]["total_anomalies"] == 5


class TestHealthEndpoints:
    """Test cases for health check endpoints"""
    
    def test_health_check_success(self, client):
        """Test successful health check"""
        # Act
        response = client.get("/health")
        
        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert "timestamp" in response_data
        assert "components" in response_data
    
    def test_metrics_endpoint_success(self, client):
        """Test successful metrics endpoint"""
        # Act
        response = client.get("/metrics")
        
        # Assert
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "smart_meter_readings_total" in response.text
