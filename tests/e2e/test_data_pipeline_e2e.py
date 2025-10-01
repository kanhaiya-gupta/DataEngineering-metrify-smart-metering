"""
End-to-end tests for the complete data pipeline
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from httpx import AsyncClient
from presentation.api.main import app
from tests.conftest import sample_smart_meter_data, sample_meter_reading_data


@pytest.mark.e2e
@pytest.mark.slow
class TestDataPipelineE2E:
    """End-to-end tests for the complete data pipeline"""
    
    @pytest.fixture
    async def client(self):
        """Async HTTP client for testing"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.fixture
    def sample_meters_data(self) -> List[Dict[str, Any]]:
        """Sample meters data for E2E testing"""
        return [
            {
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
            },
            {
                "meter_id": "SM002",
                "location": {
                    "latitude": 48.1351,
                    "longitude": 11.5820,
                    "address": "Munich, Germany"
                },
                "specifications": {
                    "manufacturer": "ABB",
                    "model": "ABM-2000",
                    "firmware_version": "2.1.0",
                    "installation_date": "2023-02-15"
                },
                "status": "ACTIVE",
                "quality_tier": "GOOD",
                "metadata": {
                    "customer_id": "CUST002",
                    "installation_type": "commercial"
                }
            }
        ]
    
    @pytest.fixture
    def sample_readings_data(self) -> List[Dict[str, Any]]:
        """Sample readings data for E2E testing"""
        base_time = datetime.utcnow()
        readings = []
        
        for i in range(10):
            reading = {
                "meter_id": "SM001",
                "timestamp": (base_time + timedelta(minutes=i * 15)).isoformat(),
                "energy_consumed_kwh": 1.0 + (i * 0.1),
                "power_factor": 0.95 + (i * 0.001),
                "voltage_v": 230.0 + (i * 0.5),
                "current_a": 6.5 + (i * 0.1),
                "frequency_hz": 50.0 + (i * 0.01),
                "temperature_c": 25.0 + (i * 0.5),
                "quality_score": 0.95 - (i * 0.01),
                "anomaly_detected": i == 5  # One anomalous reading
            }
            readings.append(reading)
        
        return readings
    
    @pytest.mark.asyncio
    async def test_complete_data_pipeline_flow(self, client, sample_meters_data, sample_readings_data):
        """Test complete data pipeline from meter registration to analytics"""
        # Step 1: Register smart meters
        meter_responses = []
        for meter_data in sample_meters_data:
            response = await client.post("/api/v1/smart-meters/", json=meter_data)
            assert response.status_code == 201
            meter_responses.append(response.json())
        
        # Step 2: Ingest meter readings
        for reading_data in sample_readings_data:
            response = await client.post(
                f"/api/v1/smart-meters/{reading_data['meter_id']}/readings/",
                json=reading_data
            )
            assert response.status_code == 201
        
        # Step 3: Verify meter readings are stored
        response = await client.get(f"/api/v1/smart-meters/SM001/readings/")
        assert response.status_code == 200
        readings = response.json()
        assert len(readings["readings"]) == 10
        
        # Step 4: Check data quality
        response = await client.get(f"/api/v1/smart-meters/SM001/quality/")
        assert response.status_code == 200
        quality_data = response.json()
        assert "quality_score" in quality_data
        assert "anomalies_detected" in quality_data
        
        # Step 5: Get analytics
        response = await client.get(f"/api/v1/smart-meters/SM001/analytics/")
        assert response.status_code == 200
        analytics = response.json()
        assert "average_consumption" in analytics
        assert "consumption_trend" in analytics
        assert "anomaly_count" in analytics
        
        # Step 6: Verify anomaly detection
        response = await client.get(f"/api/v1/smart-meters/SM001/anomalies/")
        assert response.status_code == 200
        anomalies = response.json()
        assert len(anomalies["anomalies"]) == 1  # One anomalous reading
        
        # Step 7: Get meter status
        response = await client.get(f"/api/v1/smart-meters/SM001/")
        assert response.status_code == 200
        meter = response.json()
        assert meter["status"] == "ACTIVE"
        assert meter["quality_tier"] == "EXCELLENT"
    
    @pytest.mark.asyncio
    async def test_batch_ingestion_flow(self, client, sample_meters_data, sample_readings_data):
        """Test batch ingestion of meter readings"""
        # Step 1: Register a meter
        meter_data = sample_meters_data[0]
        response = await client.post("/api/v1/smart-meters/", json=meter_data)
        assert response.status_code == 201
        
        # Step 2: Batch ingest readings
        batch_data = {
            "readings": sample_readings_data
        }
        response = await client.post("/api/v1/smart-meters/batch/readings/", json=batch_data)
        assert response.status_code == 200
        batch_result = response.json()
        assert batch_result["total_processed"] == 10
        assert batch_result["successful"] == 10
        assert batch_result["failed"] == 0
        
        # Step 3: Verify all readings are stored
        response = await client.get(f"/api/v1/smart-meters/SM001/readings/")
        assert response.status_code == 200
        readings = response.json()
        assert len(readings["readings"]) == 10
    
    @pytest.mark.asyncio
    async def test_error_handling_flow(self, client):
        """Test error handling in the data pipeline"""
        # Step 1: Try to create meter with invalid data
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
        
        response = await client.post("/api/v1/smart-meters/", json=invalid_meter_data)
        assert response.status_code == 422  # Validation error
        
        # Step 2: Try to add reading to non-existent meter
        reading_data = {
            "meter_id": "NONEXISTENT",
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
        
        response = await client.post(
            "/api/v1/smart-meters/NONEXISTENT/readings/",
            json=reading_data
        )
        assert response.status_code == 404  # Meter not found
    
    @pytest.mark.asyncio
    async def test_real_time_processing_flow(self, client, sample_meters_data):
        """Test real-time processing of meter data"""
        # Step 1: Register a meter
        meter_data = sample_meters_data[0]
        response = await client.post("/api/v1/smart-meters/", json=meter_data)
        assert response.status_code == 201
        
        # Step 2: Send real-time readings
        real_time_readings = []
        for i in range(5):
            reading_data = {
                "meter_id": "SM001",
                "timestamp": datetime.utcnow().isoformat(),
                "energy_consumed_kwh": 1.0 + (i * 0.2),
                "power_factor": 0.95,
                "voltage_v": 230.0,
                "current_a": 6.5,
                "frequency_hz": 50.0,
                "temperature_c": 25.0,
                "quality_score": 0.95,
                "anomaly_detected": False
            }
            real_time_readings.append(reading_data)
            
            response = await client.post(
                "/api/v1/smart-meters/SM001/readings/",
                json=reading_data
            )
            assert response.status_code == 201
        
        # Step 3: Verify real-time processing
        response = await client.get("/api/v1/smart-meters/SM001/readings/realtime/")
        assert response.status_code == 200
        real_time_data = response.json()
        assert len(real_time_data["readings"]) == 5
    
    @pytest.mark.asyncio
    async def test_analytics_and_reporting_flow(self, client, sample_meters_data, sample_readings_data):
        """Test analytics and reporting functionality"""
        # Step 1: Register meters and ingest data
        for meter_data in sample_meters_data:
            response = await client.post("/api/v1/smart-meters/", json=meter_data)
            assert response.status_code == 201
        
        for reading_data in sample_readings_data:
            response = await client.post(
                f"/api/v1/smart-meters/{reading_data['meter_id']}/readings/",
                json=reading_data
            )
            assert response.status_code == 201
        
        # Step 2: Get consumption analytics
        response = await client.get("/api/v1/analytics/consumption/")
        assert response.status_code == 200
        consumption_analytics = response.json()
        assert "total_consumption" in consumption_analytics
        assert "average_consumption" in consumption_analytics
        assert "consumption_by_meter" in consumption_analytics
        
        # Step 3: Get quality analytics
        response = await client.get("/api/v1/analytics/quality/")
        assert response.status_code == 200
        quality_analytics = response.json()
        assert "overall_quality_score" in quality_analytics
        assert "quality_by_meter" in quality_analytics
        assert "anomaly_summary" in quality_analytics
        
        # Step 4: Get performance analytics
        response = await client.get("/api/v1/analytics/performance/")
        assert response.status_code == 200
        performance_analytics = response.json()
        assert "processing_latency" in performance_analytics
        assert "throughput" in performance_analytics
        assert "error_rate" in performance_analytics
        
        # Step 5: Generate report
        report_data = {
            "report_type": "daily_summary",
            "start_date": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "meters": ["SM001", "SM002"]
        }
        
        response = await client.post("/api/v1/analytics/reports/", json=report_data)
        assert response.status_code == 200
        report = response.json()
        assert "report_id" in report
        assert "generated_at" in report
        assert "summary" in report
    
    @pytest.mark.asyncio
    async def test_monitoring_and_alerting_flow(self, client, sample_meters_data):
        """Test monitoring and alerting functionality"""
        # Step 1: Register a meter
        meter_data = sample_meters_data[0]
        response = await client.post("/api/v1/smart-meters/", json=meter_data)
        assert response.status_code == 201
        
        # Step 2: Send anomalous reading to trigger alert
        anomalous_reading = {
            "meter_id": "SM001",
            "timestamp": datetime.utcnow().isoformat(),
            "energy_consumed_kwh": 100.0,  # Anomalous high consumption
            "power_factor": 0.95,
            "voltage_v": 230.0,
            "current_a": 6.5,
            "frequency_hz": 50.0,
            "temperature_c": 25.0,
            "quality_score": 0.95,
            "anomaly_detected": True
        }
        
        response = await client.post(
            "/api/v1/smart-meters/SM001/readings/",
            json=anomalous_reading
        )
        assert response.status_code == 201
        
        # Step 3: Check alerts
        response = await client.get("/api/v1/alerts/")
        assert response.status_code == 200
        alerts = response.json()
        assert len(alerts["alerts"]) >= 1
        
        # Step 4: Check system health
        response = await client.get("/health")
        assert response.status_code == 200
        health = response.json()
        assert health["status"] == "healthy"
        assert "components" in health
        
        # Step 5: Check metrics
        response = await client.get("/metrics")
        assert response.status_code == 200
        metrics = response.text
        assert "smart_meter_readings_total" in metrics
        assert "data_quality_score" in metrics
        assert "anomalies_detected_total" in metrics
    
    @pytest.mark.asyncio
    async def test_data_export_flow(self, client, sample_meters_data, sample_readings_data):
        """Test data export functionality"""
        # Step 1: Register meter and ingest data
        meter_data = sample_meters_data[0]
        response = await client.post("/api/v1/smart-meters/", json=meter_data)
        assert response.status_code == 201
        
        for reading_data in sample_readings_data:
            response = await client.post(
                f"/api/v1/smart-meters/{reading_data['meter_id']}/readings/",
                json=reading_data
            )
            assert response.status_code == 201
        
        # Step 2: Export data as CSV
        response = await client.get(
            "/api/v1/smart-meters/SM001/readings/export/",
            params={"format": "csv", "start_date": "2023-01-01", "end_date": "2023-12-31"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv"
        
        # Step 3: Export data as JSON
        response = await client.get(
            "/api/v1/smart-meters/SM001/readings/export/",
            params={"format": "json", "start_date": "2023-01-01", "end_date": "2023-12-31"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Step 4: Export analytics data
        response = await client.get("/api/v1/analytics/export/")
        assert response.status_code == 200
        analytics_export = response.json()
        assert "consumption_data" in analytics_export
        assert "quality_metrics" in analytics_export
        assert "anomaly_data" in analytics_export
