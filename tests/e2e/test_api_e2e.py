"""
End-to-end tests for the API
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
class TestAPIE2E:
    """End-to-end tests for the API"""
    
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
    
    @pytest.mark.asyncio
    async def test_complete_api_workflow(self, client, sample_meters_data):
        """Test complete API workflow from meter registration to analytics"""
        # Step 1: Register smart meters
        meter_responses = []
        for meter_data in sample_meters_data:
            response = await client.post("/api/v1/smart-meters/", json=meter_data)
            assert response.status_code == 201
            meter_responses.append(response.json())
        
        # Step 2: Verify meters are registered
        for meter_data in sample_meters_data:
            response = await client.get(f"/api/v1/smart-meters/{meter_data['meter_id']}")
            assert response.status_code == 200
            meter = response.json()
            assert meter["meter_id"] == meter_data["meter_id"]
            assert meter["status"] == meter_data["status"]
        
        # Step 3: Ingest meter readings
        base_time = datetime.utcnow()
        for i in range(5):
            for meter_data in sample_meters_data:
                reading_data = {
                    "meter_id": meter_data["meter_id"],
                    "timestamp": (base_time + timedelta(minutes=i * 15)).isoformat(),
                    "energy_consumed_kwh": 1.0 + (i * 0.1),
                    "power_factor": 0.95 + (i * 0.001),
                    "voltage_v": 230.0 + (i * 0.5),
                    "current_a": 6.5 + (i * 0.1),
                    "frequency_hz": 50.0 + (i * 0.01),
                    "temperature_c": 25.0 + (i * 0.5),
                    "quality_score": 0.95 - (i * 0.01),
                    "anomaly_detected": i == 2  # One anomalous reading
                }
                
                response = await client.post(
                    f"/api/v1/smart-meters/{meter_data['meter_id']}/readings/",
                    json=reading_data
                )
                assert response.status_code == 201
        
        # Step 4: Verify readings are stored
        for meter_data in sample_meters_data:
            response = await client.get(f"/api/v1/smart-meters/{meter_data['meter_id']}/readings/")
            assert response.status_code == 200
            readings = response.json()
            assert len(readings["readings"]) == 5
        
        # Step 5: Get meter analytics
        for meter_data in sample_meters_data:
            response = await client.get(f"/api/v1/smart-meters/{meter_data['meter_id']}/analytics/")
            assert response.status_code == 200
            analytics = response.json()
            assert "average_consumption" in analytics
            assert "consumption_trend" in analytics
            assert "anomaly_count" in analytics
        
        # Step 6: Get quality metrics
        for meter_data in sample_meters_data:
            response = await client.get(f"/api/v1/smart-meters/{meter_data['meter_id']}/quality/")
            assert response.status_code == 200
            quality = response.json()
            assert "quality_score" in quality
            assert "anomalies_detected" in quality
        
        # Step 7: Get anomalies
        for meter_data in sample_meters_data:
            response = await client.get(f"/api/v1/smart-meters/{meter_data['meter_id']}/anomalies/")
            assert response.status_code == 200
            anomalies = response.json()
            assert "anomalies" in anomalies
            if meter_data["meter_id"] == "SM001":
                assert len(anomalies["anomalies"]) == 1  # One anomalous reading
        
        # Step 8: Update meter status
        response = await client.put(
            "/api/v1/smart-meters/SM001/",
            json={"status": "MAINTENANCE"}
        )
        assert response.status_code == 200
        
        # Verify status update
        response = await client.get("/api/v1/smart-meters/SM001/")
        assert response.status_code == 200
        meter = response.json()
        assert meter["status"] == "MAINTENANCE"
    
    @pytest.mark.asyncio
    async def test_batch_operations_workflow(self, client, sample_meters_data):
        """Test batch operations workflow"""
        # Step 1: Register meters
        for meter_data in sample_meters_data:
            response = await client.post("/api/v1/smart-meters/", json=meter_data)
            assert response.status_code == 201
        
        # Step 2: Batch ingest readings
        batch_readings = []
        for i in range(10):
            for meter_data in sample_meters_data:
                reading_data = {
                    "meter_id": meter_data["meter_id"],
                    "timestamp": (datetime.utcnow() + timedelta(minutes=i * 15)).isoformat(),
                    "energy_consumed_kwh": 1.0 + (i * 0.1),
                    "power_factor": 0.95,
                    "voltage_v": 230.0,
                    "current_a": 6.5,
                    "frequency_hz": 50.0,
                    "temperature_c": 25.0,
                    "quality_score": 0.95,
                    "anomaly_detected": False
                }
                batch_readings.append(reading_data)
        
        batch_data = {"readings": batch_readings}
        response = await client.post("/api/v1/smart-meters/batch/readings/", json=batch_data)
        assert response.status_code == 200
        batch_result = response.json()
        assert batch_result["total_processed"] == 20
        assert batch_result["successful"] == 20
        assert batch_result["failed"] == 0
        
        # Step 3: Verify batch ingestion
        for meter_data in sample_meters_data:
            response = await client.get(f"/api/v1/smart-meters/{meter_data['meter_id']}/readings/")
            assert response.status_code == 200
            readings = response.json()
            assert len(readings["readings"]) == 10
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, client):
        """Test error handling workflow"""
        # Step 1: Try to get non-existent meter
        response = await client.get("/api/v1/smart-meters/NONEXISTENT")
        assert response.status_code == 404
        
        # Step 2: Try to create meter with invalid data
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
        assert response.status_code == 422
        
        # Step 3: Try to add reading to non-existent meter
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
        assert response.status_code == 404
        
        # Step 4: Try to get analytics for non-existent meter
        response = await client.get("/api/v1/smart-meters/NONEXISTENT/analytics/")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_analytics_workflow(self, client, sample_meters_data):
        """Test analytics workflow"""
        # Step 1: Register meters and ingest data
        for meter_data in sample_meters_data:
            response = await client.post("/api/v1/smart-meters/", json=meter_data)
            assert response.status_code == 201
        
        # Ingest some readings
        for i in range(5):
            for meter_data in sample_meters_data:
                reading_data = {
                    "meter_id": meter_data["meter_id"],
                    "timestamp": (datetime.utcnow() + timedelta(hours=i)).isoformat(),
                    "energy_consumed_kwh": 1.0 + (i * 0.2),
                    "power_factor": 0.95,
                    "voltage_v": 230.0,
                    "current_a": 6.5,
                    "frequency_hz": 50.0,
                    "temperature_c": 25.0,
                    "quality_score": 0.95,
                    "anomaly_detected": False
                }
                
                response = await client.post(
                    f"/api/v1/smart-meters/{meter_data['meter_id']}/readings/",
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
        
        # Step 5: Get meter-specific analytics
        for meter_data in sample_meters_data:
            response = await client.get(f"/api/v1/smart-meters/{meter_data['meter_id']}/analytics/")
            assert response.status_code == 200
            analytics = response.json()
            assert "average_consumption" in analytics
            assert "consumption_trend" in analytics
            assert "anomaly_count" in analytics
    
    @pytest.mark.asyncio
    async def test_monitoring_workflow(self, client, sample_meters_data):
        """Test monitoring workflow"""
        # Step 1: Register a meter
        meter_data = sample_meters_data[0]
        response = await client.post("/api/v1/smart-meters/", json=meter_data)
        assert response.status_code == 201
        
        # Step 2: Send anomalous reading to trigger alert
        anomalous_reading = {
            "meter_id": meter_data["meter_id"],
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
            f"/api/v1/smart-meters/{meter_data['meter_id']}/readings/",
            json=anomalous_reading
        )
        assert response.status_code == 201
        
        # Step 3: Check system health
        response = await client.get("/health")
        assert response.status_code == 200
        health = response.json()
        assert health["status"] == "healthy"
        assert "components" in health
        
        # Step 4: Check metrics
        response = await client.get("/metrics")
        assert response.status_code == 200
        metrics = response.text
        assert "smart_meter_readings_total" in metrics
        assert "data_quality_score" in metrics
        assert "anomalies_detected_total" in metrics
        
        # Step 5: Check alerts
        response = await client.get("/api/v1/alerts/")
        assert response.status_code == 200
        alerts = response.json()
        assert "alerts" in alerts
    
    @pytest.mark.asyncio
    async def test_data_export_workflow(self, client, sample_meters_data):
        """Test data export workflow"""
        # Step 1: Register meter and ingest data
        meter_data = sample_meters_data[0]
        response = await client.post("/api/v1/smart-meters/", json=meter_data)
        assert response.status_code == 201
        
        # Ingest some readings
        for i in range(5):
            reading_data = {
                "meter_id": meter_data["meter_id"],
                "timestamp": (datetime.utcnow() + timedelta(hours=i)).isoformat(),
                "energy_consumed_kwh": 1.0 + (i * 0.2),
                "power_factor": 0.95,
                "voltage_v": 230.0,
                "current_a": 6.5,
                "frequency_hz": 50.0,
                "temperature_c": 25.0,
                "quality_score": 0.95,
                "anomaly_detected": False
            }
            
            response = await client.post(
                f"/api/v1/smart-meters/{meter_data['meter_id']}/readings/",
                json=reading_data
            )
            assert response.status_code == 201
        
        # Step 2: Export data as CSV
        response = await client.get(
            f"/api/v1/smart-meters/{meter_data['meter_id']}/readings/export/",
            params={"format": "csv", "start_date": "2023-01-01", "end_date": "2023-12-31"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv"
        
        # Step 3: Export data as JSON
        response = await client.get(
            f"/api/v1/smart-meters/{meter_data['meter_id']}/readings/export/",
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
    
    @pytest.mark.asyncio
    async def test_pagination_workflow(self, client, sample_meters_data):
        """Test pagination workflow"""
        # Step 1: Register multiple meters
        for i in range(10):
            meter_data = sample_meters_data[0].copy()
            meter_data["meter_id"] = f"SM{i:03d}"
            meter_data["metadata"]["customer_id"] = f"CUST{i:03d}"
            
            response = await client.post("/api/v1/smart-meters/", json=meter_data)
            assert response.status_code == 201
        
        # Step 2: Test pagination
        response = await client.get("/api/v1/smart-meters/", params={"page": 1, "page_size": 5})
        assert response.status_code == 200
        meters = response.json()
        assert len(meters["meters"]) == 5
        assert "pagination" in meters
        assert meters["pagination"]["page"] == 1
        assert meters["pagination"]["page_size"] == 5
        
        # Step 3: Test next page
        response = await client.get("/api/v1/smart-meters/", params={"page": 2, "page_size": 5})
        assert response.status_code == 200
        meters = response.json()
        assert len(meters["meters"]) == 5
        assert meters["pagination"]["page"] == 2
        
        # Step 4: Test filtering
        response = await client.get("/api/v1/smart-meters/", params={"status": "ACTIVE"})
        assert response.status_code == 200
        meters = response.json()
        assert all(meter["status"] == "ACTIVE" for meter in meters["meters"])
        
        # Step 5: Test sorting
        response = await client.get("/api/v1/smart-meters/", params={"sort_by": "meter_id", "sort_order": "asc"})
        assert response.status_code == 200
        meters = response.json()
        meter_ids = [meter["meter_id"] for meter in meters["meters"]]
        assert meter_ids == sorted(meter_ids)
    
    @pytest.mark.asyncio
    async def test_search_workflow(self, client, sample_meters_data):
        """Test search workflow"""
        # Step 1: Register meters
        for meter_data in sample_meters_data:
            response = await client.post("/api/v1/smart-meters/", json=meter_data)
            assert response.status_code == 201
        
        # Step 2: Search by meter ID
        response = await client.get("/api/v1/smart-meters/search/", params={"q": "SM001"})
        assert response.status_code == 200
        results = response.json()
        assert len(results["results"]) == 1
        assert results["results"][0]["meter_id"] == "SM001"
        
        # Step 3: Search by location
        response = await client.get("/api/v1/smart-meters/search/", params={"q": "Berlin"})
        assert response.status_code == 200
        results = response.json()
        assert len(results["results"]) == 1
        assert "Berlin" in results["results"][0]["location"]["address"]
        
        # Step 4: Search by manufacturer
        response = await client.get("/api/v1/smart-meters/search/", params={"q": "Siemens"})
        assert response.status_code == 200
        results = response.json()
        assert len(results["results"]) == 1
        assert results["results"][0]["specifications"]["manufacturer"] == "Siemens"
        
        # Step 5: Search with no results
        response = await client.get("/api/v1/smart-meters/search/", params={"q": "NONEXISTENT"})
        assert response.status_code == 200
        results = response.json()
        assert len(results["results"]) == 0
