"""
Load tests for the Metrify Smart Metering Data Pipeline
"""

import pytest
import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any
from httpx import AsyncClient
from presentation.api.main import app
from tests.conftest import performance_test_data


@pytest.mark.performance
@pytest.mark.slow
class TestLoadTests:
    """Load tests for system performance validation"""
    
    @pytest.fixture
    async def client(self):
        """Async HTTP client for testing"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.fixture
    def load_test_config(self):
        """Load test configuration"""
        return {
            "concurrent_users": 100,
            "requests_per_user": 10,
            "ramp_up_time": 30,  # seconds
            "test_duration": 300,  # seconds
            "target_response_time": 2.0,  # seconds
            "max_error_rate": 0.01  # 1%
        }
    
    @pytest.fixture
    def sample_meter_data(self):
        """Sample meter data for load testing"""
        return {
            "meter_id": "LOAD_TEST_METER",
            "location": {
                "latitude": 52.5200,
                "longitude": 13.4050,
                "address": "Load Test Location"
            },
            "specifications": {
                "manufacturer": "LoadTest",
                "model": "LT-1000",
                "firmware_version": "1.0.0",
                "installation_date": "2023-01-01"
            },
            "status": "ACTIVE",
            "quality_tier": "EXCELLENT",
            "metadata": {
                "test_type": "load_test"
            }
        }
    
    @pytest.mark.asyncio
    async def test_concurrent_meter_registration(self, client, load_test_config, sample_meter_data):
        """Test concurrent meter registration performance"""
        # Arrange
        concurrent_users = load_test_config["concurrent_users"]
        requests_per_user = load_test_config["requests_per_user"]
        
        async def register_meter(user_id: int, request_id: int):
            """Register a meter for a specific user and request"""
            meter_data = sample_meter_data.copy()
            meter_data["meter_id"] = f"LOAD_TEST_METER_{user_id}_{request_id}"
            meter_data["metadata"]["user_id"] = user_id
            meter_data["metadata"]["request_id"] = request_id
            
            start_time = time.time()
            response = await client.post("/api/v1/smart-meters/", json=meter_data)
            end_time = time.time()
            
            return {
                "user_id": user_id,
                "request_id": request_id,
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 201
            }
        
        # Act
        start_time = time.time()
        tasks = []
        
        for user_id in range(concurrent_users):
            for request_id in range(requests_per_user):
                task = register_meter(user_id, request_id)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Assert
        total_requests = concurrent_users * requests_per_user
        total_time = end_time - start_time
        throughput = total_requests / total_time
        
        successful_requests = [r for r in results if isinstance(r, dict) and r["success"]]
        failed_requests = [r for r in results if isinstance(r, dict) and not r["success"]]
        
        success_rate = len(successful_requests) / total_requests
        error_rate = len(failed_requests) / total_requests
        
        response_times = [r["response_time"] for r in successful_requests]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0
        p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else 0
        
        # Performance assertions
        assert success_rate >= 0.99, f"Success rate {success_rate:.2%} below 99%"
        assert error_rate <= load_test_config["max_error_rate"], f"Error rate {error_rate:.2%} above {load_test_config['max_error_rate']:.2%}"
        assert avg_response_time <= load_test_config["target_response_time"], f"Average response time {avg_response_time:.2f}s above {load_test_config['target_response_time']}s"
        assert p95_response_time <= load_test_config["target_response_time"] * 2, f"P95 response time {p95_response_time:.2f}s too high"
        
        # Print performance metrics
        print(f"\n=== Meter Registration Load Test Results ===")
        print(f"Total Requests: {total_requests}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Error Rate: {error_rate:.2%}")
        print(f"Average Response Time: {avg_response_time:.3f}s")
        print(f"P95 Response Time: {p95_response_time:.3f}s")
        print(f"P99 Response Time: {p99_response_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_reading_ingestion(self, client, load_test_config, sample_meter_data):
        """Test concurrent meter reading ingestion performance"""
        # Arrange
        # First register a meter
        response = await client.post("/api/v1/smart-meters/", json=sample_meter_data)
        assert response.status_code == 201
        
        concurrent_users = load_test_config["concurrent_users"]
        requests_per_user = load_test_config["requests_per_user"]
        
        async def ingest_reading(user_id: int, request_id: int):
            """Ingest a meter reading for a specific user and request"""
            reading_data = {
                "meter_id": sample_meter_data["meter_id"],
                "timestamp": datetime.utcnow().isoformat(),
                "energy_consumed_kwh": 1.0 + (request_id * 0.1),
                "power_factor": 0.95,
                "voltage_v": 230.0,
                "current_a": 6.5,
                "frequency_hz": 50.0,
                "temperature_c": 25.0,
                "quality_score": 0.95,
                "anomaly_detected": False
            }
            
            start_time = time.time()
            response = await client.post(
                f"/api/v1/smart-meters/{sample_meter_data['meter_id']}/readings/",
                json=reading_data
            )
            end_time = time.time()
            
            return {
                "user_id": user_id,
                "request_id": request_id,
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 201
            }
        
        # Act
        start_time = time.time()
        tasks = []
        
        for user_id in range(concurrent_users):
            for request_id in range(requests_per_user):
                task = ingest_reading(user_id, request_id)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Assert
        total_requests = concurrent_users * requests_per_user
        total_time = end_time - start_time
        throughput = total_requests / total_time
        
        successful_requests = [r for r in results if isinstance(r, dict) and r["success"]]
        failed_requests = [r for r in results if isinstance(r, dict) and not r["success"]]
        
        success_rate = len(successful_requests) / total_requests
        error_rate = len(failed_requests) / total_requests
        
        response_times = [r["response_time"] for r in successful_requests]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0
        
        # Performance assertions
        assert success_rate >= 0.99, f"Success rate {success_rate:.2%} below 99%"
        assert error_rate <= load_test_config["max_error_rate"], f"Error rate {error_rate:.2%} above {load_test_config['max_error_rate']:.2%}"
        assert avg_response_time <= load_test_config["target_response_time"], f"Average response time {avg_response_time:.2f}s above {load_test_config['target_response_time']}s"
        
        # Print performance metrics
        print(f"\n=== Reading Ingestion Load Test Results ===")
        print(f"Total Requests: {total_requests}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Error Rate: {error_rate:.2%}")
        print(f"Average Response Time: {avg_response_time:.3f}s")
        print(f"P95 Response Time: {p95_response_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_batch_ingestion_performance(self, client, load_test_config, sample_meter_data):
        """Test batch ingestion performance"""
        # Arrange
        # First register a meter
        response = await client.post("/api/v1/smart-meters/", json=sample_meter_data)
        assert response.status_code == 201
        
        batch_sizes = [10, 50, 100, 500, 1000]
        results = {}
        
        for batch_size in batch_sizes:
            # Generate batch data
            readings_data = []
            for i in range(batch_size):
                reading_data = {
                    "meter_id": sample_meter_data["meter_id"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "energy_consumed_kwh": 1.0 + (i * 0.1),
                    "power_factor": 0.95,
                    "voltage_v": 230.0,
                    "current_a": 6.5,
                    "frequency_hz": 50.0,
                    "temperature_c": 25.0,
                    "quality_score": 0.95,
                    "anomaly_detected": False
                }
                readings_data.append(reading_data)
            
            batch_data = {"readings": readings_data}
            
            # Act
            start_time = time.time()
            response = await client.post("/api/v1/smart-meters/batch/readings/", json=batch_data)
            end_time = time.time()
            
            # Record results
            results[batch_size] = {
                "response_time": end_time - start_time,
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "throughput": batch_size / (end_time - start_time) if end_time > start_time else 0
            }
        
        # Assert
        for batch_size, result in results.items():
            assert result["success"], f"Batch size {batch_size} failed with status {result['status_code']}"
            assert result["response_time"] <= 10.0, f"Batch size {batch_size} took too long: {result['response_time']:.2f}s"
        
        # Print performance metrics
        print(f"\n=== Batch Ingestion Performance Results ===")
        for batch_size, result in results.items():
            print(f"Batch Size: {batch_size:4d} | Response Time: {result['response_time']:.3f}s | Throughput: {result['throughput']:.2f} readings/s")
    
    @pytest.mark.asyncio
    async def test_concurrent_analytics_queries(self, client, load_test_config, sample_meter_data):
        """Test concurrent analytics queries performance"""
        # Arrange
        # First register a meter and ingest some data
        response = await client.post("/api/v1/smart-meters/", json=sample_meter_data)
        assert response.status_code == 201
        
        # Ingest some readings
        for i in range(100):
            reading_data = {
                "meter_id": sample_meter_data["meter_id"],
                "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                "energy_consumed_kwh": 1.0 + (i * 0.1),
                "power_factor": 0.95,
                "voltage_v": 230.0,
                "current_a": 6.5,
                "frequency_hz": 50.0,
                "temperature_c": 25.0,
                "quality_score": 0.95,
                "anomaly_detected": False
            }
            
            response = await client.post(
                f"/api/v1/smart-meters/{sample_meter_data['meter_id']}/readings/",
                json=reading_data
            )
            assert response.status_code == 201
        
        concurrent_users = 50  # Reduced for analytics queries
        requests_per_user = 5
        
        async def query_analytics(user_id: int, request_id: int):
            """Execute analytics query for a specific user and request"""
            query_types = [
                "/api/v1/smart-meters/LOAD_TEST_METER/analytics/",
                "/api/v1/smart-meters/LOAD_TEST_METER/quality/",
                "/api/v1/smart-meters/LOAD_TEST_METER/anomalies/",
                "/api/v1/analytics/consumption/",
                "/api/v1/analytics/quality/"
            ]
            
            query_url = query_types[request_id % len(query_types)]
            
            start_time = time.time()
            response = await client.get(query_url)
            end_time = time.time()
            
            return {
                "user_id": user_id,
                "request_id": request_id,
                "query_type": query_url,
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 200
            }
        
        # Act
        start_time = time.time()
        tasks = []
        
        for user_id in range(concurrent_users):
            for request_id in range(requests_per_user):
                task = query_analytics(user_id, request_id)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Assert
        total_requests = concurrent_users * requests_per_user
        total_time = end_time - start_time
        throughput = total_requests / total_time
        
        successful_requests = [r for r in results if isinstance(r, dict) and r["success"]]
        failed_requests = [r for r in results if isinstance(r, dict) and not r["success"]]
        
        success_rate = len(successful_requests) / total_requests
        error_rate = len(failed_requests) / total_requests
        
        response_times = [r["response_time"] for r in successful_requests]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0
        
        # Performance assertions
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%"
        assert error_rate <= 0.05, f"Error rate {error_rate:.2%} above 5%"
        assert avg_response_time <= 5.0, f"Average response time {avg_response_time:.2f}s above 5s"
        
        # Print performance metrics
        print(f"\n=== Analytics Queries Load Test Results ===")
        print(f"Total Requests: {total_requests}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Error Rate: {error_rate:.2%}")
        print(f"Average Response Time: {avg_response_time:.3f}s")
        print(f"P95 Response Time: {p95_response_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, client, load_test_config, sample_meter_data):
        """Test memory usage under load"""
        import psutil
        import os
        
        # Arrange
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # First register a meter
        response = await client.post("/api/v1/smart-meters/", json=sample_meter_data)
        assert response.status_code == 201
        
        # Act - Ingest large number of readings
        batch_size = 1000
        num_batches = 10
        
        for batch_num in range(num_batches):
            readings_data = []
            for i in range(batch_size):
                reading_data = {
                    "meter_id": sample_meter_data["meter_id"],
                    "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                    "energy_consumed_kwh": 1.0 + (i * 0.1),
                    "power_factor": 0.95,
                    "voltage_v": 230.0,
                    "current_a": 6.5,
                    "frequency_hz": 50.0,
                    "temperature_c": 25.0,
                    "quality_score": 0.95,
                    "anomaly_detected": False
                }
                readings_data.append(reading_data)
            
            batch_data = {"readings": readings_data}
            response = await client.post("/api/v1/smart-meters/batch/readings/", json=batch_data)
            assert response.status_code == 200
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            print(f"Batch {batch_num + 1}: Memory usage: {current_memory:.2f} MB (increase: {memory_increase:.2f} MB)")
        
        # Assert
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for 10k readings)
        assert total_memory_increase < 500, f"Memory increase {total_memory_increase:.2f} MB too high"
        
        print(f"\n=== Memory Usage Test Results ===")
        print(f"Initial Memory: {initial_memory:.2f} MB")
        print(f"Final Memory: {final_memory:.2f} MB")
        print(f"Total Memory Increase: {total_memory_increase:.2f} MB")
        print(f"Memory per Reading: {total_memory_increase / (batch_size * num_batches) * 1024:.2f} KB")
