"""
Stress tests for the Metrify Smart Metering Data Pipeline
"""

import pytest
import asyncio
import time
import random
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any
from httpx import AsyncClient
from presentation.api.main import app
from tests.conftest import performance_test_data


@pytest.mark.performance
@pytest.mark.slow
class TestStressTests:
    """Stress tests for system performance under extreme load"""
    
    @pytest.fixture
    async def client(self):
        """Async HTTP client for testing"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.fixture
    def stress_test_config(self):
        """Stress test configuration"""
        return {
            "max_concurrent_users": 1000,
            "max_requests_per_user": 100,
            "ramp_up_time": 60,  # seconds
            "test_duration": 600,  # seconds
            "target_response_time": 5.0,  # seconds
            "max_error_rate": 0.05  # 5%
        }
    
    @pytest.mark.asyncio
    async def test_extreme_concurrent_load(self, client, stress_test_config):
        """Test system under extreme concurrent load"""
        # Arrange
        max_users = stress_test_config["max_concurrent_users"]
        requests_per_user = stress_test_config["max_requests_per_user"]
        
        # Register a test meter first
        meter_data = {
            "meter_id": "STRESS_TEST_METER",
            "location": {
                "latitude": 52.5200,
                "longitude": 13.4050,
                "address": "Stress Test Location"
            },
            "specifications": {
                "manufacturer": "StressTest",
                "model": "ST-1000",
                "firmware_version": "1.0.0",
                "installation_date": "2023-01-01"
            },
            "status": "ACTIVE",
            "quality_tier": "EXCELLENT",
            "metadata": {"test_type": "stress_test"}
        }
        
        response = await client.post("/api/v1/smart-meters/", json=meter_data)
        assert response.status_code == 201
        
        async def stress_worker(user_id: int):
            """Stress worker for a single user"""
            results = []
            for request_id in range(requests_per_user):
                try:
                    # Randomly choose between different operations
                    operation = random.choice([
                        "add_reading",
                        "get_meter",
                        "get_analytics",
                        "get_quality"
                    ])
                    
                    start_time = time.time()
                    
                    if operation == "add_reading":
                        reading_data = {
                            "meter_id": "STRESS_TEST_METER",
                            "timestamp": datetime.utcnow().isoformat(),
                            "energy_consumed_kwh": random.uniform(0.5, 5.0),
                            "power_factor": random.uniform(0.85, 1.0),
                            "voltage_v": random.uniform(220.0, 240.0),
                            "current_a": random.uniform(5.0, 15.0),
                            "frequency_hz": random.uniform(49.8, 50.2),
                            "temperature_c": random.uniform(15.0, 35.0),
                            "quality_score": random.uniform(0.8, 1.0),
                            "anomaly_detected": random.random() < 0.01
                        }
                        response = await client.post(
                            "/api/v1/smart-meters/STRESS_TEST_METER/readings/",
                            json=reading_data
                        )
                    elif operation == "get_meter":
                        response = await client.get("/api/v1/smart-meters/STRESS_TEST_METER/")
                    elif operation == "get_analytics":
                        response = await client.get("/api/v1/smart-meters/STRESS_TEST_METER/analytics/")
                    elif operation == "get_quality":
                        response = await client.get("/api/v1/smart-meters/STRESS_TEST_METER/quality/")
                    
                    end_time = time.time()
                    
                    results.append({
                        "user_id": user_id,
                        "request_id": request_id,
                        "operation": operation,
                        "status_code": response.status_code,
                        "response_time": end_time - start_time,
                        "success": response.status_code < 400,
                        "start_time": start_time,
                        "end_time": end_time
                    })
                    
                except Exception as e:
                    end_time = time.time()
                    results.append({
                        "user_id": user_id,
                        "request_id": request_id,
                        "operation": "error",
                        "status_code": 0,
                        "response_time": end_time - start_time,
                        "success": False,
                        "error": str(e),
                        "start_time": start_time,
                        "end_time": end_time
                    })
            
            return results
        
        # Act
        print(f"\n🚀 Starting stress test with {max_users} concurrent users...")
        start_time = time.time()
        
        # Create tasks for all users
        tasks = [stress_worker(user_id) for user_id in range(max_users)]
        
        # Execute all tasks concurrently
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Flatten results
        flat_results = []
        for user_results in all_results:
            if isinstance(user_results, list):
                flat_results.extend(user_results)
            else:
                # Handle exceptions
                flat_results.append({
                    "success": False,
                    "error": str(user_results),
                    "response_time": 0,
                    "start_time": time.time(),
                    "end_time": time.time()
                })
        
        # Calculate metrics
        total_requests = len(flat_results)
        successful_requests = len([r for r in flat_results if r.get("success", False)])
        failed_requests = total_requests - successful_requests
        
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        response_times = [r["response_time"] for r in flat_results if r.get("success", False)]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0
        p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else 0
        
        throughput = total_requests / total_time if total_time > 0 else 0
        
        # Assert
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%"
        assert error_rate <= stress_test_config["max_error_rate"], f"Error rate {error_rate:.2%} above {stress_test_config['max_error_rate']:.2%}"
        assert avg_response_time <= stress_test_config["target_response_time"], f"Average response time {avg_response_time:.2f}s above {stress_test_config['target_response_time']}s"
        
        # Print results
        print(f"\n=== Stress Test Results ===")
        print(f"Total Requests: {total_requests:,}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Error Rate: {error_rate:.2%}")
        print(f"Average Response Time: {avg_response_time:.3f}s")
        print(f"P95 Response Time: {p95_response_time:.3f}s")
        print(f"P99 Response Time: {p99_response_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_memory_stress_test(self, client):
        """Test system memory usage under stress"""
        import psutil
        import os
        
        # Arrange
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Register a test meter
        meter_data = {
            "meter_id": "MEMORY_STRESS_METER",
            "location": {
                "latitude": 52.5200,
                "longitude": 13.4050,
                "address": "Memory Stress Test Location"
            },
            "specifications": {
                "manufacturer": "MemoryTest",
                "model": "MT-1000",
                "firmware_version": "1.0.0",
                "installation_date": "2023-01-01"
            },
            "status": "ACTIVE",
            "quality_tier": "EXCELLENT",
            "metadata": {"test_type": "memory_stress_test"}
        }
        
        response = await client.post("/api/v1/smart-meters/", json=meter_data)
        assert response.status_code == 201
        
        # Act - Ingest large number of readings
        batch_size = 1000
        num_batches = 50
        
        for batch_num in range(num_batches):
            readings_data = []
            for i in range(batch_size):
                reading_data = {
                    "meter_id": "MEMORY_STRESS_METER",
                    "timestamp": (datetime.utcnow() + timedelta(minutes=i)).isoformat(),
                    "energy_consumed_kwh": random.uniform(0.5, 5.0),
                    "power_factor": random.uniform(0.85, 1.0),
                    "voltage_v": random.uniform(220.0, 240.0),
                    "current_a": random.uniform(5.0, 15.0),
                    "frequency_hz": random.uniform(49.8, 50.2),
                    "temperature_c": random.uniform(15.0, 35.0),
                    "quality_score": random.uniform(0.8, 1.0),
                    "anomaly_detected": random.random() < 0.01
                }
                readings_data.append(reading_data)
            
            batch_data = {"readings": readings_data}
            response = await client.post("/api/v1/smart-meters/batch/readings/", json=batch_data)
            assert response.status_code == 200
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            print(f"Batch {batch_num + 1}: Memory usage: {current_memory:.2f} MB (increase: {memory_increase:.2f} MB)")
            
            # Check if memory usage is reasonable
            assert memory_increase < 1000, f"Memory increase {memory_increase:.2f} MB too high at batch {batch_num + 1}"
        
        # Assert
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for 50k readings)
        assert total_memory_increase < 500, f"Total memory increase {total_memory_increase:.2f} MB too high"
        
        print(f"\n=== Memory Stress Test Results ===")
        print(f"Initial Memory: {initial_memory:.2f} MB")
        print(f"Final Memory: {final_memory:.2f} MB")
        print(f"Total Memory Increase: {total_memory_increase:.2f} MB")
        print(f"Memory per Reading: {total_memory_increase / (batch_size * num_batches) * 1024:.2f} KB")
    
    @pytest.mark.asyncio
    async def test_database_stress_test(self, client):
        """Test database performance under stress"""
        # Arrange
        num_meters = 100
        readings_per_meter = 1000
        
        # Register multiple meters
        meter_ids = []
        for i in range(num_meters):
            meter_data = {
                "meter_id": f"DB_STRESS_METER_{i:03d}",
                "location": {
                    "latitude": 52.5200 + random.uniform(-0.1, 0.1),
                    "longitude": 13.4050 + random.uniform(-0.1, 0.1),
                    "address": f"Database Stress Test Location {i}"
                },
                "specifications": {
                    "manufacturer": "DBTest",
                    "model": f"DBT-{i:03d}",
                    "firmware_version": "1.0.0",
                    "installation_date": "2023-01-01"
                },
                "status": "ACTIVE",
                "quality_tier": "EXCELLENT",
                "metadata": {"test_type": "database_stress_test", "meter_index": i}
            }
            
            response = await client.post("/api/v1/smart-meters/", json=meter_data)
            assert response.status_code == 201
            meter_ids.append(meter_data["meter_id"])
        
        # Act - Ingest readings for all meters
        start_time = time.time()
        
        for meter_id in meter_ids:
            readings_data = []
            for i in range(readings_per_meter):
                reading_data = {
                    "meter_id": meter_id,
                    "timestamp": (datetime.utcnow() + timedelta(minutes=i)).isoformat(),
                    "energy_consumed_kwh": random.uniform(0.5, 5.0),
                    "power_factor": random.uniform(0.85, 1.0),
                    "voltage_v": random.uniform(220.0, 240.0),
                    "current_a": random.uniform(5.0, 15.0),
                    "frequency_hz": random.uniform(49.8, 50.2),
                    "temperature_c": random.uniform(15.0, 35.0),
                    "quality_score": random.uniform(0.8, 1.0),
                    "anomaly_detected": random.random() < 0.01
                }
                readings_data.append(reading_data)
            
            batch_data = {"readings": readings_data}
            response = await client.post("/api/v1/smart-meters/batch/readings/", json=batch_data)
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Test database queries under load
        query_start_time = time.time()
        
        # Test concurrent analytics queries
        async def query_analytics(meter_id: str):
            response = await client.get(f"/api/v1/smart-meters/{meter_id}/analytics/")
            return response.status_code == 200
        
        # Execute queries concurrently
        query_tasks = [query_analytics(meter_id) for meter_id in meter_ids]
        query_results = await asyncio.gather(*query_tasks, return_exceptions=True)
        
        query_end_time = time.time()
        query_time = query_end_time - query_start_time
        
        # Assert
        total_readings = num_meters * readings_per_meter
        ingestion_throughput = total_readings / total_time
        
        successful_queries = len([r for r in query_results if r is True])
        query_success_rate = successful_queries / len(query_results)
        
        assert ingestion_throughput > 1000, f"Ingestion throughput {ingestion_throughput:.2f} readings/s too low"
        assert query_success_rate >= 0.95, f"Query success rate {query_success_rate:.2%} below 95%"
        assert query_time < 30, f"Query time {query_time:.2f}s too high"
        
        print(f"\n=== Database Stress Test Results ===")
        print(f"Meters: {num_meters}")
        print(f"Readings per Meter: {readings_per_meter}")
        print(f"Total Readings: {total_readings:,}")
        print(f"Ingestion Time: {total_time:.2f}s")
        print(f"Ingestion Throughput: {ingestion_throughput:.2f} readings/s")
        print(f"Query Time: {query_time:.2f}s")
        print(f"Query Success Rate: {query_success_rate:.2%}")
    
    @pytest.mark.asyncio
    async def test_api_rate_limiting_stress_test(self, client):
        """Test API rate limiting under stress"""
        # Arrange
        meter_data = {
            "meter_id": "RATE_LIMIT_METER",
            "location": {
                "latitude": 52.5200,
                "longitude": 13.4050,
                "address": "Rate Limit Test Location"
            },
            "specifications": {
                "manufacturer": "RateTest",
                "model": "RT-1000",
                "firmware_version": "1.0.0",
                "installation_date": "2023-01-01"
            },
            "status": "ACTIVE",
            "quality_tier": "EXCELLENT",
            "metadata": {"test_type": "rate_limit_stress_test"}
        }
        
        response = await client.post("/api/v1/smart-meters/", json=meter_data)
        assert response.status_code == 201
        
        # Act - Send requests as fast as possible
        max_requests = 10000
        requests_per_second = 1000
        
        async def rate_limit_worker():
            """Worker that sends requests as fast as possible"""
            results = []
            for i in range(max_requests // 10):  # Divide by 10 to avoid overwhelming
                try:
                    start_time = time.time()
                    response = await client.get("/api/v1/smart-meters/RATE_LIMIT_METER/")
                    end_time = time.time()
                    
                    results.append({
                        "request_id": i,
                        "status_code": response.status_code,
                        "response_time": end_time - start_time,
                        "success": response.status_code < 400,
                        "timestamp": time.time()
                    })
                    
                    # Small delay to avoid overwhelming the system
                    await asyncio.sleep(0.001)
                    
                except Exception as e:
                    results.append({
                        "request_id": i,
                        "status_code": 0,
                        "response_time": 0,
                        "success": False,
                        "error": str(e),
                        "timestamp": time.time()
                    })
            
            return results
        
        # Execute rate limit test
        start_time = time.time()
        results = await rate_limit_worker()
        end_time = time.time()
        
        # Analyze results
        total_requests = len(results)
        successful_requests = len([r for r in results if r.get("success", False)])
        failed_requests = total_requests - successful_requests
        
        # Count rate limited requests (429 status code)
        rate_limited_requests = len([r for r in results if r.get("status_code") == 429])
        
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        rate_limit_rate = rate_limited_requests / total_requests if total_requests > 0 else 0
        
        response_times = [r["response_time"] for r in results if r.get("success", False)]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        # Assert
        assert success_rate >= 0.90, f"Success rate {success_rate:.2%} below 90%"
        assert rate_limit_rate <= 0.10, f"Rate limit rate {rate_limit_rate:.2%} above 10%"
        assert avg_response_time <= 2.0, f"Average response time {avg_response_time:.2f}s above 2s"
        
        print(f"\n=== Rate Limiting Stress Test Results ===")
        print(f"Total Requests: {total_requests:,}")
        print(f"Successful Requests: {successful_requests:,}")
        print(f"Failed Requests: {failed_requests:,}")
        print(f"Rate Limited Requests: {rate_limited_requests:,}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Rate Limit Rate: {rate_limit_rate:.2%}")
        print(f"Average Response Time: {avg_response_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_system_recovery_stress_test(self, client):
        """Test system recovery under stress"""
        # Arrange
        meter_data = {
            "meter_id": "RECOVERY_METER",
            "location": {
                "latitude": 52.5200,
                "longitude": 13.4050,
                "address": "Recovery Test Location"
            },
            "specifications": {
                "manufacturer": "RecoveryTest",
                "model": "RT-1000",
                "firmware_version": "1.0.0",
                "installation_date": "2023-01-01"
            },
            "status": "ACTIVE",
            "quality_tier": "EXCELLENT",
            "metadata": {"test_type": "recovery_stress_test"}
        }
        
        response = await client.post("/api/v1/smart-meters/", json=meter_data)
        assert response.status_code == 201
        
        # Act - Simulate system stress and recovery
        stress_cycles = 5
        recovery_times = []
        
        for cycle in range(stress_cycles):
            print(f"\n🔄 Stress Cycle {cycle + 1}/{stress_cycles}")
            
            # Phase 1: Apply stress
            stress_start = time.time()
            
            # Send burst of requests
            burst_requests = 1000
            tasks = []
            for i in range(burst_requests):
                task = client.get("/api/v1/smart-meters/RECOVERY_METER/")
                tasks.append(task)
            
            # Execute burst
            burst_results = await asyncio.gather(*tasks, return_exceptions=True)
            stress_end = time.time()
            
            # Phase 2: Measure recovery
            recovery_start = time.time()
            
            # Send normal requests and measure response time
            normal_requests = 100
            normal_tasks = []
            for i in range(normal_requests):
                task = client.get("/api/v1/smart-meters/RECOVERY_METER/")
                normal_tasks.append(task)
            
            normal_results = await asyncio.gather(*normal_tasks, return_exceptions=True)
            recovery_end = time.time()
            
            recovery_time = recovery_end - recovery_start
            recovery_times.append(recovery_time)
            
            # Calculate metrics
            burst_success_rate = len([r for r in burst_results if not isinstance(r, Exception)]) / len(burst_results)
            normal_success_rate = len([r for r in normal_results if not isinstance(r, Exception)]) / len(normal_results)
            
            print(f"Burst Success Rate: {burst_success_rate:.2%}")
            print(f"Normal Success Rate: {normal_success_rate:.2%}")
            print(f"Recovery Time: {recovery_time:.2f}s")
            
            # Wait between cycles
            await asyncio.sleep(5)
        
        # Assert
        avg_recovery_time = statistics.mean(recovery_times)
        max_recovery_time = max(recovery_times)
        
        assert avg_recovery_time <= 10.0, f"Average recovery time {avg_recovery_time:.2f}s too high"
        assert max_recovery_time <= 20.0, f"Max recovery time {max_recovery_time:.2f}s too high"
        
        print(f"\n=== System Recovery Stress Test Results ===")
        print(f"Stress Cycles: {stress_cycles}")
        print(f"Average Recovery Time: {avg_recovery_time:.2f}s")
        print(f"Max Recovery Time: {max_recovery_time:.2f}s")
        print(f"Recovery Times: {[f'{t:.2f}s' for t in recovery_times]}")
