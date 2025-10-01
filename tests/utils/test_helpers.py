"""
Test helper functions and utilities
"""

import asyncio
import time
import random
import string
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Performance metrics for testing"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float
    success_rate: float
    error_rate: float


class TestDataGenerator:
    """Utility class for generating test data"""
    
    @staticmethod
    def generate_meter_id(prefix: str = "SM") -> str:
        """Generate a random meter ID"""
        suffix = ''.join(random.choices(string.digits, k=6))
        return f"{prefix}{suffix}"
    
    @staticmethod
    def generate_customer_id(prefix: str = "CUST") -> str:
        """Generate a random customer ID"""
        suffix = ''.join(random.choices(string.digits, k=6))
        return f"{prefix}{suffix}"
    
    @staticmethod
    def generate_location() -> Dict[str, Any]:
        """Generate a random location"""
        # German cities with coordinates
        cities = [
            {"name": "Berlin", "lat": 52.5200, "lon": 13.4050},
            {"name": "Munich", "lat": 48.1351, "lon": 11.5820},
            {"name": "Hamburg", "lat": 53.5511, "lon": 9.9937},
            {"name": "Cologne", "lat": 50.9375, "lon": 6.9603},
            {"name": "Frankfurt", "lat": 50.1109, "lon": 8.6821},
            {"name": "Stuttgart", "lat": 48.7758, "lon": 9.1829},
            {"name": "Düsseldorf", "lat": 51.2277, "lon": 6.7735},
            {"name": "Dortmund", "lat": 51.5136, "lon": 7.4653},
            {"name": "Essen", "lat": 51.4556, "lon": 7.0116},
            {"name": "Leipzig", "lat": 51.3397, "lon": 12.3731}
        ]
        
        city = random.choice(cities)
        return {
            "latitude": city["lat"] + random.uniform(-0.01, 0.01),
            "longitude": city["lon"] + random.uniform(-0.01, 0.01),
            "address": f"{city['name']}, Germany"
        }
    
    @staticmethod
    def generate_meter_specifications() -> Dict[str, Any]:
        """Generate random meter specifications"""
        manufacturers = ["Siemens", "ABB", "Schneider Electric", "Landis+Gyr", "Itron"]
        models = ["SGM-1000", "ABM-2000", "SE-3000", "LG-4000", "IT-5000"]
        
        return {
            "manufacturer": random.choice(manufacturers),
            "model": random.choice(models),
            "firmware_version": f"{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
            "installation_date": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d")
        }
    
    @staticmethod
    def generate_meter_reading(meter_id: str, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a random meter reading"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        return {
            "meter_id": meter_id,
            "timestamp": timestamp.isoformat(),
            "energy_consumed_kwh": round(random.uniform(0.5, 5.0), 2),
            "power_factor": round(random.uniform(0.85, 1.0), 3),
            "voltage_v": round(random.uniform(220.0, 240.0), 1),
            "current_a": round(random.uniform(5.0, 15.0), 1),
            "frequency_hz": round(random.uniform(49.8, 50.2), 2),
            "temperature_c": round(random.uniform(15.0, 35.0), 1),
            "quality_score": round(random.uniform(0.8, 1.0), 3),
            "anomaly_detected": random.random() < 0.05  # 5% chance of anomaly
        }
    
    @staticmethod
    def generate_batch_readings(meter_id: str, count: int, 
                              start_time: Optional[datetime] = None,
                              interval_minutes: int = 15) -> List[Dict[str, Any]]:
        """Generate a batch of meter readings"""
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=count * interval_minutes / 60)
        
        readings = []
        for i in range(count):
            timestamp = start_time + timedelta(minutes=i * interval_minutes)
            reading = TestDataGenerator.generate_meter_reading(meter_id, timestamp)
            readings.append(reading)
        
        return readings
    
    @staticmethod
    def generate_smart_meter() -> Dict[str, Any]:
        """Generate a complete smart meter"""
        meter_id = TestDataGenerator.generate_meter_id()
        
        return {
            "meter_id": meter_id,
            "location": TestDataGenerator.generate_location(),
            "specifications": TestDataGenerator.generate_meter_specifications(),
            "status": random.choice(["ACTIVE", "INACTIVE", "MAINTENANCE"]),
            "quality_tier": random.choice(["EXCELLENT", "GOOD", "FAIR", "POOR"]),
            "metadata": {
                "customer_id": TestDataGenerator.generate_customer_id(),
                "installation_type": random.choice(["residential", "commercial", "industrial"])
            }
        }


class PerformanceTestHelper:
    """Helper class for performance testing"""
    
    @staticmethod
    def calculate_metrics(results: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Calculate performance metrics from test results"""
        total_requests = len(results)
        successful_requests = len([r for r in results if r.get("success", False)])
        failed_requests = total_requests - successful_requests
        
        response_times = [r.get("response_time", 0) for r in results if r.get("success", False)]
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            response_times_sorted = sorted(response_times)
            p95_index = int(len(response_times_sorted) * 0.95)
            p99_index = int(len(response_times_sorted) * 0.99)
            p95_response_time = response_times_sorted[p95_index] if p95_index < len(response_times_sorted) else 0
            p99_response_time = response_times_sorted[p99_index] if p99_index < len(response_times_sorted) else 0
        else:
            avg_response_time = 0
            p95_response_time = 0
            p99_response_time = 0
        
        total_time = max([r.get("end_time", 0) for r in results]) - min([r.get("start_time", 0) for r in results])
        throughput = total_requests / total_time if total_time > 0 else 0
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        return PerformanceMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            throughput=throughput,
            success_rate=success_rate,
            error_rate=error_rate
        )
    
    @staticmethod
    def print_metrics(metrics: PerformanceMetrics, test_name: str = "Performance Test"):
        """Print performance metrics in a formatted way"""
        print(f"\n=== {test_name} Results ===")
        print(f"Total Requests: {metrics.total_requests}")
        print(f"Successful Requests: {metrics.successful_requests}")
        print(f"Failed Requests: {metrics.failed_requests}")
        print(f"Total Time: {metrics.total_time:.2f}s")
        print(f"Throughput: {metrics.throughput:.2f} requests/second")
        print(f"Success Rate: {metrics.success_rate:.2%}")
        print(f"Error Rate: {metrics.error_rate:.2%}")
        print(f"Average Response Time: {metrics.avg_response_time:.3f}s")
        print(f"P95 Response Time: {metrics.p95_response_time:.3f}s")
        print(f"P99 Response Time: {metrics.p99_response_time:.3f}s")
    
    @staticmethod
    async def run_concurrent_requests(request_func, concurrent_users: int, 
                                    requests_per_user: int, **kwargs) -> List[Dict[str, Any]]:
        """Run concurrent requests and return results"""
        async def run_user_requests(user_id: int):
            """Run requests for a single user"""
            user_results = []
            for request_id in range(requests_per_user):
                start_time = time.time()
                try:
                    result = await request_func(user_id, request_id, **kwargs)
                    end_time = time.time()
                    user_results.append({
                        "user_id": user_id,
                        "request_id": request_id,
                        "success": True,
                        "response_time": end_time - start_time,
                        "start_time": start_time,
                        "end_time": end_time,
                        "result": result
                    })
                except Exception as e:
                    end_time = time.time()
                    user_results.append({
                        "user_id": user_id,
                        "request_id": request_id,
                        "success": False,
                        "response_time": end_time - start_time,
                        "start_time": start_time,
                        "end_time": end_time,
                        "error": str(e)
                    })
            return user_results
        
        # Run all users concurrently
        tasks = [run_user_requests(user_id) for user_id in range(concurrent_users)]
        user_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        all_results = []
        for user_result in user_results:
            if isinstance(user_result, list):
                all_results.extend(user_result)
            else:
                # Handle exceptions
                all_results.append({
                    "success": False,
                    "error": str(user_result),
                    "response_time": 0,
                    "start_time": time.time(),
                    "end_time": time.time()
                })
        
        return all_results


class DatabaseTestHelper:
    """Helper class for database testing"""
    
    @staticmethod
    def create_test_database_url() -> str:
        """Create a test database URL"""
        return "sqlite:///./test_metrify.db"
    
    @staticmethod
    async def cleanup_test_data(session, table_name: str, condition: str = None):
        """Clean up test data from database"""
        if condition:
            query = f"DELETE FROM {table_name} WHERE {condition}"
        else:
            query = f"DELETE FROM {table_name}"
        
        await session.execute(query)
        await session.commit()
    
    @staticmethod
    async def count_records(session, table_name: str, condition: str = None) -> int:
        """Count records in a table"""
        if condition:
            query = f"SELECT COUNT(*) FROM {table_name} WHERE {condition}"
        else:
            query = f"SELECT COUNT(*) FROM {table_name}"
        
        result = await session.execute(query)
        return result.scalar()


class MockDataHelper:
    """Helper class for creating mock data"""
    
    @staticmethod
    def create_mock_kafka_message(topic: str, key: str, value: Dict[str, Any]) -> Dict[str, Any]:
        """Create a mock Kafka message"""
        return {
            "topic": topic,
            "partition": 0,
            "offset": random.randint(0, 10000),
            "key": key,
            "value": value,
            "timestamp": int(time.time() * 1000)
        }
    
    @staticmethod
    def create_mock_s3_object(bucket: str, key: str, content: str) -> Dict[str, Any]:
        """Create a mock S3 object"""
        return {
            "Bucket": bucket,
            "Key": key,
            "Content": content,
            "Size": len(content),
            "LastModified": datetime.utcnow(),
            "ETag": f'"{hash(content)}"'
        }
    
    @staticmethod
    def create_mock_http_response(status_code: int, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a mock HTTP response"""
        return {
            "status_code": status_code,
            "content": content,
            "headers": {"Content-Type": "application/json"},
            "elapsed": random.uniform(0.1, 2.0)
        }


class TestAssertionHelper:
    """Helper class for test assertions"""
    
    @staticmethod
    def assert_performance_requirements(metrics: PerformanceMetrics, 
                                      max_response_time: float = 2.0,
                                      min_success_rate: float = 0.99,
                                      min_throughput: float = 100.0):
        """Assert performance requirements are met"""
        assert metrics.avg_response_time <= max_response_time, \
            f"Average response time {metrics.avg_response_time:.3f}s exceeds {max_response_time}s"
        
        assert metrics.success_rate >= min_success_rate, \
            f"Success rate {metrics.success_rate:.2%} below {min_success_rate:.2%}"
        
        assert metrics.throughput >= min_throughput, \
            f"Throughput {metrics.throughput:.2f} requests/s below {min_throughput} requests/s"
    
    @staticmethod
    def assert_data_quality(data: List[Dict[str, Any]], 
                          required_fields: List[str],
                          quality_threshold: float = 0.95):
        """Assert data quality requirements"""
        if not data:
            return
        
        total_records = len(data)
        valid_records = 0
        
        for record in data:
            if all(field in record for field in required_fields):
                valid_records += 1
        
        quality_score = valid_records / total_records
        assert quality_score >= quality_threshold, \
            f"Data quality {quality_score:.2%} below threshold {quality_threshold:.2%}"
    
    @staticmethod
    def assert_anomaly_detection(readings: List[Dict[str, Any]], 
                               expected_anomaly_rate: float = 0.05,
                               tolerance: float = 0.02):
        """Assert anomaly detection is working correctly"""
        total_readings = len(readings)
        anomalous_readings = len([r for r in readings if r.get("anomaly_detected", False)])
        
        actual_anomaly_rate = anomalous_readings / total_readings if total_readings > 0 else 0
        
        assert abs(actual_anomaly_rate - expected_anomaly_rate) <= tolerance, \
            f"Anomaly rate {actual_anomaly_rate:.2%} differs from expected {expected_anomaly_rate:.2%} by more than {tolerance:.2%}"
