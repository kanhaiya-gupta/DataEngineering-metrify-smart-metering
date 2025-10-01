"""
Smart Meter Test Data Fixtures

Comprehensive test data for smart meter entities, readings, and related components.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any


def generate_smart_meter_data(count: int = 100) -> List[Dict[str, Any]]:
    """Generate smart meter test data."""
    meters = []
    
    for i in range(count):
        meter = {
            "meter_id": f"SM{i:06d}",
            "location": {
                "latitude": 52.5200 + np.random.uniform(-0.1, 0.1),
                "longitude": 13.4050 + np.random.uniform(-0.1, 0.1),
                "address": f"Test Street {i}, Berlin, Germany"
            },
            "specifications": {
                "manufacturer": np.random.choice(["Siemens", "Landis+Gyr", "Itron", "Elster"]),
                "model": f"Model-{np.random.randint(1000, 9999)}",
                "firmware_version": f"{np.random.randint(1, 5)}.{np.random.randint(0, 9)}.{np.random.randint(0, 9)}",
                "installation_date": (datetime.utcnow() - timedelta(days=np.random.randint(30, 3650))).isoformat()
            },
            "status": np.random.choice(["ACTIVE", "INACTIVE", "MAINTENANCE"], p=[0.8, 0.15, 0.05]),
            "quality_tier": np.random.choice(["EXCELLENT", "GOOD", "FAIR", "POOR"], p=[0.4, 0.3, 0.2, 0.1]),
            "metadata": {
                "customer_id": f"CUST{i:06d}",
                "installation_type": np.random.choice(["residential", "commercial", "industrial"]),
                "tariff_type": np.random.choice(["standard", "time_of_use", "demand_response"]),
                "grid_operator": np.random.choice(["TENNET", "50HERTZ", "AMPRION"])
            }
        }
        meters.append(meter)
    
    return meters


def generate_meter_readings_data(meter_ids: List[str], hours: int = 24) -> List[Dict[str, Any]]:
    """Generate smart meter readings test data."""
    readings = []
    
    for meter_id in meter_ids:
        base_energy = np.random.uniform(50, 200)  # Base energy consumption
        
        for hour in range(hours):
            # Add some realistic patterns
            time_of_day_factor = 1 + 0.3 * np.sin(2 * np.pi * hour / 24)  # Daily pattern
            random_factor = np.random.uniform(0.8, 1.2)  # Random variation
            
            energy_consumed = base_energy * time_of_day_factor * random_factor
            
            reading = {
                "meter_id": meter_id,
                "timestamp": datetime.utcnow() - timedelta(hours=hours-hour),
                "energy_consumed_kwh": round(energy_consumed, 2),
                "power_factor": round(np.random.uniform(0.85, 0.98), 3),
                "voltage_v": round(np.random.uniform(220, 240), 1),
                "current_a": round(energy_consumed / 230, 2),  # P = V * I
                "frequency_hz": round(np.random.uniform(49.8, 50.2), 2),
                "temperature_c": round(np.random.uniform(15, 35), 1),
                "quality_score": round(np.random.uniform(0.7, 1.0), 3),
                "anomaly_detected": np.random.choice([True, False], p=[0.05, 0.95])
            }
            readings.append(reading)
    
    return readings


def generate_anomaly_data(meter_id: str, count: int = 10) -> List[Dict[str, Any]]:
    """Generate anomaly test data."""
    anomalies = []
    
    for i in range(count):
        anomaly = {
            "anomaly_id": f"ANOM_{meter_id}_{i:06d}",
            "meter_id": meter_id,
            "timestamp": datetime.utcnow() - timedelta(hours=i),
            "anomaly_type": np.random.choice([
                "energy_spike", "voltage_drop", "frequency_deviation", 
                "power_factor_anomaly", "temperature_anomaly"
            ]),
            "severity": np.random.choice(["low", "medium", "high", "critical"]),
            "confidence": round(np.random.uniform(0.7, 1.0), 3),
            "description": f"Anomaly detected in {meter_id}",
            "affected_parameters": np.random.choice([
                ["energy_consumed_kwh"],
                ["voltage_v"],
                ["frequency_hz"],
                ["power_factor", "current_a"],
                ["temperature_c"]
            ]),
            "detection_method": np.random.choice([
                "statistical", "ml_model", "rule_based", "threshold"
            ]),
            "status": np.random.choice(["new", "investigating", "resolved", "false_positive"])
        }
        anomalies.append(anomaly)
    
    return anomalies


def generate_quality_metrics_data(meter_ids: List[str]) -> List[Dict[str, Any]]:
    """Generate data quality metrics test data."""
    metrics = []
    
    for meter_id in meter_ids:
        metric = {
            "meter_id": meter_id,
            "timestamp": datetime.utcnow(),
            "completeness": round(np.random.uniform(0.85, 1.0), 3),
            "accuracy": round(np.random.uniform(0.8, 0.98), 3),
            "consistency": round(np.random.uniform(0.75, 0.95), 3),
            "timeliness": round(np.random.uniform(0.9, 1.0), 3),
            "validity": round(np.random.uniform(0.85, 0.99), 3),
            "overall_score": 0.0,  # Will be calculated
            "issues": [
                "Missing timestamps: 2 records",
                "Invalid voltage readings: 1 record"
            ] if np.random.random() < 0.3 else [],
            "recommendations": [
                "Improve data validation",
                "Check meter calibration"
            ] if np.random.random() < 0.4 else []
        }
        
        # Calculate overall score
        scores = [metric["completeness"], metric["accuracy"], metric["consistency"], 
                 metric["timeliness"], metric["validity"]]
        metric["overall_score"] = round(np.mean(scores), 3)
        
        metrics.append(metric)
    
    return metrics


def generate_forecast_data(meter_id: str, hours: int = 24) -> pd.DataFrame:
    """Generate forecast test data."""
    timestamps = pd.date_range(
        start=datetime.utcnow(),
        periods=hours,
        freq='H'
    )
    
    # Generate realistic forecast data
    base_value = np.random.uniform(80, 150)
    trend = np.random.uniform(-0.5, 0.5)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(hours) / 24)
    noise = np.random.normal(0, 2, hours)
    
    actual = base_value + trend * np.arange(hours) + seasonality + noise
    forecast = actual + np.random.normal(0, 1, hours)
    confidence_lower = forecast - np.random.uniform(5, 15, hours)
    confidence_upper = forecast + np.random.uniform(5, 15, hours)
    
    return pd.DataFrame({
        'meter_id': meter_id,
        'timestamp': timestamps,
        'actual': actual,
        'forecast': forecast,
        'confidence_lower': confidence_lower,
        'confidence_upper': confidence_upper,
        'model_name': 'prophet_v1',
        'accuracy_score': round(np.random.uniform(0.8, 0.95), 3)
    })


def generate_ml_training_data(meter_ids: List[str], days: int = 30) -> pd.DataFrame:
    """Generate ML training data."""
    data = []
    
    for meter_id in meter_ids:
        for day in range(days):
            for hour in range(24):
                # Generate features
                timestamp = datetime.utcnow() - timedelta(days=days-day, hours=23-hour)
                
                # Energy consumption with patterns
                base_energy = np.random.uniform(50, 200)
                time_factor = 1 + 0.3 * np.sin(2 * np.pi * hour / 24)
                day_factor = 1 + 0.2 * np.sin(2 * np.pi * day / 7)  # Weekly pattern
                energy = base_energy * time_factor * day_factor
                
                # Weather features
                temperature = 15 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
                humidity = 60 + 20 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 5)
                
                # Grid features
                frequency = 50 + np.random.normal(0, 0.1)
                voltage = 230 + np.random.normal(0, 5)
                
                # Anomaly label (5% chance)
                is_anomaly = np.random.choice([0, 1], p=[0.95, 0.05])
                
                data.append({
                    'meter_id': meter_id,
                    'timestamp': timestamp,
                    'energy_consumed': energy,
                    'temperature': temperature,
                    'humidity': humidity,
                    'frequency': frequency,
                    'voltage': voltage,
                    'hour': hour,
                    'day_of_week': timestamp.weekday(),
                    'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
                    'anomaly_label': is_anomaly
                })
    
    return pd.DataFrame(data)


def generate_event_data(event_type: str, count: int = 10) -> List[Dict[str, Any]]:
    """Generate domain event test data."""
    events = []
    
    for i in range(count):
        event = {
            "event_id": f"evt_{event_type}_{i:06d}",
            "event_type": event_type,
            "aggregate_id": f"SM{i:06d}",
            "timestamp": datetime.utcnow() - timedelta(minutes=i),
            "version": i + 1,
            "data": {
                "meter_id": f"SM{i:06d}",
                "status": "ACTIVE",
                "location": "Berlin, Germany"
            },
            "metadata": {
                "source": "smart_meter_service",
                "correlation_id": f"corr_{i:06d}",
                "causation_id": f"cause_{i:06d}"
            }
        }
        events.append(event)
    
    return events


def generate_performance_test_data(operation: str, count: int = 1000) -> List[Dict[str, Any]]:
    """Generate performance test data."""
    data = []
    
    for i in range(count):
        if operation == "meter_readings":
            data.append({
                "meter_id": f"SM{i:06d}",
                "timestamp": datetime.utcnow() - timedelta(minutes=i),
                "energy_consumed_kwh": np.random.uniform(50, 200),
                "power_factor": np.random.uniform(0.85, 0.98),
                "voltage_v": np.random.uniform(220, 240),
                "current_a": np.random.uniform(5, 15),
                "frequency_hz": np.random.uniform(49.8, 50.2),
                "temperature_c": np.random.uniform(15, 35),
                "quality_score": np.random.uniform(0.7, 1.0),
                "anomaly_detected": np.random.choice([True, False], p=[0.05, 0.95])
            })
        elif operation == "events":
            data.append({
                "event_id": f"evt_{i:06d}",
                "event_type": np.random.choice(["MeterReading", "AnomalyDetected", "StatusChanged"]),
                "aggregate_id": f"SM{i:06d}",
                "timestamp": datetime.utcnow() - timedelta(minutes=i),
                "data": {"value": np.random.uniform(0, 1000)}
            })
        elif operation == "analytics":
            data.append({
                "meter_id": f"SM{i:06d}",
                "timestamp": datetime.utcnow() - timedelta(minutes=i),
                "metric_name": np.random.choice(["energy_consumption", "efficiency", "quality_score"]),
                "metric_value": np.random.uniform(0, 1000),
                "dimensions": {
                    "region": "Berlin",
                    "customer_type": np.random.choice(["residential", "commercial"])
                }
            })
    
    return data
