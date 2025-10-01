"""
ML Models Test Data Fixtures

Comprehensive test data for ML models, training data, and model performance metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import json


def generate_ml_model_data(count: int = 20) -> List[Dict[str, Any]]:
    """Generate ML model test data."""
    models = []
    
    model_types = [
        {
            "type": "anomaly_detection",
            "framework": "tensorflow",
            "algorithm": "isolation_forest",
            "use_case": "detect_energy_anomalies"
        },
        {
            "type": "forecasting",
            "framework": "tensorflow",
            "algorithm": "lstm",
            "use_case": "energy_consumption_forecast"
        },
        {
            "type": "classification",
            "framework": "sklearn",
            "algorithm": "random_forest",
            "use_case": "quality_prediction"
        },
        {
            "type": "regression",
            "framework": "tensorflow",
            "algorithm": "neural_network",
            "use_case": "efficiency_prediction"
        },
        {
            "type": "clustering",
            "framework": "sklearn",
            "algorithm": "kmeans",
            "use_case": "customer_segmentation"
        }
    ]
    
    for i in range(count):
        model_type = model_types[i % len(model_types)]
        
        model = {
            "model_id": f"ML_{model_type['type'].upper()}_{i:06d}",
            "name": f"{model_type['type'].title()} Model v{i+1}",
            "type": model_type["type"],
            "framework": model_type["framework"],
            "algorithm": model_type["algorithm"],
            "use_case": model_type["use_case"],
            "version": f"v{i+1}.0.0",
            "status": np.random.choice(["training", "trained", "deployed", "retired"], p=[0.1, 0.3, 0.5, 0.1]),
            "created_at": (datetime.utcnow() - timedelta(days=np.random.randint(1, 365))).isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "performance_metrics": {
                "accuracy": round(np.random.uniform(0.8, 0.95), 3),
                "precision": round(np.random.uniform(0.75, 0.92), 3),
                "recall": round(np.random.uniform(0.78, 0.90), 3),
                "f1_score": round(np.random.uniform(0.76, 0.91), 3),
                "auc_roc": round(np.random.uniform(0.85, 0.98), 3),
                "mae": round(np.random.uniform(0.5, 3.0), 2),
                "rmse": round(np.random.uniform(1.0, 5.0), 2),
                "r2_score": round(np.random.uniform(0.7, 0.95), 3)
            },
            "training_info": {
                "training_samples": np.random.randint(10000, 100000),
                "validation_samples": np.random.randint(2000, 20000),
                "test_samples": np.random.randint(1000, 10000),
                "training_duration_hours": round(np.random.uniform(1, 24), 1),
                "features_count": np.random.randint(10, 100),
                "hyperparameters": {
                    "learning_rate": round(np.random.uniform(0.001, 0.1), 4),
                    "batch_size": np.random.choice([32, 64, 128, 256]),
                    "epochs": np.random.randint(50, 500),
                    "hidden_layers": np.random.randint(2, 8),
                    "dropout_rate": round(np.random.uniform(0.1, 0.5), 2)
                }
            },
            "deployment_info": {
                "deployment_status": np.random.choice(["not_deployed", "staging", "production", "retired"]),
                "deployment_date": (datetime.utcnow() - timedelta(days=np.random.randint(0, 180))).isoformat() if np.random.random() > 0.3 else None,
                "endpoint_url": f"https://api.metrify.com/ml/{model_type['type']}/predict" if np.random.random() > 0.3 else None,
                "replicas": np.random.randint(1, 5),
                "cpu_cores": np.random.randint(1, 8),
                "memory_gb": np.random.randint(2, 16),
                "gpu_enabled": np.random.choice([True, False], p=[0.3, 0.7])
            },
            "monitoring": {
                "prediction_count": np.random.randint(1000, 100000),
                "avg_response_time_ms": round(np.random.uniform(10, 500), 1),
                "error_rate": round(np.random.uniform(0.001, 0.05), 4),
                "throughput_per_second": round(np.random.uniform(10, 1000), 1),
                "last_prediction": (datetime.utcnow() - timedelta(minutes=np.random.randint(1, 1440))).isoformat(),
                "data_drift_score": round(np.random.uniform(0.1, 0.8), 3),
                "model_drift_score": round(np.random.uniform(0.1, 0.6), 3)
            },
            "metadata": {
                "created_by": f"data_scientist_{np.random.randint(1, 10)}",
                "team": np.random.choice(["ML_Team", "Data_Team", "Analytics_Team"]),
                "project": f"Smart_Metering_{model_type['type'].title()}",
                "tags": [model_type["type"], model_type["framework"], model_type["use_case"]],
                "description": f"ML model for {model_type['use_case']} using {model_type['algorithm']}",
                "model_size_mb": round(np.random.uniform(10, 500), 1),
                "dependencies": [
                    "tensorflow==2.12.0",
                    "numpy==1.24.0",
                    "pandas==1.5.0",
                    "scikit-learn==1.2.0"
                ]
            }
        }
        models.append(model)
    
    return models


def generate_training_data(model_type: str, samples: int = 1000) -> pd.DataFrame:
    """Generate ML training data based on model type."""
    
    if model_type == "anomaly_detection":
        return generate_anomaly_training_data(samples)
    elif model_type == "forecasting":
        return generate_forecasting_training_data(samples)
    elif model_type == "classification":
        return generate_classification_training_data(samples)
    elif model_type == "regression":
        return generate_regression_training_data(samples)
    elif model_type == "clustering":
        return generate_clustering_training_data(samples)
    else:
        return generate_generic_training_data(samples)


def generate_anomaly_training_data(samples: int) -> pd.DataFrame:
    """Generate anomaly detection training data."""
    data = []
    
    for i in range(samples):
        # Normal data (90% of samples)
        if np.random.random() < 0.9:
            energy = np.random.normal(100, 20)
            voltage = np.random.normal(230, 10)
            current = np.random.normal(5, 1)
            frequency = np.random.normal(50, 0.1)
            temperature = np.random.normal(25, 5)
            is_anomaly = 0
        else:
            # Anomalous data (10% of samples)
            energy = np.random.normal(200, 30)  # High energy consumption
            voltage = np.random.normal(200, 15)  # Low voltage
            current = np.random.normal(8, 2)  # High current
            frequency = np.random.normal(49.5, 0.2)  # Frequency deviation
            temperature = np.random.normal(40, 8)  # High temperature
            is_anomaly = 1
        
        data.append({
            'meter_id': f"SM{i:06d}",
            'timestamp': datetime.utcnow() - timedelta(hours=i),
            'energy_consumed_kwh': energy,
            'voltage_v': voltage,
            'current_a': current,
            'frequency_hz': frequency,
            'temperature_c': temperature,
            'hour': (datetime.utcnow() - timedelta(hours=i)).hour,
            'day_of_week': (datetime.utcnow() - timedelta(hours=i)).weekday(),
            'is_weekend': 1 if (datetime.utcnow() - timedelta(hours=i)).weekday() >= 5 else 0,
            'anomaly_label': is_anomaly
        })
    
    return pd.DataFrame(data)


def generate_forecasting_training_data(samples: int) -> pd.DataFrame:
    """Generate forecasting training data."""
    data = []
    
    # Generate time series data
    base_value = 100
    trend = 0.1
    seasonality_amplitude = 20
    
    for i in range(samples):
        timestamp = datetime.utcnow() - timedelta(hours=samples-i)
        
        # Time series components
        trend_component = trend * i
        seasonal_component = seasonality_amplitude * np.sin(2 * np.pi * i / 24)  # Daily seasonality
        weekly_component = 10 * np.sin(2 * np.pi * i / (24 * 7))  # Weekly seasonality
        noise = np.random.normal(0, 5)
        
        value = base_value + trend_component + seasonal_component + weekly_component + noise
        
        data.append({
            'timestamp': timestamp,
            'value': value,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'month': timestamp.month,
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'lag_1': value + np.random.normal(0, 2) if i > 0 else value,
            'lag_24': value + np.random.normal(0, 3) if i >= 24 else value,
            'lag_168': value + np.random.normal(0, 4) if i >= 168 else value
        })
    
    return pd.DataFrame(data)


def generate_classification_training_data(samples: int) -> pd.DataFrame:
    """Generate classification training data."""
    data = []
    
    for i in range(samples):
        # Features
        energy_consumption = np.random.uniform(50, 200)
        voltage_quality = np.random.uniform(0.8, 1.0)
        frequency_stability = np.random.uniform(0.9, 1.0)
        temperature = np.random.uniform(15, 35)
        humidity = np.random.uniform(30, 80)
        power_factor = np.random.uniform(0.8, 1.0)
        
        # Calculate quality score
        quality_score = (voltage_quality + frequency_stability + power_factor) / 3
        
        # Determine quality class
        if quality_score >= 0.95:
            quality_class = "EXCELLENT"
        elif quality_score >= 0.85:
            quality_class = "GOOD"
        elif quality_score >= 0.70:
            quality_class = "FAIR"
        else:
            quality_class = "POOR"
        
        data.append({
            'meter_id': f"SM{i:06d}",
            'energy_consumption_kwh': energy_consumption,
            'voltage_quality': voltage_quality,
            'frequency_stability': frequency_stability,
            'temperature_c': temperature,
            'humidity_percent': humidity,
            'power_factor': power_factor,
            'quality_score': quality_score,
            'quality_class': quality_class,
            'customer_type': np.random.choice(['residential', 'commercial', 'industrial']),
            'installation_age_years': np.random.uniform(0, 20)
        })
    
    return pd.DataFrame(data)


def generate_regression_training_data(samples: int) -> pd.DataFrame:
    """Generate regression training data."""
    data = []
    
    for i in range(samples):
        # Features
        energy_consumption = np.random.uniform(50, 200)
        temperature = np.random.uniform(15, 35)
        humidity = np.random.uniform(30, 80)
        voltage = np.random.uniform(220, 240)
        frequency = np.random.uniform(49.8, 50.2)
        hour = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        
        # Target variable (efficiency)
        efficiency = (0.8 + 
                     0.1 * (voltage - 230) / 10 +  # Voltage effect
                     0.05 * (frequency - 50) / 0.2 +  # Frequency effect
                     -0.02 * (temperature - 25) / 10 +  # Temperature effect
                     -0.01 * (humidity - 50) / 30 +  # Humidity effect
                     np.random.normal(0, 0.05))  # Noise
        
        efficiency = max(0.5, min(1.0, efficiency))  # Clamp between 0.5 and 1.0
        
        data.append({
            'meter_id': f"SM{i:06d}",
            'energy_consumption_kwh': energy_consumption,
            'temperature_c': temperature,
            'humidity_percent': humidity,
            'voltage_v': voltage,
            'frequency_hz': frequency,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'efficiency': efficiency
        })
    
    return pd.DataFrame(data)


def generate_clustering_training_data(samples: int) -> pd.DataFrame:
    """Generate clustering training data."""
    data = []
    
    # Define customer segments
    segments = [
        {"name": "low_consumption", "energy_range": (50, 100), "volatility": 0.1, "size": 0.3},
        {"name": "medium_consumption", "energy_range": (100, 150), "volatility": 0.15, "size": 0.4},
        {"name": "high_consumption", "energy_range": (150, 250), "volatility": 0.2, "size": 0.2},
        {"name": "industrial", "energy_range": (200, 500), "volatility": 0.25, "size": 0.1}
    ]
    
    for i in range(samples):
        # Select segment based on size
        segment_choice = np.random.choice(len(segments), p=[s["size"] for s in segments])
        segment = segments[segment_choice]
        
        # Generate features based on segment
        energy_min, energy_max = segment["energy_range"]
        energy_consumption = np.random.uniform(energy_min, energy_max)
        volatility = segment["volatility"]
        
        # Additional features
        peak_consumption = energy_consumption * (1 + volatility * np.random.uniform(0.5, 1.5))
        off_peak_consumption = energy_consumption * (1 - volatility * np.random.uniform(0.3, 0.8))
        weekend_consumption = energy_consumption * np.random.uniform(0.7, 1.2)
        
        data.append({
            'customer_id': f"CUST{i:06d}",
            'energy_consumption_kwh': energy_consumption,
            'peak_consumption_kwh': peak_consumption,
            'off_peak_consumption_kwh': off_peak_consumption,
            'weekend_consumption_kwh': weekend_consumption,
            'consumption_volatility': volatility,
            'customer_type': np.random.choice(['residential', 'commercial', 'industrial']),
            'tariff_type': np.random.choice(['standard', 'time_of_use', 'demand_response']),
            'payment_reliability': np.random.uniform(0.7, 1.0),
            'contract_duration_months': np.random.randint(12, 60),
            'segment': segment["name"]
        })
    
    return pd.DataFrame(data)


def generate_generic_training_data(samples: int) -> pd.DataFrame:
    """Generate generic training data."""
    data = []
    
    for i in range(samples):
        data.append({
            'id': i,
            'feature_1': np.random.normal(0, 1),
            'feature_2': np.random.normal(0, 1),
            'feature_3': np.random.normal(0, 1),
            'feature_4': np.random.normal(0, 1),
            'feature_5': np.random.normal(0, 1),
            'target': np.random.normal(0, 1)
        })
    
    return pd.DataFrame(data)


def generate_model_metrics_data(model_ids: List[str]) -> List[Dict[str, Any]]:
    """Generate model performance metrics data."""
    metrics_data = []
    
    for model_id in model_ids:
        # Generate metrics for different time periods
        for days_ago in [1, 7, 30, 90]:
            timestamp = datetime.utcnow() - timedelta(days=days_ago)
            
            metrics = {
                "model_id": model_id,
                "timestamp": timestamp,
                "period_days": days_ago,
                "predictions_count": np.random.randint(100, 10000),
                "accuracy": round(np.random.uniform(0.8, 0.95), 3),
                "precision": round(np.random.uniform(0.75, 0.92), 3),
                "recall": round(np.random.uniform(0.78, 0.90), 3),
                "f1_score": round(np.random.uniform(0.76, 0.91), 3),
                "auc_roc": round(np.random.uniform(0.85, 0.98), 3),
                "mae": round(np.random.uniform(0.5, 3.0), 2),
                "rmse": round(np.random.uniform(1.0, 5.0), 2),
                "r2_score": round(np.random.uniform(0.7, 0.95), 3),
                "avg_response_time_ms": round(np.random.uniform(10, 500), 1),
                "error_rate": round(np.random.uniform(0.001, 0.05), 4),
                "throughput_per_second": round(np.random.uniform(10, 1000), 1),
                "data_drift_score": round(np.random.uniform(0.1, 0.8), 3),
                "model_drift_score": round(np.random.uniform(0.1, 0.6), 3),
                "cpu_usage_percent": round(np.random.uniform(20, 80), 1),
                "memory_usage_percent": round(np.random.uniform(30, 90), 1),
                "gpu_usage_percent": round(np.random.uniform(0, 70), 1) if np.random.random() > 0.5 else None
            }
            metrics_data.append(metrics)
    
    return metrics_data


def generate_model_predictions_data(model_id: str, count: int = 100) -> List[Dict[str, Any]]:
    """Generate model prediction data."""
    predictions = []
    
    for i in range(count):
        prediction = {
            "prediction_id": f"PRED_{model_id}_{i:06d}",
            "model_id": model_id,
            "timestamp": datetime.utcnow() - timedelta(minutes=i),
            "input_data": {
                "meter_id": f"SM{i:06d}",
                "energy_consumption": np.random.uniform(50, 200),
                "voltage": np.random.uniform(220, 240),
                "temperature": np.random.uniform(15, 35)
            },
            "prediction": {
                "value": np.random.uniform(0, 1),
                "confidence": round(np.random.uniform(0.7, 1.0), 3),
                "class": np.random.choice(["normal", "anomaly", "high", "medium", "low"]),
                "probability": round(np.random.uniform(0.5, 1.0), 3)
            },
            "metadata": {
                "processing_time_ms": round(np.random.uniform(1, 100), 1),
                "model_version": "v1.2.0",
                "feature_count": np.random.randint(5, 20),
                "prediction_type": np.random.choice(["classification", "regression", "anomaly_detection"])
            }
        }
        predictions.append(prediction)
    
    return predictions
