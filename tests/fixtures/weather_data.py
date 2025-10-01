"""
Weather Test Data Fixtures

Comprehensive test data for weather stations, observations, and forecasts.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any


def generate_weather_station_data(count: int = 10) -> List[Dict[str, Any]]:
    """Generate weather station test data."""
    stations = []
    
    # German cities with weather stations
    cities = [
        {"name": "Berlin Weather Station", "lat": 52.5200, "lon": 13.4050, "elevation": 34},
        {"name": "Munich Weather Station", "lat": 48.1351, "lon": 11.5820, "elevation": 520},
        {"name": "Hamburg Weather Station", "lat": 53.5511, "lon": 9.9937, "elevation": 6},
        {"name": "Cologne Weather Station", "lat": 50.9375, "lon": 6.9603, "elevation": 37},
        {"name": "Frankfurt Weather Station", "lat": 50.1109, "lon": 8.6821, "elevation": 112},
        {"name": "Stuttgart Weather Station", "lat": 48.7758, "lon": 9.1829, "elevation": 245},
        {"name": "Düsseldorf Weather Station", "lat": 51.2277, "lon": 6.7735, "elevation": 38},
        {"name": "Dortmund Weather Station", "lat": 51.5136, "lon": 7.4653, "elevation": 86},
        {"name": "Essen Weather Station", "lat": 51.4556, "lon": 7.0116, "elevation": 116},
        {"name": "Leipzig Weather Station", "lat": 51.3397, "lon": 12.3731, "elevation": 113}
    ]
    
    for i in range(count):
        city = cities[i % len(cities)]
        station = {
            "station_id": f"WS{i:06d}",
            "name": city["name"],
            "location": {
                "latitude": city["lat"] + np.random.uniform(-0.01, 0.01),
                "longitude": city["lon"] + np.random.uniform(-0.01, 0.01),
                "address": f"{city['name']}, Germany"
            },
            "station_type": np.random.choice(["AUTOMATIC", "MANUAL", "HYBRID"]),
            "status": np.random.choice(["ACTIVE", "INACTIVE", "MAINTENANCE"], p=[0.8, 0.15, 0.05]),
            "elevation_m": city["elevation"] + np.random.uniform(-10, 10),
            "installation_date": (datetime.utcnow() - timedelta(days=np.random.randint(30, 3650))).isoformat(),
            "capabilities": {
                "temperature": True,
                "humidity": True,
                "pressure": True,
                "wind_speed": True,
                "wind_direction": True,
                "precipitation": np.random.choice([True, False], p=[0.7, 0.3]),
                "solar_radiation": np.random.choice([True, False], p=[0.5, 0.5])
            },
            "metadata": {
                "operator": np.random.choice(["DWD", "Local Authority", "Private"]),
                "data_quality": np.random.choice(["EXCELLENT", "GOOD", "FAIR"], p=[0.6, 0.3, 0.1]),
                "update_frequency": np.random.choice(["1min", "5min", "10min", "1hour"])
            }
        }
        stations.append(station)
    
    return stations


def generate_weather_observations_data(station_ids: List[str], hours: int = 24) -> List[Dict[str, Any]]:
    """Generate weather observations test data."""
    observations = []
    
    for station_id in station_ids:
        # Base weather conditions (seasonal variation)
        base_temp = 15 + 10 * np.sin(2 * np.pi * datetime.utcnow().timetuple().tm_yday / 365)
        base_humidity = 60 + 20 * np.sin(2 * np.pi * datetime.utcnow().timetuple().tm_yday / 365)
        base_pressure = 1013.25  # Standard atmospheric pressure
        
        for hour in range(hours):
            # Add daily patterns
            time_of_day_factor = 1 + 0.3 * np.sin(2 * np.pi * hour / 24)
            
            # Temperature with daily cycle
            temperature = base_temp + 8 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
            
            # Humidity (inverse relationship with temperature)
            humidity = base_humidity - 20 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 5)
            humidity = max(0, min(100, humidity))  # Clamp to 0-100
            
            # Pressure with slight daily variation
            pressure = base_pressure + 2 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 1)
            
            # Wind speed (higher during day)
            wind_speed = 3 + 2 * np.sin(2 * np.pi * hour / 24) + np.random.exponential(1)
            wind_speed = max(0, wind_speed)  # Ensure non-negative
            
            # Wind direction (random but with some persistence)
            wind_direction = np.random.uniform(0, 360)
            
            # Precipitation (occasional)
            precipitation = np.random.exponential(0.5) if np.random.random() < 0.1 else 0
            
            observation = {
                "station_id": station_id,
                "timestamp": datetime.utcnow() - timedelta(hours=hours-hour),
                "temperature_c": round(temperature, 1),
                "humidity_percent": round(humidity, 1),
                "pressure_hpa": round(pressure, 2),
                "wind_speed_ms": round(wind_speed, 1),
                "wind_direction_deg": round(wind_direction, 0),
                "precipitation_mm": round(precipitation, 2),
                "visibility_km": round(np.random.uniform(5, 20), 1),
                "cloud_cover_percent": round(np.random.uniform(0, 100), 0),
                "solar_radiation_wm2": round(max(0, 800 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 50)), 0),
                "quality_score": round(np.random.uniform(0.8, 1.0), 3),
                "data_source": "automatic_weather_station"
            }
            observations.append(observation)
    
    return observations


def generate_weather_forecast_data(station_id: str, hours: int = 48) -> pd.DataFrame:
    """Generate weather forecast test data."""
    timestamps = pd.date_range(
        start=datetime.utcnow(),
        periods=hours,
        freq='H'
    )
    
    # Generate forecast data with realistic patterns
    base_temp = 15 + 10 * np.sin(2 * np.pi * datetime.utcnow().timetuple().tm_yday / 365)
    
    temperatures = []
    humidities = []
    pressures = []
    wind_speeds = []
    wind_directions = []
    precipitations = []
    
    for i, timestamp in enumerate(timestamps):
        # Temperature forecast with daily cycle and trend
        temp = base_temp + 8 * np.sin(2 * np.pi * i / 24) + 0.1 * i + np.random.normal(0, 1)
        temperatures.append(temp)
        
        # Humidity forecast (inverse relationship with temperature)
        humidity = 60 - 20 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 3)
        humidities.append(max(0, min(100, humidity)))
        
        # Pressure forecast with slight variation
        pressure = 1013.25 + 2 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 0.5)
        pressures.append(pressure)
        
        # Wind speed forecast
        wind_speed = 3 + 2 * np.sin(2 * np.pi * i / 24) + np.random.exponential(0.5)
        wind_speeds.append(max(0, wind_speed))
        
        # Wind direction forecast (with some persistence)
        wind_dir = np.random.uniform(0, 360)
        wind_directions.append(wind_dir)
        
        # Precipitation forecast (occasional)
        precip = np.random.exponential(0.3) if np.random.random() < 0.15 else 0
        precipitations.append(precip)
    
    return pd.DataFrame({
        'station_id': station_id,
        'timestamp': timestamps,
        'temperature_c': temperatures,
        'humidity_percent': humidities,
        'pressure_hpa': pressures,
        'wind_speed_ms': wind_speeds,
        'wind_direction_deg': wind_directions,
        'precipitation_mm': precipitations,
        'forecast_model': 'ECMWF',
        'forecast_accuracy': round(np.random.uniform(0.8, 0.95), 3),
        'confidence_score': round(np.random.uniform(0.7, 0.9), 3)
    })


def generate_weather_alerts_data(station_ids: List[str]) -> List[Dict[str, Any]]:
    """Generate weather alerts test data."""
    alerts = []
    
    alert_types = [
        "temperature_high", "temperature_low", "humidity_high", "humidity_low",
        "pressure_high", "pressure_low", "wind_speed_high", "precipitation_high"
    ]
    
    for station_id in station_ids:
        # Generate 0-3 alerts per station
        num_alerts = np.random.poisson(1.5)
        
        for i in range(num_alerts):
            alert = {
                "alert_id": f"WA_{station_id}_{i:06d}",
                "station_id": station_id,
                "alert_type": np.random.choice(alert_types),
                "severity": np.random.choice(["low", "medium", "high", "critical"], p=[0.4, 0.3, 0.2, 0.1]),
                "timestamp": datetime.utcnow() - timedelta(hours=np.random.randint(0, 24)),
                "description": f"Weather alert for {station_id}",
                "threshold_value": np.random.uniform(20, 40),
                "actual_value": np.random.uniform(15, 45),
                "duration_hours": np.random.uniform(1, 12),
                "status": np.random.choice(["active", "resolved", "expired"], p=[0.3, 0.6, 0.1]),
                "metadata": {
                    "alert_source": "automatic_detection",
                    "confidence": round(np.random.uniform(0.7, 1.0), 3),
                    "notification_sent": np.random.choice([True, False], p=[0.8, 0.2])
                }
            }
            alerts.append(alert)
    
    return alerts


def generate_weather_quality_metrics_data(station_ids: List[str]) -> List[Dict[str, Any]]:
    """Generate weather data quality metrics test data."""
    metrics = []
    
    for station_id in station_ids:
        metric = {
            "station_id": station_id,
            "timestamp": datetime.utcnow(),
            "completeness": round(np.random.uniform(0.85, 1.0), 3),
            "accuracy": round(np.random.uniform(0.8, 0.98), 3),
            "consistency": round(np.random.uniform(0.75, 0.95), 3),
            "timeliness": round(np.random.uniform(0.9, 1.0), 3),
            "validity": round(np.random.uniform(0.85, 0.99), 3),
            "overall_score": 0.0,  # Will be calculated
            "issues": [
                "Missing temperature readings: 2 records",
                "Invalid humidity values: 1 record"
            ] if np.random.random() < 0.3 else [],
            "recommendations": [
                "Calibrate temperature sensor",
                "Check humidity sensor connection"
            ] if np.random.random() < 0.4 else [],
            "data_gaps": [
                {"start": "2024-01-15T10:00:00Z", "end": "2024-01-15T11:00:00Z", "duration_minutes": 60}
            ] if np.random.random() < 0.2 else []
        }
        
        # Calculate overall score
        scores = [metric["completeness"], metric["accuracy"], metric["consistency"], 
                 metric["timeliness"], metric["validity"]]
        metric["overall_score"] = round(np.mean(scores), 3)
        
        metrics.append(metric)
    
    return metrics


def generate_weather_correlation_data(station_ids: List[str], days: int = 30) -> pd.DataFrame:
    """Generate weather correlation data for analysis."""
    data = []
    
    for station_id in station_ids:
        for day in range(days):
            for hour in range(24):
                timestamp = datetime.utcnow() - timedelta(days=days-day, hours=23-hour)
                
                # Generate correlated weather data
                base_temp = 15 + 10 * np.sin(2 * np.pi * day / 365)  # Seasonal
                temp = base_temp + 8 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
                
                # Humidity inversely correlated with temperature
                humidity = 80 - 0.5 * temp + np.random.normal(0, 5)
                humidity = max(0, min(100, humidity))
                
                # Pressure slightly correlated with temperature
                pressure = 1013.25 + 0.1 * temp + np.random.normal(0, 1)
                
                # Wind speed correlated with pressure gradient
                wind_speed = 3 + 0.1 * abs(pressure - 1013.25) + np.random.exponential(1)
                
                data.append({
                    'station_id': station_id,
                    'timestamp': timestamp,
                    'temperature_c': temp,
                    'humidity_percent': humidity,
                    'pressure_hpa': pressure,
                    'wind_speed_ms': wind_speed,
                    'day_of_year': timestamp.timetuple().tm_yday,
                    'hour_of_day': hour,
                    'is_weekend': 1 if timestamp.weekday() >= 5 else 0
                })
    
    return pd.DataFrame(data)


def generate_weather_anomaly_data(station_id: str, count: int = 10) -> List[Dict[str, Any]]:
    """Generate weather anomaly test data."""
    anomalies = []
    
    anomaly_types = [
        "temperature_spike", "temperature_drop", "humidity_anomaly",
        "pressure_anomaly", "wind_speed_anomaly", "sensor_failure"
    ]
    
    for i in range(count):
        anomaly = {
            "anomaly_id": f"WANOM_{station_id}_{i:06d}",
            "station_id": station_id,
            "timestamp": datetime.utcnow() - timedelta(hours=i),
            "anomaly_type": np.random.choice(anomaly_types),
            "severity": np.random.choice(["low", "medium", "high", "critical"]),
            "confidence": round(np.random.uniform(0.7, 1.0), 3),
            "description": f"Weather anomaly detected in {station_id}",
            "affected_parameters": np.random.choice([
                ["temperature_c"],
                ["humidity_percent"],
                ["pressure_hpa"],
                ["wind_speed_ms"],
                ["temperature_c", "humidity_percent"]
            ]),
            "detection_method": np.random.choice([
                "statistical", "ml_model", "threshold", "pattern_matching"
            ]),
            "status": np.random.choice(["new", "investigating", "resolved", "false_positive"]),
            "metadata": {
                "expected_value": np.random.uniform(15, 25),
                "actual_value": np.random.uniform(10, 30),
                "deviation": np.random.uniform(5, 15)
            }
        }
        anomalies.append(anomaly)
    
    return anomalies
