"""
Solar Panel Test Data Fixtures

Comprehensive test data for solar panel installations, generation data, and related components.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any


def generate_solar_panel_data(count: int = 50) -> List[Dict[str, Any]]:
    """Generate solar panel installation test data."""
    panels = []
    
    # German cities with solar installations
    cities = [
        {"name": "Berlin", "lat": 52.5200, "lon": 13.4050, "solar_potential": 0.85},
        {"name": "Munich", "lat": 48.1351, "lon": 11.5820, "solar_potential": 0.90},
        {"name": "Hamburg", "lat": 53.5511, "lon": 9.9937, "solar_potential": 0.80},
        {"name": "Cologne", "lat": 50.9375, "lon": 6.9603, "solar_potential": 0.88},
        {"name": "Frankfurt", "lat": 50.1109, "lon": 8.6821, "solar_potential": 0.87},
        {"name": "Stuttgart", "lat": 48.7758, "lon": 9.1829, "solar_potential": 0.92},
        {"name": "Düsseldorf", "lat": 51.2277, "lon": 6.7735, "solar_potential": 0.86},
        {"name": "Dortmund", "lat": 51.5136, "lon": 7.4653, "solar_potential": 0.84},
        {"name": "Essen", "lat": 51.4556, "lon": 7.0116, "solar_potential": 0.83},
        {"name": "Leipzig", "lat": 51.3397, "lon": 12.3731, "solar_potential": 0.89}
    ]
    
    panel_types = [
        {"type": "monocrystalline", "efficiency": 0.22, "degradation_rate": 0.005},
        {"type": "polycrystalline", "efficiency": 0.18, "degradation_rate": 0.007},
        {"type": "thin_film", "efficiency": 0.15, "degradation_rate": 0.010},
        {"type": "bifacial", "efficiency": 0.20, "degradation_rate": 0.006}
    ]
    
    for i in range(count):
        city = cities[i % len(cities)]
        panel_type = panel_types[i % len(panel_types)]
        
        # Calculate panel capacity based on type and location
        base_capacity = np.random.uniform(3, 20)  # kW
        efficiency_factor = panel_type["efficiency"]
        location_factor = city["solar_potential"]
        capacity = base_capacity * efficiency_factor * location_factor
        
        panel = {
            "panel_id": f"SP{i:06d}",
            "installation_id": f"INST_{i:06d}",
            "location": {
                "latitude": city["lat"] + np.random.uniform(-0.01, 0.01),
                "longitude": city["lon"] + np.random.uniform(-0.01, 0.01),
                "address": f"Solar Installation {i}, {city['name']}, Germany"
            },
            "specifications": {
                "panel_type": panel_type["type"],
                "capacity_kw": round(capacity, 2),
                "efficiency": panel_type["efficiency"],
                "degradation_rate": panel_type["degradation_rate"],
                "manufacturer": np.random.choice(["SunPower", "LG", "Panasonic", "JinkoSolar", "Trina Solar"]),
                "model": f"SP-{panel_type['type'][:3].upper()}-{np.random.randint(100, 999)}",
                "warranty_years": np.random.choice([20, 25, 30]),
                "installation_date": (datetime.utcnow() - timedelta(days=np.random.randint(30, 3650))).isoformat()
            },
            "installation_details": {
                "roof_type": np.random.choice(["flat", "pitched", "tilted"]),
                "tilt_angle": np.random.uniform(15, 45),  # degrees
                "azimuth_angle": np.random.uniform(150, 210),  # degrees (south-facing)
                "shading_factor": np.random.uniform(0.0, 0.3),  # 0-30% shading
                "inverter_type": np.random.choice(["string", "micro", "power_optimizer"]),
                "inverter_capacity_kw": round(capacity * 1.1, 2)  # 10% overcapacity
            },
            "status": np.random.choice(["ACTIVE", "INACTIVE", "MAINTENANCE", "FAULT"], p=[0.75, 0.10, 0.10, 0.05]),
            "performance_tier": np.random.choice(["EXCELLENT", "GOOD", "FAIR", "POOR"], p=[0.4, 0.3, 0.2, 0.1]),
            "metadata": {
                "owner_id": f"OWNER_{i:06d}",
                "installation_company": np.random.choice(["SolarCorp", "GreenEnergy", "SunTech", "EcoSolar"]),
                "grid_connection": np.random.choice(["grid_tied", "hybrid", "off_grid"]),
                "feed_in_tariff": np.random.uniform(0.06, 0.12),  # EUR/kWh
                "monitoring_enabled": np.random.choice([True, False], p=[0.8, 0.2])
            }
        }
        panels.append(panel)
    
    return panels


def generate_solar_generation_data(panel_ids: List[str], days: int = 30) -> List[Dict[str, Any]]:
    """Generate solar panel generation data."""
    generation_data = []
    
    for panel_id in panel_ids:
        # Base generation capacity (varies by panel)
        base_capacity = np.random.uniform(3, 20)  # kW
        
        for day in range(days):
            for hour in range(24):
                timestamp = datetime.utcnow() - timedelta(days=days-day, hours=23-hour)
                
                # Solar generation pattern (only during daylight hours)
                if 6 <= hour <= 18:  # Daylight hours
                    # Solar irradiance pattern (bell curve)
                    solar_irradiance = max(0, np.sin(np.pi * (hour - 6) / 12))
                    
                    # Weather factor (clouds, etc.)
                    weather_factor = np.random.uniform(0.3, 1.0)
                    
                    # Seasonal factor (higher in summer)
                    day_of_year = timestamp.timetuple().tm_yday
                    seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                    
                    # Panel efficiency degradation over time
                    years_old = (datetime.utcnow() - timestamp).days / 365
                    degradation_factor = max(0.7, 1 - years_old * 0.005)  # 0.5% per year
                    
                    # Calculate generation
                    generation_kw = base_capacity * solar_irradiance * weather_factor * seasonal_factor * degradation_factor
                    generation_kwh = generation_kw * 0.25  # 15-minute intervals
                    
                    # Add some noise
                    generation_kwh += np.random.normal(0, 0.1)
                    generation_kwh = max(0, generation_kwh)  # Can't be negative
                    
                else:
                    generation_kwh = 0.0
                
                # Calculate additional metrics
                voltage = 400 + np.random.normal(0, 10)  # V
                current = (generation_kw * 1000) / voltage if voltage > 0 else 0  # A
                power_factor = np.random.uniform(0.95, 1.0)
                
                generation_record = {
                    "panel_id": panel_id,
                    "timestamp": timestamp,
                    "generation_kwh": round(generation_kwh, 3),
                    "generation_kw": round(generation_kw, 2),
                    "voltage_v": round(voltage, 1),
                    "current_a": round(current, 2),
                    "power_factor": round(power_factor, 3),
                    "efficiency": round(generation_kwh / (base_capacity * 0.25) if base_capacity > 0 else 0, 3),
                    "temperature_c": round(15 + 20 * np.sin(np.pi * (hour - 6) / 12) + np.random.normal(0, 3), 1),
                    "irradiance_wm2": round(800 * solar_irradiance * weather_factor if 6 <= hour <= 18 else 0, 0),
                    "quality_score": round(np.random.uniform(0.8, 1.0), 3),
                    "anomaly_detected": np.random.choice([True, False], p=[0.03, 0.97])  # 3% anomaly rate
                }
                generation_data.append(generation_record)
    
    return generation_data


def generate_solar_forecast_data(panel_id: str, days: int = 7) -> pd.DataFrame:
    """Generate solar generation forecast data."""
    timestamps = pd.date_range(
        start=datetime.utcnow(),
        periods=days * 24,
        freq='H'
    )
    
    # Generate forecast data
    base_capacity = np.random.uniform(5, 15)  # kW
    forecasts = []
    
    for i, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        
        if 6 <= hour <= 18:  # Daylight hours
            # Solar irradiance forecast
            solar_irradiance = max(0, np.sin(np.pi * (hour - 6) / 12))
            
            # Weather forecast factor
            weather_factor = np.random.uniform(0.4, 1.0)
            
            # Seasonal factor
            day_of_year = timestamp.timetuple().tm_yday
            seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            # Forecast generation
            forecast_generation = base_capacity * solar_irradiance * weather_factor * seasonal_factor
            actual_generation = forecast_generation + np.random.normal(0, 0.5)  # Add some variation
            actual_generation = max(0, actual_generation)
            
            # Confidence intervals
            confidence_lower = max(0, forecast_generation - np.random.uniform(1, 3))
            confidence_upper = forecast_generation + np.random.uniform(1, 3)
            
        else:
            forecast_generation = 0.0
            actual_generation = 0.0
            confidence_lower = 0.0
            confidence_upper = 0.0
        
        forecasts.append({
            'panel_id': panel_id,
            'timestamp': timestamp,
            'forecast_generation_kwh': round(forecast_generation, 3),
            'actual_generation_kwh': round(actual_generation, 3),
            'confidence_lower': round(confidence_lower, 3),
            'confidence_upper': round(confidence_upper, 3),
            'irradiance_forecast_wm2': round(800 * solar_irradiance * weather_factor if 6 <= hour <= 18 else 0, 0),
            'weather_condition': np.random.choice(['sunny', 'partly_cloudy', 'cloudy', 'overcast'], p=[0.4, 0.3, 0.2, 0.1]),
            'forecast_accuracy': round(np.random.uniform(0.8, 0.95), 3),
            'model_name': 'solar_forecast_v2'
        })
    
    return pd.DataFrame(forecasts)


def generate_solar_anomaly_data(panel_id: str, count: int = 10) -> List[Dict[str, Any]]:
    """Generate solar panel anomaly test data."""
    anomalies = []
    
    anomaly_types = [
        "generation_drop", "voltage_anomaly", "current_anomaly", "efficiency_loss",
        "inverter_fault", "shading_issue", "soiling", "hotspot_detection"
    ]
    
    for i in range(count):
        anomaly = {
            "anomaly_id": f"SANOM_{panel_id}_{i:06d}",
            "panel_id": panel_id,
            "timestamp": datetime.utcnow() - timedelta(hours=i),
            "anomaly_type": np.random.choice(anomaly_types),
            "severity": np.random.choice(["low", "medium", "high", "critical"]),
            "confidence": round(np.random.uniform(0.7, 1.0), 3),
            "description": f"Solar panel anomaly detected in {panel_id}",
            "affected_parameters": np.random.choice([
                ["generation_kwh"],
                ["voltage_v"],
                ["current_a"],
                ["efficiency"],
                ["generation_kwh", "efficiency"],
                ["voltage_v", "current_a"]
            ]),
            "detection_method": np.random.choice([
                "statistical", "ml_model", "threshold", "pattern_matching"
            ]),
            "status": np.random.choice(["new", "investigating", "resolved", "false_positive"]),
            "impact_assessment": {
                "generation_loss_percent": round(np.random.uniform(5, 50), 1),
                "estimated_revenue_loss_eur": round(np.random.uniform(10, 200), 2),
                "repair_priority": np.random.choice(["low", "medium", "high", "urgent"])
            }
        }
        anomalies.append(anomaly)
    
    return anomalies


def generate_solar_performance_data(panel_ids: List[str]) -> List[Dict[str, Any]]:
    """Generate solar panel performance metrics data."""
    performance_data = []
    
    for panel_id in panel_ids:
        # Calculate performance metrics
        daily_generation = np.random.uniform(15, 45)  # kWh/day
        monthly_generation = daily_generation * 30
        yearly_generation = daily_generation * 365
        
        # Performance ratios
        performance_ratio = np.random.uniform(0.75, 0.95)
        capacity_factor = np.random.uniform(0.15, 0.25)
        availability = np.random.uniform(0.95, 1.0)
        
        performance = {
            "panel_id": panel_id,
            "timestamp": datetime.utcnow(),
            "daily_generation_kwh": round(daily_generation, 2),
            "monthly_generation_kwh": round(monthly_generation, 2),
            "yearly_generation_kwh": round(yearly_generation, 2),
            "performance_ratio": round(performance_ratio, 3),
            "capacity_factor": round(capacity_factor, 3),
            "availability": round(availability, 3),
            "efficiency": round(np.random.uniform(0.15, 0.22), 3),
            "degradation_rate": round(np.random.uniform(0.003, 0.008), 4),
            "co2_savings_kg": round(yearly_generation * 0.4, 1),  # 0.4 kg CO2 per kWh
            "revenue_eur": round(yearly_generation * np.random.uniform(0.08, 0.12), 2),
            "payback_period_years": round(np.random.uniform(8, 15), 1),
            "lifetime_remaining_years": round(np.random.uniform(15, 25), 1)
        }
        performance_data.append(performance)
    
    return performance_data


def generate_solar_weather_correlation_data(panel_ids: List[str], days: int = 30) -> pd.DataFrame:
    """Generate solar generation and weather correlation data."""
    data = []
    
    for panel_id in panel_ids:
        for day in range(days):
            for hour in range(24):
                timestamp = datetime.utcnow() - timedelta(days=days-day, hours=23-hour)
                
                # Weather conditions
                temperature = 15 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 3)
                humidity = 60 + 20 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 10)
                cloud_cover = np.random.uniform(0, 100)
                wind_speed = np.random.exponential(3)
                
                # Solar irradiance (affected by weather)
                base_irradiance = 800 if 6 <= hour <= 18 else 0
                cloud_factor = 1 - (cloud_cover / 100) * 0.7  # Clouds reduce irradiance
                irradiance = base_irradiance * cloud_factor
                
                # Generation (correlated with irradiance and temperature)
                base_generation = np.random.uniform(3, 15)  # kW
                irradiance_factor = irradiance / 800
                temp_factor = 1 - 0.004 * (temperature - 25)  # Temperature coefficient
                generation = base_generation * irradiance_factor * temp_factor
                generation = max(0, generation)
                
                data.append({
                    'panel_id': panel_id,
                    'timestamp': timestamp,
                    'generation_kw': generation,
                    'irradiance_wm2': irradiance,
                    'temperature_c': temperature,
                    'humidity_percent': humidity,
                    'cloud_cover_percent': cloud_cover,
                    'wind_speed_ms': wind_speed,
                    'day_of_year': timestamp.timetuple().tm_yday,
                    'hour_of_day': hour,
                    'is_weekend': 1 if timestamp.weekday() >= 5 else 0
                })
    
    return pd.DataFrame(data)


def generate_solar_maintenance_data(panel_ids: List[str]) -> List[Dict[str, Any]]:
    """Generate solar panel maintenance data."""
    maintenance_records = []
    
    maintenance_types = [
        "cleaning", "inverter_replacement", "panel_replacement", "wiring_check",
        "monitoring_system_update", "warranty_claim", "performance_optimization"
    ]
    
    for panel_id in panel_ids:
        # Generate 0-3 maintenance records per panel
        num_records = np.random.poisson(1.2)
        
        for i in range(num_records):
            maintenance = {
                "maintenance_id": f"MAINT_{panel_id}_{i:06d}",
                "panel_id": panel_id,
                "maintenance_type": np.random.choice(maintenance_types),
                "timestamp": datetime.utcnow() - timedelta(days=np.random.randint(1, 365)),
                "duration_hours": round(np.random.uniform(0.5, 8.0), 1),
                "technician": f"Tech_{np.random.randint(1000, 9999)}",
                "cost_eur": round(np.random.uniform(50, 2000), 2),
                "description": f"Maintenance performed on {panel_id}",
                "status": np.random.choice(["completed", "in_progress", "scheduled"], p=[0.7, 0.1, 0.2]),
                "parts_replaced": np.random.choice([
                    ["inverter"],
                    ["cables"],
                    ["monitoring_device"],
                    ["inverter", "cables"],
                    []
                ]),
                "performance_improvement": round(np.random.uniform(0, 0.15), 3),
                "next_maintenance_due": (datetime.utcnow() + timedelta(days=np.random.randint(30, 365))).isoformat()
            }
            maintenance_records.append(maintenance)
    
    return maintenance_records
