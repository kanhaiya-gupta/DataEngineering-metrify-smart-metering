"""
Grid Operator Test Data Fixtures

Comprehensive test data for grid operators, grid status, and grid events.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any


def generate_grid_operator_data(count: int = 10) -> List[Dict[str, Any]]:
    """Generate grid operator test data."""
    operators = []
    
    # German grid operators
    operator_templates = [
        {
            "operator_id": "TENNET",
            "name": "TenneT TSO B.V.",
            "country": "Netherlands",
            "region": "North",
            "grid_type": "transmission",
            "voltage_levels": [380, 220, 110],
            "service_area": ["Lower Saxony", "North Rhine-Westphalia", "Hamburg", "Bremen"]
        },
        {
            "operator_id": "50HERTZ",
            "name": "50Hertz Transmission GmbH",
            "country": "Germany",
            "region": "East",
            "grid_type": "transmission",
            "voltage_levels": [380, 220, 110],
            "service_area": ["Berlin", "Brandenburg", "Mecklenburg-Vorpommern", "Saxony", "Saxony-Anhalt", "Thuringia"]
        },
        {
            "operator_id": "AMPRION",
            "name": "Amprion GmbH",
            "country": "Germany",
            "region": "West",
            "grid_type": "transmission",
            "voltage_levels": [380, 220, 110],
            "service_area": ["North Rhine-Westphalia", "Rhineland-Palatinate", "Saarland", "Hesse", "Baden-Württemberg"]
        },
        {
            "operator_id": "TRANSNETBW",
            "name": "TransnetBW GmbH",
            "country": "Germany",
            "region": "South",
            "grid_type": "transmission",
            "voltage_levels": [380, 220, 110],
            "service_area": ["Baden-Württemberg"]
        }
    ]
    
    for i in range(count):
        template = operator_templates[i % len(operator_templates)]
        
        operator = {
            "operator_id": template["operator_id"],
            "name": template["name"],
            "country": template["country"],
            "region": template["region"],
            "grid_type": template["grid_type"],
            "voltage_levels_kv": template["voltage_levels"],
            "service_area": template["service_area"],
            "status": np.random.choice(["ACTIVE", "INACTIVE", "MAINTENANCE"], p=[0.9, 0.05, 0.05]),
            "contact_info": {
                "email": f"contact@{template['operator_id'].lower()}.com",
                "phone": f"+49 {np.random.randint(200, 999)} {np.random.randint(1000000, 9999999)}",
                "website": f"https://www.{template['operator_id'].lower()}.com",
                "address": f"{template['name']} Headquarters, Germany"
            },
            "technical_specs": {
                "grid_frequency_hz": 50.0,
                "nominal_voltage_380kv": True,
                "nominal_voltage_220kv": True,
                "nominal_voltage_110kv": True,
                "grid_stability_margin": round(np.random.uniform(0.05, 0.15), 3),
                "max_load_capacity_mw": np.random.randint(5000, 15000),
                "renewable_penetration_percent": round(np.random.uniform(30, 60), 1)
            },
            "compliance": {
                "grid_code_version": "2023.1",
                "renewable_energy_act_compliant": True,
                "grid_stability_requirements": True,
                "data_protection_gdpr": True,
                "cybersecurity_standards": "ISO 27001"
            },
            "metadata": {
                "established_year": np.random.randint(1990, 2010),
                "employees": np.random.randint(500, 5000),
                "annual_revenue_million_eur": np.random.randint(100, 1000),
                "grid_length_km": np.random.randint(10000, 50000),
                "substations_count": np.random.randint(100, 500)
            }
        }
        operators.append(operator)
    
    return operators


def generate_grid_status_data(operator_ids: List[str], hours: int = 24) -> List[Dict[str, Any]]:
    """Generate grid status test data."""
    status_data = []
    
    for operator_id in operator_ids:
        for hour in range(hours):
            timestamp = datetime.utcnow() - timedelta(hours=hours-hour)
            
            # Base grid parameters
            base_frequency = 50.0
            base_voltage = 380.0  # kV
            base_load = np.random.uniform(30000, 60000)  # MW
            base_generation = np.random.uniform(25000, 55000)  # MW
            
            # Add daily patterns
            time_factor = 1 + 0.1 * np.sin(2 * np.pi * hour / 24)
            
            # Frequency variation (should be very small)
            frequency = base_frequency + np.random.normal(0, 0.05)
            
            # Voltage variation
            voltage = base_voltage + np.random.normal(0, 5)
            
            # Load and generation with daily patterns
            load = base_load * time_factor + np.random.normal(0, 1000)
            generation = base_generation * time_factor + np.random.normal(0, 1000)
            
            # Calculate grid balance
            balance = generation - load
            reserve_margin = (generation - load) / load if load > 0 else 0
            
            # Grid stability score
            stability_score = 1.0 - abs(frequency - 50.0) / 50.0 - abs(voltage - 380.0) / 380.0
            stability_score = max(0, min(1, stability_score))
            
            # Renewable energy generation
            renewable_generation = generation * np.random.uniform(0.3, 0.6)
            conventional_generation = generation - renewable_generation
            
            status = {
                "operator_id": operator_id,
                "timestamp": timestamp,
                "frequency_hz": round(frequency, 3),
                "voltage_kv": round(voltage, 1),
                "load_mw": round(load, 1),
                "generation_mw": round(generation, 1),
                "renewable_generation_mw": round(renewable_generation, 1),
                "conventional_generation_mw": round(conventional_generation, 1),
                "grid_balance_mw": round(balance, 1),
                "reserve_margin_percent": round(reserve_margin * 100, 2),
                "stability_score": round(stability_score, 3),
                "grid_status": "STABLE" if stability_score > 0.8 else "UNSTABLE",
                "alert_level": "NONE" if stability_score > 0.9 else "LOW" if stability_score > 0.7 else "HIGH",
                "quality_score": round(np.random.uniform(0.85, 1.0), 3)
            }
            status_data.append(status)
    
    return status_data


def generate_grid_events_data(operator_ids: List[str], count: int = 50) -> List[Dict[str, Any]]:
    """Generate grid events test data."""
    events = []
    
    event_types = [
        "frequency_deviation", "voltage_deviation", "load_shedding", "generation_trip",
        "transmission_line_trip", "substation_fault", "renewable_curtailment", "grid_restoration",
        "maintenance_outage", "weather_impact", "cyber_incident", "equipment_failure"
    ]
    
    for i in range(count):
        operator_id = np.random.choice(operator_ids)
        event_type = np.random.choice(event_types)
        
        # Event severity based on type
        severity_weights = {
            "frequency_deviation": [0.1, 0.3, 0.4, 0.2],
            "voltage_deviation": [0.2, 0.4, 0.3, 0.1],
            "load_shedding": [0.0, 0.1, 0.3, 0.6],
            "generation_trip": [0.1, 0.2, 0.4, 0.3],
            "transmission_line_trip": [0.0, 0.2, 0.5, 0.3],
            "substation_fault": [0.0, 0.1, 0.4, 0.5],
            "renewable_curtailment": [0.3, 0.4, 0.2, 0.1],
            "grid_restoration": [0.2, 0.3, 0.3, 0.2],
            "maintenance_outage": [0.4, 0.4, 0.2, 0.0],
            "weather_impact": [0.1, 0.2, 0.4, 0.3],
            "cyber_incident": [0.0, 0.1, 0.3, 0.6],
            "equipment_failure": [0.1, 0.3, 0.4, 0.2]
        }
        
        severity = np.random.choice(["low", "medium", "high", "critical"], 
                                  p=severity_weights.get(event_type, [0.25, 0.25, 0.25, 0.25]))
        
        event = {
            "event_id": f"GRID_EVT_{i:06d}",
            "operator_id": operator_id,
            "event_type": event_type,
            "severity": severity,
            "timestamp": datetime.utcnow() - timedelta(hours=np.random.randint(0, 168)),  # Last week
            "duration_minutes": round(np.random.exponential(30), 1),
            "description": f"Grid event: {event_type} in {operator_id}",
            "affected_components": np.random.choice([
                ["transmission_line_1"],
                ["substation_A"],
                ["generator_1", "generator_2"],
                ["transmission_line_1", "substation_A"],
                ["grid_entire"]
            ]),
            "impact_assessment": {
                "customers_affected": np.random.randint(0, 100000),
                "power_loss_mw": round(np.random.uniform(0, 1000), 1),
                "estimated_cost_eur": round(np.random.uniform(0, 1000000), 2),
                "recovery_time_hours": round(np.random.uniform(0.5, 24), 1)
            },
            "root_cause": np.random.choice([
                "equipment_failure", "weather_conditions", "human_error", "cyber_attack",
                "maintenance_issue", "overload", "external_factors", "unknown"
            ]),
            "status": np.random.choice(["ongoing", "resolved", "investigating", "false_alarm"], p=[0.1, 0.7, 0.15, 0.05]),
            "response_actions": [
                "automatic_load_shedding",
                "generator_startup",
                "transmission_rerouting",
                "customer_notification",
                "emergency_procedures"
            ][:np.random.randint(1, 4)],
            "metadata": {
                "detection_method": np.random.choice(["automatic", "manual", "customer_report"]),
                "confidence": round(np.random.uniform(0.7, 1.0), 3),
                "investigation_required": severity in ["high", "critical"],
                "regulatory_report_required": severity == "critical"
            }
        }
        events.append(event)
    
    return events


def generate_grid_forecast_data(operator_id: str, days: int = 7) -> pd.DataFrame:
    """Generate grid load and generation forecast data."""
    timestamps = pd.date_range(
        start=datetime.utcnow(),
        periods=days * 24,
        freq='H'
    )
    
    forecasts = []
    
    for i, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base load with daily and weekly patterns
        base_load = 40000  # MW
        daily_factor = 1 + 0.2 * np.sin(2 * np.pi * hour / 24)
        weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * day_of_week / 7)
        
        # Load forecast
        load_forecast = base_load * daily_factor * weekly_factor + np.random.normal(0, 1000)
        
        # Generation forecast (renewable + conventional)
        renewable_forecast = load_forecast * np.random.uniform(0.3, 0.7)
        conventional_forecast = load_forecast - renewable_forecast
        
        # Grid balance forecast
        balance_forecast = (renewable_forecast + conventional_forecast) - load_forecast
        
        # Reserve margin forecast
        reserve_margin = balance_forecast / load_forecast if load_forecast > 0 else 0
        
        forecasts.append({
            'operator_id': operator_id,
            'timestamp': timestamp,
            'load_forecast_mw': round(load_forecast, 1),
            'renewable_forecast_mw': round(renewable_forecast, 1),
            'conventional_forecast_mw': round(conventional_forecast, 1),
            'total_generation_forecast_mw': round(renewable_forecast + conventional_forecast, 1),
            'grid_balance_forecast_mw': round(balance_forecast, 1),
            'reserve_margin_forecast_percent': round(reserve_margin * 100, 2),
            'frequency_forecast_hz': round(50.0 + np.random.normal(0, 0.02), 3),
            'voltage_forecast_kv': round(380.0 + np.random.normal(0, 2), 1),
            'confidence_score': round(np.random.uniform(0.8, 0.95), 3),
            'model_name': 'grid_forecast_v3'
        })
    
    return pd.DataFrame(forecasts)


def generate_grid_anomaly_data(operator_ids: List[str], count: int = 30) -> List[Dict[str, Any]]:
    """Generate grid anomaly test data."""
    anomalies = []
    
    anomaly_types = [
        "frequency_anomaly", "voltage_anomaly", "load_anomaly", "generation_anomaly",
        "stability_anomaly", "reserve_margin_anomaly", "power_quality_anomaly"
    ]
    
    for i in range(count):
        operator_id = np.random.choice(operator_ids)
        anomaly_type = np.random.choice(anomaly_types)
        
        anomaly = {
            "anomaly_id": f"GRID_ANOM_{i:06d}",
            "operator_id": operator_id,
            "anomaly_type": anomaly_type,
            "severity": np.random.choice(["low", "medium", "high", "critical"]),
            "confidence": round(np.random.uniform(0.7, 1.0), 3),
            "timestamp": datetime.utcnow() - timedelta(hours=np.random.randint(0, 72)),
            "description": f"Grid anomaly detected in {operator_id}: {anomaly_type}",
            "affected_parameters": np.random.choice([
                ["frequency_hz"],
                ["voltage_kv"],
                ["load_mw"],
                ["generation_mw"],
                ["stability_score"],
                ["frequency_hz", "voltage_kv"],
                ["load_mw", "generation_mw"]
            ]),
            "detection_method": np.random.choice([
                "statistical", "ml_model", "threshold", "pattern_matching"
            ]),
            "status": np.random.choice(["new", "investigating", "resolved", "false_positive"]),
            "impact_assessment": {
                "grid_stability_impact": round(np.random.uniform(0, 0.3), 3),
                "customers_at_risk": np.random.randint(0, 50000),
                "potential_power_loss_mw": round(np.random.uniform(0, 500), 1),
                "urgency_level": np.random.choice(["low", "medium", "high", "urgent"])
            },
            "recommended_actions": [
                "increase_generation",
                "reduce_load",
                "adjust_voltage",
                "monitor_closely",
                "emergency_procedures"
            ][:np.random.randint(1, 4)]
        }
        anomalies.append(anomaly)
    
    return anomalies


def generate_grid_performance_data(operator_ids: List[str]) -> List[Dict[str, Any]]:
    """Generate grid performance metrics data."""
    performance_data = []
    
    for operator_id in operator_ids:
        performance = {
            "operator_id": operator_id,
            "timestamp": datetime.utcnow(),
            "frequency_stability": round(np.random.uniform(0.95, 1.0), 3),
            "voltage_stability": round(np.random.uniform(0.90, 1.0), 3),
            "grid_availability": round(np.random.uniform(0.99, 1.0), 4),
            "renewable_integration_percent": round(np.random.uniform(30, 60), 1),
            "grid_efficiency": round(np.random.uniform(0.85, 0.95), 3),
            "power_quality_score": round(np.random.uniform(0.8, 1.0), 3),
            "reserve_margin_percent": round(np.random.uniform(5, 20), 1),
            "transmission_losses_percent": round(np.random.uniform(2, 8), 2),
            "grid_flexibility_score": round(np.random.uniform(0.7, 1.0), 3),
            "cybersecurity_score": round(np.random.uniform(0.8, 1.0), 3),
            "compliance_score": round(np.random.uniform(0.9, 1.0), 3),
            "customer_satisfaction": round(np.random.uniform(0.7, 1.0), 3),
            "operational_costs_million_eur": round(np.random.uniform(50, 200), 1),
            "renewable_curtailment_percent": round(np.random.uniform(0, 10), 2)
        }
        performance_data.append(performance)
    
    return performance_data
