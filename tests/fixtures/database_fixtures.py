"""
Database Test Fixtures

Comprehensive test data for database testing, including schemas, sample data, and test utilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import psycopg2
import psycopg2.extras
import json
from sqlalchemy import create_engine, text


def generate_database_schema() -> Dict[str, str]:
    """Generate PostgreSQL database schema SQL statements."""
    
    schema = {
        "smart_meters": """
        CREATE TABLE smart_meters (
            meter_id VARCHAR(20) PRIMARY KEY,
            location_latitude DECIMAL(10, 8) NOT NULL,
            location_longitude DECIMAL(11, 8) NOT NULL,
            location_address TEXT NOT NULL,
            manufacturer VARCHAR(50),
            model VARCHAR(50),
            firmware_version VARCHAR(20),
            installation_date DATE,
            status VARCHAR(20) NOT NULL,
            quality_tier VARCHAR(20) NOT NULL,
            customer_id VARCHAR(20),
            installation_type VARCHAR(20),
            tariff_type VARCHAR(20),
            grid_operator VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        "meter_readings": """
        CREATE TABLE meter_readings (
            id SERIAL PRIMARY KEY,
            meter_id VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            energy_consumed_kwh DECIMAL(10, 3) NOT NULL,
            power_factor DECIMAL(4, 3) NOT NULL,
            voltage_v DECIMAL(6, 1) NOT NULL,
            current_a DECIMAL(6, 2) NOT NULL,
            frequency_hz DECIMAL(5, 2) NOT NULL,
            temperature_c DECIMAL(5, 1),
            quality_score DECIMAL(4, 3),
            anomaly_detected BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (meter_id) REFERENCES smart_meters(meter_id)
        );
        """,
        
        "weather_stations": """
        CREATE TABLE weather_stations (
            station_id VARCHAR(20) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            location_latitude DECIMAL(10, 8) NOT NULL,
            location_longitude DECIMAL(11, 8) NOT NULL,
            location_address TEXT NOT NULL,
            station_type VARCHAR(20) NOT NULL,
            status VARCHAR(20) NOT NULL,
            elevation_m INTEGER,
            installation_date DATE,
            operator VARCHAR(50),
            data_quality VARCHAR(20),
            update_frequency VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        "weather_observations": """
        CREATE TABLE weather_observations (
            id SERIAL PRIMARY KEY,
            station_id VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            temperature_c DECIMAL(5, 1),
            humidity_percent DECIMAL(5, 1),
            pressure_hpa DECIMAL(7, 2),
            wind_speed_ms DECIMAL(5, 1),
            wind_direction_deg DECIMAL(6, 1),
            precipitation_mm DECIMAL(6, 2),
            visibility_km DECIMAL(5, 1),
            cloud_cover_percent DECIMAL(5, 1),
            solar_radiation_wm2 DECIMAL(6, 1),
            quality_score DECIMAL(4, 3),
            data_source VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (station_id) REFERENCES weather_stations(station_id)
        );
        """,
        
        "grid_operators": """
        CREATE TABLE grid_operators (
            operator_id VARCHAR(20) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            country VARCHAR(50) NOT NULL,
            region VARCHAR(50),
            grid_type VARCHAR(20) NOT NULL,
            voltage_levels_kv TEXT,
            service_area TEXT,
            status VARCHAR(20) NOT NULL,
            contact_email VARCHAR(100),
            contact_phone VARCHAR(20),
            website VARCHAR(100),
            address TEXT,
            grid_frequency_hz DECIMAL(4, 1) DEFAULT 50.0,
            grid_stability_margin DECIMAL(4, 3),
            max_load_capacity_mw INTEGER,
            renewable_penetration_percent DECIMAL(5, 2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        "grid_status": """
        CREATE TABLE grid_status (
            id SERIAL PRIMARY KEY,
            operator_id VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            frequency_hz DECIMAL(6, 3) NOT NULL,
            voltage_kv DECIMAL(6, 1) NOT NULL,
            load_mw DECIMAL(10, 1) NOT NULL,
            generation_mw DECIMAL(10, 1) NOT NULL,
            renewable_generation_mw DECIMAL(10, 1),
            conventional_generation_mw DECIMAL(10, 1),
            grid_balance_mw DECIMAL(10, 1),
            reserve_margin_percent DECIMAL(5, 2),
            stability_score DECIMAL(4, 3),
            grid_status VARCHAR(20),
            alert_level VARCHAR(20),
            quality_score DECIMAL(4, 3),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (operator_id) REFERENCES grid_operators(operator_id)
        );
        """,
        
        "solar_panels": """
        CREATE TABLE solar_panels (
            panel_id VARCHAR(20) PRIMARY KEY,
            installation_id VARCHAR(20) NOT NULL,
            location_latitude DECIMAL(10, 8) NOT NULL,
            location_longitude DECIMAL(11, 8) NOT NULL,
            location_address TEXT NOT NULL,
            panel_type VARCHAR(20) NOT NULL,
            capacity_kw DECIMAL(6, 2) NOT NULL,
            efficiency DECIMAL(4, 3) NOT NULL,
            degradation_rate DECIMAL(6, 4),
            manufacturer VARCHAR(50),
            model VARCHAR(50),
            warranty_years INTEGER,
            installation_date DATE,
            roof_type VARCHAR(20),
            tilt_angle DECIMAL(5, 2),
            azimuth_angle DECIMAL(6, 2),
            shading_factor DECIMAL(4, 3),
            inverter_type VARCHAR(20),
            inverter_capacity_kw DECIMAL(6, 2),
            status VARCHAR(20) NOT NULL,
            performance_tier VARCHAR(20),
            owner_id VARCHAR(20),
            installation_company VARCHAR(50),
            grid_connection VARCHAR(20),
            feed_in_tariff DECIMAL(6, 4),
            monitoring_enabled BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        "solar_generation": """
        CREATE TABLE solar_generation (
            id SERIAL PRIMARY KEY,
            panel_id VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            generation_kwh DECIMAL(10, 3) NOT NULL,
            generation_kw DECIMAL(8, 2) NOT NULL,
            voltage_v DECIMAL(6, 1),
            current_a DECIMAL(8, 2),
            power_factor DECIMAL(4, 3),
            efficiency DECIMAL(4, 3),
            temperature_c DECIMAL(5, 1),
            irradiance_wm2 DECIMAL(6, 1),
            quality_score DECIMAL(4, 3),
            anomaly_detected BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (panel_id) REFERENCES solar_panels(panel_id)
        );
        """,
        
        "anomalies": """
        CREATE TABLE anomalies (
            anomaly_id VARCHAR(30) PRIMARY KEY,
            meter_id VARCHAR(20),
            panel_id VARCHAR(20),
            station_id VARCHAR(20),
            operator_id VARCHAR(20),
            timestamp TIMESTAMP NOT NULL,
            anomaly_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            confidence DECIMAL(4, 3) NOT NULL,
            description TEXT,
            affected_parameters TEXT,
            detection_method VARCHAR(30),
            status VARCHAR(20) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (meter_id) REFERENCES smart_meters(meter_id),
            FOREIGN KEY (panel_id) REFERENCES solar_panels(panel_id),
            FOREIGN KEY (station_id) REFERENCES weather_stations(station_id),
            FOREIGN KEY (operator_id) REFERENCES grid_operators(operator_id)
        );
        """,
        
        "quality_metrics": """
        CREATE TABLE quality_metrics (
            id SERIAL PRIMARY KEY,
            meter_id VARCHAR(20),
            panel_id VARCHAR(20),
            station_id VARCHAR(20),
            operator_id VARCHAR(20),
            timestamp TIMESTAMP NOT NULL,
            completeness DECIMAL(4, 3) NOT NULL,
            accuracy DECIMAL(4, 3) NOT NULL,
            consistency DECIMAL(4, 3) NOT NULL,
            timeliness DECIMAL(4, 3) NOT NULL,
            validity DECIMAL(4, 3) NOT NULL,
            overall_score DECIMAL(4, 3) NOT NULL,
            issues TEXT,
            recommendations TEXT,
            data_gaps TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (meter_id) REFERENCES smart_meters(meter_id),
            FOREIGN KEY (panel_id) REFERENCES solar_panels(panel_id),
            FOREIGN KEY (station_id) REFERENCES weather_stations(station_id),
            FOREIGN KEY (operator_id) REFERENCES grid_operators(operator_id)
        );
        """,
        
        "ml_models": """
        CREATE TABLE ml_models (
            model_id VARCHAR(30) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            type VARCHAR(30) NOT NULL,
            framework VARCHAR(20) NOT NULL,
            algorithm VARCHAR(30) NOT NULL,
            use_case VARCHAR(50) NOT NULL,
            version VARCHAR(20) NOT NULL,
            status VARCHAR(20) NOT NULL,
            created_at TIMESTAMP NOT NULL,
            last_updated TIMESTAMP NOT NULL,
            performance_metrics TEXT,
            training_info TEXT,
            deployment_info TEXT,
            monitoring TEXT,
            metadata TEXT,
            created_at_db TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at_db TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        "model_predictions": """
        CREATE TABLE model_predictions (
            prediction_id VARCHAR(30) PRIMARY KEY,
            model_id VARCHAR(30) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            input_data JSONB NOT NULL,
            prediction JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES ml_models(model_id)
        );
        """
    }
    
    return schema


def generate_sample_data() -> Dict[str, List[Dict[str, Any]]]:
    """Generate sample data for all tables."""
    
    # Smart meters data
    smart_meters = []
    for i in range(10):
        smart_meters.append({
            "meter_id": f"SM{i:06d}",
            "location_latitude": 52.5200 + np.random.uniform(-0.1, 0.1),
            "location_longitude": 13.4050 + np.random.uniform(-0.1, 0.1),
            "location_address": f"Test Street {i}, Berlin, Germany",
            "manufacturer": np.random.choice(["Siemens", "Landis+Gyr", "Itron", "Elster"]),
            "model": f"Model-{np.random.randint(1000, 9999)}",
            "firmware_version": f"{np.random.randint(1, 5)}.{np.random.randint(0, 9)}.{np.random.randint(0, 9)}",
            "installation_date": (datetime.utcnow() - timedelta(days=np.random.randint(30, 3650))).strftime("%Y-%m-%d"),
            "status": np.random.choice(["ACTIVE", "INACTIVE", "MAINTENANCE"], p=[0.8, 0.15, 0.05]),
            "quality_tier": np.random.choice(["EXCELLENT", "GOOD", "FAIR", "POOR"], p=[0.4, 0.3, 0.2, 0.1]),
            "customer_id": f"CUST{i:06d}",
            "installation_type": np.random.choice(["residential", "commercial", "industrial"]),
            "tariff_type": np.random.choice(["standard", "time_of_use", "demand_response"]),
            "grid_operator": np.random.choice(["TENNET", "50HERTZ", "AMPRION"])
        })
    
    # Meter readings data
    meter_readings = []
    for meter_id in [m["meter_id"] for m in smart_meters]:
        for hour in range(24):
            timestamp = datetime.utcnow() - timedelta(hours=23-hour)
            energy_consumed = np.random.uniform(50, 200)
            
            meter_readings.append({
                "meter_id": meter_id,
                "timestamp": timestamp,
                "energy_consumed_kwh": round(energy_consumed, 3),
                "power_factor": round(np.random.uniform(0.85, 0.98), 3),
                "voltage_v": round(np.random.uniform(220, 240), 1),
                "current_a": round(energy_consumed / 230, 2),
                "frequency_hz": round(np.random.uniform(49.8, 50.2), 2),
                "temperature_c": round(np.random.uniform(15, 35), 1),
                "quality_score": round(np.random.uniform(0.7, 1.0), 3),
                "anomaly_detected": np.random.choice([True, False], p=[0.05, 0.95])
            })
    
    # Weather stations data
    weather_stations = []
    cities = [
        {"name": "Berlin Weather Station", "lat": 52.5200, "lon": 13.4050},
        {"name": "Munich Weather Station", "lat": 48.1351, "lon": 11.5820},
        {"name": "Hamburg Weather Station", "lat": 53.5511, "lon": 9.9937}
    ]
    
    for i, city in enumerate(cities):
        weather_stations.append({
            "station_id": f"WS{i:06d}",
            "name": city["name"],
            "location_latitude": city["lat"],
            "location_longitude": city["lon"],
            "location_address": f"{city['name']}, Germany",
            "station_type": np.random.choice(["AUTOMATIC", "MANUAL", "HYBRID"]),
            "status": "ACTIVE",
            "elevation_m": np.random.randint(10, 100),
            "installation_date": (datetime.utcnow() - timedelta(days=np.random.randint(30, 3650))).strftime("%Y-%m-%d"),
            "operator": "DWD",
            "data_quality": "EXCELLENT",
            "update_frequency": "1min"
        })
    
    # Weather observations data
    weather_observations = []
    for station in weather_stations:
        for hour in range(24):
            timestamp = datetime.utcnow() - timedelta(hours=23-hour)
            
            weather_observations.append({
                "station_id": station["station_id"],
                "timestamp": timestamp,
                "temperature_c": round(np.random.uniform(10, 30), 1),
                "humidity_percent": round(np.random.uniform(40, 80), 1),
                "pressure_hpa": round(np.random.uniform(1000, 1030), 2),
                "wind_speed_ms": round(np.random.uniform(0, 10), 1),
                "wind_direction_deg": round(np.random.uniform(0, 360), 1),
                "precipitation_mm": round(np.random.uniform(0, 5), 2),
                "visibility_km": round(np.random.uniform(5, 20), 1),
                "cloud_cover_percent": round(np.random.uniform(0, 100), 1),
                "solar_radiation_wm2": round(np.random.uniform(0, 800), 1),
                "quality_score": round(np.random.uniform(0.8, 1.0), 3),
                "data_source": "automatic_weather_station"
            })
    
    # Grid operators data
    grid_operators = [
        {
            "operator_id": "TENNET",
            "name": "TenneT TSO B.V.",
            "country": "Netherlands",
            "region": "North",
            "grid_type": "transmission",
            "voltage_levels_kv": json.dumps([380, 220, 110]),
            "service_area": json.dumps(["Lower Saxony", "North Rhine-Westphalia", "Hamburg", "Bremen"]),
            "status": "ACTIVE",
            "contact_email": "contact@tennet.eu",
            "contact_phone": "+49 511 345-0",
            "website": "https://www.tennet.eu",
            "address": "TenneT TSO B.V., Germany",
            "grid_frequency_hz": 50.0,
            "grid_stability_margin": 0.10,
            "max_load_capacity_mw": 10000,
            "renewable_penetration_percent": 45.5
        },
        {
            "operator_id": "50HERTZ",
            "name": "50Hertz Transmission GmbH",
            "country": "Germany",
            "region": "East",
            "grid_type": "transmission",
            "voltage_levels_kv": json.dumps([380, 220, 110]),
            "service_area": json.dumps(["Berlin", "Brandenburg", "Mecklenburg-Vorpommern", "Saxony", "Saxony-Anhalt", "Thuringia"]),
            "status": "ACTIVE",
            "contact_email": "info@50hertz.com",
            "contact_phone": "+49 30 5150-0",
            "website": "https://www.50hertz.com",
            "address": "50Hertz Transmission GmbH, Germany",
            "grid_frequency_hz": 50.0,
            "grid_stability_margin": 0.12,
            "max_load_capacity_mw": 8000,
            "renewable_penetration_percent": 52.3
        }
    ]
    
    # Grid status data
    grid_status = []
    for operator in grid_operators:
        for hour in range(24):
            timestamp = datetime.utcnow() - timedelta(hours=23-hour)
            
            grid_status.append({
                "operator_id": operator["operator_id"],
                "timestamp": timestamp,
                "frequency_hz": round(50.0 + np.random.normal(0, 0.05), 3),
                "voltage_kv": round(380.0 + np.random.normal(0, 5), 1),
                "load_mw": round(np.random.uniform(30000, 60000), 1),
                "generation_mw": round(np.random.uniform(25000, 55000), 1),
                "renewable_generation_mw": round(np.random.uniform(10000, 30000), 1),
                "conventional_generation_mw": round(np.random.uniform(15000, 25000), 1),
                "grid_balance_mw": round(np.random.uniform(-1000, 1000), 1),
                "reserve_margin_percent": round(np.random.uniform(5, 20), 2),
                "stability_score": round(np.random.uniform(0.85, 1.0), 3),
                "grid_status": "STABLE",
                "alert_level": "NONE",
                "quality_score": round(np.random.uniform(0.85, 1.0), 3)
            })
    
    return {
        "smart_meters": smart_meters,
        "meter_readings": meter_readings,
        "weather_stations": weather_stations,
        "weather_observations": weather_observations,
        "grid_operators": grid_operators,
        "grid_status": grid_status
    }


def create_test_database(
    host: str = "localhost",
    port: int = 5432,
    database: str = "metrify_test",
    user: str = "postgres",
    password: str = "postgres"
) -> str:
    """Create a PostgreSQL test database with schema and sample data."""
    
    # Get schema and sample data
    schema = generate_database_schema()
    sample_data = generate_sample_data()
    
    # Database connection string
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    try:
        # Create database connection
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create tables
        for table_name, create_sql in schema.items():
            cursor.execute(create_sql)
        
        # Insert sample data
        for table_name, data in sample_data.items():
            if data:  # Only insert if data exists
                # Get column names from first record
                columns = list(data[0].keys())
                placeholders = ", ".join(["%s" for _ in columns])
                insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                
                # Insert data
                for record in data:
                    values = [record[col] for col in columns]
                    cursor.execute(insert_sql, values)
        
        print(f"PostgreSQL test database created successfully: {database}")
        
    except Exception as e:
        print(f"Error creating test database: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()
    
    return connection_string


def generate_test_queries() -> Dict[str, str]:
    """Generate test SQL queries for database testing."""
    
    queries = {
        "get_all_meters": "SELECT * FROM smart_meters;",
        
        "get_active_meters": "SELECT * FROM smart_meters WHERE status = 'ACTIVE';",
        
        "get_meter_readings_by_date": """
        SELECT mr.*, sm.location_address 
        FROM meter_readings mr 
        JOIN smart_meters sm ON mr.meter_id = sm.meter_id 
        WHERE DATE(mr.timestamp) = DATE('now')
        ORDER BY mr.timestamp DESC;
        """,
        
        "get_weather_data_by_station": """
        SELECT wo.*, ws.name as station_name 
        FROM weather_observations wo 
        JOIN weather_stations ws ON wo.station_id = ws.station_id 
        WHERE wo.station_id = ? 
        ORDER BY wo.timestamp DESC;
        """,
        
        "get_grid_status_latest": """
        SELECT gs.*, go.name as operator_name 
        FROM grid_status gs 
        JOIN grid_operators go ON gs.operator_id = go.operator_id 
        WHERE gs.timestamp = (
            SELECT MAX(timestamp) FROM grid_status WHERE operator_id = gs.operator_id
        );
        """,
        
        "get_anomalies_by_severity": """
        SELECT * FROM anomalies 
        WHERE severity = ? 
        ORDER BY timestamp DESC;
        """,
        
        "get_quality_metrics_summary": """
        SELECT 
            meter_id,
            AVG(overall_score) as avg_quality_score,
            COUNT(*) as total_measurements,
            MIN(timestamp) as first_measurement,
            MAX(timestamp) as last_measurement
        FROM quality_metrics 
        WHERE meter_id IS NOT NULL
        GROUP BY meter_id
        ORDER BY avg_quality_score DESC;
        """,
        
        "get_ml_models_by_status": """
        SELECT * FROM ml_models 
        WHERE status = ? 
        ORDER BY last_updated DESC;
        """,
        
        "get_prediction_count_by_model": """
        SELECT 
            model_id,
            COUNT(*) as prediction_count,
            MIN(timestamp) as first_prediction,
            MAX(timestamp) as last_prediction
        FROM model_predictions 
        GROUP BY model_id
        ORDER BY prediction_count DESC;
        """,
        
        "get_energy_consumption_trends": """
        SELECT 
            DATE(timestamp) as date,
            AVG(energy_consumed_kwh) as avg_consumption,
            MIN(energy_consumed_kwh) as min_consumption,
            MAX(energy_consumed_kwh) as max_consumption
        FROM meter_readings 
        WHERE timestamp >= DATE('now', '-7 days')
        GROUP BY DATE(timestamp)
        ORDER BY date;
        """,
        
        "get_weather_correlation": """
        SELECT 
            wo.temperature_c,
            wo.humidity_percent,
            AVG(mr.energy_consumed_kwh) as avg_energy_consumption
        FROM weather_observations wo
        JOIN meter_readings mr ON DATE(wo.timestamp) = DATE(mr.timestamp)
        WHERE wo.timestamp >= DATE('now', '-30 days')
        GROUP BY wo.temperature_c, wo.humidity_percent
        ORDER BY wo.temperature_c;
        """
    }
    
    return queries


def generate_performance_test_data(table_name: str, record_count: int = 10000) -> List[Dict[str, Any]]:
    """Generate large datasets for performance testing."""
    
    if table_name == "meter_readings":
        data = []
        meter_ids = [f"SM{i:06d}" for i in range(100)]  # 100 meters
        
        for i in range(record_count):
            meter_id = meter_ids[i % len(meter_ids)]
            timestamp = datetime.utcnow() - timedelta(minutes=i)
            
            data.append({
                "meter_id": meter_id,
                "timestamp": timestamp,
                "energy_consumed_kwh": round(np.random.uniform(50, 200), 3),
                "power_factor": round(np.random.uniform(0.85, 0.98), 3),
                "voltage_v": round(np.random.uniform(220, 240), 1),
                "current_a": round(np.random.uniform(5, 15), 2),
                "frequency_hz": round(np.random.uniform(49.8, 50.2), 2),
                "temperature_c": round(np.random.uniform(15, 35), 1),
                "quality_score": round(np.random.uniform(0.7, 1.0), 3),
                "anomaly_detected": np.random.choice([True, False], p=[0.05, 0.95])
            })
        
        return data
    
    elif table_name == "weather_observations":
        data = []
        station_ids = [f"WS{i:06d}" for i in range(10)]  # 10 weather stations
        
        for i in range(record_count):
            station_id = station_ids[i % len(station_ids)]
            timestamp = datetime.utcnow() - timedelta(minutes=i)
            
            data.append({
                "station_id": station_id,
                "timestamp": timestamp,
                "temperature_c": round(np.random.uniform(10, 30), 1),
                "humidity_percent": round(np.random.uniform(40, 80), 1),
                "pressure_hpa": round(np.random.uniform(1000, 1030), 2),
                "wind_speed_ms": round(np.random.uniform(0, 10), 1),
                "wind_direction_deg": round(np.random.uniform(0, 360), 1),
                "precipitation_mm": round(np.random.uniform(0, 5), 2),
                "visibility_km": round(np.random.uniform(5, 20), 1),
                "cloud_cover_percent": round(np.random.uniform(0, 100), 1),
                "solar_radiation_wm2": round(np.random.uniform(0, 800), 1),
                "quality_score": round(np.random.uniform(0.8, 1.0), 3),
                "data_source": "automatic_weather_station"
            })
        
        return data
    
    else:
        return []


def cleanup_test_database(
    host: str = "localhost",
    port: int = 5432,
    database: str = "metrify_test",
    user: str = "postgres",
    password: str = "postgres"
) -> None:
    """Clean up PostgreSQL test database."""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=host,
            port=port,
            database="postgres",  # Connect to default database
            user=user,
            password=password
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Drop test database
        cursor.execute(f"DROP DATABASE IF EXISTS {database}")
        print(f"PostgreSQL test database {database} cleaned up successfully")
        
    except Exception as e:
        print(f"Error cleaning up test database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()


def get_postgresql_connection_string(
    host: str = "localhost",
    port: int = 5432,
    database: str = "metrify_test",
    user: str = "postgres",
    password: str = "postgres"
) -> str:
    """Get PostgreSQL connection string for testing."""
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def create_test_database_with_docker() -> str:
    """Create test database using Docker PostgreSQL container."""
    import subprocess
    import time
    
    # Docker command to run PostgreSQL
    docker_cmd = [
        "docker", "run", "--name", "metrify-test-db",
        "-e", "POSTGRES_PASSWORD=postgres",
        "-e", "POSTGRES_DB=metrify_test",
        "-p", "5433:5432",
        "-d", "postgres:15"
    ]
    
    try:
        # Start PostgreSQL container
        subprocess.run(docker_cmd, check=True)
        
        # Wait for database to be ready
        time.sleep(10)
        
        # Create database with schema
        connection_string = create_test_database(
            host="localhost",
            port=5433,
            database="metrify_test",
            user="postgres",
            password="postgres"
        )
        
        return connection_string
        
    except subprocess.CalledProcessError as e:
        print(f"Error starting PostgreSQL container: {e}")
        raise


def stop_test_database_docker() -> None:
    """Stop and remove PostgreSQL Docker container."""
    import subprocess
    
    try:
        # Stop and remove container
        subprocess.run(["docker", "stop", "metrify-test-db"], check=True)
        subprocess.run(["docker", "rm", "metrify-test-db"], check=True)
        print("PostgreSQL test container stopped and removed")
        
    except subprocess.CalledProcessError as e:
        print(f"Error stopping PostgreSQL container: {e}")


def create_test_database_with_testcontainers() -> str:
    """Create test database using testcontainers (for pytest integration)."""
    from testcontainers.postgres import PostgresContainer
    
    # Start PostgreSQL container
    postgres = PostgresContainer("postgres:15")
    postgres.start()
    
    # Get connection details
    host = postgres.get_container_host_ip()
    port = postgres.get_exposed_port(5432)
    database = postgres.POSTGRES_DB
    user = postgres.POSTGRES_USER
    password = postgres.POSTGRES_PASSWORD
    
    # Create database with schema
    connection_string = create_test_database(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    
    return connection_string
