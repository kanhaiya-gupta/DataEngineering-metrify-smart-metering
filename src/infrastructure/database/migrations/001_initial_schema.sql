-- Initial Database Schema Migration
-- Creates all tables for the Metrify Smart Metering system

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create smart_meters table
CREATE TABLE smart_meters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meter_id VARCHAR(255) UNIQUE NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    address TEXT NOT NULL,
    manufacturer VARCHAR(255) NOT NULL,
    model VARCHAR(255) NOT NULL,
    installation_date TIMESTAMP NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
    quality_tier VARCHAR(50) NOT NULL DEFAULT 'UNKNOWN',
    installed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_reading_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    firmware_version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    metadata JSONB,
    version INTEGER NOT NULL DEFAULT 0
);

-- Create indexes for smart_meters
CREATE INDEX idx_smart_meters_meter_id ON smart_meters(meter_id);
CREATE INDEX idx_smart_meters_status ON smart_meters(status);
CREATE INDEX idx_smart_meters_quality_tier ON smart_meters(quality_tier);
CREATE INDEX idx_smart_meters_location ON smart_meters(latitude, longitude);
CREATE INDEX idx_smart_meters_created_at ON smart_meters(created_at);

-- Create meter_readings table
CREATE TABLE meter_readings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meter_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    voltage DECIMAL(10, 3) NOT NULL,
    current DECIMAL(10, 3) NOT NULL,
    power_factor DECIMAL(5, 4) NOT NULL,
    frequency DECIMAL(6, 3) NOT NULL,
    active_power DECIMAL(12, 3) NOT NULL,
    reactive_power DECIMAL(12, 3) NOT NULL,
    apparent_power DECIMAL(12, 3) NOT NULL,
    data_quality_score DECIMAL(3, 2) NOT NULL DEFAULT 1.0,
    is_anomaly BOOLEAN NOT NULL DEFAULT FALSE,
    anomaly_type VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (meter_id) REFERENCES smart_meters(meter_id) ON DELETE CASCADE
);

-- Create indexes for meter_readings
CREATE INDEX idx_meter_readings_meter_id ON meter_readings(meter_id);
CREATE INDEX idx_meter_readings_timestamp ON meter_readings(timestamp);
CREATE INDEX idx_meter_readings_quality ON meter_readings(data_quality_score);
CREATE INDEX idx_meter_readings_anomaly ON meter_readings(is_anomaly);

-- Create meter_events table
CREATE TABLE meter_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meter_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    aggregate_version INTEGER NOT NULL,
    event_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (meter_id) REFERENCES smart_meters(meter_id) ON DELETE CASCADE
);

-- Create indexes for meter_events
CREATE INDEX idx_meter_events_meter_id ON meter_events(meter_id);
CREATE INDEX idx_meter_events_event_type ON meter_events(event_type);
CREATE INDEX idx_meter_events_occurred_at ON meter_events(occurred_at);

-- Create grid_operators table
CREATE TABLE grid_operators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operator_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    operator_type VARCHAR(100) NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    address TEXT NOT NULL,
    contact_email VARCHAR(255),
    contact_phone VARCHAR(50),
    website VARCHAR(255),
    status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    grid_capacity_mw DECIMAL(12, 3),
    voltage_level_kv DECIMAL(8, 3),
    coverage_area_km2 DECIMAL(12, 3),
    metadata JSONB,
    version INTEGER NOT NULL DEFAULT 0
);

-- Create indexes for grid_operators
CREATE INDEX idx_grid_operators_operator_id ON grid_operators(operator_id);
CREATE INDEX idx_grid_operators_status ON grid_operators(status);
CREATE INDEX idx_grid_operators_type ON grid_operators(operator_type);
CREATE INDEX idx_grid_operators_location ON grid_operators(latitude, longitude);

-- Create grid_statuses table
CREATE TABLE grid_statuses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operator_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    voltage_level DECIMAL(10, 3) NOT NULL,
    frequency DECIMAL(6, 3) NOT NULL,
    load_percentage DECIMAL(5, 2) NOT NULL,
    stability_score DECIMAL(3, 2) NOT NULL,
    power_quality_score DECIMAL(3, 2) NOT NULL,
    total_generation_mw DECIMAL(12, 3),
    total_consumption_mw DECIMAL(12, 3),
    grid_frequency_hz DECIMAL(6, 3),
    voltage_deviation_percent DECIMAL(5, 2),
    data_quality_score DECIMAL(3, 2) NOT NULL DEFAULT 1.0,
    is_anomaly BOOLEAN NOT NULL DEFAULT FALSE,
    anomaly_type VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (operator_id) REFERENCES grid_operators(operator_id) ON DELETE CASCADE
);

-- Create indexes for grid_statuses
CREATE INDEX idx_grid_statuses_operator_id ON grid_statuses(operator_id);
CREATE INDEX idx_grid_statuses_timestamp ON grid_statuses(timestamp);
CREATE INDEX idx_grid_statuses_quality ON grid_statuses(data_quality_score);
CREATE INDEX idx_grid_statuses_anomaly ON grid_statuses(is_anomaly);

-- Create grid_events table
CREATE TABLE grid_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operator_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    aggregate_version INTEGER NOT NULL,
    event_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (operator_id) REFERENCES grid_operators(operator_id) ON DELETE CASCADE
);

-- Create indexes for grid_events
CREATE INDEX idx_grid_events_operator_id ON grid_events(operator_id);
CREATE INDEX idx_grid_events_event_type ON grid_events(event_type);
CREATE INDEX idx_grid_events_occurred_at ON grid_events(occurred_at);

-- Create weather_stations table
CREATE TABLE weather_stations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    station_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    station_type VARCHAR(100) NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    address TEXT NOT NULL,
    operator VARCHAR(255) NOT NULL,
    contact_email VARCHAR(255),
    contact_phone VARCHAR(50),
    status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_observations INTEGER NOT NULL DEFAULT 0,
    average_quality_score DECIMAL(3, 2) NOT NULL DEFAULT 1.0,
    last_observation_at TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 0
);

-- Create indexes for weather_stations
CREATE INDEX idx_weather_stations_station_id ON weather_stations(station_id);
CREATE INDEX idx_weather_stations_status ON weather_stations(status);
CREATE INDEX idx_weather_stations_operator ON weather_stations(operator);
CREATE INDEX idx_weather_stations_location ON weather_stations(latitude, longitude);

-- Create weather_observations table
CREATE TABLE weather_observations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    station_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    temperature_celsius DECIMAL(5, 2) NOT NULL,
    humidity_percent DECIMAL(5, 2) NOT NULL,
    pressure_hpa DECIMAL(8, 2) NOT NULL,
    wind_speed_ms DECIMAL(6, 2) NOT NULL,
    wind_direction_degrees DECIMAL(6, 2) NOT NULL,
    cloud_cover_percent DECIMAL(5, 2) NOT NULL,
    visibility_km DECIMAL(6, 2) NOT NULL,
    uv_index DECIMAL(4, 1),
    precipitation_mm DECIMAL(8, 2),
    data_quality_score DECIMAL(3, 2) NOT NULL DEFAULT 1.0,
    is_anomaly BOOLEAN NOT NULL DEFAULT FALSE,
    anomaly_type VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (station_id) REFERENCES weather_stations(station_id) ON DELETE CASCADE
);

-- Create indexes for weather_observations
CREATE INDEX idx_weather_observations_station_id ON weather_observations(station_id);
CREATE INDEX idx_weather_observations_timestamp ON weather_observations(timestamp);
CREATE INDEX idx_weather_observations_quality ON weather_observations(data_quality_score);
CREATE INDEX idx_weather_observations_anomaly ON weather_observations(is_anomaly);

-- Create weather_events table
CREATE TABLE weather_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    station_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    aggregate_version INTEGER NOT NULL,
    event_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (station_id) REFERENCES weather_stations(station_id) ON DELETE CASCADE
);

-- Create indexes for weather_events
CREATE INDEX idx_weather_events_station_id ON weather_events(station_id);
CREATE INDEX idx_weather_events_event_type ON weather_events(event_type);
CREATE INDEX idx_weather_events_occurred_at ON weather_events(occurred_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_smart_meters_updated_at BEFORE UPDATE ON smart_meters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_grid_operators_updated_at BEFORE UPDATE ON grid_operators
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_weather_stations_updated_at BEFORE UPDATE ON weather_stations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE VIEW active_smart_meters AS
SELECT * FROM smart_meters WHERE status = 'ACTIVE';

CREATE VIEW active_grid_operators AS
SELECT * FROM grid_operators WHERE status = 'ACTIVE';

CREATE VIEW active_weather_stations AS
SELECT * FROM weather_stations WHERE status = 'ACTIVE';

CREATE VIEW recent_meter_readings AS
SELECT mr.*, sm.meter_id, sm.manufacturer, sm.model
FROM meter_readings mr
JOIN smart_meters sm ON mr.meter_id = sm.meter_id
WHERE mr.timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours';

CREATE VIEW recent_grid_statuses AS
SELECT gs.*, go.name as operator_name
FROM grid_statuses gs
JOIN grid_operators go ON gs.operator_id = go.operator_id
WHERE gs.timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours';

CREATE VIEW recent_weather_observations AS
SELECT wo.*, ws.name as station_name
FROM weather_observations wo
JOIN weather_stations ws ON wo.station_id = ws.station_id
WHERE wo.timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours';
