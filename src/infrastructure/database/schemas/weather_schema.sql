-- Weather Station Schema Definition
-- Defines the complete schema for weather station domain

-- Weather Stations Table
CREATE TABLE IF NOT EXISTS weather_stations (
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
    version INTEGER NOT NULL DEFAULT 0,
    
    -- Constraints
    CONSTRAINT chk_weather_stations_location 
        CHECK (latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180),
    CONSTRAINT chk_weather_stations_status 
        CHECK (status IN ('ACTIVE', 'MAINTENANCE', 'CALIBRATION', 'ERROR', 'OFFLINE', 'MALFUNCTION', 'SENSOR_ERROR', 'INACTIVE', 'SUSPENDED', 'RETIRED')),
    CONSTRAINT chk_weather_stations_quality_score 
        CHECK (average_quality_score >= 0 AND average_quality_score <= 1),
    CONSTRAINT chk_weather_stations_total_observations 
        CHECK (total_observations >= 0)
);

-- Weather Observations Table
CREATE TABLE IF NOT EXISTS weather_observations (
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
    
    -- Foreign Key
    CONSTRAINT fk_weather_observations_station_id 
        FOREIGN KEY (station_id) REFERENCES weather_stations(station_id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT chk_weather_observations_temperature 
        CHECK (temperature_celsius >= -50 AND temperature_celsius <= 60),
    CONSTRAINT chk_weather_observations_humidity 
        CHECK (humidity_percent >= 0 AND humidity_percent <= 100),
    CONSTRAINT chk_weather_observations_pressure 
        CHECK (pressure_hpa >= 950 AND pressure_hpa <= 1050),
    CONSTRAINT chk_weather_observations_wind_speed 
        CHECK (wind_speed_ms >= 0 AND wind_speed_ms <= 100),
    CONSTRAINT chk_weather_observations_wind_direction 
        CHECK (wind_direction_degrees >= 0 AND wind_direction_degrees <= 360),
    CONSTRAINT chk_weather_observations_cloud_cover 
        CHECK (cloud_cover_percent >= 0 AND cloud_cover_percent <= 100),
    CONSTRAINT chk_weather_observations_visibility 
        CHECK (visibility_km >= 0 AND visibility_km <= 50),
    CONSTRAINT chk_weather_observations_uv_index 
        CHECK (uv_index IS NULL OR (uv_index >= 0 AND uv_index <= 15)),
    CONSTRAINT chk_weather_observations_precipitation 
        CHECK (precipitation_mm IS NULL OR precipitation_mm >= 0),
    CONSTRAINT chk_weather_observations_quality_score 
        CHECK (data_quality_score >= 0 AND data_quality_score <= 1)
);

-- Weather Events Table
CREATE TABLE IF NOT EXISTS weather_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    station_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    aggregate_version INTEGER NOT NULL,
    event_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key
    CONSTRAINT fk_weather_events_station_id 
        FOREIGN KEY (station_id) REFERENCES weather_stations(station_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_weather_stations_station_id ON weather_stations(station_id);
CREATE INDEX IF NOT EXISTS idx_weather_stations_status ON weather_stations(status);
CREATE INDEX IF NOT EXISTS idx_weather_stations_operator ON weather_stations(operator);
CREATE INDEX IF NOT EXISTS idx_weather_stations_location ON weather_stations(latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_weather_observations_station_id ON weather_observations(station_id);
CREATE INDEX IF NOT EXISTS idx_weather_observations_timestamp ON weather_observations(timestamp);
CREATE INDEX IF NOT EXISTS idx_weather_observations_quality ON weather_observations(data_quality_score);
CREATE INDEX IF NOT EXISTS idx_weather_observations_anomaly ON weather_observations(is_anomaly);

CREATE INDEX IF NOT EXISTS idx_weather_events_station_id ON weather_events(station_id);
CREATE INDEX IF NOT EXISTS idx_weather_events_event_type ON weather_events(event_type);
CREATE INDEX IF NOT EXISTS idx_weather_events_occurred_at ON weather_events(occurred_at);

-- Views
CREATE OR REPLACE VIEW active_weather_stations AS
SELECT * FROM weather_stations WHERE status = 'ACTIVE';

CREATE OR REPLACE VIEW recent_weather_observations AS
SELECT wo.*, ws.name as station_name
FROM weather_observations wo
JOIN weather_stations ws ON wo.station_id = ws.station_id
WHERE wo.timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours';
