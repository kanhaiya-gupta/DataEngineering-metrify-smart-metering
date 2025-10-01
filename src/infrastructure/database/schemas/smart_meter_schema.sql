-- Smart Meter Schema Definition
-- Defines the complete schema for smart meter domain

-- Smart Meters Table
CREATE TABLE IF NOT EXISTS smart_meters (
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
    version INTEGER NOT NULL DEFAULT 0,
    
    -- Constraints
    CONSTRAINT chk_smart_meters_location 
        CHECK (latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180),
    CONSTRAINT chk_smart_meters_quality_tier 
        CHECK (quality_tier IN ('EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'UNKNOWN')),
    CONSTRAINT chk_smart_meters_status 
        CHECK (status IN ('ACTIVE', 'INACTIVE', 'FAULTY', 'MAINTENANCE'))
);

-- Meter Readings Table
CREATE TABLE IF NOT EXISTS meter_readings (
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
    
    -- Foreign Key
    CONSTRAINT fk_meter_readings_meter_id 
        FOREIGN KEY (meter_id) REFERENCES smart_meters(meter_id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT chk_meter_readings_voltage 
        CHECK (voltage >= 0 AND voltage <= 1000),
    CONSTRAINT chk_meter_readings_current 
        CHECK (current >= 0 AND current <= 1000),
    CONSTRAINT chk_meter_readings_power_factor 
        CHECK (power_factor >= 0 AND power_factor <= 1),
    CONSTRAINT chk_meter_readings_frequency 
        CHECK (frequency >= 45 AND frequency <= 55),
    CONSTRAINT chk_meter_readings_power_values 
        CHECK (active_power >= 0 AND reactive_power >= 0 AND apparent_power >= 0),
    CONSTRAINT chk_meter_readings_quality_score 
        CHECK (data_quality_score >= 0 AND data_quality_score <= 1)
);

-- Meter Events Table
CREATE TABLE IF NOT EXISTS meter_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meter_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    aggregate_version INTEGER NOT NULL,
    event_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key
    CONSTRAINT fk_meter_events_meter_id 
        FOREIGN KEY (meter_id) REFERENCES smart_meters(meter_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_smart_meters_meter_id ON smart_meters(meter_id);
CREATE INDEX IF NOT EXISTS idx_smart_meters_status ON smart_meters(status);
CREATE INDEX IF NOT EXISTS idx_smart_meters_quality_tier ON smart_meters(quality_tier);
CREATE INDEX IF NOT EXISTS idx_smart_meters_location ON smart_meters(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_smart_meters_created_at ON smart_meters(created_at);

CREATE INDEX IF NOT EXISTS idx_meter_readings_meter_id ON meter_readings(meter_id);
CREATE INDEX IF NOT EXISTS idx_meter_readings_timestamp ON meter_readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_meter_readings_quality ON meter_readings(data_quality_score);
CREATE INDEX IF NOT EXISTS idx_meter_readings_anomaly ON meter_readings(is_anomaly);

CREATE INDEX IF NOT EXISTS idx_meter_events_meter_id ON meter_events(meter_id);
CREATE INDEX IF NOT EXISTS idx_meter_events_event_type ON meter_events(event_type);
CREATE INDEX IF NOT EXISTS idx_meter_events_occurred_at ON meter_events(occurred_at);

-- Views
CREATE OR REPLACE VIEW active_smart_meters AS
SELECT * FROM smart_meters WHERE status = 'ACTIVE';

CREATE OR REPLACE VIEW recent_meter_readings AS
SELECT mr.*, sm.meter_id, sm.manufacturer, sm.model
FROM meter_readings mr
JOIN smart_meters sm ON mr.meter_id = sm.meter_id
WHERE mr.timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours';
