-- Grid Operator Schema Definition
-- Defines the complete schema for grid operator domain

-- Grid Operators Table
CREATE TABLE IF NOT EXISTS grid_operators (
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
    version INTEGER NOT NULL DEFAULT 0,
    
    -- Constraints
    CONSTRAINT chk_grid_operators_location 
        CHECK (latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180),
    CONSTRAINT chk_grid_operators_status 
        CHECK (status IN ('ACTIVE', 'INACTIVE', 'MAINTENANCE')),
    CONSTRAINT chk_grid_operators_capacity 
        CHECK (grid_capacity_mw IS NULL OR grid_capacity_mw > 0),
    CONSTRAINT chk_grid_operators_voltage_level 
        CHECK (voltage_level_kv IS NULL OR voltage_level_kv > 0)
);

-- Grid Status Table
CREATE TABLE IF NOT EXISTS grid_statuses (
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
    
    -- Foreign Key
    CONSTRAINT fk_grid_statuses_operator_id 
        FOREIGN KEY (operator_id) REFERENCES grid_operators(operator_id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT chk_grid_statuses_voltage_level 
        CHECK (voltage_level >= 0 AND voltage_level <= 1000),
    CONSTRAINT chk_grid_statuses_frequency 
        CHECK (frequency >= 45 AND frequency <= 55),
    CONSTRAINT chk_grid_statuses_load_percentage 
        CHECK (load_percentage >= 0 AND load_percentage <= 100),
    CONSTRAINT chk_grid_statuses_stability_score 
        CHECK (stability_score >= 0 AND stability_score <= 1),
    CONSTRAINT chk_grid_statuses_power_quality_score 
        CHECK (power_quality_score >= 0 AND power_quality_score <= 1),
    CONSTRAINT chk_grid_statuses_quality_score 
        CHECK (data_quality_score >= 0 AND data_quality_score <= 1)
);

-- Grid Events Table
CREATE TABLE IF NOT EXISTS grid_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operator_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    aggregate_version INTEGER NOT NULL,
    event_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key
    CONSTRAINT fk_grid_events_operator_id 
        FOREIGN KEY (operator_id) REFERENCES grid_operators(operator_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_grid_operators_operator_id ON grid_operators(operator_id);
CREATE INDEX IF NOT EXISTS idx_grid_operators_status ON grid_operators(status);
CREATE INDEX IF NOT EXISTS idx_grid_operators_type ON grid_operators(operator_type);
CREATE INDEX IF NOT EXISTS idx_grid_operators_location ON grid_operators(latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_grid_statuses_operator_id ON grid_statuses(operator_id);
CREATE INDEX IF NOT EXISTS idx_grid_statuses_timestamp ON grid_statuses(timestamp);
CREATE INDEX IF NOT EXISTS idx_grid_statuses_quality ON grid_statuses(data_quality_score);
CREATE INDEX IF NOT EXISTS idx_grid_statuses_anomaly ON grid_statuses(is_anomaly);

CREATE INDEX IF NOT EXISTS idx_grid_events_operator_id ON grid_events(operator_id);
CREATE INDEX IF NOT EXISTS idx_grid_events_event_type ON grid_events(event_type);
CREATE INDEX IF NOT EXISTS idx_grid_events_occurred_at ON grid_events(occurred_at);

-- Views
CREATE OR REPLACE VIEW active_grid_operators AS
SELECT * FROM grid_operators WHERE status = 'ACTIVE';

CREATE OR REPLACE VIEW recent_grid_statuses AS
SELECT gs.*, go.name as operator_name
FROM grid_statuses gs
JOIN grid_operators go ON gs.operator_id = go.operator_id
WHERE gs.timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours';
