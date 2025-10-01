-- Additional Indexes Migration
-- Adds performance-optimized indexes for common queries

-- Smart Meter Indexes
CREATE INDEX CONCURRENTLY idx_smart_meters_created_at_status ON smart_meters(created_at, status);
CREATE INDEX CONCURRENTLY idx_smart_meters_quality_tier_status ON smart_meters(quality_tier, status);
CREATE INDEX CONCURRENTLY idx_smart_meters_manufacturer_model ON smart_meters(manufacturer, model);
CREATE INDEX CONCURRENTLY idx_smart_meters_installed_at ON smart_meters(installed_at);

-- Meter Reading Indexes
CREATE INDEX CONCURRENTLY idx_meter_readings_meter_timestamp ON meter_readings(meter_id, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_meter_readings_quality_timestamp ON meter_readings(data_quality_score, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_meter_readings_anomaly_timestamp ON meter_readings(is_anomaly, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_meter_readings_voltage_range ON meter_readings(voltage) WHERE voltage BETWEEN 200 AND 250;
CREATE INDEX CONCURRENTLY idx_meter_readings_current_range ON meter_readings(current) WHERE current BETWEEN 0 AND 100;

-- Meter Event Indexes
CREATE INDEX CONCURRENTLY idx_meter_events_type_timestamp ON meter_events(event_type, occurred_at DESC);
CREATE INDEX CONCURRENTLY idx_meter_events_aggregate_version ON meter_events(meter_id, aggregate_version);

-- Grid Operator Indexes
CREATE INDEX CONCURRENTLY idx_grid_operators_type_status ON grid_operators(operator_type, status);
CREATE INDEX CONCURRENTLY idx_grid_operators_capacity ON grid_operators(grid_capacity_mw) WHERE grid_capacity_mw IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_grid_operators_voltage_level ON grid_operators(voltage_level_kv) WHERE voltage_level_kv IS NOT NULL;

-- Grid Status Indexes
CREATE INDEX CONCURRENTLY idx_grid_statuses_operator_timestamp ON grid_statuses(operator_id, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_grid_statuses_stability_timestamp ON grid_statuses(stability_score, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_grid_statuses_quality_timestamp ON grid_statuses(data_quality_score, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_grid_statuses_anomaly_timestamp ON grid_statuses(is_anomaly, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_grid_statuses_frequency_range ON grid_statuses(frequency) WHERE frequency BETWEEN 49.5 AND 50.5;

-- Grid Event Indexes
CREATE INDEX CONCURRENTLY idx_grid_events_type_timestamp ON grid_events(event_type, occurred_at DESC);
CREATE INDEX CONCURRENTLY idx_grid_events_aggregate_version ON grid_events(operator_id, aggregate_version);

-- Weather Station Indexes
CREATE INDEX CONCURRENTLY idx_weather_stations_type_status ON weather_stations(station_type, status);
CREATE INDEX CONCURRENTLY idx_weather_stations_operator_status ON weather_stations(operator, status);
CREATE INDEX CONCURRENTLY idx_weather_stations_quality_score ON weather_stations(average_quality_score) WHERE average_quality_score > 0;
CREATE INDEX CONCURRENTLY idx_weather_stations_last_observation ON weather_stations(last_observation_at) WHERE last_observation_at IS NOT NULL;

-- Weather Observation Indexes
CREATE INDEX CONCURRENTLY idx_weather_observations_station_timestamp ON weather_observations(station_id, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_weather_observations_quality_timestamp ON weather_observations(data_quality_score, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_weather_observations_anomaly_timestamp ON weather_observations(is_anomaly, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_weather_observations_temperature_range ON weather_observations(temperature_celsius) WHERE temperature_celsius BETWEEN -20 AND 40;
CREATE INDEX CONCURRENTLY idx_weather_observations_humidity_range ON weather_observations(humidity_percent) WHERE humidity_percent BETWEEN 20 AND 80;

-- Weather Event Indexes
CREATE INDEX CONCURRENTLY idx_weather_events_type_timestamp ON weather_events(event_type, occurred_at DESC);
CREATE INDEX CONCURRENTLY idx_weather_events_aggregate_version ON weather_events(station_id, aggregate_version);

-- Composite Indexes for Common Queries
CREATE INDEX CONCURRENTLY idx_smart_meters_location_status ON smart_meters(latitude, longitude, status);
CREATE INDEX CONCURRENTLY idx_grid_operators_location_status ON grid_operators(latitude, longitude, status);
CREATE INDEX CONCURRENTLY idx_weather_stations_location_status ON weather_stations(latitude, longitude, status);

-- Partial Indexes for Active Records
CREATE INDEX CONCURRENTLY idx_active_smart_meters ON smart_meters(meter_id) WHERE status = 'ACTIVE';
CREATE INDEX CONCURRENTLY idx_active_grid_operators ON grid_operators(operator_id) WHERE status = 'ACTIVE';
CREATE INDEX CONCURRENTLY idx_active_weather_stations ON weather_stations(station_id) WHERE status = 'ACTIVE';

-- Partial Indexes for Anomalous Records
CREATE INDEX CONCURRENTLY idx_anomalous_meter_readings ON meter_readings(meter_id, timestamp) WHERE is_anomaly = true;
CREATE INDEX CONCURRENTLY idx_anomalous_grid_statuses ON grid_statuses(operator_id, timestamp) WHERE is_anomaly = true;
CREATE INDEX CONCURRENTLY idx_anomalous_weather_observations ON weather_observations(station_id, timestamp) WHERE is_anomaly = true;
