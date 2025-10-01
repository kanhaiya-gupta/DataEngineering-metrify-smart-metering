-- Additional Constraints Migration
-- Adds data integrity constraints and foreign key relationships

-- Add foreign key constraints
ALTER TABLE meter_readings 
ADD CONSTRAINT fk_meter_readings_meter_id 
FOREIGN KEY (meter_id) REFERENCES smart_meters(meter_id) 
ON DELETE CASCADE;

ALTER TABLE meter_events 
ADD CONSTRAINT fk_meter_events_meter_id 
FOREIGN KEY (meter_id) REFERENCES smart_meters(meter_id) 
ON DELETE CASCADE;

ALTER TABLE grid_statuses 
ADD CONSTRAINT fk_grid_statuses_operator_id 
FOREIGN KEY (operator_id) REFERENCES grid_operators(operator_id) 
ON DELETE CASCADE;

ALTER TABLE grid_events 
ADD CONSTRAINT fk_grid_events_operator_id 
FOREIGN KEY (operator_id) REFERENCES grid_operators(operator_id) 
ON DELETE CASCADE;

ALTER TABLE weather_observations 
ADD CONSTRAINT fk_weather_observations_station_id 
FOREIGN KEY (station_id) REFERENCES weather_stations(station_id) 
ON DELETE CASCADE;

ALTER TABLE weather_events 
ADD CONSTRAINT fk_weather_events_station_id 
FOREIGN KEY (station_id) REFERENCES weather_stations(station_id) 
ON DELETE CASCADE;

-- Add check constraints for data validation
ALTER TABLE smart_meters 
ADD CONSTRAINT chk_smart_meters_voltage_range 
CHECK (latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180);

ALTER TABLE smart_meters 
ADD CONSTRAINT chk_smart_meters_quality_tier 
CHECK (quality_tier IN ('EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'UNKNOWN'));

ALTER TABLE smart_meters 
ADD CONSTRAINT chk_smart_meters_status 
CHECK (status IN ('ACTIVE', 'INACTIVE', 'FAULTY', 'MAINTENANCE'));

ALTER TABLE meter_readings 
ADD CONSTRAINT chk_meter_readings_voltage 
CHECK (voltage >= 0 AND voltage <= 1000);

ALTER TABLE meter_readings 
ADD CONSTRAINT chk_meter_readings_current 
CHECK (current >= 0 AND current <= 1000);

ALTER TABLE meter_readings 
ADD CONSTRAINT chk_meter_readings_power_factor 
CHECK (power_factor >= 0 AND power_factor <= 1);

ALTER TABLE meter_readings 
ADD CONSTRAINT chk_meter_readings_frequency 
CHECK (frequency >= 45 AND frequency <= 55);

ALTER TABLE meter_readings 
ADD CONSTRAINT chk_meter_readings_power_values 
CHECK (active_power >= 0 AND reactive_power >= 0 AND apparent_power >= 0);

ALTER TABLE meter_readings 
ADD CONSTRAINT chk_meter_readings_quality_score 
CHECK (data_quality_score >= 0 AND data_quality_score <= 1);

ALTER TABLE grid_operators 
ADD CONSTRAINT chk_grid_operators_location 
CHECK (latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180);

ALTER TABLE grid_operators 
ADD CONSTRAINT chk_grid_operators_status 
CHECK (status IN ('ACTIVE', 'INACTIVE', 'MAINTENANCE'));

ALTER TABLE grid_operators 
ADD CONSTRAINT chk_grid_operators_capacity 
CHECK (grid_capacity_mw IS NULL OR grid_capacity_mw > 0);

ALTER TABLE grid_operators 
ADD CONSTRAINT chk_grid_operators_voltage_level 
CHECK (voltage_level_kv IS NULL OR voltage_level_kv > 0);

ALTER TABLE grid_statuses 
ADD CONSTRAINT chk_grid_statuses_voltage_level 
CHECK (voltage_level >= 0 AND voltage_level <= 1000);

ALTER TABLE grid_statuses 
ADD CONSTRAINT chk_grid_statuses_frequency 
CHECK (frequency >= 45 AND frequency <= 55);

ALTER TABLE grid_statuses 
ADD CONSTRAINT chk_grid_statuses_load_percentage 
CHECK (load_percentage >= 0 AND load_percentage <= 100);

ALTER TABLE grid_statuses 
ADD CONSTRAINT chk_grid_statuses_stability_score 
CHECK (stability_score >= 0 AND stability_score <= 1);

ALTER TABLE grid_statuses 
ADD CONSTRAINT chk_grid_statuses_power_quality_score 
CHECK (power_quality_score >= 0 AND power_quality_score <= 1);

ALTER TABLE grid_statuses 
ADD CONSTRAINT chk_grid_statuses_quality_score 
CHECK (data_quality_score >= 0 AND data_quality_score <= 1);

ALTER TABLE weather_stations 
ADD CONSTRAINT chk_weather_stations_location 
CHECK (latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180);

ALTER TABLE weather_stations 
ADD CONSTRAINT chk_weather_stations_status 
CHECK (status IN ('ACTIVE', 'MAINTENANCE', 'CALIBRATION', 'ERROR', 'OFFLINE', 'MALFUNCTION', 'SENSOR_ERROR', 'INACTIVE', 'SUSPENDED', 'RETIRED'));

ALTER TABLE weather_stations 
ADD CONSTRAINT chk_weather_stations_quality_score 
CHECK (average_quality_score >= 0 AND average_quality_score <= 1);

ALTER TABLE weather_stations 
ADD CONSTRAINT chk_weather_stations_total_observations 
CHECK (total_observations >= 0);

ALTER TABLE weather_observations 
ADD CONSTRAINT chk_weather_observations_temperature 
CHECK (temperature_celsius >= -50 AND temperature_celsius <= 60);

ALTER TABLE weather_observations 
ADD CONSTRAINT chk_weather_observations_humidity 
CHECK (humidity_percent >= 0 AND humidity_percent <= 100);

ALTER TABLE weather_observations 
ADD CONSTRAINT chk_weather_observations_pressure 
CHECK (pressure_hpa >= 950 AND pressure_hpa <= 1050);

ALTER TABLE weather_observations 
ADD CONSTRAINT chk_weather_observations_wind_speed 
CHECK (wind_speed_ms >= 0 AND wind_speed_ms <= 100);

ALTER TABLE weather_observations 
ADD CONSTRAINT chk_weather_observations_wind_direction 
CHECK (wind_direction_degrees >= 0 AND wind_direction_degrees <= 360);

ALTER TABLE weather_observations 
ADD CONSTRAINT chk_weather_observations_cloud_cover 
CHECK (cloud_cover_percent >= 0 AND cloud_cover_percent <= 100);

ALTER TABLE weather_observations 
ADD CONSTRAINT chk_weather_observations_visibility 
CHECK (visibility_km >= 0 AND visibility_km <= 50);

ALTER TABLE weather_observations 
ADD CONSTRAINT chk_weather_observations_uv_index 
CHECK (uv_index IS NULL OR (uv_index >= 0 AND uv_index <= 15));

ALTER TABLE weather_observations 
ADD CONSTRAINT chk_weather_observations_precipitation 
CHECK (precipitation_mm IS NULL OR precipitation_mm >= 0);

ALTER TABLE weather_observations 
ADD CONSTRAINT chk_weather_observations_quality_score 
CHECK (data_quality_score >= 0 AND data_quality_score <= 1);

-- Add unique constraints
ALTER TABLE smart_meters 
ADD CONSTRAINT uk_smart_meters_meter_id UNIQUE (meter_id);

ALTER TABLE grid_operators 
ADD CONSTRAINT uk_grid_operators_operator_id UNIQUE (operator_id);

ALTER TABLE weather_stations 
ADD CONSTRAINT uk_weather_stations_station_id UNIQUE (station_id);

-- Add not null constraints
ALTER TABLE smart_meters 
ALTER COLUMN meter_id SET NOT NULL;

ALTER TABLE smart_meters 
ALTER COLUMN latitude SET NOT NULL;

ALTER TABLE smart_meters 
ALTER COLUMN longitude SET NOT NULL;

ALTER TABLE smart_meters 
ALTER COLUMN address SET NOT NULL;

ALTER TABLE smart_meters 
ALTER COLUMN manufacturer SET NOT NULL;

ALTER TABLE smart_meters 
ALTER COLUMN model SET NOT NULL;

ALTER TABLE smart_meters 
ALTER COLUMN installation_date SET NOT NULL;

ALTER TABLE smart_meters 
ALTER COLUMN status SET NOT NULL;

ALTER TABLE smart_meters 
ALTER COLUMN quality_tier SET NOT NULL;

ALTER TABLE grid_operators 
ALTER COLUMN operator_id SET NOT NULL;

ALTER TABLE grid_operators 
ALTER COLUMN name SET NOT NULL;

ALTER TABLE grid_operators 
ALTER COLUMN operator_type SET NOT NULL;

ALTER TABLE grid_operators 
ALTER COLUMN latitude SET NOT NULL;

ALTER TABLE grid_operators 
ALTER COLUMN longitude SET NOT NULL;

ALTER TABLE grid_operators 
ALTER COLUMN address SET NOT NULL;

ALTER TABLE grid_operators 
ALTER COLUMN status SET NOT NULL;

ALTER TABLE weather_stations 
ALTER COLUMN station_id SET NOT NULL;

ALTER TABLE weather_stations 
ALTER COLUMN name SET NOT NULL;

ALTER TABLE weather_stations 
ALTER COLUMN station_type SET NOT NULL;

ALTER TABLE weather_stations 
ALTER COLUMN latitude SET NOT NULL;

ALTER TABLE weather_stations 
ALTER COLUMN longitude SET NOT NULL;

ALTER TABLE weather_stations 
ALTER COLUMN address SET NOT NULL;

ALTER TABLE weather_stations 
ALTER COLUMN operator SET NOT NULL;

ALTER TABLE weather_stations 
ALTER COLUMN status SET NOT NULL;
