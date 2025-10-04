-- Dimension table for weather stations
-- Provides master data for all weather stations in the system

with station_data as (
    select
        station_id,
        station_location_lat as latitude,
        station_location_lon as longitude,
        station_elevation_m as elevation_m,
        station_name,
        observation_type,
        observation_source,
        source_system,
        min(observation_timestamp) as first_observation_date,
        max(observation_timestamp) as last_observation_date,
        count(*) as total_observations,
        avg(temperature_celsius) as avg_temperature_celsius,
        min(temperature_celsius) as min_temperature_celsius,
        max(temperature_celsius) as max_temperature_celsius,
        avg(humidity_percent) as avg_humidity_percent,
        avg(pressure_hpa) as avg_pressure_hpa,
        avg(wind_speed_ms) as avg_wind_speed_ms,
        avg(wind_direction_degrees) as avg_wind_direction_degrees,
        sum(precipitation_mm) as total_precipitation_mm,
        avg(visibility_km) as avg_visibility_km,
        avg(cloud_cover_percent) as avg_cloud_cover_percent,
        avg(uv_index) as avg_uv_index,
        count(case when is_anomaly then 1 end) as total_anomalies,
        avg(anomaly_score) as avg_anomaly_score
        
    from {{ ref('stg_weather_data') }}
    group by 
        station_id, 
        station_location_lat, 
        station_location_lon, 
        station_elevation_m, 
        station_name, 
        observation_type, 
        observation_source, 
        source_system
),

-- Add calculated fields
enriched_stations as (
    select
        *,
        -- Temperature range
        max_temperature_celsius - min_temperature_celsius as temperature_range_celsius,
        
        -- Anomaly metrics
        case 
            when total_observations > 0 then (total_anomalies::float / total_observations) * 100
            else 0
        end as anomaly_percent,
        
        -- Station age
        current_date - first_observation_date::date as days_in_service,
        
        -- Observation frequency
        case 
            when days_in_service > 0 then total_observations::float / days_in_service
            else 0
        end as observations_per_day,
        
        -- Temperature classification
        case 
            when avg_temperature_celsius is null then 'UNKNOWN'
            when avg_temperature_celsius < -10 then 'VERY_COLD'
            when avg_temperature_celsius < 0 then 'COLD'
            when avg_temperature_celsius < 10 then 'COOL'
            when avg_temperature_celsius < 20 then 'MILD'
            when avg_temperature_celsius < 30 then 'WARM'
            when avg_temperature_celsius < 40 then 'HOT'
            else 'VERY_HOT'
        end as temperature_classification,
        
        -- Humidity classification
        case 
            when avg_humidity_percent is null then 'UNKNOWN'
            when avg_humidity_percent < 30 then 'DRY'
            when avg_humidity_percent < 50 then 'COMFORTABLE'
            when avg_humidity_percent < 70 then 'MODERATE'
            else 'HUMID'
        end as humidity_classification,
        
        -- Wind classification
        case 
            when avg_wind_speed_ms is null then 'UNKNOWN'
            when avg_wind_speed_ms < 1 then 'CALM'
            when avg_wind_speed_ms < 5 then 'LIGHT'
            when avg_wind_speed_ms < 10 then 'MODERATE'
            when avg_wind_speed_ms < 20 then 'STRONG'
            else 'VERY_STRONG'
        end as wind_classification,
        
        -- Precipitation classification
        case 
            when total_precipitation_mm is null then 'UNKNOWN'
            when total_precipitation_mm = 0 then 'NO_PRECIPITATION'
            when total_precipitation_mm < 10 then 'LIGHT_PRECIPITATION'
            when total_precipitation_mm < 50 then 'MODERATE_PRECIPITATION'
            when total_precipitation_mm < 100 then 'HEAVY_PRECIPITATION'
            else 'VERY_HEAVY_PRECIPITATION'
        end as precipitation_classification,
        
        -- Anomaly classification
        case 
            when anomaly_percent >= 20 then 'HIGH_ANOMALY'
            when anomaly_percent >= 10 then 'MEDIUM_ANOMALY'
            when anomaly_percent > 0 then 'LOW_ANOMALY'
            else 'NO_ANOMALY'
        end as anomaly_classification,
        
        -- Station status
        case 
            when last_observation_date < current_date - interval '7 days' then 'INACTIVE'
            when anomaly_percent >= 50 then 'FAULTY'
            when observations_per_day < 1 then 'LOW_ACTIVITY'
            else 'ACTIVE'
        end as station_status,
        
        -- Geographic region (simplified)
        case 
            when latitude is null or longitude is null then 'UNKNOWN'
            when latitude > 50 and longitude > 0 then 'NORTH_EAST'
            when latitude > 50 and longitude <= 0 then 'NORTH_WEST'
            when latitude <= 50 and longitude > 0 then 'SOUTH_EAST'
            when latitude <= 50 and longitude <= 0 then 'SOUTH_WEST'
            else 'UNKNOWN'
        end as geographic_region
        
    from station_data
),

-- Add additional metadata
final_stations as (
    select
        *,
        -- Station ID components
        split_part(station_id, '-', 1) as station_type,
        split_part(station_id, '-', 2) as station_region,
        split_part(station_id, '-', 3) as station_sequence,
        
        -- Station name components
        split_part(station_name, ' ', 1) as station_type,
        split_part(station_name, ' ', 2) as station_location,
        
        -- Elevation classification
        case 
            when elevation_m is null then 'UNKNOWN'
            when elevation_m < 100 then 'LOW_ELEVATION'
            when elevation_m < 500 then 'MEDIUM_ELEVATION'
            when elevation_m < 1000 then 'HIGH_ELEVATION'
            else 'VERY_HIGH_ELEVATION'
        end as elevation_classification,
        
        -- Data completeness
        case 
            when latitude is not null and longitude is not null and station_name is not null 
                 and elevation_m is not null then 'COMPLETE'
            when latitude is not null and longitude is not null and station_name is not null then 'PARTIAL'
            else 'INCOMPLETE'
        end as data_completeness,
        
        -- Last updated
        current_timestamp as last_updated
        
    from enriched_stations
)

select * from final_stations
