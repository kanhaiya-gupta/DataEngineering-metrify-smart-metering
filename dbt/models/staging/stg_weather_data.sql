-- Staging model for weather station data
-- Transforms raw weather data into a clean, consistent format

with source_data as (
    select * from {{ source('raw', 'weather_observations') }}
),

cleaned_data as (
    select
        -- Primary keys
        station_id,
        observation_id,
        
        -- Timestamps
        observation_timestamp,
        created_at,
        updated_at,
        
        -- Weather measurements
        temperature_celsius,
        humidity_percent,
        pressure_hpa,
        wind_speed_mps,
        wind_direction_degrees,
        precipitation_mm,
        visibility_km,
        cloud_cover_percent,
        uv_index,
        
        -- Quality metrics
        data_quality_score,
        is_anomaly,
        anomaly_score,
        
        -- Metadata
        observation_type,
        observation_source,
        data_quality_flags,
        
        -- Location context
        station_latitude,
        station_longitude,
        station_elevation_m,
        station_address,
        weather_station_type,
        
        -- Data lineage
        ingestion_batch_id,
        source_system,
        raw_data_hash
        
    from source_data
    where observation_timestamp is not null
      and station_id is not null
),

-- Data quality validations
validated_data as (
    select
        *,
        -- Validate temperature values
        case 
            when temperature_celsius is not null and (temperature_celsius < {{ var('data_quality').min_temperature_celsius }} 
                 or temperature_celsius > {{ var('data_quality').max_temperature_celsius }}) then false
            else true
        end as is_valid_temperature,
        
        -- Validate humidity values
        case 
            when humidity_percent is not null and (humidity_percent < {{ var('data_quality').min_humidity_percent }} 
                 or humidity_percent > {{ var('data_quality').max_humidity_percent }}) then false
            else true
        end as is_valid_humidity,
        
        -- Validate pressure values
        case 
            when pressure_hpa is not null and (pressure_hpa < 800 or pressure_hpa > 1100) then false
            else true
        end as is_valid_pressure,
        
        -- Validate wind speed values
        case 
            when wind_speed_mps is not null and wind_speed_mps < 0 then false
            else true
        end as is_valid_wind_speed,
        
        -- Validate wind direction values
        case 
            when wind_direction_degrees is not null and (wind_direction_degrees < 0 or wind_direction_degrees > 360) then false
            else true
        end as is_valid_wind_direction,
        
        -- Validate precipitation values
        case 
            when precipitation_mm is not null and precipitation_mm < 0 then false
            else true
        end as is_valid_precipitation,
        
        -- Validate visibility values
        case 
            when visibility_km is not null and visibility_km < 0 then false
            else true
        end as is_valid_visibility,
        
        -- Validate cloud cover values
        case 
            when cloud_cover_percent is not null and (cloud_cover_percent < 0 or cloud_cover_percent > 100) then false
            else true
        end as is_valid_cloud_cover,
        
        -- Validate UV index values
        case 
            when uv_index is not null and (uv_index < 0 or uv_index > 15) then false
            else true
        end as is_valid_uv_index,
        
        -- Calculate data quality score
        case 
            when is_valid_temperature and is_valid_humidity and is_valid_pressure 
                 and is_valid_wind_speed and is_valid_wind_direction then 'EXCELLENT'
            when is_valid_temperature and is_valid_humidity and is_valid_pressure then 'GOOD'
            when is_valid_temperature and is_valid_humidity then 'FAIR'
            else 'POOR'
        end as calculated_quality_tier
        
    from cleaned_data
),

-- Add business logic
enriched_data as (
    select
        *,
        -- Time-based features
        extract(hour from observation_timestamp) as observation_hour,
        extract(dow from observation_timestamp) as day_of_week,
        extract(month from observation_timestamp) as observation_month,
        extract(quarter from observation_timestamp) as observation_quarter,
        extract(year from observation_timestamp) as observation_year,
        
        -- Peak hour classification
        case 
            when extract(hour from observation_timestamp) between {{ var('business_rules').peak_hours_start }} 
                 and {{ var('business_rules').peak_hours_end }} then true
            else false
        end as is_peak_hour_calculated,
        
        -- Weekend classification
        case 
            when extract(dow from observation_timestamp) in ({{ var('business_rules').weekend_days | join(', ') }}) then true
            else false
        end as is_weekend_calculated,
        
        -- Temperature categories
        case 
            when temperature_celsius is null then 'UNKNOWN'
            when temperature_celsius < -10 then 'VERY_COLD'
            when temperature_celsius < 0 then 'COLD'
            when temperature_celsius < 10 then 'COOL'
            when temperature_celsius < 20 then 'MILD'
            when temperature_celsius < 30 then 'WARM'
            when temperature_celsius < 40 then 'HOT'
            else 'VERY_HOT'
        end as temperature_category,
        
        -- Humidity categories
        case 
            when humidity_percent is null then 'UNKNOWN'
            when humidity_percent < 30 then 'DRY'
            when humidity_percent < 50 then 'COMFORTABLE'
            when humidity_percent < 70 then 'MODERATE'
            else 'HUMID'
        end as humidity_category,
        
        -- Wind categories
        case 
            when wind_speed_mps is null then 'UNKNOWN'
            when wind_speed_mps < 1 then 'CALM'
            when wind_speed_mps < 5 then 'LIGHT'
            when wind_speed_mps < 10 then 'MODERATE'
            when wind_speed_mps < 20 then 'STRONG'
            else 'VERY_STRONG'
        end as wind_category,
        
        -- Weather condition classification
        case 
            when precipitation_mm > 0 and wind_speed_mps > 10 then 'STORMY'
            when precipitation_mm > 5 then 'HEAVY_RAIN'
            when precipitation_mm > 0 then 'RAINY'
            when cloud_cover_percent > 80 then 'CLOUDY'
            when cloud_cover_percent > 50 then 'PARTLY_CLOUDY'
            when cloud_cover_percent > 20 then 'MOSTLY_CLEAR'
            else 'CLEAR'
        end as weather_condition,
        
        -- Quality flags
        case 
            when not is_valid_temperature then 'INVALID_TEMPERATURE'
            when not is_valid_humidity then 'INVALID_HUMIDITY'
            when not is_valid_pressure then 'INVALID_PRESSURE'
            when not is_valid_wind_speed then 'INVALID_WIND_SPEED'
            when not is_valid_wind_direction then 'INVALID_WIND_DIRECTION'
            when not is_valid_precipitation then 'INVALID_PRECIPITATION'
            when not is_valid_visibility then 'INVALID_VISIBILITY'
            when not is_valid_cloud_cover then 'INVALID_CLOUD_COVER'
            when not is_valid_uv_index then 'INVALID_UV_INDEX'
            else 'VALID'
        end as quality_flag
        
    from validated_data
)

select * from enriched_data
