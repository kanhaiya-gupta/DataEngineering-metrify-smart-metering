-- Staging model for smart meter readings
-- Transforms raw smart meter data into a clean, consistent format

with source_data as (
    select * from {{ source('processed', 'smart_meter_readings') }}
),

cleaned_data as (
    select
        -- Primary keys
        meter_id,
        reading_id,
        
        -- Timestamps
        reading_timestamp,
        created_at,
        updated_at,
        
        -- Reading values
        consumption_kwh,
        voltage_v,
        current_a,
        power_factor,
        frequency_hz,
        
        -- Quality metrics
        quality_score,
        quality_tier,
        is_anomaly,
        anomaly_score,
        
        -- Metadata
        reading_type,
        reading_source,
        data_quality_flags,
        
        -- Location context
        meter_location_lat,
        meter_location_lon,
        meter_address,
        
        -- Business context
        tariff_type,
        time_of_use_period,
        is_peak_hour,
        is_weekend,
        
        -- Data lineage
        ingestion_batch_id,
        source_system,
        raw_data_hash
        
    from source_data
    where reading_timestamp is not null
      and meter_id is not null
      and consumption_kwh is not null
),

-- Data quality validations
validated_data as (
    select
        *,
        -- Validate consumption values
        case 
            when consumption_kwh < {{ var('data_quality').min_consumption_kwh }} then false
            when consumption_kwh > {{ var('data_quality').max_consumption_kwh }} then false
            else true
        end as is_valid_consumption,
        
        -- Validate voltage values
        case 
            when voltage_v is not null and (voltage_v < 200 or voltage_v > 250) then false
            else true
        end as is_valid_voltage,
        
        -- Validate current values
        case 
            when current_a is not null and current_a < 0 then false
            else true
        end as is_valid_current,
        
        -- Validate power factor
        case 
            when power_factor is not null and (power_factor < 0 or power_factor > 1) then false
            else true
        end as is_valid_power_factor,
        
        -- Validate frequency
        case 
            when frequency_hz is not null and (frequency_hz < 45 or frequency_hz > 65) then false
            else true
        end as is_valid_frequency,
        
        -- Calculate data quality score
        case 
            when is_valid_consumption and is_valid_voltage and is_valid_current 
                 and is_valid_power_factor and is_valid_frequency then 'EXCELLENT'
            when is_valid_consumption and is_valid_voltage and is_valid_current then 'GOOD'
            when is_valid_consumption then 'FAIR'
            else 'POOR'
        end as calculated_quality_tier
        
    from cleaned_data
),

-- Add business logic
enriched_data as (
    select
        *,
        -- Time-based features (Snowflake syntax)
        hour(reading_timestamp) as reading_hour,
        dayofweek(reading_timestamp) as day_of_week,
        month(reading_timestamp) as reading_month,
        quarter(reading_timestamp) as reading_quarter,
        year(reading_timestamp) as reading_year,
        
        -- Peak hour classification
        case 
            when hour(reading_timestamp) between {{ var('business_rules').peak_hours_start }} 
                 and {{ var('business_rules').peak_hours_end }} then true
            else false
        end as is_peak_hour_calculated,
        
        -- Off-peak hour classification
        case 
            when hour(reading_timestamp) between {{ var('business_rules').off_peak_hours_start }} 
                 and 23 or hour(reading_timestamp) between 0 
                 and {{ var('business_rules').off_peak_hours_end }} then true
            else false
        end as is_off_peak_hour,
        
        -- Weekend classification
        case 
            when dayofweek(reading_timestamp) in ({{ var('business_rules').weekend_days | join(', ') }}) then true
            else false
        end as is_weekend_calculated,
        
        -- Consumption categories
        case 
            when consumption_kwh = 0 then 'ZERO_CONSUMPTION'
            when consumption_kwh < 0.1 then 'VERY_LOW'
            when consumption_kwh < 1.0 then 'LOW'
            when consumption_kwh < 10.0 then 'MEDIUM'
            when consumption_kwh < 50.0 then 'HIGH'
            else 'VERY_HIGH'
        end as consumption_category,
        
        -- Quality flags
        case 
            when not is_valid_consumption then 'INVALID_CONSUMPTION'
            when not is_valid_voltage then 'INVALID_VOLTAGE'
            when not is_valid_current then 'INVALID_CURRENT'
            when not is_valid_power_factor then 'INVALID_POWER_FACTOR'
            when not is_valid_frequency then 'INVALID_FREQUENCY'
            else 'VALID'
        end as quality_flag
        
    from validated_data
)

select * from enriched_data
