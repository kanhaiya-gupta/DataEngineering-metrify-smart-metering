-- Staging model for grid operator status data
-- Transforms raw grid status data into a clean, consistent format

with source_data as (
    select * from {{ source('raw', 'grid_status') }}
),

cleaned_data as (
    select
        -- Primary keys
        operator_id,
        status_id,
        
        -- Timestamps
        status_timestamp,
        created_at,
        updated_at,
        
        -- Grid status values
        grid_frequency_hz,
        grid_voltage_kv,
        grid_load_mw,
        grid_capacity_mw,
        grid_efficiency_percent,
        
        -- Status indicators
        is_grid_stable,
        is_under_frequency,
        is_over_frequency,
        is_under_voltage,
        is_over_voltage,
        is_overload,
        is_emergency,
        
        -- Quality metrics
        data_quality_score,
        is_anomaly,
        anomaly_score,
        
        -- Metadata
        status_source,
        data_quality_flags,
        
        -- Location context
        grid_region,
        substation_id,
        control_area,
        
        -- Data lineage
        ingestion_batch_id,
        source_system,
        raw_data_hash
        
    from source_data
    where status_timestamp is not null
      and operator_id is not null
),

-- Data quality validations
validated_data as (
    select
        *,
        -- Validate frequency values
        case 
            when grid_frequency_hz is not null and (grid_frequency_hz < 45 or grid_frequency_hz > 65) then false
            else true
        end as is_valid_frequency,
        
        -- Validate voltage values
        case 
            when grid_voltage_kv is not null and (grid_voltage_kv < 100 or grid_voltage_kv > 500) then false
            else true
        end as is_valid_voltage,
        
        -- Validate load values
        case 
            when grid_load_mw is not null and grid_load_mw < 0 then false
            else true
        end as is_valid_load,
        
        -- Validate capacity values
        case 
            when grid_capacity_mw is not null and grid_capacity_mw < 0 then false
            else true
        end as is_valid_capacity,
        
        -- Validate efficiency values
        case 
            when grid_efficiency_percent is not null and (grid_efficiency_percent < 0 or grid_efficiency_percent > 100) then false
            else true
        end as is_valid_efficiency,
        
        -- Calculate data quality score
        case 
            when is_valid_frequency and is_valid_voltage and is_valid_load 
                 and is_valid_capacity and is_valid_efficiency then 'EXCELLENT'
            when is_valid_frequency and is_valid_voltage and is_valid_load then 'GOOD'
            when is_valid_frequency and is_valid_voltage then 'FAIR'
            else 'POOR'
        end as calculated_quality_tier
        
    from cleaned_data
),

-- Add business logic
enriched_data as (
    select
        *,
        -- Time-based features
        extract(hour from status_timestamp) as status_hour,
        extract(dow from status_timestamp) as day_of_week,
        extract(month from status_timestamp) as status_month,
        extract(quarter from status_timestamp) as status_quarter,
        extract(year from status_timestamp) as status_year,
        
        -- Peak hour classification
        case 
            when extract(hour from status_timestamp) between {{ var('business_rules').peak_hours_start }} 
                 and {{ var('business_rules').peak_hours_end }} then true
            else false
        end as is_peak_hour_calculated,
        
        -- Weekend classification
        case 
            when extract(dow from status_timestamp) in ({{ var('business_rules').weekend_days | join(', ') }}) then true
            else false
        end as is_weekend_calculated,
        
        -- Grid utilization
        case 
            when grid_capacity_mw > 0 then (grid_load_mw / grid_capacity_mw) * 100
            else null
        end as grid_utilization_percent,
        
        -- Grid status classification
        case 
            when is_emergency then 'EMERGENCY'
            when is_overload then 'OVERLOAD'
            when is_under_frequency or is_over_frequency then 'FREQUENCY_ISSUE'
            when is_under_voltage or is_over_voltage then 'VOLTAGE_ISSUE'
            when is_grid_stable then 'STABLE'
            else 'UNKNOWN'
        end as grid_status_classification,
        
        -- Load category
        case 
            when grid_load_mw is null then 'UNKNOWN'
            when grid_load_mw = 0 then 'NO_LOAD'
            when grid_load_mw < grid_capacity_mw * 0.3 then 'LOW_LOAD'
            when grid_load_mw < grid_capacity_mw * 0.7 then 'MEDIUM_LOAD'
            when grid_load_mw < grid_capacity_mw * 0.9 then 'HIGH_LOAD'
            else 'CRITICAL_LOAD'
        end as load_category,
        
        -- Quality flags
        case 
            when not is_valid_frequency then 'INVALID_FREQUENCY'
            when not is_valid_voltage then 'INVALID_VOLTAGE'
            when not is_valid_load then 'INVALID_LOAD'
            when not is_valid_capacity then 'INVALID_CAPACITY'
            when not is_valid_efficiency then 'INVALID_EFFICIENCY'
            else 'VALID'
        end as quality_flag
        
    from validated_data
)

select * from enriched_data
