-- Staging model for grid operator status data
-- Cleans and standardizes grid operator data

with source_data as (
    select * from {{ source('raw', 'grid_status') }}
),

cleaned_data as (
    select
        operator_name,
        timestamp,
        total_capacity_mw,
        available_capacity_mw,
        load_factor,
        frequency_hz,
        voltage_kv,
        grid_stability_score,
        renewable_percentage,
        region,
        
        -- Add calculated fields
        total_capacity_mw - available_capacity_mw as utilized_capacity_mw,
        
        case 
            when total_capacity_mw > 0 then 
                (total_capacity_mw - available_capacity_mw) / total_capacity_mw
            else 0
        end as utilization_rate,
        
        case 
            when available_capacity_mw / total_capacity_mw < 0.1 then 'critical'
            when available_capacity_mw / total_capacity_mw < 0.2 then 'low'
            when available_capacity_mw / total_capacity_mw < 0.5 then 'medium'
            else 'high'
        end as capacity_status,
        
        case 
            when frequency_hz < 49.8 or frequency_hz > 50.2 then true
            else false
        end as has_frequency_anomaly,
        
        case 
            when voltage_kv < 380 or voltage_kv > 420 then true
            else false
        end as has_voltage_anomaly,
        
        case 
            when grid_stability_score < 0.8 then 'unstable'
            when grid_stability_score < 0.9 then 'moderate'
            else 'stable'
        end as stability_status,
        
        -- Add time-based fields
        date(timestamp) as status_date,
        hour(timestamp) as status_hour,
        dayofweek(timestamp) as day_of_week,
        month(timestamp) as status_month,
        year(timestamp) as status_year,
        
        -- Add processing metadata
        current_timestamp() as processed_at,
        'staging' as processing_stage
        
    from source_data
    where 
        -- Filter out invalid data
        total_capacity_mw is not null
        and available_capacity_mw is not null
        and load_factor is not null
        and frequency_hz is not null
        and voltage_kv is not null
        and timestamp is not null
        and operator_name is not null
),

final as (
    select
        *,
        -- Add row number for deduplication
        row_number() over (
            partition by operator_name, timestamp 
            order by processed_at desc
        ) as row_num
    from cleaned_data
)

select * from final
where row_num = 1  -- Keep only the latest record for each operator/timestamp combination
