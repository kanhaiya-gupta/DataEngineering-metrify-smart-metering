-- Staging model for smart meter readings
-- Cleans and standardizes raw smart meter data

with source_data as (
    select * from {{ source('raw', 'smart_meter_readings') }}
),

cleaned_data as (
    select
        meter_id,
        timestamp,
        consumption_kwh,
        voltage,
        current,
        power_factor,
        frequency,
        temperature,
        humidity,
        data_quality_score,
        
        -- Add calculated fields
        case 
            when power_factor is not null then power_factor
            else 1.0
        end as power_factor_clean,
        
        case 
            when frequency is not null then frequency
            else 50.0
        end as frequency_clean,
        
        -- Calculate power consumption
        voltage * current as apparent_power_va,
        voltage * current * power_factor as real_power_w,
        
        -- Add time-based fields
        date(timestamp) as reading_date,
        hour(timestamp) as reading_hour,
        dayofweek(timestamp) as day_of_week,
        month(timestamp) as reading_month,
        year(timestamp) as reading_year,
        
        -- Add data quality flags
        case 
            when data_quality_score >= 0.9 then 'high'
            when data_quality_score >= 0.7 then 'medium'
            else 'low'
        end as quality_tier,
        
        case 
            when consumption_kwh < 0 then true
            else false
        end as has_negative_consumption,
        
        case 
            when voltage < 200 or voltage > 250 then true
            else false
        end as has_voltage_anomaly,
        
        case 
            when current < 0 or current > 100 then true
            else false
        end as has_current_anomaly,
        
        -- Add processing metadata
        current_timestamp() as processed_at,
        'staging' as processing_stage
        
    from source_data
    where 
        -- Filter out obviously bad data
        consumption_kwh is not null
        and voltage is not null
        and current is not null
        and timestamp is not null
        and meter_id is not null
),

final as (
    select
        *,
        -- Add row number for deduplication
        row_number() over (
            partition by meter_id, timestamp 
            order by processed_at desc
        ) as row_num
    from cleaned_data
)

select * from final
where row_num = 1  -- Keep only the latest record for each meter/timestamp combination
