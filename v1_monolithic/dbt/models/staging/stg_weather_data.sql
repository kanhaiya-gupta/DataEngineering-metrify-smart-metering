-- Staging model for weather data
-- Cleans and standardizes weather observations

with source_data as (
    select * from {{ source('raw', 'weather_data') }}
),

cleaned_data as (
    select
        city,
        timestamp,
        temperature_celsius,
        humidity_percent,
        pressure_hpa,
        wind_speed_ms,
        wind_direction_degrees,
        cloud_cover_percent,
        visibility_km,
        uv_index,
        precipitation_mm,
        energy_demand_factor,
        
        -- Add calculated fields
        case 
            when temperature_celsius < 0 then 'freezing'
            when temperature_celsius < 10 then 'cold'
            when temperature_celsius < 20 then 'cool'
            when temperature_celsius < 30 then 'warm'
            else 'hot'
        end as temperature_category,
        
        case 
            when humidity_percent < 30 then 'dry'
            when humidity_percent < 60 then 'moderate'
            when humidity_percent < 80 then 'humid'
            else 'very_humid'
        end as humidity_category,
        
        case 
            when wind_speed_ms < 2 then 'calm'
            when wind_speed_ms < 5 then 'light'
            when wind_speed_ms < 10 then 'moderate'
            when wind_speed_ms < 20 then 'strong'
            else 'very_strong'
        end as wind_category,
        
        case 
            when cloud_cover_percent < 25 then 'clear'
            when cloud_cover_percent < 50 then 'partly_cloudy'
            when cloud_cover_percent < 75 then 'mostly_cloudy'
            else 'overcast'
        end as cloud_category,
        
        -- Calculate heat index (simplified)
        case 
            when temperature_celsius >= 27 and humidity_percent >= 40 then
                temperature_celsius + 0.5 * (humidity_percent - 40)
            else temperature_celsius
        end as heat_index,
        
        -- Calculate wind chill (simplified)
        case 
            when temperature_celsius <= 10 and wind_speed_ms > 4.8 then
                temperature_celsius - (wind_speed_ms - 4.8) * 2
            else temperature_celsius
        end as wind_chill,
        
        -- Add energy demand indicators
        case 
            when energy_demand_factor > 1.5 then 'high_demand'
            when energy_demand_factor > 1.2 then 'elevated_demand'
            when energy_demand_factor < 0.8 then 'low_demand'
            else 'normal_demand'
        end as demand_category,
        
        -- Add time-based fields
        date(timestamp) as weather_date,
        hour(timestamp) as weather_hour,
        dayofweek(timestamp) as day_of_week,
        month(timestamp) as weather_month,
        year(timestamp) as weather_year,
        
        -- Add processing metadata
        current_timestamp() as processed_at,
        'staging' as processing_stage
        
    from source_data
    where 
        -- Filter out invalid data
        temperature_celsius is not null
        and humidity_percent is not null
        and pressure_hpa is not null
        and timestamp is not null
        and city is not null
),

final as (
    select
        *,
        -- Add row number for deduplication
        row_number() over (
            partition by city, timestamp 
            order by processed_at desc
        ) as row_num
    from cleaned_data
)

select * from final
where row_num = 1  -- Keep only the latest record for each city/timestamp combination
