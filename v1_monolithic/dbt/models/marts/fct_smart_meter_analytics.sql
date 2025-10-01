-- Fact table for smart meter analytics
-- Combines smart meter readings with grid status and weather data

with smart_meter_readings as (
    select * from {{ ref('stg_smart_meter_readings') }}
),

grid_status as (
    select * from {{ ref('stg_grid_status') }}
),

weather_data as (
    select * from {{ ref('stg_weather_data') }}
),

-- Create time-based aggregations
hourly_consumption as (
    select
        meter_id,
        reading_date,
        reading_hour,
        sum(consumption_kwh) as total_consumption_kwh,
        avg(consumption_kwh) as avg_consumption_kwh,
        max(consumption_kwh) as max_consumption_kwh,
        min(consumption_kwh) as min_consumption_kwh,
        count(*) as reading_count,
        avg(data_quality_score) as avg_quality_score,
        avg(voltage) as avg_voltage,
        avg(current) as avg_current,
        avg(power_factor_clean) as avg_power_factor,
        avg(frequency_clean) as avg_frequency
    from smart_meter_readings
    group by meter_id, reading_date, reading_hour
),

-- Join with grid status (closest timestamp)
grid_status_with_readings as (
    select
        smr.*,
        gs.operator_name,
        gs.total_capacity_mw,
        gs.available_capacity_mw,
        gs.load_factor,
        gs.frequency_hz as grid_frequency_hz,
        gs.voltage_kv as grid_voltage_kv,
        gs.grid_stability_score,
        gs.renewable_percentage,
        gs.region,
        gs.capacity_status,
        gs.stability_status,
        gs.utilization_rate
    from hourly_consumption smr
    left join grid_status gs
        on smr.reading_date = gs.status_date
        and smr.reading_hour = gs.status_hour
),

-- Join with weather data (closest timestamp)
final_with_weather as (
    select
        gsr.*,
        wd.city,
        wd.temperature_celsius,
        wd.humidity_percent,
        wd.pressure_hpa,
        wd.wind_speed_ms,
        wd.wind_direction_degrees,
        wd.cloud_cover_percent,
        wd.visibility_km,
        wd.energy_demand_factor,
        wd.temperature_category,
        wd.humidity_category,
        wd.wind_category,
        wd.cloud_category,
        wd.heat_index,
        wd.wind_chill,
        wd.demand_category
    from grid_status_with_readings gsr
    left join weather_data wd
        on gsr.reading_date = wd.weather_date
        and gsr.reading_hour = wd.weather_hour
),

-- Add calculated metrics
final as (
    select
        *,
        
        -- Energy efficiency metrics
        case 
            when avg_power_factor > 0.95 then 'excellent'
            when avg_power_factor > 0.9 then 'good'
            when avg_power_factor > 0.8 then 'fair'
            else 'poor'
        end as power_factor_rating,
        
        -- Consumption patterns
        case 
            when total_consumption_kwh > avg(total_consumption_kwh) over (partition by meter_id) * 1.5 then 'high'
            when total_consumption_kwh < avg(total_consumption_kwh) over (partition by meter_id) * 0.5 then 'low'
            else 'normal'
        end as consumption_pattern,
        
        -- Grid impact assessment
        case 
            when stability_status = 'unstable' and total_consumption_kwh > avg(total_consumption_kwh) over (partition by reading_date, reading_hour) * 1.2 then 'high_impact'
            when stability_status = 'moderate' and total_consumption_kwh > avg(total_consumption_kwh) over (partition by reading_date, reading_hour) * 1.5 then 'moderate_impact'
            else 'low_impact'
        end as grid_impact,
        
        -- Weather correlation
        case 
            when demand_category = 'high_demand' and total_consumption_kwh > avg(total_consumption_kwh) over (partition by reading_date, reading_hour) * 1.2 then 'weather_correlated'
            when demand_category = 'low_demand' and total_consumption_kwh < avg(total_consumption_kwh) over (partition by reading_date, reading_hour) * 0.8 then 'weather_correlated'
            else 'not_correlated'
        end as weather_correlation,
        
        -- Data quality assessment
        case 
            when avg_quality_score >= 0.9 then 'high_quality'
            when avg_quality_score >= 0.7 then 'medium_quality'
            else 'low_quality'
        end as data_quality_tier,
        
        -- Anomaly detection
        case 
            when total_consumption_kwh > avg(total_consumption_kwh) over (partition by meter_id) * 2 then true
            when total_consumption_kwh < 0 then true
            when avg_voltage < 200 or avg_voltage > 250 then true
            when avg_current < 0 or avg_current > 100 then true
            else false
        end as has_anomaly,
        
        -- Time-based features
        case 
            when reading_hour >= 6 and reading_hour <= 9 then 'morning_peak'
            when reading_hour >= 17 and reading_hour <= 20 then 'evening_peak'
            when reading_hour >= 22 or reading_hour <= 5 then 'night'
            else 'off_peak'
        end as time_period,
        
        case 
            when day_of_week in (1, 7) then 'weekend'
            else 'weekday'
        end as day_type,
        
        -- Add processing metadata
        current_timestamp() as processed_at,
        'marts' as processing_stage
        
    from final_with_weather
)

select * from final
