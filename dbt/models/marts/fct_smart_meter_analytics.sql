-- Fact table for smart meter analytics
-- Aggregates smart meter readings into hourly analytics for reporting and analysis

with hourly_readings as (
    select
        meter_id,
        date_trunc('hour', reading_timestamp) as date_hour,
        
        -- Aggregated consumption metrics
        count(*) as reading_count,
        sum(consumption_kwh) as total_consumption_kwh,
        avg(consumption_kwh) as avg_consumption_kwh,
        min(consumption_kwh) as min_consumption_kwh,
        max(consumption_kwh) as max_consumption_kwh,
        stddev(consumption_kwh) as stddev_consumption_kwh,
        
        -- Electrical metrics
        avg(voltage_v) as avg_voltage_v,
        avg(current_a) as avg_current_a,
        avg(power_factor) as avg_power_factor,
        avg(frequency_hz) as avg_frequency_hz,
        
        -- Quality metrics
        avg(quality_score) as avg_quality_score,
        count(case when quality_tier = 'EXCELLENT' then 1 end) as excellent_quality_count,
        count(case when quality_tier = 'GOOD' then 1 end) as good_quality_count,
        count(case when quality_tier = 'FAIR' then 1 end) as fair_quality_count,
        count(case when quality_tier = 'POOR' then 1 end) as poor_quality_count,
        count(case when is_anomaly then 1 end) as anomaly_count,
        avg(anomaly_score) as avg_anomaly_score,
        
        -- Business context
        max(is_peak_hour_calculated) as is_peak_hour,
        max(is_off_peak_hour) as is_off_peak_hour,
        max(is_weekend_calculated) as is_weekend,
        max(tariff_type) as tariff_type,
        max(time_of_use_period) as time_of_use_period,
        
        -- Location context
        max(meter_location_lat) as meter_latitude,
        max(meter_location_lon) as meter_longitude,
        max(meter_address) as meter_address,
        
        -- Data lineage
        max(ingestion_batch_id) as latest_batch_id,
        max(source_system) as source_system
        
    from {{ ref('stg_smart_meter_readings') }}
    where reading_timestamp >= current_date - interval '{{ var("data_retention").staging_data_days }} days'
    group by meter_id, date_trunc('hour', reading_timestamp)
),

-- Add calculated metrics
enriched_metrics as (
    select
        *,
        -- Quality percentage calculations
        case 
            when reading_count > 0 then (excellent_quality_count::float / reading_count) * 100
            else 0
        end as excellent_quality_percent,
        
        case 
            when reading_count > 0 then (good_quality_count::float / reading_count) * 100
            else 0
        end as good_quality_percent,
        
        case 
            when reading_count > 0 then (fair_quality_count::float / reading_count) * 100
            else 0
        end as fair_quality_percent,
        
        case 
            when reading_count > 0 then (poor_quality_count::float / reading_count) * 100
            else 0
        end as poor_quality_percent,
        
        case 
            when reading_count > 0 then (anomaly_count::float / reading_count) * 100
            else 0
        end as anomaly_percent,
        
        -- Consumption categories
        case 
            when total_consumption_kwh = 0 then 'ZERO_CONSUMPTION'
            when total_consumption_kwh < 0.1 then 'VERY_LOW'
            when total_consumption_kwh < 1.0 then 'LOW'
            when total_consumption_kwh < 10.0 then 'MEDIUM'
            when total_consumption_kwh < 50.0 then 'HIGH'
            else 'VERY_HIGH'
        end as consumption_category,
        
        -- Quality tier classification
        case 
            when excellent_quality_percent >= 80 then 'EXCELLENT'
            when good_quality_percent >= 70 then 'GOOD'
            when fair_quality_percent >= 50 then 'FAIR'
            else 'POOR'
        end as overall_quality_tier,
        
        -- Anomaly classification
        case 
            when anomaly_percent >= 20 then 'HIGH_ANOMALY'
            when anomaly_percent >= 10 then 'MEDIUM_ANOMALY'
            when anomaly_percent > 0 then 'LOW_ANOMALY'
            else 'NO_ANOMALY'
        end as anomaly_classification,
        
        -- Time period classification
        case 
            when is_peak_hour and not is_weekend then 'PEAK_WEEKDAY'
            when is_peak_hour and is_weekend then 'PEAK_WEEKEND'
            when is_off_peak_hour and not is_weekend then 'OFF_PEAK_WEEKDAY'
            when is_off_peak_hour and is_weekend then 'OFF_PEAK_WEEKEND'
            else 'OTHER'
        end as time_period_classification
        
    from hourly_readings
),

-- Add time-based features
final_metrics as (
    select
        *,
        -- Time features
        extract(hour from date_hour) as hour_of_day,
        extract(dow from date_hour) as day_of_week,
        extract(month from date_hour) as month_of_year,
        extract(quarter from date_hour) as quarter_of_year,
        extract(year from date_hour) as year,
        
        -- Date features
        date_hour::date as date,
        to_char(date_hour, 'Day') as day_name,
        to_char(date_hour, 'Month') as month_name,
        
        -- Business day classification
        case 
            when extract(dow from date_hour) in (1, 2, 3, 4, 5) then 'WEEKDAY'
            else 'WEEKEND'
        end as day_type,
        
        -- Season classification
        case 
            when extract(month from date_hour) in (12, 1, 2) then 'WINTER'
            when extract(month from date_hour) in (3, 4, 5) then 'SPRING'
            when extract(month from date_hour) in (6, 7, 8) then 'SUMMER'
            when extract(month from date_hour) in (9, 10, 11) then 'FALL'
        end as season,
        
        -- Peak season classification
        case 
            when extract(month from date_hour) in (6, 7, 8, 12, 1) then true
            else false
        end as is_peak_season
        
    from enriched_metrics
)

select * from final_metrics
