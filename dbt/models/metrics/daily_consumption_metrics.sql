-- Daily consumption metrics
-- Aggregates hourly smart meter data into daily metrics for reporting

with daily_consumption as (
    select
        meter_id,
        date,
        day_name,
        day_type,
        month_name,
        quarter_of_year,
        year,
        season,
        is_peak_season,
        
        -- Consumption metrics
        sum(total_consumption_kwh) as daily_consumption_kwh,
        avg(avg_consumption_kwh) as avg_hourly_consumption_kwh,
        min(min_consumption_kwh) as min_hourly_consumption_kwh,
        max(max_consumption_kwh) as max_hourly_consumption_kwh,
        stddev(avg_consumption_kwh) as stddev_hourly_consumption_kwh,
        
        -- Reading metrics
        sum(reading_count) as total_readings,
        avg(reading_count) as avg_readings_per_hour,
        min(reading_count) as min_readings_per_hour,
        max(reading_count) as max_readings_per_hour,
        
        -- Quality metrics
        avg(avg_quality_score) as avg_quality_score,
        sum(excellent_quality_count) as total_excellent_readings,
        sum(good_quality_count) as total_good_readings,
        sum(fair_quality_count) as total_fair_readings,
        sum(poor_quality_count) as total_poor_readings,
        sum(anomaly_count) as total_anomalies,
        avg(avg_anomaly_score) as avg_anomaly_score,
        
        -- Electrical metrics
        avg(avg_voltage_v) as avg_voltage_v,
        avg(avg_current_a) as avg_current_a,
        avg(avg_power_factor) as avg_power_factor,
        avg(avg_frequency_hz) as avg_frequency_hz,
        
        -- Business context
        max(tariff_type) as tariff_type,
        max(time_of_use_period) as time_of_use_period,
        max(is_peak_hour) as had_peak_hours,
        max(is_off_peak_hour) as had_off_peak_hours,
        max(is_weekend) as is_weekend,
        
        -- Location context
        max(meter_latitude) as meter_latitude,
        max(meter_longitude) as meter_longitude,
        max(meter_address) as meter_address
        
    from {{ ref('fct_smart_meter_analytics') }}
    group by 
        meter_id, 
        date, 
        day_name, 
        day_type, 
        month_name, 
        quarter_of_year, 
        year, 
        season, 
        is_peak_season
),

-- Add calculated metrics
enriched_metrics as (
    select
        *,
        -- Quality percentage calculations
        case 
            when total_readings > 0 then (total_excellent_readings::float / total_readings) * 100
            else 0
        end as excellent_quality_percent,
        
        case 
            when total_readings > 0 then (total_good_readings::float / total_readings) * 100
            else 0
        end as good_quality_percent,
        
        case 
            when total_readings > 0 then (total_fair_readings::float / total_readings) * 100
            else 0
        end as fair_quality_percent,
        
        case 
            when total_readings > 0 then (total_poor_readings::float / total_readings) * 100
            else 0
        end as poor_quality_percent,
        
        case 
            when total_readings > 0 then (total_anomalies::float / total_readings) * 100
            else 0
        end as anomaly_percent,
        
        -- Consumption categories
        case 
            when daily_consumption_kwh = 0 then 'ZERO_CONSUMPTION'
            when daily_consumption_kwh < 1 then 'VERY_LOW'
            when daily_consumption_kwh < 10 then 'LOW'
            when daily_consumption_kwh < 50 then 'MEDIUM'
            when daily_consumption_kwh < 100 then 'HIGH'
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
        
        -- Consumption efficiency
        case 
            when avg_power_factor > 0 then daily_consumption_kwh / avg_power_factor
            else daily_consumption_kwh
        end as efficiency_adjusted_consumption_kwh,
        
        -- Peak vs off-peak consumption
        case 
            when had_peak_hours and had_off_peak_hours then 'MIXED'
            when had_peak_hours then 'PEAK_ONLY'
            when had_off_peak_hours then 'OFF_PEAK_ONLY'
            else 'UNKNOWN'
        end as consumption_pattern
        
    from daily_consumption
),

-- Add time-based features
final_metrics as (
    select
        *,
        -- Week of year
        extract(week from date) as week_of_year,
        
        -- Day of year
        extract(doy from date) as day_of_year,
        
        -- Business day classification
        case 
            when day_type = 'WEEKDAY' then 'BUSINESS_DAY'
            else 'NON_BUSINESS_DAY'
        end as business_day_type,
        
        -- Month classification
        case 
            when month_name in ('December', 'January', 'February') then 'WINTER'
            when month_name in ('March', 'April', 'May') then 'SPRING'
            when month_name in ('June', 'July', 'August') then 'SUMMER'
            when month_name in ('September', 'October', 'November') then 'FALL'
        end as season_calculated,
        
        -- Peak season classification
        case 
            when month_name in ('June', 'July', 'August', 'December', 'January') then true
            else false
        end as is_peak_season_calculated,
        
        -- Data completeness
        case 
            when total_readings >= 24 then 'COMPLETE'
            when total_readings >= 12 then 'PARTIAL'
            else 'INCOMPLETE'
        end as data_completeness,
        
        -- Last updated
        current_timestamp as last_updated
        
    from enriched_metrics
)

select * from final_metrics
