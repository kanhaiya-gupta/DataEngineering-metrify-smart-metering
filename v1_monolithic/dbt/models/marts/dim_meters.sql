-- Dimension table for smart meters
-- Contains static and slowly changing meter information

with meter_metadata as (
    select
        meter_id,
        min(reading_date) as installation_date,
        max(reading_date) as last_reading_date,
        count(distinct reading_date) as days_active,
        count(*) as total_readings,
        avg(data_quality_score) as avg_quality_score,
        avg(voltage) as avg_voltage,
        avg(current) as avg_current,
        avg(power_factor_clean) as avg_power_factor,
        avg(frequency_clean) as avg_frequency,
        sum(consumption_kwh) as total_consumption_kwh,
        avg(consumption_kwh) as avg_daily_consumption_kwh,
        max(consumption_kwh) as max_consumption_kwh,
        min(consumption_kwh) as min_consumption_kwh,
        stddev(consumption_kwh) as consumption_stddev
    from {{ ref('stg_smart_meter_readings') }}
    group by meter_id
),

meter_classification as (
    select
        *,
        
        -- Meter age classification
        case 
            when days_active < 30 then 'new'
            when days_active < 365 then 'recent'
            when days_active < 1095 then 'established'
            else 'mature'
        end as meter_age_category,
        
        -- Consumption level classification
        case 
            when avg_daily_consumption_kwh > 50 then 'high_consumption'
            when avg_daily_consumption_kwh > 20 then 'medium_consumption'
            when avg_daily_consumption_kwh > 5 then 'low_consumption'
            else 'very_low_consumption'
        end as consumption_level,
        
        -- Quality classification
        case 
            when avg_quality_score >= 0.9 then 'high_quality'
            when avg_quality_score >= 0.7 then 'medium_quality'
            else 'low_quality'
        end as quality_classification,
        
        -- Reliability classification
        case 
            when days_active > 0 then total_readings / days_active
            else 0
        end as readings_per_day,
        
        case 
            when readings_per_day >= 24 then 'highly_reliable'
            when readings_per_day >= 12 then 'reliable'
            when readings_per_day >= 6 then 'moderately_reliable'
            else 'unreliable'
        end as reliability_classification,
        
        -- Voltage stability
        case 
            when avg_voltage >= 220 and avg_voltage <= 240 then 'stable'
            when avg_voltage >= 210 and avg_voltage <= 250 then 'moderate'
            else 'unstable'
        end as voltage_stability,
        
        -- Power factor efficiency
        case 
            when avg_power_factor >= 0.95 then 'excellent'
            when avg_power_factor >= 0.9 then 'good'
            when avg_power_factor >= 0.8 then 'fair'
            else 'poor'
        end as power_factor_efficiency,
        
        -- Consumption variability
        case 
            when consumption_stddev / avg_daily_consumption_kwh < 0.2 then 'stable'
            when consumption_stddev / avg_daily_consumption_kwh < 0.5 then 'variable'
            else 'highly_variable'
        end as consumption_variability,
        
        -- Anomaly detection
        case 
            when max_consumption_kwh > avg_daily_consumption_kwh * 3 then true
            when min_consumption_kwh < 0 then true
            when avg_voltage < 200 or avg_voltage > 250 then true
            when avg_current < 0 or avg_current > 100 then true
            else false
        end as has_historical_anomalies,
        
        -- Performance score (0-100)
        (
            case when avg_quality_score >= 0.9 then 25 else avg_quality_score * 25 end +
            case when readings_per_day >= 24 then 25 else readings_per_day * 25 / 24 end +
            case when avg_power_factor >= 0.95 then 25 else avg_power_factor * 25 / 0.95 end +
            case when avg_voltage >= 220 and avg_voltage <= 240 then 25 else 0 end
        ) as performance_score
        
    from meter_metadata
),

final as (
    select
        *,
        
        -- Overall meter status
        case 
            when performance_score >= 90 and not has_historical_anomalies then 'excellent'
            when performance_score >= 75 and not has_historical_anomalies then 'good'
            when performance_score >= 60 then 'fair'
            else 'needs_attention'
        end as overall_status,
        
        -- Maintenance priority
        case 
            when has_historical_anomalies and performance_score < 60 then 'high'
            when performance_score < 75 or has_historical_anomalies then 'medium'
            else 'low'
        end as maintenance_priority,
        
        -- Add processing metadata
        current_timestamp() as processed_at,
        'marts' as processing_stage
        
    from meter_classification
)

select * from final
