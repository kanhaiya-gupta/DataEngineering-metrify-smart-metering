-- Energy Efficiency Metrics
-- Provides key performance indicators for energy efficiency and grid optimization

with efficiency_metrics as (
    select
        meter_id,
        date_hour,
        date,
        hour_of_day,
        day_of_week,
        month_of_year,
        quarter_of_year,
        year,
        season,
        is_peak_hour,
        is_weekend,
        tariff_type,
        time_of_use_period,
        
        -- Consumption metrics
        total_consumption_kwh,
        avg_consumption_kwh,
        min_consumption_kwh,
        max_consumption_kwh,
        stddev_consumption_kwh,
        
        -- Electrical efficiency metrics
        avg_power_factor,
        avg_voltage_v,
        avg_current_a,
        avg_frequency_hz,
        
        -- Quality metrics
        avg_quality_score,
        anomaly_count,
        reading_count,
        excellent_quality_count,
        good_quality_count,
        fair_quality_count,
        poor_quality_count
        
    from {{ ref('fct_smart_meter_analytics') }}
),

-- Calculate efficiency ratios and indicators
efficiency_calculations as (
    select
        *,
        
        -- Load factor (average load / peak load)
        case 
            when max_consumption_kwh > 0 then avg_consumption_kwh / max_consumption_kwh
            else null
        end as load_factor,
        
        -- Power factor efficiency (closer to 1.0 is better)
        case 
            when avg_power_factor >= 0.95 then 'EXCELLENT'
            when avg_power_factor >= 0.90 then 'GOOD'
            when avg_power_factor >= 0.85 then 'FAIR'
            when avg_power_factor >= 0.80 then 'POOR'
            else 'VERY_POOR'
        end as power_factor_efficiency_tier,
        
        -- Voltage efficiency (within normal range)
        case 
            when avg_voltage_v between 220 and 240 then 'OPTIMAL'
            when avg_voltage_v between 210 and 250 then 'ACCEPTABLE'
            when avg_voltage_v between 200 and 260 then 'MARGINAL'
            else 'POOR'
        end as voltage_efficiency_tier,
        
        -- Frequency stability (within normal range)
        case 
            when avg_frequency_hz between 49.8 and 50.2 then 'STABLE'
            when avg_frequency_hz between 49.5 and 50.5 then 'ACCEPTABLE'
            when avg_frequency_hz between 49.0 and 51.0 then 'UNSTABLE'
            else 'CRITICAL'
        end as frequency_stability_tier,
        
        -- Quality efficiency
        case 
            when avg_quality_score >= 0.95 then 'EXCELLENT'
            when avg_quality_score >= 0.90 then 'GOOD'
            when avg_quality_score >= 0.80 then 'FAIR'
            when avg_quality_score >= 0.70 then 'POOR'
            else 'VERY_POOR'
        end as quality_efficiency_tier,
        
        -- Anomaly rate
        case 
            when reading_count > 0 then (anomaly_count::float / reading_count) * 100
            else 0
        end as anomaly_rate_percent,
        
        -- Quality distribution
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
        end as poor_quality_percent
        
    from efficiency_metrics
),

-- Add efficiency scores and classifications
final_metrics as (
    select
        *,
        
        -- Overall efficiency score (0-100)
        (
            case when power_factor_efficiency_tier = 'EXCELLENT' then 25
                 when power_factor_efficiency_tier = 'GOOD' then 20
                 when power_factor_efficiency_tier = 'FAIR' then 15
                 when power_factor_efficiency_tier = 'POOR' then 10
                 else 5 end +
            case when voltage_efficiency_tier = 'OPTIMAL' then 25
                 when voltage_efficiency_tier = 'ACCEPTABLE' then 20
                 when voltage_efficiency_tier = 'MARGINAL' then 15
                 else 5 end +
            case when frequency_stability_tier = 'STABLE' then 25
                 when frequency_stability_tier = 'ACCEPTABLE' then 20
                 when frequency_stability_tier = 'UNSTABLE' then 15
                 else 5 end +
            case when quality_efficiency_tier = 'EXCELLENT' then 25
                 when quality_efficiency_tier = 'GOOD' then 20
                 when quality_efficiency_tier = 'FAIR' then 15
                 when quality_efficiency_tier = 'POOR' then 10
                 else 5 end
        ) as overall_efficiency_score,
        
        -- Efficiency classification
        case 
            when (
                case when power_factor_efficiency_tier = 'EXCELLENT' then 25
                     when power_factor_efficiency_tier = 'GOOD' then 20
                     when power_factor_efficiency_tier = 'FAIR' then 15
                     when power_factor_efficiency_tier = 'POOR' then 10
                     else 5 end +
                case when voltage_efficiency_tier = 'OPTIMAL' then 25
                     when voltage_efficiency_tier = 'ACCEPTABLE' then 20
                     when voltage_efficiency_tier = 'MARGINAL' then 15
                     else 5 end +
                case when frequency_stability_tier = 'STABLE' then 25
                     when frequency_stability_tier = 'ACCEPTABLE' then 20
                     when frequency_stability_tier = 'UNSTABLE' then 15
                     else 5 end +
                case when quality_efficiency_tier = 'EXCELLENT' then 25
                     when quality_efficiency_tier = 'GOOD' then 20
                     when quality_efficiency_tier = 'FAIR' then 15
                     when quality_efficiency_tier = 'POOR' then 10
                     else 5 end
            ) >= 90 then 'EXCELLENT'
            when (
                case when power_factor_efficiency_tier = 'EXCELLENT' then 25
                     when power_factor_efficiency_tier = 'GOOD' then 20
                     when power_factor_efficiency_tier = 'FAIR' then 15
                     when power_factor_efficiency_tier = 'POOR' then 10
                     else 5 end +
                case when voltage_efficiency_tier = 'OPTIMAL' then 25
                     when voltage_efficiency_tier = 'ACCEPTABLE' then 20
                     when voltage_efficiency_tier = 'MARGINAL' then 15
                     else 5 end +
                case when frequency_stability_tier = 'STABLE' then 25
                     when frequency_stability_tier = 'ACCEPTABLE' then 20
                     when frequency_stability_tier = 'UNSTABLE' then 15
                     else 5 end +
                case when quality_efficiency_tier = 'EXCELLENT' then 25
                     when quality_efficiency_tier = 'GOOD' then 20
                     when quality_efficiency_tier = 'FAIR' then 15
                     when quality_efficiency_tier = 'POOR' then 10
                     else 5 end
            ) >= 75 then 'GOOD'
            when (
                case when power_factor_efficiency_tier = 'EXCELLENT' then 25
                     when power_factor_efficiency_tier = 'GOOD' then 20
                     when power_factor_efficiency_tier = 'FAIR' then 15
                     when power_factor_efficiency_tier = 'POOR' then 10
                     else 5 end +
                case when voltage_efficiency_tier = 'OPTIMAL' then 25
                     when voltage_efficiency_tier = 'ACCEPTABLE' then 20
                     when voltage_efficiency_tier = 'MARGINAL' then 15
                     else 5 end +
                case when frequency_stability_tier = 'STABLE' then 25
                     when frequency_stability_tier = 'ACCEPTABLE' then 20
                     when frequency_stability_tier = 'UNSTABLE' then 15
                     else 5 end +
                case when quality_efficiency_tier = 'EXCELLENT' then 25
                     when quality_efficiency_tier = 'GOOD' then 20
                     when quality_efficiency_tier = 'FAIR' then 15
                     when quality_efficiency_tier = 'POOR' then 10
                     else 5 end
            ) >= 60 then 'FAIR'
            else 'POOR'
        end as overall_efficiency_classification,
        
        -- Energy waste indicators
        case 
            when avg_power_factor < 0.85 then true
            when avg_voltage_v < 210 or avg_voltage_v > 250 then true
            when avg_frequency_hz < 49.5 or avg_frequency_hz > 50.5 then true
            when anomaly_rate_percent > 10 then true
            else false
        end as has_energy_waste,
        
        -- Optimization opportunities
        case 
            when avg_power_factor < 0.90 and avg_power_factor >= 0.85 then 'POWER_FACTOR_CORRECTION'
            when avg_voltage_v < 220 or avg_voltage_v > 240 then 'VOLTAGE_REGULATION'
            when avg_frequency_hz < 49.8 or avg_frequency_hz > 50.2 then 'FREQUENCY_STABILIZATION'
            when anomaly_rate_percent > 5 then 'DATA_QUALITY_IMPROVEMENT'
            when load_factor < 0.5 then 'LOAD_BALANCING'
            else 'NO_OPTIMIZATION_NEEDED'
        end as optimization_opportunity,
        
        -- Last updated
        current_timestamp as last_updated
        
    from efficiency_calculations
)

select * from final_metrics
