-- Grid Performance Metrics
-- Provides key performance indicators for grid stability and performance

with grid_metrics as (
    select
        operator_id,
        region_code,
        operator_name,
        last_status_date,
        date(last_status_date) as status_date,
        extract(hour from last_status_date) as status_hour,
        extract(dayofweek from last_status_date) as day_of_week,
        extract(month from last_status_date) as month_of_year,
        extract(quarter from last_status_date) as quarter_of_year,
        extract(year from last_status_date) as year,
        
        -- Grid performance metrics
        total_status_updates,
        avg_frequency_hz,
        avg_voltage_kv,
        avg_generation_mw,
        avg_consumption_mw,
        avg_stability_score,
        
        -- Stability metrics
        stability_percent,
        warning_percent,
        critical_percent,
        anomaly_percent,
        
        -- Classifications
        stability_classification,
        risk_classification,
        anomaly_classification,
        operator_status,
        
        -- Utilization metrics
        avg_utilization_percent,
        days_in_service,
        updates_per_day
        
    from {{ ref('dim_grid_operators') }}
),

-- Calculate performance indicators
performance_calculations as (
    select
        *,
        
        -- Frequency stability score (0-100)
        case 
            when avg_frequency_hz between 49.8 and 50.2 then 100
            when avg_frequency_hz between 49.5 and 50.5 then 80
            when avg_frequency_hz between 49.0 and 51.0 then 60
            when avg_frequency_hz between 48.5 and 51.5 then 40
            else 20
        end as frequency_stability_score,
        
        -- Voltage stability score (0-100)
        case 
            when avg_voltage_kv between 220 and 240 then 100
            when avg_voltage_kv between 210 and 250 then 80
            when avg_voltage_kv between 200 and 260 then 60
            when avg_voltage_kv between 190 and 270 then 40
            else 20
        end as voltage_stability_score,
        
        -- Grid utilization efficiency
        case 
            when avg_utilization_percent between 70 and 90 then 'OPTIMAL'
            when avg_utilization_percent between 60 and 95 then 'GOOD'
            when avg_utilization_percent between 50 and 100 then 'ACCEPTABLE'
            when avg_utilization_percent < 50 then 'UNDERUTILIZED'
            else 'OVERUTILIZED'
        end as utilization_efficiency_tier,
        
        -- Grid reliability score
        case 
            when stability_percent >= 95 and critical_percent = 0 then 'EXCELLENT'
            when stability_percent >= 90 and critical_percent <= 1 then 'GOOD'
            when stability_percent >= 80 and critical_percent <= 5 then 'FAIR'
            when stability_percent >= 70 and critical_percent <= 10 then 'POOR'
            else 'CRITICAL'
        end as reliability_tier,
        
        -- Performance trend indicators
        case 
            when stability_percent >= 95 and anomaly_percent <= 5 then 'IMPROVING'
            when stability_percent >= 90 and anomaly_percent <= 10 then 'STABLE'
            when stability_percent >= 80 and anomaly_percent <= 20 then 'DECLINING'
            else 'CRITICAL'
        end as performance_trend,
        
        -- Grid stress indicators
        case 
            when critical_percent > 5 then true
            when warning_percent > 20 then true
            when avg_utilization_percent > 95 then true
            when anomaly_percent > 25 then true
            else false
        end as is_grid_stressed,
        
        -- Maintenance urgency
        case 
            when critical_percent > 10 or stability_percent < 70 then 'URGENT'
            when critical_percent > 5 or stability_percent < 80 then 'HIGH'
            when warning_percent > 15 or stability_percent < 90 then 'MEDIUM'
            when warning_percent > 10 or anomaly_percent > 15 then 'LOW'
            else 'NONE'
        end as maintenance_urgency,
        
        -- Grid capacity status
        case 
            when avg_utilization_percent > 95 then 'CRITICAL_CAPACITY'
            when avg_utilization_percent > 85 then 'HIGH_CAPACITY'
            when avg_utilization_percent > 70 then 'NORMAL_CAPACITY'
            when avg_utilization_percent > 50 then 'LOW_CAPACITY'
            else 'UNDERUTILIZED'
        end as capacity_status
        
    from grid_metrics
),

-- Add overall performance score and classifications
final_metrics as (
    select
        *,
        
        -- Overall grid performance score (0-100)
        (
            (frequency_stability_score * 0.3) +
            (voltage_stability_score * 0.3) +
            (stability_percent * 0.4)
        ) as overall_performance_score,
        
        -- Performance classification
        case 
            when (
                (frequency_stability_score * 0.3) +
                (voltage_stability_score * 0.3) +
                (stability_percent * 0.4)
            ) >= 90 then 'EXCELLENT'
            when (
                (frequency_stability_score * 0.3) +
                (voltage_stability_score * 0.3) +
                (stability_percent * 0.4)
            ) >= 80 then 'GOOD'
            when (
                (frequency_stability_score * 0.3) +
                (voltage_stability_score * 0.3) +
                (stability_percent * 0.4)
            ) >= 70 then 'FAIR'
            when (
                (frequency_stability_score * 0.3) +
                (voltage_stability_score * 0.3) +
                (stability_percent * 0.4)
            ) >= 60 then 'POOR'
            else 'CRITICAL'
        end as overall_performance_classification,
        
        -- Grid health status
        case 
            when overall_performance_score >= 90 and not is_grid_stressed then 'HEALTHY'
            when overall_performance_score >= 80 and not is_grid_stressed then 'GOOD'
            when overall_performance_score >= 70 or is_grid_stressed then 'WARNING'
            when overall_performance_score >= 60 then 'CRITICAL'
            else 'EMERGENCY'
        end as grid_health_status,
        
        -- Optimization recommendations
        case 
            when frequency_stability_score < 80 then 'FREQUENCY_STABILIZATION'
            when voltage_stability_score < 80 then 'VOLTAGE_REGULATION'
            when avg_utilization_percent > 90 then 'CAPACITY_EXPANSION'
            when anomaly_percent > 15 then 'DATA_QUALITY_IMPROVEMENT'
            when stability_percent < 85 then 'GRID_STABILIZATION'
            when critical_percent > 5 then 'EMERGENCY_RESPONSE'
            else 'MAINTENANCE_SCHEDULING'
        end as optimization_recommendation,
        
        -- Risk assessment
        case 
            when critical_percent > 10 or overall_performance_score < 60 then 'HIGH_RISK'
            when critical_percent > 5 or overall_performance_score < 70 then 'MEDIUM_RISK'
            when warning_percent > 15 or overall_performance_score < 80 then 'LOW_RISK'
            else 'MINIMAL_RISK'
        end as risk_assessment,
        
        -- Last updated
        current_timestamp as last_updated
        
    from performance_calculations
)

select * from final_metrics
