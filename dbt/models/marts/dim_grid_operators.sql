-- Dimension table for grid operators
-- Provides master data for all grid operators in the system

with operator_data as (
    select
        operator_id,
        region_code,
        operator_name,
        status_source,
        source_system,
        min(status_timestamp) as first_status_date,
        max(status_timestamp) as last_status_date,
        count(*) as total_status_updates,
        avg(grid_frequency_hz) as avg_frequency_hz,
        avg(grid_voltage_kv) as avg_voltage_kv,
        avg(power_generation_mw) as avg_generation_mw,
        avg(power_consumption_mw) as avg_consumption_mw,
        avg(grid_stability_score) as avg_stability_score,
        count(case when is_stable then 1 end) as stable_status_count,
        count(case when alert_level = 'WARNING' then 1 end) as warning_count,
        count(case when alert_level = 'CRITICAL' then 1 end) as critical_count,
        count(case when is_anomaly then 1 end) as total_anomalies,
        avg(anomaly_score) as avg_anomaly_score
        
    from {{ ref('stg_grid_status') }}
    group by 
        operator_id, 
        region_code, 
        operator_name, 
        status_source, 
        source_system
),

-- Add calculated fields
enriched_operators as (
    select
        *,
        -- Stability metrics
        case 
            when total_status_updates > 0 then (stable_status_count::float / total_status_updates) * 100
            else 0
        end as stability_percent,
        
        case 
            when total_status_updates > 0 then (warning_count::float / total_status_updates) * 100
            else 0
        end as warning_percent,
        
        case 
            when total_status_updates > 0 then (critical_count::float / total_status_updates) * 100
            else 0
        end as critical_percent,
        
        case 
            when total_status_updates > 0 then (total_anomalies::float / total_status_updates) * 100
            else 0
        end as anomaly_percent,
        
        -- Grid utilization
        case 
            when avg_consumption_mw > 0 and avg_generation_mw > 0 then (avg_consumption_mw / avg_generation_mw) * 100
            else null
        end as avg_utilization_percent,
        
        -- Operator age
        current_date - first_status_date::date as days_in_service,
        
        -- Update frequency
        case 
            when days_in_service > 0 then total_status_updates::float / days_in_service
            else 0
        end as updates_per_day,
        
        -- Stability classification
        case 
            when stability_percent >= 95 then 'VERY_STABLE'
            when stability_percent >= 90 then 'STABLE'
            when stability_percent >= 80 then 'MODERATELY_STABLE'
            when stability_percent >= 70 then 'UNSTABLE'
            else 'VERY_UNSTABLE'
        end as stability_classification,
        
        -- Risk classification
        case 
            when critical_percent >= 5 then 'HIGH_RISK'
            when critical_percent >= 1 or warning_percent >= 10 then 'MEDIUM_RISK'
            when warning_percent >= 5 or anomaly_percent >= 20 then 'LOW_RISK'
            else 'LOW_RISK'
        end as risk_classification,
        
        -- Anomaly classification
        case 
            when anomaly_percent >= 20 then 'HIGH_ANOMALY'
            when anomaly_percent >= 10 then 'MEDIUM_ANOMALY'
            when anomaly_percent > 0 then 'LOW_ANOMALY'
            else 'NO_ANOMALY'
        end as anomaly_classification,
        
        -- Operator status
        case 
            when last_status_date < current_date - interval '1 day' then 'INACTIVE'
            when critical_percent >= 10 then 'CRITICAL'
            when warning_percent >= 20 then 'WARNING'
            when stability_percent < 70 then 'UNSTABLE'
            else 'ACTIVE'
        end as operator_status
        
    from operator_data
),

-- Add additional metadata
final_operators as (
    select
        *,
        -- Operator ID components
        split_part(operator_id, '-', 1) as operator_type,
        split_part(operator_id, '-', 2) as operator_region,
        split_part(operator_id, '-', 3) as operator_sequence,
        
        -- Region components
        split_part(region_code, '-', 1) as region_type,
        split_part(region_code, '-', 2) as region_code_detail,
        
        -- Data completeness
        case 
            when region_code is not null and operator_name is not null then 'COMPLETE'
            when region_code is not null or operator_name is not null then 'PARTIAL'
            else 'INCOMPLETE'
        end as data_completeness,
        
        -- Last updated
        current_timestamp as last_updated
        
    from enriched_operators
)

select * from final_operators
