-- Dimension table for grid operators
-- Provides master data for all grid operators in the system

with operator_data as (
    select
        operator_id,
        grid_region,
        substation_id,
        control_area,
        status_source,
        source_system,
        min(status_timestamp) as first_status_date,
        max(status_timestamp) as last_status_date,
        count(*) as total_status_updates,
        avg(grid_frequency_hz) as avg_frequency_hz,
        avg(grid_voltage_kv) as avg_voltage_kv,
        avg(grid_load_mw) as avg_load_mw,
        avg(grid_capacity_mw) as avg_capacity_mw,
        avg(grid_efficiency_percent) as avg_efficiency_percent,
        count(case when is_grid_stable then 1 end) as stable_status_count,
        count(case when is_under_frequency then 1 end) as under_frequency_count,
        count(case when is_over_frequency then 1 end) as over_frequency_count,
        count(case when is_under_voltage then 1 end) as under_voltage_count,
        count(case when is_over_voltage then 1 end) as over_voltage_count,
        count(case when is_overload then 1 end) as overload_count,
        count(case when is_emergency then 1 end) as emergency_count,
        count(case when is_anomaly then 1 end) as total_anomalies,
        avg(anomaly_score) as avg_anomaly_score
        
    from {{ ref('stg_grid_status') }}
    group by 
        operator_id, 
        grid_region, 
        substation_id, 
        control_area, 
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
            when total_status_updates > 0 then (under_frequency_count::float / total_status_updates) * 100
            else 0
        end as under_frequency_percent,
        
        case 
            when total_status_updates > 0 then (over_frequency_count::float / total_status_updates) * 100
            else 0
        end as over_frequency_percent,
        
        case 
            when total_status_updates > 0 then (under_voltage_count::float / total_status_updates) * 100
            else 0
        end as under_voltage_percent,
        
        case 
            when total_status_updates > 0 then (over_voltage_count::float / total_status_updates) * 100
            else 0
        end as over_voltage_percent,
        
        case 
            when total_status_updates > 0 then (overload_count::float / total_status_updates) * 100
            else 0
        end as overload_percent,
        
        case 
            when total_status_updates > 0 then (emergency_count::float / total_status_updates) * 100
            else 0
        end as emergency_percent,
        
        case 
            when total_status_updates > 0 then (total_anomalies::float / total_status_updates) * 100
            else 0
        end as anomaly_percent,
        
        -- Grid utilization
        case 
            when avg_capacity_mw > 0 then (avg_load_mw / avg_capacity_mw) * 100
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
            when emergency_percent >= 5 then 'HIGH_RISK'
            when emergency_percent >= 1 or overload_percent >= 10 then 'MEDIUM_RISK'
            when overload_percent >= 5 or anomaly_percent >= 20 then 'LOW_RISK'
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
            when emergency_percent >= 10 then 'EMERGENCY'
            when overload_percent >= 20 then 'OVERLOADED'
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
        
        -- Grid region components
        split_part(grid_region, '-', 1) as region_type,
        split_part(grid_region, '-', 2) as region_code,
        
        -- Control area components
        split_part(control_area, '-', 1) as control_area_type,
        split_part(control_area, '-', 2) as control_area_code,
        
        -- Data completeness
        case 
            when grid_region is not null and control_area is not null then 'COMPLETE'
            when grid_region is not null or control_area is not null then 'PARTIAL'
            else 'INCOMPLETE'
        end as data_completeness,
        
        -- Last updated
        current_timestamp as last_updated
        
    from enriched_operators
)

select * from final_operators
