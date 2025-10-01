-- Dimension table for smart meters
-- Provides master data for all smart meters in the system

with meter_data as (
    select
        meter_id,
        meter_address,
        meter_location_lat as latitude,
        meter_location_lon as longitude,
        tariff_type,
        time_of_use_period,
        reading_type,
        reading_source,
        source_system,
        min(reading_timestamp) as first_reading_date,
        max(reading_timestamp) as last_reading_date,
        count(*) as total_readings,
        avg(consumption_kwh) as avg_consumption_kwh,
        sum(consumption_kwh) as total_consumption_kwh,
        avg(quality_score) as avg_quality_score,
        count(case when is_anomaly then 1 end) as total_anomalies,
        count(case when quality_tier = 'EXCELLENT' then 1 end) as excellent_readings,
        count(case when quality_tier = 'GOOD' then 1 end) as good_readings,
        count(case when quality_tier = 'FAIR' then 1 end) as fair_readings,
        count(case when quality_tier = 'POOR' then 1 end) as poor_readings
        
    from {{ ref('stg_smart_meter_readings') }}
    group by 
        meter_id, 
        meter_address, 
        meter_location_lat, 
        meter_location_lon, 
        tariff_type, 
        time_of_use_period, 
        reading_type, 
        reading_source, 
        source_system
),

-- Add calculated fields
enriched_meters as (
    select
        *,
        -- Data quality metrics
        case 
            when total_readings > 0 then (excellent_readings::float / total_readings) * 100
            else 0
        end as excellent_quality_percent,
        
        case 
            when total_readings > 0 then (good_readings::float / total_readings) * 100
            else 0
        end as good_quality_percent,
        
        case 
            when total_readings > 0 then (fair_readings::float / total_readings) * 100
            else 0
        end as fair_quality_percent,
        
        case 
            when total_readings > 0 then (poor_readings::float / total_readings) * 100
            else 0
        end as poor_quality_percent,
        
        case 
            when total_readings > 0 then (total_anomalies::float / total_readings) * 100
            else 0
        end as anomaly_percent,
        
        -- Meter age
        current_date - first_reading_date::date as days_in_service,
        
        -- Reading frequency
        case 
            when days_in_service > 0 then total_readings::float / days_in_service
            else 0
        end as readings_per_day,
        
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
        
        -- Consumption tier classification
        case 
            when avg_consumption_kwh = 0 then 'ZERO_CONSUMPTION'
            when avg_consumption_kwh < 0.1 then 'VERY_LOW'
            when avg_consumption_kwh < 1.0 then 'LOW'
            when avg_consumption_kwh < 10.0 then 'MEDIUM'
            when avg_consumption_kwh < 50.0 then 'HIGH'
            else 'VERY_HIGH'
        end as consumption_tier,
        
        -- Meter status
        case 
            when last_reading_date < current_date - interval '7 days' then 'INACTIVE'
            when anomaly_percent >= 50 then 'FAULTY'
            when overall_quality_tier = 'POOR' then 'NEEDS_MAINTENANCE'
            else 'ACTIVE'
        end as meter_status,
        
        -- Geographic region (simplified)
        case 
            when latitude is null or longitude is null then 'UNKNOWN'
            when latitude > 50 and longitude > 0 then 'NORTH_EAST'
            when latitude > 50 and longitude <= 0 then 'NORTH_WEST'
            when latitude <= 50 and longitude > 0 then 'SOUTH_EAST'
            when latitude <= 50 and longitude <= 0 then 'SOUTH_WEST'
            else 'UNKNOWN'
        end as geographic_region
        
    from meter_data
),

-- Add additional metadata
final_meters as (
    select
        *,
        -- Meter ID components
        split_part(meter_id, '-', 1) as meter_type,
        split_part(meter_id, '-', 2) as meter_region,
        split_part(meter_id, '-', 3) as meter_sequence,
        
        -- Address components
        split_part(meter_address, ',', 1) as street_address,
        split_part(meter_address, ',', 2) as city,
        split_part(meter_address, ',', 3) as state,
        split_part(meter_address, ',', 4) as postal_code,
        
        -- Data completeness
        case 
            when latitude is not null and longitude is not null and meter_address is not null then 'COMPLETE'
            when latitude is not null and longitude is not null then 'PARTIAL'
            else 'INCOMPLETE'
        end as data_completeness,
        
        -- Last updated
        current_timestamp as last_updated
        
    from enriched_meters
)

select * from final_meters
