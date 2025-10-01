-- Daily consumption metrics
-- Aggregated metrics for daily consumption analysis

with daily_consumption as (
    select
        reading_date,
        count(distinct meter_id) as active_meters,
        sum(total_consumption_kwh) as total_consumption_kwh,
        avg(total_consumption_kwh) as avg_consumption_per_meter_kwh,
        max(total_consumption_kwh) as max_consumption_kwh,
        min(total_consumption_kwh) as min_consumption_kwh,
        stddev(total_consumption_kwh) as consumption_stddev,
        avg(avg_quality_score) as avg_data_quality,
        count(case when has_anomaly then 1 end) as anomaly_count,
        count(*) as total_readings
    from {{ ref('fct_smart_meter_analytics') }}
    group by reading_date
),

grid_impact as (
    select
        reading_date,
        avg(grid_stability_score) as avg_grid_stability,
        avg(utilization_rate) as avg_grid_utilization,
        count(case when grid_impact = 'high_impact' then 1 end) as high_impact_meters,
        count(case when grid_impact = 'moderate_impact' then 1 end) as moderate_impact_meters,
        count(case when grid_impact = 'low_impact' then 1 end) as low_impact_meters
    from {{ ref('fct_smart_meter_analytics') }}
    group by reading_date
),

weather_correlation as (
    select
        reading_date,
        avg(energy_demand_factor) as avg_energy_demand_factor,
        avg(temperature_celsius) as avg_temperature,
        avg(humidity_percent) as avg_humidity,
        count(case when weather_correlation = 'weather_correlated' then 1 end) as weather_correlated_meters,
        count(case when demand_category = 'high_demand' then 1 end) as high_demand_meters,
        count(case when demand_category = 'low_demand' then 1 end) as low_demand_meters
    from {{ ref('fct_smart_meter_analytics') }}
    group by reading_date
),

time_patterns as (
    select
        reading_date,
        sum(case when time_period = 'morning_peak' then total_consumption_kwh else 0 end) as morning_peak_consumption,
        sum(case when time_period = 'evening_peak' then total_consumption_kwh else 0 end) as evening_peak_consumption,
        sum(case when time_period = 'night' then total_consumption_kwh else 0 end) as night_consumption,
        sum(case when time_period = 'off_peak' then total_consumption_kwh else 0 end) as off_peak_consumption,
        sum(case when day_type = 'weekday' then total_consumption_kwh else 0 end) as weekday_consumption,
        sum(case when day_type = 'weekend' then total_consumption_kwh else 0 end) as weekend_consumption
    from {{ ref('fct_smart_meter_analytics') }}
    group by reading_date
),

final as (
    select
        dc.reading_date,
        dc.active_meters,
        dc.total_consumption_kwh,
        dc.avg_consumption_per_meter_kwh,
        dc.max_consumption_kwh,
        dc.min_consumption_kwh,
        dc.consumption_stddev,
        dc.avg_data_quality,
        dc.anomaly_count,
        dc.total_readings,
        
        -- Grid impact metrics
        gi.avg_grid_stability,
        gi.avg_grid_utilization,
        gi.high_impact_meters,
        gi.moderate_impact_meters,
        gi.low_impact_meters,
        
        -- Weather correlation metrics
        wc.avg_energy_demand_factor,
        wc.avg_temperature,
        wc.avg_humidity,
        wc.weather_correlated_meters,
        wc.high_demand_meters,
        wc.low_demand_meters,
        
        -- Time pattern metrics
        tp.morning_peak_consumption,
        tp.evening_peak_consumption,
        tp.night_consumption,
        tp.off_peak_consumption,
        tp.weekday_consumption,
        tp.weekend_consumption,
        
        -- Calculated metrics
        case 
            when dc.total_consumption_kwh > 0 then dc.anomaly_count / dc.total_readings
            else 0
        end as anomaly_rate,
        
        case 
            when dc.active_meters > 0 then dc.total_consumption_kwh / dc.active_meters
            else 0
        end as consumption_per_active_meter,
        
        case 
            when tp.morning_peak_consumption > 0 then tp.evening_peak_consumption / tp.morning_peak_consumption
            else null
        end as evening_to_morning_ratio,
        
        case 
            when tp.weekday_consumption > 0 then tp.weekend_consumption / tp.weekday_consumption
            else null
        end as weekend_to_weekday_ratio,
        
        -- Data quality metrics
        case 
            when dc.avg_data_quality >= 0.9 then 'high'
            when dc.avg_data_quality >= 0.7 then 'medium'
            else 'low'
        end as data_quality_tier,
        
        -- Grid stability assessment
        case 
            when gi.avg_grid_stability >= 0.9 then 'stable'
            when gi.avg_grid_stability >= 0.8 then 'moderate'
            else 'unstable'
        end as grid_stability_status,
        
        -- Weather impact assessment
        case 
            when wc.avg_energy_demand_factor > 1.5 then 'high_weather_impact'
            when wc.avg_energy_demand_factor > 1.2 then 'moderate_weather_impact'
            when wc.avg_energy_demand_factor < 0.8 then 'low_weather_impact'
            else 'normal_weather_impact'
        end as weather_impact_status,
        
        -- Add processing metadata
        current_timestamp() as processed_at,
        'metrics' as processing_stage
        
    from daily_consumption dc
    left join grid_impact gi on dc.reading_date = gi.reading_date
    left join weather_correlation wc on dc.reading_date = wc.reading_date
    left join time_patterns tp on dc.reading_date = tp.reading_date
)

select * from final
