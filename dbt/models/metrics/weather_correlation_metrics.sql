-- Weather Correlation Metrics
-- Analyzes correlation between weather conditions and energy consumption patterns

with weather_consumption as (
    select
        sm.meter_id,
        sm.date_hour,
        sm.date,
        sm.hour_of_day,
        sm.day_of_week,
        sm.month_of_year,
        sm.quarter_of_year,
        sm.year,
        sm.season,
        sm.is_peak_hour,
        sm.is_weekend,
        sm.tariff_type,
        
        -- Consumption metrics
        sm.total_consumption_kwh,
        sm.avg_consumption_kwh,
        sm.min_consumption_kwh,
        sm.max_consumption_kwh,
        
        -- Weather metrics
        ws.avg_temperature_celsius,
        ws.avg_humidity_percent,
        ws.avg_pressure_hpa,
        ws.avg_wind_speed_ms,
        ws.total_precipitation_mm,
        ws.avg_visibility_km,
        ws.avg_cloud_cover_percent,
        ws.avg_uv_index,
        
        -- Weather classifications
        ws.temperature_classification,
        ws.humidity_classification,
        ws.wind_classification,
        ws.precipitation_classification,
        ws.weather_condition,
        
        -- Location context
        sm.meter_latitude,
        sm.meter_longitude,
        ws.station_location_lat as weather_latitude,
        ws.station_location_lon as weather_longitude,
        
        -- Distance between meter and weather station (simplified)
        sqrt(
            power(sm.meter_latitude - ws.station_location_lat, 2) +
            power(sm.meter_longitude - ws.station_location_lon, 2)
        ) as distance_km
        
    from {{ ref('fct_smart_meter_analytics') }} sm
    left join {{ ref('dim_weather_stations') }} ws
        on 1=1  -- Cross join for correlation analysis
    where sm.date_hour is not null
),

-- Calculate weather impact indicators
weather_impact as (
    select
        *,
        
        -- Temperature impact on consumption
        case 
            when avg_temperature_celsius < 0 then 'HEATING_DEMAND'
            when avg_temperature_celsius between 0 and 15 then 'MILD_HEATING'
            when avg_temperature_celsius between 15 and 25 then 'COMFORTABLE'
            when avg_temperature_celsius between 25 and 30 then 'MILD_COOLING'
            when avg_temperature_celsius > 30 then 'COOLING_DEMAND'
            else 'UNKNOWN'
        end as temperature_impact,
        
        -- Humidity impact on consumption
        case 
            when avg_humidity_percent < 30 then 'DRY_CONDITIONS'
            when avg_humidity_percent between 30 and 50 then 'COMFORTABLE'
            when avg_humidity_percent between 50 and 70 then 'MODERATE_HUMIDITY'
            when avg_humidity_percent > 70 then 'HIGH_HUMIDITY'
            else 'UNKNOWN'
        end as humidity_impact,
        
        -- Wind impact on consumption
        case 
            when avg_wind_speed_ms < 2 then 'CALM'
            when avg_wind_speed_ms between 2 and 5 then 'LIGHT_BREEZE'
            when avg_wind_speed_ms between 5 and 10 then 'MODERATE_WIND'
            when avg_wind_speed_ms between 10 and 20 then 'STRONG_WIND'
            when avg_wind_speed_ms > 20 then 'VERY_STRONG_WIND'
            else 'UNKNOWN'
        end as wind_impact,
        
        -- Precipitation impact on consumption
        case 
            when total_precipitation_mm = 0 then 'NO_RAIN'
            when total_precipitation_mm < 5 then 'LIGHT_RAIN'
            when total_precipitation_mm between 5 and 15 then 'MODERATE_RAIN'
            when total_precipitation_mm between 15 and 30 then 'HEAVY_RAIN'
            when total_precipitation_mm > 30 then 'VERY_HEAVY_RAIN'
            else 'UNKNOWN'
        end as precipitation_impact,
        
        -- Weather comfort index (0-100)
        case 
            when avg_temperature_celsius between 18 and 24 and 
                 avg_humidity_percent between 40 and 60 and 
                 avg_wind_speed_ms between 1 and 5 then 100
            when avg_temperature_celsius between 15 and 27 and 
                 avg_humidity_percent between 30 and 70 and 
                 avg_wind_speed_ms between 0 and 10 then 80
            when avg_temperature_celsius between 10 and 30 and 
                 avg_humidity_percent between 20 and 80 and 
                 avg_wind_speed_ms between 0 and 15 then 60
            when avg_temperature_celsius between 5 and 35 and 
                 avg_humidity_percent between 10 and 90 and 
                 avg_wind_speed_ms between 0 and 20 then 40
            else 20
        end as weather_comfort_index,
        
        -- Expected consumption based on weather
        case 
            when avg_temperature_celsius < 0 then avg_consumption_kwh * 1.5
            when avg_temperature_celsius between 0 and 15 then avg_consumption_kwh * 1.2
            when avg_temperature_celsius between 15 and 25 then avg_consumption_kwh * 1.0
            when avg_temperature_celsius between 25 and 30 then avg_consumption_kwh * 1.1
            when avg_temperature_celsius > 30 then avg_consumption_kwh * 1.3
            else avg_consumption_kwh
        end as expected_consumption_kwh,
        
        -- Weather severity score (0-100, higher = more severe weather)
        case 
            when avg_temperature_celsius < -10 or avg_temperature_celsius > 40 then 100
            when avg_temperature_celsius < -5 or avg_temperature_celsius > 35 then 80
            when avg_temperature_celsius < 0 or avg_temperature_celsius > 30 then 60
            when avg_temperature_celsius < 5 or avg_temperature_celsius > 25 then 40
            when avg_temperature_celsius < 10 or avg_temperature_celsius > 20 then 20
            else 0
        end as weather_severity_score
        
    from weather_consumption
),

-- Calculate correlation metrics
correlation_metrics as (
    select
        *,
        
        -- Consumption deviation from expected
        total_consumption_kwh - expected_consumption_kwh as consumption_deviation_kwh,
        
        -- Consumption efficiency based on weather
        case 
            when expected_consumption_kwh > 0 then (total_consumption_kwh / expected_consumption_kwh) * 100
            else 100
        end as consumption_efficiency_percent,
        
        -- Weather impact classification
        case 
            when abs(total_consumption_kwh - expected_consumption_kwh) / expected_consumption_kwh > 0.3 then 'HIGH_IMPACT'
            when abs(total_consumption_kwh - expected_consumption_kwh) / expected_consumption_kwh > 0.15 then 'MEDIUM_IMPACT'
            when abs(total_consumption_kwh - expected_consumption_kwh) / expected_consumption_kwh > 0.05 then 'LOW_IMPACT'
            else 'MINIMAL_IMPACT'
        end as weather_impact_classification,
        
        -- Seasonal consumption patterns
        case 
            when season = 'WINTER' and avg_temperature_celsius < 10 then 'WINTER_HEATING'
            when season = 'SUMMER' and avg_temperature_celsius > 25 then 'SUMMER_COOLING'
            when season = 'SPRING' and avg_temperature_celsius between 15 and 25 then 'SPRING_COMFORT'
            when season = 'FALL' and avg_temperature_celsius between 10 and 20 then 'FALL_COMFORT'
            else 'SEASONAL_TRANSITION'
        end as seasonal_pattern,
        
        -- Weather anomaly indicators
        case 
            when weather_severity_score > 80 then true
            when avg_temperature_celsius < -5 or avg_temperature_celsius > 35 then true
            when total_precipitation_mm > 50 then true
            when avg_wind_speed_ms > 15 then true
            else false
        end as is_weather_anomaly,
        
        -- Energy efficiency in weather context
        case 
            when consumption_efficiency_percent > 120 then 'INEFFICIENT'
            when consumption_efficiency_percent between 90 and 120 then 'EFFICIENT'
            when consumption_efficiency_percent between 80 and 90 then 'VERY_EFFICIENT'
            when consumption_efficiency_percent < 80 then 'EXTREMELY_EFFICIENT'
            else 'UNKNOWN'
        end as weather_efficiency_tier
        
    from weather_impact
),

-- Add final classifications and metrics
final_metrics as (
    select
        *,
        
        -- Overall weather impact score (0-100)
        case 
            when weather_impact_classification = 'HIGH_IMPACT' then 100
            when weather_impact_classification = 'MEDIUM_IMPACT' then 75
            when weather_impact_classification = 'LOW_IMPACT' then 50
            when weather_impact_classification = 'MINIMAL_IMPACT' then 25
            else 0
        end as weather_impact_score,
        
        -- Weather correlation strength
        case 
            when abs(consumption_deviation_kwh) / expected_consumption_kwh > 0.5 then 'STRONG_CORRELATION'
            when abs(consumption_deviation_kwh) / expected_consumption_kwh > 0.3 then 'MODERATE_CORRELATION'
            when abs(consumption_deviation_kwh) / expected_consumption_kwh > 0.1 then 'WEAK_CORRELATION'
            else 'NO_CORRELATION'
        end as correlation_strength,
        
        -- Weather optimization opportunities
        case 
            when temperature_impact in ('HEATING_DEMAND', 'COOLING_DEMAND') and consumption_efficiency_percent < 90 then 'HVAC_OPTIMIZATION'
            when humidity_impact = 'HIGH_HUMIDITY' and consumption_efficiency_percent < 95 then 'HUMIDITY_CONTROL'
            when wind_impact in ('STRONG_WIND', 'VERY_STRONG_WIND') and consumption_efficiency_percent < 90 then 'WIND_PROTECTION'
            when precipitation_impact in ('HEAVY_RAIN', 'VERY_HEAVY_RAIN') and consumption_efficiency_percent < 95 then 'WEATHER_PROOFING'
            when weather_comfort_index < 60 and consumption_efficiency_percent < 85 then 'COMFORT_OPTIMIZATION'
            else 'NO_WEATHER_OPTIMIZATION'
        end as weather_optimization_opportunity,
        
        -- Last updated
        current_timestamp as last_updated
        
    from correlation_metrics
)

select * from final_metrics
