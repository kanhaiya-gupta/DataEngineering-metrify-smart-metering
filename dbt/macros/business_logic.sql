-- Business logic macros
-- Reusable macros for business calculations and classifications

{% macro calculate_utilization_percent(load_mw, capacity_mw) %}
    case 
        when {{ capacity_mw }} > 0 then ({{ load_mw }} / {{ capacity_mw }}) * 100
        else null
    end
{% endmacro %}

{% macro calculate_efficiency_adjusted_consumption(consumption_kwh, power_factor) %}
    case 
        when {{ power_factor }} > 0 then {{ consumption_kwh }} / {{ power_factor }}
        else {{ consumption_kwh }}
    end
{% endmacro %}

{% macro calculate_reading_frequency(total_readings, days_in_service) %}
    case 
        when {{ days_in_service }} > 0 then {{ total_readings }}::float / {{ days_in_service }}
        else 0
    end
{% endmacro %}

{% macro calculate_quality_percentage(count, total) %}
    case 
        when {{ total }} > 0 then ({{ count }}::float / {{ total }}) * 100
        else 0
    end
{% endmacro %}

{% macro calculate_anomaly_percentage(anomaly_count, total_count) %}
    case 
        when {{ total_count }} > 0 then ({{ anomaly_count }}::float / {{ total_count }}) * 100
        else 0
    end
{% endmacro %}

{% macro calculate_consumption_efficiency(consumption_kwh, voltage_v, current_a) %}
    case 
        when {{ voltage_v }} > 0 and {{ current_a }} > 0 then 
            {{ consumption_kwh }} / ({{ voltage_v }} * {{ current_a }} / 1000)
        else {{ consumption_kwh }}
    end
{% endmacro %}

{% macro calculate_power_factor_correction(current_pf, target_pf=0.95) %}
    case 
        when {{ current_pf }} < {{ target_pf }} then {{ target_pf }} - {{ current_pf }}
        else 0
    end
{% endmacro %}

{% macro calculate_voltage_deviation(actual_voltage, nominal_voltage=230) %}
    case 
        when {{ nominal_voltage }} > 0 then 
            abs({{ actual_voltage }} - {{ nominal_voltage }}) / {{ nominal_voltage }} * 100
        else null
    end
{% endmacro %}

{% macro calculate_frequency_deviation(actual_frequency, nominal_frequency=50) %}
    case 
        when {{ nominal_frequency }} > 0 then 
            abs({{ actual_frequency }} - {{ nominal_frequency }}) / {{ nominal_frequency }} * 100
        else null
    end
{% endmacro %}

{% macro calculate_weather_impact_score(temperature, humidity, wind_speed, precipitation) %}
    case 
        when {{ temperature }} is null or {{ humidity }} is null then null
        when {{ temperature }} < 0 or {{ temperature }} > 35 then 0.8
        when {{ humidity }} > 80 and {{ wind_speed }} > 10 then 0.6
        when {{ precipitation }} > 5 then 0.4
        when {{ temperature }} between 15 and 25 and {{ humidity }} between 40 and 60 then 1.0
        else 0.7
    end
{% endmacro %}

{% macro calculate_peak_demand_period(hour, day_of_week) %}
    case 
        when {{ day_of_week }} in (1, 2, 3, 4, 5) and {{ hour }} between 17 and 21 then 'PEAK_WEEKDAY'
        when {{ day_of_week }} in (0, 6) and {{ hour }} between 17 and 21 then 'PEAK_WEEKEND'
        when {{ day_of_week }} in (1, 2, 3, 4, 5) and {{ hour }} between 22 and 23 or {{ hour }} between 0 and 6 then 'OFF_PEAK_WEEKDAY'
        when {{ day_of_week }} in (0, 6) and {{ hour }} between 22 and 23 or {{ hour }} between 0 and 6 then 'OFF_PEAK_WEEKEND'
        else 'OTHER'
    end
{% endmacro %}

{% macro calculate_tariff_rate(time_period, tariff_type) %}
    case 
        when {{ tariff_type }} = 'TIME_OF_USE' then
            case 
                when {{ time_period }} = 'PEAK_WEEKDAY' then 0.25
                when {{ time_period }} = 'PEAK_WEEKEND' then 0.20
                when {{ time_period }} = 'OFF_PEAK_WEEKDAY' then 0.10
                when {{ time_period }} = 'OFF_PEAK_WEEKEND' then 0.08
                else 0.15
            end
        when {{ tariff_type }} = 'FLAT_RATE' then 0.15
        when {{ tariff_type }} = 'TIERED' then
            case 
                when {{ time_period }} in ('PEAK_WEEKDAY', 'PEAK_WEEKEND') then 0.20
                else 0.12
            end
        else 0.15
    end
{% endmacro %}

{% macro calculate_energy_cost(consumption_kwh, tariff_rate) %}
    {{ consumption_kwh }} * {{ tariff_rate }}
{% endmacro %}

{% macro calculate_carbon_footprint(consumption_kwh, carbon_intensity=0.5) %}
    {{ consumption_kwh }} * {{ carbon_intensity }}
{% endmacro %}

{% macro calculate_energy_savings_potential(consumption_kwh, efficiency_score) %}
    case 
        when {{ efficiency_score }} < 0.8 then {{ consumption_kwh }} * (0.8 - {{ efficiency_score }})
        else 0
    end
{% endmacro %}

{% macro calculate_demand_response_potential(consumption_kwh, time_period) %}
    case 
        when {{ time_period }} in ('PEAK_WEEKDAY', 'PEAK_WEEKEND') then {{ consumption_kwh }} * 0.3
        else {{ consumption_kwh }} * 0.1
    end
{% endmacro %}

{% macro calculate_grid_stability_score(frequency_hz, voltage_kv, load_mw, capacity_mw) %}
    case 
        when {{ frequency_hz }} is null or {{ voltage_kv }} is null or {{ load_mw }} is null or {{ capacity_mw }} is null then null
        when {{ frequency_hz }} between 49.5 and 50.5 and {{ voltage_kv }} between 220 and 240 and {{ load_mw }} < {{ capacity_mw }} * 0.8 then 1.0
        when {{ frequency_hz }} between 49.0 and 51.0 and {{ voltage_kv }} between 210 and 250 and {{ load_mw }} < {{ capacity_mw }} * 0.9 then 0.8
        when {{ frequency_hz }} between 48.0 and 52.0 and {{ voltage_kv }} between 200 and 260 and {{ load_mw }} < {{ capacity_mw }} then 0.6
        else 0.3
    end
{% endmacro %}

{% macro calculate_weather_correlation(temperature, consumption_kwh) %}
    case 
        when {{ temperature }} is null or {{ consumption_kwh }} is null then null
        when {{ temperature }} < 10 or {{ temperature }} > 30 then 0.8
        when {{ temperature }} between 15 and 25 then 0.3
        else 0.5
    end
{% endmacro %}

{% macro calculate_anomaly_severity(anomaly_score, consumption_kwh) %}
    case 
        when {{ anomaly_score }} >= 0.8 then 'CRITICAL'
        when {{ anomaly_score }} >= 0.6 then 'HIGH'
        when {{ anomaly_score }} >= 0.4 then 'MEDIUM'
        when {{ anomaly_score }} >= 0.2 then 'LOW'
        else 'MINIMAL'
    end
{% endmacro %}

{% macro calculate_maintenance_priority(quality_score, anomaly_percent, days_since_last_maintenance) %}
    case 
        when {{ quality_score }} < 50 or {{ anomaly_percent }} > 30 then 'URGENT'
        when {{ quality_score }} < 70 or {{ anomaly_percent }} > 20 or {{ days_since_last_maintenance }} > 365 then 'HIGH'
        when {{ quality_score }} < 80 or {{ anomaly_percent }} > 10 or {{ days_since_last_maintenance }} > 180 then 'MEDIUM'
        else 'LOW'
    end
{% endmacro %}
