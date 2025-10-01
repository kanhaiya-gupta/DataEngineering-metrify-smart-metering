-- Data quality check macros
-- Reusable macros for data quality validation

{% macro validate_consumption_range(column_name, min_value=0, max_value=1000) %}
    case 
        when {{ column_name }} < {{ min_value }} then false
        when {{ column_name }} > {{ max_value }} then false
        else true
    end
{% endmacro %}

{% macro validate_temperature_range(column_name, min_value=-50, max_value=60) %}
    case 
        when {{ column_name }} < {{ min_value }} then false
        when {{ column_name }} > {{ max_value }} then false
        else true
    end
{% endmacro %}

{% macro validate_humidity_range(column_name, min_value=0, max_value=100) %}
    case 
        when {{ column_name }} < {{ min_value }} then false
        when {{ column_name }} > {{ max_value }} then false
        else true
    end
{% endmacro %}

{% macro validate_frequency_range(column_name, min_value=45, max_value=65) %}
    case 
        when {{ column_name }} < {{ min_value }} then false
        when {{ column_name }} > {{ max_value }} then false
        else true
    end
{% endmacro %}

{% macro validate_voltage_range(column_name, min_value=100, max_value=500) %}
    case 
        when {{ column_name }} < {{ min_value }} then false
        when {{ column_name }} > {{ max_value }} then false
        else true
    end
{% endmacro %}

{% macro calculate_quality_tier(quality_score) %}
    case 
        when {{ quality_score }} >= 90 then 'EXCELLENT'
        when {{ quality_score }} >= 80 then 'GOOD'
        when {{ quality_score }} >= 70 then 'FAIR'
        else 'POOR'
    end
{% endmacro %}

{% macro calculate_anomaly_classification(anomaly_percent) %}
    case 
        when {{ anomaly_percent }} >= 20 then 'HIGH_ANOMALY'
        when {{ anomaly_percent }} >= 10 then 'MEDIUM_ANOMALY'
        when {{ anomaly_percent }} > 0 then 'LOW_ANOMALY'
        else 'NO_ANOMALY'
    end
{% endmacro %}

{% macro calculate_consumption_category(consumption_kwh) %}
    case 
        when {{ consumption_kwh }} = 0 then 'ZERO_CONSUMPTION'
        when {{ consumption_kwh }} < 0.1 then 'VERY_LOW'
        when {{ consumption_kwh }} < 1.0 then 'LOW'
        when {{ consumption_kwh }} < 10.0 then 'MEDIUM'
        when {{ consumption_kwh }} < 50.0 then 'HIGH'
        else 'VERY_HIGH'
    end
{% endmacro %}

{% macro calculate_temperature_category(temperature_celsius) %}
    case 
        when {{ temperature_celsius }} is null then 'UNKNOWN'
        when {{ temperature_celsius }} < -10 then 'VERY_COLD'
        when {{ temperature_celsius }} < 0 then 'COLD'
        when {{ temperature_celsius }} < 10 then 'COOL'
        when {{ temperature_celsius }} < 20 then 'MILD'
        when {{ temperature_celsius }} < 30 then 'WARM'
        when {{ temperature_celsius }} < 40 then 'HOT'
        else 'VERY_HOT'
    end
{% endmacro %}

{% macro calculate_humidity_category(humidity_percent) %}
    case 
        when {{ humidity_percent }} is null then 'UNKNOWN'
        when {{ humidity_percent }} < 30 then 'DRY'
        when {{ humidity_percent }} < 50 then 'COMFORTABLE'
        when {{ humidity_percent }} < 70 then 'MODERATE'
        else 'HUMID'
    end
{% endmacro %}

{% macro calculate_peak_hour_classification(hour, peak_start=17, peak_end=21) %}
    case 
        when {{ hour }} between {{ peak_start }} and {{ peak_end }} then true
        else false
    end
{% endmacro %}

{% macro calculate_weekend_classification(day_of_week, weekend_days=[0, 6]) %}
    case 
        when {{ day_of_week }} in ({{ weekend_days | join(', ') }}) then true
        else false
    end
{% endmacro %}

{% macro calculate_season(month) %}
    case 
        when {{ month }} in (12, 1, 2) then 'WINTER'
        when {{ month }} in (3, 4, 5) then 'SPRING'
        when {{ month }} in (6, 7, 8) then 'SUMMER'
        when {{ month }} in (9, 10, 11) then 'FALL'
    end
{% endmacro %}

{% macro calculate_geographic_region(latitude, longitude) %}
    case 
        when {{ latitude }} is null or {{ longitude }} is null then 'UNKNOWN'
        when {{ latitude }} > 50 and {{ longitude }} > 0 then 'NORTH_EAST'
        when {{ latitude }} > 50 and {{ longitude }} <= 0 then 'NORTH_WEST'
        when {{ latitude }} <= 50 and {{ longitude }} > 0 then 'SOUTH_EAST'
        when {{ latitude }} <= 50 and {{ longitude }} <= 0 then 'SOUTH_WEST'
        else 'UNKNOWN'
    end
{% endmacro %}

{% macro calculate_data_completeness(required_fields) %}
    case 
        when {% for field in required_fields -%}
            {{ field }} is not null{% if not loop.last %} and {% endif %}
        {%- endfor %} then 'COMPLETE'
        when {% for field in required_fields -%}
            {{ field }} is not null{% if not loop.last %} or {% endif %}
        {%- endfor %} then 'PARTIAL'
        else 'INCOMPLETE'
    end
{% endmacro %}

{% macro generate_quality_flag(validation_checks) %}
    case 
        {% for check in validation_checks -%}
            when not {{ check }} then 'INVALID_{{ check.upper() }}'
        {%- endfor %}
        else 'VALID'
    end
{% endmacro %}
