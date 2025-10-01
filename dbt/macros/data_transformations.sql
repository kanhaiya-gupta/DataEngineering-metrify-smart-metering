-- Data transformation macros
-- Reusable macros for data cleaning and transformation

{% macro clean_string_field(field_name, default_value='UNKNOWN') %}
    case 
        when {{ field_name }} is null or trim({{ field_name }}) = '' then '{{ default_value }}'
        else trim({{ field_name }})
    end
{% endmacro %}

{% macro clean_numeric_field(field_name, default_value=0) %}
    case 
        when {{ field_name }} is null then {{ default_value }}
        when {{ field_name }} = '' then {{ default_value }}
        else {{ field_name }}
    end
{% endmacro %}

{% macro clean_timestamp_field(field_name) %}
    case 
        when {{ field_name }} is null then null
        when {{ field_name }} = '' then null
        else {{ field_name }}
    end
{% endmacro %}

{% macro standardize_address(address_field) %}
    case 
        when {{ address_field }} is null or trim({{ address_field }}) = '' then 'UNKNOWN'
        else upper(trim({{ address_field }}))
    end
{% endmacro %}

{% macro standardize_tariff_type(tariff_field) %}
    case 
        when {{ tariff_field }} is null or trim({{ tariff_field }}) = '' then 'UNKNOWN'
        when upper(trim({{ tariff_field }})) in ('TOU', 'TIME_OF_USE', 'TIME OF USE') then 'TIME_OF_USE'
        when upper(trim({{ tariff_field }})) in ('FLAT', 'FLAT_RATE', 'FLAT RATE') then 'FLAT_RATE'
        when upper(trim({{ tariff_field }})) in ('TIERED', 'TIER', 'TIERED_RATE') then 'TIERED'
        else upper(trim({{ tariff_field }}))
    end
{% endmacro %}

{% macro standardize_quality_tier(quality_field) %}
    case 
        when {{ quality_field }} is null or trim({{ quality_field }}) = '' then 'UNKNOWN'
        when upper(trim({{ quality_field }})) in ('EXCELLENT', 'EXCELLENT_QUALITY') then 'EXCELLENT'
        when upper(trim({{ quality_field }})) in ('GOOD', 'GOOD_QUALITY') then 'GOOD'
        when upper(trim({{ quality_field }})) in ('FAIR', 'FAIR_QUALITY') then 'FAIR'
        when upper(trim({{ quality_field }})) in ('POOR', 'POOR_QUALITY') then 'POOR'
        else upper(trim({{ quality_field }}))
    end
{% endmacro %}

{% macro standardize_boolean_field(field_name, true_values=['true', '1', 'yes', 'y'], false_values=['false', '0', 'no', 'n']) %}
    case 
        when {{ field_name }} is null then null
        when lower(trim({{ field_name }})) in ({{ true_values | map('lower') | map('quote') | join(', ') }}) then true
        when lower(trim({{ field_name }})) in ({{ false_values | map('lower') | map('quote') | join(', ') }}) then false
        else null
    end
{% endmacro %}

{% macro extract_id_components(id_field, separator='-') %}
    case 
        when {{ id_field }} is null or trim({{ id_field }}) = '' then null
        else split_part({{ id_field }}, '{{ separator }}', 1)
    end
{% endmacro %}

{% macro extract_region_from_id(id_field, separator='-', position=2) %}
    case 
        when {{ id_field }} is null or trim({{ id_field }}) = '' then 'UNKNOWN'
        else split_part({{ id_field }}, '{{ separator }}', {{ position }})
    end
{% endmacro %}

{% macro extract_sequence_from_id(id_field, separator='-', position=3) %}
    case 
        when {{ id_field }} is null or trim({{ id_field }}) = '' then null
        else split_part({{ id_field }}, '{{ separator }}', {{ position }})
    end
{% endmacro %}

{% macro parse_json_field(json_field, key) %}
    case 
        when {{ json_field }} is null then null
        when {{ json_field }} = '' then null
        else {{ json_field }}::json->>'{{ key }}'
    end
{% endmacro %}

{% macro parse_array_field(array_field, separator=',') %}
    case 
        when {{ array_field }} is null or trim({{ array_field }}) = '' then null
        else string_to_array(trim({{ array_field }}), '{{ separator }}')
    end
{% endmacro %}

{% macro calculate_hash(fields) %}
    md5(concat({% for field in fields -%}
        coalesce({{ field }}::text, ''){% if not loop.last %} || '|' || {% endif %}
    {%- endfor %}))
{% endmacro %}

{% macro generate_surrogate_key(fields) %}
    {{ dbt_utils.generate_surrogate_key(fields) }}
{% endmacro %}

{% macro add_audit_columns() %}
    current_timestamp as created_at,
    current_timestamp as updated_at,
    '{{ invocation_id }}' as dbt_run_id
{% endmacro %}

{% macro add_data_lineage(source_system, batch_id) %}
    '{{ source_system }}' as source_system,
    '{{ batch_id }}' as ingestion_batch_id,
    current_timestamp as processed_at
{% endmacro %}

{% macro add_quality_flags(validation_checks) %}
    case 
        {% for check in validation_checks -%}
            when not {{ check }} then 'INVALID_{{ check.upper() }}'
        {%- endfor %}
        else 'VALID'
    end as quality_flag,
    case 
        {% for check in validation_checks -%}
            when not {{ check }} then 1
        {%- endfor %}
        else 0
    end as quality_issues_count
{% endmacro %}

{% macro add_time_features(timestamp_field) %}
    extract(hour from {{ timestamp_field }}) as hour_of_day,
    extract(dow from {{ timestamp_field }}) as day_of_week,
    extract(month from {{ timestamp_field }}) as month_of_year,
    extract(quarter from {{ timestamp_field }}) as quarter_of_year,
    extract(year from {{ timestamp_field }}) as year,
    extract(week from {{ timestamp_field }}) as week_of_year,
    extract(doy from {{ timestamp_field }}) as day_of_year,
    to_char({{ timestamp_field }}, 'Day') as day_name,
    to_char({{ timestamp_field }}, 'Month') as month_name,
    {{ timestamp_field }}::date as date
{% endmacro %}

{% macro add_business_time_features(timestamp_field, peak_start=17, peak_end=21, weekend_days=[0, 6]) %}
    case 
        when extract(hour from {{ timestamp_field }}) between {{ peak_start }} and {{ peak_end }} then true
        else false
    end as is_peak_hour,
    case 
        when extract(dow from {{ timestamp_field }}) in ({{ weekend_days | join(', ') }}) then true
        else false
    end as is_weekend,
    case 
        when extract(dow from {{ timestamp_field }}) in (1, 2, 3, 4, 5) then 'WEEKDAY'
        else 'WEEKEND'
    end as day_type,
    case 
        when extract(month from {{ timestamp_field }}) in (12, 1, 2) then 'WINTER'
        when extract(month from {{ timestamp_field }}) in (3, 4, 5) then 'SPRING'
        when extract(month from {{ timestamp_field }}) in (6, 7, 8) then 'SUMMER'
        when extract(month from {{ timestamp_field }}) in (9, 10, 11) then 'FALL'
    end as season
{% endmacro %}

{% macro add_geographic_features(latitude_field, longitude_field) %}
    case 
        when {{ latitude_field }} is null or {{ longitude_field }} is null then 'UNKNOWN'
        when {{ latitude_field }} > 50 and {{ longitude_field }} > 0 then 'NORTH_EAST'
        when {{ latitude_field }} > 50 and {{ longitude_field }} <= 0 then 'NORTH_WEST'
        when {{ latitude_field }} <= 50 and {{ longitude_field }} > 0 then 'SOUTH_EAST'
        when {{ latitude_field }} <= 50 and {{ longitude_field }} <= 0 then 'SOUTH_WEST'
        else 'UNKNOWN'
    end as geographic_region,
    case 
        when {{ latitude_field }} is null or {{ longitude_field }} is null then null
        else st_point({{ longitude_field }}, {{ latitude_field }})
    end as location_point
{% endmacro %}

{% macro add_data_completeness(required_fields) %}
    case 
        when {% for field in required_fields -%}
            {{ field }} is not null{% if not loop.last %} and {% endif %}
        {%- endfor %} then 'COMPLETE'
        when {% for field in required_fields -%}
            {{ field }} is not null{% if not loop.last %} or {% endif %}
        {%- endfor %} then 'PARTIAL'
        else 'INCOMPLETE'
    end as data_completeness,
    {% for field in required_fields -%}
        case when {{ field }} is not null then 1 else 0 end as {{ field }}_present{% if not loop.last %},{% endif %}
    {%- endfor %}
{% endmacro %}
