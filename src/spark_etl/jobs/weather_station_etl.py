"""
Weather Station ETL Job
Apache Spark ETL job for processing weather station data
"""

import logging
import argparse
from typing import Dict, Any, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, count, sum as spark_sum, avg, min as spark_min, max as spark_max, current_timestamp, lit, lag, lead, window, row_number, rank, dense_rank, percent_rank, cume_dist, ntile, collect_list, collect_set, array, size, explode, split, regexp_replace, trim, upper, lower, initcap, concat, concat_ws, substring, length, instr, locate, lpad, rpad, ltrim, rtrim, repeat, reverse, translate, regexp_extract, date_format, year, month, dayofmonth, dayofyear, dayofweek, hour, minute, second, weekofyear, quarter, date_add, date_sub, datediff, months_between, add_months, last_day, next_day, trunc, from_unixtime, unix_timestamp, to_timestamp, to_date, round, floor, ceil, abs, sqrt, pow, exp, log, log10, sin, cos, tan, asin, acos, atan, atan2, degrees, radians, pi, e, rand, randn, monotonically_increasing_id, spark_partition_id, input_file_name, current_user, current_database, current_catalog, current_schema, current_date, now, from_utc_timestamp, to_utc_timestamp, first, last, count_distinct, approx_count_distinct, stddev, stddev_pop, stddev_samp, variance, var_pop, var_samp, skewness, kurtosis, corr, covar_pop, covar_samp, regr_avgx, regr_avgy, regr_count, regr_intercept, regr_r2, regr_slope, regr_sxx, regr_sxy, regr_syy
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, BooleanType, IntegerType
from pyspark.sql.window import Window

from ..config.spark_config import SparkETLConfig
from ..utils.data_quality import DataQualityValidator
from ...core.config.config_loader import config_loader
from ..sources import DataSourceManager
from ..config.transformation_config import get_transformation_config_service
from ..loaders.data_loader import DataLoader

logger = logging.getLogger(__name__)


class WeatherStationETLJob:
    """
    Weather Station ETL Job
    
    Processes weather station data including:
    - Data extraction from multiple sources
    - Data quality validation and cleansing
    - Data transformation and enrichment
    - Data loading to multiple destinations
    """
    
    def __init__(self, spark: SparkSession, config: SparkETLConfig, environment: str = "development"):
        self.spark = spark
        self.config = config
        self.validator = DataQualityValidator()
        self.environment = environment
        
        # Load data sources configuration from YAML
        self.data_sources_config = config_loader.get_data_sources_config()
        self.weather_station_config = self.data_sources_config.weather_stations
        
        # Initialize data source manager for multi-source support
        self.data_source_manager = DataSourceManager(
            spark=spark,
            data_source_config=self.weather_station_config.__dict__,
            environment=environment
        )
        
        # Initialize transformation configuration service
        self.transformation_config = get_transformation_config_service(environment).get_weather_station_config()
        
        # Initialize data loader for proper client integration
        self.data_loader = DataLoader(spark, environment)
        
        # Use configuration from YAML for expected schema
        self.expected_schema = {
            "required_columns": self.weather_station_config.validation.get("required_columns", []),
            "data_types": self.weather_station_config.validation.get("data_types", {}),
            "validation_rules": self.weather_station_config.validation.get("value_ranges", {}),
            "default_values": {
                "operator": "Unknown",
                "elevation_m": 0.0,
                "installation_date": "2020-01-01T00:00:00Z",
                "metadata": "{}"
            }
        }
    
    def extract_data(self, source_path: str = None, source_type: str = None) -> DataFrame:
        """Extract weather station data from any configured data source"""
        try:
            # Use primary source if not specified
            if source_type is None:
                source_type = self.data_source_manager.primary_source
            
            logger.info(f"Extracting weather station data using {source_type} source")
            
            # Create data source instance
            data_source = self.data_source_manager.create_source(source_type)
            
            # Determine source path/identifier
            if source_path is None:
                # Use configured source path based on type
                if source_type == "csv":
                    data_root = self.weather_station_config.data_root
                    observations_file = self.weather_station_config.readings_file  # This maps to observations_file in YAML
                    source_path = f"{data_root}/{observations_file}"
                elif source_type == "api":
                    source_path = "observations_endpoint"
                elif source_type == "kafka":
                    source_path = "observations_topic"
                elif source_type == "database":
                    source_path = "observations_table"
                elif source_type == "s3":
                    source_path = "observations_file"
                elif source_type == "snowflake":
                    source_path = "observations_table"
                else:
                    raise ValueError(f"Unsupported source type: {source_type}")
            
            # Extract data using the appropriate source
            df = data_source.extract(source_path)
            
            logger.info(f"Successfully extracted {df.count()} records from {source_type} source")
            return df
            
        except Exception as e:
            logger.error(f"Data extraction failed: {str(e)}")
            raise
    
    def validate_data(self, df: DataFrame) -> Dict[str, Any]:
        """Validate weather station data quality"""
        try:
            logger.info("Starting data quality validation")
            
            # Schema validation
            schema_result = self.validator.validate_schema(df, self.expected_schema)
            
            # Data quality checks
            quality_checks = {
                "null_checks": self._check_nulls(df),
                "range_checks": self._check_ranges(df),
                "duplicate_checks": self._check_duplicates(df),
                "completeness_checks": self._check_completeness(df)
            }
            
            # Compile validation results
            validation_result = {
                "schema_validation": schema_result,
                "quality_checks": quality_checks,
                "overall_quality_score": self._calculate_quality_score(quality_checks),
                "validation_timestamp": current_timestamp().cast("string")
            }
            
            logger.info(f"Data validation completed. Quality score: {validation_result['overall_quality_score']}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
    
    def transform_data(self, df: DataFrame) -> DataFrame:
        """Transform weather station data with advanced weather analytics and spatial features"""
        try:
            logger.info("Starting advanced weather station data transformation")
            
            # 1. BASIC DATA STANDARDIZATION
            logger.info("Applying basic data standardization")
            transformed_df = df \
                .withColumn("station_id", upper(trim(col("station_id")))) \
                .withColumn("observation_timestamp", to_timestamp(col("observation_timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS")) \
                .withColumn("temperature_celsius", col("temperature_celsius").cast("double")) \
                .withColumn("humidity_percent", col("humidity_percent").cast("double")) \
                .withColumn("pressure_hpa", col("pressure_hpa").cast("double")) \
                .withColumn("wind_speed_mps", col("wind_speed_mps").cast("double")) \
                .withColumn("wind_direction_degrees", col("wind_direction_degrees").cast("double")) \
                .withColumn("precipitation_mm", col("precipitation_mm").cast("double")) \
                .withColumn("visibility_km", col("visibility_km").cast("double"))
            
            # 2. TEMPORAL FEATURE ENGINEERING
            logger.info("Creating temporal features")
            transformed_df = transformed_df \
                .withColumn("year", year(col("observation_timestamp"))) \
                .withColumn("month", month(col("observation_timestamp"))) \
                .withColumn("day", dayofmonth(col("observation_timestamp"))) \
                .withColumn("hour", hour(col("observation_timestamp"))) \
                .withColumn("day_of_week", dayofweek(col("observation_timestamp"))) \
                .withColumn("day_of_year", dayofyear(col("observation_timestamp"))) \
                .withColumn("week_of_year", weekofyear(col("observation_timestamp"))) \
                .withColumn("quarter", quarter(col("observation_timestamp"))) \
                .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), True).otherwise(False)) \
                .withColumn("is_daytime", when((col("hour") >= 6) & (col("hour") <= 18), True).otherwise(False)) \
                .withColumn("is_nighttime", when((col("hour") >= 20) | (col("hour") <= 5), True).otherwise(False)) \
                .withColumn("season",
                    when(col("month").isin([12, 1, 2]), "WINTER")
                    .when(col("month").isin([3, 4, 5]), "SPRING")
                    .when(col("month").isin([6, 7, 8]), "SUMMER")
                    .otherwise("AUTUMN"))
            
            # 3. ADVANCED WEATHER CALCULATIONS
            logger.info("Calculating advanced weather metrics")
            transformed_df = transformed_df \
                .withColumn("temperature_fahrenheit", round(col("temperature_celsius") * 9/5 + 32, 2)) \
                .withColumn("temperature_kelvin", col("temperature_celsius") + 273.15) \
                .withColumn("dew_point_celsius", 
                    col("temperature_celsius") - ((100 - col("humidity_percent")) / 5)) \
                .withColumn("heat_index", 
                    when((col("temperature_celsius") > 26.7) & (col("humidity_percent") > 40),
                         col("temperature_celsius") + 0.5 * (col("humidity_percent") - 40))
                    .otherwise(col("temperature_celsius"))) \
                .withColumn("wind_chill", 
                    when(col("temperature_celsius") <= 10,
                         13.12 + 0.6215 * col("temperature_celsius") - 11.37 * pow(col("wind_speed_mps"), 0.16) + 0.3965 * col("temperature_celsius") * pow(col("wind_speed_mps"), 0.16))
                    .otherwise(col("temperature_celsius"))) \
                .withColumn("apparent_temperature", 
                    when(col("wind_speed_mps") > 0, col("wind_chill"))
                    .otherwise(col("heat_index"))) \
                .withColumn("pressure_sea_level", 
                    col("pressure_hpa") + (col("elevation_m") / 8.3)) \
                .withColumn("air_density", 
                    (col("pressure_hpa") * 100) / (287.05 * col("temperature_kelvin")))
            
            # 4. WEATHER PATTERN ANALYSIS (Configuration-Driven)
            logger.info("Creating configuration-driven weather pattern features")
            
            # Get weather analytics configuration
            weather_analytics = self.transformation_config.get_predictive_config("weather_analytics", {})
            storm_risk_config = weather_analytics.get("storm_risk", {})
            
            # Apply comfort level based on configuration
            comfort_rule = self.transformation_config.get_business_rule("comfort_levels", "comfortable")
            if comfort_rule:
                comfortable_temp_min = comfort_rule.get("temp_min", 20)
                comfortable_temp_max = comfort_rule.get("temp_max", 25)
                comfortable_humidity_min = comfort_rule.get("humidity_min", 30)
                comfortable_humidity_max = comfort_rule.get("humidity_max", 70)
                
                cold_rule = self.transformation_config.get_business_rule("comfort_levels", "cold")
                hot_rule = self.transformation_config.get_business_rule("comfort_levels", "hot")
                humid_rule = self.transformation_config.get_business_rule("comfort_levels", "humid")
                dry_rule = self.transformation_config.get_business_rule("comfort_levels", "dry")
                
                cold_temp_max = cold_rule.get("temp_max", 10) if cold_rule else 10
                hot_temp_min = hot_rule.get("temp_min", 30) if hot_rule else 30
                humid_humidity_min = humid_rule.get("humidity_min", 80) if humid_rule else 80
                dry_humidity_max = dry_rule.get("humidity_max", 20) if dry_rule else 20
                
                transformed_df = transformed_df.withColumn("comfort_level",
                    when((col("temperature_celsius").between(comfortable_temp_min, comfortable_temp_max)) & (col("humidity_percent").between(comfortable_humidity_min, comfortable_humidity_max)), "COMFORTABLE")
                    .when(col("temperature_celsius") < cold_temp_max, "COLD")
                    .when(col("temperature_celsius") > hot_temp_min, "HOT")
                    .when(col("humidity_percent") > humid_humidity_min, "HUMID")
                    .when(col("humidity_percent") < dry_humidity_max, "DRY")
                    .otherwise("MODERATE"))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("comfort_level",
                    when((col("temperature_celsius").between(20, 25)) & (col("humidity_percent").between(30, 70)), "COMFORTABLE")
                    .when(col("temperature_celsius") < 10, "COLD")
                    .when(col("temperature_celsius") > 30, "HOT")
                    .when(col("humidity_percent") > 80, "HUMID")
                    .when(col("humidity_percent") < 20, "DRY")
                    .otherwise("MODERATE"))
            
            # Apply weather condition based on configuration
            weather_condition_rule = self.transformation_config.get_business_rule("weather_conditions", "heavy_rain")
            if weather_condition_rule:
                heavy_rain_rule = self.transformation_config.get_business_rule("weather_conditions", "heavy_rain")
                light_rain_rule = self.transformation_config.get_business_rule("weather_conditions", "light_rain")
                windy_rule = self.transformation_config.get_business_rule("weather_conditions", "windy")
                foggy_rule = self.transformation_config.get_business_rule("weather_conditions", "foggy")
                freezing_rule = self.transformation_config.get_business_rule("weather_conditions", "freezing")
                extreme_heat_rule = self.transformation_config.get_business_rule("weather_conditions", "extreme_heat")
                
                heavy_rain_min = heavy_rain_rule.get("precipitation_min", 5) if heavy_rain_rule else 5
                light_rain_min = light_rain_rule.get("precipitation_min", 0.5) if light_rain_rule else 0.5
                light_rain_max = light_rain_rule.get("precipitation_max", 5) if light_rain_rule else 5
                windy_min = windy_rule.get("wind_speed_min", 15) if windy_rule else 15
                foggy_max = foggy_rule.get("visibility_max", 1) if foggy_rule else 1
                freezing_max = freezing_rule.get("temp_max", 0) if freezing_rule else 0
                extreme_heat_min = extreme_heat_rule.get("temp_min", 35) if extreme_heat_rule else 35
                
                transformed_df = transformed_df.withColumn("weather_condition",
                    when(col("precipitation_mm") > heavy_rain_min, "HEAVY_RAIN")
                    .when(col("precipitation_mm") > light_rain_min, "LIGHT_RAIN")
                    .when(col("wind_speed_mps") > windy_min, "WINDY")
                    .when(col("visibility_km") < foggy_max, "FOGGY")
                    .when(col("temperature_celsius") < freezing_max, "FREEZING")
                    .when(col("temperature_celsius") > extreme_heat_min, "EXTREME_HEAT")
                    .otherwise("CLEAR"))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("weather_condition",
                    when(col("precipitation_mm") > 5, "HEAVY_RAIN")
                    .when(col("precipitation_mm") > 0.5, "LIGHT_RAIN")
                    .when(col("wind_speed_mps") > 15, "WINDY")
                    .when(col("visibility_km") < 1, "FOGGY")
                    .when(col("temperature_celsius") < 0, "FREEZING")
                    .when(col("temperature_celsius") > 35, "EXTREME_HEAT")
                    .otherwise("CLEAR"))
            
            # Apply UV risk level based on configuration
            uv_risk_rule = self.transformation_config.get_business_rule("uv_risk_levels", "high")
            if uv_risk_rule:
                high_rule = self.transformation_config.get_business_rule("uv_risk_levels", "high")
                moderate_rule = self.transformation_config.get_business_rule("uv_risk_levels", "moderate")
                low_rule = self.transformation_config.get_business_rule("uv_risk_levels", "low")
                
                high_hour_min = high_rule.get("hour_min", 10) if high_rule else 10
                high_hour_max = high_rule.get("hour_max", 16) if high_rule else 16
                high_temp_min = high_rule.get("temp_min", 30) if high_rule else 30
                moderate_temp_min = moderate_rule.get("temp_min", 25) if moderate_rule else 25
                moderate_temp_max = moderate_rule.get("temp_max", 30) if moderate_rule else 30
                low_temp_max = low_rule.get("temp_max", 25) if low_rule else 25
                
                transformed_df = transformed_df.withColumn("uv_risk_level",
                    when(col("hour").between(high_hour_min, high_hour_max), 
                        when(col("temperature_celsius") > high_temp_min, "HIGH")
                        .when(col("temperature_celsius") > moderate_temp_min, "MODERATE")
                        .otherwise("LOW"))
                    .otherwise("LOW"))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("uv_risk_level",
                    when(col("hour").between(10, 16), 
                        when(col("temperature_celsius") > 30, "HIGH")
                        .when(col("temperature_celsius") > 25, "MODERATE")
                        .otherwise("LOW"))
                    .otherwise("LOW"))
            
            # Apply storm risk scoring based on configuration
            if storm_risk_config:
                low_pressure_threshold = storm_risk_config.get("low_pressure", {}).get("threshold", 1000)
                low_pressure_score = storm_risk_config.get("low_pressure", {}).get("score", 0.8)
                high_wind_threshold = storm_risk_config.get("high_wind", {}).get("threshold", 20)
                high_wind_score = storm_risk_config.get("high_wind", {}).get("score", 0.9)
                heavy_precipitation_threshold = storm_risk_config.get("heavy_precipitation", {}).get("threshold", 10)
                heavy_precipitation_score = storm_risk_config.get("heavy_precipitation", {}).get("score", 0.7)
                high_humidity_threshold = storm_risk_config.get("high_humidity", {}).get("threshold", 90)
                high_humidity_score = storm_risk_config.get("high_humidity", {}).get("score", 0.6)
                default_score = storm_risk_config.get("default", {}).get("score", 0.1)
                
                transformed_df = transformed_df.withColumn("storm_risk_score",
                    when(col("pressure_hpa") < low_pressure_threshold, low_pressure_score)
                    .when(col("wind_speed_mps") > high_wind_threshold, high_wind_score)
                    .when(col("precipitation_mm") > heavy_precipitation_threshold, heavy_precipitation_score)
                    .when(col("humidity_percent") > high_humidity_threshold, high_humidity_score)
                    .otherwise(default_score))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("storm_risk_score",
                    when(col("pressure_hpa") < 1000, 0.8)
                    .when(col("wind_speed_mps") > 20, 0.9)
                    .when(col("precipitation_mm") > 10, 0.7)
                    .when(col("humidity_percent") > 90, 0.6)
                    .otherwise(0.1))
            
            # 5. TIME-SERIES FEATURES
            logger.info("Creating time-series features")
            window_1h = Window.partitionBy("station_id").orderBy("observation_timestamp").rowsBetween(-1, -1)
            window_24h = Window.partitionBy("station_id").orderBy("observation_timestamp").rowsBetween(-24, -1)
            window_7d = Window.partitionBy("station_id").orderBy("observation_timestamp").rowsBetween(-168, -1)
            
            transformed_df = transformed_df \
                .withColumn("temperature_lag_1h", lag(col("temperature_celsius"), 1).over(Window.partitionBy("station_id").orderBy("observation_timestamp"))) \
                .withColumn("pressure_lag_1h", lag(col("pressure_hpa"), 1).over(Window.partitionBy("station_id").orderBy("observation_timestamp"))) \
                .withColumn("temperature_avg_24h", avg(col("temperature_celsius")).over(window_24h)) \
                .withColumn("humidity_avg_24h", avg(col("humidity_percent")).over(window_24h)) \
                .withColumn("pressure_avg_24h", avg(col("pressure_hpa")).over(window_24h)) \
                .withColumn("wind_speed_avg_24h", avg(col("wind_speed_mps")).over(window_24h)) \
                .withColumn("temperature_avg_7d", avg(col("temperature_celsius")).over(window_7d)) \
                .withColumn("temperature_trend_24h", col("temperature_celsius") - col("temperature_avg_24h")) \
                .withColumn("pressure_trend_24h", col("pressure_hpa") - col("pressure_avg_24h")) \
                .withColumn("temperature_volatility_24h", stddev(col("temperature_celsius")).over(window_24h)) \
                .withColumn("pressure_volatility_24h", stddev(col("pressure_hpa")).over(window_24h))
            
            # 6. ANOMALY DETECTION FEATURES
            logger.info("Creating anomaly detection features")
            window_anomaly = Window.partitionBy("station_id").orderBy("observation_timestamp").rowsBetween(-10, 10)
            
            transformed_df = transformed_df \
                .withColumn("temperature_mean_20", avg(col("temperature_celsius")).over(window_anomaly)) \
                .withColumn("temperature_std_20", stddev(col("temperature_celsius")).over(window_anomaly)) \
                .withColumn("pressure_mean_20", avg(col("pressure_hpa")).over(window_anomaly)) \
                .withColumn("pressure_std_20", stddev(col("pressure_hpa")).over(window_anomaly)) \
                .withColumn("temperature_zscore", when(col("temperature_std_20") > 0, (col("temperature_celsius") - col("temperature_mean_20")) / col("temperature_std_20")).otherwise(0)) \
                .withColumn("pressure_zscore", when(col("pressure_std_20") > 0, (col("pressure_hpa") - col("pressure_mean_20")) / col("pressure_std_20")).otherwise(0)) \
                .withColumn("is_temperature_anomaly", when(abs(col("temperature_zscore")) > 3, True).otherwise(False)) \
                .withColumn("is_pressure_anomaly", when(abs(col("pressure_zscore")) > 3, True).otherwise(False)) \
                .withColumn("weather_anomaly_score", (abs(col("temperature_zscore")) + abs(col("pressure_zscore"))) / 2)
            
            # 7. SPATIAL AND GEOGRAPHIC FEATURES
            logger.info("Adding spatial and geographic features")
            transformed_df = transformed_df \
                .withColumn("station_region", 
                    when(col("station_id").rlike("^NORTH"), "NORTH")
                    .when(col("station_id").rlike("^SOUTH"), "SOUTH")
                    .when(col("station_id").rlike("^EAST"), "EAST")
                    .when(col("station_id").rlike("^WEST"), "WEST")
                    .otherwise("CENTRAL")) \
                .withColumn("elevation_category",
                    when(col("elevation_m") < 100, "LOWLAND")
                    .when(col("elevation_m") < 500, "HILLS")
                    .when(col("elevation_m") < 1000, "MOUNTAINS")
                    .otherwise("HIGH_MOUNTAINS")) \
                .withColumn("climate_zone",
                    when(col("station_region") == "NORTH", "TEMPERATE")
                    .when(col("station_region") == "SOUTH", "TROPICAL")
                    .when(col("station_region") == "EAST", "CONTINENTAL")
                    .when(col("station_region") == "WEST", "MARITIME")
                    .otherwise("MIXED")) \
                .withColumn("temperature_category",
                    when(col("temperature_celsius") < -10, "EXTREME_COLD")
                    .when(col("temperature_celsius") < 0, "COLD")
                    .when(col("temperature_celsius") < 15, "COOL")
                    .when(col("temperature_celsius") < 25, "MILD")
                    .when(col("temperature_celsius") < 35, "WARM")
                    .otherwise("HOT"))
            
            # 8. PREDICTIVE FEATURES
            logger.info("Creating predictive features")
            transformed_df = transformed_df \
                .withColumn("temperature_forecast_1h", col("temperature_avg_24h") * 0.7 + col("temperature_celsius") * 0.3) \
                .withColumn("temperature_forecast_24h", col("temperature_avg_7d") * 1.01) \
                .withColumn("precipitation_forecast_24h", 
                    when(col("pressure_hpa") < 1010, col("precipitation_mm") * 1.5)
                    .when(col("humidity_percent") > 80, col("precipitation_mm") * 1.2)
                    .otherwise(col("precipitation_mm") * 0.8)) \
                .withColumn("weather_change_probability",
                    when(col("pressure_trend_24h") < -5, 0.8)
                    .when(col("pressure_trend_24h") > 5, 0.3)
                    .when(col("temperature_trend_24h") > 5, 0.6)
                    .when(col("wind_speed_mps") > 15, 0.7)
                    .otherwise(0.2)) \
                .withColumn("extreme_weather_risk",
                    when(col("storm_risk_score") > 0.7, "HIGH")
                    .when(col("storm_risk_score") > 0.4, "MODERATE")
                    .when(col("temperature_celsius") > 40, "HEAT_WAVE")
                    .when(col("temperature_celsius") < -20, "COLD_WAVE")
                    .otherwise("LOW"))
            
            # 9. DATA QUALITY AND METADATA
            logger.info("Adding data quality and metadata")
            transformed_df = transformed_df \
                .withColumn("processed_at", current_timestamp()) \
                .withColumn("data_source", lit("spark_etl_advanced")) \
                .withColumn("processing_version", lit("2.0.0")) \
                .withColumn("transformation_batch_id", monotonically_increasing_id()) \
                .withColumn("data_quality_score", 
                    when(col("temperature_celsius").isNull(), 0.0)
                    .when(col("humidity_percent").isNull(), 0.5)
                    .when(col("pressure_hpa").isNull(), 0.7)
                    .otherwise(1.0)) \
                .withColumn("completeness_score",
                    (when(col("temperature_celsius").isNotNull(), 1).otherwise(0) +
                     when(col("humidity_percent").isNotNull(), 1).otherwise(0) +
                     when(col("pressure_hpa").isNotNull(), 1).otherwise(0) +
                     when(col("wind_speed_mps").isNotNull(), 1).otherwise(0) +
                     when(col("precipitation_mm").isNotNull(), 1).otherwise(0)) / 5.0)
            
            logger.info(f"Advanced weather station transformation completed. Output records: {transformed_df.count()}")
            logger.info(f"Total columns created: {len(transformed_df.columns)}")
            return transformed_df
            
        except Exception as e:
            logger.error(f"Advanced weather station transformation failed: {str(e)}")
            raise
    
    def load_data(self, df: DataFrame, target_path: str, format: str = "delta", 
                  postgres_output: Optional[str] = None, snowflake_output: Optional[str] = None) -> Dict[str, Any]:
        """Load processed data to multiple destinations using proper client integration"""
        try:
            logger.info("Starting data loading with proper client integration")
            
            # Prepare destinations configuration
            destinations = {}
            
            # Always load to S3 (primary destination)
            destinations["s3"] = {
                "target_path": target_path,
                "format": format,
                "write_mode": "overwrite",
                "partition_columns": ["observation_timestamp"]  # Partition by timestamp for better performance
            }
            
            # Add PostgreSQL destination if specified
            if postgres_output:
                destinations["postgresql"] = {
                    "table_name": "weather_observations",
                    "connection_string": postgres_output,
                    "write_mode": "append",
                    "batch_size": 1000
                }
            
            # Add Snowflake destination if specified
            if snowflake_output:
                destinations["snowflake"] = {
                    "table_name": "weather_station_data",
                    "schema": "PROCESSED",
                    "write_mode": "overwrite"
                }
            
            # Load to multiple destinations using DataLoader
            load_result = self.data_loader.load_to_multiple_destinations(df, destinations)
            
            logger.info(f"Data loading completed. Status: {load_result['status']}")
            return load_result
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def run_etl(self, source_path: str, target_path: str, format: str = "delta", 
                postgres_output: Optional[str] = None, snowflake_output: Optional[str] = None,
                source_type: Optional[str] = None) -> Dict[str, Any]:
        """Run complete ETL pipeline"""
        try:
            logger.info("Starting Weather Station ETL job")
            
            # Extract
            raw_df = self.extract_data(source_path, source_type)
            
            # Validate
            quality_report = self.validate_data(raw_df)
            
            # Transform
            transformed_df = self.transform_data(raw_df)
            
            # Load to multiple destinations
            load_result = self.load_data(transformed_df, target_path, format, postgres_output, snowflake_output)
            
            # Compile results
            etl_result = {
                "job_name": "weather_station_etl",
                "status": "success",
                "source_path": source_path,
                "target_path": target_path,
                "records_processed": raw_df.count(),
                "data_quality": quality_report,
                "destinations": load_result["destinations"],
                "processing_timestamp": str(current_timestamp().cast("string"))
            }
            
            logger.info("Weather Station ETL job completed successfully")
            return etl_result
            
        except Exception as e:
            logger.error(f"ETL job failed: {str(e)}")
            return {
                "job_name": "weather_station_etl",
                "status": "failed",
                "error": str(e),
                "source_path": source_path,
                "target_path": target_path
            }


def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description="Weather Station ETL Job")
    parser.add_argument("--input-path", help="Input data path (optional, uses configured source)")
    parser.add_argument("--output-path", required=True, help="Output data path")
    parser.add_argument("--output-format", default="delta", help="Output format (delta/parquet)")
    parser.add_argument("--postgres-output", help="PostgreSQL connection string")
    parser.add_argument("--snowflake-output", help="Snowflake connection string")
    parser.add_argument("--batch-id", help="Batch ID for processing")
    parser.add_argument("--data-type", default="weather_station", help="Data type")
    parser.add_argument("--source-type", help="Data source type (csv, api, kafka, database, s3, snowflake)")
    parser.add_argument("--environment", default="development", help="Environment (development, staging, production)")
    
    args = parser.parse_args()
    
    # Initialize Spark
    config = SparkETLConfig()
    spark = SparkSession.builder \
        .appName("WeatherStationETL") \
        .config(conf=config.get_spark_conf()) \
        .getOrCreate()
    
    try:
        # Create ETL job with environment
        etl_job = WeatherStationETLJob(spark, config, environment=args.environment)
        
        # Run ETL
        result = etl_job.run_etl(
            source_path=args.input_path,
            target_path=args.output_path,
            format=args.output_format,
            postgres_output=args.postgres_output,
            snowflake_output=args.snowflake_output,
            source_type=args.source_type
        )
        
        print(f"ETL Result: {result}")
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()