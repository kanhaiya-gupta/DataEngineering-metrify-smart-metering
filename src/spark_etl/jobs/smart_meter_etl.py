"""
Smart Meter ETL Job
Apache Spark ETL job for processing smart meter data
"""

import logging
import argparse
from typing import Dict, Any, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, count, sum as spark_sum, avg, min as spark_min, max as spark_max, current_timestamp, lit, lag, lead, window, row_number, rank, dense_rank, percent_rank, cume_dist, ntile, collect_list, collect_set, array, size, explode, split, regexp_replace, trim, upper, lower, initcap, concat, concat_ws, substring, length, instr, locate, lpad, rpad, ltrim, rtrim, repeat, reverse, translate, regexp_extract, date_format, year, month, dayofmonth, dayofyear, dayofweek, hour, minute, second, weekofyear, quarter, date_add, date_sub, datediff, months_between, add_months, last_day, next_day, trunc, from_unixtime, unix_timestamp, to_timestamp, to_date, round, floor, ceil, abs, sqrt, pow, exp, log, log10, sin, cos, tan, asin, acos, atan, atan2, degrees, radians, pi, e, rand, randn, monotonically_increasing_id, spark_partition_id, input_file_name, current_user, current_database, current_catalog, current_schema, current_date, now, from_utc_timestamp, to_utc_timestamp, first, last, count_distinct, approx_count_distinct, stddev, stddev_pop, stddev_samp, variance, var_pop, var_samp, skewness, kurtosis, corr, covar_pop, covar_samp, regr_avgx, regr_avgy, regr_count, regr_intercept, regr_r2, regr_slope, regr_sxx, regr_sxy, regr_syy
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, BooleanType

from ..config.spark_config import SparkETLConfig
from ..utils.data_quality import DataQualityValidator
from ...core.config.config_loader import config_loader
from ..sources import DataSourceManager
from ..config.transformation_config import get_transformation_config_service
from ..loaders.data_loader import DataLoader

logger = logging.getLogger(__name__)


class SmartMeterETLJob:
    """Smart Meter ETL Job using Apache Spark"""
    
    def __init__(self, spark: SparkSession, config: SparkETLConfig, environment: str = "development"):
        self.spark = spark
        self.config = config
        self.validator = DataQualityValidator(spark)
        self.environment = environment
        
        # Load data sources configuration from YAML
        self.data_sources_config = config_loader.get_data_sources_config()
        self.smart_meter_config = self.data_sources_config.smart_meters
        
        # Initialize data source manager for multi-source support
        self.data_source_manager = DataSourceManager(
            spark=spark,
            data_source_config=self.smart_meter_config.__dict__,
            environment=environment
        )
        
        # Initialize transformation configuration service
        self.transformation_config = get_transformation_config_service(environment).get_smart_meter_config()
        
        # Initialize data loader for proper client integration
        self.data_loader = DataLoader(spark, environment)
        
        # Use configuration from YAML for expected schema
        self.expected_schema = {
            "required_columns": self.smart_meter_config.validation.get("required_columns", []),
            "data_types": self.smart_meter_config.validation.get("data_types", {}),
            "validation_rules": self.smart_meter_config.validation.get("value_ranges", {}),
            "default_values": {
                "manufacturer": "Unknown",
                "model": "Unknown", 
                "status": "ACTIVE",
                "quality_tier": "UNKNOWN",
                "metadata": "{}"
            }
        }
        
        # Define data quality rules from YAML config
        self.quality_rules = {
            "required_fields": self.smart_meter_config.validation.get("required_columns", []),
            "range_checks": self.smart_meter_config.validation.get("value_ranges", {}),
            "duplicate_key_columns": ["meter_id"],
            "cleaning_rules": {
                "remove_duplicates": True,
                "null_handling": {
                    "firmware_version": "fill_default",
                    "metadata": "fill_default"
                },
                "default_values": {
                    "firmware_version": "1.0.0",
                    "metadata": "{}"
                }
            }
        }
    
    def extract_data(self, source_path: str = None, source_type: str = None) -> DataFrame:
        """Extract smart meter data from any configured data source"""
        try:
            # Use primary source if not specified
            if source_type is None:
                source_type = self.data_source_manager.primary_source
            
            logger.info(f"Extracting smart meter data using {source_type} source")
            
            # Create data source instance
            data_source = self.data_source_manager.create_source(source_type)
            
            # Determine source path/identifier
            if source_path is None:
                # Use configured source path based on type
                if source_type == "csv":
                    data_root = self.smart_meter_config.data_root
                    readings_file = self.smart_meter_config.readings_file
                    source_path = f"{data_root}/{readings_file}"
                elif source_type == "api":
                    source_path = "readings_endpoint"
                elif source_type == "kafka":
                    source_path = "readings_topic"
                elif source_type == "database":
                    source_path = "readings_table"
                elif source_type == "s3":
                    source_path = "readings_file"
                elif source_type == "snowflake":
                    source_path = "readings_table"
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
        """Validate smart meter data quality"""
        try:
            logger.info("Starting data quality validation")
            
            # Schema validation
            schema_result = self.validator.validate_schema(df, self.expected_schema)
            
            # Completeness validation
            completeness_result = self.validator.validate_completeness(
                df, self.quality_rules["required_fields"]
            )
            
            # Range validation
            range_result = self.validator.validate_ranges(
                df, self.quality_rules["range_checks"]
            )
            
            # Duplicate detection
            duplicate_result = self.validator.detect_duplicates(
                df, self.quality_rules["duplicate_key_columns"]
            )
            
            # Generate quality report
            quality_report = self.validator.generate_quality_report()
            
            logger.info(f"Data quality validation completed. Score: {quality_report['overall_quality_score']:.2f}%")
            return quality_report
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
    
    def transform_data(self, df: DataFrame) -> DataFrame:
        """Transform smart meter data with advanced analytics and business logic"""
        try:
            logger.info("Starting advanced data transformation")
            
            # Apply data cleaning rules
            cleaned_df = self.validator.clean_data(df, self.quality_rules["cleaning_rules"])
            
            # 1. BASIC DATA STANDARDIZATION
            logger.info("Applying basic data standardization")
            transformed_df = cleaned_df \
                .withColumn("meter_id", upper(trim(col("meter_id")))) \
                .withColumn("reading_timestamp", to_timestamp(col("reading_timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS")) \
                .withColumn("consumption_kwh", col("consumption_kwh").cast("double")) \
                .withColumn("voltage_v", col("voltage_v").cast("double")) \
                .withColumn("current_a", col("current_a").cast("double")) \
                .withColumn("power_factor", col("power_factor").cast("double")) \
                .withColumn("frequency_hz", col("frequency_hz").cast("double"))
            
            # 2. TEMPORAL FEATURE ENGINEERING
            logger.info("Creating temporal features")
            transformed_df = transformed_df \
                .withColumn("year", year(col("reading_timestamp"))) \
                .withColumn("month", month(col("reading_timestamp"))) \
                .withColumn("day", dayofmonth(col("reading_timestamp"))) \
                .withColumn("hour", hour(col("reading_timestamp"))) \
                .withColumn("day_of_week", dayofweek(col("reading_timestamp"))) \
                .withColumn("day_of_year", dayofyear(col("reading_timestamp"))) \
                .withColumn("week_of_year", weekofyear(col("reading_timestamp"))) \
                .withColumn("quarter", quarter(col("reading_timestamp"))) \
                .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), True).otherwise(False)) \
                .withColumn("is_business_hour", when((col("hour") >= 9) & (col("hour") <= 17), True).otherwise(False)) \
                .withColumn("is_peak_hour", when(col("hour").isin([18, 19, 20, 21]), True).otherwise(False)) \
                .withColumn("is_night_hour", when((col("hour") >= 22) | (col("hour") <= 6), True).otherwise(False))
            
            # 3. ADVANCED POWER CALCULATIONS
            logger.info("Calculating advanced power metrics")
            transformed_df = transformed_df \
                .withColumn("apparent_power_calculated", col("voltage_v") * col("current_a")) \
                .withColumn("active_power_calculated", col("apparent_power_calculated") * col("power_factor")) \
                .withColumn("reactive_power_calculated", sqrt(pow(col("apparent_power_calculated"), 2) - pow(col("active_power_calculated"), 2))) \
                .withColumn("power_efficiency", when(col("apparent_power_calculated") > 0, col("active_power_calculated") / col("apparent_power_calculated")).otherwise(0)) \
                .withColumn("load_factor", when(col("active_power_calculated") > 0, col("consumption_kwh") / (col("active_power_calculated") * 0.001)).otherwise(0))
            
            # 4. ANOMALY DETECTION FEATURES (Configuration-Driven)
            logger.info("Creating configuration-driven anomaly detection features")
            
            # Get anomaly detection configuration
            zscore_threshold = self.transformation_config.get_anomaly_threshold("zscore_threshold", 3.0)
            window_size = self.transformation_config.get_anomaly_threshold("window_size", 20)
            consumption_anomaly_enabled = self.transformation_config.get_anomaly_threshold("consumption_anomaly", True)
            voltage_anomaly_enabled = self.transformation_config.get_anomaly_threshold("voltage_anomaly", True)
            
            # Create window specification based on configuration
            window_spec = Window.partitionBy("meter_id").orderBy("reading_timestamp").rowsBetween(-window_size//2, window_size//2)
            
            # Apply anomaly detection based on configuration
            if consumption_anomaly_enabled and voltage_anomaly_enabled:
                transformed_df = transformed_df \
                    .withColumn("consumption_mean_20", avg(col("consumption_kwh")).over(window_spec)) \
                    .withColumn("consumption_std_20", stddev(col("consumption_kwh")).over(window_spec)) \
                    .withColumn("voltage_mean_20", avg(col("voltage_v")).over(window_spec)) \
                    .withColumn("voltage_std_20", stddev(col("voltage_v")).over(window_spec)) \
                    .withColumn("consumption_zscore", when(col("consumption_std_20") > 0, (col("consumption_kwh") - col("consumption_mean_20")) / col("consumption_std_20")).otherwise(0)) \
                    .withColumn("voltage_zscore", when(col("voltage_std_20") > 0, (col("voltage_v") - col("voltage_mean_20")) / col("voltage_std_20")).otherwise(0)) \
                    .withColumn("is_consumption_anomaly", when(abs(col("consumption_zscore")) > zscore_threshold, True).otherwise(False)) \
                    .withColumn("is_voltage_anomaly", when(abs(col("voltage_zscore")) > zscore_threshold, True).otherwise(False)) \
                    .withColumn("anomaly_score", (abs(col("consumption_zscore")) + abs(col("voltage_zscore"))) / 2)
            elif consumption_anomaly_enabled:
                transformed_df = transformed_df \
                    .withColumn("consumption_mean_20", avg(col("consumption_kwh")).over(window_spec)) \
                    .withColumn("consumption_std_20", stddev(col("consumption_kwh")).over(window_spec)) \
                    .withColumn("consumption_zscore", when(col("consumption_std_20") > 0, (col("consumption_kwh") - col("consumption_mean_20")) / col("consumption_std_20")).otherwise(0)) \
                    .withColumn("is_consumption_anomaly", when(abs(col("consumption_zscore")) > zscore_threshold, True).otherwise(False)) \
                    .withColumn("is_voltage_anomaly", lit(False)) \
                    .withColumn("anomaly_score", abs(col("consumption_zscore")))
            elif voltage_anomaly_enabled:
                transformed_df = transformed_df \
                    .withColumn("voltage_mean_20", avg(col("voltage_v")).over(window_spec)) \
                    .withColumn("voltage_std_20", stddev(col("voltage_v")).over(window_spec)) \
                    .withColumn("voltage_zscore", when(col("voltage_std_20") > 0, (col("voltage_v") - col("voltage_mean_20")) / col("voltage_std_20")).otherwise(0)) \
                    .withColumn("is_voltage_anomaly", when(abs(col("voltage_zscore")) > zscore_threshold, True).otherwise(False)) \
                    .withColumn("is_consumption_anomaly", lit(False)) \
                    .withColumn("anomaly_score", abs(col("voltage_zscore")))
            else:
                # No anomaly detection enabled
                transformed_df = transformed_df \
                    .withColumn("is_consumption_anomaly", lit(False)) \
                    .withColumn("is_voltage_anomaly", lit(False)) \
                    .withColumn("anomaly_score", lit(0.0))
            
            # 5. TIME-SERIES FEATURES
            logger.info("Creating time-series features")
            window_24h = Window.partitionBy("meter_id").orderBy("reading_timestamp").rowsBetween(-24, -1)
            window_7d = Window.partitionBy("meter_id").orderBy("reading_timestamp").rowsBetween(-168, -1)  # 7 days * 24 hours
            
            transformed_df = transformed_df \
                .withColumn("consumption_lag_1h", lag(col("consumption_kwh"), 1).over(Window.partitionBy("meter_id").orderBy("reading_timestamp"))) \
                .withColumn("consumption_lag_24h", lag(col("consumption_kwh"), 24).over(Window.partitionBy("meter_id").orderBy("reading_timestamp"))) \
                .withColumn("consumption_lead_1h", lead(col("consumption_kwh"), 1).over(Window.partitionBy("meter_id").orderBy("reading_timestamp"))) \
                .withColumn("consumption_avg_24h", avg(col("consumption_kwh")).over(window_24h)) \
                .withColumn("consumption_avg_7d", avg(col("consumption_kwh")).over(window_7d)) \
                .withColumn("consumption_trend_24h", col("consumption_kwh") - col("consumption_avg_24h")) \
                .withColumn("consumption_trend_7d", col("consumption_kwh") - col("consumption_avg_7d")) \
                .withColumn("consumption_volatility_24h", stddev(col("consumption_kwh")).over(window_24h))
            
            # 6. BUSINESS LOGIC TRANSFORMATIONS (Configuration-Driven)
            logger.info("Applying configuration-driven business logic transformations")
            
            # Get business rules from configuration
            tariff_rule = self.transformation_config.get_business_rule("tariff_categories", "residential")
            consumption_rule = self.transformation_config.get_business_rule("consumption_tiers", "low")
            efficiency_rule = self.transformation_config.get_business_rule("efficiency_ratings", "excellent")
            voltage_rule = self.transformation_config.get_business_rule("voltage_categories", "low_voltage")
            
            # Apply tariff categorization
            if tariff_rule:
                residential_values = tariff_rule.get("residential", ["RESIDENTIAL", "residential", "Residential"])
                commercial_values = tariff_rule.get("commercial", ["COMMERCIAL", "commercial", "Commercial"])
                industrial_values = tariff_rule.get("industrial", ["INDUSTRIAL", "industrial", "Industrial"])
                
                transformed_df = transformed_df.withColumn("tariff_category", 
                    when(col("tariff_type").isin(residential_values), "RESIDENTIAL")
                    .when(col("tariff_type").isin(commercial_values), "COMMERCIAL")
                    .when(col("tariff_type").isin(industrial_values), "INDUSTRIAL")
                    .otherwise("UNKNOWN"))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("tariff_category", 
                    when(col("tariff_type").isin(["RESIDENTIAL", "residential", "Residential"]), "RESIDENTIAL")
                    .when(col("tariff_type").isin(["COMMERCIAL", "commercial", "Commercial"]), "COMMERCIAL")
                    .when(col("tariff_type").isin(["INDUSTRIAL", "industrial", "Industrial"]), "INDUSTRIAL")
                    .otherwise("UNKNOWN"))
            
            # Apply consumption tier categorization
            if consumption_rule:
                low_max = consumption_rule.get("max", 50)
                medium_max = self.transformation_config.get_business_rule("consumption_tiers", "medium").get("max", 200) if self.transformation_config.get_business_rule("consumption_tiers", "medium") else 200
                high_max = self.transformation_config.get_business_rule("consumption_tiers", "high").get("max", 500) if self.transformation_config.get_business_rule("consumption_tiers", "high") else 500
                
                transformed_df = transformed_df.withColumn("consumption_tier",
                    when(col("consumption_kwh") < low_max, "LOW")
                    .when(col("consumption_kwh") < medium_max, "MEDIUM")
                    .when(col("consumption_kwh") < high_max, "HIGH")
                    .otherwise("VERY_HIGH"))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("consumption_tier",
                    when(col("consumption_kwh") < 50, "LOW")
                    .when(col("consumption_kwh") < 200, "MEDIUM")
                    .when(col("consumption_kwh") < 500, "HIGH")
                    .otherwise("VERY_HIGH"))
            
            # Apply efficiency rating
            if efficiency_rule:
                excellent_min = efficiency_rule.get("min_power_factor", 0.95)
                good_min = self.transformation_config.get_business_rule("efficiency_ratings", "good").get("min_power_factor", 0.85) if self.transformation_config.get_business_rule("efficiency_ratings", "good") else 0.85
                fair_min = self.transformation_config.get_business_rule("efficiency_ratings", "fair").get("min_power_factor", 0.75) if self.transformation_config.get_business_rule("efficiency_ratings", "fair") else 0.75
                
                transformed_df = transformed_df.withColumn("efficiency_rating",
                    when(col("power_factor") >= excellent_min, "EXCELLENT")
                    .when(col("power_factor") >= good_min, "GOOD")
                    .when(col("power_factor") >= fair_min, "FAIR")
                    .otherwise("POOR"))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("efficiency_rating",
                    when(col("power_factor") >= 0.95, "EXCELLENT")
                    .when(col("power_factor") >= 0.85, "GOOD")
                    .when(col("power_factor") >= 0.75, "FAIR")
                    .otherwise("POOR"))
            
            # Apply voltage categorization
            if voltage_rule:
                low_max = voltage_rule.get("max", 200)
                normal_max = self.transformation_config.get_business_rule("voltage_categories", "normal_voltage").get("max", 250) if self.transformation_config.get_business_rule("voltage_categories", "normal_voltage") else 250
                
                transformed_df = transformed_df.withColumn("voltage_category",
                    when(col("voltage_v") < low_max, "LOW_VOLTAGE")
                    .when(col("voltage_v") > normal_max, "HIGH_VOLTAGE")
                    .otherwise("NORMAL_VOLTAGE"))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("voltage_category",
                    when(col("voltage_v") < 200, "LOW_VOLTAGE")
                    .when(col("voltage_v") > 250, "HIGH_VOLTAGE")
                    .otherwise("NORMAL_VOLTAGE"))
            
            # 7. GEOGRAPHIC AND DEMOGRAPHIC FEATURES
            logger.info("Adding geographic and demographic features")
            transformed_df = transformed_df \
                .withColumn("region_code", substring(col("meter_id"), 1, 3)) \
                .withColumn("zone_code", substring(col("meter_id"), 4, 2)) \
                .withColumn("meter_type", 
                    when(col("meter_id").rlike("^SM"), "SMART_METER")
                    .when(col("meter_id").rlike("^AM"), "ANALOG_METER")
                    .when(col("meter_id").rlike("^DM"), "DIGITAL_METER")
                    .otherwise("UNKNOWN"))
            
            # 8. PREDICTIVE FEATURES (Configuration-Driven)
            logger.info("Creating configuration-driven predictive features")
            
            # Get predictive configuration
            forecasting_config = self.transformation_config.get_predictive_config("forecasting", {})
            risk_scoring_config = self.transformation_config.get_predictive_config("risk_scoring", {})
            
            # Apply forecasting based on configuration
            if "consumption_forecast_1h" in forecasting_config:
                forecast_1h_config = forecasting_config["consumption_forecast_1h"]
                weight_24h = forecast_1h_config.get("weight_24h", 0.98)
                weight_current = forecast_1h_config.get("weight_current", 0.02)
                transformed_df = transformed_df.withColumn("consumption_forecast_1h", 
                    col("consumption_avg_24h") * weight_24h + col("consumption_kwh") * weight_current)
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("consumption_forecast_1h", col("consumption_avg_24h") * 1.02)
            
            if "consumption_forecast_24h" in forecasting_config:
                forecast_24h_config = forecasting_config["consumption_forecast_24h"]
                weight_7d = forecast_24h_config.get("weight_7d", 0.99)
                weight_current = forecast_24h_config.get("weight_current", 0.01)
                transformed_df = transformed_df.withColumn("consumption_forecast_24h", 
                    col("consumption_avg_7d") * weight_7d + col("consumption_kwh") * weight_current)
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("consumption_forecast_24h", col("consumption_avg_7d") * 1.01)
            
            # Apply maintenance risk scoring based on configuration
            if "maintenance_risk" in risk_scoring_config:
                maintenance_config = risk_scoring_config["maintenance_risk"]
                voltage_low_threshold = maintenance_config.get("voltage_low", {}).get("threshold", 200)
                voltage_low_score = maintenance_config.get("voltage_low", {}).get("score", 0.8)
                voltage_high_threshold = maintenance_config.get("voltage_high", {}).get("threshold", 250)
                voltage_high_score = maintenance_config.get("voltage_high", {}).get("score", 0.7)
                power_factor_low_threshold = maintenance_config.get("power_factor_low", {}).get("threshold", 0.7)
                power_factor_low_score = maintenance_config.get("power_factor_low", {}).get("score", 0.6)
                frequency_low_threshold = maintenance_config.get("frequency_low", {}).get("threshold", 49.5)
                frequency_low_score = maintenance_config.get("frequency_low", {}).get("score", 0.9)
                frequency_high_threshold = maintenance_config.get("frequency_high", {}).get("threshold", 50.5)
                frequency_high_score = maintenance_config.get("frequency_high", {}).get("score", 0.9)
                default_score = maintenance_config.get("default", {}).get("score", 0.1)
                
                transformed_df = transformed_df.withColumn("maintenance_risk_score",
                    when(col("voltage_v") < voltage_low_threshold, voltage_low_score)
                    .when(col("voltage_v") > voltage_high_threshold, voltage_high_score)
                    .when(col("power_factor") < power_factor_low_threshold, power_factor_low_score)
                    .when(col("frequency_hz") < frequency_low_threshold, frequency_low_score)
                    .when(col("frequency_hz") > frequency_high_threshold, frequency_high_score)
                    .otherwise(default_score))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("maintenance_risk_score",
                    when(col("voltage_v") < 200, 0.8)
                    .when(col("voltage_v") > 250, 0.7)
                    .when(col("power_factor") < 0.7, 0.6)
                    .when(col("frequency_hz") < 49.5, 0.9)
                    .when(col("frequency_hz") > 50.5, 0.9)
                    .otherwise(0.1))
            
            # Apply energy efficiency scoring (using efficiency ratings from business rules)
            efficiency_rule = self.transformation_config.get_business_rule("efficiency_ratings", "excellent")
            if efficiency_rule:
                excellent_min = efficiency_rule.get("min_power_factor", 0.95)
                good_min = self.transformation_config.get_business_rule("efficiency_ratings", "good").get("min_power_factor", 0.85) if self.transformation_config.get_business_rule("efficiency_ratings", "good") else 0.85
                fair_min = self.transformation_config.get_business_rule("efficiency_ratings", "fair").get("min_power_factor", 0.75) if self.transformation_config.get_business_rule("efficiency_ratings", "fair") else 0.75
                
                transformed_df = transformed_df.withColumn("energy_efficiency_score",
                    when(col("power_factor") >= excellent_min, 1.0)
                    .when(col("power_factor") >= good_min, 0.8)
                    .when(col("power_factor") >= fair_min, 0.6)
                    .otherwise(0.4))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("energy_efficiency_score",
                    when(col("power_factor") >= 0.95, 1.0)
                    .when(col("power_factor") >= 0.85, 0.8)
                    .when(col("power_factor") >= 0.75, 0.6)
                    .otherwise(0.4))
            
            # 9. DATA QUALITY AND METADATA
            logger.info("Adding data quality and metadata")
            transformed_df = transformed_df \
                .withColumn("processed_at", current_timestamp()) \
                .withColumn("data_source", lit("spark_etl_advanced")) \
                .withColumn("processing_version", lit("2.0.0")) \
                .withColumn("transformation_batch_id", monotonically_increasing_id()) \
                .withColumn("data_quality_score", 
                    when(col("consumption_kwh").isNull(), 0.0)
                    .when(col("voltage_v").isNull(), 0.5)
                    .when(col("current_a").isNull(), 0.7)
                    .otherwise(1.0)) \
                .withColumn("completeness_score",
                    (when(col("consumption_kwh").isNotNull(), 1).otherwise(0) +
                     when(col("voltage_v").isNotNull(), 1).otherwise(0) +
                     when(col("current_a").isNotNull(), 1).otherwise(0) +
                     when(col("power_factor").isNotNull(), 1).otherwise(0) +
                     when(col("frequency_hz").isNotNull(), 1).otherwise(0)) / 5.0)
            
            # 10. FINAL STANDARDIZATION
            logger.info("Applying final standardization")
            transformed_df = transformed_df \
                .withColumn("status", 
                    when(col("status").isin(["ACTIVE", "active", "Active"]), "ACTIVE")
                    .when(col("status").isin(["INACTIVE", "inactive", "Inactive"]), "INACTIVE")
                    .when(col("status").isin(["MAINTENANCE", "maintenance", "Maintenance"]), "MAINTENANCE")
                    .otherwise("UNKNOWN")) \
                .withColumn("quality_tier",
                    when(col("quality_tier").isin(["HIGH", "high", "High"]), "HIGH")
                    .when(col("quality_tier").isin(["MEDIUM", "medium", "Medium"]), "MEDIUM")
                    .when(col("quality_tier").isin(["LOW", "low", "Low"]), "LOW")
                    .otherwise("UNKNOWN"))
            
            logger.info(f"Advanced data transformation completed. Output records: {transformed_df.count()}")
            logger.info(f"Total columns created: {len(transformed_df.columns)}")
            return transformed_df
            
        except Exception as e:
            logger.error(f"Advanced data transformation failed: {str(e)}")
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
                "partition_columns": ["reading_timestamp"]  # Partition by date for better performance
            }
            
            # Add PostgreSQL destination if specified
            if postgres_output:
                destinations["postgresql"] = {
                    "table_name": "meter_readings",  # We'll handle smart_meters separately
                    "connection_string": postgres_output,
                    "write_mode": "append",
                    "batch_size": 1000
                }
            
            # Add Snowflake destination if specified
            if snowflake_output:
                destinations["snowflake"] = {
                    "table_name": "smart_meter_readings",
                    "schema": "PROCESSED",
                    "write_mode": "overwrite"
                }
            
            # Load to multiple destinations using DataLoader
            load_result = self.data_loader.load_to_multiple_destinations(df, destinations)
            
            # Handle PostgreSQL smart_meters table separately if needed
            if postgres_output and load_result["destinations"].get("postgresql", {}).get("status") == "success":
                try:
                    logger.info("Loading smart_meters table to PostgreSQL")
                    
                    # Create smart_meters table with unique meters
                    smart_meters_df = df.select(
                        "meter_id",
                        "meter_location_lat",
                        "meter_location_lon", 
                        "meter_address",
                        "tariff_type"
                    ).distinct()
                    
                    # Add required columns for smart_meters table
                    smart_meters_df = smart_meters_df.withColumn("manufacturer", lit("Unknown")) \
                        .withColumn("model", lit("Unknown")) \
                        .withColumn("status", lit("ACTIVE")) \
                        .withColumn("quality_tier", lit("UNKNOWN")) \
                        .withColumn("installed_at", current_timestamp()) \
                        .withColumn("created_at", current_timestamp()) \
                        .withColumn("updated_at", current_timestamp())
                    
                    # Load smart_meters table
                    smart_meters_result = self.data_loader.load_to_postgresql(
                        df=smart_meters_df,
                        table_name="smart_meters",
                        connection_string=postgres_output,
                        write_mode="append"
                    )
                    
                    # Update results to include smart_meters table
                    if "postgresql" in load_result["destinations"]:
                        load_result["destinations"]["postgresql"]["tables"] = ["smart_meters", "meter_readings"]
                        load_result["destinations"]["postgresql"]["smart_meters_records"] = smart_meters_result["records_loaded"]
                    
                    logger.info(f"Successfully loaded {smart_meters_result['records_loaded']} smart meters to PostgreSQL")
                    
                except Exception as e:
                    logger.error(f"Smart meters table loading failed: {str(e)}")
                    # Don't fail the entire operation, just log the error
            
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
            logger.info("Starting Smart Meter ETL job")
            
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
                "job_name": "smart_meter_etl",
                "status": "success",
                "source_path": source_path,
                "target_path": target_path,
                "format": format,
                "input_records": raw_df.count(),
                "output_records": load_result["records_loaded"],
                "data_quality": quality_report,
                "destinations": load_result["destinations"],
                "processing_timestamp": str(current_timestamp().cast("string"))
            }
            
            logger.info("Smart Meter ETL job completed successfully")
            return etl_result
            
        except Exception as e:
            logger.error(f"ETL job failed: {str(e)}")
            return {
                "job_name": "smart_meter_etl",
                "status": "failed",
                "error": str(e),
                "source_path": source_path,
                "target_path": target_path
            }


def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description="Smart Meter ETL Job")
    parser.add_argument("--input-path", help="Input data path (optional, uses configured source)")
    parser.add_argument("--output-path", required=True, help="Output data path")
    parser.add_argument("--output-format", default="delta", help="Output format (delta/parquet)")
    parser.add_argument("--postgres-output", help="PostgreSQL connection string")
    parser.add_argument("--snowflake-output", help="Snowflake connection string")
    parser.add_argument("--batch-id", help="Batch ID for processing")
    parser.add_argument("--data-type", default="smart_meter", help="Data type")
    parser.add_argument("--source-type", help="Data source type (csv, api, kafka, database, s3, snowflake)")
    parser.add_argument("--environment", default="development", help="Environment (development, staging, production)")
    
    args = parser.parse_args()
    
    # Initialize Spark
    config = SparkETLConfig()
    spark = SparkSession.builder \
        .appName("SmartMeterETL") \
        .config(conf=config.get_spark_conf()) \
        .getOrCreate()
    
    try:
        # Create ETL job with environment
        etl_job = SmartMeterETLJob(spark, config, environment=args.environment)
        
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
    
    def get_processing_stats(self, df: DataFrame) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            stats = df.select(
                count("*").alias("total_records"),
                count(when(col("status") == "ACTIVE", 1)).alias("active_meters"),
                count(when(col("status") == "INACTIVE", 1)).alias("inactive_meters"),
                count(when(col("quality_tier") == "HIGH", 1)).alias("high_quality_meters"),
                count(when(col("quality_tier") == "MEDIUM", 1)).alias("medium_quality_meters"),
                count(when(col("quality_tier") == "LOW", 1)).alias("low_quality_meters"),
                avg("latitude").alias("avg_latitude"),
                avg("longitude").alias("avg_longitude")
            ).collect()[0]
            
            return {
                "total_records": stats["total_records"],
                "active_meters": stats["active_meters"],
                "inactive_meters": stats["inactive_meters"],
                "high_quality_meters": stats["high_quality_meters"],
                "medium_quality_meters": stats["medium_quality_meters"],
                "low_quality_meters": stats["low_quality_meters"],
                "avg_latitude": stats["avg_latitude"],
                "avg_longitude": stats["avg_longitude"]
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing stats: {str(e)}")
            return {"error": str(e)}
