"""
Grid Operator ETL Job
Apache Spark ETL job for processing grid operator data
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


class GridOperatorETLJob:
    """
    Grid Operator ETL Job
    
    Processes grid operator data including:
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
        self.grid_operator_config = self.data_sources_config.grid_operators
        
        # Initialize data source manager for multi-source support
        self.data_source_manager = DataSourceManager(
            spark=spark,
            data_source_config=self.grid_operator_config.__dict__,
            environment=environment
        )
        
        # Initialize transformation configuration service
        self.transformation_config = get_transformation_config_service(environment).get_grid_operator_config()
        
        # Initialize data loader for proper client integration
        self.data_loader = DataLoader(spark, environment)
        
        # Use configuration from YAML for expected schema
        self.expected_schema = {
            "required_columns": self.grid_operator_config.validation.get("required_columns", []),
            "data_types": self.grid_operator_config.validation.get("data_types", {}),
            "validation_rules": self.grid_operator_config.validation.get("value_ranges", {}),
            "default_values": {
                "contact_email": "unknown@metrify.com",
                "region": "Unknown",
                "grid_capacity_mw": 0.0,
                "voltage_level_kv": 0.0,
                "coverage_area_km2": 0.0,
                "metadata": "{}"
            }
        }
    
    def extract_data(self, source_path: str = None, source_type: str = None) -> DataFrame:
        """Extract grid operator data from any configured data source"""
        try:
            # Use primary source if not specified
            if source_type is None:
                source_type = self.data_source_manager.primary_source
            
            logger.info(f"Extracting grid operator data using {source_type} source")
            
            # Create data source instance
            data_source = self.data_source_manager.create_source(source_type)
            
            # Determine source path/identifier
            if source_path is None:
                # Use configured source path based on type
                if source_type == "csv":
                    data_root = self.grid_operator_config.data_root
                    status_file = self.grid_operator_config.readings_file  # This maps to status_file in YAML
                    source_path = f"{data_root}/{status_file}"
                elif source_type == "api":
                    source_path = "status_endpoint"
                elif source_type == "kafka":
                    source_path = "status_topic"
                elif source_type == "database":
                    source_path = "status_table"
                elif source_type == "s3":
                    source_path = "status_file"
                elif source_type == "snowflake":
                    source_path = "status_table"
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
        """Validate grid operator data quality"""
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
        """Transform grid operator data with advanced grid analytics and time-series features"""
        try:
            logger.info("Starting advanced grid operator data transformation")
            
            # 1. BASIC DATA STANDARDIZATION
            logger.info("Applying basic data standardization")
            transformed_df = df \
                .withColumn("operator_id", upper(trim(col("operator_id")))) \
                .withColumn("status_timestamp", to_timestamp(col("status_timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS")) \
                .withColumn("grid_frequency_hz", col("grid_frequency_hz").cast("double")) \
                .withColumn("grid_voltage_kv", col("grid_voltage_kv").cast("double")) \
                .withColumn("grid_load_mw", col("grid_load_mw").cast("double")) \
                .withColumn("power_generation_mw", col("power_generation_mw").cast("double")) \
                .withColumn("grid_capacity_mw", col("grid_capacity_mw").cast("double"))
            
            # 2. TEMPORAL FEATURE ENGINEERING
            logger.info("Creating temporal features")
            transformed_df = transformed_df \
                .withColumn("year", year(col("status_timestamp"))) \
                .withColumn("month", month(col("status_timestamp"))) \
                .withColumn("day", dayofmonth(col("status_timestamp"))) \
                .withColumn("hour", hour(col("status_timestamp"))) \
                .withColumn("day_of_week", dayofweek(col("status_timestamp"))) \
                .withColumn("day_of_year", dayofyear(col("status_timestamp"))) \
                .withColumn("week_of_year", weekofyear(col("status_timestamp"))) \
                .withColumn("quarter", quarter(col("status_timestamp"))) \
                .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), True).otherwise(False)) \
                .withColumn("is_peak_demand_hour", when(col("hour").isin([18, 19, 20, 21]), True).otherwise(False)) \
                .withColumn("is_off_peak_hour", when((col("hour") >= 22) | (col("hour") <= 6), True).otherwise(False)) \
                .withColumn("is_business_hour", when((col("hour") >= 9) & (col("hour") <= 17), True).otherwise(False))
            
            # 3. ADVANCED GRID ANALYTICS
            logger.info("Calculating advanced grid metrics")
            transformed_df = transformed_df \
                .withColumn("grid_utilization", when(col("grid_capacity_mw") > 0, col("grid_load_mw") / col("grid_capacity_mw")).otherwise(0)) \
                .withColumn("generation_efficiency", when(col("grid_load_mw") > 0, col("power_generation_mw") / col("grid_load_mw")).otherwise(0)) \
                .withColumn("frequency_deviation", abs(col("grid_frequency_hz") - 50.0)) \
                .withColumn("voltage_deviation", abs(col("grid_voltage_kv") - 230.0)) \
                .withColumn("power_factor", when(col("grid_voltage_kv") > 0, col("grid_load_mw") / (col("grid_voltage_kv") * 0.001)).otherwise(0)) \
                .withColumn("grid_stability_index", 1.0 - (col("frequency_deviation") / 0.5) - (col("voltage_deviation") / 20.0))
            
            # 4. GRID STABILITY AND ALERT ANALYSIS (Configuration-Driven)
            logger.info("Creating configuration-driven grid stability features")
            
            # Get grid analytics configuration
            grid_analytics = self.transformation_config.get_predictive_config("grid_analytics", {})
            frequency_target = grid_analytics.get("frequency_target", 50.0)
            voltage_target = grid_analytics.get("voltage_target", 230.0)
            stability_thresholds = grid_analytics.get("stability_thresholds", {})
            utilization_thresholds = grid_analytics.get("utilization_thresholds", {})
            
            # Get business rules for stability scores
            stability_rules = self.transformation_config.get_business_rule("grid_stability_scores", "excellent")
            alert_rules = self.transformation_config.get_business_rule("alert_levels", "critical")
            
            # Apply grid stability scoring based on configuration
            if stability_rules:
                excellent_rule = self.transformation_config.get_business_rule("grid_stability_scores", "excellent")
                good_rule = self.transformation_config.get_business_rule("grid_stability_scores", "good")
                fair_rule = self.transformation_config.get_business_rule("grid_stability_scores", "fair")
                poor_rule = self.transformation_config.get_business_rule("grid_stability_scores", "poor")
                
                if excellent_rule and good_rule and fair_rule and poor_rule:
                    transformed_df = transformed_df.withColumn("grid_stability_score", 
                        when(col("grid_frequency_hz").between(excellent_rule.get("frequency_min", 49.9), excellent_rule.get("frequency_max", 50.1)), excellent_rule.get("score", 1.0))
                        .when(col("grid_frequency_hz").between(good_rule.get("frequency_min", 49.8), good_rule.get("frequency_max", 50.2)), good_rule.get("score", 0.8))
                        .when(col("grid_frequency_hz").between(fair_rule.get("frequency_min", 49.5), fair_rule.get("frequency_max", 50.5)), fair_rule.get("score", 0.6))
                        .otherwise(poor_rule.get("score", 0.2)))
                else:
                    # Fallback to hardcoded values
                    transformed_df = transformed_df.withColumn("grid_stability_score", 
                        when(col("grid_frequency_hz").between(49.9, 50.1), 1.0)
                        .when(col("grid_frequency_hz").between(49.8, 50.2), 0.8)
                        .when(col("grid_frequency_hz").between(49.5, 50.5), 0.6)
                        .otherwise(0.2))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("grid_stability_score", 
                    when(col("grid_frequency_hz").between(49.9, 50.1), 1.0)
                    .when(col("grid_frequency_hz").between(49.8, 50.2), 0.8)
                    .when(col("grid_frequency_hz").between(49.5, 50.5), 0.6)
                    .otherwise(0.2))
            
            # Apply stability flags based on configuration
            frequency_deviation_max = stability_thresholds.get("frequency_deviation_max", 0.5)
            voltage_deviation_max = stability_thresholds.get("voltage_deviation_max", 20.0)
            
            transformed_df = transformed_df \
                .withColumn("is_frequency_stable", 
                    when(col("frequency_deviation") <= frequency_deviation_max, True)
                    .otherwise(False)) \
                .withColumn("is_voltage_stable", 
                    when(col("voltage_deviation") <= voltage_deviation_max, True)
                    .otherwise(False))
            
            # Apply alert levels based on configuration
            if alert_rules:
                critical_rule = self.transformation_config.get_business_rule("alert_levels", "critical")
                warning_rule = self.transformation_config.get_business_rule("alert_levels", "warning")
                high_load_rule = self.transformation_config.get_business_rule("alert_levels", "high_load")
                normal_rule = self.transformation_config.get_business_rule("alert_levels", "normal")
                
                if critical_rule and warning_rule and high_load_rule and normal_rule:
                    transformed_df = transformed_df.withColumn("alert_level",
                        when((col("grid_frequency_hz") < critical_rule.get("frequency_max", 49.5)) | (col("grid_frequency_hz") > critical_rule.get("frequency_max_high", 50.5)), "CRITICAL")
                        .when((col("grid_frequency_hz") < warning_rule.get("frequency_max", 49.8)) | (col("grid_frequency_hz") > warning_rule.get("frequency_max_high", 50.2)), "WARNING")
                        .when((col("grid_voltage_kv") < 200) | (col("grid_voltage_kv") > 250), "WARNING")
                        .when(col("grid_utilization") > high_load_rule.get("utilization_min", 0.95), "HIGH_LOAD")
                        .otherwise("NORMAL"))
                else:
                    # Fallback to hardcoded values
                    transformed_df = transformed_df.withColumn("alert_level",
                        when((col("grid_frequency_hz") < 49.5) | (col("grid_frequency_hz") > 50.5), "CRITICAL")
                        .when((col("grid_frequency_hz") < 49.8) | (col("grid_frequency_hz") > 50.2), "WARNING")
                        .when((col("grid_voltage_kv") < 200) | (col("grid_voltage_kv") > 250), "WARNING")
                        .when(col("grid_utilization") > 0.95, "HIGH_LOAD")
                        .otherwise("NORMAL"))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("alert_level",
                    when((col("grid_frequency_hz") < 49.5) | (col("grid_frequency_hz") > 50.5), "CRITICAL")
                    .when((col("grid_frequency_hz") < 49.8) | (col("grid_frequency_hz") > 50.2), "WARNING")
                    .when((col("grid_voltage_kv") < 200) | (col("grid_voltage_kv") > 250), "WARNING")
                    .when(col("grid_utilization") > 0.95, "HIGH_LOAD")
                    .otherwise("NORMAL"))
            
            # Apply grid health scoring based on configuration
            normal_utilization = utilization_thresholds.get("normal", 0.8)
            transformed_df = transformed_df.withColumn("grid_health_score",
                (col("grid_stability_score") * 0.4 +
                 when(col("is_voltage_stable"), 1.0).otherwise(0.0) * 0.3 +
                 when(col("grid_utilization") < normal_utilization, 1.0).otherwise(0.0) * 0.3))
            
            # 5. TIME-SERIES FEATURES
            logger.info("Creating time-series features")
            window_1h = Window.partitionBy("operator_id").orderBy("status_timestamp").rowsBetween(-1, -1)
            window_24h = Window.partitionBy("operator_id").orderBy("status_timestamp").rowsBetween(-24, -1)
            window_7d = Window.partitionBy("operator_id").orderBy("status_timestamp").rowsBetween(-168, -1)
            
            transformed_df = transformed_df \
                .withColumn("frequency_lag_1h", lag(col("grid_frequency_hz"), 1).over(Window.partitionBy("operator_id").orderBy("status_timestamp"))) \
                .withColumn("load_lag_1h", lag(col("grid_load_mw"), 1).over(Window.partitionBy("operator_id").orderBy("status_timestamp"))) \
                .withColumn("frequency_avg_24h", avg(col("grid_frequency_hz")).over(window_24h)) \
                .withColumn("load_avg_24h", avg(col("grid_load_mw")).over(window_24h)) \
                .withColumn("frequency_avg_7d", avg(col("grid_frequency_hz")).over(window_7d)) \
                .withColumn("load_avg_7d", avg(col("grid_load_mw")).over(window_7d)) \
                .withColumn("frequency_trend_24h", col("grid_frequency_hz") - col("frequency_avg_24h")) \
                .withColumn("load_trend_24h", col("grid_load_mw") - col("load_avg_24h")) \
                .withColumn("frequency_volatility_24h", stddev(col("grid_frequency_hz")).over(window_24h)) \
                .withColumn("load_volatility_24h", stddev(col("grid_load_mw")).over(window_24h))
            
            # 6. ANOMALY DETECTION FEATURES
            logger.info("Creating anomaly detection features")
            window_anomaly = Window.partitionBy("operator_id").orderBy("status_timestamp").rowsBetween(-10, 10)
            
            transformed_df = transformed_df \
                .withColumn("frequency_mean_20", avg(col("grid_frequency_hz")).over(window_anomaly)) \
                .withColumn("frequency_std_20", stddev(col("grid_frequency_hz")).over(window_anomaly)) \
                .withColumn("load_mean_20", avg(col("grid_load_mw")).over(window_anomaly)) \
                .withColumn("load_std_20", stddev(col("grid_load_mw")).over(window_anomaly)) \
                .withColumn("frequency_zscore", when(col("frequency_std_20") > 0, (col("grid_frequency_hz") - col("frequency_mean_20")) / col("frequency_std_20")).otherwise(0)) \
                .withColumn("load_zscore", when(col("load_std_20") > 0, (col("grid_load_mw") - col("load_mean_20")) / col("load_std_20")).otherwise(0)) \
                .withColumn("is_frequency_anomaly", when(abs(col("frequency_zscore")) > 3, True).otherwise(False)) \
                .withColumn("is_load_anomaly", when(abs(col("load_zscore")) > 3, True).otherwise(False)) \
                .withColumn("grid_anomaly_score", (abs(col("frequency_zscore")) + abs(col("load_zscore"))) / 2)
            
            # 7. BUSINESS LOGIC TRANSFORMATIONS (Configuration-Driven)
            logger.info("Applying configuration-driven business logic transformations")
            
            # Apply operator region categorization (hardcoded for now as it's based on ID patterns)
            transformed_df = transformed_df.withColumn("operator_region", 
                when(col("operator_id").rlike("^NORTH"), "NORTH")
                .when(col("operator_id").rlike("^SOUTH"), "SOUTH")
                .when(col("operator_id").rlike("^EAST"), "EAST")
                .when(col("operator_id").rlike("^WEST"), "WEST")
                .otherwise("CENTRAL"))
            
            # Apply load categorization based on configuration
            load_rule = self.transformation_config.get_business_rule("load_categories", "low_load")
            if load_rule:
                low_max = load_rule.get("max", 1000)
                medium_max = self.transformation_config.get_business_rule("load_categories", "medium_load").get("max", 5000) if self.transformation_config.get_business_rule("load_categories", "medium_load") else 5000
                high_max = self.transformation_config.get_business_rule("load_categories", "high_load").get("max", 10000) if self.transformation_config.get_business_rule("load_categories", "high_load") else 10000
                
                transformed_df = transformed_df.withColumn("load_category",
                    when(col("grid_load_mw") < low_max, "LOW_LOAD")
                    .when(col("grid_load_mw") < medium_max, "MEDIUM_LOAD")
                    .when(col("grid_load_mw") < high_max, "HIGH_LOAD")
                    .otherwise("VERY_HIGH_LOAD"))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("load_category",
                    when(col("grid_load_mw") < 1000, "LOW_LOAD")
                    .when(col("grid_load_mw") < 5000, "MEDIUM_LOAD")
                    .when(col("grid_load_mw") < 10000, "HIGH_LOAD")
                    .otherwise("VERY_HIGH_LOAD"))
            
            # Apply generation categorization based on configuration
            generation_rule = self.transformation_config.get_business_rule("generation_categories", "low_generation")
            if generation_rule:
                low_max = generation_rule.get("max", 500)
                medium_max = self.transformation_config.get_business_rule("generation_categories", "medium_generation").get("max", 2000) if self.transformation_config.get_business_rule("generation_categories", "medium_generation") else 2000
                high_max = self.transformation_config.get_business_rule("generation_categories", "high_generation").get("max", 5000) if self.transformation_config.get_business_rule("generation_categories", "high_generation") else 5000
                
                transformed_df = transformed_df.withColumn("generation_category",
                    when(col("power_generation_mw") < low_max, "LOW_GENERATION")
                    .when(col("power_generation_mw") < medium_max, "MEDIUM_GENERATION")
                    .when(col("power_generation_mw") < high_max, "HIGH_GENERATION")
                    .otherwise("VERY_HIGH_GENERATION"))
            else:
                # Fallback to hardcoded values
                transformed_df = transformed_df.withColumn("generation_category",
                    when(col("power_generation_mw") < 500, "LOW_GENERATION")
                    .when(col("power_generation_mw") < 2000, "MEDIUM_GENERATION")
                    .when(col("power_generation_mw") < 5000, "HIGH_GENERATION")
                    .otherwise("VERY_HIGH_GENERATION"))
            
            # Apply grid performance rating based on configuration
            transformed_df = transformed_df.withColumn("grid_performance_rating",
                when(col("grid_health_score") >= 0.9, "EXCELLENT")
                .when(col("grid_health_score") >= 0.7, "GOOD")
                .when(col("grid_health_score") >= 0.5, "FAIR")
                .otherwise("POOR"))
            
            # 8. PREDICTIVE FEATURES
            logger.info("Creating predictive features")
            transformed_df = transformed_df \
                .withColumn("frequency_forecast_1h", col("frequency_avg_24h") * 0.99 + col("grid_frequency_hz") * 0.01) \
                .withColumn("load_forecast_1h", col("load_avg_24h") * 1.01) \
                .withColumn("load_forecast_24h", col("load_avg_7d") * 1.02) \
                .withColumn("maintenance_risk_score",
                    when(col("grid_frequency_hz") < 49.5, 0.9)
                    .when(col("grid_frequency_hz") > 50.5, 0.9)
                    .when(col("grid_voltage_kv") < 200, 0.8)
                    .when(col("grid_voltage_kv") > 250, 0.8)
                    .when(col("grid_utilization") > 0.95, 0.7)
                    .when(col("frequency_volatility_24h") > 0.1, 0.6)
                    .otherwise(0.1)) \
                .withColumn("grid_reliability_score",
                    when(col("grid_health_score") >= 0.9, 1.0)
                    .when(col("grid_health_score") >= 0.7, 0.8)
                    .when(col("grid_health_score") >= 0.5, 0.6)
                    .otherwise(0.4))
            
            # 9. DATA QUALITY AND METADATA
            logger.info("Adding data quality and metadata")
            transformed_df = transformed_df \
                .withColumn("processed_at", current_timestamp()) \
                .withColumn("data_source", lit("spark_etl_advanced")) \
                .withColumn("processing_version", lit("2.0.0")) \
                .withColumn("transformation_batch_id", monotonically_increasing_id()) \
                .withColumn("data_quality_score", 
                    when(col("grid_frequency_hz").isNull(), 0.0)
                    .when(col("grid_voltage_kv").isNull(), 0.5)
                    .when(col("grid_load_mw").isNull(), 0.7)
                    .otherwise(1.0)) \
                .withColumn("completeness_score",
                    (when(col("grid_frequency_hz").isNotNull(), 1).otherwise(0) +
                     when(col("grid_voltage_kv").isNotNull(), 1).otherwise(0) +
                     when(col("grid_load_mw").isNotNull(), 1).otherwise(0) +
                     when(col("power_generation_mw").isNotNull(), 1).otherwise(0)) / 4.0)
            
            logger.info(f"Advanced grid operator transformation completed. Output records: {transformed_df.count()}")
            logger.info(f"Total columns created: {len(transformed_df.columns)}")
            return transformed_df
            
        except Exception as e:
            logger.error(f"Advanced grid operator transformation failed: {str(e)}")
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
                "partition_columns": ["status_timestamp"]  # Partition by timestamp for better performance
            }
            
            # Add PostgreSQL destination if specified
            if postgres_output:
                destinations["postgresql"] = {
                    "table_name": "grid_status",
                    "connection_string": postgres_output,
                    "write_mode": "append",
                    "batch_size": 1000
                }
            
            # Add Snowflake destination if specified
            if snowflake_output:
                destinations["snowflake"] = {
                    "table_name": "grid_operator_data",
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
            logger.info("Starting Grid Operator ETL job")
            
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
                "job_name": "grid_operator_etl",
                "status": "success",
                "source_path": source_path,
                "target_path": target_path,
                "records_processed": raw_df.count(),
                "data_quality": quality_report,
                "destinations": load_result["destinations"],
                "processing_timestamp": str(current_timestamp().cast("string"))
            }
            
            logger.info("Grid Operator ETL job completed successfully")
            return etl_result
            
        except Exception as e:
            logger.error(f"ETL job failed: {str(e)}")
            return {
                "job_name": "grid_operator_etl",
                "status": "failed",
                "error": str(e),
                "source_path": source_path,
                "target_path": target_path
            }


def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description="Grid Operator ETL Job")
    parser.add_argument("--input-path", help="Input data path (optional, uses configured source)")
    parser.add_argument("--output-path", required=True, help="Output data path")
    parser.add_argument("--output-format", default="delta", help="Output format (delta/parquet)")
    parser.add_argument("--postgres-output", help="PostgreSQL connection string")
    parser.add_argument("--snowflake-output", help="Snowflake connection string")
    parser.add_argument("--batch-id", help="Batch ID for processing")
    parser.add_argument("--data-type", default="grid_operator", help="Data type")
    parser.add_argument("--source-type", help="Data source type (csv, api, kafka, database, s3, snowflake)")
    parser.add_argument("--environment", default="development", help="Environment (development, staging, production)")
    
    args = parser.parse_args()
    
    # Initialize Spark
    config = SparkETLConfig()
    spark = SparkSession.builder \
        .appName("GridOperatorETL") \
        .config(conf=config.get_spark_conf()) \
        .getOrCreate()
    
    try:
        # Create ETL job with environment
        etl_job = GridOperatorETLJob(spark, config, environment=args.environment)
        
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