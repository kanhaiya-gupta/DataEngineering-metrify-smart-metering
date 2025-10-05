"""
Data Quality Utilities for Spark ETL
Advanced data validation and quality checks using Spark
"""

from typing import Dict, Any, List, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, count, sum as spark_sum, avg, min as spark_min, max as spark_max
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
import logging

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Advanced data quality validation using Spark"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.validation_results = {}
    
    def validate_schema(self, df: DataFrame, expected_schema: StructType) -> Dict[str, Any]:
        """Validate DataFrame schema against expected schema"""
        try:
            actual_schema = df.schema
            schema_matches = actual_schema == expected_schema
            
            result = {
                "schema_valid": schema_matches,
                "expected_schema": str(expected_schema),
                "actual_schema": str(actual_schema),
                "missing_fields": [],
                "extra_fields": [],
                "type_mismatches": []
            }
            
            if not schema_matches:
                # Find missing fields
                expected_fields = {field.name: field.dataType for field in expected_schema.fields}
                actual_fields = {field.name: field.dataType for field in actual_schema.fields}
                
                result["missing_fields"] = list(set(expected_fields.keys()) - set(actual_fields.keys()))
                result["extra_fields"] = list(set(actual_fields.keys()) - set(expected_fields.keys()))
                
                # Find type mismatches
                for field_name in set(expected_fields.keys()) & set(actual_fields.keys()):
                    if expected_fields[field_name] != actual_fields[field_name]:
                        result["type_mismatches"].append({
                            "field": field_name,
                            "expected": str(expected_fields[field_name]),
                            "actual": str(actual_fields[field_name])
                        })
            
            self.validation_results["schema"] = result
            return result
            
        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}")
            return {"schema_valid": False, "error": str(e)}
    
    def validate_completeness(self, df: DataFrame, required_fields: List[str]) -> Dict[str, Any]:
        """Validate data completeness for required fields"""
        try:
            total_rows = df.count()
            completeness_results = {}
            
            for field in required_fields:
                if field in df.columns:
                    null_count = df.filter(col(field).isNull() | isnan(col(field))).count()
                    completeness_pct = ((total_rows - null_count) / total_rows) * 100 if total_rows > 0 else 0
                    
                    completeness_results[field] = {
                        "total_rows": total_rows,
                        "null_count": null_count,
                        "completeness_percentage": completeness_pct,
                        "is_complete": completeness_pct >= 95.0  # 95% threshold
                    }
                else:
                    completeness_results[field] = {
                        "error": f"Field '{field}' not found in DataFrame"
                    }
            
            self.validation_results["completeness"] = completeness_results
            return completeness_results
            
        except Exception as e:
            logger.error(f"Completeness validation failed: {str(e)}")
            return {"error": str(e)}
    
    def validate_ranges(self, df: DataFrame, range_checks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate data ranges for numeric fields"""
        try:
            range_results = {}
            
            for field, constraints in range_checks.items():
                if field in df.columns:
                    stats = df.select(
                        count(col(field)).alias("count"),
                        spark_min(col(field)).alias("min"),
                        spark_max(col(field)).alias("max"),
                        avg(col(field)).alias("avg")
                    ).collect()[0]
                    
                    min_val = constraints.get("min")
                    max_val = constraints.get("max")
                    
                    out_of_range_count = 0
                    if min_val is not None:
                        out_of_range_count += df.filter(col(field) < min_val).count()
                    if max_val is not None:
                        out_of_range_count += df.filter(col(field) > max_val).count()
                    
                    range_results[field] = {
                        "count": stats["count"],
                        "min_value": stats["min"],
                        "max_value": stats["max"],
                        "avg_value": stats["avg"],
                        "out_of_range_count": out_of_range_count,
                        "range_valid": out_of_range_count == 0,
                        "constraints": constraints
                    }
                else:
                    range_results[field] = {"error": f"Field '{field}' not found in DataFrame"}
            
            self.validation_results["ranges"] = range_results
            return range_results
            
        except Exception as e:
            logger.error(f"Range validation failed: {str(e)}")
            return {"error": str(e)}
    
    def detect_duplicates(self, df: DataFrame, key_columns: List[str]) -> Dict[str, Any]:
        """Detect duplicate records based on key columns"""
        try:
            total_rows = df.count()
            unique_rows = df.dropDuplicates(key_columns).count()
            duplicate_count = total_rows - unique_rows
            
            result = {
                "total_rows": total_rows,
                "unique_rows": unique_rows,
                "duplicate_count": duplicate_count,
                "duplicate_percentage": (duplicate_count / total_rows) * 100 if total_rows > 0 else 0,
                "has_duplicates": duplicate_count > 0,
                "key_columns": key_columns
            }
            
            self.validation_results["duplicates"] = result
            return result
            
        except Exception as e:
            logger.error(f"Duplicate detection failed: {str(e)}")
            return {"error": str(e)}
    
    def validate_temporal_consistency(self, df: DataFrame, timestamp_field: str) -> Dict[str, Any]:
        """Validate temporal consistency of timestamp data"""
        try:
            if timestamp_field not in df.columns:
                return {"error": f"Timestamp field '{timestamp_field}' not found"}
            
            # Get timestamp statistics
            timestamp_stats = df.select(
                spark_min(col(timestamp_field)).alias("earliest"),
                spark_max(col(timestamp_field)).alias("latest"),
                count(col(timestamp_field)).alias("count")
            ).collect()[0]
            
            # Check for future timestamps (anomalies)
            from pyspark.sql.functions import current_timestamp
            future_count = df.filter(col(timestamp_field) > current_timestamp()).count()
            
            result = {
                "earliest_timestamp": timestamp_stats["earliest"],
                "latest_timestamp": timestamp_stats["latest"],
                "total_timestamps": timestamp_stats["count"],
                "future_timestamps": future_count,
                "has_future_timestamps": future_count > 0,
                "temporal_valid": future_count == 0
            }
            
            self.validation_results["temporal"] = result
            return result
            
        except Exception as e:
            logger.error(f"Temporal validation failed: {str(e)}")
            return {"error": str(e)}
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for result in self.validation_results.values() 
                               if isinstance(result, dict) and result.get("error") is None)
        
        quality_score = (passed_validations / total_validations) * 100 if total_validations > 0 else 0
        
        return {
            "overall_quality_score": quality_score,
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "failed_validations": total_validations - passed_validations,
            "validation_details": self.validation_results,
            "quality_status": "PASS" if quality_score >= 95.0 else "FAIL"
        }
    
    def clean_data(self, df: DataFrame, cleaning_rules: Dict[str, Any]) -> DataFrame:
        """Apply data cleaning rules to DataFrame"""
        try:
            cleaned_df = df
            
            # Remove duplicates if specified
            if cleaning_rules.get("remove_duplicates", False):
                key_columns = cleaning_rules.get("duplicate_key_columns", [])
                if key_columns:
                    cleaned_df = cleaned_df.dropDuplicates(key_columns)
            
            # Handle null values
            null_handling = cleaning_rules.get("null_handling", {})
            for field, strategy in null_handling.items():
                if field in cleaned_df.columns:
                    if strategy == "drop":
                        cleaned_df = cleaned_df.filter(col(field).isNotNull())
                    elif strategy == "fill_default":
                        default_value = cleaning_rules.get("default_values", {}).get(field, 0)
                        cleaned_df = cleaned_df.fillna({field: default_value})
            
            # Remove outliers if specified
            outlier_removal = cleaning_rules.get("remove_outliers", {})
            for field, constraints in outlier_removal.items():
                if field in cleaned_df.columns:
                    min_val = constraints.get("min")
                    max_val = constraints.get("max")
                    if min_val is not None:
                        cleaned_df = cleaned_df.filter(col(field) >= min_val)
                    if max_val is not None:
                        cleaned_df = cleaned_df.filter(col(field) <= max_val)
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            return df
