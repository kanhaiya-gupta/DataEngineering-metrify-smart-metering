"""
Quality Assessor
ML-based data quality assessment and scoring
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Data quality dimensions"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"

class QualityLevel(Enum):
    """Quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class QualityMetric:
    """Represents a quality metric"""
    dimension: QualityDimension
    value: float
    threshold: float
    level: QualityLevel
    description: str
    recommendations: List[str]

@dataclass
class QualityScore:
    """Represents an overall quality score"""
    overall_score: float
    dimension_scores: Dict[QualityDimension, float]
    level: QualityLevel
    assessment_date: datetime
    data_size: int
    issues_found: int

class QualityAssessor:
    """
    ML-based data quality assessment and scoring
    """
    
    def __init__(self):
        self.quality_thresholds = {
            QualityDimension.COMPLETENESS: 0.95,
            QualityDimension.ACCURACY: 0.90,
            QualityDimension.CONSISTENCY: 0.85,
            QualityDimension.VALIDITY: 0.90,
            QualityDimension.UNIQUENESS: 0.95,
            QualityDimension.TIMELINESS: 0.80
        }
        
        logger.info("QualityAssessor initialized")
    
    def assess_data_quality(self, 
                          data: pd.DataFrame,
                          schema: Dict[str, Any] = None,
                          business_rules: List[Dict[str, Any]] = None) -> QualityScore:
        """Assess overall data quality"""
        try:
            if data.empty:
                return self._create_empty_quality_score()
            
            dimension_scores = {}
            all_issues = []
            
            # Assess each quality dimension
            for dimension in QualityDimension:
                metric = self._assess_dimension(data, dimension, schema, business_rules)
                dimension_scores[dimension] = metric.value
                all_issues.extend(metric.recommendations)
            
            # Calculate overall score
            overall_score = np.mean(list(dimension_scores.values()))
            
            # Determine quality level
            level = self._determine_quality_level(overall_score)
            
            quality_score = QualityScore(
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                level=level,
                assessment_date=datetime.now(),
                data_size=len(data),
                issues_found=len(all_issues)
            )
            
            logger.info(f"Quality assessment completed: {overall_score:.3f} ({level.value})")
            return quality_score
            
        except Exception as e:
            logger.error(f"Failed to assess data quality: {str(e)}")
            return self._create_error_quality_score(str(e))
    
    def _assess_dimension(self, 
                         data: pd.DataFrame,
                         dimension: QualityDimension,
                         schema: Dict[str, Any] = None,
                         business_rules: List[Dict[str, Any]] = None) -> QualityMetric:
        """Assess a specific quality dimension"""
        try:
            if dimension == QualityDimension.COMPLETENESS:
                return self._assess_completeness(data, schema)
            elif dimension == QualityDimension.ACCURACY:
                return self._assess_accuracy(data, schema, business_rules)
            elif dimension == QualityDimension.CONSISTENCY:
                return self._assess_consistency(data, schema)
            elif dimension == QualityDimension.VALIDITY:
                return self._assess_validity(data, schema)
            elif dimension == QualityDimension.UNIQUENESS:
                return self._assess_uniqueness(data, schema)
            elif dimension == QualityDimension.TIMELINESS:
                return self._assess_timeliness(data, schema)
            else:
                return self._create_default_metric(dimension, 0.0)
                
        except Exception as e:
            logger.error(f"Failed to assess {dimension.value}: {str(e)}")
            return self._create_default_metric(dimension, 0.0)
    
    def _assess_completeness(self, data: pd.DataFrame, schema: Dict[str, Any] = None) -> QualityMetric:
        """Assess data completeness"""
        try:
            # Calculate missing values
            missing_counts = data.isnull().sum()
            total_values = len(data) * len(data.columns)
            missing_ratio = missing_counts.sum() / total_values
            completeness_score = 1.0 - missing_ratio
            
            # Identify columns with high missing values
            high_missing_cols = missing_counts[missing_counts > len(data) * 0.1].index.tolist()
            
            recommendations = []
            if high_missing_cols:
                recommendations.append(f"High missing values in columns: {', '.join(high_missing_cols)}")
                recommendations.append("Consider data imputation or source validation")
            
            if completeness_score < 0.8:
                recommendations.append("Overall completeness is low - review data collection process")
            
            level = self._determine_quality_level(completeness_score)
            
            return QualityMetric(
                dimension=QualityDimension.COMPLETENESS,
                value=completeness_score,
                threshold=self.quality_thresholds[QualityDimension.COMPLETENESS],
                level=level,
                description=f"Completeness: {completeness_score:.3f} ({missing_counts.sum()} missing values)",
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to assess completeness: {str(e)}")
            return self._create_default_metric(QualityDimension.COMPLETENESS, 0.0)
    
    def _assess_accuracy(self, data: pd.DataFrame, schema: Dict[str, Any] = None, business_rules: List[Dict[str, Any]] = None) -> QualityMetric:
        """Assess data accuracy"""
        try:
            accuracy_scores = []
            recommendations = []
            
            # Check against schema constraints
            if schema:
                for column, constraints in schema.get("fields", {}).items():
                    if column in data.columns:
                        col_accuracy = self._check_column_accuracy(data[column], constraints)
                        accuracy_scores.append(col_accuracy)
            
            # Check business rules
            if business_rules:
                for rule in business_rules:
                    rule_accuracy = self._check_business_rule(data, rule)
                    accuracy_scores.append(rule_accuracy)
            
            # If no specific rules, use basic statistical checks
            if not accuracy_scores:
                accuracy_scores = self._basic_accuracy_checks(data)
            
            overall_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.8
            
            if overall_accuracy < 0.9:
                recommendations.append("Data accuracy issues detected - review data sources")
                recommendations.append("Implement data validation at ingestion")
            
            level = self._determine_quality_level(overall_accuracy)
            
            return QualityMetric(
                dimension=QualityDimension.ACCURACY,
                value=overall_accuracy,
                threshold=self.quality_thresholds[QualityDimension.ACCURACY],
                level=level,
                description=f"Accuracy: {overall_accuracy:.3f}",
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to assess accuracy: {str(e)}")
            return self._create_default_metric(QualityDimension.ACCURACY, 0.0)
    
    def _assess_consistency(self, data: pd.DataFrame, schema: Dict[str, Any] = None) -> QualityMetric:
        """Assess data consistency"""
        try:
            consistency_scores = []
            recommendations = []
            
            # Check for duplicate records
            duplicates = data.duplicated().sum()
            duplicate_ratio = duplicates / len(data)
            consistency_scores.append(1.0 - duplicate_ratio)
            
            if duplicate_ratio > 0.05:
                recommendations.append(f"High duplicate ratio: {duplicate_ratio:.3f}")
                recommendations.append("Implement deduplication process")
            
            # Check for inconsistent formats
            format_consistency = self._check_format_consistency(data)
            consistency_scores.append(format_consistency)
            
            # Check for referential integrity
            if schema and "relationships" in schema:
                ref_consistency = self._check_referential_consistency(data, schema["relationships"])
                consistency_scores.append(ref_consistency)
            
            overall_consistency = np.mean(consistency_scores)
            
            if overall_consistency < 0.85:
                recommendations.append("Data consistency issues detected")
                recommendations.append("Review data integration processes")
            
            level = self._determine_quality_level(overall_consistency)
            
            return QualityMetric(
                dimension=QualityDimension.CONSISTENCY,
                value=overall_consistency,
                threshold=self.quality_thresholds[QualityDimension.CONSISTENCY],
                level=level,
                description=f"Consistency: {overall_consistency:.3f}",
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to assess consistency: {str(e)}")
            return self._create_default_metric(QualityDimension.CONSISTENCY, 0.0)
    
    def _assess_validity(self, data: pd.DataFrame, schema: Dict[str, Any] = None) -> QualityMetric:
        """Assess data validity"""
        try:
            validity_scores = []
            recommendations = []
            
            # Check data types
            if schema and "fields" in schema:
                for column, field_schema in schema["fields"].items():
                    if column in data.columns:
                        type_validity = self._check_data_type_validity(data[column], field_schema)
                        validity_scores.append(type_validity)
            
            # Check value ranges
            range_validity = self._check_value_ranges(data, schema)
            validity_scores.append(range_validity)
            
            # Check required fields
            if schema and "required_fields" in schema:
                required_validity = self._check_required_fields(data, schema["required_fields"])
                validity_scores.append(required_validity)
            
            overall_validity = np.mean(validity_scores) if validity_scores else 0.9
            
            if overall_validity < 0.9:
                recommendations.append("Data validity issues detected")
                recommendations.append("Implement stricter validation rules")
            
            level = self._determine_quality_level(overall_validity)
            
            return QualityMetric(
                dimension=QualityDimension.VALIDITY,
                value=overall_validity,
                threshold=self.quality_thresholds[QualityDimension.VALIDITY],
                level=level,
                description=f"Validity: {overall_validity:.3f}",
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to assess validity: {str(e)}")
            return self._create_default_metric(QualityDimension.VALIDITY, 0.0)
    
    def _assess_uniqueness(self, data: pd.DataFrame, schema: Dict[str, Any] = None) -> QualityMetric:
        """Assess data uniqueness"""
        try:
            # Check for duplicate records
            total_records = len(data)
            unique_records = len(data.drop_duplicates())
            uniqueness_ratio = unique_records / total_records
            
            # Check for duplicate keys
            key_uniqueness = 1.0
            if schema and "primary_key" in schema:
                key_columns = schema["primary_key"]
                if all(col in data.columns for col in key_columns):
                    key_duplicates = data[key_columns].duplicated().sum()
                    key_uniqueness = 1.0 - (key_duplicates / total_records)
            
            overall_uniqueness = (uniqueness_ratio + key_uniqueness) / 2
            
            recommendations = []
            if overall_uniqueness < 0.95:
                recommendations.append("Uniqueness issues detected")
                recommendations.append("Review data deduplication processes")
            
            level = self._determine_quality_level(overall_uniqueness)
            
            return QualityMetric(
                dimension=QualityDimension.UNIQUENESS,
                value=overall_uniqueness,
                threshold=self.quality_thresholds[QualityDimension.UNIQUENESS],
                level=level,
                description=f"Uniqueness: {overall_uniqueness:.3f}",
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to assess uniqueness: {str(e)}")
            return self._create_default_metric(QualityDimension.UNIQUENESS, 0.0)
    
    def _assess_timeliness(self, data: pd.DataFrame, schema: Dict[str, Any] = None) -> QualityMetric:
        """Assess data timeliness"""
        try:
            timeliness_scores = []
            recommendations = []
            
            # Check for timestamp columns
            timestamp_cols = data.select_dtypes(include=['datetime64']).columns
            
            if len(timestamp_cols) > 0:
                for col in timestamp_cols:
                    # Check for future dates
                    now = datetime.now()
                    future_dates = (data[col] > now).sum()
                    future_ratio = future_dates / len(data)
                    timeliness_scores.append(1.0 - future_ratio)
                    
                    if future_ratio > 0.01:
                        recommendations.append(f"Future dates found in {col}: {future_dates} records")
                
                # Check data freshness
                if len(timestamp_cols) > 0:
                    latest_timestamp = data[timestamp_cols[0]].max()
                    if pd.notna(latest_timestamp):
                        days_old = (now - latest_timestamp).days
                        freshness_score = max(0, 1.0 - (days_old / 30))  # 30 days threshold
                        timeliness_scores.append(freshness_score)
                        
                        if days_old > 7:
                            recommendations.append(f"Data is {days_old} days old - consider refresh")
            else:
                # No timestamp columns - assume good timeliness
                timeliness_scores.append(0.8)
            
            overall_timeliness = np.mean(timeliness_scores) if timeliness_scores else 0.8
            
            if overall_timeliness < 0.8:
                recommendations.append("Data timeliness issues detected")
                recommendations.append("Review data update frequency")
            
            level = self._determine_quality_level(overall_timeliness)
            
            return QualityMetric(
                dimension=QualityDimension.TIMELINESS,
                value=overall_timeliness,
                threshold=self.quality_thresholds[QualityDimension.TIMELINESS],
                level=level,
                description=f"Timeliness: {overall_timeliness:.3f}",
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to assess timeliness: {str(e)}")
            return self._create_default_metric(QualityDimension.TIMELINESS, 0.0)
    
    def _check_column_accuracy(self, column: pd.Series, constraints: Dict[str, Any]) -> float:
        """Check accuracy of a column against constraints"""
        try:
            accuracy_score = 1.0
            
            # Check data type
            if "type" in constraints:
                expected_type = constraints["type"]
                if expected_type == "string" and not column.dtype == "object":
                    accuracy_score -= 0.2
                elif expected_type == "numeric" and not pd.api.types.is_numeric_dtype(column):
                    accuracy_score -= 0.2
            
            # Check value constraints
            if "min_value" in constraints:
                valid_values = (column >= constraints["min_value"]).sum()
                accuracy_score = min(accuracy_score, valid_values / len(column))
            
            if "max_value" in constraints:
                valid_values = (column <= constraints["max_value"]).sum()
                accuracy_score = min(accuracy_score, valid_values / len(column))
            
            return max(0.0, accuracy_score)
            
        except Exception as e:
            logger.error(f"Failed to check column accuracy: {str(e)}")
            return 0.5
    
    def _check_business_rule(self, data: pd.DataFrame, rule: Dict[str, Any]) -> float:
        """Check data against business rule"""
        try:
            rule_type = rule.get("type", "custom")
            
            if rule_type == "sum_check":
                # Check if sum of columns equals expected value
                columns = rule.get("columns", [])
                expected_sum = rule.get("expected_sum", 0)
                actual_sum = data[columns].sum().sum()
                return 1.0 if abs(actual_sum - expected_sum) < 0.01 else 0.0
            
            elif rule_type == "ratio_check":
                # Check if ratio between columns is within range
                numerator_col = rule.get("numerator_column")
                denominator_col = rule.get("denominator_column")
                min_ratio = rule.get("min_ratio", 0)
                max_ratio = rule.get("max_ratio", 1)
                
                if numerator_col in data.columns and denominator_col in data.columns:
                    ratio = data[numerator_col] / data[denominator_col]
                    valid_ratios = ((ratio >= min_ratio) & (ratio <= max_ratio)).sum()
                    return valid_ratios / len(data)
            
            return 0.8  # Default score for unknown rules
            
        except Exception as e:
            logger.error(f"Failed to check business rule: {str(e)}")
            return 0.5
    
    def _basic_accuracy_checks(self, data: pd.DataFrame) -> List[float]:
        """Perform basic accuracy checks"""
        try:
            scores = []
            
            # Check for outliers in numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if len(data[col].dropna()) > 0:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                    outlier_ratio = outliers / len(data[col].dropna())
                    scores.append(1.0 - outlier_ratio)
            
            return scores if scores else [0.8]
            
        except Exception as e:
            logger.error(f"Failed basic accuracy checks: {str(e)}")
            return [0.5]
    
    def _check_format_consistency(self, data: pd.DataFrame) -> float:
        """Check format consistency across columns"""
        try:
            consistency_score = 1.0
            
            # Check string columns for consistent formatting
            string_cols = data.select_dtypes(include=['object']).columns
            for col in string_cols:
                if len(data[col].dropna()) > 0:
                    # Check for consistent case
                    unique_cases = data[col].str.isupper().nunique()
                    if unique_cases > 1:
                        consistency_score -= 0.1
                    
                    # Check for consistent whitespace
                    has_leading_space = data[col].str.startswith(' ').any()
                    has_trailing_space = data[col].str.endswith(' ').any()
                    if has_leading_space or has_trailing_space:
                        consistency_score -= 0.1
            
            return max(0.0, consistency_score)
            
        except Exception as e:
            logger.error(f"Failed to check format consistency: {str(e)}")
            return 0.5
    
    def _check_referential_consistency(self, data: pd.DataFrame, relationships: List[Dict[str, Any]]) -> float:
        """Check referential integrity"""
        try:
            consistency_score = 1.0
            
            for relationship in relationships:
                parent_col = relationship.get("parent_column")
                child_col = relationship.get("child_column")
                
                if parent_col in data.columns and child_col in data.columns:
                    # Check if all child values exist in parent
                    parent_values = set(data[parent_col].dropna().unique())
                    child_values = set(data[child_col].dropna().unique())
                    orphaned_values = child_values - parent_values
                    
                    if orphaned_values:
                        orphaned_ratio = len(orphaned_values) / len(child_values)
                        consistency_score -= orphaned_ratio * 0.5
            
            return max(0.0, consistency_score)
            
        except Exception as e:
            logger.error(f"Failed to check referential consistency: {str(e)}")
            return 0.5
    
    def _check_data_type_validity(self, column: pd.Series, field_schema: Dict[str, Any]) -> float:
        """Check if column data types match schema"""
        try:
            expected_type = field_schema.get("type", "string")
            actual_type = str(column.dtype)
            
            type_mapping = {
                "string": "object",
                "integer": "int64",
                "float": "float64",
                "boolean": "bool",
                "datetime": "datetime64"
            }
            
            expected_dtype = type_mapping.get(expected_type, "object")
            
            if expected_dtype in actual_type:
                return 1.0
            else:
                return 0.5  # Partial match
            
        except Exception as e:
            logger.error(f"Failed to check data type validity: {str(e)}")
            return 0.5
    
    def _check_value_ranges(self, data: pd.DataFrame, schema: Dict[str, Any]) -> float:
        """Check if values are within expected ranges"""
        try:
            if not schema or "fields" not in schema:
                return 0.8
            
            valid_values = 0
            total_values = 0
            
            for column, field_schema in schema["fields"].items():
                if column in data.columns:
                    col_data = data[column].dropna()
                    if len(col_data) > 0:
                        total_values += len(col_data)
                        
                        # Check min/max values
                        if "min_value" in field_schema:
                            valid_values += (col_data >= field_schema["min_value"]).sum()
                        else:
                            valid_values += len(col_data)
                        
                        if "max_value" in field_schema:
                            valid_values += (col_data <= field_schema["max_value"]).sum()
                        else:
                            valid_values += len(col_data)
            
            return valid_values / (total_values * 2) if total_values > 0 else 0.8
            
        except Exception as e:
            logger.error(f"Failed to check value ranges: {str(e)}")
            return 0.5
    
    def _check_required_fields(self, data: pd.DataFrame, required_fields: List[str]) -> float:
        """Check if required fields are present and non-null"""
        try:
            missing_fields = [field for field in required_fields if field not in data.columns]
            if missing_fields:
                return 0.0
            
            null_counts = data[required_fields].isnull().sum()
            total_required_values = len(data) * len(required_fields)
            null_ratio = null_counts.sum() / total_required_values
            
            return 1.0 - null_ratio
            
        except Exception as e:
            logger.error(f"Failed to check required fields: {str(e)}")
            return 0.5
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on score"""
        if score >= 0.95:
            return QualityLevel.EXCELLENT
        elif score >= 0.85:
            return QualityLevel.GOOD
        elif score >= 0.70:
            return QualityLevel.FAIR
        elif score >= 0.50:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _create_default_metric(self, dimension: QualityDimension, value: float) -> QualityMetric:
        """Create a default quality metric"""
        return QualityMetric(
            dimension=dimension,
            value=value,
            threshold=self.quality_thresholds[dimension],
            level=self._determine_quality_level(value),
            description=f"{dimension.value}: {value:.3f}",
            recommendations=[]
        )
    
    def _create_empty_quality_score(self) -> QualityScore:
        """Create quality score for empty data"""
        return QualityScore(
            overall_score=0.0,
            dimension_scores={dim: 0.0 for dim in QualityDimension},
            level=QualityLevel.CRITICAL,
            assessment_date=datetime.now(),
            data_size=0,
            issues_found=1
        )
    
    def _create_error_quality_score(self, error_message: str) -> QualityScore:
        """Create quality score for error case"""
        return QualityScore(
            overall_score=0.0,
            dimension_scores={dim: 0.0 for dim in QualityDimension},
            level=QualityLevel.CRITICAL,
            assessment_date=datetime.now(),
            data_size=0,
            issues_found=1
        )
