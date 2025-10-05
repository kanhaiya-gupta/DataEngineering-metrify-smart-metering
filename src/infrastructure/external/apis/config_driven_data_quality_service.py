"""
Configuration-Driven Data Quality Service
Implements data quality validation using configuration rules from data_sources.yaml
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass

from ....core.config.config_loader import ConfigLoader
from ....core.exceptions.domain_exceptions import DataQualityError

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    quality_score: float
    violations: List[str]
    warnings: List[str]
    recommendations: List[str]
    metrics: Dict[str, Any]


@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_types: List[str]
    confidence: float
    details: Dict[str, Any]


class ConfigDrivenDataQualityService:
    """
    Configuration-driven data quality service that validates data
    based on rules defined in data_sources.yaml
    """
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.data_sources_config = self.config_loader.get_data_sources_config()
        self.quality_threshold = 0.8
        self.anomaly_threshold = 0.7
        
    async def validate_smart_meter_data(self, data: pd.DataFrame) -> ValidationResult:
        """Validate smart meter data against configuration rules"""
        try:
            config = self.data_sources_config.smart_meters.validation
            return await self._validate_data(data, config, "smart_meter")
        except Exception as e:
            logger.error(f"Error validating smart meter data: {e}")
            raise DataQualityError(f"Failed to validate smart meter data: {e}")
    
    async def validate_grid_operator_data(self, data: pd.DataFrame) -> ValidationResult:
        """Validate grid operator data against configuration rules"""
        try:
            config = self.data_sources_config.grid_operators.validation
            return await self._validate_data(data, config, "grid_operator")
        except Exception as e:
            logger.error(f"Error validating grid operator data: {e}")
            raise DataQualityError(f"Failed to validate grid operator data: {e}")
    
    async def validate_weather_data(self, data: pd.DataFrame) -> ValidationResult:
        """Validate weather data against configuration rules"""
        try:
            config = self.data_sources_config.weather_stations.validation
            return await self._validate_data(data, config, "weather")
        except Exception as e:
            logger.error(f"Error validating weather data: {e}")
            raise DataQualityError(f"Failed to validate weather data: {e}")
    
    async def _validate_data(self, data: pd.DataFrame, config: Dict[str, Any], data_type: str) -> ValidationResult:
        """Generic data validation method"""
        violations = []
        warnings = []
        recommendations = []
        metrics = {}
        
        # 1. Check required columns
        required_columns = config.get('required_columns', [])
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            violations.append(f"Missing required columns: {missing_columns}")
        
        # 2. Check data types
        data_type_violations = await self._validate_data_types(data, config.get('data_types', {}))
        violations.extend(data_type_violations)
        
        # 3. Check value ranges
        range_violations, range_warnings = await self._validate_value_ranges(data, config.get('value_ranges', {}))
        violations.extend(range_violations)
        warnings.extend(range_warnings)
        
        # 4. Check for null values
        null_violations, null_metrics = await self._check_null_values(data, required_columns)
        violations.extend(null_violations)
        metrics.update(null_metrics)
        
        # 5. Check for duplicates
        duplicate_violations, duplicate_metrics = await self._check_duplicates(data)
        violations.extend(duplicate_violations)
        metrics.update(duplicate_metrics)
        
        # 6. Calculate quality score
        quality_score = await self._calculate_quality_score(data, violations, warnings)
        
        # 7. Generate recommendations
        recommendations = await self._generate_recommendations(violations, warnings, quality_score, data_type)
        
        # 8. Add additional metrics
        metrics.update({
            'total_records': len(data),
            'total_columns': len(data.columns),
            'validation_timestamp': datetime.utcnow().isoformat(),
            'data_type': data_type
        })
        
        is_valid = len(violations) == 0 and quality_score >= self.quality_threshold
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            metrics=metrics
        )
    
    async def _validate_data_types(self, data: pd.DataFrame, type_config: Dict[str, str]) -> List[str]:
        """Validate data types against configuration"""
        violations = []
        
        for column, expected_type in type_config.items():
            if column not in data.columns:
                continue
                
            actual_type = str(data[column].dtype)
            
            # Map pandas types to expected types
            type_mapping = {
                'string': ['object', 'string'],
                'float': ['float64', 'float32', 'int64', 'int32'],
                'int': ['int64', 'int32', 'float64'],
                'datetime': ['datetime64[ns]', 'object'],
                'bool': ['bool', 'object']
            }
            
            expected_types = type_mapping.get(expected_type, [expected_type])
            
            if actual_type not in expected_types:
                violations.append(f"Column '{column}' has type '{actual_type}' but expected '{expected_type}'")
        
        return violations
    
    async def _validate_value_ranges(self, data: pd.DataFrame, range_config: Dict[str, Dict[str, float]]) -> Tuple[List[str], List[str]]:
        """Validate value ranges against configuration"""
        violations = []
        warnings = []
        
        for column, ranges in range_config.items():
            if column not in data.columns:
                continue
                
            min_val = ranges.get('min')
            max_val = ranges.get('max')
            
            if min_val is not None:
                below_min = data[column] < min_val
                if below_min.any():
                    count = below_min.sum()
                    violations.append(f"Column '{column}' has {count} values below minimum {min_val}")
            
            if max_val is not None:
                above_max = data[column] > max_val
                if above_max.any():
                    count = above_max.sum()
                    violations.append(f"Column '{column}' has {count} values above maximum {max_val}")
            
            # Check for extreme outliers (beyond 3 standard deviations)
            if data[column].dtype in ['float64', 'int64']:
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                extreme_outliers = z_scores > 3
                if extreme_outliers.any():
                    count = extreme_outliers.sum()
                    warnings.append(f"Column '{column}' has {count} extreme outliers (z-score > 3)")
        
        return violations, warnings
    
    async def _check_null_values(self, data: pd.DataFrame, required_columns: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        """Check for null values and calculate completeness metrics"""
        violations = []
        metrics = {}
        
        null_counts = data.isnull().sum()
        total_records = len(data)
        
        for column in required_columns:
            if column in data.columns:
                null_count = null_counts[column]
                null_percentage = (null_count / total_records) * 100
                
                metrics[f'{column}_null_count'] = int(null_count)
                metrics[f'{column}_null_percentage'] = round(null_percentage, 2)
                
                if null_count > 0:
                    if null_percentage > 10:  # More than 10% nulls is a violation
                        violations.append(f"Column '{column}' has {null_count} null values ({null_percentage:.1f}%)")
                    elif null_percentage > 5:  # More than 5% nulls is a warning
                        violations.append(f"Column '{column}' has {null_count} null values ({null_percentage:.1f}%) - warning")
        
        # Overall completeness
        overall_completeness = ((total_records - data.isnull().sum().sum()) / (total_records * len(data.columns))) * 100
        metrics['overall_completeness'] = round(overall_completeness, 2)
        
        return violations, metrics
    
    async def _check_duplicates(self, data: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """Check for duplicate records"""
        violations = []
        metrics = {}
        
        # Check for exact duplicates
        exact_duplicates = data.duplicated().sum()
        metrics['exact_duplicates'] = int(exact_duplicates)
        
        if exact_duplicates > 0:
            duplicate_percentage = (exact_duplicates / len(data)) * 100
            metrics['duplicate_percentage'] = round(duplicate_percentage, 2)
            
            if duplicate_percentage > 5:  # More than 5% duplicates is a violation
                violations.append(f"Found {exact_duplicates} exact duplicate records ({duplicate_percentage:.1f}%)")
        
        return violations, metrics
    
    async def _calculate_quality_score(self, data: pd.DataFrame, violations: List[str], warnings: List[str]) -> float:
        """Calculate overall quality score"""
        base_score = 1.0
        
        # Deduct for violations (0.1 per violation)
        violation_penalty = len(violations) * 0.1
        
        # Deduct for warnings (0.05 per warning)
        warning_penalty = len(warnings) * 0.05
        
        # Deduct for null values
        null_penalty = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 0.2
        
        # Deduct for duplicates
        duplicate_penalty = (data.duplicated().sum() / len(data)) * 0.1
        
        quality_score = max(0.0, base_score - violation_penalty - warning_penalty - null_penalty - duplicate_penalty)
        
        return round(quality_score, 3)
    
    async def _generate_recommendations(self, violations: List[str], warnings: List[str], quality_score: float, data_type: str) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if quality_score < 0.5:
            recommendations.append("Data quality is critically low. Consider data source investigation.")
        elif quality_score < 0.8:
            recommendations.append("Data quality is below threshold. Review data processing pipeline.")
        
        if any("null" in v.lower() for v in violations):
            recommendations.append("Consider implementing data imputation strategies for missing values.")
        
        if any("duplicate" in v.lower() for v in violations):
            recommendations.append("Implement deduplication logic in data processing pipeline.")
        
        if any("range" in v.lower() for v in violations):
            recommendations.append("Review data source for potential sensor calibration issues.")
        
        if any("type" in v.lower() for v in violations):
            recommendations.append("Check data parsing and type conversion logic.")
        
        return recommendations
    
    async def detect_anomalies(self, data: pd.DataFrame, data_type: str) -> AnomalyResult:
        """Detect anomalies in the data using statistical methods"""
        try:
            anomaly_types = []
            anomaly_details = {}
            max_anomaly_score = 0.0
            
            # Get numeric columns for anomaly detection
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if data[column].notna().sum() < 10:  # Need at least 10 values
                    continue
                    
                # Z-score based anomaly detection
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                anomalies = z_scores > 2.5  # Threshold for anomalies
                
                if anomalies.any():
                    anomaly_count = anomalies.sum()
                    anomaly_percentage = (anomaly_count / len(data)) * 100
                    max_z_score = z_scores.max()
                    
                    anomaly_types.append(f"{column}_outliers")
                    anomaly_details[column] = {
                        'anomaly_count': int(anomaly_count),
                        'anomaly_percentage': round(anomaly_percentage, 2),
                        'max_z_score': round(max_z_score, 2)
                    }
                    
                    max_anomaly_score = max(max_anomaly_score, min(max_z_score / 5.0, 1.0))  # Normalize to 0-1
            
            # Check for temporal anomalies (if timestamp column exists)
            timestamp_columns = [col for col in data.columns if 'timestamp' in col.lower() or 'time' in col.lower()]
            if timestamp_columns:
                temporal_anomalies = await self._detect_temporal_anomalies(data, timestamp_columns[0])
                if temporal_anomalies:
                    anomaly_types.append("temporal_anomalies")
                    anomaly_details['temporal'] = temporal_anomalies
                    max_anomaly_score = max(max_anomaly_score, 0.5)
            
            is_anomaly = len(anomaly_types) > 0 and max_anomaly_score > self.anomaly_threshold
            confidence = max_anomaly_score if is_anomaly else 0.0
            
            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=max_anomaly_score,
                anomaly_types=anomaly_types,
                confidence=confidence,
                details=anomaly_details
            )
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_types=[],
                confidence=0.0,
                details={'error': str(e)}
            )
    
    async def _detect_temporal_anomalies(self, data: pd.DataFrame, timestamp_column: str) -> Optional[Dict[str, Any]]:
        """Detect temporal anomalies in timestamp data"""
        try:
            if data[timestamp_column].dtype == 'object':
                # Try to convert to datetime
                data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors='coerce')
            
            if data[timestamp_column].isna().all():
                return None
            
            # Sort by timestamp
            sorted_data = data.sort_values(timestamp_column)
            
            # Calculate time differences
            time_diffs = sorted_data[timestamp_column].diff().dt.total_seconds()
            
            # Detect unusual gaps (more than 2 standard deviations from mean)
            mean_gap = time_diffs.mean()
            std_gap = time_diffs.std()
            
            if std_gap > 0:
                unusual_gaps = time_diffs > (mean_gap + 2 * std_gap)
                if unusual_gaps.any():
                    return {
                        'unusual_gaps': int(unusual_gaps.sum()),
                        'mean_gap_seconds': round(mean_gap, 2),
                        'max_gap_seconds': round(time_diffs.max(), 2)
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error detecting temporal anomalies: {e}")
            return None
    
    async def get_quality_summary(self, data_type: str) -> Dict[str, Any]:
        """Get quality summary for a specific data type"""
        try:
            config = getattr(self.data_sources_config, data_type).validation
            
            return {
                'data_type': data_type,
                'validation_rules': {
                    'required_columns': len(config.get('required_columns', [])),
                    'data_types': len(config.get('data_types', {})),
                    'value_ranges': len(config.get('value_ranges', {}))
                },
                'quality_threshold': self.quality_threshold,
                'anomaly_threshold': self.anomaly_threshold,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting quality summary: {e}")
            return {'error': str(e)}
