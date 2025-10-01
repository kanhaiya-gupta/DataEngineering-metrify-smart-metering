"""
Quality Scorer
Automated quality scoring and metrics calculation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ScoringMethod(Enum):
    """Quality scoring methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN = "harmonic_mean"
    CUSTOM = "custom"

@dataclass
class QualityDimension:
    """Quality dimension with weight and score"""
    name: str
    weight: float
    score: float
    threshold: float
    importance: str  # low, medium, high, critical

@dataclass
class QualityScore:
    """Comprehensive quality score"""
    overall_score: float
    dimension_scores: List[QualityDimension]
    method: ScoringMethod
    calculated_at: datetime
    confidence: float
    recommendations: List[str]

class QualityScorer:
    """
    Automated quality scoring and metrics calculation
    """
    
    def __init__(self):
        self.default_weights = {
            "completeness": 0.25,
            "accuracy": 0.25,
            "consistency": 0.20,
            "validity": 0.15,
            "uniqueness": 0.10,
            "timeliness": 0.05
        }
        
        logger.info("QualityScorer initialized")
    
    def calculate_quality_score(self,
                              data: pd.DataFrame,
                              dimensions: List[QualityDimension] = None,
                              method: ScoringMethod = ScoringMethod.WEIGHTED_AVERAGE,
                              custom_weights: Dict[str, float] = None) -> QualityScore:
        """Calculate comprehensive quality score"""
        try:
            if data.empty:
                return self._create_empty_score()
            
            # Use provided dimensions or create default ones
            if dimensions is None:
                dimensions = self._create_default_dimensions(data)
            
            # Calculate scores for each dimension
            calculated_dimensions = []
            for dimension in dimensions:
                score = self._calculate_dimension_score(data, dimension)
                calculated_dimensions.append(score)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(calculated_dimensions, method)
            
            # Calculate confidence
            confidence = self._calculate_confidence(calculated_dimensions)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(calculated_dimensions, overall_score)
            
            quality_score = QualityScore(
                overall_score=overall_score,
                dimension_scores=calculated_dimensions,
                method=method,
                calculated_at=datetime.now(),
                confidence=confidence,
                recommendations=recommendations
            )
            
            logger.info(f"Quality score calculated: {overall_score:.3f} (confidence: {confidence:.3f})")
            return quality_score
            
        except Exception as e:
            logger.error(f"Failed to calculate quality score: {str(e)}")
            return self._create_error_score(str(e))
    
    def _create_default_dimensions(self, data: pd.DataFrame) -> List[QualityDimension]:
        """Create default quality dimensions"""
        try:
            dimensions = []
            
            # Completeness
            completeness_score = self._calculate_completeness_score(data)
            dimensions.append(QualityDimension(
                name="completeness",
                weight=self.default_weights["completeness"],
                score=completeness_score,
                threshold=0.95,
                importance="high"
            ))
            
            # Accuracy
            accuracy_score = self._calculate_accuracy_score(data)
            dimensions.append(QualityDimension(
                name="accuracy",
                weight=self.default_weights["accuracy"],
                score=accuracy_score,
                threshold=0.90,
                importance="high"
            ))
            
            # Consistency
            consistency_score = self._calculate_consistency_score(data)
            dimensions.append(QualityDimension(
                name="consistency",
                weight=self.default_weights["consistency"],
                score=consistency_score,
                threshold=0.85,
                importance="medium"
            ))
            
            # Validity
            validity_score = self._calculate_validity_score(data)
            dimensions.append(QualityDimension(
                name="validity",
                weight=self.default_weights["validity"],
                score=validity_score,
                threshold=0.90,
                importance="high"
            ))
            
            # Uniqueness
            uniqueness_score = self._calculate_uniqueness_score(data)
            dimensions.append(QualityDimension(
                name="uniqueness",
                weight=self.default_weights["uniqueness"],
                score=uniqueness_score,
                threshold=0.95,
                importance="medium"
            ))
            
            # Timeliness
            timeliness_score = self._calculate_timeliness_score(data)
            dimensions.append(QualityDimension(
                name="timeliness",
                weight=self.default_weights["timeliness"],
                score=timeliness_score,
                threshold=0.80,
                importance="low"
            ))
            
            return dimensions
            
        except Exception as e:
            logger.error(f"Failed to create default dimensions: {str(e)}")
            return []
    
    def _calculate_dimension_score(self, data: pd.DataFrame, dimension: QualityDimension) -> QualityDimension:
        """Calculate score for a specific dimension"""
        try:
            if dimension.name == "completeness":
                score = self._calculate_completeness_score(data)
            elif dimension.name == "accuracy":
                score = self._calculate_accuracy_score(data)
            elif dimension.name == "consistency":
                score = self._calculate_consistency_score(data)
            elif dimension.name == "validity":
                score = self._calculate_validity_score(data)
            elif dimension.name == "uniqueness":
                score = self._calculate_uniqueness_score(data)
            elif dimension.name == "timeliness":
                score = self._calculate_timeliness_score(data)
            else:
                score = 0.5  # Default score for unknown dimensions
            
            return QualityDimension(
                name=dimension.name,
                weight=dimension.weight,
                score=score,
                threshold=dimension.threshold,
                importance=dimension.importance
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate dimension score for {dimension.name}: {str(e)}")
            return QualityDimension(
                name=dimension.name,
                weight=dimension.weight,
                score=0.0,
                threshold=dimension.threshold,
                importance=dimension.importance
            )
    
    def _calculate_completeness_score(self, data: pd.DataFrame) -> float:
        """Calculate completeness score"""
        try:
            if data.empty:
                return 0.0
            
            total_cells = data.size
            null_cells = data.isnull().sum().sum()
            completeness = 1.0 - (null_cells / total_cells)
            
            return max(0.0, min(1.0, completeness))
            
        except Exception as e:
            logger.error(f"Failed to calculate completeness score: {str(e)}")
            return 0.0
    
    def _calculate_accuracy_score(self, data: pd.DataFrame) -> float:
        """Calculate accuracy score"""
        try:
            if data.empty:
                return 0.0
            
            # Basic accuracy checks
            accuracy_scores = []
            
            # Check for outliers in numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if len(data[col].dropna()) > 0:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                    outlier_ratio = outliers / len(data[col].dropna())
                    accuracy_scores.append(1.0 - outlier_ratio)
            
            # Check for impossible values
            for col in numeric_cols:
                if len(data[col].dropna()) > 0:
                    # Check for negative values where they shouldn't exist
                    if col.lower() in ['age', 'count', 'quantity']:
                        negative_ratio = (data[col] < 0).sum() / len(data[col].dropna())
                        accuracy_scores.append(1.0 - negative_ratio)
            
            return np.mean(accuracy_scores) if accuracy_scores else 0.8
            
        except Exception as e:
            logger.error(f"Failed to calculate accuracy score: {str(e)}")
            return 0.0
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate consistency score"""
        try:
            if data.empty:
                return 0.0
            
            consistency_scores = []
            
            # Check for duplicate records
            duplicates = data.duplicated().sum()
            duplicate_ratio = duplicates / len(data)
            consistency_scores.append(1.0 - duplicate_ratio)
            
            # Check for inconsistent formats in string columns
            string_cols = data.select_dtypes(include=['object']).columns
            for col in string_cols:
                if len(data[col].dropna()) > 0:
                    # Check for consistent case
                    unique_cases = data[col].str.isupper().nunique()
                    if unique_cases > 1:
                        consistency_scores.append(0.8)  # Partial penalty
                    else:
                        consistency_scores.append(1.0)
                    
                    # Check for consistent whitespace
                    has_leading_space = data[col].str.startswith(' ').any()
                    has_trailing_space = data[col].str.endswith(' ').any()
                    if has_leading_space or has_trailing_space:
                        consistency_scores.append(0.9)  # Small penalty
                    else:
                        consistency_scores.append(1.0)
            
            return np.mean(consistency_scores) if consistency_scores else 0.8
            
        except Exception as e:
            logger.error(f"Failed to calculate consistency score: {str(e)}")
            return 0.0
    
    def _calculate_validity_score(self, data: pd.DataFrame) -> float:
        """Calculate validity score"""
        try:
            if data.empty:
                return 0.0
            
            validity_scores = []
            
            # Check data types
            for col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    # Check if numeric columns contain only numbers
                    if data[col].dtype in ['int64', 'float64']:
                        non_numeric = pd.to_numeric(col_data, errors='coerce').isnull().sum()
                        validity_ratio = 1.0 - (non_numeric / len(col_data))
                        validity_scores.append(validity_ratio)
                    
                    # Check if datetime columns contain valid dates
                    elif 'datetime' in str(data[col].dtype):
                        try:
                            pd.to_datetime(col_data)
                            validity_scores.append(1.0)
                        except:
                            validity_scores.append(0.0)
                    
                    # Check string columns for reasonable length
                    elif data[col].dtype == 'object':
                        # Check for extremely long strings (potential data corruption)
                        max_length = col_data.astype(str).str.len().max()
                        if max_length > 1000:  # Arbitrary threshold
                            validity_scores.append(0.8)
                        else:
                            validity_scores.append(1.0)
            
            return np.mean(validity_scores) if validity_scores else 0.8
            
        except Exception as e:
            logger.error(f"Failed to calculate validity score: {str(e)}")
            return 0.0
    
    def _calculate_uniqueness_score(self, data: pd.DataFrame) -> float:
        """Calculate uniqueness score"""
        try:
            if data.empty:
                return 0.0
            
            # Check for duplicate records
            total_records = len(data)
            unique_records = len(data.drop_duplicates())
            uniqueness_ratio = unique_records / total_records
            
            return max(0.0, min(1.0, uniqueness_ratio))
            
        except Exception as e:
            logger.error(f"Failed to calculate uniqueness score: {str(e)}")
            return 0.0
    
    def _calculate_timeliness_score(self, data: pd.DataFrame) -> float:
        """Calculate timeliness score"""
        try:
            if data.empty:
                return 0.0
            
            # Check for timestamp columns
            timestamp_cols = data.select_dtypes(include=['datetime64']).columns
            
            if len(timestamp_cols) > 0:
                timeliness_scores = []
                
                for col in timestamp_cols:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        now = datetime.now()
                        
                        # Check for future dates
                        future_dates = (col_data > now).sum()
                        future_ratio = future_dates / len(col_data)
                        timeliness_scores.append(1.0 - future_ratio)
                        
                        # Check data freshness
                        latest_timestamp = col_data.max()
                        if pd.notna(latest_timestamp):
                            days_old = (now - latest_timestamp).days
                            freshness_score = max(0, 1.0 - (days_old / 30))  # 30 days threshold
                            timeliness_scores.append(freshness_score)
                
                return np.mean(timeliness_scores) if timeliness_scores else 0.8
            else:
                # No timestamp columns - assume good timeliness
                return 0.8
            
        except Exception as e:
            logger.error(f"Failed to calculate timeliness score: {str(e)}")
            return 0.0
    
    def _calculate_overall_score(self, 
                               dimensions: List[QualityDimension],
                               method: ScoringMethod) -> float:
        """Calculate overall quality score"""
        try:
            if not dimensions:
                return 0.0
            
            scores = [d.score for d in dimensions]
            weights = [d.weight for d in dimensions]
            
            if method == ScoringMethod.WEIGHTED_AVERAGE:
                return np.average(scores, weights=weights)
            elif method == ScoringMethod.GEOMETRIC_MEAN:
                # Geometric mean of weighted scores
                weighted_scores = [s ** w for s, w in zip(scores, weights)]
                return np.prod(weighted_scores) ** (1 / sum(weights))
            elif method == ScoringMethod.HARMONIC_MEAN:
                # Harmonic mean of weighted scores
                weighted_scores = [w / s for s, w in zip(scores, weights) if s > 0]
                return sum(weights) / sum(weighted_scores) if weighted_scores else 0.0
            else:
                return np.average(scores, weights=weights)
                
        except Exception as e:
            logger.error(f"Failed to calculate overall score: {str(e)}")
            return 0.0
    
    def _calculate_confidence(self, dimensions: List[QualityDimension]) -> float:
        """Calculate confidence in the quality score"""
        try:
            if not dimensions:
                return 0.0
            
            # Confidence based on score variance and data size
            scores = [d.score for d in dimensions]
            score_variance = np.var(scores)
            
            # Lower variance = higher confidence
            variance_confidence = max(0, 1.0 - score_variance)
            
            # More dimensions = higher confidence
            dimension_confidence = min(1.0, len(dimensions) / 6.0)
            
            # Overall confidence
            confidence = (variance_confidence + dimension_confidence) / 2
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {str(e)}")
            return 0.5
    
    def _generate_recommendations(self, 
                                dimensions: List[QualityDimension],
                                overall_score: float) -> List[str]:
        """Generate quality improvement recommendations"""
        try:
            recommendations = []
            
            # Overall score recommendations
            if overall_score < 0.5:
                recommendations.append("Overall quality is critically low - immediate attention required")
            elif overall_score < 0.7:
                recommendations.append("Overall quality is below acceptable levels - improvement needed")
            elif overall_score < 0.9:
                recommendations.append("Overall quality is good but can be improved")
            
            # Dimension-specific recommendations
            for dimension in dimensions:
                if dimension.score < dimension.threshold:
                    if dimension.importance == "critical":
                        recommendations.append(f"CRITICAL: {dimension.name} is below threshold ({dimension.score:.3f} < {dimension.threshold:.3f})")
                    elif dimension.importance == "high":
                        recommendations.append(f"HIGH PRIORITY: {dimension.name} needs improvement ({dimension.score:.3f} < {dimension.threshold:.3f})")
                    else:
                        recommendations.append(f"Consider improving {dimension.name} ({dimension.score:.3f} < {dimension.threshold:.3f})")
            
            # Specific improvement suggestions
            for dimension in dimensions:
                if dimension.name == "completeness" and dimension.score < 0.9:
                    recommendations.append("Improve data completeness by fixing data collection processes")
                elif dimension.name == "accuracy" and dimension.score < 0.9:
                    recommendations.append("Improve data accuracy by implementing validation rules")
                elif dimension.name == "consistency" and dimension.score < 0.8:
                    recommendations.append("Improve data consistency by standardizing data formats")
                elif dimension.name == "validity" and dimension.score < 0.9:
                    recommendations.append("Improve data validity by enforcing data type constraints")
                elif dimension.name == "uniqueness" and dimension.score < 0.9:
                    recommendations.append("Improve data uniqueness by implementing deduplication")
                elif dimension.name == "timeliness" and dimension.score < 0.8:
                    recommendations.append("Improve data timeliness by increasing update frequency")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return ["Unable to generate recommendations"]
    
    def _create_empty_score(self) -> QualityScore:
        """Create quality score for empty data"""
        return QualityScore(
            overall_score=0.0,
            dimension_scores=[],
            method=ScoringMethod.WEIGHTED_AVERAGE,
            calculated_at=datetime.now(),
            confidence=0.0,
            recommendations=["No data available for quality assessment"]
        )
    
    def _create_error_score(self, error_message: str) -> QualityScore:
        """Create quality score for error case"""
        return QualityScore(
            overall_score=0.0,
            dimension_scores=[],
            method=ScoringMethod.WEIGHTED_AVERAGE,
            calculated_at=datetime.now(),
            confidence=0.0,
            recommendations=[f"Quality assessment failed: {error_message}"]
        )
    
    def get_quality_benchmarks(self) -> Dict[str, Any]:
        """Get quality score benchmarks"""
        return {
            "excellent": 0.95,
            "good": 0.85,
            "fair": 0.70,
            "poor": 0.50,
            "critical": 0.30,
            "dimension_thresholds": {
                "completeness": 0.95,
                "accuracy": 0.90,
                "consistency": 0.85,
                "validity": 0.90,
                "uniqueness": 0.95,
                "timeliness": 0.80
            },
            "importance_levels": {
                "critical": ["completeness", "accuracy"],
                "high": ["validity"],
                "medium": ["consistency", "uniqueness"],
                "low": ["timeliness"]
            }
        }
