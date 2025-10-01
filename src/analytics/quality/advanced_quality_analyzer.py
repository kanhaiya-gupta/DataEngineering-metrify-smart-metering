"""
Advanced Quality Analyzer
Advanced quality analysis integrated with existing analytics
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

# Import governance quality components
from src.governance.quality import (
    QualityAssessor, ValidationEngine, QualityMonitor,
    QualityScorer, TrendAnalyzer, RemediationEngine
)

logger = logging.getLogger(__name__)

class QualityAnalysisType(Enum):
    """Types of quality analysis"""
    COMPREHENSIVE = "comprehensive"
    DIMENSIONAL = "dimensional"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"

@dataclass
class QualityInsight:
    """Quality insight result"""
    insight_type: str
    description: str
    severity: str
    confidence: float
    recommendations: List[str]
    affected_data: Dict[str, Any]
    metadata: Dict[str, Any]

class AdvancedQualityAnalyzer:
    """
    Advanced quality analysis integrated with existing analytics
    """
    
    def __init__(self):
        self.quality_assessor = QualityAssessor()
        self.validation_engine = ValidationEngine()
        self.quality_scorer = QualityScorer()
        self.trend_analyzer = TrendAnalyzer()
        self.remediation_engine = RemediationEngine()
        
        logger.info("AdvancedQualityAnalyzer initialized")
    
    def analyze_data_quality_comprehensive(self,
                                         data: pd.DataFrame,
                                         schema: Dict[str, Any] = None,
                                         business_rules: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform comprehensive quality analysis"""
        try:
            if data.empty:
                return {"error": "No data provided for analysis"}
            
            analysis_results = {
                "analysis_metadata": {
                    "analyzed_at": datetime.now().isoformat(),
                    "data_shape": data.shape,
                    "analysis_type": "comprehensive"
                },
                "quality_assessment": {},
                "validation_results": {},
                "trend_analysis": {},
                "insights": [],
                "recommendations": []
            }
            
            # 1. Quality Assessment
            quality_score = self.quality_assessor.assess_data_quality(
                data=data,
                schema=schema,
                business_rules=business_rules
            )
            
            analysis_results["quality_assessment"] = {
                "overall_score": quality_score.overall_score,
                "quality_level": quality_score.level.value,
                "dimension_scores": {
                    dim.value: score for dim, score in quality_score.dimension_scores.items()
                },
                "issues_found": quality_score.issues_found,
                "assessment_date": quality_score.assessment_date.isoformat()
            }
            
            # 2. Validation Results
            validation_results = self.validation_engine.validate_data(
                data=data,
                schema=schema
            )
            
            validation_summary = self.validation_engine.get_validation_summary(validation_results)
            analysis_results["validation_results"] = {
                "total_rules": validation_summary["total_rules"],
                "passed_rules": validation_summary["passed_rules"],
                "failed_rules": validation_summary["failed_rules"],
                "pass_rate": validation_summary["pass_rate"],
                "error_count": validation_summary["error_count"],
                "warning_count": validation_summary["warning_count"]
            }
            
            # 3. Trend Analysis (if temporal data available)
            timestamp_cols = data.select_dtypes(include=['datetime64']).columns
            if len(timestamp_cols) > 0:
                trend_analysis = self._analyze_quality_trends(data, timestamp_cols[0])
                analysis_results["trend_analysis"] = trend_analysis
            
            # 4. Generate Insights
            insights = self._generate_quality_insights(
                quality_score, validation_summary, data
            )
            analysis_results["insights"] = insights
            
            # 5. Generate Recommendations
            recommendations = self._generate_quality_recommendations(
                quality_score, validation_summary, insights
            )
            analysis_results["recommendations"] = recommendations
            
            logger.info(f"Comprehensive quality analysis completed for {data.shape[0]} rows")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to perform comprehensive quality analysis: {str(e)}")
            return {"error": str(e)}
    
    def analyze_quality_dimensions(self,
                                 data: pd.DataFrame,
                                 dimensions: List[str] = None) -> Dict[str, Any]:
        """Analyze specific quality dimensions"""
        try:
            if data.empty:
                return {"error": "No data provided for analysis"}
            
            if dimensions is None:
                dimensions = ["completeness", "accuracy", "consistency", "validity", "uniqueness", "timeliness"]
            
            dimension_analysis = {}
            
            for dimension in dimensions:
                try:
                    if dimension == "completeness":
                        analysis = self._analyze_completeness(data)
                    elif dimension == "accuracy":
                        analysis = self._analyze_accuracy(data)
                    elif dimension == "consistency":
                        analysis = self._analyze_consistency(data)
                    elif dimension == "validity":
                        analysis = self._analyze_validity(data)
                    elif dimension == "uniqueness":
                        analysis = self._analyze_uniqueness(data)
                    elif dimension == "timeliness":
                        analysis = self._analyze_timeliness(data)
                    else:
                        analysis = {"error": f"Unknown dimension: {dimension}"}
                    
                    dimension_analysis[dimension] = analysis
                    
                except Exception as e:
                    logger.error(f"Failed to analyze dimension {dimension}: {str(e)}")
                    dimension_analysis[dimension] = {"error": str(e)}
            
            return {
                "dimension_analysis": dimension_analysis,
                "analyzed_at": datetime.now().isoformat(),
                "total_dimensions": len(dimensions)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze quality dimensions: {str(e)}")
            return {"error": str(e)}
    
    def compare_quality_across_datasets(self,
                                      datasets: Dict[str, pd.DataFrame],
                                      metrics: List[str] = None) -> Dict[str, Any]:
        """Compare quality across multiple datasets"""
        try:
            if not datasets:
                return {"error": "No datasets provided for comparison"}
            
            if metrics is None:
                metrics = ["completeness", "accuracy", "consistency", "validity"]
            
            comparison_results = {
                "comparison_metadata": {
                    "compared_at": datetime.now().isoformat(),
                    "datasets": list(datasets.keys()),
                    "metrics": metrics
                },
                "dataset_scores": {},
                "rankings": {},
                "insights": []
            }
            
            # Calculate scores for each dataset
            for dataset_name, data in datasets.items():
                if data.empty:
                    comparison_results["dataset_scores"][dataset_name] = {
                        "error": "Empty dataset"
                    }
                    continue
                
                try:
                    quality_score = self.quality_assessor.assess_data_quality(data)
                    
                    comparison_results["dataset_scores"][dataset_name] = {
                        "overall_score": quality_score.overall_score,
                        "dimension_scores": {
                            dim.value: score for dim, score in quality_score.dimension_scores.items()
                        },
                        "data_size": len(data),
                        "columns": len(data.columns)
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to analyze dataset {dataset_name}: {str(e)}")
                    comparison_results["dataset_scores"][dataset_name] = {
                        "error": str(e)
                    }
            
            # Generate rankings
            valid_scores = {
                name: scores for name, scores in comparison_results["dataset_scores"].items()
                if "overall_score" in scores
            }
            
            if valid_scores:
                sorted_datasets = sorted(
                    valid_scores.items(),
                    key=lambda x: x[1]["overall_score"],
                    reverse=True
                )
                
                comparison_results["rankings"] = {
                    "by_overall_score": [name for name, _ in sorted_datasets],
                    "best_dataset": sorted_datasets[0][0] if sorted_datasets else None,
                    "worst_dataset": sorted_datasets[-1][0] if sorted_datasets else None
                }
                
                # Generate insights
                insights = self._generate_comparison_insights(valid_scores)
                comparison_results["insights"] = insights
            
            logger.info(f"Quality comparison completed for {len(datasets)} datasets")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Failed to compare quality across datasets: {str(e)}")
            return {"error": str(e)}
    
    def predict_quality_issues(self,
                             data: pd.DataFrame,
                             historical_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Predict potential quality issues"""
        try:
            if data.empty:
                return {"error": "No data provided for prediction"}
            
            predictions = {
                "prediction_metadata": {
                    "predicted_at": datetime.now().isoformat(),
                    "data_shape": data.shape
                },
                "quality_forecast": {},
                "risk_factors": [],
                "preventive_measures": []
            }
            
            # Analyze current quality
            current_quality = self.quality_assessor.assess_data_quality(data)
            
            # Predict future quality based on trends
            if historical_data is not None and not historical_data.empty:
                trend_analysis = self.trend_analyzer.analyze_trend(
                    historical_data, "quality_score"
                )
                
                predictions["quality_forecast"] = {
                    "trend_direction": trend_analysis.trend_direction.value,
                    "confidence": trend_analysis.confidence,
                    "forecast_values": trend_analysis.forecast_values,
                    "forecast_dates": [d.isoformat() for d in trend_analysis.forecast_dates]
                }
            
            # Identify risk factors
            risk_factors = self._identify_quality_risk_factors(data, current_quality)
            predictions["risk_factors"] = risk_factors
            
            # Suggest preventive measures
            preventive_measures = self._suggest_preventive_measures(
                current_quality, risk_factors
            )
            predictions["preventive_measures"] = preventive_measures
            
            logger.info("Quality issue prediction completed")
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict quality issues: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_quality_trends(self, data: pd.DataFrame, timestamp_col: str) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        try:
            # Create quality score time series
            data_with_timestamp = data.copy()
            data_with_timestamp = data_with_timestamp.sort_values(timestamp_col)
            
            # Calculate quality scores for each time period
            quality_scores = []
            timestamps = []
            
            # Group by day and calculate quality scores
            daily_groups = data_with_timestamp.groupby(
                data_with_timestamp[timestamp_col].dt.date
            )
            
            for date, group in daily_groups:
                if len(group) > 0:
                    quality_score = self.quality_assessor.assess_data_quality(group)
                    quality_scores.append(quality_score.overall_score)
                    timestamps.append(date)
            
            if len(quality_scores) < 2:
                return {"error": "Insufficient data for trend analysis"}
            
            # Create DataFrame for trend analysis
            trend_data = pd.DataFrame({
                "timestamp": timestamps,
                "quality_score": quality_scores
            })
            
            # Analyze trend
            trend_analysis = self.trend_analyzer.analyze_trend(
                trend_data, "quality_score", "timestamp"
            )
            
            return {
                "trend_direction": trend_analysis.trend_direction.value,
                "trend_type": trend_analysis.trend_type.value,
                "slope": trend_analysis.slope,
                "confidence": trend_analysis.confidence,
                "volatility": trend_analysis.volatility,
                "seasonality_detected": trend_analysis.seasonality_detected,
                "recommendations": trend_analysis.recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze quality trends: {str(e)}")
            return {"error": str(e)}
    
    def _generate_quality_insights(self,
                                 quality_score,
                                 validation_summary: Dict[str, Any],
                                 data: pd.DataFrame) -> List[QualityInsight]:
        """Generate quality insights"""
        try:
            insights = []
            
            # Overall quality insight
            if quality_score.overall_score < 0.7:
                insights.append(QualityInsight(
                    insight_type="overall_quality",
                    description=f"Overall data quality is low ({quality_score.overall_score:.3f})",
                    severity="high",
                    confidence=0.9,
                    recommendations=[
                        "Implement data quality controls at source",
                        "Review data collection processes",
                        "Consider data remediation"
                    ],
                    affected_data={"overall_score": quality_score.overall_score},
                    metadata={"data_size": len(data)}
                ))
            
            # Validation insights
            if validation_summary["failed_rules"] > 0:
                insights.append(QualityInsight(
                    insight_type="validation_failures",
                    description=f"{validation_summary['failed_rules']} validation rules failed",
                    severity="medium",
                    confidence=0.8,
                    recommendations=[
                        "Review failed validation rules",
                        "Update data validation logic",
                        "Implement automated data cleaning"
                    ],
                    affected_data={"failed_rules": validation_summary["failed_rules"]},
                    metadata={"total_rules": validation_summary["total_rules"]}
                ))
            
            # Dimension-specific insights
            for dimension, score in quality_score.dimension_scores.items():
                if score < 0.8:
                    insights.append(QualityInsight(
                        insight_type=f"{dimension.value}_quality",
                        description=f"{dimension.value} quality is below threshold ({score:.3f})",
                        severity="medium" if score > 0.6 else "high",
                        confidence=0.7,
                        recommendations=[
                            f"Improve {dimension.value} through data processing",
                            f"Implement {dimension.value} monitoring"
                        ],
                        affected_data={dimension.value: score},
                        metadata={"threshold": 0.8}
                    ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate quality insights: {str(e)}")
            return []
    
    def _generate_quality_recommendations(self,
                                        quality_score,
                                        validation_summary: Dict[str, Any],
                                        insights: List[QualityInsight]) -> List[str]:
        """Generate quality recommendations"""
        try:
            recommendations = []
            
            # Overall recommendations
            if quality_score.overall_score < 0.8:
                recommendations.append("Implement comprehensive data quality program")
                recommendations.append("Establish data quality metrics and monitoring")
            
            # Validation recommendations
            if validation_summary["error_count"] > 0:
                recommendations.append("Address critical validation errors immediately")
            
            if validation_summary["warning_count"] > 0:
                recommendations.append("Review and address validation warnings")
            
            # Insight-based recommendations
            for insight in insights:
                recommendations.extend(insight.recommendations)
            
            # Remove duplicates
            recommendations = list(set(recommendations))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate quality recommendations: {str(e)}")
            return ["Unable to generate recommendations"]
    
    def _analyze_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze completeness dimension"""
        try:
            total_cells = data.size
            null_cells = data.isnull().sum().sum()
            completeness_ratio = 1.0 - (null_cells / total_cells)
            
            # Column-level analysis
            column_completeness = {}
            for col in data.columns:
                null_count = data[col].isnull().sum()
                column_completeness[col] = {
                    "null_count": int(null_count),
                    "completeness_ratio": 1.0 - (null_count / len(data)),
                    "missing_percentage": (null_count / len(data)) * 100
                }
            
            return {
                "overall_completeness": completeness_ratio,
                "total_nulls": int(null_cells),
                "column_analysis": column_completeness,
                "recommendations": [
                    "Review data collection processes" if completeness_ratio < 0.9 else None,
                    "Implement null value handling" if null_cells > 0 else None
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze completeness: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze accuracy dimension"""
        try:
            accuracy_scores = []
            issues = []
            
            # Check numeric columns for outliers
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if len(data[col].dropna()) > 0:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                    outlier_ratio = outliers / len(data[col].dropna())
                    accuracy_scores.append(1.0 - outlier_ratio)
                    
                    if outlier_ratio > 0.05:  # More than 5% outliers
                        issues.append(f"High outlier ratio in {col}: {outlier_ratio:.3f}")
            
            overall_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.8
            
            return {
                "overall_accuracy": overall_accuracy,
                "outlier_issues": issues,
                "numeric_columns_analyzed": len(numeric_cols),
                "recommendations": [
                    "Review outlier detection logic" if issues else None,
                    "Implement data validation rules" if overall_accuracy < 0.9 else None
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze accuracy: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consistency dimension"""
        try:
            consistency_scores = []
            issues = []
            
            # Check for duplicates
            duplicates = data.duplicated().sum()
            duplicate_ratio = duplicates / len(data)
            consistency_scores.append(1.0 - duplicate_ratio)
            
            if duplicate_ratio > 0.05:
                issues.append(f"High duplicate ratio: {duplicate_ratio:.3f}")
            
            # Check string columns for format consistency
            string_cols = data.select_dtypes(include=['object']).columns
            for col in string_cols:
                if len(data[col].dropna()) > 0:
                    # Check case consistency
                    unique_cases = data[col].str.isupper().nunique()
                    if unique_cases > 1:
                        issues.append(f"Inconsistent case in {col}")
                        consistency_scores.append(0.8)
                    else:
                        consistency_scores.append(1.0)
            
            overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.8
            
            return {
                "overall_consistency": overall_consistency,
                "duplicate_count": int(duplicates),
                "consistency_issues": issues,
                "recommendations": [
                    "Implement deduplication process" if duplicate_ratio > 0.05 else None,
                    "Standardize data formats" if issues else None
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze consistency: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze validity dimension"""
        try:
            validity_scores = []
            issues = []
            
            for col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    # Check data types
                    if data[col].dtype in ['int64', 'float64']:
                        # Check for non-numeric values
                        non_numeric = pd.to_numeric(col_data, errors='coerce').isnull().sum()
                        if non_numeric > 0:
                            issues.append(f"Non-numeric values in numeric column {col}: {non_numeric}")
                            validity_scores.append(0.5)
                        else:
                            validity_scores.append(1.0)
                    else:
                        validity_scores.append(1.0)
            
            overall_validity = np.mean(validity_scores) if validity_scores else 0.8
            
            return {
                "overall_validity": overall_validity,
                "validity_issues": issues,
                "recommendations": [
                    "Implement data type validation" if issues else None,
                    "Review data conversion processes" if overall_validity < 0.9 else None
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze validity: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_uniqueness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze uniqueness dimension"""
        try:
            total_records = len(data)
            unique_records = len(data.drop_duplicates())
            uniqueness_ratio = unique_records / total_records
            
            return {
                "overall_uniqueness": uniqueness_ratio,
                "duplicate_records": total_records - unique_records,
                "recommendations": [
                    "Implement deduplication process" if uniqueness_ratio < 0.95 else None,
                    "Review data integration logic" if uniqueness_ratio < 0.9 else None
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze uniqueness: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_timeliness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze timeliness dimension"""
        try:
            timestamp_cols = data.select_dtypes(include=['datetime64']).columns
            
            if len(timestamp_cols) == 0:
                return {
                    "overall_timeliness": 0.8,  # Assume good if no timestamps
                    "timestamp_columns": 0,
                    "recommendations": ["Add timestamp columns for timeliness analysis"]
                }
            
            timeliness_scores = []
            for col in timestamp_cols:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    now = datetime.now()
                    latest_timestamp = col_data.max()
                    days_old = (now - latest_timestamp).days
                    freshness_score = max(0, 1.0 - (days_old / 30))  # 30 days threshold
                    timeliness_scores.append(freshness_score)
            
            overall_timeliness = np.mean(timeliness_scores) if timeliness_scores else 0.8
            
            return {
                "overall_timeliness": overall_timeliness,
                "timestamp_columns": len(timestamp_cols),
                "recommendations": [
                    "Increase data update frequency" if overall_timeliness < 0.8 else None,
                    "Implement real-time data processing" if overall_timeliness < 0.6 else None
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze timeliness: {str(e)}")
            return {"error": str(e)}
    
    def _generate_comparison_insights(self, dataset_scores: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate insights from dataset comparison"""
        try:
            insights = []
            
            if len(dataset_scores) < 2:
                return insights
            
            # Find best and worst datasets
            scores = [(name, data["overall_score"]) for name, data in dataset_scores.items()]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            best_dataset, best_score = scores[0]
            worst_dataset, worst_score = scores[-1]
            
            score_difference = best_score - worst_score
            
            if score_difference > 0.2:
                insights.append(f"Significant quality difference between datasets: {best_dataset} ({best_score:.3f}) vs {worst_dataset} ({worst_score:.3f})")
            
            # Check for consistent patterns
            dimension_scores = {}
            for dataset_name, data in dataset_scores.items():
                for dimension, score in data["dimension_scores"].items():
                    if dimension not in dimension_scores:
                        dimension_scores[dimension] = []
                    dimension_scores[dimension].append(score)
            
            for dimension, scores in dimension_scores.items():
                if len(scores) > 1:
                    score_variance = np.var(scores)
                    if score_variance > 0.1:  # High variance
                        insights.append(f"High variance in {dimension} across datasets (variance: {score_variance:.3f})")
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate comparison insights: {str(e)}")
            return []
    
    def _identify_quality_risk_factors(self, data: pd.DataFrame, quality_score) -> List[str]:
        """Identify quality risk factors"""
        try:
            risk_factors = []
            
            # Data size risks
            if len(data) < 100:
                risk_factors.append("Small dataset size may affect quality assessment reliability")
            
            # Quality score risks
            if quality_score.overall_score < 0.7:
                risk_factors.append("Low overall quality score indicates data quality issues")
            
            # Dimension-specific risks
            for dimension, score in quality_score.dimension_scores.items():
                if score < 0.8:
                    risk_factors.append(f"Low {dimension.value} score ({score:.3f}) poses quality risk")
            
            # Data type risks
            mixed_type_cols = []
            for col in data.columns:
                if data[col].dtype == 'object':
                    # Check if column contains mixed types
                    unique_types = data[col].apply(type).nunique()
                    if unique_types > 1:
                        mixed_type_cols.append(col)
            
            if mixed_type_cols:
                risk_factors.append(f"Mixed data types in columns: {', '.join(mixed_type_cols)}")
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Failed to identify quality risk factors: {str(e)}")
            return []
    
    def _suggest_preventive_measures(self, quality_score, risk_factors: List[str]) -> List[str]:
        """Suggest preventive measures for quality issues"""
        try:
            measures = []
            
            # General measures
            if quality_score.overall_score < 0.8:
                measures.append("Implement data quality monitoring and alerting")
                measures.append("Establish data quality standards and guidelines")
                measures.append("Train data collection teams on quality requirements")
            
            # Risk-specific measures
            for risk in risk_factors:
                if "Small dataset" in risk:
                    measures.append("Increase data collection volume")
                elif "Low overall quality" in risk:
                    measures.append("Implement comprehensive data quality program")
                elif "Mixed data types" in risk:
                    measures.append("Standardize data types and validation rules")
                elif "Low" in risk and "score" in risk:
                    measures.append("Implement targeted quality improvement for affected dimensions")
            
            # Proactive measures
            measures.extend([
                "Implement automated data validation at ingestion",
                "Set up quality dashboards and reporting",
                "Establish data quality review processes",
                "Create data quality incident response procedures"
            ])
            
            return list(set(measures))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Failed to suggest preventive measures: {str(e)}")
            return []
