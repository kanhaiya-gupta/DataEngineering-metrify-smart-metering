"""
Quality Forecasting
Predictive quality analysis and forecasting
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
from src.governance.quality import QualityAssessor, TrendAnalyzer, QualityMonitor

logger = logging.getLogger(__name__)

class ForecastMethod(Enum):
    """Forecasting methods"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SEASONAL = "seasonal"
    ARIMA = "arima"
    LSTM = "lstm"

class QualityForecastType(Enum):
    """Quality forecast types"""
    DIMENSION_SCORE = "dimension_score"
    OVERALL_SCORE = "overall_score"
    ISSUE_COUNT = "issue_count"
    TREND_DIRECTION = "trend_direction"

@dataclass
class QualityForecast:
    """Quality forecast result"""
    forecast_type: QualityForecastType
    method: ForecastMethod
    forecast_values: List[float]
    forecast_dates: List[datetime]
    confidence: float
    accuracy_metrics: Dict[str, float]
    recommendations: List[str]

class QualityForecasting:
    """
    Predictive quality analysis and forecasting
    """
    
    def __init__(self):
        self.quality_assessor = QualityAssessor()
        self.trend_analyzer = TrendAnalyzer()
        self.quality_monitor = QualityMonitor()
        
        logger.info("QualityForecasting initialized")
    
    def forecast_quality_trends(self,
                              historical_data: pd.DataFrame,
                              forecast_days: int = 30,
                              method: ForecastMethod = ForecastMethod.LINEAR) -> Dict[str, Any]:
        """Forecast quality trends based on historical data"""
        try:
            if historical_data.empty:
                return {"error": "No historical data provided"}
            
            # Ensure we have timestamp column
            timestamp_cols = historical_data.select_dtypes(include=['datetime64']).columns
            if len(timestamp_cols) == 0:
                return {"error": "No timestamp column found in historical data"}
            
            timestamp_col = timestamp_cols[0]
            
            # Calculate quality scores for historical data
            quality_history = self._calculate_quality_history(historical_data, timestamp_col)
            
            if len(quality_history) < 5:
                return {"error": "Insufficient historical data for forecasting"}
            
            # Generate forecasts for different quality metrics
            forecasts = {}
            
            # Overall quality score forecast
            overall_forecast = self._forecast_metric(
                quality_history, "overall_score", forecast_days, method
            )
            forecasts["overall_quality"] = overall_forecast
            
            # Dimension-specific forecasts
            dimensions = ["completeness", "accuracy", "consistency", "validity", "uniqueness", "timeliness"]
            for dimension in dimensions:
                if dimension in quality_history.columns:
                    dim_forecast = self._forecast_metric(
                        quality_history, dimension, forecast_days, method
                    )
                    forecasts[dimension] = dim_forecast
            
            # Issue count forecast
            if "issue_count" in quality_history.columns:
                issue_forecast = self._forecast_metric(
                    quality_history, "issue_count", forecast_days, method
                )
                forecasts["issue_count"] = issue_forecast
            
            # Generate insights and recommendations
            insights = self._generate_forecast_insights(forecasts)
            recommendations = self._generate_forecast_recommendations(forecasts)
            
            return {
                "forecast_metadata": {
                    "forecasted_at": datetime.now().isoformat(),
                    "forecast_days": forecast_days,
                    "method": method.value,
                    "historical_data_points": len(quality_history)
                },
                "forecasts": forecasts,
                "insights": insights,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to forecast quality trends: {str(e)}")
            return {"error": str(e)}
    
    def predict_quality_issues(self,
                             current_data: pd.DataFrame,
                             historical_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Predict potential quality issues"""
        try:
            if current_data.empty:
                return {"error": "No current data provided"}
            
            predictions = {
                "prediction_metadata": {
                    "predicted_at": datetime.now().isoformat(),
                    "current_data_shape": current_data.shape
                },
                "quality_risks": [],
                "issue_probabilities": {},
                "preventive_actions": []
            }
            
            # Analyze current quality
            current_quality = self.quality_assessor.assess_data_quality(current_data)
            
            # Predict based on current quality patterns
            quality_risks = self._predict_quality_risks(current_quality, current_data)
            predictions["quality_risks"] = quality_risks
            
            # Calculate issue probabilities
            issue_probabilities = self._calculate_issue_probabilities(current_quality)
            predictions["issue_probabilities"] = issue_probabilities
            
            # Historical trend analysis
            if historical_data is not None and not historical_data.empty:
                trend_insights = self._analyze_historical_trends(historical_data)
                predictions["trend_insights"] = trend_insights
            
            # Suggest preventive actions
            preventive_actions = self._suggest_preventive_actions(
                current_quality, quality_risks, issue_probabilities
            )
            predictions["preventive_actions"] = preventive_actions
            
            logger.info("Quality issue prediction completed")
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict quality issues: {str(e)}")
            return {"error": str(e)}
    
    def forecast_data_volume_impact(self,
                                  current_data: pd.DataFrame,
                                  expected_volume_increase: float,
                                  forecast_days: int = 30) -> Dict[str, Any]:
        """Forecast impact of data volume changes on quality"""
        try:
            if current_data.empty:
                return {"error": "No current data provided"}
            
            # Analyze current quality
            current_quality = self.quality_assessor.assess_data_quality(current_data)
            
            # Simulate volume increase impact
            volume_impact = self._simulate_volume_impact(
                current_quality, expected_volume_increase
            )
            
            # Generate volume-based forecasts
            forecasts = {}
            for dimension, score in current_quality.dimension_scores.items():
                impact_factor = volume_impact.get(dimension.value, 1.0)
                projected_score = score * impact_factor
                
                forecasts[dimension.value] = {
                    "current_score": score,
                    "projected_score": projected_score,
                    "impact_factor": impact_factor,
                    "change": projected_score - score
                }
            
            # Generate recommendations
            recommendations = self._generate_volume_impact_recommendations(
                volume_impact, expected_volume_increase
            )
            
            return {
                "volume_impact_metadata": {
                    "analyzed_at": datetime.now().isoformat(),
                    "current_volume": len(current_data),
                    "expected_increase": expected_volume_increase,
                    "forecast_days": forecast_days
                },
                "volume_impact": volume_impact,
                "quality_forecasts": forecasts,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to forecast volume impact: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_quality_history(self,
                                 data: pd.DataFrame,
                                 timestamp_col: str) -> pd.DataFrame:
        """Calculate quality scores over time"""
        try:
            # Sort by timestamp
            data_sorted = data.sort_values(timestamp_col)
            
            # Group by day
            daily_groups = data_sorted.groupby(
                data_sorted[timestamp_col].dt.date
            )
            
            quality_history = []
            
            for date, group in daily_groups:
                if len(group) > 10:  # Minimum group size
                    quality_score = self.quality_assessor.assess_data_quality(group)
                    
                    quality_record = {
                        "date": date,
                        "overall_score": quality_score.overall_score,
                        "issue_count": quality_score.issues_found,
                        "data_size": len(group)
                    }
                    
                    # Add dimension scores
                    for dimension, score in quality_score.dimension_scores.items():
                        quality_record[dimension.value] = score
                    
                    quality_history.append(quality_record)
            
            return pd.DataFrame(quality_history)
            
        except Exception as e:
            logger.error(f"Failed to calculate quality history: {str(e)}")
            return pd.DataFrame()
    
    def _forecast_metric(self,
                        data: pd.DataFrame,
                        metric: str,
                        forecast_days: int,
                        method: ForecastMethod) -> QualityForecast:
        """Forecast a specific metric"""
        try:
            if metric not in data.columns:
                return self._create_empty_forecast(metric, method)
            
            # Prepare data for forecasting
            values = data[metric].dropna().values
            dates = data["date"].values
            
            if len(values) < 3:
                return self._create_empty_forecast(metric, method)
            
            # Generate forecast based on method
            if method == ForecastMethod.LINEAR:
                forecast_values, forecast_dates, confidence = self._linear_forecast(
                    values, dates, forecast_days
                )
            elif method == ForecastMethod.EXPONENTIAL:
                forecast_values, forecast_dates, confidence = self._exponential_forecast(
                    values, dates, forecast_days
                )
            else:
                # Default to linear
                forecast_values, forecast_dates, confidence = self._linear_forecast(
                    values, dates, forecast_days
                )
            
            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics(values, forecast_values[:len(values)])
            
            # Generate recommendations
            recommendations = self._generate_forecast_recommendations_for_metric(
                metric, forecast_values, confidence
            )
            
            return QualityForecast(
                forecast_type=QualityForecastType.DIMENSION_SCORE,
                method=method,
                forecast_values=forecast_values,
                forecast_dates=forecast_dates,
                confidence=confidence,
                accuracy_metrics=accuracy_metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to forecast metric {metric}: {str(e)}")
            return self._create_error_forecast(metric, method, str(e))
    
    def _linear_forecast(self,
                        values: np.ndarray,
                        dates: np.ndarray,
                        forecast_days: int) -> Tuple[List[float], List[datetime], float]:
        """Linear forecasting method"""
        try:
            # Linear regression
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # Generate forecast
            last_date = pd.to_datetime(dates[-1])
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
            
            last_x = len(values) - 1
            forecast_x = np.arange(last_x + 1, last_x + 1 + forecast_days)
            forecast_values = slope * forecast_x + intercept
            
            # Calculate confidence based on R-squared
            y_pred = slope * x + intercept
            r_squared = 1 - (np.sum((values - y_pred) ** 2) / np.sum((values - np.mean(values)) ** 2))
            confidence = max(0.0, min(1.0, r_squared))
            
            return forecast_values.tolist(), forecast_dates, confidence
            
        except Exception as e:
            logger.error(f"Failed linear forecast: {str(e)}")
            return [], [], 0.0
    
    def _exponential_forecast(self,
                             values: np.ndarray,
                             dates: np.ndarray,
                             forecast_days: int) -> Tuple[List[float], List[datetime], float]:
        """Exponential forecasting method"""
        try:
            # Simple exponential smoothing
            alpha = 0.3  # Smoothing parameter
            forecast_values = []
            
            # Initialize with first value
            forecast_values.append(values[0])
            
            # Apply exponential smoothing
            for i in range(1, len(values)):
                forecast_values.append(alpha * values[i] + (1 - alpha) * forecast_values[i-1])
            
            # Generate future forecasts
            last_forecast = forecast_values[-1]
            last_date = pd.to_datetime(dates[-1])
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
            
            # Extend forecast (simplified - just use last value)
            future_forecasts = [last_forecast] * forecast_days
            
            # Calculate confidence based on recent accuracy
            recent_errors = np.abs(values[-5:] - forecast_values[-5:]) if len(values) >= 5 else np.abs(values - forecast_values)
            mae = np.mean(recent_errors)
            confidence = max(0.0, min(1.0, 1 - (mae / np.mean(values))))
            
            return future_forecasts, forecast_dates, confidence
            
        except Exception as e:
            logger.error(f"Failed exponential forecast: {str(e)}")
            return [], [], 0.0
    
    def _calculate_accuracy_metrics(self,
                                  actual: np.ndarray,
                                  predicted: np.ndarray) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        try:
            if len(actual) != len(predicted):
                min_len = min(len(actual), len(predicted))
                actual = actual[:min_len]
                predicted = predicted[:min_len]
            
            mae = np.mean(np.abs(actual - predicted))
            mse = np.mean((actual - predicted) ** 2)
            rmse = np.sqrt(mse)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if np.all(actual != 0) else 0
            
            return {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "mape": float(mape)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate accuracy metrics: {str(e)}")
            return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "mape": 0.0}
    
    def _predict_quality_risks(self, quality_score, data: pd.DataFrame) -> List[str]:
        """Predict quality risks based on current state"""
        try:
            risks = []
            
            # Overall quality risk
            if quality_score.overall_score < 0.7:
                risks.append("Critical quality degradation risk")
            elif quality_score.overall_score < 0.8:
                risks.append("Quality decline risk")
            
            # Dimension-specific risks
            for dimension, score in quality_score.dimension_scores.items():
                if score < 0.8:
                    risks.append(f"High {dimension.value} quality risk")
                elif score < 0.9:
                    risks.append(f"Medium {dimension.value} quality risk")
            
            # Data size risks
            if len(data) < 100:
                risks.append("Small dataset size may affect quality stability")
            
            # Volatility risk (simplified)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                volatility_scores = []
                for col in numeric_cols:
                    if len(data[col].dropna()) > 1:
                        cv = data[col].std() / data[col].mean() if data[col].mean() != 0 else 0
                        volatility_scores.append(cv)
                
                if volatility_scores and np.mean(volatility_scores) > 0.5:
                    risks.append("High data volatility may cause quality instability")
            
            return risks
            
        except Exception as e:
            logger.error(f"Failed to predict quality risks: {str(e)}")
            return []
    
    def _calculate_issue_probabilities(self, quality_score) -> Dict[str, float]:
        """Calculate probabilities of different quality issues"""
        try:
            probabilities = {}
            
            # Overall quality issue probability
            if quality_score.overall_score < 0.5:
                probabilities["critical_quality_issue"] = 0.9
            elif quality_score.overall_score < 0.7:
                probabilities["quality_degradation"] = 0.7
            elif quality_score.overall_score < 0.8:
                probabilities["quality_degradation"] = 0.4
            else:
                probabilities["quality_degradation"] = 0.1
            
            # Dimension-specific probabilities
            for dimension, score in quality_score.dimension_scores.items():
                if score < 0.5:
                    probabilities[f"{dimension.value}_critical_issue"] = 0.8
                elif score < 0.7:
                    probabilities[f"{dimension.value}_issue"] = 0.6
                elif score < 0.8:
                    probabilities[f"{dimension.value}_issue"] = 0.3
                else:
                    probabilities[f"{dimension.value}_issue"] = 0.1
            
            # Issue count probability
            if quality_score.issues_found > 10:
                probabilities["high_issue_count"] = 0.8
            elif quality_score.issues_found > 5:
                probabilities["high_issue_count"] = 0.5
            else:
                probabilities["high_issue_count"] = 0.2
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Failed to calculate issue probabilities: {str(e)}")
            return {}
    
    def _analyze_historical_trends(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze historical quality trends"""
        try:
            # Calculate quality history
            timestamp_cols = historical_data.select_dtypes(include=['datetime64']).columns
            if len(timestamp_cols) == 0:
                return {"error": "No timestamp column found"}
            
            quality_history = self._calculate_quality_history(historical_data, timestamp_cols[0])
            
            if len(quality_history) < 3:
                return {"error": "Insufficient historical data"}
            
            # Analyze trends
            trend_analysis = self.trend_analyzer.analyze_trend(
                quality_history, "overall_score", "date"
            )
            
            return {
                "trend_direction": trend_analysis.trend_direction.value,
                "trend_type": trend_analysis.trend_type.value,
                "confidence": trend_analysis.confidence,
                "volatility": trend_analysis.volatility,
                "seasonality_detected": trend_analysis.seasonality_detected,
                "recommendations": trend_analysis.recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze historical trends: {str(e)}")
            return {"error": str(e)}
    
    def _suggest_preventive_actions(self,
                                  quality_score,
                                  risks: List[str],
                                  probabilities: Dict[str, float]) -> List[str]:
        """Suggest preventive actions based on predictions"""
        try:
            actions = []
            
            # Risk-based actions
            for risk in risks:
                if "Critical quality" in risk:
                    actions.append("Implement immediate quality remediation")
                    actions.append("Increase quality monitoring frequency")
                elif "Quality decline" in risk:
                    actions.append("Review and improve data collection processes")
                    actions.append("Implement quality gates")
                elif "High" in risk and "quality risk" in risk:
                    actions.append(f"Focus on improving {risk.split()[1]} dimension")
            
            # Probability-based actions
            for issue, prob in probabilities.items():
                if prob > 0.7:
                    actions.append(f"High probability of {issue} - implement preventive measures")
                elif prob > 0.4:
                    actions.append(f"Medium probability of {issue} - monitor closely")
            
            # General preventive actions
            actions.extend([
                "Implement automated quality monitoring",
                "Set up quality alerting thresholds",
                "Establish quality review processes",
                "Create quality incident response procedures"
            ])
            
            return list(set(actions))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Failed to suggest preventive actions: {str(e)}")
            return []
    
    def _simulate_volume_impact(self, quality_score, volume_increase: float) -> Dict[str, float]:
        """Simulate impact of volume increase on quality"""
        try:
            impact_factors = {}
            
            # Simulate impact on different dimensions
            # Higher volume typically affects completeness and timeliness more
            impact_factors["completeness"] = max(0.5, 1.0 - (volume_increase * 0.1))
            impact_factors["timeliness"] = max(0.5, 1.0 - (volume_increase * 0.15))
            impact_factors["accuracy"] = max(0.7, 1.0 - (volume_increase * 0.05))
            impact_factors["consistency"] = max(0.8, 1.0 - (volume_increase * 0.08))
            impact_factors["validity"] = max(0.8, 1.0 - (volume_increase * 0.06))
            impact_factors["uniqueness"] = max(0.9, 1.0 - (volume_increase * 0.03))
            
            return impact_factors
            
        except Exception as e:
            logger.error(f"Failed to simulate volume impact: {str(e)}")
            return {}
    
    def _generate_volume_impact_recommendations(self,
                                              volume_impact: Dict[str, float],
                                              volume_increase: float) -> List[str]:
        """Generate recommendations for volume impact"""
        try:
            recommendations = []
            
            # Overall recommendations
            if volume_increase > 0.5:  # 50% increase
                recommendations.append("Implement scalable data processing infrastructure")
                recommendations.append("Consider data partitioning strategies")
            
            # Dimension-specific recommendations
            for dimension, impact in volume_impact.items():
                if impact < 0.8:
                    recommendations.append(f"Strengthen {dimension} controls for higher volume")
                    recommendations.append(f"Implement {dimension} monitoring and alerting")
            
            # General recommendations
            recommendations.extend([
                "Scale up quality monitoring systems",
                "Implement automated quality checks",
                "Prepare for increased data processing load",
                "Review and optimize data collection processes"
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate volume impact recommendations: {str(e)}")
            return []
    
    def _generate_forecast_insights(self, forecasts: Dict[str, QualityForecast]) -> List[str]:
        """Generate insights from forecasts"""
        try:
            insights = []
            
            for metric, forecast in forecasts.items():
                if forecast.confidence > 0.7:
                    trend = "improving" if forecast.forecast_values[-1] > forecast.forecast_values[0] else "declining"
                    insights.append(f"{metric} is forecasted to be {trend} with {forecast.confidence:.1%} confidence")
                elif forecast.confidence < 0.5:
                    insights.append(f"{metric} forecast has low confidence - more data needed")
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate forecast insights: {str(e)}")
            return []
    
    def _generate_forecast_recommendations(self, forecasts: Dict[str, QualityForecast]) -> List[str]:
        """Generate recommendations from forecasts"""
        try:
            recommendations = []
            
            for metric, forecast in forecasts.items():
                if forecast.confidence > 0.7:
                    if forecast.forecast_values[-1] < 0.8:
                        recommendations.append(f"Take action to improve {metric} before it degrades further")
                    elif forecast.forecast_values[-1] > 0.9:
                        recommendations.append(f"Maintain current practices for {metric}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate forecast recommendations: {str(e)}")
            return []
    
    def _generate_forecast_recommendations_for_metric(self,
                                                    metric: str,
                                                    forecast_values: List[float],
                                                    confidence: float) -> List[str]:
        """Generate recommendations for a specific metric forecast"""
        try:
            recommendations = []
            
            if confidence < 0.5:
                recommendations.append(f"Insufficient data for reliable {metric} forecasting")
                recommendations.append("Collect more historical data for better predictions")
            else:
                if forecast_values[-1] < 0.7:
                    recommendations.append(f"Implement measures to improve {metric}")
                elif forecast_values[-1] > 0.9:
                    recommendations.append(f"Maintain current {metric} practices")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations for {metric}: {str(e)}")
            return []
    
    def _create_empty_forecast(self, metric: str, method: ForecastMethod) -> QualityForecast:
        """Create empty forecast for missing data"""
        return QualityForecast(
            forecast_type=QualityForecastType.DIMENSION_SCORE,
            method=method,
            forecast_values=[],
            forecast_dates=[],
            confidence=0.0,
            accuracy_metrics={},
            recommendations=[f"Insufficient data for {metric} forecasting"]
        )
    
    def _create_error_forecast(self, metric: str, method: ForecastMethod, error: str) -> QualityForecast:
        """Create error forecast"""
        return QualityForecast(
            forecast_type=QualityForecastType.DIMENSION_SCORE,
            method=method,
            forecast_values=[],
            forecast_dates=[],
            confidence=0.0,
            accuracy_metrics={},
            recommendations=[f"Forecast failed for {metric}: {error}"]
        )
