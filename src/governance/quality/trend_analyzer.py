"""
Trend Analyzer
Quality trend analysis and forecasting
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

class TrendDirection(Enum):
    """Trend direction"""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"

class TrendType(Enum):
    """Trend type"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"

@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    metric_name: str
    trend_direction: TrendDirection
    trend_type: TrendType
    slope: float
    r_squared: float
    confidence: float
    forecast_values: List[float]
    forecast_dates: List[datetime]
    seasonality_detected: bool
    volatility: float
    recommendations: List[str]

class TrendAnalyzer:
    """
    Quality trend analysis and forecasting
    """
    
    def __init__(self):
        self.min_data_points = 5
        self.forecast_periods = 7  # days
        
        logger.info("TrendAnalyzer initialized")
    
    def analyze_trend(self,
                     data: pd.DataFrame,
                     metric_column: str,
                     date_column: str = "timestamp",
                     forecast_days: int = 7) -> TrendAnalysis:
        """Analyze trend for a specific metric"""
        try:
            if data.empty or metric_column not in data.columns:
                return self._create_empty_analysis(metric_column)
            
            # Prepare data
            df = data[[date_column, metric_column]].copy()
            df = df.dropna()
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
            
            if len(df) < self.min_data_points:
                return self._create_insufficient_data_analysis(metric_column)
            
            # Calculate trend components
            trend_direction = self._calculate_trend_direction(df[metric_column])
            trend_type = self._detect_trend_type(df[metric_column])
            slope, r_squared = self._calculate_linear_trend(df[metric_column])
            confidence = self._calculate_confidence(df[metric_column], r_squared)
            
            # Generate forecast
            forecast_values, forecast_dates = self._generate_forecast(
                df, metric_column, date_column, forecast_days
            )
            
            # Detect seasonality
            seasonality_detected = self._detect_seasonality(df[metric_column])
            
            # Calculate volatility
            volatility = self._calculate_volatility(df[metric_column])
            
            # Generate recommendations
            recommendations = self._generate_trend_recommendations(
                trend_direction, trend_type, slope, confidence, volatility
            )
            
            analysis = TrendAnalysis(
                metric_name=metric_column,
                trend_direction=trend_direction,
                trend_type=trend_type,
                slope=slope,
                r_squared=r_squared,
                confidence=confidence,
                forecast_values=forecast_values,
                forecast_dates=forecast_dates,
                seasonality_detected=seasonality_detected,
                volatility=volatility,
                recommendations=recommendations
            )
            
            logger.info(f"Trend analysis completed for {metric_column}: {trend_direction.value}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze trend for {metric_column}: {str(e)}")
            return self._create_error_analysis(metric_column, str(e))
    
    def _calculate_trend_direction(self, values: pd.Series) -> TrendDirection:
        """Calculate trend direction"""
        try:
            if len(values) < 2:
                return TrendDirection.STABLE
            
            # Calculate slope
            x = np.arange(len(values))
            y = values.values
            
            # Linear regression
            slope = np.polyfit(x, y, 1)[0]
            
            # Calculate change percentage
            first_value = values.iloc[0]
            last_value = values.iloc[-1]
            change_pct = (last_value - first_value) / first_value if first_value != 0 else 0
            
            # Calculate volatility
            volatility = values.std() / values.mean() if values.mean() != 0 else 0
            
            # Determine direction
            if volatility > 0.3:  # High volatility threshold
                return TrendDirection.VOLATILE
            elif abs(change_pct) < 0.05:  # Less than 5% change
                return TrendDirection.STABLE
            elif slope > 0:
                return TrendDirection.IMPROVING
            else:
                return TrendDirection.DECLINING
                
        except Exception as e:
            logger.error(f"Failed to calculate trend direction: {str(e)}")
            return TrendDirection.STABLE
    
    def _detect_trend_type(self, values: pd.Series) -> TrendType:
        """Detect trend type"""
        try:
            if len(values) < 3:
                return TrendType.LINEAR
            
            # Test for linear trend
            x = np.arange(len(values))
            y = values.values
            
            # Linear regression
            linear_slope, linear_intercept = np.polyfit(x, y, 1)
            linear_pred = linear_slope * x + linear_intercept
            linear_r2 = 1 - (np.sum((y - linear_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            # Test for exponential trend
            if np.all(y > 0):  # Only if all values are positive
                log_y = np.log(y)
                exp_slope, exp_intercept = np.polyfit(x, log_y, 1)
                exp_pred = np.exp(exp_slope * x + exp_intercept)
                exp_r2 = 1 - (np.sum((y - exp_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
            else:
                exp_r2 = 0
            
            # Test for seasonality (simplified)
            seasonal_r2 = self._test_seasonality(values)
            
            # Determine best fit
            r2_scores = {
                TrendType.LINEAR: linear_r2,
                TrendType.EXPONENTIAL: exp_r2,
                TrendType.SEASONAL: seasonal_r2
            }
            
            best_type = max(r2_scores, key=r2_scores.get)
            
            # If no clear winner, default to linear
            if r2_scores[best_type] < 0.5:
                return TrendType.LINEAR
            
            return best_type
            
        except Exception as e:
            logger.error(f"Failed to detect trend type: {str(e)}")
            return TrendType.LINEAR
    
    def _test_seasonality(self, values: pd.Series) -> float:
        """Test for seasonality (simplified)"""
        try:
            if len(values) < 7:  # Need at least a week of data
                return 0.0
            
            # Simple seasonality test: check for weekly patterns
            weekly_avg = []
            for i in range(7):
                day_values = values.iloc[i::7]
                if len(day_values) > 0:
                    weekly_avg.append(day_values.mean())
                else:
                    weekly_avg.append(0)
            
            # Calculate variance in weekly averages
            if len(weekly_avg) > 1:
                seasonality_score = 1 - (np.var(weekly_avg) / np.mean(weekly_avg) ** 2) if np.mean(weekly_avg) != 0 else 0
                return max(0, seasonality_score)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to test seasonality: {str(e)}")
            return 0.0
    
    def _calculate_linear_trend(self, values: pd.Series) -> Tuple[float, float]:
        """Calculate linear trend slope and R-squared"""
        try:
            x = np.arange(len(values))
            y = values.values
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return slope, r_squared
            
        except Exception as e:
            logger.error(f"Failed to calculate linear trend: {str(e)}")
            return 0.0, 0.0
    
    def _calculate_confidence(self, values: pd.Series, r_squared: float) -> float:
        """Calculate confidence in trend analysis"""
        try:
            # Base confidence on R-squared
            base_confidence = r_squared
            
            # Adjust for data size
            size_factor = min(1.0, len(values) / 30)  # More data = higher confidence
            
            # Adjust for volatility
            volatility = values.std() / values.mean() if values.mean() != 0 else 1
            volatility_factor = max(0.5, 1 - volatility)  # Lower volatility = higher confidence
            
            confidence = base_confidence * size_factor * volatility_factor
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {str(e)}")
            return 0.5
    
    def _generate_forecast(self,
                          df: pd.DataFrame,
                          metric_column: str,
                          date_column: str,
                          forecast_days: int) -> Tuple[List[float], List[datetime]]:
        """Generate forecast values"""
        try:
            if len(df) < 2:
                return [], []
            
            # Simple linear forecast
            x = np.arange(len(df))
            y = df[metric_column].values
            
            slope, intercept = np.polyfit(x, y, 1)
            
            # Generate forecast
            last_date = df[date_column].iloc[-1]
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
            
            last_x = len(df) - 1
            forecast_x = np.arange(last_x + 1, last_x + 1 + forecast_days)
            forecast_values = slope * forecast_x + intercept
            
            return forecast_values.tolist(), forecast_dates
            
        except Exception as e:
            logger.error(f"Failed to generate forecast: {str(e)}")
            return [], []
    
    def _detect_seasonality(self, values: pd.Series) -> bool:
        """Detect if data has seasonal patterns"""
        try:
            if len(values) < 14:  # Need at least 2 weeks
                return False
            
            # Simple seasonality detection
            # Check if there's a weekly pattern
            weekly_means = []
            for i in range(7):
                day_values = values.iloc[i::7]
                if len(day_values) > 0:
                    weekly_means.append(day_values.mean())
            
            if len(weekly_means) < 2:
                return False
            
            # Calculate coefficient of variation
            cv = np.std(weekly_means) / np.mean(weekly_means) if np.mean(weekly_means) != 0 else 0
            
            # If CV > 0.1, consider it seasonal
            return cv > 0.1
            
        except Exception as e:
            logger.error(f"Failed to detect seasonality: {str(e)}")
            return False
    
    def _calculate_volatility(self, values: pd.Series) -> float:
        """Calculate data volatility"""
        try:
            if len(values) < 2:
                return 0.0
            
            # Calculate coefficient of variation
            mean_val = values.mean()
            std_val = values.std()
            
            if mean_val == 0:
                return 0.0
            
            return std_val / mean_val
            
        except Exception as e:
            logger.error(f"Failed to calculate volatility: {str(e)}")
            return 0.0
    
    def _generate_trend_recommendations(self,
                                      trend_direction: TrendDirection,
                                      trend_type: TrendType,
                                      slope: float,
                                      confidence: float,
                                      volatility: float) -> List[str]:
        """Generate recommendations based on trend analysis"""
        try:
            recommendations = []
            
            # Trend direction recommendations
            if trend_direction == TrendDirection.DECLINING:
                recommendations.append("Quality is declining - investigate root causes")
                recommendations.append("Implement immediate corrective measures")
            elif trend_direction == TrendDirection.IMPROVING:
                recommendations.append("Quality is improving - maintain current practices")
                recommendations.append("Consider scaling successful improvements")
            elif trend_direction == TrendDirection.VOLATILE:
                recommendations.append("Quality is volatile - investigate instability causes")
                recommendations.append("Implement monitoring and alerting")
            elif trend_direction == TrendDirection.STABLE:
                recommendations.append("Quality is stable - monitor for changes")
            
            # Confidence-based recommendations
            if confidence < 0.5:
                recommendations.append("Low confidence in trend - collect more data")
                recommendations.append("Consider increasing monitoring frequency")
            
            # Volatility recommendations
            if volatility > 0.3:
                recommendations.append("High volatility detected - investigate causes")
                recommendations.append("Consider implementing smoothing techniques")
            
            # Trend type recommendations
            if trend_type == TrendType.SEASONAL:
                recommendations.append("Seasonal patterns detected - adjust for seasonality")
                recommendations.append("Consider seasonal quality targets")
            elif trend_type == TrendType.EXPONENTIAL:
                recommendations.append("Exponential trend detected - monitor for acceleration")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return ["Unable to generate recommendations"]
    
    def _create_empty_analysis(self, metric_name: str) -> TrendAnalysis:
        """Create empty trend analysis"""
        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=TrendDirection.STABLE,
            trend_type=TrendType.LINEAR,
            slope=0.0,
            r_squared=0.0,
            confidence=0.0,
            forecast_values=[],
            forecast_dates=[],
            seasonality_detected=False,
            volatility=0.0,
            recommendations=["No data available for trend analysis"]
        )
    
    def _create_insufficient_data_analysis(self, metric_name: str) -> TrendAnalysis:
        """Create trend analysis for insufficient data"""
        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=TrendDirection.STABLE,
            trend_type=TrendType.LINEAR,
            slope=0.0,
            r_squared=0.0,
            confidence=0.0,
            forecast_values=[],
            forecast_dates=[],
            seasonality_detected=False,
            volatility=0.0,
            recommendations=[f"Insufficient data for trend analysis (need at least {self.min_data_points} points)"]
        )
    
    def _create_error_analysis(self, metric_name: str, error_message: str) -> TrendAnalysis:
        """Create trend analysis for error case"""
        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=TrendDirection.STABLE,
            trend_type=TrendType.LINEAR,
            slope=0.0,
            r_squared=0.0,
            confidence=0.0,
            forecast_values=[],
            forecast_dates=[],
            seasonality_detected=False,
            volatility=0.0,
            recommendations=[f"Trend analysis failed: {error_message}"]
        )
    
    def analyze_multiple_metrics(self,
                               data: pd.DataFrame,
                               metric_columns: List[str],
                               date_column: str = "timestamp") -> Dict[str, TrendAnalysis]:
        """Analyze trends for multiple metrics"""
        try:
            results = {}
            
            for metric in metric_columns:
                if metric in data.columns:
                    analysis = self.analyze_trend(data, metric, date_column)
                    results[metric] = analysis
                else:
                    logger.warning(f"Metric column {metric} not found in data")
            
            logger.info(f"Analyzed trends for {len(results)} metrics")
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze multiple metrics: {str(e)}")
            return {}
    
    def get_trend_summary(self, analyses: Dict[str, TrendAnalysis]) -> Dict[str, Any]:
        """Get summary of multiple trend analyses"""
        try:
            if not analyses:
                return {"error": "No analyses provided"}
            
            # Count trends by direction
            direction_counts = {}
            for analysis in analyses.values():
                direction = analysis.trend_direction.value
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
            
            # Calculate average confidence
            avg_confidence = np.mean([a.confidence for a in analyses.values()])
            
            # Count seasonal patterns
            seasonal_count = sum(1 for a in analyses.values() if a.seasonality_detected)
            
            # Calculate average volatility
            avg_volatility = np.mean([a.volatility for a in analyses.values()])
            
            return {
                "total_metrics": len(analyses),
                "trend_directions": direction_counts,
                "average_confidence": avg_confidence,
                "seasonal_patterns": seasonal_count,
                "average_volatility": avg_volatility,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get trend summary: {str(e)}")
            return {"error": str(e)}
