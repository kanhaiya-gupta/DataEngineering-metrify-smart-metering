"""
Real-Time Analytics
Advanced real-time analytics and stream processing capabilities
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
import statistics
import math

logger = logging.getLogger(__name__)

class AnalyticsType(Enum):
    """Analytics types"""
    AGGREGATION = "aggregation"
    WINDOWED = "windowed"
    SLIDING = "sliding"
    SESSION = "session"
    PREDICTIVE = "predictive"
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"

class WindowFunction(Enum):
    """Window functions"""
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    STDDEV = "stddev"
    PERCENTILE = "percentile"
    CUSTOM = "custom"

@dataclass
class AnalyticsWindow:
    """Analytics window configuration"""
    window_type: AnalyticsType
    size: int  # seconds
    slide: int = None  # seconds (for sliding windows)
    gap: int = None  # seconds (for session windows)

@dataclass
class AnalyticsRecord:
    """Analytics record"""
    key: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class AnalyticsResult:
    """Analytics result"""
    key: str
    function: WindowFunction
    value: float
    timestamp: datetime
    window_start: datetime
    window_end: datetime
    record_count: int
    metadata: Dict[str, Any] = None

@dataclass
class AnomalyDetectionResult:
    """Anomaly detection result"""
    key: str
    value: float
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    method: str
    metadata: Dict[str, Any] = None

@dataclass
class TrendAnalysisResult:
    """Trend analysis result"""
    key: str
    trend_direction: str  # "up", "down", "stable"
    trend_strength: float  # 0-1
    slope: float
    r_squared: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class AnalyticsMetrics:
    """Analytics operation metrics"""
    total_records_processed: int = 0
    total_results_generated: int = 0
    average_processing_time: float = 0.0
    anomalies_detected: int = 0
    trends_identified: int = 0
    error_count: int = 0

class RealTimeAnalytics:
    """
    Advanced real-time analytics engine with multiple analysis types
    """
    
    def __init__(self, 
                 max_window_size: int = 10000,
                 anomaly_threshold: float = 2.0,
                 trend_confidence_threshold: float = 0.7):
        self.max_window_size = max_window_size
        self.anomaly_threshold = anomaly_threshold
        self.trend_confidence_threshold = trend_confidence_threshold
        
        # Data buffers
        self.data_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_window_size))
        self.window_results: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Analytics functions
        self.window_functions: Dict[WindowFunction, Callable] = {
            WindowFunction.COUNT: self._count_function,
            WindowFunction.SUM: self._sum_function,
            WindowFunction.AVG: self._avg_function,
            WindowFunction.MIN: self._min_function,
            WindowFunction.MAX: self._max_function,
            WindowFunction.MEDIAN: self._median_function,
            WindowFunction.STDDEV: self._stddev_function,
            WindowFunction.PERCENTILE: self._percentile_function
        }
        
        # Metrics
        self.metrics = AnalyticsMetrics()
        self.metrics_lock = threading.RLock()
        
        # Threading
        self.analytics_lock = threading.RLock()
        
        logger.info("RealTimeAnalytics initialized")
    
    def add_record(self, record: AnalyticsRecord) -> List[AnalyticsResult]:
        """Add record and perform analytics"""
        try:
            with self.analytics_lock:
                # Add to buffer
                self.data_buffers[record.key].append(record)
                
                # Perform analytics
                results = self._perform_analytics(record.key)
                
                # Update metrics
                with self.metrics_lock:
                    self.metrics.total_records_processed += 1
                    self.metrics.total_results_generated += len(results)
            
            logger.debug(f"Added record for key {record.key}, generated {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to add record: {str(e)}")
            with self.metrics_lock:
                self.metrics.error_count += 1
            return []
    
    def calculate_windowed_aggregation(self,
                                     key: str,
                                     window: AnalyticsWindow,
                                     functions: List[WindowFunction]) -> List[AnalyticsResult]:
        """Calculate windowed aggregations"""
        try:
            if key not in self.data_buffers:
                return []
            
            records = list(self.data_buffers[key])
            if not records:
                return []
            
            # Filter records by window
            filtered_records = self._filter_by_window(records, window)
            if not filtered_records:
                return []
            
            results = []
            window_start = min(r.timestamp for r in filtered_records)
            window_end = max(r.timestamp for r in filtered_records)
            
            for function in functions:
                if function in self.window_functions:
                    value = self.window_functions[function](filtered_records)
                    
                    result = AnalyticsResult(
                        key=key,
                        function=function,
                        value=value,
                        timestamp=datetime.now(),
                        window_start=window_start,
                        window_end=window_end,
                        record_count=len(filtered_records),
                        metadata={"window_type": window.window_type.value}
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to calculate windowed aggregation for key {key}: {str(e)}")
            return []
    
    def detect_anomalies(self, 
                        key: str,
                        method: str = "z_score",
                        threshold: float = None) -> List[AnomalyDetectionResult]:
        """Detect anomalies in data stream"""
        try:
            if key not in self.data_buffers:
                return []
            
            records = list(self.data_buffers[key])
            if len(records) < 10:  # Need minimum data for anomaly detection
                return []
            
            threshold = threshold or self.anomaly_threshold
            results = []
            
            if method == "z_score":
                results = self._z_score_anomaly_detection(records, threshold)
            elif method == "isolation_forest":
                results = self._isolation_forest_anomaly_detection(records, threshold)
            elif method == "moving_average":
                results = self._moving_average_anomaly_detection(records, threshold)
            elif method == "percentile":
                results = self._percentile_anomaly_detection(records, threshold)
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.anomalies_detected += len([r for r in results if r.is_anomaly])
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies for key {key}: {str(e)}")
            return []
    
    def analyze_trends(self, 
                      key: str,
                      window_size: int = 100,
                      min_confidence: float = None) -> List[TrendAnalysisResult]:
        """Analyze trends in data stream"""
        try:
            if key not in self.data_buffers:
                return []
            
            records = list(self.data_buffers[key])
            if len(records) < window_size:
                return []
            
            min_confidence = min_confidence or self.trend_confidence_threshold
            results = []
            
            # Use sliding window for trend analysis
            for i in range(len(records) - window_size + 1):
                window_records = records[i:i + window_size]
                
                trend_result = self._calculate_trend(window_records, min_confidence)
                if trend_result:
                    trend_result.key = key
                    trend_result.timestamp = datetime.now()
                    results.append(trend_result)
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.trends_identified += len(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze trends for key {key}: {str(e)}")
            return []
    
    def predict_next_values(self,
                           key: str,
                           prediction_count: int = 5,
                           method: str = "linear_regression") -> List[Dict[str, Any]]:
        """Predict next values in the stream"""
        try:
            if key not in self.data_buffers:
                return []
            
            records = list(self.data_buffers[key])
            if len(records) < 10:
                return []
            
            predictions = []
            
            if method == "linear_regression":
                predictions = self._linear_regression_prediction(records, prediction_count)
            elif method == "moving_average":
                predictions = self._moving_average_prediction(records, prediction_count)
            elif method == "exponential_smoothing":
                predictions = self._exponential_smoothing_prediction(records, prediction_count)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict next values for key {key}: {str(e)}")
            return []
    
    def _perform_analytics(self, key: str) -> List[AnalyticsResult]:
        """Perform analytics on data for a key"""
        try:
            records = list(self.data_buffers[key])
            if not records:
                return []
            
            # Default window for real-time analytics
            window = AnalyticsWindow(AnalyticsType.WINDOWED, 300)  # 5 minutes
            functions = [WindowFunction.COUNT, WindowFunction.AVG, WindowFunction.MAX, WindowFunction.MIN]
            
            return self.calculate_windowed_aggregation(key, window, functions)
            
        except Exception as e:
            logger.error(f"Failed to perform analytics for key {key}: {str(e)}")
            return []
    
    def _filter_by_window(self, records: List[AnalyticsRecord], window: AnalyticsWindow) -> List[AnalyticsRecord]:
        """Filter records by window configuration"""
        try:
            if not records:
                return records
            
            current_time = datetime.now()
            
            if window.window_type == AnalyticsType.WINDOWED:
                cutoff_time = current_time - timedelta(seconds=window.size)
                return [r for r in records if r.timestamp >= cutoff_time]
            
            elif window.window_type == AnalyticsType.SLIDING:
                cutoff_time = current_time - timedelta(seconds=window.size)
                return [r for r in records if r.timestamp >= cutoff_time]
            
            elif window.window_type == AnalyticsType.SESSION:
                if window.gap:
                    cutoff_time = current_time - timedelta(seconds=window.gap)
                    return [r for r in records if r.timestamp >= cutoff_time]
                return records
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to filter records by window: {str(e)}")
            return records
    
    # Window functions
    def _count_function(self, records: List[AnalyticsRecord]) -> float:
        """Count function"""
        return float(len(records))
    
    def _sum_function(self, records: List[AnalyticsRecord]) -> float:
        """Sum function"""
        return sum(r.value for r in records)
    
    def _avg_function(self, records: List[AnalyticsRecord]) -> float:
        """Average function"""
        if not records:
            return 0.0
        return sum(r.value for r in records) / len(records)
    
    def _min_function(self, records: List[AnalyticsRecord]) -> float:
        """Minimum function"""
        if not records:
            return 0.0
        return min(r.value for r in records)
    
    def _max_function(self, records: List[AnalyticsRecord]) -> float:
        """Maximum function"""
        if not records:
            return 0.0
        return max(r.value for r in records)
    
    def _median_function(self, records: List[AnalyticsRecord]) -> float:
        """Median function"""
        if not records:
            return 0.0
        values = [r.value for r in records]
        return statistics.median(values)
    
    def _stddev_function(self, records: List[AnalyticsRecord]) -> float:
        """Standard deviation function"""
        if len(records) < 2:
            return 0.0
        values = [r.value for r in records]
        return statistics.stdev(values)
    
    def _percentile_function(self, records: List[AnalyticsRecord], percentile: float = 95.0) -> float:
        """Percentile function"""
        if not records:
            return 0.0
        values = [r.value for r in records]
        return statistics.quantiles(values, n=100)[int(percentile) - 1]
    
    # Anomaly detection methods
    def _z_score_anomaly_detection(self, records: List[AnalyticsRecord], threshold: float) -> List[AnomalyDetectionResult]:
        """Z-score anomaly detection"""
        try:
            if len(records) < 3:
                return []
            
            values = [r.value for r in records]
            mean = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
            
            if std_dev == 0:
                return []
            
            results = []
            for record in records:
                z_score = abs((record.value - mean) / std_dev)
                is_anomaly = z_score > threshold
                
                result = AnomalyDetectionResult(
                    key=record.key,
                    value=record.value,
                    timestamp=record.timestamp,
                    anomaly_score=z_score,
                    is_anomaly=is_anomaly,
                    threshold=threshold,
                    method="z_score"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed z-score anomaly detection: {str(e)}")
            return []
    
    def _isolation_forest_anomaly_detection(self, records: List[AnalyticsRecord], threshold: float) -> List[AnomalyDetectionResult]:
        """Isolation Forest anomaly detection (simplified)"""
        try:
            # Simplified implementation - in real system, use scikit-learn
            results = []
            values = [r.value for r in records]
            
            # Simple outlier detection based on IQR
            q1 = statistics.quantiles(values, n=4)[0]
            q3 = statistics.quantiles(values, n=4)[2]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for record in records:
                is_anomaly = record.value < lower_bound or record.value > upper_bound
                anomaly_score = abs(record.value - statistics.median(values)) / (iqr + 1e-10)
                
                result = AnomalyDetectionResult(
                    key=record.key,
                    value=record.value,
                    timestamp=record.timestamp,
                    anomaly_score=anomaly_score,
                    is_anomaly=is_anomaly,
                    threshold=threshold,
                    method="isolation_forest"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed isolation forest anomaly detection: {str(e)}")
            return []
    
    def _moving_average_anomaly_detection(self, records: List[AnalyticsRecord], threshold: float) -> List[AnomalyDetectionResult]:
        """Moving average anomaly detection"""
        try:
            if len(records) < 5:
                return []
            
            window_size = min(10, len(records) // 2)
            results = []
            
            for i in range(len(records)):
                if i < window_size:
                    continue
                
                window_records = records[i - window_size:i]
                window_values = [r.value for r in window_records]
                moving_avg = statistics.mean(window_values)
                moving_std = statistics.stdev(window_values) if len(window_values) > 1 else 0.0
                
                if moving_std == 0:
                    continue
                
                record = records[i]
                z_score = abs((record.value - moving_avg) / moving_std)
                is_anomaly = z_score > threshold
                
                result = AnomalyDetectionResult(
                    key=record.key,
                    value=record.value,
                    timestamp=record.timestamp,
                    anomaly_score=z_score,
                    is_anomaly=is_anomaly,
                    threshold=threshold,
                    method="moving_average"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed moving average anomaly detection: {str(e)}")
            return []
    
    def _percentile_anomaly_detection(self, records: List[AnalyticsRecord], threshold: float) -> List[AnomalyDetectionResult]:
        """Percentile-based anomaly detection"""
        try:
            if len(records) < 10:
                return []
            
            values = [r.value for r in records]
            lower_percentile = statistics.quantiles(values, n=100)[int(100 - threshold * 50) - 1]
            upper_percentile = statistics.quantiles(values, n=100)[int(threshold * 50) - 1]
            
            results = []
            for record in records:
                is_anomaly = record.value < lower_percentile or record.value > upper_percentile
                anomaly_score = 0.0
                if is_anomaly:
                    if record.value < lower_percentile:
                        anomaly_score = (lower_percentile - record.value) / (upper_percentile - lower_percentile + 1e-10)
                    else:
                        anomaly_score = (record.value - upper_percentile) / (upper_percentile - lower_percentile + 1e-10)
                
                result = AnomalyDetectionResult(
                    key=record.key,
                    value=record.value,
                    timestamp=record.timestamp,
                    anomaly_score=anomaly_score,
                    is_anomaly=is_anomaly,
                    threshold=threshold,
                    method="percentile"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed percentile anomaly detection: {str(e)}")
            return []
    
    # Trend analysis
    def _calculate_trend(self, records: List[AnalyticsRecord], min_confidence: float) -> Optional[TrendAnalysisResult]:
        """Calculate trend for a window of records"""
        try:
            if len(records) < 3:
                return None
            
            # Simple linear regression
            n = len(records)
            x_values = list(range(n))
            y_values = [r.value for r in records]
            
            # Calculate slope and intercept
            x_mean = statistics.mean(x_values)
            y_mean = statistics.mean(y_values)
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            if denominator == 0:
                return None
            
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            
            # Calculate R-squared
            y_pred = [slope * x + intercept for x in x_values]
            ss_res = sum((y - pred) ** 2 for y, pred in zip(y_values, y_pred))
            ss_tot = sum((y - y_mean) ** 2 for y in y_values)
            
            if ss_tot == 0:
                r_squared = 0.0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            # Determine trend direction and strength
            if abs(slope) < 0.01:
                trend_direction = "stable"
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = "up"
                trend_strength = min(abs(slope) * 100, 1.0)
            else:
                trend_direction = "down"
                trend_strength = min(abs(slope) * 100, 1.0)
            
            confidence = r_squared * trend_strength
            
            if confidence < min_confidence:
                return None
            
            return TrendAnalysisResult(
                key="",  # Will be set by caller
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                slope=slope,
                r_squared=r_squared,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate trend: {str(e)}")
            return None
    
    # Prediction methods
    def _linear_regression_prediction(self, records: List[AnalyticsRecord], prediction_count: int) -> List[Dict[str, Any]]:
        """Linear regression prediction"""
        try:
            if len(records) < 3:
                return []
            
            n = len(records)
            x_values = list(range(n))
            y_values = [r.value for r in records]
            
            # Calculate slope and intercept
            x_mean = statistics.mean(x_values)
            y_mean = statistics.mean(y_values)
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            if denominator == 0:
                return []
            
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            
            predictions = []
            for i in range(1, prediction_count + 1):
                predicted_value = slope * (n + i - 1) + intercept
                predicted_time = records[-1].timestamp + timedelta(seconds=i * 60)  # Assume 1-minute intervals
                
                predictions.append({
                    "value": predicted_value,
                    "timestamp": predicted_time.isoformat(),
                    "confidence": 0.8,  # Simplified confidence
                    "method": "linear_regression"
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed linear regression prediction: {str(e)}")
            return []
    
    def _moving_average_prediction(self, records: List[AnalyticsRecord], prediction_count: int) -> List[Dict[str, Any]]:
        """Moving average prediction"""
        try:
            if len(records) < 5:
                return []
            
            window_size = min(10, len(records) // 2)
            recent_values = [r.value for r in records[-window_size:]]
            moving_avg = statistics.mean(recent_values)
            
            predictions = []
            for i in range(1, prediction_count + 1):
                predicted_time = records[-1].timestamp + timedelta(seconds=i * 60)
                
                predictions.append({
                    "value": moving_avg,
                    "timestamp": predicted_time.isoformat(),
                    "confidence": 0.6,
                    "method": "moving_average"
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed moving average prediction: {str(e)}")
            return []
    
    def _exponential_smoothing_prediction(self, records: List[AnalyticsRecord], prediction_count: int) -> List[Dict[str, Any]]:
        """Exponential smoothing prediction"""
        try:
            if len(records) < 3:
                return []
            
            alpha = 0.3  # Smoothing factor
            values = [r.value for r in records]
            
            # Simple exponential smoothing
            smoothed_values = [values[0]]
            for i in range(1, len(values)):
                smoothed = alpha * values[i] + (1 - alpha) * smoothed_values[-1]
                smoothed_values.append(smoothed)
            
            last_smoothed = smoothed_values[-1]
            
            predictions = []
            for i in range(1, prediction_count + 1):
                predicted_time = records[-1].timestamp + timedelta(seconds=i * 60)
                
                predictions.append({
                    "value": last_smoothed,
                    "timestamp": predicted_time.isoformat(),
                    "confidence": 0.7,
                    "method": "exponential_smoothing"
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed exponential smoothing prediction: {str(e)}")
            return []
    
    def get_metrics(self) -> AnalyticsMetrics:
        """Get analytics metrics"""
        try:
            with self.metrics_lock:
                return AnalyticsMetrics(
                    total_records_processed=self.metrics.total_records_processed,
                    total_results_generated=self.metrics.total_results_generated,
                    average_processing_time=self.metrics.average_processing_time,
                    anomalies_detected=self.metrics.anomalies_detected,
                    trends_identified=self.metrics.trends_identified,
                    error_count=self.metrics.error_count
                )
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return AnalyticsMetrics()
    
    def reset_metrics(self) -> None:
        """Reset analytics metrics"""
        try:
            with self.metrics_lock:
                self.metrics = AnalyticsMetrics()
                
        except Exception as e:
            logger.error(f"Failed to reset metrics: {str(e)}")
    
    def cleanup_expired_data(self, max_age_hours: int = 24) -> int:
        """Clean up expired data"""
        try:
            cleaned_count = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            with self.analytics_lock:
                for key in list(self.data_buffers.keys()):
                    original_size = len(self.data_buffers[key])
                    self.data_buffers[key] = deque(
                        [r for r in self.data_buffers[key] if r.timestamp >= cutoff_time],
                        maxlen=self.max_window_size
                    )
                    cleaned_count += original_size - len(self.data_buffers[key])
                    
                    if not self.data_buffers[key]:
                        del self.data_buffers[key]
            
            logger.info(f"Cleaned up {cleaned_count} expired records")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {str(e)}")
            return 0
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer information"""
        try:
            with self.analytics_lock:
                return {
                    "total_keys": len(self.data_buffers),
                    "total_records": sum(len(buf) for buf in self.data_buffers.values()),
                    "buffer_sizes": {key: len(buf) for key, buf in self.data_buffers.items()}
                }
                
        except Exception as e:
            logger.error(f"Failed to get buffer info: {str(e)}")
            return {}
