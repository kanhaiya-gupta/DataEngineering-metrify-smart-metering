# src/performance/streaming/__init__.py

"""
Streaming Performance Optimization
Advanced stream processing capabilities for real-time analytics
"""

from .flink_integration import FlinkIntegration, FlinkJobConfig, FlinkStreamConfig, FlinkJobMetrics
from .stream_joins import StreamJoiner, JoinType, WindowType, JoinWindow, StreamRecord, JoinResult
from .real_time_analytics import (
    RealTimeAnalytics, 
    AnalyticsType, 
    WindowFunction, 
    AnalyticsWindow,
    AnalyticsRecord,
    AnalyticsResult,
    AnomalyDetectionResult,
    TrendAnalysisResult
)

__all__ = [
    # Flink Integration
    "FlinkIntegration",
    "FlinkJobConfig", 
    "FlinkStreamConfig",
    "FlinkJobMetrics",
    
    # Stream Joins
    "StreamJoiner",
    "JoinType",
    "WindowType", 
    "JoinWindow",
    "StreamRecord",
    "JoinResult",
    
    # Real-time Analytics
    "RealTimeAnalytics",
    "AnalyticsType",
    "WindowFunction",
    "AnalyticsWindow", 
    "AnalyticsRecord",
    "AnalyticsResult",
    "AnomalyDetectionResult",
    "TrendAnalysisResult",
]
