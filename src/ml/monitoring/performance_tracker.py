"""
Performance Tracker Module

This module provides performance tracking and monitoring capabilities for ML models:
- Model performance metrics tracking
- Real-time performance monitoring
- Performance degradation detection
- Performance reporting and alerting
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    metric_name: str
    value: float
    timestamp: datetime
    model_version: str
    dataset_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    alert_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    model_version: str
    resolved: bool = False

class PerformanceThreshold:
    """Performance threshold configuration"""
    
    def __init__(self, metric_name: str, threshold_value: float, 
                 comparison: str = 'less_than', severity: str = 'medium'):
        self.metric_name = metric_name
        self.threshold_value = threshold_value
        self.comparison = comparison  # 'less_than', 'greater_than', 'equals'
        self.severity = severity

class PerformanceTracker:
    """
    Performance tracker for ML models
    
    Tracks model performance metrics, detects degradation,
    and generates alerts when performance drops below thresholds.
    """
    
    def __init__(self, model_name: str, model_version: str = "latest"):
        self.model_name = model_name
        self.model_version = model_version
        self.metrics_history: List[PerformanceMetric] = []
        self.alerts: List[PerformanceAlert] = []
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.monitoring_active = False
        self.monitoring_task = None
        
        logger.info(f"Performance tracker initialized for model: {model_name} v{model_version}")
    
    def add_threshold(self, threshold: PerformanceThreshold):
        """Add a performance threshold"""
        self.thresholds[threshold.metric_name] = threshold
        logger.info(f"Added threshold for {threshold.metric_name}: {threshold.comparison} {threshold.threshold_value}")
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        metric.model_version = self.model_version
        self.metrics_history.append(metric)
        
        # Check for threshold violations
        self._check_thresholds(metric)
        
        logger.debug(f"Recorded metric {metric.metric_name}: {metric.value}")
    
    def record_metrics(self, metrics: List[PerformanceMetric]):
        """Record multiple performance metrics"""
        for metric in metrics:
            self.record_metric(metric)
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric violates any thresholds"""
        if metric.metric_name not in self.thresholds:
            return
        
        threshold = self.thresholds[metric.metric_name]
        violation = False
        
        if threshold.comparison == 'less_than' and metric.value < threshold.threshold_value:
            violation = True
        elif threshold.comparison == 'greater_than' and metric.value > threshold.threshold_value:
            violation = True
        elif threshold.comparison == 'equals' and metric.value == threshold.threshold_value:
            violation = True
        
        if violation:
            self._create_alert(metric, threshold)
    
    def _create_alert(self, metric: PerformanceMetric, threshold: PerformanceThreshold):
        """Create a performance alert"""
        alert_id = f"{self.model_name}_{metric.metric_name}_{int(time.time())}"
        
        message = f"Performance threshold violated: {metric.metric_name} = {metric.value} {threshold.comparison} {threshold.threshold_value}"
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            metric_name=metric.metric_name,
            current_value=metric.value,
            threshold_value=threshold.threshold_value,
            severity=threshold.severity,
            message=message,
            timestamp=datetime.now(),
            model_version=self.model_version
        )
        
        self.alerts.append(alert)
        logger.warning(f"Performance alert created: {message}")
    
    def get_latest_metrics(self, metric_name: Optional[str] = None, 
                          limit: int = 100) -> List[PerformanceMetric]:
        """Get latest performance metrics"""
        metrics = self.metrics_history
        
        if metric_name:
            metrics = [m for m in metrics if m.metric_name == metric_name]
        
        # Sort by timestamp descending and limit
        metrics.sort(key=lambda x: x.timestamp, reverse=True)
        return metrics[:limit]
    
    def get_metrics_summary(self, metric_name: str, 
                           hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for a specific time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.metric_name == metric_name and m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No metrics found for the specified period"}
        
        values = [m.value for m in recent_metrics]
        
        return {
            "metric_name": metric_name,
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "latest": values[0] if values else None,
            "time_period_hours": hours
        }
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"Alert {alert_id} resolved")
                break
    
    def get_performance_trend(self, metric_name: str, 
                            hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trend for a metric"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.metric_name == metric_name and m.timestamp >= cutoff_time
        ]
        
        if len(recent_metrics) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Sort by timestamp
        recent_metrics.sort(key=lambda x: x.timestamp)
        values = [m.value for m in recent_metrics]
        
        # Calculate trend
        x = np.arange(len(values))
        trend_slope = np.polyfit(x, values, 1)[0]
        
        # Determine trend direction
        if abs(trend_slope) < 0.01:
            trend_direction = "stable"
        elif trend_slope > 0:
            trend_direction = "improving"
        else:
            trend_direction = "degrading"
        
        return {
            "metric_name": metric_name,
            "trend_slope": trend_slope,
            "trend_direction": trend_direction,
            "data_points": len(values),
            "time_period_hours": hours,
            "first_value": values[0],
            "last_value": values[-1],
            "change": values[-1] - values[0] if len(values) > 1 else 0
        }
    
    def export_metrics(self, format: str = 'json') -> Union[str, pd.DataFrame]:
        """Export metrics in specified format"""
        if not self.metrics_history:
            return "No metrics to export"
        
        if format == 'json':
            metrics_data = []
            for metric in self.metrics_history:
                metrics_data.append({
                    'metric_name': metric.metric_name,
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat(),
                    'model_version': metric.model_version,
                    'dataset_name': metric.dataset_name,
                    'metadata': metric.metadata
                })
            return json.dumps(metrics_data, indent=2)
        
        elif format == 'dataframe':
            data = []
            for metric in self.metrics_history:
                data.append({
                    'metric_name': metric.metric_name,
                    'value': metric.value,
                    'timestamp': metric.timestamp,
                    'model_version': metric.model_version,
                    'dataset_name': metric.dataset_name,
                    **metric.metadata
                })
            return pd.DataFrame(data)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def start_monitoring(self, interval_seconds: int = 300):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        logger.info(f"Started performance monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        if not self.monitoring_active:
            logger.warning("Monitoring is not active")
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("Stopped performance monitoring")
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                # This would integrate with actual model serving
                # to collect real-time performance metrics
                await self._collect_performance_metrics()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(interval_seconds)
    
    async def _collect_performance_metrics(self):
        """Collect performance metrics from model serving"""
        # This is a placeholder - in real implementation,
        # this would collect metrics from the actual model serving infrastructure
        logger.debug("Collecting performance metrics...")
        
        # Example: collect accuracy, latency, throughput metrics
        # This would be replaced with actual metric collection
        pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get performance tracker health status"""
        active_alerts = self.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity == 'critical']
        
        return {
            "status": "healthy" if not critical_alerts else "unhealthy",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "total_metrics": len(self.metrics_history),
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "monitoring_active": self.monitoring_active,
            "thresholds_configured": len(self.thresholds)
        }
