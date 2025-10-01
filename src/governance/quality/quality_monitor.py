"""
Quality Monitor
Real-time quality monitoring and alerting
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import threading
import time

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MonitorStatus(Enum):
    """Monitor status"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"

@dataclass
class QualityAlert:
    """Represents a quality alert"""
    alert_id: str
    monitor_id: str
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class QualityMetric:
    """Represents a quality metric"""
    metric_name: str
    value: float
    threshold: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any]

class QualityMonitor:
    """
    Real-time quality monitoring and alerting
    """
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval  # seconds
        self.monitors = {}
        self.alerts = []
        self.metrics_history = []
        self.status = MonitorStatus.STOPPED
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        logger.info(f"QualityMonitor initialized with {check_interval}s check interval")
    
    def add_monitor(self,
                   monitor_id: str,
                   metric_name: str,
                   threshold: float,
                   alert_level: AlertLevel,
                   check_function: Callable[[], float],
                   description: str = "",
                   enabled: bool = True) -> bool:
        """Add a quality monitor"""
        try:
            monitor = {
                "monitor_id": monitor_id,
                "metric_name": metric_name,
                "threshold": threshold,
                "alert_level": alert_level,
                "check_function": check_function,
                "description": description,
                "enabled": enabled,
                "created_at": datetime.now(),
                "last_check": None,
                "last_value": None
            }
            
            self.monitors[monitor_id] = monitor
            
            logger.info(f"Quality monitor added: {monitor_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add monitor {monitor_id}: {str(e)}")
            return False
    
    def remove_monitor(self, monitor_id: str) -> bool:
        """Remove a quality monitor"""
        try:
            if monitor_id in self.monitors:
                del self.monitors[monitor_id]
                logger.info(f"Quality monitor removed: {monitor_id}")
                return True
            else:
                logger.warning(f"Monitor {monitor_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove monitor {monitor_id}: {str(e)}")
            return False
    
    def start_monitoring(self) -> bool:
        """Start quality monitoring"""
        try:
            if self.status == MonitorStatus.ACTIVE:
                logger.warning("Monitoring already active")
                return True
            
            self.status = MonitorStatus.ACTIVE
            self.stop_event.clear()
            
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            logger.info("Quality monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {str(e)}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop quality monitoring"""
        try:
            if self.status == MonitorStatus.STOPPED:
                logger.warning("Monitoring already stopped")
                return True
            
            self.status = MonitorStatus.STOPPED
            self.stop_event.set()
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            logger.info("Quality monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {str(e)}")
            return False
    
    def pause_monitoring(self) -> bool:
        """Pause quality monitoring"""
        try:
            if self.status == MonitorStatus.ACTIVE:
                self.status = MonitorStatus.PAUSED
                logger.info("Quality monitoring paused")
                return True
            else:
                logger.warning("Monitoring not active")
                return False
                
        except Exception as e:
            logger.error(f"Failed to pause monitoring: {str(e)}")
            return False
    
    def resume_monitoring(self) -> bool:
        """Resume quality monitoring"""
        try:
            if self.status == MonitorStatus.PAUSED:
                self.status = MonitorStatus.ACTIVE
                logger.info("Quality monitoring resumed")
                return True
            else:
                logger.warning("Monitoring not paused")
                return False
                
        except Exception as e:
            logger.error(f"Failed to resume monitoring: {str(e)}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while not self.stop_event.is_set() and self.status == MonitorStatus.ACTIVE:
                self._check_all_monitors()
                self.stop_event.wait(self.check_interval)
                
        except Exception as e:
            logger.error(f"Monitoring loop error: {str(e)}")
        finally:
            logger.info("Monitoring loop ended")
    
    def _check_all_monitors(self):
        """Check all active monitors"""
        try:
            for monitor_id, monitor in self.monitors.items():
                if monitor["enabled"] and self.status == MonitorStatus.ACTIVE:
                    self._check_monitor(monitor_id, monitor)
                    
        except Exception as e:
            logger.error(f"Failed to check monitors: {str(e)}")
    
    def _check_monitor(self, monitor_id: str, monitor: Dict[str, Any]):
        """Check a single monitor"""
        try:
            # Execute check function
            current_value = monitor["check_function"]()
            
            # Update monitor state
            monitor["last_check"] = datetime.now()
            monitor["last_value"] = current_value
            
            # Record metric
            metric = QualityMetric(
                metric_name=monitor["metric_name"],
                value=current_value,
                threshold=monitor["threshold"],
                unit="ratio",  # Default unit
                timestamp=datetime.now(),
                metadata={"monitor_id": monitor_id}
            )
            self.metrics_history.append(metric)
            
            # Check for threshold breach
            if self._is_threshold_breached(current_value, monitor["threshold"]):
                self._create_alert(monitor_id, monitor, current_value)
            
            logger.debug(f"Monitor {monitor_id} checked: {current_value}")
            
        except Exception as e:
            logger.error(f"Failed to check monitor {monitor_id}: {str(e)}")
            # Create error alert
            self._create_error_alert(monitor_id, monitor, str(e))
    
    def _is_threshold_breached(self, current_value: float, threshold: float) -> bool:
        """Check if threshold is breached"""
        try:
            # For now, assume threshold breach when value is below threshold
            # In real implementation, this would be configurable
            return current_value < threshold
            
        except Exception as e:
            logger.error(f"Failed to check threshold breach: {str(e)}")
            return False
    
    def _create_alert(self, monitor_id: str, monitor: Dict[str, Any], current_value: float):
        """Create a quality alert"""
        try:
            alert_id = f"alert_{int(datetime.now().timestamp())}"
            
            alert = QualityAlert(
                alert_id=alert_id,
                monitor_id=monitor_id,
                level=monitor["alert_level"],
                message=f"Quality threshold breached: {monitor['metric_name']} = {current_value:.3f} (threshold: {monitor['threshold']:.3f})",
                metric_name=monitor["metric_name"],
                current_value=current_value,
                threshold_value=monitor["threshold"],
                timestamp=datetime.now()
            )
            
            self.alerts.append(alert)
            
            # Log alert
            log_level = {
                AlertLevel.INFO: logger.info,
                AlertLevel.WARNING: logger.warning,
                AlertLevel.ERROR: logger.error,
                AlertLevel.CRITICAL: logger.critical
            }
            
            log_func = log_level.get(alert.level, logger.warning)
            log_func(f"Quality Alert: {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to create alert: {str(e)}")
    
    def _create_error_alert(self, monitor_id: str, monitor: Dict[str, Any], error_message: str):
        """Create an error alert for monitor failure"""
        try:
            alert_id = f"error_alert_{int(datetime.now().timestamp())}"
            
            alert = QualityAlert(
                alert_id=alert_id,
                monitor_id=monitor_id,
                level=AlertLevel.ERROR,
                message=f"Monitor check failed: {error_message}",
                metric_name=monitor["metric_name"],
                current_value=0.0,
                threshold_value=monitor["threshold"],
                timestamp=datetime.now()
            )
            
            self.alerts.append(alert)
            logger.error(f"Monitor Error Alert: {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to create error alert: {str(e)}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a quality alert"""
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    logger.info(f"Alert resolved: {alert_id}")
                    return True
            
            logger.warning(f"Alert {alert_id} not found or already resolved")
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {str(e)}")
            return False
    
    def get_active_alerts(self) -> List[QualityAlert]:
        """Get all active (unresolved) alerts"""
        try:
            return [alert for alert in self.alerts if not alert.resolved]
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {str(e)}")
            return []
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[QualityAlert]:
        """Get alerts by level"""
        try:
            return [alert for alert in self.alerts if alert.level == level]
            
        except Exception as e:
            logger.error(f"Failed to get alerts by level: {str(e)}")
            return []
    
    def get_metrics_history(self, 
                          metric_name: Optional[str] = None,
                          hours: int = 24) -> List[QualityMetric]:
        """Get metrics history"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            filtered_metrics = [
                metric for metric in self.metrics_history
                if metric.timestamp >= cutoff_time
            ]
            
            if metric_name:
                filtered_metrics = [
                    metric for metric in filtered_metrics
                    if metric.metric_name == metric_name
                ]
            
            # Sort by timestamp
            filtered_metrics.sort(key=lambda x: x.timestamp)
            
            return filtered_metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics history: {str(e)}")
            return []
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        try:
            active_monitors = len([m for m in self.monitors.values() if m["enabled"]])
            total_alerts = len(self.alerts)
            active_alerts = len(self.get_active_alerts())
            
            return {
                "status": self.status.value,
                "active_monitors": active_monitors,
                "total_monitors": len(self.monitors),
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "check_interval": self.check_interval,
                "last_check": max([m["last_check"] for m in self.monitors.values() if m["last_check"]], default=None),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get monitoring status: {str(e)}")
            return {"error": str(e)}
    
    def get_quality_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get data for quality dashboard"""
        try:
            metrics = self.get_metrics_history(hours=hours)
            alerts = self.get_active_alerts()
            
            # Group metrics by name
            metrics_by_name = {}
            for metric in metrics:
                if metric.metric_name not in metrics_by_name:
                    metrics_by_name[metric.metric_name] = []
                metrics_by_name[metric.metric_name].append(metric)
            
            # Calculate trends
            trends = {}
            for metric_name, metric_list in metrics_by_name.items():
                if len(metric_list) >= 2:
                    values = [m.value for m in metric_list]
                    trend = "improving" if values[-1] > values[0] else "declining"
                    trends[metric_name] = {
                        "trend": trend,
                        "current_value": values[-1],
                        "previous_value": values[0],
                        "change": values[-1] - values[0]
                    }
            
            # Count alerts by level
            alert_counts = {}
            for level in AlertLevel:
                alert_counts[level.value] = len([a for a in alerts if a.level == level])
            
            return {
                "metrics_trends": trends,
                "alert_counts": alert_counts,
                "total_metrics": len(metrics),
                "active_alerts": len(alerts),
                "time_range_hours": hours,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {str(e)}")
            return {"error": str(e)}
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old metrics and alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Clean up old metrics
            old_metrics_count = len(self.metrics_history)
            self.metrics_history = [
                metric for metric in self.metrics_history
                if metric.timestamp >= cutoff_time
            ]
            removed_metrics = old_metrics_count - len(self.metrics_history)
            
            # Clean up old resolved alerts
            old_alerts_count = len(self.alerts)
            self.alerts = [
                alert for alert in self.alerts
                if not alert.resolved or alert.resolved_at >= cutoff_time
            ]
            removed_alerts = old_alerts_count - len(self.alerts)
            
            total_removed = removed_metrics + removed_alerts
            logger.info(f"Cleaned up {total_removed} old records ({removed_metrics} metrics, {removed_alerts} alerts)")
            return total_removed
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
            return 0
