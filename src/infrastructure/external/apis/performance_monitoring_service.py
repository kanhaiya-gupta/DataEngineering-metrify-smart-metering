"""
Performance Monitoring Service

Provides comprehensive performance monitoring, metrics collection, and resource tracking
for the data engineering pipeline.
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific component"""
    component_name: str
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_usage_mb: float
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int
    processing_rate: float  # records/second
    error_rate: float  # errors/second
    queue_size: int
    active_connections: int
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResourceLimits:
    """Resource limits for components"""
    max_memory_mb: int
    max_cpu_percent: float
    max_disk_io_mb: int
    max_network_io_mb: int
    max_connections: int
    max_queue_size: int

@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    component_name: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    threshold: float
    current_value: float
    resolved: bool = False

class PerformanceMonitoringService:
    """
    Comprehensive performance monitoring service
    
    Monitors system resources, application metrics, and performance indicators
    with real-time alerting and historical tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[PerformanceAlert] = []
        self.resource_limits: Dict[str, ResourceLimits] = {}
        self.custom_metrics_collectors: Dict[str, Callable] = {}
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.alert_callbacks: List[Callable] = []
        
        # Performance tracking
        self.processing_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.queue_sizes: Dict[str, int] = defaultdict(int)
        self.active_connections: Dict[str, int] = defaultdict(int)
        
        # Initialize resource limits from config
        self._load_resource_limits()
        
    def _load_resource_limits(self):
        """Load resource limits from configuration"""
        try:
            limits_config = self.config.get('resource_limits', {})
            
            for component, limits in limits_config.items():
                self.resource_limits[component] = ResourceLimits(
                    max_memory_mb=limits.get('max_memory_mb', 1024),
                    max_cpu_percent=limits.get('max_cpu_percent', 80.0),
                    max_disk_io_mb=limits.get('max_disk_io_mb', 100),
                    max_network_io_mb=limits.get('max_network_io_mb', 50),
                    max_connections=limits.get('max_connections', 100),
                    max_queue_size=limits.get('max_queue_size', 1000)
                )
                
        except Exception as e:
            logger.error(f"Error loading resource limits: {e}")
    
    def start_monitoring(self, interval: int = 30):
        """Start performance monitoring"""
        if self.monitoring_active:
            logger.warning("Performance monitoring is already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started performance monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped performance monitoring")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect application metrics
                self._collect_application_metrics()
                
                # Check for alerts
                self._check_alerts()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / (1024 * 1024)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes if disk_io else 0
            disk_write = disk_io.write_bytes if disk_io else 0
            
            # Network I/O
            network_io = psutil.net_io_counters()
            net_sent = network_io.bytes_sent if network_io else 0
            net_recv = network_io.bytes_recv if network_io else 0
            
            # Create metrics for system
            metrics = PerformanceMetrics(
                component_name="system",
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                memory_usage_mb=memory_usage_mb,
                disk_io_read=disk_read,
                disk_io_write=disk_write,
                network_io_sent=net_sent,
                network_io_recv=net_recv,
                processing_rate=0.0,  # Will be updated by application metrics
                error_rate=0.0,
                queue_size=0,
                active_connections=0
            )
            
            self._store_metrics("system", metrics)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # Collect metrics for each component
            for component in self.resource_limits.keys():
                metrics = PerformanceMetrics(
                    component_name=component,
                    timestamp=datetime.utcnow(),
                    cpu_usage=0.0,  # Will be calculated based on system usage
                    memory_usage=0.0,
                    memory_usage_mb=0.0,
                    disk_io_read=0,
                    disk_io_write=0,
                    network_io_sent=0,
                    network_io_recv=0,
                    processing_rate=self._calculate_processing_rate(component),
                    error_rate=self._calculate_error_rate(component),
                    queue_size=self.queue_sizes.get(component, 0),
                    active_connections=self.active_connections.get(component, 0)
                )
                
                # Add custom metrics if collector exists
                if component in self.custom_metrics_collectors:
                    try:
                        custom_metrics = self.custom_metrics_collectors[component]()
                        metrics.custom_metrics.update(custom_metrics)
                    except Exception as e:
                        logger.error(f"Error collecting custom metrics for {component}: {e}")
                
                self._store_metrics(component, metrics)
                
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    def _calculate_processing_rate(self, component: str) -> float:
        """Calculate processing rate for a component"""
        rates = self.processing_rates.get(component, deque())
        if not rates:
            return 0.0
        return sum(rates) / len(rates)
    
    def _calculate_error_rate(self, component: str) -> float:
        """Calculate error rate for a component"""
        rates = self.error_rates.get(component, deque())
        if not rates:
            return 0.0
        return sum(rates) / len(rates)
    
    def _store_metrics(self, component: str, metrics: PerformanceMetrics):
        """Store metrics in history"""
        self.metrics_history[component].append(metrics)
    
    def _check_alerts(self):
        """Check for performance alerts"""
        try:
            for component, metrics_list in self.metrics_history.items():
                if not metrics_list:
                    continue
                    
                latest_metrics = metrics_list[-1]
                limits = self.resource_limits.get(component)
                
                if not limits:
                    continue
                
                # Check memory usage
                if latest_metrics.memory_usage_mb > limits.max_memory_mb:
                    self._create_alert(
                        component=component,
                        alert_type="high_memory_usage",
                        severity="warning",
                        message=f"Memory usage {latest_metrics.memory_usage_mb:.1f}MB exceeds limit {limits.max_memory_mb}MB",
                        threshold=limits.max_memory_mb,
                        current_value=latest_metrics.memory_usage_mb
                    )
                
                # Check CPU usage
                if latest_metrics.cpu_usage > limits.max_cpu_percent:
                    self._create_alert(
                        component=component,
                        alert_type="high_cpu_usage",
                        severity="warning",
                        message=f"CPU usage {latest_metrics.cpu_usage:.1f}% exceeds limit {limits.max_cpu_percent}%",
                        threshold=limits.max_cpu_percent,
                        current_value=latest_metrics.cpu_usage
                    )
                
                # Check queue size
                if latest_metrics.queue_size > limits.max_queue_size:
                    self._create_alert(
                        component=component,
                        alert_type="high_queue_size",
                        severity="critical",
                        message=f"Queue size {latest_metrics.queue_size} exceeds limit {limits.max_queue_size}",
                        threshold=limits.max_queue_size,
                        current_value=latest_metrics.queue_size
                    )
                
                # Check processing rate (if too low)
                if latest_metrics.processing_rate < 10 and latest_metrics.processing_rate > 0:
                    self._create_alert(
                        component=component,
                        alert_type="low_processing_rate",
                        severity="warning",
                        message=f"Processing rate {latest_metrics.processing_rate:.1f} records/sec is low",
                        threshold=10.0,
                        current_value=latest_metrics.processing_rate
                    )
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _create_alert(self, component: str, alert_type: str, severity: str, 
                     message: str, threshold: float, current_value: float):
        """Create a performance alert"""
        alert_id = f"{component}_{alert_type}_{int(time.time())}"
        
        # Check if similar alert already exists
        for existing_alert in self.alerts:
            if (existing_alert.component_name == component and 
                existing_alert.alert_type == alert_type and 
                not existing_alert.resolved):
                return  # Don't create duplicate alerts
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            component_name=component,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.utcnow(),
            threshold=threshold,
            current_value=current_value
        )
        
        self.alerts.append(alert)
        logger.warning(f"Performance Alert: {message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        try:
            # Keep only last 24 hours of alerts
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def record_processing_event(self, component: str, records_processed: int, 
                              processing_time: float, errors: int = 0):
        """Record a processing event for rate calculation"""
        try:
            if processing_time > 0:
                rate = records_processed / processing_time
                self.processing_rates[component].append(rate)
            
            if processing_time > 0:
                error_rate = errors / processing_time
                self.error_rates[component].append(error_rate)
                
        except Exception as e:
            logger.error(f"Error recording processing event: {e}")
    
    def update_queue_size(self, component: str, size: int):
        """Update queue size for a component"""
        self.queue_sizes[component] = size
    
    def update_connection_count(self, component: str, count: int):
        """Update active connection count for a component"""
        self.active_connections[component] = count
    
    def register_custom_metrics_collector(self, component: str, collector: Callable):
        """Register a custom metrics collector for a component"""
        self.custom_metrics_collectors[component] = collector
    
    def add_alert_callback(self, callback: Callable):
        """Add an alert callback function"""
        self.alert_callbacks.append(callback)
    
    def get_metrics_summary(self, component: str = None, 
                           hours: int = 1) -> Dict[str, Any]:
        """Get performance metrics summary"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            if component:
                components = [component]
            else:
                components = list(self.metrics_history.keys())
            
            summary = {}
            
            for comp in components:
                metrics_list = self.metrics_history.get(comp, deque())
                recent_metrics = [m for m in metrics_list if m.timestamp > cutoff_time]
                
                if not recent_metrics:
                    continue
                
                summary[comp] = {
                    "total_metrics": len(recent_metrics),
                    "avg_cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
                    "avg_memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
                    "avg_processing_rate": sum(m.processing_rate for m in recent_metrics) / len(recent_metrics),
                    "avg_error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
                    "max_queue_size": max(m.queue_size for m in recent_metrics),
                    "max_connections": max(m.active_connections for m in recent_metrics),
                    "latest_timestamp": recent_metrics[-1].timestamp.isoformat()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {}
    
    def get_active_alerts(self, component: str = None) -> List[Dict[str, Any]]:
        """Get active alerts"""
        try:
            alerts = [alert for alert in self.alerts if not alert.resolved]
            
            if component:
                alerts = [alert for alert in alerts if alert.component_name == component]
            
            return [
                {
                    "alert_id": alert.alert_id,
                    "component_name": alert.component_name,
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "threshold": alert.threshold,
                    "current_value": alert.current_value
                }
                for alert in alerts
            ]
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    logger.info(f"Resolved alert {alert_id}")
                    break
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard"""
        try:
            return {
                "metrics_summary": self.get_metrics_summary(hours=1),
                "active_alerts": self.get_active_alerts(),
                "system_health": self._calculate_system_health(),
                "resource_utilization": self._calculate_resource_utilization(),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    def _calculate_system_health(self) -> str:
        """Calculate overall system health"""
        try:
            alerts = self.get_active_alerts()
            critical_alerts = [a for a in alerts if a["severity"] == "critical"]
            warning_alerts = [a for a in alerts if a["severity"] == "warning"]
            
            if critical_alerts:
                return "critical"
            elif warning_alerts:
                return "warning"
            else:
                return "healthy"
                
        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return "unknown"
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization percentages"""
        try:
            system_metrics = self.metrics_history.get("system", deque())
            if not system_metrics:
                return {}
            
            latest = system_metrics[-1]
            
            return {
                "cpu_utilization": latest.cpu_usage,
                "memory_utilization": latest.memory_usage,
                "disk_io_utilization": min(100.0, (latest.disk_io_read + latest.disk_io_write) / (1024 * 1024 * 100)),  # Rough estimate
                "network_io_utilization": min(100.0, (latest.network_io_sent + latest.network_io_recv) / (1024 * 1024 * 50))  # Rough estimate
            }
            
        except Exception as e:
            logger.error(f"Error calculating resource utilization: {e}")
            return {}
    
    def export_metrics(self, file_path: str, hours: int = 24):
        """Export metrics to file"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "time_range_hours": hours,
                "metrics": {},
                "alerts": []
            }
            
            # Export metrics
            for component, metrics_list in self.metrics_history.items():
                recent_metrics = [m for m in metrics_list if m.timestamp > cutoff_time]
                export_data["metrics"][component] = [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "cpu_usage": m.cpu_usage,
                        "memory_usage": m.memory_usage,
                        "memory_usage_mb": m.memory_usage_mb,
                        "processing_rate": m.processing_rate,
                        "error_rate": m.error_rate,
                        "queue_size": m.queue_size,
                        "active_connections": m.active_connections,
                        "custom_metrics": m.custom_metrics
                    }
                    for m in recent_metrics
                ]
            
            # Export alerts
            recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
            export_data["alerts"] = [
                {
                    "alert_id": a.alert_id,
                    "component_name": a.component_name,
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                    "threshold": a.threshold,
                    "current_value": a.current_value,
                    "resolved": a.resolved
                }
                for a in recent_alerts
            ]
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported metrics to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

# Global performance monitoring instance
performance_monitor = PerformanceMonitoringService()
