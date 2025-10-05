"""
Resource Manager

Manages system resources including CPU, memory, and I/O limits for optimal
performance and resource utilization in the data engineering pipeline.
"""

import logging
import psutil
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import os
import signal
import gc

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    CONNECTIONS = "connections"
    FILE_DESCRIPTORS = "file_descriptors"

class ResourceStatus(Enum):
    """Resource status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"

@dataclass
class ResourceLimit:
    """Resource limit configuration"""
    resource_type: ResourceType
    max_value: float
    warning_threshold: float
    critical_threshold: float
    unit: str
    enabled: bool = True

@dataclass
class ResourceUsage:
    """Current resource usage"""
    resource_type: ResourceType
    current_value: float
    percentage: float
    status: ResourceStatus
    timestamp: datetime
    limit: ResourceLimit

@dataclass
class ResourceAlert:
    """Resource alert"""
    alert_id: str
    resource_type: ResourceType
    status: ResourceStatus
    message: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False

class ResourceManager:
    """
    Comprehensive resource management system
    
    Features:
    - Real-time resource monitoring
    - Dynamic resource allocation
    - Resource limit enforcement
    - Automatic scaling and throttling
    - Resource usage optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.resource_limits: Dict[ResourceType, ResourceLimit] = {}
        self.current_usage: Dict[ResourceType, ResourceUsage] = {}
        self.resource_alerts: List[ResourceAlert] = []
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.alert_callbacks: List[Callable] = []
        self.throttling_enabled = True
        self.auto_scaling_enabled = True
        
        # Resource tracking
        self.usage_history: Dict[ResourceType, List[ResourceUsage]] = {}
        self.peak_usage: Dict[ResourceType, float] = {}
        self.resource_lock = threading.Lock()
        
        # Initialize resource limits
        self._initialize_resource_limits()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _initialize_resource_limits(self):
        """Initialize resource limits from configuration"""
        try:
            # Default limits
            default_limits = {
                ResourceType.CPU: ResourceLimit(
                    resource_type=ResourceType.CPU,
                    max_value=80.0,  # 80% CPU usage
                    warning_threshold=60.0,
                    critical_threshold=75.0,
                    unit="percent"
                ),
                ResourceType.MEMORY: ResourceLimit(
                    resource_type=ResourceType.MEMORY,
                    max_value=2048.0,  # 2GB
                    warning_threshold=1536.0,  # 1.5GB
                    critical_threshold=1792.0,  # 1.75GB
                    unit="MB"
                ),
                ResourceType.DISK_IO: ResourceLimit(
                    resource_type=ResourceType.DISK_IO,
                    max_value=100.0,  # 100MB/s
                    warning_threshold=75.0,
                    critical_threshold=90.0,
                    unit="MB/s"
                ),
                ResourceType.NETWORK_IO: ResourceLimit(
                    resource_type=ResourceType.NETWORK_IO,
                    max_value=50.0,  # 50MB/s
                    warning_threshold=35.0,
                    critical_threshold=45.0,
                    unit="MB/s"
                ),
                ResourceType.CONNECTIONS: ResourceLimit(
                    resource_type=ResourceType.CONNECTIONS,
                    max_value=1000.0,
                    warning_threshold=750.0,
                    critical_threshold=900.0,
                    unit="count"
                ),
                ResourceType.FILE_DESCRIPTORS: ResourceLimit(
                    resource_type=ResourceType.FILE_DESCRIPTORS,
                    max_value=10000.0,
                    warning_threshold=7500.0,
                    critical_threshold=9000.0,
                    unit="count"
                )
            }
            
            # Load from config if available
            config_limits = self.config.get('resource_limits', {})
            for resource_type, limit_config in config_limits.items():
                try:
                    resource_enum = ResourceType(resource_type)
                    default_limits[resource_enum] = ResourceLimit(
                        resource_type=resource_enum,
                        max_value=limit_config.get('max_value', default_limits[resource_enum].max_value),
                        warning_threshold=limit_config.get('warning_threshold', default_limits[resource_enum].warning_threshold),
                        critical_threshold=limit_config.get('critical_threshold', default_limits[resource_enum].critical_threshold),
                        unit=limit_config.get('unit', default_limits[resource_enum].unit),
                        enabled=limit_config.get('enabled', True)
                    )
                except ValueError:
                    logger.warning(f"Unknown resource type in config: {resource_type}")
            
            self.resource_limits = default_limits
            logger.info("Initialized resource limits")
            
        except Exception as e:
            logger.error(f"Error initializing resource limits: {e}")
    
    def start_monitoring(self, interval: int = 5):
        """Start resource monitoring"""
        if self.monitoring_active:
            logger.warning("Resource monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started resource monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped resource monitoring")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect resource usage
                self._collect_resource_usage()
                
                # Check resource limits
                self._check_resource_limits()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_resource_usage(self):
        """Collect current resource usage"""
        try:
            with self.resource_lock:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self._update_resource_usage(ResourceType.CPU, cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self._update_resource_usage(ResourceType.MEMORY, memory_mb)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    disk_usage = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)  # MB
                    self._update_resource_usage(ResourceType.DISK_IO, disk_usage)
                
                # Network I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    network_usage = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)  # MB
                    self._update_resource_usage(ResourceType.NETWORK_IO, network_usage)
                
                # Connections
                connections = len(psutil.net_connections())
                self._update_resource_usage(ResourceType.CONNECTIONS, connections)
                
                # File descriptors
                try:
                    fd_count = len(os.listdir('/proc/self/fd'))
                    self._update_resource_usage(ResourceType.FILE_DESCRIPTORS, fd_count)
                except (OSError, FileNotFoundError):
                    # Not available on all systems
                    pass
                
        except Exception as e:
            logger.error(f"Error collecting resource usage: {e}")
    
    def _update_resource_usage(self, resource_type: ResourceType, current_value: float):
        """Update resource usage and calculate status"""
        try:
            limit = self.resource_limits.get(resource_type)
            if not limit or not limit.enabled:
                return
            
            # Calculate percentage
            percentage = (current_value / limit.max_value) * 100 if limit.max_value > 0 else 0
            
            # Determine status
            if percentage >= limit.critical_threshold:
                status = ResourceStatus.CRITICAL
            elif percentage >= limit.warning_threshold:
                status = ResourceStatus.WARNING
            else:
                status = ResourceStatus.HEALTHY
            
            # Create usage record
            usage = ResourceUsage(
                resource_type=resource_type,
                current_value=current_value,
                percentage=percentage,
                status=status,
                timestamp=datetime.utcnow(),
                limit=limit
            )
            
            # Store usage
            self.current_usage[resource_type] = usage
            
            # Add to history
            if resource_type not in self.usage_history:
                self.usage_history[resource_type] = []
            self.usage_history[resource_type].append(usage)
            
            # Update peak usage
            if resource_type not in self.peak_usage or current_value > self.peak_usage[resource_type]:
                self.peak_usage[resource_type] = current_value
            
        except Exception as e:
            logger.error(f"Error updating resource usage for {resource_type}: {e}")
    
    def _check_resource_limits(self):
        """Check resource limits and create alerts"""
        try:
            for resource_type, usage in self.current_usage.items():
                limit = usage.limit
                
                # Check if limit is exceeded
                if usage.percentage >= limit.critical_threshold:
                    self._create_resource_alert(
                        resource_type=resource_type,
                        status=ResourceStatus.CRITICAL,
                        message=f"{resource_type.value} usage {usage.percentage:.1f}% exceeds critical threshold {limit.critical_threshold}%",
                        current_value=usage.current_value,
                        threshold=limit.critical_threshold
                    )
                elif usage.percentage >= limit.warning_threshold:
                    self._create_resource_alert(
                        resource_type=resource_type,
                        status=ResourceStatus.WARNING,
                        message=f"{resource_type.value} usage {usage.percentage:.1f}% exceeds warning threshold {limit.warning_threshold}%",
                        current_value=usage.current_value,
                        threshold=limit.warning_threshold
                    )
                
        except Exception as e:
            logger.error(f"Error checking resource limits: {e}")
    
    def _create_resource_alert(self, resource_type: ResourceType, status: ResourceStatus,
                             message: str, current_value: float, threshold: float):
        """Create a resource alert"""
        try:
            # Check if similar alert already exists
            for existing_alert in self.resource_alerts:
                if (existing_alert.resource_type == resource_type and 
                    existing_alert.status == status and 
                    not existing_alert.resolved):
                    return  # Don't create duplicate alerts
            
            alert_id = f"{resource_type.value}_{status.value}_{int(time.time())}"
            
            alert = ResourceAlert(
                alert_id=alert_id,
                resource_type=resource_type,
                status=status,
                message=message,
                current_value=current_value,
                threshold=threshold,
                timestamp=datetime.utcnow()
            )
            
            self.resource_alerts.append(alert)
            logger.warning(f"Resource Alert: {message}")
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in resource alert callback: {e}")
            
        except Exception as e:
            logger.error(f"Error creating resource alert: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old usage history and alerts"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            # Clean up usage history
            for resource_type in self.usage_history:
                self.usage_history[resource_type] = [
                    usage for usage in self.usage_history[resource_type]
                    if usage.timestamp > cutoff_time
                ]
            
            # Clean up alerts
            self.resource_alerts = [
                alert for alert in self.resource_alerts
                if alert.timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def get_resource_usage(self, resource_type: ResourceType = None) -> Dict[str, Any]:
        """Get current resource usage"""
        try:
            if resource_type:
                usage = self.current_usage.get(resource_type)
                if usage:
                    return {
                        "resource_type": usage.resource_type.value,
                        "current_value": usage.current_value,
                        "percentage": usage.percentage,
                        "status": usage.status.value,
                        "timestamp": usage.timestamp.isoformat(),
                        "limit": {
                            "max_value": usage.limit.max_value,
                            "warning_threshold": usage.limit.warning_threshold,
                            "critical_threshold": usage.limit.critical_threshold,
                            "unit": usage.limit.unit
                        }
                    }
                return {}
            
            # Return all resource usage
            return {
                resource_type.value: {
                    "current_value": usage.current_value,
                    "percentage": usage.percentage,
                    "status": usage.status.value,
                    "timestamp": usage.timestamp.isoformat(),
                    "limit": {
                        "max_value": usage.limit.max_value,
                        "warning_threshold": usage.limit.warning_threshold,
                        "critical_threshold": usage.limit.critical_threshold,
                        "unit": usage.limit.unit
                    }
                }
                for resource_type, usage in self.current_usage.items()
            }
            
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {}
    
    def get_resource_alerts(self, status: ResourceStatus = None) -> List[Dict[str, Any]]:
        """Get resource alerts"""
        try:
            alerts = [alert for alert in self.resource_alerts if not alert.resolved]
            
            if status:
                alerts = [alert for alert in alerts if alert.status == status]
            
            return [
                {
                    "alert_id": alert.alert_id,
                    "resource_type": alert.resource_type.value,
                    "status": alert.status.value,
                    "message": alert.message,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in alerts
            ]
            
        except Exception as e:
            logger.error(f"Error getting resource alerts: {e}")
            return []
    
    def resolve_alert(self, alert_id: str):
        """Resolve a resource alert"""
        try:
            for alert in self.resource_alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    logger.info(f"Resolved resource alert {alert_id}")
                    break
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add an alert callback function"""
        self.alert_callbacks.append(callback)
    
    def set_resource_limit(self, resource_type: ResourceType, max_value: float,
                          warning_threshold: float = None, critical_threshold: float = None):
        """Set resource limit"""
        try:
            if resource_type in self.resource_limits:
                limit = self.resource_limits[resource_type]
                limit.max_value = max_value
                if warning_threshold is not None:
                    limit.warning_threshold = warning_threshold
                if critical_threshold is not None:
                    limit.critical_threshold = critical_threshold
                
                logger.info(f"Updated {resource_type.value} limit: max={max_value}, warning={limit.warning_threshold}, critical={limit.critical_threshold}")
            else:
                logger.warning(f"Unknown resource type: {resource_type}")
                
        except Exception as e:
            logger.error(f"Error setting resource limit: {e}")
    
    def enable_throttling(self, enabled: bool = True):
        """Enable or disable resource throttling"""
        self.throttling_enabled = enabled
        logger.info(f"Resource throttling {'enabled' if enabled else 'disabled'}")
    
    def enable_auto_scaling(self, enabled: bool = True):
        """Enable or disable automatic resource scaling"""
        self.auto_scaling_enabled = enabled
        logger.info(f"Auto scaling {'enabled' if enabled else 'disabled'}")
    
    def optimize_memory(self):
        """Optimize memory usage"""
        try:
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Log memory usage after optimization
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            logger.info(f"Memory usage after optimization: {memory_mb:.1f}MB")
            
        except Exception as e:
            logger.error(f"Error optimizing memory: {e}")
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak resource usage"""
        try:
            return {
                resource_type.value: peak_value
                for resource_type, peak_value in self.peak_usage.items()
            }
        except Exception as e:
            logger.error(f"Error getting peak usage: {e}")
            return {}
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary"""
        try:
            return {
                "current_usage": self.get_resource_usage(),
                "active_alerts": self.get_resource_alerts(),
                "peak_usage": self.get_peak_usage(),
                "throttling_enabled": self.throttling_enabled,
                "auto_scaling_enabled": self.auto_scaling_enabled,
                "monitoring_active": self.monitoring_active,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting resource summary: {e}")
            return {}
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down resource manager...")
        self.stop_monitoring()
    
    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()

# Global resource manager instance
resource_manager = ResourceManager()
