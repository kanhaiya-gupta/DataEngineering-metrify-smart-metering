"""
DataDog Client Implementation
Handles metrics, logs, and APM integration
"""

import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

try:
    from datadog import initialize, api, statsd
except ImportError:
    api = None
    statsd = None

from src.core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class DataDogClient:
    """
    DataDog Client for monitoring and observability
    
    Handles metrics collection, log aggregation, and APM tracing
    """
    
    def __init__(
        self,
        api_key: str,
        app_key: str,
        site: str = "datadoghq.com",
        service_name: str = "metrify-smart-metering"
    ):
        if api is None or statsd is None:
            raise InfrastructureError("DataDog SDK not installed", service="datadog")
        
        self.api_key = api_key
        self.app_key = app_key
        self.site = site
        self.service_name = service_name
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize DataDog client"""
        try:
            initialize(
                api_key=self.api_key,
                app_key=self.app_key,
                site=self.site
            )
            self._initialized = True
            logger.info("DataDog client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize DataDog: {str(e)}")
            raise InfrastructureError(f"Failed to initialize DataDog: {str(e)}", service="datadog")
    
    async def send_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[List[str]] = None,
        metric_type: str = "gauge"
    ) -> None:
        """
        Send a metric to DataDog
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for the metric
            metric_type: Type of metric (gauge, count, rate, histogram)
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if metric_type == "gauge":
                statsd.gauge(metric_name, value, tags=tags)
            elif metric_type == "count":
                statsd.count(metric_name, value, tags=tags)
            elif metric_type == "rate":
                statsd.rate(metric_name, value, tags=tags)
            elif metric_type == "histogram":
                statsd.histogram(metric_name, value, tags=tags)
            
            logger.debug(f"Sent metric {metric_name}: {value}")
            
        except Exception as e:
            logger.error(f"Error sending metric: {str(e)}")
            raise InfrastructureError(f"Failed to send metric: {str(e)}", service="datadog")
    
    async def send_log(
        self,
        message: str,
        level: str = "info",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send a log to DataDog
        
        Args:
            message: Log message
            level: Log level (debug, info, warn, error, critical)
            tags: Optional tags for the log
            metadata: Optional metadata for the log
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            log_data = {
                "message": message,
                "level": level,
                "timestamp": int(time.time()),
                "service": self.service_name,
                "tags": tags or [],
                "metadata": metadata or {}
            }
            
            api.Logs.send(body=log_data)
            logger.debug(f"Sent log: {message}")
            
        except Exception as e:
            logger.error(f"Error sending log: {str(e)}")
            raise InfrastructureError(f"Failed to send log: {str(e)}", service="datadog")
    
    async def send_event(
        self,
        title: str,
        text: str,
        alert_type: str = "info",
        tags: Optional[List[str]] = None,
        source_type: str = "metrify"
    ) -> None:
        """
        Send an event to DataDog
        
        Args:
            title: Event title
            text: Event description
            alert_type: Alert type (info, success, warning, error)
            tags: Optional tags for the event
            source_type: Source type for the event
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            event_data = {
                "title": title,
                "text": text,
                "alert_type": alert_type,
                "tags": tags or [],
                "source_type_name": source_type,
                "date_happened": int(time.time())
            }
            
            api.Event.create(**event_data)
            logger.debug(f"Sent event: {title}")
            
        except Exception as e:
            logger.error(f"Error sending event: {str(e)}")
            raise InfrastructureError(f"Failed to send event: {str(e)}", service="datadog")
    
    async def track_meter_metrics(
        self,
        meter_id: str,
        reading_count: int,
        avg_quality_score: float,
        anomaly_count: int
    ) -> None:
        """Track smart meter specific metrics"""
        base_tags = [f"meter_id:{meter_id}", "service:smart_meter"]
        
        await self.send_metric("meter.reading_count", reading_count, base_tags)
        await self.send_metric("meter.quality_score", avg_quality_score, base_tags)
        await self.send_metric("meter.anomaly_count", anomaly_count, base_tags)
    
    async def track_grid_metrics(
        self,
        operator_id: str,
        stability_score: float,
        load_percentage: float,
        anomaly_count: int
    ) -> None:
        """Track grid operator specific metrics"""
        base_tags = [f"operator_id:{operator_id}", "service:grid_operator"]
        
        await self.send_metric("grid.stability_score", stability_score, base_tags)
        await self.send_metric("grid.load_percentage", load_percentage, base_tags)
        await self.send_metric("grid.anomaly_count", anomaly_count, base_tags)
    
    async def track_weather_metrics(
        self,
        station_id: str,
        observation_count: int,
        avg_quality_score: float,
        anomaly_count: int
    ) -> None:
        """Track weather station specific metrics"""
        base_tags = [f"station_id:{station_id}", "service:weather_station"]
        
        await self.send_metric("weather.observation_count", observation_count, base_tags)
        await self.send_metric("weather.quality_score", avg_quality_score, base_tags)
        await self.send_metric("weather.anomaly_count", anomaly_count, base_tags)
    
    async def track_system_metrics(
        self,
        total_meters: int,
        active_meters: int,
        total_operators: int,
        active_operators: int,
        total_stations: int,
        active_stations: int
    ) -> None:
        """Track overall system metrics"""
        base_tags = ["service:system"]
        
        await self.send_metric("system.total_meters", total_meters, base_tags)
        await self.send_metric("system.active_meters", active_meters, base_tags)
        await self.send_metric("system.total_operators", total_operators, base_tags)
        await self.send_metric("system.active_operators", active_operators, base_tags)
        await self.send_metric("system.total_stations", total_stations, base_tags)
        await self.send_metric("system.active_stations", active_stations, base_tags)
    
    async def track_data_quality_metrics(
        self,
        entity_type: str,
        entity_id: str,
        quality_score: float,
        total_records: int,
        valid_records: int
    ) -> None:
        """Track data quality metrics"""
        base_tags = [f"entity_type:{entity_type}", f"entity_id:{entity_id}"]
        
        await self.send_metric("data_quality.score", quality_score, base_tags)
        await self.send_metric("data_quality.total_records", total_records, base_tags)
        await self.send_metric("data_quality.valid_records", valid_records, base_tags)
        
        # Calculate and track quality percentage
        quality_percentage = (valid_records / total_records * 100) if total_records > 0 else 0
        await self.send_metric("data_quality.percentage", quality_percentage, base_tags)
    
    async def track_performance_metrics(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        error_count: int = 0
    ) -> None:
        """Track performance metrics"""
        base_tags = [f"operation:{operation}", f"success:{success}"]
        
        await self.send_metric("performance.duration", duration_ms, base_tags, "histogram")
        await self.send_metric("performance.error_count", error_count, base_tags)
        
        if success:
            await self.send_metric("performance.success_rate", 1, base_tags)
        else:
            await self.send_metric("performance.success_rate", 0, base_tags)
    
    async def send_alert(
        self,
        alert_title: str,
        alert_message: str,
        severity: str = "warning",
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None
    ) -> None:
        """Send an alert to DataDog"""
        tags = []
        if entity_id:
            tags.append(f"entity_id:{entity_id}")
        if entity_type:
            tags.append(f"entity_type:{entity_type}")
        
        await self.send_event(
            title=alert_title,
            text=alert_message,
            alert_type=severity,
            tags=tags
        )
    
    async def get_dashboard_url(self, dashboard_id: str) -> str:
        """Get DataDog dashboard URL"""
        return f"https://{self.site}/dashboard/{dashboard_id}"
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get DataDog client metrics"""
        return {
            "initialized": self._initialized,
            "service_name": self.service_name,
            "site": self.site
        }
