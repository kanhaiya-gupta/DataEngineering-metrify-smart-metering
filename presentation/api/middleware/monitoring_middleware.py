"""
Monitoring Middleware
Handles metrics collection and performance monitoring
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config.config_loader import get_monitoring_config

logger = logging.getLogger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Monitoring middleware for metrics collection
    
    Handles performance metrics, error tracking, and monitoring
    integration for comprehensive API observability.
    """
    
    def __init__(self, app, monitoring_service=None):
        super().__init__(app)
        self.monitoring_service = monitoring_service
        self.monitoring_config = get_monitoring_config()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timing
        start_time = time.time()
        
        # Track request start
        await self._track_request_start(request)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Track successful request
            await self._track_request_success(request, response, duration)
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Track failed request
            await self._track_request_error(request, e, duration)
            raise
    
    async def _track_request_start(self, request: Request) -> None:
        """Track request start metrics"""
        try:
            if not self.monitoring_service:
                return
            
            # Increment request counter
            self.monitoring_service.prometheus.increment_operation_requests(
                operation="api_request",
                status="started"
            )
            
        except Exception as e:
            logger.error(f"Error tracking request start: {str(e)}")
    
    async def _track_request_success(
        self,
        request: Request,
        response: Response,
        duration: float
    ) -> None:
        """Track successful request metrics"""
        try:
            if not self.monitoring_service:
                return
            
            # Determine operation type
            operation = self._get_operation_type(request)
            
            # Track performance metrics
            await self.monitoring_service.track_operation_performance(
                operation=operation,
                duration=duration,
                success=True
            )
            
            # Track specific metrics based on endpoint
            await self._track_endpoint_metrics(request, response, duration)
            
        except Exception as e:
            logger.error(f"Error tracking request success: {str(e)}")
    
    async def _track_request_error(
        self,
        request: Request,
        error: Exception,
        duration: float
    ) -> None:
        """Track failed request metrics"""
        try:
            if not self.monitoring_service:
                return
            
            # Determine operation type
            operation = self._get_operation_type(request)
            
            # Track performance metrics
            await self.monitoring_service.track_operation_performance(
                operation=operation,
                duration=duration,
                success=False,
                error_count=1
            )
            
            # Track error metrics
            self.monitoring_service.prometheus.increment_operation_requests(
                operation=operation,
                status="error"
            )
            
        except Exception as e:
            logger.error(f"Error tracking request error: {str(e)}")
    
    async def _track_endpoint_metrics(
        self,
        request: Request,
        response: Response,
        duration: float
    ) -> None:
        """Track endpoint-specific metrics"""
        try:
            if not self.monitoring_service:
                return
            
            path = request.url.path
            method = request.method
            status_code = response.status_code
            
            # Track smart meter endpoints
            if "/smart-meters" in path:
                await self._track_smart_meter_metrics(request, response, duration)
            
            # Track grid operator endpoints
            elif "/grid-operators" in path:
                await self._track_grid_operator_metrics(request, response, duration)
            
            # Track weather endpoints
            elif "/weather" in path:
                await self._track_weather_metrics(request, response, duration)
            
            # Track analytics endpoints
            elif "/analytics" in path:
                await self._track_analytics_metrics(request, response, duration)
            
        except Exception as e:
            logger.error(f"Error tracking endpoint metrics: {str(e)}")
    
    async def _track_smart_meter_metrics(
        self,
        request: Request,
        response: Response,
        duration: float
    ) -> None:
        """Track smart meter specific metrics"""
        try:
            # Extract meter ID from path if available
            path_parts = request.url.path.split("/")
            meter_id = None
            for i, part in enumerate(path_parts):
                if part == "smart-meters" and i + 1 < len(path_parts):
                    meter_id = path_parts[i + 1]
                    break
            
            if meter_id:
                # Track meter-specific metrics
                self.monitoring_service.prometheus.increment_meter_readings(
                    meter_id=meter_id,
                    status="api_request"
                )
            
        except Exception as e:
            logger.error(f"Error tracking smart meter metrics: {str(e)}")
    
    async def _track_grid_operator_metrics(
        self,
        request: Request,
        response: Response,
        duration: float
    ) -> None:
        """Track grid operator specific metrics"""
        try:
            # Extract operator ID from path if available
            path_parts = request.url.path.split("/")
            operator_id = None
            for i, part in enumerate(path_parts):
                if part == "grid-operators" and i + 1 < len(path_parts):
                    operator_id = path_parts[i + 1]
                    break
            
            if operator_id:
                # Track operator-specific metrics
                self.monitoring_service.prometheus.increment_grid_status_updates(
                    operator_id=operator_id,
                    status="api_request"
                )
            
        except Exception as e:
            logger.error(f"Error tracking grid operator metrics: {str(e)}")
    
    async def _track_weather_metrics(
        self,
        request: Request,
        response: Response,
        duration: float
    ) -> None:
        """Track weather station specific metrics"""
        try:
            # Extract station ID from path if available
            path_parts = request.url.path.split("/")
            station_id = None
            for i, part in enumerate(path_parts):
                if part == "weather" and i + 1 < len(path_parts):
                    station_id = path_parts[i + 1]
                    break
            
            if station_id:
                # Track station-specific metrics
                self.monitoring_service.prometheus.increment_weather_observations(
                    station_id=station_id,
                    status="api_request"
                )
            
        except Exception as e:
            logger.error(f"Error tracking weather metrics: {str(e)}")
    
    async def _track_analytics_metrics(
        self,
        request: Request,
        response: Response,
        duration: float
    ) -> None:
        """Track analytics specific metrics"""
        try:
            # Track analytics operation
            self.monitoring_service.prometheus.increment_operation_requests(
                operation="analytics_request",
                status="success" if response.status_code < 400 else "error"
            )
            
        except Exception as e:
            logger.error(f"Error tracking analytics metrics: {str(e)}")
    
    def _get_operation_type(self, request: Request) -> str:
        """Get operation type from request path"""
        path = request.url.path
        method = request.method
        
        # Smart meter operations
        if "/smart-meters" in path:
            if method == "GET":
                return "smart_meter_read"
            elif method == "POST":
                return "smart_meter_create"
            elif method == "PUT":
                return "smart_meter_update"
            elif method == "DELETE":
                return "smart_meter_delete"
        
        # Grid operator operations
        elif "/grid-operators" in path:
            if method == "GET":
                return "grid_operator_read"
            elif method == "POST":
                return "grid_operator_create"
            elif method == "PUT":
                return "grid_operator_update"
            elif method == "DELETE":
                return "grid_operator_delete"
        
        # Weather operations
        elif "/weather" in path:
            if method == "GET":
                return "weather_read"
            elif method == "POST":
                return "weather_create"
            elif method == "PUT":
                return "weather_update"
            elif method == "DELETE":
                return "weather_delete"
        
        # Analytics operations
        elif "/analytics" in path:
            return "analytics_request"
        
        # Default operation
        return "api_request"
