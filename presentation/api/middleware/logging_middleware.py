"""
Logging Middleware
Handles request/response logging and correlation IDs
"""

import logging
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logging middleware for request/response logging
    
    Handles structured logging, correlation IDs, and request tracking
    for comprehensive API monitoring.
    """
    
    def __init__(self, app, log_requests: bool = True, log_responses: bool = True):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        request.state.request_id = correlation_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        if self.log_requests:
            await self._log_request(request, correlation_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            if self.log_responses:
                await self._log_response(request, response, duration, correlation_id)
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Request-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            await self._log_error(request, e, duration, correlation_id)
            raise
    
    async def _log_request(self, request: Request, correlation_id: str) -> None:
        """Log incoming request"""
        try:
            # Extract client IP
            client_ip = request.client.host if request.client else "unknown"
            
            # Extract user agent
            user_agent = request.headers.get("user-agent", "unknown")
            
            # Extract authorization info
            authorization = request.headers.get("authorization", "")
            auth_type = "authenticated" if authorization.startswith("Bearer ") else "anonymous"
            
            # Log request
            logger.info(
                f"Request started",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "url": str(request.url),
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "client_ip": client_ip,
                    "user_agent": user_agent,
                    "auth_type": auth_type,
                    "content_type": request.headers.get("content-type", ""),
                    "content_length": request.headers.get("content-length", "0"),
                    "event_type": "request_started"
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging request: {str(e)}")
    
    async def _log_response(
        self,
        request: Request,
        response: Response,
        duration: float,
        correlation_id: str
    ) -> None:
        """Log outgoing response"""
        try:
            # Determine response status
            status_code = response.status_code
            status_category = self._get_status_category(status_code)
            
            # Log response
            logger.info(
                f"Request completed",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "url": str(request.url),
                    "path": request.url.path,
                    "status_code": status_code,
                    "status_category": status_category,
                    "duration_ms": round(duration * 1000, 2),
                    "response_size": response.headers.get("content-length", "0"),
                    "event_type": "request_completed"
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging response: {str(e)}")
    
    async def _log_error(
        self,
        request: Request,
        error: Exception,
        duration: float,
        correlation_id: str
    ) -> None:
        """Log request error"""
        try:
            # Log error
            logger.error(
                f"Request failed",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "url": str(request.url),
                    "path": request.url.path,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "duration_ms": round(duration * 1000, 2),
                    "event_type": "request_failed"
                },
                exc_info=True
            )
            
        except Exception as e:
            logger.error(f"Error logging error: {str(e)}")
    
    def _get_status_category(self, status_code: int) -> str:
        """Get status code category"""
        if 200 <= status_code < 300:
            return "success"
        elif 300 <= status_code < 400:
            return "redirect"
        elif 400 <= status_code < 500:
            return "client_error"
        elif 500 <= status_code < 600:
            return "server_error"
        else:
            return "unknown"
