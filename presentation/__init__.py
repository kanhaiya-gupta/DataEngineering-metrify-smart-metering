"""
Presentation Layer Module
Contains API endpoints, CLI tools, and background workers
"""

from .api.main import app, get_monitoring_service
from .api.v1.smart_meter_endpoints import router as smart_meter_router
from .api.v1.grid_operator_endpoints import router as grid_operator_router
from .api.v1.weather_endpoints import router as weather_router
from .api.v1.analytics_endpoints import router as analytics_router
from .api.middleware.auth_middleware import AuthMiddleware, AuthService
from .api.middleware.logging_middleware import LoggingMiddleware
from .api.middleware.monitoring_middleware import MonitoringMiddleware
from .cli.data_ingestion_cli import cli as data_ingestion_cli
from .cli.quality_check_cli import cli as quality_check_cli
from .cli.maintenance_cli import cli as maintenance_cli
from .workers.ingestion_worker import IngestionWorker
from .workers.processing_worker import ProcessingWorker
from .workers.monitoring_worker import MonitoringWorker

__all__ = [
    # FastAPI App
    "app",
    "get_monitoring_service",
    
    # API Routers
    "smart_meter_router",
    "grid_operator_router", 
    "weather_router",
    "analytics_router",
    
    # Middleware
    "AuthMiddleware",
    "AuthService",
    "LoggingMiddleware",
    "MonitoringMiddleware",
    
    # CLI Tools
    "data_ingestion_cli",
    "quality_check_cli",
    "maintenance_cli",
    
    # Background Workers
    "IngestionWorker",
    "ProcessingWorker",
    "MonitoringWorker"
]
