"""
Main FastAPI Application
Entry point for the Metrify Smart Metering API
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .v1.smart_meter_endpoints import router as smart_meter_router
from .v1.grid_operator_endpoints import router as grid_operator_router
from .v1.weather_endpoints import router as weather_router
from .v1.analytics_endpoints import router as analytics_router
from .middleware.auth_middleware import AuthMiddleware
from .middleware.logging_middleware import LoggingMiddleware
from .middleware.monitoring_middleware import MonitoringMiddleware
from src.core.config.config_loader import get_app_config
from src.infrastructure.external.monitoring.monitoring_service import MonitoringService
from src.infrastructure.external.monitoring.prometheus.prometheus_client import PrometheusClient
from src.infrastructure.external.monitoring.grafana.grafana_client import GrafanaClient
from src.infrastructure.external.monitoring.jaeger.jaeger_client import JaegerClient
from src.infrastructure.external.monitoring.datadog.datadog_client import DataDogClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global monitoring service instance
monitoring_service: MonitoringService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global monitoring_service
    
    try:
        # Initialize monitoring service
        app_config = get_app_config()
        
        # Create monitoring clients
        prometheus_client = PrometheusClient(
            namespace="metrify",
            subsystem="smart_metering_api"
        )
        
        grafana_client = GrafanaClient(
            base_url=app_config.grafana_url,
            username=app_config.grafana_username,
            password=app_config.grafana_password
        )
        
        jaeger_client = JaegerClient(
            service_name="metrify-smart-metering-api",
            agent_host=app_config.jaeger_agent_host,
            agent_port=app_config.jaeger_agent_port
        )
        
        datadog_client = DataDogClient(
            api_key=app_config.datadog_api_key,
            app_key=app_config.datadog_app_key,
            site=app_config.datadog_site
        )
        
        # Initialize monitoring service
        monitoring_service = MonitoringService(
            prometheus_client=prometheus_client,
            grafana_client=grafana_client,
            jaeger_client=jaeger_client,
            datadog_client=datadog_client
        )
        
        await monitoring_service.initialize()
        
        # Setup monitoring dashboards
        await monitoring_service.setup_monitoring_dashboards()
        
        logger.info("Application startup completed")
        yield
        
    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}")
        raise
    finally:
        # Cleanup
        if monitoring_service:
            await monitoring_service.shutdown()
        logger.info("Application shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="Metrify Smart Metering API",
    description="Enterprise-grade data pipeline for smart metering operations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add custom middleware
app.add_middleware(AuthMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(MonitoringMiddleware)

# Include routers
app.include_router(
    smart_meter_router,
    prefix="/api/v1/smart-meters",
    tags=["Smart Meters"]
)

app.include_router(
    grid_operator_router,
    prefix="/api/v1/grid-operators",
    tags=["Grid Operators"]
)

app.include_router(
    weather_router,
    prefix="/api/v1/weather",
    tags=["Weather Stations"]
)

app.include_router(
    analytics_router,
    prefix="/api/v1/analytics",
    tags=["Analytics"]
)


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint"""
    return {
        "service": "Metrify Smart Metering API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    try:
        # Get monitoring status
        monitoring_status = await monitoring_service.get_monitoring_status() if monitoring_service else {}
        
        return {
            "status": "healthy",
            "timestamp": "2024-01-20T15:45:00Z",
            "version": "1.0.0",
            "monitoring": monitoring_status,
            "components": {
                "api": "healthy",
                "database": "healthy",
                "kafka": "healthy",
                "monitoring": "healthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "2024-01-20T15:45:00Z"
            }
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        if monitoring_service:
            metrics_data = monitoring_service.prometheus.get_metrics()
            return Response(
                content=metrics_data,
                media_type="text/plain"
            )
        else:
            return Response(
                content="# Monitoring service not initialized\n",
                media_type="text/plain"
            )
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return Response(
            content=f"# Error getting metrics: {str(e)}\n",
            media_type="text/plain",
            status_code=500
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    # Track error in monitoring
    if monitoring_service:
        await monitoring_service.track_operation_performance(
            operation="api_request",
            duration=0.0,
            success=False,
            error_count=1
        )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, 'request_id', None)
        }
    )


def get_monitoring_service() -> MonitoringService:
    """Get the global monitoring service instance"""
    return monitoring_service


if __name__ == "__main__":
    uvicorn.run(
        "presentation.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
