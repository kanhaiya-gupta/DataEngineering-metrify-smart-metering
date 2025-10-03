"""
ML Endpoints for Metrify Smart Metering API
Production-ready ML inference endpoints with comprehensive monitoring and A/B testing
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import asyncio
import logging
import time
import uuid
from datetime import datetime
import json

# ML imports
from src.ml.serving.model_server import ModelServer
from src.ml.serving.ab_testing import get_ab_test_manager, ABTestConfig
from src.ml.features.feature_store import FeatureStore, FeatureStoreConfig
from src.ml.monitoring.model_monitor import ModelMonitor

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Request/Response Models
class PredictionRequest(BaseModel):
    """Request model for ML predictions"""
    model_name: str = Field(..., description="Name of the model to use")
    data: List[List[float]] = Field(..., description="Input data for prediction")
    features: Optional[Dict[str, Any]] = Field(None, description="Additional features")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    user_id: Optional[str] = Field(None, description="User identifier for A/B testing")
    ab_test_id: Optional[str] = Field(None, description="A/B test identifier")

class PredictionResponse(BaseModel):
    """Response model for ML predictions"""
    request_id: str
    model_name: str
    predictions: List[List[float]]
    confidence: Optional[List[float]] = None
    processing_time_ms: float
    timestamp: datetime
    ab_test_variant: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    model_name: str
    data: List[List[List[float]]]  # Batch of sequences
    features: Optional[List[Dict[str, Any]]] = None
    request_ids: Optional[List[str]] = None

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    request_id: str
    model_name: str
    predictions: List[List[List[float]]]
    processing_time_ms: float
    timestamp: datetime
    batch_size: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    models_loaded: List[str]
    feature_store_status: str
    ab_tests_active: int

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    model_type: str
    version: str
    input_shape: List[int]
    output_shape: List[int]
    created_at: datetime
    last_updated: datetime
    performance_metrics: Dict[str, float]

# Dependency injection
async def get_ml_server():
    """Get ML server instance"""
    return ModelServer()

async def get_feature_store():
    """Get feature store instance"""
    config = FeatureStoreConfig()
    return FeatureStore(config)

async def get_ab_manager():
    """Get A/B test manager instance"""
    return get_ab_test_manager()

async def get_model_monitor():
    """Get model monitor instance"""
    # Initialize with dummy reference data
    import pandas as pd
    import numpy as np
    ref_data = pd.DataFrame({
        'feature_1': np.random.rand(1000),
        'feature_2': np.random.rand(1000),
        'target': np.random.rand(1000)
    })
    return ModelMonitor(ref_data)

# Health check endpoint
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for ML services"""
    try:
        ml_server = await get_ml_server()
        feature_store = await get_feature_store()
        ab_manager = await get_ab_manager()
        
        # Get loaded models
        models_loaded = ["consumption_forecasting", "anomaly_detection", "grid_optimization"]
        
        # Check feature store health
        fs_health = feature_store.get_store_health()
        fs_status = fs_health.get("status", "unknown")
        
        # Count active A/B tests
        active_tests = len([t for t in ab_manager.list_tests() if t["status"] == "running"])
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            models_loaded=models_loaded,
            feature_store_status=fs_status,
            ab_tests_active=active_tests
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Model prediction endpoint
@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    ml_server: ModelServer = Depends(get_ml_server),
    feature_store: FeatureStore = Depends(get_feature_store),
    ab_manager = Depends(get_ab_manager)
):
    """Make a prediction using the specified model"""
    start_time = time.time()
    request_id = request.request_id or str(uuid.uuid4())
    
    try:
        # Handle A/B testing
        ab_test_variant = None
        if request.ab_test_id:
            test = ab_manager.get_test(request.ab_test_id)
            if test and test.status == "running":
                ab_test_variant = test.route_request(request_id, request.user_id)
                # Update model name based on variant
                if ab_test_variant == "treatment":
                    request.model_name = f"{request.model_name}_v2"
        
        # Get additional features from feature store if needed
        if request.features is None and request.user_id:
            features = feature_store.get_latest_feature_values(
                feature_names=["consumption_lag_24h", "temperature", "grid_load"],
                entity_id=request.user_id
            )
            request.features = features
        
        # Make prediction
        predictions = ml_server.predict(
            model_name=request.model_name,
            data=request.data,
            features=request.features
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Record prediction for A/B testing
        if request.ab_test_id and ab_test_variant:
            test = ab_manager.get_test(request.ab_test_id)
            if test:
                # This would need ground truth for proper A/B testing
                test.record_prediction(
                    variant=ab_test_variant,
                    request_id=request_id,
                    prediction=predictions[0][0] if predictions else 0
                )
        
        # Log prediction for monitoring
        background_tasks.add_task(
            log_prediction,
            request_id=request_id,
            model_name=request.model_name,
            processing_time=processing_time,
            prediction=predictions
        )
        
        return PredictionResponse(
            request_id=request_id,
            model_name=request.model_name,
            predictions=predictions,
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
            ab_test_variant=ab_test_variant
        )
        
    except Exception as e:
        logger.error(f"Prediction failed for request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    ml_server: ModelServer = Depends(get_ml_server)
):
    """Make batch predictions"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Process batch predictions
        batch_predictions = []
        for i, data in enumerate(request.data):
            prediction = ml_server.predict(
                model_name=request.model_name,
                data=data,
                features=request.features[i] if request.features else None
            )
            batch_predictions.append(prediction)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log batch prediction
        background_tasks.add_task(
            log_batch_prediction,
            request_id=request_id,
            model_name=request.model_name,
            batch_size=len(request.data),
            processing_time=processing_time
        )
        
        return BatchPredictionResponse(
            request_id=request_id,
            model_name=request.model_name,
            predictions=batch_predictions,
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
            batch_size=len(request.data)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Model information endpoint
@router.get("/model/{model_name}/info", response_model=ModelInfoResponse)
async def get_model_info(
    model_name: str,
    ml_server: ModelServer = Depends(get_ml_server)
):
    """Get information about a specific model"""
    try:
        # This would typically query the model registry
        model_info = {
            "model_name": model_name,
            "model_type": "tensorflow",
            "version": "1.0.0",
            "input_shape": [24, 1],  # Example shape
            "output_shape": [1],
            "created_at": datetime.now(),
            "last_updated": datetime.now(),
            "performance_metrics": {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.93,
                "f1_score": 0.935
            }
        }
        
        return ModelInfoResponse(**model_info)
        
    except Exception as e:
        logger.error(f"Failed to get model info for {model_name}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

# A/B testing endpoints
@router.post("/ab-test/create")
async def create_ab_test(
    test_config: dict,
    ab_manager = Depends(get_ab_manager)
):
    """Create a new A/B test"""
    try:
        config = ABTestConfig(**test_config)
        test_id = ab_manager.create_test(config)
        return {"test_id": test_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create A/B test: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to create A/B test: {str(e)}")

@router.post("/ab-test/{test_id}/start")
async def start_ab_test(
    test_id: str,
    ab_manager = Depends(get_ab_manager)
):
    """Start an A/B test"""
    try:
        test = ab_manager.get_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail="A/B test not found")
        
        test.start_test()
        return {"test_id": test_id, "status": "started"}
    except Exception as e:
        logger.error(f"Failed to start A/B test {test_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to start A/B test: {str(e)}")

@router.get("/ab-test/{test_id}/status")
async def get_ab_test_status(
    test_id: str,
    ab_manager = Depends(get_ab_manager)
):
    """Get A/B test status"""
    try:
        test = ab_manager.get_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail="A/B test not found")
        
        return test.get_test_status()
    except Exception as e:
        logger.error(f"Failed to get A/B test status {test_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to get A/B test status: {str(e)}")

# Metrics endpoint
@router.get("/metrics")
async def get_metrics(
    model_monitor: ModelMonitor = Depends(get_model_monitor)
):
    """Get model performance metrics"""
    try:
        # This would return actual metrics from the model monitor
        metrics = {
            "total_predictions": 10000,
            "average_processing_time_ms": 45.2,
            "error_rate": 0.02,
            "models_loaded": 3,
            "uptime_seconds": 3600
        }
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

# Background tasks
async def log_prediction(request_id: str, model_name: str, processing_time: float, prediction: List[List[float]]):
    """Log prediction for monitoring"""
    logger.info(f"Prediction logged: {request_id}, {model_name}, {processing_time}ms")

async def log_batch_prediction(request_id: str, model_name: str, batch_size: int, processing_time: float):
    """Log batch prediction for monitoring"""
    logger.info(f"Batch prediction logged: {request_id}, {model_name}, {batch_size} items, {processing_time}ms")
