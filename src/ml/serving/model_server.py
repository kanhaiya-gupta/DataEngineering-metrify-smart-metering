"""
Model Server

This module implements a production-ready model server for serving ML models
with features like load balancing, health checks, and metrics collection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import asyncio
import json
from dataclasses import dataclass
import threading
import time

import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.tensorflow
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# Pydantic models for API requests
class PredictionRequest(BaseModel):
    data: List[List[float]]
    model_name: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    data: List[List[List[float]]]
    model_name: Optional[str] = None


@dataclass
class ServerConfig:
    """Configuration for model server"""
    host: str = "0.0.0.0"
    port: int = 2500
    model_path: str = None
    model_name: str = "smart_meter_model"
    version: str = "1.0.0"
    max_batch_size: int = 32
    timeout: int = 30
    health_check_interval: int = 60
    enable_metrics: bool = True
    enable_logging: bool = True


class ModelServer:
    """
    Production-ready model server for ML models
    
    Features:
    - FastAPI-based REST API
    - Model loading and caching
    - Health checks and monitoring
    - Metrics collection
    - Error handling and logging
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.app = FastAPI(
            title=f"{config.model_name} Model Server",
            version=config.version,
            description="ML Model Serving API for Smart Meter Data Pipeline"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Model and state
        self.model = None
        self.model_metadata = {}
        self.is_loaded = False
        self.load_time = None
        self.request_count = 0
        self.error_count = 0
        self.last_health_check = None
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "last_request_time": None
        }
        
        # Setup routes
        self._setup_routes()
        
        # Start health check background task
        self._start_health_check()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return await self._health_check()
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get server metrics"""
            return self.metrics
        
        @self.app.get("/model/info")
        async def get_model_info():
            """Get model information"""
            return self.model_metadata
        
        @self.app.post("/predict")
        async def predict(request: PredictionRequest):
            """Make predictions"""
            return await self._predict(request)
        
        @self.app.post("/predict/batch")
        async def predict_batch(request: BatchPredictionRequest):
            """Make batch predictions"""
            return await self._predict_batch(request)
        
        @self.app.post("/model/reload")
        async def reload_model():
            """Reload the model"""
            return await self._reload_model()
    
    async def _health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        self.last_health_check = datetime.now()
        
        health_status = {
            "status": "healthy",
            "timestamp": self.last_health_check.isoformat(),
            "model_loaded": self.is_loaded,
            "uptime": self._get_uptime(),
            "metrics": self.metrics
        }
        
        # Check model status
        if not self.is_loaded:
            health_status["status"] = "unhealthy"
            health_status["issues"] = ["model_not_loaded"]
        
        # Check for recent errors
        if self.error_count > 0:
            error_rate = self.error_count / max(self.request_count, 1)
            if error_rate > 0.1:  # 10% error rate threshold
                health_status["status"] = "degraded"
                health_status["issues"] = health_status.get("issues", []) + ["high_error_rate"]
        
        return health_status
    
    async def _predict(self, request: PredictionRequest) -> Dict[str, Any]:
        """Make single prediction"""
        start_time = time.time()
        
        try:
            # Validate request
            if not self.is_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            # Convert request to numpy array
            X = np.array(request.data)
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(response_time, success=True)
            
            return {
                "prediction": prediction.tolist(),
                "model_name": self.config.model_name,
                "version": self.config.version,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self._update_metrics(time.time() - start_time, success=False)
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _predict_batch(self, request: BatchPredictionRequest) -> Dict[str, Any]:
        """Make batch predictions"""
        start_time = time.time()
        
        try:
            # Validate request
            if not self.is_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            if len(request.data) > self.config.max_batch_size:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Batch size {len(request.data)} exceeds maximum {self.config.max_batch_size}"
                )
            
            # Convert request to numpy array
            X = np.array(request.data)
            
            # Make predictions
            predictions = self.model.predict(X, verbose=0)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(response_time, success=True)
            
            return {
                "predictions": predictions.tolist(),
                "batch_size": len(request.data),
                "model_name": self.config.model_name,
                "version": self.config.version,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self._update_metrics(time.time() - start_time, success=False)
            logger.error(f"Batch prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _reload_model(self) -> Dict[str, Any]:
        """Reload the model"""
        try:
            await self.load_model(self.config.model_path)
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Model reload failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def load_model(self, model_path: str):
        """Load the model"""
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Load model
            if model_path.startswith("mlflow://"):
                model_uri = model_path.replace("mlflow://", "")
                self.model = mlflow.tensorflow.load_model(model_uri)
            else:
                self.model = keras.models.load_model(model_path)
            
            # Store model metadata
            self.model_metadata = {
                "model_name": self.config.model_name,
                "version": self.config.version,
                "model_path": model_path,
                "load_time": datetime.now().isoformat(),
                "input_shape": self.model.input_shape,
                "output_shape": self.model.output_shape
            }
            
            self.is_loaded = True
            self.load_time = datetime.now()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _update_metrics(self, response_time: float, success: bool):
        """Update server metrics"""
        self.request_count += 1
        self.metrics["total_requests"] = self.request_count
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.error_count += 1
            self.metrics["failed_requests"] = self.error_count
        
        # Update average response time
        current_avg = self.metrics["average_response_time"]
        total_requests = self.metrics["successful_requests"]
        
        if total_requests > 0:
            self.metrics["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
        
        self.metrics["last_request_time"] = datetime.now().isoformat()
    
    def _get_uptime(self) -> str:
        """Get server uptime"""
        if self.load_time is None:
            return "0s"
        
        uptime = datetime.now() - self.load_time
        return str(uptime)
    
    def _start_health_check(self):
        """Start background health check task"""
        async def health_check_loop():
            while True:
                try:
                    await self._health_check()
                except Exception as e:
                    logger.error(f"Health check failed: {str(e)}")
                
                await asyncio.sleep(self.config.health_check_interval)
        
        # Start the health check loop
        asyncio.create_task(health_check_loop())
    
    def start_server(self):
        """Start the model server"""
        logger.info(f"Starting model server on {self.config.host}:{self.config.port}")
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
    
    def run_server(self):
        """Run the server (blocking)"""
        self.start_server()


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = ServerConfig(
        model_path="models/consumption_forecasting_model.h5",
        model_name="consumption_forecasting",
        version="1.0.0"
    )
    
    # Create and start server
    server = ModelServer(config)
    server.run_server()
