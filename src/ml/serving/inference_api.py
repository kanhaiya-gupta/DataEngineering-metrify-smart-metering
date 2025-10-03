"""
Inference API

This module provides a high-level API for model inference with features like:
- Batch processing
- Async inference
- Model caching
- Error handling and retries
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for model inference"""
    data: Union[List[List[float]], np.ndarray, pd.DataFrame]
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """Response from model inference"""
    predictions: Union[List[float], np.ndarray]
    model_name: str
    model_version: str
    request_id: Optional[str] = None
    inference_time_ms: float = 0.0
    confidence_scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class InferenceAPI:
    """
    High-level API for model inference operations
    """

    def __init__(self, model_server: Any = None):
        self.model_server = model_server
        self.model_cache: Dict[str, Any] = {}
        self.request_cache: Dict[str, InferenceResponse] = {}
        self.max_cache_size = 1000
        
        if self.model_server:
            logger.info("InferenceAPI initialized with model server.")
        else:
            logger.warning("InferenceAPI initialized without model server. "
                          "Inference will be simulated.")

    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """
        Perform single prediction
        
        Args:
            request: Inference request containing data and metadata
            
        Returns:
            InferenceResponse with predictions and metadata
        """
        start_time = datetime.utcnow()
        request_id = request.request_id or f"req_{datetime.utcnow().timestamp()}"
        
        try:
            # Check cache first
            if request_id in self.request_cache:
                logger.debug(f"Returning cached result for request {request_id}")
                return self.request_cache[request_id]
            
            # Prepare data
            data = self._prepare_data(request.data)
            
            # Get model
            model = await self._get_model(request.model_name, request.model_version)
            
            # Perform inference
            if self.model_server:
                predictions = await self._inference_with_server(model, data)
            else:
                predictions = await self._simulate_inference(data)
            
            # Calculate confidence scores if available
            confidence_scores = None
            if hasattr(model, 'predict_proba'):
                try:
                    confidence_scores = model.predict_proba(data).tolist()
                except Exception as e:
                    logger.warning(f"Could not calculate confidence scores: {e}")
            
            # Create response
            inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            response = InferenceResponse(
                predictions=predictions,
                model_name=request.model_name or "default",
                model_version=request.model_version or "1.0.0",
                request_id=request_id,
                inference_time_ms=inference_time,
                confidence_scores=confidence_scores,
                metadata=request.metadata
            )
            
            # Cache response
            self._cache_response(request_id, response)
            
            logger.info(f"Inference completed for request {request_id} in {inference_time:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Inference failed for request {request_id}: {e}")
            raise

    async def predict_batch(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """
        Perform batch predictions
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of inference responses
        """
        logger.info(f"Processing batch of {len(requests)} requests")
        
        # Process requests concurrently
        tasks = [self.predict(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Request {i} failed: {response}")
                # Create error response
                error_response = InferenceResponse(
                    predictions=[],
                    model_name=requests[i].model_name or "default",
                    model_version=requests[i].model_version or "1.0.0",
                    request_id=requests[i].request_id,
                    metadata={"error": str(response)}
                )
                results.append(error_response)
            else:
                results.append(response)
        
        return results

    def _prepare_data(self, data: Union[List[List[float]], np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Prepare data for inference"""
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    async def _get_model(self, model_name: Optional[str], model_version: Optional[str]) -> Any:
        """Get model from cache or load it"""
        cache_key = f"{model_name or 'default'}_{model_version or 'latest'}"
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        # In a real implementation, this would load the model
        # For now, return a mock model
        mock_model = MockModel()
        self.model_cache[cache_key] = mock_model
        
        # Limit cache size
        if len(self.model_cache) > self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.model_cache))
            del self.model_cache[oldest_key]
        
        return mock_model

    async def _inference_with_server(self, model: Any, data: np.ndarray) -> List[float]:
        """Perform inference using model server"""
        if self.model_server:
            # Convert to the format expected by model server
            request_data = data.tolist()
            result = await self.model_server.predict(request_data)
            return result.get('predictions', [])
        else:
            return await self._simulate_inference(data)

    async def _simulate_inference(self, data: np.ndarray) -> List[float]:
        """Simulate inference for testing"""
        # Simple simulation: return random values based on input shape
        batch_size = data.shape[0] if len(data.shape) > 1 else 1
        predictions = np.random.random(batch_size).tolist()
        
        # Add some delay to simulate processing
        await asyncio.sleep(0.01)
        
        return predictions

    def _cache_response(self, request_id: str, response: InferenceResponse):
        """Cache inference response"""
        if len(self.request_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.request_cache))
            del self.request_cache[oldest_key]
        
        self.request_cache[request_id] = response

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "model_cache_size": len(self.model_cache),
            "request_cache_size": len(self.request_cache),
            "max_cache_size": self.max_cache_size
        }

    def clear_cache(self):
        """Clear all caches"""
        self.model_cache.clear()
        self.request_cache.clear()
        logger.info("All caches cleared")

    def get_health_status(self) -> Dict[str, Any]:
        """Get API health status"""
        return {
            "status": "healthy",
            "model_server_connected": self.model_server is not None,
            "cache_stats": self.get_cache_stats(),
            "last_check": datetime.utcnow().isoformat()
        }


class MockModel:
    """Mock model for testing and simulation"""
    
    def predict(self, data):
        """Mock prediction method"""
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                # Batch prediction
                return np.random.random(len(data)).tolist()
            else:
                # Single prediction
                return [np.random.random()]
        return [0.0]
    
    def predict_proba(self, data):
        """Mock probability prediction method"""
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                # Batch prediction
                batch_size = len(data)
                return np.random.random((batch_size, 2))
            else:
                # Single prediction
                return np.random.random((1, 2))
        return np.array([[0.5, 0.5]])
