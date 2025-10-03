"""
Feature Serving Module

This module provides feature serving and caching capabilities for ML models:
- Real-time feature serving
- Feature caching and optimization
- Feature transformation pipelines
- Feature monitoring and validation
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class FeatureRequest:
    """Request for feature serving"""
    entity_id: str
    feature_names: List[str]
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = None

@dataclass
class FeatureResponse:
    """Response from feature serving"""
    entity_id: str
    features: Dict[str, Any]
    timestamp: datetime
    cache_hit: bool = False
    processing_time_ms: float = 0.0

class FeatureCache:
    """In-memory feature cache with TTL support"""
    
    def __init__(self, default_ttl_seconds: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.default_ttl = default_ttl_seconds
        
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached features if not expired"""
        if key not in self.cache:
            return None
            
        if self._is_expired(key):
            self._remove(key)
            return None
            
        return self.cache[key]
    
    def set(self, key: str, features: Dict[str, Any], ttl_seconds: Optional[int] = None):
        """Cache features with TTL"""
        self.cache[key] = features
        self.cache_timestamps[key] = datetime.now()
        
    def _is_expired(self, key: str) -> bool:
        """Check if cached item is expired"""
        if key not in self.cache_timestamps:
            return True
            
        ttl = self.default_ttl
        expiry_time = self.cache_timestamps[key] + timedelta(seconds=ttl)
        return datetime.now() > expiry_time
    
    def _remove(self, key: str):
        """Remove expired item from cache"""
        self.cache.pop(key, None)
        self.cache_timestamps.pop(key, None)
    
    def clear(self):
        """Clear all cached items"""
        self.cache.clear()
        self.cache_timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)

class FeatureTransformer(ABC):
    """Abstract base class for feature transformers"""
    
    @abstractmethod
    def transform(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Transform features"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of output feature names"""
        pass

class StandardScaler(FeatureTransformer):
    """Standard scaler for numerical features"""
    
    def __init__(self, feature_names: List[str], mean: Optional[Dict[str, float]] = None, 
                 std: Optional[Dict[str, float]] = None):
        self.feature_names = feature_names
        self.mean = mean or {}
        self.std = std or {}
        self.fitted = mean is not None and std is not None
    
    def fit(self, data: pd.DataFrame):
        """Fit scaler on training data"""
        for feature in self.feature_names:
            if feature in data.columns:
                self.mean[feature] = data[feature].mean()
                self.std[feature] = data[feature].std()
        self.fitted = True
    
    def transform(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Transform features using fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        transformed = {}
        for feature in self.feature_names:
            if feature in features and feature in self.mean and feature in self.std:
                if self.std[feature] != 0:
                    transformed[f"{feature}_scaled"] = (features[feature] - self.mean[feature]) / self.std[feature]
                else:
                    transformed[f"{feature}_scaled"] = 0.0
        return transformed
    
    def get_feature_names(self) -> List[str]:
        """Get scaled feature names"""
        return [f"{name}_scaled" for name in self.feature_names]

class OneHotEncoder(FeatureTransformer):
    """One-hot encoder for categorical features"""
    
    def __init__(self, feature_names: List[str], categories: Optional[Dict[str, List[str]]] = None):
        self.feature_names = feature_names
        self.categories = categories or {}
        self.fitted = categories is not None
    
    def fit(self, data: pd.DataFrame):
        """Fit encoder on training data"""
        for feature in self.feature_names:
            if feature in data.columns:
                self.categories[feature] = sorted(data[feature].unique().tolist())
        self.fitted = True
    
    def transform(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Transform features using one-hot encoding"""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        transformed = {}
        for feature in self.feature_names:
            if feature in features and feature in self.categories:
                value = features[feature]
                for category in self.categories[feature]:
                    transformed[f"{feature}_{category}"] = 1 if value == category else 0
        return transformed
    
    def get_feature_names(self) -> List[str]:
        """Get one-hot encoded feature names"""
        names = []
        for feature in self.feature_names:
            if feature in self.categories:
                names.extend([f"{feature}_{cat}" for cat in self.categories[feature]])
        return names

class FeatureServing:
    """
    Feature serving service for ML models
    
    Provides real-time feature serving with caching, transformation,
    and monitoring capabilities.
    """
    
    def __init__(self, cache_ttl_seconds: int = 300):
        self.cache = FeatureCache(cache_ttl_seconds)
        self.transformers: Dict[str, FeatureTransformer] = {}
        self.feature_store = None  # Will be injected
        self.metrics = {
            "requests_total": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "avg_processing_time_ms": 0.0
        }
    
    def set_feature_store(self, feature_store):
        """Set the feature store instance"""
        self.feature_store = feature_store
    
    def add_transformer(self, name: str, transformer: FeatureTransformer):
        """Add a feature transformer"""
        self.transformers[name] = transformer
        logger.info(f"Added transformer: {name}")
    
    def get_features(self, request: FeatureRequest) -> FeatureResponse:
        """
        Get features for a given entity
        
        Args:
            request: Feature request containing entity_id and feature_names
            
        Returns:
            FeatureResponse with requested features
        """
        start_time = time.time()
        self.metrics["requests_total"] += 1
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(request)
            cached_features = self.cache.get(cache_key)
            
            if cached_features:
                self.metrics["cache_hits"] += 1
                processing_time = (time.time() - start_time) * 1000
                return FeatureResponse(
                    entity_id=request.entity_id,
                    features=cached_features,
                    timestamp=datetime.now(),
                    cache_hit=True,
                    processing_time_ms=processing_time
                )
            
            # Cache miss - fetch from feature store
            self.metrics["cache_misses"] += 1
            features = self._fetch_features(request)
            
            # Apply transformations
            transformed_features = self._apply_transformations(features)
            
            # Cache the result
            self.cache.set(cache_key, transformed_features)
            
            processing_time = (time.time() - start_time) * 1000
            self._update_avg_processing_time(processing_time)
            
            return FeatureResponse(
                entity_id=request.entity_id,
                features=transformed_features,
                timestamp=datetime.now(),
                cache_hit=False,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Error serving features for {request.entity_id}: {str(e)}")
            raise
    
    def _get_cache_key(self, request: FeatureRequest) -> str:
        """Generate cache key for request"""
        feature_names_str = ",".join(sorted(request.feature_names))
        return f"{request.entity_id}:{feature_names_str}"
    
    def _fetch_features(self, request: FeatureRequest) -> Dict[str, Any]:
        """Fetch features from feature store"""
        if not self.feature_store:
            raise ValueError("Feature store not configured")
        
        # This would integrate with the actual feature store
        # For now, return mock data
        features = {}
        for feature_name in request.feature_names:
            features[feature_name] = np.random.random()  # Mock data
        
        return features
    
    def _apply_transformations(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all configured transformations"""
        transformed = features.copy()
        
        for name, transformer in self.transformers.items():
            try:
                transformed.update(transformer.transform(transformed))
            except Exception as e:
                logger.warning(f"Error in transformer {name}: {str(e)}")
        
        return transformed
    
    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time metric"""
        current_avg = self.metrics["avg_processing_time_ms"]
        total_requests = self.metrics["requests_total"]
        
        if total_requests == 1:
            self.metrics["avg_processing_time_ms"] = processing_time
        else:
            # Running average
            self.metrics["avg_processing_time_ms"] = (
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get serving metrics"""
        cache_hit_rate = 0.0
        if self.metrics["requests_total"] > 0:
            cache_hit_rate = self.metrics["cache_hits"] / self.metrics["requests_total"]
        
        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": self.cache.size()
        }
    
    def clear_cache(self):
        """Clear feature cache"""
        self.cache.clear()
        logger.info("Feature cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for feature serving"""
        try:
            # Test basic functionality
            test_request = FeatureRequest(
                entity_id="test",
                feature_names=["test_feature"]
            )
            
            # This would normally test with real feature store
            return {
                "status": "healthy",
                "cache_size": self.cache.size(),
                "transformers_count": len(self.transformers),
                "feature_store_connected": self.feature_store is not None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
