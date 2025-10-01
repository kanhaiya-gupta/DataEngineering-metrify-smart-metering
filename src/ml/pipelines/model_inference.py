"""
Model Inference Pipeline

This module implements real-time model inference for smart meter data,
including consumption forecasting, anomaly detection, and grid optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import json
import asyncio
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.tensorflow
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for model inference"""
    model_path: str
    feature_columns: List[str]
    sequence_length: int = 24
    forecast_horizon: int = 1
    batch_size: int = 32
    confidence_threshold: float = 0.95
    anomaly_threshold: float = 0.05


class ModelInferencePipeline:
    """
    Real-time model inference pipeline for smart meter data
    
    Supports:
    - Consumption forecasting
    - Anomaly detection
    - Grid optimization
    - Batch and real-time inference
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_columns = config.feature_columns
        self.is_loaded = False
        
    def load_model(self, model_path: Optional[str] = None):
        """Load the trained model"""
        model_path = model_path or self.config.model_path
        
        try:
            # Load model from MLflow or local path
            if model_path.startswith("mlflow://"):
                model_uri = model_path.replace("mlflow://", "")
                self.model = mlflow.tensorflow.load_model(model_uri)
            else:
                self.model = keras.models.load_model(model_path)
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise
    
    def load_scaler(self, scaler_path: str):
        """Load the feature scaler"""
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded successfully from {scaler_path}")
        except Exception as e:
            logger.error(f"Failed to load scaler from {scaler_path}: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess data for inference"""
        # Ensure we have the required feature columns
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            # Fill missing columns with zeros
            for col in missing_cols:
                df[col] = 0
        
        # Select and order features
        X = df[self.feature_columns].values
        
        # Scale features if scaler is available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences for time series models"""
        sequences = []
        
        for i in range(self.config.sequence_length, len(X) + 1):
            seq = X[i-self.config.sequence_length:i]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def predict_consumption(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict energy consumption"""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Starting consumption prediction")
        
        # Preprocess data
        X = self.preprocess_data(df)
        
        # Create sequences
        X_seq = self.create_sequences(X)
        
        if len(X_seq) == 0:
            logger.warning("No sequences created. Need at least sequence_length data points.")
            return {
                "predictions": [],
                "confidence": [],
                "timestamp": datetime.now().isoformat(),
                "status": "insufficient_data"
            }
        
        # Make predictions
        predictions = self.model.predict(X_seq, batch_size=self.config.batch_size, verbose=0)
        
        # Calculate confidence intervals (simplified)
        pred_mean = np.mean(predictions, axis=0) if predictions.ndim > 1 else predictions
        pred_std = np.std(predictions, axis=0) if predictions.ndim > 1 else np.zeros_like(pred_mean)
        
        # Calculate confidence scores
        confidence = self._calculate_confidence(pred_mean, pred_std)
        
        # Create results
        results = {
            "predictions": pred_mean.tolist(),
            "confidence": confidence.tolist(),
            "prediction_std": pred_std.tolist(),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "model_type": "consumption_forecasting"
        }
        
        logger.info(f"Consumption prediction completed. Predicted {len(pred_mean)} values")
        return results
    
    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in smart meter data"""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Starting anomaly detection")
        
        # Preprocess data
        X = self.preprocess_data(df)
        
        # Create sequences
        X_seq = self.create_sequences(X)
        
        if len(X_seq) == 0:
            logger.warning("No sequences created. Need at least sequence_length data points.")
            return {
                "anomalies": [],
                "anomaly_scores": [],
                "timestamp": datetime.now().isoformat(),
                "status": "insufficient_data"
            }
        
        # Reconstruct data using autoencoder
        reconstructed = self.model.predict(X_seq, batch_size=self.config.batch_size, verbose=0)
        
        # Calculate reconstruction error
        reconstruction_error = np.mean(np.square(X_seq - reconstructed), axis=1)
        
        # Detect anomalies
        anomalies = reconstruction_error > self.config.anomaly_threshold
        anomaly_scores = reconstruction_error.tolist()
        
        # Create results
        results = {
            "anomalies": anomalies.tolist(),
            "anomaly_scores": anomaly_scores,
            "threshold": self.config.anomaly_threshold,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "model_type": "anomaly_detection"
        }
        
        logger.info(f"Anomaly detection completed. Found {np.sum(anomalies)} anomalies")
        return results
    
    def predict_grid_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict grid optimization recommendations"""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Starting grid optimization prediction")
        
        # Preprocess data
        X = self.preprocess_data(df)
        
        # Create sequences
        X_seq = self.create_sequences(X)
        
        if len(X_seq) == 0:
            logger.warning("No sequences created. Need at least sequence_length data points.")
            return {
                "recommendations": [],
                "optimization_score": 0.0,
                "timestamp": datetime.now().isoformat(),
                "status": "insufficient_data"
            }
        
        # Make predictions
        predictions = self.model.predict(X_seq, batch_size=self.config.batch_size, verbose=0)
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(predictions, df)
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(predictions)
        
        # Create results
        results = {
            "recommendations": recommendations,
            "optimization_score": optimization_score,
            "predictions": predictions.tolist(),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "model_type": "grid_optimization"
        }
        
        logger.info(f"Grid optimization prediction completed. Score: {optimization_score:.4f}")
        return results
    
    def batch_inference(self, df: pd.DataFrame, model_type: str = "consumption_forecasting") -> Dict[str, Any]:
        """Perform batch inference on historical data"""
        logger.info(f"Starting batch inference for {model_type}")
        
        if model_type == "consumption_forecasting":
            return self.predict_consumption(df)
        elif model_type == "anomaly_detection":
            return self.detect_anomalies(df)
        elif model_type == "grid_optimization":
            return self.predict_grid_optimization(df)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    async def real_time_inference(self, data_stream: asyncio.Queue, 
                                 output_stream: asyncio.Queue) -> None:
        """Perform real-time inference on streaming data"""
        logger.info("Starting real-time inference")
        
        while True:
            try:
                # Get data from stream
                data = await data_stream.get()
                
                # Convert to DataFrame
                df = pd.DataFrame([data])
                
                # Perform inference based on model type
                if "consumption" in self.config.model_path:
                    result = self.predict_consumption(df)
                elif "anomaly" in self.config.model_path:
                    result = self.detect_anomalies(df)
                elif "grid" in self.config.model_path:
                    result = self.predict_grid_optimization(df)
                else:
                    result = {"error": "Unknown model type"}
                
                # Send result to output stream
                await output_stream.put(result)
                
            except Exception as e:
                logger.error(f"Error in real-time inference: {str(e)}")
                await output_stream.put({
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "status": "error"
                })
    
    def _calculate_confidence(self, predictions: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for predictions"""
        # Simplified confidence calculation
        # In production, you might use more sophisticated methods
        confidence = np.exp(-std / np.mean(predictions))
        return np.clip(confidence, 0, 1)
    
    def _generate_optimization_recommendations(self, predictions: np.ndarray, 
                                             df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate grid optimization recommendations"""
        recommendations = []
        
        # Analyze predictions and generate recommendations
        if len(predictions) > 0:
            pred_value = predictions[-1] if predictions.ndim == 1 else predictions[-1][0]
            
            # Example recommendations based on predictions
            if pred_value > 1000:  # High consumption
                recommendations.append({
                    "type": "load_balancing",
                    "priority": "high",
                    "message": "High consumption detected. Consider load balancing.",
                    "action": "Activate additional grid resources"
                })
            
            if pred_value < 100:  # Low consumption
                recommendations.append({
                    "type": "efficiency",
                    "priority": "medium",
                    "message": "Low consumption detected. Check for efficiency opportunities.",
                    "action": "Review grid efficiency settings"
                })
            
            # Weather-based recommendations
            if 'temperature_c' in df.columns and len(df) > 0:
                temp = df['temperature_c'].iloc[-1]
                if temp > 30:  # Hot weather
                    recommendations.append({
                        "type": "weather_adaptation",
                        "priority": "high",
                        "message": "Hot weather detected. Expect increased cooling demand.",
                        "action": "Prepare for increased energy consumption"
                    })
        
        return recommendations
    
    def _calculate_optimization_score(self, predictions: np.ndarray) -> float:
        """Calculate grid optimization score"""
        if len(predictions) == 0:
            return 0.0
        
        # Simplified optimization score calculation
        # In production, this would be more sophisticated
        pred_value = predictions[-1] if predictions.ndim == 1 else predictions[-1][0]
        
        # Score based on how close to optimal range (e.g., 500-800)
        optimal_min, optimal_max = 500, 800
        
        if optimal_min <= pred_value <= optimal_max:
            return 1.0  # Perfect score
        else:
            # Calculate distance from optimal range
            if pred_value < optimal_min:
                distance = optimal_min - pred_value
            else:
                distance = pred_value - optimal_max
            
            # Convert distance to score (0-1)
            max_distance = max(optimal_min, 1000 - optimal_max)
            score = max(0, 1 - (distance / max_distance))
            
            return score
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": self.config.model_path,
            "sequence_length": self.config.sequence_length,
            "forecast_horizon": self.config.forecast_horizon,
            "feature_columns": self.feature_columns,
            "model_type": self._detect_model_type()
        }
    
    def _detect_model_type(self) -> str:
        """Detect the type of loaded model"""
        if "consumption" in self.config.model_path:
            return "consumption_forecasting"
        elif "anomaly" in self.config.model_path:
            return "anomaly_detection"
        elif "grid" in self.config.model_path:
            return "grid_optimization"
        else:
            return "unknown"
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the inference pipeline"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "model_loaded": self.is_loaded,
            "scaler_loaded": self.scaler is not None,
            "status": "healthy" if self.is_loaded else "unhealthy"
        }
        
        if self.is_loaded:
            try:
                # Test with dummy data
                dummy_data = np.random.random((1, len(self.feature_columns)))
                dummy_df = pd.DataFrame(dummy_data, columns=self.feature_columns)
                
                # Test prediction
                if "consumption" in self.config.model_path:
                    result = self.predict_consumption(dummy_df)
                elif "anomaly" in self.config.model_path:
                    result = self.detect_anomalies(dummy_df)
                elif "grid" in self.config.model_path:
                    result = self.predict_grid_optimization(dummy_df)
                
                health_status["prediction_test"] = "passed"
                
            except Exception as e:
                health_status["prediction_test"] = "failed"
                health_status["error"] = str(e)
                health_status["status"] = "unhealthy"
        
        return health_status
