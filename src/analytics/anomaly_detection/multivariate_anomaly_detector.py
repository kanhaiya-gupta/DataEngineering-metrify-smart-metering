"""
Multivariate Anomaly Detector
Advanced multivariate anomaly detection for smart meter data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.covariance import EmpiricalCovariance
from scipy import stats
import joblib

logger = logging.getLogger(__name__)

@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection"""
    # Model parameters
    contamination: float = 0.1  # Expected proportion of anomalies
    random_state: int = 42
    
    # Isolation Forest
    isolation_forest_n_estimators: int = 100
    isolation_forest_max_samples: Union[str, int] = 'auto'
    
    # One-Class SVM
    svm_kernel: str = 'rbf'
    svm_gamma: str = 'scale'
    svm_nu: float = 0.1
    
    # DBSCAN
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    
    # PCA
    pca_n_components: Optional[int] = None
    pca_variance_threshold: float = 0.95
    
    # Thresholds
    anomaly_score_threshold: float = 0.5
    confidence_threshold: float = 0.8

class MultivariateAnomalyDetector:
    """
    Advanced multivariate anomaly detection system
    """
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.anomaly_history = []
        
        logger.info("MultivariateAnomalyDetector initialized")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection"""
        df = data.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features for time series
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend']:
                for lag in [1, 2, 3, 6, 12, 24]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
                # Rolling statistics
                for window in [3, 6, 12, 24]:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                
                # Difference features
                df[f'{col}_diff_1'] = df[col].diff(1)
                df[f'{col}_diff_24'] = df[col].diff(24)
                df[f'{col}_pct_change'] = df[col].pct_change()
        
        # Statistical features
        for col in numeric_cols:
            if col not in ['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend']:
                # Z-score
                df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
                
                # Percentile rank
                df[f'{col}_percentile'] = df[col].rank(pct=True)
                
                # Outlier indicators
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                df[f'{col}_outlier'] = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).astype(int)
        
        return df
    
    def train_isolation_forest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train Isolation Forest model"""
        try:
            # Prepare features
            df_features = self.prepare_features(data)
            df_features = df_features.dropna()
            
            # Select numeric features
            X = df_features.select_dtypes(include=[np.number])
            
            if X.empty:
                return {"error": "No numeric features available"}
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Isolation Forest
            model = IsolationForest(
                contamination=self.config.contamination,
                n_estimators=self.config.isolation_forest_n_estimators,
                max_samples=self.config.isolation_forest_max_samples,
                random_state=self.config.random_state
            )
            
            model.fit(X_scaled)
            
            # Evaluate model
            anomaly_scores = model.decision_function(X_scaled)
            predictions = model.predict(X_scaled)
            
            # Convert predictions (-1 for anomaly, 1 for normal)
            anomaly_labels = (predictions == -1).astype(int)
            
            # Calculate metrics
            anomaly_count = anomaly_labels.sum()
            total_count = len(anomaly_labels)
            anomaly_rate = anomaly_count / total_count
            
            # Store model and scaler
            self.models['isolation_forest'] = model
            self.scalers['isolation_forest'] = scaler
            
            logger.info(f"Isolation Forest trained - Anomaly rate: {anomaly_rate:.3f}")
            
            return {
                'model_type': 'isolation_forest',
                'anomaly_count': anomaly_count,
                'total_count': total_count,
                'anomaly_rate': anomaly_rate,
                'feature_count': X.shape[1],
                'training_samples': len(X_scaled)
            }
            
        except Exception as e:
            logger.error(f"Isolation Forest training failed: {str(e)}")
            raise
    
    def train_one_class_svm(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train One-Class SVM model"""
        try:
            # Prepare features
            df_features = self.prepare_features(data)
            df_features = df_features.dropna()
            
            # Select numeric features
            X = df_features.select_dtypes(include=[np.number])
            
            if X.empty:
                return {"error": "No numeric features available"}
            
            # Scale features
            scaler = RobustScaler()  # More robust to outliers
            X_scaled = scaler.fit_transform(X)
            
            # Train One-Class SVM
            model = OneClassSVM(
                kernel=self.config.svm_kernel,
                gamma=self.config.svm_gamma,
                nu=self.config.svm_nu
            )
            
            model.fit(X_scaled)
            
            # Evaluate model
            anomaly_scores = model.decision_function(X_scaled)
            predictions = model.predict(X_scaled)
            
            # Convert predictions (-1 for anomaly, 1 for normal)
            anomaly_labels = (predictions == -1).astype(int)
            
            # Calculate metrics
            anomaly_count = anomaly_labels.sum()
            total_count = len(anomaly_labels)
            anomaly_rate = anomaly_count / total_count
            
            # Store model and scaler
            self.models['one_class_svm'] = model
            self.scalers['one_class_svm'] = scaler
            
            logger.info(f"One-Class SVM trained - Anomaly rate: {anomaly_rate:.3f}")
            
            return {
                'model_type': 'one_class_svm',
                'anomaly_count': anomaly_count,
                'total_count': total_count,
                'anomaly_rate': anomaly_rate,
                'feature_count': X.shape[1],
                'training_samples': len(X_scaled)
            }
            
        except Exception as e:
            logger.error(f"One-Class SVM training failed: {str(e)}")
            raise
    
    def train_dbscan(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train DBSCAN clustering model"""
        try:
            # Prepare features
            df_features = self.prepare_features(data)
            df_features = df_features.dropna()
            
            # Select numeric features
            X = df_features.select_dtypes(include=[np.number])
            
            if X.empty:
                return {"error": "No numeric features available"}
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=self.config.pca_n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Train DBSCAN
            model = DBSCAN(
                eps=self.config.dbscan_eps,
                min_samples=self.config.dbscan_min_samples
            )
            
            cluster_labels = model.fit_predict(X_pca)
            
            # Identify anomalies (cluster label -1)
            anomaly_labels = (cluster_labels == -1).astype(int)
            
            # Calculate metrics
            anomaly_count = anomaly_labels.sum()
            total_count = len(anomaly_labels)
            anomaly_rate = anomaly_count / total_count
            
            # Store models and scalers
            self.models['dbscan'] = model
            self.models['pca'] = pca
            self.scalers['dbscan'] = scaler
            
            logger.info(f"DBSCAN trained - Anomaly rate: {anomaly_rate:.3f}")
            
            return {
                'model_type': 'dbscan',
                'anomaly_count': anomaly_count,
                'total_count': total_count,
                'anomaly_rate': anomaly_rate,
                'feature_count': X.shape[1],
                'pca_components': X_pca.shape[1],
                'training_samples': len(X_scaled)
            }
            
        except Exception as e:
            logger.error(f"DBSCAN training failed: {str(e)}")
            raise
    
    def train_ensemble_detector(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train ensemble anomaly detector"""
        try:
            # Train individual models
            if_results = self.train_isolation_forest(data)
            svm_results = self.train_one_class_svm(data)
            dbscan_results = self.train_dbscan(data)
            
            # Create ensemble
            ensemble_weights = {
                'isolation_forest': 0.4,
                'one_class_svm': 0.3,
                'dbscan': 0.3
            }
            
            self.models['ensemble'] = {
                'models': ['isolation_forest', 'one_class_svm', 'dbscan'],
                'weights': ensemble_weights
            }
            
            # Calculate ensemble metrics
            ensemble_anomaly_rate = np.average([
                if_results.get('anomaly_rate', 0),
                svm_results.get('anomaly_rate', 0),
                dbscan_results.get('anomaly_rate', 0)
            ], weights=list(ensemble_weights.values()))
            
            logger.info(f"Ensemble detector trained - Anomaly rate: {ensemble_anomaly_rate:.3f}")
            
            return {
                'model_type': 'ensemble',
                'anomaly_rate': ensemble_anomaly_rate,
                'component_results': {
                    'isolation_forest': if_results,
                    'one_class_svm': svm_results,
                    'dbscan': dbscan_results
                }
            }
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {str(e)}")
            raise
    
    def detect_anomalies(self, data: pd.DataFrame, model_name: str = 'ensemble') -> Dict[str, Any]:
        """Detect anomalies using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        try:
            # Prepare features
            df_features = self.prepare_features(data)
            df_features = df_features.dropna()
            
            # Select numeric features
            X = df_features.select_dtypes(include=[np.number])
            
            if X.empty:
                return {"error": "No numeric features available"}
            
            if model_name == 'ensemble':
                return self._detect_anomalies_ensemble(X, df_features)
            else:
                return self._detect_anomalies_single(X, df_features, model_name)
                
        except Exception as e:
            logger.error(f"Anomaly detection failed for {model_name}: {str(e)}")
            raise
    
    def _detect_anomalies_single(self, X: pd.DataFrame, df_features: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Detect anomalies using single model"""
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        if model_name == 'dbscan':
            # Apply PCA
            pca = self.models['pca']
            X_pca = pca.transform(X_scaled)
            
            # Predict
            cluster_labels = model.fit_predict(X_pca)
            anomaly_labels = (cluster_labels == -1).astype(int)
            anomaly_scores = np.abs(cluster_labels)  # Use cluster distance as score
        else:
            # Predict
            anomaly_scores = model.decision_function(X_scaled)
            predictions = model.predict(X_scaled)
            anomaly_labels = (predictions == -1).astype(int)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(anomaly_scores, anomaly_labels)
        
        # Filter by thresholds
        high_confidence_anomalies = (anomaly_labels == 1) & (confidence_scores >= self.config.confidence_threshold)
        
        # Store results
        result = {
            'model_name': model_name,
            'anomaly_labels': anomaly_labels.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'confidence_scores': confidence_scores.tolist(),
            'high_confidence_anomalies': high_confidence_anomalies.tolist(),
            'anomaly_count': anomaly_labels.sum(),
            'high_confidence_count': high_confidence_anomalies.sum(),
            'total_count': len(anomaly_labels),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add anomaly details
        if anomaly_labels.sum() > 0:
            anomaly_indices = np.where(anomaly_labels == 1)[0]
            anomaly_details = []
            
            for idx in anomaly_indices:
                anomaly_details.append({
                    'index': int(idx),
                    'timestamp': df_features.index[idx].isoformat() if hasattr(df_features.index, 'isoformat') else str(df_features.index[idx]),
                    'anomaly_score': float(anomaly_scores[idx]),
                    'confidence_score': float(confidence_scores[idx]),
                    'features': X.iloc[idx].to_dict()
                })
            
            result['anomaly_details'] = anomaly_details
        
        # Store in history
        self.anomaly_history.append(result)
        
        return result
    
    def _detect_anomalies_ensemble(self, X: pd.DataFrame, df_features: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using ensemble model"""
        ensemble_info = self.models['ensemble']
        models = ensemble_info['models']
        weights = ensemble_info['weights']
        
        # Get predictions from individual models
        individual_results = {}
        for model_name in models:
            if model_name in self.models:
                individual_results[model_name] = self._detect_anomalies_single(X, df_features, model_name)
        
        # Combine results
        if not individual_results:
            return {"error": "No individual models available for ensemble"}
        
        # Weighted voting
        ensemble_anomaly_labels = np.zeros(len(X))
        ensemble_anomaly_scores = np.zeros(len(X))
        ensemble_confidence_scores = np.zeros(len(X))
        
        for model_name, result in individual_results.items():
            if 'anomaly_labels' in result and 'anomaly_scores' in result:
                weight = weights[model_name]
                ensemble_anomaly_labels += weight * np.array(result['anomaly_labels'])
                ensemble_anomaly_scores += weight * np.array(result['anomaly_scores'])
                
                if 'confidence_scores' in result:
                    ensemble_confidence_scores += weight * np.array(result['confidence_scores'])
        
        # Convert to binary labels
        ensemble_anomaly_labels = (ensemble_anomaly_labels >= 0.5).astype(int)
        
        # Filter by thresholds
        high_confidence_anomalies = (ensemble_anomaly_labels == 1) & (ensemble_confidence_scores >= self.config.confidence_threshold)
        
        result = {
            'model_name': 'ensemble',
            'anomaly_labels': ensemble_anomaly_labels.tolist(),
            'anomaly_scores': ensemble_anomaly_scores.tolist(),
            'confidence_scores': ensemble_confidence_scores.tolist(),
            'high_confidence_anomalies': high_confidence_anomalies.tolist(),
            'anomaly_count': ensemble_anomaly_labels.sum(),
            'high_confidence_count': high_confidence_anomalies.sum(),
            'total_count': len(ensemble_anomaly_labels),
            'individual_results': individual_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add anomaly details
        if ensemble_anomaly_labels.sum() > 0:
            anomaly_indices = np.where(ensemble_anomaly_labels == 1)[0]
            anomaly_details = []
            
            for idx in anomaly_indices:
                anomaly_details.append({
                    'index': int(idx),
                    'timestamp': df_features.index[idx].isoformat() if hasattr(df_features.index, 'isoformat') else str(df_features.index[idx]),
                    'anomaly_score': float(ensemble_anomaly_scores[idx]),
                    'confidence_score': float(ensemble_confidence_scores[idx]),
                    'features': X.iloc[idx].to_dict()
                })
            
            result['anomaly_details'] = anomaly_details
        
        # Store in history
        self.anomaly_history.append(result)
        
        return result
    
    def _calculate_confidence_scores(self, anomaly_scores: np.ndarray, anomaly_labels: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for anomaly predictions"""
        # Normalize scores to 0-1 range
        if len(anomaly_scores) > 0:
            min_score = np.min(anomaly_scores)
            max_score = np.max(anomaly_scores)
            
            if max_score > min_score:
                normalized_scores = (anomaly_scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.ones_like(anomaly_scores)
        else:
            normalized_scores = np.array([])
        
        # Higher confidence for more extreme scores
        confidence_scores = np.abs(normalized_scores - 0.5) * 2
        
        return confidence_scores
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of anomaly detection results"""
        if not self.anomaly_history:
            return {"message": "No anomaly detection results available"}
        
        # Aggregate statistics
        total_detections = sum(result.get('anomaly_count', 0) for result in self.anomaly_history)
        total_high_confidence = sum(result.get('high_confidence_count', 0) for result in self.anomaly_history)
        total_samples = sum(result.get('total_count', 0) for result in self.anomaly_history)
        
        # Model performance
        model_performance = {}
        for result in self.anomaly_history:
            model_name = result.get('model_name', 'unknown')
            if model_name not in model_performance:
                model_performance[model_name] = {
                    'detection_count': 0,
                    'total_samples': 0,
                    'detection_rate': 0.0
                }
            
            model_performance[model_name]['detection_count'] += result.get('anomaly_count', 0)
            model_performance[model_name]['total_samples'] += result.get('total_count', 0)
        
        # Calculate detection rates
        for model_name, perf in model_performance.items():
            if perf['total_samples'] > 0:
                perf['detection_rate'] = perf['detection_count'] / perf['total_samples']
        
        return {
            'total_detections': total_detections,
            'total_high_confidence_detections': total_high_confidence,
            'total_samples_processed': total_samples,
            'overall_detection_rate': total_detections / total_samples if total_samples > 0 else 0,
            'high_confidence_rate': total_high_confidence / total_detections if total_detections > 0 else 0,
            'model_performance': model_performance,
            'latest_detection': self.anomaly_history[-1]['timestamp'] if self.anomaly_history else None
        }
    
    def export_anomaly_report(self, filepath: str):
        """Export anomaly detection report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_anomaly_summary(),
            'anomaly_history': self.anomaly_history
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Anomaly detection report exported to {filepath}")
    
    def save_models(self, filepath: str):
        """Save trained models"""
        model_data = {
            'models': {name: model for name, model in self.models.items() if name != 'model'},
            'scalers': self.scalers,
            'config': self.config,
            'anomaly_history': self.anomaly_history
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Anomaly detection models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.config = model_data['config']
        self.anomaly_history = model_data.get('anomaly_history', [])
        
        logger.info(f"Anomaly detection models loaded from {filepath}")
