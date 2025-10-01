"""
Drift Detection Module
Comprehensive drift detection for data and model drift
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import json

logger = logging.getLogger(__name__)

@dataclass
class DriftConfig:
    """Configuration for drift detection"""
    # Statistical tests
    ks_threshold: float = 0.05
    psi_threshold: float = 0.2
    wasserstein_threshold: float = 0.1
    
    # Model drift
    performance_threshold: float = 0.05
    prediction_drift_threshold: float = 0.1
    
    # Data drift
    feature_drift_threshold: float = 0.1
    multivariate_drift_threshold: float = 0.1
    
    # Alerting
    alert_on_drift: bool = True
    alert_cooldown_hours: int = 24

@dataclass
class DriftResult:
    """Result of drift detection"""
    drift_type: str  # 'data', 'model', 'prediction'
    feature_name: Optional[str]
    drift_detected: bool
    drift_score: float
    confidence: float
    p_value: Optional[float]
    message: str
    timestamp: datetime
    details: Dict[str, Any]

class DriftDetector:
    """
    Comprehensive drift detection system
    """
    
    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self.reference_data = None
        self.reference_model = None
        self.drift_history = []
        self.alert_history = []
        
        logger.info("DriftDetector initialized")
    
    def set_reference_data(self, data: pd.DataFrame, data_type: str = "training"):
        """Set reference data for drift detection"""
        self.reference_data = {
            "data": data.copy(),
            "data_type": data_type,
            "timestamp": datetime.now(),
            "statistics": self._calculate_data_statistics(data)
        }
        logger.info(f"Reference data set: {data.shape[0]} samples, {data.shape[1]} features")
    
    def set_reference_model(self, model, model_type: str = "tensorflow"):
        """Set reference model for drift detection"""
        self.reference_model = {
            "model": model,
            "model_type": model_type,
            "timestamp": datetime.now()
        }
        logger.info(f"Reference model set: {model_type}")
    
    def detect_data_drift(self, 
                         current_data: pd.DataFrame,
                         features: Optional[List[str]] = None) -> List[DriftResult]:
        """Detect data drift between reference and current data"""
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        reference_data = self.reference_data["data"]
        results = []
        
        if features is None:
            features = current_data.columns.tolist()
        
        for feature in features:
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue
            
            try:
                result = self._detect_univariate_drift(
                    feature, 
                    reference_data[feature], 
                    current_data[feature]
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Drift detection failed for feature {feature}: {str(e)}")
                results.append(DriftResult(
                    drift_type="data",
                    feature_name=feature,
                    drift_detected=False,
                    drift_score=0.0,
                    confidence=0.0,
                    p_value=None,
                    message=f"Error: {str(e)}",
                    timestamp=datetime.now(),
                    details={}
                ))
        
        # Detect multivariate drift
        try:
            multivariate_result = self._detect_multivariate_drift(reference_data, current_data)
            results.append(multivariate_result)
        except Exception as e:
            logger.error(f"Multivariate drift detection failed: {str(e)}")
        
        # Store results
        self.drift_history.extend(results)
        
        return results
    
    def detect_model_drift(self, 
                          current_predictions: np.ndarray,
                          current_actuals: np.ndarray) -> List[DriftResult]:
        """Detect model drift based on performance degradation"""
        if self.reference_model is None:
            raise ValueError("Reference model not set")
        
        results = []
        
        try:
            # Performance drift
            performance_result = self._detect_performance_drift(
                current_predictions, 
                current_actuals
            )
            results.append(performance_result)
            
            # Prediction drift
            prediction_result = self._detect_prediction_drift(current_predictions)
            results.append(prediction_result)
            
        except Exception as e:
            logger.error(f"Model drift detection failed: {str(e)}")
            results.append(DriftResult(
                drift_type="model",
                feature_name=None,
                drift_detected=False,
                drift_score=0.0,
                confidence=0.0,
                p_value=None,
                message=f"Error: {str(e)}",
                timestamp=datetime.now(),
                details={}
            ))
        
        # Store results
        self.drift_history.extend(results)
        
        return results
    
    def _detect_univariate_drift(self, 
                                feature_name: str,
                                reference: pd.Series,
                                current: pd.Series) -> DriftResult:
        """Detect univariate drift for a single feature"""
        # Remove null values
        ref_clean = reference.dropna()
        curr_clean = current.dropna()
        
        if len(ref_clean) < 10 or len(curr_clean) < 10:
            return DriftResult(
                drift_type="data",
                feature_name=feature_name,
                drift_detected=False,
                drift_score=0.0,
                confidence=0.0,
                p_value=None,
                message="Insufficient data for drift detection",
                timestamp=datetime.now(),
                details={}
            )
        
        # Statistical tests
        drift_scores = {}
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(ref_clean, curr_clean)
        drift_scores["ks"] = {
            "statistic": ks_stat,
            "p_value": ks_pvalue,
            "drift_detected": ks_pvalue < self.config.ks_threshold
        }
        
        # Population Stability Index
        psi_score = self._calculate_psi(ref_clean, curr_clean)
        drift_scores["psi"] = {
            "score": psi_score,
            "drift_detected": psi_score > self.config.psi_threshold
        }
        
        # Wasserstein distance
        wasserstein_dist = stats.wasserstein_distance(ref_clean, curr_clean)
        drift_scores["wasserstein"] = {
            "distance": wasserstein_dist,
            "drift_detected": wasserstein_dist > self.config.wasserstein_threshold
        }
        
        # Jensen-Shannon divergence
        js_div = self._calculate_js_divergence(ref_clean, curr_clean)
        drift_scores["js_divergence"] = {
            "divergence": js_div,
            "drift_detected": js_div > 0.1
        }
        
        # Overall drift detection
        drift_detected = any([
            drift_scores["ks"]["drift_detected"],
            drift_scores["psi"]["drift_detected"],
            drift_scores["wasserstein"]["drift_detected"],
            drift_scores["js_divergence"]["drift_detected"]
        ])
        
        # Calculate overall drift score
        drift_score = max(
            drift_scores["ks"]["statistic"],
            drift_scores["psi"]["score"],
            drift_scores["wasserstein"]["distance"],
            drift_scores["js_divergence"]["divergence"]
        )
        
        # Calculate confidence
        confidence = 1 - min(ks_pvalue, 0.1)  # Higher confidence for lower p-values
        
        message = f"Drift detected: {drift_detected}, Score: {drift_score:.3f}"
        
        return DriftResult(
            drift_type="data",
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=drift_score,
            confidence=confidence,
            p_value=ks_pvalue,
            message=message,
            timestamp=datetime.now(),
            details=drift_scores
        )
    
    def _detect_multivariate_drift(self, 
                                  reference: pd.DataFrame,
                                  current: pd.DataFrame) -> DriftResult:
        """Detect multivariate drift using PCA and statistical tests"""
        try:
            # Align columns
            common_cols = list(set(reference.columns) & set(current.columns))
            ref_aligned = reference[common_cols].select_dtypes(include=[np.number])
            curr_aligned = current[common_cols].select_dtypes(include=[np.number])
            
            if ref_aligned.empty or curr_aligned.empty:
                return DriftResult(
                    drift_type="data",
                    feature_name="multivariate",
                    drift_detected=False,
                    drift_score=0.0,
                    confidence=0.0,
                    p_value=None,
                    message="No numeric features for multivariate drift detection",
                    timestamp=datetime.now(),
                    details={}
                )
            
            # Remove null values
            ref_clean = ref_aligned.dropna()
            curr_clean = curr_aligned.dropna()
            
            if len(ref_clean) < 10 or len(curr_clean) < 10:
                return DriftResult(
                    drift_type="data",
                    feature_name="multivariate",
                    drift_detected=False,
                    drift_score=0.0,
                    confidence=0.0,
                    p_value=None,
                    message="Insufficient data for multivariate drift detection",
                    timestamp=datetime.now(),
                    details={}
                )
            
            # Standardize data
            scaler = StandardScaler()
            ref_scaled = scaler.fit_transform(ref_clean)
            curr_scaled = scaler.transform(curr_clean)
            
            # Apply PCA
            pca = PCA(n_components=min(10, ref_scaled.shape[1]))
            ref_pca = pca.fit_transform(ref_scaled)
            curr_pca = pca.transform(curr_scaled)
            
            # Test each principal component
            drift_scores = []
            for i in range(ref_pca.shape[1]):
                ref_pc = ref_pca[:, i]
                curr_pc = curr_pca[:, i]
                
                ks_stat, ks_pvalue = stats.ks_2samp(ref_pc, curr_pc)
                drift_scores.append(ks_stat)
            
            # Overall multivariate drift
            max_drift_score = max(drift_scores)
            avg_drift_score = np.mean(drift_scores)
            
            drift_detected = max_drift_score > self.config.multivariate_drift_threshold
            confidence = 1 - min(avg_drift_score, 0.1)
            
            message = f"Multivariate drift detected: {drift_detected}, Max score: {max_drift_score:.3f}"
            
            return DriftResult(
                drift_type="data",
                feature_name="multivariate",
                drift_detected=drift_detected,
                drift_score=max_drift_score,
                confidence=confidence,
                p_value=None,
                message=message,
                timestamp=datetime.now(),
                details={
                    "pca_components": ref_pca.shape[1],
                    "drift_scores_per_component": drift_scores,
                    "average_drift_score": avg_drift_score
                }
            )
            
        except Exception as e:
            logger.error(f"Multivariate drift detection failed: {str(e)}")
            return DriftResult(
                drift_type="data",
                feature_name="multivariate",
                drift_detected=False,
                drift_score=0.0,
                confidence=0.0,
                p_value=None,
                message=f"Error: {str(e)}",
                timestamp=datetime.now(),
                details={}
            )
    
    def _detect_performance_drift(self, 
                                 predictions: np.ndarray,
                                 actuals: np.ndarray) -> DriftResult:
        """Detect performance drift"""
        try:
            # Calculate current performance metrics
            mae = np.mean(np.abs(predictions - actuals))
            mse = np.mean((predictions - actuals) ** 2)
            rmse = np.sqrt(mse)
            
            # For simplicity, assume reference performance (in practice, this would be stored)
            reference_mae = 10.0  # Example reference MAE
            reference_rmse = 15.0  # Example reference RMSE
            
            # Calculate performance degradation
            mae_degradation = (mae - reference_mae) / reference_mae
            rmse_degradation = (rmse - reference_rmse) / reference_rmse
            
            # Overall performance drift
            performance_drift = max(abs(mae_degradation), abs(rmse_degradation))
            drift_detected = performance_drift > self.config.performance_threshold
            
            confidence = min(performance_drift, 1.0)
            
            message = f"Performance drift detected: {drift_detected}, Degradation: {performance_drift:.3f}"
            
            return DriftResult(
                drift_type="model",
                feature_name="performance",
                drift_detected=drift_detected,
                drift_score=performance_drift,
                confidence=confidence,
                p_value=None,
                message=message,
                timestamp=datetime.now(),
                details={
                    "current_mae": mae,
                    "current_rmse": rmse,
                    "mae_degradation": mae_degradation,
                    "rmse_degradation": rmse_degradation,
                    "reference_mae": reference_mae,
                    "reference_rmse": reference_rmse
                }
            )
            
        except Exception as e:
            logger.error(f"Performance drift detection failed: {str(e)}")
            return DriftResult(
                drift_type="model",
                feature_name="performance",
                drift_detected=False,
                drift_score=0.0,
                confidence=0.0,
                p_value=None,
                message=f"Error: {str(e)}",
                timestamp=datetime.now(),
                details={}
            )
    
    def _detect_prediction_drift(self, predictions: np.ndarray) -> DriftResult:
        """Detect prediction drift"""
        try:
            # For simplicity, assume reference predictions (in practice, this would be stored)
            reference_predictions = np.random.normal(50, 10, len(predictions))  # Example reference
            
            # Statistical test for prediction drift
            ks_stat, ks_pvalue = stats.ks_2samp(reference_predictions, predictions)
            
            # Calculate drift score
            drift_score = ks_stat
            drift_detected = ks_pvalue < 0.05
            
            confidence = 1 - min(ks_pvalue, 0.1)
            
            message = f"Prediction drift detected: {drift_detected}, Score: {drift_score:.3f}"
            
            return DriftResult(
                drift_type="model",
                feature_name="predictions",
                drift_detected=drift_detected,
                drift_score=drift_score,
                confidence=confidence,
                p_value=ks_pvalue,
                message=message,
                timestamp=datetime.now(),
                details={
                    "ks_statistic": ks_stat,
                    "p_value": ks_pvalue,
                    "reference_mean": np.mean(reference_predictions),
                    "current_mean": np.mean(predictions)
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction drift detection failed: {str(e)}")
            return DriftResult(
                drift_type="model",
                feature_name="predictions",
                drift_detected=False,
                drift_score=0.0,
                confidence=0.0,
                p_value=None,
                message=f"Error: {str(e)}",
                timestamp=datetime.now(),
                details={}
            )
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins based on reference data
            bins = np.percentile(reference, [0, 20, 40, 60, 80, 100])
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bins)
            curr_counts, _ = np.histogram(current, bins=bins)
            
            # Normalize to probabilities
            ref_probs = ref_counts / len(reference)
            curr_probs = curr_counts / len(current)
            
            # Calculate PSI
            psi = 0
            for i in range(len(ref_probs)):
                if ref_probs[i] > 0 and curr_probs[i] > 0:
                    psi += (curr_probs[i] - ref_probs[i]) * np.log(curr_probs[i] / ref_probs[i])
            
            return psi
            
        except Exception as e:
            logger.error(f"PSI calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_js_divergence(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate Jensen-Shannon divergence"""
        try:
            # Create bins
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            bins = np.linspace(min_val, max_val, 50)
            
            # Calculate histograms
            ref_hist, _ = np.histogram(reference, bins=bins, density=True)
            curr_hist, _ = np.histogram(current, bins=bins, density=True)
            
            # Normalize
            ref_hist = ref_hist / ref_hist.sum()
            curr_hist = curr_hist / curr_hist.sum()
            
            # Calculate JS divergence
            m = 0.5 * (ref_hist + curr_hist)
            js_div = 0.5 * stats.entropy(ref_hist, m) + 0.5 * stats.entropy(curr_hist, m)
            
            return js_div
            
        except Exception as e:
            logger.error(f"JS divergence calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data statistics for reference"""
        stats = {}
        
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                stats[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'skew': data[col].skew(),
                    'kurtosis': data[col].kurtosis()
                }
        
        return stats
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection results"""
        if not self.drift_history:
            return {"message": "No drift detection results available"}
        
        # Group by drift type
        data_drift = [r for r in self.drift_history if r.drift_type == "data"]
        model_drift = [r for r in self.drift_history if r.drift_type == "model"]
        
        summary = {
            "total_drift_checks": len(self.drift_history),
            "data_drift_checks": len(data_drift),
            "model_drift_checks": len(model_drift),
            "drift_detected_count": sum(1 for r in self.drift_history if r.drift_detected),
            "high_confidence_drift": sum(1 for r in self.drift_history if r.drift_detected and r.confidence > 0.8),
            "average_drift_score": np.mean([r.drift_score for r in self.drift_history]),
            "latest_drift_check": max(self.drift_history, key=lambda x: x.timestamp).timestamp.isoformat()
        }
        
        return summary
    
    def export_drift_report(self, filepath: str):
        """Export drift detection report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_drift_summary(),
            "drift_results": []
        }
        
        for result in self.drift_history:
            report["drift_results"].append({
                "drift_type": result.drift_type,
                "feature_name": result.feature_name,
                "drift_detected": result.drift_detected,
                "drift_score": result.drift_score,
                "confidence": result.confidence,
                "p_value": result.p_value,
                "message": result.message,
                "timestamp": result.timestamp.isoformat(),
                "details": result.details
            })
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Drift report exported to {filepath}")
    
    def clear_history(self):
        """Clear drift detection history"""
        self.drift_history = []
        self.alert_history = []
        logger.info("Drift detection history cleared")
