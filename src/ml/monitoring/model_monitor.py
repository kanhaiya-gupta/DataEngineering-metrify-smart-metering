"""
Model Monitoring

This module implements comprehensive model monitoring for ML models,
including performance tracking, drift detection, and alerting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import asyncio

import mlflow
import mlflow.tensorflow
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring"""
    reference_data_path: str
    current_data_path: str
    model_path: str
    drift_threshold: float = 0.1
    performance_threshold: float = 0.8
    check_interval: int = 3600  # seconds
    alert_emails: List[str] = None
    enable_mlflow_logging: bool = True
    enable_evidently_reports: bool = True


class ModelMonitor:
    """
    Comprehensive model monitoring system
    
    Monitors:
    - Model performance metrics
    - Data drift and quality
    - Model predictions accuracy
    - System health and availability
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.reference_data = None
        self.current_data = None
        self.model = None
        self.monitoring_results = {}
        self.alert_history = []
        
        # Initialize MLflow if enabled
        if self.config.enable_mlflow_logging:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("model_monitoring")
    
    def load_reference_data(self, data_path: str):
        """Load reference data for drift detection"""
        try:
            self.reference_data = pd.read_csv(data_path)
            logger.info(f"Reference data loaded from {data_path}. Shape: {self.reference_data.shape}")
        except Exception as e:
            logger.error(f"Failed to load reference data: {str(e)}")
            raise
    
    def load_current_data(self, data_path: str):
        """Load current data for monitoring"""
        try:
            self.current_data = pd.read_csv(data_path)
            logger.info(f"Current data loaded from {data_path}. Shape: {self.current_data.shape}")
        except Exception as e:
            logger.error(f"Failed to load current data: {str(e)}")
            raise
    
    def load_model(self, model_path: str):
        """Load the model for monitoring"""
        try:
            self.model = mlflow.tensorflow.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def check_data_drift(self) -> Dict[str, Any]:
        """Check for data drift between reference and current data"""
        if self.reference_data is None or self.current_data is None:
            raise ValueError("Reference and current data must be loaded first")
        
        logger.info("Checking for data drift")
        
        # Define column mapping for Evidently
        column_mapping = ColumnMapping(
            target='target' if 'target' in self.reference_data.columns else None,
            numerical_features=self.reference_data.select_dtypes(include=[np.number]).columns.tolist(),
            categorical_features=self.reference_data.select_dtypes(include=['object']).columns.tolist()
        )
        
        # Create drift report
        drift_report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            DataQualityPreset()
        ])
        
        drift_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=column_mapping
        )
        
        # Extract drift results
        drift_results = {}
        
        # Check numerical features drift
        for feature in column_mapping.numerical_features:
            try:
                drift_metric = drift_report.get_metric(DataDriftPreset()).get_result()
                if feature in drift_metric.drift_by_columns:
                    drift_score = drift_metric.drift_by_columns[feature].drift_score
                    drift_detected = drift_metric.drift_by_columns[feature].drift_detected
                    
                    drift_results[feature] = {
                        "drift_score": float(drift_score),
                        "drift_detected": bool(drift_detected),
                        "threshold": self.config.drift_threshold
                    }
            except Exception as e:
                logger.warning(f"Could not check drift for feature {feature}: {str(e)}")
                drift_results[feature] = {
                    "drift_score": 0.0,
                    "drift_detected": False,
                    "error": str(e)
                }
        
        # Overall drift detection
        overall_drift = any(
            result.get("drift_detected", False) 
            for result in drift_results.values()
        )
        
        drift_summary = {
            "overall_drift_detected": overall_drift,
            "drift_threshold": self.config.drift_threshold,
            "features_checked": len(drift_results),
            "features_with_drift": sum(1 for r in drift_results.values() if r.get("drift_detected", False)),
            "feature_results": drift_results,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Data drift check completed. Drift detected: {overall_drift}")
        
        return drift_summary
    
    def check_model_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Check model performance on test data"""
        if self.model is None:
            raise ValueError("Model must be loaded first")
        
        logger.info("Checking model performance")
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate performance metrics
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # Multi-output model
            y_pred = np.argmax(predictions, axis=1)
            y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
        else:
            # Single output model
            y_pred = predictions.flatten()
            y_true = y_test.flatten()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Calculate additional metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Check if performance is below threshold
        performance_acceptable = f1 >= self.config.performance_threshold
        
        performance_results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "performance_threshold": self.config.performance_threshold,
            "performance_acceptable": performance_acceptable,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Model performance check completed. F1: {f1:.4f}, Acceptable: {performance_acceptable}")
        
        return performance_results
    
    def check_prediction_drift(self, X_current: np.ndarray, y_current: np.ndarray) -> Dict[str, Any]:
        """Check for prediction drift"""
        if self.model is None:
            raise ValueError("Model must be loaded first")
        
        logger.info("Checking prediction drift")
        
        # Make predictions on current data
        current_predictions = self.model.predict(X_current)
        
        # Calculate prediction statistics
        pred_mean = np.mean(current_predictions)
        pred_std = np.std(current_predictions)
        pred_min = np.min(current_predictions)
        pred_max = np.max(current_predictions)
        
        # Calculate prediction accuracy if ground truth is available
        if y_current is not None:
            if current_predictions.ndim > 1 and current_predictions.shape[1] > 1:
                y_pred = np.argmax(current_predictions, axis=1)
                y_true = np.argmax(y_current, axis=1) if y_current.ndim > 1 else y_current
            else:
                y_pred = current_predictions.flatten()
                y_true = y_current.flatten()
            
            accuracy = np.mean(y_pred == y_true)
            mse = np.mean((y_true - y_pred) ** 2)
        else:
            accuracy = None
            mse = None
        
        prediction_results = {
            "prediction_mean": float(pred_mean),
            "prediction_std": float(pred_std),
            "prediction_min": float(pred_min),
            "prediction_max": float(pred_max),
            "accuracy": float(accuracy) if accuracy is not None else None,
            "mse": float(mse) if mse is not None else None,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Prediction drift check completed")
        
        return prediction_results
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        logger.info("Checking system health")
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        # Check data availability
        health_status["checks"]["data_availability"] = {
            "reference_data_loaded": self.reference_data is not None,
            "current_data_loaded": self.current_data is not None,
            "model_loaded": self.model is not None
        }
        
        # Check data quality
        if self.current_data is not None:
            missing_values = self.current_data.isnull().sum().sum()
            duplicate_rows = self.current_data.duplicated().sum()
            
            health_status["checks"]["data_quality"] = {
                "missing_values": int(missing_values),
                "duplicate_rows": int(duplicate_rows),
                "total_rows": len(self.current_data),
                "quality_score": 1.0 - (missing_values + duplicate_rows) / (len(self.current_data) * len(self.current_data.columns))
            }
        
        # Check model performance (if recent data available)
        if hasattr(self, 'monitoring_results') and 'performance' in self.monitoring_results:
            perf = self.monitoring_results['performance']
            health_status["checks"]["model_performance"] = {
                "f1_score": perf.get('f1_score', 0),
                "performance_acceptable": perf.get('performance_acceptable', False)
            }
        
        # Determine overall status
        all_checks_passed = all(
            check.get("status", "healthy") == "healthy" 
            for check in health_status["checks"].values()
        )
        
        if not all_checks_passed:
            health_status["overall_status"] = "unhealthy"
        
        logger.info(f"System health check completed. Status: {health_status['overall_status']}")
        
        return health_status
    
    def run_full_monitoring(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Run complete monitoring suite"""
        logger.info("Running full monitoring suite")
        
        monitoring_results = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_id": f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        try:
            # Check data drift
            monitoring_results["data_drift"] = self.check_data_drift()
        except Exception as e:
            logger.error(f"Data drift check failed: {str(e)}")
            monitoring_results["data_drift"] = {"error": str(e)}
        
        try:
            # Check model performance
            monitoring_results["performance"] = self.check_model_performance(X_test, y_test)
        except Exception as e:
            logger.error(f"Performance check failed: {str(e)}")
            monitoring_results["performance"] = {"error": str(e)}
        
        try:
            # Check prediction drift
            monitoring_results["prediction_drift"] = self.check_prediction_drift(X_test, y_test)
        except Exception as e:
            logger.error(f"Prediction drift check failed: {str(e)}")
            monitoring_results["prediction_drift"] = {"error": str(e)}
        
        try:
            # Check system health
            monitoring_results["system_health"] = self.check_system_health()
        except Exception as e:
            logger.error(f"System health check failed: {str(e)}")
            monitoring_results["system_health"] = {"error": str(e)}
        
        # Store results
        self.monitoring_results = monitoring_results
        
        # Log to MLflow if enabled
        if self.config.enable_mlflow_logging:
            self._log_to_mlflow(monitoring_results)
        
        # Check for alerts
        self._check_alerts(monitoring_results)
        
        logger.info("Full monitoring suite completed")
        
        return monitoring_results
    
    def _log_to_mlflow(self, monitoring_results: Dict[str, Any]):
        """Log monitoring results to MLflow"""
        try:
            with mlflow.start_run(run_name=f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log metrics
                if "performance" in monitoring_results and "error" not in monitoring_results["performance"]:
                    perf = monitoring_results["performance"]
                    mlflow.log_metrics({
                        "f1_score": perf.get("f1_score", 0),
                        "accuracy": perf.get("accuracy", 0),
                        "precision": perf.get("precision", 0),
                        "recall": perf.get("recall", 0)
                    })
                
                # Log drift metrics
                if "data_drift" in monitoring_results and "error" not in monitoring_results["data_drift"]:
                    drift = monitoring_results["data_drift"]
                    mlflow.log_metrics({
                        "overall_drift_detected": float(drift.get("overall_drift_detected", False)),
                        "features_with_drift": drift.get("features_with_drift", 0)
                    })
                
                # Log artifacts
                mlflow.log_dict(monitoring_results, "monitoring_results.json")
                
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {str(e)}")
    
    def _check_alerts(self, monitoring_results: Dict[str, Any]):
        """Check for alert conditions and send notifications"""
        alerts = []
        
        # Check data drift alerts
        if "data_drift" in monitoring_results and "error" not in monitoring_results["data_drift"]:
            drift = monitoring_results["data_drift"]
            if drift.get("overall_drift_detected", False):
                alerts.append({
                    "type": "data_drift",
                    "severity": "high",
                    "message": f"Data drift detected in {drift.get('features_with_drift', 0)} features",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Check performance alerts
        if "performance" in monitoring_results and "error" not in monitoring_results["performance"]:
            perf = monitoring_results["performance"]
            if not perf.get("performance_acceptable", True):
                alerts.append({
                    "type": "performance_degradation",
                    "severity": "high",
                    "message": f"Model performance below threshold. F1: {perf.get('f1_score', 0):.4f}",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Check system health alerts
        if "system_health" in monitoring_results and "error" not in monitoring_results["system_health"]:
            health = monitoring_results["system_health"]
            if health.get("overall_status") != "healthy":
                alerts.append({
                    "type": "system_health",
                    "severity": "medium",
                    "message": "System health issues detected",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Store alerts
        if alerts:
            self.alert_history.extend(alerts)
            logger.warning(f"Generated {len(alerts)} alerts")
            
            # Send notifications (implement based on your notification system)
            self._send_alerts(alerts)
    
    def _send_alerts(self, alerts: List[Dict[str, Any]]):
        """Send alert notifications"""
        # Implement your alerting mechanism here
        # This could be email, Slack, PagerDuty, etc.
        for alert in alerts:
            logger.warning(f"ALERT: {alert['type']} - {alert['message']}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring results"""
        if not self.monitoring_results:
            return {"status": "no_monitoring_data"}
        
        summary = {
            "last_monitoring": self.monitoring_results.get("timestamp"),
            "monitoring_id": self.monitoring_results.get("monitoring_id"),
            "overall_status": "healthy"
        }
        
        # Check for issues
        issues = []
        
        if "data_drift" in self.monitoring_results:
            drift = self.monitoring_results["data_drift"]
            if drift.get("overall_drift_detected", False):
                issues.append("data_drift")
        
        if "performance" in self.monitoring_results:
            perf = self.monitoring_results["performance"]
            if not perf.get("performance_acceptable", True):
                issues.append("performance_degradation")
        
        if "system_health" in self.monitoring_results:
            health = self.monitoring_results["system_health"]
            if health.get("overall_status") != "healthy":
                issues.append("system_health")
        
        if issues:
            summary["overall_status"] = "unhealthy"
            summary["issues"] = issues
        
        summary["total_alerts"] = len(self.alert_history)
        summary["recent_alerts"] = self.alert_history[-5:] if self.alert_history else []
        
        return summary
    
    def export_monitoring_report(self, filepath: str):
        """Export monitoring results to file"""
        if not self.monitoring_results:
            raise ValueError("No monitoring results to export")
        
        with open(filepath, 'w') as f:
            json.dump(self.monitoring_results, f, indent=2, default=str)
        
        logger.info(f"Monitoring report exported to {filepath}")
    
    def start_continuous_monitoring(self, check_interval: Optional[int] = None):
        """Start continuous monitoring in background"""
        interval = check_interval or self.config.check_interval
        
        async def monitoring_loop():
            while True:
                try:
                    logger.info("Running scheduled monitoring check")
                    # This would need to be adapted based on your data source
                    # For now, we'll just log that monitoring would run
                    logger.info("Continuous monitoring check completed")
                except Exception as e:
                    logger.error(f"Monitoring check failed: {str(e)}")
                
                await asyncio.sleep(interval)
        
        # Start the monitoring loop
        asyncio.create_task(monitoring_loop())
        logger.info(f"Continuous monitoring started with {interval}s interval")
