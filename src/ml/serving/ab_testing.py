"""
A/B Testing Framework

This module implements A/B testing capabilities for ML models,
including traffic splitting, statistical testing, and automated promotion.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import hashlib
import uuid

from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing"""
    test_name: str
    control_model: str
    treatment_model: str
    traffic_split: float = 0.1  # 10% traffic to treatment
    success_metric: str = "f1_score"
    success_threshold: float = 0.05  # 5% improvement required
    test_duration_days: int = 7
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    enable_auto_promotion: bool = True
    enable_early_stopping: bool = True
    early_stopping_threshold: float = 0.01  # 1% significance


class ABTestingFramework:
    """
    A/B Testing framework for ML models
    
    Features:
    - Traffic splitting and routing
    - Statistical significance testing
    - Automated model promotion
    - Real-time monitoring and alerting
    - Experiment management
    """
    
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.test_id = str(uuid.uuid4())
        self.start_time = None
        self.end_time = None
        self.status = "not_started"  # not_started, running, completed, stopped
        self.results = {}
        self.metrics = {
            "control": {"requests": 0, "successes": 0, "failures": 0, "metrics": {}},
            "treatment": {"requests": 0, "successes": 0, "failures": 0, "metrics": {}}
        }
        
    def start_test(self):
        """Start the A/B test"""
        if self.status != "not_started":
            raise ValueError(f"Cannot start test. Current status: {self.status}")
        
        self.start_time = datetime.now()
        self.status = "running"
        
        logger.info(f"Started A/B test: {self.config.test_name} (ID: {self.test_id})")
        logger.info(f"Control model: {self.config.control_model}")
        logger.info(f"Treatment model: {self.config.treatment_model}")
        logger.info(f"Traffic split: {self.config.traffic_split:.1%}")
    
    def stop_test(self):
        """Stop the A/B test"""
        if self.status != "running":
            raise ValueError(f"Cannot stop test. Current status: {self.status}")
        
        self.end_time = datetime.now()
        self.status = "stopped"
        
        logger.info(f"Stopped A/B test: {self.config.test_name}")
    
    def complete_test(self):
        """Complete the A/B test and analyze results"""
        if self.status != "running":
            raise ValueError(f"Cannot complete test. Current status: {self.status}")
        
        self.end_time = datetime.now()
        self.status = "completed"
        
        # Analyze results
        self.results = self._analyze_results()
        
        logger.info(f"Completed A/B test: {self.config.test_name}")
        logger.info(f"Results: {self.results}")
    
    def route_request(self, request_id: str, user_id: str = None) -> str:
        """Route a request to either control or treatment model"""
        if self.status != "running":
            raise ValueError(f"Cannot route request. Test status: {self.status}")
        
        # Use user_id if provided, otherwise use request_id for consistent routing
        routing_key = user_id or request_id
        
        # Create deterministic hash for consistent routing
        hash_value = int(hashlib.md5(f"{routing_key}_{self.test_id}".encode()).hexdigest(), 16)
        routing_ratio = (hash_value % 100) / 100.0
        
        if routing_ratio < self.config.traffic_split:
            return "treatment"
        else:
            return "control"
    
    def record_prediction(self, 
                         variant: str,
                         request_id: str,
                         prediction: Any,
                         ground_truth: Any = None,
                         metrics: Dict[str, float] = None):
        """Record a prediction result"""
        if variant not in ["control", "treatment"]:
            raise ValueError(f"Invalid variant: {variant}")
        
        # Update basic metrics
        self.metrics[variant]["requests"] += 1
        
        if ground_truth is not None:
            # Calculate success/failure based on prediction accuracy
            if isinstance(prediction, (int, float)) and isinstance(ground_truth, (int, float)):
                # For regression tasks
                error = abs(prediction - ground_truth)
                threshold = 0.1 * abs(ground_truth)  # 10% error threshold
                if error <= threshold:
                    self.metrics[variant]["successes"] += 1
                else:
                    self.metrics[variant]["failures"] += 1
            else:
                # For classification tasks
                if prediction == ground_truth:
                    self.metrics[variant]["successes"] += 1
                else:
                    self.metrics[variant]["failures"] += 1
        
        # Update detailed metrics
        if metrics:
            for metric_name, metric_value in metrics.items():
                if metric_name not in self.metrics[variant]["metrics"]:
                    self.metrics[variant]["metrics"][metric_name] = []
                self.metrics[variant]["metrics"][metric_name].append(metric_value)
    
    def get_test_status(self) -> Dict[str, Any]:
        """Get current test status"""
        status = {
            "test_id": self.test_id,
            "test_name": self.config.test_name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": None,
            "traffic_split": self.config.traffic_split,
            "control_model": self.config.control_model,
            "treatment_model": self.config.treatment_model,
            "metrics": self.metrics
        }
        
        if self.start_time and self.end_time:
            status["duration"] = str(self.end_time - self.start_time)
        elif self.start_time:
            status["duration"] = str(datetime.now() - self.start_time)
        
        return status
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze A/B test results"""
        results = {
            "test_id": self.test_id,
            "test_name": self.config.test_name,
            "analysis_time": datetime.now().isoformat(),
            "control_metrics": {},
            "treatment_metrics": {},
            "statistical_test": {},
            "recommendation": "no_decision",
            "confidence": 0.0
        }
        
        # Calculate summary metrics for each variant
        for variant in ["control", "treatment"]:
            variant_metrics = self.metrics[variant]
            
            # Basic metrics
            total_requests = variant_metrics["requests"]
            successes = variant_metrics["successes"]
            failures = variant_metrics["failures"]
            
            results[f"{variant}_metrics"] = {
                "total_requests": total_requests,
                "successes": successes,
                "failures": failures,
                "success_rate": successes / total_requests if total_requests > 0 else 0,
                "failure_rate": failures / total_requests if total_requests > 0 else 0
            }
            
            # Detailed metrics
            for metric_name, metric_values in variant_metrics["metrics"].items():
                if metric_values:
                    results[f"{variant}_metrics"][metric_name] = {
                        "mean": np.mean(metric_values),
                        "std": np.std(metric_values),
                        "min": np.min(metric_values),
                        "max": np.max(metric_values),
                        "count": len(metric_values)
                    }
        
        # Statistical testing
        if self._has_sufficient_data():
            results["statistical_test"] = self._perform_statistical_test()
            results["recommendation"] = self._make_recommendation(results["statistical_test"])
            results["confidence"] = results["statistical_test"].get("confidence", 0.0)
        
        return results
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data for statistical testing"""
        control_requests = self.metrics["control"]["requests"]
        treatment_requests = self.metrics["treatment"]["requests"]
        
        return (control_requests >= self.config.min_sample_size and 
                treatment_requests >= self.config.min_sample_size)
    
    def _perform_statistical_test(self) -> Dict[str, Any]:
        """Perform statistical significance testing"""
        test_results = {}
        
        # Get the success metric data
        success_metric = self.config.success_metric
        
        control_values = self.metrics["control"]["metrics"].get(success_metric, [])
        treatment_values = self.metrics["treatment"]["metrics"].get(success_metric, [])
        
        if not control_values or not treatment_values:
            test_results["error"] = f"No data available for metric: {success_metric}"
            return test_results
        
        # Calculate basic statistics
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        control_std = np.std(control_values, ddof=1)
        treatment_std = np.std(treatment_values, ddof=1)
        
        # Calculate improvement
        improvement = treatment_mean - control_mean
        improvement_pct = (improvement / control_mean) * 100 if control_mean != 0 else 0
        
        test_results.update({
            "control_mean": float(control_mean),
            "treatment_mean": float(treatment_mean),
            "control_std": float(control_std),
            "treatment_std": float(treatment_std),
            "improvement": float(improvement),
            "improvement_pct": float(improvement_pct)
        })
        
        # Perform t-test
        try:
            t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
            test_results.update({
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < (1 - self.config.confidence_level)
            })
        except Exception as e:
            test_results["t_test_error"] = str(e)
        
        # Perform Mann-Whitney U test (non-parametric)
        try:
            u_stat, u_p_value = stats.mannwhitneyu(treatment_values, control_values, alternative='two-sided')
            test_results.update({
                "u_statistic": float(u_stat),
                "u_p_value": float(u_p_value),
                "u_significant": u_p_value < (1 - self.config.confidence_level)
            })
        except Exception as e:
            test_results["u_test_error"] = str(e)
        
        # Calculate confidence interval for the difference
        try:
            n1, n2 = len(control_values), len(treatment_values)
            se_diff = np.sqrt((control_std**2 / n1) + (treatment_std**2 / n2))
            df = n1 + n2 - 2  # degrees of freedom
            
            t_critical = stats.t.ppf(1 - (1 - self.config.confidence_level) / 2, df)
            margin_error = t_critical * se_diff
            
            ci_lower = improvement - margin_error
            ci_upper = improvement + margin_error
            
            test_results.update({
                "confidence_interval": {
                    "lower": float(ci_lower),
                    "upper": float(ci_upper),
                    "level": self.config.confidence_level
                }
            })
        except Exception as e:
            test_results["ci_error"] = str(e)
        
        return test_results
    
    def _make_recommendation(self, statistical_test: Dict[str, Any]) -> str:
        """Make recommendation based on statistical test results"""
        if "error" in statistical_test:
            return "insufficient_data"
        
        improvement_pct = statistical_test.get("improvement_pct", 0)
        significant = statistical_test.get("significant", False)
        
        # Check if improvement meets threshold
        meets_threshold = abs(improvement_pct) >= (self.config.success_threshold * 100)
        
        if significant and meets_threshold:
            if improvement_pct > 0:
                return "promote_treatment"
            else:
                return "keep_control"
        elif significant and not meets_threshold:
            return "no_practical_difference"
        else:
            return "insufficient_evidence"
    
    def should_promote_treatment(self) -> bool:
        """Check if treatment should be promoted based on current results"""
        if self.status != "running":
            return False
        
        # Check if we have enough data for early stopping
        if not self._has_sufficient_data():
            return False
        
        # Perform quick statistical test
        quick_results = self._perform_statistical_test()
        recommendation = self._make_recommendation(quick_results)
        
        return recommendation == "promote_treatment"
    
    def get_confidence_interval(self, metric_name: str = None) -> Dict[str, Any]:
        """Get confidence interval for a specific metric"""
        if metric_name is None:
            metric_name = self.config.success_metric
        
        control_values = self.metrics["control"]["metrics"].get(metric_name, [])
        treatment_values = self.metrics["treatment"]["metrics"].get(metric_name, [])
        
        if not control_values or not treatment_values:
            return {"error": f"No data available for metric: {metric_name}"}
        
        # Calculate confidence interval for the difference
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        control_std = np.std(control_values, ddof=1)
        treatment_std = np.std(treatment_values, ddof=1)
        
        n1, n2 = len(control_values), len(treatment_values)
        se_diff = np.sqrt((control_std**2 / n1) + (treatment_std**2 / n2))
        df = n1 + n2 - 2
        
        t_critical = stats.t.ppf(1 - (1 - self.config.confidence_level) / 2, df)
        margin_error = t_critical * se_diff
        
        difference = treatment_mean - control_mean
        ci_lower = difference - margin_error
        ci_upper = difference + margin_error
        
        return {
            "metric": metric_name,
            "difference": float(difference),
            "confidence_interval": {
                "lower": float(ci_lower),
                "upper": float(ci_upper),
                "level": self.config.confidence_level
            },
            "margin_error": float(margin_error)
        }
    
    def export_results(self, filepath: str):
        """Export test results to file"""
        results = {
            "config": {
                "test_name": self.config.test_name,
                "control_model": self.config.control_model,
                "treatment_model": self.config.treatment_model,
                "traffic_split": self.config.traffic_split,
                "success_metric": self.config.success_metric,
                "success_threshold": self.config.success_threshold,
                "test_duration_days": self.config.test_duration_days,
                "min_sample_size": self.config.min_sample_size,
                "confidence_level": self.config.confidence_level
            },
            "test_info": {
                "test_id": self.test_id,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "status": self.status
            },
            "metrics": self.metrics,
            "results": self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results exported to: {filepath}")
    
    def get_recommendation_summary(self) -> Dict[str, Any]:
        """Get a summary of the test recommendation"""
        if not self.results:
            return {"status": "no_results", "message": "Test not completed"}
        
        recommendation = self.results.get("recommendation", "no_decision")
        confidence = self.results.get("confidence", 0.0)
        
        summary = {
            "recommendation": recommendation,
            "confidence": confidence,
            "interpretation": self._interpret_recommendation(recommendation, confidence)
        }
        
        if "statistical_test" in self.results:
            stat_test = self.results["statistical_test"]
            summary.update({
                "improvement_pct": stat_test.get("improvement_pct", 0),
                "p_value": stat_test.get("p_value", 1.0),
                "significant": stat_test.get("significant", False)
            })
        
        return summary
    
    def _interpret_recommendation(self, recommendation: str, confidence: float) -> str:
        """Interpret the recommendation in human-readable terms"""
        interpretations = {
            "promote_treatment": f"Promote treatment model (confidence: {confidence:.1%})",
            "keep_control": f"Keep control model (confidence: {confidence:.1%})",
            "no_practical_difference": "No practical difference between models",
            "insufficient_evidence": "Insufficient evidence to make a decision",
            "insufficient_data": "Not enough data collected yet",
            "no_decision": "No decision made yet"
        }
        
        return interpretations.get(recommendation, f"Unknown recommendation: {recommendation}")


class ABTestManager:
    """
    Manager for multiple A/B tests
    """
    
    def __init__(self):
        self.tests = {}
    
    def create_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test"""
        test = ABTestingFramework(config)
        self.tests[test.test_id] = test
        return test.test_id
    
    def get_test(self, test_id: str) -> Optional[ABTestingFramework]:
        """Get an A/B test by ID"""
        return self.tests.get(test_id)
    
    def list_tests(self) -> List[Dict[str, Any]]:
        """List all A/B tests"""
        return [
            {
                "test_id": test_id,
                "test_name": test.config.test_name,
                "status": test.status,
                "start_time": test.start_time.isoformat() if test.start_time else None
            }
            for test_id, test in self.tests.items()
        ]
    
    def stop_all_tests(self):
        """Stop all running tests"""
        for test in self.tests.values():
            if test.status == "running":
                test.stop_test()
    
    def cleanup_completed_tests(self, older_than_days: int = 30):
        """Remove completed tests older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        tests_to_remove = []
        for test_id, test in self.tests.items():
            if (test.status in ["completed", "stopped"] and 
                test.end_time and 
                test.end_time < cutoff_date):
                tests_to_remove.append(test_id)
        
        for test_id in tests_to_remove:
            del self.tests[test_id]
        
        logger.info(f"Cleaned up {len(tests_to_remove)} old tests")


# Global A/B test manager
ab_test_manager = ABTestManager()


def get_ab_test_manager() -> ABTestManager:
    """Get the global A/B test manager"""
    return ab_test_manager
