"""
Feature Validator
Validates feature quality, consistency, and drift detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
import json

logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    """Feature validation rule"""
    name: str
    description: str
    rule_type: str  # 'range', 'distribution', 'completeness', 'consistency'
    parameters: Dict[str, Any]
    severity: str = 'error'  # 'error', 'warning', 'info'

@dataclass
class ValidationResult:
    """Feature validation result"""
    feature_name: str
    rule_name: str
    passed: bool
    score: float
    message: str
    severity: str
    timestamp: datetime
    details: Dict[str, Any] = None

class FeatureValidator:
    """
    Comprehensive feature validation and quality monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validation_rules = {}
        self.baseline_stats = {}
        self.validation_history = []
        
        # Initialize default validation rules
        self._initialize_default_rules()
        
        logger.info("FeatureValidator initialized")
    
    def _initialize_default_rules(self):
        """Initialize default validation rules"""
        # Completeness rules
        self.add_validation_rule(ValidationRule(
            name="completeness_threshold",
            description="Check if feature completeness is above threshold",
            rule_type="completeness",
            parameters={"threshold": 0.95},
            severity="error"
        ))
        
        # Range rules
        self.add_validation_rule(ValidationRule(
            name="consumption_range",
            description="Check if consumption values are within reasonable range",
            rule_type="range",
            parameters={"min_value": 0, "max_value": 1000},
            severity="error"
        ))
        
        # Distribution rules
        self.add_validation_rule(ValidationRule(
            name="distribution_stability",
            description="Check if feature distribution is stable over time",
            rule_type="distribution",
            parameters={"stability_threshold": 0.1},
            severity="warning"
        ))
        
        # Consistency rules
        self.add_validation_rule(ValidationRule(
            name="temporal_consistency",
            description="Check if feature values are temporally consistent",
            rule_type="consistency",
            parameters={"max_change_rate": 0.5},
            severity="warning"
        ))
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a custom validation rule"""
        self.validation_rules[rule.name] = rule
        logger.info(f"Added validation rule: {rule.name}")
    
    def remove_validation_rule(self, rule_name: str):
        """Remove a validation rule"""
        if rule_name in self.validation_rules:
            del self.validation_rules[rule_name]
            logger.info(f"Removed validation rule: {rule_name}")
    
    def validate_feature(self, 
                        feature_name: str, 
                        data: pd.Series,
                        rule_name: Optional[str] = None) -> List[ValidationResult]:
        """Validate a single feature against rules"""
        results = []
        
        # Get rules to apply
        rules_to_apply = [rule_name] if rule_name else list(self.validation_rules.keys())
        
        for rule_name in rules_to_apply:
            if rule_name not in self.validation_rules:
                continue
            
            rule = self.validation_rules[rule_name]
            
            try:
                result = self._apply_validation_rule(feature_name, data, rule)
                results.append(result)
            except Exception as e:
                logger.error(f"Validation rule {rule_name} failed: {str(e)}")
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_name=rule_name,
                    passed=False,
                    score=0.0,
                    message=f"Validation error: {str(e)}",
                    severity="error",
                    timestamp=datetime.now()
                ))
        
        return results
    
    def validate_features(self, 
                         data: pd.DataFrame,
                         feature_names: Optional[List[str]] = None) -> Dict[str, List[ValidationResult]]:
        """Validate multiple features"""
        if feature_names is None:
            feature_names = data.columns.tolist()
        
        validation_results = {}
        
        for feature_name in feature_names:
            if feature_name in data.columns:
                results = self.validate_feature(feature_name, data[feature_name])
                validation_results[feature_name] = results
            else:
                logger.warning(f"Feature {feature_name} not found in data")
        
        # Store validation history
        self.validation_history.append({
            'timestamp': datetime.now(),
            'feature_count': len(feature_names),
            'results': validation_results
        })
        
        return validation_results
    
    def _apply_validation_rule(self, 
                              feature_name: str, 
                              data: pd.Series, 
                              rule: ValidationRule) -> ValidationResult:
        """Apply a specific validation rule"""
        if rule.rule_type == "completeness":
            return self._validate_completeness(feature_name, data, rule)
        elif rule.rule_type == "range":
            return self._validate_range(feature_name, data, rule)
        elif rule.rule_type == "distribution":
            return self._validate_distribution(feature_name, data, rule)
        elif rule.rule_type == "consistency":
            return self._validate_consistency(feature_name, data, rule)
        else:
            raise ValueError(f"Unknown rule type: {rule.rule_type}")
    
    def _validate_completeness(self, 
                              feature_name: str, 
                              data: pd.Series, 
                              rule: ValidationRule) -> ValidationResult:
        """Validate feature completeness"""
        threshold = rule.parameters["threshold"]
        completeness = 1 - data.isnull().sum() / len(data)
        
        passed = completeness >= threshold
        score = completeness
        
        message = f"Completeness: {completeness:.3f} (threshold: {threshold})"
        
        return ValidationResult(
            feature_name=feature_name,
            rule_name=rule.name,
            passed=passed,
            score=score,
            message=message,
            severity=rule.severity,
            timestamp=datetime.now(),
            details={"completeness": completeness, "threshold": threshold}
        )
    
    def _validate_range(self, 
                       feature_name: str, 
                       data: pd.Series, 
                       rule: ValidationRule) -> ValidationResult:
        """Validate feature value range"""
        min_value = rule.parameters["min_value"]
        max_value = rule.parameters["max_value"]
        
        # Remove null values for range check
        valid_data = data.dropna()
        
        if len(valid_data) == 0:
            return ValidationResult(
                feature_name=feature_name,
                rule_name=rule.name,
                passed=False,
                score=0.0,
                message="No valid data for range validation",
                severity=rule.severity,
                timestamp=datetime.now()
            )
        
        out_of_range = ((valid_data < min_value) | (valid_data > max_value)).sum()
        range_violation_rate = out_of_range / len(valid_data)
        
        passed = range_violation_rate == 0
        score = 1 - range_violation_rate
        
        message = f"Range violations: {out_of_range}/{len(valid_data)} ({range_violation_rate:.3f})"
        
        return ValidationResult(
            feature_name=feature_name,
            rule_name=rule.name,
            passed=passed,
            score=score,
            message=message,
            severity=rule.severity,
            timestamp=datetime.now(),
            details={
                "min_value": min_value,
                "max_value": max_value,
                "out_of_range_count": out_of_range,
                "violation_rate": range_violation_rate
            }
        )
    
    def _validate_distribution(self, 
                             feature_name: str, 
                             data: pd.Series, 
                             rule: ValidationRule) -> ValidationResult:
        """Validate feature distribution stability"""
        stability_threshold = rule.parameters["stability_threshold"]
        
        # Remove null values
        valid_data = data.dropna()
        
        if len(valid_data) < 10:
            return ValidationResult(
                feature_name=feature_name,
                rule_name=rule.name,
                passed=True,
                score=1.0,
                message="Insufficient data for distribution validation",
                severity=rule.severity,
                timestamp=datetime.now()
            )
        
        # Split data into two halves
        mid_point = len(valid_data) // 2
        first_half = valid_data[:mid_point]
        second_half = valid_data[mid_point:]
        
        # Calculate distribution statistics
        first_stats = {
            'mean': first_half.mean(),
            'std': first_half.std(),
            'skew': first_half.skew(),
            'kurtosis': first_half.kurtosis()
        }
        
        second_stats = {
            'mean': second_half.mean(),
            'std': second_half.std(),
            'skew': second_half.skew(),
            'kurtosis': second_half.kurtosis()
        }
        
        # Calculate stability score
        stability_scores = []
        for stat in ['mean', 'std', 'skew', 'kurtosis']:
            if first_stats[stat] != 0:
                change = abs(second_stats[stat] - first_stats[stat]) / abs(first_stats[stat])
                stability_scores.append(change)
        
        avg_stability = np.mean(stability_scores) if stability_scores else 0
        passed = avg_stability <= stability_threshold
        score = 1 - avg_stability
        
        message = f"Distribution stability: {avg_stability:.3f} (threshold: {stability_threshold})"
        
        return ValidationResult(
            feature_name=feature_name,
            rule_name=rule.name,
            passed=passed,
            score=score,
            message=message,
            severity=rule.severity,
            timestamp=datetime.now(),
            details={
                "first_half_stats": first_stats,
                "second_half_stats": second_stats,
                "stability_score": avg_stability,
                "threshold": stability_threshold
            }
        )
    
    def _validate_consistency(self, 
                            feature_name: str, 
                            data: pd.Series, 
                            rule: ValidationRule) -> ValidationResult:
        """Validate temporal consistency"""
        max_change_rate = rule.parameters["max_change_rate"]
        
        # Remove null values
        valid_data = data.dropna()
        
        if len(valid_data) < 2:
            return ValidationResult(
                feature_name=feature_name,
                rule_name=rule.name,
                passed=True,
                score=1.0,
                message="Insufficient data for consistency validation",
                severity=rule.severity,
                timestamp=datetime.now()
            )
        
        # Calculate change rates
        changes = valid_data.diff().abs()
        change_rates = changes / valid_data.shift(1).abs()
        
        # Remove infinite and NaN values
        change_rates = change_rates.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(change_rates) == 0:
            return ValidationResult(
                feature_name=feature_name,
                rule_name=rule.name,
                passed=True,
                score=1.0,
                message="No valid change rates calculated",
                severity=rule.severity,
                timestamp=datetime.now()
            )
        
        # Check for excessive changes
        excessive_changes = (change_rates > max_change_rate).sum()
        violation_rate = excessive_changes / len(change_rates)
        
        passed = violation_rate == 0
        score = 1 - violation_rate
        
        message = f"Consistency violations: {excessive_changes}/{len(change_rates)} ({violation_rate:.3f})"
        
        return ValidationResult(
            feature_name=feature_name,
            rule_name=rule.name,
            passed=passed,
            score=score,
            message=message,
            severity=rule.severity,
            timestamp=datetime.now(),
            details={
                "max_change_rate": max_change_rate,
                "excessive_changes": excessive_changes,
                "violation_rate": violation_rate,
                "avg_change_rate": change_rates.mean()
            }
        )
    
    def detect_feature_drift(self, 
                           feature_name: str,
                           current_data: pd.Series,
                           reference_data: pd.Series) -> Dict[str, Any]:
        """Detect feature drift between reference and current data"""
        try:
            # Remove null values
            current_clean = current_data.dropna()
            reference_clean = reference_data.dropna()
            
            if len(current_clean) < 10 or len(reference_clean) < 10:
                return {
                    "drift_detected": False,
                    "drift_score": 0.0,
                    "message": "Insufficient data for drift detection"
                }
            
            # Statistical tests
            drift_results = {}
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(reference_clean, current_clean)
            drift_results["ks_test"] = {
                "statistic": ks_stat,
                "p_value": ks_pvalue,
                "drift_detected": ks_pvalue < 0.05
            }
            
            # Population Stability Index (PSI)
            psi_score = self._calculate_psi(reference_clean, current_clean)
            drift_results["psi"] = {
                "score": psi_score,
                "drift_detected": psi_score > 0.2
            }
            
            # Wasserstein distance
            wasserstein_distance = stats.wasserstein_distance(reference_clean, current_clean)
            drift_results["wasserstein_distance"] = {
                "distance": wasserstein_distance,
                "drift_detected": wasserstein_distance > 0.1
            }
            
            # Overall drift detection
            drift_detected = (
                drift_results["ks_test"]["drift_detected"] or
                drift_results["psi"]["drift_detected"] or
                drift_results["wasserstein_distance"]["drift_detected"]
            )
            
            # Calculate overall drift score
            drift_score = max(
                drift_results["ks_test"]["statistic"],
                drift_results["psi"]["score"],
                drift_results["wasserstein_distance"]["distance"]
            )
            
            return {
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "details": drift_results,
                "message": f"Drift detected: {drift_detected}, Score: {drift_score:.3f}"
            }
            
        except Exception as e:
            logger.error(f"Drift detection failed for {feature_name}: {str(e)}")
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "error": str(e)
            }
    
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
    
    def get_validation_summary(self, 
                             validation_results: Dict[str, List[ValidationResult]]) -> Dict[str, Any]:
        """Get summary of validation results"""
        summary = {
            "total_features": len(validation_results),
            "total_rules": 0,
            "passed_rules": 0,
            "failed_rules": 0,
            "warning_rules": 0,
            "error_rules": 0,
            "overall_score": 0.0,
            "feature_scores": {},
            "rule_scores": {}
        }
        
        all_scores = []
        
        for feature_name, results in validation_results.items():
            feature_scores = []
            for result in results:
                summary["total_rules"] += 1
                feature_scores.append(result.score)
                all_scores.append(result.score)
                
                if result.passed:
                    summary["passed_rules"] += 1
                else:
                    summary["failed_rules"] += 1
                
                if result.severity == "warning":
                    summary["warning_rules"] += 1
                elif result.severity == "error":
                    summary["error_rules"] += 1
                
                # Track rule scores
                if result.rule_name not in summary["rule_scores"]:
                    summary["rule_scores"][result.rule_name] = []
                summary["rule_scores"][result.rule_name].append(result.score)
            
            summary["feature_scores"][feature_name] = np.mean(feature_scores) if feature_scores else 0.0
        
        # Calculate overall score
        summary["overall_score"] = np.mean(all_scores) if all_scores else 0.0
        
        # Calculate rule scores
        for rule_name, scores in summary["rule_scores"].items():
            summary["rule_scores"][rule_name] = np.mean(scores)
        
        return summary
    
    def export_validation_report(self, 
                               validation_results: Dict[str, List[ValidationResult]],
                               filepath: str):
        """Export validation report to JSON"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_validation_summary(validation_results),
            "detailed_results": {}
        }
        
        for feature_name, results in validation_results.items():
            report["detailed_results"][feature_name] = []
            for result in results:
                report["detailed_results"][feature_name].append({
                    "rule_name": result.rule_name,
                    "passed": result.passed,
                    "score": result.score,
                    "message": result.message,
                    "severity": result.severity,
                    "timestamp": result.timestamp.isoformat(),
                    "details": result.details
                })
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report exported to {filepath}")
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history"""
        return self.validation_history
    
    def clear_validation_history(self):
        """Clear validation history"""
        self.validation_history = []
        logger.info("Validation history cleared")
