"""
Remediation Engine
Automated quality remediation and data cleaning
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class RemediationAction(Enum):
    """Remediation actions"""
    REMOVE_NULLS = "remove_nulls"
    FILL_NULLS = "fill_nulls"
    REMOVE_DUPLICATES = "remove_duplicates"
    STANDARDIZE_FORMAT = "standardize_format"
    REMOVE_OUTLIERS = "remove_outliers"
    VALIDATE_TYPES = "validate_types"
    NORMALIZE_CASE = "normalize_case"
    TRIM_WHITESPACE = "trim_whitespace"
    CUSTOM = "custom"

class RemediationSeverity(Enum):
    """Remediation severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RemediationRule:
    """Remediation rule"""
    rule_id: str
    name: str
    action: RemediationAction
    conditions: Dict[str, Any]
    severity: RemediationSeverity
    enabled: bool
    created_at: datetime

@dataclass
class RemediationResult:
    """Remediation result"""
    rule_id: str
    action: RemediationAction
    rows_affected: int
    columns_affected: List[str]
    success: bool
    message: str
    before_stats: Dict[str, Any]
    after_stats: Dict[str, Any]
    execution_time: float

class RemediationEngine:
    """
    Automated quality remediation and data cleaning
    """
    
    def __init__(self):
        self.remediation_rules = {}
        self.remediation_history = []
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("RemediationEngine initialized")
    
    def _initialize_default_rules(self):
        """Initialize default remediation rules"""
        try:
            # Remove nulls rule
            self.add_remediation_rule(
                name="Remove Null Rows",
                action=RemediationAction.REMOVE_NULLS,
                conditions={"threshold": 0.5},  # Remove rows with >50% nulls
                severity=RemediationSeverity.MEDIUM
            )
            
            # Fill nulls rule
            self.add_remediation_rule(
                name="Fill Null Values",
                action=RemediationAction.FILL_NULLS,
                conditions={"method": "forward_fill"},
                severity=RemediationSeverity.LOW
            )
            
            # Remove duplicates rule
            self.add_remediation_rule(
                name="Remove Duplicates",
                action=RemediationAction.REMOVE_DUPLICATES,
                conditions={"subset": None},  # Check all columns
                severity=RemediationSeverity.MEDIUM
            )
            
            # Standardize format rule
            self.add_remediation_rule(
                name="Standardize String Format",
                action=RemediationAction.STANDARDIZE_FORMAT,
                conditions={"columns": None},  # All string columns
                severity=RemediationSeverity.LOW
            )
            
            # Remove outliers rule
            self.add_remediation_rule(
                name="Remove Outliers",
                action=RemediationAction.REMOVE_OUTLIERS,
                conditions={"method": "iqr", "factor": 1.5},
                severity=RemediationSeverity.HIGH
            )
            
            # Normalize case rule
            self.add_remediation_rule(
                name="Normalize Case",
                action=RemediationAction.NORMALIZE_CASE,
                conditions={"case": "lower"},
                severity=RemediationSeverity.LOW
            )
            
            # Trim whitespace rule
            self.add_remediation_rule(
                name="Trim Whitespace",
                action=RemediationAction.TRIM_WHITESPACE,
                conditions={},
                severity=RemediationSeverity.LOW
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize default rules: {str(e)}")
    
    def add_remediation_rule(self,
                           name: str,
                           action: RemediationAction,
                           conditions: Dict[str, Any],
                           severity: RemediationSeverity,
                           enabled: bool = True) -> str:
        """Add a remediation rule"""
        try:
            rule_id = f"rule_{int(datetime.now().timestamp())}"
            
            rule = RemediationRule(
                rule_id=rule_id,
                name=name,
                action=action,
                conditions=conditions,
                severity=severity,
                enabled=enabled,
                created_at=datetime.now()
            )
            
            self.remediation_rules[rule_id] = rule
            
            logger.info(f"Remediation rule added: {rule_id}")
            return rule_id
            
        except Exception as e:
            logger.error(f"Failed to add remediation rule: {str(e)}")
            return ""
    
    def remediate_data(self,
                      data: pd.DataFrame,
                      rules: List[str] = None,
                      dry_run: bool = False) -> List[RemediationResult]:
        """Apply remediation rules to data"""
        try:
            if data.empty:
                return []
            
            # Filter rules to apply
            rules_to_apply = []
            if rules:
                rules_to_apply = [self.remediation_rules[rule_id] for rule_id in rules 
                                if rule_id in self.remediation_rules and self.remediation_rules[rule_id].enabled]
            else:
                rules_to_apply = [rule for rule in self.remediation_rules.values() if rule.enabled]
            
            results = []
            current_data = data.copy()
            
            for rule in rules_to_apply:
                try:
                    result = self._apply_remediation_rule(current_data, rule, dry_run)
                    results.append(result)
                    
                    # Update data if not dry run and rule was successful
                    if not dry_run and result.success:
                        current_data = self._apply_result_to_data(current_data, result)
                    
                except Exception as e:
                    logger.error(f"Failed to apply rule {rule.rule_id}: {str(e)}")
                    # Create error result
                    error_result = RemediationResult(
                        rule_id=rule.rule_id,
                        action=rule.action,
                        rows_affected=0,
                        columns_affected=[],
                        success=False,
                        message=f"Rule execution failed: {str(e)}",
                        before_stats={},
                        after_stats={},
                        execution_time=0.0
                    )
                    results.append(error_result)
            
            # Record remediation history
            if not dry_run:
                self.remediation_history.append({
                    "timestamp": datetime.now(),
                    "rules_applied": len([r for r in results if r.success]),
                    "total_rules": len(results),
                    "dry_run": dry_run
                })
            
            logger.info(f"Remediation completed: {len([r for r in results if r.success])} rules applied successfully")
            return results
            
        except Exception as e:
            logger.error(f"Failed to remediate data: {str(e)}")
            return []
    
    def _apply_remediation_rule(self,
                               data: pd.DataFrame,
                               rule: RemediationRule,
                               dry_run: bool) -> RemediationResult:
        """Apply a single remediation rule"""
        try:
            start_time = datetime.now()
            
            # Get before stats
            before_stats = self._get_data_stats(data)
            
            # Apply rule based on action
            if rule.action == RemediationAction.REMOVE_NULLS:
                result_data, affected_rows = self._remove_nulls(data, rule.conditions)
            elif rule.action == RemediationAction.FILL_NULLS:
                result_data, affected_rows = self._fill_nulls(data, rule.conditions)
            elif rule.action == RemediationAction.REMOVE_DUPLICATES:
                result_data, affected_rows = self._remove_duplicates(data, rule.conditions)
            elif rule.action == RemediationAction.STANDARDIZE_FORMAT:
                result_data, affected_rows = self._standardize_format(data, rule.conditions)
            elif rule.action == RemediationAction.REMOVE_OUTLIERS:
                result_data, affected_rows = self._remove_outliers(data, rule.conditions)
            elif rule.action == RemediationAction.NORMALIZE_CASE:
                result_data, affected_rows = self._normalize_case(data, rule.conditions)
            elif rule.action == RemediationAction.TRIM_WHITESPACE:
                result_data, affected_rows = self._trim_whitespace(data, rule.conditions)
            else:
                result_data, affected_rows = data, 0
            
            # Get after stats
            after_stats = self._get_data_stats(result_data)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Determine success
            success = not dry_run and affected_rows > 0
            
            # Get affected columns
            affected_columns = self._get_affected_columns(data, result_data)
            
            return RemediationResult(
                rule_id=rule.rule_id,
                action=rule.action,
                rows_affected=affected_rows,
                columns_affected=affected_columns,
                success=success,
                message=f"Rule {rule.name} applied successfully" if success else f"Rule {rule.name} - dry run",
                before_stats=before_stats,
                after_stats=after_stats,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Failed to apply remediation rule {rule.rule_id}: {str(e)}")
            return RemediationResult(
                rule_id=rule.rule_id,
                action=rule.action,
                rows_affected=0,
                columns_affected=[],
                success=False,
                message=f"Rule execution failed: {str(e)}",
                before_stats={},
                after_stats={},
                execution_time=0.0
            )
    
    def _remove_nulls(self, data: pd.DataFrame, conditions: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Remove rows with null values"""
        try:
            threshold = conditions.get("threshold", 0.5)
            
            # Calculate null ratio per row
            null_ratio = data.isnull().sum(axis=1) / len(data.columns)
            
            # Remove rows above threshold
            mask = null_ratio <= threshold
            result_data = data[mask]
            
            affected_rows = len(data) - len(result_data)
            return result_data, affected_rows
            
        except Exception as e:
            logger.error(f"Failed to remove nulls: {str(e)}")
            return data, 0
    
    def _fill_nulls(self, data: pd.DataFrame, conditions: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Fill null values"""
        try:
            method = conditions.get("method", "forward_fill")
            result_data = data.copy()
            
            affected_rows = 0
            
            if method == "forward_fill":
                result_data = result_data.fillna(method='ffill')
                affected_rows = data.isnull().sum().sum() - result_data.isnull().sum().sum()
            elif method == "backward_fill":
                result_data = result_data.fillna(method='bfill')
                affected_rows = data.isnull().sum().sum() - result_data.isnull().sum().sum()
            elif method == "mean":
                numeric_cols = result_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    mean_val = result_data[col].mean()
                    if pd.notna(mean_val):
                        null_count = result_data[col].isnull().sum()
                        result_data[col] = result_data[col].fillna(mean_val)
                        affected_rows += null_count
            elif method == "median":
                numeric_cols = result_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    median_val = result_data[col].median()
                    if pd.notna(median_val):
                        null_count = result_data[col].isnull().sum()
                        result_data[col] = result_data[col].fillna(median_val)
                        affected_rows += null_count
            
            return result_data, affected_rows
            
        except Exception as e:
            logger.error(f"Failed to fill nulls: {str(e)}")
            return data, 0
    
    def _remove_duplicates(self, data: pd.DataFrame, conditions: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Remove duplicate rows"""
        try:
            subset = conditions.get("subset", None)
            
            original_count = len(data)
            result_data = data.drop_duplicates(subset=subset)
            affected_rows = original_count - len(result_data)
            
            return result_data, affected_rows
            
        except Exception as e:
            logger.error(f"Failed to remove duplicates: {str(e)}")
            return data, 0
    
    def _standardize_format(self, data: pd.DataFrame, conditions: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Standardize string format"""
        try:
            columns = conditions.get("columns", None)
            result_data = data.copy()
            
            if columns is None:
                columns = result_data.select_dtypes(include=['object']).columns.tolist()
            
            affected_rows = 0
            
            for col in columns:
                if col in result_data.columns:
                    # Remove extra whitespace
                    original_values = result_data[col].astype(str)
                    standardized_values = original_values.str.strip().str.replace(r'\s+', ' ', regex=True)
                    
                    # Count changes
                    changes = (original_values != standardized_values).sum()
                    affected_rows += changes
                    
                    result_data[col] = standardized_values
            
            return result_data, affected_rows
            
        except Exception as e:
            logger.error(f"Failed to standardize format: {str(e)}")
            return data, 0
    
    def _remove_outliers(self, data: pd.DataFrame, conditions: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Remove outliers"""
        try:
            method = conditions.get("method", "iqr")
            factor = conditions.get("factor", 1.5)
            
            result_data = data.copy()
            affected_rows = 0
            
            numeric_cols = result_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if method == "iqr":
                    Q1 = result_data[col].quantile(0.25)
                    Q3 = result_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                    
                    outlier_mask = (result_data[col] < lower_bound) | (result_data[col] > upper_bound)
                    affected_rows += outlier_mask.sum()
                    
                    result_data = result_data[~outlier_mask]
                elif method == "zscore":
                    z_scores = np.abs((result_data[col] - result_data[col].mean()) / result_data[col].std())
                    outlier_mask = z_scores > factor
                    affected_rows += outlier_mask.sum()
                    
                    result_data = result_data[~outlier_mask]
            
            return result_data, affected_rows
            
        except Exception as e:
            logger.error(f"Failed to remove outliers: {str(e)}")
            return data, 0
    
    def _normalize_case(self, data: pd.DataFrame, conditions: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Normalize case of string columns"""
        try:
            case = conditions.get("case", "lower")
            result_data = data.copy()
            
            affected_rows = 0
            string_cols = result_data.select_dtypes(include=['object']).columns
            
            for col in string_cols:
                if case == "lower":
                    original_values = result_data[col].astype(str)
                    normalized_values = original_values.str.lower()
                elif case == "upper":
                    original_values = result_data[col].astype(str)
                    normalized_values = original_values.str.upper()
                elif case == "title":
                    original_values = result_data[col].astype(str)
                    normalized_values = original_values.str.title()
                else:
                    continue
                
                changes = (original_values != normalized_values).sum()
                affected_rows += changes
                
                result_data[col] = normalized_values
            
            return result_data, affected_rows
            
        except Exception as e:
            logger.error(f"Failed to normalize case: {str(e)}")
            return data, 0
    
    def _trim_whitespace(self, data: pd.DataFrame, conditions: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Trim whitespace from string columns"""
        try:
            result_data = data.copy()
            affected_rows = 0
            
            string_cols = result_data.select_dtypes(include=['object']).columns
            
            for col in string_cols:
                original_values = result_data[col].astype(str)
                trimmed_values = original_values.str.strip()
                
                changes = (original_values != trimmed_values).sum()
                affected_rows += changes
                
                result_data[col] = trimmed_values
            
            return result_data, affected_rows
            
        except Exception as e:
            logger.error(f"Failed to trim whitespace: {str(e)}")
            return data, 0
    
    def _get_data_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get data statistics"""
        try:
            return {
                "row_count": len(data),
                "column_count": len(data.columns),
                "null_count": data.isnull().sum().sum(),
                "duplicate_count": data.duplicated().sum(),
                "memory_usage": data.memory_usage(deep=True).sum()
            }
            
        except Exception as e:
            logger.error(f"Failed to get data stats: {str(e)}")
            return {}
    
    def _get_affected_columns(self, original_data: pd.DataFrame, result_data: pd.DataFrame) -> List[str]:
        """Get columns that were affected by remediation"""
        try:
            affected_columns = []
            
            for col in original_data.columns:
                if col in result_data.columns:
                    # Check if column values changed
                    if not original_data[col].equals(result_data[col]):
                        affected_columns.append(col)
            
            return affected_columns
            
        except Exception as e:
            logger.error(f"Failed to get affected columns: {str(e)}")
            return []
    
    def _apply_result_to_data(self, data: pd.DataFrame, result: RemediationResult) -> pd.DataFrame:
        """Apply remediation result to data (placeholder)"""
        # In real implementation, this would apply the specific changes
        # For now, return the original data
        return data
    
    def get_remediation_summary(self, results: List[RemediationResult]) -> Dict[str, Any]:
        """Get remediation summary"""
        try:
            total_rules = len(results)
            successful_rules = len([r for r in results if r.success])
            total_rows_affected = sum(r.rows_affected for r in results)
            total_execution_time = sum(r.execution_time for r in results)
            
            # Count by action
            action_counts = {}
            for result in results:
                action = result.action.value
                action_counts[action] = action_counts.get(action, 0) + 1
            
            return {
                "total_rules": total_rules,
                "successful_rules": successful_rules,
                "success_rate": successful_rules / total_rules if total_rules > 0 else 0,
                "total_rows_affected": total_rows_affected,
                "total_execution_time": total_execution_time,
                "action_counts": action_counts,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get remediation summary: {str(e)}")
            return {"error": str(e)}
    
    def get_remediation_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get remediation history"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            recent_history = [
                entry for entry in self.remediation_history
                if entry["timestamp"] >= cutoff_date
            ]
            
            return recent_history
            
        except Exception as e:
            logger.error(f"Failed to get remediation history: {str(e)}")
            return []
    
    def export_remediation_results(self, results: List[RemediationResult], format: str = "json") -> str:
        """Export remediation results"""
        try:
            export_data = {
                "remediation_results": [
                    {
                        "rule_id": result.rule_id,
                        "action": result.action.value,
                        "rows_affected": result.rows_affected,
                        "columns_affected": result.columns_affected,
                        "success": result.success,
                        "message": result.message,
                        "execution_time": result.execution_time
                    }
                    for result in results
                ],
                "summary": self.get_remediation_summary(results),
                "exported_at": datetime.now().isoformat()
            }
            
            if format.lower() == "json":
                return json.dumps(export_data, indent=2, default=str)
            else:
                return str(export_data)
                
        except Exception as e:
            logger.error(f"Failed to export remediation results: {str(e)}")
            return f"Export failed: {str(e)}"


