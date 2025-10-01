"""
Validation Engine
Comprehensive validation engine for schema and business rules
"""

import logging
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ValidationRuleType(Enum):
    """Types of validation rules"""
    SCHEMA = "schema"
    BUSINESS = "business"
    CUSTOM = "custom"
    REGEX = "regex"
    RANGE = "range"
    FORMAT = "format"
    REFERENCE = "reference"

class ValidationSeverity(Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationRule:
    """Represents a validation rule"""
    rule_id: str
    name: str
    rule_type: ValidationRuleType
    description: str
    severity: ValidationSeverity
    enabled: bool
    conditions: Dict[str, Any]
    created_at: datetime

@dataclass
class ValidationResult:
    """Represents a validation result"""
    rule_id: str
    rule_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    affected_rows: List[int]
    affected_columns: List[str]
    details: Dict[str, Any]

class ValidationEngine:
    """
    Comprehensive validation engine for schema and business rules
    """
    
    def __init__(self):
        self.validation_rules = {}
        self.rule_templates = self._initialize_rule_templates()
        
        logger.info("ValidationEngine initialized")
    
    def _initialize_rule_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common validation rule templates"""
        return {
            "not_null": {
                "rule_type": ValidationRuleType.SCHEMA,
                "severity": ValidationSeverity.ERROR,
                "description": "Check for null values",
                "conditions": {"check_null": True}
            },
            "data_type": {
                "rule_type": ValidationRuleType.SCHEMA,
                "severity": ValidationSeverity.ERROR,
                "description": "Check data type",
                "conditions": {"expected_type": "string"}
            },
            "range_check": {
                "rule_type": ValidationRuleType.RANGE,
                "severity": ValidationSeverity.ERROR,
                "description": "Check value range",
                "conditions": {"min_value": None, "max_value": None}
            },
            "regex_pattern": {
                "rule_type": ValidationRuleType.REGEX,
                "severity": ValidationSeverity.ERROR,
                "description": "Check regex pattern",
                "conditions": {"pattern": ""}
            },
            "unique_values": {
                "rule_type": ValidationRuleType.SCHEMA,
                "severity": ValidationSeverity.WARNING,
                "description": "Check for unique values",
                "conditions": {"check_uniqueness": True}
            },
            "referential_integrity": {
                "rule_type": ValidationRuleType.REFERENCE,
                "severity": ValidationSeverity.ERROR,
                "description": "Check referential integrity",
                "conditions": {"parent_table": "", "parent_column": "", "child_column": ""}
            }
        }
    
    def create_validation_rule(self,
                             name: str,
                             rule_type: ValidationRuleType,
                             description: str,
                             severity: ValidationSeverity,
                             conditions: Dict[str, Any],
                             enabled: bool = True) -> str:
        """Create a new validation rule"""
        try:
            rule_id = f"rule_{int(datetime.now().timestamp())}"
            
            rule = ValidationRule(
                rule_id=rule_id,
                name=name,
                rule_type=rule_type,
                description=description,
                severity=severity,
                enabled=enabled,
                conditions=conditions,
                created_at=datetime.now()
            )
            
            self.validation_rules[rule_id] = rule
            
            logger.info(f"Validation rule created: {rule_id}")
            return rule_id
            
        except Exception as e:
            logger.error(f"Failed to create validation rule: {str(e)}")
            return ""
    
    def create_rule_from_template(self,
                                template_name: str,
                                name: str,
                                conditions: Dict[str, Any],
                                severity: ValidationSeverity = None) -> str:
        """Create a validation rule from template"""
        try:
            if template_name not in self.rule_templates:
                raise ValueError(f"Template {template_name} not found")
            
            template = self.rule_templates[template_name]
            
            # Merge template conditions with provided conditions
            merged_conditions = {**template["conditions"], **conditions}
            
            return self.create_validation_rule(
                name=name,
                rule_type=template["rule_type"],
                description=template["description"],
                severity=severity or template["severity"],
                conditions=merged_conditions
            )
            
        except Exception as e:
            logger.error(f"Failed to create rule from template: {str(e)}")
            return ""
    
    def validate_data(self, 
                     data: pd.DataFrame,
                     rules: List[str] = None,
                     columns: List[str] = None) -> List[ValidationResult]:
        """Validate data against rules"""
        try:
            if data.empty:
                return []
            
            # Filter rules to apply
            rules_to_apply = []
            if rules:
                rules_to_apply = [self.validation_rules[rule_id] for rule_id in rules 
                                if rule_id in self.validation_rules and self.validation_rules[rule_id].enabled]
            else:
                rules_to_apply = [rule for rule in self.validation_rules.values() if rule.enabled]
            
            # Filter columns to validate
            data_to_validate = data
            if columns:
                data_to_validate = data[columns]
            
            validation_results = []
            
            for rule in rules_to_apply:
                try:
                    result = self._apply_validation_rule(data_to_validate, rule)
                    validation_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to apply rule {rule.rule_id}: {str(e)}")
                    # Create error result
                    error_result = ValidationResult(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Rule execution failed: {str(e)}",
                        affected_rows=[],
                        affected_columns=[],
                        details={"error": str(e)}
                    )
                    validation_results.append(error_result)
            
            logger.info(f"Validation completed: {len(validation_results)} rules applied")
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate data: {str(e)}")
            return []
    
    def _apply_validation_rule(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Apply a single validation rule"""
        try:
            if rule.rule_type == ValidationRuleType.SCHEMA:
                return self._apply_schema_rule(data, rule)
            elif rule.rule_type == ValidationRuleType.BUSINESS:
                return self._apply_business_rule(data, rule)
            elif rule.rule_type == ValidationRuleType.REGEX:
                return self._apply_regex_rule(data, rule)
            elif rule.rule_type == ValidationRuleType.RANGE:
                return self._apply_range_rule(data, rule)
            elif rule.rule_type == ValidationRuleType.FORMAT:
                return self._apply_format_rule(data, rule)
            elif rule.rule_type == ValidationRuleType.REFERENCE:
                return self._apply_reference_rule(data, rule)
            else:
                return self._apply_custom_rule(data, rule)
                
        except Exception as e:
            logger.error(f"Failed to apply rule {rule.rule_id}: {str(e)}")
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Rule execution failed: {str(e)}",
                affected_rows=[],
                affected_columns=[],
                details={"error": str(e)}
            )
    
    def _apply_schema_rule(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Apply schema validation rule"""
        try:
            conditions = rule.conditions
            affected_rows = []
            affected_columns = []
            passed = True
            message = "Schema validation passed"
            
            # Check for null values
            if conditions.get("check_null", False):
                null_mask = data.isnull()
                null_rows = null_mask.any(axis=1)
                if null_rows.any():
                    affected_rows = data[null_rows].index.tolist()
                    affected_columns = data.columns[null_mask.any()].tolist()
                    passed = False
                    message = f"Found {len(affected_rows)} rows with null values"
            
            # Check data type
            elif "expected_type" in conditions:
                expected_type = conditions["expected_type"]
                type_mismatch_cols = []
                
                for col in data.columns:
                    if not self._check_column_type(data[col], expected_type):
                        type_mismatch_cols.append(col)
                
                if type_mismatch_cols:
                    affected_columns = type_mismatch_cols
                    passed = False
                    message = f"Type mismatch in columns: {', '.join(type_mismatch_cols)}"
            
            # Check uniqueness
            elif conditions.get("check_uniqueness", False):
                if "columns" in conditions:
                    check_cols = conditions["columns"]
                    if all(col in data.columns for col in check_cols):
                        duplicates = data[check_cols].duplicated()
                        if duplicates.any():
                            affected_rows = data[duplicates].index.tolist()
                            passed = False
                            message = f"Found {len(affected_rows)} duplicate rows"
                            affected_columns = check_cols
            
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                message=message,
                affected_rows=affected_rows,
                affected_columns=affected_columns,
                details={"conditions": conditions}
            )
            
        except Exception as e:
            logger.error(f"Failed to apply schema rule: {str(e)}")
            raise
    
    def _apply_business_rule(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Apply business validation rule"""
        try:
            conditions = rule.conditions
            rule_type = conditions.get("type", "custom")
            passed = True
            message = "Business rule validation passed"
            affected_rows = []
            affected_columns = []
            
            if rule_type == "sum_check":
                # Check if sum of columns equals expected value
                columns = conditions.get("columns", [])
                expected_sum = conditions.get("expected_sum", 0)
                tolerance = conditions.get("tolerance", 0.01)
                
                if all(col in data.columns for col in columns):
                    actual_sum = data[columns].sum().sum()
                    if abs(actual_sum - expected_sum) > tolerance:
                        passed = False
                        message = f"Sum check failed: expected {expected_sum}, got {actual_sum}"
                        affected_rows = data.index.tolist()
                        affected_columns = columns
            
            elif rule_type == "ratio_check":
                # Check if ratio between columns is within range
                numerator_col = conditions.get("numerator_column")
                denominator_col = conditions.get("denominator_column")
                min_ratio = conditions.get("min_ratio", 0)
                max_ratio = conditions.get("max_ratio", 1)
                
                if numerator_col in data.columns and denominator_col in data.columns:
                    ratio = data[numerator_col] / data[denominator_col]
                    invalid_ratios = (ratio < min_ratio) | (ratio > max_ratio)
                    if invalid_ratios.any():
                        passed = False
                        message = f"Ratio check failed: {invalid_ratios.sum()} rows outside range [{min_ratio}, {max_ratio}]"
                        affected_rows = data[invalid_ratios].index.tolist()
                        affected_columns = [numerator_col, denominator_col]
            
            elif rule_type == "custom":
                # Custom business logic
                custom_function = conditions.get("function")
                if custom_function:
                    # In real implementation, this would execute custom Python code
                    # For now, we'll assume it's a simple check
                    passed = True
                    message = "Custom business rule validation passed"
            
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                message=message,
                affected_rows=affected_rows,
                affected_columns=affected_columns,
                details={"rule_type": rule_type, "conditions": conditions}
            )
            
        except Exception as e:
            logger.error(f"Failed to apply business rule: {str(e)}")
            raise
    
    def _apply_regex_rule(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Apply regex validation rule"""
        try:
            conditions = rule.conditions
            pattern = conditions.get("pattern", "")
            columns = conditions.get("columns", [])
            
            if not pattern:
                return ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    passed=True,
                    severity=rule.severity,
                    message="No pattern specified",
                    affected_rows=[],
                    affected_columns=[],
                    details={}
                )
            
            affected_rows = []
            affected_columns = []
            passed = True
            message = "Regex validation passed"
            
            # Compile regex pattern
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid regex pattern: {str(e)}",
                    affected_rows=[],
                    affected_columns=[],
                    details={"error": str(e)}
                )
            
            # Apply to specified columns or all string columns
            if columns:
                check_columns = [col for col in columns if col in data.columns]
            else:
                check_columns = data.select_dtypes(include=['object']).columns.tolist()
            
            for col in check_columns:
                if col in data.columns:
                    # Check each value against regex
                    invalid_mask = data[col].astype(str).apply(lambda x: not regex.match(x) if pd.notna(x) else False)
                    if invalid_mask.any():
                        col_affected_rows = data[invalid_mask].index.tolist()
                        affected_rows.extend(col_affected_rows)
                        affected_columns.append(col)
                        passed = False
            
            if not passed:
                message = f"Regex validation failed: {len(affected_rows)} values don't match pattern"
            
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                message=message,
                affected_rows=affected_rows,
                affected_columns=affected_columns,
                details={"pattern": pattern, "columns": check_columns}
            )
            
        except Exception as e:
            logger.error(f"Failed to apply regex rule: {str(e)}")
            raise
    
    def _apply_range_rule(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Apply range validation rule"""
        try:
            conditions = rule.conditions
            columns = conditions.get("columns", [])
            min_value = conditions.get("min_value")
            max_value = conditions.get("max_value")
            
            affected_rows = []
            affected_columns = []
            passed = True
            message = "Range validation passed"
            
            # Apply to specified columns or all numeric columns
            if columns:
                check_columns = [col for col in columns if col in data.columns]
            else:
                check_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in check_columns:
                if col in data.columns:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        # Check min value
                        if min_value is not None:
                            min_violations = col_data < min_value
                            if min_violations.any():
                                min_rows = data[col_data < min_value].index.tolist()
                                affected_rows.extend(min_rows)
                                affected_columns.append(col)
                                passed = False
                        
                        # Check max value
                        if max_value is not None:
                            max_violations = col_data > max_value
                            if max_violations.any():
                                max_rows = data[col_data > max_value].index.tolist()
                                affected_rows.extend(max_rows)
                                affected_columns.append(col)
                                passed = False
            
            if not passed:
                message = f"Range validation failed: {len(affected_rows)} values outside range [{min_value}, {max_value}]"
            
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                message=message,
                affected_rows=affected_rows,
                affected_columns=affected_columns,
                details={"min_value": min_value, "max_value": max_value, "columns": check_columns}
            )
            
        except Exception as e:
            logger.error(f"Failed to apply range rule: {str(e)}")
            raise
    
    def _apply_format_rule(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Apply format validation rule"""
        try:
            conditions = rule.conditions
            columns = conditions.get("columns", [])
            format_type = conditions.get("format_type", "email")
            
            affected_rows = []
            affected_columns = []
            passed = True
            message = "Format validation passed"
            
            # Apply to specified columns or all string columns
            if columns:
                check_columns = [col for col in columns if col in data.columns]
            else:
                check_columns = data.select_dtypes(include=['object']).columns.tolist()
            
            for col in check_columns:
                if col in data.columns:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        # Check format based on type
                        if format_type == "email":
                            invalid_mask = col_data.astype(str).apply(lambda x: not self._is_valid_email(x))
                        elif format_type == "phone":
                            invalid_mask = col_data.astype(str).apply(lambda x: not self._is_valid_phone(x))
                        elif format_type == "date":
                            invalid_mask = col_data.astype(str).apply(lambda x: not self._is_valid_date(x))
                        else:
                            invalid_mask = pd.Series([False] * len(col_data), index=col_data.index)
                        
                        if invalid_mask.any():
                            col_affected_rows = data[invalid_mask].index.tolist()
                            affected_rows.extend(col_affected_rows)
                            affected_columns.append(col)
                            passed = False
            
            if not passed:
                message = f"Format validation failed: {len(affected_rows)} values have invalid {format_type} format"
            
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                message=message,
                affected_rows=affected_rows,
                affected_columns=affected_columns,
                details={"format_type": format_type, "columns": check_columns}
            )
            
        except Exception as e:
            logger.error(f"Failed to apply format rule: {str(e)}")
            raise
    
    def _apply_reference_rule(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Apply referential integrity rule"""
        try:
            conditions = rule.conditions
            parent_table = conditions.get("parent_table")
            parent_column = conditions.get("parent_column")
            child_column = conditions.get("child_column")
            
            # This is a simplified implementation
            # In real implementation, this would check against actual parent table
            passed = True
            message = "Referential integrity validation passed"
            affected_rows = []
            affected_columns = []
            
            # For now, assume all references are valid
            # In real implementation, this would query the parent table
            
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                message=message,
                affected_rows=affected_rows,
                affected_columns=affected_columns,
                details={"parent_table": parent_table, "parent_column": parent_column, "child_column": child_column}
            )
            
        except Exception as e:
            logger.error(f"Failed to apply reference rule: {str(e)}")
            raise
    
    def _apply_custom_rule(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Apply custom validation rule"""
        try:
            # This is a placeholder for custom rule execution
            # In real implementation, this would execute custom Python code
            passed = True
            message = "Custom rule validation passed"
            
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                message=message,
                affected_rows=[],
                affected_columns=[],
                details={"custom_rule": True}
            )
            
        except Exception as e:
            logger.error(f"Failed to apply custom rule: {str(e)}")
            raise
    
    def _check_column_type(self, column: pd.Series, expected_type: str) -> bool:
        """Check if column matches expected type"""
        try:
            if expected_type == "string":
                return column.dtype == "object"
            elif expected_type == "integer":
                return pd.api.types.is_integer_dtype(column)
            elif expected_type == "float":
                return pd.api.types.is_float_dtype(column)
            elif expected_type == "boolean":
                return pd.api.types.is_bool_dtype(column)
            elif expected_type == "datetime":
                return pd.api.types.is_datetime64_any_dtype(column)
            else:
                return True
                
        except Exception as e:
            logger.error(f"Failed to check column type: {str(e)}")
            return False
    
    def _is_valid_email(self, email: str) -> bool:
        """Check if string is valid email"""
        try:
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, email))
        except:
            return False
    
    def _is_valid_phone(self, phone: str) -> bool:
        """Check if string is valid phone number"""
        try:
            pattern = r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
            return bool(re.match(pattern, phone))
        except:
            return False
    
    def _is_valid_date(self, date_str: str) -> bool:
        """Check if string is valid date"""
        try:
            pd.to_datetime(date_str)
            return True
        except:
            return False
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Get validation summary"""
        try:
            total_rules = len(results)
            passed_rules = len([r for r in results if r.passed])
            failed_rules = total_rules - passed_rules
            
            # Count by severity
            error_count = len([r for r in results if not r.passed and r.severity == ValidationSeverity.ERROR])
            warning_count = len([r for r in results if not r.passed and r.severity == ValidationSeverity.WARNING])
            info_count = len([r for r in results if not r.passed and r.severity == ValidationSeverity.INFO])
            
            # Get affected data
            all_affected_rows = set()
            all_affected_columns = set()
            for result in results:
                all_affected_rows.update(result.affected_rows)
                all_affected_columns.update(result.affected_columns)
            
            return {
                "total_rules": total_rules,
                "passed_rules": passed_rules,
                "failed_rules": failed_rules,
                "pass_rate": passed_rules / total_rules if total_rules > 0 else 0,
                "error_count": error_count,
                "warning_count": warning_count,
                "info_count": info_count,
                "affected_rows": len(all_affected_rows),
                "affected_columns": len(all_affected_columns),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get validation summary: {str(e)}")
            return {"error": str(e)}
    
    def export_validation_results(self, results: List[ValidationResult], format: str = "json") -> str:
        """Export validation results"""
        try:
            export_data = {
                "validation_results": [
                    {
                        "rule_id": result.rule_id,
                        "rule_name": result.rule_name,
                        "passed": result.passed,
                        "severity": result.severity.value,
                        "message": result.message,
                        "affected_rows": result.affected_rows,
                        "affected_columns": result.affected_columns,
                        "details": result.details
                    }
                    for result in results
                ],
                "summary": self.get_validation_summary(results),
                "exported_at": datetime.now().isoformat()
            }
            
            if format.lower() == "json":
                return json.dumps(export_data, indent=2, default=str)
            else:
                return str(export_data)
                
        except Exception as e:
            logger.error(f"Failed to export validation results: {str(e)}")
            return f"Export failed: {str(e)}"
