"""
Business Rule Engine
Business rule engine for complex event processing
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import json
import re
from collections import defaultdict

from ..sourcing.event_store import Event, EventType

logger = logging.getLogger(__name__)

class RuleType(Enum):
    """Rule types"""
    CONDITIONAL = "conditional"
    TEMPORAL = "temporal"
    AGGREGATION = "aggregation"
    THRESHOLD = "threshold"
    SEQUENCE = "sequence"
    CUSTOM = "custom"

class RuleStatus(Enum):
    """Rule status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    ERROR = "error"

class RulePriority(Enum):
    """Rule priority"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class BusinessRule:
    """Business rule definition"""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    event_types: List[EventType]
    conditions: Dict[str, Any]
    actions: List[Callable[[List[Event], Dict[str, Any]], None]]
    priority: RulePriority = RulePriority.MEDIUM
    status: RuleStatus = RuleStatus.ACTIVE
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None
    execution_count: int = 0
    last_executed: Optional[datetime] = None

@dataclass
class RuleExecution:
    """Rule execution result"""
    rule_id: str
    execution_id: str
    status: str
    events_processed: List[Event]
    actions_executed: List[str]
    execution_time: float
    timestamp: datetime
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class RuleContext:
    """Rule execution context"""
    context_id: str
    rule_id: str
    events: List[Event]
    variables: Dict[str, Any]
    created_at: datetime
    expires_at: datetime

class BusinessRuleEngine:
    """
    Business rule engine for complex event processing
    """
    
    def __init__(self):
        self.rules: Dict[str, BusinessRule] = {}
        self.active_contexts: Dict[str, RuleContext] = {}
        self.execution_history: List[RuleExecution] = []
        self.rule_lock = threading.RLock()
        
        # Rule execution statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "rules_by_type": defaultdict(int),
            "execution_times": []
        }
        
        logger.info("BusinessRuleEngine initialized")
    
    def register_rule(self, rule: BusinessRule) -> bool:
        """Register a business rule"""
        try:
            with self.rule_lock:
                rule.created_at = datetime.now()
                rule.updated_at = datetime.now()
                self.rules[rule.rule_id] = rule
                logger.info(f"Registered business rule: {rule.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register business rule {rule.rule_id}: {str(e)}")
            return False
    
    def unregister_rule(self, rule_id: str) -> bool:
        """Unregister a business rule"""
        try:
            with self.rule_lock:
                if rule_id in self.rules:
                    del self.rules[rule_id]
                    logger.info(f"Unregistered business rule: {rule_id}")
                    return True
                else:
                    logger.warning(f"Business rule {rule_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to unregister business rule {rule_id}: {str(e)}")
            return False
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update a business rule"""
        try:
            with self.rule_lock:
                if rule_id not in self.rules:
                    return False
                
                rule = self.rules[rule_id]
                
                # Update allowed fields
                allowed_fields = ["name", "description", "conditions", "actions", "priority", "status", "enabled"]
                for field, value in updates.items():
                    if field in allowed_fields:
                        setattr(rule, field, value)
                
                rule.updated_at = datetime.now()
                logger.info(f"Updated business rule: {rule_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update business rule {rule_id}: {str(e)}")
            return False
    
    def process_event(self, event: Event) -> List[RuleExecution]:
        """Process an event against all applicable rules"""
        try:
            executions = []
            
            # Find applicable rules
            applicable_rules = self._find_applicable_rules(event)
            
            # Execute rules in priority order
            for rule in sorted(applicable_rules, key=lambda r: r.priority.value, reverse=True):
                if rule.enabled and rule.status == RuleStatus.ACTIVE:
                    execution = self._execute_rule(rule, [event])
                    if execution:
                        executions.append(execution)
            
            logger.debug(f"Processed event {event.event_id} against {len(executions)} rules")
            return executions
            
        except Exception as e:
            logger.error(f"Failed to process event {event.event_id}: {str(e)}")
            return []
    
    def process_events(self, events: List[Event]) -> List[RuleExecution]:
        """Process multiple events against applicable rules"""
        try:
            executions = []
            
            # Group events by type for efficiency
            events_by_type = defaultdict(list)
            for event in events:
                events_by_type[event.event_type].append(event)
            
            # Process each event type
            for event_type, type_events in events_by_type.items():
                applicable_rules = [rule for rule in self.rules.values() 
                                 if event_type in rule.event_types and rule.enabled and rule.status == RuleStatus.ACTIVE]
                
                for rule in sorted(applicable_rules, key=lambda r: r.priority.value, reverse=True):
                    execution = self._execute_rule(rule, type_events)
                    if execution:
                        executions.append(execution)
            
            logger.debug(f"Processed {len(events)} events against {len(executions)} rules")
            return executions
            
        except Exception as e:
            logger.error(f"Failed to process events: {str(e)}")
            return []
    
    def _find_applicable_rules(self, event: Event) -> List[BusinessRule]:
        """Find rules applicable to an event"""
        try:
            applicable_rules = []
            
            for rule in self.rules.values():
                if (rule.enabled and 
                    rule.status == RuleStatus.ACTIVE and 
                    event.event_type in rule.event_types):
                    applicable_rules.append(rule)
            
            return applicable_rules
            
        except Exception as e:
            logger.error(f"Failed to find applicable rules: {str(e)}")
            return []
    
    def _execute_rule(self, rule: BusinessRule, events: List[Event]) -> Optional[RuleExecution]:
        """Execute a business rule"""
        try:
            execution_id = f"{rule.rule_id}_{int(datetime.now().timestamp())}"
            start_time = datetime.now()
            
            # Check if rule conditions are met
            if not self._evaluate_conditions(rule, events):
                return None
            
            # Execute rule actions
            actions_executed = []
            error_message = None
            
            try:
                for action in rule.actions:
                    action_name = action.__name__ if hasattr(action, '__name__') else str(action)
                    action(events, {"rule_id": rule.rule_id, "execution_id": execution_id})
                    actions_executed.append(action_name)
                
                # Update rule statistics
                rule.execution_count += 1
                rule.last_executed = datetime.now()
                
                # Update engine statistics
                self.stats["total_executions"] += 1
                self.stats["successful_executions"] += 1
                self.stats["rules_by_type"][rule.rule_type.value] += 1
                
            except Exception as e:
                error_message = str(e)
                self.stats["failed_executions"] += 1
                logger.error(f"Failed to execute rule {rule.rule_id}: {str(e)}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            self.stats["execution_times"].append(execution_time)
            
            # Create execution result
            execution = RuleExecution(
                rule_id=rule.rule_id,
                execution_id=execution_id,
                status="success" if not error_message else "failed",
                events_processed=events,
                actions_executed=actions_executed,
                execution_time=execution_time,
                timestamp=datetime.now(),
                error_message=error_message
            )
            
            # Store execution history
            with self.rule_lock:
                self.execution_history.append(execution)
                # Keep only last 1000 executions
                if len(self.execution_history) > 1000:
                    self.execution_history = self.execution_history[-1000:]
            
            logger.debug(f"Executed rule {rule.rule_id} in {execution_time:.3f}s")
            return execution
            
        except Exception as e:
            logger.error(f"Failed to execute rule {rule.rule_id}: {str(e)}")
            return None
    
    def _evaluate_conditions(self, rule: BusinessRule, events: List[Event]) -> bool:
        """Evaluate rule conditions"""
        try:
            if rule.rule_type == RuleType.CONDITIONAL:
                return self._evaluate_conditional_conditions(rule, events)
            elif rule.rule_type == RuleType.TEMPORAL:
                return self._evaluate_temporal_conditions(rule, events)
            elif rule.rule_type == RuleType.AGGREGATION:
                return self._evaluate_aggregation_conditions(rule, events)
            elif rule.rule_type == RuleType.THRESHOLD:
                return self._evaluate_threshold_conditions(rule, events)
            elif rule.rule_type == RuleType.SEQUENCE:
                return self._evaluate_sequence_conditions(rule, events)
            elif rule.rule_type == RuleType.CUSTOM:
                return self._evaluate_custom_conditions(rule, events)
            else:
                logger.warning(f"Unknown rule type: {rule.rule_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to evaluate conditions for rule {rule.rule_id}: {str(e)}")
            return False
    
    def _evaluate_conditional_conditions(self, rule: BusinessRule, events: List[Event]) -> bool:
        """Evaluate conditional rule conditions"""
        try:
            conditions = rule.conditions.get("conditions", [])
            if not conditions:
                return True
            
            # Evaluate each condition
            for condition in conditions:
                if not self._evaluate_single_condition(condition, events):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to evaluate conditional conditions: {str(e)}")
            return False
    
    def _evaluate_temporal_conditions(self, rule: BusinessRule, events: List[Event]) -> bool:
        """Evaluate temporal rule conditions"""
        try:
            conditions = rule.conditions
            
            # Check time window
            time_window = conditions.get("time_window_seconds", 3600)
            if events:
                event_times = [event.timestamp for event in events]
                time_span = (max(event_times) - min(event_times)).total_seconds()
                if time_span > time_window:
                    return False
            
            # Check frequency
            min_frequency = conditions.get("min_frequency", 1)
            max_frequency = conditions.get("max_frequency", float('inf'))
            event_count = len(events)
            
            if not (min_frequency <= event_count <= max_frequency):
                return False
            
            # Check time patterns
            time_patterns = conditions.get("time_patterns", [])
            for pattern in time_patterns:
                if not self._evaluate_time_pattern(pattern, events):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to evaluate temporal conditions: {str(e)}")
            return False
    
    def _evaluate_aggregation_conditions(self, rule: BusinessRule, events: List[Event]) -> bool:
        """Evaluate aggregation rule conditions"""
        try:
            conditions = rule.conditions
            aggregation_field = conditions.get("aggregation_field")
            aggregation_type = conditions.get("aggregation_type", "sum")
            threshold = conditions.get("threshold")
            
            if not aggregation_field or threshold is None:
                return False
            
            # Extract values
            values = []
            for event in events:
                if aggregation_field in event.data:
                    value = event.data[aggregation_field]
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            if not values:
                return False
            
            # Calculate aggregation
            if aggregation_type == "sum":
                result = sum(values)
            elif aggregation_type == "avg":
                result = sum(values) / len(values)
            elif aggregation_type == "min":
                result = min(values)
            elif aggregation_type == "max":
                result = max(values)
            elif aggregation_type == "count":
                result = len(values)
            else:
                logger.warning(f"Unknown aggregation type: {aggregation_type}")
                return False
            
            # Check threshold
            operator = conditions.get("operator", "gte")
            return self._compare_values(result, operator, threshold)
            
        except Exception as e:
            logger.error(f"Failed to evaluate aggregation conditions: {str(e)}")
            return False
    
    def _evaluate_threshold_conditions(self, rule: BusinessRule, events: List[Event]) -> bool:
        """Evaluate threshold rule conditions"""
        try:
            conditions = rule.conditions
            threshold_field = conditions.get("threshold_field")
            threshold_value = conditions.get("threshold_value")
            operator = conditions.get("operator", "gte")
            
            if not threshold_field or threshold_value is None:
                return False
            
            # Check if any event exceeds threshold
            for event in events:
                if threshold_field in event.data:
                    value = event.data[threshold_field]
                    if isinstance(value, (int, float)):
                        if self._compare_values(value, operator, threshold_value):
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate threshold conditions: {str(e)}")
            return False
    
    def _evaluate_sequence_conditions(self, rule: BusinessRule, events: List[Event]) -> bool:
        """Evaluate sequence rule conditions"""
        try:
            conditions = rule.conditions
            required_sequence = conditions.get("sequence", [])
            
            if not required_sequence:
                return True
            
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda e: e.timestamp)
            
            # Check if events match required sequence
            sequence_index = 0
            for event in sorted_events:
                if sequence_index < len(required_sequence):
                    if event.event_type.value == required_sequence[sequence_index]:
                        sequence_index += 1
            
            return sequence_index == len(required_sequence)
            
        except Exception as e:
            logger.error(f"Failed to evaluate sequence conditions: {str(e)}")
            return False
    
    def _evaluate_custom_conditions(self, rule: BusinessRule, events: List[Event]) -> bool:
        """Evaluate custom rule conditions"""
        try:
            custom_evaluator = rule.conditions.get("custom_evaluator")
            if custom_evaluator and callable(custom_evaluator):
                return custom_evaluator(events, rule.conditions)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to evaluate custom conditions: {str(e)}")
            return False
    
    def _evaluate_single_condition(self, condition: Dict[str, Any], events: List[Event]) -> bool:
        """Evaluate a single condition"""
        try:
            field = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")
            
            if not field or not operator:
                return True
            
            # Check if any event matches condition
            for event in events:
                event_value = event.data.get(field) or event.metadata.get(field)
                if event_value is not None:
                    if self._compare_values(event_value, operator, value):
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate single condition: {str(e)}")
            return False
    
    def _evaluate_time_pattern(self, pattern: Dict[str, Any], events: List[Event]) -> bool:
        """Evaluate time pattern"""
        try:
            pattern_type = pattern.get("type")
            
            if pattern_type == "time_of_day":
                target_hours = pattern.get("hours", [])
                for event in events:
                    if event.timestamp.hour in target_hours:
                        return True
                return False
            
            elif pattern_type == "day_of_week":
                target_days = pattern.get("days", [])
                for event in events:
                    if event.timestamp.weekday() in target_days:
                        return True
                return False
            
            elif pattern_type == "time_interval":
                interval_minutes = pattern.get("interval_minutes", 60)
                if len(events) < 2:
                    return True
                
                sorted_events = sorted(events, key=lambda e: e.timestamp)
                for i in range(len(sorted_events) - 1):
                    time_diff = (sorted_events[i+1].timestamp - sorted_events[i].timestamp).total_seconds() / 60
                    if abs(time_diff - interval_minutes) > 5:  # 5 minute tolerance
                        return False
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to evaluate time pattern: {str(e)}")
            return False
    
    def _compare_values(self, actual_value: Any, operator: str, expected_value: Any) -> bool:
        """Compare values using operator"""
        try:
            if operator == "eq":
                return actual_value == expected_value
            elif operator == "ne":
                return actual_value != expected_value
            elif operator == "gt":
                return actual_value > expected_value
            elif operator == "gte":
                return actual_value >= expected_value
            elif operator == "lt":
                return actual_value < expected_value
            elif operator == "lte":
                return actual_value <= expected_value
            elif operator == "in":
                return actual_value in expected_value
            elif operator == "not_in":
                return actual_value not in expected_value
            elif operator == "contains":
                return expected_value in str(actual_value)
            elif operator == "regex":
                return bool(re.search(expected_value, str(actual_value)))
            else:
                logger.warning(f"Unknown operator: {operator}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to compare values: {str(e)}")
            return False
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get rule engine statistics"""
        try:
            with self.rule_lock:
                return {
                    "total_rules": len(self.rules),
                    "active_rules": sum(1 for rule in self.rules.values() if rule.enabled and rule.status == RuleStatus.ACTIVE),
                    "rules_by_type": dict(self.stats["rules_by_type"]),
                    "execution_stats": {
                        "total_executions": self.stats["total_executions"],
                        "successful_executions": self.stats["successful_executions"],
                        "failed_executions": self.stats["failed_executions"],
                        "success_rate": (self.stats["successful_executions"] / self.stats["total_executions"] * 100) if self.stats["total_executions"] > 0 else 0
                    },
                    "performance_stats": {
                        "avg_execution_time": sum(self.stats["execution_times"]) / len(self.stats["execution_times"]) if self.stats["execution_times"] else 0,
                        "max_execution_time": max(self.stats["execution_times"]) if self.stats["execution_times"] else 0,
                        "min_execution_time": min(self.stats["execution_times"]) if self.stats["execution_times"] else 0
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get rule statistics: {str(e)}")
            return {}
    
    def get_execution_history(self, limit: int = 100) -> List[RuleExecution]:
        """Get recent execution history"""
        try:
            with self.rule_lock:
                return self.execution_history[-limit:] if self.execution_history else []
                
        except Exception as e:
            logger.error(f"Failed to get execution history: {str(e)}")
            return []
    
    def get_rule_by_id(self, rule_id: str) -> Optional[BusinessRule]:
        """Get rule by ID"""
        try:
            return self.rules.get(rule_id)
            
        except Exception as e:
            logger.error(f"Failed to get rule {rule_id}: {str(e)}")
            return None
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[BusinessRule]:
        """Get rules by type"""
        try:
            return [rule for rule in self.rules.values() if rule.rule_type == rule_type]
            
        except Exception as e:
            logger.error(f"Failed to get rules by type {rule_type}: {str(e)}")
            return []
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule"""
        try:
            if rule_id in self.rules:
                self.rules[rule_id].enabled = True
                self.rules[rule_id].updated_at = datetime.now()
                logger.info(f"Enabled rule: {rule_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to enable rule {rule_id}: {str(e)}")
            return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule"""
        try:
            if rule_id in self.rules:
                self.rules[rule_id].enabled = False
                self.rules[rule_id].updated_at = datetime.now()
                logger.info(f"Disabled rule: {rule_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to disable rule {rule_id}: {str(e)}")
            return False
    
    def suspend_rule(self, rule_id: str) -> bool:
        """Suspend a rule"""
        try:
            if rule_id in self.rules:
                self.rules[rule_id].status = RuleStatus.SUSPENDED
                self.rules[rule_id].updated_at = datetime.now()
                logger.info(f"Suspended rule: {rule_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to suspend rule {rule_id}: {str(e)}")
            return False
    
    def activate_rule(self, rule_id: str) -> bool:
        """Activate a rule"""
        try:
            if rule_id in self.rules:
                self.rules[rule_id].status = RuleStatus.ACTIVE
                self.rules[rule_id].updated_at = datetime.now()
                logger.info(f"Activated rule: {rule_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to activate rule {rule_id}: {str(e)}")
            return False
    
    def clear_execution_history(self) -> bool:
        """Clear execution history"""
        try:
            with self.rule_lock:
                self.execution_history.clear()
                logger.info("Cleared execution history")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear execution history: {str(e)}")
            return False
