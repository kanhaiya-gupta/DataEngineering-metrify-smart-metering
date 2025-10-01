"""
Event Correlator
Event correlation and pattern detection for complex event processing
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import time
from collections import defaultdict, deque

from ..sourcing.event_store import Event, EventType

logger = logging.getLogger(__name__)

class CorrelationType(Enum):
    """Correlation types"""
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    SPATIAL = "spatial"
    LOGICAL = "logical"
    SEQUENTIAL = "sequential"

class CorrelationStatus(Enum):
    """Correlation status"""
    PENDING = "pending"
    MATCHED = "matched"
    EXPIRED = "expired"
    FAILED = "failed"

@dataclass
class CorrelationRule:
    """Event correlation rule"""
    rule_id: str
    name: str
    correlation_type: CorrelationType
    event_types: List[EventType]
    time_window: timedelta
    conditions: Dict[str, Any]
    action: Callable[[List[Event]], None]
    priority: int = 0
    enabled: bool = True
    created_at: datetime = None

@dataclass
class CorrelationContext:
    """Correlation context for tracking related events"""
    context_id: str
    rule_id: str
    events: List[Event]
    created_at: datetime
    expires_at: datetime
    status: CorrelationStatus = CorrelationStatus.PENDING
    metadata: Dict[str, Any] = None

@dataclass
class CorrelationResult:
    """Correlation result"""
    rule_id: str
    context_id: str
    matched_events: List[Event]
    correlation_type: CorrelationType
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

class EventCorrelator:
    """
    Event correlator for complex event processing
    """
    
    def __init__(self):
        self.rules: Dict[str, CorrelationRule] = {}
        self.active_contexts: Dict[str, CorrelationContext] = {}
        self.event_buffer: deque = deque(maxlen=10000)  # Circular buffer
        self.correlator_lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_contexts, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("EventCorrelator initialized")
    
    def register_rule(self, rule: CorrelationRule) -> bool:
        """Register a correlation rule"""
        try:
            with self.correlator_lock:
                rule.created_at = datetime.now()
                self.rules[rule.rule_id] = rule
                logger.info(f"Registered correlation rule: {rule.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register correlation rule {rule.rule_id}: {str(e)}")
            return False
    
    def unregister_rule(self, rule_id: str) -> bool:
        """Unregister a correlation rule"""
        try:
            with self.correlator_lock:
                if rule_id in self.rules:
                    del self.rules[rule_id]
                    logger.info(f"Unregistered correlation rule: {rule_id}")
                    return True
                else:
                    logger.warning(f"Correlation rule {rule_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to unregister correlation rule {rule_id}: {str(e)}")
            return False
    
    def process_event(self, event: Event) -> List[CorrelationResult]:
        """Process an event for correlation"""
        try:
            results = []
            
            # Add event to buffer
            with self.correlator_lock:
                self.event_buffer.append(event)
            
            # Check against all enabled rules
            for rule in self.rules.values():
                if not rule.enabled:
                    continue
                
                if event.event_type in rule.event_types:
                    correlation_result = self._check_correlation(event, rule)
                    if correlation_result:
                        results.append(correlation_result)
            
            logger.debug(f"Processed event {event.event_id} for correlation, found {len(results)} matches")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process event {event.event_id} for correlation: {str(e)}")
            return []
    
    def _check_correlation(self, event: Event, rule: CorrelationRule) -> Optional[CorrelationResult]:
        """Check if event matches correlation rule"""
        try:
            if rule.correlation_type == CorrelationType.TEMPORAL:
                return self._check_temporal_correlation(event, rule)
            elif rule.correlation_type == CorrelationType.CAUSAL:
                return self._check_causal_correlation(event, rule)
            elif rule.correlation_type == CorrelationType.SPATIAL:
                return self._check_spatial_correlation(event, rule)
            elif rule.correlation_type == CorrelationType.LOGICAL:
                return self._check_logical_correlation(event, rule)
            elif rule.correlation_type == CorrelationType.SEQUENTIAL:
                return self._check_sequential_correlation(event, rule)
            else:
                logger.warning(f"Unknown correlation type: {rule.correlation_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to check correlation for rule {rule.rule_id}: {str(e)}")
            return None
    
    def _check_temporal_correlation(self, event: Event, rule: CorrelationRule) -> Optional[CorrelationResult]:
        """Check temporal correlation"""
        try:
            # Look for events within time window
            time_window_start = event.timestamp - rule.time_window
            
            # Find events in time window
            related_events = []
            for buffered_event in self.event_buffer:
                if (buffered_event.timestamp >= time_window_start and
                    buffered_event.timestamp <= event.timestamp and
                    buffered_event.event_type in rule.event_types and
                    buffered_event.event_id != event.event_id):
                    related_events.append(buffered_event)
            
            # Check conditions
            if self._check_conditions(event, related_events, rule.conditions):
                # Create or update correlation context
                context_id = self._get_or_create_context(event, rule, related_events)
                
                # Check if correlation is complete
                if self._is_correlation_complete(context_id, rule):
                    return self._complete_correlation(context_id, rule)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check temporal correlation: {str(e)}")
            return None
    
    def _check_causal_correlation(self, event: Event, rule: CorrelationRule) -> Optional[CorrelationResult]:
        """Check causal correlation"""
        try:
            # Look for events with matching correlation_id or causation_id
            related_events = []
            
            for buffered_event in self.event_buffer:
                if (buffered_event.event_id != event.event_id and
                    buffered_event.event_type in rule.event_types):
                    
                    # Check correlation IDs
                    if (event.correlation_id and 
                        event.correlation_id == buffered_event.correlation_id):
                        related_events.append(buffered_event)
                    elif (event.causation_id and 
                          event.causation_id == buffered_event.event_id):
                        related_events.append(buffered_event)
            
            # Check conditions
            if self._check_conditions(event, related_events, rule.conditions):
                context_id = self._get_or_create_context(event, rule, related_events)
                
                if self._is_correlation_complete(context_id, rule):
                    return self._complete_correlation(context_id, rule)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check causal correlation: {str(e)}")
            return None
    
    def _check_spatial_correlation(self, event: Event, rule: CorrelationRule) -> Optional[CorrelationResult]:
        """Check spatial correlation"""
        try:
            # Look for events with similar spatial attributes
            event_location = event.data.get("location") or event.metadata.get("location")
            if not event_location:
                return None
            
            related_events = []
            for buffered_event in self.event_buffer:
                if (buffered_event.event_id != event.event_id and
                    buffered_event.event_type in rule.event_types):
                    
                    buffered_location = (buffered_event.data.get("location") or 
                                       buffered_event.metadata.get("location"))
                    
                    if buffered_location and self._is_spatially_related(event_location, buffered_location):
                        related_events.append(buffered_event)
            
            # Check conditions
            if self._check_conditions(event, related_events, rule.conditions):
                context_id = self._get_or_create_context(event, rule, related_events)
                
                if self._is_correlation_complete(context_id, rule):
                    return self._complete_correlation(context_id, rule)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check spatial correlation: {str(e)}")
            return None
    
    def _check_logical_correlation(self, event: Event, rule: CorrelationRule) -> Optional[CorrelationResult]:
        """Check logical correlation based on event data"""
        try:
            # Look for events with matching logical attributes
            related_events = []
            
            for buffered_event in self.event_buffer:
                if (buffered_event.event_id != event.event_id and
                    buffered_event.event_type in rule.event_types):
                    
                    if self._is_logically_related(event, buffered_event, rule.conditions):
                        related_events.append(buffered_event)
            
            # Check conditions
            if self._check_conditions(event, related_events, rule.conditions):
                context_id = self._get_or_create_context(event, rule, related_events)
                
                if self._is_correlation_complete(context_id, rule):
                    return self._complete_correlation(context_id, rule)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check logical correlation: {str(e)}")
            return None
    
    def _check_sequential_correlation(self, event: Event, rule: CorrelationRule) -> Optional[CorrelationResult]:
        """Check sequential correlation"""
        try:
            # Look for events in sequence
            related_events = []
            
            # Get events in chronological order
            sorted_events = sorted(self.event_buffer, key=lambda e: e.timestamp)
            
            # Find sequence of events
            sequence_start = None
            for i, buffered_event in enumerate(sorted_events):
                if buffered_event.event_id == event.event_id:
                    sequence_start = i
                    break
            
            if sequence_start is not None:
                # Look for events before and after in sequence
                for i in range(max(0, sequence_start - 10), min(len(sorted_events), sequence_start + 10)):
                    if (i != sequence_start and 
                        sorted_events[i].event_type in rule.event_types):
                        related_events.append(sorted_events[i])
            
            # Check conditions
            if self._check_conditions(event, related_events, rule.conditions):
                context_id = self._get_or_create_context(event, rule, related_events)
                
                if self._is_correlation_complete(context_id, rule):
                    return self._complete_correlation(context_id, rule)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check sequential correlation: {str(e)}")
            return None
    
    def _check_conditions(self, event: Event, related_events: List[Event], conditions: Dict[str, Any]) -> bool:
        """Check if conditions are met for correlation"""
        try:
            if not conditions:
                return True
            
            # Check event count conditions
            if "min_events" in conditions:
                if len(related_events) < conditions["min_events"]:
                    return False
            
            if "max_events" in conditions:
                if len(related_events) > conditions["max_events"]:
                    return False
            
            # Check data conditions
            if "data_conditions" in conditions:
                for condition in conditions["data_conditions"]:
                    if not self._evaluate_data_condition(event, related_events, condition):
                        return False
            
            # Check time conditions
            if "time_conditions" in conditions:
                for condition in conditions["time_conditions"]:
                    if not self._evaluate_time_condition(event, related_events, condition):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check conditions: {str(e)}")
            return False
    
    def _evaluate_data_condition(self, event: Event, related_events: List[Event], condition: Dict[str, Any]) -> bool:
        """Evaluate data condition"""
        try:
            field = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")
            
            if not field or not operator:
                return True
            
            # Check event data
            event_value = event.data.get(field)
            if event_value is not None:
                if not self._compare_values(event_value, operator, value):
                    return False
            
            # Check related events
            for related_event in related_events:
                related_value = related_event.data.get(field)
                if related_value is not None:
                    if not self._compare_values(related_value, operator, value):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to evaluate data condition: {str(e)}")
            return False
    
    def _evaluate_time_condition(self, event: Event, related_events: List[Event], condition: Dict[str, Any]) -> bool:
        """Evaluate time condition"""
        try:
            condition_type = condition.get("type")
            
            if condition_type == "time_gap":
                max_gap = condition.get("max_gap_seconds", 3600)
                
                for related_event in related_events:
                    time_gap = abs((event.timestamp - related_event.timestamp).total_seconds())
                    if time_gap > max_gap:
                        return False
            
            elif condition_type == "time_order":
                # Check if events are in correct order
                sorted_events = sorted(related_events + [event], key=lambda e: e.timestamp)
                expected_order = condition.get("expected_order", [])
                
                for i, expected_type in enumerate(expected_order):
                    if i < len(sorted_events):
                        if sorted_events[i].event_type.value != expected_type:
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to evaluate time condition: {str(e)}")
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
            else:
                logger.warning(f"Unknown operator: {operator}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to compare values: {str(e)}")
            return False
    
    def _is_spatially_related(self, location1: Any, location2: Any) -> bool:
        """Check if two locations are spatially related"""
        try:
            # Simple string comparison - in real implementation, use geospatial libraries
            if isinstance(location1, str) and isinstance(location2, str):
                return location1.lower() == location2.lower()
            
            # If coordinates, calculate distance
            if (isinstance(location1, dict) and isinstance(location2, dict) and
                "lat" in location1 and "lon" in location1 and
                "lat" in location2 and "lon" in location2):
                
                # Simple distance calculation (not accurate for large distances)
                lat1, lon1 = location1["lat"], location1["lon"]
                lat2, lon2 = location2["lat"], location2["lon"]
                
                distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
                return distance < 0.01  # Within ~1km
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check spatial relation: {str(e)}")
            return False
    
    def _is_logically_related(self, event1: Event, event2: Event, conditions: Dict[str, Any]) -> bool:
        """Check if two events are logically related"""
        try:
            # Check if events share common attributes
            common_attributes = conditions.get("common_attributes", [])
            
            for attr in common_attributes:
                if (event1.data.get(attr) != event2.data.get(attr) and
                    event1.metadata.get(attr) != event2.metadata.get(attr)):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check logical relation: {str(e)}")
            return False
    
    def _get_or_create_context(self, event: Event, rule: CorrelationRule, related_events: List[Event]) -> str:
        """Get or create correlation context"""
        try:
            # Generate context ID based on rule and event attributes
            context_key = f"{rule.rule_id}_{event.aggregate_id}_{event.correlation_id or 'default'}"
            
            with self.correlator_lock:
                if context_key in self.active_contexts:
                    # Update existing context
                    context = self.active_contexts[context_key]
                    if event not in context.events:
                        context.events.append(event)
                    context.expires_at = event.timestamp + rule.time_window
                else:
                    # Create new context
                    context = CorrelationContext(
                        context_id=context_key,
                        rule_id=rule.rule_id,
                        events=[event] + related_events,
                        created_at=event.timestamp,
                        expires_at=event.timestamp + rule.time_window,
                        metadata={}
                    )
                    self.active_contexts[context_key] = context
            
            return context_key
            
        except Exception as e:
            logger.error(f"Failed to get or create context: {str(e)}")
            return ""
    
    def _is_correlation_complete(self, context_id: str, rule: CorrelationRule) -> bool:
        """Check if correlation is complete"""
        try:
            context = self.active_contexts.get(context_id)
            if not context:
                return False
            
            # Check if we have enough events
            min_events = rule.conditions.get("min_events", 1)
            if len(context.events) < min_events:
                return False
            
            # Check if we have all required event types
            required_types = set(rule.event_types)
            found_types = set(event.event_type for event in context.events)
            
            if not required_types.issubset(found_types):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check correlation completeness: {str(e)}")
            return False
    
    def _complete_correlation(self, context_id: str, rule: CorrelationRule) -> CorrelationResult:
        """Complete correlation and execute action"""
        try:
            context = self.active_contexts.get(context_id)
            if not context:
                return None
            
            # Create correlation result
            result = CorrelationResult(
                rule_id=rule.rule_id,
                context_id=context_id,
                matched_events=context.events.copy(),
                correlation_type=rule.correlation_type,
                confidence=1.0,  # Simple confidence calculation
                timestamp=datetime.now(),
                metadata=context.metadata
            )
            
            # Execute action
            try:
                rule.action(context.events)
                context.status = CorrelationStatus.MATCHED
            except Exception as e:
                logger.error(f"Failed to execute correlation action: {str(e)}")
                context.status = CorrelationStatus.FAILED
            
            # Remove context
            with self.correlator_lock:
                if context_id in self.active_contexts:
                    del self.active_contexts[context_id]
            
            logger.info(f"Completed correlation for rule {rule.rule_id} with {len(context.events)} events")
            return result
            
        except Exception as e:
            logger.error(f"Failed to complete correlation: {str(e)}")
            return None
    
    def _cleanup_expired_contexts(self):
        """Clean up expired correlation contexts"""
        try:
            while True:
                time.sleep(60)  # Run every minute
                
                current_time = datetime.now()
                expired_contexts = []
                
                with self.correlator_lock:
                    for context_id, context in self.active_contexts.items():
                        if context.expires_at < current_time:
                            expired_contexts.append(context_id)
                    
                    for context_id in expired_contexts:
                        context = self.active_contexts[context_id]
                        context.status = CorrelationStatus.EXPIRED
                        del self.active_contexts[context_id]
                        logger.debug(f"Expired correlation context: {context_id}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired contexts: {str(e)}")
    
    def get_correlation_statistics(self) -> Dict[str, Any]:
        """Get correlation statistics"""
        try:
            with self.correlator_lock:
                return {
                    "total_rules": len(self.rules),
                    "active_contexts": len(self.active_contexts),
                    "event_buffer_size": len(self.event_buffer),
                    "rules_by_type": {
                        rule_type.value: sum(1 for rule in self.rules.values() if rule.correlation_type == rule_type)
                        for rule_type in CorrelationType
                    },
                    "enabled_rules": sum(1 for rule in self.rules.values() if rule.enabled)
                }
                
        except Exception as e:
            logger.error(f"Failed to get correlation statistics: {str(e)}")
            return {}
    
    def get_active_contexts(self) -> List[CorrelationContext]:
        """Get all active correlation contexts"""
        try:
            with self.correlator_lock:
                return list(self.active_contexts.values())
                
        except Exception as e:
            logger.error(f"Failed to get active contexts: {str(e)}")
            return []
