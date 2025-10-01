"""
Pattern Detector
Advanced pattern detection for complex event processing
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import time
import re
from collections import defaultdict, deque

from ..sourcing.event_store import Event, EventType

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Pattern types"""
    SEQUENCE = "sequence"
    FREQUENCY = "frequency"
    ANOMALY = "anomaly"
    TREND = "trend"
    CYCLIC = "cyclic"
    THRESHOLD = "threshold"
    CORRELATION = "correlation"
    CUSTOM = "custom"

class PatternStatus(Enum):
    """Pattern status"""
    DETECTING = "detecting"
    DETECTED = "detected"
    EXPIRED = "expired"
    FAILED = "failed"

@dataclass
class EventPattern:
    """Event pattern definition"""
    pattern_id: str
    name: str
    pattern_type: PatternType
    event_types: List[EventType]
    time_window: timedelta
    conditions: Dict[str, Any]
    action: Callable[[List[Event], Dict[str, Any]], None]
    priority: int = 0
    enabled: bool = True
    created_at: datetime = None

@dataclass
class PatternMatch:
    """Pattern match result"""
    pattern_id: str
    match_id: str
    matched_events: List[Event]
    pattern_type: PatternType
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class PatternContext:
    """Pattern detection context"""
    context_id: str
    pattern_id: str
    events: List[Event]
    created_at: datetime
    expires_at: datetime
    status: PatternStatus = PatternStatus.DETECTING
    metadata: Dict[str, Any] = None

class PatternDetector:
    """
    Advanced pattern detector for complex event processing
    """
    
    def __init__(self):
        self.patterns: Dict[str, EventPattern] = {}
        self.active_contexts: Dict[str, PatternContext] = {}
        self.event_buffer: deque = deque(maxlen=50000)  # Larger buffer for pattern detection
        self.pattern_lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_contexts, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("PatternDetector initialized")
    
    def register_pattern(self, pattern: EventPattern) -> bool:
        """Register a pattern for detection"""
        try:
            with self.pattern_lock:
                pattern.created_at = datetime.now()
                self.patterns[pattern.pattern_id] = pattern
                logger.info(f"Registered pattern: {pattern.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register pattern {pattern.pattern_id}: {str(e)}")
            return False
    
    def unregister_pattern(self, pattern_id: str) -> bool:
        """Unregister a pattern"""
        try:
            with self.pattern_lock:
                if pattern_id in self.patterns:
                    del self.patterns[pattern_id]
                    logger.info(f"Unregistered pattern: {pattern_id}")
                    return True
                else:
                    logger.warning(f"Pattern {pattern_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to unregister pattern {pattern_id}: {str(e)}")
            return False
    
    def process_event(self, event: Event) -> List[PatternMatch]:
        """Process an event for pattern detection"""
        try:
            matches = []
            
            # Add event to buffer
            with self.pattern_lock:
                self.event_buffer.append(event)
            
            # Check against all enabled patterns
            for pattern in self.patterns.values():
                if not pattern.enabled:
                    continue
                
                if event.event_type in pattern.event_types:
                    pattern_match = self._detect_pattern(event, pattern)
                    if pattern_match:
                        matches.append(pattern_match)
            
            logger.debug(f"Processed event {event.event_id} for pattern detection, found {len(matches)} matches")
            return matches
            
        except Exception as e:
            logger.error(f"Failed to process event {event.event_id} for pattern detection: {str(e)}")
            return []
    
    def _detect_pattern(self, event: Event, pattern: EventPattern) -> Optional[PatternMatch]:
        """Detect pattern for a specific event"""
        try:
            if pattern.pattern_type == PatternType.SEQUENCE:
                return self._detect_sequence_pattern(event, pattern)
            elif pattern.pattern_type == PatternType.FREQUENCY:
                return self._detect_frequency_pattern(event, pattern)
            elif pattern.pattern_type == PatternType.ANOMALY:
                return self._detect_anomaly_pattern(event, pattern)
            elif pattern.pattern_type == PatternType.TREND:
                return self._detect_trend_pattern(event, pattern)
            elif pattern.pattern_type == PatternType.CYCLIC:
                return self._detect_cyclic_pattern(event, pattern)
            elif pattern.pattern_type == PatternType.THRESHOLD:
                return self._detect_threshold_pattern(event, pattern)
            elif pattern.pattern_type == PatternType.CORRELATION:
                return self._detect_correlation_pattern(event, pattern)
            elif pattern.pattern_type == PatternType.CUSTOM:
                return self._detect_custom_pattern(event, pattern)
            else:
                logger.warning(f"Unknown pattern type: {pattern.pattern_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to detect pattern {pattern.pattern_id}: {str(e)}")
            return None
    
    def _detect_sequence_pattern(self, event: Event, pattern: EventPattern) -> Optional[PatternMatch]:
        """Detect sequence pattern"""
        try:
            # Look for events in sequence within time window
            time_window_start = event.timestamp - pattern.time_window
            
            # Get events in time window
            relevant_events = []
            for buffered_event in self.event_buffer:
                if (buffered_event.timestamp >= time_window_start and
                    buffered_event.timestamp <= event.timestamp and
                    buffered_event.event_type in pattern.event_types):
                    relevant_events.append(buffered_event)
            
            # Sort by timestamp
            relevant_events.sort(key=lambda e: e.timestamp)
            
            # Check for required sequence
            required_sequence = pattern.conditions.get("sequence", [])
            if not required_sequence:
                return None
            
            # Find matching sequence
            sequence_matches = self._find_sequence_matches(relevant_events, required_sequence)
            
            if sequence_matches:
                # Create pattern match
                match_id = f"{pattern.pattern_id}_{int(time.time())}"
                confidence = self._calculate_sequence_confidence(sequence_matches, required_sequence)
                
                match = PatternMatch(
                    pattern_id=pattern.pattern_id,
                    match_id=match_id,
                    matched_events=sequence_matches,
                    pattern_type=pattern.pattern_type,
                    confidence=confidence,
                    timestamp=datetime.now()
                )
                
                # Execute action
                try:
                    pattern.action(sequence_matches, {"pattern_type": "sequence"})
                except Exception as e:
                    logger.error(f"Failed to execute pattern action: {str(e)}")
                
                return match
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect sequence pattern: {str(e)}")
            return None
    
    def _detect_frequency_pattern(self, event: Event, pattern: EventPattern) -> Optional[PatternMatch]:
        """Detect frequency pattern"""
        try:
            # Count events of specific types within time window
            time_window_start = event.timestamp - pattern.time_window
            
            event_counts = defaultdict(int)
            relevant_events = []
            
            for buffered_event in self.event_buffer:
                if (buffered_event.timestamp >= time_window_start and
                    buffered_event.timestamp <= event.timestamp and
                    buffered_event.event_type in pattern.event_types):
                    event_counts[buffered_event.event_type] += 1
                    relevant_events.append(buffered_event)
            
            # Check frequency conditions
            min_frequency = pattern.conditions.get("min_frequency", 1)
            max_frequency = pattern.conditions.get("max_frequency", float('inf'))
            
            total_events = sum(event_counts.values())
            if min_frequency <= total_events <= max_frequency:
                # Check specific event type frequencies
                type_frequencies = pattern.conditions.get("type_frequencies", {})
                frequency_met = True
                
                for event_type, required_count in type_frequencies.items():
                    if event_counts.get(event_type, 0) < required_count:
                        frequency_met = False
                        break
                
                if frequency_met:
                    match_id = f"{pattern.pattern_id}_{int(time.time())}"
                    confidence = self._calculate_frequency_confidence(event_counts, pattern.conditions)
                    
                    match = PatternMatch(
                        pattern_id=pattern.pattern_id,
                        match_id=match_id,
                        matched_events=relevant_events,
                        pattern_type=pattern.pattern_type,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        metadata={"event_counts": dict(event_counts)}
                    )
                    
                    # Execute action
                    try:
                        pattern.action(relevant_events, {"pattern_type": "frequency", "counts": dict(event_counts)})
                    except Exception as e:
                        logger.error(f"Failed to execute pattern action: {str(e)}")
                    
                    return match
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect frequency pattern: {str(e)}")
            return None
    
    def _detect_anomaly_pattern(self, event: Event, pattern: EventPattern) -> Optional[PatternMatch]:
        """Detect anomaly pattern"""
        try:
            # Get historical events for comparison
            time_window_start = event.timestamp - pattern.time_window
            historical_events = []
            
            for buffered_event in self.event_buffer:
                if (buffered_event.timestamp >= time_window_start and
                    buffered_event.timestamp < event.timestamp and
                    buffered_event.event_type in pattern.event_types):
                    historical_events.append(buffered_event)
            
            if len(historical_events) < 5:  # Need minimum data for anomaly detection
                return None
            
            # Check for anomalies in event data
            anomaly_detected = False
            anomaly_fields = pattern.conditions.get("anomaly_fields", [])
            
            for field in anomaly_fields:
                if field in event.data:
                    if self._is_anomalous_value(event.data[field], historical_events, field):
                        anomaly_detected = True
                        break
            
            if anomaly_detected:
                match_id = f"{pattern.pattern_id}_{int(time.time())}"
                confidence = self._calculate_anomaly_confidence(event, historical_events, anomaly_fields)
                
                match = PatternMatch(
                    pattern_id=pattern.pattern_id,
                    match_id=match_id,
                    matched_events=[event],
                    pattern_type=pattern.pattern_type,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    metadata={"anomaly_fields": anomaly_fields}
                )
                
                # Execute action
                try:
                    pattern.action([event], {"pattern_type": "anomaly", "anomaly_fields": anomaly_fields})
                except Exception as e:
                    logger.error(f"Failed to execute pattern action: {str(e)}")
                
                return match
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect anomaly pattern: {str(e)}")
            return None
    
    def _detect_trend_pattern(self, event: Event, pattern: EventPattern) -> Optional[PatternMatch]:
        """Detect trend pattern"""
        try:
            # Get events for trend analysis
            time_window_start = event.timestamp - pattern.time_window
            trend_events = []
            
            for buffered_event in self.event_buffer:
                if (buffered_event.timestamp >= time_window_start and
                    buffered_event.timestamp <= event.timestamp and
                    buffered_event.event_type in pattern.event_types):
                    trend_events.append(buffered_event)
            
            if len(trend_events) < 3:  # Need minimum data for trend analysis
                return None
            
            # Sort by timestamp
            trend_events.sort(key=lambda e: e.timestamp)
            
            # Analyze trend
            trend_field = pattern.conditions.get("trend_field")
            if not trend_field:
                return None
            
            trend_direction = self._analyze_trend(trend_events, trend_field)
            expected_direction = pattern.conditions.get("trend_direction")
            
            if trend_direction == expected_direction:
                match_id = f"{pattern.pattern_id}_{int(time.time())}"
                confidence = self._calculate_trend_confidence(trend_events, trend_field, trend_direction)
                
                match = PatternMatch(
                    pattern_id=pattern.pattern_id,
                    match_id=match_id,
                    matched_events=trend_events,
                    pattern_type=pattern.pattern_type,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    metadata={"trend_direction": trend_direction, "trend_field": trend_field}
                )
                
                # Execute action
                try:
                    pattern.action(trend_events, {"pattern_type": "trend", "direction": trend_direction})
                except Exception as e:
                    logger.error(f"Failed to execute pattern action: {str(e)}")
                
                return match
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect trend pattern: {str(e)}")
            return None
    
    def _detect_cyclic_pattern(self, event: Event, pattern: EventPattern) -> Optional[PatternMatch]:
        """Detect cyclic pattern"""
        try:
            # Get events for cyclic analysis
            time_window_start = event.timestamp - pattern.time_window
            cyclic_events = []
            
            for buffered_event in self.event_buffer:
                if (buffered_event.timestamp >= time_window_start and
                    buffered_event.timestamp <= event.timestamp and
                    buffered_event.event_type in pattern.event_types):
                    cyclic_events.append(buffered_event)
            
            if len(cyclic_events) < 6:  # Need minimum data for cyclic analysis
                return None
            
            # Sort by timestamp
            cyclic_events.sort(key=lambda e: e.timestamp)
            
            # Check for cyclic behavior
            cycle_period = pattern.conditions.get("cycle_period_seconds", 3600)  # Default 1 hour
            cycle_tolerance = pattern.conditions.get("cycle_tolerance", 0.1)  # 10% tolerance
            
            if self._detect_cycle(cyclic_events, cycle_period, cycle_tolerance):
                match_id = f"{pattern.pattern_id}_{int(time.time())}"
                confidence = self._calculate_cyclic_confidence(cyclic_events, cycle_period)
                
                match = PatternMatch(
                    pattern_id=pattern.pattern_id,
                    match_id=match_id,
                    matched_events=cyclic_events,
                    pattern_type=pattern.pattern_type,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    metadata={"cycle_period": cycle_period}
                )
                
                # Execute action
                try:
                    pattern.action(cyclic_events, {"pattern_type": "cyclic", "cycle_period": cycle_period})
                except Exception as e:
                    logger.error(f"Failed to execute pattern action: {str(e)}")
                
                return match
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect cyclic pattern: {str(e)}")
            return None
    
    def _detect_threshold_pattern(self, event: Event, pattern: EventPattern) -> Optional[PatternMatch]:
        """Detect threshold pattern"""
        try:
            # Check if event data exceeds thresholds
            threshold_field = pattern.conditions.get("threshold_field")
            if not threshold_field or threshold_field not in event.data:
                return None
            
            value = event.data[threshold_field]
            if not isinstance(value, (int, float)):
                return None
            
            # Check thresholds
            min_threshold = pattern.conditions.get("min_threshold")
            max_threshold = pattern.conditions.get("max_threshold")
            
            threshold_exceeded = False
            if min_threshold is not None and value < min_threshold:
                threshold_exceeded = True
            elif max_threshold is not None and value > max_threshold:
                threshold_exceeded = True
            
            if threshold_exceeded:
                match_id = f"{pattern.pattern_id}_{int(time.time())}"
                confidence = self._calculate_threshold_confidence(value, min_threshold, max_threshold)
                
                match = PatternMatch(
                    pattern_id=pattern.pattern_id,
                    match_id=match_id,
                    matched_events=[event],
                    pattern_type=pattern.pattern_type,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    metadata={
                        "threshold_field": threshold_field,
                        "value": value,
                        "min_threshold": min_threshold,
                        "max_threshold": max_threshold
                    }
                )
                
                # Execute action
                try:
                    pattern.action([event], {"pattern_type": "threshold", "value": value})
                except Exception as e:
                    logger.error(f"Failed to execute pattern action: {str(e)}")
                
                return match
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect threshold pattern: {str(e)}")
            return None
    
    def _detect_correlation_pattern(self, event: Event, pattern: EventPattern) -> Optional[PatternMatch]:
        """Detect correlation pattern"""
        try:
            # Look for correlated events
            time_window_start = event.timestamp - pattern.time_window
            correlated_events = []
            
            for buffered_event in self.event_buffer:
                if (buffered_event.timestamp >= time_window_start and
                    buffered_event.timestamp <= event.timestamp and
                    buffered_event.event_type in pattern.event_types and
                    buffered_event.event_id != event.event_id):
                    
                    if self._are_events_correlated(event, buffered_event, pattern.conditions):
                        correlated_events.append(buffered_event)
            
            if correlated_events:
                match_id = f"{pattern.pattern_id}_{int(time.time())}"
                confidence = self._calculate_correlation_confidence(event, correlated_events)
                
                match = PatternMatch(
                    pattern_id=pattern.pattern_id,
                    match_id=match_id,
                    matched_events=[event] + correlated_events,
                    pattern_type=pattern.pattern_type,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    metadata={"correlated_count": len(correlated_events)}
                )
                
                # Execute action
                try:
                    pattern.action([event] + correlated_events, {"pattern_type": "correlation"})
                except Exception as e:
                    logger.error(f"Failed to execute pattern action: {str(e)}")
                
                return match
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect correlation pattern: {str(e)}")
            return None
    
    def _detect_custom_pattern(self, event: Event, pattern: EventPattern) -> Optional[PatternMatch]:
        """Detect custom pattern using custom logic"""
        try:
            # Get events for custom analysis
            time_window_start = event.timestamp - pattern.time_window
            custom_events = []
            
            for buffered_event in self.event_buffer:
                if (buffered_event.timestamp >= time_window_start and
                    buffered_event.timestamp <= event.timestamp and
                    buffered_event.event_type in pattern.event_types):
                    custom_events.append(buffered_event)
            
            # Use custom detection logic
            custom_detector = pattern.conditions.get("custom_detector")
            if custom_detector and callable(custom_detector):
                result = custom_detector(event, custom_events, pattern.conditions)
                if result:
                    match_id = f"{pattern.pattern_id}_{int(time.time())}"
                    confidence = result.get("confidence", 0.5)
                    
                    match = PatternMatch(
                        pattern_id=pattern.pattern_id,
                        match_id=match_id,
                        matched_events=result.get("matched_events", [event]),
                        pattern_type=pattern.pattern_type,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        metadata=result.get("metadata", {})
                    )
                    
                    # Execute action
                    try:
                        pattern.action(match.matched_events, {"pattern_type": "custom"})
                    except Exception as e:
                        logger.error(f"Failed to execute pattern action: {str(e)}")
                    
                    return match
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect custom pattern: {str(e)}")
            return None
    
    def _find_sequence_matches(self, events: List[Event], required_sequence: List[str]) -> List[Event]:
        """Find events that match required sequence"""
        try:
            if not required_sequence:
                return events
            
            matches = []
            sequence_index = 0
            
            for event in events:
                if event.event_type.value == required_sequence[sequence_index]:
                    matches.append(event)
                    sequence_index += 1
                    
                    if sequence_index >= len(required_sequence):
                        break
            
            return matches if sequence_index == len(required_sequence) else []
            
        except Exception as e:
            logger.error(f"Failed to find sequence matches: {str(e)}")
            return []
    
    def _is_anomalous_value(self, value: Any, historical_events: List[Event], field: str) -> bool:
        """Check if value is anomalous compared to historical data"""
        try:
            if not isinstance(value, (int, float)):
                return False
            
            # Extract historical values
            historical_values = []
            for event in historical_events:
                if field in event.data and isinstance(event.data[field], (int, float)):
                    historical_values.append(event.data[field])
            
            if len(historical_values) < 3:
                return False
            
            # Calculate statistics
            mean_val = sum(historical_values) / len(historical_values)
            variance = sum((x - mean_val) ** 2 for x in historical_values) / len(historical_values)
            std_dev = variance ** 0.5
            
            if std_dev == 0:
                return value != mean_val
            
            # Check if value is more than 2 standard deviations from mean
            z_score = abs(value - mean_val) / std_dev
            return z_score > 2.0
            
        except Exception as e:
            logger.error(f"Failed to check anomalous value: {str(e)}")
            return False
    
    def _analyze_trend(self, events: List[Event], field: str) -> str:
        """Analyze trend direction"""
        try:
            values = []
            for event in events:
                if field in event.data and isinstance(event.data[field], (int, float)):
                    values.append(event.data[field])
            
            if len(values) < 3:
                return "unknown"
            
            # Simple trend analysis
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg > first_avg * 1.1:
                return "increasing"
            elif second_avg < first_avg * 0.9:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Failed to analyze trend: {str(e)}")
            return "unknown"
    
    def _detect_cycle(self, events: List[Event], cycle_period: int, tolerance: float) -> bool:
        """Detect cyclic pattern"""
        try:
            if len(events) < 6:
                return False
            
            # Group events by time periods
            time_groups = defaultdict(list)
            for event in events:
                period = int(event.timestamp.timestamp() // cycle_period)
                time_groups[period].append(event)
            
            # Check if we have events in multiple periods
            if len(time_groups) < 2:
                return False
            
            # Check if events occur with consistent intervals
            periods = sorted(time_groups.keys())
            intervals = [periods[i+1] - periods[i] for i in range(len(periods)-1)]
            
            if not intervals:
                return False
            
            avg_interval = sum(intervals) / len(intervals)
            tolerance_range = avg_interval * tolerance
            
            # Check if all intervals are within tolerance
            for interval in intervals:
                if abs(interval - avg_interval) > tolerance_range:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to detect cycle: {str(e)}")
            return False
    
    def _are_events_correlated(self, event1: Event, event2: Event, conditions: Dict[str, Any]) -> bool:
        """Check if two events are correlated"""
        try:
            # Check correlation conditions
            correlation_fields = conditions.get("correlation_fields", [])
            
            for field in correlation_fields:
                if (event1.data.get(field) != event2.data.get(field) and
                    event1.metadata.get(field) != event2.metadata.get(field)):
                    return False
            
            # Check time correlation
            max_time_gap = conditions.get("max_time_gap_seconds", 3600)
            time_gap = abs((event1.timestamp - event2.timestamp).total_seconds())
            
            return time_gap <= max_time_gap
            
        except Exception as e:
            logger.error(f"Failed to check event correlation: {str(e)}")
            return False
    
    def _calculate_sequence_confidence(self, matches: List[Event], required_sequence: List[str]) -> float:
        """Calculate confidence for sequence pattern"""
        try:
            if not matches or not required_sequence:
                return 0.0
            
            # Simple confidence based on sequence completeness
            completeness = len(matches) / len(required_sequence)
            return min(1.0, completeness)
            
        except Exception as e:
            logger.error(f"Failed to calculate sequence confidence: {str(e)}")
            return 0.0
    
    def _calculate_frequency_confidence(self, event_counts: Dict[EventType, int], conditions: Dict[str, Any]) -> float:
        """Calculate confidence for frequency pattern"""
        try:
            total_events = sum(event_counts.values())
            min_frequency = conditions.get("min_frequency", 1)
            max_frequency = conditions.get("max_frequency", float('inf'))
            
            if min_frequency <= total_events <= max_frequency:
                return 1.0
            else:
                return 0.5  # Partial match
                
        except Exception as e:
            logger.error(f"Failed to calculate frequency confidence: {str(e)}")
            return 0.0
    
    def _calculate_anomaly_confidence(self, event: Event, historical_events: List[Event], anomaly_fields: List[str]) -> float:
        """Calculate confidence for anomaly pattern"""
        try:
            if not anomaly_fields:
                return 0.0
            
            confidence_scores = []
            
            for field in anomaly_fields:
                if field in event.data:
                    value = event.data[field]
                    if isinstance(value, (int, float)):
                        historical_values = [
                            e.data[field] for e in historical_events 
                            if field in e.data and isinstance(e.data[field], (int, float))
                        ]
                        
                        if historical_values:
                            mean_val = sum(historical_values) / len(historical_values)
                            variance = sum((x - mean_val) ** 2 for x in historical_values) / len(historical_values)
                            std_dev = variance ** 0.5
                            
                            if std_dev > 0:
                                z_score = abs(value - mean_val) / std_dev
                                confidence = min(1.0, z_score / 3.0)  # Normalize to 0-1
                                confidence_scores.append(confidence)
            
            return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate anomaly confidence: {str(e)}")
            return 0.0
    
    def _calculate_trend_confidence(self, events: List[Event], field: str, trend_direction: str) -> float:
        """Calculate confidence for trend pattern"""
        try:
            if not events:
                return 0.0
            
            values = [e.data[field] for e in events if field in e.data and isinstance(e.data[field], (int, float))]
            
            if len(values) < 3:
                return 0.0
            
            # Calculate trend strength
            first_third = values[:len(values)//3]
            last_third = values[-len(values)//3:]
            
            first_avg = sum(first_third) / len(first_third)
            last_avg = sum(last_third) / len(last_third)
            
            change_ratio = abs(last_avg - first_avg) / first_avg if first_avg != 0 else 0
            
            return min(1.0, change_ratio)
            
        except Exception as e:
            logger.error(f"Failed to calculate trend confidence: {str(e)}")
            return 0.0
    
    def _calculate_cyclic_confidence(self, events: List[Event], cycle_period: int) -> float:
        """Calculate confidence for cyclic pattern"""
        try:
            if len(events) < 6:
                return 0.0
            
            # Group events by time periods
            time_groups = defaultdict(list)
            for event in events:
                period = int(event.timestamp.timestamp() // cycle_period)
                time_groups[period].append(event)
            
            # Calculate confidence based on period consistency
            periods = sorted(time_groups.keys())
            if len(periods) < 2:
                return 0.0
            
            intervals = [periods[i+1] - periods[i] for i in range(len(periods)-1)]
            avg_interval = sum(intervals) / len(intervals)
            
            # Calculate consistency
            consistency = 1.0
            for interval in intervals:
                deviation = abs(interval - avg_interval) / avg_interval
                consistency *= (1.0 - deviation)
            
            return max(0.0, consistency)
            
        except Exception as e:
            logger.error(f"Failed to calculate cyclic confidence: {str(e)}")
            return 0.0
    
    def _calculate_threshold_confidence(self, value: float, min_threshold: float, max_threshold: float) -> float:
        """Calculate confidence for threshold pattern"""
        try:
            if min_threshold is not None and value < min_threshold:
                # Calculate how far below threshold
                deviation = (min_threshold - value) / min_threshold
                return min(1.0, deviation * 2)  # Scale to 0-1
            
            elif max_threshold is not None and value > max_threshold:
                # Calculate how far above threshold
                deviation = (value - max_threshold) / max_threshold
                return min(1.0, deviation * 2)  # Scale to 0-1
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate threshold confidence: {str(e)}")
            return 0.0
    
    def _calculate_correlation_confidence(self, event: Event, correlated_events: List[Event]) -> float:
        """Calculate confidence for correlation pattern"""
        try:
            if not correlated_events:
                return 0.0
            
            # Simple confidence based on number of correlated events
            base_confidence = min(1.0, len(correlated_events) / 5.0)  # Max confidence at 5+ events
            
            # Adjust based on time proximity
            time_scores = []
            for correlated_event in correlated_events:
                time_gap = abs((event.timestamp - correlated_event.timestamp).total_seconds())
                time_score = max(0.0, 1.0 - (time_gap / 3600))  # Decay over 1 hour
                time_scores.append(time_score)
            
            avg_time_score = sum(time_scores) / len(time_scores) if time_scores else 0.0
            
            return (base_confidence + avg_time_score) / 2.0
            
        except Exception as e:
            logger.error(f"Failed to calculate correlation confidence: {str(e)}")
            return 0.0
    
    def _cleanup_expired_contexts(self):
        """Clean up expired pattern contexts"""
        try:
            while True:
                time.sleep(60)  # Run every minute
                
                current_time = datetime.now()
                expired_contexts = []
                
                with self.pattern_lock:
                    for context_id, context in self.active_contexts.items():
                        if context.expires_at < current_time:
                            expired_contexts.append(context_id)
                    
                    for context_id in expired_contexts:
                        context = self.active_contexts[context_id]
                        context.status = PatternStatus.EXPIRED
                        del self.active_contexts[context_id]
                        logger.debug(f"Expired pattern context: {context_id}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired contexts: {str(e)}")
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get pattern detection statistics"""
        try:
            with self.pattern_lock:
                return {
                    "total_patterns": len(self.patterns),
                    "active_contexts": len(self.active_contexts),
                    "event_buffer_size": len(self.event_buffer),
                    "patterns_by_type": {
                        pattern_type.value: sum(1 for pattern in self.patterns.values() if pattern.pattern_type == pattern_type)
                        for pattern_type in PatternType
                    },
                    "enabled_patterns": sum(1 for pattern in self.patterns.values() if pattern.enabled)
                }
                
        except Exception as e:
            logger.error(f"Failed to get pattern statistics: {str(e)}")
            return {}
    
    def get_active_contexts(self) -> List[PatternContext]:
        """Get all active pattern contexts"""
        try:
            with self.pattern_lock:
                return list(self.active_contexts.values())
                
        except Exception as e:
            logger.error(f"Failed to get active contexts: {str(e)}")
            return []
