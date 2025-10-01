"""
Complex Event Processing Module

This module provides complex event processing capabilities:
- Event correlation and pattern detection
- Business rule engine for event processing
- Real-time event stream analytics
- Event-driven alerting and notifications
"""

from .event_correlator import EventCorrelator, CorrelationRule
from .pattern_detector import PatternDetector, EventPattern
from .business_rule_engine import BusinessRuleEngine, BusinessRule

__all__ = [
    "EventCorrelator",
    "CorrelationRule",
    "PatternDetector",
    "EventPattern",
    "BusinessRuleEngine",
    "BusinessRule"
]
