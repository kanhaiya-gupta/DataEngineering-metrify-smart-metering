"""
Event-Driven Architecture Module

This module provides comprehensive event-driven architecture capabilities:
- Event sourcing for audit trails and state reconstruction
- CQRS (Command Query Responsibility Segregation) pattern
- Complex event processing and pattern detection
- Event-driven microservices communication
"""

from .sourcing import (
    EventStore,
    EventReplay,
    EventVersioning
)
from .cqrs import (
    CommandHandler,
    QueryHandler,
    ReadModelSync
)
from .processing import (
    EventCorrelator,
    PatternDetector,
    BusinessRuleEngine
)

__all__ = [
    # Event Sourcing
    "EventStore",
    "EventReplay", 
    "EventVersioning",
    
    # CQRS
    "CommandHandler",
    "QueryHandler",
    "ReadModelSync",
    
    # Event Processing
    "EventCorrelator",
    "PatternDetector",
    "BusinessRuleEngine"
]
