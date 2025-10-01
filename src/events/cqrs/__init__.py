"""
CQRS (Command Query Responsibility Segregation) Module

This module provides CQRS pattern implementation:
- Command handlers for write operations
- Query handlers for read operations
- Read model synchronization
- Event-driven read model updates
"""

from .command_handlers import CommandHandler, CommandBus
from .query_handlers import QueryHandler, QueryBus
from .read_model_sync import ReadModelSync, ReadModelProjection

__all__ = [
    "CommandHandler",
    "CommandBus",
    "QueryHandler", 
    "QueryBus",
    "ReadModelSync",
    "ReadModelProjection"
]
