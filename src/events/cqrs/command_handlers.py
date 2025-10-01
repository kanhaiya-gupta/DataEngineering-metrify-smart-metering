"""
Command Handlers
Command handling for CQRS pattern implementation
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Type, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import uuid
import threading
from abc import ABC, abstractmethod

from ..sourcing.event_store import EventStore, Event, EventType

logger = logging.getLogger(__name__)

class CommandStatus(Enum):
    """Command execution status"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"

class CommandResult(Enum):
    """Command execution result"""
    SUCCESS = "success"
    FAILURE = "failure"
    REJECTED = "rejected"
    PARTIAL_SUCCESS = "partial_success"

@dataclass
class Command:
    """Base command class"""
    command_id: str
    command_type: str
    aggregate_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None

@dataclass
class CommandExecution:
    """Command execution tracking"""
    command_id: str
    status: CommandStatus
    result: Optional[CommandResult] = None
    events_generated: List[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None

class CommandHandler(ABC):
    """Abstract base class for command handlers"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.handler_lock = threading.RLock()
        
    @abstractmethod
    def can_handle(self, command: Command) -> bool:
        """Check if this handler can handle the command"""
        pass
    
    @abstractmethod
    def handle(self, command: Command) -> CommandExecution:
        """Handle the command and return execution result"""
        pass
    
    def _create_event(self,
                     aggregate_id: str,
                     event_type: EventType,
                     event_name: str,
                     data: Dict[str, Any],
                     metadata: Dict[str, Any] = None,
                     correlation_id: str = None,
                     causation_id: str = None) -> Event:
        """Create a new event"""
        return Event(
            event_id=str(uuid.uuid4()),
            aggregate_id=aggregate_id,
            event_type=event_type,
            event_name=event_name,
            version=0,  # Will be set when appended
            data=data,
            metadata=metadata or {},
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            causation_id=causation_id
        )
    
    def _append_events(self, aggregate_id: str, events: List[Event]) -> bool:
        """Append events to the event store"""
        try:
            current_version = self.event_store.get_aggregate_version(aggregate_id)
            return self.event_store.append_events(aggregate_id, events, current_version)
        except Exception as e:
            logger.error(f"Failed to append events for aggregate {aggregate_id}: {str(e)}")
            return False

class CommandBus:
    """Command bus for routing commands to handlers"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.handlers: List[CommandHandler] = []
        self.bus_lock = threading.RLock()
        
        logger.info("CommandBus initialized")
    
    def register_handler(self, handler: CommandHandler) -> bool:
        """Register a command handler"""
        try:
            with self.bus_lock:
                self.handlers.append(handler)
                logger.info(f"Registered command handler: {handler.__class__.__name__}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register command handler: {str(e)}")
            return False
    
    def unregister_handler(self, handler: CommandHandler) -> bool:
        """Unregister a command handler"""
        try:
            with self.bus_lock:
                if handler in self.handlers:
                    self.handlers.remove(handler)
                    logger.info(f"Unregistered command handler: {handler.__class__.__name__}")
                    return True
                else:
                    logger.warning(f"Handler {handler.__class__.__name__} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to unregister command handler: {str(e)}")
            return False
    
    def execute_command(self, command: Command) -> CommandExecution:
        """Execute a command"""
        try:
            # Find appropriate handler
            handler = self._find_handler(command)
            if not handler:
                return CommandExecution(
                    command_id=command.command_id,
                    status=CommandStatus.REJECTED,
                    result=CommandResult.REJECTED,
                    error_message=f"No handler found for command type: {command.command_type}"
                )
            
            # Execute command
            logger.info(f"Executing command {command.command_id} with handler {handler.__class__.__name__}")
            execution = handler.handle(command)
            
            return execution
            
        except Exception as e:
            logger.error(f"Failed to execute command {command.command_id}: {str(e)}")
            return CommandExecution(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                result=CommandResult.FAILURE,
                error_message=str(e)
            )
    
    def _find_handler(self, command: Command) -> Optional[CommandHandler]:
        """Find handler for command"""
        try:
            for handler in self.handlers:
                if handler.can_handle(command):
                    return handler
            return None
            
        except Exception as e:
            logger.error(f"Failed to find handler for command {command.command_id}: {str(e)}")
            return None
    
    def create_command(self,
                      command_type: str,
                      aggregate_id: str,
                      data: Dict[str, Any],
                      metadata: Dict[str, Any] = None,
                      correlation_id: str = None,
                      causation_id: str = None) -> Command:
        """Create a new command"""
        try:
            return Command(
                command_id=str(uuid.uuid4()),
                command_type=command_type,
                aggregate_id=aggregate_id,
                data=data,
                metadata=metadata or {},
                timestamp=datetime.now(),
                correlation_id=correlation_id,
                causation_id=causation_id
            )
            
        except Exception as e:
            logger.error(f"Failed to create command: {str(e)}")
            raise
    
    def get_command_statistics(self) -> Dict[str, Any]:
        """Get command bus statistics"""
        try:
            return {
                "registered_handlers": len(self.handlers),
                "handler_types": [handler.__class__.__name__ for handler in self.handlers],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get command statistics: {str(e)}")
            return {}

# Example command handlers for smart meter data
class SmartMeterCommandHandler(CommandHandler):
    """Command handler for smart meter operations"""
    
    def can_handle(self, command: Command) -> bool:
        """Check if this handler can handle smart meter commands"""
        return command.command_type.startswith("smart_meter_")
    
    def handle(self, command: Command) -> CommandExecution:
        """Handle smart meter command"""
        try:
            execution = CommandExecution(
                command_id=command.command_id,
                status=CommandStatus.EXECUTING,
                started_at=datetime.now()
            )
            
            events = []
            
            if command.command_type == "smart_meter_create":
                events = self._handle_create_meter(command)
            elif command.command_type == "smart_meter_update":
                events = self._handle_update_meter(command)
            elif command.command_type == "smart_meter_reading":
                events = self._handle_reading(command)
            elif command.command_type == "smart_meter_deactivate":
                events = self._handle_deactivate_meter(command)
            else:
                execution.status = CommandStatus.REJECTED
                execution.result = CommandResult.REJECTED
                execution.error_message = f"Unknown command type: {command.command_type}"
                return execution
            
            # Append events to store
            if events:
                success = self._append_events(command.aggregate_id, events)
                if success:
                    execution.status = CommandStatus.COMPLETED
                    execution.result = CommandResult.SUCCESS
                    execution.events_generated = [event.event_id for event in events]
                else:
                    execution.status = CommandStatus.FAILED
                    execution.result = CommandResult.FAILURE
                    execution.error_message = "Failed to append events to store"
            else:
                execution.status = CommandStatus.REJECTED
                execution.result = CommandResult.REJECTED
                execution.error_message = "No events generated"
            
            execution.completed_at = datetime.now()
            if execution.started_at:
                execution.execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            return execution
            
        except Exception as e:
            logger.error(f"Failed to handle smart meter command {command.command_id}: {str(e)}")
            return CommandExecution(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                result=CommandResult.FAILURE,
                error_message=str(e),
                started_at=datetime.now(),
                completed_at=datetime.now()
            )
    
    def _handle_create_meter(self, command: Command) -> List[Event]:
        """Handle create meter command"""
        try:
            events = []
            
            # Create meter created event
            event = self._create_event(
                aggregate_id=command.aggregate_id,
                event_type=EventType.DOMAIN_EVENT,
                event_name="SmartMeterCreated",
                data={
                    "meter_id": command.data.get("meter_id"),
                    "location": command.data.get("location"),
                    "meter_type": command.data.get("meter_type"),
                    "installation_date": command.data.get("installation_date")
                },
                metadata=command.metadata,
                correlation_id=command.correlation_id,
                causation_id=command.causation_id
            )
            events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to handle create meter command: {str(e)}")
            return []
    
    def _handle_update_meter(self, command: Command) -> List[Event]:
        """Handle update meter command"""
        try:
            events = []
            
            # Create meter updated event
            event = self._create_event(
                aggregate_id=command.aggregate_id,
                event_type=EventType.DOMAIN_EVENT,
                event_name="SmartMeterUpdated",
                data={
                    "meter_id": command.data.get("meter_id"),
                    "updates": command.data.get("updates", {}),
                    "updated_at": datetime.now().isoformat()
                },
                metadata=command.metadata,
                correlation_id=command.correlation_id,
                causation_id=command.causation_id
            )
            events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to handle update meter command: {str(e)}")
            return []
    
    def _handle_reading(self, command: Command) -> List[Event]:
        """Handle meter reading command"""
        try:
            events = []
            
            # Create reading recorded event
            event = self._create_event(
                aggregate_id=command.aggregate_id,
                event_type=EventType.DOMAIN_EVENT,
                event_name="MeterReadingRecorded",
                data={
                    "meter_id": command.data.get("meter_id"),
                    "reading_value": command.data.get("reading_value"),
                    "reading_timestamp": command.data.get("reading_timestamp"),
                    "quality_score": command.data.get("quality_score", 1.0)
                },
                metadata=command.metadata,
                correlation_id=command.correlation_id,
                causation_id=command.causation_id
            )
            events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to handle reading command: {str(e)}")
            return []
    
    def _handle_deactivate_meter(self, command: Command) -> List[Event]:
        """Handle deactivate meter command"""
        try:
            events = []
            
            # Create meter deactivated event
            event = self._create_event(
                aggregate_id=command.aggregate_id,
                event_type=EventType.DOMAIN_EVENT,
                event_name="SmartMeterDeactivated",
                data={
                    "meter_id": command.data.get("meter_id"),
                    "deactivation_reason": command.data.get("deactivation_reason"),
                    "deactivated_at": datetime.now().isoformat()
                },
                metadata=command.metadata,
                correlation_id=command.correlation_id,
                causation_id=command.causation_id
            )
            events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to handle deactivate meter command: {str(e)}")
            return []

class GridOperatorCommandHandler(CommandHandler):
    """Command handler for grid operator operations"""
    
    def can_handle(self, command: Command) -> bool:
        """Check if this handler can handle grid operator commands"""
        return command.command_type.startswith("grid_operator_")
    
    def handle(self, command: Command) -> CommandExecution:
        """Handle grid operator command"""
        try:
            execution = CommandExecution(
                command_id=command.command_id,
                status=CommandStatus.EXECUTING,
                started_at=datetime.now()
            )
            
            events = []
            
            if command.command_type == "grid_operator_create":
                events = self._handle_create_operator(command)
            elif command.command_type == "grid_operator_update":
                events = self._handle_update_operator(command)
            elif command.command_type == "grid_operator_data":
                events = self._handle_operator_data(command)
            else:
                execution.status = CommandStatus.REJECTED
                execution.result = CommandResult.REJECTED
                execution.error_message = f"Unknown command type: {command.command_type}"
                return execution
            
            # Append events to store
            if events:
                success = self._append_events(command.aggregate_id, events)
                if success:
                    execution.status = CommandStatus.COMPLETED
                    execution.result = CommandResult.SUCCESS
                    execution.events_generated = [event.event_id for event in events]
                else:
                    execution.status = CommandStatus.FAILED
                    execution.result = CommandResult.FAILURE
                    execution.error_message = "Failed to append events to store"
            else:
                execution.status = CommandStatus.REJECTED
                execution.result = CommandResult.REJECTED
                execution.error_message = "No events generated"
            
            execution.completed_at = datetime.now()
            if execution.started_at:
                execution.execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            return execution
            
        except Exception as e:
            logger.error(f"Failed to handle grid operator command {command.command_id}: {str(e)}")
            return CommandExecution(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                result=CommandResult.FAILURE,
                error_message=str(e),
                started_at=datetime.now(),
                completed_at=datetime.now()
            )
    
    def _handle_create_operator(self, command: Command) -> List[Event]:
        """Handle create operator command"""
        try:
            events = []
            
            event = self._create_event(
                aggregate_id=command.aggregate_id,
                event_type=EventType.DOMAIN_EVENT,
                event_name="GridOperatorCreated",
                data={
                    "operator_id": command.data.get("operator_id"),
                    "operator_name": command.data.get("operator_name"),
                    "region": command.data.get("region"),
                    "contact_info": command.data.get("contact_info")
                },
                metadata=command.metadata,
                correlation_id=command.correlation_id,
                causation_id=command.causation_id
            )
            events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to handle create operator command: {str(e)}")
            return []
    
    def _handle_update_operator(self, command: Command) -> List[Event]:
        """Handle update operator command"""
        try:
            events = []
            
            event = self._create_event(
                aggregate_id=command.aggregate_id,
                event_type=EventType.DOMAIN_EVENT,
                event_name="GridOperatorUpdated",
                data={
                    "operator_id": command.data.get("operator_id"),
                    "updates": command.data.get("updates", {}),
                    "updated_at": datetime.now().isoformat()
                },
                metadata=command.metadata,
                correlation_id=command.correlation_id,
                causation_id=command.causation_id
            )
            events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to handle update operator command: {str(e)}")
            return []
    
    def _handle_operator_data(self, command: Command) -> List[Event]:
        """Handle operator data command"""
        try:
            events = []
            
            event = self._create_event(
                aggregate_id=command.aggregate_id,
                event_type=EventType.DOMAIN_EVENT,
                event_name="GridOperatorDataReceived",
                data={
                    "operator_id": command.data.get("operator_id"),
                    "data_type": command.data.get("data_type"),
                    "data": command.data.get("data"),
                    "timestamp": command.data.get("timestamp")
                },
                metadata=command.metadata,
                correlation_id=command.correlation_id,
                causation_id=command.causation_id
            )
            events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to handle operator data command: {str(e)}")
            return []
