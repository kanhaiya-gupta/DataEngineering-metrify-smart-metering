"""
Query Handlers
Query handling for CQRS pattern implementation
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Type, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import uuid
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class QueryStatus(Enum):
    """Query execution status"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_FOUND = "not_found"

class QueryResult(Enum):
    """Query execution result"""
    SUCCESS = "success"
    FAILURE = "failure"
    NOT_FOUND = "not_found"
    PARTIAL_SUCCESS = "partial_success"

@dataclass
class Query:
    """Base query class"""
    query_id: str
    query_type: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None

@dataclass
class QueryExecution:
    """Query execution tracking"""
    query_id: str
    status: QueryStatus
    result: Optional[QueryResult] = None
    data: Optional[Any] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    record_count: Optional[int] = None

class QueryHandler(ABC):
    """Abstract base class for query handlers"""
    
    def __init__(self):
        self.handler_lock = threading.RLock()
        
    @abstractmethod
    def can_handle(self, query: Query) -> bool:
        """Check if this handler can handle the query"""
        pass
    
    @abstractmethod
    def handle(self, query: Query) -> QueryExecution:
        """Handle the query and return execution result"""
        pass

class QueryBus:
    """Query bus for routing queries to handlers"""
    
    def __init__(self):
        self.handlers: List[QueryHandler] = []
        self.bus_lock = threading.RLock()
        
        logger.info("QueryBus initialized")
    
    def register_handler(self, handler: QueryHandler) -> bool:
        """Register a query handler"""
        try:
            with self.bus_lock:
                self.handlers.append(handler)
                logger.info(f"Registered query handler: {handler.__class__.__name__}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register query handler: {str(e)}")
            return False
    
    def unregister_handler(self, handler: QueryHandler) -> bool:
        """Unregister a query handler"""
        try:
            with self.bus_lock:
                if handler in self.handlers:
                    self.handlers.remove(handler)
                    logger.info(f"Unregistered query handler: {handler.__class__.__name__}")
                    return True
                else:
                    logger.warning(f"Handler {handler.__class__.__name__} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to unregister query handler: {str(e)}")
            return False
    
    def execute_query(self, query: Query) -> QueryExecution:
        """Execute a query"""
        try:
            # Find appropriate handler
            handler = self._find_handler(query)
            if not handler:
                return QueryExecution(
                    query_id=query.query_id,
                    status=QueryStatus.FAILED,
                    result=QueryResult.FAILURE,
                    error_message=f"No handler found for query type: {query.query_type}"
                )
            
            # Execute query
            logger.info(f"Executing query {query.query_id} with handler {handler.__class__.__name__}")
            execution = handler.handle(query)
            
            return execution
            
        except Exception as e:
            logger.error(f"Failed to execute query {query.query_id}: {str(e)}")
            return QueryExecution(
                query_id=query.query_id,
                status=QueryStatus.FAILED,
                result=QueryResult.FAILURE,
                error_message=str(e)
            )
    
    def _find_handler(self, query: Query) -> Optional[QueryHandler]:
        """Find handler for query"""
        try:
            for handler in self.handlers:
                if handler.can_handle(query):
                    return handler
            return None
            
        except Exception as e:
            logger.error(f"Failed to find handler for query {query.query_id}: {str(e)}")
            return None
    
    def create_query(self,
                    query_type: str,
                    parameters: Dict[str, Any],
                    metadata: Dict[str, Any] = None,
                    correlation_id: str = None) -> Query:
        """Create a new query"""
        try:
            return Query(
                query_id=str(uuid.uuid4()),
                query_type=query_type,
                parameters=parameters,
                metadata=metadata or {},
                timestamp=datetime.now(),
                correlation_id=correlation_id
            )
            
        except Exception as e:
            logger.error(f"Failed to create query: {str(e)}")
            raise
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query bus statistics"""
        try:
            return {
                "registered_handlers": len(self.handlers),
                "handler_types": [handler.__class__.__name__ for handler in self.handlers],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get query statistics: {str(e)}")
            return {}

# Example query handlers for smart meter data
class SmartMeterQueryHandler(QueryHandler):
    """Query handler for smart meter read operations"""
    
    def __init__(self, read_model_store: Dict[str, Any]):
        super().__init__()
        self.read_model_store = read_model_store
        
    def can_handle(self, query: Query) -> bool:
        """Check if this handler can handle smart meter queries"""
        return query.query_type.startswith("smart_meter_")
    
    def handle(self, query: Query) -> QueryExecution:
        """Handle smart meter query"""
        try:
            execution = QueryExecution(
                query_id=query.query_id,
                status=QueryStatus.EXECUTING,
                started_at=datetime.now()
            )
            
            data = None
            record_count = 0
            
            if query.query_type == "smart_meter_get_by_id":
                data, record_count = self._handle_get_by_id(query)
            elif query.query_type == "smart_meter_list":
                data, record_count = self._handle_list(query)
            elif query.query_type == "smart_meter_readings":
                data, record_count = self._handle_readings(query)
            elif query.query_type == "smart_meter_statistics":
                data, record_count = self._handle_statistics(query)
            elif query.query_type == "smart_meter_search":
                data, record_count = self._handle_search(query)
            else:
                execution.status = QueryStatus.FAILED
                execution.result = QueryResult.FAILURE
                execution.error_message = f"Unknown query type: {query.query_type}"
                return execution
            
            if data is not None:
                execution.status = QueryStatus.COMPLETED
                execution.result = QueryResult.SUCCESS
                execution.data = data
                execution.record_count = record_count
            else:
                execution.status = QueryStatus.NOT_FOUND
                execution.result = QueryResult.NOT_FOUND
                execution.error_message = "No data found"
            
            execution.completed_at = datetime.now()
            if execution.started_at:
                execution.execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            return execution
            
        except Exception as e:
            logger.error(f"Failed to handle smart meter query {query.query_id}: {str(e)}")
            return QueryExecution(
                query_id=query.query_id,
                status=QueryStatus.FAILED,
                result=QueryResult.FAILURE,
                error_message=str(e),
                started_at=datetime.now(),
                completed_at=datetime.now()
            )
    
    def _handle_get_by_id(self, query: Query) -> tuple[Optional[Dict[str, Any]], int]:
        """Handle get meter by ID query"""
        try:
            meter_id = query.parameters.get("meter_id")
            if not meter_id:
                return None, 0
            
            meter_data = self.read_model_store.get("meters", {}).get(meter_id)
            if meter_data:
                return meter_data, 1
            else:
                return None, 0
                
        except Exception as e:
            logger.error(f"Failed to handle get by ID query: {str(e)}")
            return None, 0
    
    def _handle_list(self, query: Query) -> tuple[Optional[List[Dict[str, Any]]], int]:
        """Handle list meters query"""
        try:
            meters = self.read_model_store.get("meters", {})
            
            # Apply filters
            filtered_meters = []
            for meter_id, meter_data in meters.items():
                if self._matches_filters(meter_data, query.parameters.get("filters", {})):
                    filtered_meters.append(meter_data)
            
            # Apply pagination
            page = query.parameters.get("page", 1)
            page_size = query.parameters.get("page_size", 10)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            paginated_meters = filtered_meters[start_idx:end_idx]
            
            return paginated_meters, len(filtered_meters)
            
        except Exception as e:
            logger.error(f"Failed to handle list query: {str(e)}")
            return None, 0
    
    def _handle_readings(self, query: Query) -> tuple[Optional[List[Dict[str, Any]]], int]:
        """Handle meter readings query"""
        try:
            meter_id = query.parameters.get("meter_id")
            if not meter_id:
                return None, 0
            
            readings = self.read_model_store.get("readings", {}).get(meter_id, [])
            
            # Apply time filters
            from_timestamp = query.parameters.get("from_timestamp")
            to_timestamp = query.parameters.get("to_timestamp")
            
            if from_timestamp or to_timestamp:
                filtered_readings = []
                for reading in readings:
                    reading_time = reading.get("timestamp")
                    if reading_time:
                        if from_timestamp and reading_time < from_timestamp:
                            continue
                        if to_timestamp and reading_time > to_timestamp:
                            continue
                        filtered_readings.append(reading)
                readings = filtered_readings
            
            # Apply pagination
            page = query.parameters.get("page", 1)
            page_size = query.parameters.get("page_size", 100)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            paginated_readings = readings[start_idx:end_idx]
            
            return paginated_readings, len(readings)
            
        except Exception as e:
            logger.error(f"Failed to handle readings query: {str(e)}")
            return None, 0
    
    def _handle_statistics(self, query: Query) -> tuple[Optional[Dict[str, Any]], int]:
        """Handle meter statistics query"""
        try:
            meter_id = query.parameters.get("meter_id")
            if not meter_id:
                return None, 0
            
            readings = self.read_model_store.get("readings", {}).get(meter_id, [])
            if not readings:
                return None, 0
            
            # Calculate statistics
            values = [r.get("value", 0) for r in readings if r.get("value") is not None]
            
            if not values:
                return None, 0
            
            stats = {
                "total_readings": len(values),
                "min_value": min(values),
                "max_value": max(values),
                "avg_value": sum(values) / len(values),
                "first_reading": readings[0].get("timestamp") if readings else None,
                "last_reading": readings[-1].get("timestamp") if readings else None
            }
            
            return stats, 1
            
        except Exception as e:
            logger.error(f"Failed to handle statistics query: {str(e)}")
            return None, 0
    
    def _handle_search(self, query: Query) -> tuple[Optional[List[Dict[str, Any]]], int]:
        """Handle meter search query"""
        try:
            search_term = query.parameters.get("search_term", "").lower()
            if not search_term:
                return [], 0
            
            meters = self.read_model_store.get("meters", {})
            matching_meters = []
            
            for meter_id, meter_data in meters.items():
                # Search in meter data
                searchable_fields = [
                    meter_data.get("location", ""),
                    meter_data.get("meter_type", ""),
                    meter_data.get("meter_id", "")
                ]
                
                if any(search_term in str(field).lower() for field in searchable_fields):
                    matching_meters.append(meter_data)
            
            return matching_meters, len(matching_meters)
            
        except Exception as e:
            logger.error(f"Failed to handle search query: {str(e)}")
            return None, 0
    
    def _matches_filters(self, meter_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if meter data matches filters"""
        try:
            for field, value in filters.items():
                if field in meter_data:
                    if isinstance(value, list):
                        if meter_data[field] not in value:
                            return False
                    else:
                        if meter_data[field] != value:
                            return False
                else:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check filters: {str(e)}")
            return False

class GridOperatorQueryHandler(QueryHandler):
    """Query handler for grid operator read operations"""
    
    def __init__(self, read_model_store: Dict[str, Any]):
        super().__init__()
        self.read_model_store = read_model_store
        
    def can_handle(self, query: Query) -> bool:
        """Check if this handler can handle grid operator queries"""
        return query.query_type.startswith("grid_operator_")
    
    def handle(self, query: Query) -> QueryExecution:
        """Handle grid operator query"""
        try:
            execution = QueryExecution(
                query_id=query.query_id,
                status=QueryStatus.EXECUTING,
                started_at=datetime.now()
            )
            
            data = None
            record_count = 0
            
            if query.query_type == "grid_operator_get_by_id":
                data, record_count = self._handle_get_by_id(query)
            elif query.query_type == "grid_operator_list":
                data, record_count = self._handle_list(query)
            elif query.query_type == "grid_operator_data":
                data, record_count = self._handle_data(query)
            elif query.query_type == "grid_operator_statistics":
                data, record_count = self._handle_statistics(query)
            else:
                execution.status = QueryStatus.FAILED
                execution.result = QueryResult.FAILURE
                execution.error_message = f"Unknown query type: {query.query_type}"
                return execution
            
            if data is not None:
                execution.status = QueryStatus.COMPLETED
                execution.result = QueryResult.SUCCESS
                execution.data = data
                execution.record_count = record_count
            else:
                execution.status = QueryStatus.NOT_FOUND
                execution.result = QueryResult.NOT_FOUND
                execution.error_message = "No data found"
            
            execution.completed_at = datetime.now()
            if execution.started_at:
                execution.execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            return execution
            
        except Exception as e:
            logger.error(f"Failed to handle grid operator query {query.query_id}: {str(e)}")
            return QueryExecution(
                query_id=query.query_id,
                status=QueryStatus.FAILED,
                result=QueryResult.FAILURE,
                error_message=str(e),
                started_at=datetime.now(),
                completed_at=datetime.now()
            )
    
    def _handle_get_by_id(self, query: Query) -> tuple[Optional[Dict[str, Any]], int]:
        """Handle get operator by ID query"""
        try:
            operator_id = query.parameters.get("operator_id")
            if not operator_id:
                return None, 0
            
            operator_data = self.read_model_store.get("operators", {}).get(operator_id)
            if operator_data:
                return operator_data, 1
            else:
                return None, 0
                
        except Exception as e:
            logger.error(f"Failed to handle get by ID query: {str(e)}")
            return None, 0
    
    def _handle_list(self, query: Query) -> tuple[Optional[List[Dict[str, Any]]], int]:
        """Handle list operators query"""
        try:
            operators = self.read_model_store.get("operators", {})
            
            # Apply filters
            filtered_operators = []
            for operator_id, operator_data in operators.items():
                if self._matches_filters(operator_data, query.parameters.get("filters", {})):
                    filtered_operators.append(operator_data)
            
            # Apply pagination
            page = query.parameters.get("page", 1)
            page_size = query.parameters.get("page_size", 10)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            paginated_operators = filtered_operators[start_idx:end_idx]
            
            return paginated_operators, len(filtered_operators)
            
        except Exception as e:
            logger.error(f"Failed to handle list query: {str(e)}")
            return None, 0
    
    def _handle_data(self, query: Query) -> tuple[Optional[List[Dict[str, Any]]], int]:
        """Handle operator data query"""
        try:
            operator_id = query.parameters.get("operator_id")
            if not operator_id:
                return None, 0
            
            data = self.read_model_store.get("operator_data", {}).get(operator_id, [])
            
            # Apply time filters
            from_timestamp = query.parameters.get("from_timestamp")
            to_timestamp = query.parameters.get("to_timestamp")
            
            if from_timestamp or to_timestamp:
                filtered_data = []
                for item in data:
                    item_time = item.get("timestamp")
                    if item_time:
                        if from_timestamp and item_time < from_timestamp:
                            continue
                        if to_timestamp and item_time > to_timestamp:
                            continue
                        filtered_data.append(item)
                data = filtered_data
            
            return data, len(data)
            
        except Exception as e:
            logger.error(f"Failed to handle data query: {str(e)}")
            return None, 0
    
    def _handle_statistics(self, query: Query) -> tuple[Optional[Dict[str, Any]], int]:
        """Handle operator statistics query"""
        try:
            operator_id = query.parameters.get("operator_id")
            if not operator_id:
                return None, 0
            
            data = self.read_model_store.get("operator_data", {}).get(operator_id, [])
            if not data:
                return None, 0
            
            # Calculate statistics
            stats = {
                "total_data_points": len(data),
                "data_types": list(set(item.get("data_type", "") for item in data)),
                "first_data": data[0].get("timestamp") if data else None,
                "last_data": data[-1].get("timestamp") if data else None
            }
            
            return stats, 1
            
        except Exception as e:
            logger.error(f"Failed to handle statistics query: {str(e)}")
            return None, 0
    
    def _matches_filters(self, operator_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if operator data matches filters"""
        try:
            for field, value in filters.items():
                if field in operator_data:
                    if isinstance(value, list):
                        if operator_data[field] not in value:
                            return False
                    else:
                        if operator_data[field] != value:
                            return False
                else:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check filters: {str(e)}")
            return False
