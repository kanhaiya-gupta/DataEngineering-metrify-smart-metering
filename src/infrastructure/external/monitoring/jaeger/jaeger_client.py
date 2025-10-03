"""
Jaeger Client Implementation
Handles distributed tracing for comprehensive observability
"""

import time
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import logging
from contextlib import asynccontextmanager
from functools import wraps

try:
    from jaeger_client import Config, Tracer
    from opentracing import tags, logs
    from opentracing.ext import tags as ext_tags
    from opentracing.propagation import Format
    from opentracing.scope_scope_manager import ScopeManager
    from opentracing.scope import Scope
except ImportError:
    Config = None
    Tracer = None
    tags = None
    logs = None
    ext_tags = None
    Format = None
    ScopeManager = None
    Scope = None

from src.core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class JaegerClient:
    """
    Jaeger Client for distributed tracing
    
    Handles distributed tracing across all services and components
    for comprehensive observability and performance monitoring.
    """
    
    def __init__(
        self,
        service_name: str = "metrify-smart-metering",
        agent_host: str = "localhost",
        agent_port: int = 6831,
        collector_endpoint: Optional[str] = None,
        sampling_rate: float = 1.0,
        flush_interval: int = 1000
    ):
        if Tracer is None:
            raise InfrastructureError("Jaeger client not installed", service="jaeger")
        
        self.service_name = service_name
        self.agent_host = agent_host
        self.agent_port = agent_port
        self.collector_endpoint = collector_endpoint
        self.sampling_rate = sampling_rate
        self.flush_interval = flush_interval
        
        self.tracer: Optional[Tracer] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Jaeger client and tracer"""
        try:
            config = Config(
                config={
                    'sampler': {
                        'type': 'const',
                        'param': 1 if self.sampling_rate >= 1.0 else 0,
                    },
                    'local_agent': {
                        'reporting_host': self.agent_host,
                        'reporting_port': self.agent_port,
                    },
                    'logging': True,
                    'reporter_batch_size': 10,
                    'reporter_flush_interval': self.flush_interval,
                },
                service_name=self.service_name,
                validate=True,
            )
            
            if self.collector_endpoint:
                config.config['reporter'] = {
                    'collector_endpoint': self.collector_endpoint,
                }
            
            self.tracer = config.initialize_tracer()
            self._initialized = True
            logger.info(f"Jaeger client initialized for service: {self.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Jaeger client: {str(e)}")
            raise InfrastructureError(f"Failed to initialize Jaeger: {str(e)}", service="jaeger")
    
    def get_tracer(self) -> Tracer:
        """Get the Jaeger tracer instance"""
        if not self._initialized or self.tracer is None:
            raise InfrastructureError("Jaeger client not initialized", service="jaeger")
        return self.tracer
    
    def start_span(
        self,
        operation_name: str,
        parent_span: Optional[Any] = None,
        tags: Optional[Dict[str, Union[str, int, float, bool]]] = None,
        start_time: Optional[float] = None
    ) -> Any:
        """Start a new span"""
        try:
            tracer = self.get_tracer()
            
            span = tracer.start_span(
                operation_name=operation_name,
                child_of=parent_span,
                tags=tags or {},
                start_time=start_time
            )
            
            return span
            
        except Exception as e:
            logger.error(f"Error starting span: {str(e)}")
            raise InfrastructureError(f"Failed to start span: {str(e)}", service="jaeger")
    
    def finish_span(self, span: Any, tags: Optional[Dict[str, Union[str, int, float, bool]]] = None) -> None:
        """Finish a span"""
        try:
            if tags:
                for key, value in tags.items():
                    span.set_tag(key, value)
            
            span.finish()
            
        except Exception as e:
            logger.error(f"Error finishing span: {str(e)}")
            raise InfrastructureError(f"Failed to finish span: {str(e)}", service="jaeger")
    
    def add_span_tag(self, span: Any, key: str, value: Union[str, int, float, bool]) -> None:
        """Add a tag to a span"""
        try:
            span.set_tag(key, value)
        except Exception as e:
            logger.error(f"Error adding span tag: {str(e)}")
    
    def add_span_log(self, span: Any, message: str, fields: Optional[Dict[str, Any]] = None) -> None:
        """Add a log to a span"""
        try:
            log_data = {
                'event': message,
                'timestamp': int(time.time() * 1000000)  # microseconds
            }
            
            if fields:
                log_data.update(fields)
            
            span.log_kv(log_data)
            
        except Exception as e:
            logger.error(f"Error adding span log: {str(e)}")
    
    def add_span_error(self, span: Any, error: Exception) -> None:
        """Add error information to a span"""
        try:
            span.set_tag(tags.ERROR, True)
            span.set_tag('error.type', type(error).__name__)
            span.set_tag('error.message', str(error))
            
            self.add_span_log(span, 'error', {
                'error.type': type(error).__name__,
                'error.message': str(error),
                'error.stack': str(error.__traceback__)
            })
            
        except Exception as e:
            logger.error(f"Error adding span error: {str(e)}")
    
    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        parent_span: Optional[Any] = None,
        tags: Optional[Dict[str, Union[str, int, float, bool]]] = None
    ):
        """Context manager for tracing operations"""
        span = None
        try:
            span = self.start_span(operation_name, parent_span, tags)
            yield span
        except Exception as e:
            if span:
                self.add_span_error(span, e)
            raise
        finally:
            if span:
                self.finish_span(span)
    
    def trace_function(
        self,
        operation_name: Optional[str] = None,
        tags: Optional[Dict[str, Union[str, int, float, bool]]] = None
    ):
        """Decorator for tracing functions"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                with self.trace_operation(op_name, tags=tags) as span:
                    try:
                        result = await func(*args, **kwargs)
                        self.add_span_tag(span, 'success', True)
                        return result
                    except Exception as e:
                        self.add_span_error(span, e)
                        raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                with self.trace_operation(op_name, tags=tags) as span:
                    try:
                        result = func(*args, **kwargs)
                        self.add_span_tag(span, 'success', True)
                        return result
                    except Exception as e:
                        self.add_span_error(span, e)
                        raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def trace_smart_meter_operation(
        self,
        operation: str,
        meter_id: str,
        parent_span: Optional[Any] = None
    ) -> Any:
        """Trace smart meter operations"""
        tags = {
            'component': 'smart_meter',
            'operation': operation,
            'meter_id': meter_id,
            'service': self.service_name
        }
        
        return self.start_span(f"smart_meter.{operation}", parent_span, tags)
    
    def trace_grid_operator_operation(
        self,
        operation: str,
        operator_id: str,
        parent_span: Optional[Any] = None
    ) -> Any:
        """Trace grid operator operations"""
        tags = {
            'component': 'grid_operator',
            'operation': operation,
            'operator_id': operator_id,
            'service': self.service_name
        }
        
        return self.start_span(f"grid_operator.{operation}", parent_span, tags)
    
    def trace_weather_operation(
        self,
        operation: str,
        station_id: str,
        parent_span: Optional[Any] = None
    ) -> Any:
        """Trace weather station operations"""
        tags = {
            'component': 'weather_station',
            'operation': operation,
            'station_id': station_id,
            'service': self.service_name
        }
        
        return self.start_span(f"weather_station.{operation}", parent_span, tags)
    
    def trace_kafka_operation(
        self,
        operation: str,
        topic: str,
        parent_span: Optional[Any] = None
    ) -> Any:
        """Trace Kafka operations"""
        tags = {
            'component': 'kafka',
            'operation': operation,
            'topic': topic,
            'service': self.service_name
        }
        
        return self.start_span(f"kafka.{operation}", parent_span, tags)
    
    def trace_database_operation(
        self,
        operation: str,
        table: str,
        parent_span: Optional[Any] = None
    ) -> Any:
        """Trace database operations"""
        tags = {
            'component': 'database',
            'operation': operation,
            'table': table,
            'service': self.service_name
        }
        
        return self.start_span(f"database.{operation}", parent_span, tags)
    
    def trace_airflow_operation(
        self,
        operation: str,
        dag_id: str,
        task_id: Optional[str] = None,
        parent_span: Optional[Any] = None
    ) -> Any:
        """Trace Airflow operations"""
        tags = {
            'component': 'airflow',
            'operation': operation,
            'dag_id': dag_id,
            'service': self.service_name
        }
        
        if task_id:
            tags['task_id'] = task_id
        
        return self.start_span(f"airflow.{operation}", parent_span, tags)
    
    def trace_s3_operation(
        self,
        operation: str,
        bucket: str,
        key: Optional[str] = None,
        parent_span: Optional[Any] = None
    ) -> Any:
        """Trace S3 operations"""
        tags = {
            'component': 's3',
            'operation': operation,
            'bucket': bucket,
            'service': self.service_name
        }
        
        if key:
            tags['key'] = key
        
        return self.start_span(f"s3.{operation}", parent_span, tags)
    
    def trace_snowflake_operation(
        self,
        operation: str,
        warehouse: str,
        parent_span: Optional[Any] = None
    ) -> Any:
        """Trace Snowflake operations"""
        tags = {
            'component': 'snowflake',
            'operation': operation,
            'warehouse': warehouse,
            'service': self.service_name
        }
        
        return self.start_span(f"snowflake.{operation}", parent_span, tags)
    
    def inject_span_context(self, span: Any, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject span context into headers for propagation"""
        try:
            tracer = self.get_tracer()
            tracer.inject(span.context, Format.HTTP_HEADERS, headers)
            return headers
        except Exception as e:
            logger.error(f"Error injecting span context: {str(e)}")
            return headers
    
    def extract_span_context(self, headers: Dict[str, str]) -> Optional[Any]:
        """Extract span context from headers"""
        try:
            tracer = self.get_tracer()
            return tracer.extract(Format.HTTP_HEADERS, headers)
        except Exception as e:
            logger.error(f"Error extracting span context: {str(e)}")
            return None
    
    def create_span_from_context(self, operation_name: str, context: Any) -> Any:
        """Create a span from extracted context"""
        try:
            return self.start_span(operation_name, child_of=context)
        except Exception as e:
            logger.error(f"Error creating span from context: {str(e)}")
            return None
    
    def close(self) -> None:
        """Close the Jaeger client and flush remaining spans"""
        try:
            if self.tracer:
                self.tracer.close()
                logger.info("Jaeger client closed")
        except Exception as e:
            logger.error(f"Error closing Jaeger client: {str(e)}")
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get Jaeger client information"""
        return {
            "service_name": self.service_name,
            "agent_host": self.agent_host,
            "agent_port": self.agent_port,
            "collector_endpoint": self.collector_endpoint,
            "sampling_rate": self.sampling_rate,
            "initialized": self._initialized
        }


# Import asyncio for the decorator
import asyncio
