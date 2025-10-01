"""
Stream Joins
Advanced stream joining capabilities for real-time data processing
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import threading
import json

logger = logging.getLogger(__name__)

class JoinType(Enum):
    """Stream join types"""
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL_OUTER = "full_outer"
    CROSS = "cross"

class WindowType(Enum):
    """Window types for stream joins"""
    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"

@dataclass
class JoinWindow:
    """Join window configuration"""
    window_type: WindowType
    size: int  # seconds
    slide: int = None  # seconds (for sliding windows)
    gap: int = None  # seconds (for session windows)

@dataclass
class StreamRecord:
    """Stream record with metadata"""
    key: str
    value: Any
    timestamp: datetime
    source: str
    partition: int = 0
    offset: int = 0

@dataclass
class JoinResult:
    """Join result"""
    left_record: StreamRecord
    right_record: StreamRecord
    join_timestamp: datetime
    join_key: str
    metadata: Dict[str, Any] = None

@dataclass
class JoinMetrics:
    """Join operation metrics"""
    total_joins: int = 0
    successful_joins: int = 0
    failed_joins: int = 0
    left_records_processed: int = 0
    right_records_processed: int = 0
    join_latency: float = 0.0  # average latency in ms
    throughput: float = 0.0  # joins per second

class StreamJoiner:
    """
    Advanced stream joining with multiple join types and windowing
    """
    
    def __init__(self, 
                 join_type: JoinType = JoinType.INNER,
                 window: JoinWindow = None,
                 key_extractor: Callable[[StreamRecord], str] = None,
                 max_window_size: int = 10000):
        self.join_type = join_type
        self.window = window or JoinWindow(WindowType.TUMBLING, 300)  # 5 minutes default
        self.key_extractor = key_extractor or self._default_key_extractor
        self.max_window_size = max_window_size
        
        # Stream buffers
        self.left_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_window_size))
        self.right_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_window_size))
        
        # Metrics
        self.metrics = JoinMetrics()
        self.metrics_lock = threading.RLock()
        
        # Threading
        self.join_lock = threading.RLock()
        
        logger.info(f"StreamJoiner initialized with {join_type.value} join and {window.window_type.value} window")
    
    def add_left_record(self, record: StreamRecord) -> List[JoinResult]:
        """Add record to left stream and perform joins"""
        try:
            join_key = self.key_extractor(record)
            
            with self.join_lock:
                # Add to left buffer
                self.left_buffer[join_key].append(record)
                
                # Perform joins
                join_results = self._perform_joins(join_key, "left")
                
                # Update metrics
                with self.metrics_lock:
                    self.metrics.left_records_processed += 1
                    self.metrics.total_joins += len(join_results)
                    self.metrics.successful_joins += len(join_results)
            
            logger.debug(f"Added left record for key {join_key}, generated {len(join_results)} joins")
            return join_results
            
        except Exception as e:
            logger.error(f"Failed to add left record: {str(e)}")
            with self.metrics_lock:
                self.metrics.failed_joins += 1
            return []
    
    def add_right_record(self, record: StreamRecord) -> List[JoinResult]:
        """Add record to right stream and perform joins"""
        try:
            join_key = self.key_extractor(record)
            
            with self.join_lock:
                # Add to right buffer
                self.right_buffer[join_key].append(record)
                
                # Perform joins
                join_results = self._perform_joins(join_key, "right")
                
                # Update metrics
                with self.metrics_lock:
                    self.metrics.right_records_processed += 1
                    self.metrics.total_joins += len(join_results)
                    self.metrics.successful_joins += len(join_results)
            
            logger.debug(f"Added right record for key {join_key}, generated {len(join_results)} joins")
            return join_results
            
        except Exception as e:
            logger.error(f"Failed to add right record: {str(e)}")
            with self.metrics_lock:
                self.metrics.failed_joins += 1
            return []
    
    def _perform_joins(self, join_key: str, side: str) -> List[JoinResult]:
        """Perform joins for a specific key"""
        try:
            join_results = []
            
            left_records = list(self.left_buffer[join_key])
            right_records = list(self.right_buffer[join_key])
            
            if not left_records or not right_records:
                return join_results
            
            # Apply window filtering
            left_records = self._filter_by_window(left_records)
            right_records = self._filter_by_window(right_records)
            
            if not left_records or not right_records:
                return join_results
            
            # Perform join based on type
            if self.join_type == JoinType.INNER:
                join_results = self._inner_join(left_records, right_records, join_key)
            elif self.join_type == JoinType.LEFT:
                join_results = self._left_join(left_records, right_records, join_key)
            elif self.join_type == JoinType.RIGHT:
                join_results = self._right_join(left_records, right_records, join_key)
            elif self.join_type == JoinType.FULL_OUTER:
                join_results = self._full_outer_join(left_records, right_records, join_key)
            elif self.join_type == JoinType.CROSS:
                join_results = self._cross_join(left_records, right_records, join_key)
            
            # Update latency metrics
            if join_results:
                with self.metrics_lock:
                    current_time = time.time()
                    total_latency = sum(
                        (current_time - r.left_record.timestamp.timestamp()) * 1000
                        for r in join_results
                    )
                    self.metrics.join_latency = total_latency / len(join_results)
            
            return join_results
            
        except Exception as e:
            logger.error(f"Failed to perform joins for key {join_key}: {str(e)}")
            return []
    
    def _inner_join(self, left_records: List[StreamRecord], right_records: List[StreamRecord], join_key: str) -> List[JoinResult]:
        """Perform inner join"""
        try:
            join_results = []
            
            for left_record in left_records:
                for right_record in right_records:
                    if self._records_match(left_record, right_record):
                        join_result = JoinResult(
                            left_record=left_record,
                            right_record=right_record,
                            join_timestamp=datetime.now(),
                            join_key=join_key,
                            metadata={"join_type": "inner"}
                        )
                        join_results.append(join_result)
            
            return join_results
            
        except Exception as e:
            logger.error(f"Failed to perform inner join: {str(e)}")
            return []
    
    def _left_join(self, left_records: List[StreamRecord], right_records: List[StreamRecord], join_key: str) -> List[JoinResult]:
        """Perform left join"""
        try:
            join_results = []
            
            for left_record in left_records:
                matched = False
                for right_record in right_records:
                    if self._records_match(left_record, right_record):
                        join_result = JoinResult(
                            left_record=left_record,
                            right_record=right_record,
                            join_timestamp=datetime.now(),
                            join_key=join_key,
                            metadata={"join_type": "left"}
                        )
                        join_results.append(join_result)
                        matched = True
                
                # If no match found, create result with None right record
                if not matched:
                    join_result = JoinResult(
                        left_record=left_record,
                        right_record=None,
                        join_timestamp=datetime.now(),
                        join_key=join_key,
                        metadata={"join_type": "left", "no_match": True}
                    )
                    join_results.append(join_result)
            
            return join_results
            
        except Exception as e:
            logger.error(f"Failed to perform left join: {str(e)}")
            return []
    
    def _right_join(self, left_records: List[StreamRecord], right_records: List[StreamRecord], join_key: str) -> List[JoinResult]:
        """Perform right join"""
        try:
            join_results = []
            
            for right_record in right_records:
                matched = False
                for left_record in left_records:
                    if self._records_match(left_record, right_record):
                        join_result = JoinResult(
                            left_record=left_record,
                            right_record=right_record,
                            join_timestamp=datetime.now(),
                            join_key=join_key,
                            metadata={"join_type": "right"}
                        )
                        join_results.append(join_result)
                        matched = True
                
                # If no match found, create result with None left record
                if not matched:
                    join_result = JoinResult(
                        left_record=None,
                        right_record=right_record,
                        join_timestamp=datetime.now(),
                        join_key=join_key,
                        metadata={"join_type": "right", "no_match": True}
                    )
                    join_results.append(join_result)
            
            return join_results
            
        except Exception as e:
            logger.error(f"Failed to perform right join: {str(e)}")
            return []
    
    def _full_outer_join(self, left_records: List[StreamRecord], right_records: List[StreamRecord], join_key: str) -> List[JoinResult]:
        """Perform full outer join"""
        try:
            join_results = []
            
            # Track which records have been matched
            left_matched = set()
            right_matched = set()
            
            # Find all matches
            for i, left_record in enumerate(left_records):
                for j, right_record in enumerate(right_records):
                    if self._records_match(left_record, right_record):
                        join_result = JoinResult(
                            left_record=left_record,
                            right_record=right_record,
                            join_timestamp=datetime.now(),
                            join_key=join_key,
                            metadata={"join_type": "full_outer"}
                        )
                        join_results.append(join_result)
                        left_matched.add(i)
                        right_matched.add(j)
            
            # Add unmatched left records
            for i, left_record in enumerate(left_records):
                if i not in left_matched:
                    join_result = JoinResult(
                        left_record=left_record,
                        right_record=None,
                        join_timestamp=datetime.now(),
                        join_key=join_key,
                        metadata={"join_type": "full_outer", "no_match": True}
                    )
                    join_results.append(join_result)
            
            # Add unmatched right records
            for j, right_record in enumerate(right_records):
                if j not in right_matched:
                    join_result = JoinResult(
                        left_record=None,
                        right_record=right_record,
                        join_timestamp=datetime.now(),
                        join_key=join_key,
                        metadata={"join_type": "full_outer", "no_match": True}
                    )
                    join_results.append(join_result)
            
            return join_results
            
        except Exception as e:
            logger.error(f"Failed to perform full outer join: {str(e)}")
            return []
    
    def _cross_join(self, left_records: List[StreamRecord], right_records: List[StreamRecord], join_key: str) -> List[JoinResult]:
        """Perform cross join (cartesian product)"""
        try:
            join_results = []
            
            for left_record in left_records:
                for right_record in right_records:
                    join_result = JoinResult(
                        left_record=left_record,
                        right_record=right_record,
                        join_timestamp=datetime.now(),
                        join_key=join_key,
                        metadata={"join_type": "cross"}
                    )
                    join_results.append(join_result)
            
            return join_results
            
        except Exception as e:
            logger.error(f"Failed to perform cross join: {str(e)}")
            return []
    
    def _records_match(self, left_record: StreamRecord, right_record: StreamRecord) -> bool:
        """Check if two records match for joining"""
        try:
            # Simple key-based matching
            # In real implementation, this could be more complex
            return left_record.key == right_record.key
            
        except Exception as e:
            logger.error(f"Failed to check record match: {str(e)}")
            return False
    
    def _filter_by_window(self, records: List[StreamRecord]) -> List[StreamRecord]:
        """Filter records by window configuration"""
        try:
            if not records:
                return records
            
            current_time = datetime.now()
            
            if self.window.window_type == WindowType.TUMBLING:
                # Keep records within the window size
                cutoff_time = current_time - timedelta(seconds=self.window.size)
                return [r for r in records if r.timestamp >= cutoff_time]
            
            elif self.window.window_type == WindowType.SLIDING:
                # Keep records within the window size
                cutoff_time = current_time - timedelta(seconds=self.window.size)
                return [r for r in records if r.timestamp >= cutoff_time]
            
            elif self.window.window_type == WindowType.SESSION:
                # Keep records within the session gap
                if self.window.gap:
                    cutoff_time = current_time - timedelta(seconds=self.window.gap)
                    return [r for r in records if r.timestamp >= cutoff_time]
                return records
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to filter records by window: {str(e)}")
            return records
    
    def _default_key_extractor(self, record: StreamRecord) -> str:
        """Default key extractor"""
        try:
            return record.key
            
        except Exception as e:
            logger.error(f"Failed to extract key from record: {str(e)}")
            return ""
    
    def cleanup_expired_records(self) -> int:
        """Clean up expired records from buffers"""
        try:
            cleaned_count = 0
            
            with self.join_lock:
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(seconds=self.window.size * 2)  # Keep 2x window size
                
                for join_key in list(self.left_buffer.keys()):
                    # Remove expired records
                    original_size = len(self.left_buffer[join_key])
                    self.left_buffer[join_key] = deque(
                        [r for r in self.left_buffer[join_key] if r.timestamp >= cutoff_time],
                        maxlen=self.max_window_size
                    )
                    cleaned_count += original_size - len(self.left_buffer[join_key])
                    
                    # Remove empty buffers
                    if not self.left_buffer[join_key]:
                        del self.left_buffer[join_key]
                
                for join_key in list(self.right_buffer.keys()):
                    # Remove expired records
                    original_size = len(self.right_buffer[join_key])
                    self.right_buffer[join_key] = deque(
                        [r for r in self.right_buffer[join_key] if r.timestamp >= cutoff_time],
                        maxlen=self.max_window_size
                    )
                    cleaned_count += original_size - len(self.right_buffer[join_key])
                    
                    # Remove empty buffers
                    if not self.right_buffer[join_key]:
                        del self.right_buffer[join_key]
            
            logger.info(f"Cleaned up {cleaned_count} expired records")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired records: {str(e)}")
            return 0
    
    def get_metrics(self) -> JoinMetrics:
        """Get join operation metrics"""
        try:
            with self.metrics_lock:
                # Calculate throughput
                current_time = time.time()
                if hasattr(self, '_last_metrics_time'):
                    time_diff = current_time - self._last_metrics_time
                    if time_diff > 0:
                        self.metrics.throughput = self.metrics.total_joins / time_diff
                
                self._last_metrics_time = current_time
                
                return JoinMetrics(
                    total_joins=self.metrics.total_joins,
                    successful_joins=self.metrics.successful_joins,
                    failed_joins=self.metrics.failed_joins,
                    left_records_processed=self.metrics.left_records_processed,
                    right_records_processed=self.metrics.right_records_processed,
                    join_latency=self.metrics.join_latency,
                    throughput=self.metrics.throughput
                )
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return JoinMetrics()
    
    def reset_metrics(self) -> None:
        """Reset join metrics"""
        try:
            with self.metrics_lock:
                self.metrics = JoinMetrics()
                self._last_metrics_time = time.time()
                
        except Exception as e:
            logger.error(f"Failed to reset metrics: {str(e)}")
    
    def get_buffer_sizes(self) -> Dict[str, int]:
        """Get current buffer sizes"""
        try:
            with self.join_lock:
                return {
                    "left_buffer_keys": len(self.left_buffer),
                    "right_buffer_keys": len(self.right_buffer),
                    "total_left_records": sum(len(buf) for buf in self.left_buffer.values()),
                    "total_right_records": sum(len(buf) for buf in self.right_buffer.values())
                }
                
        except Exception as e:
            logger.error(f"Failed to get buffer sizes: {str(e)}")
            return {}
    
    def clear_buffers(self) -> None:
        """Clear all buffers"""
        try:
            with self.join_lock:
                self.left_buffer.clear()
                self.right_buffer.clear()
                logger.info("Cleared all join buffers")
                
        except Exception as e:
            logger.error(f"Failed to clear buffers: {str(e)}")
    
    def set_join_type(self, join_type: JoinType) -> None:
        """Change join type"""
        try:
            self.join_type = join_type
            logger.info(f"Changed join type to {join_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to change join type: {str(e)}")
    
    def set_window(self, window: JoinWindow) -> None:
        """Change window configuration"""
        try:
            self.window = window
            logger.info(f"Changed window to {window.window_type.value} with size {window.size}s")
            
        except Exception as e:
            logger.error(f"Failed to change window: {str(e)}")
    
    def set_key_extractor(self, key_extractor: Callable[[StreamRecord], str]) -> None:
        """Change key extractor function"""
        try:
            self.key_extractor = key_extractor
            logger.info("Changed key extractor function")
            
        except Exception as e:
            logger.error(f"Failed to change key extractor: {str(e)}")
