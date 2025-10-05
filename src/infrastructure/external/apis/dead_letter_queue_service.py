"""
Dead Letter Queue Service
Handles failed data processing and provides retry mechanisms
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict

from ....core.exceptions.domain_exceptions import DataQualityError, InfrastructureError

logger = logging.getLogger(__name__)


class FailureReason(Enum):
    """Reasons for data processing failure"""
    VALIDATION_FAILED = "validation_failed"
    QUALITY_THRESHOLD_BREACHED = "quality_threshold_breached"
    ANOMALY_DETECTED = "anomaly_detected"
    PROCESSING_ERROR = "processing_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategies for failed data"""
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"
    MANUAL = "manual"


@dataclass
class FailedRecord:
    """Represents a failed data record"""
    record_id: str
    data_type: str
    original_data: Dict[str, Any]
    failure_reason: FailureReason
    error_message: str
    failure_timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    next_retry_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetryResult:
    """Result of a retry attempt"""
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0
    next_retry_at: Optional[datetime] = None


class DeadLetterQueueService:
    """
    Manages failed data records and provides retry mechanisms
    """
    
    def __init__(self, max_retries: int = 3, retry_delay_seconds: int = 60):
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        
        # In-memory storage for failed records
        self.failed_records: Dict[str, FailedRecord] = {}
        self.retry_queue: List[str] = []  # Queue of record IDs to retry
        self.processing_callbacks: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            'total_failures': 0,
            'total_retries': 0,
            'successful_retries': 0,
            'permanent_failures': 0,
            'by_reason': defaultdict(int),
            'by_data_type': defaultdict(int)
        }
        
        # Start background retry processor
        self._retry_task = None
        self._start_retry_processor()
    
    def _start_retry_processor(self):
        """Start background task for processing retries"""
        if self._retry_task is None or self._retry_task.done():
            self._retry_task = asyncio.create_task(self._process_retries())
    
    async def add_failed_record(self, data_type: str, original_data: Dict[str, Any], 
                               failure_reason: FailureReason, error_message: str,
                               retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a failed record to the dead letter queue"""
        try:
            record_id = str(uuid.uuid4())
            
            # Calculate next retry time based on strategy
            next_retry_at = self._calculate_next_retry_time(retry_strategy, 0)
            
            failed_record = FailedRecord(
                record_id=record_id,
                data_type=data_type,
                original_data=original_data,
                failure_reason=failure_reason,
                error_message=error_message,
                failure_timestamp=datetime.utcnow(),
                retry_count=0,
                max_retries=self.max_retries,
                retry_strategy=retry_strategy,
                next_retry_at=next_retry_at,
                metadata=metadata or {}
            )
            
            # Store the failed record
            self.failed_records[record_id] = failed_record
            
            # Add to retry queue if not manual strategy
            if retry_strategy != RetryStrategy.MANUAL:
                self.retry_queue.append(record_id)
            
            # Update statistics
            self.stats['total_failures'] += 1
            self.stats['by_reason'][failure_reason.value] += 1
            self.stats['by_data_type'][data_type] += 1
            
            logger.warning(f"Added failed record {record_id} to DLQ: {failure_reason.value} - {error_message}")
            
            return record_id
            
        except Exception as e:
            logger.error(f"Error adding failed record to DLQ: {e}")
            raise InfrastructureError(f"Failed to add record to dead letter queue: {e}")
    
    def _calculate_next_retry_time(self, strategy: RetryStrategy, retry_count: int) -> datetime:
        """Calculate next retry time based on strategy and retry count"""
        now = datetime.utcnow()
        
        if strategy == RetryStrategy.IMMEDIATE:
            return now
        elif strategy == RetryStrategy.FIXED_INTERVAL:
            return now + timedelta(seconds=self.retry_delay_seconds)
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.retry_delay_seconds * (2 ** retry_count)
            return now + timedelta(seconds=min(delay, 3600))  # Max 1 hour
        else:  # MANUAL
            return None
    
    async def _process_retries(self):
        """Background task to process retry queue"""
        while True:
            try:
                current_time = datetime.utcnow()
                records_to_retry = []
                
                # Find records ready for retry
                for record_id in self.retry_queue[:]:
                    if record_id in self.failed_records:
                        record = self.failed_records[record_id]
                        
                        if (record.next_retry_at and 
                            record.next_retry_at <= current_time and 
                            record.retry_count < record.max_retries):
                            records_to_retry.append(record_id)
                        elif record.retry_count >= record.max_retries:
                            # Remove from retry queue if max retries reached
                            self.retry_queue.remove(record_id)
                            self.stats['permanent_failures'] += 1
                            logger.error(f"Record {record_id} permanently failed after {record.max_retries} retries")
                
                # Process retries
                for record_id in records_to_retry:
                    await self._retry_record(record_id)
                
                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in retry processor: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _retry_record(self, record_id: str) -> RetryResult:
        """Retry processing a failed record"""
        try:
            if record_id not in self.failed_records:
                return RetryResult(success=False, error_message="Record not found")
            
            record = self.failed_records[record_id]
            
            # Check if we have a processing callback for this data type
            if record.data_type not in self.processing_callbacks:
                logger.warning(f"No processing callback registered for data type: {record.data_type}")
                return RetryResult(success=False, error_message="No processing callback available")
            
            # Attempt to process the record
            callback = self.processing_callbacks[record.data_type]
            success = await callback(record.original_data)
            
            if success:
                # Success - remove from DLQ
                del self.failed_records[record_id]
                if record_id in self.retry_queue:
                    self.retry_queue.remove(record_id)
                
                self.stats['successful_retries'] += 1
                logger.info(f"Successfully retried record {record_id}")
                
                return RetryResult(success=True, retry_count=record.retry_count)
            else:
                # Failed again - update retry count and schedule next retry
                record.retry_count += 1
                self.stats['total_retries'] += 1
                
                if record.retry_count < record.max_retries:
                    # Schedule next retry
                    record.next_retry_at = self._calculate_next_retry_time(
                        record.retry_strategy, record.retry_count
                    )
                    logger.warning(f"Retry {record.retry_count} failed for record {record_id}, scheduling next retry")
                else:
                    # Max retries reached - remove from retry queue
                    if record_id in self.retry_queue:
                        self.retry_queue.remove(record_id)
                    self.stats['permanent_failures'] += 1
                    logger.error(f"Record {record_id} permanently failed after {record.max_retries} retries")
                
                return RetryResult(
                    success=False, 
                    retry_count=record.retry_count,
                    next_retry_at=record.next_retry_at
                )
                
        except Exception as e:
            logger.error(f"Error retrying record {record_id}: {e}")
            return RetryResult(success=False, error_message=str(e))
    
    def register_processing_callback(self, data_type: str, callback: Callable):
        """Register a callback function for processing specific data types"""
        self.processing_callbacks[data_type] = callback
        logger.info(f"Registered processing callback for data type: {data_type}")
    
    async def get_failed_records(self, data_type: Optional[str] = None, 
                                failure_reason: Optional[FailureReason] = None,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """Get failed records with optional filtering"""
        try:
            records = list(self.failed_records.values())
            
            # Apply filters
            if data_type:
                records = [r for r in records if r.data_type == data_type]
            
            if failure_reason:
                records = [r for r in records if r.failure_reason == failure_reason]
            
            # Sort by failure timestamp (newest first)
            records.sort(key=lambda x: x.failure_timestamp, reverse=True)
            
            # Apply limit
            records = records[:limit]
            
            return [asdict(record) for record in records]
            
        except Exception as e:
            logger.error(f"Error getting failed records: {e}")
            return []
    
    async def get_dlq_statistics(self) -> Dict[str, Any]:
        """Get dead letter queue statistics"""
        try:
            total_records = len(self.failed_records)
            retry_queue_size = len(self.retry_queue)
            
            # Count by retry count
            retry_counts = defaultdict(int)
            for record in self.failed_records.values():
                retry_counts[record.retry_count] += 1
            
            # Count by failure reason
            reason_counts = defaultdict(int)
            for record in self.failed_records.values():
                reason_counts[record.failure_reason.value] += 1
            
            # Count by data type
            data_type_counts = defaultdict(int)
            for record in self.failed_records.values():
                data_type_counts[record.data_type] += 1
            
            return {
                'total_failed_records': total_records,
                'retry_queue_size': retry_queue_size,
                'statistics': self.stats,
                'retry_distribution': dict(retry_counts),
                'failure_reason_distribution': dict(reason_counts),
                'data_type_distribution': dict(data_type_counts),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting DLQ statistics: {e}")
            return {'error': str(e)}
    
    async def manually_retry_record(self, record_id: str) -> RetryResult:
        """Manually retry a specific record"""
        try:
            if record_id not in self.failed_records:
                return RetryResult(success=False, error_message="Record not found")
            
            return await self._retry_record(record_id)
            
        except Exception as e:
            logger.error(f"Error manually retrying record {record_id}: {e}")
            return RetryResult(success=False, error_message=str(e))
    
    async def remove_record(self, record_id: str) -> bool:
        """Remove a record from the dead letter queue"""
        try:
            if record_id in self.failed_records:
                del self.failed_records[record_id]
                if record_id in self.retry_queue:
                    self.retry_queue.remove(record_id)
                logger.info(f"Removed record {record_id} from DLQ")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing record {record_id}: {e}")
            return False
    
    async def clear_old_records(self, days: int = 7) -> int:
        """Clear records older than specified days"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            old_records = [
                record_id for record_id, record in self.failed_records.items()
                if record.failure_timestamp < cutoff_time
            ]
            
            for record_id in old_records:
                await self.remove_record(record_id)
            
            logger.info(f"Cleared {len(old_records)} old records from DLQ")
            return len(old_records)
            
        except Exception as e:
            logger.error(f"Error clearing old records: {e}")
            return 0
    
    async def export_dlq_data(self, data_type: Optional[str] = None) -> Dict[str, Any]:
        """Export dead letter queue data for analysis"""
        try:
            records = await self.get_failed_records(data_type=data_type, limit=1000)
            statistics = await self.get_dlq_statistics()
            
            return {
                'export_info': {
                    'exported_at': datetime.utcnow().isoformat(),
                    'data_type_filter': data_type,
                    'record_count': len(records)
                },
                'records': records,
                'statistics': statistics
            }
            
        except Exception as e:
            logger.error(f"Error exporting DLQ data: {e}")
            return {'error': str(e)}
