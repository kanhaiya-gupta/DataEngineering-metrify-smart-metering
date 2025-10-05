"""
Optimized Batch Processor

Provides high-performance batch processing with adaptive sizing, parallel processing,
and resource optimization for the data engineering pipeline.
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import json
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    optimal_batch_size: int
    max_batch_size: int
    min_batch_size: int
    batch_timeout: int  # seconds
    parallel_batches: int
    memory_limit_mb: int
    cpu_limit_percent: float
    processing_timeout: int  # seconds
    retry_attempts: int
    backoff_factor: float

@dataclass
class BatchResult:
    """Result of batch processing"""
    batch_id: str
    records_processed: int
    processing_time: float
    success: bool
    errors: int
    memory_used_mb: float
    cpu_usage_percent: float
    throughput: float  # records/second
    error_message: Optional[str] = None

@dataclass
class PerformanceStats:
    """Performance statistics for optimization"""
    avg_processing_time: float
    avg_throughput: float
    avg_memory_usage: float
    avg_cpu_usage: float
    error_rate: float
    optimal_batch_size: int
    sample_count: int

class OptimizedBatchProcessor:
    """
    High-performance batch processor with adaptive optimization
    
    Features:
    - Adaptive batch sizing based on performance metrics
    - Parallel processing with resource limits
    - Memory and CPU optimization
    - Automatic retry with exponential backoff
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: BatchConfig, performance_monitor=None):
        self.config = config
        self.performance_monitor = performance_monitor
        self.performance_stats: Dict[str, PerformanceStats] = {}
        self.adaptive_batch_sizes: Dict[str, int] = {}
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_threads: List[threading.Thread] = []
        self.processing_active = False
        self.stats_lock = threading.Lock()
        
        # Performance tracking
        self.batch_history: Dict[str, List[BatchResult]] = {}
        self.optimization_enabled = True
        self.optimization_window = 300  # seconds
        
        # Resource monitoring
        self.memory_warning_threshold = 0.8  # 80% of limit
        self.cpu_warning_threshold = 0.8  # 80% of limit
        
    def start_processing(self, num_workers: int = None):
        """Start batch processing workers"""
        if self.processing_active:
            logger.warning("Batch processing is already active")
            return
            
        if num_workers is None:
            num_workers = min(self.config.parallel_batches, psutil.cpu_count())
        
        self.processing_active = True
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"BatchWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        logger.info(f"Started batch processing with {num_workers} workers")
    
    def stop_processing(self):
        """Stop batch processing"""
        self.processing_active = False
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=10)
        
        self.worker_threads.clear()
        logger.info("Stopped batch processing")
    
    def _worker_loop(self):
        """Main worker loop"""
        while self.processing_active:
            try:
                # Get batch from queue with timeout
                try:
                    batch_data = self.processing_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process batch
                result = self._process_batch(batch_data)
                
                # Put result in result queue
                self.result_queue.put(result)
                
                # Update performance stats
                self._update_performance_stats(result)
                
                # Optimize batch size if enabled
                if self.optimization_enabled:
                    self._optimize_batch_size(result)
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1)
    
    async def process_data_async(self, data_type: str, data: List[Dict[str, Any]], 
                               processor_func: Callable) -> BatchResult:
        """Process data asynchronously with optimization"""
        try:
            # Get optimal batch size for data type
            batch_size = self._get_optimal_batch_size(data_type)
            
            # Split data into batches
            batches = self._split_into_batches(data, batch_size)
            
            # Process batches in parallel
            tasks = []
            for i, batch in enumerate(batches):
                batch_id = f"{data_type}_batch_{i}_{int(time.time())}"
                task = self._process_batch_async(batch_id, batch, processor_func)
                tasks.append(task)
            
            # Wait for all batches to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            total_records = sum(r.records_processed if isinstance(r, BatchResult) else 0 for r in results)
            total_errors = sum(r.errors if isinstance(r, BatchResult) else 0 for r in results)
            total_time = max(r.processing_time if isinstance(r, BatchResult) else 0 for r in results)
            
            # Calculate overall throughput
            throughput = total_records / total_time if total_time > 0 else 0
            
            # Record performance metrics
            if self.performance_monitor:
                self.performance_monitor.record_processing_event(
                    component=data_type,
                    records_processed=total_records,
                    processing_time=total_time,
                    errors=total_errors
                )
            
            return BatchResult(
                batch_id=f"{data_type}_aggregated_{int(time.time())}",
                records_processed=total_records,
                processing_time=total_time,
                success=total_errors == 0,
                errors=total_errors,
                memory_used_mb=psutil.virtual_memory().used / (1024 * 1024),
                cpu_usage_percent=psutil.cpu_percent(),
                throughput=throughput
            )
            
        except Exception as e:
            logger.error(f"Error in async batch processing: {e}")
            return BatchResult(
                batch_id=f"{data_type}_error_{int(time.time())}",
                records_processed=0,
                processing_time=0,
                success=False,
                errors=1,
                memory_used_mb=0,
                cpu_usage_percent=0,
                throughput=0,
                error_message=str(e)
            )
    
    async def _process_batch_async(self, batch_id: str, batch_data: List[Dict[str, Any]], 
                                 processor_func: Callable) -> BatchResult:
        """Process a single batch asynchronously"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        start_cpu = psutil.cpu_percent()
        
        try:
            # Check resource limits before processing
            if not self._check_resource_limits():
                raise Exception("Resource limits exceeded")
            
            # Process batch
            result = await processor_func(batch_data)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            memory_used = (psutil.virtual_memory().used - start_memory) / (1024 * 1024)
            cpu_usage = psutil.cpu_percent() - start_cpu
            throughput = len(batch_data) / processing_time if processing_time > 0 else 0
            
            return BatchResult(
                batch_id=batch_id,
                records_processed=len(batch_data),
                processing_time=processing_time,
                success=True,
                errors=0,
                memory_used_mb=memory_used,
                cpu_usage_percent=cpu_usage,
                throughput=throughput
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing batch {batch_id}: {e}")
            
            return BatchResult(
                batch_id=batch_id,
                records_processed=0,
                processing_time=processing_time,
                success=False,
                errors=1,
                memory_used_mb=0,
                cpu_usage_percent=0,
                throughput=0,
                error_message=str(e)
            )
    
    def _process_batch(self, batch_data: Dict[str, Any]) -> BatchResult:
        """Process a single batch (synchronous)"""
        batch_id = batch_data.get('batch_id')
        data = batch_data.get('data', [])
        processor_func = batch_data.get('processor_func')
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        try:
            # Check resource limits
            if not self._check_resource_limits():
                raise Exception("Resource limits exceeded")
            
            # Process batch
            if processor_func:
                result = processor_func(data)
            else:
                result = data  # Default: return data as-is
            
            # Calculate metrics
            processing_time = time.time() - start_time
            memory_used = (psutil.virtual_memory().used - start_memory) / (1024 * 1024)
            cpu_usage = psutil.cpu_percent()
            throughput = len(data) / processing_time if processing_time > 0 else 0
            
            return BatchResult(
                batch_id=batch_id,
                records_processed=len(data),
                processing_time=processing_time,
                success=True,
                errors=0,
                memory_used_mb=memory_used,
                cpu_usage_percent=cpu_usage,
                throughput=throughput
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing batch {batch_id}: {e}")
            
            return BatchResult(
                batch_id=batch_id,
                records_processed=0,
                processing_time=processing_time,
                success=False,
                errors=1,
                memory_used_mb=0,
                cpu_usage_percent=0,
                throughput=0,
                error_message=str(e)
            )
    
    def _split_into_batches(self, data: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        """Split data into batches of specified size"""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def _get_optimal_batch_size(self, data_type: str) -> int:
        """Get optimal batch size for data type"""
        if data_type in self.adaptive_batch_sizes:
            return self.adaptive_batch_sizes[data_type]
        
        # Use configured optimal batch size as default
        return self.config.optimal_batch_size
    
    def _check_resource_limits(self) -> bool:
        """Check if current resource usage is within limits"""
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / (1024 * 1024)
            if memory_usage_mb > self.config.memory_limit_mb:
                logger.warning(f"Memory usage {memory_usage_mb:.1f}MB exceeds limit {self.config.memory_limit_mb}MB")
                return False
            
            # Check CPU usage
            cpu_usage = psutil.cpu_percent()
            if cpu_usage > self.config.cpu_limit_percent:
                logger.warning(f"CPU usage {cpu_usage:.1f}% exceeds limit {self.config.cpu_limit_percent}%")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking resource limits: {e}")
            return True  # Allow processing if check fails
    
    def _update_performance_stats(self, result: BatchResult):
        """Update performance statistics"""
        try:
            with self.stats_lock:
                data_type = result.batch_id.split('_')[0] if '_' in result.batch_id else 'unknown'
                
                if data_type not in self.batch_history:
                    self.batch_history[data_type] = []
                
                self.batch_history[data_type].append(result)
                
                # Keep only recent history
                cutoff_time = datetime.utcnow() - timedelta(seconds=self.optimization_window)
                self.batch_history[data_type] = [
                    r for r in self.batch_history[data_type]
                    if hasattr(r, 'timestamp') and r.timestamp > cutoff_time
                ]
                
                # Update performance stats
                self._calculate_performance_stats(data_type)
                
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")
    
    def _calculate_performance_stats(self, data_type: str):
        """Calculate performance statistics for data type"""
        try:
            history = self.batch_history.get(data_type, [])
            if not history:
                return
            
            # Calculate averages
            processing_times = [r.processing_time for r in history if r.processing_time > 0]
            throughputs = [r.throughput for r in history if r.throughput > 0]
            memory_usage = [r.memory_used_mb for r in history if r.memory_used_mb > 0]
            cpu_usage = [r.cpu_usage_percent for r in history if r.cpu_usage_percent > 0]
            errors = [r.errors for r in history]
            
            avg_processing_time = np.mean(processing_times) if processing_times else 0
            avg_throughput = np.mean(throughputs) if throughputs else 0
            avg_memory_usage = np.mean(memory_usage) if memory_usage else 0
            avg_cpu_usage = np.mean(cpu_usage) if cpu_usage else 0
            error_rate = sum(errors) / len(errors) if errors else 0
            
            # Calculate optimal batch size based on throughput
            optimal_batch_size = self._calculate_optimal_batch_size(history)
            
            self.performance_stats[data_type] = PerformanceStats(
                avg_processing_time=avg_processing_time,
                avg_throughput=avg_throughput,
                avg_memory_usage=avg_memory_usage,
                avg_cpu_usage=avg_cpu_usage,
                error_rate=error_rate,
                optimal_batch_size=optimal_batch_size,
                sample_count=len(history)
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
    
    def _calculate_optimal_batch_size(self, history: List[BatchResult]) -> int:
        """Calculate optimal batch size based on performance history"""
        try:
            if len(history) < 3:
                return self.config.optimal_batch_size
            
            # Group by batch size and calculate average throughput
            batch_size_performance = {}
            for result in history:
                batch_size = result.records_processed
                if batch_size not in batch_size_performance:
                    batch_size_performance[batch_size] = []
                batch_size_performance[batch_size].append(result.throughput)
            
            # Find batch size with highest average throughput
            best_batch_size = self.config.optimal_batch_size
            best_throughput = 0
            
            for batch_size, throughputs in batch_size_performance.items():
                avg_throughput = np.mean(throughputs)
                if avg_throughput > best_throughput:
                    best_throughput = avg_throughput
                    best_batch_size = batch_size
            
            # Ensure within configured limits
            best_batch_size = max(self.config.min_batch_size, 
                                min(self.config.max_batch_size, best_batch_size))
            
            return best_batch_size
            
        except Exception as e:
            logger.error(f"Error calculating optimal batch size: {e}")
            return self.config.optimal_batch_size
    
    def _optimize_batch_size(self, result: BatchResult):
        """Optimize batch size based on performance results"""
        try:
            if not self.optimization_enabled:
                return
            
            data_type = result.batch_id.split('_')[0] if '_' in result.batch_id else 'unknown'
            
            # Get current optimal batch size
            current_optimal = self.adaptive_batch_sizes.get(data_type, self.config.optimal_batch_size)
            
            # Adjust based on performance
            if result.throughput > 0:
                # If throughput is good, try increasing batch size
                if result.throughput > current_optimal * 0.8:  # 80% of optimal
                    new_size = min(self.config.max_batch_size, 
                                 int(current_optimal * 1.1))  # Increase by 10%
                else:
                    # If throughput is poor, try decreasing batch size
                    new_size = max(self.config.min_batch_size, 
                                 int(current_optimal * 0.9))  # Decrease by 10%
                
                self.adaptive_batch_sizes[data_type] = new_size
                logger.info(f"Optimized batch size for {data_type}: {current_optimal} -> {new_size}")
            
        except Exception as e:
            logger.error(f"Error optimizing batch size: {e}")
    
    def get_performance_stats(self, data_type: str = None) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            if data_type:
                stats = self.performance_stats.get(data_type)
                if stats:
                    return {
                        "data_type": data_type,
                        "avg_processing_time": stats.avg_processing_time,
                        "avg_throughput": stats.avg_throughput,
                        "avg_memory_usage": stats.avg_memory_usage,
                        "avg_cpu_usage": stats.avg_cpu_usage,
                        "error_rate": stats.error_rate,
                        "optimal_batch_size": stats.optimal_batch_size,
                        "sample_count": stats.sample_count
                    }
                return {}
            
            # Return all stats
            return {
                data_type: {
                    "avg_processing_time": stats.avg_processing_time,
                    "avg_throughput": stats.avg_throughput,
                    "avg_memory_usage": stats.avg_memory_usage,
                    "avg_cpu_usage": stats.avg_cpu_usage,
                    "error_rate": stats.error_rate,
                    "optimal_batch_size": stats.optimal_batch_size,
                    "sample_count": stats.sample_count
                }
                for data_type, stats in self.performance_stats.items()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    def reset_performance_stats(self, data_type: str = None):
        """Reset performance statistics"""
        try:
            with self.stats_lock:
                if data_type:
                    if data_type in self.performance_stats:
                        del self.performance_stats[data_type]
                    if data_type in self.batch_history:
                        del self.batch_history[data_type]
                    if data_type in self.adaptive_batch_sizes:
                        del self.adaptive_batch_sizes[data_type]
                else:
                    self.performance_stats.clear()
                    self.batch_history.clear()
                    self.adaptive_batch_sizes.clear()
                
                logger.info(f"Reset performance stats for {data_type or 'all data types'}")
                
        except Exception as e:
            logger.error(f"Error resetting performance stats: {e}")
    
    def enable_optimization(self, enabled: bool = True):
        """Enable or disable batch size optimization"""
        self.optimization_enabled = enabled
        logger.info(f"Batch size optimization {'enabled' if enabled else 'disabled'}")
    
    def set_optimization_window(self, window_seconds: int):
        """Set optimization window for performance analysis"""
        self.optimization_window = window_seconds
        logger.info(f"Set optimization window to {window_seconds} seconds")
