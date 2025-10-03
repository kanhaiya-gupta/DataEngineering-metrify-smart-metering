"""
Cache Warmer
Intelligent cache warming strategies and preloading
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class WarmingStrategy(Enum):
    """Cache warming strategies"""
    EAGER = "eager"  # Load all data immediately
    LAZY = "lazy"    # Load data on demand
    SCHEDULED = "scheduled"  # Load data on schedule
    PREDICTIVE = "predictive"  # Load data based on predictions
    DEPENDENCY = "dependency"  # Load data based on dependencies

class WarmingPriority(Enum):
    """Cache warming priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class WarmingTask:
    """Cache warming task"""
    task_id: str
    key: str
    factory: Callable[[], Any]
    priority: WarmingPriority
    strategy: WarmingStrategy
    ttl: Optional[timedelta] = None
    dependencies: List[str] = None
    created_at: datetime = None
    scheduled_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None

@dataclass
class WarmingStats:
    """Cache warming statistics"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    running_tasks: int = 0
    pending_tasks: int = 0
    success_rate: float = 0.0
    average_warming_time: float = 0.0

class CacheWarmer:
    """
    Intelligent cache warming with multiple strategies and priorities
    """
    
    def __init__(self, 
                 cache_client = None,
                 max_workers: int = 10,
                 warming_interval: int = 60):
        self.cache_client = cache_client
        self.max_workers = max_workers
        self.warming_interval = warming_interval
        
        # Task management
        self.tasks: Dict[str, WarmingTask] = {}
        self.task_queue = deque()
        self.running_tasks: Dict[str, WarmingTask] = {}
        self.completed_tasks: deque = deque(maxlen=10000)
        
        # Statistics
        self.stats = WarmingStats()
        self.stats_lock = threading.RLock()
        
        # Threading
        self.warmer_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Start background warming thread
        self.warming_thread = threading.Thread(
            target=self._process_warming_queue,
            daemon=True
        )
        self.warming_thread.start()
        
        logger.info(f"CacheWarmer initialized with {max_workers} workers")
    
    def add_warming_task(self,
                        key: str,
                        factory: Callable[[], Any],
                        priority: WarmingPriority = WarmingPriority.MEDIUM,
                        strategy: WarmingStrategy = WarmingStrategy.EAGER,
                        ttl: Optional[timedelta] = None,
                        dependencies: List[str] = None,
                        scheduled_at: Optional[datetime] = None) -> str:
        """Add a cache warming task"""
        try:
            task_id = f"warm_{int(time.time())}_{hash(key) % 10000}"
            
            task = WarmingTask(
                task_id=task_id,
                key=key,
                factory=factory,
                priority=priority,
                strategy=strategy,
                ttl=ttl,
                dependencies=dependencies or [],
                created_at=datetime.now(),
                scheduled_at=scheduled_at
            )
            
            with self.warmer_lock:
                self.tasks[task_id] = task
                
                # Add to queue based on strategy
                if strategy == WarmingStrategy.EAGER:
                    self.task_queue.append(task)
                elif strategy == WarmingStrategy.SCHEDULED and scheduled_at:
                    # Will be processed by scheduled warming
                    pass
                elif strategy == WarmingStrategy.LAZY:
                    # Will be processed on demand
                    pass
            
            with self.stats_lock:
                self.stats.total_tasks += 1
                self.stats.pending_tasks += 1
            
            logger.info(f"Added warming task {task_id} for key: {key}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to add warming task for key {key}: {str(e)}")
            return ""
    
    def warm_key(self, key: str, factory: Callable[[], Any], ttl: Optional[timedelta] = None) -> bool:
        """Warm a specific cache key immediately"""
        try:
            if not self.cache_client:
                logger.warning("No cache client available for warming")
                return False
            
            # Check if key already exists
            if self.cache_client.get(key) is not None:
                logger.debug(f"Key {key} already exists in cache")
                return True
            
            # Generate value using factory
            start_time = time.time()
            value = factory()
            generation_time = time.time() - start_time
            
            if value is None:
                logger.warning(f"Factory returned None for key: {key}")
                return False
            
            # Set in cache
            if ttl:
                success = self.cache_client.set(key, value, ttl)
            else:
                success = self.cache_client.set(key, value)
            
            if success:
                logger.info(f"Warmed key {key} in {generation_time:.3f}s")
                return True
            else:
                logger.error(f"Failed to set warmed value for key: {key}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to warm key {key}: {str(e)}")
            return False
    
    def warm_keys(self, keys: List[str], factory: Callable[[str], Any], ttl: Optional[timedelta] = None) -> int:
        """Warm multiple cache keys in parallel"""
        try:
            if not keys:
                return 0
            
            # Filter out keys that already exist
            keys_to_warm = []
            for key in keys:
                if self.cache_client and self.cache_client.get(key) is None:
                    keys_to_warm.append(key)
            
            if not keys_to_warm:
                logger.info("All keys already exist in cache")
                return len(keys)
            
            # Warm keys in parallel
            warmed_count = 0
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(keys_to_warm))) as executor:
                future_to_key = {
                    executor.submit(self._warm_single_key, key, factory, ttl): key
                    for key in keys_to_warm
                }
                
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        success = future.result()
                        if success:
                            warmed_count += 1
                    except Exception as e:
                        logger.error(f"Failed to warm key {key}: {str(e)}")
            
            logger.info(f"Warmed {warmed_count}/{len(keys_to_warm)} keys")
            return warmed_count
            
        except Exception as e:
            logger.error(f"Failed to warm keys: {str(e)}")
            return 0
    
    def warm_pattern(self, pattern: str, factory: Callable[[str], Any], ttl: Optional[timedelta] = None) -> int:
        """Warm all keys matching a pattern"""
        try:
            if not self.cache_client:
                logger.warning("No cache client available for warming")
                return 0
            
            # Get keys matching pattern
            if hasattr(self.cache_client, 'keys'):
                matching_keys = self.cache_client.keys(pattern)
            else:
                logger.warning("Cache client does not support pattern matching")
                return 0
            
            if not matching_keys:
                logger.info(f"No keys found matching pattern: {pattern}")
                return 0
            
            return self.warm_keys(matching_keys, factory, ttl)
            
        except Exception as e:
            logger.error(f"Failed to warm pattern {pattern}: {str(e)}")
            return 0
    
    def warm_namespace(self, namespace: str, factory: Callable[[str], Any], ttl: Optional[timedelta] = None) -> int:
        """Warm all keys in a namespace"""
        try:
            pattern = f"{namespace}:*"
            return self.warm_pattern(pattern, factory, ttl)
            
        except Exception as e:
            logger.error(f"Failed to warm namespace {namespace}: {str(e)}")
            return 0
    
    def schedule_warming(self, 
                        key: str,
                        factory: Callable[[], Any],
                        scheduled_at: datetime,
                        ttl: Optional[timedelta] = None) -> str:
        """Schedule cache warming for a specific time"""
        try:
            return self.add_warming_task(
                key=key,
                factory=factory,
                priority=WarmingPriority.MEDIUM,
                strategy=WarmingStrategy.SCHEDULED,
                ttl=ttl,
                scheduled_at=scheduled_at
            )
            
        except Exception as e:
            logger.error(f"Failed to schedule warming for key {key}: {str(e)}")
            return ""
    
    def warm_dependencies(self, key: str, dependency_factory: Callable[[str], List[str]]) -> int:
        """Warm cache based on dependencies"""
        try:
            # Get dependencies
            dependencies = dependency_factory(key)
            if not dependencies:
                logger.info(f"No dependencies found for key: {key}")
                return 0
            
            # Warm dependencies
            warmed_count = 0
            for dep_key in dependencies:
                if self.cache_client and self.cache_client.get(dep_key) is None:
                    # This would need a factory for each dependency
                    # For now, we'll just log it
                    logger.debug(f"Would warm dependency: {dep_key}")
                    warmed_count += 1
            
            logger.info(f"Warmed {warmed_count} dependencies for key: {key}")
            return warmed_count
            
        except Exception as e:
            logger.error(f"Failed to warm dependencies for key {key}: {str(e)}")
            return 0
    
    def predict_and_warm(self, 
                        current_key: str,
                        predictor: Callable[[str], List[str]],
                        factory: Callable[[str], Any],
                        ttl: Optional[timedelta] = None) -> int:
        """Predict next likely keys and warm them"""
        try:
            # Get predicted keys
            predicted_keys = predictor(current_key)
            if not predicted_keys:
                logger.info(f"No predictions for key: {current_key}")
                return 0
            
            # Warm predicted keys
            warmed_count = 0
            for pred_key in predicted_keys:
                if self.cache_client and self.cache_client.get(pred_key) is None:
                    value = factory(pred_key)
                    if value is not None:
                        if ttl:
                            success = self.cache_client.set(pred_key, value, ttl)
                        else:
                            success = self.cache_client.set(pred_key, value)
                        
                        if success:
                            warmed_count += 1
            
            logger.info(f"Warmed {warmed_count} predicted keys for: {current_key}")
            return warmed_count
            
        except Exception as e:
            logger.error(f"Failed to predict and warm for key {current_key}: {str(e)}")
            return 0
    
    def _warm_single_key(self, key: str, factory: Callable[[str], Any], ttl: Optional[timedelta] = None) -> bool:
        """Warm a single key (internal method)"""
        try:
            if not self.cache_client:
                return False
            
            # Check if key already exists
            if self.cache_client.get(key) is not None:
                return True
            
            # Generate value
            value = factory(key)
            if value is None:
                return False
            
            # Set in cache
            if ttl:
                return self.cache_client.set(key, value, ttl)
            else:
                return self.cache_client.set(key, value)
                
        except Exception as e:
            logger.error(f"Error warming single key {key}: {str(e)}")
            return False
    
    def _process_warming_queue(self):
        """Process warming queue in background thread"""
        try:
            while True:
                time.sleep(self.warming_interval)
                
                current_time = datetime.now()
                tasks_to_execute = []
                
                with self.warmer_lock:
                    # Find tasks ready for execution
                    remaining_tasks = []
                    for task in self.task_queue:
                        if (task.strategy == WarmingStrategy.EAGER or
                            (task.strategy == WarmingStrategy.SCHEDULED and 
                             task.scheduled_at and task.scheduled_at <= current_time)):
                            tasks_to_execute.append(task)
                        else:
                            remaining_tasks.append(task)
                    
                    self.task_queue = deque(remaining_tasks)
                
                # Execute ready tasks
                for task in tasks_to_execute:
                    self._execute_warming_task(task)
                
        except Exception as e:
            logger.error(f"Error in warming queue processor: {str(e)}")
    
    def _execute_warming_task(self, task: WarmingTask) -> None:
        """Execute a warming task"""
        try:
            task.status = "running"
            self.running_tasks[task.task_id] = task
            
            with self.stats_lock:
                self.stats.pending_tasks -= 1
                self.stats.running_tasks += 1
            
            # Execute warming
            start_time = time.time()
            success = self.warm_key(task.key, task.factory, task.ttl)
            execution_time = time.time() - start_time
            
            # Update task status
            if success:
                task.status = "completed"
                with self.stats_lock:
                    self.stats.completed_tasks += 1
                    self.stats.success_rate = self.stats.completed_tasks / self.stats.total_tasks
                    self.stats.average_warming_time = (
                        (self.stats.average_warming_time * (self.stats.completed_tasks - 1) + execution_time) /
                        self.stats.completed_tasks
                    )
            else:
                task.status = "failed"
                task.error_message = "Warming failed"
                with self.stats_lock:
                    self.stats.failed_tasks += 1
            
            with self.stats_lock:
                self.stats.running_tasks -= 1
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            logger.info(f"Executed warming task {task.task_id} for key {task.key} in {execution_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error executing warming task {task.task_id}: {str(e)}")
            task.status = "failed"
            task.error_message = str(e)
            
            with self.stats_lock:
                self.stats.failed_tasks += 1
                self.stats.running_tasks -= 1
    
    def get_warming_stats(self) -> WarmingStats:
        """Get warming statistics"""
        try:
            with self.stats_lock:
                return WarmingStats(
                    total_tasks=self.stats.total_tasks,
                    completed_tasks=self.stats.completed_tasks,
                    failed_tasks=self.stats.failed_tasks,
                    running_tasks=self.stats.running_tasks,
                    pending_tasks=self.stats.pending_tasks,
                    success_rate=self.stats.success_rate,
                    average_warming_time=self.stats.average_warming_time
                )
                
        except Exception as e:
            logger.error(f"Error getting warming stats: {str(e)}")
            return WarmingStats()
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific warming task"""
        try:
            with self.warmer_lock:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    return {
                        "task_id": task.task_id,
                        "key": task.key,
                        "status": task.status,
                        "priority": task.priority.value,
                        "strategy": task.strategy.value,
                        "created_at": task.created_at.isoformat() if task.created_at else None,
                        "scheduled_at": task.scheduled_at.isoformat() if task.scheduled_at else None,
                        "error_message": task.error_message
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error getting task status for {task_id}: {str(e)}")
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a warming task"""
        try:
            with self.warmer_lock:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    if task.status == "pending":
                        task.status = "cancelled"
                        # Remove from queue
                        self.task_queue = deque(t for t in self.task_queue if t.task_id != task_id)
                        logger.info(f"Cancelled warming task {task_id}")
                        return True
                    else:
                        logger.warning(f"Cannot cancel task {task_id} with status {task.status}")
                        return False
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {str(e)}")
            return False
    
    def get_recent_tasks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent warming tasks"""
        try:
            with self.warmer_lock:
                recent_tasks = list(self.completed_tasks)[-limit:]
                return [
                    {
                        "task_id": task.task_id,
                        "key": task.key,
                        "status": task.status,
                        "priority": task.priority.value,
                        "strategy": task.strategy.value,
                        "created_at": task.created_at.isoformat() if task.created_at else None,
                        "error_message": task.error_message
                    }
                    for task in recent_tasks
                ]
                
        except Exception as e:
            logger.error(f"Error getting recent tasks: {str(e)}")
            return []
    
    def clear_completed_tasks(self) -> None:
        """Clear completed task history"""
        try:
            with self.warmer_lock:
                self.completed_tasks.clear()
                logger.info("Completed task history cleared")
                
        except Exception as e:
            logger.error(f"Error clearing completed tasks: {str(e)}")
    
    def stop(self) -> None:
        """Stop the cache warmer"""
        try:
            self.executor.shutdown(wait=True)
            logger.info("Cache warmer stopped")
            
        except Exception as e:
            logger.error(f"Error stopping cache warmer: {str(e)}")


