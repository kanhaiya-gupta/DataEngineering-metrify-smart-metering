"""
Cache Invalidator
Advanced cache invalidation strategies and policies
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Pattern
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class InvalidationStrategy(Enum):
    """Cache invalidation strategies"""
    IMMEDIATE = "immediate"
    LAZY = "lazy"
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    DEPENDENCY_BASED = "dependency_based"

class InvalidationScope(Enum):
    """Cache invalidation scope"""
    KEY = "key"
    PATTERN = "pattern"
    NAMESPACE = "namespace"
    ALL = "all"

@dataclass
class InvalidationRule:
    """Cache invalidation rule"""
    rule_id: str
    name: str
    pattern: str
    strategy: InvalidationStrategy
    scope: InvalidationScope
    ttl: Optional[timedelta] = None
    enabled: bool = True
    created_at: datetime = None

@dataclass
class InvalidationEvent:
    """Cache invalidation event"""
    event_id: str
    rule_id: str
    keys: List[str]
    timestamp: datetime
    reason: str
    success: bool = True
    error_message: Optional[str] = None

class CacheInvalidator:
    """
    Advanced cache invalidation with multiple strategies and policies
    """
    
    def __init__(self, cache_client = None):
        self.cache_client = cache_client
        self.rules: Dict[str, InvalidationRule] = {}
        self.event_history: deque = deque(maxlen=10000)
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.invalidation_queue: deque = deque()
        self.invalidator_lock = threading.RLock()
        
        # Start background invalidation thread
        self.invalidation_thread = threading.Thread(
            target=self._process_invalidation_queue,
            daemon=True
        )
        self.invalidation_thread.start()
        
        logger.info("CacheInvalidator initialized")
    
    def register_rule(self, rule: InvalidationRule) -> bool:
        """Register an invalidation rule"""
        try:
            with self.invalidator_lock:
                rule.created_at = datetime.now()
                self.rules[rule.rule_id] = rule
                logger.info(f"Registered invalidation rule: {rule.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register invalidation rule {rule.rule_id}: {str(e)}")
            return False
    
    def unregister_rule(self, rule_id: str) -> bool:
        """Unregister an invalidation rule"""
        try:
            with self.invalidator_lock:
                if rule_id in self.rules:
                    del self.rules[rule_id]
                    logger.info(f"Unregistered invalidation rule: {rule_id}")
                    return True
                else:
                    logger.warning(f"Invalidation rule {rule_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to unregister invalidation rule {rule_id}: {str(e)}")
            return False
    
    def invalidate_key(self, key: str, reason: str = "manual") -> bool:
        """Invalidate a specific cache key"""
        try:
            if not self.cache_client:
                logger.warning("No cache client available for invalidation")
                return False
            
            success = self.cache_client.delete(key)
            
            # Record invalidation event
            event = InvalidationEvent(
                event_id=f"inv_{int(time.time())}",
                rule_id="manual",
                keys=[key],
                timestamp=datetime.now(),
                reason=reason,
                success=success
            )
            self._record_event(event)
            
            logger.info(f"Invalidated key: {key}, reason: {reason}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to invalidate key {key}: {str(e)}")
            return False
    
    def invalidate_pattern(self, pattern: str, reason: str = "pattern_match") -> int:
        """Invalidate all keys matching a pattern"""
        try:
            if not self.cache_client:
                logger.warning("No cache client available for invalidation")
                return 0
            
            # Get keys matching pattern
            if hasattr(self.cache_client, 'keys'):
                matching_keys = self.cache_client.keys(pattern)
            else:
                # Fallback to regex matching
                all_keys = self._get_all_keys()
                matching_keys = [key for key in all_keys if re.match(pattern, key)]
            
            if not matching_keys:
                logger.info(f"No keys found matching pattern: {pattern}")
                return 0
            
            # Invalidate keys
            invalidated_count = 0
            for key in matching_keys:
                if self.cache_client.delete(key):
                    invalidated_count += 1
            
            # Record invalidation event
            event = InvalidationEvent(
                event_id=f"inv_{int(time.time())}",
                rule_id="pattern",
                keys=matching_keys,
                timestamp=datetime.now(),
                reason=reason,
                success=invalidated_count > 0
            )
            self._record_event(event)
            
            logger.info(f"Invalidated {invalidated_count} keys matching pattern: {pattern}")
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Failed to invalidate pattern {pattern}: {str(e)}")
            return 0
    
    def invalidate_namespace(self, namespace: str, reason: str = "namespace_clear") -> int:
        """Invalidate all keys in a namespace"""
        try:
            pattern = f"{namespace}:*"
            return self.invalidate_pattern(pattern, reason)
            
        except Exception as e:
            logger.error(f"Failed to invalidate namespace {namespace}: {str(e)}")
            return 0
    
    def invalidate_all(self, reason: str = "clear_all") -> bool:
        """Invalidate all cache keys"""
        try:
            if not self.cache_client:
                logger.warning("No cache client available for invalidation")
                return False
            
            success = self.cache_client.clear()
            
            # Record invalidation event
            event = InvalidationEvent(
                event_id=f"inv_{int(time.time())}",
                rule_id="all",
                keys=[],
                timestamp=datetime.now(),
                reason=reason,
                success=success
            )
            self._record_event(event)
            
            logger.info(f"Invalidated all cache keys, reason: {reason}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to invalidate all keys: {str(e)}")
            return False
    
    def schedule_invalidation(self, 
                            keys: List[str], 
                            delay_seconds: int = 0,
                            reason: str = "scheduled") -> str:
        """Schedule invalidation for later execution"""
        try:
            invalidation_id = f"sched_{int(time.time())}"
            
            invalidation_task = {
                "id": invalidation_id,
                "keys": keys,
                "execute_at": datetime.now() + timedelta(seconds=delay_seconds),
                "reason": reason,
                "created_at": datetime.now()
            }
            
            with self.invalidator_lock:
                self.invalidation_queue.append(invalidation_task)
            
            logger.info(f"Scheduled invalidation {invalidation_id} for {len(keys)} keys")
            return invalidation_id
            
        except Exception as e:
            logger.error(f"Failed to schedule invalidation: {str(e)}")
            return ""
    
    def add_dependency(self, parent_key: str, child_key: str) -> bool:
        """Add dependency relationship between keys"""
        try:
            with self.invalidator_lock:
                if child_key not in self.dependency_graph[parent_key]:
                    self.dependency_graph[parent_key].append(child_key)
                    logger.debug(f"Added dependency: {parent_key} -> {child_key}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to add dependency {parent_key} -> {child_key}: {str(e)}")
            return False
    
    def remove_dependency(self, parent_key: str, child_key: str) -> bool:
        """Remove dependency relationship between keys"""
        try:
            with self.invalidator_lock:
                if child_key in self.dependency_graph[parent_key]:
                    self.dependency_graph[parent_key].remove(child_key)
                    logger.debug(f"Removed dependency: {parent_key} -> {child_key}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove dependency {parent_key} -> {child_key}: {str(e)}")
            return False
    
    def invalidate_with_dependencies(self, key: str, reason: str = "dependency_invalidation") -> int:
        """Invalidate key and all its dependencies"""
        try:
            invalidated_count = 0
            
            # Invalidate the key itself
            if self.invalidate_key(key, reason):
                invalidated_count += 1
            
            # Invalidate dependencies
            with self.invalidator_lock:
                dependencies = self.dependency_graph.get(key, [])
            
            for dep_key in dependencies:
                if self.invalidate_key(dep_key, f"dependency of {key}"):
                    invalidated_count += 1
                
                # Recursively invalidate dependencies of dependencies
                invalidated_count += self.invalidate_with_dependencies(dep_key, f"dependency of {key}")
            
            logger.info(f"Invalidated {invalidated_count} keys including dependencies for: {key}")
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Failed to invalidate with dependencies for {key}: {str(e)}")
            return 0
    
    def process_data_change_event(self, event_type: str, entity_id: str, data: Dict[str, Any]) -> int:
        """Process data change event and invalidate related cache keys"""
        try:
            invalidated_count = 0
            
            # Find applicable rules
            applicable_rules = []
            for rule in self.rules.values():
                if not rule.enabled:
                    continue
                
                if rule.strategy == InvalidationStrategy.EVENT_BASED:
                    if self._rule_matches_event(rule, event_type, entity_id, data):
                        applicable_rules.append(rule)
            
            # Execute invalidation rules
            for rule in applicable_rules:
                if rule.scope == InvalidationScope.KEY:
                    key = self._generate_key_from_rule(rule, entity_id, data)
                    if self.invalidate_key(key, f"rule: {rule.name}"):
                        invalidated_count += 1
                
                elif rule.scope == InvalidationScope.PATTERN:
                    pattern = self._generate_pattern_from_rule(rule, entity_id, data)
                    count = self.invalidate_pattern(pattern, f"rule: {rule.name}")
                    invalidated_count += count
                
                elif rule.scope == InvalidationScope.NAMESPACE:
                    namespace = self._generate_namespace_from_rule(rule, entity_id, data)
                    count = self.invalidate_namespace(namespace, f"rule: {rule.name}")
                    invalidated_count += count
            
            logger.info(f"Processed data change event {event_type} for {entity_id}, invalidated {invalidated_count} keys")
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Failed to process data change event: {str(e)}")
            return 0
    
    def _process_invalidation_queue(self):
        """Process scheduled invalidations in background thread"""
        try:
            while True:
                time.sleep(1)  # Check every second
                
                current_time = datetime.now()
                tasks_to_execute = []
                
                with self.invalidator_lock:
                    # Find tasks ready for execution
                    remaining_tasks = []
                    for task in self.invalidation_queue:
                        if task["execute_at"] <= current_time:
                            tasks_to_execute.append(task)
                        else:
                            remaining_tasks.append(task)
                    
                    self.invalidation_queue = deque(remaining_tasks)
                
                # Execute ready tasks
                for task in tasks_to_execute:
                    try:
                        for key in task["keys"]:
                            self.invalidate_key(key, task["reason"])
                        logger.info(f"Executed scheduled invalidation {task['id']}")
                    except Exception as e:
                        logger.error(f"Failed to execute scheduled invalidation {task['id']}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in invalidation queue processor: {str(e)}")
    
    def _rule_matches_event(self, rule: InvalidationRule, event_type: str, entity_id: str, data: Dict[str, Any]) -> bool:
        """Check if rule matches the event"""
        try:
            # Simple pattern matching - can be enhanced
            if rule.pattern in event_type or rule.pattern in entity_id:
                return True
            
            # Check data fields
            for field, value in data.items():
                if rule.pattern in str(value):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking rule match: {str(e)}")
            return False
    
    def _generate_key_from_rule(self, rule: InvalidationRule, entity_id: str, data: Dict[str, Any]) -> str:
        """Generate cache key from rule and event data"""
        try:
            # Simple key generation - can be enhanced
            return f"{rule.pattern}:{entity_id}"
            
        except Exception as e:
            logger.error(f"Error generating key from rule: {str(e)}")
            return ""
    
    def _generate_pattern_from_rule(self, rule: InvalidationRule, entity_id: str, data: Dict[str, Any]) -> str:
        """Generate pattern from rule and event data"""
        try:
            # Simple pattern generation - can be enhanced
            return f"{rule.pattern}:*"
            
        except Exception as e:
            logger.error(f"Error generating pattern from rule: {str(e)}")
            return ""
    
    def _generate_namespace_from_rule(self, rule: InvalidationRule, entity_id: str, data: Dict[str, Any]) -> str:
        """Generate namespace from rule and event data"""
        try:
            # Simple namespace generation - can be enhanced
            return rule.pattern
            
        except Exception as e:
            logger.error(f"Error generating namespace from rule: {str(e)}")
            return ""
    
    def _get_all_keys(self) -> List[str]:
        """Get all cache keys (fallback method)"""
        try:
            if hasattr(self.cache_client, 'keys'):
                return self.cache_client.keys("*")
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting all keys: {str(e)}")
            return []
    
    def _record_event(self, event: InvalidationEvent) -> None:
        """Record invalidation event"""
        try:
            with self.invalidator_lock:
                self.event_history.append(event)
                
        except Exception as e:
            logger.error(f"Error recording invalidation event: {str(e)}")
    
    def get_invalidation_stats(self) -> Dict[str, Any]:
        """Get invalidation statistics"""
        try:
            with self.invalidator_lock:
                total_events = len(self.event_history)
                successful_events = sum(1 for event in self.event_history if event.success)
                failed_events = total_events - successful_events
                
                return {
                    "total_rules": len(self.rules),
                    "enabled_rules": sum(1 for rule in self.rules.values() if rule.enabled),
                    "total_invalidations": total_events,
                    "successful_invalidations": successful_events,
                    "failed_invalidations": failed_events,
                    "success_rate": successful_events / total_events if total_events > 0 else 0,
                    "pending_scheduled": len(self.invalidation_queue),
                    "dependency_relationships": sum(len(deps) for deps in self.dependency_graph.values())
                }
                
        except Exception as e:
            logger.error(f"Error getting invalidation stats: {str(e)}")
            return {}
    
    def get_recent_events(self, limit: int = 100) -> List[InvalidationEvent]:
        """Get recent invalidation events"""
        try:
            with self.invalidator_lock:
                return list(self.event_history)[-limit:]
                
        except Exception as e:
            logger.error(f"Error getting recent events: {str(e)}")
            return []
    
    def clear_event_history(self) -> None:
        """Clear invalidation event history"""
        try:
            with self.invalidator_lock:
                self.event_history.clear()
                logger.info("Invalidation event history cleared")
                
        except Exception as e:
            logger.error(f"Error clearing event history: {str(e)}")
    
    def export_rules(self) -> List[Dict[str, Any]]:
        """Export invalidation rules"""
        try:
            with self.invalidator_lock:
                return [
                    {
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "pattern": rule.pattern,
                        "strategy": rule.strategy.value,
                        "scope": rule.scope.value,
                        "ttl": rule.ttl.total_seconds() if rule.ttl else None,
                        "enabled": rule.enabled,
                        "created_at": rule.created_at.isoformat() if rule.created_at else None
                    }
                    for rule in self.rules.values()
                ]
                
        except Exception as e:
            logger.error(f"Error exporting rules: {str(e)}")
            return []
