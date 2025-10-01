"""
Multi-Level Cache
Advanced multi-level caching implementation for performance optimization
"""

import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache levels"""
    L1 = "l1"  # In-memory (fastest)
    L2 = "l2"  # Redis (fast)
    L3 = "l3"  # Database (slower)

class CacheStrategy(Enum):
    """Cache strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[timedelta] = None
    level: CacheLevel = CacheLevel.L1

@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0

class MultiLevelCache:
    """
    Multi-level cache implementation with L1 (memory), L2 (Redis), L3 (database)
    """
    
    def __init__(self, 
                 l1_size: int = 1000,
                 l2_redis_client = None,
                 l3_database_client = None,
                 default_ttl: timedelta = timedelta(hours=1),
                 strategy: CacheStrategy = CacheStrategy.LRU):
        self.l1_size = l1_size
        self.l2_client = l2_redis_client
        self.l3_client = l3_database_client
        self.default_ttl = default_ttl
        self.strategy = strategy
        
        # L1 Cache (In-memory)
        self.l1_cache = OrderedDict()
        self.l1_lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats()
        self.stats_lock = threading.RLock()
        
        logger.info(f"MultiLevelCache initialized with L1 size: {l1_size}, strategy: {strategy.value}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, checking all levels"""
        try:
            with self.stats_lock:
                self.stats.total_requests += 1
            
            # Check L1 cache first
            value = self._get_from_l1(key)
            if value is not None:
                with self.stats_lock:
                    self.stats.hits += 1
                    self.stats.hit_rate = self.stats.hits / self.stats.total_requests
                logger.debug(f"Cache hit in L1 for key: {key}")
                return value
            
            # Check L2 cache (Redis)
            if self.l2_client:
                value = self._get_from_l2(key)
                if value is not None:
                    # Promote to L1
                    self._set_in_l1(key, value)
                    with self.stats_lock:
                        self.stats.hits += 1
                        self.stats.hit_rate = self.stats.hits / self.stats.total_requests
                    logger.debug(f"Cache hit in L2 for key: {key}")
                    return value
            
            # Check L3 cache (Database)
            if self.l3_client:
                value = self._get_from_l3(key)
                if value is not None:
                    # Promote to L2 and L1
                    if self.l2_client:
                        self._set_in_l2(key, value)
                    self._set_in_l1(key, value)
                    with self.stats_lock:
                        self.stats.hits += 1
                        self.stats.hit_rate = self.stats.hits / self.stats.total_requests
                    logger.debug(f"Cache hit in L3 for key: {key}")
                    return value
            
            # Cache miss
            with self.stats_lock:
                self.stats.misses += 1
                self.stats.miss_rate = self.stats.misses / self.stats.total_requests
            logger.debug(f"Cache miss for key: {key}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting cache value for key {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in cache, storing in all levels"""
        try:
            cache_ttl = ttl or self.default_ttl
            
            # Set in L1
            self._set_in_l1(key, value, cache_ttl)
            
            # Set in L2 if available
            if self.l2_client:
                self._set_in_l2(key, value, cache_ttl)
            
            # Set in L3 if available
            if self.l3_client:
                self._set_in_l3(key, value, cache_ttl)
            
            logger.debug(f"Cache set for key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache value for key {key}: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from all cache levels"""
        try:
            # Delete from L1
            self._delete_from_l1(key)
            
            # Delete from L2 if available
            if self.l2_client:
                self._delete_from_l2(key)
            
            # Delete from L3 if available
            if self.l3_client:
                self._delete_from_l3(key)
            
            logger.debug(f"Cache deleted for key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting cache value for key {key}: {str(e)}")
            return False
    
    def clear(self, level: Optional[CacheLevel] = None) -> bool:
        """Clear cache at specified level or all levels"""
        try:
            if level is None or level == CacheLevel.L1:
                with self.l1_lock:
                    self.l1_cache.clear()
                logger.info("L1 cache cleared")
            
            if level is None or level == CacheLevel.L2:
                if self.l2_client:
                    self.l2_client.flushdb()
                    logger.info("L2 cache cleared")
            
            if level is None or level == CacheLevel.L3:
                if self.l3_client:
                    # Clear L3 cache (implementation depends on database)
                    logger.info("L3 cache cleared")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: Optional[timedelta] = None) -> Any:
        """Get value from cache or set it using factory function"""
        try:
            value = self.get(key)
            if value is not None:
                return value
            
            # Value not in cache, create it
            value = factory()
            self.set(key, value, ttl)
            return value
            
        except Exception as e:
            logger.error(f"Error in get_or_set for key {key}: {str(e)}")
            return None
    
    def _get_from_l1(self, key: str) -> Optional[Any]:
        """Get value from L1 cache"""
        try:
            with self.l1_lock:
                if key in self.l1_cache:
                    entry = self.l1_cache[key]
                    
                    # Check TTL
                    if entry.ttl and datetime.now() - entry.created_at > entry.ttl:
                        del self.l1_cache[key]
                        return None
                    
                    # Update access info
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    
                    # Move to end (LRU)
                    self.l1_cache.move_to_end(key)
                    
                    return entry.value
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting from L1 cache: {str(e)}")
            return None
    
    def _set_in_l1(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Set value in L1 cache"""
        try:
            with self.l1_lock:
                # Check if we need to evict
                if len(self.l1_cache) >= self.l1_size and key not in self.l1_cache:
                    self._evict_from_l1()
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    ttl=ttl,
                    level=CacheLevel.L1
                )
                
                self.l1_cache[key] = entry
                
        except Exception as e:
            logger.error(f"Error setting in L1 cache: {str(e)}")
    
    def _delete_from_l1(self, key: str) -> None:
        """Delete value from L1 cache"""
        try:
            with self.l1_lock:
                if key in self.l1_cache:
                    del self.l1_cache[key]
                    
        except Exception as e:
            logger.error(f"Error deleting from L1 cache: {str(e)}")
    
    def _evict_from_l1(self) -> None:
        """Evict entry from L1 cache based on strategy"""
        try:
            with self.l1_lock:
                if not self.l1_cache:
                    return
                
                if self.strategy == CacheStrategy.LRU:
                    # Remove least recently used (first item)
                    self.l1_cache.popitem(last=False)
                elif self.strategy == CacheStrategy.LFU:
                    # Remove least frequently used
                    least_frequent_key = min(
                        self.l1_cache.keys(),
                        key=lambda k: self.l1_cache[k].access_count
                    )
                    del self.l1_cache[least_frequent_key]
                elif self.strategy == CacheStrategy.FIFO:
                    # Remove first in (first item)
                    self.l1_cache.popitem(last=False)
                elif self.strategy == CacheStrategy.TTL:
                    # Remove expired entries first, then LRU
                    now = datetime.now()
                    expired_keys = [
                        k for k, v in self.l1_cache.items()
                        if v.ttl and now - v.created_at > v.ttl
                    ]
                    if expired_keys:
                        del self.l1_cache[expired_keys[0]]
                    else:
                        self.l1_cache.popitem(last=False)
                
                with self.stats_lock:
                    self.stats.evictions += 1
                    
        except Exception as e:
            logger.error(f"Error evicting from L1 cache: {str(e)}")
    
    def _get_from_l2(self, key: str) -> Optional[Any]:
        """Get value from L2 cache (Redis)"""
        try:
            if not self.l2_client:
                return None
            
            value = self.l2_client.get(key)
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"Error getting from L2 cache: {str(e)}")
            return None
    
    def _set_in_l2(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Set value in L2 cache (Redis)"""
        try:
            if not self.l2_client:
                return
            
            ttl_seconds = int(ttl.total_seconds()) if ttl else None
            self.l2_client.setex(key, ttl_seconds, json.dumps(value))
            
        except Exception as e:
            logger.error(f"Error setting in L2 cache: {str(e)}")
    
    def _delete_from_l2(self, key: str) -> None:
        """Delete value from L2 cache (Redis)"""
        try:
            if not self.l2_client:
                return
            
            self.l2_client.delete(key)
            
        except Exception as e:
            logger.error(f"Error deleting from L2 cache: {str(e)}")
    
    def _get_from_l3(self, key: str) -> Optional[Any]:
        """Get value from L3 cache (Database)"""
        try:
            if not self.l3_client:
                return None
            
            # Implementation depends on database client
            # This is a placeholder
            return None
            
        except Exception as e:
            logger.error(f"Error getting from L3 cache: {str(e)}")
            return None
    
    def _set_in_l3(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Set value in L3 cache (Database)"""
        try:
            if not self.l3_client:
                return
            
            # Implementation depends on database client
            # This is a placeholder
            
        except Exception as e:
            logger.error(f"Error setting in L3 cache: {str(e)}")
    
    def _delete_from_l3(self, key: str) -> None:
        """Delete value from L3 cache (Database)"""
        try:
            if not self.l3_client:
                return
            
            # Implementation depends on database client
            # This is a placeholder
            
        except Exception as e:
            logger.error(f"Error deleting from L3 cache: {str(e)}")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        try:
            with self.stats_lock:
                return CacheStats(
                    hits=self.stats.hits,
                    misses=self.stats.misses,
                    evictions=self.stats.evictions,
                    total_requests=self.stats.total_requests,
                    hit_rate=self.stats.hit_rate,
                    miss_rate=self.stats.miss_rate
                )
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return CacheStats()
    
    def reset_stats(self) -> None:
        """Reset cache statistics"""
        try:
            with self.stats_lock:
                self.stats = CacheStats()
                
        except Exception as e:
            logger.error(f"Error resetting cache stats: {str(e)}")
    
    def get_l1_size(self) -> int:
        """Get current L1 cache size"""
        try:
            with self.l1_lock:
                return len(self.l1_cache)
                
        except Exception as e:
            logger.error(f"Error getting L1 cache size: {str(e)}")
            return 0
    
    def warm_cache(self, keys: List[str], factory: Callable[[str], Any]) -> None:
        """Warm cache with specified keys"""
        try:
            for key in keys:
                if self.get(key) is None:
                    value = factory(key)
                    if value is not None:
                        self.set(key, value)
            
            logger.info(f"Cache warmed with {len(keys)} keys")
            
        except Exception as e:
            logger.error(f"Error warming cache: {str(e)}")
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        try:
            invalidated_count = 0
            
            # Invalidate L1
            with self.l1_lock:
                keys_to_remove = [k for k in self.l1_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self.l1_cache[key]
                    invalidated_count += 1
            
            # Invalidate L2
            if self.l2_client:
                keys = self.l2_client.keys(f"*{pattern}*")
                if keys:
                    self.l2_client.delete(*keys)
                    invalidated_count += len(keys)
            
            logger.info(f"Invalidated {invalidated_count} keys matching pattern: {pattern}")
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Error invalidating pattern {pattern}: {str(e)}")
            return 0
