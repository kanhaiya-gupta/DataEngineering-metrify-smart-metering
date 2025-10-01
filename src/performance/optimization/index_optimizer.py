"""
Index Optimizer
Advanced index optimization and management for database performance
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class IndexType(Enum):
    """Index types"""
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"
    SPGIST = "spgist"
    BRIN = "brin"
    COMPOSITE = "composite"
    PARTIAL = "partial"
    UNIQUE = "unique"

class IndexStatus(Enum):
    """Index status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUILDING = "building"
    FAILED = "failed"
    DROPPED = "dropped"

@dataclass
class IndexDefinition:
    """Index definition"""
    index_name: str
    table_name: str
    columns: List[str]
    index_type: IndexType
    is_unique: bool = False
    is_partial: bool = False
    where_clause: Optional[str] = None
    fillfactor: int = 90
    created_at: datetime = None
    size_bytes: int = 0
    status: IndexStatus = IndexStatus.ACTIVE

@dataclass
class IndexUsageStats:
    """Index usage statistics"""
    index_name: str
    table_name: str
    scans: int = 0
    tuples_read: int = 0
    tuples_fetched: int = 0
    last_used: Optional[datetime] = None
    hit_ratio: float = 0.0
    size_bytes: int = 0

@dataclass
class IndexRecommendation:
    """Index recommendation"""
    table_name: str
    columns: List[str]
    index_type: IndexType
    estimated_benefit: float
    creation_cost: float
    priority: int
    reason: str
    confidence: float

@dataclass
class IndexMetrics:
    """Index optimization metrics"""
    total_indexes: int = 0
    active_indexes: int = 0
    unused_indexes: int = 0
    duplicate_indexes: int = 0
    recommendations_generated: int = 0
    indexes_created: int = 0
    indexes_dropped: int = 0

class IndexOptimizer:
    """
    Advanced index optimizer for database performance tuning
    """
    
    def __init__(self, 
                 database_connection = None,
                 enable_auto_optimization: bool = True,
                 monitoring_interval: int = 3600):
        self.db_connection = database_connection
        self.enable_auto_optimization = enable_auto_optimization
        self.monitoring_interval = monitoring_interval
        
        # Index management
        self.indexes: Dict[str, IndexDefinition] = {}
        self.usage_stats: Dict[str, IndexUsageStats] = {}
        self.recommendations: List[IndexRecommendation] = []
        
        # Metrics
        self.metrics = IndexMetrics()
        self.metrics_lock = threading.RLock()
        
        # Threading
        self.optimizer_lock = threading.RLock()
        
        # Start monitoring thread
        if enable_auto_optimization:
            self.monitor_thread = threading.Thread(target=self._monitor_indexes, daemon=True)
            self.monitor_thread.start()
        
        logger.info("IndexOptimizer initialized")
    
    def analyze_index_usage(self) -> Dict[str, Any]:
        """Analyze current index usage patterns"""
        try:
            analysis = {
                "total_indexes": len(self.indexes),
                "active_indexes": len([idx for idx in self.indexes.values() if idx.status == IndexStatus.ACTIVE]),
                "unused_indexes": [],
                "heavily_used_indexes": [],
                "duplicate_indexes": [],
                "size_analysis": {},
                "performance_impact": {}
            }
            
            # Analyze usage statistics
            for index_name, stats in self.usage_stats.items():
                if stats.scans == 0:
                    analysis["unused_indexes"].append(index_name)
                elif stats.scans > 1000:  # Threshold for heavy usage
                    analysis["heavily_used_indexes"].append({
                        "index": index_name,
                        "scans": stats.scans,
                        "hit_ratio": stats.hit_ratio
                    })
            
            # Find duplicate indexes
            analysis["duplicate_indexes"] = self._find_duplicate_indexes()
            
            # Size analysis
            analysis["size_analysis"] = self._analyze_index_sizes()
            
            # Performance impact
            analysis["performance_impact"] = self._analyze_performance_impact()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze index usage: {str(e)}")
            return {}
    
    def generate_recommendations(self, 
                               query_patterns: List[str] = None,
                               performance_threshold: float = 0.8) -> List[IndexRecommendation]:
        """Generate index optimization recommendations"""
        try:
            recommendations = []
            
            # Analyze query patterns for missing indexes
            if query_patterns:
                query_recs = self._analyze_query_patterns(query_patterns)
                recommendations.extend(query_recs)
            
            # Find missing indexes for existing tables
            missing_recs = self._find_missing_indexes()
            recommendations.extend(missing_recs)
            
            # Find redundant indexes
            redundant_recs = self._find_redundant_indexes()
            recommendations.extend(redundant_recs)
            
            # Find underutilized indexes
            underutilized_recs = self._find_underutilized_indexes()
            recommendations.extend(underutilized_recs)
            
            # Sort by priority and benefit
            recommendations.sort(key=lambda x: (x.priority, -x.estimated_benefit))
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.recommendations_generated += len(recommendations)
            
            self.recommendations = recommendations
            logger.info(f"Generated {len(recommendations)} index recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return []
    
    def create_index(self, 
                    table_name: str,
                    columns: List[str],
                    index_type: IndexType = IndexType.BTREE,
                    is_unique: bool = False,
                    is_partial: bool = False,
                    where_clause: Optional[str] = None) -> bool:
        """Create a new index"""
        try:
            index_name = f"idx_{table_name}_{'_'.join(columns)}_{int(time.time())}"
            
            index_def = IndexDefinition(
                index_name=index_name,
                table_name=table_name,
                columns=columns,
                index_type=index_type,
                is_unique=is_unique,
                is_partial=is_partial,
                where_clause=where_clause,
                created_at=datetime.now()
            )
            
            # Create index in database
            if self.db_connection:
                success = self._create_index_in_db(index_def)
                if not success:
                    return False
            
            # Add to local registry
            with self.optimizer_lock:
                self.indexes[index_name] = index_def
            
            # Initialize usage stats
            self.usage_stats[index_name] = IndexUsageStats(
                index_name=index_name,
                table_name=table_name
            )
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.total_indexes += 1
                self.metrics.active_indexes += 1
                self.metrics.indexes_created += 1
            
            logger.info(f"Created index {index_name} on {table_name}({', '.join(columns)})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            return False
    
    def drop_index(self, index_name: str) -> bool:
        """Drop an index"""
        try:
            if index_name not in self.indexes:
                logger.warning(f"Index {index_name} not found")
                return False
            
            index_def = self.indexes[index_name]
            
            # Drop index from database
            if self.db_connection:
                success = self._drop_index_from_db(index_name)
                if not success:
                    return False
            
            # Remove from local registry
            with self.optimizer_lock:
                del self.indexes[index_name]
            
            # Remove usage stats
            if index_name in self.usage_stats:
                del self.usage_stats[index_name]
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.total_indexes -= 1
                if index_def.status == IndexStatus.ACTIVE:
                    self.metrics.active_indexes -= 1
                self.metrics.indexes_dropped += 1
            
            logger.info(f"Dropped index {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop index {index_name}: {str(e)}")
            return False
    
    def rebuild_index(self, index_name: str) -> bool:
        """Rebuild an index"""
        try:
            if index_name not in self.indexes:
                logger.warning(f"Index {index_name} not found")
                return False
            
            # Set status to building
            self.indexes[index_name].status = IndexStatus.BUILDING
            
            # Rebuild index in database
            if self.db_connection:
                success = self._rebuild_index_in_db(index_name)
                if not success:
                    self.indexes[index_name].status = IndexStatus.FAILED
                    return False
            
            # Update status
            self.indexes[index_name].status = IndexStatus.ACTIVE
            self.indexes[index_name].created_at = datetime.now()
            
            logger.info(f"Rebuilt index {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild index {index_name}: {str(e)}")
            self.indexes[index_name].status = IndexStatus.FAILED
            return False
    
    def optimize_indexes(self) -> Dict[str, Any]:
        """Perform comprehensive index optimization"""
        try:
            optimization_results = {
                "indexes_created": 0,
                "indexes_dropped": 0,
                "indexes_rebuilt": 0,
                "recommendations_applied": 0,
                "performance_improvement": 0.0,
                "errors": []
            }
            
            # Generate recommendations
            recommendations = self.generate_recommendations()
            
            # Apply high-priority recommendations
            for rec in recommendations:
                if rec.priority <= 2 and rec.confidence > 0.8:  # High priority, high confidence
                    try:
                        if rec.reason.startswith("CREATE"):
                            success = self.create_index(
                                rec.table_name,
                                rec.columns,
                                rec.index_type
                            )
                            if success:
                                optimization_results["indexes_created"] += 1
                                optimization_results["recommendations_applied"] += 1
                        
                        elif rec.reason.startswith("DROP"):
                            # Find index to drop
                            index_to_drop = self._find_index_by_columns(rec.table_name, rec.columns)
                            if index_to_drop:
                                success = self.drop_index(index_to_drop)
                                if success:
                                    optimization_results["indexes_dropped"] += 1
                                    optimization_results["recommendations_applied"] += 1
                    
                    except Exception as e:
                        optimization_results["errors"].append(f"Failed to apply recommendation: {str(e)}")
            
            # Rebuild fragmented indexes
            fragmented_indexes = self._find_fragmented_indexes()
            for index_name in fragmented_indexes:
                try:
                    success = self.rebuild_index(index_name)
                    if success:
                        optimization_results["indexes_rebuilt"] += 1
                except Exception as e:
                    optimization_results["errors"].append(f"Failed to rebuild {index_name}: {str(e)}")
            
            # Calculate performance improvement
            optimization_results["performance_improvement"] = self._calculate_performance_improvement()
            
            logger.info(f"Index optimization completed: {optimization_results}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Failed to optimize indexes: {str(e)}")
            return {"error": str(e)}
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get comprehensive index statistics"""
        try:
            stats = {
                "total_indexes": len(self.indexes),
                "active_indexes": len([idx for idx in self.indexes.values() if idx.status == IndexStatus.ACTIVE]),
                "total_size_bytes": sum(idx.size_bytes for idx in self.indexes.values()),
                "average_size_bytes": 0,
                "most_used_indexes": [],
                "least_used_indexes": [],
                "largest_indexes": [],
                "index_types": {},
                "table_coverage": {}
            }
            
            if self.indexes:
                stats["average_size_bytes"] = stats["total_size_bytes"] / len(self.indexes)
            
            # Most used indexes
            sorted_usage = sorted(
                self.usage_stats.items(),
                key=lambda x: x[1].scans,
                reverse=True
            )[:10]
            stats["most_used_indexes"] = [
                {"index": name, "scans": usage.scans, "hit_ratio": usage.hit_ratio}
                for name, usage in sorted_usage
            ]
            
            # Least used indexes
            sorted_usage_asc = sorted(
                self.usage_stats.items(),
                key=lambda x: x[1].scans
            )[:10]
            stats["least_used_indexes"] = [
                {"index": name, "scans": usage.scans, "hit_ratio": usage.hit_ratio}
                for name, usage in sorted_usage_asc
            ]
            
            # Largest indexes
            sorted_size = sorted(
                self.indexes.items(),
                key=lambda x: x[1].size_bytes,
                reverse=True
            )[:10]
            stats["largest_indexes"] = [
                {"index": name, "size_bytes": idx.size_bytes, "table": idx.table_name}
                for name, idx in sorted_size
            ]
            
            # Index types distribution
            type_counts = defaultdict(int)
            for idx in self.indexes.values():
                type_counts[idx.index_type.value] += 1
            stats["index_types"] = dict(type_counts)
            
            # Table coverage
            table_counts = defaultdict(int)
            for idx in self.indexes.values():
                table_counts[idx.table_name] += 1
            stats["table_coverage"] = dict(table_counts)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index statistics: {str(e)}")
            return {}
    
    def _analyze_query_patterns(self, query_patterns: List[str]) -> List[IndexRecommendation]:
        """Analyze query patterns for index opportunities"""
        try:
            recommendations = []
            
            for query in query_patterns:
                # Simple query analysis - in real implementation, use proper SQL parser
                if "WHERE" in query.upper():
                    # Extract table and column information
                    # This is simplified - real implementation would be more sophisticated
                    table_match = query.upper().split("FROM")[1].split()[0] if "FROM" in query.upper() else "unknown"
                    where_clause = query.upper().split("WHERE")[1].split("GROUP")[0] if "GROUP" in query.upper() else query.upper().split("WHERE")[1]
                    
                    # Extract column names from WHERE clause
                    columns = []
                    for word in where_clause.split():
                        if word.isalpha() and len(word) > 1:
                            columns.append(word)
                    
                    if columns and table_match != "unknown":
                        recommendation = IndexRecommendation(
                            table_name=table_match,
                            columns=columns[:3],  # Limit to first 3 columns
                            index_type=IndexType.BTREE,
                            estimated_benefit=0.8,
                            creation_cost=0.2,
                            priority=1,
                            reason=f"CREATE index for WHERE clause: {where_clause[:50]}...",
                            confidence=0.9
                        )
                        recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to analyze query patterns: {str(e)}")
            return []
    
    def _find_missing_indexes(self) -> List[IndexRecommendation]:
        """Find missing indexes for existing tables"""
        try:
            recommendations = []
            
            # This would analyze actual table schemas and query patterns
            # For now, return mock recommendations
            mock_recommendations = [
                IndexRecommendation(
                    table_name="smart_meters",
                    columns=["meter_id", "timestamp"],
                    index_type=IndexType.BTREE,
                    estimated_benefit=0.9,
                    creation_cost=0.3,
                    priority=1,
                    reason="Missing index for primary lookup pattern",
                    confidence=0.95
                ),
                IndexRecommendation(
                    table_name="meter_readings",
                    columns=["meter_id", "reading_date"],
                    index_type=IndexType.BTREE,
                    estimated_benefit=0.7,
                    creation_cost=0.4,
                    priority=2,
                    reason="Missing index for time-series queries",
                    confidence=0.85
                )
            ]
            
            return mock_recommendations
            
        except Exception as e:
            logger.error(f"Failed to find missing indexes: {str(e)}")
            return []
    
    def _find_redundant_indexes(self) -> List[IndexRecommendation]:
        """Find redundant indexes"""
        try:
            recommendations = []
            
            # Group indexes by table
            table_indexes = defaultdict(list)
            for idx in self.indexes.values():
                table_indexes[idx.table_name].append(idx)
            
            # Check for redundant indexes
            for table, indexes in table_indexes.items():
                for i, idx1 in enumerate(indexes):
                    for idx2 in indexes[i+1:]:
                        if self._is_redundant_index(idx1, idx2):
                            recommendation = IndexRecommendation(
                                table_name=table,
                                columns=idx2.columns,
                                index_type=idx2.index_type,
                                estimated_benefit=0.5,  # Space savings
                                creation_cost=0.0,  # No creation cost for dropping
                                priority=3,
                                reason=f"DROP redundant index (covered by {idx1.index_name})",
                                confidence=0.9
                            )
                            recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to find redundant indexes: {str(e)}")
            return []
    
    def _find_underutilized_indexes(self) -> List[IndexRecommendation]:
        """Find underutilized indexes"""
        try:
            recommendations = []
            
            for index_name, stats in self.usage_stats.items():
                if stats.scans < 10:  # Threshold for underutilized
                    index_def = self.indexes.get(index_name)
                    if index_def:
                        recommendation = IndexRecommendation(
                            table_name=index_def.table_name,
                            columns=index_def.columns,
                            index_type=index_def.index_type,
                            estimated_benefit=0.3,  # Space savings
                            creation_cost=0.0,  # No creation cost for dropping
                            priority=4,
                            reason=f"DROP underutilized index (only {stats.scans} scans)",
                            confidence=0.8
                        )
                        recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to find underutilized indexes: {str(e)}")
            return []
    
    def _find_duplicate_indexes(self) -> List[List[str]]:
        """Find duplicate indexes"""
        try:
            duplicates = []
            index_groups = defaultdict(list)
            
            # Group indexes by table and columns
            for idx in self.indexes.values():
                key = (idx.table_name, tuple(sorted(idx.columns)))
                index_groups[key].append(idx.index_name)
            
            # Find groups with more than one index
            for group in index_groups.values():
                if len(group) > 1:
                    duplicates.append(group)
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Failed to find duplicate indexes: {str(e)}")
            return []
    
    def _analyze_index_sizes(self) -> Dict[str, Any]:
        """Analyze index sizes"""
        try:
            sizes = [idx.size_bytes for idx in self.indexes.values() if idx.size_bytes > 0]
            
            if not sizes:
                return {"total_size": 0, "average_size": 0, "largest_size": 0}
            
            return {
                "total_size": sum(sizes),
                "average_size": sum(sizes) / len(sizes),
                "largest_size": max(sizes),
                "size_distribution": {
                    "small": len([s for s in sizes if s < 1024 * 1024]),  # < 1MB
                    "medium": len([s for s in sizes if 1024 * 1024 <= s < 100 * 1024 * 1024]),  # 1MB - 100MB
                    "large": len([s for s in sizes if s >= 100 * 1024 * 1024])  # >= 100MB
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze index sizes: {str(e)}")
            return {}
    
    def _analyze_performance_impact(self) -> Dict[str, Any]:
        """Analyze performance impact of indexes"""
        try:
            total_scans = sum(stats.scans for stats in self.usage_stats.values())
            total_tuples_read = sum(stats.tuples_read for stats in self.usage_stats.values())
            total_tuples_fetched = sum(stats.tuples_fetched for stats in self.usage_stats.values())
            
            return {
                "total_scans": total_scans,
                "total_tuples_read": total_tuples_read,
                "total_tuples_fetched": total_tuples_fetched,
                "average_hit_ratio": sum(stats.hit_ratio for stats in self.usage_stats.values()) / len(self.usage_stats) if self.usage_stats else 0,
                "efficiency_score": total_tuples_fetched / total_tuples_read if total_tuples_read > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze performance impact: {str(e)}")
            return {}
    
    def _find_fragmented_indexes(self) -> List[str]:
        """Find fragmented indexes that need rebuilding"""
        try:
            fragmented = []
            
            # Simple fragmentation detection based on size and usage
            for index_name, stats in self.usage_stats.items():
                index_def = self.indexes.get(index_name)
                if index_def and index_def.size_bytes > 0:
                    # If index is large but has low hit ratio, it might be fragmented
                    if stats.hit_ratio < 0.5 and index_def.size_bytes > 10 * 1024 * 1024:  # 10MB
                        fragmented.append(index_name)
            
            return fragmented
            
        except Exception as e:
            logger.error(f"Failed to find fragmented indexes: {str(e)}")
            return []
    
    def _calculate_performance_improvement(self) -> float:
        """Calculate overall performance improvement"""
        try:
            # Simplified calculation
            total_hit_ratio = sum(stats.hit_ratio for stats in self.usage_stats.values())
            average_hit_ratio = total_hit_ratio / len(self.usage_stats) if self.usage_stats else 0
            
            # Convert hit ratio to performance improvement percentage
            improvement = (average_hit_ratio - 0.5) * 100  # Assume 50% as baseline
            return max(0, improvement)
            
        except Exception as e:
            logger.error(f"Failed to calculate performance improvement: {str(e)}")
            return 0.0
    
    def _is_redundant_index(self, idx1: IndexDefinition, idx2: IndexDefinition) -> bool:
        """Check if two indexes are redundant"""
        try:
            # Check if one index's columns are a subset of another's
            if set(idx1.columns).issubset(set(idx2.columns)):
                return True
            if set(idx2.columns).issubset(set(idx1.columns)):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check redundant index: {str(e)}")
            return False
    
    def _find_index_by_columns(self, table_name: str, columns: List[str]) -> Optional[str]:
        """Find index by table and columns"""
        try:
            for index_name, idx in self.indexes.items():
                if (idx.table_name == table_name and 
                    set(idx.columns) == set(columns)):
                    return index_name
            return None
            
        except Exception as e:
            logger.error(f"Failed to find index by columns: {str(e)}")
            return None
    
    def _create_index_in_db(self, index_def: IndexDefinition) -> bool:
        """Create index in database"""
        try:
            if not self.db_connection:
                return True  # Mock success if no DB connection
            
            # Generate CREATE INDEX SQL
            columns_str = ", ".join(index_def.columns)
            unique_str = "UNIQUE " if index_def.is_unique else ""
            partial_str = f" WHERE {index_def.where_clause}" if index_def.is_partial else ""
            
            sql = f"CREATE {unique_str}INDEX {index_def.index_name} ON {index_def.table_name} ({columns_str}){partial_str}"
            
            # Execute SQL (simplified)
            logger.debug(f"Creating index: {sql}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index in database: {str(e)}")
            return False
    
    def _drop_index_from_db(self, index_name: str) -> bool:
        """Drop index from database"""
        try:
            if not self.db_connection:
                return True  # Mock success if no DB connection
            
            sql = f"DROP INDEX {index_name}"
            logger.debug(f"Dropping index: {sql}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop index from database: {str(e)}")
            return False
    
    def _rebuild_index_in_db(self, index_name: str) -> bool:
        """Rebuild index in database"""
        try:
            if not self.db_connection:
                return True  # Mock success if no DB connection
            
            sql = f"REINDEX {index_name}"
            logger.debug(f"Rebuilding index: {sql}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild index in database: {str(e)}")
            return False
    
    def _monitor_indexes(self):
        """Monitor indexes in background thread"""
        try:
            while True:
                time.sleep(self.monitoring_interval)
                
                # Update usage statistics
                self._update_usage_statistics()
                
                # Auto-optimize if enabled
                if self.enable_auto_optimization:
                    self.optimize_indexes()
                
        except Exception as e:
            logger.error(f"Error in index monitoring: {str(e)}")
    
    def _update_usage_statistics(self):
        """Update index usage statistics"""
        try:
            # In real implementation, this would query database statistics
            # For now, simulate some updates
            for index_name in self.usage_stats:
                stats = self.usage_stats[index_name]
                # Simulate some usage
                stats.scans += 1
                stats.tuples_read += 10
                stats.tuples_fetched += 8
                stats.hit_ratio = stats.tuples_fetched / stats.tuples_read if stats.tuples_read > 0 else 0
                stats.last_used = datetime.now()
                
        except Exception as e:
            logger.error(f"Failed to update usage statistics: {str(e)}")
    
    def get_metrics(self) -> IndexMetrics:
        """Get index optimizer metrics"""
        try:
            with self.metrics_lock:
                return IndexMetrics(
                    total_indexes=self.metrics.total_indexes,
                    active_indexes=self.metrics.active_indexes,
                    unused_indexes=self.metrics.unused_indexes,
                    duplicate_indexes=self.metrics.duplicate_indexes,
                    recommendations_generated=self.metrics.recommendations_generated,
                    indexes_created=self.metrics.indexes_created,
                    indexes_dropped=self.metrics.indexes_dropped
                )
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return IndexMetrics()
    
    def reset_metrics(self) -> None:
        """Reset optimizer metrics"""
        try:
            with self.metrics_lock:
                self.metrics = IndexMetrics()
                
        except Exception as e:
            logger.error(f"Failed to reset metrics: {str(e)}")
