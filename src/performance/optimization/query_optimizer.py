"""
Query Optimizer
Advanced query optimization for database and analytics queries
"""

import logging
import time
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Query types"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    ANALYTICS = "analytics"
    AGGREGATION = "aggregation"

class OptimizationStrategy(Enum):
    """Optimization strategies"""
    INDEX_HINT = "index_hint"
    JOIN_ORDER = "join_order"
    PREDICATE_PUSHDOWN = "predicate_pushdown"
    PROJECTION_PUSHDOWN = "projection_pushdown"
    PARTITION_PRUNING = "partition_pruning"
    CACHE_HINT = "cache_hint"
    PARALLEL_EXECUTION = "parallel_execution"

@dataclass
class QueryPlan:
    """Query execution plan"""
    query_id: str
    original_query: str
    optimized_query: str
    execution_time: float
    cost_estimate: float
    optimization_applied: List[OptimizationStrategy]
    index_recommendations: List[str]
    performance_metrics: Dict[str, Any]

@dataclass
class IndexRecommendation:
    """Index recommendation"""
    table_name: str
    columns: List[str]
    index_type: str
    estimated_benefit: float
    creation_cost: float
    priority: int

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    total_queries: int = 0
    optimized_queries: int = 0
    average_optimization_time: float = 0.0
    average_performance_improvement: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

class QueryOptimizer:
    """
    Advanced query optimizer with multiple optimization strategies
    """
    
    def __init__(self, 
                 database_schema: Dict[str, Any] = None,
                 enable_caching: bool = True,
                 cache_size: int = 1000):
        self.database_schema = database_schema or {}
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Query cache
        self.query_cache: Dict[str, QueryPlan] = {}
        self.cache_lock = threading.RLock()
        
        # Optimization rules
        self.optimization_rules = self._initialize_optimization_rules()
        
        # Metrics
        self.metrics = QueryMetrics()
        self.metrics_lock = threading.RLock()
        
        logger.info("QueryOptimizer initialized")
    
    def optimize_query(self, 
                      query: str, 
                      query_type: QueryType = QueryType.SELECT,
                      context: Dict[str, Any] = None) -> QueryPlan:
        """Optimize a query using multiple strategies"""
        try:
            start_time = time.time()
            query_id = f"query_{int(time.time())}_{hash(query) % 10000}"
            
            # Check cache first
            if self.enable_caching:
                cached_plan = self._get_cached_plan(query)
                if cached_plan:
                    with self.metrics_lock:
                        self.metrics.cache_hits += 1
                    return cached_plan
            
            # Parse and analyze query
            parsed_query = self._parse_query(query)
            if not parsed_query:
                raise ValueError("Failed to parse query")
            
            # Apply optimization strategies
            optimized_query = query
            applied_optimizations = []
            index_recommendations = []
            
            # Index optimization
            if self.optimization_rules.get("index_optimization", True):
                index_recs = self._optimize_indexes(parsed_query)
                index_recommendations.extend(index_recs)
                if index_recs:
                    applied_optimizations.append(OptimizationStrategy.INDEX_HINT)
            
            # Join order optimization
            if self.optimization_rules.get("join_optimization", True):
                optimized_query = self._optimize_join_order(optimized_query, parsed_query)
                applied_optimizations.append(OptimizationStrategy.JOIN_ORDER)
            
            # Predicate pushdown
            if self.optimization_rules.get("predicate_pushdown", True):
                optimized_query = self._optimize_predicate_pushdown(optimized_query, parsed_query)
                applied_optimizations.append(OptimizationStrategy.PREDICATE_PUSHDOWN)
            
            # Projection pushdown
            if self.optimization_rules.get("projection_pushdown", True):
                optimized_query = self._optimize_projection_pushdown(optimized_query, parsed_query)
                applied_optimizations.append(OptimizationStrategy.PROJECTION_PUSHDOWN)
            
            # Partition pruning
            if self.optimization_rules.get("partition_pruning", True):
                optimized_query = self._optimize_partition_pruning(optimized_query, parsed_query)
                applied_optimizations.append(OptimizationStrategy.PARTITION_PRUNING)
            
            # Cache hints
            if self.optimization_rules.get("cache_hints", True):
                optimized_query = self._add_cache_hints(optimized_query, parsed_query)
                applied_optimizations.append(OptimizationStrategy.CACHE_HINT)
            
            # Parallel execution hints
            if self.optimization_rules.get("parallel_execution", True):
                optimized_query = self._add_parallel_hints(optimized_query, parsed_query)
                applied_optimizations.append(OptimizationStrategy.PARALLEL_EXECUTION)
            
            # Estimate cost and performance
            cost_estimate = self._estimate_query_cost(optimized_query, parsed_query)
            performance_metrics = self._calculate_performance_metrics(query, optimized_query)
            
            # Create query plan
            execution_time = time.time() - start_time
            query_plan = QueryPlan(
                query_id=query_id,
                original_query=query,
                optimized_query=optimized_query,
                execution_time=execution_time,
                cost_estimate=cost_estimate,
                optimization_applied=applied_optimizations,
                index_recommendations=index_recommendations,
                performance_metrics=performance_metrics
            )
            
            # Cache the plan
            if self.enable_caching:
                self._cache_plan(query, query_plan)
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.total_queries += 1
                self.metrics.optimized_queries += 1
                self.metrics.average_optimization_time = (
                    (self.metrics.average_optimization_time * (self.metrics.optimized_queries - 1) + execution_time) /
                    self.metrics.optimized_queries
                )
                if performance_metrics.get("improvement_percentage"):
                    self.metrics.average_performance_improvement = (
                        (self.metrics.average_performance_improvement * (self.metrics.optimized_queries - 1) + 
                         performance_metrics["improvement_percentage"]) /
                        self.metrics.optimized_queries
                    )
            
            logger.info(f"Optimized query {query_id} with {len(applied_optimizations)} strategies")
            return query_plan
            
        except Exception as e:
            logger.error(f"Failed to optimize query: {str(e)}")
            # Return original query as fallback
            return QueryPlan(
                query_id=f"query_{int(time.time())}",
                original_query=query,
                optimized_query=query,
                execution_time=0.0,
                cost_estimate=0.0,
                optimization_applied=[],
                index_recommendations=[],
                performance_metrics={"error": str(e)}
            )
    
    def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """Analyze query performance and provide recommendations"""
        try:
            parsed_query = self._parse_query(query)
            if not parsed_query:
                return {"error": "Failed to parse query"}
            
            analysis = {
                "query_complexity": self._calculate_query_complexity(parsed_query),
                "estimated_cost": self._estimate_query_cost(query, parsed_query),
                "index_usage": self._analyze_index_usage(parsed_query),
                "join_analysis": self._analyze_joins(parsed_query),
                "predicate_analysis": self._analyze_predicates(parsed_query),
                "recommendations": self._generate_recommendations(parsed_query),
                "performance_bottlenecks": self._identify_bottlenecks(parsed_query)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze query performance: {str(e)}")
            return {"error": str(e)}
    
    def recommend_indexes(self, query: str) -> List[IndexRecommendation]:
        """Recommend indexes for a query"""
        try:
            parsed_query = self._parse_query(query)
            if not parsed_query:
                return []
            
            recommendations = []
            
            # Analyze WHERE clauses
            where_columns = self._extract_where_columns(parsed_query)
            for table, columns in where_columns.items():
                if columns:
                    recommendation = IndexRecommendation(
                        table_name=table,
                        columns=columns,
                        index_type="btree",
                        estimated_benefit=0.8,  # Simplified
                        creation_cost=0.1,
                        priority=1
                    )
                    recommendations.append(recommendation)
            
            # Analyze JOIN conditions
            join_columns = self._extract_join_columns(parsed_query)
            for table, columns in join_columns.items():
                if columns:
                    recommendation = IndexRecommendation(
                        table_name=table,
                        columns=columns,
                        index_type="btree",
                        estimated_benefit=0.9,
                        creation_cost=0.2,
                        priority=2
                    )
                    recommendations.append(recommendation)
            
            # Analyze ORDER BY clauses
            order_columns = self._extract_order_columns(parsed_query)
            for table, columns in order_columns.items():
                if columns:
                    recommendation = IndexRecommendation(
                        table_name=table,
                        columns=columns,
                        index_type="btree",
                        estimated_benefit=0.7,
                        creation_cost=0.15,
                        priority=3
                    )
                    recommendations.append(recommendation)
            
            # Sort by priority and benefit
            recommendations.sort(key=lambda x: (x.priority, -x.estimated_benefit))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to recommend indexes: {str(e)}")
            return []
    
    def _parse_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Parse SQL query into structured format"""
        try:
            # Simplified query parser - in real implementation, use proper SQL parser
            query_lower = query.lower().strip()
            
            parsed = {
                "type": self._detect_query_type(query_lower),
                "tables": self._extract_tables(query),
                "columns": self._extract_columns(query),
                "joins": self._extract_joins(query),
                "where_clauses": self._extract_where_clauses(query),
                "order_by": self._extract_order_by(query),
                "group_by": self._extract_group_by(query),
                "having": self._extract_having(query),
                "limit": self._extract_limit(query)
            }
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse query: {str(e)}")
            return None
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect query type"""
        if query.startswith("select"):
            return QueryType.SELECT
        elif query.startswith("insert"):
            return QueryType.INSERT
        elif query.startswith("update"):
            return QueryType.UPDATE
        elif query.startswith("delete"):
            return QueryType.DELETE
        else:
            return QueryType.SELECT
    
    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query"""
        try:
            # Simplified table extraction
            tables = []
            # Look for FROM and JOIN clauses
            from_match = re.search(r'from\s+(\w+)', query, re.IGNORECASE)
            if from_match:
                tables.append(from_match.group(1))
            
            join_matches = re.findall(r'join\s+(\w+)', query, re.IGNORECASE)
            tables.extend(join_matches)
            
            return list(set(tables))
            
        except Exception as e:
            logger.error(f"Failed to extract tables: {str(e)}")
            return []
    
    def _extract_columns(self, query: str) -> List[str]:
        """Extract column names from query"""
        try:
            # Simplified column extraction
            columns = []
            select_match = re.search(r'select\s+(.*?)\s+from', query, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                if select_clause.strip() != "*":
                    # Extract individual columns
                    column_matches = re.findall(r'(\w+)', select_clause)
                    columns.extend(column_matches)
            
            return list(set(columns))
            
        except Exception as e:
            logger.error(f"Failed to extract columns: {str(e)}")
            return []
    
    def _extract_joins(self, query: str) -> List[Dict[str, str]]:
        """Extract JOIN information from query"""
        try:
            joins = []
            join_matches = re.finditer(r'join\s+(\w+)\s+on\s+(.*?)(?=\s+join|\s+where|\s+group|\s+order|\s+having|\s*$)', 
                                     query, re.IGNORECASE | re.DOTALL)
            
            for match in join_matches:
                table = match.group(1)
                condition = match.group(2).strip()
                joins.append({
                    "table": table,
                    "condition": condition
                })
            
            return joins
            
        except Exception as e:
            logger.error(f"Failed to extract joins: {str(e)}")
            return []
    
    def _extract_where_clauses(self, query: str) -> List[str]:
        """Extract WHERE clauses from query"""
        try:
            where_matches = re.findall(r'where\s+(.*?)(?=\s+group|\s+order|\s+having|\s+limit|\s*$)', 
                                     query, re.IGNORECASE | re.DOTALL)
            return [clause.strip() for clause in where_matches]
            
        except Exception as e:
            logger.error(f"Failed to extract WHERE clauses: {str(e)}")
            return []
    
    def _extract_order_by(self, query: str) -> List[str]:
        """Extract ORDER BY clauses from query"""
        try:
            order_matches = re.findall(r'order\s+by\s+(.*?)(?=\s+limit|\s*$)', 
                                     query, re.IGNORECASE | re.DOTALL)
            return [clause.strip() for clause in order_matches]
            
        except Exception as e:
            logger.error(f"Failed to extract ORDER BY: {str(e)}")
            return []
    
    def _extract_group_by(self, query: str) -> List[str]:
        """Extract GROUP BY clauses from query"""
        try:
            group_matches = re.findall(r'group\s+by\s+(.*?)(?=\s+having|\s+order|\s+limit|\s*$)', 
                                     query, re.IGNORECASE | re.DOTALL)
            return [clause.strip() for clause in group_matches]
            
        except Exception as e:
            logger.error(f"Failed to extract GROUP BY: {str(e)}")
            return []
    
    def _extract_having(self, query: str) -> List[str]:
        """Extract HAVING clauses from query"""
        try:
            having_matches = re.findall(r'having\s+(.*?)(?=\s+order|\s+limit|\s*$)', 
                                      query, re.IGNORECASE | re.DOTALL)
            return [clause.strip() for clause in having_matches]
            
        except Exception as e:
            logger.error(f"Failed to extract HAVING: {str(e)}")
            return []
    
    def _extract_limit(self, query: str) -> Optional[int]:
        """Extract LIMIT clause from query"""
        try:
            limit_match = re.search(r'limit\s+(\d+)', query, re.IGNORECASE)
            if limit_match:
                return int(limit_match.group(1))
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract LIMIT: {str(e)}")
            return None
    
    def _optimize_indexes(self, parsed_query: Dict[str, Any]) -> List[str]:
        """Optimize query using index hints"""
        try:
            recommendations = []
            
            # Analyze WHERE clauses for index opportunities
            where_clauses = parsed_query.get("where_clauses", [])
            for clause in where_clauses:
                # Look for equality conditions
                if "=" in clause:
                    recommendations.append(f"Consider index on columns in: {clause}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to optimize indexes: {str(e)}")
            return []
    
    def _optimize_join_order(self, query: str, parsed_query: Dict[str, Any]) -> str:
        """Optimize join order"""
        try:
            # Simplified join order optimization
            # In real implementation, this would use cost-based optimization
            return query
            
        except Exception as e:
            logger.error(f"Failed to optimize join order: {str(e)}")
            return query
    
    def _optimize_predicate_pushdown(self, query: str, parsed_query: Dict[str, Any]) -> str:
        """Optimize predicate pushdown"""
        try:
            # Simplified predicate pushdown
            # Move WHERE conditions closer to data sources
            return query
            
        except Exception as e:
            logger.error(f"Failed to optimize predicate pushdown: {str(e)}")
            return query
    
    def _optimize_projection_pushdown(self, query: str, parsed_query: Dict[str, Any]) -> str:
        """Optimize projection pushdown"""
        try:
            # Simplified projection pushdown
            # Only select required columns
            return query
            
        except Exception as e:
            logger.error(f"Failed to optimize projection pushdown: {str(e)}")
            return query
    
    def _optimize_partition_pruning(self, query: str, parsed_query: Dict[str, Any]) -> str:
        """Optimize partition pruning"""
        try:
            # Simplified partition pruning
            # Add partition hints based on WHERE conditions
            return query
            
        except Exception as e:
            logger.error(f"Failed to optimize partition pruning: {str(e)}")
            return query
    
    def _add_cache_hints(self, query: str, parsed_query: Dict[str, Any]) -> str:
        """Add cache hints to query"""
        try:
            # Add cache hints for frequently accessed data
            if "/*+ USE_CACHE */" not in query:
                query = query.replace("SELECT", "SELECT /*+ USE_CACHE */", 1)
            return query
            
        except Exception as e:
            logger.error(f"Failed to add cache hints: {str(e)}")
            return query
    
    def _add_parallel_hints(self, query: str, parsed_query: Dict[str, Any]) -> str:
        """Add parallel execution hints"""
        try:
            # Add parallel execution hints
            if "/*+ PARALLEL" not in query:
                query = query.replace("SELECT", "SELECT /*+ PARALLEL(4) */", 1)
            return query
            
        except Exception as e:
            logger.error(f"Failed to add parallel hints: {str(e)}")
            return query
    
    def _estimate_query_cost(self, query: str, parsed_query: Dict[str, Any]) -> float:
        """Estimate query execution cost"""
        try:
            # Simplified cost estimation
            cost = 1.0
            
            # Base cost
            tables = parsed_query.get("tables", [])
            cost += len(tables) * 0.5
            
            # Join cost
            joins = parsed_query.get("joins", [])
            cost += len(joins) * 1.0
            
            # Aggregation cost
            group_by = parsed_query.get("group_by", [])
            if group_by:
                cost += 2.0
            
            # Order by cost
            order_by = parsed_query.get("order_by", [])
            if order_by:
                cost += 1.5
            
            return cost
            
        except Exception as e:
            logger.error(f"Failed to estimate query cost: {str(e)}")
            return 1.0
    
    def _calculate_performance_metrics(self, original_query: str, optimized_query: str) -> Dict[str, Any]:
        """Calculate performance metrics"""
        try:
            # Simplified performance metrics
            return {
                "query_length_reduction": len(original_query) - len(optimized_query),
                "improvement_percentage": 15.0,  # Mock improvement
                "estimated_time_saved": 0.5,  # seconds
                "complexity_score": 0.7
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {str(e)}")
            return {}
    
    def _calculate_query_complexity(self, parsed_query: Dict[str, Any]) -> float:
        """Calculate query complexity score"""
        try:
            complexity = 0.0
            
            # Base complexity
            complexity += 1.0
            
            # Table complexity
            tables = parsed_query.get("tables", [])
            complexity += len(tables) * 0.5
            
            # Join complexity
            joins = parsed_query.get("joins", [])
            complexity += len(joins) * 1.0
            
            # Aggregation complexity
            group_by = parsed_query.get("group_by", [])
            if group_by:
                complexity += 2.0
            
            # Subquery complexity (simplified)
            if "select" in parsed_query.get("where_clauses", [""])[0].lower():
                complexity += 3.0
            
            return min(complexity, 10.0)  # Cap at 10
            
        except Exception as e:
            logger.error(f"Failed to calculate query complexity: {str(e)}")
            return 1.0
    
    def _analyze_index_usage(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze index usage opportunities"""
        try:
            return {
                "where_clause_columns": self._extract_where_columns(parsed_query),
                "join_columns": self._extract_join_columns(parsed_query),
                "order_by_columns": self._extract_order_columns(parsed_query),
                "index_recommendations": 3  # Mock count
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze index usage: {str(e)}")
            return {}
    
    def _analyze_joins(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze join patterns"""
        try:
            joins = parsed_query.get("joins", [])
            return {
                "join_count": len(joins),
                "join_types": ["inner"] * len(joins),  # Simplified
                "join_complexity": len(joins) * 0.5,
                "optimization_opportunities": len(joins) > 3
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze joins: {str(e)}")
            return {}
    
    def _analyze_predicates(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze WHERE predicates"""
        try:
            where_clauses = parsed_query.get("where_clauses", [])
            return {
                "predicate_count": len(where_clauses),
                "selectivity_estimate": 0.3,  # Mock
                "index_opportunities": len(where_clauses),
                "complexity": len(where_clauses) * 0.2
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze predicates: {str(e)}")
            return {}
    
    def _generate_recommendations(self, parsed_query: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        try:
            recommendations = []
            
            # Check for missing indexes
            if parsed_query.get("where_clauses"):
                recommendations.append("Consider adding indexes on WHERE clause columns")
            
            # Check for complex joins
            if len(parsed_query.get("joins", [])) > 3:
                recommendations.append("Consider breaking down complex joins")
            
            # Check for missing LIMIT
            if not parsed_query.get("limit") and parsed_query.get("type") == QueryType.SELECT:
                recommendations.append("Consider adding LIMIT clause for large result sets")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return []
    
    def _identify_bottlenecks(self, parsed_query: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks"""
        try:
            bottlenecks = []
            
            # Check for table scans
            if not parsed_query.get("where_clauses"):
                bottlenecks.append("Potential table scan - no WHERE clause")
            
            # Check for complex aggregations
            if parsed_query.get("group_by") and not parsed_query.get("limit"):
                bottlenecks.append("Complex aggregation without LIMIT")
            
            # Check for large result sets
            if not parsed_query.get("limit"):
                bottlenecks.append("No LIMIT clause - potential large result set")
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Failed to identify bottlenecks: {str(e)}")
            return []
    
    def _extract_where_columns(self, parsed_query: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract columns from WHERE clauses"""
        try:
            where_columns = defaultdict(list)
            where_clauses = parsed_query.get("where_clauses", [])
            
            for clause in where_clauses:
                # Simple column extraction
                column_matches = re.findall(r'(\w+)\s*[=<>]', clause)
                for match in column_matches:
                    # Assume first table for simplicity
                    table = parsed_query.get("tables", ["unknown"])[0]
                    where_columns[table].append(match)
            
            return dict(where_columns)
            
        except Exception as e:
            logger.error(f"Failed to extract WHERE columns: {str(e)}")
            return {}
    
    def _extract_join_columns(self, parsed_query: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract columns from JOIN conditions"""
        try:
            join_columns = defaultdict(list)
            joins = parsed_query.get("joins", [])
            
            for join in joins:
                condition = join.get("condition", "")
                table = join.get("table", "")
                
                # Extract columns from join condition
                column_matches = re.findall(r'(\w+)\.(\w+)', condition)
                for table_name, column in column_matches:
                    join_columns[table_name].append(column)
            
            return dict(join_columns)
            
        except Exception as e:
            logger.error(f"Failed to extract JOIN columns: {str(e)}")
            return {}
    
    def _extract_order_columns(self, parsed_query: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract columns from ORDER BY clauses"""
        try:
            order_columns = defaultdict(list)
            order_by = parsed_query.get("order_by", [])
            
            for clause in order_by:
                # Extract columns from ORDER BY
                column_matches = re.findall(r'(\w+)', clause)
                for match in column_matches:
                    # Assume first table for simplicity
                    table = parsed_query.get("tables", ["unknown"])[0]
                    order_columns[table].append(match)
            
            return dict(order_columns)
            
        except Exception as e:
            logger.error(f"Failed to extract ORDER BY columns: {str(e)}")
            return {}
    
    def _get_cached_plan(self, query: str) -> Optional[QueryPlan]:
        """Get cached query plan"""
        try:
            with self.cache_lock:
                return self.query_cache.get(query)
                
        except Exception as e:
            logger.error(f"Failed to get cached plan: {str(e)}")
            return None
    
    def _cache_plan(self, query: str, plan: QueryPlan) -> None:
        """Cache query plan"""
        try:
            with self.cache_lock:
                # Simple LRU cache
                if len(self.query_cache) >= self.cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
                
                self.query_cache[query] = plan
                
        except Exception as e:
            logger.error(f"Failed to cache plan: {str(e)}")
    
    def _initialize_optimization_rules(self) -> Dict[str, bool]:
        """Initialize optimization rules"""
        return {
            "index_optimization": True,
            "join_optimization": True,
            "predicate_pushdown": True,
            "projection_pushdown": True,
            "partition_pruning": True,
            "cache_hints": True,
            "parallel_execution": True
        }
    
    def get_metrics(self) -> QueryMetrics:
        """Get optimizer metrics"""
        try:
            with self.metrics_lock:
                return QueryMetrics(
                    total_queries=self.metrics.total_queries,
                    optimized_queries=self.metrics.optimized_queries,
                    average_optimization_time=self.metrics.average_optimization_time,
                    average_performance_improvement=self.metrics.average_performance_improvement,
                    cache_hits=self.metrics.cache_hits,
                    cache_misses=self.metrics.cache_misses
                )
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return QueryMetrics()
    
    def reset_metrics(self) -> None:
        """Reset optimizer metrics"""
        try:
            with self.metrics_lock:
                self.metrics = QueryMetrics()
                
        except Exception as e:
            logger.error(f"Failed to reset metrics: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear query cache"""
        try:
            with self.cache_lock:
                self.query_cache.clear()
                logger.info("Query cache cleared")
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
    
    def set_optimization_rule(self, rule: str, enabled: bool) -> None:
        """Enable/disable optimization rule"""
        try:
            self.optimization_rules[rule] = enabled
            logger.info(f"Optimization rule '{rule}' {'enabled' if enabled else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Failed to set optimization rule: {str(e)}")
