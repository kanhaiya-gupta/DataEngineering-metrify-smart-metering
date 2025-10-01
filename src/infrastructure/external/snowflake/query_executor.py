"""
Snowflake Query Executor
Handles complex analytics queries and data transformations
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from .snowflake_client import SnowflakeClient
from ....core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class SnowflakeQueryExecutor:
    """
    Snowflake Query Executor
    
    Provides high-level methods for executing complex analytics queries
    and data transformations in Snowflake.
    """
    
    def __init__(self, snowflake_client: SnowflakeClient):
        self.client = snowflake_client
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get overall system overview and metrics"""
        try:
            # Get meter statistics
            meter_stats = await self.client.execute_query("""
                SELECT 
                    COUNT(*) as total_meters,
                    COUNT(CASE WHEN status = 'ACTIVE' THEN 1 END) as active_meters,
                    AVG(average_quality_score) as avg_quality_score
                FROM smart_meters
            """)
            
            # Get grid statistics
            grid_stats = await self.client.execute_query("""
                SELECT 
                    COUNT(*) as total_operators,
                    COUNT(CASE WHEN status = 'ACTIVE' THEN 1 END) as active_operators
                FROM grid_operators
            """)
            
            # Get weather statistics
            weather_stats = await self.client.execute_query("""
                SELECT 
                    COUNT(*) as total_stations,
                    COUNT(CASE WHEN status = 'ACTIVE' THEN 1 END) as active_stations,
                    AVG(average_quality_score) as avg_quality_score
                FROM weather_stations
            """)
            
            # Get recent data counts
            recent_data = await self.client.execute_query("""
                SELECT 
                    (SELECT COUNT(*) FROM meter_readings WHERE timestamp >= CURRENT_DATE) as today_readings,
                    (SELECT COUNT(*) FROM grid_statuses WHERE timestamp >= CURRENT_DATE) as today_statuses,
                    (SELECT COUNT(*) FROM weather_observations WHERE timestamp >= CURRENT_DATE) as today_observations
            """)
            
            return {
                "meters": meter_stats[0] if meter_stats else {},
                "grid_operators": grid_stats[0] if grid_stats else {},
                "weather_stations": weather_stats[0] if weather_stats else {},
                "recent_data": recent_data[0] if recent_data else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting system overview: {str(e)}")
            raise InfrastructureError(f"Failed to get system overview: {str(e)}", service="snowflake")
    
    async def get_quality_trends(
        self,
        days: int = 30,
        entity_type: str = "all"
    ) -> List[Dict[str, Any]]:
        """Get data quality trends over time"""
        try:
            queries = []
            
            if entity_type in ["all", "meters"]:
                queries.append("""
                    SELECT 
                        'meters' as entity_type,
                        DATE(timestamp) as date,
                        AVG(data_quality_score) as avg_quality_score,
                        COUNT(*) as record_count
                    FROM meter_readings
                    WHERE timestamp >= CURRENT_DATE - INTERVAL '%(days)s days'
                    GROUP BY DATE(timestamp)
                """)
            
            if entity_type in ["all", "grid"]:
                queries.append("""
                    SELECT 
                        'grid' as entity_type,
                        DATE(timestamp) as date,
                        AVG(data_quality_score) as avg_quality_score,
                        COUNT(*) as record_count
                    FROM grid_statuses
                    WHERE timestamp >= CURRENT_DATE - INTERVAL '%(days)s days'
                    GROUP BY DATE(timestamp)
                """)
            
            if entity_type in ["all", "weather"]:
                queries.append("""
                    SELECT 
                        'weather' as entity_type,
                        DATE(timestamp) as date,
                        AVG(data_quality_score) as avg_quality_score,
                        COUNT(*) as record_count
                    FROM weather_observations
                    WHERE timestamp >= CURRENT_DATE - INTERVAL '%(days)s days'
                    GROUP BY DATE(timestamp)
                """)
            
            if not queries:
                return []
            
            union_query = " UNION ALL ".join(queries) + " ORDER BY entity_type, date"
            
            return await self.client.execute_query(union_query, {"days": days})
            
        except Exception as e:
            logger.error(f"Error getting quality trends: {str(e)}")
            raise InfrastructureError(f"Failed to get quality trends: {str(e)}", service="snowflake")
    
    async def get_anomaly_analysis(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get comprehensive anomaly analysis"""
        try:
            # Get anomaly counts by type
            anomaly_counts = await self.client.execute_query("""
                SELECT 
                    'meter_readings' as source,
                    anomaly_type,
                    COUNT(*) as count
                FROM meter_readings
                WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
                AND is_anomaly = true
                GROUP BY anomaly_type
                
                UNION ALL
                
                SELECT 
                    'grid_statuses' as source,
                    anomaly_type,
                    COUNT(*) as count
                FROM grid_statuses
                WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
                AND is_anomaly = true
                GROUP BY anomaly_type
                
                UNION ALL
                
                SELECT 
                    'weather_observations' as source,
                    anomaly_type,
                    COUNT(*) as count
                FROM weather_observations
                WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
                AND is_anomaly = true
                GROUP BY anomaly_type
                
                ORDER BY source, count DESC
            """, {
                "start_date": start_date,
                "end_date": end_date
            })
            
            # Get anomaly trends over time
            anomaly_trends = await self.client.execute_query("""
                SELECT 
                    DATE(timestamp) as date,
                    SUM(CASE WHEN source = 'meter_readings' THEN 1 ELSE 0 END) as meter_anomalies,
                    SUM(CASE WHEN source = 'grid_statuses' THEN 1 ELSE 0 END) as grid_anomalies,
                    SUM(CASE WHEN source = 'weather_observations' THEN 1 ELSE 0 END) as weather_anomalies
                FROM (
                    SELECT timestamp, 'meter_readings' as source FROM meter_readings WHERE is_anomaly = true
                    UNION ALL
                    SELECT timestamp, 'grid_statuses' as source FROM grid_statuses WHERE is_anomaly = true
                    UNION ALL
                    SELECT timestamp, 'weather_observations' as source FROM weather_observations WHERE is_anomaly = true
                )
                WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, {
                "start_date": start_date,
                "end_date": end_date
            })
            
            return {
                "anomaly_counts": anomaly_counts,
                "anomaly_trends": anomaly_trends
            }
            
        except Exception as e:
            logger.error(f"Error getting anomaly analysis: {str(e)}")
            raise InfrastructureError(f"Failed to get anomaly analysis: {str(e)}", service="snowflake")
    
    async def get_performance_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            # Get data volume metrics
            volume_metrics = await self.client.execute_query("""
                SELECT 
                    'meter_readings' as data_type,
                    COUNT(*) as total_records,
                    COUNT(DISTINCT meter_id) as unique_entities,
                    AVG(data_quality_score) as avg_quality_score
                FROM meter_readings
                WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
                
                UNION ALL
                
                SELECT 
                    'grid_statuses' as data_type,
                    COUNT(*) as total_records,
                    COUNT(DISTINCT operator_id) as unique_entities,
                    AVG(data_quality_score) as avg_quality_score
                FROM grid_statuses
                WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
                
                UNION ALL
                
                SELECT 
                    'weather_observations' as data_type,
                    COUNT(*) as total_records,
                    COUNT(DISTINCT station_id) as unique_entities,
                    AVG(data_quality_score) as avg_quality_score
                FROM weather_observations
                WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
            """, {
                "start_date": start_date,
                "end_date": end_date
            })
            
            # Get processing efficiency
            efficiency_metrics = await self.client.execute_query("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as daily_records,
                    AVG(data_quality_score) as avg_quality,
                    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count
                FROM (
                    SELECT timestamp, data_quality_score, is_anomaly FROM meter_readings
                    UNION ALL
                    SELECT timestamp, data_quality_score, is_anomaly FROM grid_statuses
                    UNION ALL
                    SELECT timestamp, data_quality_score, is_anomaly FROM weather_observations
                )
                WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, {
                "start_date": start_date,
                "end_date": end_date
            })
            
            return {
                "volume_metrics": volume_metrics,
                "efficiency_metrics": efficiency_metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            raise InfrastructureError(f"Failed to get performance metrics: {str(e)}", service="snowflake")
    
    async def get_energy_demand_forecast(
        self,
        forecast_days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get energy demand forecast based on historical data and weather patterns"""
        try:
            query = """
            WITH historical_demand AS (
                SELECT 
                    DATE(timestamp) as date,
                    AVG(active_power) as avg_demand,
                    AVG(temperature_celsius) as avg_temperature,
                    AVG(humidity_percent) as avg_humidity
                FROM meter_readings mr
                LEFT JOIN weather_observations wo ON DATE(mr.timestamp) = DATE(wo.timestamp)
                WHERE mr.timestamp >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE(timestamp)
            ),
            weather_forecast AS (
                SELECT 
                    CURRENT_DATE + INTERVAL '1 day' * (ROW_NUMBER() OVER (ORDER BY 1) - 1) as forecast_date,
                    AVG(avg_temperature) as forecast_temp,
                    AVG(avg_humidity) as forecast_humidity
                FROM historical_demand
                CROSS JOIN TABLE(GENERATOR(ROWCOUNT => %(forecast_days)s))
            )
            SELECT 
                wf.forecast_date,
                wf.forecast_temp,
                wf.forecast_humidity,
                -- Simple linear regression for demand forecast
                AVG(hd.avg_demand) + 
                (wf.forecast_temp - AVG(hd.avg_temperature)) * 0.1 +
                (wf.forecast_humidity - AVG(hd.avg_humidity)) * 0.05 as forecast_demand
            FROM weather_forecast wf
            CROSS JOIN historical_demand hd
            GROUP BY wf.forecast_date, wf.forecast_temp, wf.forecast_humidity
            ORDER BY wf.forecast_date
            """
            
            return await self.client.execute_query(query, {"forecast_days": forecast_days})
            
        except Exception as e:
            logger.error(f"Error getting energy demand forecast: {str(e)}")
            raise InfrastructureError(f"Failed to get energy demand forecast: {str(e)}", service="snowflake")
    
    async def get_correlation_analysis(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get correlation analysis between different data sources"""
        try:
            # Weather-energy correlation
            weather_energy_corr = await self.client.execute_query("""
                WITH daily_metrics AS (
                    SELECT 
                        DATE(mr.timestamp) as date,
                        AVG(mr.active_power) as avg_power,
                        AVG(wo.temperature_celsius) as avg_temp,
                        AVG(wo.humidity_percent) as avg_humidity,
                        AVG(wo.pressure_hpa) as avg_pressure
                    FROM meter_readings mr
                    LEFT JOIN weather_observations wo ON DATE(mr.timestamp) = DATE(wo.timestamp)
                    WHERE mr.timestamp BETWEEN %(start_date)s AND %(end_date)s
                    GROUP BY DATE(mr.timestamp)
                )
                SELECT 
                    CORR(avg_power, avg_temp) as temp_correlation,
                    CORR(avg_power, avg_humidity) as humidity_correlation,
                    CORR(avg_power, avg_pressure) as pressure_correlation,
                    CORR(avg_temp, avg_humidity) as temp_humidity_correlation
                FROM daily_metrics
            """, {
                "start_date": start_date,
                "end_date": end_date
            })
            
            # Grid stability correlation
            grid_stability_corr = await self.client.execute_query("""
                WITH daily_grid AS (
                    SELECT 
                        DATE(timestamp) as date,
                        AVG(stability_score) as avg_stability,
                        AVG(load_percentage) as avg_load,
                        AVG(frequency) as avg_frequency
                    FROM grid_statuses
                    WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
                    GROUP BY DATE(timestamp)
                )
                SELECT 
                    CORR(avg_stability, avg_load) as stability_load_correlation,
                    CORR(avg_stability, avg_frequency) as stability_frequency_correlation,
                    CORR(avg_load, avg_frequency) as load_frequency_correlation
                FROM daily_grid
            """, {
                "start_date": start_date,
                "end_date": end_date
            })
            
            return {
                "weather_energy_correlation": weather_energy_corr[0] if weather_energy_corr else {},
                "grid_stability_correlation": grid_stability_corr[0] if grid_stability_corr else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting correlation analysis: {str(e)}")
            raise InfrastructureError(f"Failed to get correlation analysis: {str(e)}", service="snowflake")
