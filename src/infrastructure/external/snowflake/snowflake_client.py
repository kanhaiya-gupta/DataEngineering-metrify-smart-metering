"""
Snowflake Client Implementation
Handles data warehouse operations and analytics queries
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

try:
    import snowflake.connector
    from snowflake.connector import DictCursor
except ImportError:
    snowflake = None
    DictCursor = None

from ....core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class SnowflakeClient:
    """
    Snowflake Client for data warehouse operations
    
    Handles connection management, query execution, and data analytics
    """
    
    def __init__(
        self,
        account: str,
        user: str,
        password: str,
        warehouse: str,
        database: str,
        schema: str = "PUBLIC",
        role: Optional[str] = None
    ):
        if snowflake is None:
            raise InfrastructureError("Snowflake connector not installed", service="snowflake")
        
        self.account = account
        self.user = user
        self.password = password
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.role = role
        
        self._connection = None
        self._is_connected = False
    
    async def connect(self) -> None:
        """Connect to Snowflake"""
        try:
            self._connection = snowflake.connector.connect(
                account=self.account,
                user=self.user,
                password=self.password,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema,
                role=self.role
            )
            self._is_connected = True
            logger.info(f"Connected to Snowflake: {self.account}.{self.database}.{self.schema}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            raise InfrastructureError(f"Failed to connect to Snowflake: {str(e)}", service="snowflake")
    
    async def disconnect(self) -> None:
        """Disconnect from Snowflake"""
        if self._connection:
            self._connection.close()
            self._connection = None
            self._is_connected = False
            logger.info("Disconnected from Snowflake")
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a SQL query
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Query results as list of dictionaries
        """
        if not self._is_connected or not self._connection:
            await self.connect()
        
        try:
            cursor = self._connection.cursor(DictCursor)
            cursor.execute(query, params or {})
            results = cursor.fetchall()
            cursor.close()
            
            logger.debug(f"Executed query, returned {len(results)} rows")
            return results
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise InfrastructureError(f"Failed to execute query: {str(e)}", service="snowflake")
    
    async def execute_ddl(self, ddl_statement: str) -> None:
        """
        Execute a DDL statement (CREATE, ALTER, DROP)
        
        Args:
            ddl_statement: DDL statement to execute
        """
        if not self._is_connected or not self._connection:
            await self.connect()
        
        try:
            cursor = self._connection.cursor()
            cursor.execute(ddl_statement)
            cursor.close()
            
            logger.info(f"Executed DDL: {ddl_statement[:100]}...")
            
        except Exception as e:
            logger.error(f"Error executing DDL: {str(e)}")
            raise InfrastructureError(f"Failed to execute DDL: {str(e)}", service="snowflake")
    
    async def create_table_from_s3(
        self,
        table_name: str,
        s3_path: str,
        file_format: str = "JSON",
        columns: Optional[List[Dict[str, str]]] = None
    ) -> None:
        """
        Create a table from S3 data
        
        Args:
            table_name: Name of the table to create
            s3_path: S3 path to the data files
            file_format: File format (JSON, CSV, PARQUET)
            columns: Column definitions
        """
        try:
            if columns:
                column_defs = ", ".join([f"{col['name']} {col['type']}" for col in columns])
                create_sql = f"""
                CREATE OR REPLACE TABLE {table_name} (
                    {column_defs}
                )
                """
            else:
                create_sql = f"CREATE OR REPLACE TABLE {table_name} (data VARIANT)"
            
            await self.execute_ddl(create_sql)
            
            # Create stage for S3 data
            stage_name = f"{table_name}_stage"
            create_stage_sql = f"""
            CREATE OR REPLACE STAGE {stage_name}
            URL = '{s3_path}'
            FILE_FORMAT = {file_format}
            """
            await self.execute_ddl(create_stage_sql)
            
            # Copy data from stage to table
            copy_sql = f"""
            COPY INTO {table_name}
            FROM @{stage_name}
            FILE_FORMAT = {file_format}
            """
            await self.execute_ddl(copy_sql)
            
            logger.info(f"Created table {table_name} from S3 data")
            
        except Exception as e:
            logger.error(f"Error creating table from S3: {str(e)}")
            raise InfrastructureError(f"Failed to create table from S3: {str(e)}", service="snowflake")
    
    async def get_meter_analytics(
        self,
        start_date: datetime,
        end_date: datetime,
        meter_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get smart meter analytics data"""
        query = """
        SELECT 
            meter_id,
            DATE(timestamp) as reading_date,
            COUNT(*) as reading_count,
            AVG(voltage) as avg_voltage,
            AVG(current) as avg_current,
            AVG(active_power) as avg_active_power,
            AVG(data_quality_score) as avg_quality_score,
            SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count
        FROM meter_readings
        WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
        """
        
        if meter_ids:
            query += " AND meter_id IN %(meter_ids)s"
        
        query += """
        GROUP BY meter_id, DATE(timestamp)
        ORDER BY meter_id, reading_date
        """
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'meter_ids': tuple(meter_ids) if meter_ids else None
        }
        
        return await self.execute_query(query, params)
    
    async def get_grid_analytics(
        self,
        start_date: datetime,
        end_date: datetime,
        operator_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get grid operator analytics data"""
        query = """
        SELECT 
            operator_id,
            DATE(timestamp) as status_date,
            COUNT(*) as status_count,
            AVG(voltage_level) as avg_voltage_level,
            AVG(frequency) as avg_frequency,
            AVG(load_percentage) as avg_load_percentage,
            AVG(stability_score) as avg_stability_score,
            AVG(power_quality_score) as avg_power_quality_score,
            SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count
        FROM grid_statuses
        WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
        """
        
        if operator_ids:
            query += " AND operator_id IN %(operator_ids)s"
        
        query += """
        GROUP BY operator_id, DATE(timestamp)
        ORDER BY operator_id, status_date
        """
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'operator_ids': tuple(operator_ids) if operator_ids else None
        }
        
        return await self.execute_query(query, params)
    
    async def get_weather_analytics(
        self,
        start_date: datetime,
        end_date: datetime,
        station_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get weather station analytics data"""
        query = """
        SELECT 
            station_id,
            DATE(timestamp) as observation_date,
            COUNT(*) as observation_count,
            AVG(temperature_celsius) as avg_temperature,
            AVG(humidity_percent) as avg_humidity,
            AVG(pressure_hpa) as avg_pressure,
            AVG(wind_speed_ms) as avg_wind_speed,
            AVG(data_quality_score) as avg_quality_score,
            SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count
        FROM weather_observations
        WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
        """
        
        if station_ids:
            query += " AND station_id IN %(station_ids)s"
        
        query += """
        GROUP BY station_id, DATE(timestamp)
        ORDER BY station_id, observation_date
        """
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'station_ids': tuple(station_ids) if station_ids else None
        }
        
        return await self.execute_query(query, params)
    
    async def get_energy_demand_correlation(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get energy demand correlation with weather data"""
        query = """
        WITH weather_daily AS (
            SELECT 
                DATE(timestamp) as observation_date,
                AVG(temperature_celsius) as avg_temperature,
                AVG(humidity_percent) as avg_humidity,
                AVG(pressure_hpa) as avg_pressure
            FROM weather_observations
            WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
            GROUP BY DATE(timestamp)
        ),
        energy_daily AS (
            SELECT 
                DATE(timestamp) as reading_date,
                AVG(active_power) as avg_power_consumption
            FROM meter_readings
            WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
            GROUP BY DATE(timestamp)
        )
        SELECT 
            w.observation_date,
            w.avg_temperature,
            w.avg_humidity,
            w.avg_pressure,
            e.avg_power_consumption,
            CORR(w.avg_temperature, e.avg_power_consumption) as temp_correlation,
            CORR(w.avg_humidity, e.avg_power_consumption) as humidity_correlation
        FROM weather_daily w
        JOIN energy_daily e ON w.observation_date = e.reading_date
        ORDER BY w.observation_date
        """
        
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        return await self.execute_query(query, params)
    
    def is_connected(self) -> bool:
        """Check if client is connected to Snowflake"""
        return self._is_connected and self._connection is not None
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get Snowflake client metrics"""
        if not self._connection:
            return {"connected": False}
        
        try:
            # Get warehouse status
            warehouse_query = f"SHOW WAREHOUSES LIKE '{self.warehouse}'"
            warehouse_info = await self.execute_query(warehouse_query)
            
            return {
                "connected": self._is_connected,
                "account": self.account,
                "database": self.database,
                "schema": self.schema,
                "warehouse": self.warehouse,
                "warehouse_info": warehouse_info[0] if warehouse_info else {}
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return {"connected": self._is_connected, "error": str(e)}
