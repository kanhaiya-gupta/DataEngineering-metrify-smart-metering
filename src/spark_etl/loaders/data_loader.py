"""
Data Loader for Spark ETL
Integrates dedicated clients with Spark DataFrame operations for proper data loading
"""

import logging
from typing import Dict, Any, Optional, List
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import lit, current_timestamp

from ...infrastructure.external.s3.s3_client import S3Client
from ...infrastructure.external.snowflake.snowflake_client import SnowflakeClient
from ...core.config.config_loader import get_s3_config, get_snowflake_config
from ...core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data Loader that integrates dedicated clients with Spark DataFrame operations
    
    Provides proper connection management, error handling, and monitoring
    while using Spark connectors for efficient data loading
    """
    
    def __init__(self, spark: SparkSession, environment: str = "development"):
        self.spark = spark
        self.environment = environment
        
        # Initialize clients for configuration and monitoring
        self.s3_config = get_s3_config()
        self.snowflake_config = get_snowflake_config()
        
        # Initialize clients (for connection validation and monitoring)
        self.s3_client = S3Client(self.s3_config)
        self.snowflake_client = SnowflakeClient(self.snowflake_config)
    
    async def validate_connections(self) -> Dict[str, bool]:
        """Validate all external service connections"""
        try:
            results = {}
            
            # Validate S3 connection
            try:
                await self.s3_client.connect()
                results["s3"] = True
                logger.info("✅ S3 connection validated")
            except Exception as e:
                results["s3"] = False
                logger.error(f"❌ S3 connection failed: {str(e)}")
            
            # Validate Snowflake connection
            try:
                await self.snowflake_client.connect()
                results["snowflake"] = True
                logger.info("✅ Snowflake connection validated")
            except Exception as e:
                results["snowflake"] = False
                logger.error(f"❌ Snowflake connection failed: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Connection validation failed: {str(e)}")
            raise InfrastructureError(f"Connection validation failed: {str(e)}")
    
    def load_to_s3(
        self, 
        df: DataFrame, 
        target_path: str, 
        format: str = "delta",
        partition_columns: Optional[List[str]] = None,
        write_mode: str = "overwrite"
    ) -> Dict[str, Any]:
        """
        Load DataFrame to S3 using Spark connectors with proper configuration
        
        Args:
            df: Spark DataFrame to load
            target_path: S3 target path (s3://bucket/path)
            format: Output format (delta, parquet, json)
            partition_columns: Columns to partition by
            write_mode: Write mode (overwrite, append, ignore)
        
        Returns:
            Loading result with metadata
        """
        try:
            logger.info(f"Loading data to S3: {target_path}")
            
            # Validate S3 path format
            if not target_path.startswith("s3://"):
                raise ValueError(f"Invalid S3 path format: {target_path}")
            
            # Get record count
            record_count = df.count()
            
            # Configure Spark for S3 access
            spark_conf = {
                "spark.hadoop.fs.s3a.access.key": self.s3_config.access_key_id,
                "spark.hadoop.fs.s3a.secret.key": self.s3_config.secret_access_key,
                "spark.hadoop.fs.s3a.endpoint": self.s3_config.endpoint_url,
                "spark.hadoop.fs.s3a.path.style.access": "true",
                "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem"
            }
            
            # Apply S3 configuration
            for key, value in spark_conf.items():
                self.spark.conf.set(key, value)
            
            # Prepare write operation
            writer = df.write.format(format.lower())
            
            # Set write mode
            writer = writer.mode(write_mode)
            
            # Add format-specific options
            if format.lower() == "delta":
                writer = writer.option("mergeSchema", "true")
            elif format.lower() == "parquet":
                writer = writer.option("compression", "snappy")
            elif format.lower() == "json":
                writer = writer.option("compression", "gzip")
            
            # Add partitioning if specified
            if partition_columns:
                writer = writer.partitionBy(*partition_columns)
            
            # Execute write operation
            writer.save(target_path)
            
            result = {
                "status": "success",
                "records_loaded": record_count,
                "target_path": target_path,
                "format": format,
                "write_mode": write_mode,
                "partition_columns": partition_columns or [],
                "s3_bucket": target_path.split("/")[2],
                "s3_key": "/".join(target_path.split("/")[3:])
            }
            
            logger.info(f"✅ Successfully loaded {record_count} records to S3: {target_path}")
            return result
            
        except Exception as e:
            logger.error(f"❌ S3 loading failed: {str(e)}")
            raise InfrastructureError(f"S3 loading failed: {str(e)}", service="s3")
    
    def load_to_snowflake(
        self, 
        df: DataFrame, 
        table_name: str,
        schema: str = "PROCESSED",
        write_mode: str = "overwrite"
    ) -> Dict[str, Any]:
        """
        Load DataFrame to Snowflake using Spark-Snowflake connector with proper configuration
        
        Args:
            df: Spark DataFrame to load
            table_name: Target table name
            schema: Target schema
            write_mode: Write mode (overwrite, append, ignore)
        
        Returns:
            Loading result with metadata
        """
        try:
            logger.info(f"Loading data to Snowflake: {schema}.{table_name}")
            
            # Get record count
            record_count = df.count()
            
            # Configure Spark for Snowflake access
            snowflake_options = {
                "sfURL": f"{self.snowflake_config.account}.snowflakecomputing.com",
                "sfUser": self.snowflake_config.user,
                "sfPassword": self.snowflake_config.password,
                "sfDatabase": self.snowflake_config.database,
                "sfSchema": schema,
                "sfWarehouse": self.snowflake_config.warehouse,
                "sfRole": self.snowflake_config.role,
                "dbtable": table_name
            }
            
            # Execute write operation
            df.write \
                .format("net.snowflake.spark.snowflake") \
                .options(**snowflake_options) \
                .mode(write_mode) \
                .save()
            
            result = {
                "status": "success",
                "records_loaded": record_count,
                "table_name": table_name,
                "schema": schema,
                "write_mode": write_mode,
                "snowflake_account": self.snowflake_config.account,
                "snowflake_database": self.snowflake_config.database,
                "snowflake_warehouse": self.snowflake_config.warehouse
            }
            
            logger.info(f"✅ Successfully loaded {record_count} records to Snowflake: {schema}.{table_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Snowflake loading failed: {str(e)}")
            raise InfrastructureError(f"Snowflake loading failed: {str(e)}", service="snowflake")
    
    def load_to_postgresql(
        self, 
        df: DataFrame, 
        table_name: str,
        connection_string: str,
        write_mode: str = "append",
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Load DataFrame to PostgreSQL using JDBC connector with proper configuration
        
        Args:
            df: Spark DataFrame to load
            table_name: Target table name
            connection_string: PostgreSQL connection string
            write_mode: Write mode (overwrite, append, ignore)
            batch_size: Batch size for JDBC operations
        
        Returns:
            Loading result with metadata
        """
        try:
            logger.info(f"Loading data to PostgreSQL: {table_name}")
            
            # Get record count
            record_count = df.count()
            
            # Configure JDBC options
            jdbc_options = {
                "url": f"jdbc:postgresql://{connection_string}",
                "dbtable": table_name,
                "user": self.spark.conf.get("spark.sql.execution.arrow.pyspark.enabled", "false"),
                "password": "placeholder",  # Should be configured via Spark conf
                "batchsize": str(batch_size),
                "isolationLevel": "NONE"
            }
            
            # Execute write operation
            df.write \
                .format("jdbc") \
                .options(**jdbc_options) \
                .mode(write_mode) \
                .save()
            
            result = {
                "status": "success",
                "records_loaded": record_count,
                "table_name": table_name,
                "connection_string": connection_string,
                "write_mode": write_mode,
                "batch_size": batch_size
            }
            
            logger.info(f"✅ Successfully loaded {record_count} records to PostgreSQL: {table_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL loading failed: {str(e)}")
            raise InfrastructureError(f"PostgreSQL loading failed: {str(e)}", service="postgresql")
    
    def load_to_multiple_destinations(
        self,
        df: DataFrame,
        destinations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Load DataFrame to multiple destinations with proper error handling
        
        Args:
            df: Spark DataFrame to load
            destinations: Dictionary of destination configurations
        
        Returns:
            Combined loading results
        """
        try:
            results = {}
            total_records = df.count()
            
            for destination, config in destinations.items():
                try:
                    if destination == "s3":
                        result = self.load_to_s3(
                            df=df,
                            target_path=config["target_path"],
                            format=config.get("format", "delta"),
                            partition_columns=config.get("partition_columns"),
                            write_mode=config.get("write_mode", "overwrite")
                        )
                    elif destination == "snowflake":
                        result = self.load_to_snowflake(
                            df=df,
                            table_name=config["table_name"],
                            schema=config.get("schema", "PROCESSED"),
                            write_mode=config.get("write_mode", "overwrite")
                        )
                    elif destination == "postgresql":
                        result = self.load_to_postgresql(
                            df=df,
                            table_name=config["table_name"],
                            connection_string=config["connection_string"],
                            write_mode=config.get("write_mode", "append"),
                            batch_size=config.get("batch_size", 1000)
                        )
                    else:
                        raise ValueError(f"Unsupported destination: {destination}")
                    
                    results[destination] = result
                    
                except Exception as e:
                    logger.error(f"Failed to load to {destination}: {str(e)}")
                    results[destination] = {
                        "status": "failed",
                        "error": str(e),
                        "destination": destination
                    }
            
            # Summary
            successful_destinations = [d for d, r in results.items() if r.get("status") == "success"]
            failed_destinations = [d for d, r in results.items() if r.get("status") == "failed"]
            
            return {
                "status": "success" if not failed_destinations else "partial_success",
                "total_records": total_records,
                "successful_destinations": successful_destinations,
                "failed_destinations": failed_destinations,
                "destinations": results
            }
            
        except Exception as e:
            logger.error(f"Multi-destination loading failed: {str(e)}")
            raise InfrastructureError(f"Multi-destination loading failed: {str(e)}")
