"""
Spark ETL Configuration
Configuration management for Apache Spark ETL jobs
"""

import os
from typing import Dict, Any, Optional
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf


class SparkETLConfig:
    """Configuration class for Spark ETL jobs"""
    
    def __init__(self):
        self.app_name = os.getenv("SPARK_APP_NAME", "metrify-spark-etl")
        self.master = os.getenv("SPARK_MASTER", "local[*]")
        self.driver_memory = os.getenv("SPARK_DRIVER_MEMORY", "2g")
        self.executor_memory = os.getenv("SPARK_EXECUTOR_MEMORY", "2g")
        self.executor_cores = os.getenv("SPARK_EXECUTOR_CORES", "2")
        self.max_result_size = os.getenv("SPARK_MAX_RESULT_SIZE", "1g")
        
        # S3 Configuration
        self.s3_endpoint = os.getenv("S3_ENDPOINT_URL", "https://s3.eu-central-1.amazonaws.com")
        self.s3_bucket = os.getenv("S3_BUCKET_NAME", "metrify-data-lake")
        self.s3_region = os.getenv("S3_REGION", "eu-central-1")
        
        # Database Configuration
        self.postgres_host = os.getenv("POSTGRES_HOST", "postgres")
        self.postgres_port = os.getenv("POSTGRES_PORT", "5432")
        self.postgres_db = os.getenv("POSTGRES_DB", "metrify")
        self.postgres_user = os.getenv("POSTGRES_USER", "metrify")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD", "metrify")
        
        # Snowflake Configuration
        self.snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT", "")
        self.snowflake_user = os.getenv("SNOWFLAKE_USER", "")
        self.snowflake_password = os.getenv("SNOWFLAKE_PASSWORD", "")
        self.snowflake_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "METRIFY_WAREHOUSE")
        self.snowflake_database = os.getenv("SNOWFLAKE_DATABASE", "METRIFY_ANALYTICS")
        self.snowflake_schema = os.getenv("SNOWFLAKE_SCHEMA", "RAW")
    
    def get_spark_conf(self) -> SparkConf:
        """Get Spark configuration"""
        conf = SparkConf()
        
        # Basic configuration
        conf.set("spark.app.name", self.app_name)
        conf.set("spark.master", self.master)
        conf.set("spark.driver.memory", self.driver_memory)
        conf.set("spark.executor.memory", self.executor_memory)
        conf.set("spark.executor.cores", self.executor_cores)
        conf.set("spark.driver.maxResultSize", self.max_result_size)
        
        # S3 Configuration
        conf.set("spark.hadoop.fs.s3a.endpoint", self.s3_endpoint)
        conf.set("spark.hadoop.fs.s3a.region", self.s3_region)
        conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        conf.set("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider")
        
        # Performance optimizations
        conf.set("spark.sql.adaptive.enabled", "true")
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        
        # Delta Lake configuration
        conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        conf.set("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        
        return conf
    
    def create_spark_session(self) -> SparkSession:
        """Create and configure Spark session"""
        conf = self.get_spark_conf()
        
        spark = SparkSession.builder \
            .config(conf=conf) \
            .getOrCreate()
        
        # Set log level
        spark.sparkContext.setLogLevel("INFO")
        
        return spark
    
    def get_s3_path(self, data_type: str, partition: str = None) -> str:
        """Get S3 path for data type"""
        base_path = f"s3a://{self.s3_bucket}/processed/{data_type}"
        if partition:
            return f"{base_path}/{partition}"
        return base_path
    
    def get_postgres_url(self) -> str:
        """Get PostgreSQL JDBC URL"""
        return f"jdbc:postgresql://{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    def get_postgres_properties(self) -> Dict[str, str]:
        """Get PostgreSQL connection properties"""
        return {
            "user": self.postgres_user,
            "password": self.postgres_password,
            "driver": "org.postgresql.Driver"
        }
