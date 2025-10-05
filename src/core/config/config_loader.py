"""
Configuration Loader
Handles loading and validation of environment configuration
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    name: str
    user: str
    password: str
    ssl_mode: str
    pool_size: int
    max_overflow: int
    pool_timeout: int
    pool_recycle: int
    pool_pre_ping: bool
    echo: bool


@dataclass
class KafkaConfig:
    """Kafka configuration"""
    bootstrap_servers: str
    security_protocol: str
    sasl_mechanism: str
    sasl_username: str
    sasl_password: str
    ssl_ca_location: str
    producer_acks: str
    producer_retries: int
    producer_batch_size: int
    producer_linger_ms: int
    producer_compression_type: str
    consumer_group_id: str
    consumer_auto_offset_reset: str
    consumer_enable_auto_commit: bool
    consumer_auto_commit_interval_ms: int
    consumer_session_timeout_ms: int
    consumer_heartbeat_interval_ms: int


@dataclass
class S3Config:
    """S3 configuration"""
    access_key_id: str
    secret_access_key: str
    default_region: str
    bucket: str
    prefix: str
    endpoint_url: str
    archive_bucket: str
    archive_prefix: str
    archive_retention_days: int
    archive_compression: bool
    backup_bucket: str


@dataclass
class SnowflakeConfig:
    """Snowflake configuration"""
    account: str
    user: str
    password: str
    warehouse: str
    database: str
    schema: str
    role: str
    connection_timeout: int
    query_timeout: int
    autocommit: bool
    client_session_keep_alive: bool


@dataclass
class AirflowConfig:
    """Airflow configuration"""
    home: str
    dags_folder: str
    logs_folder: str
    plugins_folder: str
    base_url: str
    db_conn_id: str
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    executor: str
    celery_broker_url: str
    celery_result_backend: str
    worker_concurrency: int
    worker_autoscale_min: int
    worker_autoscale_max: int


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    datadog_api_key: str
    datadog_app_key: str
    datadog_site: str
    datadog_service_name: str
    datadog_env: str
    datadog_version: str
    prometheus_port: int
    prometheus_metrics_path: str
    prometheus_enabled: bool
    log_level: str
    log_format: str
    log_file: str
    log_max_size: str
    log_backup_count: int
    log_compression: bool


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str
    jwt_algorithm: str
    jwt_access_token_expire_minutes: int
    jwt_refresh_token_expire_days: int
    api_rate_limit: int
    api_rate_limit_window: int
    api_cors_origins: str
    api_cors_methods: str
    api_cors_headers: str
    encryption_key: str
    encryption_algorithm: str
    encryption_iv_length: int


@dataclass
class DataSourceConfig:
    """Data source configuration"""
    data_root: str
    readings_file: str
    meters_file: str
    file_patterns: Dict[str, str]
    batch_size: int
    processing_interval: int
    kafka_topic: str
    csv_settings: Dict[str, Any]
    validation: Dict[str, Any]

@dataclass
class DataSourcesConfig:
    """All data sources configuration"""
    smart_meters: DataSourceConfig
    grid_operators: DataSourceConfig
    weather_stations: DataSourceConfig
    processing: Dict[str, Any]
    environments: Dict[str, Any]

@dataclass
class AppConfig:
    """Application configuration"""
    name: str
    version: str
    description: str
    environment: str
    debug: bool
    testing: bool
    workers: int
    worker_class: str
    worker_connections: int
    max_requests: int
    max_requests_jitter: int
    timeout: int
    keepalive: int
    memory_limit: str
    memory_warning_threshold: str
    memory_critical_threshold: str
    cpu_limit: int
    cpu_warning_threshold: str
    cpu_critical_threshold: str
    database_url: str
    grafana_url: str
    grafana_username: str
    grafana_password: str
    jaeger_agent_host: str
    jaeger_agent_port: int
    datadog_api_key: str
    datadog_app_key: str
    datadog_site: str


class ConfigLoader:
    """
    Configuration loader and validator
    
    Loads configuration from environment variables and validates them
    """
    
    def __init__(self, env_file: Optional[str] = None):
        self.env_file = env_file
        self.yaml_config = {}
        self._load_env_file()
        self._load_yaml_config()
    
    def _load_env_file(self) -> None:
        """Load environment variables from file"""
        if self.env_file and Path(self.env_file).exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            logger.info(f"Loaded environment from {self.env_file}")
    
    def _load_yaml_config(self) -> None:
        """Load YAML configuration files"""
        try:
            # Load data sources configuration
            data_sources_path = Path("config/external_services/data_sources.yaml")
            if data_sources_path.exists():
                with open(data_sources_path, 'r') as f:
                    self.yaml_config['data_sources'] = yaml.safe_load(f)
                logger.info(f"Loaded data sources config from {data_sources_path}")
            else:
                logger.warning(f"Data sources config not found at {data_sources_path}")
            
            # Load performance configuration
            performance_path = Path("config/external_services/performance.yaml")
            if performance_path.exists():
                with open(performance_path, 'r') as f:
                    self.yaml_config['performance'] = yaml.safe_load(f)
                logger.info(f"Loaded performance config from {performance_path}")
            else:
                logger.warning(f"Performance config not found at {performance_path}")
        except Exception as e:
            logger.error(f"Error loading YAML config: {e}")
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            name=os.getenv('DB_NAME', 'metrify'),
            user=os.getenv('DB_USER', 'metrify_user'),
            password=os.getenv('DB_PASSWORD', ''),
            ssl_mode=os.getenv('DB_SSL_MODE', 'disable'),
            pool_size=int(os.getenv('DB_POOL_SIZE', '5')),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '10')),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30')),
            pool_recycle=int(os.getenv('DB_POOL_RECYCLE', '3600')),
            pool_pre_ping=os.getenv('DB_POOL_PRE_PING', 'true').lower() == 'true',
            echo=os.getenv('DB_ECHO', 'false').lower() == 'true'
        )
    
    def get_kafka_config(self) -> KafkaConfig:
        """Get Kafka configuration"""
        return KafkaConfig(
            bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
            security_protocol=os.getenv('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT'),
            sasl_mechanism=os.getenv('KAFKA_SASL_MECHANISM', 'PLAIN'),
            sasl_username=os.getenv('KAFKA_SASL_USERNAME', ''),
            sasl_password=os.getenv('KAFKA_SASL_PASSWORD', ''),
            ssl_ca_location=os.getenv('KAFKA_SSL_CA_LOCATION', ''),
            producer_acks=os.getenv('KAFKA_PRODUCER_ACKS', '1'),
            producer_retries=int(os.getenv('KAFKA_PRODUCER_RETRIES', '1')),
            producer_batch_size=int(os.getenv('KAFKA_PRODUCER_BATCH_SIZE', '16384')),
            producer_linger_ms=int(os.getenv('KAFKA_PRODUCER_LINGER_MS', '10')),
            producer_compression_type=os.getenv('KAFKA_PRODUCER_COMPRESSION_TYPE', 'none'),
            consumer_group_id=os.getenv('KAFKA_CONSUMER_GROUP_ID', 'metrify-processors'),
            consumer_auto_offset_reset=os.getenv('KAFKA_CONSUMER_AUTO_OFFSET_RESET', 'latest'),
            consumer_enable_auto_commit=os.getenv('KAFKA_CONSUMER_ENABLE_AUTO_COMMIT', 'true').lower() == 'true',
            consumer_auto_commit_interval_ms=int(os.getenv('KAFKA_CONSUMER_AUTO_COMMIT_INTERVAL_MS', '1000')),
            consumer_session_timeout_ms=int(os.getenv('KAFKA_CONSUMER_SESSION_TIMEOUT_MS', '30000')),
            consumer_heartbeat_interval_ms=int(os.getenv('KAFKA_CONSUMER_HEARTBEAT_INTERVAL_MS', '10000'))
        )
    
    def get_s3_config(self) -> S3Config:
        """Get S3 configuration"""
        return S3Config(
            access_key_id=os.getenv('S3_ACCESS_KEY_ID', ''),
            secret_access_key=os.getenv('S3_SECRET_ACCESS_KEY', ''),
            default_region=os.getenv('S3_REGION', 'eu-central-1'),
            bucket=os.getenv('S3_BUCKET_NAME', 'metrify-data-lake'),
            prefix=os.getenv('S3_UPLOAD_PREFIX', 'data/uploads/'),
            endpoint_url=os.getenv('S3_ENDPOINT_URL', 'https://s3.eu-central-1.amazonaws.com'),
            archive_bucket=os.getenv('S3_ARCHIVE_BUCKET', 'metrify-archive'),
            archive_prefix=os.getenv('S3_ARCHIVE_PREFIX', 'data/archive/'),
            archive_retention_days=int(os.getenv('S3_ARCHIVE_RETENTION_DAYS', '2555')),
            archive_compression=os.getenv('S3_ARCHIVE_COMPRESSION', 'true').lower() == 'true',
            backup_bucket=os.getenv('S3_BACKUP_BUCKET', 'metrify-backups')
        )
    
    def get_snowflake_config(self) -> SnowflakeConfig:
        """Get Snowflake configuration"""
        return SnowflakeConfig(
            account=os.getenv('SNOWFLAKE_ACCOUNT', ''),
            user=os.getenv('SNOWFLAKE_USER', ''),
            password=os.getenv('SNOWFLAKE_PASSWORD', ''),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', ''),
            database=os.getenv('SNOWFLAKE_DATABASE', ''),
            schema=os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC'),
            role=os.getenv('SNOWFLAKE_ROLE', ''),
            connection_timeout=int(os.getenv('SNOWFLAKE_CONNECTION_TIMEOUT', '60')),
            query_timeout=int(os.getenv('SNOWFLAKE_QUERY_TIMEOUT', '300')),
            autocommit=os.getenv('SNOWFLAKE_AUTOCOMMIT', 'true').lower() == 'true',
            client_session_keep_alive=os.getenv('SNOWFLAKE_CLIENT_SESSION_KEEP_ALIVE', 'true').lower() == 'true'
        )
    
    def get_data_sources_config(self) -> DataSourcesConfig:
        """Get data sources configuration"""
        # Load from YAML config if available
        if hasattr(self, 'yaml_config') and 'data_sources' in self.yaml_config:
            data_sources = self.yaml_config['data_sources']
            
            return DataSourcesConfig(
                smart_meters=DataSourceConfig(
                    data_root=data_sources['smart_meters'].get('data_root', 'data/raw'),
                    readings_file=data_sources['smart_meters'].get('readings_file', 'smart_meters/smart_meter_readings.csv'),
                    meters_file=data_sources['smart_meters'].get('meters_file', 'smart_meters/smart_meters.csv'),
                    file_patterns=data_sources['smart_meters'].get('file_patterns', {}),
                    batch_size=data_sources['smart_meters'].get('batch_size', 1000),
                    processing_interval=data_sources['smart_meters'].get('processing_interval', 60),
                    kafka_topic=data_sources['smart_meters'].get('kafka_topic', 'smart_meter_data'),
                    csv_settings=data_sources['smart_meters'].get('csv_settings', {}),
                    validation=data_sources['smart_meters'].get('validation', {})
                ),
                grid_operators=DataSourceConfig(
                    data_root=data_sources['grid_operators'].get('data_root', 'data/raw'),
                    readings_file=data_sources['grid_operators'].get('status_file', 'grid_operators/grid_status.csv'),
                    meters_file=data_sources['grid_operators'].get('operators_file', 'grid_operators/grid_operators.csv'),
                    file_patterns=data_sources['grid_operators'].get('file_patterns', {}),
                    batch_size=data_sources['grid_operators'].get('batch_size', 100),
                    processing_interval=data_sources['grid_operators'].get('processing_interval', 60),
                    kafka_topic=data_sources['grid_operators'].get('kafka_topic', 'grid_operator_data'),
                    csv_settings=data_sources['grid_operators'].get('csv_settings', {}),
                    validation=data_sources['grid_operators'].get('validation', {})
                ),
                weather_stations=DataSourceConfig(
                    data_root=data_sources['weather_stations'].get('data_root', 'data/raw'),
                    readings_file=data_sources['weather_stations'].get('observations_file', 'weather_stations/weather_observations.csv'),
                    meters_file=data_sources['weather_stations'].get('stations_file', 'weather_stations/weather_stations.csv'),
                    file_patterns=data_sources['weather_stations'].get('file_patterns', {}),
                    batch_size=data_sources['weather_stations'].get('batch_size', 100),
                    processing_interval=data_sources['weather_stations'].get('processing_interval', 300),
                    kafka_topic=data_sources['weather_stations'].get('kafka_topic', 'weather_data'),
                    csv_settings=data_sources['weather_stations'].get('csv_settings', {}),
                    validation=data_sources['weather_stations'].get('validation', {})
                ),
                processing=data_sources.get('processing', {}),
                environments=data_sources.get('environments', {})
            )
        
        # Fallback to default configuration
        return DataSourcesConfig(
            smart_meters=DataSourceConfig(
                data_root='data/raw',
                readings_file='smart_meters/smart_meter_readings.csv',
                meters_file='smart_meters/smart_meters.csv',
                file_patterns={},
                batch_size=1000,
                processing_interval=60,
                kafka_topic='smart-meter-data',
                csv_settings={},
                validation={}
            ),
            grid_operators=DataSourceConfig(
                data_root='data/raw',
                readings_file='grid_operators/grid_status.csv',
                meters_file='grid_operators/grid_operators.csv',
                file_patterns={},
                batch_size=100,
                processing_interval=60,
                kafka_topic='grid-operator-data',
                csv_settings={},
                validation={}
            ),
            weather_stations=DataSourceConfig(
                data_root='data/raw',
                readings_file='weather_stations/weather_observations.csv',
                meters_file='weather_stations/weather_stations.csv',
                file_patterns={},
                batch_size=100,
                processing_interval=300,
                kafka_topic='weather-data',
                csv_settings={},
                validation={}
            ),
            processing={},
            environments={}
        )
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.yaml_config.get('performance', {})

    def get_airflow_config(self) -> AirflowConfig:
        """Get Airflow configuration"""
        return AirflowConfig(
            home=os.getenv('AIRFLOW_HOME', './airflow'),
            dags_folder=os.getenv('AIRFLOW_DAGS_FOLDER', './airflow/dags'),
            logs_folder=os.getenv('AIRFLOW_LOGS_FOLDER', './airflow/logs'),
            plugins_folder=os.getenv('AIRFLOW_PLUGINS_FOLDER', './airflow/plugins'),
            base_url=os.getenv('AIRFLOW_BASE_URL', 'http://localhost:8080'),
            db_conn_id=os.getenv('AIRFLOW_DB_CONN_ID', 'postgres_default'),
            db_host=os.getenv('AIRFLOW_DB_HOST', 'localhost'),
            db_port=int(os.getenv('AIRFLOW_DB_PORT', '5432')),
            db_name=os.getenv('AIRFLOW_DB_NAME', 'airflow'),
            db_user=os.getenv('AIRFLOW_DB_USER', 'airflow_user'),
            db_password=os.getenv('AIRFLOW_DB_PASSWORD', ''),
            executor=os.getenv('AIRFLOW_EXECUTOR', 'LocalExecutor'),
            celery_broker_url=os.getenv('AIRFLOW_CELERY_BROKER_URL', ''),
            celery_result_backend=os.getenv('AIRFLOW_CELERY_RESULT_BACKEND', ''),
            worker_concurrency=int(os.getenv('AIRFLOW_WORKER_CONCURRENCY', '2')),
            worker_autoscale_min=int(os.getenv('AIRFLOW_WORKER_AUTOSCALE_MIN', '1')),
            worker_autoscale_max=int(os.getenv('AIRFLOW_WORKER_AUTOSCALE_MAX', '4'))
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return MonitoringConfig(
            datadog_api_key=os.getenv('DATADOG_API_KEY', ''),
            datadog_app_key=os.getenv('DATADOG_APP_KEY', ''),
            datadog_site=os.getenv('DATADOG_SITE', 'datadoghq.com'),
            datadog_service_name=os.getenv('DATADOG_SERVICE_NAME', 'metrify-smart-metering'),
            datadog_env=os.getenv('DATADOG_ENV', 'development'),
            datadog_version=os.getenv('DATADOG_VERSION', '1.0.0'),
            prometheus_port=int(os.getenv('PROMETHEUS_PORT', '9090')),
            prometheus_metrics_path=os.getenv('PROMETHEUS_METRICS_PATH', '/metrics'),
            prometheus_enabled=os.getenv('PROMETHEUS_ENABLED', 'true').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_format=os.getenv('LOG_FORMAT', 'text'),
            log_file=os.getenv('LOG_FILE', './logs/application.log'),
            log_max_size=os.getenv('LOG_MAX_SIZE', '10MB'),
            log_backup_count=int(os.getenv('LOG_BACKUP_COUNT', '5')),
            log_compression=os.getenv('LOG_COMPRESSION', 'false').lower() == 'true'
        )
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return SecurityConfig(
            jwt_secret_key=os.getenv('JWT_SECRET_KEY', ''),
            jwt_algorithm=os.getenv('JWT_ALGORITHM', 'HS256'),
            jwt_access_token_expire_minutes=int(os.getenv('JWT_ACCESS_TOKEN_EXPIRE_MINUTES', '30')),
            jwt_refresh_token_expire_days=int(os.getenv('JWT_REFRESH_TOKEN_EXPIRE_DAYS', '7')),
            api_rate_limit=int(os.getenv('API_RATE_LIMIT', '1000')),
            api_rate_limit_window=int(os.getenv('API_RATE_LIMIT_WINDOW', '3600')),
            api_cors_origins=os.getenv('API_CORS_ORIGINS', ''),
            api_cors_methods=os.getenv('API_CORS_METHODS', 'GET,POST,PUT,DELETE,OPTIONS'),
            api_cors_headers=os.getenv('API_CORS_HEADERS', 'Content-Type,Authorization'),
            encryption_key=os.getenv('ENCRYPTION_KEY', ''),
            encryption_algorithm=os.getenv('ENCRYPTION_ALGORITHM', 'AES-256-GCM'),
            encryption_iv_length=int(os.getenv('ENCRYPTION_IV_LENGTH', '12'))
        )
    
    def get_app_config(self) -> AppConfig:
        """Get application configuration"""
        return AppConfig(
            name=os.getenv('APP_NAME', 'Metrify Smart Metering'),
            version=os.getenv('APP_VERSION', '1.0.0'),
            description=os.getenv('APP_DESCRIPTION', 'Smart Meter Data Pipeline'),
            environment=os.getenv('ENVIRONMENT', 'development'),
            debug=os.getenv('DEBUG', 'false').lower() == 'true',
            testing=os.getenv('TESTING', 'false').lower() == 'true',
            workers=int(os.getenv('APP_WORKERS', '1')),
            worker_class=os.getenv('APP_WORKER_CLASS', 'uvicorn.workers.UvicornWorker'),
            worker_connections=int(os.getenv('APP_WORKER_CONNECTIONS', '100')),
            max_requests=int(os.getenv('APP_MAX_REQUESTS', '1000')),
            max_requests_jitter=int(os.getenv('APP_MAX_REQUESTS_JITTER', '100')),
            timeout=int(os.getenv('APP_TIMEOUT', '30')),
            keepalive=int(os.getenv('APP_KEEPALIVE', '2')),
            memory_limit=os.getenv('MEMORY_LIMIT', '1GB'),
            memory_warning_threshold=os.getenv('MEMORY_WARNING_THRESHOLD', '80%'),
            memory_critical_threshold=os.getenv('MEMORY_CRITICAL_THRESHOLD', '90%'),
            cpu_limit=int(os.getenv('CPU_LIMIT', '2')),
            cpu_warning_threshold=os.getenv('CPU_WARNING_THRESHOLD', '80%'),
            cpu_critical_threshold=os.getenv('CPU_CRITICAL_THRESHOLD', '90%'),
            database_url=os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/metrify'),
            grafana_url=os.getenv('GRAFANA_URL', 'http://localhost:3000'),
            grafana_username=os.getenv('GRAFANA_USERNAME', 'admin'),
            grafana_password=os.getenv('GRAFANA_PASSWORD', 'admin'),
            jaeger_agent_host=os.getenv('JAEGER_AGENT_HOST', 'localhost'),
            jaeger_agent_port=int(os.getenv('JAEGER_AGENT_PORT', '6831')),
            datadog_api_key=os.getenv('DATADOG_API_KEY', ''),
            datadog_app_key=os.getenv('DATADOG_APP_KEY', ''),
            datadog_site=os.getenv('DATADOG_SITE', 'datadoghq.com')
        )
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configuration objects"""
        return {
            'database': self.get_database_config(),
            'kafka': self.get_kafka_config(),
            's3': self.get_s3_config(),
            'snowflake': self.get_snowflake_config(),
            'airflow': self.get_airflow_config(),
            'monitoring': self.get_monitoring_config(),
            'security': self.get_security_config(),
            'app': self.get_app_config()
        }
    
    def validate_config(self) -> bool:
        """Validate configuration and return True if valid"""
        try:
            configs = self.get_all_configs()
            
            # Validate required fields
            required_fields = [
                ('database', ['host', 'name', 'user']),
                ('kafka', ['bootstrap_servers']),
                ('s3', ['access_key_id', 'secret_access_key', 'bucket']),
                ('snowflake', ['account', 'user', 'password']),
                ('monitoring', ['datadog_api_key', 'datadog_app_key']),
                ('security', ['jwt_secret_key', 'encryption_key'])
            ]
            
            for config_name, fields in required_fields:
                config = configs[config_name]
                for field in fields:
                    if not getattr(config, field):
                        logger.error(f"Missing required configuration: {config_name}.{field}")
                        return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False


# Global configuration instance
config_loader = ConfigLoader()

# Convenience functions
def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return config_loader.get_database_config()

def get_kafka_config() -> KafkaConfig:
    """Get Kafka configuration"""
    return config_loader.get_kafka_config()

def get_s3_config() -> S3Config:
    """Get S3 configuration"""
    return config_loader.get_s3_config()

def get_snowflake_config() -> SnowflakeConfig:
    """Get Snowflake configuration"""
    return config_loader.get_snowflake_config()

def get_airflow_config() -> AirflowConfig:
    """Get Airflow configuration"""
    return config_loader.get_airflow_config()

def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return config_loader.get_monitoring_config()

def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return config_loader.get_security_config()

def get_app_config() -> AppConfig:
    """Get application configuration"""
    return config_loader.get_app_config()
