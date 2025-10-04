"""
Configuration Loader
Handles loading and merging of configuration from multiple sources
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConfigLoader:
    """
    Configuration loader that merges YAML files with environment variables
    
    Priority order (highest to lowest):
    1. Environment variables
    2. YAML configuration files
    3. Default values
    """
    
    environment: str = "development"
    config_dir: str = "config"
    yaml_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    env_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize configuration after object creation"""
        self.config_dir = Path(self.config_dir)
        self._load_yaml_config()
        self._load_env_config()
    
    def _load_yaml_config(self) -> None:
        """Load configuration from YAML files"""
        try:
            # Load environment-specific config
            env_config_file = self.config_dir / "environments" / f"{self.environment}.yaml"
            if env_config_file.exists():
                with open(env_config_file, 'r') as f:
                    self.yaml_config = yaml.safe_load(f)
            else:
                # Fallback to development config
                dev_config_file = self.config_dir / "environments" / "development.yaml"
                if dev_config_file.exists():
                    with open(dev_config_file, 'r') as f:
                        self.yaml_config = yaml.safe_load(f)
                else:
                    self.yaml_config = {}
            
            # Load additional config files
            self._load_database_config()
            self._load_external_services_config()
            
        except Exception as e:
            print(f"Warning: Could not load YAML configuration: {e}")
            self.yaml_config = {}
    
    def _load_database_config(self) -> None:
        """Load database-specific configuration"""
        try:
            # Load connection pools config
            pools_file = self.config_dir / "database" / "connection_pools.yaml"
            if pools_file.exists():
                with open(pools_file, 'r') as f:
                    pools_config = yaml.safe_load(f)
                    if 'database' not in self.yaml_config:
                        self.yaml_config['database'] = {}
                    self.yaml_config['database']['connection_pools'] = pools_config
            
            # Load query optimization config
            query_file = self.config_dir / "database" / "query_optimization.yaml"
            if query_file.exists():
                with open(query_file, 'r') as f:
                    query_config = yaml.safe_load(f)
                    if 'database' not in self.yaml_config:
                        self.yaml_config['database'] = {}
                    self.yaml_config['database']['query_optimization'] = query_config
            
        except Exception as e:
            print(f"Warning: Could not load database configuration: {e}")
    
    def _load_external_services_config(self) -> None:
        """Load external services configuration"""
        try:
            # Load Kafka config
            kafka_file = self.config_dir / "external_services" / "kafka.yaml"
            if kafka_file.exists():
                with open(kafka_file, 'r') as f:
                    kafka_config = yaml.safe_load(f)
                    if 'kafka' not in self.yaml_config:
                        self.yaml_config['kafka'] = {}
                    self.yaml_config['kafka'].update(kafka_config)
            
            # Load S3 config
            s3_file = self.config_dir / "external_services" / "s3.yaml"
            if s3_file.exists():
                with open(s3_file, 'r') as f:
                    s3_config = yaml.safe_load(f)
                    if 's3' not in self.yaml_config:
                        self.yaml_config['s3'] = {}
                    self.yaml_config['s3'].update(s3_config)
            
            # Load Data Sources config
            data_sources_file = self.config_dir / "external_services" / "data_sources.yaml"
            if data_sources_file.exists():
                with open(data_sources_file, 'r') as f:
                    data_sources_config = yaml.safe_load(f)
                    if 'data_sources' not in self.yaml_config:
                        self.yaml_config['data_sources'] = {}
                    self.yaml_config['data_sources'].update(data_sources_config)
            
            # Load Snowflake config
            snowflake_file = self.config_dir / "external_services" / "snowflake.yaml"
            if snowflake_file.exists():
                with open(snowflake_file, 'r') as f:
                    snowflake_config = yaml.safe_load(f)
                    if 'snowflake' not in self.yaml_config:
                        self.yaml_config['snowflake'] = {}
                    self.yaml_config['snowflake'].update(snowflake_config)
            
            # Load monitoring config
            monitoring_file = self.config_dir / "external_services" / "monitoring.yaml"
            if monitoring_file.exists():
                with open(monitoring_file, 'r') as f:
                    monitoring_config = yaml.safe_load(f)
                    if 'monitoring' not in self.yaml_config:
                        self.yaml_config['monitoring'] = {}
                    self.yaml_config['monitoring'].update(monitoring_config)
            
        except Exception as e:
            print(f"Warning: Could not load external services configuration: {e}")
    
    def _load_env_config(self) -> None:
        """Load configuration from environment variables"""
        self.env_config = {}
        
        # Load all environment variables that start with METRIFY_
        for key, value in os.environ.items():
            if key.startswith('METRIFY_'):
                # Convert METRIFY_DATABASE_HOST to database.host
                config_key = key[8:].lower().replace('_', '.')
                self.env_config[config_key] = self._parse_env_value(value)
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool, None]:
        """Parse environment variable value to appropriate type"""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        elif value.lower() in ('null', 'none'):
            return None
        elif value.isdigit():
            return int(value)
        elif value.replace('.', '').isdigit():
            return float(value)
        else:
            return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key
        
        Args:
            key: Configuration key (e.g., 'database.host')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        # Check environment variables first
        if key in self.env_config:
            return self.env_config[key]
        
        # Check YAML configuration
        value = self._get_nested_value(self.yaml_config, key)
        if value is not None:
            return value
        
        return default
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """Get nested value from configuration dictionary"""
        keys = key.split('.')
        current = config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            'host': self.get('database.host', 'localhost'),
            'port': self.get('database.port', 5432),
            'name': self.get('database.name', 'metrify'),
            'username': self.get('database.username', 'metrify'),
            'password': self.get('database.password', 'password'),
            'pool_size': self.get('database.pool_size', 5),
            'max_overflow': self.get('database.max_overflow', 10),
            'pool_timeout': self.get('database.pool_timeout', 30),
            'pool_recycle': self.get('database.pool_recycle', 3600),
            'echo': self.get('database.echo', False)
        }
    
    def get_kafka_config(self) -> Dict[str, Any]:
        """Get Kafka configuration"""
        return {
            'bootstrap_servers': self.get('kafka.bootstrap_servers', ['localhost:9092']),
            'security_protocol': self.get('kafka.security_protocol', 'PLAINTEXT'),
            'sasl_mechanism': self.get('kafka.sasl_mechanism'),
            'sasl_username': self.get('kafka.sasl_username'),
            'sasl_password': self.get('kafka.sasl_password'),
            'consumer_group': self.get('kafka.consumer_group', 'metrify'),
            'auto_offset_reset': self.get('kafka.auto_offset_reset', 'latest'),
            'enable_auto_commit': self.get('kafka.enable_auto_commit', True),
            'session_timeout_ms': self.get('kafka.session_timeout_ms', 30000),
            'heartbeat_interval_ms': self.get('kafka.heartbeat_interval_ms', 10000)
        }
    
    def get_s3_config(self) -> Dict[str, Any]:
        """Get S3 configuration"""
        return {
            'region': self.get('s3.region', 'eu-central-1'),
            'bucket_name': self.get('s3.bucket_name', 'metrify-data'),
            'access_key_id': self.get('s3.access_key_id'),
            'secret_access_key': self.get('s3.secret_access_key'),
            'endpoint_url': self.get('s3.endpoint_url'),
            'use_ssl': self.get('s3.use_ssl', True)
        }
    
    def get_snowflake_config(self) -> Dict[str, Any]:
        """Get Snowflake configuration"""
        return {
            'account': self.get('snowflake.account'),
            'warehouse': self.get('snowflake.warehouse'),
            'database': self.get('snowflake.database'),
            'schema': self.get('snowflake.schema', 'PUBLIC'),
            'username': self.get('snowflake.username'),
            'password': self.get('snowflake.password'),
            'role': self.get('snowflake.role'),
            'region': self.get('snowflake.region', 'eu-central-1')
        }
    
    def get_airflow_config(self) -> Dict[str, Any]:
        """Get Airflow configuration"""
        return {
            'webserver_url': self.get('airflow.webserver_url', 'http://localhost:8080'),
            'username': self.get('airflow.username', 'admin'),
            'password': self.get('airflow.password', 'admin'),
            'api_url': self.get('airflow.api_url', 'http://localhost:8080/api/v1'),
            'dag_folder': self.get('airflow.dag_folder', '/opt/airflow/dags')
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return {
            'prometheus': {
                'enabled': self.get('monitoring.prometheus.enabled', True),
                'port': self.get('monitoring.prometheus.port', 9090),
                'namespace': self.get('monitoring.prometheus.namespace', 'metrify'),
                'subsystem': self.get('monitoring.prometheus.subsystem', 'smart_metering')
            },
            'grafana': {
                'enabled': self.get('monitoring.grafana.enabled', True),
                'url': self.get('monitoring.grafana.url', 'http://localhost:3000'),
                'username': self.get('monitoring.grafana.username', 'admin'),
                'password': self.get('monitoring.grafana.password', 'admin'),
                'api_key': self.get('monitoring.grafana.api_key')
            },
            'jaeger': {
                'enabled': self.get('monitoring.jaeger.enabled', True),
                'agent_host': self.get('monitoring.jaeger.agent_host', 'localhost'),
                'agent_port': self.get('monitoring.jaeger.agent_port', 6831),
                'collector_endpoint': self.get('monitoring.jaeger.collector_endpoint')
            },
            'datadog': {
                'enabled': self.get('monitoring.datadog.enabled', False),
                'api_key': self.get('monitoring.datadog.api_key'),
                'app_key': self.get('monitoring.datadog.app_key'),
                'site': self.get('monitoring.datadog.site', 'datadoghq.com')
            }
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            'jwt_secret_key': self.get('security.jwt_secret_key', 'change-me-in-production'),
            'jwt_algorithm': self.get('security.jwt_algorithm', 'HS256'),
            'jwt_expire_minutes': self.get('security.jwt_expire_minutes', 60),
            'refresh_token_expire_days': self.get('security.refresh_token_expire_days', 7),
            'password_min_length': self.get('security.password_min_length', 8),
            'max_login_attempts': self.get('security.max_login_attempts', 5),
            'lockout_duration_minutes': self.get('security.lockout_duration_minutes', 15)
        }
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration"""
        return {
            'name': self.get('app.name', 'Metrify Smart Metering API'),
            'version': self.get('app.version', '1.0.0'),
            'description': self.get('app.description', 'Smart metering data pipeline'),
            'host': self.get('app.host', '0.0.0.0'),
            'port': self.get('app.port', 8000),
            'workers': self.get('app.workers', 1),
            'reload': self.get('app.reload', False),
            'debug': self.get('debug', False),
            'log_level': self.get('log_level', 'INFO')
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config = self.yaml_config.copy()
        
        # Merge environment variables
        for key, value in self.env_config.items():
            self._set_nested_value(config, key, value)
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested value in configuration dictionary"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def save_to_file(self, filepath: str) -> None:
        """Save current configuration to YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = [
            'database.host',
            'database.name',
            'database.username',
            'database.password'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                print(f"Error: Required configuration key '{key}' is missing")
                return False
        
        return True


# Global configuration loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(environment: str = None) -> ConfigLoader:
    """Get global configuration loader instance"""
    global _config_loader
    
    if _config_loader is None:
        env = environment or os.getenv('METRIFY_ENVIRONMENT', 'development')
        _config_loader = ConfigLoader(environment=env)
    
    return _config_loader


def get_database_config() -> Dict[str, Any]:
    """Get database configuration"""
    return get_config_loader().get_database_config()


def get_kafka_config() -> Dict[str, Any]:
    """Get Kafka configuration"""
    return get_config_loader().get_kafka_config()


def get_s3_config() -> Dict[str, Any]:
    """Get S3 configuration"""
    return get_config_loader().get_s3_config()


def get_snowflake_config() -> Dict[str, Any]:
    """Get Snowflake configuration"""
    return get_config_loader().get_snowflake_config()


def get_airflow_config() -> Dict[str, Any]:
    """Get Airflow configuration"""
    return get_config_loader().get_airflow_config()


def get_monitoring_config() -> Dict[str, Any]:
    """Get monitoring configuration"""
    return get_config_loader().get_monitoring_config()


def get_security_config() -> Dict[str, Any]:
    """Get security configuration"""
    return get_config_loader().get_security_config()


def get_app_config() -> Dict[str, Any]:
    """Get application configuration"""
    return get_config_loader().get_app_config()
