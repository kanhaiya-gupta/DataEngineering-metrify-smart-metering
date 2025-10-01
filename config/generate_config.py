#!/usr/bin/env python3
"""
Configuration Template Generator
Generates configuration templates and examples
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
import json


class ConfigGenerator:
    """Generates configuration templates and examples"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
    
    def generate_all(self) -> None:
        """Generate all configuration templates"""
        print("🔧 Generating Metrify Smart Metering Configuration Templates...")
        print("=" * 60)
        
        # Create directories
        self._create_directories()
        
        # Generate environment templates
        self._generate_environment_templates()
        
        # Generate database templates
        self._generate_database_templates()
        
        # Generate external services templates
        self._generate_external_services_templates()
        
        # Generate monitoring templates
        self._generate_monitoring_templates()
        
        # Generate .env templates
        self._generate_env_templates()
        
        # Generate Docker templates
        self._generate_docker_templates()
        
        print("\n✅ Configuration templates generated successfully!")
        print("📝 Next steps:")
        print("  1. Copy and customize the generated templates")
        print("  2. Set your environment variables")
        print("  3. Run: python config/validate_config.py")
    
    def _create_directories(self) -> None:
        """Create configuration directories"""
        directories = [
            self.config_dir,
            self.config_dir / "environments",
            self.config_dir / "database",
            self.config_dir / "external_services",
            self.config_dir / "monitoring",
            self.config_dir / "monitoring" / "grafana",
            self.config_dir / "docker",
            self.config_dir / "kubernetes"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {directory}")
    
    def _generate_environment_templates(self) -> None:
        """Generate environment-specific configuration templates"""
        print("🌍 Generating environment templates...")
        
        # Development template
        dev_template = {
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG",
            "app": {
                "name": "Metrify Smart Metering API",
                "version": "1.0.0",
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "reload": True
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "metrify_dev",
                "username": "metrify_dev",
                "password": "dev_password",
                "pool_size": 5,
                "max_overflow": 10
            },
            "kafka": {
                "bootstrap_servers": ["localhost:9092"],
                "security_protocol": "PLAINTEXT",
                "consumer_group": "metrify-dev"
            },
            "s3": {
                "region": "eu-central-1",
                "bucket_name": "metrify-dev-data",
                "access_key_id": "dev_access_key",
                "secret_access_key": "dev_secret_key"
            },
            "monitoring": {
                "prometheus": {"enabled": True, "port": 9090},
                "grafana": {"enabled": True, "url": "http://localhost:3000"},
                "jaeger": {"enabled": True, "agent_host": "localhost"}
            }
        }
        
        self._save_yaml(dev_template, self.config_dir / "environments" / "development.template.yaml")
        
        # Production template
        prod_template = {
            "environment": "production",
            "debug": False,
            "log_level": "WARNING",
            "app": {
                "name": "Metrify Smart Metering API",
                "version": "1.0.0",
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 8,
                "reload": False
            },
            "database": {
                "host": "${DB_HOST}",
                "port": 5432,
                "name": "metrify_prod",
                "username": "${DB_USERNAME}",
                "password": "${DB_PASSWORD}",
                "pool_size": 50,
                "max_overflow": 100
            },
            "kafka": {
                "bootstrap_servers": ["${KAFKA_BROKER_1}", "${KAFKA_BROKER_2}"],
                "security_protocol": "SASL_SSL",
                "sasl_username": "${KAFKA_USERNAME}",
                "sasl_password": "${KAFKA_PASSWORD}",
                "consumer_group": "metrify-prod"
            },
            "s3": {
                "region": "eu-central-1",
                "bucket_name": "metrify-prod-data",
                "access_key_id": "${AWS_ACCESS_KEY_ID}",
                "secret_access_key": "${AWS_SECRET_ACCESS_KEY}"
            },
            "monitoring": {
                "prometheus": {"enabled": True, "port": 9090},
                "grafana": {"enabled": True, "url": "https://grafana.metrify.com"},
                "jaeger": {"enabled": True, "agent_host": "jaeger-agent.metrify.com"}
            }
        }
        
        self._save_yaml(prod_template, self.config_dir / "environments" / "production.template.yaml")
    
    def _generate_database_templates(self) -> None:
        """Generate database configuration templates"""
        print("🗄️ Generating database templates...")
        
        # Connection pools template
        pools_template = {
            "development": {
                "pool_size": 5,
                "max_overflow": 10,
                "pool_timeout": 30,
                "pool_recycle": 3600,
                "pool_pre_ping": True,
                "echo": True
            },
            "staging": {
                "pool_size": 20,
                "max_overflow": 30,
                "pool_timeout": 30,
                "pool_recycle": 3600,
                "pool_pre_ping": True,
                "echo": False
            },
            "production": {
                "pool_size": 50,
                "max_overflow": 100,
                "pool_timeout": 30,
                "pool_recycle": 3600,
                "pool_pre_ping": True,
                "echo": False
            }
        }
        
        self._save_yaml(pools_template, self.config_dir / "database" / "connection_pools.template.yaml")
        
        # Query optimization template
        query_template = {
            "timeouts": {
                "default": 30,
                "analytics": 300,
                "batch_operations": 600,
                "migration": 3600
            },
            "indexes": {
                "smart_meter_readings": [
                    {"column": "meter_id", "type": "btree"},
                    {"column": "timestamp", "type": "btree"},
                    {"column": "meter_id, timestamp", "type": "btree", "name": "idx_meter_timestamp"}
                ]
            },
            "partitioning": {
                "smart_meter_readings": {
                    "strategy": "range",
                    "column": "timestamp",
                    "interval": "monthly",
                    "retention_months": 12
                }
            }
        }
        
        self._save_yaml(query_template, self.config_dir / "database" / "query_optimization.template.yaml")
    
    def _generate_external_services_templates(self) -> None:
        """Generate external services configuration templates"""
        print("🔌 Generating external services templates...")
        
        # Kafka template
        kafka_template = {
            "topics": {
                "smart_meter_data": {
                    "name": "smart-meter-data",
                    "partitions": 12,
                    "replication_factor": 3,
                    "retention_hours": 168
                }
            },
            "producer": {
                "bootstrap_servers": ["kafka-1:9092", "kafka-2:9092"],
                "client_id": "metrify-producer",
                "acks": "all",
                "retries": 10,
                "batch_size": 16384
            },
            "consumer": {
                "bootstrap_servers": ["kafka-1:9092", "kafka-2:9092"],
                "group_id": "metrify-consumer",
                "auto_offset_reset": "latest",
                "enable_auto_commit": True
            }
        }
        
        self._save_yaml(kafka_template, self.config_dir / "external_services" / "kafka.template.yaml")
        
        # S3 template
        s3_template = {
            "buckets": {
                "primary": {
                    "name": "metrify-data",
                    "region": "eu-central-1",
                    "versioning": True,
                    "encryption": "AES256"
                }
            },
            "data_organization": {
                "smart_meter": {
                    "prefix": "smart-meter-data/",
                    "structure": [
                        "year={year}/",
                        "month={month}/",
                        "day={day}/",
                        "meter_id={meter_id}/",
                        "data.parquet"
                    ]
                }
            }
        }
        
        self._save_yaml(s3_template, self.config_dir / "external_services" / "s3.template.yaml")
    
    def _generate_monitoring_templates(self) -> None:
        """Generate monitoring configuration templates"""
        print("📊 Generating monitoring templates...")
        
        # Prometheus template
        prometheus_template = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": "metrify-api",
                    "static_configs": [{"targets": ["api:8000"]}],
                    "metrics_path": "/metrics",
                    "scrape_interval": "15s"
                }
            ]
        }
        
        self._save_yaml(prometheus_template, self.config_dir / "monitoring" / "prometheus.template.yml")
        
        # Grafana datasources template
        grafana_template = {
            "datasources": [
                {
                    "name": "Prometheus",
                    "type": "prometheus",
                    "url": "http://prometheus:9090",
                    "access": "proxy",
                    "is_default": True
                },
                {
                    "name": "PostgreSQL",
                    "type": "postgres",
                    "url": "postgres:5432",
                    "database": "metrify_prod",
                    "user": "grafana",
                    "password": "${GRAFANA_DB_PASSWORD}"
                }
            ]
        }
        
        self._save_yaml(grafana_template, self.config_dir / "monitoring" / "grafana" / "datasources.template.yml")
    
    def _generate_env_templates(self) -> None:
        """Generate .env file templates"""
        print("🌍 Generating .env templates...")
        
        # Development .env template
        dev_env_template = """# Development Environment Variables
# Metrify Smart Metering Data Pipeline

# Environment
METRIFY_ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=metrify_dev
DB_USERNAME=metrify_dev
DB_PASSWORD=dev_password

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_SECURITY_PROTOCOL=PLAINTEXT
KAFKA_CONSUMER_GROUP=metrify-dev

# S3
AWS_ACCESS_KEY_ID=dev_access_key
AWS_SECRET_ACCESS_KEY=dev_secret_key
S3_REGION=eu-central-1
S3_BUCKET_NAME=metrify-dev-data

# Snowflake
SNOWFLAKE_ACCOUNT=metrify-dev
SNOWFLAKE_WAREHOUSE=DEV_WH
SNOWFLAKE_DATABASE=METRIFY_DEV
SNOWFLAKE_USERNAME=dev_user
SNOWFLAKE_PASSWORD=dev_password

# Security
JWT_SECRET_KEY=dev_secret_key_change_in_production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
JAEGER_ENABLED=true
DATADOG_ENABLED=false
"""
        
        self._save_text(dev_env_template, "development.env.template")
        
        # Production .env template
        prod_env_template = """# Production Environment Variables
# Metrify Smart Metering Data Pipeline

# Environment
METRIFY_ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Database
DB_HOST=${DB_HOST}
DB_PORT=5432
DB_NAME=metrify_prod
DB_USERNAME=${DB_USERNAME}
DB_PASSWORD=${DB_PASSWORD}

# Kafka
KAFKA_BOOTSTRAP_SERVERS=${KAFKA_BROKER_1},${KAFKA_BROKER_2}
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_USERNAME=${KAFKA_USERNAME}
KAFKA_PASSWORD=${KAFKA_PASSWORD}
KAFKA_CONSUMER_GROUP=metrify-prod

# S3
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
S3_REGION=eu-central-1
S3_BUCKET_NAME=metrify-prod-data

# Snowflake
SNOWFLAKE_ACCOUNT=${SNOWFLAKE_ACCOUNT}
SNOWFLAKE_WAREHOUSE=PROD_WH
SNOWFLAKE_DATABASE=METRIFY_PROD
SNOWFLAKE_USERNAME=${SNOWFLAKE_USERNAME}
SNOWFLAKE_PASSWORD=${SNOWFLAKE_PASSWORD}

# Security
JWT_SECRET_KEY=${JWT_SECRET_KEY}
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=15

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
JAEGER_ENABLED=true
DATADOG_ENABLED=true
DATADOG_API_KEY=${DATADOG_API_KEY}
DATADOG_APP_KEY=${DATADOG_APP_KEY}
"""
        
        self._save_text(prod_env_template, "production.env.template")
    
    def _generate_docker_templates(self) -> None:
        """Generate Docker configuration templates"""
        print("🐳 Generating Docker templates...")
        
        # Docker Compose template
        docker_compose_template = """version: '3.8'

services:
  # API Service
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - METRIFY_ENVIRONMENT=development
    depends_on:
      - postgres
      - kafka
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs

  # Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: metrify_dev
      POSTGRES_USER: metrify_dev
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/database/migrations:/docker-entrypoint-initdb.d

  # Kafka
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"
    depends_on:
      - zookeeper

  # Zookeeper
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/monitoring/grafana:/etc/grafana/provisioning

volumes:
  postgres_data:
  prometheus_data:
  grafana_data:
"""
        
        self._save_text(docker_compose_template, self.config_dir / "docker" / "docker-compose.template.yml")
    
    def _save_yaml(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save data as YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
        print(f"  ✅ Generated: {filepath}")
    
    def _save_text(self, content: str, filepath: str) -> None:
        """Save text content to file"""
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✅ Generated: {filepath}")


def main():
    """Main generation function"""
    config_dir = sys.argv[1] if len(sys.argv) > 1 else "config"
    
    generator = ConfigGenerator(config_dir)
    generator.generate_all()


if __name__ == "__main__":
    main()
