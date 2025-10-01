#!/usr/bin/env python3
"""
Configuration Validation Script
Validates configuration files and environment variables
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
import json


class ConfigValidator:
    """Validates configuration files and environment variables"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> bool:
        """Validate all configuration files"""
        print("🔍 Validating Metrify Smart Metering Configuration...")
        print("=" * 60)
        
        # Validate environment files
        self._validate_environment_files()
        
        # Validate database configuration
        self._validate_database_config()
        
        # Validate external services configuration
        self._validate_external_services_config()
        
        # Validate monitoring configuration
        self._validate_monitoring_config()
        
        # Validate environment variables
        self._validate_environment_variables()
        
        # Print results
        self._print_results()
        
        return len(self.errors) == 0
    
    def _validate_environment_files(self) -> None:
        """Validate environment-specific YAML files"""
        print("📁 Validating environment files...")
        
        environments = ["development", "staging", "production"]
        
        for env in environments:
            env_file = self.config_dir / "environments" / f"{env}.yaml"
            
            if not env_file.exists():
                self.errors.append(f"Missing environment file: {env_file}")
                continue
            
            try:
                with open(env_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Validate required sections
                required_sections = [
                    'app', 'database', 'kafka', 's3', 'snowflake', 
                    'airflow', 'monitoring', 'security'
                ]
                
                for section in required_sections:
                    if section not in config:
                        self.errors.append(f"Missing section '{section}' in {env_file}")
                    else:
                        self._validate_section(f"{env}.{section}", config[section], env_file)
                
                print(f"  ✅ {env}.yaml - Valid")
                
            except yaml.YAMLError as e:
                self.errors.append(f"Invalid YAML in {env_file}: {e}")
            except Exception as e:
                self.errors.append(f"Error reading {env_file}: {e}")
    
    def _validate_database_config(self) -> None:
        """Validate database configuration files"""
        print("🗄️ Validating database configuration...")
        
        db_config_dir = self.config_dir / "database"
        
        if not db_config_dir.exists():
            self.warnings.append("Database configuration directory not found")
            return
        
        # Validate connection pools
        pools_file = db_config_dir / "connection_pools.yaml"
        if pools_file.exists():
            try:
                with open(pools_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                required_environments = ["development", "staging", "production"]
                for env in required_environments:
                    if env not in config:
                        self.warnings.append(f"Missing {env} pool configuration in {pools_file}")
                    else:
                        self._validate_pool_config(env, config[env], pools_file)
                
                print("  ✅ connection_pools.yaml - Valid")
                
            except Exception as e:
                self.errors.append(f"Error validating {pools_file}: {e}")
        
        # Validate query optimization
        query_file = db_config_dir / "query_optimization.yaml"
        if query_file.exists():
            try:
                with open(query_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                required_sections = ["timeouts", "indexes", "partitioning"]
                for section in required_sections:
                    if section not in config:
                        self.warnings.append(f"Missing section '{section}' in {query_file}")
                
                print("  ✅ query_optimization.yaml - Valid")
                
            except Exception as e:
                self.errors.append(f"Error validating {query_file}: {e}")
    
    def _validate_external_services_config(self) -> None:
        """Validate external services configuration"""
        print("🔌 Validating external services configuration...")
        
        services_dir = self.config_dir / "external_services"
        
        if not services_dir.exists():
            self.warnings.append("External services configuration directory not found")
            return
        
        services = ["kafka", "s3", "snowflake", "monitoring"]
        
        for service in services:
            service_file = services_dir / f"{service}.yaml"
            
            if not service_file.exists():
                self.warnings.append(f"Missing {service} configuration file")
                continue
            
            try:
                with open(service_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if service == "kafka":
                    self._validate_kafka_config(config, service_file)
                elif service == "s3":
                    self._validate_s3_config(config, service_file)
                elif service == "snowflake":
                    self._validate_snowflake_config(config, service_file)
                elif service == "monitoring":
                    self._validate_monitoring_config_file(config, service_file)
                
                print(f"  ✅ {service}.yaml - Valid")
                
            except Exception as e:
                self.errors.append(f"Error validating {service_file}: {e}")
    
    def _validate_monitoring_config(self) -> None:
        """Validate monitoring configuration"""
        print("📊 Validating monitoring configuration...")
        
        monitoring_dir = self.config_dir / "monitoring"
        
        if not monitoring_dir.exists():
            self.warnings.append("Monitoring configuration directory not found")
            return
        
        # Validate Prometheus config
        prometheus_file = monitoring_dir / "prometheus.yml"
        if prometheus_file.exists():
            try:
                with open(prometheus_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                required_sections = ["global", "scrape_configs"]
                for section in required_sections:
                    if section not in config:
                        self.errors.append(f"Missing section '{section}' in {prometheus_file}")
                
                print("  ✅ prometheus.yml - Valid")
                
            except Exception as e:
                self.errors.append(f"Error validating {prometheus_file}: {e}")
        
        # Validate Grafana config
        grafana_dir = monitoring_dir / "grafana"
        if grafana_dir.exists():
            datasources_file = grafana_dir / "datasources.yml"
            if datasources_file.exists():
                try:
                    with open(datasources_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    if "datasources" not in config:
                        self.errors.append(f"Missing 'datasources' section in {datasources_file}")
                    
                    print("  ✅ grafana/datasources.yml - Valid")
                    
                except Exception as e:
                    self.errors.append(f"Error validating {datasources_file}: {e}")
        
        # Validate Jaeger config
        jaeger_file = monitoring_dir / "jaeger.yml"
        if jaeger_file.exists():
            try:
                with open(jaeger_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                required_sections = ["service_name", "sampler", "reporter"]
                for section in required_sections:
                    if section not in config:
                        self.errors.append(f"Missing section '{section}' in {jaeger_file}")
                
                print("  ✅ jaeger.yml - Valid")
                
            except Exception as e:
                self.errors.append(f"Error validating {jaeger_file}: {e}")
    
    def _validate_environment_variables(self) -> None:
        """Validate required environment variables"""
        print("🌍 Validating environment variables...")
        
        # Check if .env files exist
        env_files = [".env", "development.env", "production.env"]
        env_file_found = False
        
        for env_file in env_files:
            if Path(env_file).exists():
                env_file_found = True
                print(f"  ✅ {env_file} - Found")
                break
        
        if not env_file_found:
            self.warnings.append("No .env files found")
        
        # Check for critical environment variables
        critical_vars = [
            "DB_PASSWORD",
            "JWT_SECRET_KEY",
            "KAFKA_PASSWORD",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY"
        ]
        
        for var in critical_vars:
            if not os.getenv(var):
                self.warnings.append(f"Critical environment variable {var} not set")
    
    def _validate_section(self, section_name: str, section_config: Dict[str, Any], file_path: Path) -> None:
        """Validate a configuration section"""
        if section_name.endswith(".database"):
            self._validate_database_section(section_config, file_path)
        elif section_name.endswith(".kafka"):
            self._validate_kafka_section(section_config, file_path)
        elif section_name.endswith(".s3"):
            self._validate_s3_section(section_config, file_path)
        elif section_name.endswith(".monitoring"):
            self._validate_monitoring_section(section_config, file_path)
    
    def _validate_database_section(self, config: Dict[str, Any], file_path: Path) -> None:
        """Validate database section"""
        required_fields = ["host", "port", "name", "username", "password"]
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field '{field}' in database section of {file_path}")
    
    def _validate_kafka_section(self, config: Dict[str, Any], file_path: Path) -> None:
        """Validate Kafka section"""
        required_fields = ["bootstrap_servers", "consumer_group"]
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field '{field}' in kafka section of {file_path}")
    
    def _validate_s3_section(self, config: Dict[str, Any], file_path: Path) -> None:
        """Validate S3 section"""
        required_fields = ["region", "bucket_name"]
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field '{field}' in s3 section of {file_path}")
    
    def _validate_monitoring_section(self, config: Dict[str, Any], file_path: Path) -> None:
        """Validate monitoring section"""
        required_services = ["prometheus", "grafana", "jaeger"]
        for service in required_services:
            if service not in config:
                self.warnings.append(f"Missing {service} configuration in monitoring section of {file_path}")
    
    def _validate_pool_config(self, env: str, config: Dict[str, Any], file_path: Path) -> None:
        """Validate connection pool configuration"""
        required_fields = ["pool_size", "max_overflow", "pool_timeout", "pool_recycle"]
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field '{field}' in {env} pool config of {file_path}")
    
    def _validate_kafka_config(self, config: Dict[str, Any], file_path: Path) -> None:
        """Validate Kafka configuration"""
        required_sections = ["topics", "producer", "consumer"]
        for section in required_sections:
            if section not in config:
                self.errors.append(f"Missing required section '{section}' in {file_path}")
    
    def _validate_s3_config(self, config: Dict[str, Any], file_path: Path) -> None:
        """Validate S3 configuration"""
        required_sections = ["buckets", "data_organization", "upload", "download"]
        for section in required_sections:
            if section not in config:
                self.errors.append(f"Missing required section '{section}' in {file_path}")
    
    def _validate_snowflake_config(self, config: Dict[str, Any], file_path: Path) -> None:
        """Validate Snowflake configuration"""
        required_sections = ["connection", "warehouses", "databases"]
        for section in required_sections:
            if section not in config:
                self.errors.append(f"Missing required section '{section}' in {file_path}")
    
    def _validate_monitoring_config_file(self, config: Dict[str, Any], file_path: Path) -> None:
        """Validate monitoring configuration file"""
        required_sections = ["prometheus", "grafana", "jaeger", "datadog"]
        for section in required_sections:
            if section not in config:
                self.warnings.append(f"Missing {section} configuration in {file_path}")
    
    def _print_results(self) -> None:
        """Print validation results"""
        print("\n" + "=" * 60)
        print("📋 Validation Results")
        print("=" * 60)
        
        if self.errors:
            print(f"❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("✅ All configuration files are valid!")
        elif not self.errors:
            print("✅ Configuration is valid with warnings")
        else:
            print("❌ Configuration has errors that need to be fixed")
        
        print("=" * 60)


def main():
    """Main validation function"""
    config_dir = sys.argv[1] if len(sys.argv) > 1 else "config"
    
    validator = ConfigValidator(config_dir)
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
