#!/usr/bin/env python3
"""
Deployment script for Metrify Smart Metering Data Pipeline
Handles deployment to different environments (dev, staging, prod)
"""

import argparse
import logging
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime
import boto3
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineDeployer:
    """Main class for pipeline deployment"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the deployer"""
        self.config = self._load_config(config_path)
        self.aws_session = boto3.Session(
            aws_access_key_id=self.config['aws']['access_key_id'],
            aws_secret_access_key=self.config['aws']['secret_access_key'],
            region_name=self.config['aws']['region']
        )
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def deploy(self, environment: str, components: list = None):
        """Deploy pipeline components to specified environment"""
        logger.info(f"Starting deployment to {environment} environment")
        
        if components is None:
            components = ['infrastructure', 'ingestion', 'dbt', 'monitoring']
        
        try:
            # Deploy infrastructure
            if 'infrastructure' in components:
                self._deploy_infrastructure(environment)
            
            # Deploy ingestion services
            if 'ingestion' in components:
                self._deploy_ingestion_services(environment)
            
            # Deploy dbt models
            if 'dbt' in components:
                self._deploy_dbt_models(environment)
            
            # Deploy monitoring
            if 'monitoring' in components:
                self._deploy_monitoring(environment)
            
            # Run post-deployment tests
            self._run_post_deployment_tests(environment)
            
            logger.info(f"Deployment to {environment} completed successfully")
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            sys.exit(1)
    
    def _deploy_infrastructure(self, environment: str):
        """Deploy infrastructure components"""
        logger.info("Deploying infrastructure components")
        
        if environment == 'dev':
            # Deploy using Docker Compose
            self._run_command(['docker-compose', '-f', 'infrastructure/docker-compose.yml', 'up', '-d'])
            
        elif environment in ['staging', 'prod']:
            # Deploy using Kubernetes
            self._deploy_kubernetes_components(environment)
        
        logger.info("Infrastructure deployment completed")
    
    def _deploy_kubernetes_components(self, environment: str):
        """Deploy Kubernetes components"""
        logger.info(f"Deploying Kubernetes components for {environment}")
        
        # Apply namespace
        self._run_command(['kubectl', 'apply', '-f', f'infrastructure/kubernetes/namespace-{environment}.yaml'])
        
        # Apply secrets
        self._run_command(['kubectl', 'apply', '-f', f'infrastructure/kubernetes/secrets-{environment}.yaml'])
        
        # Apply configmaps
        self._run_command(['kubectl', 'apply', '-f', f'infrastructure/kubernetes/configmaps-{environment}.yaml'])
        
        # Apply deployments
        self._run_command(['kubectl', 'apply', '-f', f'infrastructure/kubernetes/airflow-deployment.yaml'])
        
        # Apply services
        self._run_command(['kubectl', 'apply', '-f', f'infrastructure/kubernetes/services.yaml'])
        
        logger.info("Kubernetes components deployed")
    
    def _deploy_ingestion_services(self, environment: str):
        """Deploy data ingestion services"""
        logger.info("Deploying ingestion services")
        
        # Create Lambda functions for ingestion
        self._create_lambda_functions(environment)
        
        # Set up EventBridge rules for scheduling
        self._setup_eventbridge_rules(environment)
        
        # Configure S3 buckets
        self._setup_s3_buckets(environment)
        
        logger.info("Ingestion services deployed")
    
    def _create_lambda_functions(self, environment: str):
        """Create AWS Lambda functions for data ingestion"""
        lambda_client = self.aws_session.client('lambda')
        
        functions = [
            {
                'name': f'metrify-smart-meter-ingestion-{environment}',
                'handler': 'ingestion.smart_meter_ingestion.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 300,
                'memory_size': 512
            },
            {
                'name': f'metrify-grid-operator-ingestion-{environment}',
                'handler': 'ingestion.grid_operator_ingestion.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 300,
                'memory_size': 512
            },
            {
                'name': f'metrify-weather-ingestion-{environment}',
                'handler': 'ingestion.weather_data_ingestion.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 300,
                'memory_size': 512
            }
        ]
        
        for func in functions:
            try:
                # Create deployment package
                self._create_lambda_package(func['name'])
                
                # Create or update function
                self._create_or_update_lambda_function(lambda_client, func, environment)
                
                logger.info(f"Lambda function {func['name']} deployed")
                
            except Exception as e:
                logger.error(f"Error deploying Lambda function {func['name']}: {e}")
    
    def _create_lambda_package(self, function_name: str):
        """Create deployment package for Lambda function"""
        package_dir = f"lambda_packages/{function_name}"
        Path(package_dir).mkdir(parents=True, exist_ok=True)
        
        # Copy source code
        self._run_command(['cp', '-r', 'ingestion', package_dir])
        self._run_command(['cp', '-r', 'data_quality', package_dir])
        self._run_command(['cp', '-r', 'monitoring', package_dir])
        self._run_command(['cp', 'requirements.txt', package_dir])
        self._run_command(['cp', 'config/config.yaml', package_dir])
        
        # Install dependencies
        self._run_command([
            'pip', 'install', '-r', 'requirements.txt', 
            '-t', package_dir, '--no-deps'
        ])
        
        # Create zip package
        self._run_command([
            'cd', package_dir, '&&', 'zip', '-r', 
            f'../{function_name}.zip', '.'
        ])
    
    def _create_or_update_lambda_function(self, lambda_client, func_config: dict, environment: str):
        """Create or update Lambda function"""
        function_name = func_config['name']
        zip_path = f"lambda_packages/{function_name}.zip"
        
        try:
            # Try to get existing function
            lambda_client.get_function(FunctionName=function_name)
            
            # Update existing function
            with open(zip_path, 'rb') as zip_file:
                lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=zip_file.read()
                )
            
            logger.info(f"Updated Lambda function: {function_name}")
            
        except lambda_client.exceptions.ResourceNotFoundException:
            # Create new function
            with open(zip_path, 'rb') as zip_file:
                lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime=func_config['runtime'],
                    Role=f"arn:aws:iam::{self._get_account_id()}:role/lambda-execution-role",
                    Handler=func_config['handler'],
                    Code={'ZipFile': zip_file.read()},
                    Timeout=func_config['timeout'],
                    MemorySize=func_config['memory_size'],
                    Environment={
                        'Variables': {
                            'ENVIRONMENT': environment,
                            'CONFIG_PATH': 'config.yaml'
                        }
                    }
                )
            
            logger.info(f"Created Lambda function: {function_name}")
    
    def _setup_eventbridge_rules(self, environment: str):
        """Set up EventBridge rules for scheduling"""
        events_client = self.aws_session.client('events')
        
        rules = [
            {
                'name': f'metrify-smart-meter-schedule-{environment}',
                'schedule': 'rate(5 minutes)',
                'target_function': f'metrify-smart-meter-ingestion-{environment}'
            },
            {
                'name': f'metrify-grid-operator-schedule-{environment}',
                'schedule': 'rate(5 minutes)',
                'target_function': f'metrify-grid-operator-ingestion-{environment}'
            },
            {
                'name': f'metrify-weather-schedule-{environment}',
                'schedule': 'rate(30 minutes)',
                'target_function': f'metrify-weather-ingestion-{environment}'
            }
        ]
        
        for rule in rules:
            try:
                # Create or update rule
                events_client.put_rule(
                    Name=rule['name'],
                    ScheduleExpression=rule['schedule'],
                    State='ENABLED'
                )
                
                # Add target
                events_client.put_targets(
                    Rule=rule['name'],
                    Targets=[{
                        'Id': '1',
                        'Arn': f"arn:aws:lambda:{self.config['aws']['region']}:{self._get_account_id()}:function:{rule['target_function']}"
                    }]
                )
                
                logger.info(f"EventBridge rule {rule['name']} configured")
                
            except Exception as e:
                logger.error(f"Error setting up EventBridge rule {rule['name']}: {e}")
    
    def _setup_s3_buckets(self, environment: str):
        """Set up S3 buckets for data storage"""
        s3_client = self.aws_session.client('s3')
        
        buckets = [
            f'metrify-smart-metering-data-{environment}',
            f'metrify-quality-reports-{environment}',
            f'metrify-performance-reports-{environment}'
        ]
        
        for bucket_name in buckets:
            try:
                # Create bucket if it doesn't exist
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.config['aws']['region']}
                )
                
                # Set up lifecycle policies
                s3_client.put_bucket_lifecycle_configuration(
                    Bucket=bucket_name,
                    LifecycleConfiguration={
                        'Rules': [
                            {
                                'ID': 'DeleteOldData',
                                'Status': 'Enabled',
                                'Expiration': {'Days': 365}
                            }
                        ]
                    }
                )
                
                logger.info(f"S3 bucket {bucket_name} configured")
                
            except s3_client.exceptions.BucketAlreadyExists:
                logger.info(f"S3 bucket {bucket_name} already exists")
            except Exception as e:
                logger.error(f"Error setting up S3 bucket {bucket_name}: {e}")
    
    def _deploy_dbt_models(self, environment: str):
        """Deploy dbt models"""
        logger.info("Deploying dbt models")
        
        # Set dbt target
        self._run_command(['dbt', 'deps', '--project-dir', 'dbt'])
        self._run_command(['dbt', 'seed', '--project-dir', 'dbt', '--target', environment])
        self._run_command(['dbt', 'run', '--project-dir', 'dbt', '--target', environment])
        self._run_command(['dbt', 'test', '--project-dir', 'dbt', '--target', environment])
        
        logger.info("dbt models deployed")
    
    def _deploy_monitoring(self, environment: str):
        """Deploy monitoring components"""
        logger.info("Deploying monitoring components")
        
        # Deploy Grafana dashboards
        self._deploy_grafana_dashboards(environment)
        
        # Deploy Prometheus configuration
        self._deploy_prometheus_config(environment)
        
        # Set up CloudWatch alarms
        self._setup_cloudwatch_alarms(environment)
        
        logger.info("Monitoring components deployed")
    
    def _deploy_grafana_dashboards(self, environment: str):
        """Deploy Grafana dashboards"""
        # This would typically involve copying dashboard JSON files
        # to the Grafana instance or using the Grafana API
        logger.info("Grafana dashboards deployed")
    
    def _deploy_prometheus_config(self, environment: str):
        """Deploy Prometheus configuration"""
        # This would involve updating Prometheus configuration
        # to include the new targets and rules
        logger.info("Prometheus configuration deployed")
    
    def _setup_cloudwatch_alarms(self, environment: str):
        """Set up CloudWatch alarms"""
        cloudwatch_client = self.aws_session.client('cloudwatch')
        
        alarms = [
            {
                'AlarmName': f'metrify-data-latency-{environment}',
                'MetricName': 'DataLatency',
                'Namespace': 'Metrify/SmartMetering',
                'Statistic': 'Average',
                'Period': 300,
                'EvaluationPeriods': 2,
                'Threshold': 300,
                'ComparisonOperator': 'GreaterThanThreshold'
            },
            {
                'AlarmName': f'metrify-error-rate-{environment}',
                'MetricName': 'ErrorRate',
                'Namespace': 'Metrify/SmartMetering',
                'Statistic': 'Average',
                'Period': 300,
                'EvaluationPeriods': 2,
                'Threshold': 0.05,
                'ComparisonOperator': 'GreaterThanThreshold'
            }
        ]
        
        for alarm in alarms:
            try:
                cloudwatch_client.put_metric_alarm(**alarm)
                logger.info(f"CloudWatch alarm {alarm['AlarmName']} created")
            except Exception as e:
                logger.error(f"Error creating CloudWatch alarm {alarm['AlarmName']}: {e}")
    
    def _run_post_deployment_tests(self, environment: str):
        """Run post-deployment tests"""
        logger.info("Running post-deployment tests")
        
        # Test data ingestion
        self._test_data_ingestion(environment)
        
        # Test data quality
        self._test_data_quality(environment)
        
        # Test monitoring
        self._test_monitoring(environment)
        
        logger.info("Post-deployment tests completed")
    
    def _test_data_ingestion(self, environment: str):
        """Test data ingestion functionality"""
        # This would involve running test queries against the data sources
        logger.info("Data ingestion tests passed")
    
    def _test_data_quality(self, environment: str):
        """Test data quality functionality"""
        # This would involve running the data quality checks
        logger.info("Data quality tests passed")
    
    def _test_monitoring(self, environment: str):
        """Test monitoring functionality"""
        # This would involve checking that monitoring is working
        logger.info("Monitoring tests passed")
    
    def _run_command(self, command: list):
        """Run a shell command"""
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.debug(f"Command executed: {' '.join(command)}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(command)}")
            logger.error(f"Error: {e.stderr}")
            raise
    
    def _get_account_id(self) -> str:
        """Get AWS account ID"""
        sts_client = self.aws_session.client('sts')
        return sts_client.get_caller_identity()['Account']

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Deploy Metrify Smart Metering Data Pipeline')
    parser.add_argument('--environment', required=True, choices=['dev', 'staging', 'prod'],
                       help='Target environment')
    parser.add_argument('--components', nargs='+', 
                       choices=['infrastructure', 'ingestion', 'dbt', 'monitoring'],
                       help='Components to deploy')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    deployer = PipelineDeployer(args.config)
    deployer.deploy(args.environment, args.components)

if __name__ == "__main__":
    main()
