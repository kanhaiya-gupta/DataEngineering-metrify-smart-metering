# Metrify Smart Metering Data Pipeline - Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Metrify Smart Metering Data Pipeline across different environments (development, staging, and production).

## Prerequisites

### 1. System Requirements

#### Development Environment
- **OS**: Windows 10/11, macOS 10.15+, or Ubuntu 20.04+
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 100GB free space
- **CPU**: 4 cores minimum, 8 cores recommended

#### Production Environment
- **OS**: Ubuntu 20.04+ or Amazon Linux 2
- **RAM**: 64GB minimum, 128GB recommended
- **Storage**: 1TB free space minimum
- **CPU**: 16 cores minimum, 32 cores recommended

### 2. Software Requirements

#### Required Software
- **Python**: 3.9 or higher
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **Kubernetes**: 1.21 or higher (for production)
- **kubectl**: 1.21 or higher
- **AWS CLI**: 2.0 or higher
- **dbt**: 1.6 or higher
- **Git**: 2.30 or higher

#### Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Cloud Requirements

#### AWS Services
- **S3**: Data storage buckets
- **Lambda**: Serverless functions
- **EventBridge**: Event scheduling
- **CloudWatch**: Monitoring and logging
- **IAM**: Access management
- **VPC**: Network isolation
- **RDS**: PostgreSQL database
- **EMR**: Spark processing (optional)

#### Snowflake
- **Account**: Snowflake account with appropriate permissions
- **Warehouse**: Compute warehouse for data processing
- **Database**: Database for storing processed data
- **Schema**: Schemas for different data layers

## Environment Setup

### 1. Development Environment

#### Step 1: Clone Repository
```bash
git clone https://github.com/metrify/smart-metering-pipeline.git
cd smart-metering-pipeline
```

#### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Configure Environment
```bash
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your settings
```

#### Step 5: Start Local Services
```bash
docker-compose -f infrastructure/docker-compose.yml up -d
```

#### Step 6: Initialize Airflow
```bash
# Wait for services to start
sleep 30

# Initialize Airflow database
docker-compose -f infrastructure/docker-compose.yml exec airflow-webserver airflow db init

# Create Airflow admin user
docker-compose -f infrastructure/docker-compose.yml exec airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@metrify.com \
    --password admin
```

#### Step 7: Access Services
- **Airflow**: http://localhost:8080 (admin/admin)
- **Kafka UI**: http://localhost:8081
- **Jupyter**: http://localhost:8888
- **Grafana**: http://localhost:3000 (admin/admin)
- **MinIO**: http://localhost:9000 (minioadmin/minioadmin)

### 2. Staging Environment

#### Step 1: Prepare Kubernetes Cluster
```bash
# Create EKS cluster
eksctl create cluster \
    --name metrify-staging \
    --region eu-central-1 \
    --nodegroup-name workers \
    --node-type t3.large \
    --nodes 3 \
    --nodes-min 1 \
    --nodes-max 5
```

#### Step 2: Configure kubectl
```bash
aws eks update-kubeconfig --region eu-central-1 --name metrify-staging
```

#### Step 3: Create Namespace
```bash
kubectl create namespace metrify-staging
kubectl config set-context --current --namespace=metrify-staging
```

#### Step 4: Deploy Secrets
```bash
# Create secrets file
cat > infrastructure/kubernetes/secrets-staging.yaml << EOF
apiVersion: v1
kind: Secret
metadata:
  name: airflow-secrets
  namespace: metrify-staging
type: Opaque
data:
  sql-alchemy-conn: $(echo -n "postgresql://airflow:airflow@postgres:5432/airflow" | base64)
  secret-key: $(echo -n "your-secret-key-here" | base64)
  snowflake-conn: $(echo -n "your-snowflake-connection-string" | base64)
EOF

kubectl apply -f infrastructure/kubernetes/secrets-staging.yaml
```

#### Step 5: Deploy Configuration
```bash
# Create configmaps
kubectl create configmap airflow-config \
    --from-file=config/config.yaml \
    --namespace=metrify-staging

kubectl create configmap airflow-dags \
    --from-file=dags/ \
    --namespace=metrify-staging
```

#### Step 6: Deploy Applications
```bash
# Deploy PostgreSQL
kubectl apply -f infrastructure/kubernetes/postgres-deployment.yaml

# Deploy Airflow
kubectl apply -f infrastructure/kubernetes/airflow-deployment.yaml

# Deploy Kafka
kubectl apply -f infrastructure/kubernetes/kafka-deployment.yaml
```

#### Step 7: Verify Deployment
```bash
kubectl get pods
kubectl get services
kubectl logs -f deployment/airflow-webserver
```

### 3. Production Environment

#### Step 1: Prepare Production Infrastructure
```bash
# Create production EKS cluster
eksctl create cluster \
    --name metrify-production \
    --region eu-central-1 \
    --nodegroup-name workers \
    --node-type t3.xlarge \
    --nodes 5 \
    --nodes-min 3 \
    --nodes-max 10 \
    --ssh-access \
    --ssh-public-key ~/.ssh/id_rsa.pub
```

#### Step 2: Configure High Availability
```bash
# Deploy across multiple AZs
kubectl apply -f infrastructure/kubernetes/multi-az-deployment.yaml
```

#### Step 3: Set Up Monitoring
```bash
# Deploy Prometheus
kubectl apply -f infrastructure/kubernetes/prometheus-deployment.yaml

# Deploy Grafana
kubectl apply -f infrastructure/kubernetes/grafana-deployment.yaml

# Deploy DataDog agent
kubectl apply -f infrastructure/kubernetes/datadog-agent.yaml
```

#### Step 4: Configure Security
```bash
# Deploy network policies
kubectl apply -f infrastructure/kubernetes/network-policies.yaml

# Deploy Pod Security Policies
kubectl apply -f infrastructure/kubernetes/pod-security-policies.yaml

# Deploy RBAC
kubectl apply -f infrastructure/kubernetes/rbac.yaml
```

## Data Pipeline Deployment

### 1. Deploy Data Ingestion

#### Step 1: Create Lambda Functions
```bash
python scripts/deploy.py --environment production --components ingestion
```

#### Step 2: Set Up EventBridge Rules
```bash
# Smart meter ingestion (every 5 minutes)
aws events put-rule \
    --name metrify-smart-meter-schedule-prod \
    --schedule-expression "rate(5 minutes)"

# Grid operator ingestion (every 5 minutes)
aws events put-rule \
    --name metrify-grid-operator-schedule-prod \
    --schedule-expression "rate(5 minutes)"

# Weather data ingestion (every 30 minutes)
aws events put-rule \
    --name metrify-weather-schedule-prod \
    --schedule-expression "rate(30 minutes)"
```

#### Step 3: Configure S3 Buckets
```bash
# Create data buckets
aws s3 mb s3://metrify-smart-metering-data-prod
aws s3 mb s3://metrify-quality-reports-prod
aws s3 mb s3://metrify-performance-reports-prod

# Set up lifecycle policies
aws s3api put-bucket-lifecycle-configuration \
    --bucket metrify-smart-metering-data-prod \
    --lifecycle-configuration file://infrastructure/s3-lifecycle.json
```

### 2. Deploy Data Processing

#### Step 1: Deploy Airflow DAGs
```bash
# Copy DAGs to Airflow
kubectl cp dags/ airflow-webserver-0:/opt/airflow/dags/ -n metrify-production

# Restart Airflow scheduler
kubectl rollout restart deployment/airflow-scheduler -n metrify-production
```

#### Step 2: Deploy dbt Models
```bash
# Set up dbt profiles
mkdir -p ~/.dbt
cp dbt/profiles.yml ~/.dbt/

# Deploy dbt models
dbt deps --project-dir dbt
dbt seed --project-dir dbt --target prod
dbt run --project-dir dbt --target prod
dbt test --project-dir dbt --target prod
```

#### Step 3: Configure Data Quality
```bash
# Deploy data quality checks
kubectl apply -f infrastructure/kubernetes/data-quality-deployment.yaml

# Set up quality monitoring
kubectl apply -f infrastructure/kubernetes/quality-monitoring.yaml
```

### 3. Deploy Monitoring

#### Step 1: Deploy Grafana Dashboards
```bash
# Import dashboards
kubectl cp monitoring/grafana/dashboards/ grafana-0:/var/lib/grafana/dashboards/ -n metrify-production

# Restart Grafana
kubectl rollout restart deployment/grafana -n metrify-production
```

#### Step 2: Configure Prometheus
```bash
# Deploy Prometheus configuration
kubectl apply -f infrastructure/kubernetes/prometheus-config.yaml

# Deploy alerting rules
kubectl apply -f infrastructure/kubernetes/prometheus-alerts.yaml
```

#### Step 3: Set Up DataDog
```bash
# Deploy DataDog agent
kubectl apply -f infrastructure/kubernetes/datadog-agent.yaml

# Configure DataDog dashboards
python scripts/setup_datadog.py --environment production
```

## Configuration Management

### 1. Environment Variables

#### Development
```bash
export ENVIRONMENT=dev
export SNOWFLAKE_ACCOUNT=your-account.snowflakecomputing.com
export SNOWFLAKE_USER=your-username
export SNOWFLAKE_PASSWORD=your-password
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

#### Staging
```bash
export ENVIRONMENT=staging
export SNOWFLAKE_ACCOUNT=your-account.snowflakecomputing.com
export SNOWFLAKE_USER=your-username
export SNOWFLAKE_PASSWORD=your-password
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

#### Production
```bash
export ENVIRONMENT=prod
export SNOWFLAKE_ACCOUNT=your-account.snowflakecomputing.com
export SNOWFLAKE_USER=your-username
export SNOWFLAKE_PASSWORD=your-password
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

### 2. Configuration Files

#### config/config.yaml
```yaml
environment: "prod"
databases:
  snowflake:
    account: "your-account.snowflakecomputing.com"
    user: "your-username"
    password: "your-password"
    warehouse: "COMPUTE_WH"
    database: "METRIFY_ANALYTICS"
    schema: "RAW"
aws:
  region: "eu-central-1"
  s3_bucket: "metrify-smart-metering-data-prod"
  access_key_id: "your-access-key"
  secret_access_key: "your-secret-key"
```

#### dbt/profiles.yml
```yaml
metrify_analytics:
  target: prod
  outputs:
    prod:
      type: snowflake
      account: "your-account.snowflakecomputing.com"
      user: "your-username"
      password: "your-password"
      warehouse: "COMPUTE_WH"
      database: "METRIFY_ANALYTICS"
      schema: "RAW"
      threads: 4
```

## Testing and Validation

### 1. Unit Tests
```bash
# Run unit tests
python -m pytest tests/unit/ -v

# Run with coverage
python -m pytest tests/unit/ --cov=ingestion --cov=data_quality --cov=monitoring
```

### 2. Integration Tests
```bash
# Run integration tests
python -m pytest tests/integration/ -v

# Test data ingestion
python tests/integration/test_ingestion.py

# Test data quality
python tests/integration/test_data_quality.py
```

### 3. End-to-End Tests
```bash
# Run end-to-end tests
python -m pytest tests/e2e/ -v

# Test complete pipeline
python tests/e2e/test_pipeline.py
```

### 4. Performance Tests
```bash
# Run performance tests
python tests/performance/test_performance.py

# Load testing
python tests/performance/load_test.py
```

## Monitoring and Maintenance

### 1. Health Checks
```bash
# Check Airflow health
kubectl get pods -n metrify-production | grep airflow

# Check data quality
python monitoring/data_quality_monitor.py --date $(date +%Y-%m-%d)

# Check performance
python monitoring/performance_monitor.py --date $(date +%Y-%m-%d)
```

### 2. Log Monitoring
```bash
# View Airflow logs
kubectl logs -f deployment/airflow-webserver -n metrify-production

# View data quality logs
kubectl logs -f deployment/data-quality-monitor -n metrify-production

# View performance logs
kubectl logs -f deployment/performance-monitor -n metrify-production
```

### 3. Backup and Recovery
```bash
# Backup Airflow database
kubectl exec -it postgres-0 -n metrify-production -- pg_dump -U airflow airflow > airflow_backup.sql

# Backup Snowflake data
snowsql -c production -q "CREATE OR REPLACE TABLE backup.smart_meter_readings AS SELECT * FROM raw.smart_meter_readings;"

# Backup S3 data
aws s3 sync s3://metrify-smart-metering-data-prod s3://metrify-backup-bucket/
```

## Troubleshooting

### 1. Common Issues

#### Airflow DAG Not Running
```bash
# Check DAG status
kubectl exec -it airflow-webserver-0 -n metrify-production -- airflow dags list

# Check DAG logs
kubectl exec -it airflow-webserver-0 -n metrify-production -- airflow tasks logs smart_meter_pipeline extract_smart_meter_data 2024-01-01
```

#### Data Quality Issues
```bash
# Check data quality logs
kubectl logs -f deployment/data-quality-monitor -n metrify-production

# Run manual quality check
python data_quality/quality_checks.py --date 2024-01-01
```

#### Performance Issues
```bash
# Check performance metrics
kubectl logs -f deployment/performance-monitor -n metrify-production

# Check resource usage
kubectl top pods -n metrify-production
```

### 2. Debugging Commands

#### Check Pod Status
```bash
kubectl get pods -n metrify-production
kubectl describe pod <pod-name> -n metrify-production
```

#### Check Service Status
```bash
kubectl get services -n metrify-production
kubectl describe service <service-name> -n metrify-production
```

#### Check Logs
```bash
kubectl logs <pod-name> -n metrify-production
kubectl logs <pod-name> -n metrify-production --previous
```

## Security Considerations

### 1. Access Control
- Use IAM roles for AWS services
- Implement RBAC for Kubernetes
- Use secrets management for sensitive data
- Enable audit logging

### 2. Network Security
- Use VPC for network isolation
- Implement security groups
- Use private subnets for databases
- Enable VPC flow logs

### 3. Data Security
- Encrypt data at rest and in transit
- Use key management services
- Implement data masking
- Regular security audits

## Cost Optimization

### 1. Resource Optimization
- Use spot instances for non-critical workloads
- Implement auto-scaling
- Use reserved instances for predictable workloads
- Regular resource right-sizing

### 2. Data Lifecycle Management
- Implement data archiving
- Use appropriate storage classes
- Regular data cleanup
- Monitor storage costs

### 3. Query Optimization
- Optimize slow queries
- Use appropriate indexes
- Implement query caching
- Regular query performance analysis

## Conclusion

This deployment guide provides comprehensive instructions for deploying the Metrify Smart Metering Data Pipeline across different environments. Follow the steps carefully and ensure all prerequisites are met before proceeding with deployment.

For additional support or questions, please refer to the troubleshooting section or contact the data engineering team.
