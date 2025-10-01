#!/bin/bash
# Kubernetes deployment script for Metrify Smart Metering Data Pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE=${NAMESPACE:-"metrify"}
ENVIRONMENT=${ENVIRONMENT:-"production"}
DRY_RUN=${DRY_RUN:-"false"}

echo -e "${BLUE}🚀 Deploying Metrify Smart Metering to Kubernetes${NC}"
echo "=================================================="
echo "Namespace: $NAMESPACE"
echo "Environment: $ENVIRONMENT"
echo "Dry Run: $DRY_RUN"
echo "=================================================="

# Function to apply manifests
apply_manifests() {
    local directory=$1
    local description=$2
    
    echo -e "${YELLOW}Deploying $description...${NC}"
    
    if [ "$DRY_RUN" = "true" ]; then
        kubectl apply -f "$directory" --dry-run=client -o yaml
    else
        kubectl apply -f "$directory"
    fi
    
    echo -e "${GREEN}✅ $description deployed successfully${NC}"
}

# Function to check if namespace exists
check_namespace() {
    if ! kubectl get namespace "$NAMESPACE" > /dev/null 2>&1; then
        echo -e "${YELLOW}Creating namespace $NAMESPACE...${NC}"
        kubectl create namespace "$NAMESPACE"
        echo -e "${GREEN}✅ Namespace $NAMESPACE created${NC}"
    else
        echo -e "${GREEN}✅ Namespace $NAMESPACE already exists${NC}"
    fi
}

# Function to create secrets
create_secrets() {
    echo -e "${YELLOW}Creating secrets...${NC}"
    
    # Database secret
    if ! kubectl get secret metrify-database-secret -n "$NAMESPACE" > /dev/null 2>&1; then
        kubectl create secret generic metrify-database-secret \
            --from-literal=host="${DB_HOST:-postgres.metrify.internal}" \
            --from-literal=database="${DB_NAME:-metrify_prod}" \
            --from-literal=username="${DB_USERNAME:-metrify_prod}" \
            --from-literal=password="${DB_PASSWORD:-change-me}" \
            -n "$NAMESPACE"
        echo -e "${GREEN}✅ Database secret created${NC}"
    else
        echo -e "${GREEN}✅ Database secret already exists${NC}"
    fi
    
    # Kafka secret
    if ! kubectl get secret metrify-kafka-secret -n "$NAMESPACE" > /dev/null 2>&1; then
        kubectl create secret generic metrify-kafka-secret \
            --from-literal=username="${KAFKA_USERNAME:-metrify}" \
            --from-literal=password="${KAFKA_PASSWORD:-change-me}" \
            -n "$NAMESPACE"
        echo -e "${GREEN}✅ Kafka secret created${NC}"
    else
        echo -e "${GREEN}✅ Kafka secret already exists${NC}"
    fi
    
    # S3 secret
    if ! kubectl get secret metrify-s3-secret -n "$NAMESPACE" > /dev/null 2>&1; then
        kubectl create secret generic metrify-s3-secret \
            --from-literal=access_key_id="${AWS_ACCESS_KEY_ID:-change-me}" \
            --from-literal=secret_access_key="${AWS_SECRET_ACCESS_KEY:-change-me}" \
            -n "$NAMESPACE"
        echo -e "${GREEN}✅ S3 secret created${NC}"
    else
        echo -e "${GREEN}✅ S3 secret already exists${NC}"
    fi
    
    # JWT secret
    if ! kubectl get secret metrify-jwt-secret -n "$NAMESPACE" > /dev/null 2>&1; then
        kubectl create secret generic metrify-jwt-secret \
            --from-literal=secret_key="${JWT_SECRET_KEY:-change-me-in-production}" \
            -n "$NAMESPACE"
        echo -e "${GREEN}✅ JWT secret created${NC}"
    else
        echo -e "${GREEN}✅ JWT secret already exists${NC}"
    fi
    
    # Snowflake secret
    if ! kubectl get secret metrify-snowflake-secret -n "$NAMESPACE" > /dev/null 2>&1; then
        kubectl create secret generic metrify-snowflake-secret \
            --from-literal=account="${SNOWFLAKE_ACCOUNT:-change-me}" \
            --from-literal=warehouse="${SNOWFLAKE_WAREHOUSE:-PROD_WH}" \
            --from-literal=database="${SNOWFLAKE_DATABASE:-METRIFY_PROD}" \
            --from-literal=username="${SNOWFLAKE_USERNAME:-change-me}" \
            --from-literal=password="${SNOWFLAKE_PASSWORD:-change-me}" \
            -n "$NAMESPACE"
        echo -e "${GREEN}✅ Snowflake secret created${NC}"
    else
        echo -e "${GREEN}✅ Snowflake secret already exists${NC}"
    fi
}

# Function to create service accounts
create_service_accounts() {
    echo -e "${YELLOW}Creating service accounts...${NC}"
    
    # API service account
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: metrify-api
  namespace: $NAMESPACE
  labels:
    app: metrify-api
    component: api
EOF
    
    # Worker service account
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: metrify-worker
  namespace: $NAMESPACE
  labels:
    app: metrify-worker
    component: worker
EOF
    
    echo -e "${GREEN}✅ Service accounts created${NC}"
}

# Function to check deployment status
check_deployment_status() {
    local deployment=$1
    local namespace=$2
    
    echo -e "${YELLOW}Checking status of $deployment...${NC}"
    
    if kubectl get deployment "$deployment" -n "$namespace" > /dev/null 2>&1; then
        kubectl rollout status deployment/"$deployment" -n "$namespace" --timeout=300s
        echo -e "${GREEN}✅ $deployment is ready${NC}"
    else
        echo -e "${RED}❌ $deployment not found${NC}"
    fi
}

# Function to show deployment information
show_deployment_info() {
    echo -e "${BLUE}📋 Deployment Information${NC}"
    echo "=================================================="
    
    # Show pods
    echo -e "${YELLOW}Pods:${NC}"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo ""
    
    # Show services
    echo -e "${YELLOW}Services:${NC}"
    kubectl get services -n "$NAMESPACE" -o wide
    
    echo ""
    
    # Show ingress
    echo -e "${YELLOW}Ingress:${NC}"
    kubectl get ingress -n "$NAMESPACE" -o wide 2>/dev/null || echo "No ingress found"
    
    echo ""
    
    # Show secrets
    echo -e "${YELLOW}Secrets:${NC}"
    kubectl get secrets -n "$NAMESPACE"
}

# Main deployment process
main() {
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl is not installed or not in PATH${NC}"
        exit 1
    fi
    
    # Check if we can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
        exit 1
    fi
    
    # Create namespace
    check_namespace
    
    # Create secrets
    create_secrets
    
    # Create service accounts
    create_service_accounts
    
    # Deploy configurations
    apply_manifests "configmaps/" "ConfigMaps"
    
    # Deploy services
    apply_manifests "services/" "Services"
    
    # Deploy applications
    apply_manifests "deployments/" "Deployments"
    
    # Wait for deployments to be ready
    if [ "$DRY_RUN" = "false" ]; then
        echo -e "${YELLOW}Waiting for deployments to be ready...${NC}"
        
        check_deployment_status "metrify-api" "$NAMESPACE"
        check_deployment_status "metrify-ingestion-worker" "$NAMESPACE"
        check_deployment_status "metrify-processing-worker" "$NAMESPACE"
        
        # Show deployment information
        show_deployment_info
        
        echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
        echo ""
        echo -e "${BLUE}📖 Next Steps:${NC}"
        echo "1. Check logs: kubectl logs -f deployment/metrify-api -n $NAMESPACE"
        echo "2. Scale workers: kubectl scale deployment metrify-ingestion-worker --replicas=10 -n $NAMESPACE"
        echo "3. Port forward: kubectl port-forward service/metrify-api 8000:80 -n $NAMESPACE"
        echo "4. Monitor: kubectl top pods -n $NAMESPACE"
    else
        echo -e "${GREEN}🎉 Dry run completed successfully!${NC}"
    fi
}

# Run main function
main "$@"
