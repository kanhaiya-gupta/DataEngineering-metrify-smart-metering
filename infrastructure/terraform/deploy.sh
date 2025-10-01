#!/bin/bash
# Terraform deployment script for Metrify Smart Metering Infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-"staging"}
ACTION=${2:-"plan"}
AWS_REGION=${AWS_REGION:-"eu-central-1"}
TF_VAR_database_password=${TF_VAR_database_password:-"change-me-in-production"}

echo -e "${BLUE}🏗️  Deploying Metrify Smart Metering Infrastructure${NC}"
echo "=================================================="
echo "Environment: $ENVIRONMENT"
echo "Action: $ACTION"
echo "AWS Region: $AWS_REGION"
echo "=================================================="

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        echo -e "${RED}❌ Terraform is not installed${NC}"
        exit 1
    fi
    
    # Check if aws cli is installed
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}❌ AWS CLI is not installed${NC}"
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl is not installed${NC}"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED}❌ AWS credentials not configured${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All prerequisites met${NC}"
}

# Function to initialize Terraform
init_terraform() {
    echo -e "${YELLOW}Initializing Terraform...${NC}"
    
    terraform init \
        -backend-config="bucket=metrify-terraform-state" \
        -backend-config="key=infrastructure/${ENVIRONMENT}/terraform.tfstate" \
        -backend-config="region=${AWS_REGION}" \
        -backend-config="dynamodb_table=metrify-terraform-locks"
    
    echo -e "${GREEN}✅ Terraform initialized${NC}"
}

# Function to validate Terraform configuration
validate_terraform() {
    echo -e "${YELLOW}Validating Terraform configuration...${NC}"
    
    terraform validate
    
    echo -e "${GREEN}✅ Terraform configuration is valid${NC}"
}

# Function to plan Terraform changes
plan_terraform() {
    echo -e "${YELLOW}Planning Terraform changes...${NC}"
    
    terraform plan \
        -var-file="environments/${ENVIRONMENT}.tfvars" \
        -var="database_password=${TF_VAR_database_password}" \
        -out="terraform-${ENVIRONMENT}.plan"
    
    echo -e "${GREEN}✅ Terraform plan completed${NC}"
}

# Function to apply Terraform changes
apply_terraform() {
    echo -e "${YELLOW}Applying Terraform changes...${NC}"
    
    if [ -f "terraform-${ENVIRONMENT}.plan" ]; then
        terraform apply "terraform-${ENVIRONMENT}.plan"
    else
        terraform apply \
            -var-file="environments/${ENVIRONMENT}.tfvars" \
            -var="database_password=${TF_VAR_database_password}" \
            -auto-approve
    fi
    
    echo -e "${GREEN}✅ Terraform apply completed${NC}"
}

# Function to destroy Terraform resources
destroy_terraform() {
    echo -e "${YELLOW}Destroying Terraform resources...${NC}"
    
    read -p "Are you sure you want to destroy all resources? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        terraform destroy \
            -var-file="environments/${ENVIRONMENT}.tfvars" \
            -var="database_password=${TF_VAR_database_password}" \
            -auto-approve
        
        echo -e "${GREEN}✅ Terraform destroy completed${NC}"
    else
        echo -e "${YELLOW}Destroy cancelled${NC}"
    fi
}

# Function to show Terraform outputs
show_outputs() {
    echo -e "${YELLOW}Showing Terraform outputs...${NC}"
    
    terraform output
    
    echo -e "${GREEN}✅ Terraform outputs displayed${NC}"
}

# Function to update kubeconfig
update_kubeconfig() {
    echo -e "${YELLOW}Updating kubeconfig...${NC}"
    
    CLUSTER_NAME=$(terraform output -raw cluster_name)
    
    aws eks update-kubeconfig \
        --region "${AWS_REGION}" \
        --name "${CLUSTER_NAME}"
    
    echo -e "${GREEN}✅ kubeconfig updated${NC}"
}

# Function to deploy Kubernetes manifests
deploy_kubernetes() {
    echo -e "${YELLOW}Deploying Kubernetes manifests...${NC}"
    
    # Update kubeconfig first
    update_kubeconfig
    
    # Deploy to Kubernetes
    cd ../kubernetes
    ./deploy.sh
    
    echo -e "${GREEN}✅ Kubernetes deployment completed${NC}"
}

# Function to show deployment status
show_status() {
    echo -e "${BLUE}📋 Deployment Status${NC}"
    echo "=================================================="
    
    # Show Terraform outputs
    echo -e "${YELLOW}Terraform Outputs:${NC}"
    terraform output
    
    echo ""
    
    # Show Kubernetes resources
    echo -e "${YELLOW}Kubernetes Resources:${NC}"
    kubectl get pods -n metrify 2>/dev/null || echo "Kubernetes not deployed yet"
    
    echo ""
    
    # Show AWS resources
    echo -e "${YELLOW}AWS Resources:${NC}"
    aws ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType]' --output table 2>/dev/null || echo "No EC2 instances found"
}

# Function to show help
show_help() {
    echo -e "${BLUE}Metrify Smart Metering Infrastructure Deployment${NC}"
    echo "=================================================="
    echo ""
    echo "Usage: $0 [ENVIRONMENT] [ACTION]"
    echo ""
    echo "Environments:"
    echo "  staging    - Deploy to staging environment"
    echo "  production - Deploy to production environment"
    echo ""
    echo "Actions:"
    echo "  init       - Initialize Terraform"
    echo "  plan       - Plan Terraform changes"
    echo "  apply      - Apply Terraform changes"
    echo "  destroy    - Destroy Terraform resources"
    echo "  output     - Show Terraform outputs"
    echo "  status     - Show deployment status"
    echo "  k8s        - Deploy Kubernetes manifests"
    echo "  all        - Full deployment (plan + apply + k8s)"
    echo ""
    echo "Environment Variables:"
    echo "  AWS_REGION                    - AWS region (default: eu-central-1)"
    echo "  TF_VAR_database_password      - Database password"
    echo ""
    echo "Examples:"
    echo "  $0 staging plan"
    echo "  $0 production apply"
    echo "  $0 staging all"
}

# Main function
main() {
    case $ACTION in
        "init")
            check_prerequisites
            init_terraform
            ;;
        "plan")
            check_prerequisites
            init_terraform
            validate_terraform
            plan_terraform
            ;;
        "apply")
            check_prerequisites
            init_terraform
            validate_terraform
            apply_terraform
            show_outputs
            ;;
        "destroy")
            check_prerequisites
            init_terraform
            destroy_terraform
            ;;
        "output")
            check_prerequisites
            init_terraform
            show_outputs
            ;;
        "status")
            check_prerequisites
            init_terraform
            show_status
            ;;
        "k8s")
            check_prerequisites
            deploy_kubernetes
            ;;
        "all")
            check_prerequisites
            init_terraform
            validate_terraform
            plan_terraform
            apply_terraform
            show_outputs
            deploy_kubernetes
            show_status
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            echo -e "${RED}❌ Unknown action: $ACTION${NC}"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
