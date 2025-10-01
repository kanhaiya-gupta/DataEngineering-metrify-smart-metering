# Cost Optimization Terraform Variables

# Project Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "smart-meter-platform"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

# Budget Configuration
variable "budget_limit" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 1000
}

variable "cost_anomaly_threshold" {
  description = "Cost anomaly detection threshold in USD"
  type        = number
  default     = 100
}

variable "admin_email" {
  description = "Admin email for cost alerts"
  type        = string
  default     = "admin@example.com"
}

# AWS Configuration
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "aws_availability_zone" {
  description = "AWS availability zone"
  type        = string
  default     = "us-west-2a"
}

variable "aws_subnet_ids" {
  description = "List of AWS subnet IDs"
  type        = list(string)
  default     = []
}

variable "aws_target_group_arns" {
  description = "List of AWS target group ARNs"
  type        = list(string)
  default     = []
}

variable "aws_security_group_ids" {
  description = "List of AWS security group IDs"
  type        = list(string)
  default     = []
}

# AWS Reserved Instances
variable "enable_reserved_instances" {
  description = "Enable AWS reserved instances"
  type        = bool
  default     = false
}

variable "reserved_instance_type" {
  description = "AWS reserved instance type"
  type        = string
  default     = "t3.micro"
}

variable "reserved_instance_count" {
  description = "Number of AWS reserved instances"
  type        = number
  default     = 1
}

# AWS Spot Instances
variable "enable_spot_instances" {
  description = "Enable AWS spot instances"
  type        = bool
  default     = false
}

variable "spot_instance_count" {
  description = "Number of AWS spot instances"
  type        = number
  default     = 1
}

variable "spot_instance_ami" {
  description = "AMI for AWS spot instances"
  type        = string
  default     = "ami-0c02fb55956c7d316"
}

variable "spot_instance_type" {
  description = "Instance type for AWS spot instances"
  type        = string
  default     = "t3.micro"
}

variable "spot_instance_price" {
  description = "Maximum price for AWS spot instances"
  type        = string
  default     = "0.01"
}

# AWS Auto Scaling
variable "enable_auto_scaling" {
  description = "Enable AWS auto scaling"
  type        = bool
  default     = true
}

variable "min_capacity" {
  description = "Minimum capacity for auto scaling"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum capacity for auto scaling"
  type        = number
  default     = 10
}

variable "desired_capacity" {
  description = "Desired capacity for auto scaling"
  type        = number
  default     = 3
}

variable "launch_template_ami" {
  description = "AMI for launch template"
  type        = string
  default     = "ami-0c02fb55956c7d316"
}

variable "launch_template_instance_type" {
  description = "Instance type for launch template"
  type        = string
  default     = "t3.micro"
}

# Azure Configuration
variable "azure_region" {
  description = "Azure region"
  type        = string
  default     = "West US 2"
}

variable "azure_resource_group_name" {
  description = "Azure resource group name"
  type        = string
  default     = "smart-meter-platform-rg"
}

# Azure Reserved Instances
variable "enable_azure_reserved_instances" {
  description = "Enable Azure reserved instances"
  type        = bool
  default     = false
}

variable "azure_reserved_instance_sku" {
  description = "Azure reserved instance SKU"
  type        = string
  default     = "Standard_D2s_v3"
}

variable "azure_reserved_instance_quantity" {
  description = "Number of Azure reserved instances"
  type        = number
  default     = 1
}

# Azure Spot Instances
variable "enable_azure_spot_instances" {
  description = "Enable Azure spot instances"
  type        = bool
  default     = false
}

variable "azure_spot_instance_count" {
  description = "Number of Azure spot instances"
  type        = number
  default     = 1
}

variable "azure_spot_instance_size" {
  description = "Size for Azure spot instances"
  type        = string
  default     = "Standard_D2s_v3"
}

variable "azure_ssh_public_key" {
  description = "SSH public key for Azure instances"
  type        = string
  default     = ""
}

# Azure Auto Scaling
variable "enable_azure_auto_scaling" {
  description = "Enable Azure auto scaling"
  type        = bool
  default     = true
}

variable "azure_vmss_id" {
  description = "Azure VMSS ID for auto scaling"
  type        = string
  default     = ""
}

variable "azure_min_capacity" {
  description = "Minimum capacity for Azure auto scaling"
  type        = number
  default     = 1
}

variable "azure_max_capacity" {
  description = "Maximum capacity for Azure auto scaling"
  type        = number
  default     = 10
}

# GCP Configuration
variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-west1"
}

variable "gcp_zone" {
  description = "GCP zone"
  type        = string
  default     = "us-west1-a"
}

variable "gcp_billing_account_id" {
  description = "GCP billing account ID"
  type        = string
}

variable "gcp_network_name" {
  description = "GCP network name"
  type        = string
  default     = "default"
}

# GCP Committed Use Discounts
variable "enable_gcp_committed_use" {
  description = "Enable GCP committed use discounts"
  type        = bool
  default     = false
}

variable "gcp_commitment_vcpu" {
  description = "Number of vCPUs for GCP commitment"
  type        = number
  default     = 4
}

# GCP Preemptible Instances
variable "enable_gcp_preemptible_instances" {
  description = "Enable GCP preemptible instances"
  type        = bool
  default     = false
}

variable "gcp_preemptible_count" {
  description = "Number of GCP preemptible instances"
  type        = number
  default     = 1
}

variable "gcp_preemptible_machine_type" {
  description = "Machine type for GCP preemptible instances"
  type        = string
  default     = "e2-medium"
}

variable "gcp_preemptible_image" {
  description = "Image for GCP preemptible instances"
  type        = string
  default     = "ubuntu-os-cloud/ubuntu-2004-lts"
}

# GCP Auto Scaling
variable "enable_gcp_auto_scaling" {
  description = "Enable GCP auto scaling"
  type        = bool
  default     = true
}

variable "gcp_instance_group_manager_id" {
  description = "GCP instance group manager ID"
  type        = string
  default     = ""
}

variable "gcp_min_replicas" {
  description = "Minimum replicas for GCP auto scaling"
  type        = number
  default     = 1
}

variable "gcp_max_replicas" {
  description = "Maximum replicas for GCP auto scaling"
  type        = number
  default     = 10
}

# Cost Optimization Strategy
variable "cost_optimization_strategy" {
  description = "Cost optimization strategy"
  type        = string
  default     = "balanced"
  validation {
    condition     = contains(["performance", "balanced", "cost"], var.cost_optimization_strategy)
    error_message = "Cost optimization strategy must be one of: performance, balanced, cost."
  }
}

# Resource Tags
variable "additional_tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Monitoring
variable "enable_cost_monitoring" {
  description = "Enable cost monitoring"
  type        = bool
  default     = true
}

variable "cost_alert_threshold_percent" {
  description = "Cost alert threshold percentage"
  type        = number
  default     = 80
}

# Compliance
variable "compliance_framework" {
  description = "Compliance framework to follow"
  type        = string
  default     = "SOC2"
  validation {
    condition     = contains(["SOC2", "HIPAA", "PCI-DSS", "GDPR"], var.compliance_framework)
    error_message = "Compliance framework must be one of: SOC2, HIPAA, PCI-DSS, GDPR."
  }
}

# Data Residency
variable "data_residency_region" {
  description = "Primary data residency region"
  type        = string
  default     = "us-west-2"
}

# Performance
variable "enable_performance_optimization" {
  description = "Enable performance optimization"
  type        = bool
  default     = true
}

# Disaster Recovery
variable "enable_cost_aware_disaster_recovery" {
  description = "Enable cost-aware disaster recovery"
  type        = bool
  default     = false
}

# Multi-Cloud Cost Analysis
variable "enable_cross_cloud_cost_analysis" {
  description = "Enable cross-cloud cost analysis"
  type        = bool
  default     = true
}

# Cost Allocation
variable "enable_cost_allocation" {
  description = "Enable cost allocation"
  type        = bool
  default     = true
}

variable "cost_allocation_tags" {
  description = "Tags for cost allocation"
  type        = list(string)
  default     = ["Project", "Environment", "Service", "Team"]
}
