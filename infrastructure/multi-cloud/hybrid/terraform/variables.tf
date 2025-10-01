# Hybrid Cloud Terraform Variables

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

# AWS Configuration
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "aws_vpc_cidr" {
  description = "CIDR block for AWS VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "aws_private_subnets" {
  description = "CIDR blocks for AWS private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

# Azure Configuration
variable "azure_region" {
  description = "Azure region"
  type        = string
  default     = "West US 2"
}

variable "azure_vnet_cidr" {
  description = "CIDR block for Azure VNet"
  type        = string
  default     = "10.1.0.0/16"
}

variable "azure_private_subnets" {
  description = "CIDR blocks for Azure private subnets"
  type        = list(string)
  default     = ["10.1.1.0/24", "10.1.2.0/24", "10.1.3.0/24"]
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

variable "gcp_vpc_cidr" {
  description = "CIDR block for GCP VPC"
  type        = string
  default     = "10.2.0.0/16"
}

variable "gcp_private_subnets" {
  description = "CIDR blocks for GCP private subnets"
  type        = list(string)
  default     = ["10.2.1.0/24", "10.2.2.0/24", "10.2.3.0/24"]
}

# Database Configuration
variable "postgres_instance_class" {
  description = "RDS instance class for PostgreSQL"
  type        = string
  default     = "db.t3.micro"
}

variable "postgres_sku_name" {
  description = "SKU name for Azure PostgreSQL"
  type        = string
  default     = "GP_Standard_D2s_v3"
}

variable "postgres_database_name" {
  description = "Name of the PostgreSQL database"
  type        = string
  default     = "smartmeterdb"
}

variable "postgres_username" {
  description = "Username for PostgreSQL"
  type        = string
  default     = "postgres"
}

variable "postgres_password" {
  description = "Password for PostgreSQL"
  type        = string
  sensitive   = true
}

# Cross-Cloud Features
variable "enable_cross_cloud_replication" {
  description = "Enable cross-cloud data replication"
  type        = bool
  default     = false
}

variable "enable_database_replication" {
  description = "Enable cross-cloud database replication"
  type        = bool
  default     = false
}

variable "enable_message_queues" {
  description = "Enable cross-cloud message queues"
  type        = bool
  default     = true
}

variable "enable_cross_cloud_monitoring" {
  description = "Enable cross-cloud monitoring"
  type        = bool
  default     = true
}

variable "enable_cross_cloud_lb" {
  description = "Enable cross-cloud load balancing"
  type        = bool
  default     = false
}

variable "enable_cross_cloud_backup" {
  description = "Enable cross-cloud backup"
  type        = bool
  default     = false
}

# Cost Optimization
variable "enable_cost_anomaly_detection" {
  description = "Enable cost anomaly detection"
  type        = bool
  default     = true
}

variable "cost_anomaly_threshold" {
  description = "Cost anomaly detection threshold in USD"
  type        = number
  default     = 100
}

variable "budget_limit" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 1000
}

# Monitoring
variable "log_retention_days" {
  description = "Number of days to retain logs"
  type        = number
  default     = 30
}

variable "admin_email" {
  description = "Admin email for alerts"
  type        = string
  default     = "admin@example.com"
}

# Security
variable "enable_encryption" {
  description = "Enable encryption at rest"
  type        = bool
  default     = true
}

variable "enable_network_encryption" {
  description = "Enable encryption in transit"
  type        = bool
  default     = true
}

# Backup
variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 7
}

variable "enable_geo_redundant_backup" {
  description = "Enable geo-redundant backup"
  type        = bool
  default     = false
}

# High Availability
variable "enable_high_availability" {
  description = "Enable high availability"
  type        = bool
  default     = false
}

variable "enable_zone_redundancy" {
  description = "Enable zone redundancy"
  type        = bool
  default     = false
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "enable_auto_scaling" {
  description = "Enable auto scaling"
  type        = bool
  default     = true
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

variable "enable_data_sovereignty" {
  description = "Enable data sovereignty controls"
  type        = bool
  default     = false
}

# Resource Tags
variable "additional_tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Performance
variable "enable_performance_insights" {
  description = "Enable performance insights"
  type        = bool
  default     = true
}

variable "enable_enhanced_monitoring" {
  description = "Enable enhanced monitoring"
  type        = bool
  default     = true
}

# Disaster Recovery
variable "enable_cross_region_replication" {
  description = "Enable cross-region replication"
  type        = bool
  default     = false
}

variable "enable_geo_redundant_storage" {
  description = "Enable geo-redundant storage"
  type        = bool
  default     = false
}

# Networking
variable "enable_private_endpoints" {
  description = "Enable private endpoints for services"
  type        = bool
  default     = false
}

variable "enable_service_endpoints" {
  description = "Enable service endpoints"
  type        = bool
  default     = true
}

# Identity and Access Management
variable "enable_managed_identity" {
  description = "Enable managed identity"
  type        = bool
  default     = true
}

variable "enable_rbac" {
  description = "Enable role-based access control"
  type        = bool
  default     = true
}

# Kubernetes Configuration
variable "kubeconfig_path" {
  description = "Path to kubeconfig file"
  type        = string
  default     = "~/.kube/config"
}

# Edge Computing
variable "enable_edge_computing" {
  description = "Enable edge computing capabilities"
  type        = bool
  default     = false
}

variable "edge_locations" {
  description = "List of edge computing locations"
  type        = list(string)
  default     = []
}

# Data Sovereignty
variable "data_sovereignty_regions" {
  description = "List of regions for data sovereignty"
  type        = list(string)
  default     = ["us-west-2", "West US 2", "us-west1"]
}

# Multi-Cloud Load Balancing
variable "load_balancing_strategy" {
  description = "Load balancing strategy across clouds"
  type        = string
  default     = "round_robin"
  validation {
    condition     = contains(["round_robin", "least_connections", "geographic", "cost_optimized"], var.load_balancing_strategy)
    error_message = "Load balancing strategy must be one of: round_robin, least_connections, geographic, cost_optimized."
  }
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

# Disaster Recovery
variable "disaster_recovery_strategy" {
  description = "Disaster recovery strategy"
  type        = string
  default     = "active_passive"
  validation {
    condition     = contains(["active_passive", "active_active", "pilot_light"], var.disaster_recovery_strategy)
    error_message = "Disaster recovery strategy must be one of: active_passive, active_active, pilot_light."
  }
}

# Data Migration
variable "enable_data_migration" {
  description = "Enable data migration capabilities"
  type        = bool
  default     = false
}

variable "migration_strategy" {
  description = "Data migration strategy"
  type        = string
  default     = "lift_and_shift"
  validation {
    condition     = contains(["lift_and_shift", "replatform", "refactor"], var.migration_strategy)
    error_message = "Migration strategy must be one of: lift_and_shift, replatform, refactor."
  }
}
