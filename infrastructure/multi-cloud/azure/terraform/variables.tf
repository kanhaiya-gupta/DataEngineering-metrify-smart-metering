# Azure Terraform Variables

variable "azure_region" {
  description = "Azure region"
  type        = string
  default     = "Germany West Central"
}

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

# Network Configuration
variable "vnet_address_space" {
  description = "Address space for virtual network"
  type        = string
  default     = "10.1.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.1.1.0/24", "10.1.2.0/24", "10.1.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.1.101.0/24", "10.1.102.0/24", "10.1.103.0/24"]
}

# Database Configuration
variable "postgres_sku_name" {
  description = "SKU name for PostgreSQL server"
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

# Redis Configuration
variable "redis_capacity" {
  description = "Capacity for Redis cache"
  type        = number
  default     = 1
}

variable "redis_family" {
  description = "Family for Redis cache"
  type        = string
  default     = "C"
}

variable "redis_sku_name" {
  description = "SKU name for Redis cache"
  type        = string
  default     = "Standard"
}

# Event Hub Configuration
variable "eventhub_sku" {
  description = "SKU for Event Hub namespace"
  type        = string
  default     = "Standard"
}

variable "eventhub_capacity" {
  description = "Capacity for Event Hub namespace"
  type        = number
  default     = 1
}

# AKS Configuration
variable "aks_node_vm_size" {
  description = "VM size for AKS nodes"
  type        = string
  default     = "Standard_D2s_v3"
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
  default     = "Germany West Central"
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

# Cost Management
variable "budget_limit" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 1000
}

variable "enable_cost_anomaly_detection" {
  description = "Enable cost anomaly detection"
  type        = bool
  default     = true
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

# Application Gateway
variable "appgw_sku_name" {
  description = "SKU name for Application Gateway"
  type        = string
  default     = "Standard_v2"
}

variable "appgw_tier" {
  description = "Tier for Application Gateway"
  type        = string
  default     = "Standard_v2"
}

variable "appgw_capacity" {
  description = "Capacity for Application Gateway"
  type        = number
  default     = 2
}

# Storage
variable "storage_account_tier" {
  description = "Tier for storage accounts"
  type        = string
  default     = "Standard"
}

variable "storage_replication_type" {
  description = "Replication type for storage accounts"
  type        = string
  default     = "LRS"
}

variable "enable_hierarchical_namespace" {
  description = "Enable hierarchical namespace for storage"
  type        = bool
  default     = true
}
