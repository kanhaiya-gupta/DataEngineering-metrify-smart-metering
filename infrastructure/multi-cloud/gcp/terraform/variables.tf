# GCP Terraform Variables

variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "europe-west3"
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

variable "create_project" {
  description = "Whether to create a new project"
  type        = bool
  default     = false
}

variable "billing_account_id" {
  description = "Billing account ID"
  type        = string
  default     = ""
}

variable "organization_id" {
  description = "Organization ID"
  type        = string
  default     = ""
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "pods_cidr_ranges" {
  description = "CIDR ranges for pods"
  type        = list(string)
  default     = ["10.1.0.0/20", "10.2.0.0/20", "10.3.0.0/20"]
}

variable "services_cidr_ranges" {
  description = "CIDR ranges for services"
  type        = list(string)
  default     = ["10.10.0.0/24", "10.11.0.0/24", "10.12.0.0/24"]
}

# Database Configuration
variable "postgres_machine_type" {
  description = "Machine type for PostgreSQL"
  type        = string
  default     = "db-f1-micro"
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
variable "redis_tier" {
  description = "Tier for Redis instance"
  type        = string
  default     = "BASIC"
}

variable "redis_memory_size_gb" {
  description = "Memory size for Redis instance"
  type        = number
  default     = 1
}

variable "redis_reserved_ip_range" {
  description = "Reserved IP range for Redis"
  type        = string
  default     = "10.0.4.0/24"
}

# GKE Configuration
variable "gke_node_machine_type" {
  description = "Machine type for GKE nodes"
  type        = string
  default     = "e2-medium"
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

variable "enable_point_in_time_recovery" {
  description = "Enable point-in-time recovery"
  type        = bool
  default     = true
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
  default     = "europe-west3"
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
variable "enable_workload_identity" {
  description = "Enable workload identity"
  type        = bool
  default     = true
}

variable "enable_rbac" {
  description = "Enable role-based access control"
  type        = bool
  default     = true
}

# Load Balancer
variable "domain_name" {
  description = "Domain name for SSL certificate"
  type        = string
  default     = "smartmeter.example.com"
}

# Storage
variable "storage_class" {
  description = "Storage class for buckets"
  type        = string
  default     = "STANDARD"
}

variable "enable_versioning" {
  description = "Enable versioning for storage buckets"
  type        = bool
  default     = true
}

# Security
variable "enable_binary_authorization" {
  description = "Enable binary authorization"
  type        = bool
  default     = true
}

variable "enable_pod_security_policy" {
  description = "Enable pod security policy"
  type        = bool
  default     = true
}

variable "enable_network_policy" {
  description = "Enable network policy"
  type        = bool
  default     = true
}

# Monitoring
variable "enable_cloud_monitoring" {
  description = "Enable Cloud Monitoring"
  type        = bool
  default     = true
}

variable "enable_cloud_logging" {
  description = "Enable Cloud Logging"
  type        = bool
  default     = true
}

# Secret Management
variable "enable_secret_manager" {
  description = "Enable Secret Manager"
  type        = bool
  default     = true
}

# KMS
variable "enable_kms" {
  description = "Enable Cloud KMS"
  type        = bool
  default     = true
}

variable "kms_key_rotation_period" {
  description = "KMS key rotation period in seconds"
  type        = string
  default     = "7776000s" # 90 days
}
