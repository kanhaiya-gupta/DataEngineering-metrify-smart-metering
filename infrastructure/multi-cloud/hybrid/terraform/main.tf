# Hybrid Cloud Infrastructure
# Terraform configuration for hybrid cloud deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Cloud       = "aws"
    }
  }
}

# Azure Provider
provider "azurerm" {
  features {}
}

# GCP Provider
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Kubernetes Provider (for cross-cloud management)
provider "kubernetes" {
  config_path = var.kubeconfig_path
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "azurerm_client_config" "current" {}
data "google_client_config" "current" {}

# Cross-Cloud VPC Peering
# AWS VPC
resource "aws_vpc" "main" {
  cidr_block           = var.aws_vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-aws-vpc"
  }
}

resource "aws_subnet" "private" {
  count = length(var.aws_private_subnets)

  vpc_id            = aws_vpc.main.id
  cidr_block        = var.aws_private_subnets[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "${var.project_name}-aws-private-${count.index + 1}"
  }
}

# Azure VNet
resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-rg"
  location = var.azure_region

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Cloud       = "azure"
  }
}

resource "azurerm_virtual_network" "main" {
  name                = "${var.project_name}-vnet"
  address_space       = [var.azure_vnet_cidr]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  tags = {
    Name = "${var.project_name}-azure-vnet"
  }
}

resource "azurerm_subnet" "private" {
  count = length(var.azure_private_subnets)

  name                 = "${var.project_name}-azure-private-${count.index + 1}"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.azure_private_subnets[count.index]]
}

# GCP VPC
resource "google_compute_network" "main" {
  name                    = "${var.project_name}-vpc"
  auto_create_subnetworks = false
  mtu                     = 1460
}

resource "google_compute_subnetwork" "private" {
  count = length(var.gcp_private_subnets)

  name          = "${var.project_name}-gcp-private-${count.index + 1}"
  ip_cidr_range = var.gcp_private_subnets[count.index]
  region        = var.gcp_region
  network       = google_compute_network.main.id

  private_ip_google_access = true
}

# Cross-Cloud Storage Replication
# AWS S3
resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.project_name}-data-lake-${random_string.bucket_suffix.result}"

  tags = {
    Name = "${var.project_name}-data-lake"
  }
}

resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Azure Storage
resource "azurerm_storage_account" "data_lake" {
  name                     = "${var.project_name}dl${random_string.storage_suffix.result}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"

  is_hns_enabled = true

  tags = {
    Name = "${var.project_name}-data-lake"
  }
}

# GCP Storage
resource "google_storage_bucket" "data_lake" {
  name          = "${var.project_name}-data-lake-${random_string.bucket_suffix.result}"
  location      = var.gcp_region
  force_destroy = var.environment != "prod"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }
}

# Cross-Cloud Data Replication
resource "aws_s3_bucket_replication_configuration" "data_lake" {
  count = var.enable_cross_cloud_replication ? 1 : 0

  bucket = aws_s3_bucket.data_lake.id
  role   = aws_iam_role.replication[0].arn

  rule {
    id     = "replicate-to-azure"
    status = "Enabled"

    destination {
      bucket        = azurerm_storage_account.data_lake.primary_blob_endpoint
      storage_class = "STANDARD"
    }
  }

  rule {
    id     = "replicate-to-gcp"
    status = "Enabled"

    destination {
      bucket        = google_storage_bucket.data_lake.name
      storage_class = "STANDARD"
    }
  }
}

# IAM Role for Replication
resource "aws_iam_role" "replication" {
  count = var.enable_cross_cloud_replication ? 1 : 0

  name = "${var.project_name}-replication-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })
}

# Cross-Cloud Database Replication
# Primary Database (AWS RDS)
resource "aws_db_instance" "postgres_primary" {
  identifier = "${var.project_name}-postgres-primary"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.postgres_instance_class

  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = var.postgres_database_name
  username = var.postgres_username
  password = var.postgres_password

  vpc_security_group_ids = [aws_security_group.postgres.id]
  db_subnet_group_name   = aws_db_subnet_group.postgres.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = var.environment == "dev"
  deletion_protection = var.environment == "prod"

  tags = {
    Name = "${var.project_name}-postgres-primary"
  }
}

# Read Replica (Azure)
resource "azurerm_postgresql_flexible_server" "postgres_replica" {
  count = var.enable_database_replication ? 1 : 0

  name                   = "${var.project_name}-postgres-replica"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "15"
  administrator_login    = var.postgres_username
  administrator_password = var.postgres_password
  zone                   = "1"

  storage_mb = 32768
  sku_name   = var.postgres_sku_name

  backup_retention_days        = 7
  geo_redundant_backup_enabled = false

  tags = {
    Name = "${var.project_name}-postgres-replica"
  }
}

# Cross-Cloud Message Queues
# AWS SQS
resource "aws_sqs_queue" "data_events" {
  name = "${var.project_name}-data-events"

  visibility_timeout_seconds = 300
  message_retention_seconds  = 1209600 # 14 days
  max_message_size          = 262144   # 256 KB
  delay_seconds             = 0
  receive_wait_time_seconds = 0

  tags = {
    Name = "${var.project_name}-data-events"
  }
}

# Azure Service Bus
resource "azurerm_servicebus_namespace" "main" {
  count = var.enable_message_queues ? 1 : 0

  name                = "${var.project_name}-servicebus"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "Standard"
  capacity            = 1

  tags = {
    Name = "${var.project_name}-servicebus"
  }
}

# GCP Pub/Sub
resource "google_pubsub_topic" "data_events" {
  count = var.enable_message_queues ? 1 : 0

  name = "data-events"

  message_retention_duration = "604800s" # 7 days
}

# Cross-Cloud Monitoring
# AWS CloudWatch
resource "aws_cloudwatch_log_group" "cross_cloud" {
  name              = "/aws/cross-cloud/${var.project_name}"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "${var.project_name}-cross-cloud-logs"
  }
}

# Azure Monitor
resource "azurerm_log_analytics_workspace" "cross_cloud" {
  count = var.enable_cross_cloud_monitoring ? 1 : 0

  name                = "${var.project_name}-cross-cloud-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = var.log_retention_days

  tags = {
    Name = "${var.project_name}-cross-cloud-logs"
  }
}

# GCP Cloud Logging
resource "google_logging_project_sink" "cross_cloud" {
  count = var.enable_cross_cloud_monitoring ? 1 : 0

  name        = "${var.project_name}-cross-cloud-logs"
  destination = "storage.googleapis.com/${google_storage_bucket.logs[0].name}"

  filter = "resource.type=\"gke_cluster\" OR resource.type=\"gce_instance\""

  unique_writer_identity = true
}

resource "google_storage_bucket" "logs" {
  count = var.enable_cross_cloud_monitoring ? 1 : 0

  name          = "${var.project_name}-cross-cloud-logs-${random_string.bucket_suffix.result}"
  location      = var.gcp_region
  force_destroy = var.environment != "prod"

  lifecycle_rule {
    condition {
      age = var.log_retention_days
    }
    action {
      type = "Delete"
    }
  }
}

# Cross-Cloud Load Balancing
# AWS ALB
resource "aws_lb" "cross_cloud" {
  name               = "${var.project_name}-cross-cloud-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.private[*].id

  enable_deletion_protection = var.environment == "prod"

  tags = {
    Name = "${var.project_name}-cross-cloud-alb"
  }
}

# Azure Application Gateway
resource "azurerm_public_ip" "appgw" {
  count = var.enable_cross_cloud_lb ? 1 : 0

  name                = "${var.project_name}-cross-cloud-appgw-pip"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = {
    Name = "${var.project_name}-cross-cloud-appgw-pip"
  }
}

# GCP Load Balancer
resource "google_compute_global_address" "cross_cloud" {
  count = var.enable_cross_cloud_lb ? 1 : 0

  name = "${var.project_name}-cross-cloud-lb-ip"
}

# Cross-Cloud Security
# AWS Security Group
resource "aws_security_group" "postgres" {
  name_prefix = "${var.project_name}-postgres"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.aws_vpc_cidr, var.azure_vnet_cidr, var.gcp_vpc_cidr]
    description = "PostgreSQL"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = {
    Name = "${var.project_name}-postgres-sg"
  }
}

resource "aws_security_group" "alb" {
  name_prefix = "${var.project_name}-alb"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP"
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = {
    Name = "${var.project_name}-alb-sg"
  }
}

# DB Subnet Group
resource "aws_db_subnet_group" "postgres" {
  name       = "${var.project_name}-postgres-subnet-group"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name = "${var.project_name}-postgres-subnet-group"
  }
}

# Cost Optimization
# AWS Cost Anomaly Detection
resource "aws_ce_anomaly_detector" "main" {
  count = var.enable_cost_anomaly_detection ? 1 : 0

  name = "${var.project_name}-cost-anomaly-detector"

  specification = jsonencode({
    threshold = {
      threshold_value = var.cost_anomaly_threshold
      threshold_type  = "ABSOLUTE"
    }
  })

  monitor_arn_lists = [aws_ce_cost_category.main[0].arn]
}

resource "aws_ce_cost_category" "main" {
  count = var.enable_cost_anomaly_detection ? 1 : 0

  name = "${var.project_name}-cost-category"

  rule {
    value = "Smart Meter Platform"
    rule {
      dimension {
        key           = "SERVICE"
        values        = ["Amazon Elastic Compute Cloud - Compute", "Amazon Relational Database Service"]
        match_options = ["EQUALS"]
      }
    }
  }
}

# Cross-Cloud Backup
# AWS Backup
resource "aws_backup_vault" "main" {
  count = var.enable_cross_cloud_backup ? 1 : 0

  name        = "${var.project_name}-backup-vault"
  kms_key_arn = aws_kms_key.backup[0].arn

  tags = {
    Name = "${var.project_name}-backup-vault"
  }
}

resource "aws_kms_key" "backup" {
  count = var.enable_cross_cloud_backup ? 1 : 0

  description             = "KMS key for cross-cloud backup"
  deletion_window_in_days = 7

  tags = {
    Name = "${var.project_name}-backup-key"
  }
}

# Random strings for unique resource names
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "random_string" "storage_suffix" {
  length  = 8
  special = false
  upper   = false
}

# Outputs
output "aws_vpc_id" {
  description = "ID of the AWS VPC"
  value       = aws_vpc.main.id
}

output "azure_vnet_id" {
  description = "ID of the Azure VNet"
  value       = azurerm_virtual_network.main.id
}

output "gcp_vpc_id" {
  description = "ID of the GCP VPC"
  value       = google_compute_network.main.id
}

output "aws_s3_bucket" {
  description = "Name of the AWS S3 bucket"
  value       = aws_s3_bucket.data_lake.bucket
}

output "azure_storage_account" {
  description = "Name of the Azure storage account"
  value       = azurerm_storage_account.data_lake.name
}

output "gcp_storage_bucket" {
  description = "Name of the GCP storage bucket"
  value       = google_storage_bucket.data_lake.name
}

output "postgres_primary_endpoint" {
  description = "Endpoint of the primary PostgreSQL instance"
  value       = aws_db_instance.postgres_primary.endpoint
}

output "cross_cloud_alb_dns" {
  description = "DNS name of the cross-cloud ALB"
  value       = aws_lb.cross_cloud.dns_name
}
