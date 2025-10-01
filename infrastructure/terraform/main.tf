# Metrify Smart Metering Infrastructure
# Main Terraform configuration for AWS resources

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
  
  backend "s3" {
    bucket         = "metrify-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "eu-central-1"
    encrypt        = true
    dynamodb_table = "metrify-terraform-locks"
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "Metrify Smart Metering"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "Platform Team"
    }
  }
}

# Configure Kubernetes Provider
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

# Configure Helm Provider
provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

# Data sources
data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_name
}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  name = "${var.project_name}-${var.environment}"
  
  vpc_cidr = "10.0.0.0/16"
  azs      = slice(data.aws_availability_zones.available.names, 0, 3)
  
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# VPC Module
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  name = local.name
  cidr = local.vpc_cidr
  
  azs             = local.azs
  private_subnets = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 4, k)]
  public_subnets  = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 48)]
  
  enable_nat_gateway     = true
  single_nat_gateway     = false
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true
  
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }
  
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }
  
  tags = local.tags
}

# EKS Module
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"
  
  cluster_name    = local.name
  cluster_version = var.kubernetes_version
  
  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true
  cluster_endpoint_private_access = true
  
  # EKS Managed Node Groups
  eks_managed_node_groups = {
    general = {
      name = "general"
      
      instance_types = ["t3.medium", "t3.large"]
      capacity_type  = "ON_DEMAND"
      
      min_size     = 2
      max_size     = 10
      desired_size = 3
      
      disk_size = 50
      disk_type = "gp3"
      
      labels = {
        role = "general"
      }
    }
    
    workers = {
      name = "workers"
      
      instance_types = ["c5.xlarge", "c5.2xlarge"]
      capacity_type  = "SPOT"
      
      min_size     = 1
      max_size     = 20
      desired_size = 5
      
      disk_size = 100
      disk_type = "gp3"
      
      labels = {
        role = "worker"
      }
      
      taints = [
        {
          key    = "worker"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  # Cluster access entry
  cluster_access_entries = {
    admin = {
      kubernetes_groups = ["system:masters"]
      principal_arn     = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
      policy_associations = {
        admin = {
          policy_arn = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
          access_scope = {
            type = "cluster"
          }
        }
      }
    }
  }
  
  tags = local.tags
}

# RDS PostgreSQL
module "rds" {
  source = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"
  
  identifier = "${local.name}-postgres"
  
  engine               = "postgres"
  engine_version       = "15.4"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = var.database_instance_class
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "metrify_prod"
  username = "metrify_admin"
  password = var.database_password
  port     = 5432
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "${local.name}-postgres-final-snapshot"
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_enhanced_monitoring.arn
  
  tags = local.tags
}

# ElastiCache Redis
module "redis" {
  source = "terraform-aws-modules/elasticache/aws"
  version = "~> 1.0"
  
  cluster_id = "${local.name}-redis"
  
  engine               = "redis"
  node_type            = var.redis_node_type
  num_cache_nodes      = 2
  parameter_group_name = "default.redis7"
  port                 = 6379
  
  subnet_ids = module.vpc.private_subnets
  vpc_id     = module.vpc.vpc_id
  security_group_ids = [aws_security_group.redis.id]
  
  tags = local.tags
}

# MSK Kafka
module "msk" {
  source = "terraform-aws-modules/msk/aws"
  version = "~> 4.0"
  
  cluster_name = local.name
  
  kafka_version = "3.5.1"
  number_of_broker_nodes = 3
  
  broker_node_group_info = {
    instance_type   = var.kafka_instance_type
    ebs_volume_size = 100
    client_subnets  = module.vpc.private_subnets
    security_groups = [aws_security_group.kafka.id]
  }
  
  configuration_info = {
    arn  = aws_msk_configuration.main.arn
    revision = aws_msk_configuration.main.latest_revision
  }
  
  client_authentication = {
    sasl = {
      scram = true
    }
  }
  
  encryption_info = {
    encryption_at_rest_kms_key_id = aws_kms_key.msk.arn
    encryption_in_transit = {
      client_broker = "TLS"
      in_cluster    = true
    }
  }
  
  tags = local.tags
}

# S3 Buckets
resource "aws_s3_bucket" "data" {
  bucket = "${local.name}-data"
  
  tags = local.tags
}

resource "aws_s3_bucket" "analytics" {
  bucket = "${local.name}-analytics"
  
  tags = local.tags
}

resource "aws_s3_bucket" "backup" {
  bucket = "${local.name}-backup"
  
  tags = local.tags
}

# S3 Bucket Configurations
resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "data" {
  bucket = aws_s3_bucket.data.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  
  rule {
    id     = "smart_meter_data"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }
    
    expiration {
      days = 2555  # 7 years
    }
  }
}

# KMS Keys
resource "aws_kms_key" "msk" {
  description             = "KMS key for MSK encryption"
  deletion_window_in_days = 7
  
  tags = local.tags
}

resource "aws_kms_key" "rds" {
  description             = "KMS key for RDS encryption"
  deletion_window_in_days = 7
  
  tags = local.tags
}

# Security Groups
resource "aws_security_group" "rds" {
  name_prefix = "${local.name}-rds-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.tags, {
    Name = "${local.name}-rds"
  })
}

resource "aws_security_group" "redis" {
  name_prefix = "${local.name}-redis-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.tags, {
    Name = "${local.name}-redis"
  })
}

resource "aws_security_group" "kafka" {
  name_prefix = "${local.name}-kafka-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 9092
    to_port     = 9092
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  ingress {
    from_port   = 9094
    to_port     = 9094
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.tags, {
    Name = "${local.name}-kafka"
  })
}

# Database Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "${local.name}-db-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = merge(local.tags, {
    Name = "${local.name}-db-subnet-group"
  })
}

# MSK Configuration
resource "aws_msk_configuration" "main" {
  kafka_versions = ["3.5.1"]
  name           = "${local.name}-config"
  
  server_properties = <<PROPERTIES
auto.create.topics.enable=true
default.replication.factor=3
min.insync.replicas=2
num.partitions=12
log.retention.hours=168
compression.type=snappy
PROPERTIES
}

# RDS Enhanced Monitoring Role
resource "aws_iam_role" "rds_enhanced_monitoring" {
  name = "${local.name}-rds-enhanced-monitoring"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  role       = aws_iam_role.rds_enhanced_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}
