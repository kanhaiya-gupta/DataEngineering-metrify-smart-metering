# Outputs for Metrify Smart Metering Infrastructure

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

# EKS Outputs
output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_primary_security_group_id" {
  description = "The cluster primary security group ID created by the EKS service"
  value       = module.eks.cluster_primary_security_group_id
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

# RDS Outputs
output "rds_instance_id" {
  description = "RDS instance ID"
  value       = module.rds.db_instance_id
}

output "rds_instance_arn" {
  description = "RDS instance ARN"
  value       = module.rds.db_instance_arn
}

output "rds_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.db_instance_endpoint
}

output "rds_instance_hosted_zone_id" {
  description = "RDS instance hosted zone ID"
  value       = module.rds.db_instance_hosted_zone_id
}

output "rds_instance_name" {
  description = "RDS instance name"
  value       = module.rds.db_instance_name
}

output "rds_instance_username" {
  description = "RDS instance root username"
  value       = module.rds.db_instance_username
  sensitive   = true
}

# Redis Outputs
output "redis_cluster_id" {
  description = "Redis cluster ID"
  value       = module.redis.cluster_id
}

output "redis_cluster_arn" {
  description = "Redis cluster ARN"
  value       = module.redis.cluster_arn
}

output "redis_cluster_endpoint" {
  description = "Redis cluster endpoint"
  value       = module.redis.cluster_endpoint
}

output "redis_cluster_port" {
  description = "Redis cluster port"
  value       = module.redis.cluster_port
}

# MSK Outputs
output "msk_cluster_arn" {
  description = "MSK cluster ARN"
  value       = module.msk.cluster_arn
}

output "msk_cluster_name" {
  description = "MSK cluster name"
  value       = module.msk.cluster_name
}

output "msk_bootstrap_brokers" {
  description = "Plaintext connection host:port pairs"
  value       = module.msk.bootstrap_brokers
  sensitive   = true
}

output "msk_bootstrap_brokers_sasl_scram" {
  description = "SASL/SCRAM connection host:port pairs"
  value       = module.msk.bootstrap_brokers_sasl_scram
  sensitive   = true
}

output "msk_bootstrap_brokers_tls" {
  description = "TLS connection host:port pairs"
  value       = module.msk.bootstrap_brokers_tls
  sensitive   = true
}

# S3 Outputs
output "s3_data_bucket_id" {
  description = "S3 data bucket ID"
  value       = aws_s3_bucket.data.id
}

output "s3_data_bucket_arn" {
  description = "S3 data bucket ARN"
  value       = aws_s3_bucket.data.arn
}

output "s3_analytics_bucket_id" {
  description = "S3 analytics bucket ID"
  value       = aws_s3_bucket.analytics.id
}

output "s3_analytics_bucket_arn" {
  description = "S3 analytics bucket ARN"
  value       = aws_s3_bucket.analytics.arn
}

output "s3_backup_bucket_id" {
  description = "S3 backup bucket ID"
  value       = aws_s3_bucket.backup.id
}

output "s3_backup_bucket_arn" {
  description = "S3 backup bucket ARN"
  value       = aws_s3_bucket.backup.arn
}

# Security Group Outputs
output "rds_security_group_id" {
  description = "RDS security group ID"
  value       = aws_security_group.rds.id
}

output "redis_security_group_id" {
  description = "Redis security group ID"
  value       = aws_security_group.redis.id
}

output "kafka_security_group_id" {
  description = "Kafka security group ID"
  value       = aws_security_group.kafka.id
}

# KMS Outputs
output "msk_kms_key_id" {
  description = "MSK KMS key ID"
  value       = aws_kms_key.msk.key_id
}

output "msk_kms_key_arn" {
  description = "MSK KMS key ARN"
  value       = aws_kms_key.msk.arn
}

output "rds_kms_key_id" {
  description = "RDS KMS key ID"
  value       = aws_kms_key.rds.key_id
}

output "rds_kms_key_arn" {
  description = "RDS KMS key ARN"
  value       = aws_kms_key.rds.arn
}

# Connection Information
output "connection_info" {
  description = "Connection information for the deployed infrastructure"
  value = {
    # EKS
    cluster_name = module.eks.cluster_name
    cluster_endpoint = module.eks.cluster_endpoint
    
    # Database
    database_endpoint = module.rds.db_instance_endpoint
    database_name = module.rds.db_instance_name
    
    # Redis
    redis_endpoint = module.redis.cluster_endpoint
    redis_port = module.redis.cluster_port
    
    # Kafka
    kafka_bootstrap_brokers = module.msk.bootstrap_brokers_tls
    
    # S3
    data_bucket = aws_s3_bucket.data.id
    analytics_bucket = aws_s3_bucket.analytics.id
    backup_bucket = aws_s3_bucket.backup.id
  }
  sensitive = true
}

# Deployment Commands
output "deployment_commands" {
  description = "Commands to deploy the application"
  value = {
    # Update kubeconfig
    update_kubeconfig = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
    
    # Deploy application
    deploy_app = "kubectl apply -f infrastructure/kubernetes/"
    
    # Check status
    check_pods = "kubectl get pods -n metrify"
    check_services = "kubectl get services -n metrify"
    
    # Port forward for testing
    port_forward_api = "kubectl port-forward service/metrify-api 8000:80 -n metrify"
  }
}
