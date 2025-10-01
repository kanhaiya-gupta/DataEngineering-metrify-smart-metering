# GCP Multi-Cloud Infrastructure
# Terraform configuration for GCP deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Data sources
data "google_client_config" "current" {}

# Project
resource "google_project" "main" {
  count           = var.create_project ? 1 : 0
  name            = var.project_name
  project_id      = var.gcp_project_id
  billing_account = var.billing_account_id
  org_id          = var.organization_id

  labels = {
    project     = var.project_name
    environment = var.environment
    managed_by  = "terraform"
  }
}

# Enable APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "sqladmin.googleapis.com",
    "redis.googleapis.com",
    "pubsub.googleapis.com",
    "storage.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "cloudkms.googleapis.com",
    "secretmanager.googleapis.com"
  ])

  service = each.value
  project = var.gcp_project_id

  disable_dependent_services = false
  disable_on_destroy        = false
}

# VPC Network
resource "google_compute_network" "main" {
  name                    = "${var.project_name}-vpc"
  auto_create_subnetworks = false
  mtu                     = 1460

  depends_on = [google_project_service.apis]
}

# Subnets
resource "google_compute_subnetwork" "private" {
  count = length(var.private_subnet_cidrs)

  name          = "${var.project_name}-private-subnet-${count.index + 1}"
  ip_cidr_range = var.private_subnet_cidrs[count.index]
  region        = var.gcp_region
  network       = google_compute_network.main.id

  private_ip_google_access = true

  secondary_range {
    range_name    = "pods-${count.index + 1}"
    ip_cidr_range = var.pods_cidr_ranges[count.index]
  }

  secondary_range {
    range_name    = "services-${count.index + 1}"
    ip_cidr_range = var.services_cidr_ranges[count.index]
  }
}

resource "google_compute_subnetwork" "public" {
  count = length(var.public_subnet_cidrs)

  name          = "${var.project_name}-public-subnet-${count.index + 1}"
  ip_cidr_range = var.public_subnet_cidrs[count.index]
  region        = var.gcp_region
  network       = google_compute_network.main.id
}

# Firewall Rules
resource "google_compute_firewall" "data_platform" {
  name    = "${var.project_name}-data-platform"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "9092", "8080"]
  }

  source_ranges = [var.vpc_cidr]
  target_tags   = ["data-platform"]

  depends_on = [google_project_service.apis]
}

resource "google_compute_firewall" "ssh" {
  name    = "${var.project_name}-ssh"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ssh"]
}

# Cloud Storage Buckets
resource "google_storage_bucket" "data_lake" {
  name          = "${var.project_name}-data-lake-${random_string.bucket_suffix.result}"
  location      = var.gcp_region
  force_destroy = var.environment != "prod"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "data_warehouse" {
  name          = "${var.project_name}-data-warehouse-${random_string.bucket_suffix.result}"
  location      = var.gcp_region
  force_destroy = var.environment != "prod"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "backup" {
  name          = "${var.project_name}-backup-${random_string.bucket_suffix.result}"
  location      = var.gcp_region
  force_destroy = var.environment != "prod"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.apis]
}

# Cloud SQL PostgreSQL
resource "google_sql_database_instance" "postgres" {
  name             = "${var.project_name}-postgres"
  database_version = "POSTGRES_15"
  region           = var.gcp_region

  settings {
    tier              = var.postgres_machine_type
    availability_type = var.environment == "prod" ? "REGIONAL" : "ZONAL"

    disk_type = "PD_SSD"
    disk_size = 100
    disk_autoresize = true

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      location                       = var.gcp_region
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 7
        retention_unit   = "COUNT"
      }
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.main.id
      require_ssl     = true
    }

    maintenance_window {
      day          = 7
      hour         = 3
      update_track = "stable"
    }

    database_flags {
      name  = "log_statement"
      value = "all"
    }

    database_flags {
      name  = "log_min_duration_statement"
      value = "1000"
    }
  }

  deletion_protection = var.environment == "prod"

  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "main" {
  name     = var.postgres_database_name
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "main" {
  name     = var.postgres_username
  instance = google_sql_database_instance.postgres.name
  password = var.postgres_password
}

# Memorystore Redis
resource "google_redis_instance" "main" {
  name           = "${var.project_name}-redis"
  tier           = var.redis_tier
  memory_size_gb = var.redis_memory_size_gb
  region         = var.gcp_region

  location_id             = "${var.gcp_region}-a"
  alternative_location_id = "${var.gcp_region}-b"

  redis_version     = "REDIS_7_0"
  display_name      = "Smart Meter Redis"
  reserved_ip_range = var.redis_reserved_ip_range

  auth_enabled = true

  depends_on = [google_project_service.apis]
}

# Pub/Sub Topics
resource "google_pubsub_topic" "smart_meter" {
  name = "smart-meter-events"

  message_retention_duration = "604800s" # 7 days

  depends_on = [google_project_service.apis]
}

resource "google_pubsub_topic" "grid_operator" {
  name = "grid-operator-events"

  message_retention_duration = "604800s" # 7 days

  depends_on = [google_project_service.apis]
}

resource "google_pubsub_topic" "weather" {
  name = "weather-events"

  message_retention_duration = "604800s" # 7 days

  depends_on = [google_project_service.apis]
}

# Pub/Sub Subscriptions
resource "google_pubsub_subscription" "smart_meter_processing" {
  name  = "smart-meter-processing"
  topic = google_pubsub_topic.smart_meter.name

  ack_deadline_seconds = 60

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.smart_meter_dlq.id
    max_delivery_attempts = 5
  }
}

resource "google_pubsub_subscription" "grid_operator_processing" {
  name  = "grid-operator-processing"
  topic = google_pubsub_topic.grid_operator.name

  ack_deadline_seconds = 60

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
}

resource "google_pubsub_subscription" "weather_processing" {
  name  = "weather-processing"
  topic = google_pubsub_topic.weather.name

  ack_deadline_seconds = 60

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
}

# Dead Letter Queue
resource "google_pubsub_topic" "smart_meter_dlq" {
  name = "smart-meter-dlq"

  message_retention_duration = "1209600s" # 14 days

  depends_on = [google_project_service.apis]
}

# GKE Cluster
resource "google_container_cluster" "main" {
  name     = "${var.project_name}-gke"
  location = var.gcp_region

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.main.name
  subnetwork = google_compute_subnetwork.private[0].name

  # Enable private nodes
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  # Enable IP aliasing
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods-1"
    services_secondary_range_name = "services-1"
  }

  # Enable network policy
  network_policy {
    enabled = true
  }

  # Enable workload identity
  workload_identity_config {
    workload_pool = "${var.gcp_project_id}.svc.id.goog"
  }

  # Enable binary authorization
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  # Enable pod security policy
  pod_security_policy_config {
    enabled = true
  }

  # Enable master authorized networks
  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "All"
    }
  }

  # Enable legacy authorization
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }

  depends_on = [google_project_service.apis]
}

# Node Pool
resource "google_container_node_pool" "data_platform" {
  name       = "data-platform"
  location   = var.gcp_region
  cluster    = google_container_cluster.main.name
  node_count = 3

  node_config {
    preemptible  = var.enable_spot_instances
    machine_type = var.gke_node_machine_type
    disk_size_gb = 50
    disk_type    = "pd-ssd"

    # Enable workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # Enable secure boot
    shielded_instance_config {
      enable_secure_boot = true
    }

    # Enable integrity monitoring
    shielded_instance_config {
      enable_integrity_monitoring = true
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      workload    = "data-platform"
    }

    tags = ["data-platform"]
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 10
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Cloud Load Balancer
resource "google_compute_global_address" "main" {
  name = "${var.project_name}-lb-ip"
}

resource "google_compute_global_forwarding_rule" "main" {
  name       = "${var.project_name}-lb-rule"
  target     = google_compute_target_https_proxy.main.id
  port_range = "443"
  ip_address = google_compute_global_address.main.address
}

resource "google_compute_target_https_proxy" "main" {
  name             = "${var.project_name}-https-proxy"
  url_map          = google_compute_url_map.main.id
  ssl_certificates = [google_compute_managed_ssl_certificate.main.id]
}

resource "google_compute_url_map" "main" {
  name            = "${var.project_name}-url-map"
  default_service = google_compute_backend_service.main.id
}

resource "google_compute_backend_service" "main" {
  name        = "${var.project_name}-backend"
  protocol    = "HTTP"
  port_name   = "http"
  timeout_sec = 10

  backend {
    group = google_container_cluster.main.node_pool[0].instance_group_urls[0]
  }

  health_checks = [google_compute_health_check.main.id]
}

resource "google_compute_health_check" "main" {
  name               = "${var.project_name}-health-check"
  check_interval_sec = 5
  timeout_sec        = 5
  healthy_threshold  = 2
  unhealthy_threshold = 3

  http_health_check {
    request_path = "/health"
    port         = "80"
  }
}

resource "google_compute_managed_ssl_certificate" "main" {
  name = "${var.project_name}-ssl-cert"

  managed {
    domains = [var.domain_name]
  }
}

# Cloud Logging
resource "google_logging_project_sink" "main" {
  name        = "${var.project_name}-logs"
  destination = "storage.googleapis.com/${google_storage_bucket.logs.name}"

  filter = "resource.type=\"gke_cluster\" OR resource.type=\"gce_instance\""

  unique_writer_identity = true
}

resource "google_storage_bucket" "logs" {
  name          = "${var.project_name}-logs-${random_string.bucket_suffix.result}"
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

# Cloud Monitoring
resource "google_monitoring_notification_channel" "email" {
  display_name = "Email Notification"
  type         = "email"

  labels = {
    email_address = var.admin_email
  }
}

resource "google_monitoring_alert_policy" "high_cpu" {
  display_name = "High CPU Usage"
  combiner     = "OR"

  conditions {
    display_name = "CPU usage is high"
    condition_threshold {
      filter          = "resource.type=\"gke_container\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.8

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "1800s"
  }
}

# Secret Manager
resource "google_secret_manager_secret" "postgres_password" {
  secret_id = "postgres-password"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "postgres_password" {
  secret      = google_secret_manager_secret.postgres_password.id
  secret_data = var.postgres_password
}

# Cloud KMS
resource "google_kms_key_ring" "main" {
  name     = "${var.project_name}-keyring"
  location = var.gcp_region
}

resource "google_kms_crypto_key" "main" {
  name            = "${var.project_name}-key"
  key_ring        = google_kms_key_ring.main.id
  rotation_period = "7776000s" # 90 days

  version_template {
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }
}

# Random string for unique resource names
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# Outputs
output "project_id" {
  description = "GCP Project ID"
  value       = var.gcp_project_id
}

output "vpc_network" {
  description = "Name of the VPC network"
  value       = google_compute_network.main.name
}

output "private_subnets" {
  description = "Names of private subnets"
  value       = google_compute_subnetwork.private[*].name
}

output "public_subnets" {
  description = "Names of public subnets"
  value       = google_compute_subnetwork.public[*].name
}

output "data_lake_bucket" {
  description = "Name of the data lake bucket"
  value       = google_storage_bucket.data_lake.name
}

output "data_warehouse_bucket" {
  description = "Name of the data warehouse bucket"
  value       = google_storage_bucket.data_warehouse.name
}

output "postgres_connection_name" {
  description = "Connection name of the Cloud SQL instance"
  value       = google_sql_database_instance.postgres.connection_name
}

output "redis_host" {
  description = "Host of the Redis instance"
  value       = google_redis_instance.main.host
}

output "gke_cluster_name" {
  description = "Name of the GKE cluster"
  value       = google_container_cluster.main.name
}

output "gke_cluster_endpoint" {
  description = "Endpoint of the GKE cluster"
  value       = google_container_cluster.main.endpoint
}

output "load_balancer_ip" {
  description = "IP address of the load balancer"
  value       = google_compute_global_address.main.address
}
