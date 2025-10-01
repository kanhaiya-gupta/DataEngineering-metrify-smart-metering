# Multi-Cloud Cost Optimization
# Terraform configuration for cost optimization across AWS, Azure, and GCP

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
  }
}

# AWS Provider
provider "aws" {
  region = var.aws_region
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

# Data sources
data "aws_caller_identity" "current" {}
data "azurerm_client_config" "current" {}
data "google_client_config" "current" {}

# AWS Cost Optimization
# Budget
resource "aws_budgets_budget" "main" {
  name         = "${var.project_name}-budget"
  budget_type  = "COST"
  limit_amount = var.budget_limit
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filters = {
    Tag = ["Project:${var.project_name}"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = [var.admin_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 100
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = [var.admin_email]
  }
}

# Cost Anomaly Detection
resource "aws_ce_anomaly_detector" "main" {
  name = "${var.project_name}-cost-anomaly-detector"

  specification = jsonencode({
    threshold = {
      threshold_value = var.cost_anomaly_threshold
      threshold_type  = "ABSOLUTE"
    }
  })

  monitor_arn_lists = [aws_ce_cost_category.main.arn]
}

resource "aws_ce_cost_category" "main" {
  name = "${var.project_name}-cost-category"

  rule {
    value = "Smart Meter Platform"
    rule {
      dimension {
        key           = "SERVICE"
        values        = ["Amazon Elastic Compute Cloud - Compute", "Amazon Relational Database Service", "Amazon Simple Storage Service"]
        match_options = ["EQUALS"]
      }
    }
  }

  rule {
    value = "Data Processing"
    rule {
      dimension {
        key           = "SERVICE"
        values        = ["Amazon Kinesis", "Amazon MSK", "Amazon EMR"]
        match_options = ["EQUALS"]
      }
    }
  }
}

# Reserved Instances
resource "aws_ec2_capacity_reservation" "main" {
  count = var.enable_reserved_instances ? 1 : 0

  instance_type     = var.reserved_instance_type
  instance_platform = "Linux/UNIX"
  instance_count    = var.reserved_instance_count
  availability_zone = var.aws_availability_zone

  tags = {
    Name = "${var.project_name}-reserved-instance"
  }
}

# Spot Instances
resource "aws_spot_instance_request" "main" {
  count = var.enable_spot_instances ? var.spot_instance_count : 0

  ami           = var.spot_instance_ami
  instance_type = var.spot_instance_type
  spot_price    = var.spot_instance_price

  tags = {
    Name = "${var.project_name}-spot-instance-${count.index + 1}"
  }
}

# Auto Scaling
resource "aws_autoscaling_group" "main" {
  count = var.enable_auto_scaling ? 1 : 0

  name                = "${var.project_name}-asg"
  vpc_zone_identifier = var.aws_subnet_ids
  target_group_arns   = var.aws_target_group_arns
  health_check_type   = "ELB"
  health_check_grace_period = 300

  min_size         = var.min_capacity
  max_size         = var.max_capacity
  desired_capacity = var.desired_capacity

  launch_template {
    id      = aws_launch_template.main[0].id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "${var.project_name}-asg-instance"
    propagate_at_launch = true
  }
}

resource "aws_launch_template" "main" {
  count = var.enable_auto_scaling ? 1 : 0

  name_prefix   = "${var.project_name}-"
  image_id      = var.launch_template_ami
  instance_type = var.launch_template_instance_type

  vpc_security_group_ids = var.aws_security_group_ids

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.project_name}-asg-instance"
    }
  }
}

# Azure Cost Optimization
# Budget
resource "azurerm_consumption_budget_subscription" "main" {
  name            = "${var.project_name}-budget"
  subscription_id = data.azurerm_client_config.current.subscription_id

  amount     = var.budget_limit
  time_grain = "Monthly"

  time_period {
    start_date = "2024-01-01T00:00:00Z"
    end_date   = "2025-12-31T23:59:59Z"
  }

  filter {
    tag {
      name = "Project"
      values = [var.project_name]
    }
  }

  notification {
    enabled   = true
    threshold = 80
    operator  = "GreaterThan"

    contact_emails = [var.admin_email]
  }

  notification {
    enabled   = true
    threshold = 100
    operator  = "GreaterThan"

    contact_emails = [var.admin_email]
  }
}

# Reserved Instances
resource "azurerm_reserved_vm_instance" "main" {
  count = var.enable_azure_reserved_instances ? 1 : 0

  name                = "${var.project_name}-reserved-instance"
  location            = var.azure_region
  resource_group_name = var.azure_resource_group_name
  scope               = "Subscription"
  billing_scope_id    = data.azurerm_client_config.current.subscription_id
  sku_name            = var.azure_reserved_instance_sku
  quantity            = var.azure_reserved_instance_quantity
}

# Spot Instances
resource "azurerm_linux_virtual_machine" "spot" {
  count = var.enable_azure_spot_instances ? var.azure_spot_instance_count : 0

  name                = "${var.project_name}-spot-vm-${count.index + 1}"
  resource_group_name = var.azure_resource_group_name
  location            = var.azure_region
  size                = var.azure_spot_instance_size
  admin_username      = "adminuser"

  disable_password_authentication = true

  admin_ssh_key {
    username   = "adminuser"
    public_key = var.azure_ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-focal"
    sku       = "20_04-lts"
    version   = "latest"
  }

  priority        = "Spot"
  eviction_policy = "Deallocate"

  tags = {
    Name = "${var.project_name}-spot-vm-${count.index + 1}"
  }
}

# Auto Scaling
resource "azurerm_monitor_autoscale_setting" "main" {
  count = var.enable_azure_auto_scaling ? 1 : 0

  name                = "${var.project_name}-autoscale"
  resource_group_name = var.azure_resource_group_name
  location            = var.azure_region
  target_resource_id  = var.azure_vmss_id

  profile {
    name = "defaultProfile"

    capacity {
      default = var.azure_min_capacity
      minimum = var.azure_min_capacity
      maximum = var.azure_max_capacity
    }

    rule {
      metric_trigger {
        metric_name        = "Percentage CPU"
        metric_resource_id = var.azure_vmss_id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "GreaterThan"
        threshold          = 75
      }

      scale_action {
        direction = "Increase"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT1M"
      }
    }

    rule {
      metric_trigger {
        metric_name        = "Percentage CPU"
        metric_resource_id = var.azure_vmss_id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "LessThan"
        threshold          = 25
      }

      scale_action {
        direction = "Decrease"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT1M"
      }
    }
  }
}

# GCP Cost Optimization
# Budget
resource "google_billing_budget" "main" {
  billing_account = var.gcp_billing_account_id
  display_name    = "${var.project_name}-budget"

  budget_filter {
    projects = ["projects/${var.gcp_project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_limit)
    }
  }

  threshold_rules {
    threshold_percent = 0.8
    spend_basis      = "CURRENT_SPEND"
  }

  threshold_rules {
    threshold_percent = 1.0
    spend_basis      = "CURRENT_SPEND"
  }

  all_updates_rule {
    monitoring_notification_channels = [google_monitoring_notification_channel.email.id]
    disable_default_iam_recipients   = false
  }
}

# Committed Use Discounts
resource "google_compute_commitment" "main" {
  count = var.enable_gcp_committed_use ? 1 : 0

  name   = "${var.project_name}-commitment"
  plan   = "TWELVE_MONTH"
  type   = "GENERAL_PURPOSE_N1"
  region = var.gcp_region

  resources {
    type       = "VCPU"
    amount     = var.gcp_commitment_vcpu
    accelerator_type = "NVIDIA_TESLA_V100"
  }
}

# Preemptible Instances
resource "google_compute_instance" "preemptible" {
  count = var.enable_gcp_preemptible_instances ? var.gcp_preemptible_count : 0

  name         = "${var.project_name}-preemptible-${count.index + 1}"
  machine_type = var.gcp_preemptible_machine_type
  zone         = var.gcp_zone

  boot_disk {
    initialize_params {
      image = var.gcp_preemptible_image
    }
  }

  network_interface {
    network = var.gcp_network_name
  }

  scheduling {
    preemptible       = true
    automatic_restart = false
  }

  tags = ["preemptible"]
}

# Auto Scaling
resource "google_compute_autoscaler" "main" {
  count = var.enable_gcp_auto_scaling ? 1 : 0

  name   = "${var.project_name}-autoscaler"
  zone   = var.gcp_zone
  target = var.gcp_instance_group_manager_id

  autoscaling_policy {
    max_replicas    = var.gcp_max_replicas
    min_replicas    = var.gcp_min_replicas
    cooldown_period = 60

    cpu_utilization {
      target = 0.6
    }

    load_balancing_utilization {
      target = 0.5
    }
  }
}

# Cost Monitoring
resource "google_monitoring_notification_channel" "email" {
  display_name = "Email Notification"
  type         = "email"

  labels = {
    email_address = var.admin_email
  }
}

# Cost Alerts
resource "google_monitoring_alert_policy" "cost_alert" {
  display_name = "Cost Alert"
  combiner     = "OR"

  conditions {
    display_name = "Cost threshold exceeded"
    condition_threshold {
      filter          = "resource.type=\"billing_account\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = var.budget_limit * 0.8

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

# Cross-Cloud Cost Analysis
# Cost Comparison Dashboard
resource "aws_cloudwatch_dashboard" "cost_analysis" {
  dashboard_name = "${var.project_name}-cost-analysis"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/Billing", "EstimatedCharges", "Currency", "USD"],
            ["AWS/Billing", "EstimatedCharges", "ServiceName", "Amazon Elastic Compute Cloud - Compute"],
            ["AWS/Billing", "EstimatedCharges", "ServiceName", "Amazon Relational Database Service"],
            ["AWS/Billing", "EstimatedCharges", "ServiceName", "Amazon Simple Storage Service"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "AWS Cost Breakdown"
          period  = 86400
        }
      }
    ]
  })
}

# Cost Optimization Recommendations
resource "aws_ce_cost_category" "optimization" {
  name = "${var.project_name}-optimization"

  rule {
    value = "Optimize"
    rule {
      dimension {
        key           = "SERVICE"
        values        = ["Amazon Elastic Compute Cloud - Compute"]
        match_options = ["EQUALS"]
      }
    }
  }

  rule {
    value = "Review"
    rule {
      dimension {
        key           = "SERVICE"
        values        = ["Amazon Relational Database Service"]
        match_options = ["EQUALS"]
      }
    }
  }
}

# Outputs
output "aws_budget_name" {
  description = "Name of the AWS budget"
  value       = aws_budgets_budget.main.name
}

output "azure_budget_name" {
  description = "Name of the Azure budget"
  value       = azurerm_consumption_budget_subscription.main.name
}

output "gcp_budget_name" {
  description = "Name of the GCP budget"
  value       = google_billing_budget.main.display_name
}

output "cost_anomaly_detector_name" {
  description = "Name of the cost anomaly detector"
  value       = aws_ce_anomaly_detector.main.name
}

output "cost_dashboard_url" {
  description = "URL of the cost analysis dashboard"
  value       = "https://console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.cost_analysis.dashboard_name}"
}
