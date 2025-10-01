#!/usr/bin/env python3
"""
DataDog Setup Script
Configures DataDog dashboards and monitors for the Metrify Smart Metering pipeline
"""

import argparse
import logging
import yaml
import requests
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDogSetup:
    """Main class for DataDog setup"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the DataDog setup"""
        self.config = self._load_config(config_path)
        self.api_key = self.config['monitoring']['datadog']['api_key']
        self.app_key = self.config['monitoring']['datadog']['app_key']
        self.site = self.config['monitoring']['datadog']['site']
        self.base_url = f"https://api.{self.site}"
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def setup_dashboards(self, environment: str):
        """Set up DataDog dashboards"""
        logger.info(f"Setting up DataDog dashboards for {environment}")
        
        dashboards = [
            self._create_data_quality_dashboard(environment),
            self._create_performance_dashboard(environment),
            self._create_smart_meter_dashboard(environment),
            self._create_grid_status_dashboard(environment),
            self._create_weather_dashboard(environment)
        ]
        
        for dashboard in dashboards:
            self._create_dashboard(dashboard)
        
        logger.info("DataDog dashboards created successfully")
    
    def _create_data_quality_dashboard(self, environment: str) -> dict:
        """Create data quality dashboard"""
        return {
            "title": f"Metrify Data Quality - {environment.title()}",
            "description": "Data quality metrics and alerts for smart metering pipeline",
            "widgets": [
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "avg:metrify.data_quality_score{environment:" + environment + "}",
                                "display_type": "line"
                            }
                        ],
                        "title": "Average Data Quality Score"
                    }
                },
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "sum:metrify.data_quality_checks{environment:" + environment + "}",
                                "display_type": "bars"
                            }
                        ],
                        "title": "Data Quality Checks"
                    }
                },
                {
                    "definition": {
                        "type": "query_value",
                        "requests": [
                            {
                                "q": "avg:metrify.data_quality_score{environment:" + environment + "}",
                                "aggregator": "avg"
                            }
                        ],
                        "title": "Overall Quality Score"
                    }
                }
            ]
        }
    
    def _create_performance_dashboard(self, environment: str) -> dict:
        """Create performance dashboard"""
        return {
            "title": f"Metrify Performance - {environment.title()}",
            "description": "Performance metrics for smart metering pipeline",
            "widgets": [
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "avg:metrify.ingestion_rate{environment:" + environment + "}",
                                "display_type": "line"
                            }
                        ],
                        "title": "Data Ingestion Rate"
                    }
                },
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "avg:metrify.processing_time{environment:" + environment + "}",
                                "display_type": "line"
                            }
                        ],
                        "title": "Processing Time"
                    }
                },
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "sum:metrify.records_processed{environment:" + environment + "}",
                                "display_type": "bars"
                            }
                        ],
                        "title": "Records Processed"
                    }
                }
            ]
        }
    
    def _create_smart_meter_dashboard(self, environment: str) -> dict:
        """Create smart meter dashboard"""
        return {
            "title": f"Smart Meter Analytics - {environment.title()}",
            "description": "Smart meter data analytics and insights",
            "widgets": [
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "sum:metrify.smart_meter_consumption{environment:" + environment + "}",
                                "display_type": "line"
                            }
                        ],
                        "title": "Total Consumption"
                    }
                },
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "avg:metrify.smart_meter_voltage{environment:" + environment + "}",
                                "display_type": "line"
                            }
                        ],
                        "title": "Average Voltage"
                    }
                },
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "avg:metrify.smart_meter_current{environment:" + environment + "}",
                                "display_type": "line"
                            }
                        ],
                        "title": "Average Current"
                    }
                }
            ]
        }
    
    def _create_grid_status_dashboard(self, environment: str) -> dict:
        """Create grid status dashboard"""
        return {
            "title": f"Grid Status - {environment.title()}",
            "description": "Grid operator status and stability metrics",
            "widgets": [
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "avg:metrify.grid_stability_score{environment:" + environment + "}",
                                "display_type": "line"
                            }
                        ],
                        "title": "Grid Stability Score"
                    }
                },
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "avg:metrify.grid_utilization{environment:" + environment + "}",
                                "display_type": "line"
                            }
                        ],
                        "title": "Grid Utilization"
                    }
                },
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "avg:metrify.grid_frequency{environment:" + environment + "}",
                                "display_type": "line"
                            }
                        ],
                        "title": "Grid Frequency"
                    }
                }
            ]
        }
    
    def _create_weather_dashboard(self, environment: str) -> dict:
        """Create weather dashboard"""
        return {
            "title": f"Weather Impact - {environment.title()}",
            "description": "Weather data and energy demand correlation",
            "widgets": [
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "avg:metrify.temperature{environment:" + environment + "}",
                                "display_type": "line"
                            }
                        ],
                        "title": "Average Temperature"
                    }
                },
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "avg:metrify.energy_demand_factor{environment:" + environment + "}",
                                "display_type": "line"
                            }
                        ],
                        "title": "Energy Demand Factor"
                    }
                },
                {
                    "definition": {
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "avg:metrify.humidity{environment:" + environment + "}",
                                "display_type": "line"
                            }
                        ],
                        "title": "Average Humidity"
                    }
                }
            ]
        }
    
    def _create_dashboard(self, dashboard: dict):
        """Create a DataDog dashboard"""
        url = f"{self.base_url}/api/v1/dashboard"
        headers = {
            "DD-API-KEY": self.api_key,
            "DD-APPLICATION-KEY": self.app_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, headers=headers, json=dashboard)
            response.raise_for_status()
            logger.info(f"Dashboard '{dashboard['title']}' created successfully")
        except Exception as e:
            logger.error(f"Error creating dashboard '{dashboard['title']}': {e}")
    
    def setup_monitors(self, environment: str):
        """Set up DataDog monitors"""
        logger.info(f"Setting up DataDog monitors for {environment}")
        
        monitors = [
            self._create_data_quality_monitor(environment),
            self._create_performance_monitor(environment),
            self._create_error_rate_monitor(environment),
            self._create_data_latency_monitor(environment)
        ]
        
        for monitor in monitors:
            self._create_monitor(monitor)
        
        logger.info("DataDog monitors created successfully")
    
    def _create_data_quality_monitor(self, environment: str) -> dict:
        """Create data quality monitor"""
        return {
            "name": f"Metrify Data Quality - {environment.title()}",
            "type": "metric alert",
            "query": f"avg(last_5m):avg:metrify.data_quality_score{{environment:{environment}}} < 0.8",
            "message": f"Data quality score is below threshold for {environment} environment",
            "tags": [f"environment:{environment}", "service:data-quality"],
            "options": {
                "thresholds": {
                    "critical": 0.8,
                    "warning": 0.9
                },
                "notify_audit": False,
                "require_full_window": True,
                "notify_no_data": True,
                "no_data_timeframe": 10
            }
        }
    
    def _create_performance_monitor(self, environment: str) -> dict:
        """Create performance monitor"""
        return {
            "name": f"Metrify Performance - {environment.title()}",
            "type": "metric alert",
            "query": f"avg(last_5m):avg:metrify.processing_time{{environment:{environment}}} > 300",
            "message": f"Processing time is above threshold for {environment} environment",
            "tags": [f"environment:{environment}", "service:performance"],
            "options": {
                "thresholds": {
                    "critical": 300,
                    "warning": 200
                },
                "notify_audit": False,
                "require_full_window": True,
                "notify_no_data": True,
                "no_data_timeframe": 10
            }
        }
    
    def _create_error_rate_monitor(self, environment: str) -> dict:
        """Create error rate monitor"""
        return {
            "name": f"Metrify Error Rate - {environment.title()}",
            "type": "metric alert",
            "query": f"avg(last_5m):avg:metrify.error_rate{{environment:{environment}}} > 0.05",
            "message": f"Error rate is above threshold for {environment} environment",
            "tags": [f"environment:{environment}", "service:error-rate"],
            "options": {
                "thresholds": {
                    "critical": 0.05,
                    "warning": 0.02
                },
                "notify_audit": False,
                "require_full_window": True,
                "notify_no_data": True,
                "no_data_timeframe": 10
            }
        }
    
    def _create_data_latency_monitor(self, environment: str) -> dict:
        """Create data latency monitor"""
        return {
            "name": f"Metrify Data Latency - {environment.title()}",
            "type": "metric alert",
            "query": f"avg(last_5m):avg:metrify.data_latency{{environment:{environment}}} > 300",
            "message": f"Data latency is above threshold for {environment} environment",
            "tags": [f"environment:{environment}", "service:data-latency"],
            "options": {
                "thresholds": {
                    "critical": 300,
                    "warning": 200
                },
                "notify_audit": False,
                "require_full_window": True,
                "notify_no_data": True,
                "no_data_timeframe": 10
            }
        }
    
    def _create_monitor(self, monitor: dict):
        """Create a DataDog monitor"""
        url = f"{self.base_url}/api/v1/monitor"
        headers = {
            "DD-API-KEY": self.api_key,
            "DD-APPLICATION-KEY": self.app_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, headers=headers, json=monitor)
            response.raise_for_status()
            logger.info(f"Monitor '{monitor['name']}' created successfully")
        except Exception as e:
            logger.error(f"Error creating monitor '{monitor['name']}': {e}")
    
    def setup_logs(self, environment: str):
        """Set up DataDog log collection"""
        logger.info(f"Setting up DataDog log collection for {environment}")
        
        # This would typically involve configuring log collection
        # through the DataDog agent or Kubernetes DaemonSet
        logger.info("DataDog log collection configured")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Set up DataDog for Metrify Smart Metering Pipeline')
    parser.add_argument('--environment', required=True, choices=['dev', 'staging', 'prod'],
                       help='Target environment')
    parser.add_argument('--components', nargs='+', 
                       choices=['dashboards', 'monitors', 'logs'],
                       default=['dashboards', 'monitors', 'logs'],
                       help='Components to set up')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    setup = DataDogSetup(args.config)
    
    if 'dashboards' in args.components:
        setup.setup_dashboards(args.environment)
    
    if 'monitors' in args.components:
        setup.setup_monitors(args.environment)
    
    if 'logs' in args.components:
        setup.setup_logs(args.environment)

if __name__ == "__main__":
    main()
