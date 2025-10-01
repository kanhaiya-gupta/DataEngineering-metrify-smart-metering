"""
Grafana Client Implementation
Handles dashboard and data source management for Grafana
"""

import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

try:
    import requests
    from requests.auth import HTTPBasicAuth
except ImportError:
    requests = None
    HTTPBasicAuth = None

from ....core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class GrafanaClient:
    """
    Grafana Client for dashboard and data source management
    
    Handles creation, updating, and management of Grafana dashboards
    and data sources for comprehensive monitoring visualization.
    """
    
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        api_key: Optional[str] = None,
        organization_id: int = 1
    ):
        if requests is None:
            raise InfrastructureError("Requests library not installed", service="grafana")
        
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.api_key = api_key
        self.organization_id = organization_id
        self.session = requests.Session()
        
        # Set up authentication
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            })
        else:
            self.session.auth = HTTPBasicAuth(username, password)
            self.session.headers.update({'Content-Type': 'application/json'})
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Grafana client and verify connection"""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            response.raise_for_status()
            
            # Get user info to verify authentication
            user_response = self.session.get(f"{self.base_url}/api/user")
            user_response.raise_for_status()
            
            self._initialized = True
            logger.info("Grafana client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Grafana client: {str(e)}")
            raise InfrastructureError(f"Failed to initialize Grafana: {str(e)}", service="grafana")
    
    async def create_data_source(self, data_source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new data source in Grafana"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/datasources",
                json=data_source_config
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Created data source: {result.get('name', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Error creating data source: {str(e)}")
            raise InfrastructureError(f"Failed to create data source: {str(e)}", service="grafana")
    
    async def create_prometheus_data_source(self, name: str, url: str) -> Dict[str, Any]:
        """Create Prometheus data source"""
        config = {
            "name": name,
            "type": "prometheus",
            "url": url,
            "access": "proxy",
            "isDefault": False,
            "jsonData": {
                "httpMethod": "POST",
                "queryTimeout": "60s",
                "timeInterval": "5s"
            }
        }
        return await self.create_data_source(config)
    
    async def create_influxdb_data_source(self, name: str, url: str, database: str) -> Dict[str, Any]:
        """Create InfluxDB data source"""
        config = {
            "name": name,
            "type": "influxdb",
            "url": url,
            "database": database,
            "access": "proxy",
            "isDefault": False,
            "jsonData": {
                "httpMode": "POST",
                "timeInterval": "5s"
            }
        }
        return await self.create_data_source(config)
    
    async def create_dashboard(self, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new dashboard in Grafana"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/dashboards/db",
                json=dashboard_config
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Created dashboard: {result.get('title', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise InfrastructureError(f"Failed to create dashboard: {str(e)}", service="grafana")
    
    async def create_smart_meter_dashboard(self, prometheus_datasource_id: int) -> Dict[str, Any]:
        """Create smart meter monitoring dashboard"""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "Smart Meter Monitoring",
                "tags": ["smart-meter", "monitoring"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Meter Readings Rate",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "rate(metrify_smart_metering_meter_readings_total[5m])",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "ops/sec",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Data Quality Score",
                        "type": "gauge",
                        "targets": [
                            {
                                "expr": "metrify_smart_metering_meter_quality_score",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "min": 0,
                                "max": 1,
                                "unit": "short",
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "yellow", "value": 0.7},
                                        {"color": "green", "value": 0.9}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Anomaly Detection Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(metrify_smart_metering_meter_anomalies_total[5m])",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "ops/sec",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        return await self.create_dashboard(dashboard_config)
    
    async def create_grid_operator_dashboard(self, prometheus_datasource_id: int) -> Dict[str, Any]:
        """Create grid operator monitoring dashboard"""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "Grid Operator Monitoring",
                "tags": ["grid-operator", "monitoring"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Grid Stability Score",
                        "type": "gauge",
                        "targets": [
                            {
                                "expr": "metrify_smart_metering_grid_stability_score",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "min": 0,
                                "max": 1,
                                "unit": "short",
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "yellow", "value": 0.7},
                                        {"color": "green", "value": 0.9}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Grid Load Percentage",
                        "type": "gauge",
                        "targets": [
                            {
                                "expr": "metrify_smart_metering_grid_load_percentage",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "min": 0,
                                "max": 100,
                                "unit": "percent",
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 80},
                                        {"color": "red", "value": 95}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Grid Status Updates Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(metrify_smart_metering_grid_status_updates_total[5m])",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "ops/sec",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        return await self.create_dashboard(dashboard_config)
    
    async def create_weather_dashboard(self, prometheus_datasource_id: int) -> Dict[str, Any]:
        """Create weather station monitoring dashboard"""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "Weather Station Monitoring",
                "tags": ["weather", "monitoring"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Temperature",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "metrify_smart_metering_weather_temperature",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "celsius",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Humidity",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "metrify_smart_metering_weather_humidity",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percent",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Weather Observations Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(metrify_smart_metering_weather_observations_total[5m])",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "ops/sec",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        return await self.create_dashboard(dashboard_config)
    
    async def create_system_overview_dashboard(self, prometheus_datasource_id: int) -> Dict[str, Any]:
        """Create system overview dashboard"""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "System Overview",
                "tags": ["system", "overview"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Operation Success Rate",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "rate(metrify_smart_metering_operation_requests_total{status=\"success\"}[5m]) / rate(metrify_smart_metering_operation_requests_total[5m]) * 100",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percent",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Data Quality Score",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "avg(metrify_smart_metering_data_quality_score)",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "short",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Kafka Message Rate",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "rate(metrify_smart_metering_kafka_messages_produced_total[5m])",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "ops/sec",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 0}
                    },
                    {
                        "id": 4,
                        "title": "Operation Duration",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(metrify_smart_metering_operation_duration_seconds_bucket[5m]))",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "s",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        return await self.create_dashboard(dashboard_config)
    
    async def create_airflow_dashboard(self, prometheus_datasource_id: int) -> Dict[str, Any]:
        """Create Airflow monitoring dashboard"""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "Airflow Monitoring",
                "tags": ["airflow", "monitoring"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "DAG Run Success Rate",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "rate(metrify_smart_metering_airflow_dag_runs_total{status=\"success\"}[5m]) / rate(metrify_smart_metering_airflow_dag_runs_total[5m]) * 100",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percent",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Task Run Success Rate",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "rate(metrify_smart_metering_airflow_task_runs_total{status=\"success\"}[5m]) / rate(metrify_smart_metering_airflow_task_runs_total[5m]) * 100",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percent",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "DAG Duration",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(metrify_smart_metering_airflow_dag_duration_seconds_bucket[5m]))",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "s",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        return await self.create_dashboard(dashboard_config)
    
    async def create_kafka_dashboard(self, prometheus_datasource_id: int) -> Dict[str, Any]:
        """Create Kafka monitoring dashboard"""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "Kafka Monitoring",
                "tags": ["kafka", "monitoring"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Message Production Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(metrify_smart_metering_kafka_messages_produced_total[5m])",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "ops/sec",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Message Consumption Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(metrify_smart_metering_kafka_messages_consumed_total[5m])",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "ops/sec",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Consumer Lag",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "metrify_smart_metering_kafka_consumer_lag",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "short",
                                "color": {"mode": "palette-classic"}
                            }
                        },
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        return await self.create_dashboard(dashboard_config)
    
    async def update_dashboard(self, dashboard_id: int, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing dashboard"""
        try:
            response = self.session.put(
                f"{self.base_url}/api/dashboards/db",
                json=dashboard_config
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Updated dashboard: {result.get('title', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {str(e)}")
            raise InfrastructureError(f"Failed to update dashboard: {str(e)}", service="grafana")
    
    async def delete_dashboard(self, dashboard_id: int) -> bool:
        """Delete a dashboard"""
        try:
            response = self.session.delete(f"{self.base_url}/api/dashboards/uid/{dashboard_id}")
            response.raise_for_status()
            
            logger.info(f"Deleted dashboard: {dashboard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting dashboard: {str(e)}")
            raise InfrastructureError(f"Failed to delete dashboard: {str(e)}", service="grafana")
    
    async def get_dashboard(self, dashboard_id: int) -> Dict[str, Any]:
        """Get dashboard by ID"""
        try:
            response = self.session.get(f"{self.base_url}/api/dashboards/uid/{dashboard_id}")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting dashboard: {str(e)}")
            raise InfrastructureError(f"Failed to get dashboard: {str(e)}", service="grafana")
    
    async def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all dashboards"""
        try:
            response = self.session.get(f"{self.base_url}/api/search?type=dash-db")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error listing dashboards: {str(e)}")
            raise InfrastructureError(f"Failed to list dashboards: {str(e)}", service="grafana")
    
    async def get_dashboard_url(self, dashboard_id: int) -> str:
        """Get dashboard URL"""
        return f"{self.base_url}/d/{dashboard_id}"
    
    async def setup_monitoring_dashboards(self, prometheus_datasource_id: int) -> Dict[str, Any]:
        """Setup all monitoring dashboards"""
        try:
            dashboards = {}
            
            # Create all dashboards
            dashboards['smart_meter'] = await self.create_smart_meter_dashboard(prometheus_datasource_id)
            dashboards['grid_operator'] = await self.create_grid_operator_dashboard(prometheus_datasource_id)
            dashboards['weather'] = await self.create_weather_dashboard(prometheus_datasource_id)
            dashboards['system_overview'] = await self.create_system_overview_dashboard(prometheus_datasource_id)
            dashboards['airflow'] = await self.create_airflow_dashboard(prometheus_datasource_id)
            dashboards['kafka'] = await self.create_kafka_dashboard(prometheus_datasource_id)
            
            logger.info("All monitoring dashboards created successfully")
            return dashboards
            
        except Exception as e:
            logger.error(f"Error setting up monitoring dashboards: {str(e)}")
            raise InfrastructureError(f"Failed to setup monitoring dashboards: {str(e)}", service="grafana")
    
    async def get_client_info(self) -> Dict[str, Any]:
        """Get Grafana client information"""
        return {
            "base_url": self.base_url,
            "username": self.username,
            "organization_id": self.organization_id,
            "initialized": self._initialized,
            "api_key_configured": self.api_key is not None
        }
