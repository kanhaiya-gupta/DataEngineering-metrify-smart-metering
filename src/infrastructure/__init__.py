"""
Infrastructure Module
Contains infrastructure implementations for external dependencies
"""

from .database.models.smart_meter_model import SmartMeterModel, SmartMeterReadingModel, MeterEventModel
from .database.models.grid_operator_model import GridOperatorModel, GridStatusModel, GridEventModel
from .database.models.weather_station_model import WeatherStationModel, WeatherObservationModel, WeatherEventModel
from .database.repositories.smart_meter_repository import SmartMeterRepository
from .database.repositories.grid_operator_repository import GridOperatorRepository
from .database.repositories.weather_station_repository import WeatherStationRepository
from .database.config import DatabaseConfig, get_database_config, get_session, get_session_context
from .external.kafka.kafka_producer import KafkaProducer
from .external.kafka.kafka_consumer import KafkaConsumer
from .external.s3.s3_client import S3Client
from .external.apis.data_quality_service import DataQualityService
from .external.apis.anomaly_detection_service import AnomalyDetectionService
from .external.apis.grid_data_service import GridDataService
from .external.apis.weather_data_service import WeatherDataService
from .external.apis.alerting_service import AlertingService
from .external.snowflake.snowflake_client import SnowflakeClient
from .external.snowflake.query_executor import SnowflakeQueryExecutor
from .external.kafka.message_serializer import MessageSerializer
from .external.s3.data_archiver import S3DataArchiver
from .external.monitoring.datadog.datadog_client import DataDogClient
from .external.monitoring.prometheus.prometheus_client import PrometheusClient
from .external.monitoring.grafana.grafana_client import GrafanaClient
from .external.monitoring.jaeger.jaeger_client import JaegerClient
from .external.monitoring.monitoring_service import MonitoringService
from .external.airflow.airflow_client import AirflowClient

__all__ = [
    # Database Models
    "SmartMeterModel",
    "SmartMeterReadingModel", 
    "MeterEventModel",
    "GridOperatorModel",
    "GridStatusModel",
    "GridEventModel",
    "WeatherStationModel",
    "WeatherObservationModel",
    "WeatherEventModel",
    
    # Repository Implementations
    "SmartMeterRepository",
    "GridOperatorRepository",
    "WeatherStationRepository",
    
    # Database Configuration
    "DatabaseConfig",
    "get_database_config",
    "get_session",
    "get_session_context",
    
    # Kafka Services
    "KafkaProducer",
    "KafkaConsumer",
    
    # S3 Services
    "S3Client",
    
    # API Services
    "DataQualityService",
    "AnomalyDetectionService",
    "GridDataService",
    "WeatherDataService",
    "AlertingService",
    
    # Snowflake Services
    "SnowflakeClient",
    "SnowflakeQueryExecutor",
    
    # Kafka Services
    "MessageSerializer",
    
    # S3 Services
    "S3DataArchiver",
    
    # Monitoring Services
    "DataDogClient",
    "PrometheusClient",
    "GrafanaClient",
    "JaegerClient",
    "MonitoringService",
    
    # Airflow Services
    "AirflowClient",
]
