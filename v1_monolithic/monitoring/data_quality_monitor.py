"""
Data Quality Monitoring and Alerting
Real-time monitoring of data quality metrics and alerting
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import yaml
from pathlib import Path
import boto3
from snowflake.connector import connect
import pandas as pd
import numpy as np
from datadog import initialize, api
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityAlert:
    """Data class for quality alerts"""
    alert_id: str
    table_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    message: str
    timestamp: datetime

class DataQualityMonitor:
    """Main class for data quality monitoring"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data quality monitor"""
        self.config = self._load_config(config_path)
        self.snowflake_conn = self._init_snowflake_connection()
        self.s3_client = boto3.client('s3')
        self._init_datadog()
        self.alert_history = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _init_snowflake_connection(self):
        """Initialize Snowflake connection"""
        db_config = self.config['databases']['snowflake']
        return connect(
            user=db_config['user'],
            password=db_config['password'],
            account=db_config['account'],
            warehouse=db_config['warehouse'],
            database=db_config['database'],
            schema=db_config['schema']
        )
    
    def _init_datadog(self):
        """Initialize Datadog monitoring"""
        if 'datadog' in self.config['monitoring']:
            dd_config = self.config['monitoring']['datadog']
            initialize(
                api_key=dd_config['api_key'],
                app_key=dd_config['app_key'],
                api_host=dd_config.get('site', 'datadoghq.com')
            )
    
    def monitor_data_quality(self, date: str = None):
        """Monitor data quality for all tables"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Starting data quality monitoring for date: {date}")
        
        # Monitor each data source
        smart_meter_alerts = self._monitor_smart_meter_quality(date)
        grid_alerts = self._monitor_grid_operator_quality(date)
        weather_alerts = self._monitor_weather_quality(date)
        
        # Combine all alerts
        all_alerts = smart_meter_alerts + grid_alerts + weather_alerts
        
        # Process alerts
        self._process_alerts(all_alerts)
        
        # Send summary report
        self._send_quality_report(date, all_alerts)
        
        logger.info(f"Data quality monitoring completed. Found {len(all_alerts)} alerts")
    
    def _monitor_smart_meter_quality(self, date: str) -> List[QualityAlert]:
        """Monitor smart meter data quality"""
        alerts = []
        
        try:
            # Query quality metrics
            query = f"""
            SELECT 
                COUNT(*) as total_readings,
                COUNT(CASE WHEN data_quality_score >= 0.9 THEN 1 END) as high_quality_readings,
                COUNT(CASE WHEN data_quality_score < 0.7 THEN 1 END) as low_quality_readings,
                AVG(data_quality_score) as avg_quality_score,
                COUNT(CASE WHEN consumption_kwh < 0 THEN 1 END) as negative_consumption,
                COUNT(CASE WHEN voltage < 200 OR voltage > 250 THEN 1 END) as voltage_anomalies,
                COUNT(CASE WHEN current < 0 OR current > 100 THEN 1 END) as current_anomalies
            FROM raw.smart_meter_readings 
            WHERE DATE(timestamp) = '{date}'
            """
            
            df = pd.read_sql(query, self.snowflake_conn)
            
            if df.empty or df['total_readings'].iloc[0] == 0:
                alerts.append(QualityAlert(
                    alert_id=f"smart_meter_no_data_{date}",
                    table_name="smart_meter_readings",
                    metric_name="data_availability",
                    current_value=0,
                    threshold=1,
                    severity="critical",
                    message="No smart meter data found for the date",
                    timestamp=datetime.utcnow()
                ))
                return alerts
            
            row = df.iloc[0]
            
            # Check data availability
            if row['total_readings'] < 1000:  # Expected minimum readings
                alerts.append(QualityAlert(
                    alert_id=f"smart_meter_low_volume_{date}",
                    table_name="smart_meter_readings",
                    metric_name="data_volume",
                    current_value=row['total_readings'],
                    threshold=1000,
                    severity="warning",
                    message=f"Low data volume: {row['total_readings']} readings",
                    timestamp=datetime.utcnow()
                ))
            
            # Check quality score
            if row['avg_quality_score'] < 0.8:
                alerts.append(QualityAlert(
                    alert_id=f"smart_meter_low_quality_{date}",
                    table_name="smart_meter_readings",
                    metric_name="avg_quality_score",
                    current_value=row['avg_quality_score'],
                    threshold=0.8,
                    severity="warning",
                    message=f"Low average quality score: {row['avg_quality_score']:.2f}",
                    timestamp=datetime.utcnow()
                ))
            
            # Check for data anomalies
            if row['negative_consumption'] > 0:
                alerts.append(QualityAlert(
                    alert_id=f"smart_meter_negative_consumption_{date}",
                    table_name="smart_meter_readings",
                    metric_name="negative_consumption",
                    current_value=row['negative_consumption'],
                    threshold=0,
                    severity="critical",
                    message=f"Found {row['negative_consumption']} negative consumption readings",
                    timestamp=datetime.utcnow()
                ))
            
            if row['voltage_anomalies'] > row['total_readings'] * 0.05:  # 5% threshold
                alerts.append(QualityAlert(
                    alert_id=f"smart_meter_voltage_anomalies_{date}",
                    table_name="smart_meter_readings",
                    metric_name="voltage_anomalies",
                    current_value=row['voltage_anomalies'],
                    threshold=row['total_readings'] * 0.05,
                    severity="warning",
                    message=f"High voltage anomalies: {row['voltage_anomalies']} readings",
                    timestamp=datetime.utcnow()
                ))
            
            if row['current_anomalies'] > row['total_readings'] * 0.05:  # 5% threshold
                alerts.append(QualityAlert(
                    alert_id=f"smart_meter_current_anomalies_{date}",
                    table_name="smart_meter_readings",
                    metric_name="current_anomalies",
                    current_value=row['current_anomalies'],
                    threshold=row['total_readings'] * 0.05,
                    severity="warning",
                    message=f"High current anomalies: {row['current_anomalies']} readings",
                    timestamp=datetime.utcnow()
                ))
            
        except Exception as e:
            logger.error(f"Error monitoring smart meter quality: {e}")
            alerts.append(QualityAlert(
                alert_id=f"smart_meter_monitoring_error_{date}",
                table_name="smart_meter_readings",
                metric_name="monitoring_error",
                current_value=0,
                threshold=1,
                severity="critical",
                message=f"Error monitoring smart meter data: {str(e)}",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    def _monitor_grid_operator_quality(self, date: str) -> List[QualityAlert]:
        """Monitor grid operator data quality"""
        alerts = []
        
        try:
            # Query quality metrics
            query = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT operator_name) as unique_operators,
                AVG(grid_stability_score) as avg_stability_score,
                COUNT(CASE WHEN frequency_hz < 49.5 OR frequency_hz > 50.5 THEN 1 END) as frequency_anomalies,
                COUNT(CASE WHEN voltage_kv < 380 OR voltage_kv > 420 THEN 1 END) as voltage_anomalies,
                COUNT(CASE WHEN available_capacity_mw > total_capacity_mw THEN 1 END) as capacity_anomalies
            FROM raw.grid_status 
            WHERE DATE(timestamp) = '{date}'
            """
            
            df = pd.read_sql(query, self.snowflake_conn)
            
            if df.empty or df['total_records'].iloc[0] == 0:
                alerts.append(QualityAlert(
                    alert_id=f"grid_no_data_{date}",
                    table_name="grid_status",
                    metric_name="data_availability",
                    current_value=0,
                    threshold=1,
                    severity="critical",
                    message="No grid operator data found for the date",
                    timestamp=datetime.utcnow()
                ))
                return alerts
            
            row = df.iloc[0]
            
            # Check data availability
            if row['unique_operators'] < 2:  # Expected at least 2 operators
                alerts.append(QualityAlert(
                    alert_id=f"grid_low_operators_{date}",
                    table_name="grid_status",
                    metric_name="unique_operators",
                    current_value=row['unique_operators'],
                    threshold=2,
                    severity="warning",
                    message=f"Low operator coverage: {row['unique_operators']} operators",
                    timestamp=datetime.utcnow()
                ))
            
            # Check stability score
            if row['avg_stability_score'] < 0.8:
                alerts.append(QualityAlert(
                    alert_id=f"grid_low_stability_{date}",
                    table_name="grid_status",
                    metric_name="avg_stability_score",
                    current_value=row['avg_stability_score'],
                    threshold=0.8,
                    severity="warning",
                    message=f"Low average stability score: {row['avg_stability_score']:.2f}",
                    timestamp=datetime.utcnow()
                ))
            
            # Check for frequency anomalies
            if row['frequency_anomalies'] > 0:
                alerts.append(QualityAlert(
                    alert_id=f"grid_frequency_anomalies_{date}",
                    table_name="grid_status",
                    metric_name="frequency_anomalies",
                    current_value=row['frequency_anomalies'],
                    threshold=0,
                    severity="critical",
                    message=f"Found {row['frequency_anomalies']} frequency anomalies",
                    timestamp=datetime.utcnow()
                ))
            
            # Check for voltage anomalies
            if row['voltage_anomalies'] > 0:
                alerts.append(QualityAlert(
                    alert_id=f"grid_voltage_anomalies_{date}",
                    table_name="grid_status",
                    metric_name="voltage_anomalies",
                    current_value=row['voltage_anomalies'],
                    threshold=0,
                    severity="critical",
                    message=f"Found {row['voltage_anomalies']} voltage anomalies",
                    timestamp=datetime.utcnow()
                ))
            
            # Check for capacity anomalies
            if row['capacity_anomalies'] > 0:
                alerts.append(QualityAlert(
                    alert_id=f"grid_capacity_anomalies_{date}",
                    table_name="grid_status",
                    metric_name="capacity_anomalies",
                    current_value=row['capacity_anomalies'],
                    threshold=0,
                    severity="warning",
                    message=f"Found {row['capacity_anomalies']} capacity anomalies",
                    timestamp=datetime.utcnow()
                ))
            
        except Exception as e:
            logger.error(f"Error monitoring grid operator quality: {e}")
            alerts.append(QualityAlert(
                alert_id=f"grid_monitoring_error_{date}",
                table_name="grid_status",
                metric_name="monitoring_error",
                current_value=0,
                threshold=1,
                severity="critical",
                message=f"Error monitoring grid operator data: {str(e)}",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    def _monitor_weather_quality(self, date: str) -> List[QualityAlert]:
        """Monitor weather data quality"""
        alerts = []
        
        try:
            # Query quality metrics
            query = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT city) as unique_cities,
                AVG(temperature_celsius) as avg_temperature,
                AVG(humidity_percent) as avg_humidity,
                COUNT(CASE WHEN temperature_celsius < -30 OR temperature_celsius > 50 THEN 1 END) as temperature_anomalies,
                COUNT(CASE WHEN humidity_percent < 0 OR humidity_percent > 100 THEN 1 END) as humidity_anomalies,
                COUNT(CASE WHEN pressure_hpa < 950 OR pressure_hpa > 1050 THEN 1 END) as pressure_anomalies
            FROM raw.weather_data 
            WHERE DATE(timestamp) = '{date}'
            """
            
            df = pd.read_sql(query, self.snowflake_conn)
            
            if df.empty or df['total_records'].iloc[0] == 0:
                alerts.append(QualityAlert(
                    alert_id=f"weather_no_data_{date}",
                    table_name="weather_data",
                    metric_name="data_availability",
                    current_value=0,
                    threshold=1,
                    severity="critical",
                    message="No weather data found for the date",
                    timestamp=datetime.utcnow()
                ))
                return alerts
            
            row = df.iloc[0]
            
            # Check data availability
            if row['unique_cities'] < 4:  # Expected at least 4 cities
                alerts.append(QualityAlert(
                    alert_id=f"weather_low_cities_{date}",
                    table_name="weather_data",
                    metric_name="unique_cities",
                    current_value=row['unique_cities'],
                    threshold=4,
                    severity="warning",
                    message=f"Low city coverage: {row['unique_cities']} cities",
                    timestamp=datetime.utcnow()
                ))
            
            # Check for temperature anomalies
            if row['temperature_anomalies'] > 0:
                alerts.append(QualityAlert(
                    alert_id=f"weather_temperature_anomalies_{date}",
                    table_name="weather_data",
                    metric_name="temperature_anomalies",
                    current_value=row['temperature_anomalies'],
                    threshold=0,
                    severity="warning",
                    message=f"Found {row['temperature_anomalies']} temperature anomalies",
                    timestamp=datetime.utcnow()
                ))
            
            # Check for humidity anomalies
            if row['humidity_anomalies'] > 0:
                alerts.append(QualityAlert(
                    alert_id=f"weather_humidity_anomalies_{date}",
                    table_name="weather_data",
                    metric_name="humidity_anomalies",
                    current_value=row['humidity_anomalies'],
                    threshold=0,
                    severity="warning",
                    message=f"Found {row['humidity_anomalies']} humidity anomalies",
                    timestamp=datetime.utcnow()
                ))
            
            # Check for pressure anomalies
            if row['pressure_anomalies'] > 0:
                alerts.append(QualityAlert(
                    alert_id=f"weather_pressure_anomalies_{date}",
                    table_name="weather_data",
                    metric_name="pressure_anomalies",
                    current_value=row['pressure_anomalies'],
                    threshold=0,
                    severity="warning",
                    message=f"Found {row['pressure_anomalies']} pressure anomalies",
                    timestamp=datetime.utcnow()
                ))
            
        except Exception as e:
            logger.error(f"Error monitoring weather quality: {e}")
            alerts.append(QualityAlert(
                alert_id=f"weather_monitoring_error_{date}",
                table_name="weather_data",
                metric_name="monitoring_error",
                current_value=0,
                threshold=1,
                severity="critical",
                message=f"Error monitoring weather data: {str(e)}",
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    def _process_alerts(self, alerts: List[QualityAlert]):
        """Process and send alerts"""
        for alert in alerts:
            # Store alert in history
            self.alert_history.append(alert)
            
            # Send to Datadog
            self._send_datadog_alert(alert)
            
            # Send email for critical alerts
            if alert.severity == "critical":
                self._send_email_alert(alert)
            
            logger.info(f"Processed alert: {alert.alert_id} - {alert.message}")
    
    def _send_datadog_alert(self, alert: QualityAlert):
        """Send alert to Datadog"""
        try:
            if 'datadog' in self.config['monitoring']:
                api.Event.create(
                    title=f"Data Quality Alert: {alert.table_name}",
                    text=alert.message,
                    alert_type="error" if alert.severity == "critical" else "warning",
                    tags=[
                        f"table:{alert.table_name}",
                        f"metric:{alert.metric_name}",
                        f"severity:{alert.severity}"
                    ]
                )
        except Exception as e:
            logger.error(f"Error sending Datadog alert: {e}")
    
    def _send_email_alert(self, alert: QualityAlert):
        """Send email alert for critical issues"""
        try:
            # This would need to be configured with actual email settings
            logger.info(f"Email alert would be sent for: {alert.alert_id}")
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def _send_quality_report(self, date: str, alerts: List[QualityAlert]):
        """Send daily quality report"""
        try:
            # Store report in S3
            report = {
                "date": date,
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a.severity == "critical"]),
                "warning_alerts": len([a for a in alerts if a.severity == "warning"]),
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "table_name": alert.table_name,
                        "metric_name": alert.metric_name,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold,
                        "severity": alert.severity,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in alerts
                ],
                "generated_at": datetime.utcnow().isoformat()
            }
            
            s3_key = f"quality_reports/{date}/daily_quality_report.json"
            
            self.s3_client.put_object(
                Bucket=self.config['aws']['s3_bucket'],
                Key=s3_key,
                Body=json.dumps(report, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Quality report stored for {date}")
            
        except Exception as e:
            logger.error(f"Error sending quality report: {e}")

def main():
    """Main function to run data quality monitoring"""
    import sys
    
    monitor = DataQualityMonitor()
    
    # Get date from command line or use today
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y-%m-%d')
    
    # Run monitoring
    monitor.monitor_data_quality(date)

if __name__ == "__main__":
    main()
