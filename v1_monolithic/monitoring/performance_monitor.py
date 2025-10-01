"""
Performance Monitoring and Optimization
Monitor pipeline performance and identify optimization opportunities
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Data class for performance metrics"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any]

class PerformanceMonitor:
    """Main class for performance monitoring"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the performance monitor"""
        self.config = self._load_config(config_path)
        self.snowflake_conn = self._init_snowflake_connection()
        self.s3_client = boto3.client('s3')
        self._init_datadog()
        self.metrics_history = []
        
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
    
    def monitor_pipeline_performance(self, date: str = None):
        """Monitor overall pipeline performance"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Starting performance monitoring for date: {date}")
        
        # Monitor different aspects
        ingestion_metrics = self._monitor_ingestion_performance(date)
        processing_metrics = self._monitor_processing_performance(date)
        storage_metrics = self._monitor_storage_performance(date)
        query_metrics = self._monitor_query_performance(date)
        
        # Combine all metrics
        all_metrics = ingestion_metrics + processing_metrics + storage_metrics + query_metrics
        
        # Process metrics
        self._process_metrics(all_metrics)
        
        # Generate performance report
        self._generate_performance_report(date, all_metrics)
        
        # Check for performance issues
        self._check_performance_issues(all_metrics)
        
        logger.info(f"Performance monitoring completed. Collected {len(all_metrics)} metrics")
    
    def _monitor_ingestion_performance(self, date: str) -> List[PerformanceMetric]:
        """Monitor data ingestion performance"""
        metrics = []
        
        try:
            # Query ingestion metrics
            query = f"""
            SELECT 
                'smart_meter_readings' as table_name,
                COUNT(*) as record_count,
                MIN(timestamp) as earliest_timestamp,
                MAX(timestamp) as latest_timestamp,
                COUNT(DISTINCT meter_id) as unique_meters
            FROM raw.smart_meter_readings 
            WHERE DATE(timestamp) = '{date}'
            
            UNION ALL
            
            SELECT 
                'grid_status' as table_name,
                COUNT(*) as record_count,
                MIN(timestamp) as earliest_timestamp,
                MAX(timestamp) as latest_timestamp,
                COUNT(DISTINCT operator_name) as unique_meters
            FROM raw.grid_status 
            WHERE DATE(timestamp) = '{date}'
            
            UNION ALL
            
            SELECT 
                'weather_data' as table_name,
                COUNT(*) as record_count,
                MIN(timestamp) as earliest_timestamp,
                MAX(timestamp) as latest_timestamp,
                COUNT(DISTINCT city) as unique_meters
            FROM raw.weather_data 
            WHERE DATE(timestamp) = '{date}'
            """
            
            df = pd.read_sql(query, self.snowflake_conn)
            
            for _, row in df.iterrows():
                # Calculate ingestion rate
                if row['record_count'] > 0:
                    time_diff = pd.to_datetime(row['latest_timestamp']) - pd.to_datetime(row['earliest_timestamp'])
                    hours = time_diff.total_seconds() / 3600
                    ingestion_rate = row['record_count'] / max(hours, 1)
                    
                    metrics.append(PerformanceMetric(
                        metric_name=f"{row['table_name']}_ingestion_rate",
                        value=ingestion_rate,
                        unit="records_per_hour",
                        timestamp=datetime.utcnow(),
                        context={
                            "table_name": row['table_name'],
                            "record_count": row['record_count'],
                            "unique_entities": row['unique_meters']
                        }
                    ))
                
                # Record count metric
                metrics.append(PerformanceMetric(
                    metric_name=f"{row['table_name']}_record_count",
                    value=row['record_count'],
                    unit="records",
                    timestamp=datetime.utcnow(),
                    context={
                        "table_name": row['table_name'],
                        "date": date
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error monitoring ingestion performance: {e}")
        
        return metrics
    
    def _monitor_processing_performance(self, date: str) -> List[PerformanceMetric]:
        """Monitor data processing performance"""
        metrics = []
        
        try:
            # Query processing metrics
            query = f"""
            SELECT 
                'staging' as processing_stage,
                COUNT(*) as record_count,
                AVG(data_quality_score) as avg_quality_score
            FROM staging.smart_meter_readings 
            WHERE DATE(timestamp) = '{date}'
            
            UNION ALL
            
            SELECT 
                'marts' as processing_stage,
                COUNT(*) as record_count,
                AVG(data_quality_score) as avg_quality_score
            FROM marts.fct_smart_meter_analytics 
            WHERE DATE(reading_date) = '{date}'
            """
            
            df = pd.read_sql(query, self.snowflake_conn)
            
            for _, row in df.iterrows():
                # Processing volume metric
                metrics.append(PerformanceMetric(
                    metric_name=f"{row['processing_stage']}_processing_volume",
                    value=row['record_count'],
                    unit="records",
                    timestamp=datetime.utcnow(),
                    context={
                        "processing_stage": row['processing_stage'],
                        "date": date
                    }
                ))
                
                # Data quality metric
                if row['avg_quality_score'] is not None:
                    metrics.append(PerformanceMetric(
                        metric_name=f"{row['processing_stage']}_data_quality",
                        value=row['avg_quality_score'],
                        unit="score",
                        timestamp=datetime.utcnow(),
                        context={
                            "processing_stage": row['processing_stage'],
                            "date": date
                        }
                    ))
            
        except Exception as e:
            logger.error(f"Error monitoring processing performance: {e}")
        
        return metrics
    
    def _monitor_storage_performance(self, date: str) -> List[PerformanceMetric]:
        """Monitor storage performance"""
        metrics = []
        
        try:
            # Query storage metrics
            query = f"""
            SELECT 
                table_name,
                row_count,
                bytes,
                bytes / row_count as avg_row_size
            FROM information_schema.tables 
            WHERE table_schema IN ('raw', 'staging', 'marts', 'metrics')
            AND table_name IN (
                'smart_meter_readings', 'grid_status', 'weather_data',
                'stg_smart_meter_readings', 'stg_grid_status', 'stg_weather_data',
                'fct_smart_meter_analytics', 'dim_meters', 'daily_consumption_metrics'
            )
            """
            
            df = pd.read_sql(query, self.snowflake_conn)
            
            for _, row in df.iterrows():
                # Storage size metric
                metrics.append(PerformanceMetric(
                    metric_name=f"{row['table_name']}_storage_size",
                    value=row['bytes'],
                    unit="bytes",
                    timestamp=datetime.utcnow(),
                    context={
                        "table_name": row['table_name'],
                        "row_count": row['row_count']
                    }
                ))
                
                # Average row size metric
                if row['avg_row_size'] is not None:
                    metrics.append(PerformanceMetric(
                        metric_name=f"{row['table_name']}_avg_row_size",
                        value=row['avg_row_size'],
                        unit="bytes",
                        timestamp=datetime.utcnow(),
                        context={
                            "table_name": row['table_name']
                        }
                    ))
            
        except Exception as e:
            logger.error(f"Error monitoring storage performance: {e}")
        
        return metrics
    
    def _monitor_query_performance(self, date: str) -> List[PerformanceMetric]:
        """Monitor query performance"""
        metrics = []
        
        try:
            # Query query performance metrics
            query = f"""
            SELECT 
                query_id,
                query_text,
                total_elapsed_time,
                bytes_scanned,
                rows_produced,
                warehouse_name,
                start_time
            FROM snowflake.account_usage.query_history 
            WHERE start_time >= '{date} 00:00:00'
            AND start_time < '{date} 23:59:59'
            AND query_text ILIKE '%smart_meter%'
            ORDER BY total_elapsed_time DESC
            LIMIT 10
            """
            
            df = pd.read_sql(query, self.snowflake_conn)
            
            if not df.empty:
                # Average query execution time
                avg_execution_time = df['total_elapsed_time'].mean()
                metrics.append(PerformanceMetric(
                    metric_name="avg_query_execution_time",
                    value=avg_execution_time,
                    unit="milliseconds",
                    timestamp=datetime.utcnow(),
                    context={
                        "query_count": len(df),
                        "date": date
                    }
                ))
                
                # Total bytes scanned
                total_bytes_scanned = df['bytes_scanned'].sum()
                metrics.append(PerformanceMetric(
                    metric_name="total_bytes_scanned",
                    value=total_bytes_scanned,
                    unit="bytes",
                    timestamp=datetime.utcnow(),
                    context={
                        "query_count": len(df),
                        "date": date
                    }
                ))
                
                # Slowest query
                slowest_query = df.iloc[0]
                metrics.append(PerformanceMetric(
                    metric_name="slowest_query_time",
                    value=slowest_query['total_elapsed_time'],
                    unit="milliseconds",
                    timestamp=datetime.utcnow(),
                    context={
                        "query_id": slowest_query['query_id'],
                        "warehouse": slowest_query['warehouse_name']
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error monitoring query performance: {e}")
        
        return metrics
    
    def _process_metrics(self, metrics: List[PerformanceMetric]):
        """Process and store metrics"""
        for metric in metrics:
            # Store in history
            self.metrics_history.append(metric)
            
            # Send to Datadog
            self._send_datadog_metric(metric)
            
            logger.debug(f"Processed metric: {metric.metric_name} = {metric.value} {metric.unit}")
    
    def _send_datadog_metric(self, metric: PerformanceMetric):
        """Send metric to Datadog"""
        try:
            if 'datadog' in self.config['monitoring']:
                api.Metric.send(
                    metric=metric.metric_name,
                    points=[(int(metric.timestamp.timestamp()), metric.value)],
                    tags=[f"unit:{metric.unit}"]
                )
        except Exception as e:
            logger.error(f"Error sending Datadog metric: {e}")
    
    def _check_performance_issues(self, metrics: List[PerformanceMetric]):
        """Check for performance issues and generate alerts"""
        issues = []
        
        # Check for low ingestion rates
        ingestion_metrics = [m for m in metrics if m.metric_name.endswith('_ingestion_rate')]
        for metric in ingestion_metrics:
            if metric.value < 100:  # Less than 100 records per hour
                issues.append(f"Low ingestion rate for {metric.context.get('table_name', 'unknown')}: {metric.value:.2f} records/hour")
        
        # Check for high query execution times
        query_time_metrics = [m for m in metrics if m.metric_name == 'avg_query_execution_time']
        for metric in query_time_metrics:
            if metric.value > 30000:  # More than 30 seconds
                issues.append(f"High average query execution time: {metric.value:.2f} ms")
        
        # Check for large storage usage
        storage_metrics = [m for m in metrics if m.metric_name.endswith('_storage_size')]
        for metric in storage_metrics:
            if metric.value > 1000000000:  # More than 1GB
                issues.append(f"Large storage usage for {metric.context.get('table_name', 'unknown')}: {metric.value / 1000000000:.2f} GB")
        
        # Log issues
        if issues:
            logger.warning(f"Performance issues detected: {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("No performance issues detected")
    
    def _generate_performance_report(self, date: str, metrics: List[PerformanceMetric]):
        """Generate performance report"""
        try:
            # Group metrics by category
            ingestion_metrics = [m for m in metrics if 'ingestion' in m.metric_name]
            processing_metrics = [m for m in metrics if 'processing' in m.metric_name]
            storage_metrics = [m for m in metrics if 'storage' in m.metric_name]
            query_metrics = [m for m in metrics if 'query' in m.metric_name]
            
            # Calculate summary statistics
            report = {
                "date": date,
                "summary": {
                    "total_metrics": len(metrics),
                    "ingestion_metrics": len(ingestion_metrics),
                    "processing_metrics": len(processing_metrics),
                    "storage_metrics": len(storage_metrics),
                    "query_metrics": len(query_metrics)
                },
                "ingestion_performance": {
                    "avg_ingestion_rate": np.mean([m.value for m in ingestion_metrics if 'rate' in m.metric_name]) if ingestion_metrics else 0,
                    "total_records": sum([m.value for m in ingestion_metrics if 'count' in m.metric_name]) if ingestion_metrics else 0
                },
                "processing_performance": {
                    "avg_quality_score": np.mean([m.value for m in processing_metrics if 'quality' in m.metric_name]) if processing_metrics else 0,
                    "total_processed_records": sum([m.value for m in processing_metrics if 'volume' in m.metric_name]) if processing_metrics else 0
                },
                "storage_performance": {
                    "total_storage_bytes": sum([m.value for m in storage_metrics if 'size' in m.metric_name]) if storage_metrics else 0,
                    "avg_row_size": np.mean([m.value for m in storage_metrics if 'row_size' in m.metric_name]) if storage_metrics else 0
                },
                "query_performance": {
                    "avg_execution_time": np.mean([m.value for m in query_metrics if 'execution_time' in m.metric_name]) if query_metrics else 0,
                    "total_bytes_scanned": sum([m.value for m in query_metrics if 'bytes_scanned' in m.metric_name]) if query_metrics else 0
                },
                "metrics": [
                    {
                        "metric_name": metric.metric_name,
                        "value": metric.value,
                        "unit": metric.unit,
                        "timestamp": metric.timestamp.isoformat(),
                        "context": metric.context
                    }
                    for metric in metrics
                ],
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Store report in S3
            s3_key = f"performance_reports/{date}/performance_report.json"
            
            self.s3_client.put_object(
                Bucket=self.config['aws']['s3_bucket'],
                Key=s3_key,
                Body=json.dumps(report, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Performance report generated for {date}")
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")

def main():
    """Main function to run performance monitoring"""
    import sys
    
    monitor = PerformanceMonitor()
    
    # Get date from command line or use today
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y-%m-%d')
    
    # Run monitoring
    monitor.monitor_pipeline_performance(date)

if __name__ == "__main__":
    main()
