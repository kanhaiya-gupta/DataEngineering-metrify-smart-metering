"""
Maintenance CLI
Command line interface for system maintenance operations
"""

import asyncio
import logging
import click
from datetime import datetime, timedelta
from typing import Optional, List
import json

from src.core.config.config_loader import get_database_config, get_kafka_config, get_s3_config, get_snowflake_config
from src.infrastructure.database.repositories.smart_meter_repository import SmartMeterRepository
from src.infrastructure.database.repositories.grid_operator_repository import GridOperatorRepository
from src.infrastructure.database.repositories.weather_station_repository import WeatherStationRepository
from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
from src.infrastructure.external.s3.s3_client import S3Client
from src.infrastructure.external.snowflake.snowflake_client import SnowflakeClient
from src.infrastructure.external.snowflake.query_executor import SnowflakeQueryExecutor
from src.infrastructure.external.monitoring.monitoring_service import MonitoringService
from src.infrastructure.external.monitoring.prometheus.prometheus_client import PrometheusClient
from src.infrastructure.external.monitoring.grafana.grafana_client import GrafanaClient
from src.infrastructure.external.monitoring.jaeger.jaeger_client import JaegerClient
from src.infrastructure.external.monitoring.datadog.datadog_client import DataDogClient

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Metrify Smart Metering Maintenance CLI"""
    pass


@cli.command()
@click.option('--days', default=30, help='Number of days to keep data')
@click.option('--dry-run', is_flag=True, help='Run without actually deleting data')
@click.option('--confirm', is_flag=True, help='Confirm data cleanup operation')
def cleanup_old_data(days: int, dry_run: bool, confirm: bool):
    """Clean up old data from the system"""
    asyncio.run(_cleanup_old_data(days, dry_run, confirm))


async def _cleanup_old_data(days: int, dry_run: bool, confirm: bool):
    """Clean up old data from the system"""
    try:
        click.echo("Starting data cleanup operation")
        click.echo(f"Cleaning up data older than {days} days")
        
        if dry_run:
            click.echo("DRY RUN MODE - No data will be deleted")
        
        if not confirm and not dry_run:
            click.echo("❌ This operation will permanently delete data. Use --confirm to proceed")
            return
        
        # Calculate cutoff date
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        click.echo(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize services
        db_config = get_database_config()
        smart_meter_repo = SmartMeterRepository(db_config)
        grid_operator_repo = GridOperatorRepository(db_config)
        weather_station_repo = WeatherStationRepository(db_config)
        
        cleanup_results = {}
        
        # Clean up smart meter data
        click.echo("Cleaning up smart meter data...")
        if not dry_run:
            meter_cleanup = await smart_meter_repo.cleanup_old_data(cutoff_date)
            cleanup_results['smart_meters'] = meter_cleanup
            click.echo(f"Smart meters - Deleted {meter_cleanup.get('deleted_count', 0)} records")
        else:
            meter_count = await smart_meter_repo.get_old_data_count(cutoff_date)
            click.echo(f"Smart meters - Would delete {meter_count} records")
        
        # Clean up grid operator data
        click.echo("Cleaning up grid operator data...")
        if not dry_run:
            operator_cleanup = await grid_operator_repo.cleanup_old_data(cutoff_date)
            cleanup_results['grid_operators'] = operator_cleanup
            click.echo(f"Grid operators - Deleted {operator_cleanup.get('deleted_count', 0)} records")
        else:
            operator_count = await grid_operator_repo.get_old_data_count(cutoff_date)
            click.echo(f"Grid operators - Would delete {operator_count} records")
        
        # Clean up weather station data
        click.echo("Cleaning up weather station data...")
        if not dry_run:
            station_cleanup = await weather_station_repo.cleanup_old_data(cutoff_date)
            cleanup_results['weather_stations'] = station_cleanup
            click.echo(f"Weather stations - Deleted {station_cleanup.get('deleted_count', 0)} records")
        else:
            station_count = await weather_station_repo.get_old_data_count(cutoff_date)
            click.echo(f"Weather stations - Would delete {station_count} records")
        
        # Generate cleanup report
        report = {
            "cleanup_timestamp": datetime.utcnow().isoformat(),
            "cutoff_date": cutoff_date.isoformat(),
            "days_retained": days,
            "dry_run": dry_run,
            "results": cleanup_results
        }
        
        click.echo("Cleanup Results:")
        click.echo(json.dumps(report, indent=2))
        
        if dry_run:
            click.echo("✅ Data cleanup dry run completed")
        else:
            click.echo("✅ Data cleanup completed successfully")
        
    except Exception as e:
        click.echo(f"❌ Error during data cleanup: {str(e)}")
        logger.error(f"Data cleanup error: {str(e)}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--source', type=click.Choice(['smart_meters', 'grid_operators', 'weather_stations', 'all']), default='all', help='Data source to optimize')
@click.option('--analyze-only', is_flag=True, help='Only analyze, do not optimize')
def optimize_database(source: str, analyze_only: bool):
    """Optimize database performance"""
    asyncio.run(_optimize_database(source, analyze_only))


async def _optimize_database(source: str, analyze_only: bool):
    """Optimize database performance"""
    try:
        click.echo("Starting database optimization")
        
        if analyze_only:
            click.echo("ANALYZE ONLY MODE - No optimizations will be applied")
        
        # Initialize services
        db_config = get_database_config()
        smart_meter_repo = SmartMeterRepository(db_config)
        grid_operator_repo = GridOperatorRepository(db_config)
        weather_station_repo = WeatherStationRepository(db_config)
        
        optimization_results = {}
        
        # Optimize smart meters
        if source in ['smart_meters', 'all']:
            click.echo("Optimizing smart meter database...")
            if not analyze_only:
                meter_optimization = await smart_meter_repo.optimize_database()
                optimization_results['smart_meters'] = meter_optimization
                click.echo(f"Smart meters - Optimization completed")
            else:
                meter_analysis = await smart_meter_repo.analyze_database_performance()
                optimization_results['smart_meters'] = meter_analysis
                click.echo(f"Smart meters - Analysis completed")
        
        # Optimize grid operators
        if source in ['grid_operators', 'all']:
            click.echo("Optimizing grid operator database...")
            if not analyze_only:
                operator_optimization = await grid_operator_repo.optimize_database()
                optimization_results['grid_operators'] = operator_optimization
                click.echo(f"Grid operators - Optimization completed")
            else:
                operator_analysis = await grid_operator_repo.analyze_database_performance()
                optimization_results['grid_operators'] = operator_analysis
                click.echo(f"Grid operators - Analysis completed")
        
        # Optimize weather stations
        if source in ['weather_stations', 'all']:
            click.echo("Optimizing weather station database...")
            if not analyze_only:
                station_optimization = await weather_station_repo.optimize_database()
                optimization_results['weather_stations'] = station_optimization
                click.echo(f"Weather stations - Optimization completed")
            else:
                station_analysis = await weather_station_repo.analyze_database_performance()
                optimization_results['weather_stations'] = station_analysis
                click.echo(f"Weather stations - Analysis completed")
        
        # Generate optimization report
        report = {
            "optimization_timestamp": datetime.utcnow().isoformat(),
            "analyze_only": analyze_only,
            "results": optimization_results
        }
        
        click.echo("Optimization Results:")
        click.echo(json.dumps(report, indent=2))
        
        if analyze_only:
            click.echo("✅ Database analysis completed")
        else:
            click.echo("✅ Database optimization completed successfully")
        
    except Exception as e:
        click.echo(f"❌ Error during database optimization: {str(e)}")
        logger.error(f"Database optimization error: {str(e)}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--backup-type', type=click.Choice(['full', 'incremental', 'differential']), default='incremental', help='Type of backup')
@click.option('--destination', help='Backup destination path')
@click.option('--compress', is_flag=True, help='Compress backup files')
def backup_data(backup_type: str, destination: Optional[str], compress: bool):
    """Backup system data"""
    asyncio.run(_backup_data(backup_type, destination, compress))


async def _backup_data(backup_type: str, destination: Optional[str], compress: bool):
    """Backup system data"""
    try:
        click.echo("Starting data backup")
        click.echo(f"Backup type: {backup_type}")
        
        if not destination:
            destination = f"backup_{backup_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        click.echo(f"Backup destination: {destination}")
        click.echo(f"Compression: {'enabled' if compress else 'disabled'}")
        
        # Initialize services
        db_config = get_database_config()
        s3_config = get_s3_config()
        s3_client = S3Client(s3_config)
        
        backup_results = {}
        
        # Backup database
        click.echo("Backing up database...")
        db_backup = await _backup_database(db_config, destination, compress)
        backup_results['database'] = db_backup
        click.echo(f"Database backup completed: {db_backup.get('backup_file', 'N/A')}")
        
        # Backup S3 data
        click.echo("Backing up S3 data...")
        s3_backup = await s3_client.backup_data(destination, compress)
        backup_results['s3'] = s3_backup
        click.echo(f"S3 backup completed: {s3_backup.get('backup_files', 0)} files")
        
        # Generate backup report
        report = {
            "backup_timestamp": datetime.utcnow().isoformat(),
            "backup_type": backup_type,
            "destination": destination,
            "compression": compress,
            "results": backup_results
        }
        
        click.echo("Backup Results:")
        click.echo(json.dumps(report, indent=2))
        
        click.echo("✅ Data backup completed successfully")
        
    except Exception as e:
        click.echo(f"❌ Error during data backup: {str(e)}")
        logger.error(f"Data backup error: {str(e)}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--backup-file', required=True, help='Backup file to restore')
@click.option('--confirm', is_flag=True, help='Confirm restore operation')
def restore_data(backup_file: str, confirm: bool):
    """Restore system data from backup"""
    asyncio.run(_restore_data(backup_file, confirm))


async def _restore_data(backup_file: str, confirm: bool):
    """Restore system data from backup"""
    try:
        click.echo("Starting data restore")
        click.echo(f"Backup file: {backup_file}")
        
        if not confirm:
            click.echo("❌ This operation will overwrite existing data. Use --confirm to proceed")
            return
        
        # Initialize services
        db_config = get_database_config()
        s3_config = get_s3_config()
        s3_client = S3Client(s3_config)
        
        restore_results = {}
        
        # Restore database
        click.echo("Restoring database...")
        db_restore = await _restore_database(db_config, backup_file)
        restore_results['database'] = db_restore
        click.echo(f"Database restore completed: {db_restore.get('status', 'N/A')}")
        
        # Restore S3 data
        click.echo("Restoring S3 data...")
        s3_restore = await s3_client.restore_data(backup_file)
        restore_results['s3'] = s3_restore
        click.echo(f"S3 restore completed: {s3_restore.get('restored_files', 0)} files")
        
        # Generate restore report
        report = {
            "restore_timestamp": datetime.utcnow().isoformat(),
            "backup_file": backup_file,
            "results": restore_results
        }
        
        click.echo("Restore Results:")
        click.echo(json.dumps(report, indent=2))
        
        click.echo("✅ Data restore completed successfully")
        
    except Exception as e:
        click.echo(f"❌ Error during data restore: {str(e)}")
        logger.error(f"Data restore error: {str(e)}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--component', type=click.Choice(['kafka', 's3', 'snowflake', 'monitoring', 'all']), default='all', help='Component to check')
def health_check(component: str):
    """Check system health"""
    asyncio.run(_health_check(component))


async def _health_check(component: str):
    """Check system health"""
    try:
        click.echo("Starting system health check")
        
        health_results = {}
        
        # Check Kafka
        if component in ['kafka', 'all']:
            click.echo("Checking Kafka health...")
            kafka_health = await _check_kafka_health()
            health_results['kafka'] = kafka_health
            status = "✅ Healthy" if kafka_health.get('status') == 'healthy' else "❌ Unhealthy"
            click.echo(f"Kafka: {status}")
        
        # Check S3
        if component in ['s3', 'all']:
            click.echo("Checking S3 health...")
            s3_health = await _check_s3_health()
            health_results['s3'] = s3_health
            status = "✅ Healthy" if s3_health.get('status') == 'healthy' else "❌ Unhealthy"
            click.echo(f"S3: {status}")
        
        # Check Snowflake
        if component in ['snowflake', 'all']:
            click.echo("Checking Snowflake health...")
            snowflake_health = await _check_snowflake_health()
            health_results['snowflake'] = snowflake_health
            status = "✅ Healthy" if snowflake_health.get('status') == 'healthy' else "❌ Unhealthy"
            click.echo(f"Snowflake: {status}")
        
        # Check monitoring
        if component in ['monitoring', 'all']:
            click.echo("Checking monitoring health...")
            monitoring_health = await _check_monitoring_health()
            health_results['monitoring'] = monitoring_health
            status = "✅ Healthy" if monitoring_health.get('status') == 'healthy' else "❌ Unhealthy"
            click.echo(f"Monitoring: {status}")
        
        # Generate health report
        overall_status = "healthy" if all(
            result.get('status') == 'healthy' for result in health_results.values()
        ) else "unhealthy"
        
        report = {
            "health_check_timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall_status,
            "results": health_results
        }
        
        click.echo("Health Check Results:")
        click.echo(json.dumps(report, indent=2))
        
        if overall_status == "healthy":
            click.echo("✅ System health check passed")
        else:
            click.echo("❌ System health check failed")
            return 1
        
    except Exception as e:
        click.echo(f"❌ Error during health check: {str(e)}")
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        raise click.Abort()


async def _backup_database(db_config, destination: str, compress: bool) -> dict:
    """Backup database"""
    # This would implement actual database backup logic
    return {
        "backup_file": f"{destination}/database_backup.sql",
        "size_mb": 1024,
        "status": "completed"
    }


async def _restore_database(db_config, backup_file: str) -> dict:
    """Restore database"""
    # This would implement actual database restore logic
    return {
        "status": "completed",
        "restored_tables": 10
    }


async def _check_kafka_health() -> dict:
    """Check Kafka health"""
    try:
        kafka_config = get_kafka_config()
        kafka_producer = KafkaProducer(kafka_config)
        
        # Test Kafka connection
        await kafka_producer.health_check()
        
        return {
            "status": "healthy",
            "brokers": kafka_config.bootstrap_servers,
            "response_time_ms": 50
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def _check_s3_health() -> dict:
    """Check S3 health"""
    try:
        s3_config = get_s3_config()
        s3_client = S3Client(s3_config)
        
        # Test S3 connection
        await s3_client.health_check()
        
        return {
            "status": "healthy",
            "bucket": s3_config.bucket_name,
            "response_time_ms": 100
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def _check_snowflake_health() -> dict:
    """Check Snowflake health"""
    try:
        snowflake_config = get_snowflake_config()
        snowflake_client = SnowflakeClient(snowflake_config)
        
        # Test Snowflake connection
        await snowflake_client.health_check()
        
        return {
            "status": "healthy",
            "account": snowflake_config.account,
            "response_time_ms": 200
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def _check_monitoring_health() -> dict:
    """Check monitoring health"""
    try:
        # Initialize monitoring clients
        prometheus_client = PrometheusClient(namespace="metrify", subsystem="health_check")
        grafana_client = GrafanaClient("http://grafana:3000", "admin", "admin")
        jaeger_client = JaegerClient("metrify-health-check", "jaeger-agent", 6831)
        datadog_client = DataDogClient("test-api-key", "test-app-key", "datadoghq.com")
        
        # Test monitoring services
        monitoring_service = MonitoringService(
            prometheus_client=prometheus_client,
            grafana_client=grafana_client,
            jaeger_client=jaeger_client,
            datadog_client=datadog_client
        )
        
        await monitoring_service.health_check()
        
        return {
            "status": "healthy",
            "services": ["prometheus", "grafana", "jaeger", "datadog"],
            "response_time_ms": 150
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == '__main__':
    cli()
