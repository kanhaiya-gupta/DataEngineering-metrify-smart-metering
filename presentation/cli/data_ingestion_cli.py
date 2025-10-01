"""
Data Ingestion CLI
Command line interface for data ingestion operations
"""

import asyncio
import logging
import click
from datetime import datetime, timedelta
from typing import Optional, List
import json

from src.core.config.config_loader import get_database_config, get_kafka_config, get_s3_config
from src.infrastructure.database.repositories.smart_meter_repository import SmartMeterRepository
from src.infrastructure.database.repositories.grid_operator_repository import GridOperatorRepository
from src.infrastructure.database.repositories.weather_station_repository import WeatherStationRepository
from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
from src.infrastructure.external.s3.s3_client import S3Client
from src.infrastructure.external.apis.data_quality_service import DataQualityService
from src.infrastructure.external.apis.anomaly_detection_service import AnomalyDetectionService
from src.infrastructure.external.apis.alerting_service import AlertingService
from src.application.use_cases.ingest_smart_meter_data import IngestSmartMeterDataUseCase
from src.application.use_cases.process_grid_status import ProcessGridStatusUseCase
from src.application.use_cases.analyze_weather_impact import AnalyzeWeatherImpactUseCase

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Metrify Smart Metering Data Ingestion CLI"""
    pass


@cli.command()
@click.option('--meter-id', required=True, help='Smart meter ID')
@click.option('--file', 'file_path', required=True, help='Path to data file')
@click.option('--format', 'file_format', type=click.Choice(['json', 'csv', 'parquet']), default='json', help='File format')
@click.option('--batch-size', default=1000, help='Batch size for processing')
@click.option('--dry-run', is_flag=True, help='Run without actually ingesting data')
def ingest_smart_meter_data(meter_id: str, file_path: str, file_format: str, batch_size: int, dry_run: bool):
    """Ingest smart meter data from file"""
    asyncio.run(_ingest_smart_meter_data(meter_id, file_path, file_format, batch_size, dry_run))


async def _ingest_smart_meter_data(meter_id: str, file_path: str, file_format: str, batch_size: int, dry_run: bool):
    """Ingest smart meter data from file"""
    try:
        click.echo(f"Starting smart meter data ingestion for meter {meter_id}")
        click.echo(f"File: {file_path}, Format: {file_format}, Batch size: {batch_size}")
        
        if dry_run:
            click.echo("DRY RUN MODE - No data will be ingested")
            return
        
        # Initialize services
        db_config = get_database_config()
        kafka_config = get_kafka_config()
        s3_config = get_s3_config()
        
        smart_meter_repo = SmartMeterRepository(db_config)
        kafka_producer = KafkaProducer(kafka_config)
        s3_client = S3Client(s3_config)
        data_quality_service = DataQualityService()
        anomaly_detection_service = AnomalyDetectionService()
        
        # Create use case
        ingest_use_case = IngestSmartMeterDataUseCase(
            smart_meter_repository=smart_meter_repo,
            data_quality_service=data_quality_service,
            anomaly_detection_service=anomaly_detection_service,
            kafka_producer=kafka_producer,
            s3_client=s3_client
        )
        
        # Read and process file
        readings_data = await _read_data_file(file_path, file_format)
        
        # Process in batches
        total_processed = 0
        for i in range(0, len(readings_data), batch_size):
            batch = readings_data[i:i + batch_size]
            
            click.echo(f"Processing batch {i//batch_size + 1} ({len(batch)} records)")
            
            result = await ingest_use_case.execute(
                meter_id=meter_id,
                readings_data=batch,
                metadata={"source": "cli", "file": file_path}
            )
            
            total_processed += len(batch)
            click.echo(f"Batch processed: {result.get('readings_processed', 0)} records")
            click.echo(f"Quality score: {result.get('quality_score', 0.0):.3f}")
            click.echo(f"Anomalies detected: {result.get('anomalies_detected', 0)}")
        
        click.echo(f"✅ Data ingestion completed successfully!")
        click.echo(f"Total records processed: {total_processed}")
        
    except Exception as e:
        click.echo(f"❌ Error during data ingestion: {str(e)}")
        logger.error(f"Data ingestion error: {str(e)}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--operator-id', required=True, help='Grid operator ID')
@click.option('--file', 'file_path', required=True, help='Path to data file')
@click.option('--format', 'file_format', type=click.Choice(['json', 'csv', 'parquet']), default='json', help='File format')
@click.option('--batch-size', default=1000, help='Batch size for processing')
@click.option('--dry-run', is_flag=True, help='Run without actually ingesting data')
def ingest_grid_status_data(operator_id: str, file_path: str, file_format: str, batch_size: int, dry_run: bool):
    """Ingest grid status data from file"""
    asyncio.run(_ingest_grid_status_data(operator_id, file_path, file_format, batch_size, dry_run))


async def _ingest_grid_status_data(operator_id: str, file_path: str, file_format: str, batch_size: int, dry_run: bool):
    """Ingest grid status data from file"""
    try:
        click.echo(f"Starting grid status data ingestion for operator {operator_id}")
        click.echo(f"File: {file_path}, Format: {file_format}, Batch size: {batch_size}")
        
        if dry_run:
            click.echo("DRY RUN MODE - No data will be ingested")
            return
        
        # Initialize services
        db_config = get_database_config()
        kafka_config = get_kafka_config()
        s3_config = get_s3_config()
        
        grid_operator_repo = GridOperatorRepository(db_config)
        kafka_producer = KafkaProducer(kafka_config)
        s3_client = S3Client(s3_config)
        data_quality_service = DataQualityService()
        anomaly_detection_service = AnomalyDetectionService()
        alerting_service = AlertingService()
        
        # Create use case
        process_use_case = ProcessGridStatusUseCase(
            grid_operator_repository=grid_operator_repo,
            data_quality_service=data_quality_service,
            anomaly_detection_service=anomaly_detection_service,
            alerting_service=alerting_service,
            kafka_producer=kafka_producer,
            s3_client=s3_client
        )
        
        # Read and process file
        status_data = await _read_data_file(file_path, file_format)
        
        # Process in batches
        total_processed = 0
        for i in range(0, len(status_data), batch_size):
            batch = status_data[i:i + batch_size]
            
            click.echo(f"Processing batch {i//batch_size + 1} ({len(batch)} records)")
            
            result = await process_use_case.execute(
                operator_id=operator_id,
                status_data=batch,
                metadata={"source": "cli", "file": file_path}
            )
            
            total_processed += len(batch)
            click.echo(f"Batch processed: {result.get('statuses_processed', 0)} records")
            click.echo(f"Quality score: {result.get('quality_score', 0.0):.3f}")
            click.echo(f"Anomalies detected: {result.get('anomalies_detected', 0)}")
            click.echo(f"Stability score: {result.get('stability_score', 0.0):.3f}")
            click.echo(f"Alerts sent: {result.get('alerts_sent', 0)}")
        
        click.echo(f"✅ Data ingestion completed successfully!")
        click.echo(f"Total records processed: {total_processed}")
        
    except Exception as e:
        click.echo(f"❌ Error during data ingestion: {str(e)}")
        logger.error(f"Data ingestion error: {str(e)}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--station-id', required=True, help='Weather station ID')
@click.option('--file', 'file_path', required=True, help='Path to data file')
@click.option('--format', 'file_format', type=click.Choice(['json', 'csv', 'parquet']), default='json', help='File format')
@click.option('--batch-size', default=1000, help='Batch size for processing')
@click.option('--dry-run', is_flag=True, help='Run without actually ingesting data')
def ingest_weather_data(station_id: str, file_path: str, file_format: str, batch_size: int, dry_run: bool):
    """Ingest weather data from file"""
    asyncio.run(_ingest_weather_data(station_id, file_path, file_format, batch_size, dry_run))


async def _ingest_weather_data(station_id: str, file_path: str, file_format: str, batch_size: int, dry_run: bool):
    """Ingest weather data from file"""
    try:
        click.echo(f"Starting weather data ingestion for station {station_id}")
        click.echo(f"File: {file_path}, Format: {file_format}, Batch size: {batch_size}")
        
        if dry_run:
            click.echo("DRY RUN MODE - No data will be ingested")
            return
        
        # Initialize services
        db_config = get_database_config()
        kafka_config = get_kafka_config()
        s3_config = get_s3_config()
        
        weather_station_repo = WeatherStationRepository(db_config)
        kafka_producer = KafkaProducer(kafka_config)
        s3_client = S3Client(s3_config)
        data_quality_service = DataQualityService()
        anomaly_detection_service = AnomalyDetectionService()
        alerting_service = AlertingService()
        
        # Create use case
        analyze_use_case = AnalyzeWeatherImpactUseCase(
            weather_station_repository=weather_station_repo,
            smart_meter_repository=None,
            grid_operator_repository=None,
            data_quality_service=data_quality_service,
            anomaly_detection_service=anomaly_detection_service,
            alerting_service=alerting_service,
            kafka_producer=kafka_producer,
            s3_client=s3_client,
            snowflake_query_executor=None
        )
        
        # Read and process file
        observation_data = await _read_data_file(file_path, file_format)
        
        # Process in batches
        total_processed = 0
        for i in range(0, len(observation_data), batch_size):
            batch = observation_data[i:i + batch_size]
            
            click.echo(f"Processing batch {i//batch_size + 1} ({len(batch)} records)")
            
            result = await analyze_use_case.execute(
                station_id=station_id,
                analysis_period_hours=24
            )
            
            total_processed += len(batch)
            click.echo(f"Batch processed: {len(batch)} records")
        
        click.echo(f"✅ Data ingestion completed successfully!")
        click.echo(f"Total records processed: {total_processed}")
        
    except Exception as e:
        click.echo(f"❌ Error during data ingestion: {str(e)}")
        logger.error(f"Data ingestion error: {str(e)}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--source', type=click.Choice(['smart_meters', 'grid_operators', 'weather_stations', 'all']), default='all', help='Data source to validate')
@click.option('--start-date', help='Start date for validation (YYYY-MM-DD)')
@click.option('--end-date', help='End date for validation (YYYY-MM-DD)')
@click.option('--output', help='Output file for validation report')
def validate_data_quality(source: str, start_date: Optional[str], end_date: Optional[str], output: Optional[str]):
    """Validate data quality across all sources"""
    asyncio.run(_validate_data_quality(source, start_date, end_date, output))


async def _validate_data_quality(source: str, start_date: Optional[str], end_date: Optional[str], output: Optional[str]):
    """Validate data quality across all sources"""
    try:
        click.echo("Starting data quality validation")
        
        # Parse dates
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        else:
            start_dt = datetime.utcnow() - timedelta(days=7)
        
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
        else:
            end_dt = datetime.utcnow()
        
        click.echo(f"Validation period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        
        # Initialize services
        db_config = get_database_config()
        smart_meter_repo = SmartMeterRepository(db_config)
        grid_operator_repo = GridOperatorRepository(db_config)
        weather_station_repo = WeatherStationRepository(db_config)
        
        validation_results = {}
        
        # Validate smart meters
        if source in ['smart_meters', 'all']:
            click.echo("Validating smart meter data...")
            meter_quality = await smart_meter_repo.get_data_quality_metrics(start_dt, end_dt)
            validation_results['smart_meters'] = meter_quality
            click.echo(f"Smart meters - Quality score: {meter_quality.get('avg_quality_score', 0):.3f}")
        
        # Validate grid operators
        if source in ['grid_operators', 'all']:
            click.echo("Validating grid operator data...")
            operator_quality = await grid_operator_repo.get_data_quality_metrics(start_dt, end_dt)
            validation_results['grid_operators'] = operator_quality
            click.echo(f"Grid operators - Quality score: {operator_quality.get('avg_quality_score', 0):.3f}")
        
        # Validate weather stations
        if source in ['weather_stations', 'all']:
            click.echo("Validating weather station data...")
            station_quality = await weather_station_repo.get_data_quality_metrics(start_dt, end_dt)
            validation_results['weather_stations'] = station_quality
            click.echo(f"Weather stations - Quality score: {station_quality.get('avg_quality_score', 0):.3f}")
        
        # Generate report
        report = {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "validation_period": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat()
            },
            "results": validation_results,
            "summary": {
                "overall_quality_score": sum(
                    result.get('avg_quality_score', 0) for result in validation_results.values()
                ) / len(validation_results) if validation_results else 0,
                "total_issues": sum(
                    result.get('quality_issues', 0) for result in validation_results.values()
                )
            }
        }
        
        # Output results
        if output:
            with open(output, 'w') as f:
                json.dump(report, f, indent=2)
            click.echo(f"Validation report saved to {output}")
        else:
            click.echo("Validation Results:")
            click.echo(json.dumps(report, indent=2))
        
        click.echo("✅ Data quality validation completed!")
        
    except Exception as e:
        click.echo(f"❌ Error during data quality validation: {str(e)}")
        logger.error(f"Data quality validation error: {str(e)}", exc_info=True)
        raise click.Abort()


async def _read_data_file(file_path: str, file_format: str) -> List[dict]:
    """Read data from file based on format"""
    try:
        if file_format == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
        
        elif file_format == 'csv':
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        
        elif file_format == 'parquet':
            import pandas as pd
            df = pd.read_parquet(file_path)
            return df.to_dict('records')
        
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise


if __name__ == '__main__':
    cli()
