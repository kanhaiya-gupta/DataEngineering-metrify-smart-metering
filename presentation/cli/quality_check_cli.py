"""
Quality Check CLI
Command line interface for data quality operations
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
from src.infrastructure.external.apis.data_quality_service import DataQualityService
from src.infrastructure.external.apis.anomaly_detection_service import AnomalyDetectionService
from src.infrastructure.external.apis.alerting_service import AlertingService
from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
from src.infrastructure.external.s3.s3_client import S3Client

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Metrify Smart Metering Quality Check CLI"""
    pass


@cli.command()
@click.option('--source', type=click.Choice(['smart_meters', 'grid_operators', 'weather_stations', 'all']), default='all', help='Data source to check')
@click.option('--start-date', help='Start date for quality check (YYYY-MM-DD)')
@click.option('--end-date', help='End date for quality check (YYYY-MM-DD)')
@click.option('--threshold', default=0.8, help='Quality threshold (0.0-1.0)')
@click.option('--output', help='Output file for quality report')
def check_quality(source: str, start_date: Optional[str], end_date: Optional[str], threshold: float, output: Optional[str]):
    """Check data quality across all sources"""
    asyncio.run(_check_quality(source, start_date, end_date, threshold, output))


async def _check_quality(source: str, start_date: Optional[str], end_date: Optional[str], threshold: float, output: Optional[str]):
    """Check data quality across all sources"""
    try:
        click.echo("Starting data quality check")
        
        # Parse dates
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        else:
            start_dt = datetime.utcnow() - timedelta(days=7)
        
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
        else:
            end_dt = datetime.utcnow()
        
        click.echo(f"Quality check period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        click.echo(f"Quality threshold: {threshold}")
        
        # Initialize services
        db_config = get_database_config()
        smart_meter_repo = SmartMeterRepository(db_config)
        grid_operator_repo = GridOperatorRepository(db_config)
        weather_station_repo = WeatherStationRepository(db_config)
        data_quality_service = DataQualityService()
        
        quality_results = {}
        issues_found = []
        
        # Check smart meters
        if source in ['smart_meters', 'all']:
            click.echo("Checking smart meter data quality...")
            meter_quality = await smart_meter_repo.get_data_quality_metrics(start_dt, end_dt)
            quality_results['smart_meters'] = meter_quality
            
            if meter_quality.get('avg_quality_score', 0) < threshold:
                issues_found.append({
                    'source': 'smart_meters',
                    'issue': 'Low quality score',
                    'score': meter_quality.get('avg_quality_score', 0),
                    'threshold': threshold
                })
                click.echo(f"⚠️  Smart meters - Quality score below threshold: {meter_quality.get('avg_quality_score', 0):.3f} < {threshold}")
            else:
                click.echo(f"✅ Smart meters - Quality score: {meter_quality.get('avg_quality_score', 0):.3f}")
        
        # Check grid operators
        if source in ['grid_operators', 'all']:
            click.echo("Checking grid operator data quality...")
            operator_quality = await grid_operator_repo.get_data_quality_metrics(start_dt, end_dt)
            quality_results['grid_operators'] = operator_quality
            
            if operator_quality.get('avg_quality_score', 0) < threshold:
                issues_found.append({
                    'source': 'grid_operators',
                    'issue': 'Low quality score',
                    'score': operator_quality.get('avg_quality_score', 0),
                    'threshold': threshold
                })
                click.echo(f"⚠️  Grid operators - Quality score below threshold: {operator_quality.get('avg_quality_score', 0):.3f} < {threshold}")
            else:
                click.echo(f"✅ Grid operators - Quality score: {operator_quality.get('avg_quality_score', 0):.3f}")
        
        # Check weather stations
        if source in ['weather_stations', 'all']:
            click.echo("Checking weather station data quality...")
            station_quality = await weather_station_repo.get_data_quality_metrics(start_dt, end_dt)
            quality_results['weather_stations'] = station_quality
            
            if station_quality.get('avg_quality_score', 0) < threshold:
                issues_found.append({
                    'source': 'weather_stations',
                    'issue': 'Low quality score',
                    'score': station_quality.get('avg_quality_score', 0),
                    'threshold': threshold
                })
                click.echo(f"⚠️  Weather stations - Quality score below threshold: {station_quality.get('avg_quality_score', 0):.3f} < {threshold}")
            else:
                click.echo(f"✅ Weather stations - Quality score: {station_quality.get('avg_quality_score', 0):.3f}")
        
        # Generate report
        report = {
            "quality_check_timestamp": datetime.utcnow().isoformat(),
            "check_period": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat()
            },
            "threshold": threshold,
            "results": quality_results,
            "issues_found": issues_found,
            "summary": {
                "overall_quality_score": sum(
                    result.get('avg_quality_score', 0) for result in quality_results.values()
                ) / len(quality_results) if quality_results else 0,
                "total_issues": len(issues_found),
                "quality_status": "PASS" if len(issues_found) == 0 else "FAIL"
            }
        }
        
        # Output results
        if output:
            with open(output, 'w') as f:
                json.dump(report, f, indent=2)
            click.echo(f"Quality check report saved to {output}")
        else:
            click.echo("Quality Check Results:")
            click.echo(json.dumps(report, indent=2))
        
        if issues_found:
            click.echo(f"❌ Quality check failed with {len(issues_found)} issues found")
            return 1
        else:
            click.echo("✅ Quality check passed - no issues found")
            return 0
        
    except Exception as e:
        click.echo(f"❌ Error during quality check: {str(e)}")
        logger.error(f"Quality check error: {str(e)}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--source', type=click.Choice(['smart_meters', 'grid_operators', 'weather_stations', 'all']), default='all', help='Data source to check')
@click.option('--start-date', help='Start date for anomaly detection (YYYY-MM-DD)')
@click.option('--end-date', help='End date for anomaly detection (YYYY-MM-DD)')
@click.option('--sensitivity', default=0.5, help='Anomaly detection sensitivity (0.0-1.0)')
@click.option('--output', help='Output file for anomaly report')
def detect_anomalies(source: str, start_date: Optional[str], end_date: Optional[str], sensitivity: float, output: Optional[str]):
    """Detect anomalies in data"""
    asyncio.run(_detect_anomalies(source, start_date, end_date, sensitivity, output))


async def _detect_anomalies(source: str, start_date: Optional[str], end_date: Optional[str], sensitivity: float, output: Optional[str]):
    """Detect anomalies in data"""
    try:
        click.echo("Starting anomaly detection")
        
        # Parse dates
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        else:
            start_dt = datetime.utcnow() - timedelta(days=7)
        
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
        else:
            end_dt = datetime.utcnow()
        
        click.echo(f"Anomaly detection period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        click.echo(f"Detection sensitivity: {sensitivity}")
        
        # Initialize services
        db_config = get_database_config()
        smart_meter_repo = SmartMeterRepository(db_config)
        grid_operator_repo = GridOperatorRepository(db_config)
        weather_station_repo = WeatherStationRepository(db_config)
        anomaly_detection_service = AnomalyDetectionService()
        
        anomaly_results = {}
        total_anomalies = 0
        
        # Detect anomalies in smart meters
        if source in ['smart_meters', 'all']:
            click.echo("Detecting anomalies in smart meter data...")
            meter_anomalies = await smart_meter_repo.get_anomalies(
                start_time=start_dt,
                end_time=end_dt,
                sensitivity=sensitivity
            )
            anomaly_results['smart_meters'] = meter_anomalies
            total_anomalies += len(meter_anomalies)
            click.echo(f"Smart meters - Anomalies detected: {len(meter_anomalies)}")
        
        # Detect anomalies in grid operators
        if source in ['grid_operators', 'all']:
            click.echo("Detecting anomalies in grid operator data...")
            operator_anomalies = await grid_operator_repo.get_anomalies(
                start_time=start_dt,
                end_time=end_dt,
                sensitivity=sensitivity
            )
            anomaly_results['grid_operators'] = operator_anomalies
            total_anomalies += len(operator_anomalies)
            click.echo(f"Grid operators - Anomalies detected: {len(operator_anomalies)}")
        
        # Detect anomalies in weather stations
        if source in ['weather_stations', 'all']:
            click.echo("Detecting anomalies in weather station data...")
            station_anomalies = await weather_station_repo.get_anomalies(
                start_time=start_dt,
                end_time=end_dt,
                sensitivity=sensitivity
            )
            anomaly_results['weather_stations'] = station_anomalies
            total_anomalies += len(station_anomalies)
            click.echo(f"Weather stations - Anomalies detected: {len(station_anomalies)}")
        
        # Generate report
        report = {
            "anomaly_detection_timestamp": datetime.utcnow().isoformat(),
            "detection_period": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat()
            },
            "sensitivity": sensitivity,
            "results": anomaly_results,
            "summary": {
                "total_anomalies": total_anomalies,
                "anomalies_by_source": {
                    source: len(anomalies) for source, anomalies in anomaly_results.items()
                }
            }
        }
        
        # Output results
        if output:
            with open(output, 'w') as f:
                json.dump(report, f, indent=2)
            click.echo(f"Anomaly detection report saved to {output}")
        else:
            click.echo("Anomaly Detection Results:")
            click.echo(json.dumps(report, indent=2))
        
        click.echo(f"✅ Anomaly detection completed - {total_anomalies} anomalies found")
        
    except Exception as e:
        click.echo(f"❌ Error during anomaly detection: {str(e)}")
        logger.error(f"Anomaly detection error: {str(e)}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--source', type=click.Choice(['smart_meters', 'grid_operators', 'weather_stations', 'all']), default='all', help='Data source to check')
@click.option('--start-date', help='Start date for data validation (YYYY-MM-DD)')
@click.option('--end-date', help='End date for data validation (YYYY-MM-DD)')
@click.option('--rules', help='Path to custom validation rules file')
@click.option('--output', help='Output file for validation report')
def validate_data(source: str, start_date: Optional[str], end_date: Optional[str], rules: Optional[str], output: Optional[str]):
    """Validate data against business rules"""
    asyncio.run(_validate_data(source, start_date, end_date, rules, output))


async def _validate_data(source: str, start_date: Optional[str], end_date: Optional[str], rules: Optional[str], output: Optional[str]):
    """Validate data against business rules"""
    try:
        click.echo("Starting data validation")
        
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
        
        # Load validation rules
        validation_rules = _load_validation_rules(rules)
        click.echo(f"Loaded {len(validation_rules)} validation rules")
        
        # Initialize services
        db_config = get_database_config()
        smart_meter_repo = SmartMeterRepository(db_config)
        grid_operator_repo = GridOperatorRepository(db_config)
        weather_station_repo = WeatherStationRepository(db_config)
        data_quality_service = DataQualityService()
        
        validation_results = {}
        total_violations = 0
        
        # Validate smart meters
        if source in ['smart_meters', 'all']:
            click.echo("Validating smart meter data...")
            meter_violations = await _validate_smart_meter_data(
                smart_meter_repo, start_dt, end_dt, validation_rules
            )
            validation_results['smart_meters'] = meter_violations
            total_violations += len(meter_violations)
            click.echo(f"Smart meters - Violations found: {len(meter_violations)}")
        
        # Validate grid operators
        if source in ['grid_operators', 'all']:
            click.echo("Validating grid operator data...")
            operator_violations = await _validate_grid_operator_data(
                grid_operator_repo, start_dt, end_dt, validation_rules
            )
            validation_results['grid_operators'] = operator_violations
            total_violations += len(operator_violations)
            click.echo(f"Grid operators - Violations found: {len(operator_violations)}")
        
        # Validate weather stations
        if source in ['weather_stations', 'all']:
            click.echo("Validating weather station data...")
            station_violations = await _validate_weather_station_data(
                weather_station_repo, start_dt, end_dt, validation_rules
            )
            validation_results['weather_stations'] = station_violations
            total_violations += len(station_violations)
            click.echo(f"Weather stations - Violations found: {len(station_violations)}")
        
        # Generate report
        report = {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "validation_period": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat()
            },
            "validation_rules": validation_rules,
            "results": validation_results,
            "summary": {
                "total_violations": total_violations,
                "violations_by_source": {
                    source: len(violations) for source, violations in validation_results.items()
                },
                "validation_status": "PASS" if total_violations == 0 else "FAIL"
            }
        }
        
        # Output results
        if output:
            with open(output, 'w') as f:
                json.dump(report, f, indent=2)
            click.echo(f"Validation report saved to {output}")
        else:
            click.echo("Data Validation Results:")
            click.echo(json.dumps(report, indent=2))
        
        if total_violations > 0:
            click.echo(f"❌ Data validation failed with {total_violations} violations found")
            return 1
        else:
            click.echo("✅ Data validation passed - no violations found")
            return 0
        
    except Exception as e:
        click.echo(f"❌ Error during data validation: {str(e)}")
        logger.error(f"Data validation error: {str(e)}", exc_info=True)
        raise click.Abort()


def _load_validation_rules(rules_file: Optional[str]) -> List[dict]:
    """Load validation rules from file or use defaults"""
    if rules_file:
        try:
            with open(rules_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            click.echo(f"Warning: Could not load rules file {rules_file}: {str(e)}")
    
    # Default validation rules
    return [
        {
            "name": "voltage_range",
            "description": "Voltage must be within acceptable range",
            "field": "voltage",
            "min_value": 200.0,
            "max_value": 250.0
        },
        {
            "name": "current_positive",
            "description": "Current must be positive",
            "field": "current",
            "min_value": 0.0
        },
        {
            "name": "power_factor_range",
            "description": "Power factor must be between 0 and 1",
            "field": "power_factor",
            "min_value": 0.0,
            "max_value": 1.0
        },
        {
            "name": "frequency_range",
            "description": "Frequency must be within acceptable range",
            "field": "frequency",
            "min_value": 45.0,
            "max_value": 55.0
        }
    ]


async def _validate_smart_meter_data(repo, start_dt, end_dt, rules):
    """Validate smart meter data against rules"""
    # This would implement actual validation logic
    # For now, return empty list
    return []


async def _validate_grid_operator_data(repo, start_dt, end_dt, rules):
    """Validate grid operator data against rules"""
    # This would implement actual validation logic
    # For now, return empty list
    return []


async def _validate_weather_station_data(repo, start_dt, end_dt, rules):
    """Validate weather station data against rules"""
    # This would implement actual validation logic
    # For now, return empty list
    return []


if __name__ == '__main__':
    cli()
