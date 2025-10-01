"""
Analytics API Endpoints
REST API endpoints for analytics and reporting
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta

from src.application.use_cases.analyze_weather_impact import AnalyzeWeatherImpactUseCase
from src.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
from src.core.config.config_loader import get_database_config, get_kafka_config, get_s3_config, get_snowflake_config
from src.infrastructure.database.repositories.smart_meter_repository import SmartMeterRepository
from src.infrastructure.database.repositories.grid_operator_repository import GridOperatorRepository
from src.infrastructure.database.repositories.weather_station_repository import WeatherStationRepository
from src.infrastructure.external.apis.data_quality_service import DataQualityService
from src.infrastructure.external.apis.anomaly_detection_service import AnomalyDetectionService
from src.infrastructure.external.apis.alerting_service import AlertingService
from src.infrastructure.external.kafka.kafka_producer import KafkaProducer
from src.infrastructure.external.s3.s3_client import S3Client
from src.infrastructure.external.snowflake.query_executor import SnowflakeQueryExecutor
from src.infrastructure.external.snowflake.snowflake_client import SnowflakeClient

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection
def get_smart_meter_repository():
    """Get smart meter repository instance"""
    db_config = get_database_config()
    return SmartMeterRepository(db_config)

def get_grid_operator_repository():
    """Get grid operator repository instance"""
    db_config = get_database_config()
    return GridOperatorRepository(db_config)

def get_weather_station_repository():
    """Get weather station repository instance"""
    db_config = get_database_config()
    return WeatherStationRepository(db_config)

def get_data_quality_service():
    """Get data quality service instance"""
    return DataQualityService()

def get_anomaly_detection_service():
    """Get anomaly detection service instance"""
    return AnomalyDetectionService()

def get_alerting_service():
    """Get alerting service instance"""
    return AlertingService()

def get_kafka_producer():
    """Get Kafka producer instance"""
    kafka_config = get_kafka_config()
    return KafkaProducer(kafka_config)

def get_s3_client():
    """Get S3 client instance"""
    s3_config = get_s3_config()
    return S3Client(s3_config)

def get_snowflake_query_executor():
    """Get Snowflake query executor instance"""
    snowflake_config = get_snowflake_config()
    return SnowflakeQueryExecutor(snowflake_config)

def get_snowflake_client():
    """Get Snowflake client instance"""
    snowflake_config = get_snowflake_config()
    return SnowflakeClient(snowflake_config)


@router.get("/overview", response_model=Dict[str, Any])
async def get_analytics_overview(
    period: str = Query("24h", description="Analysis period"),
    smart_meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository),
    grid_operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository),
    weather_station_repo: WeatherStationRepository = Depends(get_weather_station_repository)
):
    """Get analytics overview for the entire system"""
    try:
        # Parse period
        if period == "24h":
            hours = 24
        elif period == "7d":
            hours = 168
        elif period == "30d":
            hours = 720
        else:
            hours = 24
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get system statistics
        total_meters = await smart_meter_repo.get_total_count()
        active_meters = await smart_meter_repo.get_active_count()
        total_operators = await grid_operator_repo.get_total_count()
        active_operators = await grid_operator_repo.get_active_count()
        total_stations = await weather_station_repo.get_total_count()
        active_stations = await weather_station_repo.get_active_count()
        
        # Get recent data counts
        recent_readings = await smart_meter_repo.get_readings_count_in_period(start_time, end_time)
        recent_statuses = await grid_operator_repo.get_statuses_count_in_period(start_time, end_time)
        recent_observations = await weather_station_repo.get_observations_count_in_period(start_time, end_time)
        
        # Calculate data quality scores
        avg_meter_quality = await smart_meter_repo.get_average_quality_score()
        avg_operator_quality = await grid_operator_repo.get_average_quality_score()
        avg_station_quality = await weather_station_repo.get_average_quality_score()
        
        # Calculate anomaly rates
        meter_anomaly_rate = await smart_meter_repo.get_anomaly_rate()
        operator_anomaly_rate = await grid_operator_repo.get_anomaly_rate()
        station_anomaly_rate = await weather_station_repo.get_anomaly_rate()
        
        response = {
            "period": period,
            "timestamp": end_time.isoformat(),
            "system_health": {
                "overall_status": "healthy",
                "data_quality_score": round((avg_meter_quality + avg_operator_quality + avg_station_quality) / 3, 3),
                "anomaly_rate": round((meter_anomaly_rate + operator_anomaly_rate + station_anomaly_rate) / 3, 2)
            },
            "smart_meters": {
                "total": total_meters,
                "active": active_meters,
                "inactive": total_meters - active_meters,
                "recent_readings": recent_readings,
                "avg_quality_score": round(avg_meter_quality, 3),
                "anomaly_rate": round(meter_anomaly_rate, 2)
            },
            "grid_operators": {
                "total": total_operators,
                "active": active_operators,
                "inactive": total_operators - active_operators,
                "recent_statuses": recent_statuses,
                "avg_quality_score": round(avg_operator_quality, 3),
                "anomaly_rate": round(operator_anomaly_rate, 2)
            },
            "weather_stations": {
                "total": total_stations,
                "active": active_stations,
                "inactive": total_stations - active_stations,
                "recent_observations": recent_observations,
                "avg_quality_score": round(avg_station_quality, 3),
                "anomaly_rate": round(station_anomaly_rate, 2)
            },
            "data_ingestion": {
                "total_records": recent_readings + recent_statuses + recent_observations,
                "readings_per_hour": round(recent_readings / hours, 2),
                "statuses_per_hour": round(recent_statuses / hours, 2),
                "observations_per_hour": round(recent_observations / hours, 2)
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting analytics overview: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analytics overview"
        )


@router.get("/energy-consumption", response_model=Dict[str, Any])
async def get_energy_consumption_analytics(
    period: str = Query("24h", description="Analysis period"),
    meter_ids: Optional[List[str]] = Query(None, description="Filter by meter IDs"),
    operator_ids: Optional[List[str]] = Query(None, description="Filter by operator IDs"),
    smart_meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository),
    grid_operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository)
):
    """Get energy consumption analytics"""
    try:
        # Parse period
        if period == "24h":
            hours = 24
        elif period == "7d":
            hours = 168
        elif period == "30d":
            hours = 720
        else:
            hours = 24
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get energy consumption data
        consumption_data = await smart_meter_repo.get_energy_consumption_analytics(
            start_time=start_time,
            end_time=end_time,
            meter_ids=meter_ids
        )
        
        # Get grid status data
        grid_data = await grid_operator_repo.get_grid_analytics(
            start_time=start_time,
            end_time=end_time,
            operator_ids=operator_ids
        )
        
        # Calculate totals
        total_consumption = sum(reading.get('total_consumption', 0) for reading in consumption_data)
        total_generation = sum(status.get('total_generation', 0) for status in grid_data)
        net_consumption = total_consumption - total_generation
        
        # Calculate averages
        avg_consumption = total_consumption / len(consumption_data) if consumption_data else 0
        avg_generation = total_generation / len(grid_data) if grid_data else 0
        
        # Find peak consumption
        peak_consumption = max((reading.get('peak_consumption', 0) for reading in consumption_data), default=0)
        peak_consumption_time = max(
            (reading.get('peak_consumption_time') for reading in consumption_data),
            default=None
        )
        
        response = {
            "period": period,
            "timestamp": end_time.isoformat(),
            "total_consumption_kwh": round(total_consumption, 2),
            "total_generation_kwh": round(total_generation, 2),
            "net_consumption_kwh": round(net_consumption, 2),
            "avg_consumption_kwh": round(avg_consumption, 2),
            "avg_generation_kwh": round(avg_generation, 2),
            "peak_consumption_kwh": round(peak_consumption, 2),
            "peak_consumption_time": peak_consumption_time,
            "consumption_by_meter": consumption_data,
            "grid_status": grid_data,
            "efficiency_metrics": {
                "consumption_efficiency": round(avg_consumption / max(avg_generation, 1), 3),
                "grid_utilization": round(total_consumption / max(total_generation, 1), 3)
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting energy consumption analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get energy consumption analytics"
        )


@router.get("/weather-impact", response_model=Dict[str, Any])
async def get_weather_impact_analytics(
    period: str = Query("24h", description="Analysis period"),
    station_ids: Optional[List[str]] = Query(None, description="Filter by station IDs"),
    weather_station_repo: WeatherStationRepository = Depends(get_weather_station_repository),
    data_quality_service: DataQualityService = Depends(get_data_quality_service),
    anomaly_detection_service: AnomalyDetectionService = Depends(get_anomaly_detection_service),
    alerting_service: AlertingService = Depends(get_alerting_service),
    kafka_producer: KafkaProducer = Depends(get_kafka_producer),
    s3_client: S3Client = Depends(get_s3_client),
    snowflake_query_executor: SnowflakeQueryExecutor = Depends(get_snowflake_query_executor)
):
    """Get weather impact analytics"""
    try:
        # Parse period
        if period == "24h":
            hours = 24
        elif period == "7d":
            hours = 168
        elif period == "30d":
            hours = 720
        else:
            hours = 24
        
        # Create use case for weather impact analysis
        analyze_use_case = AnalyzeWeatherImpactUseCase(
            weather_station_repository=weather_station_repo,
            smart_meter_repository=None,  # Not needed for weather analysis
            grid_operator_repository=None,  # Not needed for weather analysis
            data_quality_service=data_quality_service,
            anomaly_detection_service=anomaly_detection_service,
            alerting_service=alerting_service,
            kafka_producer=kafka_producer,
            s3_client=s3_client,
            snowflake_query_executor=snowflake_query_executor
        )
        
        # Execute weather impact analysis
        result = await analyze_use_case.execute(
            station_ids=station_ids,
            analysis_period_hours=hours
        )
        
        # Extract weather impact data
        weather_impact = result.get("weather_impact", {})
        energy_forecast = result.get("energy_forecast", {})
        correlation_analysis = result.get("correlation_analysis", {})
        
        response = {
            "period": period,
            "timestamp": datetime.utcnow().isoformat(),
            "weather_impact": {
                "temperature_impact": weather_impact.get("temperature_impact", {}),
                "humidity_impact": weather_impact.get("humidity_impact", {}),
                "wind_impact": weather_impact.get("wind_impact", {}),
                "precipitation_impact": weather_impact.get("precipitation_impact", {}),
                "overall_impact_score": weather_impact.get("overall_impact_score", 0.0)
            },
            "energy_forecast": {
                "predictions": energy_forecast.get("predictions", []),
                "confidence": energy_forecast.get("confidence", 0.0),
                "generated_at": energy_forecast.get("generated_at", datetime.utcnow().isoformat())
            },
            "correlation_analysis": {
                "temperature_correlation": correlation_analysis.get("temperature_correlation", 0.0),
                "humidity_correlation": correlation_analysis.get("humidity_correlation", 0.0),
                "wind_correlation": correlation_analysis.get("wind_correlation", 0.0),
                "precipitation_correlation": correlation_analysis.get("precipitation_correlation", 0.0)
            },
            "recommendations": result.get("recommendations", [])
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting weather impact analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get weather impact analytics"
        )


@router.get("/anomalies", response_model=Dict[str, Any])
async def get_anomaly_analytics(
    period: str = Query("24h", description="Analysis period"),
    anomaly_type: Optional[str] = Query(None, description="Filter by anomaly type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    smart_meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository),
    grid_operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository),
    weather_station_repo: WeatherStationRepository = Depends(get_weather_station_repository)
):
    """Get anomaly analytics"""
    try:
        # Parse period
        if period == "24h":
            hours = 24
        elif period == "7d":
            hours = 168
        elif period == "30d":
            hours = 720
        else:
            hours = 24
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get anomaly data from all sources
        meter_anomalies = await smart_meter_repo.get_anomalies(
            start_time=start_time,
            end_time=end_time,
            anomaly_type=anomaly_type,
            severity=severity
        )
        
        operator_anomalies = await grid_operator_repo.get_anomalies(
            start_time=start_time,
            end_time=end_time,
            anomaly_type=anomaly_type,
            severity=severity
        )
        
        station_anomalies = await weather_station_repo.get_anomalies(
            start_time=start_time,
            end_time=end_time,
            anomaly_type=anomaly_type,
            severity=severity
        )
        
        # Calculate anomaly statistics
        total_anomalies = len(meter_anomalies) + len(operator_anomalies) + len(station_anomalies)
        
        # Group by type
        anomaly_types = {}
        for anomaly in meter_anomalies + operator_anomalies + station_anomalies:
            anomaly_type = anomaly.get('anomaly_type', 'unknown')
            if anomaly_type not in anomaly_types:
                anomaly_types[anomaly_type] = 0
            anomaly_types[anomaly_type] += 1
        
        # Group by severity
        severity_counts = {}
        for anomaly in meter_anomalies + operator_anomalies + station_anomalies:
            severity = anomaly.get('severity', 'unknown')
            if severity not in severity_counts:
                severity_counts[severity] = 0
            severity_counts[severity] += 1
        
        # Calculate anomaly rates
        meter_anomaly_rate = len(meter_anomalies) / max(len(meter_anomalies), 1) * 100
        operator_anomaly_rate = len(operator_anomalies) / max(len(operator_anomalies), 1) * 100
        station_anomaly_rate = len(station_anomalies) / max(len(station_anomalies), 1) * 100
        
        response = {
            "period": period,
            "timestamp": end_time.isoformat(),
            "total_anomalies": total_anomalies,
            "anomaly_rate": round(total_anomalies / hours, 2),
            "anomalies_by_type": anomaly_types,
            "anomalies_by_severity": severity_counts,
            "anomalies_by_source": {
                "smart_meters": {
                    "count": len(meter_anomalies),
                    "rate": round(meter_anomaly_rate, 2)
                },
                "grid_operators": {
                    "count": len(operator_anomalies),
                    "rate": round(operator_anomaly_rate, 2)
                },
                "weather_stations": {
                    "count": len(station_anomalies),
                    "rate": round(station_anomaly_rate, 2)
                }
            },
            "recent_anomalies": {
                "smart_meters": meter_anomalies[:10],  # Last 10
                "grid_operators": operator_anomalies[:10],
                "weather_stations": station_anomalies[:10]
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting anomaly analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get anomaly analytics"
        )


@router.get("/data-quality", response_model=Dict[str, Any])
async def get_data_quality_analytics(
    period: str = Query("24h", description="Analysis period"),
    smart_meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository),
    grid_operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository),
    weather_station_repo: WeatherStationRepository = Depends(get_weather_station_repository)
):
    """Get data quality analytics"""
    try:
        # Parse period
        if period == "24h":
            hours = 24
        elif period == "7d":
            hours = 168
        elif period == "30d":
            hours = 720
        else:
            hours = 24
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get data quality metrics
        meter_quality = await smart_meter_repo.get_data_quality_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        operator_quality = await grid_operator_repo.get_data_quality_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        station_quality = await weather_station_repo.get_data_quality_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        # Calculate overall quality score
        overall_quality = (
            meter_quality.get('avg_quality_score', 0) +
            operator_quality.get('avg_quality_score', 0) +
            station_quality.get('avg_quality_score', 0)
        ) / 3
        
        # Calculate quality trends
        quality_trend = "stable"
        if meter_quality.get('quality_trend', 0) > 0.1:
            quality_trend = "improving"
        elif meter_quality.get('quality_trend', 0) < -0.1:
            quality_trend = "declining"
        
        response = {
            "period": period,
            "timestamp": end_time.isoformat(),
            "overall_quality_score": round(overall_quality, 3),
            "quality_trend": quality_trend,
            "quality_by_source": {
                "smart_meters": {
                    "avg_quality_score": round(meter_quality.get('avg_quality_score', 0), 3),
                    "quality_trend": round(meter_quality.get('quality_trend', 0), 3),
                    "total_records": meter_quality.get('total_records', 0),
                    "quality_issues": meter_quality.get('quality_issues', 0)
                },
                "grid_operators": {
                    "avg_quality_score": round(operator_quality.get('avg_quality_score', 0), 3),
                    "quality_trend": round(operator_quality.get('quality_trend', 0), 3),
                    "total_records": operator_quality.get('total_records', 0),
                    "quality_issues": operator_quality.get('quality_issues', 0)
                },
                "weather_stations": {
                    "avg_quality_score": round(station_quality.get('avg_quality_score', 0), 3),
                    "quality_trend": round(station_quality.get('quality_trend', 0), 3),
                    "total_records": station_quality.get('total_records', 0),
                    "quality_issues": station_quality.get('quality_issues', 0)
                }
            },
            "quality_issues": {
                "missing_data": meter_quality.get('missing_data', 0) + operator_quality.get('missing_data', 0) + station_quality.get('missing_data', 0),
                "invalid_data": meter_quality.get('invalid_data', 0) + operator_quality.get('invalid_data', 0) + station_quality.get('invalid_data', 0),
                "outliers": meter_quality.get('outliers', 0) + operator_quality.get('outliers', 0) + station_quality.get('outliers', 0)
            },
            "recommendations": [
                "Monitor data quality trends closely",
                "Investigate sources of quality issues",
                "Implement automated quality checks",
                "Set up quality alerts for critical thresholds"
            ]
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting data quality analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get data quality analytics"
        )


@router.get("/reports/daily", response_model=Dict[str, Any])
async def get_daily_report(
    date: Optional[str] = Query(None, description="Report date (YYYY-MM-DD)"),
    smart_meter_repo: SmartMeterRepository = Depends(get_smart_meter_repository),
    grid_operator_repo: GridOperatorRepository = Depends(get_grid_operator_repository),
    weather_station_repo: WeatherStationRepository = Depends(get_weather_station_repository)
):
    """Get daily analytics report"""
    try:
        # Parse date
        if date:
            report_date = datetime.fromisoformat(date)
        else:
            report_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        start_time = report_date
        end_time = report_date + timedelta(days=1)
        
        # Get daily statistics
        daily_stats = {
            "date": report_date.strftime("%Y-%m-%d"),
            "smart_meters": await smart_meter_repo.get_daily_stats(start_time, end_time),
            "grid_operators": await grid_operator_repo.get_daily_stats(start_time, end_time),
            "weather_stations": await weather_station_repo.get_daily_stats(start_time, end_time)
        }
        
        # Generate report summary
        total_readings = daily_stats["smart_meters"].get("total_readings", 0)
        total_statuses = daily_stats["grid_operators"].get("total_statuses", 0)
        total_observations = daily_stats["weather_stations"].get("total_observations", 0)
        
        response = {
            "report_date": report_date.strftime("%Y-%m-%d"),
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_data_points": total_readings + total_statuses + total_observations,
                "data_quality_score": round(
                    (daily_stats["smart_meters"].get("avg_quality_score", 0) +
                     daily_stats["grid_operators"].get("avg_quality_score", 0) +
                     daily_stats["weather_stations"].get("avg_quality_score", 0)) / 3, 3
                ),
                "anomaly_count": (
                    daily_stats["smart_meters"].get("anomaly_count", 0) +
                    daily_stats["grid_operators"].get("anomaly_count", 0) +
                    daily_stats["weather_stations"].get("anomaly_count", 0)
                )
            },
            "detailed_stats": daily_stats,
            "recommendations": [
                "Review daily data quality trends",
                "Investigate any anomalies detected",
                "Monitor system performance metrics",
                "Update operational procedures if needed"
            ]
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting daily report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get daily report"
        )
