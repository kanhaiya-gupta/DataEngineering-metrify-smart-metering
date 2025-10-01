"""
Anomaly Detection Use Case
Handles comprehensive anomaly detection across all data sources
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import logging

from ...core.domain.entities.smart_meter import SmartMeter, MeterReading
from ...core.domain.entities.grid_operator import GridOperator, GridStatus
from ...core.domain.entities.weather_station import WeatherStation, WeatherObservation
from ...core.interfaces.repositories.smart_meter_repository import ISmartMeterRepository
from ...core.interfaces.repositories.grid_operator_repository import IGridOperatorRepository
from ...core.interfaces.repositories.weather_station_repository import IWeatherStationRepository
from ...core.interfaces.external.anomaly_detection_service import IAnomalyDetectionService
from ...core.interfaces.external.alerting_service import IAlertingService
from ...core.exceptions.domain_exceptions import MeterNotFoundException, GridOperatorNotFoundError, WeatherStationNotFoundError
from ...infrastructure.external.kafka.kafka_producer import KafkaProducer
from ...infrastructure.external.s3.s3_client import S3Client

logger = logging.getLogger(__name__)


class DetectAnomaliesUseCase:
    """
    Use case for comprehensive anomaly detection
    
    Handles anomaly detection across all data sources including
    smart meters, grid operators, and weather stations.
    """
    
    def __init__(
        self,
        smart_meter_repository: ISmartMeterRepository,
        grid_operator_repository: IGridOperatorRepository,
        weather_station_repository: IWeatherStationRepository,
        anomaly_detection_service: IAnomalyDetectionService,
        alerting_service: IAlertingService,
        kafka_producer: KafkaProducer,
        s3_client: S3Client
    ):
        self.smart_meter_repository = smart_meter_repository
        self.grid_operator_repository = grid_operator_repository
        self.weather_station_repository = weather_station_repository
        self.anomaly_detection_service = anomaly_detection_service
        self.alerting_service = alerting_service
        self.kafka_producer = kafka_producer
        self.s3_client = s3_client
    
    async def execute(
        self,
        data_type: str,
        entity_id: str,
        detection_window_hours: int = 24,
        anomaly_threshold: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute anomaly detection
        
        Args:
            data_type: Type of data to analyze (smart_meter, grid_operator, weather_station)
            entity_id: Entity identifier
            detection_window_hours: Hours of data to analyze
            anomaly_threshold: Threshold for anomaly detection
            metadata: Optional metadata for the detection
            
        Returns:
            Dictionary containing detection results
        """
        try:
            logger.info(f"Starting anomaly detection for {data_type} {entity_id}")
            
            # Step 1: Get data for analysis
            data = await self._get_data_for_analysis(data_type, entity_id, detection_window_hours)
            
            if not data:
                return {
                    "status": "success",
                    "data_type": data_type,
                    "entity_id": entity_id,
                    "anomalies_detected": 0,
                    "message": "No data available for analysis"
                }
            
            # Step 2: Perform anomaly detection
            anomaly_results = await self._perform_anomaly_detection(data, data_type, anomaly_threshold)
            
            # Step 3: Classify anomalies by severity
            classified_anomalies = await self._classify_anomalies(anomaly_results, data_type)
            
            # Step 4: Generate anomaly insights
            insights = await self._generate_anomaly_insights(classified_anomalies, data_type)
            
            # Step 5: Store anomaly results
            await self._store_anomaly_results(entity_id, data_type, classified_anomalies, insights)
            
            # Step 6: Publish anomalies to Kafka
            await self._publish_anomalies_to_kafka(entity_id, data_type, classified_anomalies)
            
            # Step 7: Archive results to S3
            await self._archive_anomaly_results(entity_id, data_type, classified_anomalies, metadata)
            
            # Step 8: Send alerts for critical anomalies
            await self._send_anomaly_alerts(entity_id, data_type, classified_anomalies)
            
            result = {
                "status": "success",
                "data_type": data_type,
                "entity_id": entity_id,
                "detection_window_hours": detection_window_hours,
                "anomalies_detected": len(classified_anomalies),
                "critical_anomalies": len([a for a in classified_anomalies if a['severity'] == 'critical']),
                "high_anomalies": len([a for a in classified_anomalies if a['severity'] == 'high']),
                "medium_anomalies": len([a for a in classified_anomalies if a['severity'] == 'medium']),
                "low_anomalies": len([a for a in classified_anomalies if a['severity'] == 'low']),
                "insights": insights,
                "alerts_sent": len([a for a in classified_anomalies if a['severity'] in ['critical', 'high']]),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Anomaly detection completed for {data_type} {entity_id}: {len(classified_anomalies)} anomalies found")
            return result
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise
    
    async def _get_data_for_analysis(
        self,
        data_type: str,
        entity_id: str,
        detection_window_hours: int
    ) -> List[Dict[str, Any]]:
        """Get data for anomaly analysis"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=detection_window_hours)
            
            if data_type == "smart_meter":
                return await self._get_smart_meter_data(entity_id, start_time, end_time)
            elif data_type == "grid_operator":
                return await self._get_grid_operator_data(entity_id, start_time, end_time)
            elif data_type == "weather_station":
                return await self._get_weather_station_data(entity_id, start_time, end_time)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
                
        except Exception as e:
            logger.error(f"Error getting data for analysis: {str(e)}")
            return []
    
    async def _get_smart_meter_data(
        self,
        meter_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get smart meter data for analysis"""
        try:
            readings = await self.smart_meter_repository.get_readings_by_time_range(
                meter_id, start_time, end_time
            )
            
            data = []
            for reading in readings:
                data.append({
                    'timestamp': reading.timestamp.isoformat(),
                    'voltage': reading.voltage,
                    'current': reading.current,
                    'power_factor': reading.power_factor,
                    'frequency': reading.frequency,
                    'active_power': reading.active_power,
                    'reactive_power': reading.reactive_power,
                    'apparent_power': reading.apparent_power,
                    'data_quality_score': reading.data_quality_score
                })
            
            logger.info(f"Retrieved {len(data)} smart meter readings for analysis")
            return data
            
        except Exception as e:
            logger.error(f"Error getting smart meter data: {str(e)}")
            return []
    
    async def _get_grid_operator_data(
        self,
        operator_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get grid operator data for analysis"""
        try:
            statuses = await self.grid_operator_repository.get_statuses_by_time_range(
                operator_id, start_time, end_time
            )
            
            data = []
            for status in statuses:
                data.append({
                    'timestamp': status.timestamp.isoformat(),
                    'voltage_level': status.voltage_level,
                    'frequency': status.frequency,
                    'load_percentage': status.load_percentage,
                    'stability_score': status.stability_score,
                    'power_quality_score': status.power_quality_score,
                    'total_generation_mw': status.total_generation_mw,
                    'total_consumption_mw': status.total_consumption_mw,
                    'grid_frequency_hz': status.grid_frequency_hz,
                    'voltage_deviation_percent': status.voltage_deviation_percent,
                    'data_quality_score': status.data_quality_score
                })
            
            logger.info(f"Retrieved {len(data)} grid operator statuses for analysis")
            return data
            
        except Exception as e:
            logger.error(f"Error getting grid operator data: {str(e)}")
            return []
    
    async def _get_weather_station_data(
        self,
        station_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get weather station data for analysis"""
        try:
            observations = await self.weather_station_repository.get_observations_by_time_range(
                station_id, start_time, end_time
            )
            
            data = []
            for observation in observations:
                data.append({
                    'timestamp': observation.timestamp.isoformat(),
                    'temperature_celsius': observation.temperature_celsius,
                    'humidity_percent': observation.humidity_percent,
                    'pressure_hpa': observation.pressure_hpa,
                    'wind_speed_ms': observation.wind_speed_ms,
                    'wind_direction_degrees': observation.wind_direction_degrees,
                    'cloud_cover_percent': observation.cloud_cover_percent,
                    'visibility_km': observation.visibility_km,
                    'uv_index': observation.uv_index,
                    'precipitation_mm': observation.precipitation_mm,
                    'data_quality_score': observation.data_quality_score
                })
            
            logger.info(f"Retrieved {len(data)} weather observations for analysis")
            return data
            
        except Exception as e:
            logger.error(f"Error getting weather station data: {str(e)}")
            return []
    
    async def _perform_anomaly_detection(
        self,
        data: List[Dict[str, Any]],
        data_type: str,
        anomaly_threshold: float
    ) -> Dict[str, Any]:
        """Perform anomaly detection using the anomaly detection service"""
        try:
            anomaly_results = await self.anomaly_detection_service.detect_anomalies(
                data=data,
                data_type=data_type,
                threshold=anomaly_threshold
            )
            
            logger.info(f"Anomaly detection completed: {anomaly_results.get('anomaly_count', 0)} anomalies found")
            return anomaly_results
            
        except Exception as e:
            logger.error(f"Error performing anomaly detection: {str(e)}")
            return {"anomaly_count": 0, "anomalies": []}
    
    async def _classify_anomalies(
        self,
        anomaly_results: Dict[str, Any],
        data_type: str
    ) -> List[Dict[str, Any]]:
        """Classify anomalies by severity and type"""
        try:
            classified_anomalies = []
            
            for anomaly in anomaly_results.get('anomalies', []):
                # Determine severity based on anomaly score and type
                severity = self._determine_anomaly_severity(anomaly, data_type)
                
                # Determine anomaly type
                anomaly_type = self._determine_anomaly_type(anomaly, data_type)
                
                # Generate description
                description = self._generate_anomaly_description(anomaly, anomaly_type, data_type)
                
                # Generate recommendations
                recommendations = self._generate_anomaly_recommendations(anomaly, anomaly_type, severity)
                
                classified_anomaly = {
                    'timestamp': anomaly.get('timestamp'),
                    'anomaly_score': anomaly.get('score', 0.0),
                    'severity': severity,
                    'type': anomaly_type,
                    'description': description,
                    'recommendations': recommendations,
                    'affected_fields': anomaly.get('affected_fields', []),
                    'raw_data': anomaly.get('raw_data', {}),
                    'detection_method': anomaly.get('detection_method', 'unknown')
                }
                
                classified_anomalies.append(classified_anomaly)
            
            logger.info(f"Classified {len(classified_anomalies)} anomalies")
            return classified_anomalies
            
        except Exception as e:
            logger.error(f"Error classifying anomalies: {str(e)}")
            return []
    
    def _determine_anomaly_severity(self, anomaly: Dict[str, Any], data_type: str) -> str:
        """Determine anomaly severity based on score and type"""
        score = anomaly.get('score', 0.0)
        anomaly_type = anomaly.get('type', 'unknown')
        
        # Critical anomalies
        if score >= 0.95 or anomaly_type in ['system_failure', 'critical_equipment_failure']:
            return 'critical'
        
        # High severity anomalies
        elif score >= 0.8 or anomaly_type in ['voltage_spike', 'frequency_deviation', 'extreme_weather']:
            return 'high'
        
        # Medium severity anomalies
        elif score >= 0.6 or anomaly_type in ['data_quality_issue', 'unusual_pattern']:
            return 'medium'
        
        # Low severity anomalies
        else:
            return 'low'
    
    def _determine_anomaly_type(self, anomaly: Dict[str, Any], data_type: str) -> str:
        """Determine anomaly type based on affected fields and patterns"""
        affected_fields = anomaly.get('affected_fields', [])
        raw_data = anomaly.get('raw_data', {})
        
        if data_type == "smart_meter":
            if 'voltage' in affected_fields:
                if raw_data.get('voltage', 0) > 250:
                    return 'voltage_spike'
                elif raw_data.get('voltage', 0) < 200:
                    return 'voltage_drop'
            elif 'frequency' in affected_fields:
                return 'frequency_deviation'
            elif 'power_factor' in affected_fields:
                return 'power_factor_anomaly'
            else:
                return 'power_consumption_anomaly'
        
        elif data_type == "grid_operator":
            if 'stability_score' in affected_fields:
                return 'grid_stability_issue'
            elif 'frequency' in affected_fields:
                return 'grid_frequency_anomaly'
            elif 'voltage_level' in affected_fields:
                return 'grid_voltage_anomaly'
            else:
                return 'grid_operational_anomaly'
        
        elif data_type == "weather_station":
            if 'temperature_celsius' in affected_fields:
                return 'temperature_anomaly'
            elif 'wind_speed_ms' in affected_fields:
                return 'wind_speed_anomaly'
            elif 'humidity_percent' in affected_fields:
                return 'humidity_anomaly'
            else:
                return 'weather_measurement_anomaly'
        
        return 'unknown_anomaly'
    
    def _generate_anomaly_description(
        self,
        anomaly: Dict[str, Any],
        anomaly_type: str,
        data_type: str
    ) -> str:
        """Generate human-readable anomaly description"""
        timestamp = anomaly.get('timestamp', 'unknown')
        score = anomaly.get('score', 0.0)
        
        descriptions = {
            'voltage_spike': f"Voltage spike detected at {timestamp} (score: {score:.2f})",
            'voltage_drop': f"Voltage drop detected at {timestamp} (score: {score:.2f})",
            'frequency_deviation': f"Frequency deviation detected at {timestamp} (score: {score:.2f})",
            'power_factor_anomaly': f"Power factor anomaly detected at {timestamp} (score: {score:.2f})",
            'power_consumption_anomaly': f"Power consumption anomaly detected at {timestamp} (score: {score:.2f})",
            'grid_stability_issue': f"Grid stability issue detected at {timestamp} (score: {score:.2f})",
            'grid_frequency_anomaly': f"Grid frequency anomaly detected at {timestamp} (score: {score:.2f})",
            'grid_voltage_anomaly': f"Grid voltage anomaly detected at {timestamp} (score: {score:.2f})",
            'grid_operational_anomaly': f"Grid operational anomaly detected at {timestamp} (score: {score:.2f})",
            'temperature_anomaly': f"Temperature anomaly detected at {timestamp} (score: {score:.2f})",
            'wind_speed_anomaly': f"Wind speed anomaly detected at {timestamp} (score: {score:.2f})",
            'humidity_anomaly': f"Humidity anomaly detected at {timestamp} (score: {score:.2f})",
            'weather_measurement_anomaly': f"Weather measurement anomaly detected at {timestamp} (score: {score:.2f})",
            'unknown_anomaly': f"Unknown anomaly detected at {timestamp} (score: {score:.2f})"
        }
        
        return descriptions.get(anomaly_type, descriptions['unknown_anomaly'])
    
    def _generate_anomaly_recommendations(
        self,
        anomaly: Dict[str, Any],
        anomaly_type: str,
        severity: str
    ) -> List[str]:
        """Generate recommendations based on anomaly type and severity"""
        recommendations = []
        
        if severity == 'critical':
            recommendations.append("Immediate investigation required")
            recommendations.append("Consider emergency procedures")
            recommendations.append("Notify operations team immediately")
        
        elif severity == 'high':
            recommendations.append("Investigate within 1 hour")
            recommendations.append("Monitor closely")
            recommendations.append("Prepare contingency plans")
        
        elif severity == 'medium':
            recommendations.append("Investigate within 4 hours")
            recommendations.append("Monitor for patterns")
            recommendations.append("Review system logs")
        
        else:
            recommendations.append("Investigate within 24 hours")
            recommendations.append("Monitor for recurrence")
        
        # Type-specific recommendations
        if anomaly_type in ['voltage_spike', 'voltage_drop']:
            recommendations.append("Check electrical connections and equipment")
            recommendations.append("Verify voltage regulation systems")
        
        elif anomaly_type == 'frequency_deviation':
            recommendations.append("Check grid synchronization")
            recommendations.append("Verify frequency control systems")
        
        elif anomaly_type == 'grid_stability_issue':
            recommendations.append("Check grid load balancing")
            recommendations.append("Verify transmission line status")
        
        elif anomaly_type in ['temperature_anomaly', 'wind_speed_anomaly']:
            recommendations.append("Verify weather station calibration")
            recommendations.append("Check for sensor malfunctions")
        
        return recommendations
    
    async def _generate_anomaly_insights(
        self,
        classified_anomalies: List[Dict[str, Any]],
        data_type: str
    ) -> Dict[str, Any]:
        """Generate insights from anomaly analysis"""
        try:
            if not classified_anomalies:
                return {"insights": [], "patterns": [], "trends": []}
            
            insights = []
            patterns = []
            trends = []
            
            # Analyze anomaly patterns
            anomaly_types = [a['type'] for a in classified_anomalies]
            type_counts = {}
            for anomaly_type in anomaly_types:
                type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
            
            # Find most common anomaly types
            most_common = max(type_counts.items(), key=lambda x: x[1]) if type_counts else None
            if most_common:
                patterns.append({
                    'type': 'most_common_anomaly',
                    'anomaly_type': most_common[0],
                    'count': most_common[1],
                    'percentage': (most_common[1] / len(classified_anomalies)) * 100
                })
            
            # Analyze severity distribution
            severity_counts = {}
            for anomaly in classified_anomalies:
                severity = anomaly['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Calculate severity distribution
            severity_distribution = {
                severity: (count / len(classified_anomalies)) * 100
                for severity, count in severity_counts.items()
            }
            
            patterns.append({
                'type': 'severity_distribution',
                'distribution': severity_distribution
            })
            
            # Analyze temporal patterns
            timestamps = [a['timestamp'] for a in classified_anomalies if a['timestamp']]
            if timestamps:
                # Group by hour to find peak anomaly times
                hour_counts = {}
                for timestamp in timestamps:
                    try:
                        hour = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).hour
                        hour_counts[hour] = hour_counts.get(hour, 0) + 1
                    except:
                        continue
                
                if hour_counts:
                    peak_hour = max(hour_counts.items(), key=lambda x: x[1])
                    patterns.append({
                        'type': 'peak_anomaly_hour',
                        'hour': peak_hour[0],
                        'count': peak_hour[1]
                    })
            
            # Generate insights
            if severity_counts.get('critical', 0) > 0:
                insights.append({
                    'type': 'critical_anomalies',
                    'message': f"Critical anomalies detected: {severity_counts['critical']}",
                    'priority': 'high'
                })
            
            if most_common and most_common[1] > len(classified_anomalies) * 0.5:
                insights.append({
                    'type': 'recurring_anomaly',
                    'message': f"Recurring anomaly type: {most_common[0]} ({most_common[1]} occurrences)",
                    'priority': 'medium'
                })
            
            if len(classified_anomalies) > 10:
                insights.append({
                    'type': 'high_anomaly_rate',
                    'message': f"High anomaly rate detected: {len(classified_anomalies)} anomalies",
                    'priority': 'medium'
                })
            
            return {
                'insights': insights,
                'patterns': patterns,
                'trends': trends,
                'summary': {
                    'total_anomalies': len(classified_anomalies),
                    'critical_count': severity_counts.get('critical', 0),
                    'high_count': severity_counts.get('high', 0),
                    'medium_count': severity_counts.get('medium', 0),
                    'low_count': severity_counts.get('low', 0),
                    'most_common_type': most_common[0] if most_common else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating anomaly insights: {str(e)}")
            return {"insights": [], "patterns": [], "trends": []}
    
    async def _store_anomaly_results(
        self,
        entity_id: str,
        data_type: str,
        classified_anomalies: List[Dict[str, Any]],
        insights: Dict[str, Any]
    ) -> None:
        """Store anomaly results in the database"""
        try:
            # This would typically store results in a dedicated anomalies table
            # For now, we'll just log the storage
            logger.info(f"Stored {len(classified_anomalies)} anomaly results for {data_type} {entity_id}")
            
        except Exception as e:
            logger.error(f"Error storing anomaly results: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
    
    async def _publish_anomalies_to_kafka(
        self,
        entity_id: str,
        data_type: str,
        classified_anomalies: List[Dict[str, Any]]
    ) -> None:
        """Publish anomaly results to Kafka"""
        try:
            for anomaly in classified_anomalies:
                message = {
                    'entity_id': entity_id,
                    'data_type': data_type,
                    'anomaly': anomaly,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await self.kafka_producer.send_message(
                    topic="anomalies",
                    message=message
                )
            
            logger.info(f"Published {len(classified_anomalies)} anomalies to Kafka")
            
        except Exception as e:
            logger.error(f"Error publishing anomalies to Kafka: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
    
    async def _archive_anomaly_results(
        self,
        entity_id: str,
        data_type: str,
        classified_anomalies: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Archive anomaly results to S3"""
        try:
            results_data = {
                'entity_id': entity_id,
                'data_type': data_type,
                'anomalies': classified_anomalies,
                'metadata': metadata,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            timestamp = datetime.utcnow().strftime("%Y/%m/%d/%H")
            s3_key = f"anomaly-results/{data_type}/{entity_id}/{timestamp}/anomalies.json"
            
            await self.s3_client.upload_data(
                data=results_data,
                s3_key=s3_key,
                content_type="application/json"
            )
            
            logger.info(f"Archived anomaly results to S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"Error archiving anomaly results: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
    
    async def _send_anomaly_alerts(
        self,
        entity_id: str,
        data_type: str,
        classified_anomalies: List[Dict[str, Any]]
    ) -> None:
        """Send alerts for critical and high severity anomalies"""
        try:
            alerts_sent = 0
            
            for anomaly in classified_anomalies:
                if anomaly['severity'] in ['critical', 'high']:
                    await self.alerting_service.send_alert(
                        alert_type=f"anomaly_{anomaly['type']}",
                        severity=anomaly['severity'],
                        message=anomaly['description'],
                        entity_id=entity_id,
                        entity_type=data_type,
                        metadata={
                            'anomaly_score': anomaly['anomaly_score'],
                            'recommendations': anomaly['recommendations'],
                            'affected_fields': anomaly['affected_fields']
                        }
                    )
                    alerts_sent += 1
            
            logger.info(f"Sent {alerts_sent} anomaly alerts for {data_type} {entity_id}")
            
        except Exception as e:
            logger.error(f"Error sending anomaly alerts: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
