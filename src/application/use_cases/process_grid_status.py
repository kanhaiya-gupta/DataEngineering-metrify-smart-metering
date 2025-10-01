"""
Grid Status Processing Use Case
Handles the processing and monitoring of grid operator status data
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ...core.domain.entities.grid_operator import GridOperator, GridStatus
from ...core.domain.value_objects.location import Location
from ...core.domain.enums.grid_operator_status import GridOperatorStatus
from ...core.interfaces.repositories.grid_operator_repository import IGridOperatorRepository
from ...core.interfaces.external.data_quality_service import IDataQualityService
from ...core.interfaces.external.anomaly_detection_service import IAnomalyDetectionService
from ...core.interfaces.external.alerting_service import IAlertingService
from ...core.exceptions.domain_exceptions import GridOperatorNotFoundError, InvalidGridStatusError
from ...infrastructure.external.kafka.kafka_producer import KafkaProducer
from ...infrastructure.external.s3.s3_client import S3Client

logger = logging.getLogger(__name__)


class ProcessGridStatusUseCase:
    """
    Use case for processing grid operator status data
    
    Handles the complete flow of grid status processing,
    including validation, quality assessment, anomaly detection,
    stability monitoring, and alerting.
    """
    
    def __init__(
        self,
        grid_operator_repository: IGridOperatorRepository,
        data_quality_service: IDataQualityService,
        anomaly_detection_service: IAnomalyDetectionService,
        alerting_service: IAlertingService,
        kafka_producer: KafkaProducer,
        s3_client: S3Client
    ):
        self.grid_operator_repository = grid_operator_repository
        self.data_quality_service = data_quality_service
        self.anomaly_detection_service = anomaly_detection_service
        self.alerting_service = alerting_service
        self.kafka_producer = kafka_producer
        self.s3_client = s3_client
    
    async def execute(
        self,
        operator_id: str,
        status_data: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute grid status processing
        
        Args:
            operator_id: Grid operator identifier
            status_data: List of grid status data
            metadata: Optional metadata for the processing
            
        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info(f"Starting grid status processing for operator {operator_id}")
            
            # Step 1: Validate operator exists
            operator = await self._validate_operator_exists(operator_id)
            
            # Step 2: Process and validate status data
            processed_statuses = await self._process_status_data(operator_id, status_data)
            
            # Step 3: Assess data quality
            quality_results = await self._assess_data_quality(processed_statuses)
            
            # Step 4: Detect anomalies
            anomaly_results = await self._detect_anomalies(processed_statuses)
            
            # Step 5: Analyze grid stability
            stability_results = await self._analyze_grid_stability(processed_statuses)
            
            # Step 6: Store status data in database
            stored_statuses = await self._store_status_data(processed_statuses, quality_results, anomaly_results)
            
            # Step 7: Publish to Kafka for real-time monitoring
            await self._publish_to_kafka(operator_id, stored_statuses)
            
            # Step 8: Archive to S3 for long-term storage
            await self._archive_to_s3(operator_id, stored_statuses, metadata)
            
            # Step 9: Check for alerts and send notifications
            await self._check_and_send_alerts(operator, stability_results, anomaly_results)
            
            # Step 10: Update operator status if needed
            await self._update_operator_status(operator, stability_results, anomaly_results)
            
            result = {
                "status": "success",
                "operator_id": operator_id,
                "statuses_processed": len(processed_statuses),
                "quality_score": quality_results.get("overall_score", 0.0),
                "anomalies_detected": anomaly_results.get("anomaly_count", 0),
                "stability_score": stability_results.get("overall_stability", 0.0),
                "alerts_sent": stability_results.get("alerts_sent", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Grid status processing completed for operator {operator_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in grid status processing: {str(e)}")
            raise
    
    async def _validate_operator_exists(self, operator_id: str) -> GridOperator:
        """Validate that the grid operator exists"""
        try:
            operator = await self.grid_operator_repository.get_by_id(operator_id)
            if not operator:
                raise GridOperatorNotFoundError(f"Grid operator {operator_id} not found")
            return operator
        except Exception as e:
            logger.error(f"Error validating operator {operator_id}: {str(e)}")
            raise
    
    async def _process_status_data(
        self,
        operator_id: str,
        status_data: List[Dict[str, Any]]
    ) -> List[GridStatus]:
        """Process and validate grid status data"""
        processed_statuses = []
        
        for status_item in status_data:
            try:
                # Validate required fields
                required_fields = ['timestamp', 'voltage_level', 'frequency', 'load_percentage']
                for field in required_fields:
                    if field not in status_item:
                        raise InvalidGridStatusError(f"Missing required field: {field}")
                
                # Create GridStatus object
                status = GridStatus(
                    operator_id=operator_id,
                    timestamp=datetime.fromisoformat(status_item['timestamp'].replace('Z', '+00:00')),
                    voltage_level=float(status_item['voltage_level']),
                    frequency=float(status_item['frequency']),
                    load_percentage=float(status_item['load_percentage']),
                    stability_score=float(status_item.get('stability_score', 1.0)),
                    power_quality_score=float(status_item.get('power_quality_score', 1.0)),
                    total_generation_mw=float(status_item.get('total_generation_mw', 0.0)),
                    total_consumption_mw=float(status_item.get('total_consumption_mw', 0.0)),
                    grid_frequency_hz=float(status_item.get('grid_frequency_hz', 50.0)),
                    voltage_deviation_percent=float(status_item.get('voltage_deviation_percent', 0.0))
                )
                
                processed_statuses.append(status)
                
            except Exception as e:
                logger.warning(f"Error processing status: {str(e)}")
                continue
        
        logger.info(f"Processed {len(processed_statuses)} status records for operator {operator_id}")
        return processed_statuses
    
    async def _assess_data_quality(self, statuses: List[GridStatus]) -> Dict[str, Any]:
        """Assess data quality of the status records"""
        try:
            # Convert statuses to format expected by quality service
            status_data = []
            for status in statuses:
                status_data.append({
                    'timestamp': status.timestamp.isoformat(),
                    'voltage_level': status.voltage_level,
                    'frequency': status.frequency,
                    'load_percentage': status.load_percentage,
                    'stability_score': status.stability_score,
                    'power_quality_score': status.power_quality_score,
                    'total_generation_mw': status.total_generation_mw,
                    'total_consumption_mw': status.total_consumption_mw,
                    'grid_frequency_hz': status.grid_frequency_hz,
                    'voltage_deviation_percent': status.voltage_deviation_percent
                })
            
            quality_results = await self.data_quality_service.assess_quality(
                data=status_data,
                data_type="grid_status"
            )
            
            logger.info(f"Data quality assessment completed: {quality_results.get('overall_score', 0.0)}")
            return quality_results
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            return {"overall_score": 0.0, "quality_issues": []}
    
    async def _detect_anomalies(self, statuses: List[GridStatus]) -> Dict[str, Any]:
        """Detect anomalies in the status records"""
        try:
            # Convert statuses to format expected by anomaly detection service
            status_data = []
            for status in statuses:
                status_data.append({
                    'timestamp': status.timestamp.isoformat(),
                    'voltage_level': status.voltage_level,
                    'frequency': status.frequency,
                    'load_percentage': status.load_percentage,
                    'stability_score': status.stability_score,
                    'power_quality_score': status.power_quality_score,
                    'total_generation_mw': status.total_generation_mw,
                    'total_consumption_mw': status.total_consumption_mw,
                    'grid_frequency_hz': status.grid_frequency_hz,
                    'voltage_deviation_percent': status.voltage_deviation_percent
                })
            
            anomaly_results = await self.anomaly_detection_service.detect_anomalies(
                data=status_data,
                data_type="grid_status"
            )
            
            logger.info(f"Anomaly detection completed: {anomaly_results.get('anomaly_count', 0)} anomalies found")
            return anomaly_results
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return {"anomaly_count": 0, "anomalies": []}
    
    async def _analyze_grid_stability(self, statuses: List[GridStatus]) -> Dict[str, Any]:
        """Analyze grid stability and identify potential issues"""
        try:
            if not statuses:
                return {"overall_stability": 0.0, "stability_issues": [], "alerts_sent": 0}
            
            # Calculate stability metrics
            avg_stability = sum(s.stability_score for s in statuses) / len(statuses)
            avg_frequency = sum(s.frequency for s in statuses) / len(statuses)
            avg_voltage = sum(s.voltage_level for s in statuses) / len(statuses)
            avg_load = sum(s.load_percentage for s in statuses) / len(statuses)
            
            # Identify stability issues
            stability_issues = []
            alerts_sent = 0
            
            # Check frequency stability (should be around 50Hz)
            if abs(avg_frequency - 50.0) > 0.5:
                stability_issues.append({
                    "type": "frequency_deviation",
                    "severity": "high" if abs(avg_frequency - 50.0) > 1.0 else "medium",
                    "value": avg_frequency,
                    "threshold": 50.0
                })
                alerts_sent += 1
            
            # Check voltage stability
            if avg_voltage < 220 or avg_voltage > 240:
                stability_issues.append({
                    "type": "voltage_deviation",
                    "severity": "high" if avg_voltage < 200 or avg_voltage > 260 else "medium",
                    "value": avg_voltage,
                    "threshold": 230
                })
                alerts_sent += 1
            
            # Check load percentage
            if avg_load > 90:
                stability_issues.append({
                    "type": "high_load",
                    "severity": "high" if avg_load > 95 else "medium",
                    "value": avg_load,
                    "threshold": 90
                })
                alerts_sent += 1
            
            # Check overall stability score
            if avg_stability < 0.7:
                stability_issues.append({
                    "type": "low_stability",
                    "severity": "high" if avg_stability < 0.5 else "medium",
                    "value": avg_stability,
                    "threshold": 0.7
                })
                alerts_sent += 1
            
            stability_results = {
                "overall_stability": avg_stability,
                "stability_issues": stability_issues,
                "alerts_sent": alerts_sent,
                "metrics": {
                    "avg_frequency": avg_frequency,
                    "avg_voltage": avg_voltage,
                    "avg_load": avg_load,
                    "avg_stability": avg_stability
                }
            }
            
            logger.info(f"Grid stability analysis completed: {avg_stability:.2f} stability score")
            return stability_results
            
        except Exception as e:
            logger.error(f"Error analyzing grid stability: {str(e)}")
            return {"overall_stability": 0.0, "stability_issues": [], "alerts_sent": 0}
    
    async def _store_status_data(
        self,
        statuses: List[GridStatus],
        quality_results: Dict[str, Any],
        anomaly_results: Dict[str, Any]
    ) -> List[GridStatus]:
        """Store status data in the database"""
        try:
            stored_statuses = []
            
            for status in statuses:
                # Add quality and anomaly information
                status.data_quality_score = quality_results.get('overall_score', 1.0)
                status.is_anomaly = status in anomaly_results.get('anomalies', [])
                status.anomaly_type = anomaly_results.get('anomaly_type', None)
                
                # Store in database
                stored_status = await self.grid_operator_repository.add_status(status)
                stored_statuses.append(stored_status)
            
            logger.info(f"Stored {len(stored_statuses)} status records in database")
            return stored_statuses
            
        except Exception as e:
            logger.error(f"Error storing status data: {str(e)}")
            raise
    
    async def _publish_to_kafka(self, operator_id: str, statuses: List[GridStatus]) -> None:
        """Publish status data to Kafka for real-time monitoring"""
        try:
            for status in statuses:
                message = {
                    'operator_id': operator_id,
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
                    'data_quality_score': status.data_quality_score,
                    'is_anomaly': status.is_anomaly,
                    'anomaly_type': status.anomaly_type
                }
                
                await self.kafka_producer.send_message(
                    topic="grid-status",
                    message=message
                )
            
            logger.info(f"Published {len(statuses)} status records to Kafka")
            
        except Exception as e:
            logger.error(f"Error publishing to Kafka: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
    
    async def _archive_to_s3(
        self,
        operator_id: str,
        statuses: List[GridStatus],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Archive status data to S3 for long-term storage"""
        try:
            # Convert statuses to JSON format
            status_data = []
            for status in statuses:
                status_data.append({
                    'operator_id': operator_id,
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
                    'data_quality_score': status.data_quality_score,
                    'is_anomaly': status.is_anomaly,
                    'anomaly_type': status.anomaly_type
                })
            
            # Create S3 key with timestamp
            timestamp = datetime.utcnow().strftime("%Y/%m/%d/%H")
            s3_key = f"grid-status-data/{operator_id}/{timestamp}/status.json"
            
            # Upload to S3
            await self.s3_client.upload_data(
                data=status_data,
                s3_key=s3_key,
                content_type="application/json"
            )
            
            logger.info(f"Archived {len(statuses)} status records to S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"Error archiving to S3: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
    
    async def _check_and_send_alerts(
        self,
        operator: GridOperator,
        stability_results: Dict[str, Any],
        anomaly_results: Dict[str, Any]
    ) -> None:
        """Check for alerts and send notifications"""
        try:
            alerts_sent = 0
            
            # Check stability issues
            for issue in stability_results.get('stability_issues', []):
                if issue['severity'] == 'high':
                    await self.alerting_service.send_alert(
                        alert_type="grid_stability_high",
                        severity="critical",
                        message=f"High severity grid stability issue: {issue['type']}",
                        entity_id=operator.operator_id,
                        entity_type="grid_operator",
                        metadata=issue
                    )
                    alerts_sent += 1
                elif issue['severity'] == 'medium':
                    await self.alerting_service.send_alert(
                        alert_type="grid_stability_medium",
                        severity="warning",
                        message=f"Medium severity grid stability issue: {issue['type']}",
                        entity_id=operator.operator_id,
                        entity_type="grid_operator",
                        metadata=issue
                    )
                    alerts_sent += 1
            
            # Check anomalies
            if anomaly_results.get('anomaly_count', 0) > 5:
                await self.alerting_service.send_alert(
                    alert_type="grid_anomalies",
                    severity="warning",
                    message=f"Multiple grid anomalies detected: {anomaly_results.get('anomaly_count', 0)}",
                    entity_id=operator.operator_id,
                    entity_type="grid_operator",
                    metadata=anomaly_results
                )
                alerts_sent += 1
            
            logger.info(f"Sent {alerts_sent} alerts for operator {operator.operator_id}")
            
        except Exception as e:
            logger.error(f"Error sending alerts: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
    
    async def _update_operator_status(
        self,
        operator: GridOperator,
        stability_results: Dict[str, Any],
        anomaly_results: Dict[str, Any]
    ) -> None:
        """Update operator status based on stability and anomaly results"""
        try:
            stability_score = stability_results.get('overall_stability', 1.0)
            anomaly_count = anomaly_results.get('anomaly_count', 0)
            
            # Check if operator should be marked as maintenance due to stability issues
            if stability_score < 0.5 or anomaly_count > 20:
                if operator.status != GridOperatorStatus.MAINTENANCE:
                    operator.mark_as_maintenance()
                    await self.grid_operator_repository.update(operator)
                    logger.warning(f"Marked operator {operator.operator_id} as maintenance due to stability issues")
            
            # Check if operator should be marked as inactive due to critical issues
            elif stability_score < 0.3 or anomaly_count > 50:
                if operator.status != GridOperatorStatus.INACTIVE:
                    operator.mark_as_inactive()
                    await self.grid_operator_repository.update(operator)
                    logger.error(f"Marked operator {operator.operator_id} as inactive due to critical issues")
            
        except Exception as e:
            logger.error(f"Error updating operator status: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
