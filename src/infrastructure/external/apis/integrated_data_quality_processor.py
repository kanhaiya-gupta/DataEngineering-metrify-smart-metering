"""
Integrated Data Quality Processor
Combines validation, anomaly detection, monitoring, and dead letter queue management
"""

import asyncio
import time
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from ....core.config.config_loader import ConfigLoader
from ....core.exceptions.domain_exceptions import DataQualityError
from .config_driven_data_quality_service import ConfigDrivenDataQualityService, ValidationResult, AnomalyResult
from .data_quality_monitoring_service import DataQualityMonitoringService, QualityMetrics
from .dead_letter_queue_service import DeadLetterQueueService, FailureReason, RetryStrategy

logger = logging.getLogger(__name__)


class IntegratedDataQualityProcessor:
    """
    Integrated data quality processor that orchestrates all quality-related services
    """
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.data_sources_config = self.config_loader.get_data_sources_config()
        
        # Initialize services
        self.quality_service = ConfigDrivenDataQualityService()
        self.monitoring_service = DataQualityMonitoringService()
        self.dlq_service = DeadLetterQueueService()
        
        # Quality thresholds
        self.quality_thresholds = {
            'smart_meter': 0.8,
            'grid_operator': 0.8,
            'weather': 0.8
        }
        
        # Anomaly thresholds
        self.anomaly_thresholds = {
            'smart_meter': 0.7,
            'grid_operator': 0.7,
            'weather': 0.7
        }
        
        # Register processing callbacks for DLQ
        self._register_dlq_callbacks()
    
    def _register_dlq_callbacks(self):
        """Register callbacks for dead letter queue processing"""
        self.dlq_service.register_processing_callback('smart_meter', self._retry_smart_meter_processing)
        self.dlq_service.register_processing_callback('grid_operator', self._retry_grid_operator_processing)
        self.dlq_service.register_processing_callback('weather', self._retry_weather_processing)
    
    async def process_smart_meter_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Process smart meter data with full quality pipeline"""
        return await self._process_data(data, 'smart_meter')
    
    async def process_grid_operator_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Process grid operator data with full quality pipeline"""
        return await self._process_data(data, 'grid_operator')
    
    async def process_weather_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Process weather data with full quality pipeline"""
        return await self._process_data(data, 'weather')
    
    async def _process_data(self, data: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Generic data processing with quality pipeline"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting quality processing for {data_type} data: {len(data)} records")
            
            # 1. Validate data
            validation_result = await self._validate_data(data, data_type)
            
            # 2. Detect anomalies
            anomaly_result = await self._detect_anomalies(data, data_type)
            
            # 3. Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # 4. Process quality metrics
            quality_metrics = await self.monitoring_service.process_quality_metrics(
                data_type, validation_result, anomaly_result, processing_time_ms
            )
            
            # 5. Determine if data should be processed or sent to DLQ
            should_process = await self._should_process_data(
                validation_result, anomaly_result, data_type
            )
            
            # 6. Handle data based on quality assessment
            if should_process:
                result = await self._handle_quality_data(data, validation_result, anomaly_result, data_type)
            else:
                result = await self._handle_poor_quality_data(data, validation_result, anomaly_result, data_type)
            
            # 7. Add quality metrics to result
            result.update({
                'quality_metrics': {
                    'quality_score': quality_metrics.quality_score,
                    'validation_passed': quality_metrics.validation_passed,
                    'anomaly_detected': quality_metrics.anomaly_detected,
                    'violation_count': quality_metrics.violation_count,
                    'processing_time_ms': quality_metrics.processing_time_ms
                },
                'processing_timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info(f"Completed quality processing for {data_type}: "
                       f"score={quality_metrics.quality_score:.3f}, "
                       f"processed={should_process}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {data_type} data: {e}")
            
            # Send to DLQ on processing error
            await self._send_to_dlq(data, data_type, FailureReason.PROCESSING_ERROR, str(e))
            
            return {
                'success': False,
                'error': str(e),
                'data_sent_to_dlq': True,
                'processing_timestamp': datetime.utcnow().isoformat()
            }
    
    async def _validate_data(self, data: pd.DataFrame, data_type: str) -> ValidationResult:
        """Validate data using configuration-driven rules"""
        try:
            if data_type == 'smart_meter':
                return await self.quality_service.validate_smart_meter_data(data)
            elif data_type == 'grid_operator':
                return await self.quality_service.validate_grid_operator_data(data)
            elif data_type == 'weather':
                return await self.quality_service.validate_weather_data(data)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
                
        except Exception as e:
            logger.error(f"Error validating {data_type} data: {e}")
            raise DataQualityError(f"Validation failed: {e}")
    
    async def _detect_anomalies(self, data: pd.DataFrame, data_type: str) -> AnomalyResult:
        """Detect anomalies in data"""
        try:
            return await self.quality_service.detect_anomalies(data, data_type)
            
        except Exception as e:
            logger.error(f"Error detecting anomalies in {data_type} data: {e}")
            # Return no anomalies on error
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_types=[],
                confidence=0.0,
                details={'error': str(e)}
            )
    
    async def _should_process_data(self, validation_result: ValidationResult, 
                                 anomaly_result: AnomalyResult, data_type: str) -> bool:
        """Determine if data should be processed or sent to DLQ"""
        try:
            # Check quality threshold
            quality_threshold = self.quality_thresholds.get(data_type, 0.8)
            if validation_result.quality_score < quality_threshold:
                logger.warning(f"Quality score {validation_result.quality_score:.3f} below threshold {quality_threshold}")
                return False
            
            # Check anomaly threshold
            anomaly_threshold = self.anomaly_thresholds.get(data_type, 0.7)
            if anomaly_result.is_anomaly and anomaly_result.anomaly_score > anomaly_threshold:
                logger.warning(f"High anomaly score {anomaly_result.anomaly_score:.3f} above threshold {anomaly_threshold}")
                return False
            
            # Check for critical violations
            critical_violations = [
                v for v in validation_result.violations 
                if any(keyword in v.lower() for keyword in ['missing required', 'type mismatch', 'null values'])
            ]
            if critical_violations:
                logger.warning(f"Critical violations detected: {critical_violations}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error determining data processing decision: {e}")
            return False
    
    async def _handle_quality_data(self, data: pd.DataFrame, validation_result: ValidationResult,
                                 anomaly_result: AnomalyResult, data_type: str) -> Dict[str, Any]:
        """Handle data that passes quality checks"""
        try:
            # Add quality metadata to data
            data_with_quality = data.copy()
            data_with_quality['_quality_score'] = validation_result.quality_score
            data_with_quality['_is_anomaly'] = anomaly_result.is_anomaly
            data_with_quality['_anomaly_score'] = anomaly_result.anomaly_score
            data_with_quality['_validation_timestamp'] = datetime.utcnow().isoformat()
            
            # Process the data (this would typically send to next stage)
            processed_count = len(data_with_quality)
            
            logger.info(f"Successfully processed {processed_count} {data_type} records")
            
            return {
                'success': True,
                'processed_records': processed_count,
                'quality_score': validation_result.quality_score,
                'anomaly_detected': anomaly_result.is_anomaly,
                'data_ready_for_processing': True,
                'quality_metadata_added': True
            }
            
        except Exception as e:
            logger.error(f"Error handling quality data: {e}")
            raise DataQualityError(f"Failed to handle quality data: {e}")
    
    async def _handle_poor_quality_data(self, data: pd.DataFrame, validation_result: ValidationResult,
                                      anomaly_result: AnomalyResult, data_type: str) -> Dict[str, Any]:
        """Handle data that fails quality checks"""
        try:
            # Determine failure reason
            failure_reason = self._determine_failure_reason(validation_result, anomaly_result)
            
            # Send to dead letter queue
            for _, record in data.iterrows():
                await self._send_to_dlq(record.to_dict(), data_type, failure_reason, 
                                      f"Quality score: {validation_result.quality_score:.3f}")
            
            logger.warning(f"Sent {len(data)} {data_type} records to DLQ due to poor quality")
            
            return {
                'success': False,
                'processed_records': 0,
                'data_sent_to_dlq': True,
                'failure_reason': failure_reason.value,
                'quality_score': validation_result.quality_score,
                'violation_count': len(validation_result.violations),
                'anomaly_detected': anomaly_result.is_anomaly
            }
            
        except Exception as e:
            logger.error(f"Error handling poor quality data: {e}")
            raise DataQualityError(f"Failed to handle poor quality data: {e}")
    
    def _determine_failure_reason(self, validation_result: ValidationResult, 
                                anomaly_result: AnomalyResult) -> FailureReason:
        """Determine the primary reason for data failure"""
        if not validation_result.is_valid:
            return FailureReason.VALIDATION_FAILED
        elif validation_result.quality_score < 0.5:
            return FailureReason.QUALITY_THRESHOLD_BREACHED
        elif anomaly_result.is_anomaly and anomaly_result.anomaly_score > 0.8:
            return FailureReason.ANOMALY_DETECTED
        else:
            return FailureReason.UNKNOWN
    
    async def _send_to_dlq(self, data: Dict[str, Any], data_type: str, 
                          failure_reason: FailureReason, error_message: str):
        """Send data to dead letter queue"""
        try:
            await self.dlq_service.add_failed_record(
                data_type=data_type,
                original_data=data,
                failure_reason=failure_reason,
                error_message=error_message,
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
            )
            
        except Exception as e:
            logger.error(f"Error sending data to DLQ: {e}")
    
    # DLQ retry callbacks
    async def _retry_smart_meter_processing(self, data: Dict[str, Any]) -> bool:
        """Retry processing smart meter data"""
        try:
            # Convert dict back to DataFrame
            df = pd.DataFrame([data])
            result = await self.process_smart_meter_data(df)
            return result.get('success', False)
        except Exception as e:
            logger.error(f"Error retrying smart meter processing: {e}")
            return False
    
    async def _retry_grid_operator_processing(self, data: Dict[str, Any]) -> bool:
        """Retry processing grid operator data"""
        try:
            df = pd.DataFrame([data])
            result = await self.process_grid_operator_data(df)
            return result.get('success', False)
        except Exception as e:
            logger.error(f"Error retrying grid operator processing: {e}")
            return False
    
    async def _retry_weather_processing(self, data: Dict[str, Any]) -> bool:
        """Retry processing weather data"""
        try:
            df = pd.DataFrame([data])
            result = await self.process_weather_data(df)
            return result.get('success', False)
        except Exception as e:
            logger.error(f"Error retrying weather processing: {e}")
            return False
    
    # Monitoring and reporting methods
    async def get_quality_dashboard(self) -> Dict[str, Any]:
        """Get quality dashboard data"""
        try:
            return await self.monitoring_service.get_quality_dashboard_data()
        except Exception as e:
            logger.error(f"Error getting quality dashboard: {e}")
            return {'error': str(e)}
    
    async def get_dlq_status(self) -> Dict[str, Any]:
        """Get dead letter queue status"""
        try:
            return await self.dlq_service.get_dlq_statistics()
        except Exception as e:
            logger.error(f"Error getting DLQ status: {e}")
            return {'error': str(e)}
    
    async def get_quality_summary(self, data_type: str, hours: int = 24) -> Dict[str, Any]:
        """Get quality summary for a specific data type"""
        try:
            return await self.monitoring_service.get_quality_summary(data_type, hours)
        except Exception as e:
            logger.error(f"Error getting quality summary: {e}")
            return {'error': str(e)}
    
    async def get_active_alerts(self, data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active quality alerts"""
        try:
            return await self.monitoring_service.get_active_alerts(data_type)
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    async def export_quality_report(self, data_type: str, hours: int = 24) -> Dict[str, Any]:
        """Export comprehensive quality report"""
        try:
            quality_report = await self.monitoring_service.export_quality_report(data_type, hours)
            dlq_data = await self.dlq_service.export_dlq_data(data_type)
            
            return {
                'quality_report': quality_report,
                'dlq_data': dlq_data,
                'export_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error exporting quality report: {e}")
            return {'error': str(e)}
