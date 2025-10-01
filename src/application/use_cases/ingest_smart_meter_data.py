"""
Smart Meter Data Ingestion Use Case
Handles the ingestion and processing of smart meter data
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ...core.domain.entities.smart_meter import SmartMeter, MeterReading
from ...core.domain.value_objects.meter_id import MeterId
from ...core.domain.value_objects.location import Location
from ...core.domain.value_objects.meter_specifications import MeterSpecifications
from ...core.domain.enums.meter_status import MeterStatus
from ...core.domain.enums.quality_tier import QualityTier
from ...core.interfaces.repositories.smart_meter_repository import ISmartMeterRepository
from ...core.interfaces.external.data_quality_service import IDataQualityService
from ...core.interfaces.external.anomaly_detection_service import IAnomalyDetectionService
from ...core.exceptions.domain_exceptions import MeterNotFoundException, InvalidMeterReadingError
from ...infrastructure.external.kafka.kafka_producer import KafkaProducer
from ...infrastructure.external.s3.s3_client import S3Client

logger = logging.getLogger(__name__)


class IngestSmartMeterDataUseCase:
    """
    Use case for ingesting smart meter data
    
    Handles the complete flow of smart meter data ingestion,
    including validation, quality assessment, anomaly detection,
    and storage.
    """
    
    def __init__(
        self,
        smart_meter_repository: ISmartMeterRepository,
        data_quality_service: IDataQualityService,
        anomaly_detection_service: IAnomalyDetectionService,
        kafka_producer: KafkaProducer,
        s3_client: S3Client
    ):
        self.smart_meter_repository = smart_meter_repository
        self.data_quality_service = data_quality_service
        self.anomaly_detection_service = anomaly_detection_service
        self.kafka_producer = kafka_producer
        self.s3_client = s3_client
    
    async def execute(
        self,
        meter_id: str,
        readings_data: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute smart meter data ingestion
        
        Args:
            meter_id: Smart meter identifier
            readings_data: List of meter reading data
            metadata: Optional metadata for the ingestion
            
        Returns:
            Dictionary containing ingestion results
        """
        try:
            logger.info(f"Starting smart meter data ingestion for meter {meter_id}")
            
            # Step 1: Validate meter exists
            meter = await self._validate_meter_exists(meter_id)
            
            # Step 2: Process and validate readings
            processed_readings = await self._process_readings(meter_id, readings_data)
            
            # Step 3: Assess data quality
            quality_results = await self._assess_data_quality(processed_readings)
            
            # Step 4: Detect anomalies
            anomaly_results = await self._detect_anomalies(processed_readings)
            
            # Step 5: Store readings in database
            stored_readings = await self._store_readings(processed_readings, quality_results, anomaly_results)
            
            # Step 6: Publish to Kafka for real-time processing
            await self._publish_to_kafka(meter_id, stored_readings)
            
            # Step 7: Archive to S3 for long-term storage
            await self._archive_to_s3(meter_id, stored_readings, metadata)
            
            # Step 8: Update meter status if needed
            await self._update_meter_status(meter, quality_results, anomaly_results)
            
            result = {
                "status": "success",
                "meter_id": meter_id,
                "readings_processed": len(processed_readings),
                "quality_score": quality_results.get("overall_score", 0.0),
                "anomalies_detected": anomaly_results.get("anomaly_count", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Smart meter data ingestion completed for meter {meter_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in smart meter data ingestion: {str(e)}")
            raise
    
    async def _validate_meter_exists(self, meter_id: str) -> SmartMeter:
        """Validate that the smart meter exists"""
        try:
            meter = await self.smart_meter_repository.get_by_id(MeterId(meter_id))
            if not meter:
                raise MeterNotFoundException(f"Smart meter {meter_id} not found")
            return meter
        except Exception as e:
            logger.error(f"Error validating meter {meter_id}: {str(e)}")
            raise
    
    async def _process_readings(
        self,
        meter_id: str,
        readings_data: List[Dict[str, Any]]
    ) -> List[MeterReading]:
        """Process and validate meter readings"""
        processed_readings = []
        
        for reading_data in readings_data:
            try:
                # Validate required fields
                required_fields = ['timestamp', 'voltage', 'current', 'power_factor', 'frequency']
                for field in required_fields:
                    if field not in reading_data:
                        raise InvalidMeterReadingError(f"Missing required field: {field}")
                
                # Create MeterReading object
                reading = MeterReading(
                    meter_id=MeterId(meter_id),
                    timestamp=datetime.fromisoformat(reading_data['timestamp'].replace('Z', '+00:00')),
                    voltage=float(reading_data['voltage']),
                    current=float(reading_data['current']),
                    power_factor=float(reading_data['power_factor']),
                    frequency=float(reading_data['frequency']),
                    active_power=float(reading_data.get('active_power', 0.0)),
                    reactive_power=float(reading_data.get('reactive_power', 0.0)),
                    apparent_power=float(reading_data.get('apparent_power', 0.0))
                )
                
                processed_readings.append(reading)
                
            except Exception as e:
                logger.warning(f"Error processing reading: {str(e)}")
                continue
        
        logger.info(f"Processed {len(processed_readings)} readings for meter {meter_id}")
        return processed_readings
    
    async def _assess_data_quality(self, readings: List[MeterReading]) -> Dict[str, Any]:
        """Assess data quality of the readings"""
        try:
            # Convert readings to format expected by quality service
            readings_data = []
            for reading in readings:
                readings_data.append({
                    'timestamp': reading.timestamp.isoformat(),
                    'voltage': reading.voltage,
                    'current': reading.current,
                    'power_factor': reading.power_factor,
                    'frequency': reading.frequency,
                    'active_power': reading.active_power,
                    'reactive_power': reading.reactive_power,
                    'apparent_power': reading.apparent_power
                })
            
            quality_results = await self.data_quality_service.assess_quality(
                data=readings_data,
                data_type="smart_meter_readings"
            )
            
            logger.info(f"Data quality assessment completed: {quality_results.get('overall_score', 0.0)}")
            return quality_results
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            return {"overall_score": 0.0, "quality_issues": []}
    
    async def _detect_anomalies(self, readings: List[MeterReading]) -> Dict[str, Any]:
        """Detect anomalies in the readings"""
        try:
            # Convert readings to format expected by anomaly detection service
            readings_data = []
            for reading in readings:
                readings_data.append({
                    'timestamp': reading.timestamp.isoformat(),
                    'voltage': reading.voltage,
                    'current': reading.current,
                    'power_factor': reading.power_factor,
                    'frequency': reading.frequency,
                    'active_power': reading.active_power,
                    'reactive_power': reading.reactive_power,
                    'apparent_power': reading.apparent_power
                })
            
            anomaly_results = await self.anomaly_detection_service.detect_anomalies(
                data=readings_data,
                data_type="smart_meter_readings"
            )
            
            logger.info(f"Anomaly detection completed: {anomaly_results.get('anomaly_count', 0)} anomalies found")
            return anomaly_results
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return {"anomaly_count": 0, "anomalies": []}
    
    async def _store_readings(
        self,
        readings: List[MeterReading],
        quality_results: Dict[str, Any],
        anomaly_results: Dict[str, Any]
    ) -> List[MeterReading]:
        """Store readings in the database"""
        try:
            stored_readings = []
            
            for reading in readings:
                # Add quality and anomaly information
                reading.data_quality_score = quality_results.get('overall_score', 1.0)
                reading.is_anomaly = reading in anomaly_results.get('anomalies', [])
                reading.anomaly_type = anomaly_results.get('anomaly_type', None)
                
                # Store in database
                stored_reading = await self.smart_meter_repository.add_reading(reading)
                stored_readings.append(stored_reading)
            
            logger.info(f"Stored {len(stored_readings)} readings in database")
            return stored_readings
            
        except Exception as e:
            logger.error(f"Error storing readings: {str(e)}")
            raise
    
    async def _publish_to_kafka(self, meter_id: str, readings: List[MeterReading]) -> None:
        """Publish readings to Kafka for real-time processing"""
        try:
            for reading in readings:
                message = {
                    'meter_id': meter_id,
                    'timestamp': reading.timestamp.isoformat(),
                    'voltage': reading.voltage,
                    'current': reading.current,
                    'power_factor': reading.power_factor,
                    'frequency': reading.frequency,
                    'active_power': reading.active_power,
                    'reactive_power': reading.reactive_power,
                    'apparent_power': reading.apparent_power,
                    'data_quality_score': reading.data_quality_score,
                    'is_anomaly': reading.is_anomaly,
                    'anomaly_type': reading.anomaly_type
                }
                
                await self.kafka_producer.send_message(
                    topic="smart-meter-readings",
                    message=message
                )
            
            logger.info(f"Published {len(readings)} readings to Kafka")
            
        except Exception as e:
            logger.error(f"Error publishing to Kafka: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
    
    async def _archive_to_s3(
        self,
        meter_id: str,
        readings: List[MeterReading],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Archive readings to S3 for long-term storage"""
        try:
            # Convert readings to JSON format
            readings_data = []
            for reading in readings:
                readings_data.append({
                    'meter_id': meter_id,
                    'timestamp': reading.timestamp.isoformat(),
                    'voltage': reading.voltage,
                    'current': reading.current,
                    'power_factor': reading.power_factor,
                    'frequency': reading.frequency,
                    'active_power': reading.active_power,
                    'reactive_power': reading.reactive_power,
                    'apparent_power': reading.apparent_power,
                    'data_quality_score': reading.data_quality_score,
                    'is_anomaly': reading.is_anomaly,
                    'anomaly_type': reading.anomaly_type
                })
            
            # Create S3 key with timestamp
            timestamp = datetime.utcnow().strftime("%Y/%m/%d/%H")
            s3_key = f"smart-meter-data/{meter_id}/{timestamp}/readings.json"
            
            # Upload to S3
            await self.s3_client.upload_data(
                data=readings_data,
                s3_key=s3_key,
                content_type="application/json"
            )
            
            logger.info(f"Archived {len(readings)} readings to S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"Error archiving to S3: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
    
    async def _update_meter_status(
        self,
        meter: SmartMeter,
        quality_results: Dict[str, Any],
        anomaly_results: Dict[str, Any]
    ) -> None:
        """Update meter status based on quality and anomaly results"""
        try:
            quality_score = quality_results.get('overall_score', 1.0)
            anomaly_count = anomaly_results.get('anomaly_count', 0)
            
            # Determine quality tier based on score
            if quality_score >= 0.9:
                new_quality_tier = QualityTier.EXCELLENT
            elif quality_score >= 0.8:
                new_quality_tier = QualityTier.GOOD
            elif quality_score >= 0.7:
                new_quality_tier = QualityTier.FAIR
            elif quality_score >= 0.5:
                new_quality_tier = QualityTier.POOR
            else:
                new_quality_tier = QualityTier.UNKNOWN
            
            # Update meter if quality tier changed
            if meter.quality_tier != new_quality_tier:
                meter.update_quality_tier(new_quality_tier)
                await self.smart_meter_repository.update(meter)
                logger.info(f"Updated meter {meter.meter_id.value} quality tier to {new_quality_tier.value}")
            
            # Check if meter should be marked as faulty due to high anomaly rate
            if anomaly_count > 10:  # Threshold for marking as faulty
                if meter.status != MeterStatus.FAULTY:
                    meter.mark_as_faulty()
                    await self.smart_meter_repository.update(meter)
                    logger.warning(f"Marked meter {meter.meter_id.value} as faulty due to high anomaly count")
            
        except Exception as e:
            logger.error(f"Error updating meter status: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
