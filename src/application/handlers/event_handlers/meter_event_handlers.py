"""
Smart Meter Event Handlers
Handles domain events from smart meter entities
"""

from typing import Dict, Any
import logging

from ....core.domain.events.meter_events import (
    MeterRegisteredEvent,
    MeterStatusUpdatedEvent,
    MeterReadingAddedEvent,
    MeterQualityUpdatedEvent,
    MeterFaultDetectedEvent
)
from ....core.interfaces.external.alerting_service import IAlertingService
from ....infrastructure.external.kafka.kafka_producer import KafkaProducer
from ....infrastructure.external.s3.s3_client import S3Client

logger = logging.getLogger(__name__)


class MeterEventHandlers:
    """
    Event handlers for smart meter domain events
    
    Handles the side effects of smart meter domain events
    such as notifications, logging, and external system updates.
    """
    
    def __init__(
        self,
        alerting_service: IAlertingService,
        kafka_producer: KafkaProducer,
        s3_client: S3Client
    ):
        self.alerting_service = alerting_service
        self.kafka_producer = kafka_producer
        self.s3_client = s3_client
    
    async def handle_meter_registered(self, event: MeterRegisteredEvent) -> None:
        """Handle meter registration event"""
        try:
            logger.info(f"Handling meter registered event: {event.meter_id}")
            
            # Publish to Kafka for real-time processing
            await self.kafka_producer.send_message(
                topic="meter-events",
                message={
                    "event_type": "meter_registered",
                    "meter_id": event.meter_id,
                    "timestamp": event.occurred_at.isoformat(),
                    "data": event.event_data
                }
            )
            
            # Send notification to operations team
            await self.alerting_service.send_alert(
                alert_type="meter_registered",
                severity="info",
                message=f"New smart meter registered: {event.meter_id}",
                entity_id=event.meter_id,
                entity_type="smart_meter",
                metadata=event.event_data
            )
            
            # Archive event to S3 for audit trail
            await self._archive_event_to_s3("meter_registered", event)
            
            logger.info(f"Successfully handled meter registered event: {event.meter_id}")
            
        except Exception as e:
            logger.error(f"Error handling meter registered event: {str(e)}")
            # Don't raise exception to avoid breaking the main flow
    
    async def handle_meter_status_updated(self, event: MeterStatusUpdatedEvent) -> None:
        """Handle meter status update event"""
        try:
            logger.info(f"Handling meter status updated event: {event.meter_id}")
            
            # Publish to Kafka for real-time processing
            await self.kafka_producer.send_message(
                topic="meter-events",
                message={
                    "event_type": "meter_status_updated",
                    "meter_id": event.meter_id,
                    "timestamp": event.occurred_at.isoformat(),
                    "data": event.event_data
                }
            )
            
            # Send alert if status changed to faulty or maintenance
            new_status = event.event_data.get("new_status")
            if new_status in ["FAULTY", "MAINTENANCE"]:
                await self.alerting_service.send_alert(
                    alert_type="meter_status_change",
                    severity="warning" if new_status == "MAINTENANCE" else "critical",
                    message=f"Smart meter status changed to {new_status}: {event.meter_id}",
                    entity_id=event.meter_id,
                    entity_type="smart_meter",
                    metadata=event.event_data
                )
            
            # Archive event to S3 for audit trail
            await self._archive_event_to_s3("meter_status_updated", event)
            
            logger.info(f"Successfully handled meter status updated event: {event.meter_id}")
            
        except Exception as e:
            logger.error(f"Error handling meter status updated event: {str(e)}")
            # Don't raise exception to avoid breaking the main flow
    
    async def handle_meter_reading_added(self, event: MeterReadingAddedEvent) -> None:
        """Handle meter reading added event"""
        try:
            logger.info(f"Handling meter reading added event: {event.meter_id}")
            
            # Publish to Kafka for real-time processing
            await self.kafka_producer.send_message(
                topic="meter-readings",
                message={
                    "event_type": "meter_reading_added",
                    "meter_id": event.meter_id,
                    "timestamp": event.occurred_at.isoformat(),
                    "data": event.event_data
                }
            )
            
            # Check for unusual readings and send alerts
            reading_data = event.event_data.get("reading", {})
            if self._is_unusual_reading(reading_data):
                await self.alerting_service.send_alert(
                    alert_type="unusual_meter_reading",
                    severity="warning",
                    message=f"Unusual reading detected for meter {event.meter_id}",
                    entity_id=event.meter_id,
                    entity_type="smart_meter",
                    metadata=reading_data
                )
            
            # Archive event to S3 for audit trail
            await self._archive_event_to_s3("meter_reading_added", event)
            
            logger.info(f"Successfully handled meter reading added event: {event.meter_id}")
            
        except Exception as e:
            logger.error(f"Error handling meter reading added event: {str(e)}")
            # Don't raise exception to avoid breaking the main flow
    
    async def handle_meter_quality_updated(self, event: MeterQualityUpdatedEvent) -> None:
        """Handle meter quality update event"""
        try:
            logger.info(f"Handling meter quality updated event: {event.meter_id}")
            
            # Publish to Kafka for real-time processing
            await self.kafka_producer.send_message(
                topic="meter-events",
                message={
                    "event_type": "meter_quality_updated",
                    "meter_id": event.meter_id,
                    "timestamp": event.occurred_at.isoformat(),
                    "data": event.event_data
                }
            )
            
            # Send alert if quality dropped significantly
            new_quality_tier = event.event_data.get("new_quality_tier")
            old_quality_tier = event.event_data.get("old_quality_tier")
            
            if self._is_quality_degradation(old_quality_tier, new_quality_tier):
                await self.alerting_service.send_alert(
                    alert_type="meter_quality_degradation",
                    severity="warning",
                    message=f"Meter quality degraded from {old_quality_tier} to {new_quality_tier}: {event.meter_id}",
                    entity_id=event.meter_id,
                    entity_type="smart_meter",
                    metadata=event.event_data
                )
            
            # Archive event to S3 for audit trail
            await self._archive_event_to_s3("meter_quality_updated", event)
            
            logger.info(f"Successfully handled meter quality updated event: {event.meter_id}")
            
        except Exception as e:
            logger.error(f"Error handling meter quality updated event: {str(e)}")
            # Don't raise exception to avoid breaking the main flow
    
    async def handle_meter_fault_detected(self, event: MeterFaultDetectedEvent) -> None:
        """Handle meter fault detected event"""
        try:
            logger.info(f"Handling meter fault detected event: {event.meter_id}")
            
            # Publish to Kafka for real-time processing
            await self.kafka_producer.send_message(
                topic="meter-events",
                message={
                    "event_type": "meter_fault_detected",
                    "meter_id": event.meter_id,
                    "timestamp": event.occurred_at.isoformat(),
                    "data": event.event_data
                }
            )
            
            # Send critical alert for fault detection
            await self.alerting_service.send_alert(
                alert_type="meter_fault_detected",
                severity="critical",
                message=f"Fault detected in smart meter: {event.meter_id}",
                entity_id=event.meter_id,
                entity_type="smart_meter",
                metadata=event.event_data
            )
            
            # Archive event to S3 for audit trail
            await self._archive_event_to_s3("meter_fault_detected", event)
            
            logger.info(f"Successfully handled meter fault detected event: {event.meter_id}")
            
        except Exception as e:
            logger.error(f"Error handling meter fault detected event: {str(e)}")
            # Don't raise exception to avoid breaking the main flow
    
    def _is_unusual_reading(self, reading_data: Dict[str, Any]) -> bool:
        """Check if a reading is unusual and requires attention"""
        try:
            voltage = reading_data.get("voltage", 0)
            current = reading_data.get("current", 0)
            frequency = reading_data.get("frequency", 0)
            
            # Check for voltage anomalies
            if voltage < 200 or voltage > 250:
                return True
            
            # Check for current anomalies
            if current < 0 or current > 100:
                return True
            
            # Check for frequency anomalies
            if frequency < 49.5 or frequency > 50.5:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking unusual reading: {str(e)}")
            return False
    
    def _is_quality_degradation(self, old_tier: str, new_tier: str) -> bool:
        """Check if there's a significant quality degradation"""
        quality_order = ["EXCELLENT", "GOOD", "FAIR", "POOR", "UNKNOWN"]
        
        try:
            old_index = quality_order.index(old_tier) if old_tier in quality_order else 4
            new_index = quality_order.index(new_tier) if new_tier in quality_order else 4
            
            # Consider it degradation if quality dropped by 2 or more levels
            return new_index - old_index >= 2
            
        except Exception as e:
            logger.error(f"Error checking quality degradation: {str(e)}")
            return False
    
    async def _archive_event_to_s3(self, event_type: str, event: Any) -> None:
        """Archive event to S3 for audit trail"""
        try:
            event_data = {
                "event_type": event_type,
                "meter_id": event.meter_id,
                "timestamp": event.occurred_at.isoformat(),
                "event_data": event.event_data,
                "aggregate_version": event.aggregate_version,
                "event_version": event.event_version
            }
            
            # Create S3 key with timestamp
            from datetime import datetime
            timestamp = event.occurred_at.strftime("%Y/%m/%d/%H")
            s3_key = f"meter-events/{event.meter_id}/{timestamp}/{event_type}_{event.occurred_at.timestamp()}.json"
            
            # Upload to S3
            await self.s3_client.upload_data(
                data=event_data,
                s3_key=s3_key,
                content_type="application/json"
            )
            
            logger.debug(f"Archived {event_type} event to S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"Error archiving event to S3: {str(e)}")
            # Don't raise exception as this is not critical
