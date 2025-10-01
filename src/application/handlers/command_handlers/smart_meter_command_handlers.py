"""
Smart Meter Command Handlers
Handles commands for smart meter operations
"""

from typing import Dict, Any, List
import logging

from ....core.domain.entities.smart_meter import SmartMeter
from ....core.domain.value_objects.meter_id import MeterId
from ....core.domain.value_objects.location import Location
from ....core.domain.value_objects.meter_specifications import MeterSpecifications
from ....core.domain.enums.meter_status import MeterStatus
from ....core.domain.enums.quality_tier import QualityTier
from ....core.interfaces.repositories.smart_meter_repository import ISmartMeterRepository
from ....core.interfaces.external.data_quality_service import IDataQualityService
from ....core.interfaces.external.anomaly_detection_service import IAnomalyDetectionService
from ....core.exceptions.domain_exceptions import MeterNotFoundException, InvalidMeterOperationError
from ....infrastructure.external.kafka.kafka_producer import KafkaProducer
from ....infrastructure.external.s3.s3_client import S3Client

logger = logging.getLogger(__name__)


class SmartMeterCommandHandlers:
    """
    Command handlers for smart meter operations
    
    Handles high-level commands for smart meter management
    and coordinates between different services.
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
    
    async def handle_register_meter_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle register meter command"""
        try:
            logger.info(f"Handling register meter command: {command_data.get('meter_id')}")
            
            # Extract command data
            meter_id = command_data["meter_id"]
            location_data = command_data["location"]
            specifications_data = command_data["specifications"]
            status = MeterStatus(command_data.get("status", "ACTIVE"))
            quality_tier = QualityTier(command_data.get("quality_tier", "UNKNOWN"))
            metadata = command_data.get("metadata", {})
            
            # Create domain objects
            location = Location(
                latitude=location_data["latitude"],
                longitude=location_data["longitude"],
                address=location_data["address"]
            )
            
            specifications = MeterSpecifications(
                manufacturer=specifications_data["manufacturer"],
                model=specifications_data["model"],
                firmware_version=specifications_data.get("firmware_version", "1.0.0"),
                installation_date=specifications_data["installation_date"]
            )
            
            # Create smart meter entity
            meter = SmartMeter(
                meter_id=MeterId(meter_id),
                location=location,
                specifications=specifications,
                status=status,
                quality_tier=quality_tier,
                metadata=metadata
            )
            
            # Save to repository
            saved_meter = await self.smart_meter_repository.save(meter)
            
            # Publish event to Kafka
            await self.kafka_producer.send_message(
                topic="meter-commands",
                message={
                    "command_type": "register_meter",
                    "meter_id": meter_id,
                    "timestamp": saved_meter.created_at.isoformat(),
                    "data": command_data
                }
            )
            
            # Archive command to S3
            await self._archive_command_to_s3("register_meter", command_data)
            
            result = {
                "status": "success",
                "meter_id": meter_id,
                "message": "Meter registered successfully",
                "meter_data": {
                    "meter_id": saved_meter.meter_id.value,
                    "status": saved_meter.status.value,
                    "quality_tier": saved_meter.quality_tier.value,
                    "created_at": saved_meter.created_at.isoformat()
                }
            }
            
            logger.info(f"Successfully handled register meter command: {meter_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error handling register meter command: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to register meter: {str(e)}",
                "error_type": type(e).__name__
            }
    
    async def handle_update_meter_status_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update meter status command"""
        try:
            meter_id = command_data["meter_id"]
            new_status = MeterStatus(command_data["new_status"])
            reason = command_data.get("reason", "")
            
            logger.info(f"Handling update meter status command: {meter_id} -> {new_status.value}")
            
            # Get existing meter
            meter = await self.smart_meter_repository.get_by_id(MeterId(meter_id))
            if not meter:
                raise MeterNotFoundException(f"Meter {meter_id} not found")
            
            # Update status
            old_status = meter.status
            meter.update_status(new_status, reason)
            
            # Save updated meter
            updated_meter = await self.smart_meter_repository.update(meter)
            
            # Publish event to Kafka
            await self.kafka_producer.send_message(
                topic="meter-commands",
                message={
                    "command_type": "update_meter_status",
                    "meter_id": meter_id,
                    "timestamp": updated_meter.updated_at.isoformat(),
                    "data": {
                        "old_status": old_status.value,
                        "new_status": new_status.value,
                        "reason": reason
                    }
                }
            )
            
            # Archive command to S3
            await self._archive_command_to_s3("update_meter_status", command_data)
            
            result = {
                "status": "success",
                "meter_id": meter_id,
                "message": f"Meter status updated from {old_status.value} to {new_status.value}",
                "old_status": old_status.value,
                "new_status": new_status.value,
                "reason": reason
            }
            
            logger.info(f"Successfully handled update meter status command: {meter_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error handling update meter status command: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to update meter status: {str(e)}",
                "error_type": type(e).__name__
            }
    
    async def handle_update_meter_quality_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update meter quality command"""
        try:
            meter_id = command_data["meter_id"]
            new_quality_tier = QualityTier(command_data["new_quality_tier"])
            reason = command_data.get("reason", "")
            
            logger.info(f"Handling update meter quality command: {meter_id} -> {new_quality_tier.value}")
            
            # Get existing meter
            meter = await self.smart_meter_repository.get_by_id(MeterId(meter_id))
            if not meter:
                raise MeterNotFoundException(f"Meter {meter_id} not found")
            
            # Update quality tier
            old_quality_tier = meter.quality_tier
            meter.update_quality_tier(new_quality_tier)
            
            # Save updated meter
            updated_meter = await self.smart_meter_repository.update(meter)
            
            # Publish event to Kafka
            await self.kafka_producer.send_message(
                topic="meter-commands",
                message={
                    "command_type": "update_meter_quality",
                    "meter_id": meter_id,
                    "timestamp": updated_meter.updated_at.isoformat(),
                    "data": {
                        "old_quality_tier": old_quality_tier.value,
                        "new_quality_tier": new_quality_tier.value,
                        "reason": reason
                    }
                }
            )
            
            # Archive command to S3
            await self._archive_command_to_s3("update_meter_quality", command_data)
            
            result = {
                "status": "success",
                "meter_id": meter_id,
                "message": f"Meter quality updated from {old_quality_tier.value} to {new_quality_tier.value}",
                "old_quality_tier": old_quality_tier.value,
                "new_quality_tier": new_quality_tier.value,
                "reason": reason
            }
            
            logger.info(f"Successfully handled update meter quality command: {meter_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error handling update meter quality command: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to update meter quality: {str(e)}",
                "error_type": type(e).__name__
            }
    
    async def handle_mark_meter_faulty_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mark meter faulty command"""
        try:
            meter_id = command_data["meter_id"]
            fault_type = command_data.get("fault_type", "unknown")
            description = command_data.get("description", "")
            
            logger.info(f"Handling mark meter faulty command: {meter_id}")
            
            # Get existing meter
            meter = await self.smart_meter_repository.get_by_id(MeterId(meter_id))
            if not meter:
                raise MeterNotFoundException(f"Meter {meter_id} not found")
            
            # Mark as faulty
            meter.mark_as_faulty()
            
            # Save updated meter
            updated_meter = await self.smart_meter_repository.update(meter)
            
            # Publish event to Kafka
            await self.kafka_producer.send_message(
                topic="meter-commands",
                message={
                    "command_type": "mark_meter_faulty",
                    "meter_id": meter_id,
                    "timestamp": updated_meter.updated_at.isoformat(),
                    "data": {
                        "fault_type": fault_type,
                        "description": description
                    }
                }
            )
            
            # Archive command to S3
            await self._archive_command_to_s3("mark_meter_faulty", command_data)
            
            result = {
                "status": "success",
                "meter_id": meter_id,
                "message": f"Meter marked as faulty: {fault_type}",
                "fault_type": fault_type,
                "description": description
            }
            
            logger.info(f"Successfully handled mark meter faulty command: {meter_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error handling mark meter faulty command: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to mark meter as faulty: {str(e)}",
                "error_type": type(e).__name__
            }
    
    async def handle_bulk_meter_operation_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle bulk meter operation command"""
        try:
            operation = command_data["operation"]
            meter_ids = command_data["meter_ids"]
            operation_data = command_data.get("operation_data", {})
            
            logger.info(f"Handling bulk meter operation command: {operation} for {len(meter_ids)} meters")
            
            results = []
            success_count = 0
            error_count = 0
            
            for meter_id in meter_ids:
                try:
                    # Create individual command data
                    individual_command = {
                        "meter_id": meter_id,
                        **operation_data
                    }
                    
                    # Execute individual command based on operation
                    if operation == "update_status":
                        result = await self.handle_update_meter_status_command(individual_command)
                    elif operation == "update_quality":
                        result = await self.handle_update_meter_quality_command(individual_command)
                    elif operation == "mark_faulty":
                        result = await self.handle_mark_meter_faulty_command(individual_command)
                    else:
                        result = {
                            "status": "error",
                            "message": f"Unknown operation: {operation}",
                            "meter_id": meter_id
                        }
                    
                    results.append(result)
                    
                    if result["status"] == "success":
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    results.append({
                        "status": "error",
                        "meter_id": meter_id,
                        "message": str(e)
                    })
            
            # Publish bulk operation event to Kafka
            await self.kafka_producer.send_message(
                topic="meter-commands",
                message={
                    "command_type": "bulk_meter_operation",
                    "operation": operation,
                    "meter_count": len(meter_ids),
                    "success_count": success_count,
                    "error_count": error_count,
                    "timestamp": meter.updated_at.isoformat() if 'meter' in locals() else None,
                    "data": command_data
                }
            )
            
            # Archive command to S3
            await self._archive_command_to_s3("bulk_meter_operation", command_data)
            
            result = {
                "status": "success",
                "operation": operation,
                "total_meters": len(meter_ids),
                "success_count": success_count,
                "error_count": error_count,
                "results": results
            }
            
            logger.info(f"Successfully handled bulk meter operation command: {operation}")
            return result
            
        except Exception as e:
            logger.error(f"Error handling bulk meter operation command: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to execute bulk operation: {str(e)}",
                "error_type": type(e).__name__
            }
    
    async def _archive_command_to_s3(self, command_type: str, command_data: Dict[str, Any]) -> None:
        """Archive command to S3 for audit trail"""
        try:
            from datetime import datetime
            
            # Add metadata
            command_data["archived_at"] = datetime.utcnow().isoformat()
            command_data["command_type"] = command_type
            
            # Create S3 key with timestamp
            timestamp = datetime.utcnow().strftime("%Y/%m/%d/%H")
            s3_key = f"meter-commands/{command_type}/{timestamp}/{command_type}_{datetime.utcnow().timestamp()}.json"
            
            # Upload to S3
            await self.s3_client.upload_data(
                data=command_data,
                s3_key=s3_key,
                content_type="application/json"
            )
            
            logger.debug(f"Archived {command_type} command to S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"Error archiving command to S3: {str(e)}")
            # Don't raise exception as this is not critical
