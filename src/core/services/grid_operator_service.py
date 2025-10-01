"""
Grid Operator Service
Business logic for grid operator operations
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..domain.entities.grid_operator import GridOperator, GridStatus
from ..domain.value_objects.location import Location
from ..domain.enums.grid_operator_status import GridOperatorStatus
from ..interfaces.repositories.grid_operator_repository import IGridOperatorRepository
from ..interfaces.external.grid_data_service import IGridDataService
from ..interfaces.external.alerting_service import IAlertingService
from ..exceptions.domain_exceptions import GridOperatorNotFoundError, InvalidGridOperationError


@dataclass
class GridOperatorRegistrationRequest:
    """Request to register a new grid operator"""
    operator_id: str
    name: str
    operator_type: str
    headquarters: Location
    coverage_regions: List[str]
    contact_email: str
    contact_phone: Optional[str] = None
    website: Optional[str] = None
    api_endpoint: Optional[str] = None


@dataclass
class GridStatusUpdateRequest:
    """Request to update grid status"""
    operator_id: str
    timestamp: datetime
    total_capacity_mw: float
    available_capacity_mw: float
    load_factor: float
    frequency_hz: float
    voltage_kv: float
    grid_stability_score: float
    renewable_percentage: float
    region: str


class GridOperatorService:
    """
    Grid Operator Service
    
    Contains all business logic related to grid operator operations.
    This service orchestrates domain entities and coordinates with external services.
    """
    
    def __init__(
        self,
        operator_repository: IGridOperatorRepository,
        grid_data_service: IGridDataService,
        alerting_service: IAlertingService
    ):
        self.operator_repository = operator_repository
        self.grid_data_service = grid_data_service
        self.alerting_service = alerting_service
    
    async def register_operator(self, request: GridOperatorRegistrationRequest) -> GridOperator:
        """
        Register a new grid operator
        
        Args:
            request: Grid operator registration request
            
        Returns:
            Registered grid operator
            
        Raises:
            InvalidGridOperationError: If operator ID already exists
        """
        # Check if operator already exists
        existing_operator = await self.operator_repository.get_by_id(request.operator_id)
        if existing_operator:
            raise InvalidGridOperationError(f"Operator {request.operator_id} already exists")
        
        # Create new operator
        operator = GridOperator(
            operator_id=request.operator_id,
            name=request.name,
            operator_type=request.operator_type,
            headquarters=request.headquarters,
            coverage_regions=request.coverage_regions,
            contact_email=request.contact_email,
            contact_phone=request.contact_phone,
            website=request.website,
            api_endpoint=request.api_endpoint
        )
        
        # Save to repository
        await self.operator_repository.save(operator)
        
        return operator
    
    async def get_operator(self, operator_id: str) -> GridOperator:
        """
        Get a grid operator by ID
        
        Args:
            operator_id: Operator ID
            
        Returns:
            Grid operator
            
        Raises:
            GridOperatorNotFoundError: If operator doesn't exist
        """
        operator = await self.operator_repository.get_by_id(operator_id)
        if not operator:
            raise GridOperatorNotFoundError(f"Operator {operator_id} not found")
        
        return operator
    
    async def update_grid_status(self, request: GridStatusUpdateRequest) -> GridOperator:
        """
        Update grid status for an operator
        
        Args:
            request: Grid status update request
            
        Returns:
            Updated grid operator
            
        Raises:
            GridOperatorNotFoundError: If operator doesn't exist
        """
        # Get the operator
        operator = await self.get_operator(request.operator_id)
        
        # Create grid status
        status = GridStatus(
            timestamp=request.timestamp,
            total_capacity_mw=request.total_capacity_mw,
            available_capacity_mw=request.available_capacity_mw,
            load_factor=request.load_factor,
            frequency_hz=request.frequency_hz,
            voltage_kv=request.voltage_kv,
            grid_stability_score=request.grid_stability_score,
            renewable_percentage=request.renewable_percentage,
            region=request.region
        )
        
        # Update operator status
        operator.update_grid_status(status)
        
        # Save updated operator
        await self.operator_repository.save(operator)
        
        # Check for critical conditions and send alerts
        await self._check_critical_conditions(operator, status)
        
        return operator
    
    async def deactivate_operator(self, operator_id: str, reason: str) -> GridOperator:
        """
        Deactivate a grid operator
        
        Args:
            operator_id: Operator ID
            reason: Reason for deactivation
            
        Returns:
            Deactivated grid operator
        """
        operator = await self.get_operator(operator_id)
        operator.deactivate_operator(reason)
        
        await self.operator_repository.save(operator)
        return operator
    
    async def reactivate_operator(self, operator_id: str) -> GridOperator:
        """
        Reactivate a grid operator
        
        Args:
            operator_id: Operator ID
            
        Returns:
            Reactivated grid operator
        """
        operator = await self.get_operator(operator_id)
        operator.reactivate_operator()
        
        await self.operator_repository.save(operator)
        return operator
    
    async def get_operators_by_status(self, status: GridOperatorStatus) -> List[GridOperator]:
        """
        Get all operators with a specific status
        
        Args:
            status: Operator status
            
        Returns:
            List of operators with the specified status
        """
        return await self.operator_repository.get_by_status(status)
    
    async def get_operators_requiring_attention(self) -> List[GridOperator]:
        """
        Get all operators that require attention
        
        Returns:
            List of operators requiring attention
        """
        all_operators = await self.operator_repository.get_all()
        return [op for op in all_operators if op.requires_attention()]
    
    async def get_operational_operators(self) -> List[GridOperator]:
        """
        Get all operational operators
        
        Returns:
            List of operational operators
        """
        all_operators = await self.operator_repository.get_all()
        return [op for op in all_operators if op.is_operational()]
    
    async def get_operator_performance_summary(self, operator_id: str) -> Dict[str, Any]:
        """
        Get performance summary for an operator
        
        Args:
            operator_id: Operator ID
            
        Returns:
            Performance summary dictionary
        """
        operator = await self.get_operator(operator_id)
        
        return {
            "operator_id": operator.operator_id,
            "name": operator.name,
            "status": operator.status.value,
            "uptime_percentage": operator.uptime_percentage,
            "average_response_time_ms": operator.average_response_time_ms,
            "data_quality_score": operator.data_quality_score,
            "is_operational": operator.is_operational(),
            "requires_attention": operator.requires_attention(),
            "last_status_update": operator.last_status_update.isoformat() if operator.last_status_update else None,
            "current_capacity_status": operator.get_capacity_status() if operator.current_status else "Unknown",
            "current_stability_status": operator.get_stability_status() if operator.current_status else "Unknown",
            "created_at": operator.created_at.isoformat(),
            "updated_at": operator.updated_at.isoformat()
        }
    
    async def get_system_grid_summary(self) -> Dict[str, Any]:
        """
        Get overall grid system summary
        
        Returns:
            Grid system summary dictionary
        """
        all_operators = await self.operator_repository.get_all()
        
        if not all_operators:
            return {
                "total_operators": 0,
                "operational_operators": 0,
                "average_uptime": 0.0,
                "operators_requiring_attention": 0,
                "total_capacity_mw": 0.0,
                "available_capacity_mw": 0.0
            }
        
        operational_operators = [op for op in all_operators if op.is_operational()]
        operators_requiring_attention = [op for op in all_operators if op.requires_attention()]
        
        average_uptime = sum(op.uptime_percentage for op in all_operators) / len(all_operators)
        
        # Calculate total capacity
        total_capacity = sum(
            op.current_status.total_capacity_mw 
            for op in all_operators 
            if op.current_status
        )
        
        available_capacity = sum(
            op.current_status.available_capacity_mw 
            for op in all_operators 
            if op.current_status
        )
        
        return {
            "total_operators": len(all_operators),
            "operational_operators": len(operational_operators),
            "average_uptime": average_uptime,
            "operators_requiring_attention": len(operators_requiring_attention),
            "total_capacity_mw": total_capacity,
            "available_capacity_mw": available_capacity,
            "utilization_rate": (total_capacity - available_capacity) / total_capacity if total_capacity > 0 else 0.0,
            "status_distribution": {
                status.value: len([op for op in all_operators if op.status == status])
                for status in GridOperatorStatus
            }
        }
    
    async def _check_critical_conditions(self, operator: GridOperator, status: GridStatus) -> None:
        """
        Check for critical grid conditions and send alerts
        
        Args:
            operator: Grid operator
            status: Current grid status
        """
        # Check for critical capacity utilization
        utilization_rate = operator.get_utilization_rate()
        if utilization_rate > 0.9:  # 90% utilization threshold
            await self.alerting_service.send_alert(
                alert_type="grid_capacity_critical",
                operator_id=operator.operator_id,
                message=f"Grid capacity utilization is {utilization_rate:.1%}",
                severity="high",
                data={
                    "utilization_rate": utilization_rate,
                    "available_capacity_mw": status.available_capacity_mw,
                    "total_capacity_mw": status.total_capacity_mw
                }
            )
        
        # Check for frequency anomalies
        if not (49.8 <= status.frequency_hz <= 50.2):
            await self.alerting_service.send_alert(
                alert_type="grid_frequency_anomaly",
                operator_id=operator.operator_id,
                message=f"Grid frequency is {status.frequency_hz}Hz (outside normal range)",
                severity="critical",
                data={
                    "frequency_hz": status.frequency_hz,
                    "normal_range": "49.8-50.2Hz"
                }
            )
        
        # Check for voltage anomalies
        if not (380 <= status.voltage_kv <= 420):
            await self.alerting_service.send_alert(
                alert_type="grid_voltage_anomaly",
                operator_id=operator.operator_id,
                message=f"Grid voltage is {status.voltage_kv}kV (outside normal range)",
                severity="critical",
                data={
                    "voltage_kv": status.voltage_kv,
                    "normal_range": "380-420kV"
                }
            )
        
        # Check for low grid stability
        if status.grid_stability_score < 0.7:
            await self.alerting_service.send_alert(
                alert_type="grid_stability_low",
                operator_id=operator.operator_id,
                message=f"Grid stability score is {status.grid_stability_score:.2f} (below threshold)",
                severity="high",
                data={
                    "stability_score": status.grid_stability_score,
                    "threshold": 0.7
                }
            )
