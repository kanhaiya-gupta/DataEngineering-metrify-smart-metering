"""
Grid Operator Repository Interface
Abstract interface for grid operator data access
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

from ...domain.entities.grid_operator import GridOperator
from ...domain.enums.grid_operator_status import GridOperatorStatus


class IGridOperatorRepository(ABC):
    """
    Abstract interface for grid operator repository
    
    Defines the contract for grid operator data access operations.
    This interface allows for different implementations (database, cache, etc.)
    while maintaining the same interface.
    """
    
    @abstractmethod
    async def get_by_id(self, operator_id: str) -> Optional[GridOperator]:
        """
        Get a grid operator by ID
        
        Args:
            operator_id: Operator ID
            
        Returns:
            Grid operator if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def save(self, operator: GridOperator) -> None:
        """
        Save a grid operator
        
        Args:
            operator: Grid operator to save
        """
        pass
    
    @abstractmethod
    async def delete(self, operator_id: str) -> None:
        """
        Delete a grid operator
        
        Args:
            operator_id: Operator ID to delete
        """
        pass
    
    @abstractmethod
    async def get_all(self) -> List[GridOperator]:
        """
        Get all grid operators
        
        Returns:
            List of all grid operators
        """
        pass
    
    @abstractmethod
    async def get_by_status(self, status: GridOperatorStatus) -> List[GridOperator]:
        """
        Get grid operators by status
        
        Args:
            status: Operator status
            
        Returns:
            List of grid operators with the specified status
        """
        pass
    
    @abstractmethod
    async def get_by_region(self, region: str) -> List[GridOperator]:
        """
        Get grid operators by region
        
        Args:
            region: Region name
            
        Returns:
            List of grid operators in the specified region
        """
        pass
    
    @abstractmethod
    async def get_operational(self) -> List[GridOperator]:
        """
        Get operational grid operators
        
        Returns:
            List of operational grid operators
        """
        pass
    
    @abstractmethod
    async def get_requiring_attention(self) -> List[GridOperator]:
        """
        Get grid operators that require attention
        
        Returns:
            List of grid operators requiring attention
        """
        pass
    
    @abstractmethod
    async def get_by_uptime_threshold(self, min_uptime: float) -> List[GridOperator]:
        """
        Get grid operators with uptime above threshold
        
        Args:
            min_uptime: Minimum uptime percentage
            
        Returns:
            List of grid operators with uptime above threshold
        """
        pass
    
    @abstractmethod
    async def get_by_data_quality_threshold(self, min_quality: float) -> List[GridOperator]:
        """
        Get grid operators with data quality above threshold
        
        Args:
            min_quality: Minimum data quality score
            
        Returns:
            List of grid operators with data quality above threshold
        """
        pass
    
    @abstractmethod
    async def get_with_recent_updates(self, since: datetime) -> List[GridOperator]:
        """
        Get grid operators with recent status updates
        
        Args:
            since: Datetime to filter updates since
            
        Returns:
            List of grid operators with recent updates
        """
        pass
    
    @abstractmethod
    async def count_by_status(self, status: GridOperatorStatus) -> int:
        """
        Count grid operators by status
        
        Args:
            status: Operator status
            
        Returns:
            Number of grid operators with the specified status
        """
        pass
    
    @abstractmethod
    async def get_average_uptime(self) -> float:
        """
        Get average uptime across all grid operators
        
        Returns:
            Average uptime percentage
        """
        pass
    
    @abstractmethod
    async def get_average_data_quality(self) -> float:
        """
        Get average data quality score across all grid operators
        
        Returns:
            Average data quality score
        """
        pass
    
    @abstractmethod
    async def get_performance_statistics(self) -> dict:
        """
        Get performance statistics for all grid operators
        
        Returns:
            Dictionary with performance statistics
        """
        pass
