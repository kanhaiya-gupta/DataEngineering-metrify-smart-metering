"""
Grid Data Service Interface
Abstract interface for grid data operations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...domain.entities.grid_operator import GridStatus


class IGridDataService(ABC):
    """
    Abstract interface for grid data service
    
    Defines the contract for grid data operations.
    This interface allows for different implementations (API-based, file-based, etc.)
    while maintaining the same interface.
    """
    
    @abstractmethod
    async def fetch_grid_status(self, operator_id: str) -> Optional[GridStatus]:
        """
        Fetch current grid status from an operator
        
        Args:
            operator_id: Grid operator ID
            
        Returns:
            Current grid status if available, None otherwise
        """
        pass
    
    @abstractmethod
    async def fetch_historical_grid_data(self, operator_id: str, start_date: datetime, end_date: datetime) -> List[GridStatus]:
        """
        Fetch historical grid data for an operator
        
        Args:
            operator_id: Grid operator ID
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            List of historical grid statuses
        """
        pass
    
    @abstractmethod
    async def get_available_operators(self) -> List[Dict[str, Any]]:
        """
        Get list of available grid operators
        
        Returns:
            List of available operators with their metadata
        """
        pass
    
    @abstractmethod
    async def validate_operator_connection(self, operator_id: str) -> bool:
        """
        Validate connection to a grid operator
        
        Args:
            operator_id: Grid operator ID
            
        Returns:
            True if connection is valid, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_operator_capabilities(self, operator_id: str) -> Dict[str, Any]:
        """
        Get capabilities of a grid operator
        
        Args:
            operator_id: Grid operator ID
            
        Returns:
            Dictionary of operator capabilities
        """
        pass
    
    @abstractmethod
    async def subscribe_to_real_time_updates(self, operator_id: str, callback) -> str:
        """
        Subscribe to real-time grid status updates
        
        Args:
            operator_id: Grid operator ID
            callback: Callback function to handle updates
            
        Returns:
            Subscription ID for unsubscribing
        """
        pass
    
    @abstractmethod
    async def unsubscribe_from_updates(self, subscription_id: str) -> None:
        """
        Unsubscribe from real-time updates
        
        Args:
            subscription_id: Subscription ID to cancel
        """
        pass
    
    @abstractmethod
    async def get_data_quality_metrics(self, operator_id: str, since: datetime) -> Dict[str, Any]:
        """
        Get data quality metrics for an operator
        
        Args:
            operator_id: Grid operator ID
            since: Start datetime for metrics
            
        Returns:
            Data quality metrics dictionary
        """
        pass
    
    @abstractmethod
    async def get_operator_performance(self, operator_id: str, days: int) -> Dict[str, Any]:
        """
        Get performance metrics for an operator
        
        Args:
            operator_id: Grid operator ID
            days: Number of days to analyze
            
        Returns:
            Performance metrics dictionary
        """
        pass
