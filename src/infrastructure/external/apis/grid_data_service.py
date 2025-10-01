"""
Grid Data Service Implementation
Concrete implementation of IGridDataService using external APIs
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from ....core.domain.entities.grid_operator import GridStatus
from ....core.interfaces.external.grid_data_service import IGridDataService
from ....core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class GridDataService(IGridDataService):
    """
    Grid Data Service Implementation
    
    Provides grid data operations by integrating with external
    grid operator APIs and data sources.
    """
    
    def __init__(
        self,
        api_base_url: str = "https://api.grid-operators.com",
        api_key: str = "",
        timeout: int = 30
    ):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._operators_cache = {}
        self._last_cache_update = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'Metrify-SmartMetering/1.0'
            }
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def fetch_grid_status(self, operator_id: str) -> Optional[GridStatus]:
        """
        Fetch current grid status from an operator
        
        Args:
            operator_id: Grid operator ID
            
        Returns:
            Current grid status if available, None otherwise
        """
        try:
            session = await self._get_session()
            url = f"{self.api_base_url}/operators/{operator_id}/status"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_grid_status(data)
                elif response.status == 404:
                    logger.warning(f"Grid operator {operator_id} not found")
                    return None
                else:
                    logger.error(f"Failed to fetch grid status: {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching grid status for operator {operator_id}")
            raise InfrastructureError(f"Timeout fetching grid status", service="grid_api")
        except Exception as e:
            logger.error(f"Error fetching grid status: {str(e)}")
            raise InfrastructureError(f"Failed to fetch grid status: {str(e)}", service="grid_api")
    
    async def fetch_historical_grid_data(
        self, 
        operator_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[GridStatus]:
        """
        Fetch historical grid data for an operator
        
        Args:
            operator_id: Grid operator ID
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            List of historical grid statuses
        """
        try:
            session = await self._get_session()
            url = f"{self.api_base_url}/operators/{operator_id}/history"
            
            params = {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'limit': 1000  # Adjust based on API limits
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [self._parse_grid_status(item) for item in data.get('statuses', [])]
                else:
                    logger.error(f"Failed to fetch historical data: {response.status}")
                    return []
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching historical data for operator {operator_id}")
            raise InfrastructureError(f"Timeout fetching historical data", service="grid_api")
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise InfrastructureError(f"Failed to fetch historical data: {str(e)}", service="grid_api")
    
    async def get_available_operators(self) -> List[Dict[str, Any]]:
        """Get list of available grid operators"""
        try:
            # Check cache first
            if (self._last_cache_update and 
                datetime.utcnow() - self._last_cache_update < timedelta(minutes=30)):
                return list(self._operators_cache.values())
            
            session = await self._get_session()
            url = f"{self.api_base_url}/operators"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    operators = data.get('operators', [])
                    
                    # Update cache
                    self._operators_cache = {
                        op['id']: op for op in operators
                    }
                    self._last_cache_update = datetime.utcnow()
                    
                    return operators
                else:
                    logger.error(f"Failed to fetch operators: {response.status}")
                    return []
                    
        except asyncio.TimeoutError:
            logger.error("Timeout fetching available operators")
            raise InfrastructureError(f"Timeout fetching operators", service="grid_api")
        except Exception as e:
            logger.error(f"Error fetching operators: {str(e)}")
            raise InfrastructureError(f"Failed to fetch operators: {str(e)}", service="grid_api")
    
    async def validate_operator_connection(self, operator_id: str) -> bool:
        """
        Validate connection to a grid operator
        
        Args:
            operator_id: Grid operator ID
            
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            session = await self._get_session()
            url = f"{self.api_base_url}/operators/{operator_id}/health"
            
            async with session.get(url) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Error validating operator connection: {str(e)}")
            return False
    
    async def get_operator_capabilities(self, operator_id: str) -> Dict[str, Any]:
        """
        Get capabilities of a grid operator
        
        Args:
            operator_id: Grid operator ID
            
        Returns:
            Dictionary of operator capabilities
        """
        try:
            session = await self._get_session()
            url = f"{self.api_base_url}/operators/{operator_id}/capabilities"
            
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch operator capabilities: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching operator capabilities: {str(e)}")
            raise InfrastructureError(f"Failed to fetch operator capabilities: {str(e)}", service="grid_api")
    
    async def subscribe_to_real_time_updates(self, operator_id: str, callback) -> str:
        """
        Subscribe to real-time grid status updates
        
        Args:
            operator_id: Grid operator ID
            callback: Callback function to handle updates
            
        Returns:
            Subscription ID for unsubscribing
        """
        try:
            # This would typically establish a WebSocket connection
            # For now, return a mock subscription ID
            subscription_id = f"sub_{operator_id}_{datetime.utcnow().timestamp()}"
            
            # In a real implementation, this would start a background task
            # to listen for WebSocket messages and call the callback
            logger.info(f"Subscribed to real-time updates for operator {operator_id}")
            
            return subscription_id
            
        except Exception as e:
            logger.error(f"Error subscribing to real-time updates: {str(e)}")
            raise InfrastructureError(f"Failed to subscribe to updates: {str(e)}", service="grid_api")
    
    async def unsubscribe_from_updates(self, subscription_id: str) -> None:
        """
        Unsubscribe from real-time updates
        
        Args:
            subscription_id: Subscription ID to cancel
        """
        try:
            # This would typically close the WebSocket connection
            logger.info(f"Unsubscribed from updates: {subscription_id}")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from updates: {str(e)}")
            raise InfrastructureError(f"Failed to unsubscribe: {str(e)}", service="grid_api")
    
    async def get_data_quality_metrics(self, operator_id: str, since: datetime) -> Dict[str, Any]:
        """
        Get data quality metrics for an operator
        
        Args:
            operator_id: Grid operator ID
            since: Start datetime for metrics
            
        Returns:
            Data quality metrics dictionary
        """
        try:
            session = await self._get_session()
            url = f"{self.api_base_url}/operators/{operator_id}/quality"
            
            params = {
                'since': since.isoformat()
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch quality metrics: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching quality metrics: {str(e)}")
            raise InfrastructureError(f"Failed to fetch quality metrics: {str(e)}", service="grid_api")
    
    async def get_operator_performance(self, operator_id: str, days: int) -> Dict[str, Any]:
        """
        Get performance metrics for an operator
        
        Args:
            operator_id: Grid operator ID
            days: Number of days to analyze
            
        Returns:
            Performance metrics dictionary
        """
        try:
            session = await self._get_session()
            url = f"{self.api_base_url}/operators/{operator_id}/performance"
            
            params = {
                'days': days
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch performance metrics: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching performance metrics: {str(e)}")
            raise InfrastructureError(f"Failed to fetch performance metrics: {str(e)}", service="grid_api")
    
    def _parse_grid_status(self, data: Dict[str, Any]) -> GridStatus:
        """Parse API response into GridStatus object"""
        try:
            return GridStatus(
                operator_id=data.get('operator_id', ''),
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat())),
                voltage_level=data.get('voltage_level', 0.0),
                frequency=data.get('frequency', 0.0),
                load_percentage=data.get('load_percentage', 0.0),
                stability_score=data.get('stability_score', 0.0),
                power_quality_score=data.get('power_quality_score', 0.0),
                total_generation_mw=data.get('total_generation_mw'),
                total_consumption_mw=data.get('total_consumption_mw'),
                grid_frequency_hz=data.get('grid_frequency_hz'),
                voltage_deviation_percent=data.get('voltage_deviation_percent'),
                data_quality_score=data.get('data_quality_score', 1.0),
                is_anomaly=data.get('is_anomaly', False),
                anomaly_type=data.get('anomaly_type')
            )
        except Exception as e:
            logger.error(f"Error parsing grid status: {str(e)}")
            # Return a default status
            return GridStatus(
                operator_id=data.get('operator_id', ''),
                timestamp=datetime.utcnow(),
                voltage_level=0.0,
                frequency=0.0,
                load_percentage=0.0,
                stability_score=0.0,
                power_quality_score=0.0
            )
