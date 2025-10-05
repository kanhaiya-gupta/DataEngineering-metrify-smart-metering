"""
API Data Source Implementation
Handles extraction from REST APIs with authentication and rate limiting
"""

import logging
import requests
import time
from typing import Dict, Any, Optional, List
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.functions import lit, current_timestamp

from .data_source_factory import BaseDataSource

logger = logging.getLogger(__name__)


class APIDataSource(BaseDataSource):
    """
    REST API data source implementation
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any], schema: Optional[StructType] = None):
        super().__init__(spark, config, schema)
        self.authentication = config.get('authentication', {})
        self.rate_limiting = config.get('rate_limiting', {})
        self.session = requests.Session()
        self._setup_authentication()
        self._setup_rate_limiting()
    
    def _setup_authentication(self):
        """Setup authentication for API requests"""
        auth_type = self.authentication.get('type', 'none')
        
        if auth_type == 'bearer_token':
            token = self.authentication.get('token', '')
            self.session.headers.update({'Authorization': f'Bearer {token}'})
        elif auth_type == 'api_key':
            api_key = self.authentication.get('api_key', '')
            key_name = self.authentication.get('key_name', 'X-API-Key')
            self.session.headers.update({key_name: api_key})
        elif auth_type == 'basic_auth':
            username = self.authentication.get('username', '')
            password = self.authentication.get('password', '')
            self.session.auth = (username, password)
        elif auth_type == 'oauth2':
            # OAuth2 implementation would go here
            self.logger.warning("OAuth2 authentication not yet implemented")
        
        self.logger.info(f"API authentication setup: {auth_type}")
    
    def _setup_rate_limiting(self):
        """Setup rate limiting configuration"""
        self.requests_per_minute = self.rate_limiting.get('requests_per_minute', 1000)
        self.burst_limit = self.rate_limiting.get('burst_limit', 100)
        self.request_times = []
        
        self.logger.info(f"Rate limiting setup: {self.requests_per_minute} requests/minute, burst: {self.burst_limit}")
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(current_time)
    
    def extract(self, source_path: str) -> DataFrame:
        """
        Extract data from API endpoint
        
        Args:
            source_path: API endpoint URL or endpoint name from config
            
        Returns:
            DataFrame containing the API data
        """
        try:
            # Determine the actual endpoint URL
            if source_path.startswith('http'):
                endpoint_url = source_path
            else:
                # Look up endpoint in config
                endpoint_key = f"{source_path}_endpoint"
                endpoint_url = self.config.get(endpoint_key)
                if not endpoint_url:
                    raise ValueError(f"Endpoint not found in config: {source_path}")
            
            self.logger.info(f"Extracting API data from: {endpoint_url}")
            
            # Apply rate limiting
            self._check_rate_limit()
            
            # Make API request
            response = self.session.get(endpoint_url, timeout=30)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Convert to list if single object
            if isinstance(data, dict):
                data = [data]
            
            # Create DataFrame from JSON data
            if data:
                df = self.spark.createDataFrame(data)
                
                # Add metadata columns
                df = df.withColumn("api_endpoint", lit(endpoint_url)) \
                      .withColumn("extraction_timestamp", current_timestamp())
                
                record_count = df.count()
                self.logger.info(f"Successfully extracted {record_count} records from API")
                
                return df
            else:
                # Return empty DataFrame with schema if no data
                if self.schema:
                    df = self.spark.createDataFrame([], self.schema)
                else:
                    df = self.spark.createDataFrame([], "api_endpoint STRING, extraction_timestamp TIMESTAMP")
                
                self.logger.warning("No data returned from API endpoint")
                return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"API extraction failed: {str(e)}")
            raise
    
    def extract_paginated(self, source_path: str, page_size: int = 100) -> DataFrame:
        """
        Extract data from paginated API endpoint
        
        Args:
            source_path: API endpoint URL or endpoint name from config
            page_size: Number of records per page
            
        Returns:
            DataFrame containing all paginated data
        """
        try:
            # Determine the actual endpoint URL
            if source_path.startswith('http'):
                base_url = source_path
            else:
                endpoint_key = f"{source_path}_endpoint"
                base_url = self.config.get(endpoint_key)
                if not base_url:
                    raise ValueError(f"Endpoint not found in config: {source_path}")
            
            self.logger.info(f"Extracting paginated API data from: {base_url}")
            
            all_data = []
            page = 1
            
            while True:
                # Apply rate limiting
                self._check_rate_limit()
                
                # Build paginated URL
                paginated_url = f"{base_url}?page={page}&size={page_size}"
                
                # Make API request
                response = self.session.get(paginated_url, timeout=30)
                response.raise_for_status()
                
                # Parse JSON response
                data = response.json()
                
                # Handle different pagination formats
                if isinstance(data, dict):
                    if 'data' in data:
                        page_data = data['data']
                        has_more = data.get('has_more', False)
                    elif 'results' in data:
                        page_data = data['results']
                        has_more = data.get('next') is not None
                    else:
                        page_data = [data]
                        has_more = False
                else:
                    page_data = data
                    has_more = len(data) == page_size
                
                if not page_data:
                    break
                
                all_data.extend(page_data)
                self.logger.info(f"Fetched page {page}: {len(page_data)} records")
                
                if not has_more:
                    break
                
                page += 1
            
            # Create DataFrame from all data
            if all_data:
                df = self.spark.createDataFrame(all_data)
                df = df.withColumn("api_endpoint", lit(base_url)) \
                      .withColumn("extraction_timestamp", current_timestamp())
                
                record_count = df.count()
                self.logger.info(f"Successfully extracted {record_count} records from paginated API")
                
                return df
            else:
                # Return empty DataFrame
                if self.schema:
                    df = self.spark.createDataFrame([], self.schema)
                else:
                    df = self.spark.createDataFrame([], "api_endpoint STRING, extraction_timestamp TIMESTAMP")
                
                self.logger.warning("No data returned from paginated API endpoint")
                return df
            
        except Exception as e:
            self.logger.error(f"Paginated API extraction failed: {str(e)}")
            raise
    
    def validate_config(self) -> bool:
        """
        Validate API data source configuration
        
        Returns:
            True if configuration is valid
        """
        try:
            # Check authentication configuration
            auth_type = self.authentication.get('type', 'none')
            if auth_type != 'none':
                if auth_type == 'bearer_token' and not self.authentication.get('token'):
                    self.logger.error("Bearer token authentication requires 'token' field")
                    return False
                elif auth_type == 'api_key' and not self.authentication.get('api_key'):
                    self.logger.error("API key authentication requires 'api_key' field")
                    return False
                elif auth_type == 'basic_auth':
                    if not self.authentication.get('username') or not self.authentication.get('password'):
                        self.logger.error("Basic auth requires 'username' and 'password' fields")
                        return False
            
            # Check rate limiting configuration
            if self.requests_per_minute <= 0:
                self.logger.error("requests_per_minute must be positive")
                return False
            
            if self.burst_limit <= 0:
                self.logger.error("burst_limit must be positive")
                return False
            
            self.logger.info("API configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"API configuration validation failed: {str(e)}")
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get API-specific metadata"""
        metadata = super().get_metadata()
        metadata.update({
            'authentication_type': self.authentication.get('type', 'none'),
            'rate_limiting': {
                'requests_per_minute': self.requests_per_minute,
                'burst_limit': self.burst_limit
            }
        })
        return metadata
