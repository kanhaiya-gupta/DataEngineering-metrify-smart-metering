"""
S3 Client Implementation
Handles data storage and retrieval from AWS S3
"""

import json
import gzip
from typing import Dict, Any, Optional, List, BinaryIO
from datetime import datetime, timedelta
from io import BytesIO
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import logging

from ....core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class S3Client:
    """
    S3 Client for data storage and retrieval
    
    Handles file uploads, downloads, and data lake operations
    """
    
    def __init__(
        self,
        bucket_name: str,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ):
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.endpoint_url = endpoint_url
        
        self._s3_client = None
        self._is_connected = False
    
    async def connect(self) -> None:
        """Connect to S3 service"""
        try:
            session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name
            )
            
            self._s3_client = session.client(
                's3',
                endpoint_url=self.endpoint_url
            )
            
            # Test connection by checking if bucket exists
            self._s3_client.head_bucket(Bucket=self.bucket_name)
            self._is_connected = True
            logger.info(f"Connected to S3 bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise InfrastructureError("AWS credentials not found", service="s3")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket {self.bucket_name} not found")
                raise InfrastructureError(f"S3 bucket {self.bucket_name} not found", service="s3")
            else:
                logger.error(f"Failed to connect to S3: {str(e)}")
                raise InfrastructureError(f"Failed to connect to S3: {str(e)}", service="s3")
        except Exception as e:
            logger.error(f"Unexpected error connecting to S3: {str(e)}")
            raise InfrastructureError(f"Unexpected error: {str(e)}", service="s3")
    
    async def upload_file(
        self,
        file_path: str,
        s3_key: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
        compress: bool = False
    ) -> str:
        """
        Upload a file to S3
        
        Args:
            file_path: Local file path
            s3_key: S3 object key
            content_type: MIME type of the file
            metadata: Optional metadata to attach
            compress: Whether to compress the file with gzip
            
        Returns:
            S3 object URL
        """
        if not self._is_connected or not self._s3_client:
            await self.connect()
        
        try:
            extra_args = {
                'ContentType': content_type
            }
            
            if metadata:
                extra_args['Metadata'] = metadata
            
            if compress:
                # Compress file before upload
                with open(file_path, 'rb') as f_in:
                    with gzip.open(f_in, 'rb') as f_out:
                        compressed_data = f_out.read()
                
                self._s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=compressed_data,
                    **extra_args
                )
            else:
                self._s3_client.upload_file(
                    file_path,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs=extra_args
                )
            
            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"Uploaded file to S3: {s3_url}")
            return s3_url
            
        except ClientError as e:
            logger.error(f"Failed to upload file to S3: {str(e)}")
            raise InfrastructureError(f"Failed to upload file: {str(e)}", service="s3")
        except Exception as e:
            logger.error(f"Unexpected error uploading file: {str(e)}")
            raise InfrastructureError(f"Unexpected error: {str(e)}", service="s3")
    
    async def upload_data(
        self,
        data: bytes,
        s3_key: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
        compress: bool = False
    ) -> str:
        """
        Upload data bytes to S3
        
        Args:
            data: Data bytes to upload
            s3_key: S3 object key
            content_type: MIME type of the data
            metadata: Optional metadata to attach
            compress: Whether to compress the data with gzip
            
        Returns:
            S3 object URL
        """
        if not self._is_connected or not self._s3_client:
            await self.connect()
        
        try:
            extra_args = {
                'ContentType': content_type
            }
            
            if metadata:
                extra_args['Metadata'] = metadata
            
            if compress:
                # Compress data before upload
                compressed_data = gzip.compress(data)
                extra_args['ContentEncoding'] = 'gzip'
                upload_data = compressed_data
            else:
                upload_data = data
            
            self._s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=upload_data,
                **extra_args
            )
            
            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"Uploaded data to S3: {s3_url}")
            return s3_url
            
        except ClientError as e:
            logger.error(f"Failed to upload data to S3: {str(e)}")
            raise InfrastructureError(f"Failed to upload data: {str(e)}", service="s3")
        except Exception as e:
            logger.error(f"Unexpected error uploading data: {str(e)}")
            raise InfrastructureError(f"Unexpected error: {str(e)}", service="s3")
    
    async def download_file(
        self,
        s3_key: str,
        local_path: str,
        decompress: bool = False
    ) -> None:
        """
        Download a file from S3
        
        Args:
            s3_key: S3 object key
            local_path: Local file path to save to
            decompress: Whether to decompress gzipped content
        """
        if not self._is_connected or not self._s3_client:
            await self.connect()
        
        try:
            response = self._s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            data = response['Body'].read()
            
            if decompress or response.get('ContentEncoding') == 'gzip':
                data = gzip.decompress(data)
            
            with open(local_path, 'wb') as f:
                f.write(data)
            
            logger.info(f"Downloaded file from S3: {s3_key} -> {local_path}")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"S3 object not found: {s3_key}")
                raise InfrastructureError(f"S3 object not found: {s3_key}", service="s3")
            else:
                logger.error(f"Failed to download file from S3: {str(e)}")
                raise InfrastructureError(f"Failed to download file: {str(e)}", service="s3")
        except Exception as e:
            logger.error(f"Unexpected error downloading file: {str(e)}")
            raise InfrastructureError(f"Unexpected error: {str(e)}", service="s3")
    
    async def get_object(
        self,
        s3_key: str,
        decompress: bool = False
    ) -> bytes:
        """
        Get object data from S3
        
        Args:
            s3_key: S3 object key
            decompress: Whether to decompress gzipped content
            
        Returns:
            Object data as bytes
        """
        if not self._is_connected or not self._s3_client:
            await self.connect()
        
        try:
            response = self._s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            data = response['Body'].read()
            
            if decompress or response.get('ContentEncoding') == 'gzip':
                data = gzip.decompress(data)
            
            logger.info(f"Retrieved object from S3: {s3_key}")
            return data
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"S3 object not found: {s3_key}")
                raise InfrastructureError(f"S3 object not found: {s3_key}", service="s3")
            else:
                logger.error(f"Failed to get object from S3: {str(e)}")
                raise InfrastructureError(f"Failed to get object: {str(e)}", service="s3")
        except Exception as e:
            logger.error(f"Unexpected error getting object: {str(e)}")
            raise InfrastructureError(f"Unexpected error: {str(e)}", service="s3")
    
    async def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List objects in S3 bucket
        
        Args:
            prefix: Object key prefix to filter by
            max_keys: Maximum number of objects to return
            
        Returns:
            List of object metadata
        """
        if not self._is_connected or not self._s3_client:
            await self.connect()
        
        try:
            response = self._s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag']
                })
            
            logger.info(f"Listed {len(objects)} objects with prefix: {prefix}")
            return objects
            
        except ClientError as e:
            logger.error(f"Failed to list objects in S3: {str(e)}")
            raise InfrastructureError(f"Failed to list objects: {str(e)}", service="s3")
        except Exception as e:
            logger.error(f"Unexpected error listing objects: {str(e)}")
            raise InfrastructureError(f"Unexpected error: {str(e)}", service="s3")
    
    async def delete_object(self, s3_key: str) -> None:
        """
        Delete an object from S3
        
        Args:
            s3_key: S3 object key to delete
        """
        if not self._is_connected or not self._s3_client:
            await self.connect()
        
        try:
            self._s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            logger.info(f"Deleted object from S3: {s3_key}")
            
        except ClientError as e:
            logger.error(f"Failed to delete object from S3: {str(e)}")
            raise InfrastructureError(f"Failed to delete object: {str(e)}", service="s3")
        except Exception as e:
            logger.error(f"Unexpected error deleting object: {str(e)}")
            raise InfrastructureError(f"Unexpected error: {str(e)}", service="s3")
    
    async def store_meter_readings(
        self,
        meter_id: str,
        readings: List[Dict[str, Any]],
        date: datetime
    ) -> str:
        """Store smart meter readings in S3 data lake"""
        s3_key = f"raw/smart_meters/{meter_id}/{date.strftime('%Y/%m/%d')}/readings_{date.strftime('%Y%m%d_%H%M%S')}.json"
        
        data = json.dumps(readings, indent=2).encode('utf-8')
        metadata = {
            'meter_id': meter_id,
            'date': date.isoformat(),
            'record_count': str(len(readings))
        }
        
        return await self.upload_data(
            data=data,
            s3_key=s3_key,
            content_type="application/json",
            metadata=metadata,
            compress=True
        )
    
    async def store_grid_status(
        self,
        operator_id: str,
        status_data: Dict[str, Any],
        date: datetime
    ) -> str:
        """Store grid status data in S3 data lake"""
        s3_key = f"raw/grid_operators/{operator_id}/{date.strftime('%Y/%m/%d')}/status_{date.strftime('%Y%m%d_%H%M%S')}.json"
        
        data = json.dumps(status_data, indent=2).encode('utf-8')
        metadata = {
            'operator_id': operator_id,
            'date': date.isoformat()
        }
        
        return await self.upload_data(
            data=data,
            s3_key=s3_key,
            content_type="application/json",
            metadata=metadata,
            compress=True
        )
    
    async def store_weather_observations(
        self,
        station_id: str,
        observations: List[Dict[str, Any]],
        date: datetime
    ) -> str:
        """Store weather observations in S3 data lake"""
        s3_key = f"raw/weather_stations/{station_id}/{date.strftime('%Y/%m/%d')}/observations_{date.strftime('%Y%m%d_%H%M%S')}.json"
        
        data = json.dumps(observations, indent=2).encode('utf-8')
        metadata = {
            'station_id': station_id,
            'date': date.isoformat(),
            'record_count': str(len(observations))
        }
        
        return await self.upload_data(
            data=data,
            s3_key=s3_key,
            content_type="application/json",
            metadata=metadata,
            compress=True
        )
    
    def is_connected(self) -> bool:
        """Check if client is connected to S3"""
        return self._is_connected and self._s3_client is not None
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get S3 client metrics"""
        if not self._s3_client:
            return {"connected": False}
        
        try:
            # Get bucket metrics
            response = self._s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                MaxKeys=1
            )
            
            return {
                "connected": self._is_connected,
                "bucket_name": self.bucket_name,
                "region": self.region_name,
                "object_count": response.get('KeyCount', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return {"connected": self._is_connected, "error": str(e)}
