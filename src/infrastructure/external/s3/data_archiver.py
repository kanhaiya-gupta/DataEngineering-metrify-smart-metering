"""
S3 Data Archiver
Handles data archiving and lifecycle management
"""

import json
import gzip
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .s3_client import S3Client
from ....core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class S3DataArchiver:
    """
    S3 Data Archiver
    
    Handles data archiving, compression, and lifecycle management
    for the data lake storage.
    """
    
    def __init__(self, s3_client: S3Client, archive_bucket: Optional[str] = None):
        self.s3_client = s3_client
        self.archive_bucket = archive_bucket or s3_client.bucket_name
        self._archive_policies = {}
        self._initialize_archive_policies()
    
    def _initialize_archive_policies(self) -> None:
        """Initialize data archiving policies"""
        self._archive_policies = {
            "meter_readings": {
                "retention_days": 365,  # 1 year
                "archive_after_days": 30,  # Archive after 30 days
                "compress": True,
                "partition_by": "year/month/day"
            },
            "grid_statuses": {
                "retention_days": 365,
                "archive_after_days": 30,
                "compress": True,
                "partition_by": "year/month/day"
            },
            "weather_observations": {
                "retention_days": 365,
                "archive_after_days": 30,
                "compress": True,
                "partition_by": "year/month/day"
            },
            "events": {
                "retention_days": 2555,  # 7 years for audit
                "archive_after_days": 90,
                "compress": True,
                "partition_by": "year/month"
            }
        }
    
    async def archive_data(
        self,
        data_type: str,
        data: List[Dict[str, Any]],
        date: datetime,
        entity_id: Optional[str] = None
    ) -> str:
        """
        Archive data to S3 with appropriate partitioning and compression
        
        Args:
            data_type: Type of data being archived
            data: Data to archive
            date: Date for partitioning
            entity_id: Optional entity ID for partitioning
            
        Returns:
            S3 path of archived data
        """
        try:
            if data_type not in self._archive_policies:
                raise ValueError(f"Unknown data type: {data_type}")
            
            policy = self._archive_policies[data_type]
            
            # Create S3 key with partitioning
            s3_key = self._create_archive_key(data_type, date, entity_id)
            
            # Prepare data for archiving
            archive_data = {
                "data_type": data_type,
                "archive_date": datetime.utcnow().isoformat(),
                "data_date": date.isoformat(),
                "entity_id": entity_id,
                "record_count": len(data),
                "records": data
            }
            
            # Serialize data
            json_data = json.dumps(archive_data, indent=2, default=str).encode('utf-8')
            
            # Compress if policy requires
            if policy["compress"]:
                compressed_data = gzip.compress(json_data)
                content_encoding = "gzip"
            else:
                compressed_data = json_data
                content_encoding = None
            
            # Upload to S3
            metadata = {
                "data_type": data_type,
                "archive_date": datetime.utcnow().isoformat(),
                "data_date": date.isoformat(),
                "record_count": str(len(data)),
                "compressed": str(policy["compress"])
            }
            
            if entity_id:
                metadata["entity_id"] = entity_id
            
            s3_url = await self.s3_client.upload_data(
                data=compressed_data,
                s3_key=s3_key,
                content_type="application/json",
                metadata=metadata,
                compress=False  # Already compressed
            )
            
            logger.info(f"Archived {len(data)} {data_type} records to {s3_key}")
            return s3_url
            
        except Exception as e:
            logger.error(f"Error archiving data: {str(e)}")
            raise InfrastructureError(f"Failed to archive data: {str(e)}", service="s3")
    
    async def restore_data(
        self,
        s3_key: str,
        decompress: bool = True
    ) -> Dict[str, Any]:
        """
        Restore archived data from S3
        
        Args:
            s3_key: S3 key of archived data
            decompress: Whether to decompress the data
            
        Returns:
            Restored data dictionary
        """
        try:
            # Download data from S3
            data = await self.s3_client.get_object(s3_key, decompress=decompress)
            
            # Parse JSON data
            archive_data = json.loads(data.decode('utf-8'))
            
            logger.info(f"Restored {archive_data.get('record_count', 0)} records from {s3_key}")
            return archive_data
            
        except Exception as e:
            logger.error(f"Error restoring data: {str(e)}")
            raise InfrastructureError(f"Failed to restore data: {str(e)}", service="s3")
    
    async def list_archived_data(
        self,
        data_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        entity_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List archived data matching criteria
        
        Args:
            data_type: Type of data to list
            start_date: Optional start date filter
            end_date: Optional end date filter
            entity_id: Optional entity ID filter
            
        Returns:
            List of archived data metadata
        """
        try:
            # Build prefix for filtering
            prefix = f"archive/{data_type}/"
            if entity_id:
                prefix += f"{entity_id}/"
            
            # List objects with prefix
            objects = await self.s3_client.list_objects(prefix=prefix)
            
            # Filter by date if specified
            filtered_objects = []
            for obj in objects:
                # Extract date from key
                key_parts = obj['key'].split('/')
                if len(key_parts) >= 4:  # archive/data_type/year/month/day/...
                    try:
                        year = int(key_parts[-4])
                        month = int(key_parts[-3])
                        day = int(key_parts[-2])
                        obj_date = datetime(year, month, day)
                        
                        # Apply date filters
                        if start_date and obj_date < start_date:
                            continue
                        if end_date and obj_date > end_date:
                            continue
                        
                        filtered_objects.append(obj)
                    except (ValueError, IndexError):
                        # Skip objects with invalid date format
                        continue
            
            return filtered_objects
            
        except Exception as e:
            logger.error(f"Error listing archived data: {str(e)}")
            raise InfrastructureError(f"Failed to list archived data: {str(e)}", service="s3")
    
    async def cleanup_expired_data(self, data_type: str) -> int:
        """
        Clean up expired data based on retention policy
        
        Args:
            data_type: Type of data to clean up
            
        Returns:
            Number of objects deleted
        """
        try:
            if data_type not in self._archive_policies:
                raise ValueError(f"Unknown data type: {data_type}")
            
            policy = self._archive_policies[data_type]
            retention_days = policy["retention_days"]
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            # List all archived data for this type
            objects = await self.list_archived_data(data_type)
            
            # Find expired objects
            expired_objects = []
            for obj in objects:
                # Extract date from key
                key_parts = obj['key'].split('/')
                if len(key_parts) >= 4:
                    try:
                        year = int(key_parts[-4])
                        month = int(key_parts[-3])
                        day = int(key_parts[-2])
                        obj_date = datetime(year, month, day)
                        
                        if obj_date < cutoff_date:
                            expired_objects.append(obj['key'])
                    except (ValueError, IndexError):
                        # Skip objects with invalid date format
                        continue
            
            # Delete expired objects
            deleted_count = 0
            for key in expired_objects:
                try:
                    await self.s3_client.delete_object(key)
                    deleted_count += 1
                    logger.debug(f"Deleted expired object: {key}")
                except Exception as e:
                    logger.warning(f"Failed to delete object {key}: {str(e)}")
            
            logger.info(f"Cleaned up {deleted_count} expired {data_type} objects")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired data: {str(e)}")
            raise InfrastructureError(f"Failed to cleanup expired data: {str(e)}", service="s3")
    
    async def get_archive_statistics(self, data_type: str) -> Dict[str, Any]:
        """
        Get archive statistics for a data type
        
        Args:
            data_type: Type of data to get statistics for
            
        Returns:
            Archive statistics dictionary
        """
        try:
            objects = await self.list_archived_data(data_type)
            
            if not objects:
                return {
                    "data_type": data_type,
                    "total_objects": 0,
                    "total_size_bytes": 0,
                    "date_range": None
                }
            
            total_size = sum(obj['size'] for obj in objects)
            dates = []
            
            for obj in objects:
                key_parts = obj['key'].split('/')
                if len(key_parts) >= 4:
                    try:
                        year = int(key_parts[-4])
                        month = int(key_parts[-3])
                        day = int(key_parts[-2])
                        dates.append(datetime(year, month, day))
                    except (ValueError, IndexError):
                        continue
            
            return {
                "data_type": data_type,
                "total_objects": len(objects),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "date_range": {
                    "earliest": min(dates).isoformat() if dates else None,
                    "latest": max(dates).isoformat() if dates else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting archive statistics: {str(e)}")
            raise InfrastructureError(f"Failed to get archive statistics: {str(e)}", service="s3")
    
    def _create_archive_key(
        self,
        data_type: str,
        date: datetime,
        entity_id: Optional[str] = None
    ) -> str:
        """Create S3 key for archived data"""
        year = date.year
        month = date.month
        day = date.day
        
        key_parts = ["archive", data_type, str(year), str(month).zfill(2), str(day).zfill(2)]
        
        if entity_id:
            key_parts.append(entity_id)
        
        # Add timestamp for uniqueness
        timestamp = date.strftime("%Y%m%d_%H%M%S")
        key_parts.append(f"{data_type}_{timestamp}.json.gz")
        
        return "/".join(key_parts)
    
    def update_archive_policy(self, data_type: str, policy: Dict[str, Any]) -> None:
        """
        Update archive policy for a data type
        
        Args:
            data_type: Type of data
            policy: New archive policy
        """
        self._archive_policies[data_type] = policy
        logger.info(f"Updated archive policy for {data_type}")
    
    def get_archive_policies(self) -> Dict[str, Dict[str, Any]]:
        """Get all archive policies"""
        return self._archive_policies.copy()
