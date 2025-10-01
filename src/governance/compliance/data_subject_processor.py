"""
Data Subject Processor
Handles data subject rights automation (GDPR Articles 15-22)
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid
import json

logger = logging.getLogger(__name__)

class DataSubjectRight(Enum):
    """Data subject rights under GDPR"""
    ACCESS = "access"  # Article 15
    RECTIFICATION = "rectification"  # Article 16
    ERASURE = "erasure"  # Article 17
    RESTRICTION = "restriction"  # Article 18
    PORTABILITY = "portability"  # Article 20
    OBJECTION = "objection"  # Article 21

class RequestStatus(Enum):
    """Request status values"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class DataSubjectRequest:
    """Represents a data subject rights request"""
    request_id: str
    subject_id: str
    right_type: DataSubjectRight
    status: RequestStatus
    submitted_at: datetime
    due_date: datetime
    description: str
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]] = None
    completed_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    assigned_to: Optional[str] = None

class DataSubjectProcessor:
    """
    Handles data subject rights automation
    """
    
    def __init__(self, response_days: int = 30):
        self.response_days = response_days
        self.requests = {}
        self.request_templates = {}
        
        logger.info(f"DataSubjectProcessor initialized with {response_days} days response time")
    
    def submit_request(self,
                      subject_id: str,
                      right_type: DataSubjectRight,
                      description: str,
                      request_data: Dict[str, Any] = None) -> str:
        """Submit a data subject rights request"""
        try:
            request_id = str(uuid.uuid4())
            due_date = datetime.now() + timedelta(days=self.response_days)
            
            request = DataSubjectRequest(
                request_id=request_id,
                subject_id=subject_id,
                right_type=right_type,
                status=RequestStatus.PENDING,
                submitted_at=datetime.now(),
                due_date=due_date,
                description=description,
                request_data=request_data or {}
            )
            
            self.requests[request_id] = request
            
            logger.info(f"Data subject request submitted: {request_id} for {right_type.value}")
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to submit data subject request: {str(e)}")
            return ""
    
    def process_request(self, request_id: str, assigned_to: str) -> bool:
        """Start processing a data subject request"""
        try:
            if request_id not in self.requests:
                raise ValueError(f"Request {request_id} not found")
            
            request = self.requests[request_id]
            request.status = RequestStatus.IN_PROGRESS
            request.assigned_to = assigned_to
            
            logger.info(f"Request {request_id} assigned to {assigned_to}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process request: {str(e)}")
            return False
    
    def complete_request(self,
                       request_id: str,
                       response_data: Dict[str, Any],
                       success: bool = True) -> bool:
        """Complete a data subject request"""
        try:
            if request_id not in self.requests:
                raise ValueError(f"Request {request_id} not found")
            
            request = self.requests[request_id]
            request.status = RequestStatus.COMPLETED if success else RequestStatus.REJECTED
            request.response_data = response_data
            request.completed_at = datetime.now()
            
            if not success:
                request.rejection_reason = response_data.get("reason", "Request could not be fulfilled")
            
            logger.info(f"Request {request_id} completed with status: {request.status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete request: {str(e)}")
            return False
    
    def get_request(self, request_id: str) -> Optional[DataSubjectRequest]:
        """Get a specific request"""
        try:
            return self.requests.get(request_id)
            
        except Exception as e:
            logger.error(f"Failed to get request: {str(e)}")
            return None
    
    def get_subject_requests(self, subject_id: str) -> List[DataSubjectRequest]:
        """Get all requests for a data subject"""
        try:
            subject_requests = [
                request for request in self.requests.values()
                if request.subject_id == subject_id
            ]
            
            logger.debug(f"Retrieved {len(subject_requests)} requests for subject {subject_id}")
            return subject_requests
            
        except Exception as e:
            logger.error(f"Failed to get subject requests: {str(e)}")
            return []
    
    def get_pending_requests(self) -> List[DataSubjectRequest]:
        """Get all pending requests"""
        try:
            pending_requests = [
                request for request in self.requests.values()
                if request.status == RequestStatus.PENDING
            ]
            
            logger.debug(f"Retrieved {len(pending_requests)} pending requests")
            return pending_requests
            
        except Exception as e:
            logger.error(f"Failed to get pending requests: {str(e)}")
            return []
    
    def get_overdue_requests(self) -> List[DataSubjectRequest]:
        """Get all overdue requests"""
        try:
            now = datetime.now()
            overdue_requests = [
                request for request in self.requests.values()
                if request.status in [RequestStatus.PENDING, RequestStatus.IN_PROGRESS]
                and request.due_date < now
            ]
            
            logger.debug(f"Retrieved {len(overdue_requests)} overdue requests")
            return overdue_requests
            
        except Exception as e:
            logger.error(f"Failed to get overdue requests: {str(e)}")
            return []
    
    def generate_request_report(self, 
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a request processing report"""
        try:
            # Filter requests by date range
            filtered_requests = self.requests.values()
            if start_date:
                filtered_requests = [r for r in filtered_requests if r.submitted_at >= start_date]
            if end_date:
                filtered_requests = [r for r in filtered_requests if r.submitted_at <= end_date]
            
            # Generate statistics
            total_requests = len(filtered_requests)
            
            status_counts = {}
            right_type_counts = {}
            completion_times = []
            
            for request in filtered_requests:
                # Count by status
                status = request.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Count by right type
                right_type = request.right_type.value
                right_type_counts[right_type] = right_type_counts.get(right_type, 0) + 1
                
                # Calculate completion time
                if request.completed_at:
                    completion_time = (request.completed_at - request.submitted_at).total_seconds() / 3600  # hours
                    completion_times.append(completion_time)
            
            # Calculate average completion time
            avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
            
            # Check for overdue requests
            overdue_count = len(self.get_overdue_requests())
            
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "date_range": {
                        "start": start_date.isoformat() if start_date else None,
                        "end": end_date.isoformat() if end_date else None
                    },
                    "total_requests": total_requests
                },
                "statistics": {
                    "status_counts": status_counts,
                    "right_type_counts": right_type_counts,
                    "overdue_count": overdue_count,
                    "average_completion_time_hours": round(avg_completion_time, 2)
                },
                "compliance_metrics": {
                    "response_time_compliance": (total_requests - overdue_count) / total_requests if total_requests > 0 else 1.0,
                    "completion_rate": status_counts.get(RequestStatus.COMPLETED.value, 0) / total_requests if total_requests > 0 else 0
                }
            }
            
            logger.info(f"Request report generated: {total_requests} requests analyzed")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate request report: {str(e)}")
            return {"error": str(e)}
    
    def auto_process_access_request(self, request_id: str, subject_data: Dict[str, Any]) -> bool:
        """Automatically process an access request"""
        try:
            if request_id not in self.requests:
                raise ValueError(f"Request {request_id} not found")
            
            request = self.requests[request_id]
            if request.right_type != DataSubjectRight.ACCESS:
                raise ValueError("Request is not an access request")
            
            # Prepare response data
            response_data = {
                "personal_data": subject_data.get("personal_data", {}),
                "processing_activities": subject_data.get("processing_activities", []),
                "consent_records": subject_data.get("consent_records", []),
                "data_categories": subject_data.get("data_categories", []),
                "processing_purposes": subject_data.get("processing_purposes", []),
                "generated_at": datetime.now().isoformat()
            }
            
            return self.complete_request(request_id, response_data, success=True)
            
        except Exception as e:
            logger.error(f"Failed to auto-process access request: {str(e)}")
            return False
    
    def auto_process_erasure_request(self, request_id: str, can_erase: bool, reason: str = None) -> bool:
        """Automatically process an erasure request"""
        try:
            if request_id not in self.requests:
                raise ValueError(f"Request {request_id} not found")
            
            request = self.requests[request_id]
            if request.right_type != DataSubjectRight.ERASURE:
                raise ValueError("Request is not an erasure request")
            
            if can_erase:
                response_data = {
                    "status": "erased",
                    "message": "Data has been marked for erasure",
                    "erased_at": datetime.now().isoformat()
                }
                return self.complete_request(request_id, response_data, success=True)
            else:
                response_data = {
                    "status": "rejected",
                    "reason": reason or "Erasure not lawful due to legal obligations",
                    "rejected_at": datetime.now().isoformat()
                }
                return self.complete_request(request_id, response_data, success=False)
            
        except Exception as e:
            logger.error(f"Failed to auto-process erasure request: {str(e)}")
            return False
    
    def get_request_statistics(self) -> Dict[str, Any]:
        """Get request processing statistics"""
        try:
            total_requests = len(self.requests)
            
            status_counts = {}
            right_type_counts = {}
            
            for request in self.requests.values():
                # Count by status
                status = request.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Count by right type
                right_type = request.right_type.value
                right_type_counts[right_type] = right_type_counts.get(right_type, 0) + 1
            
            # Calculate overdue requests
            overdue_requests = self.get_overdue_requests()
            
            return {
                "total_requests": total_requests,
                "status_counts": status_counts,
                "right_type_counts": right_type_counts,
                "overdue_requests": len(overdue_requests),
                "response_time_days": self.response_days,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get request statistics: {str(e)}")
            return {"error": str(e)}
