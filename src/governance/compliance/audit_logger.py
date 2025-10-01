"""
Audit Logger
Provides comprehensive audit logging and reporting capabilities
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_CHANGE = "permission_change"
    SYSTEM_CONFIGURATION = "system_configuration"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    CONSENT_CHANGE = "consent_change"
    PRIVACY_SETTING_CHANGE = "privacy_setting_change"
    BREACH_INCIDENT = "breach_incident"
    COMPLIANCE_CHECK = "compliance_check"

class AuditSeverity(Enum):
    """Audit event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Represents an audit event"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    resource_type: str
    resource_id: str
    action: str
    description: str
    severity: AuditSeverity
    ip_address: Optional[str]
    user_agent: Optional[str]
    metadata: Dict[str, Any]
    result: str  # success, failure, error
    error_message: Optional[str] = None

@dataclass
class AuditQuery:
    """Represents an audit query"""
    query_id: str
    user_id: str
    query_type: str
    filters: Dict[str, Any]
    timestamp: datetime
    result_count: int

class AuditLogger:
    """
    Provides comprehensive audit logging and reporting capabilities
    """
    
    def __init__(self, retention_days: int = 2555):  # 7 years default
        self.retention_days = retention_days
        self.audit_events = []
        self.audit_queries = []
        self.audit_config = {
            "log_data_access": True,
            "log_data_modification": True,
            "log_user_actions": True,
            "log_system_events": True,
            "log_privacy_events": True,
            "log_compliance_events": True
        }
        
        logger.info(f"AuditLogger initialized with {retention_days} days retention")
    
    def log_event(self,
                 event_type: AuditEventType,
                 user_id: Optional[str],
                 resource_type: str,
                 resource_id: str,
                 action: str,
                 description: str,
                 severity: AuditSeverity = AuditSeverity.MEDIUM,
                 session_id: Optional[str] = None,
                 ip_address: Optional[str] = None,
                 user_agent: Optional[str] = None,
                 metadata: Dict[str, Any] = None,
                 result: str = "success",
                 error_message: Optional[str] = None) -> str:
        """Log an audit event"""
        try:
            # Check if this event type should be logged
            if not self._should_log_event(event_type):
                return ""
            
            event_id = str(uuid.uuid4())
            
            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                description=description,
                severity=severity,
                ip_address=ip_address,
                user_agent=user_agent,
                metadata=metadata or {},
                result=result,
                error_message=error_message
            )
            
            self.audit_events.append(event)
            
            # Log to application logger
            log_message = f"Audit Event: {event_type.value} - {action} on {resource_type}:{resource_id} by {user_id or 'system'}"
            if result != "success":
                log_message += f" - {result}: {error_message or 'Unknown error'}"
            
            if severity == AuditSeverity.CRITICAL:
                logger.critical(log_message)
            elif severity == AuditSeverity.HIGH:
                logger.error(log_message)
            elif severity == AuditSeverity.MEDIUM:
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            logger.debug(f"Audit event logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")
            return ""
    
    def log_data_access(self,
                       user_id: str,
                       resource_type: str,
                       resource_id: str,
                       access_type: str,
                       query_params: Dict[str, Any] = None,
                       session_id: Optional[str] = None,
                       ip_address: Optional[str] = None) -> str:
        """Log data access event"""
        try:
            return self.log_event(
                event_type=AuditEventType.DATA_ACCESS,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=f"access_{access_type}",
                description=f"Data access: {access_type} on {resource_type}",
                severity=AuditSeverity.LOW,
                session_id=session_id,
                ip_address=ip_address,
                metadata={"query_params": query_params or {}}
            )
            
        except Exception as e:
            logger.error(f"Failed to log data access: {str(e)}")
            return ""
    
    def log_data_modification(self,
                            user_id: str,
                            resource_type: str,
                            resource_id: str,
                            modification_type: str,
                            changes: Dict[str, Any] = None,
                            session_id: Optional[str] = None,
                            ip_address: Optional[str] = None) -> str:
        """Log data modification event"""
        try:
            return self.log_event(
                event_type=AuditEventType.DATA_MODIFICATION,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=f"modify_{modification_type}",
                description=f"Data modification: {modification_type} on {resource_type}",
                severity=AuditSeverity.MEDIUM,
                session_id=session_id,
                ip_address=ip_address,
                metadata={"changes": changes or {}}
            )
            
        except Exception as e:
            logger.error(f"Failed to log data modification: {str(e)}")
            return ""
    
    def log_user_authentication(self,
                               user_id: str,
                               action: str,
                               success: bool,
                               session_id: Optional[str] = None,
                               ip_address: Optional[str] = None,
                               user_agent: Optional[str] = None) -> str:
        """Log user authentication event"""
        try:
            event_type = AuditEventType.USER_LOGIN if action == "login" else AuditEventType.USER_LOGOUT
            
            return self.log_event(
                event_type=event_type,
                user_id=user_id,
                resource_type="user",
                resource_id=user_id,
                action=action,
                description=f"User {action}",
                severity=AuditSeverity.MEDIUM if success else AuditSeverity.HIGH,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                result="success" if success else "failure"
            )
            
        except Exception as e:
            logger.error(f"Failed to log user authentication: {str(e)}")
            return ""
    
    def log_privacy_event(self,
                         user_id: str,
                         subject_id: str,
                         event_type: str,
                         description: str,
                         metadata: Dict[str, Any] = None,
                         session_id: Optional[str] = None) -> str:
        """Log privacy-related event"""
        try:
            audit_event_type = AuditEventType.CONSENT_CHANGE if "consent" in event_type else AuditEventType.PRIVACY_SETTING_CHANGE
            
            return self.log_event(
                event_type=audit_event_type,
                user_id=user_id,
                resource_type="privacy",
                resource_id=subject_id,
                action=event_type,
                description=description,
                severity=AuditSeverity.HIGH,
                session_id=session_id,
                metadata=metadata or {}
            )
            
        except Exception as e:
            logger.error(f"Failed to log privacy event: {str(e)}")
            return ""
    
    def log_compliance_event(self,
                           user_id: str,
                           compliance_type: str,
                           resource_id: str,
                           action: str,
                           description: str,
                           result: str = "success",
                           metadata: Dict[str, Any] = None) -> str:
        """Log compliance-related event"""
        try:
            return self.log_event(
                event_type=AuditEventType.COMPLIANCE_CHECK,
                user_id=user_id,
                resource_type="compliance",
                resource_id=resource_id,
                action=action,
                description=description,
                severity=AuditSeverity.HIGH if result != "success" else AuditSeverity.MEDIUM,
                metadata=metadata or {},
                result=result
            )
            
        except Exception as e:
            logger.error(f"Failed to log compliance event: {str(e)}")
            return ""
    
    def log_breach_incident(self,
                           incident_id: str,
                           breach_type: str,
                           affected_resources: List[str],
                           severity: AuditSeverity,
                           description: str,
                           metadata: Dict[str, Any] = None) -> str:
        """Log data breach incident"""
        try:
            return self.log_event(
                event_type=AuditEventType.BREACH_INCIDENT,
                user_id=None,  # System event
                resource_type="breach",
                resource_id=incident_id,
                action="breach_detected",
                description=description,
                severity=severity,
                metadata={
                    "breach_type": breach_type,
                    "affected_resources": affected_resources,
                    **(metadata or {})
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to log breach incident: {str(e)}")
            return ""
    
    def query_audit_events(self,
                          user_id: Optional[str] = None,
                          event_types: List[AuditEventType] = None,
                          resource_types: List[str] = None,
                          resource_ids: List[str] = None,
                          severity_levels: List[AuditSeverity] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: int = 1000) -> List[AuditEvent]:
        """Query audit events with filters"""
        try:
            filtered_events = self.audit_events.copy()
            
            # Apply filters
            if user_id:
                filtered_events = [e for e in filtered_events if e.user_id == user_id]
            
            if event_types:
                filtered_events = [e for e in filtered_events if e.event_type in event_types]
            
            if resource_types:
                filtered_events = [e for e in filtered_events if e.resource_type in resource_types]
            
            if resource_ids:
                filtered_events = [e for e in filtered_events if e.resource_id in resource_ids]
            
            if severity_levels:
                filtered_events = [e for e in filtered_events if e.severity in severity_levels]
            
            if start_time:
                filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
            
            if end_time:
                filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
            
            # Sort by timestamp (newest first)
            filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply limit
            filtered_events = filtered_events[:limit]
            
            # Log the query
            query_id = self._log_audit_query(user_id, "event_query", {
                "event_types": [e.value for e in (event_types or [])],
                "resource_types": resource_types or [],
                "severity_levels": [s.value for s in (severity_levels or [])],
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "limit": limit
            }, len(filtered_events))
            
            logger.info(f"Audit query executed: {query_id}, returned {len(filtered_events)} events")
            return filtered_events
            
        except Exception as e:
            logger.error(f"Failed to query audit events: {str(e)}")
            return []
    
    def _log_audit_query(self, user_id: str, query_type: str, filters: Dict[str, Any], result_count: int) -> str:
        """Log an audit query"""
        try:
            query_id = str(uuid.uuid4())
            
            query = AuditQuery(
                query_id=query_id,
                user_id=user_id,
                query_type=query_type,
                filters=filters,
                timestamp=datetime.now(),
                result_count=result_count
            )
            
            self.audit_queries.append(query)
            return query_id
            
        except Exception as e:
            logger.error(f"Failed to log audit query: {str(e)}")
            return ""
    
    def generate_audit_report(self,
                            start_time: datetime,
                            end_time: datetime,
                            report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate an audit report"""
        try:
            # Get events in time range
            events = self.query_audit_events(
                start_time=start_time,
                end_time=end_time,
                limit=10000
            )
            
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "report_type": report_type,
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    },
                    "total_events": len(events)
                },
                "event_summary": {},
                "user_activity": {},
                "resource_activity": {},
                "security_events": [],
                "compliance_events": [],
                "recommendations": []
            }
            
            # Event summary by type
            event_counts = {}
            for event in events:
                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            report["event_summary"] = event_counts
            
            # User activity
            user_activity = {}
            for event in events:
                if event.user_id:
                    if event.user_id not in user_activity:
                        user_activity[event.user_id] = {
                            "total_events": 0,
                            "event_types": {},
                            "last_activity": event.timestamp
                        }
                    user_activity[event.user_id]["total_events"] += 1
                    event_type = event.event_type.value
                    user_activity[event.user_id]["event_types"][event_type] = \
                        user_activity[event.user_id]["event_types"].get(event_type, 0) + 1
            report["user_activity"] = user_activity
            
            # Resource activity
            resource_activity = {}
            for event in events:
                resource_key = f"{event.resource_type}:{event.resource_id}"
                if resource_key not in resource_activity:
                    resource_activity[resource_key] = {
                        "total_events": 0,
                        "event_types": {},
                        "last_accessed": event.timestamp
                    }
                resource_activity[resource_key]["total_events"] += 1
                event_type = event.event_type.value
                resource_activity[resource_key]["event_types"][event_type] = \
                    resource_activity[resource_key]["event_types"].get(event_type, 0) + 1
            report["resource_activity"] = resource_activity
            
            # Security events (high severity or failed operations)
            security_events = [
                {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "user_id": event.user_id,
                    "event_type": event.event_type.value,
                    "action": event.action,
                    "description": event.description,
                    "severity": event.severity.value,
                    "result": event.result
                }
                for event in events
                if event.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL] or event.result != "success"
            ]
            report["security_events"] = security_events
            
            # Compliance events
            compliance_events = [
                {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "user_id": event.user_id,
                    "action": event.action,
                    "description": event.description,
                    "result": event.result
                }
                for event in events
                if event.event_type == AuditEventType.COMPLIANCE_CHECK
            ]
            report["compliance_events"] = compliance_events
            
            # Generate recommendations
            report["recommendations"] = self._generate_audit_recommendations(events)
            
            logger.info(f"Audit report generated: {len(events)} events analyzed")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate audit report: {str(e)}")
            return {"error": str(e)}
    
    def _generate_audit_recommendations(self, events: List[AuditEvent]) -> List[str]:
        """Generate audit recommendations"""
        recommendations = []
        
        # Check for failed authentication attempts
        failed_logins = [e for e in events if e.event_type == AuditEventType.USER_LOGIN and e.result != "success"]
        if len(failed_logins) > 10:
            recommendations.append(f"High number of failed login attempts ({len(failed_logins)}) - review security")
        
        # Check for unusual access patterns
        data_access_events = [e for e in events if e.event_type == AuditEventType.DATA_ACCESS]
        if len(data_access_events) > 1000:
            recommendations.append("High volume of data access - consider access pattern analysis")
        
        # Check for critical events
        critical_events = [e for e in events if e.severity == AuditSeverity.CRITICAL]
        if critical_events:
            recommendations.append(f"{len(critical_events)} critical events detected - immediate review required")
        
        # Check for compliance issues
        failed_compliance = [e for e in events if e.event_type == AuditEventType.COMPLIANCE_CHECK and e.result != "success"]
        if failed_compliance:
            recommendations.append(f"{len(failed_compliance)} compliance check failures - review processes")
        
        return recommendations
    
    def _should_log_event(self, event_type: AuditEventType) -> bool:
        """Check if event type should be logged based on configuration"""
        try:
            if event_type in [AuditEventType.DATA_ACCESS, AuditEventType.DATA_MODIFICATION]:
                return self.audit_config["log_data_access"]
            elif event_type in [AuditEventType.USER_LOGIN, AuditEventType.USER_LOGOUT]:
                return self.audit_config["log_user_actions"]
            elif event_type in [AuditEventType.SYSTEM_CONFIGURATION]:
                return self.audit_config["log_system_events"]
            elif event_type in [AuditEventType.CONSENT_CHANGE, AuditEventType.PRIVACY_SETTING_CHANGE]:
                return self.audit_config["log_privacy_events"]
            elif event_type in [AuditEventType.COMPLIANCE_CHECK]:
                return self.audit_config["log_compliance_events"]
            else:
                return True
                
        except Exception as e:
            logger.error(f"Failed to check if event should be logged: {str(e)}")
            return True
    
    def cleanup_old_events(self) -> int:
        """Clean up events older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            old_events = [e for e in self.audit_events if e.timestamp < cutoff_date]
            self.audit_events = [e for e in self.audit_events if e.timestamp >= cutoff_date]
            
            old_queries = [q for q in self.audit_queries if q.timestamp < cutoff_date]
            self.audit_queries = [q for q in self.audit_queries if q.timestamp >= cutoff_date]
            
            cleaned_count = len(old_events) + len(old_queries)
            logger.info(f"Cleaned up {cleaned_count} old audit records")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old events: {str(e)}")
            return 0
    
    def export_audit_data(self, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         format: str = "json") -> str:
        """Export audit data"""
        try:
            events = self.query_audit_events(
                start_time=start_time,
                end_time=end_time,
                limit=50000
            )
            
            export_data = {
                "export_metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "total_events": len(events),
                    "time_range": {
                        "start": start_time.isoformat() if start_time else None,
                        "end": end_time.isoformat() if end_time else None
                    }
                },
                "events": [asdict(event) for event in events],
                "queries": [asdict(query) for query in self.audit_queries]
            }
            
            if format.lower() == "json":
                return json.dumps(export_data, indent=2, default=str)
            else:
                return str(export_data)
                
        except Exception as e:
            logger.error(f"Failed to export audit data: {str(e)}")
            return f"Export failed: {str(e)}"
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics"""
        try:
            total_events = len(self.audit_events)
            total_queries = len(self.audit_queries)
            
            # Event counts by type
            event_counts = {}
            for event in self.audit_events:
                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Event counts by severity
            severity_counts = {}
            for event in self.audit_events:
                severity = event.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Success/failure counts
            success_count = len([e for e in self.audit_events if e.result == "success"])
            failure_count = len([e for e in self.audit_events if e.result != "success"])
            
            return {
                "total_events": total_events,
                "total_queries": total_queries,
                "event_counts_by_type": event_counts,
                "event_counts_by_severity": severity_counts,
                "success_rate": success_count / total_events if total_events > 0 else 0,
                "failure_count": failure_count,
                "oldest_event": min([e.timestamp for e in self.audit_events]) if self.audit_events else None,
                "newest_event": max([e.timestamp for e in self.audit_events]) if self.audit_events else None,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {str(e)}")
            return {"error": str(e)}
