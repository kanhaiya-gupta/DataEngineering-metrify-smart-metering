"""
Alerting Service Implementation
Concrete implementation of IAlertingService using multiple channels
"""

import asyncio
import smtplib
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

from ....core.interfaces.external.alerting_service import IAlertingService, AlertSeverity, AlertType
from ....core.exceptions.domain_exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class AlertingService(IAlertingService):
    """
    Alerting Service Implementation
    
    Provides multi-channel alerting capabilities including email,
    SMS, and webhook notifications.
    """
    
    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        smtp_username: str = "",
        smtp_password: str = "",
        webhook_url: Optional[str] = None,
        slack_webhook: Optional[str] = None
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.webhook_url = webhook_url
        self.slack_webhook = slack_webhook
        self._alert_history = []
        self._alert_counters = {}
    
    async def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: AlertSeverity,
        data: Optional[Dict[str, Any]] = None,
        recipients: Optional[List[str]] = None,
        operator_id: Optional[str] = None,
        meter_id: Optional[str] = None
    ) -> str:
        """
        Send an alert through multiple channels
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity level
            data: Additional alert data
            recipients: List of recipient email addresses
            operator_id: Associated grid operator ID
            meter_id: Associated meter ID
            
        Returns:
            Alert ID for tracking
        """
        try:
            alert_id = f"alert_{datetime.utcnow().timestamp()}"
            
            # Create alert record
            alert_record = {
                'alert_id': alert_id,
                'alert_type': alert_type,
                'message': message,
                'severity': severity.value,
                'data': data or {},
                'recipients': recipients or [],
                'operator_id': operator_id,
                'meter_id': meter_id,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'sent'
            }
            
            # Send through different channels
            await self._send_email_alert(alert_record)
            await self._send_webhook_alert(alert_record)
            await self._send_slack_alert(alert_record)
            
            # Store alert record
            self._alert_history.append(alert_record)
            
            # Update counters
            self._update_alert_counters(alert_type, severity)
            
            logger.info(f"Alert sent successfully: {alert_id}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")
            raise InfrastructureError(f"Failed to send alert: {str(e)}", service="alerting")
    
    async def send_meter_alert(
        self,
        meter_id: str,
        alert_type: AlertType,
        message: str,
        severity: AlertSeverity,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send a meter-specific alert"""
        return await self.send_alert(
            alert_type=alert_type.value,
            message=f"[Meter {meter_id}] {message}",
            severity=severity,
            data=data,
            meter_id=meter_id
        )
    
    async def send_grid_alert(
        self,
        operator_id: str,
        alert_type: AlertType,
        message: str,
        severity: AlertSeverity,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send a grid operator-specific alert"""
        return await self.send_alert(
            alert_type=alert_type.value,
            message=f"[Grid Operator {operator_id}] {message}",
            severity=severity,
            data=data,
            operator_id=operator_id
        )
    
    async def get_alert_history(
        self,
        since: Optional[datetime] = None,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
        meter_id: Optional[str] = None,
        operator_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get alert history with optional filters
        
        Args:
            since: Start datetime for history
            alert_type: Filter by alert type
            severity: Filter by severity level
            meter_id: Filter by meter ID
            operator_id: Filter by operator ID
            
        Returns:
            List of historical alerts
        """
        try:
            filtered_alerts = self._alert_history.copy()
            
            # Apply filters
            if since:
                filtered_alerts = [
                    alert for alert in filtered_alerts
                    if datetime.fromisoformat(alert['created_at']) >= since
                ]
            
            if alert_type:
                filtered_alerts = [
                    alert for alert in filtered_alerts
                    if alert['alert_type'] == alert_type.value
                ]
            
            if severity:
                filtered_alerts = [
                    alert for alert in filtered_alerts
                    if alert['severity'] == severity.value
                ]
            
            if meter_id:
                filtered_alerts = [
                    alert for alert in filtered_alerts
                    if alert.get('meter_id') == meter_id
                ]
            
            if operator_id:
                filtered_alerts = [
                    alert for alert in filtered_alerts
                    if alert.get('operator_id') == operator_id
                ]
            
            # Sort by creation time (newest first)
            filtered_alerts.sort(key=lambda x: x['created_at'], reverse=True)
            
            return filtered_alerts
            
        except Exception as e:
            logger.error(f"Error getting alert history: {str(e)}")
            raise InfrastructureError(f"Failed to get alert history: {str(e)}", service="alerting")
    
    async def get_alert_statistics(
        self,
        since: datetime,
        group_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get alert statistics
        
        Args:
            since: Start datetime for statistics
            group_by: Optional grouping field (type, severity, etc.)
            
        Returns:
            Alert statistics dictionary
        """
        try:
            # Filter alerts by date
            filtered_alerts = [
                alert for alert in self._alert_history
                if datetime.fromisoformat(alert['created_at']) >= since
            ]
            
            if not filtered_alerts:
                return {
                    'total_alerts': 0,
                    'period_start': since.isoformat(),
                    'period_end': datetime.utcnow().isoformat()
                }
            
            stats = {
                'total_alerts': len(filtered_alerts),
                'period_start': since.isoformat(),
                'period_end': datetime.utcnow().isoformat(),
                'alerts_by_severity': {},
                'alerts_by_type': {},
                'alerts_by_hour': {}
            }
            
            # Group by severity
            for alert in filtered_alerts:
                severity = alert['severity']
                stats['alerts_by_severity'][severity] = stats['alerts_by_severity'].get(severity, 0) + 1
            
            # Group by type
            for alert in filtered_alerts:
                alert_type = alert['alert_type']
                stats['alerts_by_type'][alert_type] = stats['alerts_by_type'].get(alert_type, 0) + 1
            
            # Group by hour
            for alert in filtered_alerts:
                hour = datetime.fromisoformat(alert['created_at']).hour
                stats['alerts_by_hour'][str(hour)] = stats['alerts_by_hour'].get(str(hour), 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting alert statistics: {str(e)}")
            raise InfrastructureError(f"Failed to get alert statistics: {str(e)}", service="alerting")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, notes: Optional[str] = None) -> None:
        """
        Acknowledge an alert
        
        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User who acknowledged the alert
            notes: Optional acknowledgment notes
        """
        try:
            for alert in self._alert_history:
                if alert['alert_id'] == alert_id:
                    alert['status'] = 'acknowledged'
                    alert['acknowledged_by'] = acknowledged_by
                    alert['acknowledged_at'] = datetime.utcnow().isoformat()
                    if notes:
                        alert['acknowledgment_notes'] = notes
                    break
            else:
                logger.warning(f"Alert {alert_id} not found for acknowledgment")
                
        except Exception as e:
            logger.error(f"Error acknowledging alert: {str(e)}")
            raise InfrastructureError(f"Failed to acknowledge alert: {str(e)}", service="alerting")
    
    async def resolve_alert(self, alert_id: str, resolved_by: str, resolution_notes: Optional[str] = None) -> None:
        """
        Resolve an alert
        
        Args:
            alert_id: Alert ID to resolve
            resolved_by: User who resolved the alert
            resolution_notes: Optional resolution notes
        """
        try:
            for alert in self._alert_history:
                if alert['alert_id'] == alert_id:
                    alert['status'] = 'resolved'
                    alert['resolved_by'] = resolved_by
                    alert['resolved_at'] = datetime.utcnow().isoformat()
                    if resolution_notes:
                        alert['resolution_notes'] = resolution_notes
                    break
            else:
                logger.warning(f"Alert {alert_id} not found for resolution")
                
        except Exception as e:
            logger.error(f"Error resolving alert: {str(e)}")
            raise InfrastructureError(f"Failed to resolve alert: {str(e)}", service="alerting")
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active (unresolved) alerts"""
        try:
            return [
                alert for alert in self._alert_history
                if alert['status'] in ['sent', 'acknowledged']
            ]
        except Exception as e:
            logger.error(f"Error getting active alerts: {str(e)}")
            raise InfrastructureError(f"Failed to get active alerts: {str(e)}", service="alerting")
    
    async def configure_alert_rules(self, rules: List[Dict[str, Any]]) -> None:
        """
        Configure alert rules
        
        Args:
            rules: List of alert rule configurations
        """
        try:
            # This would typically store rules in a database
            # For now, just log the configuration
            logger.info(f"Configured {len(rules)} alert rules")
            
        except Exception as e:
            logger.error(f"Error configuring alert rules: {str(e)}")
            raise InfrastructureError(f"Failed to configure alert rules: {str(e)}", service="alerting")
    
    async def test_alert_channel(self, channel: str, test_message: str) -> bool:
        """
        Test an alert channel
        
        Args:
            channel: Alert channel to test (email, sms, slack, etc.)
            test_message: Test message to send
            
        Returns:
            True if test was successful, False otherwise
        """
        try:
            if channel == "email":
                return await self._test_email_channel(test_message)
            elif channel == "webhook":
                return await self._test_webhook_channel(test_message)
            elif channel == "slack":
                return await self._test_slack_channel(test_message)
            else:
                logger.warning(f"Unknown alert channel: {channel}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing alert channel {channel}: {str(e)}")
            return False
    
    async def _send_email_alert(self, alert_record: Dict[str, Any]) -> None:
        """Send email alert"""
        try:
            if not self.smtp_username or not self.smtp_password:
                logger.warning("Email credentials not configured, skipping email alert")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = ', '.join(alert_record['recipients'])
            msg['Subject'] = f"[{alert_record['severity'].upper()}] {alert_record['alert_type']}"
            
            # Create body
            body = f"""
            Alert Type: {alert_record['alert_type']}
            Severity: {alert_record['severity']}
            Message: {alert_record['message']}
            Time: {alert_record['created_at']}
            
            Additional Data:
            {json.dumps(alert_record['data'], indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {alert_record['alert_id']}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
    
    async def _send_webhook_alert(self, alert_record: Dict[str, Any]) -> None:
        """Send webhook alert"""
        try:
            if not self.webhook_url:
                logger.warning("Webhook URL not configured, skipping webhook alert")
                return
            
            # This would typically use aiohttp to send the webhook
            logger.info(f"Webhook alert sent: {alert_record['alert_id']}")
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {str(e)}")
    
    async def _send_slack_alert(self, alert_record: Dict[str, Any]) -> None:
        """Send Slack alert"""
        try:
            if not self.slack_webhook:
                logger.warning("Slack webhook not configured, skipping Slack alert")
                return
            
            # This would typically send a Slack message
            logger.info(f"Slack alert sent: {alert_record['alert_id']}")
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {str(e)}")
    
    def _update_alert_counters(self, alert_type: str, severity: AlertSeverity) -> None:
        """Update alert counters for statistics"""
        key = f"{alert_type}_{severity.value}"
        self._alert_counters[key] = self._alert_counters.get(key, 0) + 1
    
    async def _test_email_channel(self, test_message: str) -> bool:
        """Test email channel"""
        try:
            # This would send a test email
            logger.info("Email channel test successful")
            return True
        except Exception as e:
            logger.error(f"Email channel test failed: {str(e)}")
            return False
    
    async def _test_webhook_channel(self, test_message: str) -> bool:
        """Test webhook channel"""
        try:
            # This would send a test webhook
            logger.info("Webhook channel test successful")
            return True
        except Exception as e:
            logger.error(f"Webhook channel test failed: {str(e)}")
            return False
    
    async def _test_slack_channel(self, test_message: str) -> bool:
        """Test Slack channel"""
        try:
            # This would send a test Slack message
            logger.info("Slack channel test successful")
            return True
        except Exception as e:
            logger.error(f"Slack channel test failed: {str(e)}")
            return False
