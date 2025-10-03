"""
ML Alerting Module

This module provides alerting capabilities for ML model monitoring:
- Alert generation and management
- Alert routing and escalation
- Alert templates and formatting
- Integration with external alerting systems
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    condition: str  # Python expression to evaluate
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 15
    escalation_minutes: int = 60
    recipients: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)  # email, slack, webhook, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AlertChannel(ABC):
    """Abstract base class for alert channels"""
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel"""
        pass
    
    @abstractmethod
    def get_channel_name(self) -> str:
        """Get channel name"""
        pass

class EmailAlertChannel(AlertChannel):
    """Email alert channel"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, from_email: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(alert.metadata.get('recipients', []))
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
Alert ID: {alert.alert_id}
Rule ID: {alert.rule_id}
Severity: {alert.severity.value.upper()}
Status: {alert.status.value}
Created: {alert.created_at.isoformat()}

Message:
{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Alert {alert.alert_id} sent via email")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert {alert.alert_id}: {str(e)}")
            return False
    
    def get_channel_name(self) -> str:
        return "email"

class SlackAlertChannel(AlertChannel):
    """Slack alert channel"""
    
    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        self.webhook_url = webhook_url
        self.channel = channel
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via Slack webhook"""
        try:
            import requests
            
            color_map = {
                AlertSeverity.LOW: "#36a64f",
                AlertSeverity.MEDIUM: "#ffaa00",
                AlertSeverity.HIGH: "#ff6600",
                AlertSeverity.CRITICAL: "#ff0000"
            }
            
            payload = {
                "channel": self.channel,
                "attachments": [{
                    "color": color_map.get(alert.severity, "#36a64f"),
                    "title": f"[{alert.severity.value.upper()}] {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Alert ID", "value": alert.alert_id, "short": True},
                        {"title": "Rule ID", "value": alert.rule_id, "short": True},
                        {"title": "Status", "value": alert.status.value, "short": True},
                        {"title": "Created", "value": alert.created_at.isoformat(), "short": True}
                    ],
                    "footer": "ML Monitoring System",
                    "ts": int(alert.created_at.timestamp())
                }]
            }
            
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Alert {alert.alert_id} sent via Slack")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert {alert.alert_id}: {str(e)}")
            return False
    
    def get_channel_name(self) -> str:
        return "slack"

class WebhookAlertChannel(AlertChannel):
    """Generic webhook alert channel"""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook"""
        try:
            import requests
            
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "created_at": alert.created_at.isoformat(),
                "metadata": alert.metadata
            }
            
            response = requests.post(
                self.webhook_url, 
                json=payload, 
                headers=self.headers
            )
            response.raise_for_status()
            
            logger.info(f"Alert {alert.alert_id} sent via webhook")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert {alert.alert_id}: {str(e)}")
            return False
    
    def get_channel_name(self) -> str:
        return "webhook"

class MLAlerting:
    """
    ML Alerting system for model monitoring
    
    Manages alert rules, generates alerts based on conditions,
    and routes alerts through various channels.
    """
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.channels: Dict[str, AlertChannel] = {}
        self.alert_history: List[Alert] = []
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.monitoring_active = False
        self.monitoring_task = None
        
        logger.info("ML Alerting system initialized")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})")
    
    def add_channel(self, channel: AlertChannel):
        """Add an alert channel"""
        channel_name = channel.get_channel_name()
        self.channels[channel_name] = channel
        logger.info(f"Added alert channel: {channel_name}")
    
    def evaluate_alert_rules(self, context: Dict[str, Any]) -> List[Alert]:
        """Evaluate all alert rules against the given context"""
        triggered_alerts = []
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule_id in self.alert_cooldowns:
                if datetime.now() < self.alert_cooldowns[rule_id]:
                    continue
            
            try:
                # Evaluate rule condition
                if self._evaluate_condition(rule.condition, context):
                    alert = self._create_alert(rule, context)
                    triggered_alerts.append(alert)
                    
                    # Set cooldown
                    self.alert_cooldowns[rule_id] = datetime.now() + timedelta(minutes=rule.cooldown_minutes)
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_id}: {str(e)}")
        
        return triggered_alerts
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate alert rule condition safely"""
        try:
            # Create a safe evaluation context
            safe_context = {
                '__builtins__': {},
                'len': len,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                **context
            }
            
            return eval(condition, safe_context)
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {str(e)}")
            return False
    
    def _create_alert(self, rule: AlertRule, context: Dict[str, Any]) -> Alert:
        """Create an alert from a triggered rule"""
        alert_id = f"{rule.rule_id}_{int(time.time())}"
        
        # Format title and message with context
        title = self._format_template(rule.name, context)
        message = self._format_template(rule.condition, context)
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            title=title,
            message=message,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                'recipients': rule.recipients,
                'channels': rule.channels,
                'context': context
            }
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        return alert
    
    def _format_template(self, template: str, context: Dict[str, Any]) -> str:
        """Format template string with context variables"""
        try:
            return template.format(**context)
        except KeyError as e:
            logger.warning(f"Missing context variable {e} in template: {template}")
            return template
        except Exception as e:
            logger.error(f"Error formatting template '{template}': {str(e)}")
            return template
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert through configured channels"""
        success_count = 0
        total_channels = 0
        
        for channel_name in alert.metadata.get('channels', []):
            if channel_name in self.channels:
                total_channels += 1
                channel = self.channels[channel_name]
                
                try:
                    success = await channel.send_alert(alert)
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error sending alert {alert.alert_id} via {channel_name}: {str(e)}")
        
        # If no specific channels specified, send to all channels
        if total_channels == 0:
            for channel in self.channels.values():
                total_channels += 1
                try:
                    success = await channel.send_alert(alert)
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error sending alert {alert.alert_id} via {channel.get_channel_name()}: {str(e)}")
        
        success_rate = success_count / total_channels if total_channels > 0 else 0
        logger.info(f"Alert {alert.alert_id} sent to {success_count}/{total_channels} channels")
        
        return success_rate > 0
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by
        alert.updated_at = datetime.now()
        
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.resolved_by = resolved_by
        alert.updated_at = datetime.now()
        
        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [alert for alert in self.alerts.values() 
                if alert.status == AlertStatus.ACTIVE]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity"""
        return [alert for alert in self.alerts.values() 
                if alert.severity == severity]
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.alert_history 
                        if alert.created_at >= cutoff_time]
        
        severity_counts = {}
        status_counts = {}
        
        for alert in recent_alerts:
            severity_counts[alert.severity.value] = severity_counts.get(alert.severity.value, 0) + 1
            status_counts[alert.status.value] = status_counts.get(alert.status.value, 0) + 1
        
        return {
            "time_period_hours": hours,
            "total_alerts": len(recent_alerts),
            "severity_breakdown": severity_counts,
            "status_breakdown": status_counts,
            "active_alerts": len(self.get_active_alerts()),
            "rules_configured": len(self.alert_rules),
            "channels_configured": len(self.channels)
        }
    
    def start_monitoring(self, evaluation_interval: int = 60):
        """Start continuous alert monitoring"""
        if self.monitoring_active:
            logger.warning("Alert monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(evaluation_interval)
        )
        logger.info(f"Started alert monitoring with {evaluation_interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous alert monitoring"""
        if not self.monitoring_active:
            logger.warning("Alert monitoring is not active")
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("Stopped alert monitoring")
    
    async def _monitoring_loop(self, evaluation_interval: int):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                # This would integrate with actual monitoring data
                # For now, this is a placeholder
                context = await self._collect_monitoring_context()
                triggered_alerts = self.evaluate_alert_rules(context)
                
                for alert in triggered_alerts:
                    await self.send_alert(alert)
                
                await asyncio.sleep(evaluation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {str(e)}")
                await asyncio.sleep(evaluation_interval)
    
    async def _collect_monitoring_context(self) -> Dict[str, Any]:
        """Collect monitoring context for alert evaluation"""
        # This is a placeholder - in real implementation,
        # this would collect actual monitoring data
        return {
            "timestamp": datetime.now().isoformat(),
            "model_performance": 0.85,
            "data_drift_score": 0.12,
            "prediction_latency_ms": 150,
            "error_rate": 0.02
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get alerting system health status"""
        active_alerts = self.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        
        return {
            "status": "healthy" if not critical_alerts else "unhealthy",
            "monitoring_active": self.monitoring_active,
            "rules_configured": len(self.alert_rules),
            "channels_configured": len(self.channels),
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "total_alerts_history": len(self.alert_history)
        }
