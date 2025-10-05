"""
Data Quality Monitoring Service
Tracks and monitors data quality metrics over time
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict, deque

from ....core.config.config_loader import ConfigLoader
from ....core.exceptions.domain_exceptions import DataQualityError
from .config_driven_data_quality_service import ConfigDrivenDataQualityService, ValidationResult, AnomalyResult

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a specific time period"""
    data_type: str
    timestamp: datetime
    total_records: int
    quality_score: float
    validation_passed: bool
    anomaly_detected: bool
    violation_count: int
    warning_count: int
    null_percentage: float
    duplicate_percentage: float
    processing_time_ms: float


@dataclass
class QualityAlert:
    """Quality alert when thresholds are breached"""
    alert_id: str
    data_type: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    metrics: Dict[str, Any]
    resolved: bool = False


class DataQualityMonitoringService:
    """
    Monitors data quality metrics and generates alerts
    """
    
    def __init__(self, alert_threshold: float = 0.7, history_size: int = 1000):
        self.alert_threshold = alert_threshold
        self.history_size = history_size
        self.quality_service = ConfigDrivenDataQualityService()
        self.config_loader = ConfigLoader()
        
        # In-memory storage for metrics and alerts
        self.quality_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.active_alerts: Dict[str, QualityAlert] = {}
        self.alert_counter = 0
        
        # Quality thresholds
        self.thresholds = {
            'quality_score': 0.8,
            'null_percentage': 10.0,
            'duplicate_percentage': 5.0,
            'violation_count': 10,
            'anomaly_score': 0.7
        }
    
    async def process_quality_metrics(self, data_type: str, validation_result: ValidationResult, 
                                    anomaly_result: AnomalyResult, processing_time_ms: float) -> QualityMetrics:
        """Process and store quality metrics"""
        try:
            metrics = QualityMetrics(
                data_type=data_type,
                timestamp=datetime.utcnow(),
                total_records=validation_result.metrics.get('total_records', 0),
                quality_score=validation_result.quality_score,
                validation_passed=validation_result.is_valid,
                anomaly_detected=anomaly_result.is_anomaly,
                violation_count=len(validation_result.violations),
                warning_count=len(validation_result.warnings),
                null_percentage=validation_result.metrics.get('overall_completeness', 0),
                duplicate_percentage=validation_result.metrics.get('duplicate_percentage', 0),
                processing_time_ms=processing_time_ms
            )
            
            # Store in history
            self.quality_history[data_type].append(metrics)
            
            # Check for alerts
            await self._check_quality_alerts(metrics)
            
            logger.info(f"Processed quality metrics for {data_type}: score={metrics.quality_score:.3f}, "
                       f"records={metrics.total_records}, violations={metrics.violation_count}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error processing quality metrics: {e}")
            raise DataQualityError(f"Failed to process quality metrics: {e}")
    
    async def _check_quality_alerts(self, metrics: QualityMetrics) -> None:
        """Check if quality metrics trigger any alerts"""
        try:
            alerts = []
            
            # Quality score alert
            if metrics.quality_score < self.thresholds['quality_score']:
                alerts.append(self._create_alert(
                    data_type=metrics.data_type,
                    alert_type="low_quality_score",
                    severity="high" if metrics.quality_score < 0.5 else "medium",
                    message=f"Quality score {metrics.quality_score:.3f} below threshold {self.thresholds['quality_score']}",
                    metrics=asdict(metrics)
                ))
            
            # High null percentage alert
            if metrics.null_percentage > self.thresholds['null_percentage']:
                alerts.append(self._create_alert(
                    data_type=metrics.data_type,
                    alert_type="high_null_percentage",
                    severity="high" if metrics.null_percentage > 20 else "medium",
                    message=f"Null percentage {metrics.null_percentage:.1f}% above threshold {self.thresholds['null_percentage']}%",
                    metrics=asdict(metrics)
                ))
            
            # High duplicate percentage alert
            if metrics.duplicate_percentage > self.thresholds['duplicate_percentage']:
                alerts.append(self._create_alert(
                    data_type=metrics.data_type,
                    alert_type="high_duplicate_percentage",
                    severity="medium",
                    message=f"Duplicate percentage {metrics.duplicate_percentage:.1f}% above threshold {self.thresholds['duplicate_percentage']}%",
                    metrics=asdict(metrics)
                ))
            
            # High violation count alert
            if metrics.violation_count > self.thresholds['violation_count']:
                alerts.append(self._create_alert(
                    data_type=metrics.data_type,
                    alert_type="high_violation_count",
                    severity="high" if metrics.violation_count > 50 else "medium",
                    message=f"Violation count {metrics.violation_count} above threshold {self.thresholds['violation_count']}",
                    metrics=asdict(metrics)
                ))
            
            # Anomaly detection alert
            if metrics.anomaly_detected:
                alerts.append(self._create_alert(
                    data_type=metrics.data_type,
                    alert_type="anomaly_detected",
                    severity="high",
                    message="Anomalies detected in data",
                    metrics=asdict(metrics)
                ))
            
            # Store alerts
            for alert in alerts:
                self.active_alerts[alert.alert_id] = alert
                logger.warning(f"Quality alert generated: {alert.alert_type} for {alert.data_type}")
            
        except Exception as e:
            logger.error(f"Error checking quality alerts: {e}")
    
    def _create_alert(self, data_type: str, alert_type: str, severity: str, 
                     message: str, metrics: Dict[str, Any]) -> QualityAlert:
        """Create a new quality alert"""
        self.alert_counter += 1
        alert_id = f"{data_type}_{alert_type}_{self.alert_counter}_{int(datetime.utcnow().timestamp())}"
        
        return QualityAlert(
            alert_id=alert_id,
            data_type=data_type,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.utcnow(),
            metrics=metrics
        )
    
    async def get_quality_summary(self, data_type: str, hours: int = 24) -> Dict[str, Any]:
        """Get quality summary for a specific data type over time"""
        try:
            if data_type not in self.quality_history:
                return {'error': f'No quality history found for {data_type}'}
            
            # Filter metrics by time range
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_metrics = [m for m in self.quality_history[data_type] if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {'error': f'No recent quality metrics found for {data_type}'}
            
            # Calculate summary statistics
            quality_scores = [m.quality_score for m in recent_metrics]
            total_records = sum(m.total_records for m in recent_metrics)
            violation_counts = [m.violation_count for m in recent_metrics]
            null_percentages = [m.null_percentage for m in recent_metrics]
            
            summary = {
                'data_type': data_type,
                'time_range_hours': hours,
                'total_metrics': len(recent_metrics),
                'total_records_processed': total_records,
                'quality_score': {
                    'current': quality_scores[-1] if quality_scores else 0,
                    'average': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                    'min': min(quality_scores) if quality_scores else 0,
                    'max': max(quality_scores) if quality_scores else 0,
                    'trend': self._calculate_trend(quality_scores)
                },
                'violations': {
                    'total': sum(violation_counts),
                    'average_per_batch': sum(violation_counts) / len(violation_counts) if violation_counts else 0,
                    'max_per_batch': max(violation_counts) if violation_counts else 0
                },
                'data_completeness': {
                    'average_null_percentage': sum(null_percentages) / len(null_percentages) if null_percentages else 0,
                    'min_null_percentage': min(null_percentages) if null_percentages else 0,
                    'max_null_percentage': max(null_percentages) if null_percentages else 0
                },
                'alerts': {
                    'active_count': len([a for a in self.active_alerts.values() if a.data_type == data_type and not a.resolved]),
                    'total_generated': len([a for a in self.active_alerts.values() if a.data_type == data_type])
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting quality summary: {e}")
            return {'error': str(e)}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        recent_avg = sum(values[-3:]) / min(3, len(values))
        older_avg = sum(values[:-3]) / max(1, len(values) - 3) if len(values) > 3 else recent_avg
        
        if recent_avg > older_avg * 1.05:
            return "improving"
        elif recent_avg < older_avg * 0.95:
            return "declining"
        else:
            return "stable"
    
    async def get_active_alerts(self, data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts, optionally filtered by data type"""
        try:
            alerts = [a for a in self.active_alerts.values() if not a.resolved]
            
            if data_type:
                alerts = [a for a in alerts if a.data_type == data_type]
            
            return [asdict(alert) for alert in alerts]
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                logger.info(f"Alert {alert_id} resolved")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    async def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """Get data for quality dashboard"""
        try:
            dashboard_data = {
                'overview': {
                    'total_data_types': len(self.quality_history),
                    'total_alerts': len(self.active_alerts),
                    'active_alerts': len([a for a in self.active_alerts.values() if not a.resolved]),
                    'last_updated': datetime.utcnow().isoformat()
                },
                'data_types': {},
                'recent_alerts': []
            }
            
            # Get summary for each data type
            for data_type in self.quality_history.keys():
                summary = await self.get_quality_summary(data_type, hours=24)
                dashboard_data['data_types'][data_type] = summary
            
            # Get recent alerts (last 10)
            recent_alerts = sorted(
                [a for a in self.active_alerts.values() if not a.resolved],
                key=lambda x: x.timestamp,
                reverse=True
            )[:10]
            
            dashboard_data['recent_alerts'] = [asdict(alert) for alert in recent_alerts]
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}
    
    async def export_quality_report(self, data_type: str, hours: int = 24) -> Dict[str, Any]:
        """Export detailed quality report"""
        try:
            if data_type not in self.quality_history:
                return {'error': f'No quality history found for {data_type}'}
            
            # Get metrics for the time range
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            metrics = [m for m in self.quality_history[data_type] if m.timestamp >= cutoff_time]
            
            # Get alerts for the time range
            alerts = [a for a in self.active_alerts.values() 
                     if a.data_type == data_type and a.timestamp >= cutoff_time]
            
            report = {
                'report_info': {
                    'data_type': data_type,
                    'time_range_hours': hours,
                    'generated_at': datetime.utcnow().isoformat(),
                    'total_metrics': len(metrics),
                    'total_alerts': len(alerts)
                },
                'quality_metrics': [asdict(m) for m in metrics],
                'alerts': [asdict(a) for a in alerts],
                'summary': await self.get_quality_summary(data_type, hours)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error exporting quality report: {e}")
            return {'error': str(e)}
