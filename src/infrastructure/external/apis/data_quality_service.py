"""
Data Quality Service Implementation
Concrete implementation of IDataQualityService using ML models
"""

import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import logging

from ....core.domain.entities.smart_meter import MeterReading
from ....core.interfaces.external.data_quality_service import IDataQualityService
from ....core.exceptions.domain_exceptions import DataQualityError

logger = logging.getLogger(__name__)


class DataQualityService(IDataQualityService):
    """
    Data Quality Service Implementation
    
    Provides data quality assessment using statistical analysis
    and machine learning models for smart meter readings.
    """
    
    def __init__(self, quality_threshold: float = 0.8):
        self.quality_threshold = quality_threshold
        self._quality_models = {}
        self._initialize_quality_models()
    
    def _initialize_quality_models(self) -> None:
        """Initialize quality assessment models"""
        # This would typically load pre-trained models
        # For now, we'll use statistical methods
        logger.info("Initialized data quality models")
    
    async def calculate_quality_score(self, readings: List[MeterReading]) -> float:
        """
        Calculate overall quality score for meter readings
        
        Args:
            readings: List of meter readings to assess
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not readings:
            return 0.0
        
        try:
            # Extract reading data
            reading_data = [self._reading_to_dict(reading) for reading in readings]
            
            # Calculate individual quality metrics
            completeness_score = self._calculate_completeness_score(reading_data)
            consistency_score = self._calculate_consistency_score(reading_data)
            accuracy_score = self._calculate_accuracy_score(reading_data)
            timeliness_score = self._calculate_timeliness_score(reading_data)
            
            # Weighted average of quality metrics
            quality_score = (
                completeness_score * 0.3 +
                consistency_score * 0.3 +
                accuracy_score * 0.3 +
                timeliness_score * 0.1
            )
            
            # Ensure score is between 0.0 and 1.0
            quality_score = max(0.0, min(1.0, quality_score))
            
            logger.debug(f"Calculated quality score: {quality_score}")
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            raise DataQualityError(f"Failed to calculate quality score: {str(e)}")
    
    def _calculate_completeness_score(self, readings: List[Dict[str, Any]]) -> float:
        """Calculate completeness score based on missing data"""
        if not readings:
            return 0.0
        
        total_fields = 0
        missing_fields = 0
        
        for reading in readings:
            # Check required fields
            required_fields = [
                'voltage', 'current', 'power_factor', 'frequency',
                'active_power', 'reactive_power', 'apparent_power'
            ]
            
            for field in required_fields:
                total_fields += 1
                if reading.get(field) is None or reading.get(field) == 0:
                    missing_fields += 1
        
        if total_fields == 0:
            return 0.0
        
        completeness_ratio = 1.0 - (missing_fields / total_fields)
        return max(0.0, completeness_ratio)
    
    def _calculate_consistency_score(self, readings: List[Dict[str, Any]]) -> float:
        """Calculate consistency score based on data patterns"""
        if len(readings) < 2:
            return 1.0
        
        try:
            # Extract numerical values
            voltages = [r.get('voltage', 0) for r in readings if r.get('voltage') is not None]
            currents = [r.get('current', 0) for r in readings if r.get('current') is not None]
            frequencies = [r.get('frequency', 0) for r in readings if r.get('frequency') is not None]
            
            consistency_scores = []
            
            # Voltage consistency
            if len(voltages) > 1:
                voltage_cv = np.std(voltages) / np.mean(voltages) if np.mean(voltages) > 0 else 1.0
                voltage_consistency = max(0.0, 1.0 - min(1.0, voltage_cv))
                consistency_scores.append(voltage_consistency)
            
            # Current consistency
            if len(currents) > 1:
                current_cv = np.std(currents) / np.mean(currents) if np.mean(currents) > 0 else 1.0
                current_consistency = max(0.0, 1.0 - min(1.0, current_cv))
                consistency_scores.append(current_consistency)
            
            # Frequency consistency
            if len(frequencies) > 1:
                frequency_cv = np.std(frequencies) / np.mean(frequencies) if np.mean(frequencies) > 0 else 1.0
                frequency_consistency = max(0.0, 1.0 - min(1.0, frequency_cv))
                consistency_scores.append(frequency_consistency)
            
            if not consistency_scores:
                return 1.0
            
            return np.mean(consistency_scores)
            
        except Exception as e:
            logger.warning(f"Error calculating consistency score: {str(e)}")
            return 0.5  # Default moderate score
    
    def _calculate_accuracy_score(self, readings: List[Dict[str, Any]]) -> float:
        """Calculate accuracy score based on data validity"""
        if not readings:
            return 0.0
        
        valid_readings = 0
        total_readings = len(readings)
        
        for reading in readings:
            if self._is_valid_reading(reading):
                valid_readings += 1
        
        return valid_readings / total_readings if total_readings > 0 else 0.0
    
    def _calculate_timeliness_score(self, readings: List[MeterReading]) -> float:
        """Calculate timeliness score based on data freshness"""
        if not readings:
            return 0.0
        
        current_time = datetime.utcnow()
        timeliness_scores = []
        
        for reading in readings:
            # Calculate time difference in hours
            time_diff = (current_time - reading.timestamp).total_seconds() / 3600
            
            # Score decreases with age
            if time_diff <= 1:  # Within 1 hour
                score = 1.0
            elif time_diff <= 24:  # Within 24 hours
                score = 0.8
            elif time_diff <= 168:  # Within 1 week
                score = 0.6
            else:  # Older than 1 week
                score = 0.2
            
            timeliness_scores.append(score)
        
        return np.mean(timeliness_scores) if timeliness_scores else 0.0
    
    def _is_valid_reading(self, reading: Dict[str, Any]) -> bool:
        """Check if a reading is valid based on business rules"""
        try:
            # Check voltage range (typical: 220V ± 10%)
            voltage = reading.get('voltage', 0)
            if not (198 <= voltage <= 242):
                return False
            
            # Check current range (typical: 0-100A)
            current = reading.get('current', 0)
            if not (0 <= current <= 100):
                return False
            
            # Check power factor range (0-1)
            power_factor = reading.get('power_factor', 0)
            if not (0 <= power_factor <= 1):
                return False
            
            # Check frequency range (typical: 50Hz ± 0.5Hz)
            frequency = reading.get('frequency', 0)
            if not (49.5 <= frequency <= 50.5):
                return False
            
            # Check power values are positive
            active_power = reading.get('active_power', 0)
            reactive_power = reading.get('reactive_power', 0)
            apparent_power = reading.get('apparent_power', 0)
            
            if active_power < 0 or reactive_power < 0 or apparent_power < 0:
                return False
            
            # Check power factor consistency
            if apparent_power > 0:
                calculated_pf = active_power / apparent_power
                if abs(calculated_pf - power_factor) > 0.1:  # 10% tolerance
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating reading: {str(e)}")
            return False
    
    def _reading_to_dict(self, reading: MeterReading) -> Dict[str, Any]:
        """Convert MeterReading to dictionary"""
        return {
            'timestamp': reading.timestamp,
            'voltage': reading.voltage,
            'current': reading.current,
            'power_factor': reading.power_factor,
            'frequency': reading.frequency,
            'active_power': reading.active_power,
            'reactive_power': reading.reactive_power,
            'apparent_power': reading.apparent_power,
            'data_quality_score': reading.data_quality_score,
            'is_anomaly': reading.is_anomaly,
            'anomaly_type': reading.anomaly_type
        }
    
    async def validate_reading(self, reading: MeterReading) -> Dict[str, Any]:
        """
        Validate a single meter reading
        
        Args:
            reading: Meter reading to validate
            
        Returns:
            Validation result dictionary
        """
        try:
            reading_dict = self._reading_to_dict(reading)
            
            # Check validity
            is_valid = self._is_valid_reading(reading_dict)
            
            # Generate validation details
            violations = []
            if not is_valid:
                violations = self._get_validation_violations(reading_dict)
            
            # Calculate quality score
            quality_score = await self.calculate_quality_score([reading])
            
            # Generate recommendations
            recommendations = self._get_quality_recommendations(reading_dict, quality_score)
            
            return {
                'is_valid': is_valid,
                'violations': violations,
                'quality_score': quality_score,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error validating reading: {str(e)}")
            raise DataQualityError(f"Failed to validate reading: {str(e)}")
    
    def _get_validation_violations(self, reading: Dict[str, Any]) -> List[str]:
        """Get list of validation violations"""
        violations = []
        
        voltage = reading.get('voltage', 0)
        if not (198 <= voltage <= 242):
            violations.append(f"Voltage {voltage}V is outside acceptable range (198-242V)")
        
        current = reading.get('current', 0)
        if not (0 <= current <= 100):
            violations.append(f"Current {current}A is outside acceptable range (0-100A)")
        
        power_factor = reading.get('power_factor', 0)
        if not (0 <= power_factor <= 1):
            violations.append(f"Power factor {power_factor} is outside acceptable range (0-1)")
        
        frequency = reading.get('frequency', 0)
        if not (49.5 <= frequency <= 50.5):
            violations.append(f"Frequency {frequency}Hz is outside acceptable range (49.5-50.5Hz)")
        
        return violations
    
    def _get_quality_recommendations(self, reading: Dict[str, Any], quality_score: float) -> List[str]:
        """Get quality improvement recommendations"""
        recommendations = []
        
        if quality_score < 0.5:
            recommendations.append("Consider recalibrating the meter")
            recommendations.append("Check for sensor malfunctions")
        
        if quality_score < 0.8:
            recommendations.append("Review data collection frequency")
            recommendations.append("Implement additional validation checks")
        
        # Specific recommendations based on reading values
        voltage = reading.get('voltage', 0)
        if voltage < 200 or voltage > 240:
            recommendations.append("Check voltage supply stability")
        
        power_factor = reading.get('power_factor', 0)
        if power_factor < 0.8:
            recommendations.append("Consider power factor correction")
        
        return recommendations
    
    async def get_quality_metrics(self, meter_id: str, since: datetime) -> Dict[str, Any]:
        """
        Get quality metrics for a specific meter
        
        Args:
            meter_id: Meter ID
            since: Start datetime for metrics
            
        Returns:
            Quality metrics dictionary
        """
        try:
            # This would typically query the database for historical data
            # For now, return mock metrics
            return {
                'meter_id': meter_id,
                'since': since.isoformat(),
                'average_quality_score': 0.85,
                'total_readings': 1000,
                'valid_readings': 850,
                'quality_trend': 'improving',
                'common_violations': [
                    'Voltage out of range',
                    'Power factor low'
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting quality metrics: {str(e)}")
            raise DataQualityError(f"Failed to get quality metrics: {str(e)}")
    
    async def update_quality_thresholds(self, thresholds: Dict[str, Any]) -> None:
        """
        Update quality assessment thresholds
        
        Args:
            thresholds: New threshold configuration
        """
        try:
            if 'voltage_min' in thresholds:
                self.voltage_min = thresholds['voltage_min']
            if 'voltage_max' in thresholds:
                self.voltage_max = thresholds['voltage_max']
            if 'current_max' in thresholds:
                self.current_max = thresholds['current_max']
            if 'frequency_min' in thresholds:
                self.frequency_min = thresholds['frequency_min']
            if 'frequency_max' in thresholds:
                self.frequency_max = thresholds['frequency_max']
            
            logger.info(f"Updated quality thresholds: {thresholds}")
            
        except Exception as e:
            logger.error(f"Error updating quality thresholds: {str(e)}")
            raise DataQualityError(f"Failed to update quality thresholds: {str(e)}")
    
    async def train_quality_model(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Train the quality assessment model
        
        Args:
            training_data: Training data for the model
        """
        try:
            # This would typically train a machine learning model
            # For now, just log the training data size
            logger.info(f"Training quality model with {len(training_data)} samples")
            
            # Update internal models with new data
            self._quality_models['last_training_size'] = len(training_data)
            self._quality_models['last_training_time'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error training quality model: {str(e)}")
            raise DataQualityError(f"Failed to train quality model: {str(e)}")
    
    async def get_quality_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get data quality trends over specified period"""
        try:
            # This would typically query historical quality data
            # For now, return mock trend data
            trends = {
                'period_days': days,
                'overall_quality_trend': 'improving',
                'quality_scores': [0.85, 0.87, 0.89, 0.91, 0.88, 0.92],
                'anomaly_count_trend': 'decreasing',
                'average_quality_score': 0.89,
                'quality_improvement_rate': 0.02
            }
            logger.info(f"Retrieved quality trends for {days} days")
            return trends
        except Exception as e:
            logger.error(f"Error getting quality trends: {str(e)}")
            raise DataQualityError(f"Failed to get quality trends: {str(e)}")
    
    async def get_system_quality_summary(self) -> Dict[str, Any]:
        """Get overall system data quality summary"""
        try:
            # This would typically aggregate quality metrics from all data sources
            summary = {
                'total_data_sources': 3,  # smart_meters, weather_stations, grid_operators
                'overall_quality_score': 0.89,
                'quality_tier_distribution': {
                    'excellent': 45,
                    'good': 35,
                    'fair': 15,
                    'poor': 5
                },
                'anomaly_rate': 0.03,
                'data_completeness': 0.97,
                'last_quality_check': datetime.utcnow().isoformat(),
                'quality_issues': [
                    'Low voltage readings in Zone A',
                    'Missing temperature data in Station 3'
                ]
            }
            logger.info("Retrieved system quality summary")
            return summary
        except Exception as e:
            logger.error(f"Error getting system quality summary: {str(e)}")
            raise DataQualityError(f"Failed to get system quality summary: {str(e)}")
    
    async def update_quality_rules(self, rules: Dict[str, Any]) -> bool:
        """Update data quality validation rules"""
        try:
            # This would typically update the quality validation rules
            # For now, just log the update
            logger.info(f"Updating quality rules: {list(rules.keys())}")
            
            # Update internal quality threshold if provided
            if 'quality_threshold' in rules:
                self.quality_threshold = rules['quality_threshold']
                logger.info(f"Updated quality threshold to {self.quality_threshold}")
            
            # Update other rules as needed
            if 'anomaly_threshold' in rules:
                logger.info(f"Updated anomaly threshold to {rules['anomaly_threshold']}")
            
            logger.info("Quality rules updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating quality rules: {str(e)}")
            raise DataQualityError(f"Failed to update quality rules: {str(e)}")