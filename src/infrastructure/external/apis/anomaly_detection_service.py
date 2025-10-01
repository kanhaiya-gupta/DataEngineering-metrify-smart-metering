"""
Anomaly Detection Service Implementation
Concrete implementation of IAnomalyDetectionService using ML models
"""

import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

from ....core.domain.entities.smart_meter import SmartMeter, MeterReading
from ....core.interfaces.external.anomaly_detection_service import IAnomalyDetectionService
from ....core.exceptions.domain_exceptions import DataQualityError

logger = logging.getLogger(__name__)


class AnomalyDetectionService(IAnomalyDetectionService):
    """
    Anomaly Detection Service Implementation
    
    Provides anomaly detection using statistical methods and
    machine learning models for smart meter readings.
    """
    
    def __init__(self, anomaly_threshold: float = 0.7):
        self.anomaly_threshold = anomaly_threshold
        self._anomaly_models = {}
        self._historical_data = {}
        self._initialize_anomaly_models()
    
    def _initialize_anomaly_models(self) -> None:
        """Initialize anomaly detection models"""
        # This would typically load pre-trained models
        # For now, we'll use statistical methods
        logger.info("Initialized anomaly detection models")
    
    async def detect_anomalies(self, reading: MeterReading, meter: SmartMeter) -> List[Dict[str, Any]]:
        """
        Detect anomalies in a meter reading
        
        Args:
            reading: Meter reading to analyze
            meter: Smart meter that generated the reading
            
        Returns:
            List of detected anomalies
        """
        try:
            anomalies = []
            
            # Detect different types of anomalies
            voltage_anomalies = await self.detect_voltage_anomalies(reading, meter)
            current_anomalies = await self.detect_current_anomalies(reading, meter)
            power_factor_anomalies = await self.detect_power_factor_anomalies(reading, meter)
            frequency_anomalies = await self.detect_frequency_anomalies(reading, meter)
            consumption_anomalies = await self.detect_consumption_anomalies(reading, meter)
            
            # Combine all anomalies
            anomalies.extend(voltage_anomalies)
            anomalies.extend(current_anomalies)
            anomalies.extend(power_factor_anomalies)
            anomalies.extend(frequency_anomalies)
            anomalies.extend(consumption_anomalies)
            
            # Update historical data for future analysis
            await self._update_historical_data(meter.meter_id.value, reading)
            
            logger.debug(f"Detected {len(anomalies)} anomalies for meter {meter.meter_id.value}")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            raise DataQualityError(f"Failed to detect anomalies: {str(e)}")
    
    async def detect_voltage_anomalies(self, reading: MeterReading, meter: SmartMeter) -> List[Dict[str, Any]]:
        """Detect voltage-related anomalies"""
        anomalies = []
        
        try:
            voltage = reading.voltage
            
            # Check for voltage spikes or drops
            if voltage > 250:  # High voltage
                anomalies.append({
                    'type': 'voltage_spike',
                    'description': f'Voltage spike detected: {voltage}V',
                    'severity': 'high',
                    'confidence': 0.9,
                    'data': {'voltage': voltage, 'threshold': 250}
                })
            elif voltage < 200:  # Low voltage
                anomalies.append({
                    'type': 'voltage_drop',
                    'description': f'Voltage drop detected: {voltage}V',
                    'severity': 'medium',
                    'confidence': 0.8,
                    'data': {'voltage': voltage, 'threshold': 200}
                })
            
            # Check for voltage instability
            if meter.meter_id.value in self._historical_data:
                historical_voltages = self._historical_data[meter.meter_id.value].get('voltages', [])
                if len(historical_voltages) > 5:
                    voltage_std = np.std(historical_voltages[-10:])  # Last 10 readings
                    if voltage_std > 10:  # High voltage variation
                        anomalies.append({
                            'type': 'voltage_instability',
                            'description': f'Voltage instability detected: std={voltage_std:.2f}V',
                            'severity': 'medium',
                            'confidence': 0.7,
                            'data': {'voltage_std': voltage_std, 'threshold': 10}
                        })
            
        except Exception as e:
            logger.warning(f"Error detecting voltage anomalies: {str(e)}")
        
        return anomalies
    
    async def detect_current_anomalies(self, reading: MeterReading, meter: SmartMeter) -> List[Dict[str, Any]]:
        """Detect current-related anomalies"""
        anomalies = []
        
        try:
            current = reading.current
            
            # Check for current spikes
            if current > 80:  # High current
                anomalies.append({
                    'type': 'current_spike',
                    'description': f'Current spike detected: {current}A',
                    'severity': 'high',
                    'confidence': 0.9,
                    'data': {'current': current, 'threshold': 80}
                })
            
            # Check for current patterns
            if meter.meter_id.value in self._historical_data:
                historical_currents = self._historical_data[meter.meter_id.value].get('currents', [])
                if len(historical_currents) > 5:
                    current_trend = np.polyfit(range(len(historical_currents[-10:])), historical_currents[-10:], 1)[0]
                    if abs(current_trend) > 5:  # Rapid current change
                        anomalies.append({
                            'type': 'current_trend',
                            'description': f'Rapid current change detected: trend={current_trend:.2f}A/h',
                            'severity': 'medium',
                            'confidence': 0.6,
                            'data': {'current_trend': current_trend, 'threshold': 5}
                        })
            
        except Exception as e:
            logger.warning(f"Error detecting current anomalies: {str(e)}")
        
        return anomalies
    
    async def detect_power_factor_anomalies(self, reading: MeterReading, meter: SmartMeter) -> List[Dict[str, Any]]:
        """Detect power factor anomalies"""
        anomalies = []
        
        try:
            power_factor = reading.power_factor
            
            # Check for low power factor
            if power_factor < 0.7:  # Low power factor
                anomalies.append({
                    'type': 'low_power_factor',
                    'description': f'Low power factor detected: {power_factor:.3f}',
                    'severity': 'medium',
                    'confidence': 0.8,
                    'data': {'power_factor': power_factor, 'threshold': 0.7}
                })
            
            # Check for power factor consistency
            active_power = reading.active_power
            apparent_power = reading.apparent_power
            
            if apparent_power > 0:
                calculated_pf = active_power / apparent_power
                pf_difference = abs(calculated_pf - power_factor)
                
                if pf_difference > 0.1:  # Inconsistent power factor
                    anomalies.append({
                        'type': 'power_factor_inconsistency',
                        'description': f'Power factor inconsistency: diff={pf_difference:.3f}',
                        'severity': 'low',
                        'confidence': 0.7,
                        'data': {
                            'calculated_pf': calculated_pf,
                            'reported_pf': power_factor,
                            'difference': pf_difference
                        }
                    })
            
        except Exception as e:
            logger.warning(f"Error detecting power factor anomalies: {str(e)}")
        
        return anomalies
    
    async def detect_frequency_anomalies(self, reading: MeterReading, meter: SmartMeter) -> List[Dict[str, Any]]:
        """Detect frequency anomalies"""
        anomalies = []
        
        try:
            frequency = reading.frequency
            
            # Check for frequency deviation
            if frequency < 49.5 or frequency > 50.5:  # Frequency out of range
                severity = 'high' if frequency < 49.0 or frequency > 51.0 else 'medium'
                anomalies.append({
                    'type': 'frequency_deviation',
                    'description': f'Frequency deviation detected: {frequency}Hz',
                    'severity': severity,
                    'confidence': 0.9,
                    'data': {'frequency': frequency, 'range': [49.5, 50.5]}
                })
            
        except Exception as e:
            logger.warning(f"Error detecting frequency anomalies: {str(e)}")
        
        return anomalies
    
    async def detect_consumption_anomalies(self, reading: MeterReading, meter: SmartMeter) -> List[Dict[str, Any]]:
        """Detect consumption pattern anomalies"""
        anomalies = []
        
        try:
            active_power = reading.active_power
            
            # Check for consumption spikes
            if meter.meter_id.value in self._historical_data:
                historical_power = self._historical_data[meter.meter_id.value].get('active_power', [])
                if len(historical_power) > 5:
                    avg_power = np.mean(historical_power[-10:])  # Last 10 readings
                    power_std = np.std(historical_power[-10:])
                    
                    # Check for significant deviation from average
                    if power_std > 0:
                        z_score = abs(active_power - avg_power) / power_std
                        if z_score > 3:  # 3-sigma rule
                            anomalies.append({
                                'type': 'consumption_spike',
                                'description': f'Consumption spike detected: {active_power}W (z-score: {z_score:.2f})',
                                'severity': 'high' if z_score > 5 else 'medium',
                                'confidence': min(0.9, z_score / 5),
                                'data': {
                                    'active_power': active_power,
                                    'average_power': avg_power,
                                    'z_score': z_score
                                }
                            })
            
        except Exception as e:
            logger.warning(f"Error detecting consumption anomalies: {str(e)}")
        
        return anomalies
    
    async def get_anomaly_history(self, meter_id: str, since: datetime) -> List[Dict[str, Any]]:
        """Get anomaly history for a meter"""
        try:
            # This would typically query the database for historical anomalies
            # For now, return mock data
            return [
                {
                    'meter_id': meter_id,
                    'anomaly_type': 'voltage_spike',
                    'detected_at': (since + timedelta(hours=1)).isoformat(),
                    'severity': 'high',
                    'description': 'Voltage spike detected: 255V'
                },
                {
                    'meter_id': meter_id,
                    'anomaly_type': 'current_spike',
                    'detected_at': (since + timedelta(hours=2)).isoformat(),
                    'severity': 'medium',
                    'description': 'Current spike detected: 85A'
                }
            ]
            
        except Exception as e:
            logger.error(f"Error getting anomaly history: {str(e)}")
            raise DataQualityError(f"Failed to get anomaly history: {str(e)}")
    
    async def get_anomaly_statistics(self, meter_id: str, days: int) -> Dict[str, Any]:
        """Get anomaly statistics for a meter"""
        try:
            # This would typically calculate statistics from historical data
            # For now, return mock statistics
            return {
                'meter_id': meter_id,
                'period_days': days,
                'total_anomalies': 15,
                'anomalies_by_type': {
                    'voltage_spike': 5,
                    'current_spike': 3,
                    'low_power_factor': 4,
                    'frequency_deviation': 2,
                    'consumption_spike': 1
                },
                'anomalies_by_severity': {
                    'high': 8,
                    'medium': 5,
                    'low': 2
                },
                'average_confidence': 0.75,
                'anomaly_rate': 0.05  # 5% of readings are anomalous
            }
            
        except Exception as e:
            logger.error(f"Error getting anomaly statistics: {str(e)}")
            raise DataQualityError(f"Failed to get anomaly statistics: {str(e)}")
    
    async def update_anomaly_thresholds(self, thresholds: Dict[str, Any]) -> None:
        """Update anomaly detection thresholds"""
        try:
            if 'voltage_max' in thresholds:
                self.voltage_max_threshold = thresholds['voltage_max']
            if 'current_max' in thresholds:
                self.current_max_threshold = thresholds['current_max']
            if 'power_factor_min' in thresholds:
                self.power_factor_min_threshold = thresholds['power_factor_min']
            if 'frequency_min' in thresholds:
                self.frequency_min_threshold = thresholds['frequency_min']
            if 'frequency_max' in thresholds:
                self.frequency_max_threshold = thresholds['frequency_max']
            if 'z_score_threshold' in thresholds:
                self.z_score_threshold = thresholds['z_score_threshold']
            
            logger.info(f"Updated anomaly thresholds: {thresholds}")
            
        except Exception as e:
            logger.error(f"Error updating anomaly thresholds: {str(e)}")
            raise DataQualityError(f"Failed to update anomaly thresholds: {str(e)}")
    
    async def train_anomaly_model(self, training_data: List[Dict[str, Any]]) -> None:
        """Train the anomaly detection model"""
        try:
            # This would typically train a machine learning model
            # For now, just log the training data size
            logger.info(f"Training anomaly model with {len(training_data)} samples")
            
            # Update internal models with new data
            self._anomaly_models['last_training_size'] = len(training_data)
            self._anomaly_models['last_training_time'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error training anomaly model: {str(e)}")
            raise DataQualityError(f"Failed to train anomaly model: {str(e)}")
    
    async def _update_historical_data(self, meter_id: str, reading: MeterReading) -> None:
        """Update historical data for anomaly detection"""
        if meter_id not in self._historical_data:
            self._historical_data[meter_id] = {
                'voltages': [],
                'currents': [],
                'active_power': [],
                'timestamps': []
            }
        
        # Add new reading data
        self._historical_data[meter_id]['voltages'].append(reading.voltage)
        self._historical_data[meter_id]['currents'].append(reading.current)
        self._historical_data[meter_id]['active_power'].append(reading.active_power)
        self._historical_data[meter_id]['timestamps'].append(reading.timestamp)
        
        # Keep only last 100 readings to prevent memory issues
        max_readings = 100
        for key in ['voltages', 'currents', 'active_power', 'timestamps']:
            if len(self._historical_data[meter_id][key]) > max_readings:
                self._historical_data[meter_id][key] = self._historical_data[meter_id][key][-max_readings:]
