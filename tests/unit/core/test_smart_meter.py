"""
Unit tests for SmartMeter domain entity
"""

import pytest
from datetime import datetime, timedelta
from src.core.domain.entities.smart_meter import SmartMeter, MeterReading
from src.core.domain.value_objects.meter_id import MeterId
from src.core.domain.value_objects.location import Location
from src.core.domain.value_objects.meter_specifications import MeterSpecifications
from src.core.domain.enums.meter_status import MeterStatus
from src.core.domain.enums.quality_tier import QualityTier
from src.core.exceptions.domain_exceptions import InvalidMeterOperationError


class TestSmartMeter:
    """Test cases for SmartMeter entity"""
    
    def test_create_smart_meter(self, sample_smart_meter_data):
        """Test creating a smart meter with valid data"""
        # Arrange
        meter_id = MeterId(sample_smart_meter_data["meter_id"])
        location = Location(
            latitude=sample_smart_meter_data["location"]["latitude"],
            longitude=sample_smart_meter_data["location"]["longitude"],
            address=sample_smart_meter_data["location"]["address"]
        )
        specifications = MeterSpecifications(
            manufacturer=sample_smart_meter_data["specifications"]["manufacturer"],
            model=sample_smart_meter_data["specifications"]["model"],
            firmware_version=sample_smart_meter_data["specifications"]["firmware_version"],
            installation_date=sample_smart_meter_data["specifications"]["installation_date"]
        )
        
        # Act
        meter = SmartMeter(
            meter_id=meter_id,
            location=location,
            specifications=specifications,
            status=MeterStatus.ACTIVE,
            quality_tier=QualityTier.EXCELLENT,
            metadata=sample_smart_meter_data["metadata"]
        )
        
        # Assert
        assert meter.meter_id == meter_id
        assert meter.location == location
        assert meter.specifications == specifications
        assert meter.status == MeterStatus.ACTIVE
        assert meter.quality_tier == QualityTier.EXCELLENT
        assert meter.metadata == sample_smart_meter_data["metadata"]
        assert meter.readings == []
        assert meter.created_at is not None
        assert meter.updated_at is not None
        assert meter.version == 1
    
    def test_add_meter_reading(self, sample_smart_meter_data, sample_meter_reading_data):
        """Test adding a meter reading to a smart meter"""
        # Arrange
        meter = self._create_test_meter(sample_smart_meter_data)
        reading_data = sample_meter_reading_data.copy()
        reading_data["meter_id"] = meter.meter_id.value
        
        # Act
        reading = MeterReading(**reading_data)
        meter.add_reading(reading)
        
        # Assert
        assert len(meter.readings) == 1
        assert meter.readings[0] == reading
        assert meter.last_reading_at == reading.timestamp
        assert meter.version == 2
    
    def test_add_meter_reading_updates_last_reading_time(self, sample_smart_meter_data, sample_meter_reading_data):
        """Test that adding a reading updates the last reading time"""
        # Arrange
        meter = self._create_test_meter(sample_smart_meter_data)
        reading_data = sample_meter_reading_data.copy()
        reading_data["meter_id"] = meter.meter_id.value
        
        # Act
        reading = MeterReading(**reading_data)
        meter.add_reading(reading)
        
        # Assert
        assert meter.last_reading_at == reading.timestamp
    
    def test_add_reading_to_inactive_meter_raises_error(self, sample_smart_meter_data, sample_meter_reading_data):
        """Test that adding a reading to an inactive meter raises an error"""
        # Arrange
        meter = self._create_test_meter(sample_smart_meter_data)
        meter.status = MeterStatus.INACTIVE
        reading_data = sample_meter_reading_data.copy()
        reading_data["meter_id"] = meter.meter_id.value
        
        # Act & Assert
        reading = MeterReading(**reading_data)
        with pytest.raises(InvalidMeterOperationError, match="Cannot add reading to inactive meter"):
            meter.add_reading(reading)
    
    def test_update_status(self, sample_smart_meter_data):
        """Test updating meter status"""
        # Arrange
        meter = self._create_test_meter(sample_smart_meter_data)
        original_version = meter.version
        
        # Act
        meter.update_status(MeterStatus.MAINTENANCE)
        
        # Assert
        assert meter.status == MeterStatus.MAINTENANCE
        assert meter.version == original_version + 1
        assert meter.updated_at > meter.created_at
    
    def test_update_quality_tier(self, sample_smart_meter_data):
        """Test updating meter quality tier"""
        # Arrange
        meter = self._create_test_meter(sample_smart_meter_data)
        original_version = meter.version
        
        # Act
        meter.update_quality_tier(QualityTier.GOOD)
        
        # Assert
        assert meter.quality_tier == QualityTier.GOOD
        assert meter.version == original_version + 1
        assert meter.updated_at > meter.created_at
    
    def test_update_metadata(self, sample_smart_meter_data):
        """Test updating meter metadata"""
        # Arrange
        meter = self._create_test_meter(sample_smart_meter_data)
        new_metadata = {"customer_id": "CUST002", "installation_type": "commercial"}
        original_version = meter.version
        
        # Act
        meter.update_metadata(new_metadata)
        
        # Assert
        assert meter.metadata == new_metadata
        assert meter.version == original_version + 1
        assert meter.updated_at > meter.created_at
    
    def test_get_readings_in_time_range(self, sample_smart_meter_data, sample_meter_reading_data):
        """Test getting readings within a specific time range"""
        # Arrange
        meter = self._create_test_meter(sample_smart_meter_data)
        base_time = datetime.utcnow()
        
        # Add readings at different times
        for i in range(5):
            reading_data = sample_meter_reading_data.copy()
            reading_data["meter_id"] = meter.meter_id.value
            reading_data["timestamp"] = base_time + timedelta(hours=i)
            reading = MeterReading(**reading_data)
            meter.add_reading(reading)
        
        # Act
        start_time = base_time + timedelta(hours=1)
        end_time = base_time + timedelta(hours=3)
        readings = meter.get_readings_in_time_range(start_time, end_time)
        
        # Assert
        assert len(readings) == 3
        for reading in readings:
            assert start_time <= reading.timestamp <= end_time
    
    def test_get_latest_reading(self, sample_smart_meter_data, sample_meter_reading_data):
        """Test getting the latest reading"""
        # Arrange
        meter = self._create_test_meter(sample_smart_meter_data)
        base_time = datetime.utcnow()
        
        # Add readings at different times
        for i in range(3):
            reading_data = sample_meter_reading_data.copy()
            reading_data["meter_id"] = meter.meter_id.value
            reading_data["timestamp"] = base_time + timedelta(hours=i)
            reading = MeterReading(**reading_data)
            meter.add_reading(reading)
        
        # Act
        latest_reading = meter.get_latest_reading()
        
        # Assert
        assert latest_reading is not None
        assert latest_reading.timestamp == base_time + timedelta(hours=2)
    
    def test_get_latest_reading_returns_none_when_no_readings(self, sample_smart_meter_data):
        """Test that get_latest_reading returns None when no readings exist"""
        # Arrange
        meter = self._create_test_meter(sample_smart_meter_data)
        
        # Act
        latest_reading = meter.get_latest_reading()
        
        # Assert
        assert latest_reading is None
    
    def test_calculate_average_consumption(self, sample_smart_meter_data, sample_meter_reading_data):
        """Test calculating average energy consumption"""
        # Arrange
        meter = self._create_test_meter(sample_smart_meter_data)
        base_time = datetime.utcnow()
        
        # Add readings with different consumption values
        consumptions = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, consumption in enumerate(consumptions):
            reading_data = sample_meter_reading_data.copy()
            reading_data["meter_id"] = meter.meter_id.value
            reading_data["timestamp"] = base_time + timedelta(hours=i)
            reading_data["energy_consumed_kwh"] = consumption
            reading = MeterReading(**reading_data)
            meter.add_reading(reading)
        
        # Act
        avg_consumption = meter.calculate_average_consumption()
        
        # Assert
        assert avg_consumption == 3.0  # (1+2+3+4+5)/5
    
    def test_calculate_average_consumption_returns_zero_when_no_readings(self, sample_smart_meter_data):
        """Test that calculate_average_consumption returns 0 when no readings exist"""
        # Arrange
        meter = self._create_test_meter(sample_smart_meter_data)
        
        # Act
        avg_consumption = meter.calculate_average_consumption()
        
        # Assert
        assert avg_consumption == 0.0
    
    def test_detect_anomalies(self, sample_smart_meter_data, sample_meter_reading_data):
        """Test detecting anomalous readings"""
        # Arrange
        meter = self._create_test_meter(sample_smart_meter_data)
        base_time = datetime.utcnow()
        
        # Add normal readings
        for i in range(5):
            reading_data = sample_meter_reading_data.copy()
            reading_data["meter_id"] = meter.meter_id.value
            reading_data["timestamp"] = base_time + timedelta(hours=i)
            reading_data["energy_consumed_kwh"] = 2.0  # Normal consumption
            reading = MeterReading(**reading_data)
            meter.add_reading(reading)
        
        # Add anomalous reading
        anomalous_reading_data = sample_meter_reading_data.copy()
        anomalous_reading_data["meter_id"] = meter.meter_id.value
        anomalous_reading_data["timestamp"] = base_time + timedelta(hours=5)
        anomalous_reading_data["energy_consumed_kwh"] = 100.0  # Anomalous consumption
        anomalous_reading = MeterReading(**anomalous_reading_data)
        meter.add_reading(anomalous_reading)
        
        # Act
        anomalies = meter.detect_anomalies(threshold=3.0)  # 3x standard deviation
        
        # Assert
        assert len(anomalies) == 1
        assert anomalies[0] == anomalous_reading
    
    def test_equality(self, sample_smart_meter_data):
        """Test meter equality based on meter_id"""
        # Arrange
        meter1 = self._create_test_meter(sample_smart_meter_data)
        meter2 = self._create_test_meter(sample_smart_meter_data)
        
        # Act & Assert
        assert meter1 == meter2  # Same meter_id
        assert hash(meter1) == hash(meter2)
    
    def test_inequality(self, sample_smart_meter_data):
        """Test meter inequality with different meter_ids"""
        # Arrange
        meter1 = self._create_test_meter(sample_smart_meter_data)
        sample_smart_meter_data["meter_id"] = "SM002"
        meter2 = self._create_test_meter(sample_smart_meter_data)
        
        # Act & Assert
        assert meter1 != meter2  # Different meter_id
        assert hash(meter1) != hash(meter2)
    
    def _create_test_meter(self, sample_data):
        """Helper method to create a test meter"""
        meter_id = MeterId(sample_data["meter_id"])
        location = Location(
            latitude=sample_data["location"]["latitude"],
            longitude=sample_data["location"]["longitude"],
            address=sample_data["location"]["address"]
        )
        specifications = MeterSpecifications(
            manufacturer=sample_data["specifications"]["manufacturer"],
            model=sample_data["specifications"]["model"],
            firmware_version=sample_data["specifications"]["firmware_version"],
            installation_date=sample_data["specifications"]["installation_date"]
        )
        
        return SmartMeter(
            meter_id=meter_id,
            location=location,
            specifications=specifications,
            status=MeterStatus.ACTIVE,
            quality_tier=QualityTier.EXCELLENT,
            metadata=sample_data["metadata"]
        )


class TestMeterReading:
    """Test cases for MeterReading value object"""
    
    def test_create_meter_reading(self, sample_meter_reading_data):
        """Test creating a meter reading with valid data"""
        # Act
        reading = MeterReading(**sample_meter_reading_data)
        
        # Assert
        assert reading.meter_id == sample_meter_reading_data["meter_id"]
        assert reading.timestamp == sample_meter_reading_data["timestamp"]
        assert reading.energy_consumed_kwh == sample_meter_reading_data["energy_consumed_kwh"]
        assert reading.power_factor == sample_meter_reading_data["power_factor"]
        assert reading.voltage_v == sample_meter_reading_data["voltage_v"]
        assert reading.current_a == sample_meter_reading_data["current_a"]
        assert reading.frequency_hz == sample_meter_reading_data["frequency_hz"]
        assert reading.temperature_c == sample_meter_reading_data["temperature_c"]
        assert reading.quality_score == sample_meter_reading_data["quality_score"]
        assert reading.anomaly_detected == sample_meter_reading_data["anomaly_detected"]
    
    def test_calculate_power_consumption(self, sample_meter_reading_data):
        """Test calculating power consumption from voltage and current"""
        # Arrange
        reading = MeterReading(**sample_meter_reading_data)
        expected_power = reading.voltage_v * reading.current_a * reading.power_factor
        
        # Act
        power = reading.calculate_power_consumption()
        
        # Assert
        assert power == expected_power
    
    def test_is_anomalous(self, sample_meter_reading_data):
        """Test checking if reading is anomalous"""
        # Arrange
        normal_reading = MeterReading(**sample_meter_reading_data)
        anomalous_reading_data = sample_meter_reading_data.copy()
        anomalous_reading_data["anomaly_detected"] = True
        anomalous_reading = MeterReading(**anomalous_reading_data)
        
        # Act & Assert
        assert not normal_reading.is_anomalous()
        assert anomalous_reading.is_anomalous()
    
    def test_equality(self, sample_meter_reading_data):
        """Test reading equality based on meter_id and timestamp"""
        # Arrange
        reading1 = MeterReading(**sample_meter_reading_data)
        reading2 = MeterReading(**sample_meter_reading_data)
        
        # Act & Assert
        assert reading1 == reading2
        assert hash(reading1) == hash(reading2)
    
    def test_inequality(self, sample_meter_reading_data):
        """Test reading inequality with different timestamps"""
        # Arrange
        reading1 = MeterReading(**sample_meter_reading_data)
        different_time_data = sample_meter_reading_data.copy()
        different_time_data["timestamp"] = sample_meter_reading_data["timestamp"] + timedelta(hours=1)
        reading2 = MeterReading(**different_time_data)
        
        # Act & Assert
        assert reading1 != reading2
        assert hash(reading1) != hash(reading2)
