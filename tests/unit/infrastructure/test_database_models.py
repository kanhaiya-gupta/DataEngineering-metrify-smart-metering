"""
Unit tests for database models
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.infrastructure.database.models.smart_meter_model import SmartMeterModel, MeterReadingModel, MeterEventModel
from src.infrastructure.database.models.grid_operator_model import GridOperatorModel, GridStatusModel, GridEventModel
from src.infrastructure.database.models.weather_station_model import WeatherStationModel, WeatherObservationModel, WeatherEventModel


@pytest.fixture
def test_engine():
    """Test database engine"""
    engine = create_engine("postgresql://postgres:postgres@localhost:5432/metrify_test", echo=True)
    return engine


@pytest.fixture
def test_session_factory(test_engine):
    """Test session factory"""
    return sessionmaker(bind=test_engine)


@pytest.fixture
def test_session(test_session_factory):
    """Test database session"""
    session = test_session_factory()
    yield session
    session.close()


class TestSmartMeterModel:
    """Test cases for SmartMeterModel"""
    
    def test_create_smart_meter_model(self, test_session):
        """Test creating a smart meter model"""
        # Arrange
        meter_data = {
            "meter_id": "SM001",
            "latitude": 52.5200,
            "longitude": 13.4050,
            "address": "Berlin, Germany",
            "manufacturer": "Siemens",
            "model": "SGM-1000",
            "firmware_version": "1.2.3",
            "installation_date": "2023-01-15",
            "status": "ACTIVE",
            "quality_tier": "EXCELLENT",
            "metadata": {"customer_id": "CUST001"},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "version": 1
        }
        
        # Act
        meter = SmartMeterModel(**meter_data)
        test_session.add(meter)
        test_session.commit()
        
        # Assert
        assert meter.meter_id == "SM001"
        assert meter.latitude == 52.5200
        assert meter.longitude == 13.4050
        assert meter.address == "Berlin, Germany"
        assert meter.manufacturer == "Siemens"
        assert meter.model == "SGM-1000"
        assert meter.firmware_version == "1.2.3"
        assert meter.installation_date == "2023-01-15"
        assert meter.status == "ACTIVE"
        assert meter.quality_tier == "EXCELLENT"
        assert meter.metadata == {"customer_id": "CUST001"}
        assert meter.version == 1
    
    def test_smart_meter_model_relationships(self, test_session):
        """Test smart meter model relationships"""
        # Arrange
        meter = SmartMeterModel(
            meter_id="SM001",
            latitude=52.5200,
            longitude=13.4050,
            address="Berlin, Germany",
            manufacturer="Siemens",
            model="SGM-1000",
            firmware_version="1.2.3",
            installation_date="2023-01-15",
            status="ACTIVE",
            quality_tier="EXCELLENT",
            metadata={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1
        )
        test_session.add(meter)
        test_session.commit()
        
        # Add readings
        reading1 = MeterReadingModel(
            meter_id="SM001",
            timestamp=datetime.utcnow(),
            energy_consumed_kwh=1.5,
            power_factor=0.95,
            voltage_v=230.0,
            current_a=6.5,
            frequency_hz=50.0,
            temperature_c=25.0,
            quality_score=0.95,
            anomaly_detected=False
        )
        
        reading2 = MeterReadingModel(
            meter_id="SM001",
            timestamp=datetime.utcnow() + timedelta(hours=1),
            energy_consumed_kwh=2.0,
            power_factor=0.96,
            voltage_v=231.0,
            current_a=7.0,
            frequency_hz=50.1,
            temperature_c=26.0,
            quality_score=0.96,
            anomaly_detected=False
        )
        
        test_session.add(reading1)
        test_session.add(reading2)
        test_session.commit()
        
        # Act
        meter_readings = test_session.query(MeterReadingModel).filter_by(meter_id="SM001").all()
        
        # Assert
        assert len(meter_readings) == 2
        assert meter_readings[0].energy_consumed_kwh == 1.5
        assert meter_readings[1].energy_consumed_kwh == 2.0


class TestMeterReadingModel:
    """Test cases for MeterReadingModel"""
    
    def test_create_meter_reading_model(self, test_session):
        """Test creating a meter reading model"""
        # Arrange
        reading_data = {
            "meter_id": "SM001",
            "timestamp": datetime.utcnow(),
            "energy_consumed_kwh": 1.5,
            "power_factor": 0.95,
            "voltage_v": 230.0,
            "current_a": 6.5,
            "frequency_hz": 50.0,
            "temperature_c": 25.0,
            "quality_score": 0.95,
            "anomaly_detected": False
        }
        
        # Act
        reading = MeterReadingModel(**reading_data)
        test_session.add(reading)
        test_session.commit()
        
        # Assert
        assert reading.meter_id == "SM001"
        assert reading.energy_consumed_kwh == 1.5
        assert reading.power_factor == 0.95
        assert reading.voltage_v == 230.0
        assert reading.current_a == 6.5
        assert reading.frequency_hz == 50.0
        assert reading.temperature_c == 25.0
        assert reading.quality_score == 0.95
        assert reading.anomaly_detected == False
    
    def test_meter_reading_model_calculations(self, test_session):
        """Test meter reading model calculations"""
        # Arrange
        reading = MeterReadingModel(
            meter_id="SM001",
            timestamp=datetime.utcnow(),
            energy_consumed_kwh=1.5,
            power_factor=0.95,
            voltage_v=230.0,
            current_a=6.5,
            frequency_hz=50.0,
            temperature_c=25.0,
            quality_score=0.95,
            anomaly_detected=False
        )
        
        # Act
        power_consumption = reading.voltage_v * reading.current_a * reading.power_factor
        
        # Assert
        expected_power = 230.0 * 6.5 * 0.95
        assert power_consumption == expected_power


class TestGridOperatorModel:
    """Test cases for GridOperatorModel"""
    
    def test_create_grid_operator_model(self, test_session):
        """Test creating a grid operator model"""
        # Arrange
        operator_data = {
            "operator_id": "TENNET",
            "name": "TenneT TSO B.V.",
            "country": "Netherlands",
            "status": "ACTIVE",
            "contact_email": "info@tennet.eu",
            "contact_phone": "+31 26 373 1000",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "version": 1
        }
        
        # Act
        operator = GridOperatorModel(**operator_data)
        test_session.add(operator)
        test_session.commit()
        
        # Assert
        assert operator.operator_id == "TENNET"
        assert operator.name == "TenneT TSO B.V."
        assert operator.country == "Netherlands"
        assert operator.status == "ACTIVE"
        assert operator.contact_email == "info@tennet.eu"
        assert operator.contact_phone == "+31 26 373 1000"
        assert operator.version == 1


class TestWeatherStationModel:
    """Test cases for WeatherStationModel"""
    
    def test_create_weather_station_model(self, test_session):
        """Test creating a weather station model"""
        # Arrange
        station_data = {
            "station_id": "WS001",
            "name": "Berlin Weather Station",
            "latitude": 52.5200,
            "longitude": 13.4050,
            "address": "Berlin, Germany",
            "station_type": "AUTOMATIC",
            "status": "ACTIVE",
            "elevation_m": 34.0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "version": 1
        }
        
        # Act
        station = WeatherStationModel(**station_data)
        test_session.add(station)
        test_session.commit()
        
        # Assert
        assert station.station_id == "WS001"
        assert station.name == "Berlin Weather Station"
        assert station.latitude == 52.5200
        assert station.longitude == 13.4050
        assert station.address == "Berlin, Germany"
        assert station.station_type == "AUTOMATIC"
        assert station.status == "ACTIVE"
        assert station.elevation_m == 34.0
        assert station.version == 1


class TestWeatherObservationModel:
    """Test cases for WeatherObservationModel"""
    
    def test_create_weather_observation_model(self, test_session):
        """Test creating a weather observation model"""
        # Arrange
        observation_data = {
            "station_id": "WS001",
            "timestamp": datetime.utcnow(),
            "temperature_c": 15.5,
            "humidity_percent": 65.0,
            "pressure_hpa": 1013.25,
            "wind_speed_ms": 3.2,
            "wind_direction_deg": 180.0,
            "precipitation_mm": 0.0,
            "cloud_cover_percent": 30.0,
            "visibility_km": 10.0
        }
        
        # Act
        observation = WeatherObservationModel(**observation_data)
        test_session.add(observation)
        test_session.commit()
        
        # Assert
        assert observation.station_id == "WS001"
        assert observation.temperature_c == 15.5
        assert observation.humidity_percent == 65.0
        assert observation.pressure_hpa == 1013.25
        assert observation.wind_speed_ms == 3.2
        assert observation.wind_direction_deg == 180.0
        assert observation.precipitation_mm == 0.0
        assert observation.cloud_cover_percent == 30.0
        assert observation.visibility_km == 10.0
