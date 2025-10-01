"""
Integration tests for SmartMeterRepository
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.infrastructure.database.config import DatabaseConfig
from src.infrastructure.database.repositories.smart_meter_repository import SmartMeterRepository
from src.core.domain.entities.smart_meter import SmartMeter, MeterReading
from src.core.domain.value_objects.meter_id import MeterId
from src.core.domain.value_objects.location import Location
from src.core.domain.value_objects.meter_specifications import MeterSpecifications
from src.core.domain.enums.meter_status import MeterStatus
from src.core.domain.enums.quality_tier import QualityTier


@pytest.fixture
def test_database_url():
    """Test database URL"""
    return "postgresql://postgres:postgres@localhost:5432/metrify_test"


@pytest.fixture
def test_database_config(test_database_url):
    """Test database configuration"""
    return DatabaseConfig(
        host="localhost",
        port=5432,
        name="test_metrify",
        username="test",
        password="test",
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,
        echo=False
    )


@pytest.fixture
def test_engine(test_database_url):
    """Test database engine"""
    engine = create_engine(test_database_url, echo=True)
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


@pytest.fixture
def repository(test_database_config, test_session):
    """SmartMeterRepository instance for testing"""
    return SmartMeterRepository(test_database_config, test_session)


@pytest.fixture
def sample_meter():
    """Sample smart meter for testing"""
    meter_id = MeterId("SM001")
    location = Location(
        latitude=52.5200,
        longitude=13.4050,
        address="Berlin, Germany"
    )
    specifications = MeterSpecifications(
        manufacturer="Siemens",
        model="SGM-1000",
        firmware_version="1.2.3",
        installation_date="2023-01-15"
    )
    
    return SmartMeter(
        meter_id=meter_id,
        location=location,
        specifications=specifications,
        status=MeterStatus.ACTIVE,
        quality_tier=QualityTier.EXCELLENT,
        metadata={"customer_id": "CUST001"}
    )


@pytest.fixture
def sample_reading():
    """Sample meter reading for testing"""
    return MeterReading(
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


@pytest.mark.integration
@pytest.mark.database
class TestSmartMeterRepository:
    """Integration tests for SmartMeterRepository"""
    
    @pytest.mark.asyncio
    async def test_save_meter(self, repository, sample_meter, test_session):
        """Test saving a smart meter to database"""
        # Act
        saved_meter = await repository.save(sample_meter)
        
        # Assert
        assert saved_meter is not None
        assert saved_meter.meter_id == sample_meter.meter_id
        assert saved_meter.location == sample_meter.location
        assert saved_meter.specifications == sample_meter.specifications
        assert saved_meter.status == sample_meter.status
        assert saved_meter.quality_tier == sample_meter.quality_tier
        assert saved_meter.metadata == sample_meter.metadata
        
        # Verify in database
        db_meter = await repository.find_by_id(sample_meter.meter_id.value)
        assert db_meter is not None
        assert db_meter.meter_id == sample_meter.meter_id
    
    @pytest.mark.asyncio
    async def test_find_by_id(self, repository, sample_meter, test_session):
        """Test finding a meter by ID"""
        # Arrange
        await repository.save(sample_meter)
        
        # Act
        found_meter = await repository.find_by_id(sample_meter.meter_id.value)
        
        # Assert
        assert found_meter is not None
        assert found_meter.meter_id == sample_meter.meter_id
        assert found_meter.location == sample_meter.location
        assert found_meter.specifications == sample_meter.specifications
    
    @pytest.mark.asyncio
    async def test_find_by_id_not_found(self, repository, test_session):
        """Test finding a meter by ID when not found"""
        # Act
        found_meter = await repository.find_by_id("NONEXISTENT")
        
        # Assert
        assert found_meter is None
    
    @pytest.mark.asyncio
    async def test_find_all(self, repository, test_session):
        """Test finding all meters"""
        # Arrange
        meter1 = sample_meter
        meter2 = SmartMeter(
            meter_id=MeterId("SM002"),
            location=Location(52.5200, 13.4050, "Berlin, Germany"),
            specifications=MeterSpecifications("ABB", "ABM-2000", "2.1.0", "2023-02-15"),
            status=MeterStatus.ACTIVE,
            quality_tier=QualityTier.GOOD,
            metadata={"customer_id": "CUST002"}
        )
        
        await repository.save(meter1)
        await repository.save(meter2)
        
        # Act
        all_meters = await repository.find_all()
        
        # Assert
        assert len(all_meters) == 2
        meter_ids = [meter.meter_id.value for meter in all_meters]
        assert "SM001" in meter_ids
        assert "SM002" in meter_ids
    
    @pytest.mark.asyncio
    async def test_find_by_status(self, repository, test_session):
        """Test finding meters by status"""
        # Arrange
        active_meter = sample_meter
        inactive_meter = SmartMeter(
            meter_id=MeterId("SM002"),
            location=Location(52.5200, 13.4050, "Berlin, Germany"),
            specifications=MeterSpecifications("ABB", "ABM-2000", "2.1.0", "2023-02-15"),
            status=MeterStatus.INACTIVE,
            quality_tier=QualityTier.GOOD,
            metadata={"customer_id": "CUST002"}
        )
        
        await repository.save(active_meter)
        await repository.save(inactive_meter)
        
        # Act
        active_meters = await repository.find_by_status(MeterStatus.ACTIVE)
        inactive_meters = await repository.find_by_status(MeterStatus.INACTIVE)
        
        # Assert
        assert len(active_meters) == 1
        assert len(inactive_meters) == 1
        assert active_meters[0].meter_id == active_meter.meter_id
        assert inactive_meters[0].meter_id == inactive_meter.meter_id
    
    @pytest.mark.asyncio
    async def test_find_by_location(self, repository, test_session):
        """Test finding meters by location"""
        # Arrange
        berlin_meter = sample_meter
        munich_meter = SmartMeter(
            meter_id=MeterId("SM002"),
            location=Location(48.1351, 11.5820, "Munich, Germany"),
            specifications=MeterSpecifications("ABB", "ABM-2000", "2.1.0", "2023-02-15"),
            status=MeterStatus.ACTIVE,
            quality_tier=QualityTier.GOOD,
            metadata={"customer_id": "CUST002"}
        )
        
        await repository.save(berlin_meter)
        await repository.save(munich_meter)
        
        # Act
        berlin_meters = await repository.find_by_location("Berlin")
        munich_meters = await repository.find_by_location("Munich")
        
        # Assert
        assert len(berlin_meters) == 1
        assert len(munich_meters) == 1
        assert berlin_meters[0].meter_id == berlin_meter.meter_id
        assert munich_meters[0].meter_id == munich_meter.meter_id
    
    @pytest.mark.asyncio
    async def test_update_meter(self, repository, sample_meter, test_session):
        """Test updating a meter"""
        # Arrange
        await repository.save(sample_meter)
        
        # Act
        sample_meter.update_status(MeterStatus.MAINTENANCE)
        sample_meter.update_quality_tier(QualityTier.GOOD)
        updated_meter = await repository.save(sample_meter)
        
        # Assert
        assert updated_meter.status == MeterStatus.MAINTENANCE
        assert updated_meter.quality_tier == QualityTier.GOOD
        assert updated_meter.version == 3  # Initial + 2 updates
        
        # Verify in database
        db_meter = await repository.find_by_id(sample_meter.meter_id.value)
        assert db_meter.status == MeterStatus.MAINTENANCE
        assert db_meter.quality_tier == QualityTier.GOOD
    
    @pytest.mark.asyncio
    async def test_delete_meter(self, repository, sample_meter, test_session):
        """Test deleting a meter"""
        # Arrange
        await repository.save(sample_meter)
        
        # Act
        await repository.delete(sample_meter.meter_id.value)
        
        # Assert
        deleted_meter = await repository.find_by_id(sample_meter.meter_id.value)
        assert deleted_meter is None
    
    @pytest.mark.asyncio
    async def test_add_reading(self, repository, sample_meter, sample_reading, test_session):
        """Test adding a reading to a meter"""
        # Arrange
        await repository.save(sample_meter)
        
        # Act
        sample_meter.add_reading(sample_reading)
        await repository.save(sample_meter)
        
        # Assert
        updated_meter = await repository.find_by_id(sample_meter.meter_id.value)
        assert len(updated_meter.readings) == 1
        assert updated_meter.readings[0].meter_id == sample_reading.meter_id
        assert updated_meter.readings[0].energy_consumed_kwh == sample_reading.energy_consumed_kwh
    
    @pytest.mark.asyncio
    async def test_get_readings_in_time_range(self, repository, sample_meter, test_session):
        """Test getting readings in a time range"""
        # Arrange
        await repository.save(sample_meter)
        
        base_time = datetime.utcnow()
        readings = []
        
        for i in range(5):
            reading = MeterReading(
                meter_id=sample_meter.meter_id.value,
                timestamp=base_time + timedelta(hours=i),
                energy_consumed_kwh=1.0 + i,
                power_factor=0.95,
                voltage_v=230.0,
                current_a=6.5,
                frequency_hz=50.0,
                temperature_c=25.0,
                quality_score=0.95,
                anomaly_detected=False
            )
            readings.append(reading)
            sample_meter.add_reading(reading)
        
        await repository.save(sample_meter)
        
        # Act
        start_time = base_time + timedelta(hours=1)
        end_time = base_time + timedelta(hours=3)
        found_readings = await repository.get_readings_in_time_range(
            sample_meter.meter_id.value, start_time, end_time
        )
        
        # Assert
        assert len(found_readings) == 3
        for reading in found_readings:
            assert start_time <= reading.timestamp <= end_time
    
    @pytest.mark.asyncio
    async def test_count_meters(self, repository, test_session):
        """Test counting meters"""
        # Arrange
        meter1 = sample_meter
        meter2 = SmartMeter(
            meter_id=MeterId("SM002"),
            location=Location(52.5200, 13.4050, "Berlin, Germany"),
            specifications=MeterSpecifications("ABB", "ABM-2000", "2.1.0", "2023-02-15"),
            status=MeterStatus.ACTIVE,
            quality_tier=QualityTier.GOOD,
            metadata={"customer_id": "CUST002"}
        )
        
        await repository.save(meter1)
        await repository.save(meter2)
        
        # Act
        count = await repository.count()
        
        # Assert
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_find_with_pagination(self, repository, test_session):
        """Test finding meters with pagination"""
        # Arrange
        meters = []
        for i in range(10):
            meter = SmartMeter(
                meter_id=MeterId(f"SM{i:03d}"),
                location=Location(52.5200, 13.4050, f"Berlin {i}"),
                specifications=MeterSpecifications("Siemens", f"SGM-{i}", "1.0.0", "2023-01-01"),
                status=MeterStatus.ACTIVE,
                quality_tier=QualityTier.EXCELLENT,
                metadata={"customer_id": f"CUST{i:03d}"}
            )
            meters.append(meter)
            await repository.save(meter)
        
        # Act
        page1 = await repository.find_with_pagination(page=1, page_size=5)
        page2 = await repository.find_with_pagination(page=2, page_size=5)
        
        # Assert
        assert len(page1) == 5
        assert len(page2) == 5
        assert page1[0].meter_id.value != page2[0].meter_id.value  # Different pages
    
    @pytest.mark.asyncio
    async def test_find_by_quality_tier(self, repository, test_session):
        """Test finding meters by quality tier"""
        # Arrange
        excellent_meter = sample_meter
        good_meter = SmartMeter(
            meter_id=MeterId("SM002"),
            location=Location(52.5200, 13.4050, "Berlin, Germany"),
            specifications=MeterSpecifications("ABB", "ABM-2000", "2.1.0", "2023-02-15"),
            status=MeterStatus.ACTIVE,
            quality_tier=QualityTier.GOOD,
            metadata={"customer_id": "CUST002"}
        )
        
        await repository.save(excellent_meter)
        await repository.save(good_meter)
        
        # Act
        excellent_meters = await repository.find_by_quality_tier(QualityTier.EXCELLENT)
        good_meters = await repository.find_by_quality_tier(QualityTier.GOOD)
        
        # Assert
        assert len(excellent_meters) == 1
        assert len(good_meters) == 1
        assert excellent_meters[0].quality_tier == QualityTier.EXCELLENT
        assert good_meters[0].quality_tier == QualityTier.GOOD
