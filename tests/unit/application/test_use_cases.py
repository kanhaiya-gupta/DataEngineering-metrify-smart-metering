"""
Unit tests for application use cases
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta
from src.application.use_cases.ingest_smart_meter_data import IngestSmartMeterDataUseCase
from src.application.use_cases.process_grid_status import ProcessGridStatusUseCase
from src.application.use_cases.analyze_weather_impact import AnalyzeWeatherImpactUseCase
from src.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
from src.application.dto.smart_meter_dto import MeterReadingCreateDTO, MeterReadingBatchDTO
from src.application.dto.grid_status_dto import GridStatusCreateDTO, GridStatusBatchDTO
from src.application.dto.weather_dto import WeatherObservationCreateDTO, WeatherObservationBatchDTO
from src.core.exceptions.domain_exceptions import MeterNotFoundError, ValidationError


class TestIngestSmartMeterDataUseCase:
    """Test cases for IngestSmartMeterDataUseCase"""
    
    @pytest.fixture
    def mock_meter_repository(self):
        """Mock smart meter repository"""
        repo = Mock()
        repo.find_by_id = AsyncMock()
        repo.save = AsyncMock()
        return repo
    
    @pytest.fixture
    def mock_quality_service(self):
        """Mock data quality service"""
        service = Mock()
        service.validate_reading = AsyncMock()
        service.calculate_quality_score = AsyncMock()
        return service
    
    @pytest.fixture
    def mock_kafka_producer(self):
        """Mock Kafka producer"""
        producer = Mock()
        producer.send = AsyncMock()
        return producer
    
    @pytest.fixture
    def use_case(self, mock_meter_repository, mock_quality_service, mock_kafka_producer):
        """Create use case instance with mocked dependencies"""
        return IngestSmartMeterDataUseCase(
            meter_repository=mock_meter_repository,
            quality_service=mock_quality_service,
            kafka_producer=mock_kafka_producer
        )
    
    @pytest.mark.asyncio
    async def test_execute_single_reading_success(self, use_case, mock_meter_repository, 
                                                mock_quality_service, mock_kafka_producer, 
                                                sample_smart_meter_data, sample_meter_reading_data):
        """Test successful ingestion of a single meter reading"""
        # Arrange
        meter_data = sample_smart_meter_data
        reading_data = MeterReadingCreateDTO(**sample_meter_reading_data)
        
        # Mock meter exists
        mock_meter = Mock()
        mock_meter.meter_id.value = reading_data.meter_id
        mock_meter_repository.find_by_id.return_value = mock_meter
        
        # Mock quality validation
        mock_quality_service.validate_reading.return_value = True
        mock_quality_service.calculate_quality_score.return_value = 0.95
        
        # Act
        result = await use_case.execute(reading_data)
        
        # Assert
        assert result is not None
        mock_meter_repository.find_by_id.assert_called_once_with(reading_data.meter_id)
        mock_quality_service.validate_reading.assert_called_once()
        mock_quality_service.calculate_quality_score.assert_called_once()
        mock_meter_repository.save.assert_called_once()
        mock_kafka_producer.send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_single_reading_meter_not_found(self, use_case, mock_meter_repository, 
                                                        sample_meter_reading_data):
        """Test ingestion when meter is not found"""
        # Arrange
        reading_data = MeterReadingCreateDTO(**sample_meter_reading_data)
        mock_meter_repository.find_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(MeterNotFoundError):
            await use_case.execute(reading_data)
        
        mock_meter_repository.find_by_id.assert_called_once_with(reading_data.meter_id)
        mock_meter_repository.save.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_execute_single_reading_validation_failure(self, use_case, mock_meter_repository, 
                                                           mock_quality_service, sample_meter_reading_data):
        """Test ingestion when reading validation fails"""
        # Arrange
        reading_data = MeterReadingCreateDTO(**sample_meter_reading_data)
        mock_meter = Mock()
        mock_meter.meter_id.value = reading_data.meter_id
        mock_meter_repository.find_by_id.return_value = mock_meter
        mock_quality_service.validate_reading.return_value = False
        
        # Act & Assert
        with pytest.raises(ValidationError, match="Reading validation failed"):
            await use_case.execute(reading_data)
        
        mock_quality_service.validate_reading.assert_called_once()
        mock_meter_repository.save.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_execute_batch_readings_success(self, use_case, mock_meter_repository, 
                                                mock_quality_service, mock_kafka_producer, 
                                                sample_meter_reading_data):
        """Test successful ingestion of batch meter readings"""
        # Arrange
        readings_data = [
            MeterReadingCreateDTO(**sample_meter_reading_data),
            MeterReadingCreateDTO(**{**sample_meter_reading_data, "timestamp": datetime.utcnow() + timedelta(hours=1)})
        ]
        batch_data = MeterReadingBatchDTO(readings=readings_data)
        
        # Mock meter exists
        mock_meter = Mock()
        mock_meter.meter_id.value = readings_data[0].meter_id
        mock_meter_repository.find_by_id.return_value = mock_meter
        
        # Mock quality validation
        mock_quality_service.validate_reading.return_value = True
        mock_quality_service.calculate_quality_score.return_value = 0.95
        
        # Act
        result = await use_case.execute_batch(batch_data)
        
        # Assert
        assert result is not None
        assert result.processed_count == 2
        assert result.success_count == 2
        assert result.failed_count == 0
        assert len(result.errors) == 0
        
        mock_meter_repository.find_by_id.assert_called()
        mock_quality_service.validate_reading.assert_called()
        mock_meter_repository.save.assert_called()
        mock_kafka_producer.send.assert_called()
    
    @pytest.mark.asyncio
    async def test_execute_batch_readings_partial_failure(self, use_case, mock_meter_repository, 
                                                        mock_quality_service, mock_kafka_producer, 
                                                        sample_meter_reading_data):
        """Test batch ingestion with partial failures"""
        # Arrange
        readings_data = [
            MeterReadingCreateDTO(**sample_meter_reading_data),
            MeterReadingCreateDTO(**{**sample_meter_reading_data, "meter_id": "INVALID"})
        ]
        batch_data = MeterReadingBatchDTO(readings=readings_data)
        
        # Mock meter exists for first reading, not for second
        mock_meter = Mock()
        mock_meter.meter_id.value = readings_data[0].meter_id
        
        def mock_find_by_id(meter_id):
            if meter_id == readings_data[0].meter_id:
                return mock_meter
            return None
        
        mock_meter_repository.find_by_id.side_effect = mock_find_by_id
        
        # Mock quality validation
        mock_quality_service.validate_reading.return_value = True
        mock_quality_service.calculate_quality_score.return_value = 0.95
        
        # Act
        result = await use_case.execute_batch(batch_data)
        
        # Assert
        assert result is not None
        assert result.processed_count == 2
        assert result.success_count == 1
        assert result.failed_count == 1
        assert len(result.errors) == 1
        assert "INVALID" in result.errors


class TestProcessGridStatusUseCase:
    """Test cases for ProcessGridStatusUseCase"""
    
    @pytest.fixture
    def mock_grid_repository(self):
        """Mock grid operator repository"""
        repo = Mock()
        repo.find_by_id = AsyncMock()
        repo.save = AsyncMock()
        return repo
    
    @pytest.fixture
    def mock_grid_data_service(self):
        """Mock grid data service"""
        service = Mock()
        service.get_grid_status = AsyncMock()
        service.get_operator_info = AsyncMock()
        return service
    
    @pytest.fixture
    def use_case(self, mock_grid_repository, mock_grid_data_service):
        """Create use case instance with mocked dependencies"""
        return ProcessGridStatusUseCase(
            grid_repository=mock_grid_repository,
            grid_data_service=mock_grid_data_service
        )
    
    @pytest.mark.asyncio
    async def test_execute_single_status_success(self, use_case, mock_grid_repository, 
                                               mock_grid_data_service, sample_grid_status_data):
        """Test successful processing of a single grid status"""
        # Arrange
        status_data = GridStatusCreateDTO(**sample_grid_status_data)
        mock_operator = Mock()
        mock_operator.operator_id = status_data.operator_id
        mock_grid_repository.find_by_id.return_value = mock_operator
        
        # Act
        result = await use_case.execute(status_data)
        
        # Assert
        assert result is not None
        mock_grid_repository.find_by_id.assert_called_once_with(status_data.operator_id)
        mock_grid_repository.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_batch_status_success(self, use_case, mock_grid_repository, 
                                              sample_grid_status_data):
        """Test successful processing of batch grid status"""
        # Arrange
        statuses_data = [
            GridStatusCreateDTO(**sample_grid_status_data),
            GridStatusCreateDTO(**{**sample_grid_status_data, "operator_id": "50HERTZ"})
        ]
        batch_data = GridStatusBatchDTO(statuses=statuses_data)
        
        # Mock operators exist
        mock_operator1 = Mock()
        mock_operator1.operator_id = statuses_data[0].operator_id
        mock_operator2 = Mock()
        mock_operator2.operator_id = statuses_data[1].operator_id
        
        def mock_find_by_id(operator_id):
            if operator_id == statuses_data[0].operator_id:
                return mock_operator1
            elif operator_id == statuses_data[1].operator_id:
                return mock_operator2
            return None
        
        mock_grid_repository.find_by_id.side_effect = mock_find_by_id
        
        # Act
        result = await use_case.execute_batch(batch_data)
        
        # Assert
        assert result is not None
        assert result.processed_count == 2
        assert result.success_count == 2
        assert result.failed_count == 0
        assert len(result.errors) == 0


class TestAnalyzeWeatherImpactUseCase:
    """Test cases for AnalyzeWeatherImpactUseCase"""
    
    @pytest.fixture
    def mock_weather_repository(self):
        """Mock weather station repository"""
        repo = Mock()
        repo.find_by_id = AsyncMock()
        repo.find_by_location = AsyncMock()
        return repo
    
    @pytest.fixture
    def mock_weather_data_service(self):
        """Mock weather data service"""
        service = Mock()
        service.get_current_weather = AsyncMock()
        service.get_weather_forecast = AsyncMock()
        return service
    
    @pytest.fixture
    def use_case(self, mock_weather_repository, mock_weather_data_service):
        """Create use case instance with mocked dependencies"""
        return AnalyzeWeatherImpactUseCase(
            weather_repository=mock_weather_repository,
            weather_data_service=mock_weather_data_service
        )
    
    @pytest.mark.asyncio
    async def test_analyze_weather_impact_success(self, use_case, mock_weather_repository, 
                                                mock_weather_data_service, sample_weather_observation_data):
        """Test successful weather impact analysis"""
        # Arrange
        observation_data = WeatherObservationCreateDTO(**sample_weather_observation_data)
        mock_station = Mock()
        mock_station.station_id = observation_data.station_id
        mock_weather_repository.find_by_id.return_value = mock_station
        
        # Mock weather data service
        mock_weather_data_service.get_current_weather.return_value = {
            "temperature": 20.0,
            "humidity": 60.0,
            "pressure": 1013.25
        }
        
        # Act
        result = await use_case.execute(observation_data)
        
        # Assert
        assert result is not None
        assert "temperature_impact" in result
        assert "humidity_impact" in result
        assert "pressure_impact" in result
        
        mock_weather_repository.find_by_id.assert_called_once_with(observation_data.station_id)
        mock_weather_data_service.get_current_weather.assert_called_once()


class TestDetectAnomaliesUseCase:
    """Test cases for DetectAnomaliesUseCase"""
    
    @pytest.fixture
    def mock_anomaly_service(self):
        """Mock anomaly detection service"""
        service = Mock()
        service.detect_anomalies = AsyncMock()
        service.train_model = AsyncMock()
        return service
    
    @pytest.fixture
    def mock_meter_repository(self):
        """Mock smart meter repository"""
        repo = Mock()
        repo.find_by_id = AsyncMock()
        repo.get_readings_in_time_range = AsyncMock()
        return repo
    
    @pytest.fixture
    def use_case(self, mock_anomaly_service, mock_meter_repository):
        """Create use case instance with mocked dependencies"""
        return DetectAnomaliesUseCase(
            anomaly_service=mock_anomaly_service,
            meter_repository=mock_meter_repository
        )
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_success(self, use_case, mock_anomaly_service, 
                                          mock_meter_repository, sample_meter_reading_data):
        """Test successful anomaly detection"""
        # Arrange
        meter_id = "SM001"
        start_time = datetime.utcnow() - timedelta(hours=24)
        end_time = datetime.utcnow()
        
        # Mock meter exists
        mock_meter = Mock()
        mock_meter.meter_id.value = meter_id
        mock_meter_repository.find_by_id.return_value = mock_meter
        
        # Mock readings
        mock_readings = [Mock() for _ in range(10)]
        mock_meter_repository.get_readings_in_time_range.return_value = mock_readings
        
        # Mock anomaly detection
        mock_anomaly_service.detect_anomalies.return_value = {
            "anomalies": [mock_readings[2], mock_readings[7]],
            "anomaly_score": 0.15,
            "confidence": 0.85
        }
        
        # Act
        result = await use_case.execute(meter_id, start_time, end_time)
        
        # Assert
        assert result is not None
        assert "anomalies" in result
        assert "anomaly_score" in result
        assert "confidence" in result
        assert len(result["anomalies"]) == 2
        
        mock_meter_repository.find_by_id.assert_called_once_with(meter_id)
        mock_meter_repository.get_readings_in_time_range.assert_called_once_with(
            meter_id, start_time, end_time
        )
        mock_anomaly_service.detect_anomalies.assert_called_once_with(mock_readings)
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_meter_not_found(self, use_case, mock_meter_repository):
        """Test anomaly detection when meter is not found"""
        # Arrange
        meter_id = "INVALID"
        start_time = datetime.utcnow() - timedelta(hours=24)
        end_time = datetime.utcnow()
        mock_meter_repository.find_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(MeterNotFoundError):
            await use_case.execute(meter_id, start_time, end_time)
        
        mock_meter_repository.find_by_id.assert_called_once_with(meter_id)
    
    @pytest.mark.asyncio
    async def test_train_anomaly_model_success(self, use_case, mock_anomaly_service, 
                                             mock_meter_repository):
        """Test successful anomaly model training"""
        # Arrange
        meter_id = "SM001"
        training_data = [Mock() for _ in range(100)]
        mock_meter_repository.get_readings_in_time_range.return_value = training_data
        
        # Mock training
        mock_anomaly_service.train_model.return_value = {
            "model_id": "model_123",
            "accuracy": 0.92,
            "training_samples": 100
        }
        
        # Act
        result = await use_case.train_model(meter_id, training_data)
        
        # Assert
        assert result is not None
        assert result["model_id"] == "model_123"
        assert result["accuracy"] == 0.92
        assert result["training_samples"] == 100
        
        mock_anomaly_service.train_model.assert_called_once_with(training_data)
