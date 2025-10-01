"""
Integration tests for data quality service
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from src.infrastructure.external.apis.data_quality_service import DataQualityService
from src.core.domain.entities.smart_meter import MeterReading


@pytest.mark.integration
@pytest.mark.external
class TestDataQualityServiceIntegration:
    """Integration tests for DataQualityService"""
    
    @pytest.fixture
    def data_quality_service(self):
        """Data quality service instance"""
        return DataQualityService(
            api_url="https://api.dataquality.example.com",
            api_key="test-api-key",
            timeout=30
        )
    
    @pytest.fixture
    def sample_reading(self):
        """Sample meter reading for testing"""
        return MeterReading(
            meter_id="SM001",
            timestamp="2023-01-15T10:00:00Z",
            energy_consumed_kwh=1.5,
            power_factor=0.95,
            voltage_v=230.0,
            current_a=6.5,
            frequency_hz=50.0,
            temperature_c=25.0,
            quality_score=0.95,
            anomaly_detected=False
        )
    
    @pytest.mark.asyncio
    async def test_validate_reading_success(self, data_quality_service, sample_reading):
        """Test successful reading validation"""
        # Mock successful API response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "valid": True,
                "quality_score": 0.95,
                "issues": []
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act
            result = await data_quality_service.validate_reading(sample_reading)
            
            # Assert
            assert result is True
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_reading_validation_failure(self, data_quality_service, sample_reading):
        """Test reading validation failure"""
        # Mock API response with validation failure
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "valid": False,
                "quality_score": 0.75,
                "issues": ["Voltage out of range", "Power factor too low"]
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act
            result = await data_quality_service.validate_reading(sample_reading)
            
            # Assert
            assert result is False
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_calculate_quality_score_success(self, data_quality_service, sample_reading):
        """Test successful quality score calculation"""
        # Mock successful API response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "quality_score": 0.95,
                "factors": {
                    "voltage_stability": 0.98,
                    "power_factor": 0.92,
                    "frequency_consistency": 0.96
                }
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act
            result = await data_quality_service.calculate_quality_score(sample_reading)
            
            # Assert
            assert result == 0.95
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_success(self, data_quality_service, sample_reading):
        """Test successful anomaly detection"""
        # Mock successful API response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "anomalies_detected": True,
                "anomaly_score": 0.85,
                "anomaly_types": ["voltage_spike", "frequency_deviation"]
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act
            result = await data_quality_service.detect_anomalies(sample_reading)
            
            # Assert
            assert result is True
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_timeout_handling(self, data_quality_service, sample_reading):
        """Test API timeout handling"""
        # Mock timeout exception
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError("Request timeout")
            
            # Act & Assert
            with pytest.raises(asyncio.TimeoutError):
                await data_quality_service.validate_reading(sample_reading)
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, data_quality_service, sample_reading):
        """Test API error handling"""
        # Mock API error response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act & Assert
            with pytest.raises(Exception, match="API request failed"):
                await data_quality_service.validate_reading(sample_reading)
    
    @pytest.mark.asyncio
    async def test_batch_validation_success(self, data_quality_service):
        """Test successful batch validation"""
        # Arrange
        readings = [
            MeterReading(
                meter_id="SM001",
                timestamp="2023-01-15T10:00:00Z",
                energy_consumed_kwh=1.5,
                power_factor=0.95,
                voltage_v=230.0,
                current_a=6.5,
                frequency_hz=50.0,
                temperature_c=25.0,
                quality_score=0.95,
                anomaly_detected=False
            ),
            MeterReading(
                meter_id="SM002",
                timestamp="2023-01-15T10:00:00Z",
                energy_consumed_kwh=2.0,
                power_factor=0.96,
                voltage_v=231.0,
                current_a=7.0,
                frequency_hz=50.1,
                temperature_c=26.0,
                quality_score=0.96,
                anomaly_detected=False
            )
        ]
        
        # Mock successful API response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "results": [
                    {"valid": True, "quality_score": 0.95},
                    {"valid": True, "quality_score": 0.96}
                ]
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act
            result = await data_quality_service.validate_readings_batch(readings)
            
            # Assert
            assert len(result) == 2
            assert result[0]["valid"] is True
            assert result[1]["valid"] is True
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, data_quality_service, sample_reading):
        """Test retry mechanism for failed requests"""
        # Mock API response with retry
        with patch('aiohttp.ClientSession.post') as mock_post:
            # First call fails, second call succeeds
            mock_response_fail = Mock()
            mock_response_fail.status = 503
            mock_response_fail.text.return_value = "Service Unavailable"
            
            mock_response_success = Mock()
            mock_response_success.status = 200
            mock_response_success.json.return_value = {
                "valid": True,
                "quality_score": 0.95,
                "issues": []
            }
            
            mock_post.return_value.__aenter__.side_effect = [mock_response_fail, mock_response_success]
            
            # Act
            result = await data_quality_service.validate_reading(sample_reading)
            
            # Assert
            assert result is True
            assert mock_post.call_count == 2  # Should retry once
    
    @pytest.mark.asyncio
    async def test_rate_limiting_handling(self, data_quality_service, sample_reading):
        """Test rate limiting handling"""
        # Mock rate limiting response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 429
            mock_response.headers = {"Retry-After": "60"}
            mock_response.text.return_value = "Rate limit exceeded"
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act & Assert
            with pytest.raises(Exception, match="Rate limit exceeded"):
                await data_quality_service.validate_reading(sample_reading)
    
    @pytest.mark.asyncio
    async def test_authentication_handling(self, data_quality_service, sample_reading):
        """Test authentication handling"""
        # Mock authentication error response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 401
            mock_response.text.return_value = "Unauthorized"
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act & Assert
            with pytest.raises(Exception, match="Authentication failed"):
                await data_quality_service.validate_reading(sample_reading)
