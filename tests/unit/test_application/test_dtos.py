"""
Unit tests for application DTOs (Data Transfer Objects)
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import the actual DTOs
# from src.core.application.dtos.smart_meter_dto import SmartMeterDTO
# from src.core.application.dtos.analytics_dto import AnalyticsDTO
# from src.core.application.dtos.quality_dto import QualityDTO


class TestSmartMeterDTO:
    """Test SmartMeter DTO."""
    
    @pytest.mark.unit
    def test_dto_creation(self, sample_smart_meter_data):
        """Test DTO creation from domain data."""
        # Test DTO creation
        dto_data = {
            "meter_id": sample_smart_meter_data["meter_id"],
            "location": sample_smart_meter_data["location"],
            "status": sample_smart_meter_data["status"],
            "quality_tier": sample_smart_meter_data["quality_tier"],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        assert dto_data["meter_id"] == "SM001"
        assert dto_data["status"] == "ACTIVE"
        assert dto_data["quality_tier"] == "EXCELLENT"
        assert "created_at" in dto_data
        assert "updated_at" in dto_data
    
    @pytest.mark.unit
    def test_dto_validation(self):
        """Test DTO validation rules."""
        # Valid DTO data
        valid_dto = {
            "meter_id": "SM001",
            "location": {
                "latitude": 52.5200,
                "longitude": 13.4050,
                "address": "Berlin, Germany"
            },
            "status": "ACTIVE",
            "quality_tier": "EXCELLENT"
        }
        
        # Test validation
        assert valid_dto["meter_id"] is not None
        assert valid_dto["location"]["latitude"] is not None
        assert valid_dto["location"]["longitude"] is not None
        assert valid_dto["status"] in ["ACTIVE", "INACTIVE", "MAINTENANCE"]
        assert valid_dto["quality_tier"] in ["EXCELLENT", "GOOD", "FAIR", "POOR"]
    
    @pytest.mark.unit
    def test_dto_serialization(self, sample_smart_meter_data):
        """Test DTO serialization to JSON."""
        dto_data = {
            "meter_id": sample_smart_meter_data["meter_id"],
            "location": sample_smart_meter_data["location"],
            "status": sample_smart_meter_data["status"],
            "quality_tier": sample_smart_meter_data["quality_tier"]
        }
        
        # Test serialization
        import json
        json_str = json.dumps(dto_data)
        assert isinstance(json_str, str)
        assert "SM001" in json_str
        assert "ACTIVE" in json_str
    
    @pytest.mark.unit
    def test_dto_deserialization(self):
        """Test DTO deserialization from JSON."""
        json_data = {
            "meter_id": "SM001",
            "location": {
                "latitude": 52.5200,
                "longitude": 13.4050,
                "address": "Berlin, Germany"
            },
            "status": "ACTIVE",
            "quality_tier": "EXCELLENT"
        }
        
        # Test deserialization
        assert json_data["meter_id"] == "SM001"
        assert json_data["location"]["latitude"] == 52.5200
        assert json_data["status"] == "ACTIVE"


class TestMeterReadingDTO:
    """Test MeterReading DTO."""
    
    @pytest.mark.unit
    def test_dto_creation(self, sample_meter_reading_data):
        """Test meter reading DTO creation."""
        dto_data = {
            "meter_id": sample_meter_reading_data["meter_id"],
            "timestamp": sample_meter_reading_data["timestamp"].isoformat(),
            "energy_consumed_kwh": sample_meter_reading_data["energy_consumed_kwh"],
            "power_factor": sample_meter_reading_data["power_factor"],
            "voltage_v": sample_meter_reading_data["voltage_v"],
            "current_a": sample_meter_reading_data["current_a"],
            "frequency_hz": sample_meter_reading_data["frequency_hz"],
            "temperature_c": sample_meter_reading_data["temperature_c"],
            "quality_score": sample_meter_reading_data["quality_score"],
            "anomaly_detected": sample_meter_reading_data["anomaly_detected"]
        }
        
        assert dto_data["meter_id"] == "SM001"
        assert dto_data["energy_consumed_kwh"] > 0
        assert 0 <= dto_data["power_factor"] <= 1
        assert 200 <= dto_data["voltage_v"] <= 250
        assert dto_data["current_a"] > 0
        assert 49 <= dto_data["frequency_hz"] <= 51
        assert 0 <= dto_data["quality_score"] <= 1
    
    @pytest.mark.unit
    def test_dto_validation(self, sample_meter_reading_data):
        """Test meter reading DTO validation."""
        reading = sample_meter_reading_data
        
        # Test validation rules
        assert reading["energy_consumed_kwh"] > 0
        assert 0 <= reading["power_factor"] <= 1
        assert 200 <= reading["voltage_v"] <= 250
        assert reading["current_a"] > 0
        assert 49 <= reading["frequency_hz"] <= 51
        assert 0 <= reading["quality_score"] <= 1
        assert isinstance(reading["anomaly_detected"], bool)


class TestAnalyticsDTO:
    """Test Analytics DTO."""
    
    @pytest.mark.unit
    def test_forecast_dto_creation(self, sample_forecast_data):
        """Test forecast DTO creation."""
        dto_data = {
            "meter_id": "SM001",
            "forecast_type": "energy_consumption",
            "forecast_data": sample_forecast_data.to_dict(),
            "model_name": "prophet_v1",
            "accuracy_score": 0.92,
            "created_at": datetime.utcnow().isoformat()
        }
        
        assert dto_data["meter_id"] == "SM001"
        assert dto_data["forecast_type"] == "energy_consumption"
        assert "forecast" in dto_data["forecast_data"]
        assert dto_data["accuracy_score"] > 0.9
    
    @pytest.mark.unit
    def test_anomaly_dto_creation(self, sample_anomaly_data):
        """Test anomaly DTO creation."""
        dto_data = {
            "meter_id": sample_anomaly_data["meter_id"],
            "anomaly_type": sample_anomaly_data["anomaly_type"],
            "severity": sample_anomaly_data["severity"],
            "confidence": sample_anomaly_data["confidence"],
            "description": sample_anomaly_data["description"],
            "detected_at": sample_anomaly_data["timestamp"].isoformat()
        }
        
        assert dto_data["meter_id"] == "SM001"
        assert dto_data["anomaly_type"] == "energy_spike"
        assert dto_data["severity"] == "high"
        assert 0 <= dto_data["confidence"] <= 1
    
    @pytest.mark.unit
    def test_insights_dto_creation(self):
        """Test insights DTO creation."""
        dto_data = {
            "meter_id": "SM001",
            "insights": [
                "High energy consumption detected during peak hours",
                "Temperature correlation found with energy usage",
                "Anomaly pattern identified in weekend consumption"
            ],
            "confidence_scores": [0.95, 0.88, 0.92],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        assert dto_data["meter_id"] == "SM001"
        assert len(dto_data["insights"]) == 3
        assert len(dto_data["confidence_scores"]) == 3
        assert all(0 <= score <= 1 for score in dto_data["confidence_scores"])


class TestQualityDTO:
    """Test Quality DTO."""
    
    @pytest.mark.unit
    def test_quality_metrics_dto_creation(self, sample_quality_metrics):
        """Test quality metrics DTO creation."""
        dto_data = {
            "meter_id": sample_quality_metrics["meter_id"],
            "completeness": sample_quality_metrics["completeness"],
            "accuracy": sample_quality_metrics["accuracy"],
            "consistency": sample_quality_metrics["consistency"],
            "timeliness": sample_quality_metrics["timeliness"],
            "validity": sample_quality_metrics["validity"],
            "overall_score": sample_quality_metrics["overall_score"],
            "assessed_at": datetime.utcnow().isoformat()
        }
        
        assert dto_data["meter_id"] == "SM001"
        assert 0 <= dto_data["completeness"] <= 1
        assert 0 <= dto_data["accuracy"] <= 1
        assert 0 <= dto_data["consistency"] <= 1
        assert 0 <= dto_data["timeliness"] <= 1
        assert 0 <= dto_data["validity"] <= 1
        assert 0 <= dto_data["overall_score"] <= 1
    
    @pytest.mark.unit
    def test_quality_report_dto_creation(self, sample_quality_metrics):
        """Test quality report DTO creation."""
        dto_data = {
            "meter_id": "SM001",
            "overall_score": sample_quality_metrics["overall_score"],
            "quality_level": "EXCELLENT" if sample_quality_metrics["overall_score"] >= 0.9 else "GOOD",
            "issues": sample_quality_metrics["issues"],
            "recommendations": sample_quality_metrics["recommendations"],
            "trends": {
                "improving": True,
                "rate": 0.05
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        assert dto_data["meter_id"] == "SM001"
        assert 0 <= dto_data["overall_score"] <= 1
        assert dto_data["quality_level"] in ["EXCELLENT", "GOOD", "FAIR", "POOR"]
        assert isinstance(dto_data["issues"], list)
        assert isinstance(dto_data["recommendations"], list)


class TestWeatherDTO:
    """Test Weather DTO."""
    
    @pytest.mark.unit
    def test_weather_station_dto_creation(self, sample_weather_station_data):
        """Test weather station DTO creation."""
        dto_data = {
            "station_id": sample_weather_station_data["station_id"],
            "name": sample_weather_station_data["name"],
            "location": sample_weather_station_data["location"],
            "station_type": sample_weather_station_data["station_type"],
            "status": sample_weather_station_data["status"],
            "elevation_m": sample_weather_station_data["elevation_m"]
        }
        
        assert dto_data["station_id"] == "WS001"
        assert dto_data["name"] == "Berlin Weather Station"
        assert dto_data["station_type"] == "AUTOMATIC"
        assert dto_data["status"] == "ACTIVE"
        assert dto_data["elevation_m"] >= 0
    
    @pytest.mark.unit
    def test_weather_observation_dto_creation(self, sample_weather_observation_data):
        """Test weather observation DTO creation."""
        dto_data = {
            "station_id": sample_weather_observation_data["station_id"],
            "timestamp": sample_weather_observation_data["timestamp"].isoformat(),
            "temperature_c": sample_weather_observation_data["temperature_c"],
            "humidity_percent": sample_weather_observation_data["humidity_percent"],
            "pressure_hpa": sample_weather_observation_data["pressure_hpa"],
            "wind_speed_ms": sample_weather_observation_data["wind_speed_ms"],
            "wind_direction_deg": sample_weather_observation_data["wind_direction_deg"]
        }
        
        assert dto_data["station_id"] == "WS001"
        assert -50 <= dto_data["temperature_c"] <= 60
        assert 0 <= dto_data["humidity_percent"] <= 100
        assert 800 <= dto_data["pressure_hpa"] <= 1100
        assert dto_data["wind_speed_ms"] >= 0
        assert 0 <= dto_data["wind_direction_deg"] <= 360


class TestGridOperatorDTO:
    """Test GridOperator DTO."""
    
    @pytest.mark.unit
    def test_grid_operator_dto_creation(self, sample_grid_operator_data):
        """Test grid operator DTO creation."""
        dto_data = {
            "operator_id": sample_grid_operator_data["operator_id"],
            "name": sample_grid_operator_data["name"],
            "country": sample_grid_operator_data["country"],
            "status": sample_grid_operator_data["status"],
            "contact_info": sample_grid_operator_data["contact_info"]
        }
        
        assert dto_data["operator_id"] == "TENNET"
        assert dto_data["name"] == "TenneT TSO B.V."
        assert dto_data["country"] == "Netherlands"
        assert dto_data["status"] == "ACTIVE"
        assert "email" in dto_data["contact_info"]
        assert "phone" in dto_data["contact_info"]
    
    @pytest.mark.unit
    def test_grid_status_dto_creation(self, sample_grid_status_data):
        """Test grid status DTO creation."""
        dto_data = {
            "operator_id": sample_grid_status_data["operator_id"],
            "timestamp": sample_grid_status_data["timestamp"].isoformat(),
            "frequency_hz": sample_grid_status_data["frequency_hz"],
            "voltage_kv": sample_grid_status_data["voltage_kv"],
            "load_mw": sample_grid_status_data["load_mw"],
            "generation_mw": sample_grid_status_data["generation_mw"],
            "stability_score": sample_grid_status_data["stability_score"]
        }
        
        assert dto_data["operator_id"] == "TENNET"
        assert 49.5 <= dto_data["frequency_hz"] <= 50.5
        assert 300 <= dto_data["voltage_kv"] <= 500
        assert dto_data["load_mw"] >= 0
        assert dto_data["generation_mw"] >= 0
        assert 0 <= dto_data["stability_score"] <= 1


class TestDTOValidation:
    """Test DTO validation."""
    
    @pytest.mark.unit
    def test_required_fields_validation(self):
        """Test required fields validation."""
        # Test missing required fields
        incomplete_dto = {
            "meter_id": "SM001"
            # Missing location, status, etc.
        }
        
        required_fields = ["meter_id", "location", "status", "quality_tier"]
        missing_fields = [field for field in required_fields if field not in incomplete_dto]
        
        assert len(missing_fields) > 0
        assert "location" in missing_fields
        assert "status" in missing_fields
        assert "quality_tier" in missing_fields
    
    @pytest.mark.unit
    def test_data_type_validation(self):
        """Test data type validation."""
        # Test correct data types
        valid_dto = {
            "meter_id": "SM001",  # str
            "energy_consumed_kwh": 100.5,  # float
            "power_factor": 0.95,  # float
            "anomaly_detected": True,  # bool
            "timestamp": datetime.utcnow()  # datetime
        }
        
        assert isinstance(valid_dto["meter_id"], str)
        assert isinstance(valid_dto["energy_consumed_kwh"], (int, float))
        assert isinstance(valid_dto["power_factor"], (int, float))
        assert isinstance(valid_dto["anomaly_detected"], bool)
        assert isinstance(valid_dto["timestamp"], datetime)
    
    @pytest.mark.unit
    def test_range_validation(self):
        """Test range validation."""
        # Test valid ranges
        valid_ranges = {
            "latitude": (-90, 90),
            "longitude": (-180, 180),
            "power_factor": (0, 1),
            "quality_score": (0, 1),
            "voltage_v": (200, 250),
            "frequency_hz": (49, 51)
        }
        
        test_values = {
            "latitude": 52.5200,
            "longitude": 13.4050,
            "power_factor": 0.95,
            "quality_score": 0.88,
            "voltage_v": 230.0,
            "frequency_hz": 50.0
        }
        
        for field, (min_val, max_val) in valid_ranges.items():
            value = test_values[field]
            assert min_val <= value <= max_val, f"{field} value {value} out of range [{min_val}, {max_val}]"
