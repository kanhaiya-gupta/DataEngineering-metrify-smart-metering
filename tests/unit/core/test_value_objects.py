"""
Unit tests for value objects
"""

import pytest
from datetime import datetime
from src.core.domain.value_objects.meter_id import MeterId
from src.core.domain.value_objects.location import Location
from src.core.domain.value_objects.meter_specifications import MeterSpecifications
from src.core.domain.value_objects.quality_score import QualityScore
from src.core.exceptions.domain_exceptions import ValidationError


class TestMeterId:
    """Test cases for MeterId value object"""
    
    def test_create_valid_meter_id(self):
        """Test creating a valid meter ID"""
        # Act
        meter_id = MeterId("SM001")
        
        # Assert
        assert meter_id.value == "SM001"
    
    def test_meter_id_equality(self):
        """Test meter ID equality"""
        # Arrange
        meter_id1 = MeterId("SM001")
        meter_id2 = MeterId("SM001")
        meter_id3 = MeterId("SM002")
        
        # Act & Assert
        assert meter_id1 == meter_id2
        assert meter_id1 != meter_id3
        assert hash(meter_id1) == hash(meter_id2)
        assert hash(meter_id1) != hash(meter_id3)
    
    def test_meter_id_string_representation(self):
        """Test meter ID string representation"""
        # Arrange
        meter_id = MeterId("SM001")
        
        # Act & Assert
        assert str(meter_id) == "SM001"
        assert repr(meter_id) == "MeterId('SM001')"
    
    def test_meter_id_immutability(self):
        """Test that meter ID is immutable"""
        # Arrange
        meter_id = MeterId("SM001")
        
        # Act & Assert
        with pytest.raises(AttributeError):
            meter_id.value = "SM002"


class TestLocation:
    """Test cases for Location value object"""
    
    def test_create_valid_location(self):
        """Test creating a valid location"""
        # Act
        location = Location(
            latitude=52.5200,
            longitude=13.4050,
            address="Berlin, Germany"
        )
        
        # Assert
        assert location.latitude == 52.5200
        assert location.longitude == 13.4050
        assert location.address == "Berlin, Germany"
    
    def test_location_validation_latitude(self):
        """Test location latitude validation"""
        # Act & Assert
        with pytest.raises(ValidationError, match="Latitude must be between -90 and 90"):
            Location(latitude=91.0, longitude=13.4050, address="Test")
        
        with pytest.raises(ValidationError, match="Latitude must be between -90 and 90"):
            Location(latitude=-91.0, longitude=13.4050, address="Test")
    
    def test_location_validation_longitude(self):
        """Test location longitude validation"""
        # Act & Assert
        with pytest.raises(ValidationError, match="Longitude must be between -180 and 180"):
            Location(latitude=52.5200, longitude=181.0, address="Test")
        
        with pytest.raises(ValidationError, match="Longitude must be between -180 and 180"):
            Location(latitude=52.5200, longitude=-181.0, address="Test")
    
    def test_location_validation_address(self):
        """Test location address validation"""
        # Act & Assert
        with pytest.raises(ValidationError, match="Address cannot be empty"):
            Location(latitude=52.5200, longitude=13.4050, address="")
        
        with pytest.raises(ValidationError, match="Address cannot be empty"):
            Location(latitude=52.5200, longitude=13.4050, address=None)
    
    def test_location_equality(self):
        """Test location equality"""
        # Arrange
        location1 = Location(52.5200, 13.4050, "Berlin, Germany")
        location2 = Location(52.5200, 13.4050, "Berlin, Germany")
        location3 = Location(52.5200, 13.4050, "Munich, Germany")
        
        # Act & Assert
        assert location1 == location2
        assert location1 != location3
        assert hash(location1) == hash(location2)
        assert hash(location1) != hash(location3)
    
    def test_location_immutability(self):
        """Test that location is immutable"""
        # Arrange
        location = Location(52.5200, 13.4050, "Berlin, Germany")
        
        # Act & Assert
        with pytest.raises(AttributeError):
            location.latitude = 53.5200


class TestMeterSpecifications:
    """Test cases for MeterSpecifications value object"""
    
    def test_create_valid_specifications(self):
        """Test creating valid meter specifications"""
        # Act
        specs = MeterSpecifications(
            manufacturer="Siemens",
            model="SGM-1000",
            firmware_version="1.2.3",
            installation_date="2023-01-15"
        )
        
        # Assert
        assert specs.manufacturer == "Siemens"
        assert specs.model == "SGM-1000"
        assert specs.firmware_version == "1.2.3"
        assert specs.installation_date == "2023-01-15"
    
    def test_specifications_validation_manufacturer(self):
        """Test manufacturer validation"""
        # Act & Assert
        with pytest.raises(ValidationError, match="Manufacturer cannot be empty"):
            MeterSpecifications(
                manufacturer="",
                model="SGM-1000",
                firmware_version="1.2.3",
                installation_date="2023-01-15"
            )
    
    def test_specifications_validation_model(self):
        """Test model validation"""
        # Act & Assert
        with pytest.raises(ValidationError, match="Model cannot be empty"):
            MeterSpecifications(
                manufacturer="Siemens",
                model="",
                firmware_version="1.2.3",
                installation_date="2023-01-15"
            )
    
    def test_specifications_validation_firmware_version(self):
        """Test firmware version validation"""
        # Act & Assert
        with pytest.raises(ValidationError, match="Firmware version cannot be empty"):
            MeterSpecifications(
                manufacturer="Siemens",
                model="SGM-1000",
                firmware_version="",
                installation_date="2023-01-15"
            )
    
    def test_specifications_validation_installation_date(self):
        """Test installation date validation"""
        # Act & Assert
        with pytest.raises(ValidationError, match="Installation date cannot be empty"):
            MeterSpecifications(
                manufacturer="Siemens",
                model="SGM-1000",
                firmware_version="1.2.3",
                installation_date=""
            )
    
    def test_specifications_equality(self):
        """Test specifications equality"""
        # Arrange
        specs1 = MeterSpecifications("Siemens", "SGM-1000", "1.2.3", "2023-01-15")
        specs2 = MeterSpecifications("Siemens", "SGM-1000", "1.2.3", "2023-01-15")
        specs3 = MeterSpecifications("Siemens", "SGM-2000", "1.2.3", "2023-01-15")
        
        # Act & Assert
        assert specs1 == specs2
        assert specs1 != specs3
        assert hash(specs1) == hash(specs2)
        assert hash(specs1) != hash(specs3)
    
    def test_specifications_immutability(self):
        """Test that specifications are immutable"""
        # Arrange
        specs = MeterSpecifications("Siemens", "SGM-1000", "1.2.3", "2023-01-15")
        
        # Act & Assert
        with pytest.raises(AttributeError):
            specs.manufacturer = "ABB"


class TestQualityScore:
    """Test cases for QualityScore value object"""
    
    def test_create_valid_quality_score(self):
        """Test creating a valid quality score"""
        # Act
        score = QualityScore(0.85)
        
        # Assert
        assert score.value == 0.85
        assert score.percentage == 85.0
        assert score.grade == "B"
    
    def test_quality_score_validation(self):
        """Test quality score validation"""
        # Act & Assert
        with pytest.raises(ValueError, match="Quality score must be between 0.0 and 1.0"):
            QualityScore(1.5)
        
        with pytest.raises(ValueError, match="Quality score must be between 0.0 and 1.0"):
            QualityScore(-0.1)
    
    def test_quality_score_from_percentage(self):
        """Test creating quality score from percentage"""
        # Act
        score = QualityScore.from_percentage(75.0)
        
        # Assert
        assert score.value == 0.75
        assert score.percentage == 75.0
        assert score.grade == "C"
    
    def test_quality_score_from_percentage_validation(self):
        """Test percentage validation"""
        # Act & Assert
        with pytest.raises(ValueError, match="Percentage must be between 0.0 and 100.0"):
            QualityScore.from_percentage(150.0)
        
        with pytest.raises(ValueError, match="Percentage must be between 0.0 and 100.0"):
            QualityScore.from_percentage(-10.0)
    
    def test_quality_score_from_errors(self):
        """Test creating quality score from error count"""
        # Act
        score = QualityScore.from_errors(100, 5)
        
        # Assert
        assert score.value == 0.95
        assert score.percentage == 95.0
        assert score.grade == "A"
    
    def test_quality_score_from_errors_validation(self):
        """Test error count validation"""
        # Act & Assert
        with pytest.raises(ValueError, match="Error count cannot exceed total checks"):
            QualityScore.from_errors(10, 15)
        
        with pytest.raises(ValueError, match="Total checks must be non-negative"):
            QualityScore.from_errors(-5, 2)
    
    def test_quality_score_grade_assignment(self):
        """Test quality score grade assignment"""
        # Test cases: (score, expected_grade)
        test_cases = [
            (0.95, "A"),
            (0.85, "B"),
            (0.75, "C"),
            (0.65, "D"),
            (0.45, "F")
        ]
        
        for score_value, expected_grade in test_cases:
            # Act
            score = QualityScore(score_value)
            
            # Assert
            assert score.grade == expected_grade
    
    def test_quality_score_quality_checks(self):
        """Test quality score quality checks"""
        # Arrange
        excellent_score = QualityScore(0.95)
        good_score = QualityScore(0.85)
        acceptable_score = QualityScore(0.75)
        poor_score = QualityScore(0.65)
        
        # Act & Assert
        assert excellent_score.is_excellent()
        assert not excellent_score.is_poor()
        
        assert good_score.is_good()
        assert not good_score.is_poor()
        
        assert acceptable_score.is_acceptable()
        assert not acceptable_score.is_poor()
        
        assert poor_score.is_poor()
        assert not poor_score.is_acceptable()
    
    def test_quality_score_addition(self):
        """Test adding two quality scores"""
        # Arrange
        score1 = QualityScore(0.8)
        score2 = QualityScore(0.6)
        
        # Act
        result = score1 + score2
        
        # Assert
        assert result.value == 0.7  # (0.8 + 0.6) / 2
        assert result.percentage == 70.0
        assert result.grade == "C"
    
    def test_quality_score_comparison(self):
        """Test comparing quality scores"""
        # Arrange
        score1 = QualityScore(0.8)
        score2 = QualityScore(0.6)
        score3 = QualityScore(0.8)
        
        # Act & Assert
        assert score1 > score2
        assert score2 < score1
        assert score1 >= score2
        assert score2 <= score1
        assert score1 == score3
        assert score1 >= score3
        assert score1 <= score3
    
    def test_quality_score_equality_with_tolerance(self):
        """Test quality score equality with floating point tolerance"""
        # Arrange
        score1 = QualityScore(0.8)
        score2 = QualityScore(0.800000001)  # Very close to 0.8
        
        # Act & Assert
        assert score1 == score2  # Should be equal due to tolerance
    
    def test_quality_score_equality(self):
        """Test quality score equality"""
        # Arrange
        score1 = QualityScore(0.8)
        score2 = QualityScore(0.8)
        score3 = QualityScore(0.6)
        
        # Act & Assert
        assert score1 == score2
        assert score1 != score3
        assert hash(score1) == hash(score2)
        assert hash(score1) != hash(score3)
    
    def test_quality_score_immutability(self):
        """Test that quality score is immutable"""
        # Arrange
        score = QualityScore(0.8)
        
        # Act & Assert
        with pytest.raises(AttributeError):
            score.value = 0.9
    
    def test_quality_score_string_representation(self):
        """Test quality score string representation"""
        # Arrange
        score = QualityScore(0.85)
        
        # Act & Assert
        assert str(score) == "QualityScore(85.0%)"
        assert repr(score) == "QualityScore(value=0.85, grade=B)"
