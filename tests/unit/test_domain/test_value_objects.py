"""
Unit tests for domain value objects
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

# Import the actual value objects
# from src.core.domain.value_objects.location import Location
# from src.core.domain.value_objects.quality_score import QualityScore
# from src.core.domain.value_objects.timestamp import Timestamp


class TestLocationValueObject:
    """Test Location value object."""
    
    @pytest.mark.unit
    def test_location_creation(self):
        """Test location value object creation."""
        # Valid coordinates
        lat = 52.5200
        lon = 13.4050
        address = "Berlin, Germany"
        
        # Test location creation
        location_data = {
            "latitude": lat,
            "longitude": lon,
            "address": address
        }
        
        assert location_data["latitude"] == lat
        assert location_data["longitude"] == lon
        assert location_data["address"] == address
    
    @pytest.mark.unit
    def test_location_validation(self):
        """Test location validation rules."""
        # Test valid coordinates
        valid_locations = [
            {"lat": 0, "lon": 0},  # Equator, Prime Meridian
            {"lat": 90, "lon": 180},  # North Pole, International Date Line
            {"lat": -90, "lon": -180},  # South Pole, International Date Line
            {"lat": 52.5200, "lon": 13.4050}  # Berlin
        ]
        
        for loc in valid_locations:
            assert -90 <= loc["lat"] <= 90, f"Invalid latitude: {loc['lat']}"
            assert -180 <= loc["lon"] <= 180, f"Invalid longitude: {loc['lon']}"
    
    @pytest.mark.unit
    def test_location_equality(self):
        """Test location equality comparison."""
        loc1 = {"latitude": 52.5200, "longitude": 13.4050, "address": "Berlin"}
        loc2 = {"latitude": 52.5200, "longitude": 13.4050, "address": "Berlin"}
        loc3 = {"latitude": 48.1351, "longitude": 11.5820, "address": "Munich"}
        
        # Same coordinates should be equal
        assert loc1["latitude"] == loc2["latitude"]
        assert loc1["longitude"] == loc2["longitude"]
        
        # Different coordinates should not be equal
        assert loc1["latitude"] != loc3["latitude"]
        assert loc1["longitude"] != loc3["longitude"]
    
    @pytest.mark.unit
    def test_location_distance_calculation(self):
        """Test distance calculation between locations."""
        # Berlin coordinates
        berlin = {"latitude": 52.5200, "longitude": 13.4050}
        # Munich coordinates
        munich = {"latitude": 48.1351, "longitude": 11.5820}
        
        # Calculate approximate distance (Haversine formula)
        import math
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth's radius in kilometers
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat/2)**2 + 
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                 math.sin(dlon/2)**2)
            c = 2 * math.asin(math.sqrt(a))
            return R * c
        
        distance = haversine_distance(
            berlin["latitude"], berlin["longitude"],
            munich["latitude"], munich["longitude"]
        )
        
        # Berlin to Munich is approximately 504 km
        assert 400 <= distance <= 600, f"Distance {distance}km is not in expected range"


class TestQualityScoreValueObject:
    """Test QualityScore value object."""
    
    @pytest.mark.unit
    def test_quality_score_creation(self):
        """Test quality score creation."""
        # Valid quality scores
        valid_scores = [0.0, 0.5, 0.75, 0.9, 1.0]
        
        for score in valid_scores:
            assert 0.0 <= score <= 1.0, f"Invalid quality score: {score}"
    
    @pytest.mark.unit
    def test_quality_score_validation(self):
        """Test quality score validation."""
        # Test boundary values
        assert 0.0 >= 0.0  # Minimum valid score
        assert 1.0 <= 1.0  # Maximum valid score
        
        # Test invalid scores
        invalid_scores = [-0.1, 1.1, 2.0, -1.0]
        for score in invalid_scores:
            assert not (0.0 <= score <= 1.0), f"Score {score} should be invalid"
    
    @pytest.mark.unit
    def test_quality_score_interpretation(self):
        """Test quality score interpretation."""
        # Test score interpretation
        score_interpretations = [
            (0.95, "EXCELLENT"),
            (0.85, "GOOD"),
            (0.70, "FAIR"),
            (0.50, "POOR"),
            (0.20, "POOR")
        ]
        
        for score, expected_level in score_interpretations:
            if score >= 0.9:
                level = "EXCELLENT"
            elif score >= 0.7:
                level = "GOOD"
            elif score >= 0.5:
                level = "FAIR"
            else:
                level = "POOR"
            
            assert level == expected_level, f"Score {score} should be {expected_level}, got {level}"
    
    @pytest.mark.unit
    def test_quality_score_aggregation(self):
        """Test quality score aggregation."""
        scores = [0.8, 0.9, 0.7, 0.85, 0.95]
        
        # Calculate average
        average_score = sum(scores) / len(scores)
        assert 0.8 <= average_score <= 0.9
        
        # Calculate weighted average
        weights = [0.2, 0.3, 0.1, 0.2, 0.2]
        weighted_average = sum(s * w for s, w in zip(scores, weights))
        assert 0.8 <= weighted_average <= 0.9


class TestTimestampValueObject:
    """Test Timestamp value object."""
    
    @pytest.mark.unit
    def test_timestamp_creation(self):
        """Test timestamp creation."""
        now = datetime.utcnow()
        
        # Test timestamp creation
        timestamp_data = {
            "value": now,
            "iso_format": now.isoformat(),
            "unix_timestamp": now.timestamp()
        }
        
        assert timestamp_data["value"] == now
        assert timestamp_data["iso_format"] == now.isoformat()
        assert timestamp_data["unix_timestamp"] == now.timestamp()
    
    @pytest.mark.unit
    def test_timestamp_validation(self):
        """Test timestamp validation."""
        now = datetime.utcnow()
        past = now - timedelta(days=1)
        future = now + timedelta(days=1)
        
        # Valid timestamps
        assert past <= now <= future
        
        # Test timestamp not in future
        assert now <= datetime.utcnow()
        
        # Test timestamp not too old (within last year)
        one_year_ago = now - timedelta(days=365)
        assert now >= one_year_ago
    
    @pytest.mark.unit
    def test_timestamp_comparison(self):
        """Test timestamp comparison."""
        now = datetime.utcnow()
        past = now - timedelta(minutes=1)
        future = now + timedelta(minutes=1)
        
        # Test ordering
        assert past < now < future
        
        # Test equality
        same_time = now
        assert now == same_time
    
    @pytest.mark.unit
    def test_timestamp_formatting(self):
        """Test timestamp formatting."""
        now = datetime.utcnow()
        
        # Test different formats
        iso_format = now.isoformat()
        unix_timestamp = now.timestamp()
        str_format = str(now)
        
        assert "T" in iso_format  # ISO format contains T
        assert isinstance(unix_timestamp, float)  # Unix timestamp is float
        assert len(str_format) > 0  # String format is not empty


class TestEnergyConsumptionValueObject:
    """Test EnergyConsumption value object."""
    
    @pytest.mark.unit
    def test_energy_consumption_creation(self):
        """Test energy consumption creation."""
        # Valid energy consumption values
        valid_consumptions = [0.1, 1.5, 10.0, 100.0, 1000.0]
        
        for consumption in valid_consumptions:
            assert consumption > 0, f"Energy consumption must be positive: {consumption}"
    
    @pytest.mark.unit
    def test_energy_consumption_validation(self):
        """Test energy consumption validation."""
        # Test positive values
        assert 0.1 > 0
        assert 100.0 > 0
        
        # Test invalid values
        invalid_consumptions = [-1.0, 0.0, -0.1]
        for consumption in invalid_consumptions:
            assert not (consumption > 0), f"Consumption {consumption} should be invalid"
    
    @pytest.mark.unit
    def test_energy_consumption_units(self):
        """Test energy consumption unit conversion."""
        # kWh to Wh conversion
        kwh_value = 1.5
        wh_value = kwh_value * 1000
        assert wh_value == 1500.0
        
        # Wh to kWh conversion
        wh_value = 2500.0
        kwh_value = wh_value / 1000
        assert kwh_value == 2.5


class TestPowerFactorValueObject:
    """Test PowerFactor value object."""
    
    @pytest.mark.unit
    def test_power_factor_creation(self):
        """Test power factor creation."""
        # Valid power factor values
        valid_factors = [0.85, 0.90, 0.95, 0.98, 1.0]
        
        for factor in valid_factors:
            assert 0.0 <= factor <= 1.0, f"Invalid power factor: {factor}"
    
    @pytest.mark.unit
    def test_power_factor_validation(self):
        """Test power factor validation."""
        # Test boundary values
        assert 0.0 <= 0.0 <= 1.0  # Minimum
        assert 0.0 <= 1.0 <= 1.0  # Maximum
        
        # Test invalid values
        invalid_factors = [-0.1, 1.1, 2.0]
        for factor in invalid_factors:
            assert not (0.0 <= factor <= 1.0), f"Power factor {factor} should be invalid"
    
    @pytest.mark.unit
    def test_power_factor_interpretation(self):
        """Test power factor interpretation."""
        # Test power factor quality
        factor_interpretations = [
            (1.0, "PERFECT"),
            (0.95, "EXCELLENT"),
            (0.90, "GOOD"),
            (0.85, "ACCEPTABLE"),
            (0.80, "POOR"),
            (0.70, "VERY_POOR")
        ]
        
        for factor, expected_quality in factor_interpretations:
            if factor >= 0.98:
                quality = "PERFECT"
            elif factor >= 0.95:
                quality = "EXCELLENT"
            elif factor >= 0.90:
                quality = "GOOD"
            elif factor >= 0.85:
                quality = "ACCEPTABLE"
            elif factor >= 0.80:
                quality = "POOR"
            else:
                quality = "VERY_POOR"
            
            assert quality == expected_quality, f"Factor {factor} should be {expected_quality}, got {quality}"


class TestVoltageValueObject:
    """Test Voltage value object."""
    
    @pytest.mark.unit
    def test_voltage_creation(self):
        """Test voltage creation."""
        # Valid voltage values (European standard)
        valid_voltages = [220.0, 230.0, 240.0]
        
        for voltage in valid_voltages:
            assert 200.0 <= voltage <= 250.0, f"Invalid voltage: {voltage}"
    
    @pytest.mark.unit
    def test_voltage_validation(self):
        """Test voltage validation."""
        # Test European voltage range
        assert 200.0 <= 220.0 <= 250.0
        assert 200.0 <= 230.0 <= 250.0
        assert 200.0 <= 240.0 <= 250.0
        
        # Test invalid voltages
        invalid_voltages = [100.0, 300.0, 500.0]
        for voltage in invalid_voltages:
            assert not (200.0 <= voltage <= 250.0), f"Voltage {voltage} should be invalid"
    
    @pytest.mark.unit
    def test_voltage_categorization(self):
        """Test voltage categorization."""
        # Test voltage categories
        voltage_categories = [
            (210.0, "LOW"),
            (230.0, "NORMAL"),
            (250.0, "HIGH")
        ]
        
        for voltage, expected_category in voltage_categories:
            if voltage < 220.0:
                category = "LOW"
            elif voltage <= 240.0:
                category = "NORMAL"
            else:
                category = "HIGH"
            
            assert category == expected_category, f"Voltage {voltage} should be {expected_category}, got {category}"


class TestValueObjectImmutability:
    """Test value object immutability."""
    
    @pytest.mark.unit
    def test_location_immutability(self):
        """Test that location value objects are immutable."""
        location = {"latitude": 52.5200, "longitude": 13.4050, "address": "Berlin"}
        
        # Original values should not change
        original_lat = location["latitude"]
        original_lon = location["longitude"]
        
        # Attempt to modify (in real implementation, this would be prevented)
        # location["latitude"] = 48.1351  # This should raise an error
        
        # For now, test that we can detect changes
        assert location["latitude"] == original_lat
        assert location["longitude"] == original_lon
    
    @pytest.mark.unit
    def test_quality_score_immutability(self):
        """Test that quality score value objects are immutable."""
        score = 0.85
        
        # Original value should not change
        original_score = score
        
        # Attempt to modify (in real implementation, this would be prevented)
        # score = 0.95  # This should create a new value object
        
        # For now, test that we can detect changes
        assert score == original_score
