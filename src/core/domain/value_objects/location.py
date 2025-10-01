"""
Location Value Object
Represents a geographical location for smart meters
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class Location:
    """
    Immutable value object representing a geographical location
    
    Contains coordinates, address information, and grid zone details
    """
    
    latitude: float
    longitude: float
    address: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    country: str = "Germany"
    grid_zone: Optional[str] = None
    
    def __post_init__(self):
        """Validate location data after initialization"""
        # Validate latitude
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Latitude must be between -90 and 90, got {self.latitude}")
        
        # Validate longitude
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Longitude must be between -180 and 180, got {self.longitude}")
        
        # Validate country
        if not self.country or not isinstance(self.country, str):
            raise ValueError("Country must be a non-empty string")
    
    @property
    def coordinates(self) -> tuple[float, float]:
        """Get coordinates as a tuple"""
        return (self.latitude, self.longitude)
    
    def distance_to(self, other: 'Location') -> float:
        """
        Calculate distance to another location using Haversine formula
        
        Returns distance in kilometers
        """
        if not isinstance(other, Location):
            raise ValueError("Can only calculate distance to another Location")
        
        # Earth's radius in kilometers
        R = 6371.0
        
        # Convert to radians
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        delta_lat = math.radians(other.latitude - self.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)
        
        # Haversine formula
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def is_in_germany(self) -> bool:
        """Check if location is within Germany's boundaries"""
        # Approximate Germany boundaries
        min_lat, max_lat = 47.27, 55.06
        min_lon, max_lon = 5.87, 15.04
        
        return (min_lat <= self.latitude <= max_lat and 
                min_lon <= self.longitude <= max_lon)
    
    def get_grid_zone(self) -> str:
        """Get the German grid zone for this location"""
        if self.grid_zone:
            return self.grid_zone
        
        # Determine grid zone based on location
        # This is a simplified mapping - in reality, this would be more complex
        if self.longitude < 8.0:
            return "TenneT"
        elif self.longitude < 10.0:
            return "50Hertz"
        elif self.longitude < 12.0:
            return "Amprion"
        else:
            return "TransnetBW"
    
    def __str__(self) -> str:
        """String representation"""
        if self.address:
            return f"{self.address}, {self.city or 'Unknown City'}, {self.country}"
        return f"({self.latitude:.6f}, {self.longitude:.6f})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"Location("
            f"lat={self.latitude:.6f}, "
            f"lon={self.longitude:.6f}, "
            f"city={self.city}, "
            f"country={self.country}"
            f")"
        )
    
    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if not isinstance(other, Location):
            return False
        return (self.latitude == other.latitude and 
                self.longitude == other.longitude)
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries"""
        return hash((self.latitude, self.longitude))
