"""
Meter ID Value Object
Immutable identifier for smart meters
"""

from dataclasses import dataclass
from typing import Union
import re


@dataclass(frozen=True)
class MeterId:
    """
    Immutable value object representing a smart meter ID
    
    Meter IDs follow the format: MET-{REGION}-{TYPE}-{SEQUENCE}
    Example: MET-BER-E-000001
    """
    
    value: str
    
    def __post_init__(self):
        """Validate meter ID format after initialization"""
        if not self.value:
            raise ValueError("Meter ID cannot be empty")
        
        # Validate format: MET-{REGION}-{TYPE}-{SEQUENCE}
        pattern = r'^MET-[A-Z]{3}-[A-Z]-\d{6}$'
        if not re.match(pattern, self.value):
            raise ValueError(
                f"Invalid meter ID format: {self.value}. "
                f"Expected format: MET-{{REGION}}-{{TYPE}}-{{SEQUENCE}}"
            )
        
        # Validate region code (must be 3 letters)
        region = self.value.split('-')[1]
        if not region.isalpha() or len(region) != 3:
            raise ValueError(f"Invalid region code: {region}")
        
        # Validate type code (must be single letter)
        meter_type = self.value.split('-')[2]
        if not meter_type.isalpha() or len(meter_type) != 1:
            raise ValueError(f"Invalid meter type: {meter_type}")
        
        # Validate sequence (must be 6 digits)
        sequence = self.value.split('-')[3]
        if not sequence.isdigit() or len(sequence) != 6:
            raise ValueError(f"Invalid sequence: {sequence}")
    
    @property
    def region(self) -> str:
        """Get the region code from the meter ID"""
        return self.value.split('-')[1]
    
    @property
    def meter_type(self) -> str:
        """Get the meter type from the meter ID"""
        return self.value.split('-')[2]
    
    @property
    def sequence(self) -> str:
        """Get the sequence number from the meter ID"""
        return self.value.split('-')[3]
    
    @classmethod
    def create(cls, region: str, meter_type: str, sequence: int) -> 'MeterId':
        """Create a new MeterId from components"""
        if not region or len(region) != 3 or not region.isalpha():
            raise ValueError("Region must be 3 letters")
        
        if not meter_type or len(meter_type) != 1 or not meter_type.isalpha():
            raise ValueError("Meter type must be 1 letter")
        
        if not isinstance(sequence, int) or sequence < 0 or sequence > 999999:
            raise ValueError("Sequence must be between 0 and 999999")
        
        meter_id_value = f"MET-{region.upper()}-{meter_type.upper()}-{sequence:06d}"
        return cls(meter_id_value)
    
    def __str__(self) -> str:
        """String representation"""
        return self.value
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"MeterId('{self.value}')"
    
    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if not isinstance(other, MeterId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries"""
        return hash(self.value)
    
    def __lt__(self, other) -> bool:
        """Less than comparison for sorting"""
        if not isinstance(other, MeterId):
            return NotImplemented
        return self.value < other.value
