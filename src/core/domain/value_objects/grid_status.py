"""
Grid Status Value Object
Represents the current status of a grid operator
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class GridStatus:
    """Current status of the grid operator"""
    timestamp: datetime
    total_capacity_mw: float
    available_capacity_mw: float
    load_factor: float
    frequency_hz: float
    voltage_kv: float
    grid_stability_score: float
    renewable_percentage: float
    region: str
    
    def __post_init__(self):
        """Validate grid status data"""
        if self.total_capacity_mw <= 0:
            raise ValueError("Total capacity must be positive")
        if self.available_capacity_mw < 0:
            raise ValueError("Available capacity cannot be negative")
        if not (0 <= self.load_factor <= 1):
            raise ValueError("Load factor must be between 0 and 1")
        if not (49.5 <= self.frequency_hz <= 50.5):
            raise ValueError("Frequency must be between 49.5Hz and 50.5Hz")
        if not (380 <= self.voltage_kv <= 420):
            raise ValueError("Voltage must be between 380kV and 420kV")
        if not (0 <= self.grid_stability_score <= 1):
            raise ValueError("Grid stability score must be between 0 and 1")
        if not (0 <= self.renewable_percentage <= 100):
            raise ValueError("Renewable percentage must be between 0 and 100")
