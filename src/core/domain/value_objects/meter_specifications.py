"""
Meter Specifications Value Object
Contains technical specifications of a smart meter
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class MeterAccuracy(Enum):
    """Meter accuracy classes"""
    CLASS_1 = "1"      # ±1% accuracy
    CLASS_2 = "2"      # ±2% accuracy
    CLASS_3 = "3"      # ±3% accuracy


class CommunicationProtocol(Enum):
    """Communication protocols supported by the meter"""
    DLMS_COSEM = "DLMS/COSEM"
    MODBUS = "Modbus"
    M_BUS = "M-Bus"
    LORA_WAN = "LoRaWAN"
    NB_IOT = "NB-IoT"
    WIFI = "WiFi"
    ETHERNET = "Ethernet"


@dataclass(frozen=True)
class MeterSpecifications:
    """
    Immutable value object containing technical specifications of a smart meter
    
    Contains all the technical details needed for proper operation and maintenance
    """
    
    # Basic specifications
    manufacturer: str
    model: str
    serial_number: str
    accuracy_class: MeterAccuracy
    
    # Electrical specifications
    voltage_rating: float  # V
    current_rating: float  # A
    power_rating: float    # kW
    frequency_rating: float  # Hz
    
    # Communication
    communication_protocol: CommunicationProtocol
    communication_frequency: Optional[float] = None  # Hz for wireless
    
    # Physical specifications
    dimensions: Optional[tuple[float, float, float]] = None  # (width, height, depth) in mm
    weight: Optional[float] = None  # kg
    operating_temperature_range: Optional[tuple[float, float]] = None  # (min, max) in °C
    
    # Installation details
    installation_date: Optional[str] = None  # ISO date string
    calibration_date: Optional[str] = None   # ISO date string
    next_calibration_due: Optional[str] = None  # ISO date string
    
    # Features
    supports_remote_reading: bool = True
    supports_load_control: bool = False
    supports_time_of_use: bool = False
    supports_net_metering: bool = False
    
    def __post_init__(self):
        """Validate specifications after initialization"""
        # Validate manufacturer
        if not self.manufacturer or not isinstance(self.manufacturer, str):
            raise ValueError("Manufacturer must be a non-empty string")
        
        # Validate model
        if not self.model or not isinstance(self.model, str):
            raise ValueError("Model must be a non-empty string")
        
        # Validate serial number
        if not self.serial_number or not isinstance(self.serial_number, str):
            raise ValueError("Serial number must be a non-empty string")
        
        # Validate electrical specifications
        if self.voltage_rating <= 0:
            raise ValueError("Voltage rating must be positive")
        
        if self.current_rating <= 0:
            raise ValueError("Current rating must be positive")
        
        if self.power_rating <= 0:
            raise ValueError("Power rating must be positive")
        
        if not (45 <= self.frequency_rating <= 65):
            raise ValueError("Frequency rating must be between 45Hz and 65Hz")
        
        # Validate dimensions if provided
        if self.dimensions:
            if len(self.dimensions) != 3:
                raise ValueError("Dimensions must be a tuple of 3 values (width, height, depth)")
            if any(d <= 0 for d in self.dimensions):
                raise ValueError("All dimension values must be positive")
        
        # Validate weight if provided
        if self.weight is not None and self.weight <= 0:
            raise ValueError("Weight must be positive")
        
        # Validate temperature range if provided
        if self.operating_temperature_range:
            min_temp, max_temp = self.operating_temperature_range
            if min_temp >= max_temp:
                raise ValueError("Minimum temperature must be less than maximum temperature")
    
    @property
    def power_factor_range(self) -> tuple[float, float]:
        """Get the expected power factor range for this meter"""
        # Most smart meters can handle power factors from 0.5 to 1.0
        return (0.5, 1.0)
    
    @property
    def voltage_range(self) -> tuple[float, float]:
        """Get the operating voltage range for this meter"""
        # Typically ±10% of rated voltage
        tolerance = 0.1
        min_voltage = self.voltage_rating * (1 - tolerance)
        max_voltage = self.voltage_rating * (1 + tolerance)
        return (min_voltage, max_voltage)
    
    @property
    def current_range(self) -> tuple[float, float]:
        """Get the operating current range for this meter"""
        # Typically 0.1% to 120% of rated current
        min_current = self.current_rating * 0.001
        max_current = self.current_rating * 1.2
        return (min_current, max_current)
    
    def is_voltage_valid(self, voltage: float) -> bool:
        """Check if a voltage reading is within valid range"""
        min_voltage, max_voltage = self.voltage_range
        return min_voltage <= voltage <= max_voltage
    
    def is_current_valid(self, current: float) -> bool:
        """Check if a current reading is within valid range"""
        min_current, max_current = self.current_range
        return min_current <= current <= max_current
    
    def is_power_factor_valid(self, power_factor: float) -> bool:
        """Check if a power factor reading is within valid range"""
        min_pf, max_pf = self.power_factor_range
        return min_pf <= power_factor <= max_pf
    
    def get_accuracy_tolerance(self) -> float:
        """Get the accuracy tolerance as a percentage"""
        accuracy_map = {
            MeterAccuracy.CLASS_1: 1.0,
            MeterAccuracy.CLASS_2: 2.0,
            MeterAccuracy.CLASS_3: 3.0
        }
        return accuracy_map[self.accuracy_class]
    
    def is_calibration_due(self) -> bool:
        """Check if the meter is due for calibration"""
        if not self.next_calibration_due:
            return False
        
        from datetime import datetime, date
        try:
            calibration_due = datetime.fromisoformat(self.next_calibration_due).date()
            return date.today() >= calibration_due
        except ValueError:
            return False
    
    def get_remaining_calibration_days(self) -> Optional[int]:
        """Get the number of days until calibration is due"""
        if not self.next_calibration_due:
            return None
        
        from datetime import datetime, date
        try:
            calibration_due = datetime.fromisoformat(self.next_calibration_due).date()
            today = date.today()
            delta = calibration_due - today
            return delta.days
        except ValueError:
            return None
    
    def supports_feature(self, feature: str) -> bool:
        """Check if the meter supports a specific feature"""
        feature_map = {
            "remote_reading": self.supports_remote_reading,
            "load_control": self.supports_load_control,
            "time_of_use": self.supports_time_of_use,
            "net_metering": self.supports_net_metering
        }
        return feature_map.get(feature, False)
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.manufacturer} {self.model} ({self.serial_number})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"MeterSpecifications("
            f"manufacturer='{self.manufacturer}', "
            f"model='{self.model}', "
            f"serial='{self.serial_number}', "
            f"accuracy={self.accuracy_class.value}, "
            f"voltage={self.voltage_rating}V, "
            f"current={self.current_rating}A"
            f")"
        )
    
    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if not isinstance(other, MeterSpecifications):
            return False
        return self.serial_number == other.serial_number
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries"""
        return hash(self.serial_number)
