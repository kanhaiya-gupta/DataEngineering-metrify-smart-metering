"""
Weather Station Status Enumeration
Defines the possible states of a weather station
"""

from enum import Enum


class WeatherStationStatus(Enum):
    """
    Enumeration of possible weather station statuses
    
    Represents the current operational state of a weather station
    """
    
    # Operational states
    ACTIVE = "active"                    # Station is active and collecting data
    MAINTENANCE = "maintenance"          # Station is under maintenance
    CALIBRATION = "calibration"          # Station is being calibrated
    
    # Problem states
    ERROR = "error"                      # Station has encountered an error
    OFFLINE = "offline"                  # Station is not communicating
    MALFUNCTION = "malfunction"          # Station has a hardware malfunction
    SENSOR_ERROR = "sensor_error"        # Station has sensor issues
    
    # Administrative states
    INACTIVE = "inactive"                # Station has been deactivated
    SUSPENDED = "suspended"              # Station has been suspended
    RETIRED = "retired"                  # Station has been retired
    
    @property
    def is_operational(self) -> bool:
        """Check if the station is in an operational state"""
        return self in {
            WeatherStationStatus.ACTIVE,
            WeatherStationStatus.MAINTENANCE,
            WeatherStationStatus.CALIBRATION
        }
    
    @property
    def is_problematic(self) -> bool:
        """Check if the station is in a problematic state"""
        return self in {
            WeatherStationStatus.ERROR,
            WeatherStationStatus.OFFLINE,
            WeatherStationStatus.MALFUNCTION,
            WeatherStationStatus.SENSOR_ERROR
        }
    
    @property
    def is_inactive(self) -> bool:
        """Check if the station is in an inactive state"""
        return self in {
            WeatherStationStatus.INACTIVE,
            WeatherStationStatus.SUSPENDED,
            WeatherStationStatus.RETIRED
        }
    
    @property
    def requires_attention(self) -> bool:
        """Check if the station requires immediate attention"""
        return self in {
            WeatherStationStatus.ERROR,
            WeatherStationStatus.OFFLINE,
            WeatherStationStatus.MALFUNCTION,
            WeatherStationStatus.SENSOR_ERROR,
            WeatherStationStatus.MAINTENANCE
        }
    
    @property
    def can_collect_data(self) -> bool:
        """Check if the station can collect weather data"""
        return self == WeatherStationStatus.ACTIVE
    
    @property
    def is_communicating(self) -> bool:
        """Check if the station is communicating with the system"""
        return self in {
            WeatherStationStatus.ACTIVE,
            WeatherStationStatus.MAINTENANCE,
            WeatherStationStatus.CALIBRATION,
            WeatherStationStatus.ERROR
        }
    
    @classmethod
    def get_operational_statuses(cls) -> list['WeatherStationStatus']:
        """Get all operational statuses"""
        return [status for status in cls if status.is_operational]
    
    @classmethod
    def get_problematic_statuses(cls) -> list['WeatherStationStatus']:
        """Get all problematic statuses"""
        return [status for status in cls if status.is_problematic]
    
    @classmethod
    def get_inactive_statuses(cls) -> list['WeatherStationStatus']:
        """Get all inactive statuses"""
        return [status for status in cls if status.is_inactive]
    
    def __str__(self) -> str:
        """String representation"""
        return self.value
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"WeatherStationStatus.{self.name}"
