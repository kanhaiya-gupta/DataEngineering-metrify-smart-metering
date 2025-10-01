"""
Meter Status Enumeration
Defines the possible states of a smart meter
"""

from enum import Enum, auto


class MeterStatus(Enum):
    """
    Enumeration of possible smart meter statuses
    
    Represents the current operational state of a smart meter device
    """
    
    # Initial states
    PENDING = "pending"              # Meter registered but not yet activated
    ACTIVE = "active"                # Meter is operational and collecting data
    
    # Operational states
    MAINTENANCE = "maintenance"      # Meter is under maintenance
    CALIBRATION = "calibration"      # Meter is being calibrated
    
    # Problem states
    ERROR = "error"                  # Meter has encountered an error
    OFFLINE = "offline"              # Meter is not communicating
    MALFUNCTION = "malfunction"      # Meter has a hardware malfunction
    
    # Administrative states
    DEACTIVATED = "deactivated"      # Meter has been deactivated
    RETIRED = "retired"              # Meter has been retired from service
    DISPOSED = "disposed"            # Meter has been disposed of
    
    @property
    def is_operational(self) -> bool:
        """Check if the meter is in an operational state"""
        return self in {
            MeterStatus.ACTIVE,
            MeterStatus.MAINTENANCE,
            MeterStatus.CALIBRATION
        }
    
    @property
    def is_problematic(self) -> bool:
        """Check if the meter is in a problematic state"""
        return self in {
            MeterStatus.ERROR,
            MeterStatus.OFFLINE,
            MeterStatus.MALFUNCTION
        }
    
    @property
    def is_inactive(self) -> bool:
        """Check if the meter is in an inactive state"""
        return self in {
            MeterStatus.DEACTIVATED,
            MeterStatus.RETIRED,
            MeterStatus.DISPOSED
        }
    
    @property
    def requires_attention(self) -> bool:
        """Check if the meter requires immediate attention"""
        return self in {
            MeterStatus.ERROR,
            MeterStatus.OFFLINE,
            MeterStatus.MALFUNCTION,
            MeterStatus.MAINTENANCE
        }
    
    @property
    def can_record_readings(self) -> bool:
        """Check if the meter can record new readings"""
        return self == MeterStatus.ACTIVE
    
    @property
    def is_communicating(self) -> bool:
        """Check if the meter is communicating with the system"""
        return self in {
            MeterStatus.ACTIVE,
            MeterStatus.MAINTENANCE,
            MeterStatus.CALIBRATION,
            MeterStatus.ERROR
        }
    
    @classmethod
    def get_operational_statuses(cls) -> list['MeterStatus']:
        """Get all operational statuses"""
        return [status for status in cls if status.is_operational]
    
    @classmethod
    def get_problematic_statuses(cls) -> list['MeterStatus']:
        """Get all problematic statuses"""
        return [status for status in cls if status.is_problematic]
    
    @classmethod
    def get_inactive_statuses(cls) -> list['MeterStatus']:
        """Get all inactive statuses"""
        return [status for status in cls if status.is_inactive]
    
    def __str__(self) -> str:
        """String representation"""
        return self.value
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"MeterStatus.{self.name}"
