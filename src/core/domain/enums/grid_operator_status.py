"""
Grid Operator Status Enumeration
Defines the possible states of a grid operator
"""

from enum import Enum


class GridOperatorStatus(Enum):
    """
    Enumeration of possible grid operator statuses
    
    Represents the current operational state of a grid operator
    """
    
    # Operational states
    ACTIVE = "active"                    # Operator is active and providing data
    MAINTENANCE = "maintenance"          # Operator is under maintenance
    CALIBRATION = "calibration"          # Operator systems are being calibrated
    
    # Problem states
    ERROR = "error"                      # Operator has encountered an error
    OFFLINE = "offline"                  # Operator is not communicating
    MALFUNCTION = "malfunction"          # Operator has a system malfunction
    OVERLOADED = "overloaded"            # Operator is experiencing high load
    
    # Administrative states
    INACTIVE = "inactive"                # Operator has been deactivated
    SUSPENDED = "suspended"              # Operator has been suspended
    RETIRED = "retired"                  # Operator has been retired
    
    @property
    def is_operational(self) -> bool:
        """Check if the operator is in an operational state"""
        return self in {
            GridOperatorStatus.ACTIVE,
            GridOperatorStatus.MAINTENANCE,
            GridOperatorStatus.CALIBRATION
        }
    
    @property
    def is_problematic(self) -> bool:
        """Check if the operator is in a problematic state"""
        return self in {
            GridOperatorStatus.ERROR,
            GridOperatorStatus.OFFLINE,
            GridOperatorStatus.MALFUNCTION,
            GridOperatorStatus.OVERLOADED
        }
    
    @property
    def is_inactive(self) -> bool:
        """Check if the operator is in an inactive state"""
        return self in {
            GridOperatorStatus.INACTIVE,
            GridOperatorStatus.SUSPENDED,
            GridOperatorStatus.RETIRED
        }
    
    @property
    def requires_attention(self) -> bool:
        """Check if the operator requires immediate attention"""
        return self in {
            GridOperatorStatus.ERROR,
            GridOperatorStatus.OFFLINE,
            GridOperatorStatus.MALFUNCTION,
            GridOperatorStatus.OVERLOADED,
            GridOperatorStatus.MAINTENANCE
        }
    
    @property
    def can_provide_data(self) -> bool:
        """Check if the operator can provide grid data"""
        return self == GridOperatorStatus.ACTIVE
    
    @property
    def is_communicating(self) -> bool:
        """Check if the operator is communicating with the system"""
        return self in {
            GridOperatorStatus.ACTIVE,
            GridOperatorStatus.MAINTENANCE,
            GridOperatorStatus.CALIBRATION,
            GridOperatorStatus.ERROR,
            GridOperatorStatus.OVERLOADED
        }
    
    @classmethod
    def get_operational_statuses(cls) -> list['GridOperatorStatus']:
        """Get all operational statuses"""
        return [status for status in cls if status.is_operational]
    
    @classmethod
    def get_problematic_statuses(cls) -> list['GridOperatorStatus']:
        """Get all problematic statuses"""
        return [status for status in cls if status.is_problematic]
    
    @classmethod
    def get_inactive_statuses(cls) -> list['GridOperatorStatus']:
        """Get all inactive statuses"""
        return [status for status in cls if status.is_inactive]
    
    def __str__(self) -> str:
        """String representation"""
        return self.value
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"GridOperatorStatus.{self.name}"
