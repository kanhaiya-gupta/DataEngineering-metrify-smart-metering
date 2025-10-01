"""
Alert Level Enum
Defines different levels of alerts in the system
"""

from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels"""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def priority(self) -> int:
        """Get numeric priority for sorting (higher = more urgent)"""
        priority_map = {
            AlertLevel.INFO: 1,
            AlertLevel.WARNING: 2,
            AlertLevel.ERROR: 3,
            AlertLevel.CRITICAL: 4
        }
        return priority_map[self]
    
    @property
    def color(self) -> str:
        """Get color code for UI display"""
        color_map = {
            AlertLevel.INFO: "#17a2b8",      # Info blue
            AlertLevel.WARNING: "#ffc107",   # Warning yellow
            AlertLevel.ERROR: "#dc3545",     # Error red
            AlertLevel.CRITICAL: "#6f42c1"   # Critical purple
        }
        return color_map[self]
    
    @property
    def icon(self) -> str:
        """Get icon name for UI display"""
        icon_map = {
            AlertLevel.INFO: "info-circle",
            AlertLevel.WARNING: "exclamation-triangle",
            AlertLevel.ERROR: "times-circle",
            AlertLevel.CRITICAL: "exclamation-circle"
        }
        return icon_map[self]
