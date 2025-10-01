"""
Quality Score Value Object
Represents data quality scores with validation and comparison logic
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class QualityScore:
    """
    Value object representing a data quality score
    
    Quality scores range from 0.0 (poor) to 1.0 (excellent)
    """
    
    value: float
    
    def __post_init__(self):
        """Validate quality score value"""
        if not isinstance(self.value, (int, float)):
            raise ValueError("Quality score must be a number")
        
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")
    
    @classmethod
    def from_percentage(cls, percentage: float) -> "QualityScore":
        """Create quality score from percentage (0-100)"""
        if not 0.0 <= percentage <= 100.0:
            raise ValueError("Percentage must be between 0.0 and 100.0")
        
        return cls(value=percentage / 100.0)
    
    @classmethod
    def from_errors(cls, total_checks: int, error_count: int) -> "QualityScore":
        """Create quality score from error count"""
        if total_checks < 0:
            raise ValueError("Total checks must be non-negative")
        
        if error_count < 0:
            raise ValueError("Error count must be non-negative")
        
        if error_count > total_checks:
            raise ValueError("Error count cannot exceed total checks")
        
        if total_checks == 0:
            return cls(value=1.0)  # Perfect score if no checks
        
        success_rate = (total_checks - error_count) / total_checks
        return cls(value=success_rate)
    
    @property
    def percentage(self) -> float:
        """Get quality score as percentage"""
        return round(self.value * 100, 2)
    
    @property
    def grade(self) -> str:
        """Get quality grade (A, B, C, D, F)"""
        if self.value >= 0.9:
            return "A"
        elif self.value >= 0.8:
            return "B"
        elif self.value >= 0.7:
            return "C"
        elif self.value >= 0.6:
            return "D"
        else:
            return "F"
    
    @property
    def is_excellent(self) -> bool:
        """Check if quality score is excellent (>= 0.9)"""
        return self.value >= 0.9
    
    @property
    def is_good(self) -> bool:
        """Check if quality score is good (>= 0.8)"""
        return self.value >= 0.8
    
    @property
    def is_acceptable(self) -> bool:
        """Check if quality score is acceptable (>= 0.7)"""
        return self.value >= 0.7
    
    @property
    def is_poor(self) -> bool:
        """Check if quality score is poor (< 0.7)"""
        return self.value < 0.7
    
    def __add__(self, other: "QualityScore") -> "QualityScore":
        """Add two quality scores (weighted average)"""
        if not isinstance(other, QualityScore):
            raise TypeError("Can only add QualityScore to QualityScore")
        
        # Simple average for now - could be weighted
        return QualityScore(value=(self.value + other.value) / 2)
    
    def __lt__(self, other: "QualityScore") -> bool:
        """Compare quality scores"""
        if not isinstance(other, QualityScore):
            raise TypeError("Can only compare QualityScore with QualityScore")
        
        return self.value < other.value
    
    def __le__(self, other: "QualityScore") -> bool:
        """Compare quality scores"""
        if not isinstance(other, QualityScore):
            raise TypeError("Can only compare QualityScore with QualityScore")
        
        return self.value <= other.value
    
    def __gt__(self, other: "QualityScore") -> bool:
        """Compare quality scores"""
        if not isinstance(other, QualityScore):
            raise TypeError("Can only compare QualityScore with QualityScore")
        
        return self.value > other.value
    
    def __ge__(self, other: "QualityScore") -> bool:
        """Compare quality scores"""
        if not isinstance(other, QualityScore):
            raise TypeError("Can only compare QualityScore with QualityScore")
        
        return self.value >= other.value
    
    def __eq__(self, other: object) -> bool:
        """Check equality with tolerance for floating point precision"""
        if not isinstance(other, QualityScore):
            return False
        
        return math.isclose(self.value, other.value, abs_tol=1e-9)
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries"""
        return hash(self.value)
    
    def __str__(self) -> str:
        """String representation"""
        return f"QualityScore({self.percentage}%)"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"QualityScore(value={self.value}, grade={self.grade})"
