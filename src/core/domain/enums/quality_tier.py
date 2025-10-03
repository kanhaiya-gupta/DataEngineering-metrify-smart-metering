"""
Quality Tier Enumeration
Defines data quality tiers for smart meter readings
"""

from enum import Enum, auto


class QualityTier(Enum):
    """
    Enumeration of data quality tiers
    
    Represents the quality level of data collected from smart meters
    """
    
    UNKNOWN = "unknown"     # Unknown quality (default for new data)
    HIGH = "high"           # Excellent data quality (90-100%)
    MEDIUM = "medium"       # Good data quality (70-89%)
    LOW = "low"             # Poor data quality (50-69%)
    CRITICAL = "critical"   # Very poor data quality (<50%)
    
    @property
    def min_score(self) -> float:
        """Get the minimum quality score for this tier"""
        score_map = {
            QualityTier.UNKNOWN: 0.0,
            QualityTier.HIGH: 0.9,
            QualityTier.MEDIUM: 0.7,
            QualityTier.LOW: 0.5,
            QualityTier.CRITICAL: 0.0
        }
        return score_map[self]
    
    @property
    def max_score(self) -> float:
        """Get the maximum quality score for this tier"""
        score_map = {
            QualityTier.UNKNOWN: 1.0,
            QualityTier.HIGH: 1.0,
            QualityTier.MEDIUM: 0.89,
            QualityTier.LOW: 0.69,
            QualityTier.CRITICAL: 0.49
        }
        return score_map[self]
    
    @property
    def is_acceptable(self) -> bool:
        """Check if this quality tier is acceptable for production use"""
        return self in {QualityTier.HIGH, QualityTier.MEDIUM}
    
    @property
    def requires_attention(self) -> bool:
        """Check if this quality tier requires attention"""
        return self in {QualityTier.LOW, QualityTier.CRITICAL}
    
    @property
    def priority(self) -> int:
        """Get the priority level (higher number = higher priority)"""
        priority_map = {
            QualityTier.UNKNOWN: 0,
            QualityTier.CRITICAL: 4,
            QualityTier.LOW: 3,
            QualityTier.MEDIUM: 2,
            QualityTier.HIGH: 1
        }
        return priority_map[self]
    
    @classmethod
    def from_score(cls, score: float) -> 'QualityTier':
        """
        Determine quality tier from a quality score
        
        Args:
            score: Quality score between 0.0 and 1.0
            
        Returns:
            QualityTier corresponding to the score
        """
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Quality score must be between 0.0 and 1.0, got {score}")
        
        if score >= 0.9:
            return cls.HIGH
        elif score >= 0.7:
            return cls.MEDIUM
        elif score >= 0.5:
            return cls.LOW
        else:
            return cls.CRITICAL
    
    @classmethod
    def get_acceptable_tiers(cls) -> list['QualityTier']:
        """Get all acceptable quality tiers"""
        return [tier for tier in cls if tier.is_acceptable]
    
    @classmethod
    def get_attention_required_tiers(cls) -> list['QualityTier']:
        """Get all quality tiers that require attention"""
        return [tier for tier in cls if tier.requires_attention]
    
    def __str__(self) -> str:
        """String representation"""
        return self.value
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"QualityTier.{self.name}"
