"""
Data Governance Module

This module provides comprehensive data governance capabilities:
- Data lineage tracking and visualization
- Compliance automation (GDPR, privacy controls)
- Advanced data quality assessment and monitoring
- Audit logging and reporting
"""

from .lineage import (
    AtlasIntegration,
    LineageTracker,
    LineageVisualizer
)
from .compliance import (
    GDPRProcessor,
    PrivacyController,
    AuditLogger,
    ConsentManager,
    DataSubjectProcessor,
    RetentionPolicyEngine
)
from .quality import (
    QualityAssessor,
    ValidationEngine,
    QualityMonitor,
    QualityScorer,
    TrendAnalyzer,
    RemediationEngine
)

__all__ = [
    # Lineage
    "AtlasIntegration",
    "LineageTracker", 
    "LineageVisualizer",
    
    # Compliance
    "GDPRProcessor",
    "PrivacyController",
    "AuditLogger",
    "ConsentManager",
    "DataSubjectProcessor",
    "RetentionPolicyEngine",
    
    # Quality
    "QualityAssessor",
    "ValidationEngine",
    "QualityMonitor",
    "QualityScorer",
    "TrendAnalyzer",
    "RemediationEngine"
]
