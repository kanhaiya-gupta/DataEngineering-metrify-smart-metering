"""
Data Lineage Module

This module provides data lineage tracking and visualization capabilities:
- Apache Atlas integration for metadata management
- Automated lineage tracking across data pipelines
- Interactive lineage visualization and impact analysis
"""

from .atlas_integration import AtlasIntegration, AtlasConfig
from .lineage_tracker import LineageTracker
from .lineage_visualizer import LineageVisualizer

__all__ = [
    "AtlasIntegration",
    "AtlasConfig",
    "LineageTracker",
    "LineageVisualizer"
]
