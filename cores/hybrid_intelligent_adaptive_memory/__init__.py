"""
Hybrid Intelligent Adaptive Memory System (HIAMS).

System 4 for robust multi-turn memory:
- Structured Protected Memory (deterministic source of truth)
- Adaptive episodic layers (Layer 0 / Layer 1 / Layer 2)
- Query-aware projection with dynamic token budget
"""

from pathlib import Path

from .adapter import HIAMSAdapter
from .models import (
    AdaptiveLayer1Block,
    AdaptiveLayer2Memory,
    AdaptiveSnapshot,
    CoverageClass,
    HIAMSConfig,
    HIAMSProfileSettings,
    HIAMSSessionState,
    OrderItem,
    OrderState,
    ProcessingResult,
    ProfileType,
    ProjectionIntent,
    ProjectionResult,
    StructuredSlots,
)
from .providers import HIAMSProvider

__version__ = "1.0.0"

__all__ = [
    "AdaptiveLayer1Block",
    "AdaptiveLayer2Memory",
    "AdaptiveSnapshot",
    "CoverageClass",
    "HIAMSAdapter",
    "HIAMSConfig",
    "HIAMSProfileSettings",
    "HIAMSProvider",
    "HIAMSSessionState",
    "OrderItem",
    "OrderState",
    "ProcessingResult",
    "ProfileType",
    "ProjectionIntent",
    "ProjectionResult",
    "StructuredSlots",
    "create_module",
]


def create_module(module_path: Path, **kwargs) -> HIAMSAdapter:
    """Factory function used by the UBP module loader."""
    return HIAMSAdapter(module_path, **kwargs)
