"""
investigation_pipeline - Enterprise RAG Investigation Engine

Core module for UBP Enterprise Hybrid.

Features:
- Multi-Strategy Investigation Generation (decomposition, chain-of-thought, semantic, cross-reference)
- Parallel Worker Pool with async task execution
- Quality Assurance System with multi-dimensional scoring
- Adaptive Strategy Selection via query classification
- Session Management with history tracking
- Redis Caching with environment isolation
- Comprehensive Observability and Metrics
- Debug logging for fallback and automation triggers

v1.0.0: Initial release with full enterprise features
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .adapter import InvestigationPipelineAdapter

__version__ = "1.0.0"
__all__ = ["create_module", "InvestigationPipelineAdapter"]


def create_module(
    module_path: Path,
    di_container: Optional[Any] = None,
    event_bus: Optional[Any] = None,
) -> InvestigationPipelineAdapter:
    """
    Factory function for creating the investigation_pipeline adapter.
    
    This is the entry point used by ModuleLoader.
    
    Args:
        module_path: Path to the module directory
        di_container: DI container for dependency resolution
        event_bus: Event bus for publishing events
    
    Returns:
        Configured InvestigationPipelineAdapter instance
    """
    return InvestigationPipelineAdapter(
        module_path=module_path,
        di_container=di_container,
        event_bus=event_bus,
    )
