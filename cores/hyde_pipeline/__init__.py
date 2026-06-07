"""
hyde_pipeline - Enterprise HyDE (Hypothetical Document Embedding) Engine

Core module for UBP Enterprise Hybrid.

HyDE generates hypothetical documents that would answer the user's query,
then uses these documents for embedding-based retrieval instead of the
raw query. This improves retrieval for complex, abstract, or poorly-formed queries.

Features:
- Multi-Format Generation (answer, technical_doc, faq, code_snippet, tutorial)
- Domain-Adaptive Templates (AI/ML, DevOps, API, Database, Security)
- Ensemble HyDE with strategy fusion
- Iterative Refinement with quality feedback
- Hallucination Detection and confidence scoring
- Cross-Lingual Support (EN/IT) with auto-detection
- Semantic Chunking for optimal embedding
- Adaptive Parameters based on query classification
- Redis Caching with semantic similarity matching
- Comprehensive Observability and Metrics

v1.0.0: Initial release with full enterprise features

Architecture:
- adapter.py: Bridge layer exposing all operations
- providers.py: Core logic with zero backend dependencies
- delegation.py: LLM delegation with multi-format support
- pipeline.py: Configurable multi-step orchestrator
- prompts.py: Domain-specific prompt templates
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .adapter import HyDEPipelineAdapter

__version__ = "1.0.0"
__all__ = ["create_module", "HyDEPipelineAdapter"]


def create_module(
    module_path: Path,
    di_container: Optional[Any] = None,
    event_bus: Optional[Any] = None,
) -> HyDEPipelineAdapter:
    """
    Factory function for creating the hyde_pipeline adapter.
    
    This is the entry point used by ModuleLoader.
    
    Args:
        module_path: Path to the module directory
        di_container: DI container for dependency resolution
        event_bus: Event bus for publishing events
    
    Returns:
        Configured HyDEPipelineAdapter instance
    """
    return HyDEPipelineAdapter(
        module_path=module_path,
        di_container=di_container,
        event_bus=event_bus,
    )
