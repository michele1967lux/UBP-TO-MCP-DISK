"""
enrichment_pipeline - Advanced RAG enrichment pipeline

Core module for UBP Enterprise Hybrid.

Features:
- BGE Reranker (cross-encoder)
- Query Expansion via LLM
- HyDE (Hypothetical Document Embedding)
- Context Compression (extractive + abstractive)
- Chunk Fusion (overlap, semantic, adjacent)
- Deduplication
- Metadata Injection
- Configurable pipeline orchestration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .adapter import EnrichmentPipelineAdapter

__version__ = "1.0.0"
__all__ = ["create_module", "EnrichmentPipelineAdapter"]


def create_module(
    module_path: Path,
    di_container: Optional[Any] = None,
    event_bus: Optional[Any] = None,
) -> EnrichmentPipelineAdapter:
    """
    Factory function for creating the enrichment_pipeline adapter.
    
    This is the entry point used by ModuleLoader.
    
    Args:
        module_path: Path to the module directory
        di_container: DI container for dependency resolution
        event_bus: Event bus for publishing events
    
    Returns:
        Configured EnrichmentPipelineAdapter instance
    """
    return EnrichmentPipelineAdapter(
        module_path=module_path,
        di_container=di_container,
        event_bus=event_bus,
    )
