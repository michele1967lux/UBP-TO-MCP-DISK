"""
retrieval_strategy - Advanced Retrieval Strategies Engine

Core module for UBP Enterprise Hybrid.

Implements multiple retrieval strategies for RAG systems:

1. Hybrid Retrieval (BM25 + Vector):
   - BM25 keyword-based retrieval
   - Vector semantic retrieval
   - Reciprocal Rank Fusion (RRF)
   - Weighted score fusion
   - Configurable alpha blending

2. Hierarchical Retrieval:
   - Document-level retrieval (coarse)
   - Section-level retrieval (medium)
   - Paragraph-level retrieval (fine)
   - Parent-child chunk linking
   - Context window expansion

3. Router-based Retrieval:
   - LLM-based query analysis
   - Dynamic strategy selection
   - Multi-index routing
   - Tool-aware decisions
   - Fallback chains

Features:
- Multiple fusion algorithms (RRF, weighted, max, sum)
- Configurable chunk hierarchies
- Query classification for routing
- Multi-index support
- Reranking integration
- Comprehensive caching
- Cross-lingual support (EN/IT)

v1.0.0: Initial release with full enterprise features

Architecture:
- adapter.py: Bridge layer exposing all operations
- providers.py: Core retrievers and algorithms
- strategies.py: Strategy implementations
- router.py: LLM-based routing logic
- fusion.py: Score fusion algorithms
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .adapter import RetrievalStrategyAdapter

__version__ = "1.0.0"
__all__ = ["create_module", "RetrievalStrategyAdapter"]


def create_module(
    module_path: Path,
    di_container: Optional[Any] = None,
    event_bus: Optional[Any] = None,
) -> RetrievalStrategyAdapter:
    """
    Factory function for creating the retrieval_strategy adapter.
    
    This is the entry point used by ModuleLoader.
    
    Args:
        module_path: Path to the module directory
        di_container: DI container for dependency resolution
        event_bus: Event bus for publishing events
    
    Returns:
        Configured RetrievalStrategyAdapter instance
    """
    return RetrievalStrategyAdapter(
        module_path=module_path,
        di_container=di_container,
        event_bus=event_bus,
    )
