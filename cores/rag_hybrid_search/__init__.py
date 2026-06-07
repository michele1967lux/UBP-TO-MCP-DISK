"""
RAG Hybrid Search Module

Combines dense (vector) and sparse (BM25) search with score fusion
for improved retrieval quality on both semantic and keyword queries.

Features:
- BM25 (Okapi BM25) sparse search for exact keyword matching
- Reciprocal Rank Fusion (RRF) for combining results
- Weighted score fusion with configurable weights
- Automatic index synchronization with rag_qdrant

Usage:
    from ubp_enterprise_hybrid.modules.cores.rag_hybrid_search import create_module

    module = create_module(module_path)
    await module.initialize()

    # Hybrid search
    results = await module.hybrid_search(
        query="product SKU-12345 manual",
        collection_name="docs",
        top_k=10,
        ctx=security_context
    )
"""

from pathlib import Path
from .adapter import HybridSearchAdapter

__version__ = "1.0.0"
__all__ = ["HybridSearchAdapter", "create_module"]


def create_module(module_path: Path, **kwargs) -> HybridSearchAdapter:
    """
    Factory function to create module instance.

    Args:
        module_path: Path to module directory
        **kwargs: Additional arguments for adapter

    Returns:
        HybridSearchAdapter instance
    """
    return HybridSearchAdapter(module_path, **kwargs)
