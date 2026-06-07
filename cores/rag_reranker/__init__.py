"""
RAG Reranker Module

Provides document reranking using cross-encoder models for improved
retrieval accuracy. Supports multiple backends:

- LOCAL_CROSS_ENCODER: Local sentence-transformers cross-encoder
- COHERE: Cohere Rerank API
- JINA: Jina Rerank API
- NONE: Pass-through (no reranking)

Cross-encoders score (query, document) pairs directly, providing
more accurate relevance scores than bi-encoder similarity.

Usage:
    from ubp_enterprise_hybrid.modules.cores.rag_reranker import create_module

    module = create_module(module_path)
    await module.initialize()

    # Direct reranking
    results = await module.rerank(
        query="product manual",
        documents=[{"content": "...", "doc_id": "1"}],
        top_k=10,
        ctx=security_context
    )

    # Search + rerank
    results = await module.rerank_search_results(
        query="product manual",
        collection_name="docs",
        initial_top_k=50,
        final_top_k=10,
        ctx=security_context
    )
"""

from pathlib import Path
from .adapter import RerankerAdapter

__version__ = "1.0.0"
__all__ = ["RerankerAdapter", "create_module"]


def create_module(module_path: Path, **kwargs) -> RerankerAdapter:
    """
    Factory function to create module instance.

    Args:
        module_path: Path to module directory
        **kwargs: Additional arguments for adapter

    Returns:
        RerankerAdapter instance
    """
    return RerankerAdapter(module_path, **kwargs)
