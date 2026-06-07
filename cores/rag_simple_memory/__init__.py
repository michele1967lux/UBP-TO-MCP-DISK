"""
Simple In-Memory RAG Module

Provides a lightweight Retrieval-Augmented Generation system with in-memory vector storage.

Architecture:
- __init__.py: Entry point (this file)
- adapter.py: UBP framework bridge
- providers.py: Pure technical logic (zero UBP dependencies)
"""

from pathlib import Path
from .adapter import RAGSimpleMemoryAdapter

# Module version
__version__ = "2.0.0"

# Public API
__all__ = ["RAGSimpleMemoryAdapter", "create_module"]


def create_module(module_path: Path, **kwargs) -> RAGSimpleMemoryAdapter:
    """
    Factory function to create module instance.

    Args:
        module_path: Path to module directory
        **kwargs: Additional arguments

    Returns:
        Initialized module instance
    """
    return RAGSimpleMemoryAdapter(module_path, **kwargs)
