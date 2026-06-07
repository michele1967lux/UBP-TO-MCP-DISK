"""
Filter RAG Context Module — Chunk Filtering and Ranking

Filters retrieved RAG chunks by relevance, quality, and diversity
before they reach the LLM prompt builder.

Architecture:
- __init__.py: Entry point (this file)
- adapter.py: UBP framework bridge
- providers.py: Pure filtering logic (zero UBP dependencies)
"""

from pathlib import Path

__version__ = "2.0.0"
__all__ = ["FilterRagContextAdapter", "create_module"]


def __getattr__(name):
    """Lazy import to allow standalone testing without UBP deps."""
    if name == "FilterRagContextAdapter":
        from .adapter import FilterRagContextAdapter
        return FilterRagContextAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_module(module_path: Path, **kwargs):
    """Factory function to create module instance."""
    from .adapter import FilterRagContextAdapter
    return FilterRagContextAdapter(module_path, **kwargs)
