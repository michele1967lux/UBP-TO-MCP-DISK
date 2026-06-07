"""rag_qdrant module package.

This core module follows the UBP Hybrid 3-file pattern:
- manifest.json (contract)
- adapter.py (UBP bridge)
- providers.py (pure implementation)

IMPORTANT:
This package must remain *import-light*:
- expose a `create_module(...)` factory
- avoid importing backend-dependent symbols at import time

Adapter/Provider classes are exposed via lazy imports (PEP 562 __getattr__).
"""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0"

__all__ = [
    "create_module",
    "RagQdrantAdapter",
    "RAGQdrant",
]

def create_module(module_path, **kwargs: Any):
    """Factory entry point expected by ModuleLoader (Pattern 1)."""
    from .adapter import RagQdrantAdapter

    return RagQdrantAdapter(module_path, **kwargs)


def __getattr__(name: str):
    """Lazy symbol export to keep package import lightweight."""
    if name == "RagQdrantAdapter":
        from .adapter import RagQdrantAdapter

        return RagQdrantAdapter

    if name == "RAGQdrant":
        from .providers import RAGQdrant

        return RAGQdrant

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
