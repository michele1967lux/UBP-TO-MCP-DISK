"""embedding_prefilter module package.

Embedding-based meta-routing brain with 4-layer decision engine.
Follows the UBP Hybrid 3-file pattern.

v1.0.0: Initial release (Phase 1 — Stabilization)
"""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0"

__all__ = [
    "create_module",
    "EmbeddingPrefilterAdapter",
]

def create_module(module_path, **kwargs: Any):
    """Factory entry point expected by ModuleLoader."""
    from .adapter import EmbeddingPrefilterAdapter

    return EmbeddingPrefilterAdapter(module_path, **kwargs)


def __getattr__(name: str):
    """Lazy symbol export."""
    if name == "EmbeddingPrefilterAdapter":
        from .adapter import EmbeddingPrefilterAdapter

        return EmbeddingPrefilterAdapter

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
