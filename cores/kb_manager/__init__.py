"""kb_manager module package.

Domain-agnostic CRUD manager for structured KB items.
Follows the UBP Hybrid 3-file pattern.

v1.0.0: Initial release (KB-MANAGER)
"""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0"

__all__ = [
    "create_module",
    "KBManagerAdapter",
]

def create_module(module_path, **kwargs: Any):
    """Factory entry point expected by ModuleLoader."""
    from .adapter import KBManagerAdapter

    return KBManagerAdapter(module_path, **kwargs)


def __getattr__(name: str):
    """Lazy symbol export."""
    if name == "KBManagerAdapter":
        from .adapter import KBManagerAdapter

        return KBManagerAdapter

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
