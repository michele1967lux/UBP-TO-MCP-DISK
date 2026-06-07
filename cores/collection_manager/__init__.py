"""
Collection Manager Module

Provides collection management for organizing and managing data collections with PostgreSQL backend.

Architecture:
- __init__.py: Entry point (this file)
- adapter.py: UBP framework bridge
- providers.py: Pure technical logic (zero UBP dependencies)
"""

from pathlib import Path
from .adapter import CollectionManagerAdapter

# Module version
__version__ = "2.0.0"

# Public API
__all__ = ["CollectionManagerAdapter", "create_module"]


def create_module(module_path: Path, **kwargs) -> CollectionManagerAdapter:
    """
    Factory function to create module instance.

    Args:
        module_path: Path to module directory
        **kwargs: Additional arguments

    Returns:
        Initialized module instance
    """
    return CollectionManagerAdapter(module_path, **kwargs)
