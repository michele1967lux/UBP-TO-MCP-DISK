"""kb_relevance_scorer module package.

Composite KB relevance scoring — insertable as pipeline step.
Follows the UBP Hybrid 3-file pattern.

v1.0.0: Initial release — multi-feature composite scoring engine.
"""
from __future__ import annotations
from pathlib import Path

from typing import Any

__version__ = "1.0.0"

__all__ = [
    "create_module",
    "KBRelevanceScorerAdapter",
]

def create_module(module_path, di_container=None,
                  event_bus=None, **kwargs) -> Any:
    """Factory entry point called by ModuleLoader."""
    mp = Path(module_path) if isinstance(module_path, str) else module_path
    from .adapter import KBRelevanceScorerAdapter
    return KBRelevanceScorerAdapter(
        module_path=mp,
        di_container=di_container,
        event_bus=event_bus,
        **kwargs,
    )
