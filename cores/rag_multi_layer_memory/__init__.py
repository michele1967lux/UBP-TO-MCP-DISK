"""
RAG Multi-Layer Memory Module

Centralized, client-aware, multi-domain contextual memory management
with progressive layers (Layer 0, Layer 1, Layer 2).

Architecture:
- __init__.py: Entry point (this file)
- adapter.py: UBP framework bridge (BaseHybridModule)
- layer_manager.py: Orchestration of 3 layers and trigger logic
- sub_layer_zero.py: Layer 0 snapshot generation and validation
- compression_engine.py: LLM compression via ProviderMapper
- models.py: Pydantic models for layer structures
- utils.py: Token counting and merge utilities
- prompts/: LLM prompt templates
"""

from pathlib import Path
from .adapter import MultiLayerMemoryAdapter

# Module version
__version__ = "1.0.0"

# Public API
__all__ = ["MultiLayerMemoryAdapter", "create_module"]


def create_module(module_path: Path, **kwargs) -> MultiLayerMemoryAdapter:
    """
    Factory function to create module instance.

    Args:
        module_path: Path to module directory
        **kwargs: Additional arguments (event_bus, di_container)

    Returns:
        Initialized module instance
    """
    return MultiLayerMemoryAdapter(module_path, **kwargs)
