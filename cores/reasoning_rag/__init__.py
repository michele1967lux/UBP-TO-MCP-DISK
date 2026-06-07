"""
reasoning_rag - Enterprise Reasoning-Aware RAG Engine

Core module for UBP Enterprise Hybrid.

Implements advanced RAG strategies that combine reasoning with retrieval:

1. Self-Ask RAG: Iterative sub-question decomposition
   - LLM generates sub-questions
   - Each sub-question triggers retrieval
   - Answers are integrated
   - Process repeats until complete

2. Chain-of-Thought RAG (Interleaved RAG):
   - Reasoning steps interleaved with retrieval
   - Each thought can trigger targeted retrieval
   - Context builds progressively
   - Used in advanced agents

3. Evidence Attribution:
   - Every claim is traced to sources
   - Citations with confidence scores
   - Verifiable references
   - Compliance-ready output

4. Verification Pipeline:
   - Multi-source fact checking
   - Contradiction detection
   - Confidence aggregation
   - Grounding validation

Features:
- Multi-strategy orchestration
- Automatic strategy selection based on query complexity
- Complete reasoning trace logging
- Session continuity
- Redis caching
- Comprehensive observability

v1.0.0: Initial release with full enterprise features

Architecture:
- adapter.py: Bridge layer exposing all operations
- providers.py: Core logic with zero backend dependencies
- delegation.py: LLM and retrieval delegation
- pipeline.py: Multi-strategy orchestrator
- prompts.py: Strategy-specific prompt templates
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .adapter import ReasoningRAGAdapter

__version__ = "1.0.0"
__all__ = ["create_module", "ReasoningRAGAdapter"]


def create_module(
    module_path: Path,
    di_container: Optional[Any] = None,
    event_bus: Optional[Any] = None,
) -> ReasoningRAGAdapter:
    """
    Factory function for creating the reasoning_rag adapter.
    
    This is the entry point used by ModuleLoader.
    
    Args:
        module_path: Path to the module directory
        di_container: DI container for dependency resolution
        event_bus: Event bus for publishing events
    
    Returns:
        Configured ReasoningRAGAdapter instance
    """
    return ReasoningRAGAdapter(
        module_path=module_path,
        di_container=di_container,
        event_bus=event_bus,
    )
