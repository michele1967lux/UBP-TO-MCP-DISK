"""
graph_rag - Enterprise Knowledge Graph RAG Engine

Core module for UBP Enterprise Hybrid.

Implements Graph-based RAG with knowledge graph construction and traversal:

1. Entity Extraction:
   - Named Entity Recognition (NER)
   - Entity typing and classification
   - Coreference resolution
   - Entity linking and disambiguation

2. Relation Extraction:
   - Semantic relation detection
   - Relation typing and classification
   - Evidence span extraction
   - Confidence scoring

3. Knowledge Graph Construction:
   - Incremental graph building
   - Entity and relation merging
   - Graph persistence (in-memory, Redis, Neo4j)
   - Schema validation

4. Graph-based Retrieval:
   - Entity-centric search
   - Relation-guided traversal
   - Subgraph extraction
   - Multi-hop path finding

5. Subgraph Reasoning:
   - Local context aggregation
   - Path-based inference
   - Community detection
   - Graph neural reasoning

Features:
- Multiple graph backends (in-memory, Redis, Neo4j)
- Hybrid retrieval (graph + vector)
- Incremental graph updates
- Graph visualization export
- Cross-lingual entity linking (EN/IT)
- Comprehensive observability

v1.0.0: Initial release with full enterprise features

Architecture:
- adapter.py: Bridge layer exposing all operations
- providers.py: Core logic with zero backend dependencies
- delegation.py: LLM delegation for extraction
- graph_ops.py: Graph operations and algorithms
- pipeline.py: Multi-step orchestrator
- prompts.py: Extraction prompt templates
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .adapter import GraphRAGAdapter

__version__ = "1.0.0"
__all__ = ["create_module", "GraphRAGAdapter"]


def create_module(
    module_path: Path,
    di_container: Optional[Any] = None,
    event_bus: Optional[Any] = None,
) -> GraphRAGAdapter:
    """
    Factory function for creating the graph_rag adapter.
    
    This is the entry point used by ModuleLoader.
    
    Args:
        module_path: Path to the module directory
        di_container: DI container for dependency resolution
        event_bus: Event bus for publishing events
    
    Returns:
        Configured GraphRAGAdapter instance
    """
    return GraphRAGAdapter(
        module_path=module_path,
        di_container=di_container,
        event_bus=event_bus,
    )
