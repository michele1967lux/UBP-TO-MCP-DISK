"""
query_expansion_pipeline - Advanced Query Expansion for RAG

Enterprise module for UBP Hybrid System.

Features:
- Multi-Strategy Expansion (semantic, synonym, decomposition)
- Intent Detection and Classification
- Entity Extraction and Enhancement
- Contextual Expansion (chat history aware)
- Query Decomposition (complex → simple)
- Synonym and Hypernym Expansion
- Cross-lingual Expansion
- LLM-based Generation
- Rule-based Patterns
- Quality Scoring and Filtering
- Ensemble Methods (voting, weighted)
- Query Normalization and Cleaning

Strategies:
- semantic: Generate semantic variations
- synonym: Expand with synonyms/hypernyms
- decompose: Break complex queries into sub-queries
- reformulate: Rephrase as different question types
- keywords: Extract and expand key terms
- contextual: Use conversation context
- hybrid: Combine multiple strategies

Version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .adapter import QueryExpansionAdapter

__version__ = "1.0.0"
__all__ = ["create_module", "QueryExpansionAdapter"]


def create_module(
    module_path: Path,
    di_container: Optional[Any] = None,
    event_bus: Optional[Any] = None,
) -> QueryExpansionAdapter:
    """
    Factory function for creating the query_expansion_pipeline adapter.
    
    This is the entry point used by ModuleLoader.
    
    Args:
        module_path: Path to the module directory
        di_container: DI container for dependency resolution
        event_bus: Event bus for publishing events
    
    Returns:
        Configured QueryExpansionAdapter instance
    """
    return QueryExpansionAdapter(
        module_path=module_path,
        di_container=di_container,
        event_bus=event_bus,
    )
