"""
RAG Orchestrator Module

Enterprise RAG orchestration with Access Control Lists.

Features:
- Knowledge base management (create collections, ingest documents)
- Access Control Lists (ACL) per user/client
- Per-user/client RAG configuration
- Complete RAG pipeline (Retrieve, Augment, Generate)
- Dependency injection (rag_qdrant, inference_ollama_grok)

Main Operations:
- create_knowledge_base: Create a new knowledge base
- ingest_document: Ingest text documents with chunking
- set_permission: Set ACL for user/client on collection
- set_rag_config: Set RAG config for user/client
- rag_chat: Execute RAG pipeline with ACL checks (THE MAGIC FUNCTION)
"""

from pathlib import Path
from .adapter import RagOrchestratorAdapter
from .semantic_router import SemanticRouter, RouteType, RouterResult

__version__ = "1.0.0"

__all__ = [
    "RagOrchestratorAdapter",
    "create_module",
    "SemanticRouter",
    "RouteType",
    "RouterResult",
]


def create_module(module_path: Path, **kwargs) -> RagOrchestratorAdapter:
    """
    Factory function to create rag_orchestrator module instance.

    Args:
        module_path: Path to module directory
        **kwargs: Additional arguments (di_container, event_bus, etc.)

    Returns:
        Initialized RagOrchestratorAdapter instance
    """
    return RagOrchestratorAdapter(module_path, **kwargs)
