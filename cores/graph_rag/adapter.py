"""
graph_rag/adapter.py

Bridge layer that exposes all Graph-RAG operations to the UBP system.
Handles initialization, configuration, DI resolution, and operation routing.

Operations:
- initialize: Start components
- query: Full Graph-RAG pipeline
- extract_entities: Extract entities from text
- extract_relations: Extract relations between entities
- build_graph: Add documents to knowledge graph
- search_entities: Search entities in graph
- get_subgraph: Extract subgraph around entities
- find_paths: Find paths between entities
- get_graph_stats: Get graph statistics
- export_graph: Export graph in various formats
- clear_graph: Clear knowledge graph
- get_session / delete_session: Session management
- get_stats: Metrics and statistics (admin)
- reload_config: Hot-reload configuration (admin)
- shutdown: Graceful shutdown
- health_check: Component health status

v1.0.0: Initial release with full enterprise features
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

from .providers import (
    # Data classes
    Entity,
    Relation,
    Triple,
    Subgraph,
    GraphPath,
    ExtractionResult,
    GraphQueryResult,
    GraphRAGResult,
    GraphSession,
    KnowledgeGraph,
    # Enums
    EntityType,
    RelationType,
    GraphBackend,
    RetrievalStrategy,
    # Configs
    GraphConfig,
    EntityExtractionConfig,
    RelationExtractionConfig,
    GraphConstructionConfig,
    GraphRetrievalConfig,
    HybridRetrievalConfig,
    SubgraphReasoningConfig,
    CacheConfig,
    SessionConfig,
    MetricsConfig,
    DebugConfig,
    # Providers
    GraphCacheProvider,
    GraphSessionManager,
    GraphMetricsCollector,
)
from .delegation import (
    GraphDelegator,
    LLMDelegationConfig,
)
from .pipeline import (
    GraphRAGPipeline,
    PipelineConfig,
    StepConfig,
)
from .prompts import detect_language, get_entity_types, get_relation_types

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class IModuleRegistry(Protocol):
    """Protocol for module registry."""
    def get_module(self, module_name: str) -> Optional[Any]: ...


# ============================================================================
# DI Container Module Registry Wrapper
# ============================================================================


class DIContainerModuleRegistry:
    """Wraps DI container to provide module registry interface."""

    def __init__(self, di_container: Optional[Any] = None):
        self._container = di_container
        self._cached_modules: Dict[str, Any] = {}

    def get_module(self, module_name: str) -> Optional[Any]:
        """Get a module by name (sync - cache only)."""
        if module_name in self._cached_modules:
            return self._cached_modules[module_name]
        return None

    async def resolve_module(self, module_name: str) -> Optional[Any]:
        """Async module resolution via DI container."""
        if module_name in self._cached_modules:
            return self._cached_modules[module_name]

        if not self._container:
            return None

        # DI container.resolve() is async - must be awaited
        if hasattr(self._container, "resolve"):
            try:
                module = await self._container.resolve(module_name)
                if module:
                    self._cached_modules[module_name] = module
                    return module
            except Exception as e:
                logger.warning(f"Failed to resolve module '{module_name}': {e}")

        return None


# ============================================================================
# Configuration Utilities
# ============================================================================


def resolve_env_value(value: Any) -> Any:
    """Resolve environment variable placeholders."""
    if not isinstance(value, str):
        return value
    
    pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
    
    def replace(match):
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)
    
    return re.sub(pattern, replace, value)


def coerce_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively coerce configuration values to appropriate types."""
    result = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = coerce_config_types(value)
        elif isinstance(value, list):
            result[key] = [
                coerce_config_types(v) if isinstance(v, dict) else _coerce_value(v)
                for v in value
            ]
        else:
            result[key] = _coerce_value(value)
    
    return result


def _coerce_value(value: Any) -> Any:
    """Coerce a single value to appropriate type."""
    if not isinstance(value, str):
        return value
    
    value = resolve_env_value(value)
    
    if not isinstance(value, str):
        return value
    
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False
    
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    return value


# ============================================================================
# Graph RAG Adapter
# ============================================================================


class GraphRAGAdapter:
    """
    Adapter that exposes Graph-RAG operations to the UBP system.
    
    Features:
    - Knowledge graph construction
    - Entity and relation extraction
    - Graph-based retrieval
    - Subgraph reasoning
    - Path finding
    """
    
    def __init__(
        self,
        module_path: Path,
        di_container: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        self.module_path = Path(module_path)
        self._di_container = di_container
        self._event_bus = event_bus
        
        self._module_registry = DIContainerModuleRegistry(di_container)
        
        # Configuration
        self._config: Dict[str, Any] = {}
        self._graph_config: Optional[GraphConfig] = None
        self._entity_config: Optional[EntityExtractionConfig] = None
        self._relation_config: Optional[RelationExtractionConfig] = None
        self._construction_config: Optional[GraphConstructionConfig] = None
        self._retrieval_config: Optional[GraphRetrievalConfig] = None
        self._cache_config: Optional[CacheConfig] = None
        self._session_config: Optional[SessionConfig] = None
        self._metrics_config: Optional[MetricsConfig] = None
        self._debug_config: Optional[DebugConfig] = None
        self._llm_config: Optional[LLMDelegationConfig] = None
        self._pipeline_config: Optional[PipelineConfig] = None
        
        # Components
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._cache: Optional[GraphCacheProvider] = None
        self._session_manager: Optional[GraphSessionManager] = None
        self._metrics: Optional[GraphMetricsCollector] = None
        self._delegator: Optional[GraphDelegator] = None
        self._pipeline: Optional[GraphRAGPipeline] = None
        
        # State
        self._initialized = False
        self._redis_client: Optional[Any] = None
    
    # MCP-COMPAT: OperationContext helpers (ARCH-008)
    def _build_context_from_di(self) -> OperationContext:
        """Build OperationContext from DI — backward compatibility for REST path."""
        return OperationContext(
            client_id="default",
            user_id=None,
            session_id=None,
            source="rest",
        )

    def _normalize_ctx(self, ctx: Any) -> OperationContext:
        """Normalize any context format to OperationContext."""
        if ctx is None:
            return self._build_context_from_di()
        if isinstance(ctx, OperationContext):
            return ctx
        if hasattr(ctx, "user") and ctx.user:
            user_id = getattr(ctx.user, "user_id", None)
            roles = getattr(ctx.user, "roles", [])
            client_id = getattr(ctx.user, "client_id", "default")
            if not isinstance(roles, (list, tuple)):
                roles = []
            return OperationContext(
                client_id=str(client_id) if client_id else "default",
                user_id=str(user_id) if user_id else None,
                roles=list(roles),
                source="rest",
            )
        return self._build_context_from_di()
    
    # ========================================================================
    # Configuration Loading
    # ========================================================================
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json."""
        config_path = self.module_path / "config.json"
        
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}")
            return {}
        
        with open(config_path, "r") as f:
            raw_config = json.load(f)
        
        return coerce_config_types(raw_config)
    
    def _resolve_llm_module_name(self) -> str:
        """Resolve LLM module name from ProviderMapper chain (role=rag)."""
        try:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper
            chain = ProviderMapper.resolve_chain("rag")
            if chain:
                module_name = chain[0][0]  # First in chain = primary
                logger.info(f"[GRAPH] LLM resolved via ProviderMapper: {module_name}")
                return module_name
        except Exception as e:
            logger.warning(
                f"[GRAPH] ProviderMapper NOT AVAILABLE - using hardcoded fallback "
                f"'inference_ollama_grok'. Centralized provider config (UBP_ROLES__RAG_PROVIDER) "
                f"is IGNORED for this module. Cause: {e}"
            )
        return "inference_ollama_grok"

    def _build_configs(self) -> None:
        """Build all configuration objects."""
        cfg = self._config
        
        self._graph_config = GraphConfig(
            enabled=cfg.get("graph_rag", {}).get("enabled", True),
            default_backend=cfg.get("graph_rag", {}).get("default_backend", "memory"),
            max_graph_size=cfg.get("graph_rag", {}).get("max_graph_size", 100000),
        )
        
        entity_cfg = cfg.get("entity_extraction", {})
        self._entity_config = EntityExtractionConfig(
            enabled=entity_cfg.get("enabled", True),
            method=entity_cfg.get("method", "llm"),
            max_entities_per_chunk=entity_cfg.get("max_entities_per_chunk", 20),
            min_confidence=entity_cfg.get("min_confidence", 0.6),
            temperature=entity_cfg.get("temperature", 0.1),
        )
        
        relation_cfg = cfg.get("relation_extraction", {})
        self._relation_config = RelationExtractionConfig(
            enabled=relation_cfg.get("enabled", True),
            method=relation_cfg.get("method", "llm"),
            max_relations_per_chunk=relation_cfg.get("max_relations_per_chunk", 30),
            min_confidence=relation_cfg.get("min_confidence", 0.5),
            temperature=relation_cfg.get("temperature", 0.1),
        )
        
        construction_cfg = cfg.get("graph_construction", {})
        self._construction_config = GraphConstructionConfig(
            merge_strategy=construction_cfg.get("merge_strategy", "smart"),
            entity_merge_threshold=construction_cfg.get("entity_merge_threshold", 0.9),
            max_nodes=construction_cfg.get("max_nodes", 50000),
            max_edges=construction_cfg.get("max_edges", 200000),
        )
        
        cache_cfg = cfg.get("cache", {})
        self._cache_config = CacheConfig(
            enabled=cache_cfg.get("enabled", True),
            ttl_seconds=cache_cfg.get("ttl_seconds", 3600),
        )
        
        session_cfg = cfg.get("session_management", {})
        self._session_config = SessionConfig(
            enabled=session_cfg.get("enabled", True),
            ttl_seconds=session_cfg.get("ttl_seconds", 3600),
            max_history_size=session_cfg.get("max_history_size", 30),
        )
        
        metrics_cfg = cfg.get("metrics", {})
        self._metrics_config = MetricsConfig(
            enabled=metrics_cfg.get("enabled", True),
        )
        
        debug_cfg = cfg.get("debug", {})
        self._debug_config = DebugConfig(
            enabled=debug_cfg.get("enabled", False),
            log_extractions=debug_cfg.get("log_extractions", True),
            log_graph_ops=debug_cfg.get("log_graph_ops", True),
        )
        
        delegation_cfg = cfg.get("delegation", {})
        # v3.7: Use ProviderMapper for centralized resolution, cfg override still honored
        default_module = self._resolve_llm_module_name()
        self._llm_config = LLMDelegationConfig(
            llm_module=delegation_cfg.get("llm_module", default_module),
            llm_operation=delegation_cfg.get("llm_operation", "generate"),
            timeout_seconds=delegation_cfg.get("timeout_seconds", 30),
        )
        
        pipeline_cfg = cfg.get("pipeline", {})
        steps = {}
        for step_name, step_data in pipeline_cfg.get("steps", {}).items():
            if isinstance(step_data, dict):
                steps[step_name] = StepConfig(
                    enabled=step_data.get("enabled", True),
                    timeout=step_data.get("timeout", 30),
                )
        self._pipeline_config = PipelineConfig(
            default_timeout_seconds=pipeline_cfg.get("default_timeout_seconds", 120),
            fail_fast=pipeline_cfg.get("fail_fast", False),
            steps=steps,
        )
    
    # ========================================================================
    # Operations
    # ========================================================================
    
    async def initialize(self, ctx: Any = None) -> Dict[str, Any]:
        """Initialize all Graph-RAG components."""
        if self._initialized:
            return {"status": "already_initialized"}
        
        try:
            self._config = self._load_config()
            self._build_configs()
            
            # Initialize knowledge graph
            self._knowledge_graph = KnowledgeGraph(self._construction_config)
            
            # Try to get Redis client
            if self._di_container:
                self._redis_client = getattr(self._di_container, "redis", None)
            
            self._cache = GraphCacheProvider(
                config=self._cache_config,
                redis_client=self._redis_client,
            )
            
            self._session_manager = GraphSessionManager(self._session_config)
            self._metrics = GraphMetricsCollector(self._metrics_config)
            
            self._delegator = GraphDelegator(
                llm_config=self._llm_config,
                entity_config=self._entity_config,
                relation_config=self._relation_config,
                module_registry=self._module_registry,
                event_publisher=self._event_bus,
                debug_config=self._debug_config,
            )
            
            self._pipeline = GraphRAGPipeline(
                delegator=self._delegator,
                knowledge_graph=self._knowledge_graph,
                config=self._pipeline_config,
                debug_config=self._debug_config,
            )
            
            self._initialized = True
            
            logger.info("Graph-RAG pipeline initialized successfully")
            
            if self._event_bus:
                await self._event_bus.publish(
                    "graph.initialized",
                    {"module": "graph_rag", "status": "success"},
                )
            
            return {
                "status": "initialized",
                "components": {
                    "knowledge_graph": True,
                    "cache": self._cache_config.enabled,
                    "session_manager": self._session_config.enabled,
                    "metrics": self._metrics_config.enabled,
                    "delegator": True,
                    "pipeline": True,
                },
                "graph_stats": self._knowledge_graph.get_stats(),
            }
            
        except Exception as e:
            logger.error(f"Graph-RAG initialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def query(
        self,
        query: str,
        language: Optional[str] = None,
        max_hops: int = 2,
        max_subgraph_nodes: int = 100,
        retrieval_strategy: str = "hybrid",
        session_id: Optional[str] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
        include_paths: bool = False,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Full Graph-RAG pipeline.
        
        Args:
            query: User's query
            language: Language override (en, it, auto)
            max_hops: Maximum graph traversal hops
            max_subgraph_nodes: Maximum nodes in extracted subgraph
            retrieval_strategy: Strategy (entity_centric, relation_guided, path_based, subgraph, hybrid)
            session_id: Existing session ID
            pipeline_config: Step configuration override
            include_paths: Whether to include graph paths in result (accepted but not yet implemented)
            ctx: Security context
            
        Returns:
            GraphRAGResult with answer and graph context
        """
        if not self._initialized:
            await self.initialize(ctx)
        
        # Get or create session
        if session_id:
            session = await self._session_manager.get_session(session_id)
        else:
            session = await self._session_manager.create_session()
        
        if not session:
            session = await self._session_manager.create_session()
        
        # Execute pipeline
        result = await self._pipeline.execute(
            query=query,
            session_id=session.session_id,
            language=language,
            max_hops=max_hops,
            max_subgraph_nodes=max_subgraph_nodes,
            retrieval_strategy=retrieval_strategy,
            pipeline_config=pipeline_config,
            ctx=ctx,
        )
        
        # Update session
        await self._session_manager.update_session(
            session_id=session.session_id,
            query=query,
            result=result,
        )
        
        return result.to_dict()
    
    async def extract_entities(
        self,
        text: str,
        doc_id: Optional[str] = None,
        language: str = "auto",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Extract entities from text."""
        if not self._initialized:
            await self.initialize(ctx)
        
        doc_id = doc_id or str(uuid.uuid4())
        entities = await self._delegator.extract_entities(text, doc_id, language)
        
        return {
            "doc_id": doc_id,
            "entity_count": len(entities),
            "entities": [e.to_dict() for e in entities],
        }
    
    async def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        doc_id: Optional[str] = None,
        language: str = "auto",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Extract relations between entities."""
        if not self._initialized:
            await self.initialize(ctx)
        
        doc_id = doc_id or str(uuid.uuid4())
        
        # Convert dicts to Entity objects
        entity_objects = []
        for e in entities:
            entity_type = EntityType(e.get("entity_type", "custom"))
            entity_objects.append(Entity(
                entity_id=e.get("entity_id", str(uuid.uuid4())),
                text=e.get("text", ""),
                normalized=e.get("normalized", e.get("text", "")),
                entity_type=entity_type,
                confidence=e.get("confidence", 1.0),
            ))
        
        relations = await self._delegator.extract_relations(text, entity_objects, doc_id, language)
        
        return {
            "doc_id": doc_id,
            "relation_count": len(relations),
            "relations": [r.to_dict() for r in relations],
        }
    
    async def build_graph(
        self,
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
        language: str = "auto",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Build knowledge graph from texts."""
        if not self._initialized:
            await self.initialize(ctx)
        
        if doc_ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in texts]
        
        total_entities = 0
        total_relations = 0
        
        for text, doc_id in zip(texts, doc_ids):
            result = await self._delegator.extract_combined(text, doc_id, language)
            
            # Add to graph
            for entity in result.entities:
                self._knowledge_graph.add_entity(entity)
                total_entities += 1
            
            for relation in result.relations:
                self._knowledge_graph.add_relation(relation)
                total_relations += 1
        
        if self._metrics:
            await self._metrics.record_extraction(total_entities, total_relations, 0)
        
        return {
            "documents_processed": len(texts),
            "entities_added": total_entities,
            "relations_added": total_relations,
            "graph_stats": self._knowledge_graph.get_stats(),
        }
    
    async def search_entities(
        self,
        query: str,
        top_k: int = 10,
        entity_types: Optional[List[str]] = None,
        fuzzy: bool = True,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Search entities in the knowledge graph."""
        if not self._initialized:
            await self.initialize(ctx)
        
        types = None
        if entity_types:
            types = [EntityType(t) for t in entity_types if t in get_entity_types()]
        
        results = self._knowledge_graph.search_entities(
            query=query,
            top_k=top_k,
            entity_types=types,
            fuzzy=fuzzy,
        )
        
        return {
            "query": query,
            "result_count": len(results),
            "results": [
                {"entity": e.to_dict(), "score": round(s, 3)}
                for e, s in results
            ],
        }
    
    async def get_subgraph(
        self,
        entity_ids: List[str],
        max_depth: int = 2,
        max_nodes: int = 100,
        relation_types: Optional[List[str]] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Extract subgraph around specified entities."""
        if not self._initialized:
            await self.initialize(ctx)
        
        types = None
        if relation_types:
            types = [RelationType(t) for t in relation_types if t in get_relation_types()]
        
        subgraph = self._knowledge_graph.extract_subgraph(
            seed_entity_ids=entity_ids,
            max_depth=max_depth,
            max_nodes=max_nodes,
            relation_types=types,
        )
        
        return subgraph.to_dict()
    
    async def find_paths(
        self,
        source_entity_id: str,
        target_entity_id: str,
        max_length: int = 5,
        max_paths: int = 10,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Find paths between two entities."""
        if not self._initialized:
            await self.initialize(ctx)
        
        paths = self._knowledge_graph.find_paths(
            source_id=source_entity_id,
            target_id=target_entity_id,
            max_length=max_length,
            max_paths=max_paths,
        )
        
        return {
            "source_id": source_entity_id,
            "target_id": target_entity_id,
            "path_count": len(paths),
            "paths": [p.to_dict() for p in paths],
        }
    
    async def get_graph_stats(self, ctx: Any = None) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        if not self._initialized:
            await self.initialize(ctx)
        
        return self._knowledge_graph.get_stats()
    
    async def export_graph(
        self,
        format: str = "triples",
        max_triples: int = 10000,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Export knowledge graph."""
        if not self._initialized:
            await self.initialize(ctx)
        
        if format == "triples":
            triples = self._knowledge_graph.to_triples()[:max_triples]
            return {
                "format": "triples",
                "count": len(triples),
                "triples": [t.to_dict() for t in triples],
            }
        elif format == "json":
            return {
                "format": "json",
                "stats": self._knowledge_graph.get_stats(),
                "entities": [e.to_dict() for e in list(self._knowledge_graph._entities.values())[:1000]],
                "relations": [r.to_dict() for r in list(self._knowledge_graph._relations.values())[:5000]],
            }
        else:
            return {"error": f"Unknown format: {format}"}
    
    async def clear_graph(self, ctx: Any = None) -> Dict[str, Any]:
        """Clear the knowledge graph."""
        if not self._initialized:
            await self.initialize(ctx)
        
        self._knowledge_graph.clear()
        
        return {"status": "cleared", "graph_stats": self._knowledge_graph.get_stats()}
    
    async def add_entity(
        self,
        text: str,
        normalized: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Add a single entity to the graph."""
        if not self._initialized:
            await self.initialize(ctx)
        
        try:
            etype = EntityType(entity_type)
        except ValueError:
            etype = EntityType.CUSTOM
        
        entity = Entity(
            entity_id=str(uuid.uuid4()),
            text=text,
            normalized=normalized,
            entity_type=etype,
            properties=properties or {},
        )
        
        success = self._knowledge_graph.add_entity(entity)
        
        return {
            "success": success,
            "entity": entity.to_dict() if success else None,
        }
    
    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        evidence: str = "",
        properties: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Add a relation between entities."""
        if not self._initialized:
            await self.initialize(ctx)
        
        try:
            rtype = RelationType(relation_type)
        except ValueError:
            rtype = RelationType.RELATED_TO
        
        relation = Relation(
            relation_id=str(uuid.uuid4()),
            source_id=source_id,
            target_id=target_id,
            relation_type=rtype,
            evidence=evidence,
            properties=properties or {},
        )
        
        success = self._knowledge_graph.add_relation(relation)
        
        return {
            "success": success,
            "relation": relation.to_dict() if success else None,
        }
    
    async def get_session(self, session_id: str, ctx: Any = None) -> Dict[str, Any]:
        """Get session state."""
        if not self._initialized:
            await self.initialize(ctx)
        
        session = await self._session_manager.get_session(session_id)
        if session:
            return session.to_dict()
        return {"error": "Session not found"}
    
    async def delete_session(self, session_id: str, ctx: Any = None) -> Dict[str, Any]:
        """Delete a session."""
        if not self._initialized:
            await self.initialize(ctx)
        
        deleted = await self._session_manager.delete_session(session_id)
        return {"deleted": deleted, "session_id": session_id}
    
    async def get_stats(self, ctx: Any = None) -> Dict[str, Any]:
        """Get metrics and statistics."""
        if not self._initialized:
            await self.initialize(ctx)
        
        return {
            "graph_stats": self._knowledge_graph.get_stats(),
            "metrics": self._metrics.get_metrics() if self._metrics else {},
            "cache": self._cache.get_stats() if self._cache else {},
        }
    
    async def reload_config(self, ctx: Any = None) -> Dict[str, Any]:
        """Hot-reload configuration."""
        try:
            self._config = self._load_config()
            self._build_configs()
            
            return {"status": "reloaded"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def shutdown(self, ctx: Any = None) -> Dict[str, Any]:
        """Graceful shutdown."""
        try:
            self._initialized = False
            
            if self._event_bus:
                await self._event_bus.publish(
                    "graph.shutdown",
                    {"module": "graph_rag"},
                )
            
            logger.info("Graph-RAG pipeline shut down")
            return {"status": "shutdown"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def health_check(self, ctx: Any = None) -> Dict[str, Any]:
        """Check component health."""
        if not self._initialized:
            return {"module": "graph_rag", "status": "not_initialized"}
        
        llm_health = await self._delegator.health_check()
        
        return {
            "module": "graph_rag",
            "status": "healthy" if llm_health.get("status") == "available" else "degraded",
            "initialized": self._initialized,
            "llm_delegation": llm_health,
            "graph_stats": self._knowledge_graph.get_stats(),
            "cache": self._cache.get_stats() if self._cache else None,
        }
    
    async def get_entity_types(self, ctx: Any = None) -> Dict[str, Any]:
        """Get available entity types."""
        return {"entity_types": get_entity_types()}
    
    async def get_relation_types(self, ctx: Any = None) -> Dict[str, Any]:
        """Get available relation types."""
        return {"relation_types": get_relation_types()}


# Import uuid for entity/relation creation
import uuid
