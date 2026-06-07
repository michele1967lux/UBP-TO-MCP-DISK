"""
retrieval_strategy/adapter.py

Bridge layer that exposes all retrieval strategy operations to the UBP system.

Operations:
- initialize: Start components
- retrieve: Execute retrieval with specified strategy
- hybrid_retrieve: Force hybrid strategy
- hierarchical_retrieve: Force hierarchical strategy
- multi_index_retrieve: Search across multiple indexes
- router_retrieve: LLM-routed retrieval
- bm25_search: BM25-only search
- add_documents: Add documents to index
- add_hierarchical_documents: Add documents with hierarchy
- create_index: Create a new named index
- get_indexes: List available indexes
- classify_query: Classify query without retrieval
- get_stats: Get metrics (admin)
- clear_index: Clear an index (admin)
- reload_config: Hot-reload (admin)
- shutdown: Graceful shutdown
- health_check: Component health

v1.0.0: Initial release with full enterprise features
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from ubp_enterprise_hybrid.modules.cores._shared.constants import SYSTEM_COLLECTIONS

from .providers import (
    # Data classes
    RetrievalResult,
    RetrievalResponse,
    RouterDecision,
    HierarchicalChunk,
    # Enums
    RetrievalStrategy,
    FusionMethod,
    HierarchyLevel,
    QueryClass,
    # Indexes
    BM25Index,
    HierarchicalIndex,
    IndexRegistry,
    # Configs
    BM25Config,
    VectorConfig,
    HybridConfig,
    HierarchicalConfig,
    RouterConfig,
    CacheConfig,
    MetricsConfig,
    DebugConfig,
    # Providers
    RetrievalCacheProvider,
    RetrievalMetricsCollector,
)
from .fusion import FusionEngine, FusionMethod, RRFConfig, WeightedConfig
from .router import QueryRouter, StrategySelector
from .strategies import (
    BaseRetrievalStrategy,
    HybridRetrievalStrategy,
    HierarchicalRetrievalStrategy,
    MultiIndexRetrievalStrategy,
    RouterRetrievalStrategy,
    StrategyFactory,
)

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class IModuleRegistry(Protocol):
    """Protocol for module registry."""
    def get_module(self, module_name: str) -> Optional[Any]: ...


# ============================================================================
# DI Container Wrapper
# ============================================================================


class DIContainerModuleRegistry:
    """Wraps DI container to provide module registry interface."""
    
    def __init__(self, di_container: Optional[Any] = None):
        self._container = di_container
        self._cached_modules: Dict[str, Any] = {}
    
    def get_module(self, module_name: str) -> Optional[Any]:
        """Get a module by name."""
        if module_name in self._cached_modules:
            return self._cached_modules[module_name]
        
        if not self._container:
            return None
        
        module = None
        
        if hasattr(self._container, "get"):
            try:
                module = self._container.get(module_name)
            except Exception:
                pass
        
        if not module and hasattr(self._container, "resolve"):
            try:
                module = self._container.resolve(module_name)
            except Exception:
                pass
        
        if not module and hasattr(self._container, module_name):
            module = getattr(self._container, module_name)
        
        if module:
            self._cached_modules[module_name] = module
        
        return module


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
    """Recursively coerce configuration values."""
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
    """Coerce a single value."""
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
# Retrieval Strategy Adapter
# ============================================================================


class RetrievalStrategyAdapter:
    """
    Adapter exposing retrieval strategy operations.
    
    Features:
    - Multiple retrieval strategies
    - Hybrid (BM25 + Vector) retrieval
    - Hierarchical retrieval
    - Multi-index support
    - LLM-based routing
    - Caching
    - Metrics
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
        self._bm25_config: Optional[BM25Config] = None
        self._vector_config: Optional[VectorConfig] = None
        self._hybrid_config: Optional[HybridConfig] = None
        self._hierarchical_config: Optional[HierarchicalConfig] = None
        self._router_config: Optional[RouterConfig] = None
        self._cache_config: Optional[CacheConfig] = None
        self._metrics_config: Optional[MetricsConfig] = None
        self._debug_config: Optional[DebugConfig] = None
        
        # Components
        self._bm25_index: Optional[BM25Index] = None
        self._hierarchical_index: Optional[HierarchicalIndex] = None
        self._index_registry: Optional[IndexRegistry] = None
        self._cache: Optional[RetrievalCacheProvider] = None
        self._metrics: Optional[RetrievalMetricsCollector] = None
        self._router: Optional[QueryRouter] = None
        self._strategy_factory: Optional[StrategyFactory] = None
        self._reranker: Optional[Any] = None  # v3.7.0: Optional rag_reranker module
        self._qdrant_module: Optional[Any] = None  # Cached rag_qdrant module
        
        # Strategies
        self._strategies: Dict[RetrievalStrategy, BaseRetrievalStrategy] = {}
        
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
    # Configuration
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
    
    def _build_configs(self) -> None:
        """Build configuration objects."""
        cfg = self._config
        
        bm25_cfg = cfg.get("bm25_retriever", {})
        self._bm25_config = BM25Config(
            k1=bm25_cfg.get("k1", 1.5),
            b=bm25_cfg.get("b", 0.75),
            top_k=bm25_cfg.get("top_k", 50),
            stopwords=bm25_cfg.get("stopwords", True),
            stemming=bm25_cfg.get("stemming", True),
            lowercase=bm25_cfg.get("lowercase", True),
            language=bm25_cfg.get("language", "auto"),
        )
        
        vector_cfg = cfg.get("vector_retriever", {})
        self._vector_config = VectorConfig(
            embedding_module=vector_cfg.get("embedding_module", "embedding_service"),
            embedding_operation=vector_cfg.get("embedding_operation", "embed"),
            vector_store_module=vector_cfg.get("vector_store_module", "qdrant_store"),
            vector_store_operation=vector_cfg.get("vector_store_operation", "search"),
            top_k=vector_cfg.get("top_k", 50),
        )
        
        hybrid_cfg = cfg.get("hybrid_retrieval", {})
        self._hybrid_config = HybridConfig(
            fusion_method=hybrid_cfg.get("fusion_method", "rrf"),
            alpha=hybrid_cfg.get("alpha", 0.5),
            bm25_weight=hybrid_cfg.get("bm25_weight", 0.4),
            vector_weight=hybrid_cfg.get("vector_weight", 0.6),
            normalize_scores=hybrid_cfg.get("normalize_scores", True),
            deduplicate=hybrid_cfg.get("deduplicate", True),
        )
        
        hier_cfg = cfg.get("hierarchical_retrieval", {}).get("levels", {})
        self._hierarchical_config = HierarchicalConfig(
            document_chunk_size=hier_cfg.get("document", {}).get("chunk_size", 4000),
            document_top_k=hier_cfg.get("document", {}).get("top_k", 5),
            section_chunk_size=hier_cfg.get("section", {}).get("chunk_size", 1000),
            section_top_k=hier_cfg.get("section", {}).get("top_k", 10),
            paragraph_chunk_size=hier_cfg.get("paragraph", {}).get("chunk_size", 300),
            paragraph_top_k=hier_cfg.get("paragraph", {}).get("top_k", 20),
        )
        
        router_cfg = cfg.get("router_retrieval", {})
        self._router_config = RouterConfig(
            llm_module=router_cfg.get("llm_module", "inference_ollama_grok"),
            llm_operation=router_cfg.get("llm_operation", "generate"),
            temperature=router_cfg.get("temperature", 0.1),
            timeout_seconds=router_cfg.get("timeout_seconds", 10),
            fallback_strategy=router_cfg.get("fallback_strategy", "hybrid"),
            cache_decisions=router_cfg.get("cache_decisions", True),
        )
        
        cache_cfg = cfg.get("cache", {})
        self._cache_config = CacheConfig(
            enabled=cache_cfg.get("enabled", True),
            ttl_seconds=cache_cfg.get("ttl_seconds", 1800),
            semantic_cache=cache_cfg.get("semantic_cache", True),
            semantic_threshold=cache_cfg.get("semantic_threshold", 0.95),
        )
        
        metrics_cfg = cfg.get("metrics", {})
        self._metrics_config = MetricsConfig(
            enabled=metrics_cfg.get("enabled", True),
            track_latency=metrics_cfg.get("track_latency", True),
        )
        
        debug_cfg = cfg.get("debug", {})
        self._debug_config = DebugConfig(
            enabled=debug_cfg.get("enabled", False),
            log_queries=debug_cfg.get("log_queries", True),
            log_scores=debug_cfg.get("log_scores", True),
        )
    
    # ========================================================================
    # Operations
    # ========================================================================
    
    async def initialize(self, ctx: Any = None) -> Dict[str, Any]:
        """Initialize retrieval strategy components."""
        if self._initialized:
            return {"status": "already_initialized"}
        
        try:
            self._config = self._load_config()
            self._build_configs()
            
            # Initialize indexes
            self._bm25_index = BM25Index(self._bm25_config)
            self._hierarchical_index = HierarchicalIndex(self._hierarchical_config)
            self._index_registry = IndexRegistry()
            self._index_registry.register_bm25_index("default", self._bm25_index)
            
            # Try to get Redis
            if self._di_container:
                self._redis_client = getattr(self._di_container, "redis", None)
            
            # Initialize cache and metrics
            self._cache = RetrievalCacheProvider(self._cache_config, self._redis_client)
            self._metrics = RetrievalMetricsCollector(self._metrics_config)
            
            # Initialize router
            available_indexes = list(self._index_registry.list_indexes().get("bm25", []))
            self._router = QueryRouter(
                config=self._router_config,
                module_registry=self._module_registry,
                available_indexes=available_indexes,
            )
            
            # Initialize strategy factory
            self._strategy_factory = StrategyFactory(
                module_registry=self._module_registry,
                bm25_index=self._bm25_index,
                hierarchical_index=self._hierarchical_index,
                index_registry=self._index_registry,
                router=self._router,
            )
            
            # v3.7.0: Try to resolve rag_reranker for optional reranking support
            try:
                if self._di_container and hasattr(self._di_container, 'resolve'):
                    self._reranker = await self._di_container.resolve("rag_reranker")
                    logger.info("✅ rag_reranker resolved for retrieval_strategy")
                else:
                    self._reranker = self._module_registry.get_module("rag_reranker")
                    if self._reranker:
                        logger.info("✅ rag_reranker found via module registry")
            except Exception as e:
                logger.debug(f"rag_reranker not available for retrieval_strategy: {e}")
                self._reranker = None
            
            # Create strategies
            self._strategies = {
                RetrievalStrategy.HYBRID: self._strategy_factory.create_hybrid(
                    self._hybrid_config, self._vector_config
                ),
                RetrievalStrategy.HIERARCHICAL: self._strategy_factory.create_hierarchical(
                    self._hierarchical_config
                ),
                RetrievalStrategy.MULTI_INDEX: self._strategy_factory.create_multi_index(),
            }
            
            self._initialized = True
            
            logger.info("Retrieval strategy adapter initialized")
            
            if self._event_bus:
                await self._event_bus.publish(
                    "retrieval.initialized",
                    {"module": "retrieval_strategy", "status": "success"},
                )
            
            return {
                "status": "initialized",
                "strategies": [s.value for s in self._strategies.keys()],
                "indexes": self._index_registry.list_indexes(),
            }
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _retrieve_from_collections(
        self,
        query: str,
        collections: List[str],
        top_k: int = 10,
        rerank: bool = False,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Retrieve from Qdrant collections via rag_qdrant module.
        
        v3.7.1: Added deduplication and reranking support to restore
        feature parity with strategy-based retrieval.
        """
        try:
            # Resolve rag_qdrant — use DI container directly (async resolve)
            if not hasattr(self, '_qdrant_module') or self._qdrant_module is None:
                if self._di_container and hasattr(self._di_container, 'resolve'):
                    self._qdrant_module = await self._di_container.resolve("rag_qdrant")
                    # Validate resolution succeeded
                    if self._qdrant_module is None:
                        logger.error("[RETRIEVAL] Failed to resolve rag_qdrant from DI container")
                        return {"results": [], "count": 0, "collections": collections}
                else:
                    logger.error("[RETRIEVAL] No DI container to resolve rag_qdrant")
                    return {"results": [], "count": 0, "collections": collections}

            # Pre-check: get current embedding dimension to detect incompatible collections
            active_dim = None
            qdrant_client = None
            try:
                provider = getattr(self._qdrant_module, 'provider', None)
                if provider:
                    emb_mgr = getattr(provider, 'embedding_manager', None)
                    if emb_mgr:
                        active_dim = getattr(emb_mgr, 'dimension', None)
                    # Access client for collection dimension checks
                    qdrant_client = getattr(provider, 'client', None)
            except Exception as e:
                logger.debug(f"[RETRIEVAL] Could not access qdrant client for dimension check: {e}")
                qdrant_client = None

            all_results = []
            queried_collections = []
            # Defensive: never retrieve from system/internal collections
            collections = [c for c in collections if c not in SYSTEM_COLLECTIONS]
            for collection in collections:
                try:
                    # Dimension compatibility check — skip if mismatch
                    if active_dim and qdrant_client:
                        try:
                            coll_dim = await qdrant_client.get_vector_dimension_safe(collection)
                            if coll_dim and coll_dim != active_dim:
                                logger.warning(
                                    f"[RETRIEVAL] Skipping collection '{collection}': "
                                    f"dimension mismatch ({coll_dim} vs active {active_dim})"
                                )
                                continue
                        except Exception as dim_err:
                            logger.warning(f"[RETRIEVAL] Dimension check failed for {collection}: {dim_err}")
                            # If check fails, proceed and let query_internal handle it

                    result = await self._qdrant_module.query_internal(
                        query_text=query,
                        collection=collection,
                        top_k=top_k,
                    )
                    if result and isinstance(result, dict):
                        chunks = result.get("results", [])
                        # Tag each chunk with source collection
                        for chunk in chunks:
                            if isinstance(chunk, dict):
                                chunk["collection"] = collection
                        all_results.extend(chunks)
                        queried_collections.append(collection)
                except RuntimeError as e:
                    if "dimension" in str(e).lower() or "incompatible" in str(e).lower():
                        logger.warning(
                            f"[RETRIEVAL] Skipping collection '{collection}': {e}"
                        )
                    else:
                        logger.error(f"[RETRIEVAL] Runtime error querying collection {collection}: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"[RETRIEVAL] Error querying collection {collection}: {e}", exc_info=True)

            # v3.7.1: Add deduplication (same as hybrid strategy)
            if all_results:
                from .fusion import deduplicate_results_dict_format
                before_dedup = len(all_results)
                all_results = deduplicate_results_dict_format(all_results, similarity_threshold=0.95)
                if len(all_results) < before_dedup:
                    logger.info(f"[RETRIEVAL] Deduplicated {before_dedup} → {len(all_results)} results")

            # Sort by score descending with deterministic tiebreaker on id
            all_results.sort(
                key=lambda x: (-round(x.get("score", 0), 6), x.get("id", "")),
            )
            all_results = all_results[:top_k]

            logger.info(f"[RETRIEVAL] {len(all_results)} results from {queried_collections}")

            # v3.7.1: Apply reranking if requested (same as strategy path)
            if rerank and self._reranker and len(all_results) > 15:
                try:
                    logger.info(f"[RETRIEVAL] Applying reranking to {len(all_results)} collection results")
                    
                    # Convert to chunks format expected by reranker
                    chunks = []
                    for result in all_results:
                        chunks.append({
                            "text": result.get("text", result.get("content", "")),
                            "content": result.get("text", result.get("content", "")),
                            "doc_id": result.get("id", result.get("doc_id")),
                            "score": result.get("score", 0),
                            "metadata": result.get("metadata", {}),
                        })
                    
                    # Call reranker
                    rerank_result = await self._reranker.rerank_internal(
                        query=query,
                        chunks=chunks,
                        top_k=top_k,
                        return_scores=True,
                    )
                    
                    # Update results with reranked order and scores
                    if rerank_result and "reranked_chunks" in rerank_result:
                        reranked_chunks = rerank_result["reranked_chunks"]
                        
                        # Map back to original format
                        reranked_results = []
                        for chunk in reranked_chunks:
                            # Find original result
                            doc_id = chunk.get("doc_id")
                            original = next(
                                (r for r in all_results if r.get("id") == doc_id or r.get("doc_id") == doc_id),
                                None
                            )
                            if original:
                                # Update with rerank score
                                reranked_result = {**original}
                                reranked_result["score"] = chunk.get("rerank_score", original.get("score", 0))
                                if "metadata" not in reranked_result:
                                    reranked_result["metadata"] = {}
                                reranked_result["metadata"]["rerank_score"] = chunk.get("rerank_score")
                                reranked_result["metadata"]["original_score"] = original.get("score")
                                reranked_results.append(reranked_result)
                        
                        all_results = reranked_results
                        logger.info(f"[RETRIEVAL] Reranking complete: {len(all_results)} results")
                        
                except Exception as rerank_err:
                    logger.error(f"[RETRIEVAL] Reranking failed, using original scores: {rerank_err}", exc_info=True)

            return {
                "results": all_results,
                "count": len(all_results),
                "collections": queried_collections,
            }

        except Exception as e:
            logger.error(f"[RETRIEVAL] Failed to retrieve from collections: {e}", exc_info=True)
            return {"results": [], "count": 0, "collections": collections}

    async def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 10,
        collections: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        language: str = "en",
        rerank: bool = False,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Execute retrieval with specified strategy.
        
        Args:
            query: Search query
            strategy: Strategy to use (hybrid, hierarchical, multi_index, router)
            top_k: Number of results
            collections: Qdrant collections to search (ACL-validated by caller)
            filters: Optional metadata filters
            language: Query language
            rerank: Whether to apply reranking (v3.7.0 - now implemented via rag_reranker)
            ctx: Security context
        
        Returns:
            RetrievalResponse as dict
        """
        # When collections are provided, use collection-specific path
        # v3.7.1: Now includes deduplication and reranking for feature parity
        if collections:
            return await self._retrieve_from_collections(
                query=query,
                collections=collections,
                top_k=top_k,
                rerank=rerank,
                ctx=ctx,
            )

        if not self._initialized:
            await self.initialize(ctx)
        
        # Check cache
        if self._cache:
            cached = await self._cache.get(query)
            if cached:
                if self._metrics:
                    await self._metrics.record_retrieval(
                        strategy=cached.strategy_used,
                        fusion_method=cached.fusion_method,
                        latency_ms=0,
                        result_count=len(cached.results),
                        cache_hit=True,
                    )
                return cached.to_dict()
        
        # Parse strategy
        try:
            strategy_enum = RetrievalStrategy(strategy.lower())
        except ValueError:
            strategy_enum = RetrievalStrategy.HYBRID
        
        # Get strategy implementation
        strategy_impl = self._strategies.get(strategy_enum)
        if not strategy_impl:
            strategy_impl = self._strategies.get(RetrievalStrategy.HYBRID)
        
        # Execute retrieval
        response = await strategy_impl.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
            language=language,
        )
        
        # v3.7.0: Apply reranking if requested and reranker available
        if rerank and self._reranker and len(response.results) > 5:
            try:
                logger.info(f"Applying reranking to {len(response.results)} results")
                
                # Convert to chunks format expected by reranker
                # Note: Both 'text' and 'content' fields are included for compatibility
                # with different reranker implementations
                chunks = []
                for result in response.results:
                    chunks.append({
                        "text": result.text,
                        "content": result.text,
                        "doc_id": result.id,
                        "score": result.score,
                        "metadata": result.metadata,
                    })
                
                # Call reranker
                rerank_result = await self._reranker.rerank_internal(
                    query=query,
                    chunks=chunks,
                    top_k=top_k,
                    return_scores=True,
                )
                
                # Update results with reranked order and scores
                if rerank_result and "reranked_chunks" in rerank_result:
                    reranked_chunks = rerank_result["reranked_chunks"]
                    
                    # Convert back to RetrievalResult format
                    reranked_results = []
                    unmatched_count = 0
                    for chunk in reranked_chunks:
                        # Find original result to preserve other fields
                        original = next(
                            (r for r in response.results if r.id == chunk.get("doc_id")),
                            None
                        )
                        if original:
                            # Update score with rerank score
                            reranked_results.append(
                                RetrievalResult(
                                    id=original.id,
                                    text=original.text,
                                    score=chunk.get("rerank_score", original.score),
                                    metadata={
                                        **original.metadata,
                                        "rerank_score": chunk.get("rerank_score"),
                                        "original_score": original.score,
                                    },
                                    chunk_type=original.chunk_type,
                                    chunk_id=original.chunk_id,
                                )
                            )
                        else:
                            unmatched_count += 1
                            logger.warning(f"Could not match reranked chunk to original result: {chunk.get('doc_id')}")
                    
                    if unmatched_count > 0:
                        logger.warning(f"Reranking: {unmatched_count} chunks could not be matched to original results")
                    
                    # Update response with reranked results
                    response.results = reranked_results
                    response.metadata["reranked"] = True
                    response.metadata["reranker_model"] = rerank_result.get("model_used", "unknown")
                    logger.info(f"Reranking complete: {len(reranked_results)} results")
                    
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")
                # Continue with original results on error
        
        # Cache result
        if self._cache:
            await self._cache.set(query, response)
        
        # Record metrics
        if self._metrics:
            await self._metrics.record_retrieval(
                strategy=response.strategy_used,
                fusion_method=response.fusion_method,
                latency_ms=response.retrieval_time_ms,
                result_count=len(response.results),
            )
        
        return response.to_dict()
    
    async def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 10,
        fusion_method: str = "rrf",
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Force hybrid retrieval strategy."""
        return await self.retrieve(
            query=query,
            strategy="hybrid",
            top_k=top_k,
            filters=filters,
            ctx=ctx,
        )
    
    async def hierarchical_retrieve(
        self,
        query: str,
        top_k: int = 10,
        levels: Optional[List[str]] = None,
        expand_context: bool = True,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Force hierarchical retrieval strategy."""
        if not self._initialized:
            await self.initialize(ctx)
        
        strategy = self._strategies.get(RetrievalStrategy.HIERARCHICAL)
        if not strategy:
            return {"error": "Hierarchical strategy not available"}
        
        level_enums = None
        if levels:
            level_enums = [HierarchyLevel(l) for l in levels if l in ["document", "section", "paragraph"]]
        
        response = await strategy.retrieve(
            query=query,
            top_k=top_k,
            levels=level_enums,
            expand_context=expand_context,
        )
        
        return response.to_dict()
    
    async def multi_index_retrieve(
        self,
        query: str,
        top_k: int = 10,
        index_names: Optional[List[str]] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Search across multiple indexes."""
        if not self._initialized:
            await self.initialize(ctx)
        
        strategy = self._strategies.get(RetrievalStrategy.MULTI_INDEX)
        if not strategy:
            return {"error": "Multi-index strategy not available"}
        
        response = await strategy.retrieve(
            query=query,
            top_k=top_k,
            index_names=index_names,
        )
        
        return response.to_dict()
    
    async def router_retrieve(
        self,
        query: str,
        top_k: int = 10,
        language: str = "en",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """LLM-routed retrieval."""
        if not self._initialized:
            await self.initialize(ctx)
        
        router_strategy = RouterRetrievalStrategy(
            router=self._router,
            strategies=self._strategies,
            fallback_strategy=RetrievalStrategy.HYBRID,
        )
        
        response = await router_strategy.retrieve(
            query=query,
            top_k=top_k,
            language=language,
        )
        
        if self._metrics:
            decision = response.metadata.get("routing_decision", {})
            query_class = QueryClass(decision.get("query_class", "unknown"))
            await self._metrics.record_router_decision(query_class)
        
        return response.to_dict()
    
    async def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        index_name: str = "default",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """BM25-only search."""
        if not self._initialized:
            await self.initialize(ctx)
        
        index = self._index_registry.get_bm25_index(index_name)
        if not index:
            return {"error": f"Index '{index_name}' not found"}
        
        results = index.search(query, top_k)
        
        return {
            "query": query,
            "index": index_name,
            "result_count": len(results),
            "results": [
                {
                    "doc_id": doc_id,
                    "score": round(score, 4),
                    "content": (index.get_document(doc_id) or "")[:500],
                }
                for doc_id, score in results
            ],
        }
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        index_name: str = "default",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Add documents to BM25 index.
        
        Args:
            documents: List of {"id": str, "content": str, "metadata": dict}
            index_name: Target index name
        """
        if not self._initialized:
            await self.initialize(ctx)
        
        index = self._index_registry.get_bm25_index(index_name)
        if not index:
            # Create new index
            index = BM25Index(self._bm25_config)
            self._index_registry.register_bm25_index(index_name, index)
        
        added = 0
        for doc in documents:
            doc_id = doc.get("id") or str(uuid.uuid4())
            content = doc.get("content", "")
            if content:
                index.add_document(doc_id, content)
                added += 1
        
        return {
            "index": index_name,
            "documents_added": added,
            "index_stats": index.get_stats(),
        }
    
    async def add_hierarchical_documents(
        self,
        documents: List[Dict[str, Any]],
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Add documents to hierarchical index with chunking.
        
        Args:
            documents: List of {"id": str, "content": str, "metadata": dict}
        """
        if not self._initialized:
            await self.initialize(ctx)
        
        total_counts = {"document": 0, "section": 0, "paragraph": 0}
        
        for doc in documents:
            doc_id = doc.get("id") or str(uuid.uuid4())
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            if content:
                counts = self._hierarchical_index.add_document(doc_id, content, metadata)
                for level, count in counts.items():
                    total_counts[level] += count
        
        return {
            "documents_processed": len(documents),
            "chunks_created": total_counts,
            "index_stats": self._hierarchical_index.get_stats(),
        }
    
    async def create_index(
        self,
        index_name: str,
        index_type: str = "bm25",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Create a new named index."""
        if not self._initialized:
            await self.initialize(ctx)
        
        if index_type == "bm25":
            index = BM25Index(self._bm25_config)
            self._index_registry.register_bm25_index(index_name, index)
        elif index_type == "hierarchical":
            index = HierarchicalIndex(self._hierarchical_config)
            self._index_registry.register_hierarchical_index(index_name, index)
        else:
            return {"error": f"Unknown index type: {index_type}"}
        
        # Update router available indexes
        if self._router:
            self._router.available_indexes = list(
                self._index_registry.list_indexes().get("bm25", [])
            )
        
        return {
            "created": True,
            "index_name": index_name,
            "index_type": index_type,
        }
    
    async def get_indexes(self, ctx: Any = None) -> Dict[str, Any]:
        """List available indexes."""
        if not self._initialized:
            await self.initialize(ctx)
        
        indexes = self._index_registry.list_indexes()
        
        stats = {}
        for idx_name in indexes.get("bm25", []):
            idx = self._index_registry.get_bm25_index(idx_name)
            if idx:
                stats[idx_name] = idx.get_stats()
        
        return {
            "indexes": indexes,
            "stats": stats,
        }
    
    async def classify_query(
        self,
        query: str,
        language: str = "en",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Classify query without executing retrieval."""
        if not self._initialized:
            await self.initialize(ctx)
        
        decision = await self._router.route(query, language)
        return decision.to_dict()
    
    async def get_stats(self, ctx: Any = None) -> Dict[str, Any]:
        """Get metrics and statistics."""
        if not self._initialized:
            await self.initialize(ctx)
        
        return {
            "metrics": self._metrics.get_metrics() if self._metrics else {},
            "cache": self._cache.get_stats() if self._cache else {},
            "indexes": self._index_registry.list_indexes(),
        }
    
    async def clear_index(
        self,
        index_name: str = "default",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Clear an index."""
        if not self._initialized:
            await self.initialize(ctx)
        
        index = self._index_registry.get_bm25_index(index_name)
        if index:
            index.clear()
            return {"cleared": True, "index": index_name}
        
        return {"error": f"Index '{index_name}' not found"}
    
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
        self._initialized = False
        
        if self._event_bus:
            await self._event_bus.publish(
                "retrieval.shutdown",
                {"module": "retrieval_strategy"},
            )
        
        logger.info("Retrieval strategy adapter shut down")
        return {"status": "shutdown"}
    
    async def health_check(self, ctx: Any = None) -> Dict[str, Any]:
        """Check component health."""
        if not self._initialized:
            return {"module": "retrieval_strategy", "status": "not_initialized"}
        
        return {
            "module": "retrieval_strategy",
            "status": "healthy",
            "initialized": self._initialized,
            "strategies_available": [s.value for s in self._strategies.keys()],
            "indexes": self._index_registry.list_indexes(),
        }
    
    async def get_available_strategies(self, ctx: Any = None) -> Dict[str, Any]:
        """Get list of available strategies."""
        return {
            "strategies": [
                {"name": "hybrid", "description": "BM25 + Vector with fusion"},
                {"name": "hierarchical", "description": "Multi-level document retrieval"},
                {"name": "multi_index", "description": "Search across multiple indexes"},
                {"name": "router", "description": "LLM-routed strategy selection"},
                {"name": "bm25", "description": "Keyword-based BM25 only"},
            ],
            "fusion_methods": [f.value for f in FusionMethod],
            "hierarchy_levels": [l.value for l in HierarchyLevel],
        }
