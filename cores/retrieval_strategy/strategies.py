"""
retrieval_strategy/strategies.py

Strategy implementations for retrieval operations.

Implements:
- HybridRetrievalStrategy
- HierarchicalRetrievalStrategy
- MultiIndexRetrievalStrategy
- RouterRetrievalStrategy

v1.0.0: Initial release
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .providers import (
    RetrievalResult,
    RetrievalResponse,
    RetrievalStrategy,
    FusionMethod,
    HierarchyLevel,
    QueryClass,
    RouterDecision,
    BM25Index,
    HierarchicalIndex,
    IndexRegistry,
    BM25Config,
    VectorConfig,
    HybridConfig,
    HierarchicalConfig,
)
from .fusion import FusionEngine, RRFConfig, WeightedConfig, deduplicate_results
from .router import QueryRouter, StrategySelector

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class IModuleRegistry(Protocol):
    """Protocol for module registry."""
    def get_module(self, module_name: str) -> Optional[Any]: ...


class IVectorRetriever(Protocol):
    """Protocol for vector retrieval."""
    async def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, str]]: ...


# ============================================================================
# Base Strategy
# ============================================================================


class BaseRetrievalStrategy(ABC):
    """Base class for retrieval strategies."""
    
    @property
    @abstractmethod
    def strategy_type(self) -> RetrievalStrategy:
        """Return the strategy type."""
        pass
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> RetrievalResponse:
        """Execute retrieval."""
        pass


# ============================================================================
# Hybrid Retrieval Strategy
# ============================================================================


class HybridRetrievalStrategy(BaseRetrievalStrategy):
    """
    Hybrid retrieval combining BM25 and vector search.
    
    Features:
    - BM25 keyword retrieval
    - Vector semantic retrieval
    - Multiple fusion methods (RRF, weighted, etc.)
    - Score normalization
    - Deduplication
    """
    
    @property
    def strategy_type(self) -> RetrievalStrategy:
        return RetrievalStrategy.HYBRID
    
    def __init__(
        self,
        bm25_index: BM25Index,
        module_registry: IModuleRegistry,
        config: HybridConfig,
        vector_config: VectorConfig,
    ):
        self.bm25_index = bm25_index
        self._module_registry = module_registry
        self.config = config
        self.vector_config = vector_config
        
        # Setup fusion engine
        fusion_method = FusionMethod(config.fusion_method)
        self.fusion_engine = FusionEngine(
            method=fusion_method,
            rrf_config=RRFConfig(
                weight_bm25=config.bm25_weight,
                weight_vector=config.vector_weight,
            ),
            weighted_config=WeightedConfig(
                bm25_weight=config.bm25_weight,
                vector_weight=config.vector_weight,
            ),
            alpha=config.alpha,
        )
        
        self._vector_module: Optional[Any] = None
        self._embedding_module: Optional[Any] = None
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> RetrievalResponse:
        """Execute hybrid retrieval."""
        start_time = time.perf_counter()
        
        # Get expanded top_k for fusion
        retrieval_top_k = top_k * 3
        
        # Run BM25 and vector in parallel
        bm25_task = asyncio.create_task(self._bm25_search(query, retrieval_top_k))
        vector_task = asyncio.create_task(self._vector_search(query, retrieval_top_k, filters))
        
        bm25_results, vector_results = await asyncio.gather(
            bm25_task, vector_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(bm25_results, Exception):
            logger.error(f"BM25 search failed: {bm25_results}")
            bm25_results = []
        if isinstance(vector_results, Exception):
            logger.error(f"Vector search failed: {vector_results}")
            vector_results = []
        
        # Fuse results
        fused_results = self.fusion_engine.fuse(bm25_results, vector_results)
        
        # Deduplicate if enabled
        if self.config.deduplicate:
            content_lookup = {
                doc_id: self.bm25_index.get_document(doc_id) or ""
                for doc_id, _ in fused_results
            }
            fused_results = deduplicate_results(
                fused_results,
                content_lookup,
                self.config.dedupe_threshold,
            )
        
        # Convert to RetrievalResult objects
        results = []
        for rank, (doc_id, score) in enumerate(fused_results[:top_k], start=1):
            content = self.bm25_index.get_document(doc_id) or ""
            results.append(RetrievalResult(
                doc_id=doc_id,
                content=content,
                score=score,
                source="hybrid",
                rank=rank,
            ))
        
        retrieval_time = (time.perf_counter() - start_time) * 1000
        
        return RetrievalResponse(
            query=query,
            results=results,
            strategy_used=RetrievalStrategy.HYBRID,
            fusion_method=FusionMethod(self.config.fusion_method),
            total_candidates=len(bm25_results) + len(vector_results),
            retrieval_time_ms=retrieval_time,
            metadata={
                "bm25_count": len(bm25_results),
                "vector_count": len(vector_results),
                "fusion_method": self.config.fusion_method,
            },
        )
    
    async def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Execute BM25 search."""
        return self.bm25_index.search(query, top_k)
    
    async def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """Execute vector search via delegated modules."""
        # Get vector store module
        vector_module = self._module_registry.get_module(self.vector_config.vector_store_module)
        if not vector_module:
            logger.warning(f"Vector store module '{self.vector_config.vector_store_module}' not available")
            return []
        
        # Get embedding first
        embedding_module = self._module_registry.get_module(self.vector_config.embedding_module)
        if embedding_module:
            try:
                embed_op = getattr(embedding_module, self.vector_config.embedding_operation, None)
                if embed_op:
                    embedding_result = await embed_op(text=query)
                    query_embedding = embedding_result.get("embedding") if isinstance(embedding_result, dict) else None
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                query_embedding = None
        else:
            query_embedding = None
        
        # Search vector store
        try:
            search_op = getattr(vector_module, self.vector_config.vector_store_operation, None)
            if search_op:
                if query_embedding:
                    results = await search_op(
                        query_vector=query_embedding,
                        top_k=top_k,
                        filters=filters,
                    )
                else:
                    results = await search_op(
                        query=query,
                        top_k=top_k,
                        filters=filters,
                    )
                
                # Convert to standard format
                if isinstance(results, list):
                    return [
                        (r.get("id", r.get("doc_id", "")), r.get("score", 0.0))
                        for r in results
                    ]
                elif isinstance(results, dict):
                    return [
                        (r.get("id", r.get("doc_id", "")), r.get("score", 0.0))
                        for r in results.get("results", [])
                    ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        
        return []


# ============================================================================
# Hierarchical Retrieval Strategy
# ============================================================================


class HierarchicalRetrievalStrategy(BaseRetrievalStrategy):
    """
    Hierarchical retrieval across document levels.
    
    Features:
    - Multi-level search (document, section, paragraph)
    - Parent-child context linking
    - Context window expansion
    - Weighted level combination
    """
    
    @property
    def strategy_type(self) -> RetrievalStrategy:
        return RetrievalStrategy.HIERARCHICAL
    
    def __init__(
        self,
        hierarchical_index: HierarchicalIndex,
        config: HierarchicalConfig,
    ):
        self.index = hierarchical_index
        self.config = config
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        levels: Optional[List[HierarchyLevel]] = None,
        expand_context: bool = True,
        **kwargs,
    ) -> RetrievalResponse:
        """Execute hierarchical retrieval."""
        start_time = time.perf_counter()
        
        levels = levels or [
            HierarchyLevel.DOCUMENT,
            HierarchyLevel.SECTION,
            HierarchyLevel.PARAGRAPH,
        ]
        
        # Search across levels
        level_results = self.index.search(query, levels)
        
        # Combine with weights
        combined_scores: Dict[str, float] = {}
        chunk_data: Dict[str, Any] = {}
        
        level_weights = {
            HierarchyLevel.DOCUMENT: self.config.document_weight,
            HierarchyLevel.SECTION: self.config.section_weight,
            HierarchyLevel.PARAGRAPH: self.config.paragraph_weight,
        }
        
        for level, results in level_results.items():
            weight = level_weights.get(level, 0.33)
            
            for chunk_id, score in results:
                weighted_score = weight * score
                
                if chunk_id in combined_scores:
                    combined_scores[chunk_id] = max(combined_scores[chunk_id], weighted_score)
                else:
                    combined_scores[chunk_id] = weighted_score
                
                chunk = self.index.get_chunk(chunk_id)
                if chunk:
                    chunk_data[chunk_id] = chunk
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build results with context expansion
        results = []
        seen_content = set()
        
        for rank, (chunk_id, score) in enumerate(sorted_results, start=1):
            if len(results) >= top_k:
                break
            
            chunk = chunk_data.get(chunk_id)
            if not chunk:
                continue
            
            content = chunk.content
            
            # Context expansion for paragraph level
            if expand_context and chunk.level == HierarchyLevel.PARAGRAPH:
                context_chunks = self.index.get_context_window(
                    chunk_id,
                    self.config.context_window_sentences,
                )
                if len(context_chunks) > 1:
                    content = " ".join([c.content for c in context_chunks])
            
            # Skip near-duplicates
            content_hash = hash(content[:200])
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            results.append(RetrievalResult(
                doc_id=chunk_id,
                content=content,
                score=score,
                source="hierarchical",
                rank=rank,
                hierarchy_level=chunk.level,
                parent_id=chunk.parent_id,
                metadata={
                    "doc_id": chunk.doc_id,
                    "level": chunk.level.value,
                },
            ))
        
        retrieval_time = (time.perf_counter() - start_time) * 1000
        
        return RetrievalResponse(
            query=query,
            results=results,
            strategy_used=RetrievalStrategy.HIERARCHICAL,
            total_candidates=sum(len(r) for r in level_results.values()),
            retrieval_time_ms=retrieval_time,
            metadata={
                "levels_searched": [l.value for l in levels],
                "level_counts": {l.value: len(r) for l, r in level_results.items()},
            },
        )


# ============================================================================
# Multi-Index Retrieval Strategy
# ============================================================================


class MultiIndexRetrievalStrategy(BaseRetrievalStrategy):
    """
    Retrieval across multiple indexes.
    
    Features:
    - Parallel search across indexes
    - Index-specific weighting
    - Result fusion
    """
    
    @property
    def strategy_type(self) -> RetrievalStrategy:
        return RetrievalStrategy.MULTI_INDEX
    
    def __init__(
        self,
        index_registry: IndexRegistry,
        fusion_method: FusionMethod = FusionMethod.RRF,
        index_weights: Optional[Dict[str, float]] = None,
    ):
        self.registry = index_registry
        self.fusion_method = fusion_method
        self.index_weights = index_weights or {}
        
        self.fusion_engine = FusionEngine(method=fusion_method)
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        index_names: Optional[List[str]] = None,
        **kwargs,
    ) -> RetrievalResponse:
        """Execute multi-index retrieval."""
        start_time = time.perf_counter()
        
        # Get available indexes
        available = self.registry.list_indexes()
        all_indexes = available.get("bm25", [])
        
        if index_names:
            indexes_to_search = [idx for idx in index_names if idx in all_indexes]
        else:
            indexes_to_search = all_indexes
        
        if not indexes_to_search:
            return RetrievalResponse(
                query=query,
                results=[],
                strategy_used=RetrievalStrategy.MULTI_INDEX,
                retrieval_time_ms=0,
            )
        
        # Search all indexes in parallel
        search_results = self.registry.search_multiple(
            query=query,
            index_names=indexes_to_search,
            top_k=top_k * 2,
        )
        
        # Prepare for fusion
        result_lists = []
        weights = []
        
        for idx_name, results in search_results.items():
            result_lists.append(results)
            weights.append(self.index_weights.get(idx_name, 1.0))
        
        # Fuse results
        if result_lists:
            fused = self.fusion_engine.fuse_multi(result_lists, weights)
        else:
            fused = []
        
        # Build results
        results = []
        for rank, (doc_id, score) in enumerate(fused[:top_k], start=1):
            # Find content from appropriate index
            content = ""
            for idx_name in indexes_to_search:
                index = self.registry.get_bm25_index(idx_name)
                if index:
                    doc = index.get_document(doc_id)
                    if doc:
                        content = doc
                        break
            
            results.append(RetrievalResult(
                doc_id=doc_id,
                content=content,
                score=score,
                source="multi_index",
                rank=rank,
            ))
        
        retrieval_time = (time.perf_counter() - start_time) * 1000
        
        return RetrievalResponse(
            query=query,
            results=results,
            strategy_used=RetrievalStrategy.MULTI_INDEX,
            fusion_method=self.fusion_method,
            total_candidates=sum(len(r) for r in search_results.values()),
            retrieval_time_ms=retrieval_time,
            metadata={
                "indexes_searched": indexes_to_search,
                "index_result_counts": {k: len(v) for k, v in search_results.items()},
            },
        )


# ============================================================================
# Router Retrieval Strategy
# ============================================================================


class RouterRetrievalStrategy(BaseRetrievalStrategy):
    """
    LLM-routed retrieval that dynamically selects strategy.
    
    Features:
    - Query classification
    - Dynamic strategy selection
    - Index routing
    - Fallback handling
    """
    
    @property
    def strategy_type(self) -> RetrievalStrategy:
        return RetrievalStrategy.ROUTER
    
    def __init__(
        self,
        router: QueryRouter,
        strategies: Dict[RetrievalStrategy, BaseRetrievalStrategy],
        fallback_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
    ):
        self.router = router
        self.strategies = strategies
        self.fallback_strategy = fallback_strategy
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        language: str = "en",
        force_strategy: Optional[RetrievalStrategy] = None,
        **kwargs,
    ) -> RetrievalResponse:
        """Execute router-based retrieval."""
        start_time = time.perf_counter()
        
        # Get routing decision
        decision = await self.router.route(query, language, force_strategy)
        
        # Handle skip retrieval
        if decision.skip_retrieval:
            return RetrievalResponse(
                query=query,
                results=[],
                strategy_used=RetrievalStrategy.ROUTER,
                retrieval_time_ms=(time.perf_counter() - start_time) * 1000,
                metadata={
                    "decision": decision.to_dict(),
                    "skipped": True,
                },
            )
        
        # Get strategy
        selected_strategy = decision.selected_strategy
        strategy = self.strategies.get(selected_strategy)
        
        if not strategy:
            logger.warning(f"Strategy {selected_strategy} not available, using fallback")
            strategy = self.strategies.get(self.fallback_strategy)
        
        if not strategy:
            return RetrievalResponse(
                query=query,
                results=[],
                strategy_used=RetrievalStrategy.ROUTER,
                retrieval_time_ms=(time.perf_counter() - start_time) * 1000,
                metadata={"error": "No available strategy"},
            )
        
        # Execute selected strategy
        result = await strategy.retrieve(
            query=query,
            top_k=decision.suggested_top_k or top_k,
            **kwargs,
        )
        
        # Add routing metadata
        result.metadata["routing_decision"] = decision.to_dict()
        result.metadata["routed_to"] = selected_strategy.value
        
        return result


# ============================================================================
# Strategy Factory
# ============================================================================


class StrategyFactory:
    """Factory for creating retrieval strategies."""
    
    def __init__(
        self,
        module_registry: IModuleRegistry,
        bm25_index: Optional[BM25Index] = None,
        hierarchical_index: Optional[HierarchicalIndex] = None,
        index_registry: Optional[IndexRegistry] = None,
        router: Optional[QueryRouter] = None,
    ):
        self.module_registry = module_registry
        self.bm25_index = bm25_index or BM25Index(BM25Config())
        self.hierarchical_index = hierarchical_index or HierarchicalIndex(HierarchicalConfig())
        self.index_registry = index_registry or IndexRegistry()
        self.router = router
        
        # Register default index
        self.index_registry.register_bm25_index("default", self.bm25_index)
    
    def create_hybrid(
        self,
        config: Optional[HybridConfig] = None,
        vector_config: Optional[VectorConfig] = None,
    ) -> HybridRetrievalStrategy:
        """Create hybrid retrieval strategy."""
        return HybridRetrievalStrategy(
            bm25_index=self.bm25_index,
            module_registry=self.module_registry,
            config=config or HybridConfig(),
            vector_config=vector_config or VectorConfig(),
        )
    
    def create_hierarchical(
        self,
        config: Optional[HierarchicalConfig] = None,
    ) -> HierarchicalRetrievalStrategy:
        """Create hierarchical retrieval strategy."""
        return HierarchicalRetrievalStrategy(
            hierarchical_index=self.hierarchical_index,
            config=config or HierarchicalConfig(),
        )
    
    def create_multi_index(
        self,
        fusion_method: FusionMethod = FusionMethod.RRF,
        index_weights: Optional[Dict[str, float]] = None,
    ) -> MultiIndexRetrievalStrategy:
        """Create multi-index retrieval strategy."""
        return MultiIndexRetrievalStrategy(
            index_registry=self.index_registry,
            fusion_method=fusion_method,
            index_weights=index_weights,
        )
    
    def create_router(
        self,
        strategies: Optional[Dict[RetrievalStrategy, BaseRetrievalStrategy]] = None,
        fallback: RetrievalStrategy = RetrievalStrategy.HYBRID,
    ) -> RouterRetrievalStrategy:
        """Create router retrieval strategy."""
        if not self.router:
            raise ValueError("Router not configured")
        
        if not strategies:
            strategies = {
                RetrievalStrategy.HYBRID: self.create_hybrid(),
                RetrievalStrategy.HIERARCHICAL: self.create_hierarchical(),
                RetrievalStrategy.MULTI_INDEX: self.create_multi_index(),
                RetrievalStrategy.BM25: self._create_bm25_only(),
            }
        
        return RouterRetrievalStrategy(
            router=self.router,
            strategies=strategies,
            fallback_strategy=fallback,
        )
    
    def _create_bm25_only(self) -> BaseRetrievalStrategy:
        """Create BM25-only strategy (using hybrid with alpha=1)."""
        return HybridRetrievalStrategy(
            bm25_index=self.bm25_index,
            module_registry=self.module_registry,
            config=HybridConfig(alpha=1.0, bm25_weight=1.0, vector_weight=0.0),
            vector_config=VectorConfig(),
        )
