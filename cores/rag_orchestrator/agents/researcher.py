"""
ARCHITECTURE v2.3/v2.4: Interactive Analyst - Researcher & Swarm Executor

v2.3: RAG-First data gathering logic with intelligent fallback to web search.
v2.4: Parallel Swarm execution for report sections.

Source Preferences:
    - rag_only:     Internal KB only, no fallback
    - web_only:     External web only, no RAG
    - rag_first:    Try RAG, fallback to web if insufficient
    - web_first:    Try web, fallback to RAG if insufficient
    - mixed:        Combine both sources with deduplication
    - llm_reasoning: No retrieval, pure LLM generation

Swarm Architecture (v2.4):
    - Planner Model (Big Brain): Creates intelligent report plans
    - Worker Model (Small Workers): Parallel section drafting
    - asyncio.gather for parallel research and drafting

Author: UBP Team
Version: 2.4.0
"""

import asyncio
import hashlib
import logging
import os
import time
import uuid
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .report_session import SectionPlan, ReportPlan

from .report_session import SourcePreference

try:
    from . import report_metrics  # v5.0.4: Prometheus metrics
except ImportError:
    report_metrics = None  # v6.4.1: Graceful degradation if metrics not available

# BUG-2 fix: ProviderMapper for LLM delegation fallback
try:
    from ubp_enterprise_hybrid.modules.cores._shared import ProviderMapper
    _PROVIDER_MAPPER_OK = True
except ImportError:
    _PROVIDER_MAPPER_OK = False

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================


@dataclass
class ResearchResult:
    """Result of a research query."""

    query: str
    source_preference: SourcePreference
    documents: List[Dict[str, Any]]
    sources_used: List[str]  # ["rag", "web"]
    fallback_triggered: bool
    fallback_reason: Optional[str]
    total_time_ms: float
    rag_results_count: int = 0
    web_results_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "source_preference": self.source_preference.value,
            "documents_count": len(self.documents),
            "sources_used": self.sources_used,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason,
            "total_time_ms": round(self.total_time_ms, 2),
            "rag_results_count": self.rag_results_count,
            "web_results_count": self.web_results_count,
            "metadata": self.metadata,
        }


@dataclass
class ResearchConfig:
    """Configuration for research operations."""

    # Fallback thresholds
    min_docs_for_confidence: int = 2
    min_score_threshold: float = 0.5

    # RAG settings
    rag_top_k: int = 5
    rag_collections: List[str] = field(default_factory=list)

    # Web settings
    web_max_results: int = 5
    web_timeout_ms: int = 10000

    # Deduplication
    dedup_similarity_threshold: float = 0.85


# =============================================================================
# v2.4: SWARM EXECUTION DATACLASSES
# =============================================================================


@dataclass
class WorkerConfig:
    """Configuration for Worker model (Small/Fast LLM for parallel drafting)."""

    # Provider settings (v6.0.1: model resolved by inference module)
    worker_provider: str = ""
    temperature: float = 0.35  # v6.8.5: lowered from 0.5 for professional tone
    max_tokens: int = 1500

    # Parallelism settings
    max_parallel_workers: int = 8

    # Dynamic timeout: base + (workers * per_worker), capped at max
    base_section_timeout: int = 60
    timeout_per_worker: int = 15
    max_section_timeout: int = 300
    _section_timeout_override: Optional[int] = None

    # Execution flags
    parallel_research: bool = True
    parallel_drafting: bool = True

    # Caching
    cache_research: bool = True
    cache_drafts: bool = True

    @property
    def section_timeout(self) -> int:
        """Calculate dynamic timeout based on worker count."""
        if self._section_timeout_override is not None:
            return self._section_timeout_override
        return self.get_section_timeout(max_tokens=1500)

    def get_section_timeout(self, max_tokens: int = 1500) -> int:
        """v6.4.1: Adaptive timeout based on token count + workers."""
        token_factor = max(1.0, max_tokens / 1000)
        timeout = (self.base_section_timeout * token_factor) + (self.max_parallel_workers * self.timeout_per_worker)
        return min(int(timeout), self.max_section_timeout)

    @section_timeout.setter
    def section_timeout(self, value: int) -> None:
        """Allow explicit override of section timeout."""
        self._section_timeout_override = value

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        """Load configuration from environment variables."""
        config = cls(
            worker_provider=os.getenv("UBP_REPORT__WORKER_PROVIDER", ""),
            temperature=float(os.getenv("UBP_REPORT__WORKER_TEMPERATURE", "0.35")),
            max_tokens=int(os.getenv("UBP_REPORT__WORKER_MAX_TOKENS", "1500")),
            max_parallel_workers=int(os.getenv("UBP_REPORT__MAX_PARALLEL_WORKERS", "8")),
            parallel_research=os.getenv("UBP_REPORT__PARALLEL_RESEARCH", "true").lower() == "true",
            parallel_drafting=os.getenv("UBP_REPORT__PARALLEL_DRAFTING", "true").lower() == "true",
            cache_research=os.getenv("UBP_REPORT__CACHE_RESEARCH", "true").lower() == "true",
            cache_drafts=os.getenv("UBP_REPORT__CACHE_DRAFTS", "true").lower() == "true",
        )
        # Allow explicit override from env var
        env_timeout = os.getenv("UBP_REPORT__SECTION_TIMEOUT")
        if env_timeout is not None:
            config._section_timeout_override = int(env_timeout)
        logger.info(
            f"WorkerConfig: timeout={config.section_timeout}s "
            f"(base={config.base_section_timeout} + {config.max_parallel_workers}*{config.timeout_per_worker}, "
            f"max={config.max_section_timeout})"
        )
        return config


@dataclass
class SectionDraft:
    """Draft content for a single report section."""

    section_title: str
    content: str
    word_count: int
    sources_used: List[str]
    documents_count: int
    generation_time_ms: float
    status: str  # "success", "partial", "error"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section_title": self.section_title,
            "content": self.content,
            "word_count": self.word_count,
            "sources_used": self.sources_used,
            "documents_count": self.documents_count,
            "generation_time_ms": round(self.generation_time_ms, 2),
            "status": self.status,
            "error_message": self.error_message,
            "metadata": self.metadata,
            # Bug 12 Fix: Add aliases for compatibility
            "title": self.section_title,
            "sources": self.sources_used,
        }


@dataclass
class SwarmResult:
    """Result of parallel swarm execution for a complete report."""

    plan_title: str
    sections: List[SectionDraft]
    total_time_ms: float
    parallel_efficiency: float  # actual_time / sequential_estimate
    sections_succeeded: int
    sections_failed: int
    worker_provider: str  # v6.0.1: renamed from worker_model
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_title": self.plan_title,
            "sections": [s.to_dict() for s in self.sections],
            "total_time_ms": round(self.total_time_ms, 2),
            "parallel_efficiency": round(self.parallel_efficiency, 2),
            "sections_succeeded": self.sections_succeeded,
            "sections_failed": self.sections_failed,
            "worker_provider": self.worker_provider,
            "metadata": self.metadata,
        }

    @property
    def full_draft(self) -> str:
        """Concatenate all successful section drafts into a full report."""
        parts = []
        for section in self.sections:
            if section.status == "success":
                parts.append(f"## {section.section_title}\n\n{section.content}")
        return "\n\n".join(parts)


# =============================================================================
# RESEARCHER AGENT
# =============================================================================


class Researcher:
    """
    RAG-First research agent for intelligent data gathering.

    Implements the source preference logic:
    - rag_only: Query internal knowledge bases exclusively
    - web_only: Query external web sources exclusively
    - rag_first: Try RAG, fallback to web if results insufficient
    - web_first: Try web, fallback to RAG if results insufficient
    - mixed: Query both and merge results with deduplication

    Usage:
        researcher = Researcher(rag_module, web_module)

        result = await researcher.gather_data(
            query="sicurezza UBP",
            preference=SourcePreference.RAG_FIRST,
            collections=["ubp_system_docs"],
        )
    """

    def __init__(
        self,
        rag_module=None,
        web_module=None,
        enrichment_module=None,
    ):
        """
        Initialize Researcher.

        Args:
            rag_module: RAG Qdrant module for internal KB queries
            web_module: Web search module for external queries
            enrichment_module: Optional enrichment module for reranking
        """
        self.rag = rag_module
        self.web = web_module
        self.enrichment = enrichment_module

    async def gather_data(
        self,
        query: str,
        preference: SourcePreference,
        collections: Optional[List[str]] = None,
        config: Optional[ResearchConfig] = None,
        ctx: Any = None,
    ) -> ResearchResult:
        """
        Gather research data based on source preference.

        Args:
            query: Search query
            preference: Source preference (rag_only, rag_first, etc.)
            collections: Optional list of RAG collections
            config: Optional research configuration
            ctx: Security context

        Returns:
            ResearchResult with gathered documents
        """
        start_time = time.time()
        config = config or ResearchConfig()

        logger.debug(
            "[COLLECTIONS] gather_data: collections param",
            extra={"collections": collections},
        )

        if collections:
            config.rag_collections = collections

        logger.debug(
            "[COLLECTIONS] gather_data: config.rag_collections",
            extra={"rag_collections": config.rag_collections},
        )

        logger.info(
            f"[RESEARCHER] Gathering data: '{query[:50]}...'",
            extra={
                "preference": preference.value,
                "collections": config.rag_collections,
            },
        )

        # Execute based on preference
        if preference == SourcePreference.RAG_ONLY:
            result = await self._gather_rag_only(query, config, ctx)

        elif preference == SourcePreference.WEB_ONLY:
            result = await self._gather_web_only(query, config, ctx)

        elif preference == SourcePreference.RAG_FIRST:
            result = await self._gather_rag_first(query, config, ctx)

        elif preference == SourcePreference.WEB_FIRST:
            result = await self._gather_web_first(query, config, ctx)

        elif preference == SourcePreference.MIXED:
            result = await self._gather_mixed(query, config, ctx)

        elif preference == SourcePreference.LLM_REASONING:
            # No retrieval needed
            result = ResearchResult(
                query=query,
                source_preference=preference,
                documents=[],
                sources_used=[],
                fallback_triggered=False,
                fallback_reason=None,
                total_time_ms=0,
            )

        else:
            raise ValueError(f"Unknown source preference: {preference}")

        result.total_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"[RESEARCHER] Research complete: {len(result.documents)} docs",
            extra={
                "sources_used": result.sources_used,
                "fallback": result.fallback_triggered,
                "time_ms": result.total_time_ms,
            },
        )

        return result

    # =========================================================================
    # GATHERING STRATEGIES
    # =========================================================================

    async def _gather_rag_only(
        self,
        query: str,
        config: ResearchConfig,
        ctx: Any,
    ) -> ResearchResult:
        """Gather from RAG only, no fallback."""
        docs = await self._query_rag(query, config, ctx)

        return ResearchResult(
            query=query,
            source_preference=SourcePreference.RAG_ONLY,
            documents=docs,
            sources_used=["rag"] if docs else [],
            fallback_triggered=False,
            fallback_reason=None,
            total_time_ms=0,
            rag_results_count=len(docs),
        )

    async def _gather_web_only(
        self,
        query: str,
        config: ResearchConfig,
        ctx: Any,
    ) -> ResearchResult:
        """Gather from web only, no fallback."""
        docs = await self._query_web(query, config, ctx)

        return ResearchResult(
            query=query,
            source_preference=SourcePreference.WEB_ONLY,
            documents=docs,
            sources_used=["web"] if docs else [],
            fallback_triggered=False,
            fallback_reason=None,
            total_time_ms=0,
            web_results_count=len(docs),
        )

    async def _gather_rag_first(
        self,
        query: str,
        config: ResearchConfig,
        ctx: Any,
    ) -> ResearchResult:
        """
        Try RAG first, fallback to web if insufficient.

        Fallback triggers:
        - No results
        - Results below min_docs_for_confidence
        - Best score below min_score_threshold
        """
        # Query RAG
        rag_docs = await self._query_rag(query, config, ctx)

        # Check if RAG is sufficient
        is_sufficient, reason = self._check_sufficiency(rag_docs, config)

        if is_sufficient:
            return ResearchResult(
                query=query,
                source_preference=SourcePreference.RAG_FIRST,
                documents=rag_docs,
                sources_used=["rag"],
                fallback_triggered=False,
                fallback_reason=None,
                total_time_ms=0,
                rag_results_count=len(rag_docs),
            )

        # Fallback to web
        logger.info(
            f"[RESEARCHER] RAG insufficient ({reason}), falling back to WEB",
            extra={"rag_count": len(rag_docs)},
        )

        web_docs = await self._query_web(query, config, ctx)

        # Combine and deduplicate
        all_docs = self._deduplicate_docs(rag_docs + web_docs, config)

        return ResearchResult(
            query=query,
            source_preference=SourcePreference.RAG_FIRST,
            documents=all_docs,
            sources_used=["rag", "web"],
            fallback_triggered=True,
            fallback_reason=reason,
            total_time_ms=0,
            rag_results_count=len(rag_docs),
            web_results_count=len(web_docs),
        )

    async def _gather_web_first(
        self,
        query: str,
        config: ResearchConfig,
        ctx: Any,
    ) -> ResearchResult:
        """
        Try web first, fallback to RAG if insufficient.
        """
        # Query web
        web_docs = await self._query_web(query, config, ctx)

        # Check if web is sufficient
        is_sufficient, reason = self._check_sufficiency(web_docs, config)

        if is_sufficient:
            return ResearchResult(
                query=query,
                source_preference=SourcePreference.WEB_FIRST,
                documents=web_docs,
                sources_used=["web"],
                fallback_triggered=False,
                fallback_reason=None,
                total_time_ms=0,
                web_results_count=len(web_docs),
            )

        # Fallback to RAG
        logger.info(
            f"[RESEARCHER] Web insufficient ({reason}), falling back to RAG",
            extra={"web_count": len(web_docs)},
        )

        rag_docs = await self._query_rag(query, config, ctx)

        # Combine and deduplicate
        all_docs = self._deduplicate_docs(web_docs + rag_docs, config)

        return ResearchResult(
            query=query,
            source_preference=SourcePreference.WEB_FIRST,
            documents=all_docs,
            sources_used=["web", "rag"],
            fallback_triggered=True,
            fallback_reason=reason,
            total_time_ms=0,
            rag_results_count=len(rag_docs),
            web_results_count=len(web_docs),
        )

    async def _gather_mixed(
        self,
        query: str,
        config: ResearchConfig,
        ctx: Any,
    ) -> ResearchResult:
        """
        Gather from both RAG and web in parallel, merge results.
        """
        # Execute both queries in parallel
        rag_task = asyncio.create_task(self._query_rag(query, config, ctx))
        web_task = asyncio.create_task(self._query_web(query, config, ctx))

        rag_docs, web_docs = await asyncio.gather(rag_task, web_task)

        # Merge and deduplicate
        all_docs = self._deduplicate_docs(rag_docs + web_docs, config)

        # Optional: Rerank merged results
        if self.enrichment:
            try:
                rerank_result = await self.enrichment.rerank(
                    query=query,
                    chunks=all_docs,
                    top_k=config.rag_top_k,
                )
                all_docs = rerank_result.get("reranked_chunks", all_docs)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        sources_used = []
        if rag_docs:
            sources_used.append("rag")
        if web_docs:
            sources_used.append("web")

        return ResearchResult(
            query=query,
            source_preference=SourcePreference.MIXED,
            documents=all_docs,
            sources_used=sources_used,
            fallback_triggered=False,
            fallback_reason=None,
            total_time_ms=0,
            rag_results_count=len(rag_docs),
            web_results_count=len(web_docs),
        )

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    async def _query_rag(
        self,
        query: str,
        config: ResearchConfig,
        ctx: Any,
    ) -> List[Dict[str, Any]]:
        """Query RAG module for internal documents."""
        logger.debug(
            "[COLLECTIONS] _query_rag: rag_collections",
            extra={"rag_collections": config.rag_collections},
        )

        if not self.rag:
            logger.debug("[COLLECTIONS] _query_rag: RAG module not available")
            return []

        if not config.rag_collections:
            logger.debug("[COLLECTIONS] _query_rag: EMPTY collections, skipping RAG query")
            return []

        try:
            # v5.0.4: Parallel collection queries
            async def _query_single_collection(collection: str) -> List[Dict[str, Any]]:
                """Query a single collection and normalize results."""
                response = await self.rag.query_internal(
                    collection_name=collection,
                    query_text=query,
                    top_k=config.rag_top_k,
                )
                docs = []
                for doc in response.get("results", []):
                    docs.append({
                        "text": doc.get("text", doc.get("payload", {}).get("text", "")),
                        "score": doc.get("score", 0.0),
                        "source": "rag",
                        "collection": collection,
                        "metadata": doc.get("metadata", doc.get("payload", {})),
                    })
                return docs

            if len(config.rag_collections) == 1:
                # Single collection — no parallelization overhead
                results = await _query_single_collection(config.rag_collections[0])
            else:
                # Multiple collections — query in parallel
                # v6.4.1: Limit parallel collection queries
                _collection_sem = asyncio.Semaphore(min(10, len(config.rag_collections)))
                async def _bounded_query(c):
                    async with _collection_sem:
                        return await _query_single_collection(c)
                tasks = [_bounded_query(c) for c in config.rag_collections]
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                results = []
                for collection, resp in zip(config.rag_collections, responses):
                    if isinstance(resp, Exception):
                        logger.error(f"[RESEARCHER] RAG query failed for {collection}: {resp}")
                        continue
                    results.extend(resp)

            logger.debug(
                f"[RESEARCHER] RAG returned {len(results)} documents",
                extra={"collections": config.rag_collections},
            )
            return results

        except Exception as e:
            logger.error(f"[RESEARCHER] RAG query failed: {e}")
            return []

    async def _query_web(
        self,
        query: str,
        config: ResearchConfig,
        ctx: Any,
    ) -> List[Dict[str, Any]]:
        """Query web search module for external documents."""
        if not self.web:
            logger.warning("[RESEARCHER] Web module not available")
            return []

        try:
            # Call web search module
            response = await self.web.search(
                query=query,
                max_results=config.web_max_results,
            )

            # Normalize results
            results = []
            for item in response.get("results", []):
                results.append({
                    "text": item.get("snippet", item.get("content", "")),
                    "score": item.get("relevance_score", 0.7),
                    "source": "web",
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "metadata": {
                        "url": item.get("url"),
                        "title": item.get("title"),
                        "source_type": "web",
                    },
                })

            logger.debug(f"[RESEARCHER] Web returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"[RESEARCHER] Web query failed: {e}")
            return []

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _check_sufficiency(
        self,
        docs: List[Dict[str, Any]],
        config: ResearchConfig,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if gathered documents are sufficient.

        Returns:
            Tuple of (is_sufficient, reason_if_not)
        """
        if len(docs) == 0:
            return False, "no_results"

        if len(docs) < config.min_docs_for_confidence:
            return False, f"insufficient_count ({len(docs)} < {config.min_docs_for_confidence})"

        # Check best score
        best_score = max(doc.get("score", 0) for doc in docs)
        if best_score < config.min_score_threshold:
            return False, f"low_relevance (best={best_score:.2f} < {config.min_score_threshold})"

        return True, None

    def _deduplicate_docs(
        self,
        docs: List[Dict[str, Any]],
        config: ResearchConfig,
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate documents based on text similarity.

        Simple approach: exact text matching + fuzzy similarity check.
        """
        if not docs:
            return []

        seen_texts = set()
        unique_docs = []

        for doc in docs:
            text = doc.get("text", "")
            text_key = hashlib.md5(text.lower().strip().encode()).hexdigest()

            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_docs.append(doc)

        # Sort by score (highest first)
        unique_docs.sort(key=lambda x: x.get("score", 0), reverse=True)

        logger.debug(
            f"[RESEARCHER] Deduplication: {len(docs)} -> {len(unique_docs)} docs"
        )

        return unique_docs

    async def gather_for_section(
        self,
        section_title: str,
        section_queries: List[str],
        preference: SourcePreference,
        collections: Optional[List[str]] = None,
        config: Optional[ResearchConfig] = None,
        ctx: Any = None,
        enrichment_config: Optional["SectionEnrichmentConfig"] = None,  # v2.6
        enrichment_module: Any = None,  # v2.6
    ) -> Dict[str, Any]:
        """
        Gather research data for a report section.

        Executes multiple queries and aggregates results.

        v2.6: Now supports per-section enrichment configuration.

        Args:
            section_title: Section title for logging
            section_queries: List of queries to execute
            preference: Source preference
            collections: Optional RAG collections
            config: Optional research config
            ctx: Security context
            enrichment_config: v2.6 - Per-section enrichment settings
            enrichment_module: v2.6 - Enrichment pipeline module

        Returns:
            Dict with aggregated research results
        """
        logger.debug(
            "[COLLECTIONS] gather_for_section",
            extra={"section": section_title, "collections": collections},
        )

        config = config or ResearchConfig()
        if collections:
            config.rag_collections = collections

        logger.debug(
            "[COLLECTIONS] gather_for_section: rag_collections set",
            extra={"rag_collections": config.rag_collections},
        )

        # v2.6: Track enrichment stats
        enrichment_stats = {
            "hyde_applied": False,
            "expansion_applied": False,
            "rerank_applied": False,
            "investigative_applied": False,
            "metadata_injected": False,
        }

        logger.info(
            f"[RESEARCHER] Gathering for section: {section_title}",
            extra={
                "queries_count": len(section_queries),
                "enrichment_enabled": enrichment_config is not None,
            },
        )

        # v2.6: Apply investigative decomposition if enabled
        queries_to_execute = section_queries
        if enrichment_config and enrichment_config.investigative_enabled and enrichment_module:
            try:
                decomposed = await self._apply_investigative_decomposition(
                    section_queries, enrichment_module
                )
                if decomposed:
                    queries_to_execute = decomposed
                    enrichment_stats["investigative_applied"] = True
                    logger.debug(f"[RESEARCHER] Investigative decomposition: {len(section_queries)} -> {len(queries_to_execute)} queries")
            except Exception as e:
                logger.warning(f"[RESEARCHER] Investigative decomposition failed: {e}")

        all_results: List[ResearchResult] = []

        # v5.0.4: Parallel query execution — each query is independent
        async def _execute_single_query(query: str) -> Tuple[ResearchResult, Dict[str, bool]]:
            """Execute enrichment + gather for a single query."""
            query_stats = {"hyde": False, "expansion": False}
            enriched_query = query
            if enrichment_config and enrichment_module:
                enriched_query, query_stats = await self._apply_query_enrichment(
                    query, enrichment_config, enrichment_module
                )

            result = await self.gather_data(
                query=enriched_query,
                preference=preference,
                config=config,
                ctx=ctx,
            )
            return result, query_stats

        if len(queries_to_execute) <= 1:
            # Single query — no parallelization overhead
            for query in queries_to_execute:
                result, stats = await _execute_single_query(query)
                all_results.append(result)
                enrichment_stats["hyde_applied"] = enrichment_stats["hyde_applied"] or stats.get("hyde")
                enrichment_stats["expansion_applied"] = enrichment_stats["expansion_applied"] or stats.get("expansion")
        else:
            # Multiple queries — execute in parallel
            logger.debug(f"[RESEARCHER] Parallel query execution: {len(queries_to_execute)} queries")
            tasks = [_execute_single_query(q) for q in queries_to_execute]
            results_with_stats = await asyncio.gather(*tasks, return_exceptions=True)

            for i, item in enumerate(results_with_stats):
                if isinstance(item, Exception):
                    logger.error(f"[RESEARCHER] Query {i} failed: {item}")
                    continue
                result, stats = item
                all_results.append(result)
                enrichment_stats["hyde_applied"] = enrichment_stats["hyde_applied"] or stats.get("hyde")
                enrichment_stats["expansion_applied"] = enrichment_stats["expansion_applied"] or stats.get("expansion")

        # Aggregate documents from all queries
        all_docs = []
        for result in all_results:
            all_docs.extend(result.documents)

        # Deduplicate across all queries
        unique_docs = self._deduplicate_docs(all_docs, config)

        # v2.6: Apply post-retrieval enrichment (reranking, metadata injection)
        if enrichment_config and enrichment_module and unique_docs:
            unique_docs, post_stats = await self._apply_post_retrieval_enrichment(
                unique_docs, section_queries[0] if section_queries else section_title,
                enrichment_config, enrichment_module, config
            )
            enrichment_stats["rerank_applied"] = post_stats.get("rerank", False)
            enrichment_stats["metadata_injected"] = post_stats.get("metadata", False)

        # Calculate aggregate stats
        total_rag = sum(r.rag_results_count for r in all_results)
        total_web = sum(r.web_results_count for r in all_results)
        any_fallback = any(r.fallback_triggered for r in all_results)

        # v5.0.3 RPT-005: Aggregate sources_used from all research results
        sources_used = []
        for r in all_results:
            for src in r.sources_used:
                if src not in sources_used:
                    sources_used.append(src)

        return {
            "section_title": section_title,
            "queries_executed": len(queries_to_execute),
            "original_queries": len(section_queries),
            "documents": unique_docs,
            "documents_count": len(unique_docs),
            "rag_total": total_rag,
            "web_total": total_web,
            "fallback_triggered": any_fallback,
            "source_preference": preference.value,
            "enrichment_stats": enrichment_stats,  # v2.6
            "sources_used": sources_used,  # v5.0.3 RPT-005
        }

    # =========================================================================
    # v2.6: ENRICHMENT HELPER METHODS
    # =========================================================================

    async def _apply_query_enrichment(
        self,
        query: str,
        enrichment_config: "SectionEnrichmentConfig",
        enrichment_module: Any,
    ) -> Tuple[str, Dict[str, bool]]:
        """
        v2.6: Apply query-level enrichment (HyDE, expansion).

        Returns:
            Tuple of (enriched_query, stats_dict)
        """
        stats = {"hyde": False, "expansion": False}
        enriched_query = query

        try:
            # Apply HyDE if enabled
            if enrichment_config.hyde_enabled:
                hyde_result = await enrichment_module.generate_hyde(query=query)
                if hyde_result and hyde_result.get("hyde_document"):
                    enriched_query = hyde_result["hyde_document"]
                    stats["hyde"] = True
                    logger.debug(f"[RESEARCHER] HyDE applied to query")

            # Apply query expansion if enabled
            if enrichment_config.query_expansion_enabled:
                expansion_result = await enrichment_module.expand_query(query=query)
                if expansion_result and expansion_result.get("expanded_query"):
                    if enrichment_config.hyde_enabled and enriched_query:
                        # v6.4.1: Compose — apply expansion ON the HyDE result
                        enriched_query = expansion_result.get("expanded_query", enriched_query)
                        logger.debug("[RESEARCHER] Composed HyDE + expansion enrichment")
                    else:
                        enriched_query = expansion_result["expanded_query"]
                    stats["expansion"] = True
                    logger.debug(f"[RESEARCHER] Query expansion applied")

        except Exception as e:
            logger.warning(f"[RESEARCHER] Query enrichment failed: {e}")

        return enriched_query, stats

    async def _apply_post_retrieval_enrichment(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        enrichment_config: "SectionEnrichmentConfig",
        enrichment_module: Any,
        config: ResearchConfig,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, bool]]:
        """
        v2.6: Apply post-retrieval enrichment (reranking, metadata injection).

        Returns:
            Tuple of (enriched_documents, stats_dict)
        """
        stats = {"rerank": False, "metadata": False}
        enriched_docs = documents

        try:
            # Apply reranking if enabled
            if enrichment_config.rerank_enabled and documents:
                rerank_result = await enrichment_module.rerank(
                    query=query,
                    chunks=documents,
                    top_k=config.rag_top_k,
                    reranker_type=getattr(enrichment_config, "reranker_type", "primary"),
                )
                if rerank_result and rerank_result.get("reranked_chunks"):
                    enriched_docs = rerank_result["reranked_chunks"]
                    stats["rerank"] = True
                    logger.debug(f"[RESEARCHER] Reranking applied: {len(documents)} -> {len(enriched_docs)} docs")

            # Apply metadata injection if enabled
            if enrichment_config.metadata_injection_enabled:
                metadata_result = await enrichment_module.inject_metadata(
                    chunks=enriched_docs,
                    query=query,
                )
                if metadata_result and metadata_result.get("enriched_chunks"):
                    enriched_docs = metadata_result["enriched_chunks"]
                    stats["metadata"] = True
                    logger.debug(f"[RESEARCHER] Metadata injection applied")

        except Exception as e:
            logger.warning(f"[RESEARCHER] Post-retrieval enrichment failed: {e}")

        return enriched_docs, stats

    async def _apply_investigative_decomposition(
        self,
        queries: List[str],
        enrichment_module: Any,
    ) -> Optional[List[str]]:
        """
        v2.6: Decompose queries into investigative sub-queries.

        Returns:
            List of decomposed queries, or None if decomposition failed
        """
        try:
            decomposed = []
            for query in queries:
                result = await enrichment_module.decompose_query(query=query)
                if result and result.get("sub_queries"):
                    decomposed.extend(result["sub_queries"])
                else:
                    decomposed.append(query)
            return decomposed if len(decomposed) > len(queries) else None
        except Exception as e:
            logger.warning(f"[RESEARCHER] Investigative decomposition failed: {e}")
            return None


# =============================================================================
# v2.4: SWARM EXECUTOR
# =============================================================================


# Worker Model Prompts
# v5.1.0 G1: Normative prompt — epistemic constraints for professional output.
# IMPORTANT: Keep compact, no structured markers (Qwen3-4B-AWQ echoes them).
SECTION_WRITER_SYSTEM_PROMPT = """\
Sei un redattore tecnico-scientifico. Regole vincolanti:
- Scrivi in italiano, in paragrafi densi
- Cita fonti come [1], [2] solo se presenti nel contesto fornito
- Non inventare citazioni ne' fonti
- Non ripetere concetti gia' espressi nel contesto
- Usa linguaggio prudente per evidenze non consolidate: "suggerisce", "sembra indicare", "i dati preliminari mostrano"
- Usa linguaggio assertivo solo per fatti ben documentati nelle fonti
- Max 1 conclusione per sezione, alla fine
- Non fare raccomandazioni cliniche dirette
- Distingui fatti documentati da interpretazioni
- Fermati quando il contenuto e' completo, non riempire"""

SECTION_WRITER_USER_PROMPT = """\
Sezione: {section_title}
Report: {report_subject}
Descrizione: {section_description}
Strategia fonti: {source_preference}

Contesto documentale:
{context}

Scrivi circa {max_tokens} token. Cita solo fonti numerate nel contesto. Non ripetere informazioni gia' trattate in altre sezioni del report."""


# v5.1.2: Exploratory mode prompts (memo-style output)
EXPLORATORY_WRITER_SYSTEM_PROMPT = """\
Sei un analista di ricerca. Scrivi un memo esplorativo in italiano.
Regole:
- Sintetizza le evidenze disponibili in modo conciso
- Evidenzia pattern emergenti e lacune nella letteratura
- Usa linguaggio esplorativo: "i dati suggeriscono", "emerge un pattern"
- NON fare raccomandazioni cliniche
- Includi "Evidenze chiave" e "Lacune identificate"
- Fermati quando il contenuto e' completo"""

EXPLORATORY_WRITER_USER_PROMPT = """\
Sezione: {section_title}
Argomento: {report_subject}

Matrice evidenze:
{evidence_summary}

Contesto documentale:
{context}

Scrivi un memo esplorativo conciso (max {max_tokens} token)."""


class SwarmExecutor:
    """
    v2.4/v2.6: Parallel Swarm Executor for Report Generation.

    The "Workers" that process sections in parallel using fast local models.
    Uses asyncio.gather for parallel research AND drafting.

    Architecture:
        1. Research Phase (parallel): Gather data for all sections simultaneously
        2. Drafting Phase (parallel): Generate section drafts simultaneously
        3. Assembly Phase: Combine all drafts into final report

    v2.6 Enhancements:
        - Per-section enrichment configuration (rerank, HyDE, expansion, etc.)
        - Debug event emission for real-time worker monitoring
        - Enrichment config passed from UI to each worker

    Usage:
        executor = SwarmExecutor(researcher, llm_module)
        result = await executor.execute_plan(plan, ctx)
        print(result.full_draft)
    """

    def __init__(
        self,
        researcher: Researcher,
        llm_module=None,
        config: Optional[WorkerConfig] = None,
        redis_client=None,  # v2.6: For debug events
        enrichment_module=None,  # v2.6: For enrichment pipeline
    ):
        """
        Initialize SwarmExecutor.

        Args:
            researcher: Researcher instance for data gathering
            llm_module: LLM module with generate() method for drafting
            config: Optional worker configuration
            redis_client: v2.6 - Redis client for debug event streaming
            enrichment_module: v2.6 - Enrichment pipeline module
        """
        self.researcher = researcher
        self.llm = llm_module
        self.config = config or WorkerConfig.from_env()
        self.redis_client = redis_client  # v2.6
        self.enrichment_module = enrichment_module  # v2.6

        # v6.4.2: Provider resolution with ProviderMapper fallback (BUG-2 fix)
        env_explicit = os.getenv("UBP_REPORT__WORKER_PROVIDER")
        if env_explicit:
            self._provider = env_explicit
        elif _PROVIDER_MAPPER_OK:
            try:
                chain = ProviderMapper.resolve_chain("enrichment")
                if chain:
                    self._provider = chain[0][1]
                    logger.info(f"[SWARM] ProviderMapper resolved worker: {self._provider}")
                else:
                    self._provider = self.config.worker_provider
            except Exception as e:
                logger.warning(f"[SWARM] ProviderMapper failed: {e}")
                self._provider = self.config.worker_provider
        else:
            self._provider = self.config.worker_provider

        logger.info(
            f"[SWARM] Executor initialized: provider={self._provider}, "
            f"max_parallel={self.config.max_parallel_workers}, "
            f"enrichment={'enabled' if enrichment_module else 'disabled'}"
        )

    async def execute_plan(
        self,
        plan: "ReportPlan",
        ctx: Any = None,
        conversation_id: Optional[str] = None,
        session_id: Optional[str] = None,  # v2.6: For debug events
        execution_mode: str = "report",           # v5.1.2: report/insight/exploratory
        planner_llm_module=None,                  # v5.1.2: Grok for M2 ReasoningPass
    ) -> SwarmResult:
        """
        Execute a report plan using parallel swarm processing.

        This is the main entry point for swarm execution.

        Args:
            plan: ReportPlan with sections to process
            ctx: Security context
            conversation_id: Optional conversation ID for tracking
            session_id: v2.6 - Session ID for debug event emission

        Returns:
            SwarmResult with all section drafts and metrics
        """
        start_time = time.time()
        self._current_session_id = session_id  # v2.6: Store for debug events

        logger.info(
            f"[SWARM] Starting execution: '{plan.template_name}'",
            extra={
                "sections": len(plan.sections),
                "parallel": self.config.parallel_research,
                "session_id": session_id,
            },
        )

        # Phase 1: Research (parallel or sequential)
        research_results = await self._execute_research_phase(plan, ctx)

        # v5.1.2: Conditional Evidence Abstraction + Reasoning
        evidence_matrices = None
        reasoning_output = None
        if execution_mode in ("insight", "exploratory"):
            try:
                from .reasoning import EvidenceAbstractor, ReasoningPass

                abstractor = EvidenceAbstractor(llm_module=self.llm)
                evidence_matrices = await abstractor.extract_all_sections(research_results)

                if planner_llm_module:
                    reasoner = ReasoningPass(llm_module=planner_llm_module)
                    reasoning_output = await reasoner.reason(
                        evidence_matrices, plan.subject
                    )
            except Exception as e:
                logger.warning(f"[SWARM] Evidence/Reasoning phase failed: {e}")

        # Phase 2: Conditional drafting based on execution_mode
        if execution_mode == "insight":
            section_drafts = self._build_insight_output(
                research_results, evidence_matrices, reasoning_output, plan
            )
        elif execution_mode == "exploratory":
            section_drafts = await self._execute_drafting_phase(
                plan=plan,
                research_results=research_results,
                ctx=ctx,
                execution_mode="exploratory",
                evidence_matrices=evidence_matrices,
            )
        else:
            section_drafts = await self._execute_drafting_phase(
                plan=plan,
                research_results=research_results,
                ctx=ctx,
            )

        # Calculate metrics
        total_time_ms = (time.time() - start_time) * 1000
        sequential_estimate = sum(d.generation_time_ms for d in section_drafts)
        parallel_efficiency = sequential_estimate / total_time_ms if total_time_ms > 0 else 1.0

        succeeded = sum(1 for d in section_drafts if d.status == "success")
        failed = len(section_drafts) - succeeded

        # v5.1.2: Serialize evidence/reasoning for metadata
        _evidence_dict = None
        _reasoning_dict = None
        if evidence_matrices:
            from dataclasses import asdict as _asdict
            _evidence_dict = {k: _asdict(v) for k, v in evidence_matrices.items()}
        if reasoning_output:
            from dataclasses import asdict as _asdict
            _reasoning_dict = _asdict(reasoning_output)

        result = SwarmResult(
            plan_title=plan.template_name,
            sections=section_drafts,
            total_time_ms=total_time_ms,
            parallel_efficiency=parallel_efficiency,
            sections_succeeded=succeeded,
            sections_failed=failed,
            worker_provider=self.config.worker_provider,
            metadata={
                "plan_subject": plan.subject,
                "collections": plan.collections,
                "conversation_id": conversation_id,
                "evidence_matrices": _evidence_dict,
                "reasoning_output": _reasoning_dict,
                "execution_mode": execution_mode,
            },
        )

        logger.info(
            f"[SWARM] Execution complete: {succeeded}/{len(section_drafts)} sections",
            extra={
                "total_time_ms": round(total_time_ms, 2),
                "parallel_efficiency": round(parallel_efficiency, 2),
            },
        )

        return result

    def _build_insight_output(
        self,
        research_results: Dict[str, Dict[str, Any]],
        evidence_matrices: Optional[Dict] = None,
        reasoning_output: Optional[Any] = None,
        plan: Optional["ReportPlan"] = None,
    ) -> List[SectionDraft]:
        """
        v5.1.2: Build structured insight output (JSON) for insight mode.

        Instead of narrative drafting, assembles evidence + reasoning into
        structured JSON content. No section writer LLM calls needed.

        Returns:
            List[SectionDraft] with content=json.dumps(insight_data)
        """
        import json as _json
        from dataclasses import asdict as _asdict

        sections = plan.sections if plan else []
        drafts = []

        for section in sections:
            title = section.title
            research = research_results.get(title, {})

            # Build per-section insight data
            insight_data = {
                "section_title": title,
                "output_type": "structured_insight",
                "documents_count": research.get("documents_count", 0),
                "sources_used": research.get("sources_used", []),
            }

            # Add evidence matrix if available
            if evidence_matrices and title in evidence_matrices:
                matrix = evidence_matrices[title]
                insight_data["evidence_matrix"] = _asdict(matrix)

            # Add reasoning (same for all sections, but included for completeness)
            if reasoning_output:
                insight_data["reasoning"] = _asdict(reasoning_output)

            content = _json.dumps(insight_data, ensure_ascii=False, indent=2)

            drafts.append(SectionDraft(
                section_title=title,
                content=content,
                word_count=len(content.split()),
                sources_used=research.get("sources_used", []),
                documents_count=research.get("documents_count", 0),
                generation_time_ms=0.0,
                status="success",
                metadata={"output_type": "structured_insight"},
            ))

        return drafts

    async def _execute_research_phase(
        self,
        plan: "ReportPlan",
        ctx: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute research phase for all sections.

        Returns:
            Dict mapping section_title -> research_result
        """
        logger.debug(
            "[COLLECTIONS] _execute_research_phase",
            extra={"plan_collections": plan.collections},
        )

        logger.info(f"[SWARM] Research phase: {len(plan.sections)} sections")

        if self.config.parallel_research:
            # Parallel research with semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.config.max_parallel_workers)
            tasks = []

            for section_index, section in enumerate(plan.sections):  # v2.6: Track index
                task = self._research_section_with_semaphore(
                    section=section,
                    collections=plan.collections,
                    semaphore=semaphore,
                    ctx=ctx,
                    section_index=section_index,  # v2.6: Pass index for debug
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Build results dict
            research_results = {}
            for section, result in zip(plan.sections, results):
                if isinstance(result, Exception):
                    logger.error(f"[SWARM] Research failed for '{section.title}': {result}")
                    research_results[section.title] = {
                        "documents": [],
                        "error": str(result),
                    }
                else:
                    research_results[section.title] = result

            return research_results

        else:
            # Sequential research (fallback)
            research_results = {}
            for section_index, section in enumerate(plan.sections):  # v2.6: Track index
                try:
                    result = await self._research_section(
                        section, plan.collections, ctx, section_index  # v2.6: Pass index
                    )
                    research_results[section.title] = result
                except Exception as e:
                    logger.error(f"[SWARM] Research failed for '{section.title}': {e}")
                    research_results[section.title] = {"documents": [], "error": str(e)}

            return research_results

    async def _research_section_with_semaphore(
        self,
        section: "SectionPlan",
        collections: List[str],
        semaphore: asyncio.Semaphore,
        ctx: Any,
        section_index: int = 0,  # v2.6: For debug events
    ) -> Dict[str, Any]:
        """Research a section with semaphore for concurrency control."""
        async with semaphore:
            return await self._research_section(section, collections, ctx, section_index)

    async def _research_section(
        self,
        section: "SectionPlan",
        collections: List[str],
        ctx: Any,
        section_index: int = 0,  # v2.6: For debug events
    ) -> Dict[str, Any]:
        """
        Research a single section using the Researcher.

        v2.6: Now supports per-section enrichment configuration and debug events.
        """
        import json as _json
        from datetime import datetime as _dt, timezone as _tz

        logger.debug(
            "[COLLECTIONS] _research_section",
            extra={"section": section.title, "collections": collections},
        )

        # Determine queries to use
        queries = section.suggested_queries if section.suggested_queries else [section.title]

        # Get source preference and enrichment config
        preference = section.source_preference
        enrichment_config = section.enrichment_config  # v2.6
        debug_enabled = enrichment_config and enrichment_config.debug_enabled

        # v2.6: Emit debug event - INIT phase
        if debug_enabled and self.redis_client and self._current_session_id:
            await self._emit_debug_event(
                section_index=section_index,
                section_title=section.title,
                phase="init",
                enrichment_config=enrichment_config,
                input_data={
                    "queries": queries,
                    "collections": collections,
                    "source_preference": preference.value,
                },
            )

        # Skip retrieval for LLM reasoning sections
        if preference == SourcePreference.LLM_REASONING:
            if debug_enabled and self.redis_client and self._current_session_id:
                await self._emit_debug_event(
                    section_index=section_index,
                    section_title=section.title,
                    phase="complete",
                    enrichment_config=enrichment_config,
                    output_data={"skipped": True, "reason": "LLM_REASONING mode"},
                )
            return {
                "section_title": section.title,
                "documents": [],
                "documents_count": 0,
                "source_preference": preference.value,
                "no_retrieval": True,
            }

        # v2.6: Gather data using Researcher with enrichment config
        start_time = time.time()
        result = await self.researcher.gather_for_section(
            section_title=section.title,
            section_queries=queries,
            preference=preference,
            collections=collections,
            ctx=ctx,
            enrichment_config=enrichment_config,  # v2.6: Pass enrichment config
            enrichment_module=self.enrichment_module,  # v2.6: Pass enrichment module
        )
        research_time_ms = (time.time() - start_time) * 1000

        # v2.6: Emit debug event - COMPLETE phase
        if debug_enabled and self.redis_client and self._current_session_id:
            await self._emit_debug_event(
                section_index=section_index,
                section_title=section.title,
                phase="complete",
                enrichment_config=enrichment_config,
                output_data={
                    "documents_count": result.get("documents_count", 0),
                    "rag_total": result.get("rag_total", 0),
                    "web_total": result.get("web_total", 0),
                    "fallback_triggered": result.get("fallback_triggered", False),
                },
                metrics={
                    "research_time_ms": round(research_time_ms, 2),
                },
            )

        return result

    async def _emit_debug_event(
        self,
        section_index: int,
        section_title: str,
        phase: str,
        enrichment_config: Optional["SectionEnrichmentConfig"] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        v2.6: Emit a debug event to Redis for frontend polling.
        """
        import json as _json
        from datetime import datetime as _dt, timezone as _tz

        if not self.redis_client or not self._current_session_id:
            return

        try:
            event = {
                "event_id": str(uuid.uuid4()),
                "session_id": self._current_session_id,
                "section_index": section_index,
                "section_title": section_title,
                "phase": phase,
                "timestamp": _dt.now(_tz.utc).isoformat(),
                "worker_id": f"worker_{section_index}",
                "enrichment_config": enrichment_config.to_dict() if enrichment_config else {},
                "model_info": {
                    "provider": self._provider,
                    "worker_provider": self.config.worker_provider,
                },
                "input_data": input_data,
                "output_data": output_data,
                "metrics": metrics,
                "error": error,
            }

            debug_key = f"ubp:report:debug:{self._current_session_id}"
            await self.redis_client.rpush(debug_key, _json.dumps(event))
            # v5.0.4: Cap debug events list to prevent memory leak
            await self.redis_client.ltrim(debug_key, -200, -1)
            # Set TTL of 1 hour for debug events
            await self.redis_client.expire(debug_key, 3600)

            logger.debug(
                f"[SWARM] Debug event emitted: {phase} for section {section_index}",
                extra={"session_id": self._current_session_id},
            )
        except Exception as e:
            logger.warning(f"[SWARM] Failed to emit debug event: {e}")

    async def _execute_drafting_phase(
        self,
        plan: "ReportPlan",
        research_results: Dict[str, Dict[str, Any]],
        ctx: Any,
        execution_mode: str = "report",           # v5.1.2
        evidence_matrices: Optional[Dict] = None,  # v5.1.2
    ) -> List[SectionDraft]:
        """
        Execute drafting phase for all sections.

        v5.1.2: Supports execution_mode and evidence_matrices for exploratory mode.

        Returns:
            List of SectionDraft objects
        """
        logger.info(f"[SWARM] Drafting phase: {len(plan.sections)} sections (mode={execution_mode})")

        if self.config.parallel_drafting:
            # Parallel drafting with semaphore
            semaphore = asyncio.Semaphore(self.config.max_parallel_workers)
            tasks = []

            for section in plan.sections:
                research = research_results.get(section.title, {})
                # v5.1.2: Get per-section evidence matrix
                evidence_matrix = None
                if evidence_matrices and section.title in evidence_matrices:
                    evidence_matrix = evidence_matrices[section.title]
                task = self._draft_section_with_semaphore(
                    section=section,
                    research=research,
                    plan_subject=plan.subject,
                    semaphore=semaphore,
                    execution_mode=execution_mode,
                    evidence_matrix=evidence_matrix,
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle results
            drafts = []
            for section, result in zip(plan.sections, results):
                if isinstance(result, Exception):
                    logger.error(f"[SWARM] Drafting failed for '{section.title}': {result}")
                    drafts.append(SectionDraft(
                        section_title=section.title,
                        content="",
                        word_count=0,
                        sources_used=[],
                        documents_count=0,
                        generation_time_ms=0,
                        status="error",
                        error_message=str(result),
                    ))
                else:
                    drafts.append(result)

            return drafts

        else:
            # Sequential drafting (fallback)
            drafts = []
            for section in plan.sections:
                research = research_results.get(section.title, {})
                evidence_matrix = None
                if evidence_matrices and section.title in evidence_matrices:
                    evidence_matrix = evidence_matrices[section.title]
                try:
                    draft = await self._draft_section(
                        section, research, plan.subject,
                        execution_mode=execution_mode,
                        evidence_matrix=evidence_matrix,
                    )
                    drafts.append(draft)
                except Exception as e:
                    logger.error(f"[SWARM] Drafting failed for '{section.title}': {e}")
                    drafts.append(SectionDraft(
                        section_title=section.title,
                        content="",
                        word_count=0,
                        sources_used=[],
                        documents_count=0,
                        generation_time_ms=0,
                        status="error",
                        error_message=str(e),
                    ))

            return drafts

    async def _draft_section_with_semaphore(
        self,
        section: "SectionPlan",
        research: Dict[str, Any],
        plan_subject: str,
        semaphore: asyncio.Semaphore,
        execution_mode: str = "report",
        evidence_matrix=None,
    ) -> SectionDraft:
        """Draft a section with semaphore for concurrency control."""
        async with semaphore:
            return await self._draft_section(
                section, research, plan_subject,
                execution_mode=execution_mode,
                evidence_matrix=evidence_matrix,
            )

    async def _draft_section(
        self,
        section: "SectionPlan",
        research: Dict[str, Any],
        plan_subject: str,
        execution_mode: str = "report",
        evidence_matrix=None,
    ) -> SectionDraft:
        """
        Draft a single section using the Worker LLM.

        v5.1.2: Supports execution_mode for exploratory prompts.

        Args:
            section: Section plan
            research: Research results for this section
            plan_subject: Overall report subject
            execution_mode: report/exploratory (insight handled separately)
            evidence_matrix: Optional evidence matrix for exploratory mode

        Returns:
            SectionDraft with generated content
        """
        start_time = time.time()

        if not self.llm:
            return SectionDraft(
                section_title=section.title,
                content=f"[LLM not available - Section: {section.title}]",
                word_count=0,
                sources_used=[],
                documents_count=0,
                generation_time_ms=0,
                status="error",
                error_message="LLM module not available",
            )

        # Build context from research documents
        documents = research.get("documents", [])
        context = self._build_context(documents)

        # v5.1.2: Build prompt based on execution mode
        if execution_mode == "exploratory":
            # Build evidence summary for exploratory prompt
            evidence_summary = "Nessuna matrice evidenze disponibile."
            if evidence_matrix and hasattr(evidence_matrix, "entries") and evidence_matrix.entries:
                lines = []
                for entry in evidence_matrix.entries:
                    lines.append(
                        f"- [{entry.source_index}] {entry.condition}: "
                        f"{entry.outcomes} (forza: {entry.evidence_strength})"
                    )
                evidence_summary = "\n".join(lines)

            user_prompt = EXPLORATORY_WRITER_USER_PROMPT.format(
                section_title=section.title,
                report_subject=plan_subject,
                evidence_summary=evidence_summary,
                context=context if context else "Nessun contesto documentale disponibile.",
                max_tokens=section.max_tokens,
            )
            system_prompt = EXPLORATORY_WRITER_SYSTEM_PROMPT
        else:
            # Standard report mode — unchanged
            user_prompt = SECTION_WRITER_USER_PROMPT.format(
                section_title=section.title,
                report_subject=plan_subject,
                section_description=section.description,
                context=context if context else "No specific context available. Use your knowledge.",
                max_tokens=section.max_tokens,
                source_preference=section.source_preference.value,
            )
            system_prompt = SECTION_WRITER_SYSTEM_PROMPT

        try:
            # Call Worker LLM with timeout
            # v5.1.2: Pass system_prompt for mode-specific prompts
            response = await asyncio.wait_for(
                self._call_worker_llm(user_prompt, system_prompt=system_prompt),
                timeout=self.config.section_timeout,
            )

            # v5.0.4 RPT-006: Verify citations against actual document count
            # _build_context() caps at 10 docs, so valid citations are [1]..[min(N,10)]
            response = self._verify_citations(response, min(len(documents), 10))

            generation_time_ms = (time.time() - start_time) * 1000

            # v5.0.4: Record section-level metric
            if report_metrics:
                report_metrics.record_section_time(generation_time_ms / 1000.0)

            # v5.0.4: Compute heuristic quality score (no LLM calls)
            word_count = len(response.split())
            quality = self._compute_section_quality(response, word_count, len(documents))

            return SectionDraft(
                section_title=section.title,
                content=response,
                word_count=word_count,
                sources_used=research.get("sources_used", []),
                documents_count=len(documents),
                generation_time_ms=generation_time_ms,
                status="success",
                metadata={
                    "source_preference": section.source_preference.value,
                    "quality_score": quality["score"],
                    "quality_details": quality,
                },
            )

        except asyncio.TimeoutError:
            return SectionDraft(
                section_title=section.title,
                content="",
                word_count=0,
                sources_used=[],
                documents_count=len(documents),
                generation_time_ms=(time.time() - start_time) * 1000,
                status="error",
                error_message=f"Timeout after {self.config.section_timeout}s",
            )

        except Exception as e:
            return SectionDraft(
                section_title=section.title,
                content="",
                word_count=0,
                sources_used=[],
                documents_count=len(documents),
                generation_time_ms=(time.time() - start_time) * 1000,
                status="error",
                error_message=str(e),
            )

    async def _call_worker_llm(
        self, user_prompt: str, system_prompt: str = None,
    ) -> str:
        """Call the Worker LLM with section writing prompts."""
        # v5.0.3 RPT-002: Minimal prompt format — no structured markers
        # that Qwen3-4B-AWQ would echo back in output
        # v5.1.2: Accept custom system_prompt for mode-specific prompts
        sys_prompt = system_prompt or SECTION_WRITER_SYSTEM_PROMPT
        combined_prompt = f"""{sys_prompt}

{user_prompt}"""

        params = {
            "prompt": combined_prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # v6.0.1: Pass provider only — model resolved by inference module
        if self._provider:
            params["provider"] = self._provider

        # Call LLM
        result = await self.llm.generate(**params)

        # Extract response text
        if isinstance(result, dict):
            text = result.get("response", result.get("text", ""))
        else:
            text = str(result)

        # v5.0.3 RPT-002: Post-process to strip leaked prompt fragments
        return self._clean_draft_output(text)

    # v5.0.3 RPT-002: Patterns that signal end-of-content
    _STOP_PATTERNS = [
        "### Notes", "### Note", "### References", "### Riferimenti",
        "### SECTION", "### CONTEXT", "### REQUIREMENT",
        "### SYSTEM", "### TASK", "### WRITE",
        "### SEZIONE", "### FINE",
        "[End of section]", "[END OF SECTION]", "[Fine della sezione]",
        "This section is now complete",
        "Write the section", "Scrivi la sezione",
        "Scrivi circa", "Lunghezza target",
        "Section Description:",
        "Research Context", "Contesto di ricerca:",
        "Fonti:\n[",
    ]

    # v5.0.3 RPT-011: Tail degeneration markers
    _TAIL_MARKERS = [
        "Fermato.", "Finito.", "Fine del testo.", "Fine del report.",
        "Fine della sezione.", "Fine.", "Stop.", "Fatto.", "Completato.",
        "(Parole:", "(100%", "(senza ripetizioni",
        "Fonti citate:", "Fonti utilizzate:",
        "Fai attenzione:",  # prompt echo
    ]

    # v5.0.4: Pre-compiled regex patterns for _clean_draft_output / _verify_citations
    import re as _re
    _RE_HEADER = _re.compile(r"(#{1,3}\s+.+)")
    _RE_PARA_SPLIT = _re.compile(r"\n\s*\n")
    _RE_WHITESPACE = _re.compile(r"\s+")
    _RE_CITATION_ONLY = _re.compile(r"^(\[?\d+\]?\s*[,;]?\s*)+$")
    _RE_META_LINE = _re.compile(r"^-\s*(Stay focused|Target length|Source preference|Lunghezza target)")
    _RE_TOKEN_COUNT = _re.compile(r"^\(\d+\s*tokens?\)$", _re.IGNORECASE)
    _RE_CITATION_REF = _re.compile(r"\[(\d+)\]")
    _RE_DOUBLE_SPACE = _re.compile(r"  +")
    _RE_SPACE_BEFORE_PUNCT = _re.compile(r" ([.,;:!?])")
    _RE_EMPTY_PARENS = _re.compile(r"\(\s*\)")

    @staticmethod
    def _clean_draft_output(text: str) -> str:
        """Strip leaked prompt, deduplicate content, and truncate tail degeneration.

        Strategy:
        1. Truncate at first stop pattern (after 300 chars min)
        2. Truncate at first tail degeneration marker
        3. Detect repeated section headers → keep first block
        4. Detect line repetition loops → truncate
        5. Paragraph-level dedup
        6. Strip citation-only blocks
        7. Line cleanup
        """

        if not text or not text.strip():
            return ""

        # Phase 1: Truncate at first stop pattern (after min content)
        min_content = 300
        truncate_at = len(text)
        for pattern in SwarmExecutor._STOP_PATTERNS:
            idx = text.find(pattern, min_content)
            if idx != -1 and idx < truncate_at:
                truncate_at = idx
        text = text[:truncate_at]

        # Phase 2: Truncate at first tail degeneration marker
        truncate_at = len(text)
        for marker in SwarmExecutor._TAIL_MARKERS:
            idx = text.find(marker, min_content)
            if idx != -1 and idx < truncate_at:
                truncate_at = idx
        text = text[:truncate_at]

        # Phase 3: Detect repeated section headers (## Title appears 2+ times)
        header_match = SwarmExecutor._RE_HEADER.match(text)
        if header_match:
            header = header_match.group(1).strip()
            second_idx = text.find(header, len(header) + 1)
            if second_idx != -1 and second_idx > min_content:
                text = text[:second_idx].rstrip()

        # Phase 4: Detect line repetition loops
        # If the same line appears 3+ times consecutively, truncate at first occurrence
        lines = text.split("\n")
        delooped = []
        repeat_count = 0
        prev_stripped = None
        for line in lines:
            stripped = line.strip()
            if stripped and stripped == prev_stripped:
                repeat_count += 1
                if repeat_count >= 2:
                    # 3rd consecutive repeat — truncate here
                    break
            else:
                repeat_count = 0
            prev_stripped = stripped
            delooped.append(line)
        text = "\n".join(delooped)

        # Phase 5: Paragraph-level dedup
        paragraphs = SwarmExecutor._RE_PARA_SPLIT.split(text)
        seen = set()
        unique_paras = []
        for para in paragraphs:
            norm = SwarmExecutor._RE_WHITESPACE.sub(" ", para.strip().lower())
            if len(norm) < 20:
                unique_paras.append(para)
                continue
            fingerprint = norm[:80]
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique_paras.append(para)
        text = "\n\n".join(unique_paras)

        # Phase 6: Strip citation-only blocks (e.g. "[1] [2] [3] [4] [5]" repeated)
        lines = text.split("\n")
        cleaned = []
        citation_only_count = 0
        for line in lines:
            stripped = line.strip()
            # Detect lines that are only citations like "[1] [2] [3]" or "[1], [2], [3]"
            if stripped and SwarmExecutor._RE_CITATION_ONLY.match(stripped):
                citation_only_count += 1
                if citation_only_count <= 1:
                    cleaned.append(line)
                continue
            else:
                citation_only_count = 0
            # Skip meta lines
            if SwarmExecutor._RE_META_LINE.match(stripped):
                continue
            if SwarmExecutor._RE_TOKEN_COUNT.match(stripped):
                continue
            # v5.0.4 RPT-013: Strip stray --- separators mid-content
            if stripped in ("---", "----", "-----") and cleaned:
                continue
            if not stripped and not cleaned:
                continue
            cleaned.append(line)

        # v5.0.4 RPT-012: Strip orphan trailing citations like "[1]", "[2] [3]"
        while cleaned and (
            cleaned[-1].strip() in ("", "**", "***")
            or SwarmExecutor._RE_CITATION_ONLY.match(cleaned[-1].strip())
        ):
            cleaned.pop()
        # Also strip leading separators
        while cleaned and cleaned[0].strip() in ("", "---", "----"):
            cleaned.pop(0)

        return "\n".join(cleaned).strip()

    @staticmethod
    def _verify_citations(text: str, num_documents: int) -> str:
        """v5.0.4 RPT-006: Verify and fix citations in draft output.

        Removes or replaces citations [N] where N > num_documents (hallucinated).
        Strips orphan citation references that point to non-existent sources.
        Returns cleaned text with only valid citations.
        """
        if not text or num_documents <= 0:
            return text

        valid_range = set(range(1, num_documents + 1))

        def _replace_citation(match):
            """Replace invalid citations, keep valid ones."""
            n = int(match.group(1))
            if n in valid_range:
                return match.group(0)
            return ""

        # Replace invalid [N] citations (standalone or inline)
        cleaned = SwarmExecutor._RE_CITATION_REF.sub(_replace_citation, text)

        # Clean up artifacts from removal: double spaces, space before punctuation
        cleaned = SwarmExecutor._RE_DOUBLE_SPACE.sub(" ", cleaned)
        cleaned = SwarmExecutor._RE_SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
        cleaned = SwarmExecutor._RE_EMPTY_PARENS.sub("", cleaned)

        return cleaned.strip()

    @staticmethod
    def _compute_section_quality(
        content: str, word_count: int, documents_count: int
    ) -> Dict[str, Any]:
        """v5.0.4: Compute heuristic quality score for a section draft.

        Pure local computation — no LLM calls, zero latency overhead.
        Score 0.0-1.0 based on:
        - Word count adequacy (target: 200-800 words)
        - Structure (paragraphs, headers, bullets)
        - Citation presence (relative to available documents)
        - Content density (non-empty lines ratio)
        """
        if not content or word_count == 0:
            return {"score": 0.0, "grade": "F", "flags": ["empty_content"]}

        flags = []

        # 1. Word count score (0.0-1.0): sweet spot 200-800 words
        if word_count < 50:
            wc_score = 0.1
            flags.append("too_short")
        elif word_count < 150:
            wc_score = 0.4
            flags.append("short")
        elif word_count <= 800:
            wc_score = 1.0
        elif word_count <= 1200:
            wc_score = 0.8
        else:
            wc_score = 0.6
            flags.append("verbose")

        # 2. Structure score (0.0-1.0): paragraphs, lists, headers
        paragraphs = content.count("\n\n") + 1
        has_bullets = "- " in content or "* " in content
        has_headers = content.startswith("#") or "\n#" in content
        struct_score = min(1.0, (paragraphs / 4) * 0.5 + (0.25 if has_bullets else 0) + (0.25 if has_headers else 0))

        # 3. Citation score (0.0-1.0): citations relative to available docs
        citations = set(SwarmExecutor._RE_CITATION_REF.findall(content))
        if documents_count > 0:
            cite_score = min(1.0, len(citations) / min(documents_count, 5))
        else:
            # v6.4.1: LLM_REASONING sections don't need citations — higher base score
            cite_score = 0.8
        if not citations and documents_count > 0:
            flags.append("no_citations")

        # 4. Density score (0.0-1.0): ratio of non-empty lines
        lines = content.split("\n")
        non_empty = sum(1 for l in lines if l.strip())
        density_score = non_empty / max(len(lines), 1)

        # Weighted composite
        score = round(
            wc_score * 0.30 + struct_score * 0.25 + cite_score * 0.25 + density_score * 0.20,
            2,
        )

        # Grade
        if score >= 0.8:
            grade = "A"
        elif score >= 0.65:
            grade = "B"
        elif score >= 0.5:
            grade = "C"
        elif score >= 0.35:
            grade = "D"
        else:
            grade = "F"

        return {
            "score": score,
            "grade": grade,
            "word_count_score": round(wc_score, 2),
            "structure_score": round(struct_score, 2),
            "citation_score": round(cite_score, 2),
            "density_score": round(density_score, 2),
            "citations_found": len(citations),
            "paragraphs": paragraphs,
            "flags": flags,
        }

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from research documents."""
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents[:10], 1):  # Limit to top 10 docs
            text = doc.get("text", "")
            source = doc.get("source", "unknown")
            score = doc.get("score", 0)

            # Add source reference
            ref = f"[{i}]"
            if source == "rag":
                collection = doc.get("collection", "")
                ref = f"[{i}] (RAG: {collection})"
            elif source == "web":
                url = doc.get("url", "")
                ref = f"[{i}] (Web: {url[:50]}...)" if url else f"[{i}] (Web)"

            # v5.0.4: Smart truncation at sentence boundary instead of hard cut
            snippet = self._truncate_at_sentence(text, 1000) if len(text) > 1000 else text
            context_parts.append(f"{ref}\n{snippet}")

        return "\n\n".join(context_parts)

    @staticmethod
    def _truncate_at_sentence(text: str, max_length: int) -> str:
        """Truncate text at last complete sentence before max_length."""
        if len(text) <= max_length:
            return text

        truncated = text[:max_length]

        # Find last sentence boundary (keep at least 70% of target)
        min_keep = int(max_length * 0.7)
        for delimiter in (". ", "! ", "? ", ".\n"):
            last_idx = truncated.rfind(delimiter)
            if last_idx > min_keep:
                return text[:last_idx + 1]

        return truncated + "..."

    async def execute_section(
        self,
        section: "SectionPlan",
        collections: List[str],
        plan_subject: str,
        ctx: Any = None,
    ) -> SectionDraft:
        """
        Execute a single section (research + draft).

        Convenience method for processing one section at a time.

        Args:
            section: Section plan
            collections: RAG collections
            plan_subject: Overall report subject
            ctx: Security context

        Returns:
            SectionDraft with generated content
        """
        # Research
        research = await self._research_section(section, collections, ctx)

        # Draft
        draft = await self._draft_section(section, research, plan_subject)

        return draft
