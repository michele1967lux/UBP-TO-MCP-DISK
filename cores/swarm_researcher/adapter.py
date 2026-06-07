"""
swarm_researcher/adapter.py

Bridge Layer - Multi-source parallel research operations.

Provides:
- Parallel research execution
- Citation tracking
- Result aggregation
- Quality analysis
- Coverage reporting

Version: 1.0.0
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .providers import (
    # Enums
    SourceType,
    SourcePreference,
    AggregationStrategy,
    CitationStyle,
    # Configuration
    SourceConfig,
    RAGConfig,
    WebConfig,
    FallbackThreshold,
    # Tasks
    ResearchTask,
    RetrievedDocument,
    # Results
    ResearchResult,
    SectionResearchResult,
    AggregatedResults,
    # Citations
    Citation,
    # Quality
    SourceQuality,
    CoverageReport,
    # Execution
    ExecutionProgress,
    ExecutionStats,
    # Components
    ParallelExecutor,
    SourceRouter,
    Aggregator,
    CitationTracker,
    QualityScorer,
    CoverageAnalyzer,
)

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

# Aliases for backwards compatibility
AggregatedResult = AggregatedResults
ResultAggregator = Aggregator
SectionAggregator = Aggregator

logger = logging.getLogger(__name__)


class SwarmResearcherAdapter:
    """
    Main adapter for multi-source parallel research.
    
    Provides operations for:
    - Parallel query execution
    - Multi-source retrieval (RAG + Web)
    - Citation tracking and bibliography
    - Result aggregation and deduplication
    - Quality scoring and coverage analysis
    """
    
    def __init__(
        self,
        module_path: Path,
        di_container: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        self.module_path = Path(module_path)
        self.di_container = di_container
        self.event_bus = event_bus
        
        # Components
        self._executor: Optional[ParallelExecutor] = None
        self._router: Optional[SourceRouter] = None
        self._aggregator: Optional[ResultAggregator] = None
        self._section_aggregator: Optional[SectionAggregator] = None
        self._citation_tracker: Optional[CitationTracker] = None
        self._quality_scorer: Optional[QualityScorer] = None
        self._coverage_analyzer: Optional[CoverageAnalyzer] = None
        
        # External modules (resolved at runtime)
        self._rag_module: Optional[Any] = None
        self._web_module: Optional[Any] = None
        self._rerank_module: Optional[Any] = None
        
        # State
        self._initialized = False
        
        # Configuration
        self._max_workers = 5
        self._default_timeout = 60
        self._default_citation_style = CitationStyle.APA
    
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
    # Lifecycle Operations
    # ========================================================================
    
    async def initialize(
        self,
        max_workers: int = 5,
        default_timeout: int = 60,
        citation_style: str = "apa",
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Initialize the swarm researcher.
        
        Args:
            max_workers: Maximum parallel workers
            default_timeout: Default timeout per task (seconds)
            citation_style: Default citation style
        """
        self._max_workers = max_workers
        self._default_timeout = default_timeout
        self._default_citation_style = CitationStyle(citation_style)
        
        # Resolve external modules
        await self._resolve_modules()
        
        # Initialize components
        self._router = SourceRouter()
        self._router.set_di_container(self.di_container)

        self._executor = ParallelExecutor(max_concurrent=max_workers)

        self._aggregator = Aggregator()
        self._citation_tracker = CitationTracker()
        self._quality_scorer = QualityScorer()
        self._coverage_analyzer = CoverageAnalyzer()
        
        self._initialized = True
        
        logger.info("swarm_researcher initialized")
        
        return {
            "status": "initialized",
            "module": "swarm_researcher",
            "version": "1.0.0",
            "max_workers": max_workers,
            "rag_available": self._rag_module is not None,
            "web_available": self._web_module is not None,
            "rerank_available": self._rerank_module is not None,
        }
    
    async def shutdown(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Shutdown the swarm researcher."""
        self._initialized = False
        return {"status": "shutdown"}
    
    async def health_check(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Health check."""
        return {
            "module": "swarm_researcher",
            "version": "1.0.0",
            "status": "healthy" if self._initialized else "not_initialized",
            "rag_available": self._rag_module is not None,
            "web_available": self._web_module is not None,
        }
    
    async def _resolve_modules(self) -> None:
        """Resolve external modules from DI container."""
        if not self.di_container:
            return
        
        # Try to resolve RAG module
        for rag_name in ["rag_qdrant", "retrieval_strategy", "rag_module"]:
            try:
                if hasattr(self.di_container, "resolve"):
                    self._rag_module = await self.di_container.resolve(rag_name)
                elif hasattr(self.di_container, "get"):
                    self._rag_module = self.di_container.get(rag_name)
                if self._rag_module:
                    break
            except Exception:
                continue
        
        # Try to resolve web search module
        for web_name in ["web_search", "web_module", "serp_module"]:
            try:
                if hasattr(self.di_container, "resolve"):
                    self._web_module = await self.di_container.resolve(web_name)
                elif hasattr(self.di_container, "get"):
                    self._web_module = self.di_container.get(web_name)
                if self._web_module:
                    break
            except Exception:
                continue
        
        # Try to resolve rerank module
        for rerank_name in ["reranker", "rerank_module", "cross_encoder"]:
            try:
                if hasattr(self.di_container, "resolve"):
                    self._rerank_module = await self.di_container.resolve(rerank_name)
                elif hasattr(self.di_container, "get"):
                    self._rerank_module = self.di_container.get(rerank_name)
                if self._rerank_module:
                    break
            except Exception:
                continue
    
    # ========================================================================
    # Core Research Operations
    # ========================================================================
    
    async def research_single(
        self,
        query: str,
        source_config: Optional[Dict[str, Any]] = None,
        section_id: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute research for a single query.
        
        Args:
            query: Search query
            source_config: Source configuration
            section_id: Optional section identifier
        
        Returns:
            Dict with research result
        """
        if not self._initialized:
            await self.initialize()
        
        config = self._build_source_config(source_config)
        
        task = ResearchTask(
            id=f"task_{uuid.uuid4().hex[:8]}",
            query=query,
            section_id=section_id,
            source_preference=config.preference,
            source_config=config,
        )
        
        result = await self._router.execute_research(task)
        
        # Track citations
        if config.enable_citation_tracking and result.documents:
            result.citations = self._citation_tracker.track_batch(result.documents)
        
        return {
            "success": result.success,
            "result": result.to_dict(),
            "documents": [d.to_dict() for d in result.documents],
            "citations": [c.to_dict() for c in result.citations],
            "execution_time_ms": result.execution_time_ms,
        }
    
    async def research_parallel(
        self,
        queries: List[str],
        source_config: Optional[Dict[str, Any]] = None,
        extract_citations: bool = True,
        deduplicate: bool = True,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute research for multiple queries in parallel.
        
        Args:
            queries: List of search queries
            source_config: Source configuration
            extract_citations: Whether to extract citations
            deduplicate: Whether to deduplicate results
        
        Returns:
            Dict with aggregated research results
        """
        if not self._initialized:
            await self.initialize()
        
        if not queries:
            return {
                "success": True,
                "results": [],
                "aggregated": {"documents": [], "citations": []},
            }
        
        config = self._build_source_config(source_config)
        
        # Create tasks
        tasks = [
            ResearchTask(
                id=f"task_{i}_{uuid.uuid4().hex[:6]}",
                query=query,
                source_preference=config.preference,
                source_config=config,
            )
            for i, query in enumerate(queries)
        ]
        
        # Execute in parallel
        results = await self._executor.execute(
            tasks=tasks,
            executor_func=self._router.execute_research,
            respect_dependencies=False,
        )
        
        # Aggregate results
        aggregated = self._aggregator.aggregate(
            results=results,
            deduplicate=deduplicate,
        )
        
        # Track citations
        if extract_citations:
            for doc in aggregated.documents:
                self._citation_tracker.track(doc)
            aggregated.citations = self._citation_tracker.get_all_citations()
        
        # Get execution stats
        stats = self._executor.get_stats()
        
        return {
            "success": True,
            "results": [r.to_dict() for r in results],
            "aggregated": aggregated.to_dict(),
            "documents": [d.to_dict() for d in aggregated.documents],
            "citations": [c.to_dict() for c in aggregated.citations],
            "stats": stats.to_dict(),
        }
    
    async def research_section(
        self,
        section_id: str,
        queries: List[str],
        source_config: Optional[Dict[str, Any]] = None,
        max_documents: int = 10,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute research optimized for a document section.
        
        Args:
            section_id: Section identifier
            queries: Queries for this section
            source_config: Source configuration
            max_documents: Maximum documents per section
        
        Returns:
            Dict with section research result
        """
        if not self._initialized:
            await self.initialize()
        
        config = self._build_source_config(source_config)
        
        # Create tasks with section context
        tasks = [
            ResearchTask(
                id=f"sect_{section_id}_{i}",
                query=query,
                section_id=section_id,
                source_preference=config.preference,
                source_config=config,
            )
            for i, query in enumerate(queries)
        ]
        
        # Execute
        results = await self._executor.execute(
            tasks=tasks,
            executor_func=self._router.execute_research,
        )
        
        # Aggregate for section
        section_data = self._section_aggregator.aggregate_for_section(
            section_id=section_id,
            results=results,
            max_documents=max_documents,
        )
        
        return {
            "success": True,
            "section_id": section_id,
            **section_data,
        }
    
    async def research_with_dependencies(
        self,
        task_graph: List[Dict[str, Any]],
        source_config: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute research tasks respecting dependencies.
        
        Args:
            task_graph: List of task definitions with dependencies
            source_config: Source configuration
        
        Returns:
            Dict with ordered results
        """
        if not self._initialized:
            await self.initialize()
        
        config = self._build_source_config(source_config)
        
        # Build tasks from graph
        tasks = []
        for item in task_graph:
            # Handle section plan format
            if "suggested_queries" in item:
                queries = item.get("suggested_queries", [item.get("title", "")])
                for i, query in enumerate(queries):
                    task = ResearchTask(
                        id=f"{item.get('id', 'task')}_{i}",
                        query=query,
                        section_id=item.get("id"),
                        source_preference=SourcePreference(
                            item.get("source_preference", config.preference.value)
                        ),
                        source_config=config,
                        depends_on=item.get("depends_on", []),
                        priority=item.get("order", 0),
                    )
                    tasks.append(task)
            else:
                task = ResearchTask(
                    id=item.get("id", f"task_{len(tasks)}"),
                    query=item.get("query", ""),
                    section_id=item.get("section_id"),
                    source_preference=SourcePreference(
                        item.get("source_preference", config.preference.value)
                    ),
                    source_config=config,
                    depends_on=item.get("depends_on", []),
                    priority=item.get("priority", 0),
                )
                tasks.append(task)
        
        # Execute with dependency awareness
        results = await self._executor.execute(
            tasks=tasks,
            executor_func=self._router.execute_research,
            respect_dependencies=True,
        )
        
        # Group results by section
        results_by_section: Dict[str, List[ResearchResult]] = {}
        for result in results:
            task = next((t for t in tasks if t.id == result.task_id), None)
            if task and task.section_id:
                if task.section_id not in results_by_section:
                    results_by_section[task.section_id] = []
                results_by_section[task.section_id].append(result)
        
        # Aggregate per section
        section_results = {}
        for section_id, section_results_list in results_by_section.items():
            section_results[section_id] = self._section_aggregator.aggregate_for_section(
                section_id=section_id,
                results=section_results_list,
            )
        
        return {
            "success": True,
            "tasks_executed": len(tasks),
            "results": [r.to_dict() for r in results],
            "by_section": section_results,
            "stats": self._executor.get_stats().to_dict(),
        }
    
    # ========================================================================
    # Aggregation Operations
    # ========================================================================
    
    async def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        strategy: str = "rrf",
        weights: Optional[Dict[str, float]] = None,
        deduplicate: bool = True,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Aggregate multiple research results.
        
        Args:
            results: List of research results
            strategy: Aggregation strategy (rrf, weighted, interleave, union)
            weights: Optional weights for weighted aggregation
            deduplicate: Whether to deduplicate
        
        Returns:
            Dict with aggregated result
        """
        if not self._initialized:
            await self.initialize()
        
        # Convert dicts back to ResearchResult objects
        research_results = []
        for r in results:
            documents = [
                RetrievedDocument(
                    id=d.get("id", ""),
                    content=d.get("content", ""),
                    source_type=SourceType(d.get("source_type", "rag")),
                    title=d.get("title", ""),
                    url=d.get("url"),
                    relevance_score=d.get("relevance_score", 0.0),
                )
                for d in r.get("documents", [])
            ]
            
            research_results.append(ResearchResult(
                task_id=r.get("task_id", ""),
                query=r.get("query", ""),
                source_type=SourceType(r.get("source_type", "rag")),
                documents=documents,
            ))
        
        aggregated = self._aggregator.aggregate(
            results=research_results,
            strategy=AggregationStrategy(strategy),
            weights=weights,
            deduplicate=deduplicate,
        )
        
        return {
            "success": True,
            "aggregated": aggregated.to_dict(),
            "documents": [d.to_dict() for d in aggregated.documents],
        }
    
    async def deduplicate(
        self,
        documents: List[Dict[str, Any]],
        similarity_threshold: float = 0.85,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Deduplicate documents.
        
        Args:
            documents: List of documents
            similarity_threshold: Similarity threshold for deduplication
        
        Returns:
            Dict with deduplicated documents
        """
        if not self._initialized:
            await self.initialize()
        
        # Convert to RetrievedDocument
        docs = [
            RetrievedDocument(
                id=d.get("id", str(i)),
                content=d.get("content", ""),
                source_type=SourceType(d.get("source_type", "rag")),
                title=d.get("title", ""),
                relevance_score=d.get("relevance_score", 0.0),
            )
            for i, d in enumerate(documents)
        ]
        
        # Deduplicate
        aggregator = ResultAggregator(similarity_threshold=similarity_threshold)
        unique = aggregator._deduplicate_documents(docs)
        
        return {
            "success": True,
            "original_count": len(documents),
            "unique_count": len(unique),
            "duplicates_removed": len(documents) - len(unique),
            "documents": [d.to_dict() for d in unique],
        }
    
    # ========================================================================
    # Citation Operations
    # ========================================================================
    
    async def track_citations(
        self,
        documents: List[Dict[str, Any]],
        style: str = "apa",
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Track citations for documents.
        
        Args:
            documents: List of documents to cite
            style: Citation style
        
        Returns:
            Dict with tracked citations
        """
        if not self._initialized:
            await self.initialize()
        
        # Convert to RetrievedDocument
        docs = [
            RetrievedDocument(
                id=d.get("id", str(i)),
                content=d.get("content", ""),
                source_type=SourceType(d.get("source_type", "rag")),
                title=d.get("title", ""),
                authors=d.get("authors", []),
                date=d.get("date"),
                url=d.get("url"),
                collection=d.get("collection"),
                document_id=d.get("document_id"),
            )
            for i, d in enumerate(documents)
        ]
        
        # Track
        tracker = CitationTracker(default_style=CitationStyle(style))
        citations = tracker.track_batch(docs)
        
        return {
            "success": True,
            "citations": [c.to_dict() for c in citations],
            "count": len(citations),
        }
    
    async def generate_bibliography(
        self,
        citations: Optional[List[Dict[str, Any]]] = None,
        style: str = "apa",
        numbered: bool = True,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate formatted bibliography.
        
        Args:
            citations: Optional list of citations (uses tracked if not provided)
            style: Citation style
            numbered: Whether to number entries
        
        Returns:
            Dict with formatted bibliography
        """
        if not self._initialized:
            await self.initialize()
        
        citation_style = CitationStyle(style)
        
        if citations:
            # Convert to Citation objects
            citation_objects = [
                Citation(
                    id=c.get("id", str(i)),
                    source_type=SourceType(c.get("source_type", "rag")),
                    text=c.get("text", ""),
                    title=c.get("title", ""),
                    authors=c.get("authors", []),
                    date=c.get("date"),
                    url=c.get("url"),
                    relevance_score=c.get("relevance_score", 0.0),
                )
                for i, c in enumerate(citations)
            ]
            bibliography = Bibliography(citations=citation_objects, style=citation_style)
        else:
            bibliography = self._citation_tracker.generate_bibliography(style=citation_style)
        
        # Format text
        formatted_text = self._citation_tracker.format_bibliography_text(
            style=citation_style,
            numbered=numbered,
        )
        
        return {
            "success": True,
            "bibliography": bibliography.to_dict(),
            "formatted_text": formatted_text,
            "style": style,
        }
    
    # ========================================================================
    # Quality Operations
    # ========================================================================
    
    async def score_quality(
        self,
        documents: List[Dict[str, Any]],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Score quality of documents.
        
        Args:
            documents: Documents to score
        
        Returns:
            Dict with quality scores
        """
        if not self._initialized:
            await self.initialize()
        
        # Convert to RetrievedDocument
        docs = [
            RetrievedDocument(
                id=d.get("id", str(i)),
                content=d.get("content", ""),
                source_type=SourceType(d.get("source_type", "rag")),
                title=d.get("title", ""),
                date=d.get("date"),
                url=d.get("url"),
                relevance_score=d.get("relevance_score", 0.0),
            )
            for i, d in enumerate(documents)
        ]
        
        # Score each document
        scores = [self._quality_scorer.score_document(doc) for doc in docs]
        
        return {
            "success": True,
            "scores": [
                {
                    "document_id": s.source_id,
                    "relevance": s.relevance,
                    "authority": s.authority,
                    "freshness": s.freshness,
                    "overall": s.overall_score,
                }
                for s in scores
            ],
            "average_quality": sum(s.overall_score for s in scores) / len(scores) if scores else 0,
        }
    
    async def analyze_coverage(
        self,
        queries: List[str],
        results: List[Dict[str, Any]],
        required_topics: Optional[List[str]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze research coverage.
        
        Args:
            queries: Original queries
            results: Research results
            required_topics: Required topics to cover
        
        Returns:
            Dict with coverage analysis
        """
        if not self._initialized:
            await self.initialize()
        
        # Convert results
        research_results = []
        for r in results:
            documents = [
                RetrievedDocument(
                    id=d.get("id", ""),
                    content=d.get("content", ""),
                    source_type=SourceType(d.get("source_type", "rag")),
                    title=d.get("title", ""),
                )
                for d in r.get("documents", [])
            ]
            
            research_results.append(ResearchResult(
                task_id=r.get("task_id", ""),
                query=r.get("query", ""),
                source_type=SourceType(r.get("source_type", "rag")),
                documents=documents,
                success=r.get("success", True),
            ))
        
        report = self._coverage_analyzer.analyze_coverage(
            queries=queries,
            results=research_results,
            required_topics=required_topics,
        )
        
        return {
            "success": True,
            "coverage": report.to_dict(),
        }
    
    # ========================================================================
    # Utility Operations
    # ========================================================================
    
    async def get_progress(
        self,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get current execution progress."""
        if not self._executor:
            return {"progress": None}
        
        return {
            "success": True,
            "progress": self._executor.get_progress().to_dict(),
        }
    
    async def get_stats(
        self,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self._executor:
            return {"stats": None}
        
        return {
            "success": True,
            "stats": self._executor.get_stats().to_dict(),
        }
    
    async def clear_citations(
        self,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Clear tracked citations."""
        if self._citation_tracker:
            self._citation_tracker.clear()
        
        return {"success": True, "message": "Citations cleared"}
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _build_source_config(
        self,
        config: Optional[Dict[str, Any]],
    ) -> SourceConfig:
        """Build SourceConfig from dict."""
        if not config:
            return SourceConfig()
        
        rag_config = RAGConfig(
            collections=config.get("collections", config.get("rag_collections", [])),
            top_k=config.get("top_k", config.get("rag_top_k", 10)),
            min_score=config.get("min_score", config.get("rag_min_score", 0.5)),
            rerank=config.get("rerank", True),
            rerank_top_k=config.get("rerank_top_k", 5),
            enable_hyde=config.get("enable_hyde", False),
            enable_expansion=config.get("enable_expansion", True),
            metadata_filters=config.get("metadata_filters"),
        )
        
        web_config = WebConfig(
            enabled=config.get("web_enabled", True),
            max_results=config.get("web_max_results", 5),
        )
        
        return SourceConfig(
            preference=SourcePreference(config.get("preference", config.get("source_preference", "rag_first"))),
            rag_config=rag_config,
            web_config=web_config,
            fallback_enabled=config.get("fallback_enabled", True),
            quality_threshold=config.get("quality_threshold", 0.5),
            max_parallel_queries=config.get("max_parallel", self._max_workers),
            timeout_seconds=config.get("timeout", self._default_timeout),
            enable_deduplication=config.get("deduplicate", True),
            enable_citation_tracking=config.get("extract_citations", True),
            citation_style=CitationStyle(config.get("citation_style", self._default_citation_style.value)),
        )
    
    async def _emit_event(self, event: Dict[str, Any]) -> None:
        """Emit event to event bus."""
        if self.event_bus:
            try:
                await self.event_bus.publish(f"swarm_researcher.{event['type']}", event)
            except Exception as e:
                logger.debug(f"Event emission failed: {e}")
