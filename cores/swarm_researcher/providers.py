"""
swarm_researcher/providers.py

Domain Layer - Data classes and business logic for parallel research.

Contains:
- Enums: SourceType, AggregationStrategy, QualityLevel
- Data classes: ResearchQuery, ResearchResult, Citation, etc.
- Components: ParallelExecutor, SourceRouter, Aggregator, QualityScorer, CitationTracker

Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class SourceType(str, Enum):
    """Types of research sources."""
    RAG = "rag"
    WEB = "web"
    DATABASE = "database"
    API = "api"
    LLM = "llm"
    CACHE = "cache"


class AggregationStrategy(str, Enum):
    """Strategies for aggregating results."""
    CONCAT = "concat"
    UNION = "union"
    INTERLEAVE = "interleave"
    RRF = "rrf"
    WEIGHTED = "weighted"
    BEST_MATCH = "best_match"


class QualityLevel(str, Enum):
    """Quality levels for results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ResearchStatus(str, Enum):
    """Status of a research task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# ============================================================================
# Core Data Classes
# ============================================================================


@dataclass
class ResearchQuery:
    """A research query."""
    id: str
    query: str
    section_id: Optional[str] = None
    sources: List[SourceType] = field(default_factory=lambda: [SourceType.RAG])
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    source_preference: Optional[Any] = None
    source_config: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "section_id": self.section_id,
            "sources": [s.value for s in self.sources],
            "priority": self.priority,
        }


@dataclass
class Document:
    """A retrieved document."""
    id: str
    content: str
    score: float = 0.0
    source: SourceType = SourceType.RAG
    metadata: Dict[str, Any] = field(default_factory=dict)
    collection: Optional[str] = None
    chunk_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "score": self.score,
            "source": self.source.value,
            "metadata": self.metadata,
        }


@dataclass
class Citation:
    """A citation reference."""
    id: str
    title: str
    source: str = ""
    url: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[str] = None
    document_id: Optional[str] = None
    relevance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "authors": self.authors,
            "year": self.year,
        }


@dataclass
class ResearchResult:
    """Result of a research query."""
    query_id: str
    query: str
    documents: List[Document] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    status: ResearchStatus = ResearchStatus.COMPLETED
    success: bool = True
    error: Optional[str] = None
    time_ms: float = 0.0
    source_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "document_count": len(self.documents),
            "citation_count": len(self.citations),
            "status": self.status.value,
            "time_ms": self.time_ms,
            "source_breakdown": self.source_breakdown,
        }


@dataclass
class AggregatedResults:
    """Aggregated results from multiple queries."""
    documents: List[Document] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    by_section: Dict[str, List[Document]] = field(default_factory=dict)
    total_sources: int = 0
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_count": len(self.documents),
            "citation_count": len(self.citations),
            "sections_covered": len(self.by_section),
            "total_sources": self.total_sources,
            "quality_score": self.quality_score,
        }


@dataclass
class QualityAssessment:
    """Quality assessment of results."""
    overall_score: float = 0.0
    coverage_score: float = 0.0
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    recency_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "coverage_score": self.coverage_score,
            "relevance_score": self.relevance_score,
            "diversity_score": self.diversity_score,
            "issues_count": len(self.issues),
        }


@dataclass
class CoverageAnalysis:
    """Analysis of research coverage."""
    sections_covered: List[str] = field(default_factory=list)
    sections_missing: List[str] = field(default_factory=list)
    coverage_percentage: float = 0.0
    gaps: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sections_covered": self.sections_covered,
            "sections_missing": self.sections_missing,
            "coverage_percentage": self.coverage_percentage,
            "gaps": self.gaps,
        }


# ============================================================================
# Parallel Executor
# ============================================================================


class ParallelExecutor:
    """Executes research queries in parallel."""

    def __init__(
        self,
        max_concurrent: int = 5,
        timeout: float = 30.0,
    ):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def execute_parallel(
        self,
        queries: List[ResearchQuery],
        executor_fn: Callable,
        progress_callback: Optional[Callable] = None,
    ) -> List[ResearchResult]:
        """Execute queries in parallel."""
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def execute_with_semaphore(query: ResearchQuery) -> ResearchResult:
            async with self._semaphore:
                try:
                    start = time.perf_counter()
                    result = await asyncio.wait_for(
                        executor_fn(query),
                        timeout=self.timeout
                    )
                    elapsed = (time.perf_counter() - start) * 1000
                    result.time_ms = elapsed
                    
                    if progress_callback:
                        await progress_callback({
                            "query_id": query.id,
                            "status": "completed",
                            "time_ms": elapsed,
                        })
                    
                    return result
                except asyncio.TimeoutError:
                    return ResearchResult(
                        query_id=query.id,
                        query=query.query,
                        status=ResearchStatus.FAILED,
                        error=f"Timeout after {self.timeout}s",
                    )
                except Exception as e:
                    return ResearchResult(
                        query_id=query.id,
                        query=query.query,
                        status=ResearchStatus.FAILED,
                        error=str(e),
                    )

        # Sort by priority
        sorted_queries = sorted(queries, key=lambda q: -q.priority)
        
        # Execute in parallel
        tasks = [execute_with_semaphore(q) for q in sorted_queries]
        results = await asyncio.gather(*tasks)
        
        return list(results)

    async def execute_with_dependencies(
        self,
        queries: List[ResearchQuery],
        executor_fn: Callable,
    ) -> List[ResearchResult]:
        """Execute respecting dependencies."""
        results = {}
        remaining = {q.id: q for q in queries}
        
        while remaining:
            # Find ready queries (no pending dependencies)
            ready = [
                q for q in remaining.values()
                if all(d in results for d in q.dependencies)
            ]
            
            if not ready:
                # Break deadlock
                ready = list(remaining.values())[:self.max_concurrent]
            
            # Execute batch
            batch_results = await self.execute_parallel(ready, executor_fn)
            
            for result in batch_results:
                results[result.query_id] = result
                remaining.pop(result.query_id, None)
        
        return list(results.values())

    async def execute(
        self,
        tasks: List[ResearchQuery],
        executor_func: Callable,
        respect_dependencies: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> List[ResearchResult]:
        """Unified execute method — dispatches to parallel or dependency-aware execution."""
        if respect_dependencies:
            results = await self.execute_with_dependencies(tasks, executor_func)
        else:
            results = await self.execute_parallel(tasks, executor_func, progress_callback)
        self._last_results = results
        return results

    def get_stats(self) -> "ExecutionStats":
        """Return stats from last execution."""
        results = getattr(self, "_last_results", [])
        total_docs = sum(len(r.documents) for r in results)
        total_cites = sum(len(r.citations) for r in results)
        total_time = sum(r.time_ms for r in results)
        return ExecutionStats(
            total_queries=len(results),
            total_documents=total_docs,
            total_citations=total_cites,
            total_time_ms=total_time,
        )

    def get_progress(self) -> "ExecutionProgress":
        """Return progress of current/last execution."""
        results = getattr(self, "_last_results", [])
        completed = len([r for r in results if r.status == ResearchStatus.COMPLETED])
        failed = len([r for r in results if r.status == ResearchStatus.FAILED])
        total = len(results)
        return ExecutionProgress(
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            percentage=(completed / total * 100) if total > 0 else 0.0,
        )


# ============================================================================
# Source Router
# ============================================================================


class SourceRouter:
    """Routes queries to appropriate sources."""

    def __init__(self):
        self._source_handlers: Dict[SourceType, Callable] = {}
        self._di_container: Any = None

    def set_di_container(self, di_container: Any) -> None:
        """Set DI container for module resolution."""
        self._di_container = di_container

    def register_handler(self, source: SourceType, handler: Callable) -> None:
        """Register a source handler."""
        self._source_handlers[source] = handler

    async def execute_research(self, task: ResearchQuery) -> ResearchResult:
        """Execute research for a single task, returning a ResearchResult."""
        try:
            documents = await self.route(task, di_container=self._di_container)
            return ResearchResult(
                query_id=task.id,
                query=task.query,
                documents=documents,
                status=ResearchStatus.COMPLETED,
                success=True,
            )
        except Exception as e:
            logger.error(f"Research failed for task {task.id}: {e}")
            return ResearchResult(
                query_id=task.id,
                query=task.query,
                status=ResearchStatus.FAILED,
                error=str(e),
            )

    async def route(
        self,
        query: ResearchQuery,
        di_container: Any = None,
    ) -> List[Document]:
        """Route query to sources and collect results."""
        all_documents = []
        
        for source in query.sources:
            try:
                handler = self._source_handlers.get(source)
                if handler:
                    docs = await handler(query.query, query.constraints)
                    for doc in docs:
                        doc.source = source
                    all_documents.extend(docs)
                elif di_container:
                    docs = await self._route_via_di(query, source, di_container)
                    all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Source {source} failed: {e}")
        
        return all_documents

    async def _route_via_di(
        self,
        query: ResearchQuery,
        source: SourceType,
        di_container: Any,
    ) -> List[Document]:
        """Route via DI container."""
        if source == SourceType.RAG:
            module = await di_container.resolve("retrieval_strategy")
            if module:
                result = await module.retrieve(
                    query=query.query,
                    top_k=query.constraints.get("top_k", 10),
                )
                return [
                    Document(
                        id=d.get("id", str(i)),
                        content=d.get("content", d.get("text", "")),
                        score=d.get("score", 0.0),
                        source=SourceType.RAG,
                        metadata=d.get("metadata", {}),
                    )
                    for i, d in enumerate(result.get("documents", []))
                ]
        return []


# ============================================================================
# Aggregator
# ============================================================================


class Aggregator:
    """Aggregates results from multiple sources."""

    def aggregate(
        self,
        results: List[ResearchResult],
        strategy: AggregationStrategy = AggregationStrategy.RRF,
        weights: Optional[Dict[str, float]] = None,
        deduplicate: bool = True,
    ) -> AggregatedResults:
        """Aggregate results using specified strategy."""
        if strategy == AggregationStrategy.RRF:
            return self._rrf_aggregate(results)
        elif strategy == AggregationStrategy.CONCAT:
            return self._concat_aggregate(results)
        elif strategy == AggregationStrategy.INTERLEAVE:
            return self._interleave_aggregate(results)
        elif strategy == AggregationStrategy.WEIGHTED:
            return self._weighted_aggregate(results, weights or {})
        else:
            return self._concat_aggregate(results)

    def _rrf_aggregate(
        self,
        results: List[ResearchResult],
        k: int = 60,
    ) -> AggregatedResults:
        """Reciprocal Rank Fusion aggregation."""
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        by_section: Dict[str, List[Document]] = {}
        all_citations = []
        
        for result in results:
            for rank, doc in enumerate(result.documents):
                doc_id = doc.id or hashlib.md5(doc.content[:100].encode()).hexdigest()
                rrf_score = 1.0 / (k + rank + 1)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
                doc_map[doc_id] = doc
                
                # Track by section
                section_id = result.query_id
                if section_id not in by_section:
                    by_section[section_id] = []
                if doc not in by_section[section_id]:
                    by_section[section_id].append(doc)
            
            all_citations.extend(result.citations)
        
        # Sort by RRF score
        sorted_ids = sorted(doc_scores.keys(), key=lambda x: -doc_scores[x])
        sorted_docs = [doc_map[did] for did in sorted_ids]
        
        # Update scores
        for doc in sorted_docs:
            doc_id = doc.id or hashlib.md5(doc.content[:100].encode()).hexdigest()
            doc.score = doc_scores[doc_id]
        
        return AggregatedResults(
            documents=sorted_docs,
            citations=self._deduplicate_citations(all_citations),
            by_section=by_section,
            total_sources=len(results),
        )

    def _concat_aggregate(self, results: List[ResearchResult]) -> AggregatedResults:
        """Simple concatenation."""
        all_docs = []
        all_citations = []
        by_section = {}
        
        for result in results:
            all_docs.extend(result.documents)
            all_citations.extend(result.citations)
            by_section[result.query_id] = result.documents
        
        return AggregatedResults(
            documents=all_docs,
            citations=self._deduplicate_citations(all_citations),
            by_section=by_section,
            total_sources=len(results),
        )

    def _interleave_aggregate(self, results: List[ResearchResult]) -> AggregatedResults:
        """Interleave documents from different sources."""
        all_docs = []
        max_len = max(len(r.documents) for r in results) if results else 0
        
        for i in range(max_len):
            for result in results:
                if i < len(result.documents):
                    all_docs.append(result.documents[i])
        
        all_citations = []
        by_section = {}
        for result in results:
            all_citations.extend(result.citations)
            by_section[result.query_id] = result.documents
        
        return AggregatedResults(
            documents=all_docs,
            citations=self._deduplicate_citations(all_citations),
            by_section=by_section,
            total_sources=len(results),
        )

    def _weighted_aggregate(
        self,
        results: List[ResearchResult],
        weights: Dict[str, float],
    ) -> AggregatedResults:
        """Weighted aggregation."""
        all_docs = []
        
        for result in results:
            weight = weights.get(result.query_id, 1.0)
            for doc in result.documents:
                doc.score *= weight
            all_docs.extend(result.documents)
        
        all_docs.sort(key=lambda d: -d.score)
        
        all_citations = []
        by_section = {}
        for result in results:
            all_citations.extend(result.citations)
            by_section[result.query_id] = result.documents
        
        return AggregatedResults(
            documents=all_docs,
            citations=self._deduplicate_citations(all_citations),
            by_section=by_section,
            total_sources=len(results),
        )

    def _deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """Remove duplicate citations."""
        seen = set()
        unique = []
        for cit in citations:
            key = (cit.title.lower(), cit.source)
            if key not in seen:
                seen.add(key)
                unique.append(cit)
        return unique

    def aggregate_for_section(
        self,
        section_id: str,
        results: List[ResearchResult],
        strategy: AggregationStrategy = AggregationStrategy.RRF,
    ) -> AggregatedResults:
        """Aggregate results for a specific section."""
        section_results = [r for r in results if r.query_id == section_id]
        if not section_results:
            section_results = results
        return self.aggregate(section_results, strategy=strategy)


# ============================================================================
# Quality Scorer
# ============================================================================


class QualityScorer:
    """Scores quality of research results."""

    def score_document(self, document: Document) -> float:
        """Score a single document's quality."""
        score = document.score if document.score > 0 else 0.5
        if document.content and len(document.content) > 200:
            score = min(1.0, score + 0.1)
        return score

    def __init__(
        self,
        min_relevance: float = 0.5,
        min_sources: int = 3,
    ):
        self.min_relevance = min_relevance
        self.min_sources = min_sources

    def assess_quality(
        self,
        results: AggregatedResults,
        expected_sections: Optional[List[str]] = None,
    ) -> QualityAssessment:
        """Assess quality of aggregated results."""
        issues = []
        recommendations = []
        
        # Coverage score
        if expected_sections:
            covered = set(results.by_section.keys())
            expected = set(expected_sections)
            coverage = len(covered & expected) / len(expected) if expected else 1.0
            missing = expected - covered
            if missing:
                issues.append(f"Missing sections: {', '.join(missing)}")
                recommendations.append("Run additional research for missing sections")
        else:
            coverage = 1.0 if results.documents else 0.0
        
        # Relevance score
        if results.documents:
            avg_score = sum(d.score for d in results.documents) / len(results.documents)
            relevance = min(1.0, avg_score / self.min_relevance) if self.min_relevance > 0 else 1.0
        else:
            relevance = 0.0
            issues.append("No documents retrieved")
        
        # Diversity score
        sources = set(d.source for d in results.documents)
        diversity = len(sources) / len(SourceType) if results.documents else 0.0
        
        # Recency (placeholder - would need date metadata)
        recency = 0.8
        
        # Overall score
        overall = (coverage * 0.3 + relevance * 0.4 + diversity * 0.2 + recency * 0.1)
        
        if overall < 0.5:
            recommendations.append("Consider expanding research scope")
        
        return QualityAssessment(
            overall_score=overall,
            coverage_score=coverage,
            relevance_score=relevance,
            diversity_score=diversity,
            recency_score=recency,
            issues=issues,
            recommendations=recommendations,
        )


# ============================================================================
# Citation Tracker
# ============================================================================


class CitationTracker:
    """Tracks citations from retrieved documents."""

    def __init__(self):
        self._citations: Dict[str, Citation] = {}
        self._doc_to_citations: Dict[str, List[str]] = {}

    def extract_citations(self, documents: List[Document]) -> List[Citation]:
        """Extract citations from documents."""
        citations = []
        
        for doc in documents:
            citation = self._create_citation(doc)
            if citation.id not in self._citations:
                self._citations[citation.id] = citation
                citations.append(citation)
            
            if doc.id not in self._doc_to_citations:
                self._doc_to_citations[doc.id] = []
            self._doc_to_citations[doc.id].append(citation.id)
        
        return citations

    def _create_citation(self, doc: Document) -> Citation:
        """Create citation from document."""
        metadata = doc.metadata
        
        return Citation(
            id=hashlib.md5(f"{doc.id}{doc.content[:50]}".encode()).hexdigest()[:12],
            title=metadata.get("title", f"Document {doc.id}"),
            source=metadata.get("source", doc.source.value),
            url=metadata.get("url"),
            authors=metadata.get("authors", []),
            year=metadata.get("year"),
            document_id=doc.id,
            relevance_score=doc.score,
        )

    def get_all_citations(self) -> List[Citation]:
        """Get all tracked citations."""
        return list(self._citations.values())

    def get_citation_ids(self) -> List[str]:
        """Get all citation IDs."""
        return list(self._citations.keys())

    def clear(self) -> None:
        """Clear all tracked citations."""
        self._citations.clear()
        self._doc_to_citations.clear()

    def track(self, doc: Document) -> Citation:
        """Track a single document and return its citation."""
        citations = self.extract_citations([doc])
        return citations[0] if citations else self._create_citation(doc)

    def track_batch(self, documents: List[Document]) -> List[Citation]:
        """Track multiple documents and return their citations."""
        return self.extract_citations(documents)

    def generate_bibliography(self, style: str = "apa") -> List[Dict[str, Any]]:
        """Generate bibliography entries from tracked citations."""
        return [{"id": c.id, "title": c.title, "source": c.source, 
                 "authors": c.authors, "year": c.year, "url": c.url}
                for c in self._citations.values()]

    def format_bibliography_text(self, style: str = "apa") -> str:
        """Format bibliography as text."""
        lines = []
        for c in self._citations.values():
            authors = ", ".join(c.authors) if c.authors else "Unknown"
            year = f"({c.year})" if c.year else ""
            lines.append(f"{authors} {year}. {c.title}. {c.source or ''}")
        return "\n".join(lines)


# ============================================================================
# Exports and Aliases
# ============================================================================

# Aliases for backward compatibility with adapter
ResearchTask = ResearchQuery
RetrievedDocument = Document
SectionResearchResult = ResearchResult


@dataclass
class RAGConfig:
    """Configuration for RAG source."""
    collections: List[str] = field(default_factory=list)
    top_k: int = 10
    min_score: float = 0.5
    rerank: bool = True
    rerank_top_k: int = 5
    hybrid: bool = True
    enable_hyde: bool = False
    enable_expansion: bool = True
    metadata_filters: Optional[Dict[str, Any]] = None


@dataclass
class WebConfig:
    """Configuration for web search source."""
    enabled: bool = True
    engines: List[str] = field(default_factory=lambda: ["google"])
    max_results: int = 5
    include_snippets: bool = True
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)


@dataclass
class FallbackThreshold:
    """Thresholds for fallback behavior."""
    min_documents: int = 3
    min_coverage: float = 0.5
    min_relevance: float = 0.5
    enable_web_fallback: bool = True


class SourcePreference(str, Enum):
    """Source preference for research."""
    RAG_ONLY = "rag_only"
    WEB_ONLY = "web_only"
    RAG_FIRST = "rag_first"
    WEB_FIRST = "web_first"
    MIXED = "mixed"
    ADAPTIVE = "adaptive"


class CitationStyle(str, Enum):
    """Citation styles."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    HARVARD = "harvard"


@dataclass
class SourceConfig:
    """Complete source configuration for research."""
    preference: SourcePreference = SourcePreference.RAG_FIRST
    rag_config: RAGConfig = field(default_factory=RAGConfig)
    web_config: WebConfig = field(default_factory=WebConfig)
    fallback_enabled: bool = True
    fallback_threshold: FallbackThreshold = field(default_factory=FallbackThreshold)
    quality_threshold: float = 0.5
    max_sources_per_query: int = 10
    enable_deduplication: bool = True
    enable_citation_tracking: bool = True
    citation_style: CitationStyle = CitationStyle.APA
    max_parallel_queries: int = 5
    timeout_seconds: int = 60


@dataclass
class SourceQuality:
    """Quality metrics for a source."""
    relevance_score: float = 0.0
    coverage_score: float = 0.0
    recency_score: float = 0.0
    authority_score: float = 0.0
    overall_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevance": self.relevance_score,
            "coverage": self.coverage_score,
            "overall": self.overall_score,
        }


@dataclass
class CoverageReport:
    """Report on research coverage."""
    total_sections: int = 0
    covered_sections: int = 0
    coverage_percentage: float = 0.0
    missing_sections: List[str] = field(default_factory=list)
    low_coverage_sections: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sections": self.total_sections,
            "covered_sections": self.covered_sections,
            "coverage_percentage": self.coverage_percentage,
            "missing_sections": self.missing_sections,
        }


@dataclass
class ExecutionProgress:
    """Progress of research execution."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    current_task: Optional[str] = None
    percentage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total_tasks,
            "completed": self.completed_tasks,
            "failed": self.failed_tasks,
            "percentage": self.percentage,
        }


@dataclass
class ExecutionStats:
    """Statistics from research execution."""
    total_queries: int = 0
    total_documents: int = 0
    total_citations: int = 0
    total_time_ms: float = 0.0
    avg_relevance: float = 0.0
    sources_used: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "total_documents": self.total_documents,
            "total_citations": self.total_citations,
            "total_time_ms": self.total_time_ms,
        }


class CoverageAnalyzer:
    """Analyzes research coverage across queries and sections."""

    def analyze_coverage(
        self,
        queries: List[str],
        results: List[ResearchResult],
        required_topics: Optional[List[str]] = None,
    ) -> CoverageReport:
        """Analyze coverage of research results."""
        covered = set()
        for r in results:
            if r.documents:
                covered.add(r.query_id)
        
        total = len(required_topics) if required_topics else len(queries)
        missing = []
        if required_topics:
            for topic in required_topics:
                if not any(topic.lower() in r.query.lower() for r in results if r.documents):
                    missing.append(topic)
        
        covered_count = total - len(missing)
        pct = (covered_count / total * 100) if total > 0 else 0.0
        
        return CoverageReport(
            total_sections=total,
            covered_sections=covered_count,
            coverage_percentage=pct,
            missing_sections=missing,
        )


__all__ = [
    # Enums
    "SourceType",
    "AggregationStrategy",
    "QualityLevel",
    "ResearchStatus",
    "SourcePreference",
    "CitationStyle",
    # Data classes
    "ResearchQuery",
    "Document",
    "Citation",
    "ResearchResult",
    "AggregatedResults",
    "QualityAssessment",
    "CoverageAnalysis",
    # Aliases
    "ResearchTask",
    "RetrievedDocument",
    "SectionResearchResult",
    # Config classes
    "SourceConfig",
    "RAGConfig",
    "WebConfig",
    "FallbackThreshold",
    # Additional classes
    "SourceQuality",
    "CoverageReport",
    "ExecutionProgress",
    "ExecutionStats",
    # Also export as AggregatedResult for adapter
    "AggregatedResults",
    # Components
    "ParallelExecutor",
    "SourceRouter",
    "Aggregator",
    "QualityScorer",
    "CitationTracker",
    "CoverageAnalyzer",
]

# Alias
AggregatedResult = AggregatedResults
Bibliography = List[Citation]
