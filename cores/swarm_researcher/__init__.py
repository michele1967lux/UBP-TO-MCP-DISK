"""
swarm_researcher/providers/__init__.py

Data classes and enums for multi-source research.

Version: 1.0.0
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# ============================================================================
# Enums
# ============================================================================


class SourceType(str, Enum):
    """Type of data source."""
    RAG = "rag"
    WEB = "web"
    HYBRID = "hybrid"
    LLM = "llm"


class SourcePreference(str, Enum):
    """Source preference strategy."""
    RAG_ONLY = "rag_only"
    WEB_ONLY = "web_only"
    RAG_FIRST = "rag_first"
    WEB_FIRST = "web_first"
    MIXED = "mixed"
    LLM_REASONING = "llm_reasoning"
    ADAPTIVE = "adaptive"


class AggregationStrategy(str, Enum):
    """Strategy for aggregating results."""
    CONCAT = "concat"
    UNION = "union"
    INTERLEAVE = "interleave"
    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted"


class CitationStyle(str, Enum):
    """Citation formatting style."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    HARVARD = "harvard"
    NUMERIC = "numeric"
    INLINE = "inline"


# ============================================================================
# Citation Data Classes
# ============================================================================


@dataclass
class Citation:
    """
    Structured citation for a source.
    
    Tracks source information for proper attribution
    in generated documents.
    """
    id: str
    source_type: SourceType
    
    # Content
    text: str  # Cited text/snippet
    context: str = ""  # Surrounding context
    
    # Source identification
    title: str = ""
    authors: List[str] = field(default_factory=list)
    date: Optional[str] = None
    url: Optional[str] = None
    page: Optional[str] = None
    
    # RAG-specific
    collection: Optional[str] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    
    # Quality metrics
    relevance_score: float = 0.0
    confidence: float = 0.0
    
    # Metadata
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_footnote(self, style: CitationStyle = CitationStyle.NUMERIC, number: int = 1) -> str:
        """Format as footnote."""
        if style == CitationStyle.NUMERIC:
            return f"[{number}]"
        elif style == CitationStyle.INLINE:
            author = self.authors[0] if self.authors else "Unknown"
            year = self.date[:4] if self.date else "n.d."
            return f"({author}, {year})"
        else:
            return f"[{number}]"
    
    def to_bibliography(self, style: CitationStyle = CitationStyle.APA) -> str:
        """Format for bibliography."""
        authors_str = ", ".join(self.authors) if self.authors else "Unknown"
        date_str = self.date if self.date else "n.d."
        title_str = self.title or "Untitled"
        
        if style == CitationStyle.APA:
            result = f"{authors_str} ({date_str}). {title_str}."
            if self.url:
                result += f" Retrieved from {self.url}"
            return result
        elif style == CitationStyle.IEEE:
            return f"{authors_str}, \"{title_str},\" {date_str}."
        else:
            return f"{authors_str}. {title_str}. {date_str}."
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type.value,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "title": self.title,
            "authors": self.authors,
            "date": self.date,
            "url": self.url,
            "collection": self.collection,
            "document_id": self.document_id,
            "relevance_score": self.relevance_score,
        }


@dataclass
class Bibliography:
    """Complete bibliography for a document."""
    citations: List[Citation] = field(default_factory=list)
    style: CitationStyle = CitationStyle.APA
    
    def add_citation(self, citation: Citation) -> None:
        """Add citation if not duplicate."""
        existing_ids = {c.id for c in self.citations}
        if citation.id not in existing_ids:
            self.citations.append(citation)
    
    def get_formatted(self) -> List[str]:
        """Get formatted bibliography entries."""
        return [c.to_bibliography(self.style) for c in self.citations]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "style": self.style.value,
            "count": len(self.citations),
            "entries": self.get_formatted(),
            "citations": [c.to_dict() for c in self.citations],
        }


# ============================================================================
# Source Configuration
# ============================================================================


@dataclass
class RAGConfig:
    """Configuration for RAG retrieval."""
    collections: List[str] = field(default_factory=list)
    top_k: int = 10
    min_score: float = 0.5
    rerank: bool = True
    rerank_top_k: int = 5
    enable_hyde: bool = False
    enable_expansion: bool = True
    metadata_filters: Optional[Dict[str, Any]] = None


@dataclass
class WebConfig:
    """Configuration for web search."""
    enabled: bool = True
    max_results: int = 5
    domains_whitelist: List[str] = field(default_factory=list)
    domains_blacklist: List[str] = field(default_factory=list)
    search_depth: str = "standard"  # standard, deep
    include_snippets: bool = True


@dataclass
class FallbackThreshold:
    """Thresholds for fallback triggering."""
    min_docs: int = 3
    min_score: float = 0.5
    min_coverage: float = 0.6


@dataclass
class SourceConfig:
    """Complete source configuration for research."""
    preference: SourcePreference = SourcePreference.RAG_FIRST
    
    # Source-specific configs
    rag_config: RAGConfig = field(default_factory=RAGConfig)
    web_config: WebConfig = field(default_factory=WebConfig)
    
    # Fallback settings
    fallback_enabled: bool = True
    fallback_threshold: FallbackThreshold = field(default_factory=FallbackThreshold)
    
    # Quality settings
    quality_threshold: float = 0.5
    max_sources_per_query: int = 10
    
    # Processing
    enable_deduplication: bool = True
    enable_citation_tracking: bool = True
    citation_style: CitationStyle = CitationStyle.APA
    
    # Parallel execution
    max_parallel_queries: int = 5
    timeout_seconds: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "preference": self.preference.value,
            "rag_collections": self.rag_config.collections,
            "rag_top_k": self.rag_config.top_k,
            "web_enabled": self.web_config.enabled,
            "fallback_enabled": self.fallback_enabled,
            "max_parallel": self.max_parallel_queries,
        }


# ============================================================================
# Research Task Data Classes
# ============================================================================


@dataclass
class ResearchTask:
    """A single research task to execute."""
    id: str
    query: str
    section_id: Optional[str] = None
    
    # Configuration
    source_preference: SourcePreference = SourcePreference.RAG_FIRST
    source_config: Optional[SourceConfig] = None
    
    # Context
    context: str = ""
    required_data_types: List[str] = field(default_factory=list)
    
    # Priority and dependencies
    priority: int = 0
    depends_on: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RetrievedDocument:
    """A document retrieved from a source."""
    id: str
    content: str
    source_type: SourceType
    
    # Metadata
    title: str = ""
    url: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    date: Optional[str] = None
    
    # RAG-specific
    collection: Optional[str] = None
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    
    # Scores
    relevance_score: float = 0.0
    rerank_score: Optional[float] = None
    
    # Position
    original_rank: int = 0
    
    def content_hash(self) -> str:
        """Generate hash for deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()[:16]
    
    def to_citation(self) -> Citation:
        """Convert to citation."""
        return Citation(
            id=f"cite_{self.id}",
            source_type=self.source_type,
            text=self.content[:500],
            title=self.title,
            authors=self.authors,
            date=self.date,
            url=self.url,
            collection=self.collection,
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            relevance_score=self.relevance_score,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:300] + "..." if len(self.content) > 300 else self.content,
            "source_type": self.source_type.value,
            "title": self.title,
            "url": self.url,
            "collection": self.collection,
            "relevance_score": self.relevance_score,
            "rerank_score": self.rerank_score,
        }


# ============================================================================
# Research Result Data Classes
# ============================================================================


@dataclass
class ResearchResult:
    """Result of a single research query."""
    task_id: str
    query: str
    source_type: SourceType
    
    # Results
    documents: List[RetrievedDocument] = field(default_factory=list)
    total_found: int = 0
    
    # Quality metrics
    relevance_scores: List[float] = field(default_factory=list)
    average_score: float = 0.0
    coverage_score: float = 0.0
    
    # Citations
    citations: List[Citation] = field(default_factory=list)
    
    # Execution info
    execution_time_ms: int = 0
    source_breakdown: Dict[str, int] = field(default_factory=dict)
    
    # Status
    success: bool = True
    error: Optional[str] = None
    used_fallback: bool = False
    fallback_source: Optional[SourceType] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "source_type": self.source_type.value,
            "documents_count": len(self.documents),
            "total_found": self.total_found,
            "average_score": self.average_score,
            "coverage_score": self.coverage_score,
            "citations_count": len(self.citations),
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
            "used_fallback": self.used_fallback,
        }


@dataclass
class SectionResearchResult:
    """Aggregated research result for a document section."""
    section_id: str
    
    # Research results
    primary_results: List[ResearchResult] = field(default_factory=list)
    fallback_results: List[ResearchResult] = field(default_factory=list)
    
    # Aggregated content
    documents: List[RetrievedDocument] = field(default_factory=list)
    relevant_content: List[str] = field(default_factory=list)
    
    # Citations for this section
    citations: List[Citation] = field(default_factory=list)
    
    # Coverage analysis
    queries_executed: List[str] = field(default_factory=list)
    coverage_gaps: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Quality
    overall_quality: float = 0.0
    confidence: float = 0.0
    
    # Execution
    total_time_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "documents_count": len(self.documents),
            "citations_count": len(self.citations),
            "queries_executed": len(self.queries_executed),
            "coverage_gaps": self.coverage_gaps,
            "overall_quality": self.overall_quality,
            "confidence": self.confidence,
            "total_time_ms": self.total_time_ms,
        }


@dataclass
class AggregatedResult:
    """Result of aggregating multiple research results."""
    documents: List[RetrievedDocument] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    
    # Aggregation info
    strategy_used: AggregationStrategy = AggregationStrategy.RRF
    sources_merged: int = 0
    duplicates_removed: int = 0
    
    # Quality
    overall_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents_count": len(self.documents),
            "citations_count": len(self.citations),
            "strategy": self.strategy_used.value,
            "sources_merged": self.sources_merged,
            "duplicates_removed": self.duplicates_removed,
            "overall_score": self.overall_score,
        }


# ============================================================================
# Quality Data Classes
# ============================================================================


@dataclass
class SourceQuality:
    """Quality assessment of a source."""
    source_id: str
    source_type: SourceType
    
    # Scores
    relevance: float = 0.0
    authority: float = 0.0  # Based on source reputation
    freshness: float = 0.0  # Based on date
    completeness: float = 0.0
    
    # Overall
    overall_score: float = 0.0
    
    # Flags
    is_primary_source: bool = False
    is_verified: bool = False


@dataclass
class CoverageReport:
    """Report on research coverage."""
    queries_total: int = 0
    queries_successful: int = 0
    queries_failed: int = 0
    
    # Coverage
    topics_covered: List[str] = field(default_factory=list)
    topics_missing: List[str] = field(default_factory=list)
    coverage_percentage: float = 0.0
    
    # Suggestions
    additional_queries: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "queries_total": self.queries_total,
            "queries_successful": self.queries_successful,
            "coverage_percentage": self.coverage_percentage,
            "topics_missing": self.topics_missing,
            "additional_queries": self.additional_queries,
        }


# ============================================================================
# Execution Data Classes
# ============================================================================


@dataclass
class ExecutionProgress:
    """Progress tracking for research execution."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    current_task: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    @property
    def progress_percentage(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "progress_percentage": self.progress_percentage,
            "current_task": self.current_task,
        }


@dataclass
class ExecutionStats:
    """Statistics for research execution."""
    total_queries: int = 0
    total_documents: int = 0
    total_citations: int = 0
    
    # By source
    rag_queries: int = 0
    web_queries: int = 0
    rag_documents: int = 0
    web_documents: int = 0
    
    # Fallbacks
    fallbacks_triggered: int = 0
    
    # Timing
    total_time_ms: int = 0
    average_query_time_ms: int = 0
    
    # Quality
    average_relevance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "total_documents": self.total_documents,
            "total_citations": self.total_citations,
            "rag_queries": self.rag_queries,
            "web_queries": self.web_queries,
            "fallbacks_triggered": self.fallbacks_triggered,
            "total_time_ms": self.total_time_ms,
            "average_relevance": self.average_relevance,
        }


# Export all
__all__ = [
    # Enums
    "SourceType",
    "SourcePreference",
    "AggregationStrategy",
    "CitationStyle",
    # Citation
    "Citation",
    "Bibliography",
    # Configuration
    "RAGConfig",
    "WebConfig",
    "FallbackThreshold",
    "SourceConfig",
    # Tasks
    "ResearchTask",
    "RetrievedDocument",
    # Results
    "ResearchResult",
    "SectionResearchResult",
    "AggregatedResult",
    # Quality
    "SourceQuality",
    "CoverageReport",
    # Execution
    "ExecutionProgress",
    "ExecutionStats",
]
