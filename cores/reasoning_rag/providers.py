"""
reasoning_rag/providers.py

Logic Layer - ZERO dependencies from backend.app
Must be testable standalone.

Provides:
- ReasoningResult: Complete reasoning result with trace
- SubQuestion: Self-Ask sub-question representation
- ReasoningStep: Chain-of-Thought step
- Evidence: Source attribution with spans
- Claim: Extracted factual claim
- Verification: Fact-check result
- QueryAnalysis: Query complexity and intent analysis
- ReasoningTrace: Complete reasoning log
- SelfAskReasoner: Iterative sub-question decomposition
- ChainOfThoughtReasoner: Interleaved reasoning and retrieval
- EvidenceAttributor: Citation and source tracking
- VerificationProvider: Multi-source fact checking
- QueryAnalyzer: Query analysis and strategy selection
- ReasoningSessionManager: Session management
- ReasoningCacheProvider: Redis caching
- ReasoningWorkerPool: Parallel task execution
- ReasoningMetricsCollector: Comprehensive statistics

v1.0.0: Initial release with full reasoning capabilities
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Protocol

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class ReasoningStrategy(Enum):
    """Available reasoning strategies."""
    SELF_ASK = "self_ask"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    EVIDENCE_ATTRIBUTION = "evidence_attribution"
    VERIFICATION = "verification"
    DIRECT = "direct"
    HYBRID = "hybrid"


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"


class QueryIntent(Enum):
    """Query intent types."""
    FACTUAL = "factual"
    EXPLANATORY = "explanatory"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    CAUSAL = "causal"
    DEFINITIONAL = "definitional"
    EVALUATIVE = "evaluative"


class VerificationStatus(Enum):
    """Claim verification status."""
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


class AttributionType(Enum):
    """Evidence attribution type."""
    DIRECT = "direct"
    INFERRED = "inferred"
    PARTIAL = "partial"
    NONE = "none"


class TaskStatus(Enum):
    """Worker task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Worker task priority."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class ReasoningConfig:
    """Core reasoning configuration."""
    enabled: bool = True
    default_strategy: str = "auto"
    max_reasoning_depth: int = 5
    temperature: float = 0.3
    max_tokens: int = 800
    timeout_seconds: int = 60
    retry_enabled: bool = True
    max_retries: int = 2


@dataclass
class SelfAskConfig:
    """Self-Ask strategy configuration."""
    enabled: bool = True
    max_iterations: int = 5
    min_iterations: int = 1
    convergence_threshold: float = 0.85
    sub_question_temperature: float = 0.4
    integration_temperature: float = 0.2
    max_sub_questions_per_iteration: int = 3
    retrieval_top_k: int = 5
    early_stop_on_confidence: bool = True
    confidence_threshold: float = 0.8


@dataclass
class ChainOfThoughtConfig:
    """Chain-of-Thought strategy configuration."""
    enabled: bool = True
    max_reasoning_steps: int = 8
    min_reasoning_steps: int = 2
    auto_retrieval_threshold: float = 0.6
    reasoning_temperature: float = 0.3
    synthesis_temperature: float = 0.2
    retrieval_top_k: int = 3
    interleave_mode: str = "adaptive"
    thought_verification: bool = True
    step_by_step_logging: bool = True


@dataclass
class EvidenceConfig:
    """Evidence attribution configuration."""
    enabled: bool = True
    min_confidence: float = 0.5
    citation_format: str = "inline"
    track_spans: bool = True
    source_deduplication: bool = True
    claim_extraction_enabled: bool = True
    verification_level: str = "standard"


@dataclass
class VerificationConfig:
    """Verification configuration."""
    enabled: bool = True
    multi_source_check: bool = True
    min_sources_for_verification: int = 2
    contradiction_detection: bool = True
    contradiction_threshold: float = 0.7
    confidence_aggregation: str = "weighted_average"
    grounding_validation: bool = True
    hallucination_check: bool = True
    fact_check_temperature: float = 0.1


@dataclass
class RetrievalConfig:
    """Retrieval delegation configuration."""
    module: str = "retrieval_strategy"
    operation: str = "retrieve"
    default_top_k: int = 5
    rerank_enabled: bool = True
    rerank_top_k: int = 3
    min_relevance_score: float = 0.5
    hybrid_search: bool = True
    timeout_seconds: int = 10
    fallback_to_llm: bool = True


@dataclass
class CacheConfig:
    """Redis cache configuration."""
    enabled: bool = True
    ttl_seconds: int = 3600
    base_prefix: str = "ubp"
    env: str = "dev"
    cache_sub_questions: bool = True
    cache_retrievals: bool = True
    cache_reasoning_steps: bool = False
    cache_final_answers: bool = True
    semantic_matching: bool = True
    semantic_threshold: float = 0.92

    @property
    def prefix(self) -> str:
        return f"{self.base_prefix}:{self.env}:reasoning:cache"


@dataclass
class SessionConfig:
    """Session management configuration."""
    enabled: bool = True
    ttl_seconds: int = 3600
    max_history_size: int = 50
    persist_reasoning_trace: bool = True
    persist_retrievals: bool = True
    auto_cleanup: bool = True


@dataclass
class WorkerPoolConfig:
    """Worker pool configuration."""
    enabled: bool = True
    pool_size: int = 4
    max_pool_size: int = 8
    task_timeout_seconds: int = 30
    queue_max_size: int = 100
    retry_on_failure: bool = True
    max_task_retries: int = 2
    backoff_multiplier: float = 1.5


@dataclass
class MetricsConfig:
    """Metrics collection configuration."""
    enabled: bool = True
    collect_timings: bool = True
    collect_strategy_distribution: bool = True
    collect_iteration_counts: bool = True
    collect_retrieval_stats: bool = True
    collect_confidence_scores: bool = True
    collect_verification_stats: bool = True
    retention_hours: int = 24


@dataclass
class DebugConfig:
    """Debug configuration."""
    enabled: bool = False
    log_prompts: bool = False
    log_responses: bool = False
    log_sub_questions: bool = True
    log_reasoning_steps: bool = True
    log_retrievals: bool = True
    log_evidence: bool = True
    log_verification: bool = True
    trace_execution: bool = False


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class RetrievedDocument:
    """A document retrieved from the knowledge base."""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    page: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "score": round(self.score, 3),
            "source": self.source,
            "page": self.page,
            "metadata": self.metadata,
        }


@dataclass
class SubQuestion:
    """A sub-question generated during Self-Ask reasoning."""
    question_id: str
    question: str
    iteration: int
    answer: Optional[str] = None
    retrieved_docs: List[RetrievedDocument] = field(default_factory=list)
    confidence: float = 0.0
    answered: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "iteration": self.iteration,
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "answered": self.answered,
            "doc_count": len(self.retrieved_docs),
        }


@dataclass
class ReasoningStep:
    """A single step in Chain-of-Thought reasoning."""
    step_id: str
    step_number: int
    thought: str
    needs_retrieval: bool = False
    retrieval_query: Optional[str] = None
    retrieved_docs: List[RetrievedDocument] = field(default_factory=list)
    intermediate_conclusion: Optional[str] = None
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "thought": self.thought,
            "needs_retrieval": self.needs_retrieval,
            "retrieval_query": self.retrieval_query,
            "intermediate_conclusion": self.intermediate_conclusion,
            "confidence": round(self.confidence, 3),
            "doc_count": len(self.retrieved_docs),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Claim:
    """An extracted factual claim."""
    claim_id: str
    text: str
    claim_type: str  # fact, opinion, inference
    importance: str  # high, medium, low
    needs_verification: bool = True
    source_statement: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "claim_type": self.claim_type,
            "importance": self.importance,
            "needs_verification": self.needs_verification,
        }


@dataclass
class Evidence:
    """Evidence supporting a claim with attribution."""
    evidence_id: str
    claim_id: str
    source_doc_id: str
    supporting_text: str
    confidence: float
    attribution_type: AttributionType
    text_span: Optional[Tuple[int, int]] = None  # (start, end) in source
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "claim_id": self.claim_id,
            "source_doc_id": self.source_doc_id,
            "supporting_text": self.supporting_text[:200] + "..." if len(self.supporting_text) > 200 else self.supporting_text,
            "confidence": round(self.confidence, 3),
            "attribution_type": self.attribution_type.value,
            "text_span": self.text_span,
        }


@dataclass
class Citation:
    """A citation reference."""
    citation_id: int
    source_doc_id: str
    cited_text: str
    page: Optional[int] = None
    source_title: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "source_doc_id": self.source_doc_id,
            "cited_text": self.cited_text,
            "page": self.page,
            "source_title": self.source_title,
        }


@dataclass
class Verification:
    """Verification result for a claim."""
    claim_id: str
    claim_text: str
    status: VerificationStatus
    supporting_sources: List[str] = field(default_factory=list)
    contradicting_sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "status": self.status.value,
            "supporting_sources": self.supporting_sources,
            "contradicting_sources": self.contradicting_sources,
            "confidence": round(self.confidence, 3),
            "notes": self.notes,
        }


@dataclass
class Contradiction:
    """A detected contradiction between sources."""
    claim: str
    source_a_id: str
    source_a_says: str
    source_b_id: str
    source_b_says: str
    severity: str  # high, medium, low
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "source_a_id": self.source_a_id,
            "source_a_says": self.source_a_says,
            "source_b_id": self.source_b_id,
            "source_b_says": self.source_b_says,
            "severity": self.severity,
        }


@dataclass
class QueryAnalysis:
    """Query analysis result."""
    query: str
    complexity: QueryComplexity
    intent: QueryIntent
    requires_multi_source: bool
    requires_reasoning: bool
    requires_verification: bool
    recommended_strategy: ReasoningStrategy
    strategy_reason: str
    estimated_steps: int
    key_entities: List[str] = field(default_factory=list)
    language: str = "en"
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "complexity": self.complexity.value,
            "intent": self.intent.value,
            "requires_multi_source": self.requires_multi_source,
            "requires_reasoning": self.requires_reasoning,
            "requires_verification": self.requires_verification,
            "recommended_strategy": self.recommended_strategy.value,
            "strategy_reason": self.strategy_reason,
            "estimated_steps": self.estimated_steps,
            "key_entities": self.key_entities,
            "language": self.language,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ReasoningTraceEntry:
    """A single entry in the reasoning trace."""
    entry_id: str
    entry_type: str  # sub_question, reasoning_step, retrieval, evidence, verification
    timestamp: datetime
    content: Dict[str, Any]
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "entry_type": self.entry_type,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a query."""
    trace_id: str
    query: str
    strategy: ReasoningStrategy
    entries: List[ReasoningTraceEntry] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def add_entry(
        self,
        entry_type: str,
        content: Dict[str, Any],
        duration_ms: float = 0.0,
    ) -> ReasoningTraceEntry:
        entry = ReasoningTraceEntry(
            entry_id=str(uuid.uuid4()),
            entry_type=entry_type,
            timestamp=datetime.utcnow(),
            content=content,
            duration_ms=duration_ms,
        )
        self.entries.append(entry)
        return entry
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "strategy": self.strategy.value,
            "entry_count": len(self.entries),
            "entries": [e.to_dict() for e in self.entries[-20:]],  # Last 20 entries
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_ms": sum(e.duration_ms for e in self.entries),
        }


@dataclass
class ReasoningResult:
    """Complete result from reasoning process."""
    session_id: str
    query: str
    strategy_used: ReasoningStrategy
    answer: str
    confidence: float
    reasoning_trace: ReasoningTrace
    sub_questions: List[SubQuestion] = field(default_factory=list)
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    verifications: List[Verification] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    retrieved_docs: List[RetrievedDocument] = field(default_factory=list)
    time_ms: float = 0.0
    iteration_count: int = 0
    step_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "strategy_used": self.strategy_used.value,
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "sub_questions": [sq.to_dict() for sq in self.sub_questions],
            "reasoning_steps": [rs.to_dict() for rs in self.reasoning_steps],
            "claims": [c.to_dict() for c in self.claims],
            "evidence": [e.to_dict() for e in self.evidence],
            "citations": [c.to_dict() for c in self.citations],
            "verifications": [v.to_dict() for v in self.verifications],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "reasoning_trace": self.reasoning_trace.to_dict(),
            "time_ms": round(self.time_ms, 2),
            "iteration_count": self.iteration_count,
            "step_count": self.step_count,
            "total_docs_retrieved": len(self.retrieved_docs),
        }


@dataclass
class ReasoningSession:
    """Session state for reasoning interactions."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    history: List[Dict[str, Any]] = field(default_factory=list)
    traces: List[ReasoningTrace] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "history_count": len(self.history),
            "trace_count": len(self.traces),
            "history": self.history[-10:],
            "metadata": self.metadata,
        }


@dataclass
class WorkerTask:
    """Task for worker pool execution."""
    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0


@dataclass
class WorkerStats:
    """Worker pool statistics."""
    active_workers: int
    pending_tasks: int
    completed_tasks: int
    failed_tasks: int
    avg_task_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_workers": self.active_workers,
            "pending_tasks": self.pending_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "avg_task_time_ms": round(self.avg_task_time_ms, 2),
        }


# ============================================================================
# Query Analyzer
# ============================================================================


class QueryAnalyzer:
    """
    Analyzes queries to determine complexity, intent, and best strategy.
    
    Features:
    - Complexity detection (simple to multi-hop)
    - Intent classification
    - Strategy recommendation
    - Language detection
    - Entity extraction (basic)
    """
    
    def __init__(
        self,
        complexity_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.complexity_thresholds = complexity_thresholds or {
            "simple": 0.3,
            "moderate": 0.6,
            "complex": 0.8,
        }
        
        # Italian markers for language detection
        self._italian_markers = {
            "come", "cosa", "perché", "quando", "dove", "chi", "quale",
            "il", "la", "lo", "gli", "le", "un", "una", "uno",
            "è", "sono", "sei", "siamo", "essere", "avere", "fare",
            "non", "che", "per", "con", "su", "da", "in", "di",
        }
        
        # Multi-hop indicators
        self._multi_hop_indicators = {
            "and", "also", "additionally", "furthermore", "compare",
            "versus", "vs", "difference", "both", "between",
            "e", "anche", "inoltre", "confronta", "differenza",
        }
        
        # Complex query indicators
        self._complex_indicators = {
            "why", "how", "explain", "analyze", "evaluate",
            "implications", "consequences", "impact", "effect",
            "perché", "come", "spiega", "analizza", "valuta",
        }
        
        # Intent patterns
        self._intent_patterns = {
            QueryIntent.DEFINITIONAL: ["what is", "what are", "define", "meaning", "cos'è", "cosa sono", "definisci"],
            QueryIntent.EXPLANATORY: ["explain", "how does", "how do", "describe", "spiega", "come funziona"],
            QueryIntent.COMPARATIVE: ["compare", "difference", "versus", "vs", "better", "confronta", "differenza", "meglio"],
            QueryIntent.PROCEDURAL: ["how to", "steps", "process", "procedure", "guide", "come fare", "procedura", "passi"],
            QueryIntent.CAUSAL: ["why", "cause", "reason", "because", "effect", "perché", "causa", "motivo"],
            QueryIntent.EVALUATIVE: ["should", "best", "recommend", "evaluate", "worth", "consiglia", "migliore", "vale"],
        }
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a query and return comprehensive analysis.
        
        Args:
            query: The query to analyze
            
        Returns:
            QueryAnalysis with complexity, intent, and strategy recommendation
        """
        query_lower = query.lower()
        words = set(query_lower.split())
        
        # Detect language
        language = self._detect_language(words)
        
        # Detect complexity
        complexity, complexity_score = self._detect_complexity(query_lower, words)
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Extract entities (basic)
        entities = self._extract_entities(query)
        
        # Determine requirements
        requires_multi_source = complexity in (QueryComplexity.COMPLEX, QueryComplexity.MULTI_HOP)
        requires_reasoning = complexity != QueryComplexity.SIMPLE or intent in (QueryIntent.CAUSAL, QueryIntent.EXPLANATORY)
        requires_verification = intent == QueryIntent.FACTUAL and "verify" in query_lower
        
        # Recommend strategy
        strategy, reason = self._recommend_strategy(complexity, intent, requires_verification)
        
        # Estimate steps
        estimated_steps = self._estimate_steps(complexity, strategy)
        
        return QueryAnalysis(
            query=query,
            complexity=complexity,
            intent=intent,
            requires_multi_source=requires_multi_source,
            requires_reasoning=requires_reasoning,
            requires_verification=requires_verification,
            recommended_strategy=strategy,
            strategy_reason=reason,
            estimated_steps=estimated_steps,
            key_entities=entities,
            language=language,
            confidence=complexity_score,
        )
    
    def _detect_language(self, words: Set[str]) -> str:
        """Detect language from word set."""
        italian_count = len(words & self._italian_markers)
        italian_ratio = italian_count / max(len(words), 1)
        return "it" if italian_ratio > 0.15 else "en"
    
    def _detect_complexity(self, query: str, words: Set[str]) -> Tuple[QueryComplexity, float]:
        """Detect query complexity."""
        score = 0.0
        
        # Multi-hop indicators
        multi_hop_count = len(words & self._multi_hop_indicators)
        score += multi_hop_count * 0.2
        
        # Complex indicators
        complex_count = len(words & self._complex_indicators)
        score += complex_count * 0.15
        
        # Multiple questions
        question_count = query.count("?")
        if question_count > 1:
            score += (question_count - 1) * 0.25
        
        # Length-based
        word_count = len(words)
        score += min(word_count / 50, 0.3)
        
        # Determine level
        if score >= self.complexity_thresholds["complex"]:
            return QueryComplexity.MULTI_HOP, min(score, 1.0)
        elif score >= self.complexity_thresholds["moderate"]:
            return QueryComplexity.COMPLEX, score
        elif score >= self.complexity_thresholds["simple"]:
            return QueryComplexity.MODERATE, score
        else:
            return QueryComplexity.SIMPLE, 1.0 - score
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent."""
        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return intent
        return QueryIntent.FACTUAL
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities (basic implementation)."""
        # Simple: extract capitalized words that aren't at sentence start
        words = query.split()
        entities = []
        
        for i, word in enumerate(words):
            # Skip first word and common words
            if i == 0:
                continue
            # Check if capitalized
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                entities.append(clean_word)
        
        return list(set(entities))[:5]  # Max 5 entities
    
    def _recommend_strategy(
        self,
        complexity: QueryComplexity,
        intent: QueryIntent,
        requires_verification: bool,
    ) -> Tuple[ReasoningStrategy, str]:
        """Recommend best strategy."""
        if requires_verification:
            return ReasoningStrategy.VERIFICATION, "Query requires fact verification"
        
        if complexity == QueryComplexity.SIMPLE:
            if intent == QueryIntent.FACTUAL:
                return ReasoningStrategy.EVIDENCE_ATTRIBUTION, "Simple factual query benefits from evidence attribution"
            return ReasoningStrategy.DIRECT, "Simple query can be answered directly"
        
        if complexity == QueryComplexity.MULTI_HOP:
            return ReasoningStrategy.SELF_ASK, "Multi-hop query benefits from sub-question decomposition"
        
        if complexity in (QueryComplexity.COMPLEX, QueryComplexity.MODERATE):
            if intent in (QueryIntent.EXPLANATORY, QueryIntent.CAUSAL, QueryIntent.PROCEDURAL):
                return ReasoningStrategy.CHAIN_OF_THOUGHT, "Explanatory/causal query benefits from step-by-step reasoning"
            if intent == QueryIntent.COMPARATIVE:
                return ReasoningStrategy.SELF_ASK, "Comparative query benefits from aspect decomposition"
        
        return ReasoningStrategy.CHAIN_OF_THOUGHT, "Default to chain-of-thought for complex queries"
    
    def _estimate_steps(self, complexity: QueryComplexity, strategy: ReasoningStrategy) -> int:
        """Estimate number of reasoning steps needed."""
        base_steps = {
            QueryComplexity.SIMPLE: 1,
            QueryComplexity.MODERATE: 3,
            QueryComplexity.COMPLEX: 5,
            QueryComplexity.MULTI_HOP: 7,
        }
        
        strategy_multiplier = {
            ReasoningStrategy.DIRECT: 1,
            ReasoningStrategy.EVIDENCE_ATTRIBUTION: 2,
            ReasoningStrategy.CHAIN_OF_THOUGHT: 1,
            ReasoningStrategy.SELF_ASK: 1.5,
            ReasoningStrategy.VERIFICATION: 2,
        }
        
        base = base_steps.get(complexity, 3)
        multiplier = strategy_multiplier.get(strategy, 1)
        return int(base * multiplier)


# ============================================================================
# Redis Cache Provider
# ============================================================================


class ReasoningCacheProvider:
    """
    Environment-aware Redis caching for reasoning results.
    
    Features:
    - Environment isolation (dev/test/prod)
    - TTL management
    - Hit/miss statistics
    - Semantic similarity matching (optional)
    """
    
    def __init__(self, config: CacheConfig, redis_client: Optional[Any] = None):
        self.config = config
        self._redis = redis_client
        self._local_cache: Dict[str, Any] = {}
        self._stats = {"hits": 0, "misses": 0}
    
    async def get(self, operation: str, key_data: str) -> Optional[Dict[str, Any]]:
        """Get cached result."""
        if not self.config.enabled:
            return None
        
        cache_key = self._build_key(operation, key_data)
        
        # Try Redis first
        if self._redis:
            try:
                data = await self._redis.get(cache_key)
                if data:
                    self._stats["hits"] += 1
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Fall back to local cache
        if cache_key in self._local_cache:
            entry = self._local_cache[cache_key]
            if entry["expires_at"] > datetime.utcnow():
                self._stats["hits"] += 1
                return entry["data"]
            else:
                del self._local_cache[cache_key]
        
        self._stats["misses"] += 1
        return None
    
    async def set(
        self,
        operation: str,
        key_data: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """Set cached result."""
        if not self.config.enabled:
            return False
        
        cache_key = self._build_key(operation, key_data)
        ttl = ttl or self.config.ttl_seconds
        
        # Try Redis
        if self._redis:
            try:
                await self._redis.set(cache_key, json.dumps(value), ex=ttl)
                return True
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        # Fall back to local cache
        self._local_cache[cache_key] = {
            "data": value,
            "expires_at": datetime.utcnow() + timedelta(seconds=ttl),
        }
        return True
    
    async def invalidate(self, operation: str, key_data: str) -> bool:
        """Invalidate a cache entry."""
        cache_key = self._build_key(operation, key_data)
        
        if self._redis:
            try:
                await self._redis.delete(cache_key)
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        
        if cache_key in self._local_cache:
            del self._local_cache[cache_key]
        
        return True
    
    async def clear(self) -> int:
        """Clear all cached entries."""
        count = len(self._local_cache)
        self._local_cache.clear()
        
        if self._redis:
            try:
                pattern = f"{self.config.prefix}:*"
                keys = await self._redis.keys(pattern)
                if keys:
                    await self._redis.delete(*keys)
                    count = len(keys)
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        
        return count
    
    def _build_key(self, operation: str, key_data: str) -> str:
        """Build cache key with environment isolation."""
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"{self.config.prefix}:{operation}:{key_hash}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        
        return {
            "enabled": self.config.enabled,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": round(hit_rate, 3),
            "local_entries": len(self._local_cache),
        }


# ============================================================================
# Session Manager
# ============================================================================


class ReasoningSessionManager:
    """
    Manages reasoning sessions with history tracking.
    
    Features:
    - UUID-based sessions
    - History tracking
    - Reasoning trace storage
    - TTL-based expiration
    - Auto-cleanup
    """
    
    def __init__(self, config: SessionConfig):
        self.config = config
        self._sessions: Dict[str, ReasoningSession] = {}
        self._last_cleanup = datetime.utcnow()
    
    async def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> ReasoningSession:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session = ReasoningSession(
            session_id=session_id,
            metadata=metadata or {},
        )
        self._sessions[session_id] = session
        
        await self._maybe_cleanup()
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[ReasoningSession]:
        """Get session by ID."""
        session = self._sessions.get(session_id)
        if session:
            elapsed = (datetime.utcnow() - session.updated_at).total_seconds()
            if elapsed > self.config.ttl_seconds:
                del self._sessions[session_id]
                return None
        return session
    
    async def update_session(
        self,
        session_id: str,
        query: str,
        result: ReasoningResult,
    ) -> Optional[ReasoningSession]:
        """Update session with new interaction."""
        session = await self.get_session(session_id)
        if not session:
            return None
        
        # Add to history
        session.history.append({
            "query": query,
            "strategy": result.strategy_used.value,
            "confidence": result.confidence,
            "answer_preview": result.answer[:200] + "..." if len(result.answer) > 200 else result.answer,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Trim history if needed
        if len(session.history) > self.config.max_history_size:
            session.history = session.history[-self.config.max_history_size:]
        
        # Store trace if enabled
        if self.config.persist_reasoning_trace:
            session.traces.append(result.reasoning_trace)
            if len(session.traces) > self.config.max_history_size:
                session.traces = session.traces[-self.config.max_history_size:]
        
        session.updated_at = datetime.utcnow()
        
        return session
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    async def _maybe_cleanup(self) -> None:
        """Run cleanup if needed."""
        if not self.config.auto_cleanup:
            return
        
        now = datetime.utcnow()
        if (now - self._last_cleanup).total_seconds() < 300:
            return
        
        self._last_cleanup = now
        expired = []
        
        for session_id, session in self._sessions.items():
            elapsed = (now - session.updated_at).total_seconds()
            if elapsed > self.config.ttl_seconds:
                expired.append(session_id)
        
        for session_id in expired:
            del self._sessions[session_id]
        
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired reasoning sessions")


# ============================================================================
# Worker Pool
# ============================================================================


class ReasoningWorkerPool:
    """
    Async worker pool for parallel reasoning tasks.
    
    Features:
    - Configurable pool size
    - Priority queue
    - Timeout handling
    - Retry with backoff
    - Statistics tracking
    """
    
    def __init__(self, config: WorkerPoolConfig):
        self.config = config
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=config.queue_max_size)
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._stats = {
            "completed": 0,
            "failed": 0,
            "total_time_ms": 0.0,
        }
    
    async def start(self) -> None:
        """Start worker pool."""
        if self._running:
            return
        
        self._running = True
        for i in range(self.config.pool_size):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)
        
        logger.info(f"Reasoning worker pool started with {self.config.pool_size} workers")
    
    async def stop(self) -> None:
        """Stop worker pool."""
        self._running = False
        
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("Reasoning worker pool stopped")
    
    async def submit(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs,
    ) -> WorkerTask:
        """Submit a task to the pool."""
        task = WorkerTask(
            task_id=str(uuid.uuid4()),
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
        )
        
        await self._queue.put((-priority.value, task))
        return task
    
    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop processing tasks from queue."""
        while self._running:
            try:
                try:
                    _, task = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue
                
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                start_time = time.perf_counter()
                
                try:
                    result = await asyncio.wait_for(
                        task.func(*task.args, **task.kwargs),
                        timeout=self.config.task_timeout_seconds,
                    )
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    self._stats["completed"] += 1
                    
                except asyncio.TimeoutError:
                    task.error = "Task timeout"
                    task.status = TaskStatus.FAILED
                    self._stats["failed"] += 1
                    
                except Exception as e:
                    task.error = str(e)
                    
                    if self.config.retry_on_failure and task.retries < self.config.max_task_retries:
                        task.retries += 1
                        task.status = TaskStatus.PENDING
                        delay = self.config.backoff_multiplier ** task.retries
                        await asyncio.sleep(delay)
                        await self._queue.put((-task.priority.value, task))
                    else:
                        task.status = TaskStatus.FAILED
                        self._stats["failed"] += 1
                
                task.completed_at = datetime.utcnow()
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._stats["total_time_ms"] += elapsed_ms
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    def get_stats(self) -> WorkerStats:
        """Get pool statistics."""
        total_tasks = self._stats["completed"] + self._stats["failed"]
        avg_time = self._stats["total_time_ms"] / total_tasks if total_tasks > 0 else 0
        
        return WorkerStats(
            active_workers=len([w for w in self._workers if not w.done()]),
            pending_tasks=self._queue.qsize(),
            completed_tasks=self._stats["completed"],
            failed_tasks=self._stats["failed"],
            avg_task_time_ms=avg_time,
        )


# ============================================================================
# Metrics Collector
# ============================================================================


class ReasoningMetricsCollector:
    """
    Comprehensive metrics collection for reasoning operations.
    
    Tracks:
    - Strategy distribution
    - Iteration counts
    - Retrieval statistics
    - Confidence scores
    - Verification statistics
    - Execution times
    """
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self._metrics = {
            "total_queries": 0,
            "strategy_distribution": {},
            "iteration_counts": [],
            "step_counts": [],
            "retrieval_counts": [],
            "confidence_scores": [],
            "verification_counts": {"verified": 0, "partially": 0, "unverified": 0, "contradicted": 0},
            "execution_times_ms": [],
        }
    
    async def record_reasoning(
        self,
        strategy: ReasoningStrategy,
        iterations: int,
        steps: int,
        retrievals: int,
        confidence: float,
        verifications: Dict[str, int],
        execution_time_ms: float,
    ) -> None:
        """Record a reasoning event."""
        if not self.config.enabled:
            return
        
        self._metrics["total_queries"] += 1
        
        if self.config.collect_strategy_distribution:
            strategy_name = strategy.value
            self._metrics["strategy_distribution"][strategy_name] = \
                self._metrics["strategy_distribution"].get(strategy_name, 0) + 1
        
        if self.config.collect_iteration_counts:
            self._metrics["iteration_counts"].append(iterations)
            self._metrics["step_counts"].append(steps)
            if len(self._metrics["iteration_counts"]) > 1000:
                self._metrics["iteration_counts"] = self._metrics["iteration_counts"][-1000:]
                self._metrics["step_counts"] = self._metrics["step_counts"][-1000:]
        
        if self.config.collect_retrieval_stats:
            self._metrics["retrieval_counts"].append(retrievals)
            if len(self._metrics["retrieval_counts"]) > 1000:
                self._metrics["retrieval_counts"] = self._metrics["retrieval_counts"][-1000:]
        
        if self.config.collect_confidence_scores:
            self._metrics["confidence_scores"].append(confidence)
            if len(self._metrics["confidence_scores"]) > 1000:
                self._metrics["confidence_scores"] = self._metrics["confidence_scores"][-1000:]
        
        if self.config.collect_verification_stats:
            for status, count in verifications.items():
                if status in self._metrics["verification_counts"]:
                    self._metrics["verification_counts"][status] += count
        
        if self.config.collect_timings:
            self._metrics["execution_times_ms"].append(execution_time_ms)
            if len(self._metrics["execution_times_ms"]) > 1000:
                self._metrics["execution_times_ms"] = self._metrics["execution_times_ms"][-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        iterations = self._metrics["iteration_counts"]
        steps = self._metrics["step_counts"]
        retrievals = self._metrics["retrieval_counts"]
        confidence = self._metrics["confidence_scores"]
        times = self._metrics["execution_times_ms"]
        
        return {
            "total_queries": self._metrics["total_queries"],
            "strategy_distribution": self._metrics["strategy_distribution"],
            "iterations": {
                "avg": sum(iterations) / len(iterations) if iterations else 0,
                "max": max(iterations) if iterations else 0,
            },
            "steps": {
                "avg": sum(steps) / len(steps) if steps else 0,
                "max": max(steps) if steps else 0,
            },
            "retrievals": {
                "avg": sum(retrievals) / len(retrievals) if retrievals else 0,
                "total": sum(retrievals),
            },
            "confidence": {
                "avg": sum(confidence) / len(confidence) if confidence else 0,
                "min": min(confidence) if confidence else 0,
                "max": max(confidence) if confidence else 0,
            },
            "verification": self._metrics["verification_counts"],
            "execution_times": {
                "avg_ms": sum(times) / len(times) if times else 0,
                "min_ms": min(times) if times else 0,
                "max_ms": max(times) if times else 0,
            },
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics = {
            "total_queries": 0,
            "strategy_distribution": {},
            "iteration_counts": [],
            "step_counts": [],
            "retrieval_counts": [],
            "confidence_scores": [],
            "verification_counts": {"verified": 0, "partially": 0, "unverified": 0, "contradicted": 0},
            "execution_times_ms": [],
        }
