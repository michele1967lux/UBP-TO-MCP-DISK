"""
hyde_pipeline/providers.py

Logic Layer - ZERO dependencies from backend.app
Must be testable standalone.

Provides:
- HyDEDocument: Core data structure for generated documents
- DomainClassifier: Query domain and language detection
- QualityAssuranceProvider: Multi-dimensional document scoring
- HallucinationDetector: Detect potentially fabricated content
- DocumentChunker: Semantic chunking for optimal embedding
- EnsembleFusion: Combine multiple documents intelligently
- DocumentRefiner: Iterative document improvement
- RedisCacheProvider: Environment-aware caching
- HyDESessionManager: Session management with history
- HyDEWorkerPool: Parallel async task execution
- MetricsCollector: Comprehensive statistics

v1.0.0: Initial release with innovative features
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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# v6.4.0: Import unified chunker from rag_qdrant
try:
    from rag_qdrant.chunker import ChunkingManager as _MainChunkingManager
    from rag_qdrant.chunker import ChunkingConfig as _MainChunkingConfig
    _HAS_MAIN_CHUNKER = True
except ImportError:
    _HAS_MAIN_CHUNKER = False


# ============================================================================
# Enums
# ============================================================================


class DocumentFormat(Enum):
    """Available HyDE document formats."""
    ANSWER = "answer"
    TECHNICAL_DOC = "technical_doc"
    FAQ = "faq"
    CODE_SNIPPET = "code_snippet"
    TUTORIAL = "tutorial"
    TROUBLESHOOTING = "troubleshooting"
    ARTICLE = "article"


class Domain(Enum):
    """Query domains for context adaptation."""
    AI_ML = "ai_ml"
    DEVOPS = "devops"
    API_INTEGRATION = "api_integration"
    DATABASE = "database"
    SECURITY = "security"
    CLOUD = "cloud"
    GENERAL = "general"


class QualityLevel(Enum):
    """Document quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


class RefinementStrategy(Enum):
    """Document refinement strategies."""
    EXPAND = "expand"
    FOCUS = "focus"
    TECHNICAL = "technical"
    SIMPLIFY = "simplify"


class ChunkingStrategy(Enum):
    """Document chunking strategies."""
    SEMANTIC = "semantic"
    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


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
class HyDEConfig:
    """Core HyDE configuration."""
    enabled: bool = True
    default_format: str = "answer"
    default_domain: str = "auto"
    default_length: int = 300
    min_length: int = 100
    max_length: int = 1000
    temperature: float = 0.5
    max_tokens: int = 600
    timeout_seconds: int = 30
    retry_enabled: bool = True
    max_retries: int = 2
    retry_delay_seconds: float = 1.0


@dataclass
class QualityAssuranceConfig:
    """QA scoring configuration."""
    enabled: bool = True
    weight_relevance: float = 0.35
    weight_coherence: float = 0.25
    weight_informativeness: float = 0.20
    weight_format_adherence: float = 0.10
    weight_terminology: float = 0.10
    threshold_excellent: float = 8.0
    threshold_good: float = 6.0
    threshold_acceptable: float = 4.0
    min_acceptable_score: float = 4.0


@dataclass
class EnsembleConfig:
    """Ensemble generation configuration."""
    enabled: bool = True
    default_count: int = 3
    max_count: int = 5
    fusion_strategy: str = "weighted_concat"
    diversity_penalty: float = 0.1
    parallel_generation: bool = True
    temperature_spread: float = 0.2
    format_diversity: bool = True


@dataclass
class RefinementConfig:
    """Document refinement configuration."""
    enabled: bool = True
    max_iterations: int = 2
    quality_threshold: float = 6.0
    improvement_min: float = 0.5
    strategies_enabled: Dict[str, bool] = field(default_factory=lambda: {
        "expand": True,
        "focus": True,
        "technical": True,
        "simplify": True,
    })


@dataclass
class HallucinationConfig:
    """Hallucination detection configuration."""
    enabled: bool = True
    check_unknown_terms: bool = True
    check_invented_apis: bool = True
    check_fake_versions: bool = True
    confidence_penalty: float = 0.2
    max_unknown_ratio: float = 0.15


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    enabled: bool = True
    strategy: str = "semantic"
    chunk_size: int = 256
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    preserve_sentences: bool = True
    preserve_code_blocks: bool = True


@dataclass
class CacheConfig:
    """Redis cache configuration."""
    enabled: bool = True
    ttl_seconds: int = 3600
    base_prefix: str = "ubp"
    env: str = "dev"
    cache_documents: bool = True
    cache_chunks: bool = True
    cache_qa_results: bool = True
    semantic_matching: bool = True
    semantic_threshold: float = 0.92

    @property
    def prefix(self) -> str:
        """Generate environment-isolated Redis key prefix."""
        return f"{self.base_prefix}:{self.env}:hyde:cache"


@dataclass
class SessionConfig:
    """Session management configuration."""
    enabled: bool = True
    ttl_seconds: int = 3600
    max_history_size: int = 30
    persist_documents: bool = True
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
    enable_priorities: bool = True


@dataclass
class MetricsConfig:
    """Metrics collection configuration."""
    enabled: bool = True
    collect_timings: bool = True
    collect_format_distribution: bool = True
    collect_domain_distribution: bool = True
    collect_qa_scores: bool = True
    collect_hallucination_rates: bool = True
    collect_refinement_stats: bool = True
    retention_hours: int = 24


@dataclass
class DebugConfig:
    """Debug configuration."""
    enabled: bool = False
    log_prompts: bool = False
    log_responses: bool = False
    log_qa_scores: bool = True
    log_hallucination_checks: bool = True
    log_refinement_steps: bool = True
    log_ensemble_details: bool = True
    log_chunking: bool = False
    trace_execution: bool = False


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class HyDEDocument:
    """Core HyDE document representation."""
    document_id: str
    content: str
    query: str
    format_type: str
    domain: str
    language: str
    quality_score: float = 0.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.document_id:
            self.document_id = str(uuid.uuid4())
    
    @property
    def length(self) -> int:
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        return len(self.content.split())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "content": self.content,
            "query": self.query,
            "format_type": self.format_type,
            "domain": self.domain,
            "language": self.language,
            "quality_score": self.quality_score,
            "confidence": self.confidence,
            "length": self.length,
            "word_count": self.word_count,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyDEDocument":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()
        
        return cls(
            document_id=data.get("document_id", ""),
            content=data.get("content", ""),
            query=data.get("query", ""),
            format_type=data.get("format_type", "answer"),
            domain=data.get("domain", "general"),
            language=data.get("language", "en"),
            quality_score=float(data.get("quality_score", 0.0)),
            confidence=float(data.get("confidence", 1.0)),
            created_at=created_at,
            metadata=data.get("metadata", {}),
        )


@dataclass
class DocumentChunk:
    """A chunk of a HyDE document for embedding."""
    chunk_id: str
    document_id: str
    content: str
    position: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> int:
        return len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "position": self.position,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "length": self.length,
            "metadata": self.metadata,
        }


@dataclass
class QualityAssessment:
    """Quality assessment result for a document."""
    document_id: str
    overall_score: float
    relevance_score: float
    coherence_score: float
    informativeness_score: float
    format_score: float
    terminology_score: float
    quality_level: QualityLevel
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "overall_score": round(self.overall_score, 2),
            "relevance_score": round(self.relevance_score, 2),
            "coherence_score": round(self.coherence_score, 2),
            "informativeness_score": round(self.informativeness_score, 2),
            "format_score": round(self.format_score, 2),
            "terminology_score": round(self.terminology_score, 2),
            "quality_level": self.quality_level.value,
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


@dataclass
class HallucinationCheck:
    """Result of hallucination detection."""
    document_id: str
    hallucination_detected: bool
    confidence: float
    suspicious_elements: List[Dict[str, Any]] = field(default_factory=list)
    recommendation: str = "accept"  # accept, review, reject
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "hallucination_detected": self.hallucination_detected,
            "confidence": round(self.confidence, 3),
            "suspicious_elements": self.suspicious_elements,
            "recommendation": self.recommendation,
            "suspicious_count": len(self.suspicious_elements),
        }


@dataclass
class DomainClassification:
    """Query domain classification result."""
    query: str
    domain: str
    confidence: float
    language: str
    preferred_formats: List[str] = field(default_factory=list)
    detected_keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "domain": self.domain,
            "confidence": round(self.confidence, 3),
            "language": self.language,
            "preferred_formats": self.preferred_formats,
            "detected_keywords": self.detected_keywords,
        }


@dataclass
class EnsembleResult:
    """Result of ensemble document generation."""
    documents: List[HyDEDocument]
    fused_document: Optional[HyDEDocument]
    strategy: str
    diversity_score: float
    generation_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": [d.to_dict() for d in self.documents],
            "fused_document": self.fused_document.to_dict() if self.fused_document else None,
            "strategy": self.strategy,
            "diversity_score": round(self.diversity_score, 3),
            "generation_time_ms": round(self.generation_time_ms, 2),
            "document_count": len(self.documents),
        }


@dataclass
class RefinementResult:
    """Result of document refinement."""
    original_document: HyDEDocument
    refined_document: HyDEDocument
    iterations: int
    score_improvement: float
    strategies_applied: List[str]
    refinement_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_document": self.original_document.to_dict(),
            "refined_document": self.refined_document.to_dict(),
            "iterations": self.iterations,
            "score_improvement": round(self.score_improvement, 2),
            "strategies_applied": self.strategies_applied,
            "refinement_time_ms": round(self.refinement_time_ms, 2),
        }


@dataclass
class HyDESession:
    """Session state for HyDE interactions."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    history: List[Dict[str, Any]] = field(default_factory=list)
    documents: List[HyDEDocument] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "history_count": len(self.history),
            "document_count": len(self.documents),
            "history": self.history[-10:],  # Last 10 entries
            "metadata": self.metadata,
        }


@dataclass
class HyDEResult:
    """Complete HyDE pipeline result."""
    session_id: str
    query: str
    document: HyDEDocument
    chunks: List[DocumentChunk]
    classification: DomainClassification
    quality_assessment: Optional[QualityAssessment]
    hallucination_check: Optional[HallucinationCheck]
    refinement_applied: bool
    time_ms: float
    step_times: Dict[str, float] = field(default_factory=dict)
    step_errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "query": self.query,
            "document": self.document.to_dict(),
            "chunks": [c.to_dict() for c in self.chunks],
            "classification": self.classification.to_dict(),
            "quality_assessment": self.quality_assessment.to_dict() if self.quality_assessment else None,
            "hallucination_check": self.hallucination_check.to_dict() if self.hallucination_check else None,
            "refinement_applied": self.refinement_applied,
            "time_ms": round(self.time_ms, 2),
            "step_times": {k: round(v, 2) for k, v in self.step_times.items()},
            "chunk_count": len(self.chunks),
        }
        # Include step_errors only when non-empty for cleaner responses
        if self.step_errors:
            result["step_errors"] = self.step_errors
        return result


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
# Domain Classifier
# ============================================================================


class DomainClassifier:
    """
    Classifies queries by domain and detects language.
    
    Uses keyword matching with weighted scoring and
    language detection based on common word patterns.
    """
    
    def __init__(
        self,
        domains_config: Dict[str, Dict[str, Any]],
        default_domain: str = "general",
        min_confidence: float = 0.3,
    ):
        self.domains_config = domains_config
        self.default_domain = default_domain
        self.min_confidence = min_confidence
        
        # Italian markers for language detection
        self._italian_markers = {
            "come", "cosa", "perché", "quando", "dove", "chi", "quale",
            "il", "la", "lo", "gli", "le", "un", "una", "uno",
            "è", "sono", "sei", "siamo", "essere", "avere", "fare",
            "non", "che", "per", "con", "su", "da", "in", "di",
        }
    
    def classify(self, query: str) -> DomainClassification:
        """
        Classify a query by domain and language.
        
        Args:
            query: User's query
            
        Returns:
            DomainClassification with domain, confidence, and language
        """
        query_lower = query.lower()
        
        # Detect language
        language = self._detect_language(query_lower)
        
        # Detect domain
        domain, confidence, keywords = self._detect_domain(query_lower)
        
        # Get preferred formats for domain
        domain_info = self.domains_config.get(domain, {})
        preferred_formats = domain_info.get("preferred_formats", ["answer"])
        
        return DomainClassification(
            query=query,
            domain=domain,
            confidence=confidence,
            language=language,
            preferred_formats=preferred_formats,
            detected_keywords=keywords,
        )
    
    def _detect_language(self, text: str) -> str:
        """Detect language based on common word patterns."""
        words = set(text.split())
        italian_count = len(words & self._italian_markers)
        italian_ratio = italian_count / max(len(words), 1)
        return "it" if italian_ratio > 0.15 else "en"
    
    def _detect_domain(self, query: str) -> Tuple[str, float, List[str]]:
        """Detect domain using keyword matching."""
        best_domain = self.default_domain
        best_score = 0.0
        best_keywords: List[str] = []
        
        for domain_name, domain_info in self.domains_config.items():
            if domain_name == "general":
                continue
            
            keywords = domain_info.get("keywords", [])
            if not keywords:
                continue
            
            # Find matching keywords
            matched = [kw for kw in keywords if kw.lower() in query]
            if not matched:
                continue
            
            # Calculate weighted score
            weight = float(domain_info.get("weight", 1.0))
            score = (len(matched) / len(keywords)) * weight
            
            if score > best_score:
                best_score = score
                best_domain = domain_name
                best_keywords = matched
        
        # Apply minimum confidence threshold
        confidence = min(best_score * 2, 1.0)
        if confidence < self.min_confidence:
            best_domain = self.default_domain
            confidence = 1.0 - confidence  # Invert for "general" confidence
            best_keywords = []
        
        return best_domain, confidence, best_keywords


# ============================================================================
# Quality Assurance Provider
# ============================================================================


class QualityAssuranceProvider:
    """
    Multi-dimensional quality scoring for HyDE documents.
    
    Evaluates:
    - Relevance: Keyword overlap with query
    - Coherence: Sentence structure and flow
    - Informativeness: Information density
    - Format Adherence: Matches expected format
    - Terminology: Technical term usage
    """
    
    def __init__(self, config: QualityAssuranceConfig):
        self.config = config
    
    def assess(
        self,
        document: HyDEDocument,
        original_query: str,
    ) -> QualityAssessment:
        """
        Assess document quality across multiple dimensions.
        
        Args:
            document: HyDE document to assess
            original_query: Original user query
            
        Returns:
            QualityAssessment with scores and feedback
        """
        # Calculate individual scores
        relevance = self._score_relevance(document.content, original_query)
        coherence = self._score_coherence(document.content)
        informativeness = self._score_informativeness(document.content)
        format_score = self._score_format(document.content, document.format_type)
        terminology = self._score_terminology(document.content, document.domain)
        
        # Calculate weighted overall score
        overall = (
            relevance * self.config.weight_relevance +
            coherence * self.config.weight_coherence +
            informativeness * self.config.weight_informativeness +
            format_score * self.config.weight_format_adherence +
            terminology * self.config.weight_terminology
        )
        
        # Determine quality level
        quality_level = self._determine_level(overall)
        
        # Generate feedback
        issues, suggestions = self._generate_feedback(
            relevance, coherence, informativeness, format_score, terminology
        )
        
        return QualityAssessment(
            document_id=document.document_id,
            overall_score=overall,
            relevance_score=relevance,
            coherence_score=coherence,
            informativeness_score=informativeness,
            format_score=format_score,
            terminology_score=terminology,
            quality_level=quality_level,
            issues=issues,
            suggestions=suggestions,
        )
    
    def _score_relevance(self, content: str, query: str) -> float:
        """Score based on keyword overlap with query."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "to", "of", "and", "in", "that", "it", "for", "on", "with"}
        query_words -= stop_words
        
        if not query_words:
            return 5.0
        
        overlap = len(query_words & content_words) / len(query_words)
        return min(overlap * 12, 10.0)  # Scale to 0-10
    
    def _score_coherence(self, content: str) -> float:
        """Score based on sentence structure and flow."""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 3.0
        
        # Check sentence length variance (lower is better for coherence)
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # Score based on reasonable sentence lengths and consistency
        length_score = 1.0 if 10 <= avg_length <= 25 else 0.5
        variance_score = 1.0 if variance < 100 else 0.5
        
        # Check for transition words
        transition_words = ["however", "therefore", "moreover", "additionally",
                          "furthermore", "consequently", "thus", "hence"]
        content_lower = content.lower()
        transition_count = sum(1 for w in transition_words if w in content_lower)
        transition_score = min(transition_count / 3, 1.0)
        
        base_score = 5.0 + (length_score + variance_score + transition_score) * 1.5
        return min(base_score, 10.0)
    
    def _score_informativeness(self, content: str) -> float:
        """Score based on information density."""
        words = content.split()
        unique_words = set(w.lower() for w in words)
        
        if not words:
            return 3.0
        
        # Lexical diversity
        diversity = len(unique_words) / len(words)
        
        # Information indicators
        has_numbers = bool(re.search(r'\d+', content))
        has_lists = ":" in content or "-" in content
        has_examples = any(w in content.lower() for w in ["example", "such as", "for instance"])
        has_specifics = bool(re.search(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', content))  # CamelCase
        
        base_score = diversity * 8
        bonus = sum([has_numbers, has_lists, has_examples, has_specifics]) * 0.5
        
        return min(base_score + bonus, 10.0)
    
    def _score_format(self, content: str, format_type: str) -> float:
        """Score based on format adherence."""
        content_lower = content.lower()
        
        format_indicators = {
            "answer": lambda c: len(c) > 100 and not c.startswith("Step"),
            "technical_doc": lambda c: len(c) > 200 and (":" in c or "." in c),
            "faq": lambda c: "?" in c or "answer" in c.lower(),
            "code_snippet": lambda c: "```" in c or "def " in c or "function" in c or "import" in c,
            "tutorial": lambda c: any(x in c.lower() for x in ["step", "first", "then", "next"]),
            "troubleshooting": lambda c: any(x in c.lower() for x in ["issue", "problem", "solution", "fix", "error"]),
            "article": lambda c: len(c) > 300,
        }
        
        checker = format_indicators.get(format_type, lambda c: True)
        matches_format = checker(content)
        
        # Base score
        base = 7.0 if matches_format else 4.0
        
        # Length appropriateness
        length = len(content)
        length_bonus = 0
        if format_type in ["technical_doc", "tutorial", "article"]:
            length_bonus = 1.5 if length > 400 else 0
        elif format_type in ["answer", "faq"]:
            length_bonus = 1.5 if 100 < length < 500 else 0
        elif format_type == "code_snippet":
            length_bonus = 1.5 if 150 < length < 600 else 0
        
        return min(base + length_bonus, 10.0)
    
    def _score_terminology(self, content: str, domain: str) -> float:
        """Score based on domain-appropriate terminology."""
        # Domain-specific technical terms
        domain_terms = {
            "ai_ml": ["model", "training", "inference", "embedding", "neural", "layer", "parameter", "epoch"],
            "devops": ["container", "deploy", "pipeline", "cluster", "node", "pod", "image", "config"],
            "api_integration": ["endpoint", "request", "response", "authentication", "token", "header", "api"],
            "database": ["query", "index", "table", "schema", "transaction", "join", "select", "insert"],
            "security": ["authentication", "encryption", "certificate", "token", "permission", "access", "secure"],
            "cloud": ["service", "instance", "region", "bucket", "function", "serverless", "scale"],
            "general": [],
        }
        
        terms = domain_terms.get(domain, [])
        if not terms:
            return 7.0  # Neutral for general domain
        
        content_lower = content.lower()
        matches = sum(1 for t in terms if t in content_lower)
        ratio = matches / len(terms)
        
        return min(5.0 + ratio * 7, 10.0)
    
    def _determine_level(self, score: float) -> QualityLevel:
        """Determine quality level from score."""
        if score >= self.config.threshold_excellent:
            return QualityLevel.EXCELLENT
        elif score >= self.config.threshold_good:
            return QualityLevel.GOOD
        elif score >= self.config.threshold_acceptable:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.POOR
    
    def _generate_feedback(
        self,
        relevance: float,
        coherence: float,
        informativeness: float,
        format_score: float,
        terminology: float,
    ) -> Tuple[List[str], List[str]]:
        """Generate issues and suggestions based on scores."""
        issues = []
        suggestions = []
        
        if relevance < 5.0:
            issues.append("Low relevance to query")
            suggestions.append("Include more keywords from the original query")
        
        if coherence < 5.0:
            issues.append("Poor sentence structure or flow")
            suggestions.append("Use transition words and vary sentence length")
        
        if informativeness < 5.0:
            issues.append("Low information density")
            suggestions.append("Add specific examples, numbers, or technical details")
        
        if format_score < 5.0:
            issues.append("Format doesn't match expected style")
            suggestions.append("Adjust structure to match the target format")
        
        if terminology < 5.0:
            issues.append("Insufficient domain terminology")
            suggestions.append("Include more domain-specific technical terms")
        
        return issues, suggestions


# ============================================================================
# Hallucination Detector
# ============================================================================


class HallucinationDetector:
    """
    Detects potentially fabricated content in HyDE documents.
    
    Checks for:
    - Invented API endpoints
    - Fake version numbers
    - Non-existent configuration options
    - Made-up technical terms
    """
    
    def __init__(self, config: HallucinationConfig):
        self.config = config
        
        # Known valid patterns
        self._valid_api_patterns = [
            r'/api/v\d+/', r'/v\d+/', r'api\.', r'/rest/',
            r'GET|POST|PUT|DELETE|PATCH',
        ]
        
        # Suspicious patterns
        self._suspicious_version_pattern = re.compile(
            r'(?:version|v|ver\.?)\s*(\d+\.\d+(?:\.\d+)?(?:\.\d+)?)',
            re.IGNORECASE
        )
        
        # Common valid version numbers (major.minor)
        self._known_versions = {
            "3.8", "3.9", "3.10", "3.11", "3.12",  # Python
            "18", "20", "22",  # Node.js LTS
            "1.0", "2.0", "3.0", "4.0", "5.0",  # Common major versions
        }
    
    def check(self, document: HyDEDocument) -> HallucinationCheck:
        """
        Check document for potential hallucinations.
        
        Args:
            document: HyDE document to check
            
        Returns:
            HallucinationCheck with detection results
        """
        suspicious_elements = []
        
        if self.config.check_invented_apis:
            api_issues = self._check_invented_apis(document.content)
            suspicious_elements.extend(api_issues)
        
        if self.config.check_fake_versions:
            version_issues = self._check_fake_versions(document.content)
            suspicious_elements.extend(version_issues)
        
        if self.config.check_unknown_terms:
            term_issues = self._check_unknown_terms(document.content, document.domain)
            suspicious_elements.extend(term_issues)
        
        # Calculate confidence
        high_severity = sum(1 for e in suspicious_elements if e.get("severity") == "high")
        medium_severity = sum(1 for e in suspicious_elements if e.get("severity") == "medium")
        
        # Confidence in the document (lower if more issues)
        confidence = 1.0 - (high_severity * 0.2 + medium_severity * 0.1)
        confidence = max(confidence, 0.0)
        
        # Determine recommendation
        hallucination_detected = len(suspicious_elements) > 0
        if high_severity >= 2 or (high_severity >= 1 and medium_severity >= 2):
            recommendation = "reject"
        elif high_severity >= 1 or medium_severity >= 2:
            recommendation = "review"
        else:
            recommendation = "accept"
        
        return HallucinationCheck(
            document_id=document.document_id,
            hallucination_detected=hallucination_detected,
            confidence=confidence,
            suspicious_elements=suspicious_elements,
            recommendation=recommendation,
        )
    
    def _check_invented_apis(self, content: str) -> List[Dict[str, Any]]:
        """Check for potentially invented API endpoints."""
        issues = []
        
        # Look for API-like patterns
        api_pattern = re.compile(r'(?:api|endpoint|url).*?["\']([^"\']+)["\']', re.IGNORECASE)
        matches = api_pattern.findall(content)
        
        for match in matches:
            # Check if it looks suspicious (very specific but unusual)
            if len(match) > 50 or match.count('/') > 6:
                issues.append({
                    "text": match[:100],
                    "reason": "Unusually complex API endpoint",
                    "severity": "medium",
                })
        
        return issues
    
    def _check_fake_versions(self, content: str) -> List[Dict[str, Any]]:
        """Check for potentially fake version numbers."""
        issues = []
        
        matches = self._suspicious_version_pattern.findall(content)
        for version in matches:
            # Check for unusually high major versions
            try:
                major = int(version.split('.')[0])
                if major > 50:  # Suspiciously high
                    issues.append({
                        "text": f"version {version}",
                        "reason": f"Unusually high version number ({major})",
                        "severity": "high",
                    })
            except (ValueError, IndexError):
                pass
        
        return issues
    
    def _check_unknown_terms(self, content: str, domain: str) -> List[Dict[str, Any]]:
        """Check for potentially made-up technical terms."""
        issues = []
        
        # Look for CamelCase terms that might be invented
        camel_pattern = re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+){2,})\b')
        matches = camel_pattern.findall(content)
        
        # Known valid terms (simplified)
        known_terms = {
            "JavaScript", "TypeScript", "PostgreSQL", "MongoDB", "GraphQL",
            "TensorFlow", "PyTorch", "FastAPI", "NextJS", "ReactJS",
            "Kubernetes", "DockerFile", "CloudFormation", "OpenAPI",
        }
        
        for term in matches:
            if term not in known_terms and len(term) > 20:
                issues.append({
                    "text": term,
                    "reason": "Potentially invented technical term",
                    "severity": "low",
                })
        
        return issues


# ============================================================================
# Document Chunker
# ============================================================================


class DocumentChunker:
    """
    Semantic chunking for optimal embedding of HyDE documents.
    
    Strategies:
    - semantic: Split on natural boundaries (paragraphs, sections)
    - sentence: Split on sentence boundaries
    - fixed: Fixed-size chunks with overlap
    - paragraph: Split on paragraph boundaries
    """
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
    
    def chunk(self, document: HyDEDocument) -> List[DocumentChunk]:
        """
        Chunk a document using the configured strategy.

        Args:
            document: HyDE document to chunk

        Returns:
            List of DocumentChunk objects
        """
        # v6.4.0: Delegate to unified chunker from rag_qdrant
        if _HAS_MAIN_CHUNKER:
            _strategy_map = {"semantic": "sentence", "sentence": "sentence",
                             "paragraph": "paragraph", "fixed": "fixed"}
            _strategy = _strategy_map.get(self.config.strategy, "sentence")
            config = _MainChunkingConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                strategy=_strategy,
                min_chunk_size=self.config.min_chunk_size,
            )
            manager = _MainChunkingManager(config)
            raw_chunks = manager.chunk(document.content)
            return [
                DocumentChunk(
                    chunk_id=f"{document.document_id}-{i}",
                    document_id=document.document_id,
                    content=c.text,
                    position=i,
                    start_char=c.start_char,
                    end_char=c.end_char,
                    metadata={**c.metadata, "strategy": _strategy},
                )
                for i, c in enumerate(raw_chunks)
            ] or self._validate_chunks([], document)

        return self._legacy_chunk(document)

    def _legacy_chunk(self, document: HyDEDocument) -> List[DocumentChunk]:
        """Fallback chunker when rag_qdrant is not available."""
        strategy = ChunkingStrategy(self.config.strategy)

        if strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(document)
        elif strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_sentence(document)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_paragraph(document)
        else:  # FIXED
            return self._chunk_fixed(document)
    
    def _chunk_semantic(self, document: HyDEDocument) -> List[DocumentChunk]:
        """Chunk on natural semantic boundaries."""
        content = document.content
        chunks = []
        
        # Try paragraph splitting first
        paragraphs = re.split(r'\n\n+', content)
        
        if len(paragraphs) <= 1:
            # Fall back to sentence splitting
            return self._chunk_sentence(document)
        
        current_chunk = ""
        current_start = 0
        position = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.config.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(DocumentChunk(
                    chunk_id=f"{document.document_id}-{position}",
                    document_id=document.document_id,
                    content=current_chunk.strip(),
                    position=position,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={"strategy": "semantic"},
                ))
                position += 1
                current_start += len(current_chunk)
                current_chunk = para + "\n\n"
            else:
                current_chunk += para + "\n\n"
        
        # Add remaining content
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                chunk_id=f"{document.document_id}-{position}",
                document_id=document.document_id,
                content=current_chunk.strip(),
                position=position,
                start_char=current_start,
                end_char=len(content),
                metadata={"strategy": "semantic"},
            ))
        
        return self._validate_chunks(chunks, document)
    
    def _chunk_sentence(self, document: HyDEDocument) -> List[DocumentChunk]:
        """Chunk on sentence boundaries."""
        content = document.content
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        position = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) > self.config.chunk_size and current_chunk:
                chunks.append(DocumentChunk(
                    chunk_id=f"{document.document_id}-{position}",
                    document_id=document.document_id,
                    content=current_chunk.strip(),
                    position=position,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={"strategy": "sentence"},
                ))
                position += 1
                
                # Apply overlap
                overlap_text = current_chunk[-self.config.chunk_overlap:] if self.config.chunk_overlap > 0 else ""
                current_start += len(current_chunk) - len(overlap_text)
                current_chunk = overlap_text + sentence + " "
            else:
                current_chunk += sentence + " "
        
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                chunk_id=f"{document.document_id}-{position}",
                document_id=document.document_id,
                content=current_chunk.strip(),
                position=position,
                start_char=current_start,
                end_char=len(content),
                metadata={"strategy": "sentence"},
            ))
        
        return self._validate_chunks(chunks, document)
    
    def _chunk_paragraph(self, document: HyDEDocument) -> List[DocumentChunk]:
        """Chunk on paragraph boundaries."""
        content = document.content
        paragraphs = re.split(r'\n\n+', content)
        
        chunks = []
        position = 0
        current_start = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < self.config.min_chunk_size:
                continue
            
            chunks.append(DocumentChunk(
                chunk_id=f"{document.document_id}-{position}",
                document_id=document.document_id,
                content=para,
                position=position,
                start_char=current_start,
                end_char=current_start + len(para),
                metadata={"strategy": "paragraph"},
            ))
            position += 1
            current_start += len(para) + 2  # +2 for \n\n
        
        return self._validate_chunks(chunks, document)
    
    def _chunk_fixed(self, document: HyDEDocument) -> List[DocumentChunk]:
        """Fixed-size chunking with overlap."""
        content = document.content
        chunks = []
        position = 0
        start = 0
        prev_start = -1  # v6.4.1: Track for loop guard
        
        while start < len(content):
            # v6.4.1: Loop guard — detect non-advancing start
            if start == prev_start:
                logger.warning("[HYDE] _chunk_fixed loop detected, breaking")
                break
            prev_start = start

            end = start + self.config.chunk_size
            
            # Adjust to word boundary if possible
            if end < len(content) and self.config.preserve_sentences:
                # Look for sentence end
                sentence_end = content.rfind('.', start, end)
                if sentence_end > start + self.config.min_chunk_size:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = content.rfind(' ', start, end)
                    if word_end > start + self.config.min_chunk_size:
                        end = word_end
            
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunks.append(DocumentChunk(
                    chunk_id=f"{document.document_id}-{position}",
                    document_id=document.document_id,
                    content=chunk_text,
                    position=position,
                    start_char=start,
                    end_char=end,
                    metadata={"strategy": "fixed"},
                ))
                position += 1
            
            start = end - self.config.chunk_overlap
            if start <= 0 or self.config.chunk_overlap == 0:
                start = end
        
        return chunks
    
    def _validate_chunks(
        self,
        chunks: List[DocumentChunk],
        document: HyDEDocument,
    ) -> List[DocumentChunk]:
        """Validate and potentially merge small chunks."""
        if not chunks:
            # Return whole document as single chunk
            return [DocumentChunk(
                chunk_id=f"{document.document_id}-0",
                document_id=document.document_id,
                content=document.content,
                position=0,
                start_char=0,
                end_char=len(document.content),
                metadata={"strategy": "single"},
            )]
        
        # Merge chunks that are too small
        validated = []
        pending_merge = None
        
        for chunk in chunks:
            if chunk.length < self.config.min_chunk_size:
                if pending_merge:
                    # Merge with pending
                    pending_merge = DocumentChunk(
                        chunk_id=pending_merge.chunk_id,
                        document_id=pending_merge.document_id,
                        content=pending_merge.content + " " + chunk.content,
                        position=pending_merge.position,
                        start_char=pending_merge.start_char,
                        end_char=chunk.end_char,
                        metadata=pending_merge.metadata,
                    )
                else:
                    pending_merge = chunk
            else:
                if pending_merge:
                    # Merge pending with current
                    merged = DocumentChunk(
                        chunk_id=pending_merge.chunk_id,
                        document_id=pending_merge.document_id,
                        content=pending_merge.content + " " + chunk.content,
                        position=pending_merge.position,
                        start_char=pending_merge.start_char,
                        end_char=chunk.end_char,
                        metadata=pending_merge.metadata,
                    )
                    validated.append(merged)
                    pending_merge = None
                else:
                    validated.append(chunk)
        
        if pending_merge:
            if validated:
                # Merge with last
                last = validated[-1]
                validated[-1] = DocumentChunk(
                    chunk_id=last.chunk_id,
                    document_id=last.document_id,
                    content=last.content + " " + pending_merge.content,
                    position=last.position,
                    start_char=last.start_char,
                    end_char=pending_merge.end_char,
                    metadata=last.metadata,
                )
            else:
                validated.append(pending_merge)
        
        return validated


# ============================================================================
# Ensemble Fusion
# ============================================================================


class EnsembleFusion:
    """
    Combines multiple HyDE documents into an optimal result.
    
    Strategies:
    - weighted_concat: Weighted concatenation based on quality
    - best_selection: Select highest quality document
    - merge_unique: Merge unique information from each
    """
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
    
    def fuse(
        self,
        documents: List[HyDEDocument],
        strategy: Optional[str] = None,
    ) -> Optional[HyDEDocument]:
        """
        Fuse multiple documents using the specified strategy.
        
        Args:
            documents: List of documents to fuse
            strategy: Fusion strategy override
            
        Returns:
            Fused document or None if empty
        """
        if not documents:
            return None
        
        if len(documents) == 1:
            return documents[0]
        
        strategy = strategy or self.config.fusion_strategy
        
        if strategy == "best_selection":
            return self._best_selection(documents)
        elif strategy == "merge_unique":
            return self._merge_unique(documents)
        else:  # weighted_concat
            return self._weighted_concat(documents)
    
    def _best_selection(self, documents: List[HyDEDocument]) -> HyDEDocument:
        """Select the highest quality document."""
        return max(documents, key=lambda d: d.quality_score)
    
    def _weighted_concat(self, documents: List[HyDEDocument]) -> HyDEDocument:
        """Concatenate documents weighted by quality."""
        # Sort by quality
        sorted_docs = sorted(documents, key=lambda d: d.quality_score, reverse=True)
        
        # Take top portions based on quality
        total_quality = sum(d.quality_score for d in sorted_docs)
        if total_quality == 0:
            total_quality = len(sorted_docs)
        
        parts = []
        target_length = 600  # Target fused length
        current_length = 0
        
        for doc in sorted_docs:
            weight = doc.quality_score / total_quality if total_quality > 0 else 1 / len(sorted_docs)
            doc_target = int(target_length * weight)
            
            if doc_target > 50:  # Minimum contribution
                content = doc.content[:doc_target]
                # Trim to sentence boundary
                last_period = content.rfind('.')
                if last_period > doc_target * 0.7:
                    content = content[:last_period + 1]
                parts.append(content)
                current_length += len(content)
        
        fused_content = "\n\n".join(parts)
        
        # Use best document as base
        best = sorted_docs[0]
        return HyDEDocument(
            document_id=str(uuid.uuid4()),
            content=fused_content,
            query=best.query,
            format_type=best.format_type,
            domain=best.domain,
            language=best.language,
            quality_score=sum(d.quality_score for d in sorted_docs) / len(sorted_docs),
            confidence=min(d.confidence for d in sorted_docs),
            metadata={
                "fused": True,
                "source_count": len(documents),
                "strategy": "weighted_concat",
            },
        )
    
    def _merge_unique(self, documents: List[HyDEDocument]) -> HyDEDocument:
        """Merge unique sentences from each document."""
        seen_sentences: Set[str] = set()
        unique_parts = []
        
        for doc in sorted(documents, key=lambda d: d.quality_score, reverse=True):
            sentences = re.split(r'(?<=[.!?])\s+', doc.content)
            for sentence in sentences:
                # Normalize for comparison
                normalized = sentence.lower().strip()
                if len(normalized) > 20 and normalized not in seen_sentences:
                    # Check similarity with existing
                    is_unique = True
                    for seen in seen_sentences:
                        ratio = SequenceMatcher(None, normalized, seen).ratio()
                        if ratio > 0.8:
                            is_unique = False
                            break
                    
                    if is_unique:
                        seen_sentences.add(normalized)
                        unique_parts.append(sentence.strip())
        
        fused_content = " ".join(unique_parts)
        best = max(documents, key=lambda d: d.quality_score)
        
        return HyDEDocument(
            document_id=str(uuid.uuid4()),
            content=fused_content,
            query=best.query,
            format_type=best.format_type,
            domain=best.domain,
            language=best.language,
            quality_score=sum(d.quality_score for d in documents) / len(documents),
            confidence=min(d.confidence for d in documents),
            metadata={
                "fused": True,
                "source_count": len(documents),
                "strategy": "merge_unique",
                "unique_sentences": len(unique_parts),
            },
        )
    
    def calculate_diversity(self, documents: List[HyDEDocument]) -> float:
        """Calculate diversity score for document set."""
        if len(documents) < 2:
            return 1.0
        
        similarities = []
        for i, doc1 in enumerate(documents):
            for doc2 in documents[i + 1:]:
                ratio = SequenceMatcher(None, doc1.content.lower(), doc2.content.lower()).ratio()
                similarities.append(ratio)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        return 1.0 - avg_similarity  # Higher diversity = lower similarity


# ============================================================================
# Redis Cache Provider
# ============================================================================


class RedisCacheProvider:
    """
    Environment-aware Redis caching for HyDE results.
    
    Features:
    - Environment isolation (dev/test/prod)
    - Semantic similarity matching for cache hits
    - TTL management
    - Hit/miss statistics
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
                    logger.debug("[HYDE-CACHE] HIT (redis) op=%s key=%s", operation, cache_key)
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        # Fall back to local cache
        if cache_key in self._local_cache:
            entry = self._local_cache[cache_key]
            if entry["expires_at"] > datetime.utcnow():
                self._stats["hits"] += 1
                logger.debug("[HYDE-CACHE] HIT (local) op=%s key=%s", operation, cache_key)
                return entry["data"]
            else:
                del self._local_cache[cache_key]

        self._stats["misses"] += 1
        logger.debug("[HYDE-CACHE] MISS op=%s key=%s", operation, cache_key)
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


class HyDESessionManager:
    """
    Manages HyDE sessions with history tracking.
    
    Features:
    - UUID-based sessions
    - History tracking
    - Document storage
    - TTL-based expiration
    - Auto-cleanup
    """
    
    def __init__(self, config: SessionConfig):
        self.config = config
        self._sessions: Dict[str, HyDESession] = {}
        self._last_cleanup = datetime.utcnow()
    
    async def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> HyDESession:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session = HyDESession(
            session_id=session_id,
            metadata=metadata or {},
        )
        self._sessions[session_id] = session
        
        # Trigger cleanup if needed
        await self._maybe_cleanup()
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[HyDESession]:
        """Get session by ID."""
        session = self._sessions.get(session_id)
        if session:
            # Check TTL
            elapsed = (datetime.utcnow() - session.updated_at).total_seconds()
            if elapsed > self.config.ttl_seconds:
                del self._sessions[session_id]
                return None
        return session
    
    async def update_session(
        self,
        session_id: str,
        query: str,
        document: HyDEDocument,
        format_type: str,
        domain: str,
    ) -> Optional[HyDESession]:
        """Update session with new interaction."""
        session = await self.get_session(session_id)
        if not session:
            return None
        
        # Add to history
        session.history.append({
            "query": query,
            "format_type": format_type,
            "domain": domain,
            "document_id": document.document_id,
            "quality_score": document.quality_score,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Trim history if needed
        if len(session.history) > self.config.max_history_size:
            session.history = session.history[-self.config.max_history_size:]
        
        # Store document if enabled
        if self.config.persist_documents:
            session.documents.append(document)
            if len(session.documents) > self.config.max_history_size:
                session.documents = session.documents[-self.config.max_history_size:]
        
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
        if (now - self._last_cleanup).total_seconds() < 300:  # Every 5 minutes
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
            logger.debug(f"Cleaned up {len(expired)} expired sessions")


# ============================================================================
# Worker Pool
# ============================================================================


class HyDEWorkerPool:
    """
    Async worker pool for parallel task execution.
    
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
        
        logger.info(f"HyDE worker pool started with {self.config.pool_size} workers")
    
    async def stop(self) -> None:
        """Stop worker pool."""
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("HyDE worker pool stopped")
    
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
        
        # Priority queue uses (priority, task) tuples
        # Lower number = higher priority, so negate
        await self._queue.put((-priority.value, task))
        
        return task
    
    async def execute_batch(
        self,
        tasks: List[Tuple[Callable, tuple, Dict[str, Any]]],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> List[Any]:
        """Execute multiple tasks and wait for results."""
        submitted = []
        for func, args, kwargs in tasks:
            task = await self.submit(func, *args, priority=priority, **kwargs)
            submitted.append(task)
        
        # Wait for all tasks
        results = []
        for task in submitted:
            while task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                await asyncio.sleep(0.1)
            results.append(task.result if task.status == TaskStatus.COMPLETED else task.error)
        
        return results
    
    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop processing tasks from queue."""
        while self._running:
            try:
                # Get task with timeout
                try:
                    _, task = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Execute task
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                start_time = time.perf_counter()
                
                try:
                    # Apply timeout
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
                    
                    # Retry logic
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


class MetricsCollector:
    """
    Comprehensive metrics collection for HyDE operations.
    
    Tracks:
    - Generation counts and times
    - Format distribution
    - Domain distribution
    - Quality scores
    - Hallucination rates
    - Refinement statistics
    """
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self._metrics = {
            "total_generations": 0,
            "format_distribution": {},
            "domain_distribution": {},
            "qa_scores": [],
            "hallucination_detected": 0,
            "refinements_applied": 0,
            "execution_times_ms": [],
            "ensemble_generations": 0,
        }
    
    async def record_generation(
        self,
        format_type: str,
        domain: str,
        quality_score: float,
        execution_time_ms: float,
        hallucination_detected: bool,
        refinement_applied: bool,
        ensemble_used: bool,
    ) -> None:
        """Record a generation event."""
        if not self.config.enabled:
            return
        
        self._metrics["total_generations"] += 1
        
        if self.config.collect_format_distribution:
            self._metrics["format_distribution"][format_type] = \
                self._metrics["format_distribution"].get(format_type, 0) + 1
        
        if self.config.collect_domain_distribution:
            self._metrics["domain_distribution"][domain] = \
                self._metrics["domain_distribution"].get(domain, 0) + 1
        
        if self.config.collect_qa_scores:
            self._metrics["qa_scores"].append(quality_score)
            # Keep only recent scores
            if len(self._metrics["qa_scores"]) > 1000:
                self._metrics["qa_scores"] = self._metrics["qa_scores"][-1000:]
        
        if self.config.collect_hallucination_rates and hallucination_detected:
            self._metrics["hallucination_detected"] += 1
        
        if self.config.collect_refinement_stats and refinement_applied:
            self._metrics["refinements_applied"] += 1
        
        if self.config.collect_timings:
            self._metrics["execution_times_ms"].append(execution_time_ms)
            if len(self._metrics["execution_times_ms"]) > 1000:
                self._metrics["execution_times_ms"] = self._metrics["execution_times_ms"][-1000:]
        
        if ensemble_used:
            self._metrics["ensemble_generations"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        qa_scores = self._metrics["qa_scores"]
        times = self._metrics["execution_times_ms"]
        total = self._metrics["total_generations"]
        
        return {
            "total_generations": total,
            "format_distribution": self._metrics["format_distribution"],
            "domain_distribution": self._metrics["domain_distribution"],
            "qa_scores": {
                "avg": sum(qa_scores) / len(qa_scores) if qa_scores else 0,
                "min": min(qa_scores) if qa_scores else 0,
                "max": max(qa_scores) if qa_scores else 0,
            },
            "execution_times": {
                "avg_ms": sum(times) / len(times) if times else 0,
                "min_ms": min(times) if times else 0,
                "max_ms": max(times) if times else 0,
            },
            "hallucination_rate": self._metrics["hallucination_detected"] / total if total > 0 else 0,
            "refinement_rate": self._metrics["refinements_applied"] / total if total > 0 else 0,
            "ensemble_rate": self._metrics["ensemble_generations"] / total if total > 0 else 0,
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics = {
            "total_generations": 0,
            "format_distribution": {},
            "domain_distribution": {},
            "qa_scores": [],
            "hallucination_detected": 0,
            "refinements_applied": 0,
            "execution_times_ms": [],
            "ensemble_generations": 0,
        }
