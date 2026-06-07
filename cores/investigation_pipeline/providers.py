"""
investigation_pipeline/providers.py

Logic Layer - ZERO dependencies from backend.app
Must be testable standalone.

Provides:
- InvestigationWorkerPool: Parallel async task execution
- QualityAssuranceProvider: Multi-dimensional scoring
- InvestigationDeduplicator: Remove duplicate questions
- InvestigationSessionManager: Session state with history
- RedisCacheProvider: Environment-aware caching
- MetricsCollector: Statistics and observability
- QueryClassifier: Category detection and strategy selection
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class InvestigationStrategy(str, Enum):
    """Available investigation strategies."""
    DECOMPOSITION = "decomposition"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SEMANTIC_EXPANSION = "semantic_expansion"
    CROSS_REFERENCE = "cross_reference"
    ADAPTIVE = "adaptive"


class QualityLevel(str, Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


class TaskStatus(str, Enum):
    """Worker task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Task priority levels (lower = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


# ============================================================================
# Data Classes - Configuration
# ============================================================================


@dataclass
class InvestigationConfig:
    """Main investigation configuration."""
    enabled: bool = True
    default_num_questions: int = 5
    min_questions: int = 3
    max_questions: int = 10
    default_strategy: str = "adaptive"
    temperature: float = 0.7
    max_tokens: int = 500
    timeout_seconds: int = 30
    retry_enabled: bool = True
    max_retries: int = 2
    retry_delay_seconds: float = 1.0


@dataclass
class QualityAssuranceConfig:
    """Configuration for quality assurance."""
    enabled: bool = True
    auto_retry_on_low_quality: bool = True
    max_qa_retries: int = 2
    min_acceptable_score: float = 4.0
    
    # Scoring weights (must sum to 1.0)
    weight_relevance: float = 0.40
    weight_specificity: float = 0.25
    weight_length: float = 0.20
    weight_structure: float = 0.15
    
    # Thresholds
    threshold_excellent: float = 8.0
    threshold_good: float = 6.0
    threshold_acceptable: float = 4.0
    
    # Length constraints
    length_optimal_min: int = 50
    length_optimal_max: int = 150
    min_words: int = 4
    min_length: int = 10
    max_length: int = 500
    
    # Validation
    require_question_mark: bool = True
    require_capitalization: bool = False
    min_keyword_overlap: float = 0.1
    check_off_topic: bool = True


@dataclass
class WorkerPoolConfig:
    """Configuration for worker pool."""
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
class SessionConfig:
    """Configuration for session management."""
    enabled: bool = True
    ttl_seconds: int = 3600
    max_history_size: int = 50
    persist_results: bool = True
    auto_cleanup: bool = True


@dataclass
class CacheConfig:
    """Configuration for Redis cache with environment isolation."""
    enabled: bool = True
    ttl_seconds: int = 3600
    base_prefix: str = "ubp"
    env: str = "dev"
    cache_questions: bool = True
    cache_qa_results: bool = True
    cache_sessions: bool = True

    @property
    def prefix(self) -> str:
        """Generate environment-isolated Redis key prefix."""
        return f"{self.base_prefix}:{self.env}:investigation:cache"


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""
    enabled: bool = True
    similarity_threshold: float = 0.85
    method: str = "fuzzy"
    min_unique_ratio: float = 0.70


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    enabled: bool = True
    collect_timings: bool = True
    collect_strategy_distribution: bool = True
    collect_qa_scores: bool = True
    retention_hours: int = 24


@dataclass
class DebugConfig:
    """Configuration for debugging."""
    enabled: bool = False
    log_prompts: bool = False
    log_responses: bool = False
    log_qa_scores: bool = True
    log_fallback_triggers: bool = True
    log_strategy_selection: bool = True
    log_worker_stats: bool = True
    trace_execution: bool = False


# ============================================================================
# Data Classes - Results
# ============================================================================


@dataclass
class InvestigationQuestion:
    """A single investigation question with metadata."""
    id: str
    text: str
    strategy: str
    category: str
    score: float = 0.0
    relevance_score: float = 0.0
    specificity_score: float = 0.0
    length_score: float = 0.0
    structure_score: float = 0.0
    quality_level: str = "pending"
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "strategy": self.strategy,
            "category": self.category,
            "score": self.score,
            "relevance_score": self.relevance_score,
            "specificity_score": self.specificity_score,
            "length_score": self.length_score,
            "structure_score": self.structure_score,
            "quality_level": self.quality_level,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvestigationQuestion":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            text=data.get("text", ""),
            strategy=data.get("strategy", "unknown"),
            category=data.get("category", "unknown"),
            score=data.get("score", 0.0),
            relevance_score=data.get("relevance_score", 0.0),
            specificity_score=data.get("specificity_score", 0.0),
            length_score=data.get("length_score", 0.0),
            structure_score=data.get("structure_score", 0.0),
            quality_level=data.get("quality_level", "pending"),
            is_valid=data.get("is_valid", True),
            validation_errors=data.get("validation_errors", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class QualityAssessment:
    """Result of quality assessment."""
    overall_score: float
    relevance_score: float
    specificity_score: float
    length_score: float
    structure_score: float
    quality_level: QualityLevel
    is_acceptable: bool
    issues: List[str]
    suggestions: List[str]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "relevance_score": self.relevance_score,
            "specificity_score": self.specificity_score,
            "length_score": self.length_score,
            "structure_score": self.structure_score,
            "quality_level": self.quality_level.value,
            "is_acceptable": self.is_acceptable,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "details": self.details,
        }


@dataclass
class QueryClassification:
    """Result of query classification."""
    category: str
    confidence: float
    keywords_matched: List[str]
    preferred_strategy: str
    secondary_strategy: Optional[str]
    all_matches: Dict[str, float]
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "keywords_matched": self.keywords_matched,
            "preferred_strategy": self.preferred_strategy,
            "secondary_strategy": self.secondary_strategy,
            "all_matches": self.all_matches,
            "language": self.language,
        }


@dataclass
class WorkerTask:
    """A task for the worker pool."""
    id: str
    name: str
    coroutine: Awaitable[Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retries: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def execution_time_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None


@dataclass
class WorkerPoolStats:
    """Statistics for worker pool."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    timeout_tasks: int = 0
    avg_execution_time_ms: float = 0.0
    queue_size: int = 0
    active_workers: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "timeout_tasks": self.timeout_tasks,
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "queue_size": self.queue_size,
            "active_workers": self.active_workers,
        }


@dataclass
class InvestigationSession:
    """Session state for investigation."""
    session_id: str
    user_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    queries: List[str] = field(default_factory=list)
    questions_generated: List[Dict[str, Any]] = field(default_factory=list)
    strategies_used: List[str] = field(default_factory=list)
    categories_detected: List[str] = field(default_factory=list)
    total_investigations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "queries": self.queries,
            "questions_generated": self.questions_generated,
            "strategies_used": self.strategies_used,
            "categories_detected": self.categories_detected,
            "total_investigations": self.total_investigations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvestigationSession":
        return cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            user_id=data.get("user_id"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            queries=data.get("queries", []),
            questions_generated=data.get("questions_generated", []),
            strategies_used=data.get("strategies_used", []),
            categories_detected=data.get("categories_detected", []),
            total_investigations=data.get("total_investigations", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class InvestigationResult:
    """Complete result from investigation pipeline."""
    session_id: str
    original_query: str
    questions: List[InvestigationQuestion]
    strategy_used: str
    category_detected: str
    classification: QueryClassification
    quality_assessment: Optional[QualityAssessment]
    deduplication_stats: Dict[str, Any]
    pipeline_stats: Dict[str, Any]
    time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "questions": [q.to_dict() for q in self.questions],
            "strategy_used": self.strategy_used,
            "category_detected": self.category_detected,
            "classification": self.classification.to_dict(),
            "quality_assessment": self.quality_assessment.to_dict() if self.quality_assessment else None,
            "deduplication_stats": self.deduplication_stats,
            "pipeline_stats": self.pipeline_stats,
            "time_ms": self.time_ms,
        }


# ============================================================================
# Exceptions
# ============================================================================


class InvestigationError(Exception):
    """Base exception for investigation errors."""
    pass


class WorkerPoolError(InvestigationError):
    """Worker pool operation failed."""
    pass


class QualityAssuranceError(InvestigationError):
    """QA validation failed."""
    pass


class SessionError(InvestigationError):
    """Session operation failed."""
    pass


class TaskTimeoutError(InvestigationError):
    """Task execution timed out."""
    pass


class ClassificationError(InvestigationError):
    """Query classification failed."""
    pass


# ============================================================================
# QueryClassifier
# ============================================================================


class QueryClassifier:
    """
    Classifies queries to determine optimal investigation strategy.
    
    Uses keyword matching with weighted scoring to detect category.
    """

    def __init__(self, categories: Dict[str, Dict[str, Any]], default_category: str = "technical"):
        self.categories = categories
        self.default_category = default_category

    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query and determine optimal strategy.
        
        Args:
            query: User's query text
            
        Returns:
            QueryClassification with detected category and strategy
        """
        query_lower = query.lower()
        language = self._detect_language(query_lower)
        
        all_matches: Dict[str, float] = {}
        keywords_found: Dict[str, List[str]] = {}
        
        for cat_name, cat_config in self.categories.items():
            keywords = cat_config.get("keywords", [])
            exclusive = cat_config.get("exclusive", [])
            weight = cat_config.get("weight", 1.0)
            
            matched_keywords = []
            score = 0.0
            
            # Check exclusive keywords first (higher weight)
            for kw in exclusive:
                if kw.lower() in query_lower:
                    matched_keywords.append(kw)
                    score += 2.0  # Exclusive keywords have double weight
            
            # Check regular keywords
            for kw in keywords:
                if kw.lower() in query_lower and kw not in matched_keywords:
                    matched_keywords.append(kw)
                    score += 1.0
            
            # Apply category weight
            final_score = score * float(weight)
            
            if final_score > 0:
                all_matches[cat_name] = final_score
                keywords_found[cat_name] = matched_keywords
        
        # Determine best category
        if all_matches:
            best_category = max(all_matches, key=all_matches.get)
            confidence = all_matches[best_category] / (sum(all_matches.values()) + 0.001)
        else:
            best_category = self.default_category
            confidence = 0.5
        
        # Get strategy from category config
        cat_config = self.categories.get(best_category, {})
        preferred_strategy = cat_config.get("preferred_strategy", "decomposition")
        secondary_strategy = cat_config.get("secondary_strategy")
        
        return QueryClassification(
            category=best_category,
            confidence=round(confidence, 3),
            keywords_matched=keywords_found.get(best_category, []),
            preferred_strategy=preferred_strategy,
            secondary_strategy=secondary_strategy,
            all_matches=all_matches,
            language=language,
        )

    def _detect_language(self, query_lower: str) -> str:
        """Detect query language."""
        italian_markers = [
            "come", "cosa", "perché", "quando", "dove", "chi",
            "non", "sono", "essere", "fare", "avere", "questo",
            "della", "delle", "degli", "nella", "nelle",
        ]
        italian_count = sum(1 for m in italian_markers if m in query_lower)
        return "it" if italian_count >= 2 else "en"


# ============================================================================
# QualityAssuranceProvider
# ============================================================================


class QualityAssuranceProvider:
    """
    Multi-dimensional quality assessment for investigation questions.
    
    Scoring dimensions:
    - Relevance: Keyword overlap with original query
    - Specificity: Technical terms and precision
    - Length: Optimal character count
    - Structure: Question format validation
    """

    # Technical terms for specificity detection
    TECHNICAL_TERMS = {
        "api", "database", "server", "config", "deploy", "error",
        "performance", "security", "authentication", "integration",
        "architecture", "endpoint", "cache", "query", "index",
        "async", "sync", "protocol", "format", "schema", "model",
        "training", "inference", "embedding", "vector", "neural",
    }

    # Off-topic indicators
    OFF_TOPIC_INDICATORS = [
        "weather", "sports", "entertainment", "politics", "religion",
        "favorite color", "meaning of life", "personal opinion",
    ]

    def __init__(self, config: QualityAssuranceConfig):
        self.config = config

    def assess_questions(
        self,
        questions: List[str],
        original_query: str,
        strategy: str,
    ) -> Tuple[List[InvestigationQuestion], QualityAssessment]:
        """
        Assess quality of multiple questions.
        
        Args:
            questions: List of question texts
            original_query: Original user query
            strategy: Strategy used to generate
            
        Returns:
            Tuple of (assessed questions, overall assessment)
        """
        assessed_questions: List[InvestigationQuestion] = []
        all_scores: List[float] = []
        all_issues: List[str] = []
        
        query_keywords = self._extract_keywords(original_query)
        
        for i, q_text in enumerate(questions):
            q = self._assess_single(
                text=q_text,
                original_query=original_query,
                query_keywords=query_keywords,
                strategy=strategy,
                index=i,
            )
            assessed_questions.append(q)
            all_scores.append(q.score)
            all_issues.extend(q.validation_errors)
        
        # Calculate overall assessment
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        quality_level = self._score_to_level(avg_score)
        
        overall = QualityAssessment(
            overall_score=round(avg_score, 2),
            relevance_score=round(
                sum(q.relevance_score for q in assessed_questions) / len(assessed_questions), 2
            ) if assessed_questions else 0.0,
            specificity_score=round(
                sum(q.specificity_score for q in assessed_questions) / len(assessed_questions), 2
            ) if assessed_questions else 0.0,
            length_score=round(
                sum(q.length_score for q in assessed_questions) / len(assessed_questions), 2
            ) if assessed_questions else 0.0,
            structure_score=round(
                sum(q.structure_score for q in assessed_questions) / len(assessed_questions), 2
            ) if assessed_questions else 0.0,
            quality_level=quality_level,
            is_acceptable=avg_score >= self.config.min_acceptable_score,
            issues=list(set(all_issues)),
            suggestions=self._generate_suggestions(assessed_questions, avg_score),
            details={
                "questions_assessed": len(questions),
                "valid_questions": sum(1 for q in assessed_questions if q.is_valid),
                "score_distribution": {
                    "excellent": sum(1 for q in assessed_questions if q.score >= self.config.threshold_excellent),
                    "good": sum(1 for q in assessed_questions if self.config.threshold_good <= q.score < self.config.threshold_excellent),
                    "acceptable": sum(1 for q in assessed_questions if self.config.threshold_acceptable <= q.score < self.config.threshold_good),
                    "poor": sum(1 for q in assessed_questions if q.score < self.config.threshold_acceptable),
                },
            },
        )
        
        return assessed_questions, overall

    def _assess_single(
        self,
        text: str,
        original_query: str,
        query_keywords: Set[str],
        strategy: str,
        index: int,
    ) -> InvestigationQuestion:
        """Assess a single question."""
        text = text.strip()
        validation_errors: List[str] = []
        
        # Calculate individual scores
        relevance_score = self._score_relevance(text, query_keywords)
        specificity_score = self._score_specificity(text)
        length_score = self._score_length(text)
        structure_score, struct_errors = self._score_structure(text)
        validation_errors.extend(struct_errors)
        
        # Check for off-topic content
        if self.config.check_off_topic:
            if self._is_off_topic(text):
                validation_errors.append("Question appears off-topic")
                relevance_score *= 0.3  # Heavy penalty
        
        # Calculate weighted score
        overall_score = (
            relevance_score * self.config.weight_relevance +
            specificity_score * self.config.weight_specificity +
            length_score * self.config.weight_length +
            structure_score * self.config.weight_structure
        )
        
        # Normalize to 0-10 scale
        overall_score = overall_score * 10
        
        quality_level = self._score_to_level(overall_score)
        is_valid = len(validation_errors) == 0 and overall_score >= self.config.min_acceptable_score
        
        return InvestigationQuestion(
            id=str(uuid.uuid4()),
            text=text,
            strategy=strategy,
            category="assessed",
            score=round(overall_score, 2),
            relevance_score=round(relevance_score, 3),
            specificity_score=round(specificity_score, 3),
            length_score=round(length_score, 3),
            structure_score=round(structure_score, 3),
            quality_level=quality_level.value,
            is_valid=is_valid,
            validation_errors=validation_errors,
            metadata={"index": index},
        )

    def _score_relevance(self, text: str, query_keywords: Set[str]) -> float:
        """Score relevance based on keyword overlap."""
        text_keywords = self._extract_keywords(text)
        if not query_keywords:
            return 0.5  # Neutral if no keywords
        
        overlap = query_keywords.intersection(text_keywords)
        ratio = len(overlap) / len(query_keywords)
        
        # Boost if at least one keyword matches
        if overlap:
            return min(0.5 + ratio, 1.0)
        return max(0.0, ratio - 0.1)

    def _score_specificity(self, text: str) -> float:
        """Score specificity based on technical terms."""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        technical_matches = words.intersection(self.TECHNICAL_TERMS)
        
        # Score based on technical term density
        if len(words) == 0:
            return 0.0
        
        density = len(technical_matches) / len(words)
        
        # Optimal: 10-30% technical terms
        if 0.1 <= density <= 0.3:
            return 1.0
        elif density < 0.1:
            return 0.5 + (density / 0.1) * 0.5
        else:
            return max(0.5, 1.0 - (density - 0.3))

    def _score_length(self, text: str) -> float:
        """Score based on optimal length."""
        length = len(text)
        
        if length < self.config.min_length:
            return 0.0
        elif length > self.config.max_length:
            return 0.3
        elif self.config.length_optimal_min <= length <= self.config.length_optimal_max:
            return 1.0
        elif length < self.config.length_optimal_min:
            return 0.6 + 0.4 * (length / self.config.length_optimal_min)
        else:
            overage = length - self.config.length_optimal_max
            return max(0.5, 1.0 - (overage / 200))

    def _score_structure(self, text: str) -> Tuple[float, List[str]]:
        """Score structure and return validation errors."""
        score = 1.0
        errors: List[str] = []
        
        # Check question mark
        if self.config.require_question_mark and not text.strip().endswith("?"):
            score -= 0.3
            errors.append("Missing question mark")
        
        # Check capitalization
        if self.config.require_capitalization and text and not text[0].isupper():
            score -= 0.1
            errors.append("Missing capitalization")
        
        # Check word count
        words = text.split()
        if len(words) < self.config.min_words:
            score -= 0.3
            errors.append(f"Too few words (min: {self.config.min_words})")
        
        # Check for complete sentence structure
        if not any(text.lower().startswith(q) for q in ["what", "how", "why", "when", "where", "which", "can", "is", "are", "does", "do", "qual", "come", "perché", "quando", "dove"]):
            score -= 0.1
        
        return max(0.0, score), errors

    def _score_to_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level."""
        if score >= self.config.threshold_excellent:
            return QualityLevel.EXCELLENT
        elif score >= self.config.threshold_good:
            return QualityLevel.GOOD
        elif score >= self.config.threshold_acceptable:
            return QualityLevel.ACCEPTABLE
        return QualityLevel.POOR

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once",
            "il", "la", "lo", "i", "gli", "le", "un", "una", "di", "da",
            "in", "con", "su", "per", "tra", "fra", "che", "è", "sono",
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if len(w) > 2 and w not in stopwords}

    def _is_off_topic(self, text: str) -> bool:
        """Check if question appears off-topic."""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in self.OFF_TOPIC_INDICATORS)

    def _generate_suggestions(
        self,
        questions: List[InvestigationQuestion],
        avg_score: float,
    ) -> List[str]:
        """Generate improvement suggestions."""
        suggestions: List[str] = []
        
        if avg_score < self.config.threshold_acceptable:
            suggestions.append("Consider regenerating with a different strategy")
        
        low_relevance = [q for q in questions if q.relevance_score < 0.5]
        if low_relevance:
            suggestions.append(f"{len(low_relevance)} questions have low relevance - ensure questions relate to the original query")
        
        low_specificity = [q for q in questions if q.specificity_score < 0.5]
        if low_specificity:
            suggestions.append(f"{len(low_specificity)} questions lack specificity - add more technical terms")
        
        invalid = [q for q in questions if not q.is_valid]
        if invalid:
            suggestions.append(f"{len(invalid)} questions failed validation - check format requirements")
        
        return suggestions


# ============================================================================
# InvestigationDeduplicator
# ============================================================================


class InvestigationDeduplicator:
    """Remove duplicate or near-duplicate questions."""

    def __init__(self, config: DeduplicationConfig):
        self.config = config

    def deduplicate(
        self,
        questions: List[InvestigationQuestion],
    ) -> Tuple[List[InvestigationQuestion], Dict[str, Any]]:
        """
        Remove duplicate questions.
        
        Returns:
            Tuple of (unique questions, stats)
        """
        if not self.config.enabled or not questions:
            return questions, {"enabled": False, "removed": 0}
        
        method = self.config.method
        threshold = self.config.similarity_threshold
        
        if method == "hash":
            unique, removed, groups = self._dedupe_by_hash(questions)
        elif method == "semantic":
            unique, removed, groups = self._dedupe_by_fuzzy(questions, threshold)
        else:  # fuzzy
            unique, removed, groups = self._dedupe_by_fuzzy(questions, threshold)
        
        unique_ratio = len(unique) / len(questions) if questions else 1.0
        
        stats = {
            "enabled": True,
            "method": method,
            "threshold": threshold,
            "original_count": len(questions),
            "unique_count": len(unique),
            "removed": removed,
            "duplicate_groups": len(groups),
            "unique_ratio": round(unique_ratio, 3),
            "meets_threshold": unique_ratio >= self.config.min_unique_ratio,
        }
        
        return unique, stats

    def _dedupe_by_hash(
        self,
        questions: List[InvestigationQuestion],
    ) -> Tuple[List[InvestigationQuestion], int, List[List[str]]]:
        """Deduplicate by exact text hash."""
        seen_hashes: Dict[str, InvestigationQuestion] = {}
        duplicate_groups: List[List[str]] = []
        
        for q in questions:
            h = hashlib.sha256(q.text.lower().strip().encode()).hexdigest()[:16]
            if h in seen_hashes:
                found = False
                for group in duplicate_groups:
                    if seen_hashes[h].id in group:
                        group.append(q.id)
                        found = True
                        break
                if not found:
                    duplicate_groups.append([seen_hashes[h].id, q.id])
            else:
                seen_hashes[h] = q
        
        unique = list(seen_hashes.values())
        removed = len(questions) - len(unique)
        return unique, removed, duplicate_groups

    def _dedupe_by_fuzzy(
        self,
        questions: List[InvestigationQuestion],
        threshold: float,
    ) -> Tuple[List[InvestigationQuestion], int, List[List[str]]]:
        """Deduplicate by fuzzy similarity."""
        unique: List[InvestigationQuestion] = []
        duplicate_groups: List[List[str]] = []
        
        for q in questions:
            is_duplicate = False
            for existing in unique:
                similarity = SequenceMatcher(
                    None,
                    q.text.lower().strip(),
                    existing.text.lower().strip()
                ).ratio()
                
                if similarity >= threshold:
                    found = False
                    for group in duplicate_groups:
                        if existing.id in group:
                            group.append(q.id)
                            found = True
                            break
                    if not found:
                        duplicate_groups.append([existing.id, q.id])
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(q)
        
        removed = len(questions) - len(unique)
        return unique, removed, duplicate_groups


# ============================================================================
# InvestigationWorkerPool
# ============================================================================


class InvestigationWorkerPool:
    """
    Parallel async task execution with priorities and retries.
    """

    def __init__(self, config: WorkerPoolConfig):
        self.config = config
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=config.queue_max_size
        )
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._stats = WorkerPoolStats()
        self._execution_times: List[float] = []
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return
        
        self._running = True
        for i in range(self.config.pool_size):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        
        logger.info(f"Worker pool started with {self.config.pool_size} workers")

    async def stop(self) -> None:
        """Stop the worker pool gracefully."""
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("Worker pool stopped")

    async def submit(
        self,
        name: str,
        coroutine: Awaitable[Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkerTask:
        """
        Submit a task to the pool.
        
        Args:
            name: Task name for identification
            coroutine: Async coroutine to execute
            priority: Task priority
            metadata: Optional task metadata
            
        Returns:
            WorkerTask with assigned ID
        """
        task = WorkerTask(
            id=str(uuid.uuid4()),
            name=name,
            coroutine=coroutine,
            priority=priority,
            metadata=metadata or {},
        )
        
        await self._queue.put((priority.value, task.created_at, task))
        
        async with self._lock:
            self._stats.total_tasks += 1
            self._stats.queue_size = self._queue.qsize()
        
        return task

    async def execute_batch(
        self,
        tasks: List[Tuple[str, Awaitable[Any]]],
        timeout: Optional[float] = None,
    ) -> List[WorkerTask]:
        """
        Execute multiple tasks and wait for all results.
        
        Args:
            tasks: List of (name, coroutine) tuples
            timeout: Optional timeout for all tasks
            
        Returns:
            List of completed WorkerTask objects
        """
        timeout = timeout or self.config.task_timeout_seconds
        
        # Submit all tasks
        submitted: List[WorkerTask] = []
        for name, coro in tasks:
            task = await self.submit(name, coro)
            submitted.append(task)
        
        # Wait for completion with timeout
        try:
            await asyncio.wait_for(
                self._wait_for_tasks(submitted),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Batch execution timed out after {timeout}s")
            for task in submitted:
                if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    task.status = TaskStatus.TIMEOUT
        
        return submitted

    async def _wait_for_tasks(self, tasks: List[WorkerTask]) -> None:
        """Wait for all tasks to complete."""
        while any(t.status in (TaskStatus.PENDING, TaskStatus.RUNNING) for t in tasks):
            await asyncio.sleep(0.1)

    async def _worker(self, name: str) -> None:
        """Worker coroutine that processes tasks from the queue."""
        logger.debug(f"Worker {name} started")
        
        while self._running:
            try:
                # Get task with timeout to allow checking _running
                try:
                    _, _, task = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue
                
                async with self._lock:
                    self._stats.active_workers += 1
                    self._stats.queue_size = self._queue.qsize()
                
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                
                try:
                    task.result = await asyncio.wait_for(
                        task.coroutine,
                        timeout=self.config.task_timeout_seconds,
                    )
                    task.status = TaskStatus.COMPLETED
                    
                    async with self._lock:
                        self._stats.completed_tasks += 1
                    
                except asyncio.TimeoutError:
                    task.status = TaskStatus.TIMEOUT
                    task.error = f"Task timed out after {self.config.task_timeout_seconds}s"
                    
                    async with self._lock:
                        self._stats.timeout_tasks += 1
                    
                    # Retry if enabled
                    if self.config.retry_on_failure and task.retries < self.config.max_task_retries:
                        task.retries += 1
                        await self._retry_task(task)
                
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    
                    async with self._lock:
                        self._stats.failed_tasks += 1
                    
                    # Retry if enabled
                    if self.config.retry_on_failure and task.retries < self.config.max_task_retries:
                        task.retries += 1
                        await self._retry_task(task)
                
                finally:
                    task.completed_at = time.time()
                    
                    if task.execution_time_ms:
                        self._execution_times.append(task.execution_time_ms)
                        # Keep only last 100 execution times
                        if len(self._execution_times) > 100:
                            self._execution_times = self._execution_times[-100:]
                        
                        async with self._lock:
                            if self._execution_times:
                                self._stats.avg_execution_time_ms = (
                                    sum(self._execution_times) / len(self._execution_times)
                                )
                    
                    async with self._lock:
                        self._stats.active_workers -= 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {name} error: {e}")
        
        logger.debug(f"Worker {name} stopped")

    async def _retry_task(self, task: WorkerTask) -> None:
        """Retry a failed task with exponential backoff."""
        delay = self.config.backoff_multiplier ** task.retries
        await asyncio.sleep(delay)
        
        task.status = TaskStatus.PENDING
        task.error = None
        
        await self._queue.put((task.priority.value, task.created_at, task))
        logger.info(f"Retrying task {task.name} (attempt {task.retries + 1})")

    def get_stats(self) -> WorkerPoolStats:
        """Get current worker pool statistics."""
        return self._stats


# ============================================================================
# InvestigationSessionManager
# ============================================================================


class InvestigationSessionManager:
    """Manage investigation sessions with history tracking."""

    def __init__(self, config: SessionConfig):
        self.config = config
        self._sessions: Dict[str, InvestigationSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InvestigationSession:
        """Create a new investigation session."""
        session = InvestigationSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self._sessions[session.session_id] = session
        
        return session

    async def get_session(
        self,
        session_id: str,
    ) -> Optional[InvestigationSession]:
        """Get a session by ID."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def update_session(
        self,
        session_id: str,
        query: str,
        questions: List[InvestigationQuestion],
        strategy: str,
        category: str,
    ) -> Optional[InvestigationSession]:
        """Update session with new investigation results."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            
            session.queries.append(query)
            session.questions_generated.extend([q.to_dict() for q in questions])
            session.strategies_used.append(strategy)
            session.categories_detected.append(category)
            session.total_investigations += 1
            session.updated_at = datetime.now(timezone.utc).isoformat()
            
            # Trim history if needed
            if len(session.questions_generated) > self.config.max_history_size:
                session.questions_generated = session.questions_generated[-self.config.max_history_size:]
            
            return session

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    async def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        if not self.config.auto_cleanup:
            return 0
        
        now = datetime.now(timezone.utc)
        expired: List[str] = []
        
        async with self._lock:
            for sid, session in self._sessions.items():
                updated = datetime.fromisoformat(session.updated_at.replace("Z", "+00:00"))
                age_seconds = (now - updated).total_seconds()
                if age_seconds > self.config.ttl_seconds:
                    expired.append(sid)
            
            for sid in expired:
                del self._sessions[sid]
        
        return len(expired)


# ============================================================================
# RedisCacheProvider - Environment-Aware Caching
# ============================================================================


class RedisCacheProvider:
    """
    Redis cache for investigation pipeline with environment isolation.
    
    Key format: ubp:{env}:investigation:cache:{operation}:{hash}
    """

    def __init__(self, config: CacheConfig, redis_client: Optional[Any] = None):
        self.config = config
        self._redis = redis_client
        self._stats = {
            "hits": 0,
            "misses": 0,
        }

    def _generate_key(self, operation: str, *args) -> str:
        """Generate a cache key with environment isolation."""
        content = json.dumps(args, sort_keys=True, default=str)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{self.config.prefix}:{operation}:{content_hash}"

    async def get(self, operation: str, *args) -> Optional[Any]:
        """Get cached value."""
        if not self.config.enabled or not self._redis:
            return None
        
        try:
            key = self._generate_key(operation, *args)
            cached = await self._redis.get(key)
            
            if cached:
                self._stats["hits"] += 1
                return json.loads(cached)
            else:
                self._stats["misses"] += 1
                return None
        
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            self._stats["misses"] += 1
            return None

    async def set(self, operation: str, value: Any, *args) -> bool:
        """Set cached value with TTL."""
        if not self.config.enabled or not self._redis:
            return False
        
        try:
            key = self._generate_key(operation, *args)
            serialized = json.dumps(value, default=str)
            
            await self._redis.setex(
                key,
                self.config.ttl_seconds,
                serialized,
            )
            return True
        
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    async def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries."""
        if not self._redis:
            return 0
        
        try:
            if pattern:
                full_pattern = f"{self.config.prefix}:{pattern}"
            else:
                full_pattern = f"{self.config.prefix}:*"
            
            deleted = 0
            async for key in self._redis.scan_iter(match=full_pattern):
                await self._redis.delete(key)
                deleted += 1
            
            return deleted
        
        except Exception as e:
            logger.warning(f"Cache invalidate error: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": round(hit_rate, 4),
            "env": self.config.env,
            "prefix": self.config.prefix,
            "enabled": self.config.enabled,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {"hits": 0, "misses": 0}

    async def clear(self) -> int:
        """Clear all cache entries for current environment."""
        cleared = await self.invalidate()
        self.reset_stats()
        return cleared


# ============================================================================
# MetricsCollector
# ============================================================================


class MetricsCollector:
    """Collect and aggregate investigation metrics."""

    def __init__(self, config: MetricsConfig):
        self.config = config
        self._metrics: Dict[str, Any] = {
            "total_investigations": 0,
            "questions_generated": 0,
            "strategies_used": {},
            "categories_detected": {},
            "qa_scores": [],
            "execution_times": [],
            "fallback_triggers": 0,
        }
        self._lock = asyncio.Lock()

    async def record_investigation(
        self,
        questions_count: int,
        strategy: str,
        category: str,
        qa_score: float,
        execution_time_ms: float,
        used_fallback: bool = False,
    ) -> None:
        """Record metrics for an investigation."""
        if not self.config.enabled:
            return
        
        async with self._lock:
            self._metrics["total_investigations"] += 1
            self._metrics["questions_generated"] += questions_count
            
            if self.config.collect_strategy_distribution:
                self._metrics["strategies_used"][strategy] = (
                    self._metrics["strategies_used"].get(strategy, 0) + 1
                )
                self._metrics["categories_detected"][category] = (
                    self._metrics["categories_detected"].get(category, 0) + 1
                )
            
            if self.config.collect_qa_scores:
                self._metrics["qa_scores"].append(qa_score)
                # Keep only last 1000 scores
                if len(self._metrics["qa_scores"]) > 1000:
                    self._metrics["qa_scores"] = self._metrics["qa_scores"][-1000:]
            
            if self.config.collect_timings:
                self._metrics["execution_times"].append(execution_time_ms)
                # Keep only last 1000 times
                if len(self._metrics["execution_times"]) > 1000:
                    self._metrics["execution_times"] = self._metrics["execution_times"][-1000:]
            
            if used_fallback:
                self._metrics["fallback_triggers"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics."""
        qa_scores = self._metrics["qa_scores"]
        exec_times = self._metrics["execution_times"]
        
        return {
            "total_investigations": self._metrics["total_investigations"],
            "questions_generated": self._metrics["questions_generated"],
            "strategies_used": self._metrics["strategies_used"],
            "categories_detected": self._metrics["categories_detected"],
            "qa_scores": {
                "count": len(qa_scores),
                "avg": round(sum(qa_scores) / len(qa_scores), 2) if qa_scores else 0,
                "min": round(min(qa_scores), 2) if qa_scores else 0,
                "max": round(max(qa_scores), 2) if qa_scores else 0,
            },
            "execution_times": {
                "count": len(exec_times),
                "avg_ms": round(sum(exec_times) / len(exec_times), 2) if exec_times else 0,
                "min_ms": round(min(exec_times), 2) if exec_times else 0,
                "max_ms": round(max(exec_times), 2) if exec_times else 0,
            },
            "fallback_triggers": self._metrics["fallback_triggers"],
        }

    async def reset(self) -> None:
        """Reset all metrics."""
        async with self._lock:
            self._metrics = {
                "total_investigations": 0,
                "questions_generated": 0,
                "strategies_used": {},
                "categories_detected": {},
                "qa_scores": [],
                "execution_times": [],
                "fallback_triggers": 0,
            }
