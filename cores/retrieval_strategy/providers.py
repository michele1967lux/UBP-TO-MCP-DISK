"""
retrieval_strategy/providers.py

Core retrieval components and algorithms.
ZERO dependencies from backend.app - must be testable standalone.

Provides:
- RetrievalResult: Single retrieval result
- RetrievalResponse: Collection of results with metadata
- BM25Index: In-memory BM25 index
- BM25Retriever: BM25-based retrieval
- HierarchicalChunk: Multi-level chunk representation
- HierarchicalIndex: Document/Section/Paragraph hierarchy
- IndexRegistry: Multi-index management
- RetrievalCacheProvider: Query caching
- RetrievalMetricsCollector: Performance metrics

v1.0.0: Initial release
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from enum import Enum
from typing import (
    Any, Callable, Dict, FrozenSet, Generator, Iterable,
    List, Optional, Protocol, Set, Tuple, Union,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"
    MULTI_INDEX = "multi_index"
    ROUTER = "router"


class FusionMethod(Enum):
    """Score fusion methods."""
    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted"  # Weighted combination
    MAX = "max"  # Take maximum score
    SUM = "sum"  # Sum of scores
    DBSF = "dbsf"  # Distribution-Based Score Fusion


class HierarchyLevel(Enum):
    """Hierarchical retrieval levels."""
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"


class QueryClass(Enum):
    """Query classification for routing."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    UNKNOWN = "unknown"


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class BM25Config:
    """BM25 retriever configuration."""
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    top_k: int = 50
    stopwords: bool = True
    stemming: bool = True
    lowercase: bool = True
    min_token_length: int = 2
    language: str = "auto"


@dataclass
class VectorConfig:
    """Vector retriever configuration."""
    embedding_module: str = "embedding_service"
    embedding_operation: str = "embed"
    vector_store_module: str = "qdrant_store"
    vector_store_operation: str = "search"
    top_k: int = 50
    similarity_metric: str = "cosine"
    score_threshold: float = 0.0


@dataclass
class HybridConfig:
    """Hybrid retrieval configuration."""
    fusion_method: str = "rrf"
    alpha: float = 0.5
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    normalize_scores: bool = True
    deduplicate: bool = True
    dedupe_threshold: float = 0.95


@dataclass
class HierarchicalConfig:
    """Hierarchical retrieval configuration."""
    document_chunk_size: int = 4000
    document_overlap: int = 200
    document_top_k: int = 5
    document_weight: float = 0.2
    
    section_chunk_size: int = 1000
    section_overlap: int = 100
    section_top_k: int = 10
    section_weight: float = 0.3
    
    paragraph_chunk_size: int = 300
    paragraph_overlap: int = 50
    paragraph_top_k: int = 20
    paragraph_weight: float = 0.5
    
    parent_child_linking: bool = True
    expand_context: bool = True
    context_window_sentences: int = 2


@dataclass
class RouterConfig:
    """Router configuration."""
    llm_module: str = "inference_ollama_grok"
    llm_operation: str = "generate"
    temperature: float = 0.1
    timeout_seconds: int = 10
    fallback_strategy: str = "hybrid"
    cache_decisions: bool = True


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    ttl_seconds: int = 1800
    semantic_cache: bool = True
    semantic_threshold: float = 0.95


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    enabled: bool = True
    track_latency: bool = True
    track_hit_rates: bool = True
    retention_hours: int = 24


@dataclass
class DebugConfig:
    """Debug configuration."""
    enabled: bool = False
    log_queries: bool = True
    log_scores: bool = True
    log_fusion: bool = True
    log_routing: bool = True


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # Which retriever produced this
    rank: int = 0
    hierarchy_level: Optional[HierarchyLevel] = None
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "score": round(self.score, 4),
            "metadata": self.metadata,
            "source": self.source,
            "rank": self.rank,
            "hierarchy_level": self.hierarchy_level.value if self.hierarchy_level else None,
        }
    
    def __hash__(self):
        return hash(self.doc_id)
    
    def __eq__(self, other):
        if isinstance(other, RetrievalResult):
            return self.doc_id == other.doc_id
        return False


@dataclass
class RetrievalResponse:
    """Collection of retrieval results."""
    query: str
    results: List[RetrievalResult] = field(default_factory=list)
    strategy_used: RetrievalStrategy = RetrievalStrategy.HYBRID
    fusion_method: Optional[FusionMethod] = None
    total_candidates: int = 0
    retrieval_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "result_count": len(self.results),
            "strategy_used": self.strategy_used.value,
            "fusion_method": self.fusion_method.value if self.fusion_method else None,
            "total_candidates": self.total_candidates,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }


@dataclass
class RouterDecision:
    """Router decision for a query."""
    query: str
    query_class: QueryClass
    selected_strategy: RetrievalStrategy
    selected_indexes: List[str]
    skip_retrieval: bool = False
    confidence: float = 1.0
    reasoning: str = ""
    suggested_top_k: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_class": self.query_class.value,
            "selected_strategy": self.selected_strategy.value,
            "selected_indexes": self.selected_indexes,
            "skip_retrieval": self.skip_retrieval,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "suggested_top_k": self.suggested_top_k,
        }


@dataclass
class HierarchicalChunk:
    """Chunk with hierarchical relationships."""
    chunk_id: str
    content: str
    level: HierarchyLevel
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    doc_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content_preview": self.content[:200],
            "level": self.level.value,
            "parent_id": self.parent_id,
            "child_count": len(self.child_ids),
            "doc_id": self.doc_id,
        }


# ============================================================================
# Stopwords
# ============================================================================


STOPWORDS_EN = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "the", "this", "but", "they",
    "have", "had", "what", "when", "where", "who", "which", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "just", "should", "now", "i", "you",
    "your", "we", "our", "them", "their", "been", "being", "do", "does",
    "did", "doing", "would", "could", "might", "must", "shall",
}

STOPWORDS_IT = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a",
    "da", "in", "con", "su", "per", "tra", "fra", "e", "o", "ma", "se",
    "che", "chi", "cui", "non", "come", "dove", "quando", "perché",
    "anche", "solo", "già", "ancora", "sempre", "mai", "più", "meno",
    "molto", "poco", "tutto", "niente", "nulla", "ogni", "qualche",
    "questo", "quello", "quale", "quanto", "sono", "è", "essere",
    "avere", "fare", "dire", "potere", "volere", "dovere", "stare",
    "io", "tu", "lui", "lei", "noi", "voi", "loro", "mi", "ti", "ci",
    "vi", "si", "ne", "lo", "la", "li", "le", "gli",
}


# ============================================================================
# Text Processing
# ============================================================================


class SimpleTokenizer:
    """Simple tokenizer for BM25."""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        min_length: int = 2,
        language: str = "en",
    ):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.min_length = min_length
        self.language = language
        
        self.stopwords = STOPWORDS_EN if language == "en" else STOPWORDS_IT
        if language == "auto":
            self.stopwords = STOPWORDS_EN | STOPWORDS_IT
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.lowercase:
            text = text.lower()
        
        # Split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text)
        
        # Filter
        result = []
        for token in tokens:
            if len(token) < self.min_length:
                continue
            if self.remove_stopwords and token in self.stopwords:
                continue
            result.append(token)
        
        return result


# ============================================================================
# BM25 Implementation
# ============================================================================


class BM25Index:
    """
    In-memory BM25 index.
    
    Implements Okapi BM25 scoring algorithm.
    """
    
    def __init__(self, config: BM25Config):
        self.config = config
        self.tokenizer = SimpleTokenizer(
            lowercase=config.lowercase,
            remove_stopwords=config.stopwords,
            min_length=config.min_token_length,
            language=config.language,
        )
        
        # Index structures
        self._documents: Dict[str, str] = {}  # doc_id -> content
        self._doc_lengths: Dict[str, int] = {}  # doc_id -> token count
        self._doc_tokens: Dict[str, List[str]] = {}  # doc_id -> tokens
        self._term_doc_freq: Dict[str, Set[str]] = defaultdict(set)  # term -> doc_ids
        self._term_freq: Dict[str, Dict[str, int]] = defaultdict(dict)  # term -> {doc_id: freq}
        
        self._avg_doc_length: float = 0.0
        self._doc_count: int = 0
    
    def add_document(self, doc_id: str, content: str) -> None:
        """Add a document to the index."""
        tokens = self.tokenizer.tokenize(content)
        
        self._documents[doc_id] = content
        self._doc_tokens[doc_id] = tokens
        self._doc_lengths[doc_id] = len(tokens)
        
        # Update term frequencies
        term_counts = Counter(tokens)
        for term, count in term_counts.items():
            self._term_doc_freq[term].add(doc_id)
            self._term_freq[term][doc_id] = count
        
        # Update stats
        self._doc_count = len(self._documents)
        self._avg_doc_length = sum(self._doc_lengths.values()) / max(self._doc_count, 1)
    
    def add_documents(self, documents: List[Tuple[str, str]]) -> None:
        """Add multiple documents."""
        for doc_id, content in documents:
            self.add_document(doc_id, content)
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Search the index.
        
        Returns list of (doc_id, score) tuples sorted by score descending.
        """
        top_k = top_k or self.config.top_k
        query_tokens = self.tokenizer.tokenize(query)
        
        if not query_tokens:
            return []
        
        scores: Dict[str, float] = defaultdict(float)
        
        for term in query_tokens:
            if term not in self._term_doc_freq:
                continue
            
            # IDF calculation
            doc_freq = len(self._term_doc_freq[term])
            idf = math.log(
                (self._doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
            )
            
            # Score each document containing this term
            for doc_id in self._term_doc_freq[term]:
                tf = self._term_freq[term].get(doc_id, 0)
                doc_len = self._doc_lengths.get(doc_id, 1)
                
                # BM25 scoring
                numerator = tf * (self.config.k1 + 1)
                denominator = tf + self.config.k1 * (
                    1 - self.config.b + self.config.b * (doc_len / self._avg_doc_length)
                )
                
                scores[doc_id] += idf * (numerator / denominator)
        
        # Sort and return top_k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Get document content by ID."""
        return self._documents.get(doc_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "document_count": self._doc_count,
            "unique_terms": len(self._term_doc_freq),
            "avg_doc_length": round(self._avg_doc_length, 2),
            "total_tokens": sum(self._doc_lengths.values()),
        }
    
    def clear(self) -> None:
        """Clear the index."""
        self._documents.clear()
        self._doc_lengths.clear()
        self._doc_tokens.clear()
        self._term_doc_freq.clear()
        self._term_freq.clear()
        self._avg_doc_length = 0.0
        self._doc_count = 0


# ============================================================================
# Hierarchical Index
# ============================================================================


class HierarchicalIndex:
    """
    Multi-level hierarchical index.
    
    Supports document, section, and paragraph levels with
    parent-child relationships.
    """
    
    def __init__(self, config: HierarchicalConfig):
        self.config = config
        
        # Separate indexes for each level
        self._document_index = BM25Index(BM25Config(top_k=config.document_top_k))
        self._section_index = BM25Index(BM25Config(top_k=config.section_top_k))
        self._paragraph_index = BM25Index(BM25Config(top_k=config.paragraph_top_k))
        
        # Chunk storage
        self._chunks: Dict[str, HierarchicalChunk] = {}
        
        # Parent-child mappings
        self._parent_to_children: Dict[str, List[str]] = defaultdict(list)
        self._child_to_parent: Dict[str, str] = {}
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Add a document and create hierarchical chunks.
        
        Returns counts of chunks created at each level.
        """
        metadata = metadata or {}
        counts = {"document": 0, "section": 0, "paragraph": 0}
        
        # Document level
        doc_chunk = HierarchicalChunk(
            chunk_id=f"doc_{doc_id}",
            content=content,
            level=HierarchyLevel.DOCUMENT,
            doc_id=doc_id,
            metadata=metadata,
            start_char=0,
            end_char=len(content),
        )
        self._chunks[doc_chunk.chunk_id] = doc_chunk
        self._document_index.add_document(doc_chunk.chunk_id, content)
        counts["document"] = 1
        
        # Section level
        section_chunks = self._create_chunks(
            content=content,
            chunk_size=self.config.section_chunk_size,
            overlap=self.config.section_overlap,
            level=HierarchyLevel.SECTION,
            parent_id=doc_chunk.chunk_id,
            doc_id=doc_id,
            metadata=metadata,
        )
        
        for chunk in section_chunks:
            self._chunks[chunk.chunk_id] = chunk
            self._section_index.add_document(chunk.chunk_id, chunk.content)
            self._parent_to_children[doc_chunk.chunk_id].append(chunk.chunk_id)
            self._child_to_parent[chunk.chunk_id] = doc_chunk.chunk_id
            doc_chunk.child_ids.append(chunk.chunk_id)
        
        counts["section"] = len(section_chunks)
        
        # Paragraph level
        for section in section_chunks:
            para_chunks = self._create_chunks(
                content=section.content,
                chunk_size=self.config.paragraph_chunk_size,
                overlap=self.config.paragraph_overlap,
                level=HierarchyLevel.PARAGRAPH,
                parent_id=section.chunk_id,
                doc_id=doc_id,
                metadata=metadata,
                base_offset=section.start_char,
            )
            
            for chunk in para_chunks:
                self._chunks[chunk.chunk_id] = chunk
                self._paragraph_index.add_document(chunk.chunk_id, chunk.content)
                self._parent_to_children[section.chunk_id].append(chunk.chunk_id)
                self._child_to_parent[chunk.chunk_id] = section.chunk_id
                section.child_ids.append(chunk.chunk_id)
            
            counts["paragraph"] += len(para_chunks)
        
        return counts
    
    def _create_chunks(
        self,
        content: str,
        chunk_size: int,
        overlap: int,
        level: HierarchyLevel,
        parent_id: str,
        doc_id: str,
        metadata: Dict[str, Any],
        base_offset: int = 0,
    ) -> List[HierarchicalChunk]:
        """Create chunks from content with specified parameters."""
        chunks = []
        
        if len(content) <= chunk_size:
            chunk = HierarchicalChunk(
                chunk_id=f"{level.value}_{doc_id}_{len(chunks)}",
                content=content,
                level=level,
                parent_id=parent_id,
                doc_id=doc_id,
                metadata=metadata.copy(),
                start_char=base_offset,
                end_char=base_offset + len(content),
            )
            chunks.append(chunk)
            return chunks
        
        start = 0
        chunk_num = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # Try to break at sentence boundary
            if end < len(content):
                for sep in ['. ', '.\n', '! ', '? ', '\n\n']:
                    last_sep = content[start:end].rfind(sep)
                    if last_sep > chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunk = HierarchicalChunk(
                    chunk_id=f"{level.value}_{doc_id}_{chunk_num}",
                    content=chunk_content,
                    level=level,
                    parent_id=parent_id,
                    doc_id=doc_id,
                    metadata=metadata.copy(),
                    start_char=base_offset + start,
                    end_char=base_offset + end,
                )
                chunks.append(chunk)
                chunk_num += 1
            
            start = end - overlap
            if start >= len(content) - overlap:
                break
        
        return chunks
    
    def search(
        self,
        query: str,
        levels: Optional[List[HierarchyLevel]] = None,
        expand_context: bool = True,
    ) -> Dict[HierarchyLevel, List[Tuple[str, float]]]:
        """
        Search across hierarchy levels.
        
        Returns results grouped by level.
        """
        levels = levels or [HierarchyLevel.DOCUMENT, HierarchyLevel.SECTION, HierarchyLevel.PARAGRAPH]
        results = {}
        
        if HierarchyLevel.DOCUMENT in levels:
            results[HierarchyLevel.DOCUMENT] = self._document_index.search(query)
        
        if HierarchyLevel.SECTION in levels:
            results[HierarchyLevel.SECTION] = self._section_index.search(query)
        
        if HierarchyLevel.PARAGRAPH in levels:
            results[HierarchyLevel.PARAGRAPH] = self._paragraph_index.search(query)
        
        return results
    
    def get_chunk(self, chunk_id: str) -> Optional[HierarchicalChunk]:
        """Get chunk by ID."""
        return self._chunks.get(chunk_id)
    
    def get_parent(self, chunk_id: str) -> Optional[HierarchicalChunk]:
        """Get parent chunk."""
        parent_id = self._child_to_parent.get(chunk_id)
        if parent_id:
            return self._chunks.get(parent_id)
        return None
    
    def get_children(self, chunk_id: str) -> List[HierarchicalChunk]:
        """Get child chunks."""
        child_ids = self._parent_to_children.get(chunk_id, [])
        return [self._chunks[cid] for cid in child_ids if cid in self._chunks]
    
    def get_context_window(
        self,
        chunk_id: str,
        window_size: int = 2,
    ) -> List[HierarchicalChunk]:
        """Get surrounding chunks for context expansion."""
        chunk = self._chunks.get(chunk_id)
        if not chunk or not chunk.parent_id:
            return [chunk] if chunk else []
        
        siblings = self._parent_to_children.get(chunk.parent_id, [])
        
        try:
            idx = siblings.index(chunk_id)
        except ValueError:
            return [chunk]
        
        start = max(0, idx - window_size)
        end = min(len(siblings), idx + window_size + 1)
        
        return [self._chunks[sid] for sid in siblings[start:end] if sid in self._chunks]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_chunks": len(self._chunks),
            "document_count": self._document_index._doc_count,
            "section_count": self._section_index._doc_count,
            "paragraph_count": self._paragraph_index._doc_count,
        }


# ============================================================================
# Index Registry (Multi-Index Support)
# ============================================================================


class IndexRegistry:
    """
    Registry for managing multiple indexes.
    
    Supports routing queries to specific indexes.
    """
    
    def __init__(self):
        self._bm25_indexes: Dict[str, BM25Index] = {}
        self._hierarchical_indexes: Dict[str, HierarchicalIndex] = {}
        self._default_index = "default"
    
    def register_bm25_index(self, name: str, index: BM25Index) -> None:
        """Register a BM25 index."""
        self._bm25_indexes[name] = index
    
    def register_hierarchical_index(self, name: str, index: HierarchicalIndex) -> None:
        """Register a hierarchical index."""
        self._hierarchical_indexes[name] = index
    
    def get_bm25_index(self, name: str) -> Optional[BM25Index]:
        """Get BM25 index by name."""
        return self._bm25_indexes.get(name)
    
    def get_hierarchical_index(self, name: str) -> Optional[HierarchicalIndex]:
        """Get hierarchical index by name."""
        return self._hierarchical_indexes.get(name)
    
    def list_indexes(self) -> Dict[str, List[str]]:
        """List all registered indexes."""
        return {
            "bm25": list(self._bm25_indexes.keys()),
            "hierarchical": list(self._hierarchical_indexes.keys()),
        }
    
    def search_multiple(
        self,
        query: str,
        index_names: List[str],
        index_type: str = "bm25",
        top_k: int = 50,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Search multiple indexes."""
        results = {}
        
        indexes = self._bm25_indexes if index_type == "bm25" else self._hierarchical_indexes
        
        for name in index_names:
            if name in indexes:
                index = indexes[name]
                if isinstance(index, BM25Index):
                    results[name] = index.search(query, top_k)
                else:
                    # Hierarchical - return paragraph level
                    hier_results = index.search(query, [HierarchyLevel.PARAGRAPH])
                    results[name] = hier_results.get(HierarchyLevel.PARAGRAPH, [])
        
        return results


# ============================================================================
# Cache Provider
# ============================================================================


class RetrievalCacheProvider:
    """
    Caching for retrieval results.
    
    Features:
    - Exact query caching
    - Semantic similarity caching
    - TTL expiration
    """
    
    def __init__(self, config: CacheConfig, redis_client: Optional[Any] = None):
        self.config = config
        self._redis = redis_client
        self._local_cache: Dict[str, Any] = {}
        self._query_embeddings: Dict[str, List[float]] = {}
        self._stats = {"hits": 0, "misses": 0, "semantic_hits": 0}
    
    async def get(self, query: str) -> Optional[RetrievalResponse]:
        """Get cached results for query."""
        if not self.config.enabled:
            return None
        
        cache_key = self._hash_query(query)
        
        # Check Redis
        if self._redis:
            try:
                data = await self._redis.get(f"ubp:retrieval:cache:{cache_key}")
                if data:
                    self._stats["hits"] += 1
                    return self._deserialize_response(data)
            except Exception:
                pass
        
        # Check local cache
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
        query: str,
        response: RetrievalResponse,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Cache retrieval results."""
        if not self.config.enabled:
            return False
        
        cache_key = self._hash_query(query)
        serialized = self._serialize_response(response)
        
        # Store in Redis
        if self._redis:
            try:
                await self._redis.set(
                    f"ubp:retrieval:cache:{cache_key}",
                    serialized,
                    ex=self.config.ttl_seconds,
                )
            except Exception:
                pass
        
        # Store locally
        self._local_cache[cache_key] = {
            "data": response,
            "expires_at": datetime.utcnow() + timedelta(seconds=self.config.ttl_seconds),
        }
        
        if embedding and self.config.semantic_cache:
            self._query_embeddings[cache_key] = embedding
        
        return True
    
    def _hash_query(self, query: str) -> str:
        """Hash query for cache key."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _serialize_response(self, response: RetrievalResponse) -> str:
        """Serialize response for caching."""
        return json.dumps(response.to_dict())
    
    def _deserialize_response(self, data: str) -> RetrievalResponse:
        """Deserialize cached response."""
        d = json.loads(data)
        return RetrievalResponse(
            query=d["query"],
            results=[
                RetrievalResult(
                    doc_id=r["doc_id"],
                    content=r["content"],
                    score=r["score"],
                    metadata=r.get("metadata", {}),
                    source=r.get("source", ""),
                    rank=r.get("rank", 0),
                )
                for r in d.get("results", [])
            ],
            strategy_used=RetrievalStrategy(d["strategy_used"]),
            retrieval_time_ms=d["retrieval_time_ms"],
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        return {
            "enabled": self.config.enabled,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": self._stats["hits"] / total if total > 0 else 0,
            "semantic_hits": self._stats["semantic_hits"],
            "local_entries": len(self._local_cache),
        }


# ============================================================================
# Metrics Collector
# ============================================================================


class RetrievalMetricsCollector:
    """Metrics collection for retrieval operations."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self._metrics = {
            "total_queries": 0,
            "strategy_counts": defaultdict(int),
            "fusion_counts": defaultdict(int),
            "latencies": [],
            "result_counts": [],
            "cache_hits": 0,
            "router_decisions": defaultdict(int),
        }
    
    async def record_retrieval(
        self,
        strategy: RetrievalStrategy,
        fusion_method: Optional[FusionMethod],
        latency_ms: float,
        result_count: int,
        cache_hit: bool = False,
    ) -> None:
        """Record retrieval metrics."""
        if not self.config.enabled:
            return
        
        self._metrics["total_queries"] += 1
        self._metrics["strategy_counts"][strategy.value] += 1
        
        if fusion_method:
            self._metrics["fusion_counts"][fusion_method.value] += 1
        
        if self.config.track_latency:
            self._metrics["latencies"].append(latency_ms)
            if len(self._metrics["latencies"]) > 1000:
                self._metrics["latencies"] = self._metrics["latencies"][-1000:]
        
        self._metrics["result_counts"].append(result_count)
        if len(self._metrics["result_counts"]) > 1000:
            self._metrics["result_counts"] = self._metrics["result_counts"][-1000:]
        
        if cache_hit:
            self._metrics["cache_hits"] += 1
    
    async def record_router_decision(self, query_class: QueryClass) -> None:
        """Record router decision metrics."""
        if self.config.enabled:
            self._metrics["router_decisions"][query_class.value] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        latencies = self._metrics["latencies"]
        result_counts = self._metrics["result_counts"]
        
        return {
            "total_queries": self._metrics["total_queries"],
            "strategy_distribution": dict(self._metrics["strategy_counts"]),
            "fusion_distribution": dict(self._metrics["fusion_counts"]),
            "latency_stats": {
                "avg_ms": sum(latencies) / len(latencies) if latencies else 0,
                "min_ms": min(latencies) if latencies else 0,
                "max_ms": max(latencies) if latencies else 0,
                "p50_ms": sorted(latencies)[len(latencies)//2] if latencies else 0,
            },
            "avg_result_count": sum(result_counts) / len(result_counts) if result_counts else 0,
            "cache_hit_rate": self._metrics["cache_hits"] / max(self._metrics["total_queries"], 1),
            "router_decisions": dict(self._metrics["router_decisions"]),
        }
