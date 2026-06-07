"""
enrichment_pipeline/providers.py

Logic Layer - ZERO dependencies from backend.app
Must be testable standalone.

Provides:
- RerankerProvider: Cross-encoder reranking with BGE
- ContextCompressor: Extractive and abstractive compression
- ChunkFusion: Merge overlapping/similar chunks
- Deduplicator: Remove duplicate chunks
- MetadataInjector: Add enrichment metadata
- RedisCacheProvider: Environment-aware caching
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class RerankerConfig:
    """Configuration for reranker."""

    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 512
    normalize_scores: bool = True
    cache_model: bool = True


@dataclass
class MedicalRerankerConfig:
    """Configuration for medical domain reranker (ncbi/MedCPT-Cross-Encoder)."""

    model: str = "ncbi/MedCPT-Cross-Encoder"
    device: str = "auto"
    batch_size: int = 64
    max_length: int = 512
    normalize_scores: bool = True
    cache_model: bool = True


@dataclass
class CacheConfig:
    """Configuration for Redis cache with environment isolation."""

    enabled: bool = True
    ttl_seconds: int = 3600
    # Environment-aware prefix: ubp:{env}:enrichment:cache
    base_prefix: str = "ubp"
    env: str = "dev"  # Injected from adapter
    cache_rerank: bool = True
    cache_hyde: bool = True

    @property
    def prefix(self) -> str:
        """Generate environment-isolated Redis key prefix."""
        return f"{self.base_prefix}:{self.env}:enrichment:cache"


@dataclass
class CompressionConfig:
    """Configuration for context compression."""

    default_ratio: float = 0.5
    method: str = "extractive"  # extractive, abstractive, hybrid
    min_chunk_length: int = 50
    preserve_sentences: bool = True


@dataclass
class FusionConfig:
    """Configuration for chunk fusion."""

    overlap_threshold: float = 0.3
    semantic_threshold: float = 0.93
    max_fused_length: int = 1000
    strategies: List[str] = field(
        default_factory=lambda: ["overlap", "adjacent", "semantic"]
    )


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""

    similarity_threshold: float = 0.95
    method: str = "semantic"  # hash, semantic, fuzzy


@dataclass
class EnrichedChunk:
    """Chunk with enrichment metadata."""

    kb_id: str
    chunk_id: str
    text: str
    score: float
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    enrichment_metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # v6.1.3: vector for dedup/fusion

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "kb_id": self.kb_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "metadata": self.metadata,
            "enrichment_metadata": self.enrichment_metadata,
        }
        if self.embedding is not None:
            d["embedding"] = self.embedding
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnrichedChunk":
        return cls(
            kb_id=data.get("kb_id", ""),
            chunk_id=data.get("chunk_id", ""),
            text=data.get("text", ""),
            score=data.get("score", 0.0),
            rerank_score=data.get("rerank_score"),
            metadata=data.get("metadata", {}),
            enrichment_metadata=data.get("enrichment_metadata", {}),
            embedding=data.get("embedding"),
        )


@dataclass
class RerankerResult:
    """Result from reranking operation."""

    reranked_chunks: List[EnrichedChunk]
    model_used: str
    time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reranked_chunks": [c.to_dict() for c in self.reranked_chunks],
            "model_used": self.model_used,
            "time_ms": self.time_ms,
        }


@dataclass
class CompressionResult:
    """Result from compression operation."""

    compressed_chunks: List[EnrichedChunk]
    original_tokens: int
    compressed_tokens: int
    actual_ratio: float
    method_used: str
    time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compressed_chunks": [c.to_dict() for c in self.compressed_chunks],
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "actual_ratio": self.actual_ratio,
            "method_used": self.method_used,
            "time_ms": self.time_ms,
        }


@dataclass
class FusionResult:
    """Result from fusion operation."""

    fused_chunks: List[EnrichedChunk]
    chunks_before: int
    chunks_after: int
    fusions_applied: List[Dict[str, Any]]
    time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fused_chunks": [c.to_dict() for c in self.fused_chunks],
            "chunks_before": self.chunks_before,
            "chunks_after": self.chunks_after,
            "fusions_applied": self.fusions_applied,
            "time_ms": self.time_ms,
        }


# ============================================================================
# Exceptions
# ============================================================================


class EnrichmentError(Exception):
    """Base exception for enrichment errors."""

    pass


class RerankerError(EnrichmentError):
    """Reranker operation failed."""

    pass


class ModelLoadError(EnrichmentError):
    """Model loading failed."""

    pass


# ============================================================================
# RerankerProvider
# ============================================================================


class RerankerProvider:
    """
    Cross-encoder reranking using BGE or compatible models.

    Supports:
    - BAAI/bge-reranker-v2-m3 (multilingual, recommended)
    - BAAI/bge-reranker-base
    - cross-encoder/ms-marco-MiniLM-L-6-v2

    ZERO dependencies from backend.app - fully standalone.
    """

    def __init__(self, config: RerankerConfig):
        self.config = config
        self._model = None
        self._is_loaded = False
        self._device = None

    def set_shared_model(self, model) -> None:
        """Inject pre-loaded model from SharedModelPool.

        Called by adapter.py to share a single GPU model across modules.
        When set, _ensure_model_loaded() becomes a no-op.
        """
        self._model = model
        self._is_loaded = True
        # Detect device from model
        self._device = getattr(model, 'device', None)
        if self._device and hasattr(self._device, 'type'):
            self._device = self._device.type
        logger.info(
            "Reranker (CrossEncoder) shared model injected: %s on %s",
            self.config.model,
            self._device or "unknown",
        )

    def _ensure_model_loaded(self) -> None:
        """Lazy load — fallback if no shared model was injected."""
        if self._is_loaded:
            return

        try:
            import torch
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker model (CrossEncoder): {self.config.model}")
            start = time.perf_counter()

            # Determine device with auto-detection support
            device_config = self.config.device.lower()

            if device_config == "auto":
                # Auto-detect: prefer CUDA if available
                if torch.cuda.is_available():
                    self._device = "cuda"
                    logger.info("Auto-detected CUDA, using GPU")
                else:
                    self._device = "cpu"
                    logger.info("Auto-detected: CUDA not available, using CPU")
            elif device_config == "cuda":
                if torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    self._device = "cpu"
                    logger.warning(
                        "CUDA requested but not available, falling back to CPU"
                    )
            else:
                self._device = "cpu"
                if device_config != "cpu":
                    logger.warning(f"Unknown device '{device_config}', using CPU")

            # Load model with sentence_transformers CrossEncoder
            # This uses the same backend as rag_reranker module
            # v4.2.9: FP16 for reduced VRAM (~50% less memory)
            model_kwargs = {}
            if self._device == "cuda":
                model_kwargs["torch_dtype"] = "float16"
                logger.info(f"[RERANKER] Using FP16 precision for {self.config.model}")

            self._model = CrossEncoder(
                self.config.model,
                device=self._device,
                max_length=self.config.max_length,
                model_kwargs=model_kwargs,
            )

            elapsed = (time.perf_counter() - start) * 1000
            precision = "FP16" if model_kwargs.get("torch_dtype") == "float16" else "FP32"
            logger.info(f"[RERANKER] CrossEncoder loaded in {elapsed:.0f}ms on {self._device.upper()} ({precision})")

            self._is_loaded = True

        except ImportError as e:
            raise ModelLoadError(
                f"Required packages not installed: {e}. "
                "Install with: pip install sentence-transformers torch"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load reranker model: {e}")

    def unload_model(self) -> None:
        """Release model reference. SharedModelPool owns actual lifecycle."""
        self._model = None
        self._is_loaded = False
        logger.info("Reranker (CrossEncoder) model reference released")

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> RerankerResult:
        """
        Rerank chunks by relevance to query using sentence_transformers CrossEncoder.

        Args:
            query: Search query
            chunks: List of chunk dicts with 'text' field
            top_k: Number of top chunks to return (None = all)

        Returns:
            RerankerResult with reranked chunks
        """
        self._ensure_model_loaded()

        if not chunks:
            return RerankerResult(
                reranked_chunks=[],
                model_used=self.config.model,
                time_ms=0,
            )

        start = time.perf_counter()

        # Prepare pairs for scoring - CrossEncoder expects list of [query, doc] pairs
        pairs = [[query, chunk.get("text", "")] for chunk in chunks]

        # Score using CrossEncoder.predict - handles batching internally
        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )

            # Normalize if configured (CrossEncoder returns raw logits)
            if self.config.normalize_scores:
                import torch
                scores = torch.sigmoid(torch.tensor(scores)).tolist()
            else:
                scores = scores.tolist() if hasattr(scores, 'tolist') else list(scores)

        except Exception as e:
            logger.error(f"CrossEncoder prediction failed: {e}")
            # Return chunks unchanged on error
            scores = [chunk.get("score", 0.0) for chunk in chunks]

        all_scores: List[float] = scores

        # Create enriched chunks with rerank scores
        enriched: List[EnrichedChunk] = []
        for chunk, score in zip(chunks, all_scores):
            enriched.append(
                EnrichedChunk(
                    kb_id=chunk.get("kb_id", "") or chunk.get("collection", ""),
                    chunk_id=chunk.get("chunk_id", "") or chunk.get("id", "") or chunk.get("doc_id", ""),
                    text=chunk.get("text", ""),
                    score=chunk.get("score", 0.0),
                    rerank_score=float(score),
                    metadata=chunk.get("metadata", {}),
                    embedding=chunk.get("embedding") or chunk.get("vector"),
                )
            )

        # Sort by rerank score descending
        enriched.sort(key=lambda x: x.rerank_score or 0, reverse=True)

        # Apply top_k
        if top_k is not None and top_k < len(enriched):
            enriched = enriched[:top_k]

        elapsed = (time.perf_counter() - start) * 1000

        return RerankerResult(
            reranked_chunks=enriched,
            model_used=self.config.model,
            time_ms=elapsed,
        )

    def health_check(self) -> Dict[str, Any]:
        """Check reranker health."""
        try:
            self._ensure_model_loaded()

            # Quick test rerank
            start = time.perf_counter()
            _ = self.rerank("test query", [{"text": "test document"}])
            latency = (time.perf_counter() - start) * 1000

            return {
                "status": "healthy",
                "model": self.config.model,
                "device": str(self._device),
                "latency_ms": round(latency, 2),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.config.model,
                "error": str(e),
            }

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


# ============================================================================
# ContextCompressor
# ============================================================================


class ContextCompressor:
    """
    Compress chunks to reduce token count.

    Methods:
    - Extractive: Keep most relevant sentences
    - Abstractive: Use LLM to summarize (requires delegation)
    - Hybrid: Combine both approaches
    """

    def __init__(self, config: CompressionConfig):
        self.config = config

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation."""
        return len(text) // 4

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentence_relevance(
        self,
        sentence: str,
        query: str,
        full_text: str,
    ) -> float:
        """Score sentence relevance to query."""
        # Simple keyword overlap scoring
        query_words = set(query.lower().split())
        sentence_words = set(sentence.lower().split())

        if not query_words:
            return 0.5

        overlap = len(query_words & sentence_words)
        keyword_score = overlap / len(query_words)

        # Position bonus (earlier sentences often more important)
        position = full_text.find(sentence)
        position_score = 1.0 - (position / max(len(full_text), 1)) * 0.3

        # Length bonus (avoid very short sentences)
        length_score = min(len(sentence) / 100, 1.0)

        return keyword_score * 0.6 + position_score * 0.2 + length_score * 0.2

    def compress_extractive(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        target_ratio: float,
    ) -> CompressionResult:
        """Extractive compression - keep most relevant sentences."""
        start = time.perf_counter()

        original_tokens = sum(self._estimate_tokens(c.get("text", "")) for c in chunks)
        target_tokens = int(original_tokens * target_ratio)

        compressed_chunks: List[EnrichedChunk] = []
        current_tokens = 0

        for chunk in chunks:
            text = chunk.get("text", "")
            sentences = self._split_sentences(text)

            if not sentences:
                continue

            # Score and sort sentences
            scored = [
                (s, self._score_sentence_relevance(s, query, text)) for s in sentences
            ]
            scored.sort(key=lambda x: x[1], reverse=True)

            # Select sentences up to target ratio for this chunk
            chunk_target = int(self._estimate_tokens(text) * target_ratio)
            selected: List[str] = []
            chunk_tokens = 0

            for sentence, score in scored:
                sent_tokens = self._estimate_tokens(sentence)
                if chunk_tokens + sent_tokens <= chunk_target or not selected:
                    selected.append(sentence)
                    chunk_tokens += sent_tokens

            # Reorder by original position
            if self.config.preserve_sentences:
                original_order = {s: i for i, s in enumerate(sentences)}
                selected.sort(key=lambda s: original_order.get(s, 0))

            compressed_text = " ".join(selected)

            if len(compressed_text) >= self.config.min_chunk_length:
                compressed_chunks.append(
                    EnrichedChunk(
                        kb_id=chunk.get("kb_id", ""),
                        chunk_id=chunk.get("chunk_id", ""),
                        text=compressed_text,
                        score=chunk.get("score", 0.0),
                        rerank_score=chunk.get("rerank_score"),
                        metadata=chunk.get("metadata", {}),
                        enrichment_metadata={"compression": "extractive"},
                        embedding=chunk.get("embedding") or chunk.get("vector"),
                    )
                )
                current_tokens += self._estimate_tokens(compressed_text)

        elapsed = (time.perf_counter() - start) * 1000
        actual_ratio = current_tokens / original_tokens if original_tokens > 0 else 1.0

        return CompressionResult(
            compressed_chunks=compressed_chunks,
            original_tokens=original_tokens,
            compressed_tokens=current_tokens,
            actual_ratio=actual_ratio,
            method_used="extractive",
            time_ms=elapsed,
        )

    def compress(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        target_ratio: Optional[float] = None,
        method: Optional[str] = None,
    ) -> CompressionResult:
        """
        Compress chunks using configured method.

        For abstractive/hybrid, requires external LLM callback.
        """
        ratio = target_ratio or self.config.default_ratio
        method = method or self.config.method

        if method == "extractive":
            return self.compress_extractive(query, chunks, ratio)
        else:
            # For abstractive/hybrid, fall back to extractive
            # Actual abstractive requires LLM delegation from adapter
            logger.warning(
                f"Method '{method}' requires LLM, falling back to extractive"
            )
            return self.compress_extractive(query, chunks, ratio)


# ============================================================================
# ChunkFusion
# ============================================================================


class ChunkFusion:
    """
    Merge overlapping or semantically similar chunks.

    Strategies:
    - Overlap: Merge chunks with text overlap
    - Adjacent: Merge consecutive chunks from same document
    - Semantic: Merge semantically similar chunks
    """

    def __init__(self, config: FusionConfig):
        self.config = config

    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """Calculate text overlap ratio (SequenceMatcher fallback)."""
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()

    @staticmethod
    def _build_sim_matrix(chunks: List[EnrichedChunk]):
        """Pre-compute cosine similarity matrix from chunk embeddings.

        v6.1.3: BLAS-optimized — ~1ms for 20 chunks of 384-dim vectors.
        Returns None if any chunk lacks an embedding.
        """
        if not chunks or not all(c.embedding is not None for c in chunks):
            return None
        import numpy as np
        embeddings = np.array([c.embedding for c in chunks], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embeddings / norms
        return normed @ normed.T

    def _merge_texts(self, text1: str, text2: str) -> str:
        """Merge two overlapping texts (SequenceMatcher fallback, O(n*m))."""
        matcher = SequenceMatcher(None, text1, text2)
        blocks = matcher.get_matching_blocks()

        if not blocks:
            return f"{text1}\n\n{text2}"

        for block in blocks:
            if block.a + block.size == len(text1) and block.b == 0:
                return text1 + text2[block.size :]

        return f"{text1}\n\n{text2}"

    @staticmethod
    def _merge_texts_fast(text1: str, text2: str) -> str:
        """Suffix-prefix word overlap merge, O(n) instead of O(n*m).

        Used when overlap detection is embedding-based (cosine similarity).
        Finds the longest suffix of text1 that matches a prefix of text2
        at word boundaries, then merges without duplication.
        """
        words1 = text1.split()
        words2 = text2.split()

        max_overlap = min(len(words1), len(words2), 50)
        for k in range(max_overlap, 0, -1):
            if words1[-k:] == words2[:k]:
                return " ".join(words1 + words2[k:])

        return text1 + "\n\n" + text2

    # Maximum fusions per fuse_by_overlap call (safety cap)
    _MAX_FUSIONS = 30

    def fuse_by_overlap(
        self,
        chunks: List[EnrichedChunk],
    ) -> Tuple[List[EnrichedChunk], List[Dict[str, Any]]]:
        """Merge chunks with significant text overlap.

        v6.1.4: Three-level perf fix:
        1. _merge_texts_fast() O(n) for embedding path (no SequenceMatcher)
        2. semantic_threshold raised to 0.93 (was 0.85)
        3. _MAX_FUSIONS cap prevents runaway merge loops
        """
        if len(chunks) < 2:
            return chunks, []

        _t_start = time.perf_counter()
        sim_matrix = self._build_sim_matrix(chunks)
        use_embeddings = sim_matrix is not None
        method = "embedding" if use_embeddings else "SequenceMatcher"
        threshold = self.config.semantic_threshold if use_embeddings else self.config.overlap_threshold

        if not use_embeddings:
            logger.warning(f"[FUSION] SequenceMatcher fallback — no embeddings ({len(chunks)} chunks, threshold={threshold})")
        else:
            logger.info(f"[FUSION] Cosine matrix ({len(chunks)} chunks, threshold={threshold})")

        fused = list(chunks)
        fusions: List[Dict[str, Any]] = []
        orig_indices = list(range(len(chunks)))
        merge_fn = self._merge_texts_fast if use_embeddings else self._merge_texts
        _fusion_count = 0
        _cap_hit = False

        i = 0
        while i < len(fused) - 1:
            j = i + 1
            while j < len(fused):
                if use_embeddings:
                    overlap = float(sim_matrix[orig_indices[i]][orig_indices[j]])
                else:
                    overlap = self._calculate_overlap(fused[i].text, fused[j].text)

                if overlap >= threshold:
                    if _fusion_count >= self._MAX_FUSIONS:
                        _cap_hit = True
                        break

                    # Skip merge if combined text would clearly exceed limit
                    if len(fused[i].text) + len(fused[j].text) > self.config.max_fused_length:
                        j += 1
                        continue

                    merged_text = merge_fn(fused[i].text, fused[j].text)

                    if len(merged_text) <= self.config.max_fused_length:
                        fused[i] = EnrichedChunk(
                            kb_id=fused[i].kb_id,
                            chunk_id=f"{fused[i].chunk_id}+{fused[j].chunk_id}",
                            text=merged_text,
                            score=max(fused[i].score, fused[j].score),
                            rerank_score=max(
                                fused[i].rerank_score or 0,
                                fused[j].rerank_score or 0,
                            )
                            if fused[i].rerank_score or fused[j].rerank_score
                            else None,
                            metadata={**fused[i].metadata, **fused[j].metadata},
                            enrichment_metadata={"fusion": "overlap"},
                            embedding=fused[i].embedding if fused[i].score >= fused[j].score else fused[j].embedding,
                        )

                        fusions.append(
                            {
                                "type": "overlap",
                                "merged_ids": [fused[i].chunk_id, fused[j].chunk_id],
                                "overlap_ratio": overlap,
                            }
                        )

                        fused.pop(j)
                        orig_indices.pop(j)
                        _fusion_count += 1
                        continue
                j += 1
            if _cap_hit:
                break
            i += 1

        _elapsed_ms = (time.perf_counter() - _t_start) * 1000
        logger.info(
            f"[FUSION] {method} {len(chunks)}→{len(fused)} chunks, "
            f"{_fusion_count} merges, {_elapsed_ms:.1f}ms"
            f"{' (CAP_HIT)' if _cap_hit else ''}"
        )

        return fused, fusions

    def fuse_by_adjacency(
        self,
        chunks: List[EnrichedChunk],
    ) -> Tuple[List[EnrichedChunk], List[Dict[str, Any]]]:
        """Merge consecutive chunks from same document."""
        if len(chunks) < 2:
            return chunks, []

        fused: List[EnrichedChunk] = []
        fusions: List[Dict[str, Any]] = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            # Check if from same KB and adjacent
            same_kb = current.kb_id == next_chunk.kb_id

            # Simple adjacency check based on chunk_id pattern
            # Assumes chunk_id contains sequence info
            adjacent = False
            if same_kb:
                try:
                    # Try to extract sequence numbers
                    current_num = int(re.search(r"\d+", current.chunk_id).group())
                    next_num = int(re.search(r"\d+", next_chunk.chunk_id).group())
                    adjacent = abs(next_num - current_num) == 1
                except (AttributeError, ValueError):
                    pass

            if same_kb and adjacent:
                merged_text = f"{current.text}\n\n{next_chunk.text}"

                if len(merged_text) <= self.config.max_fused_length:
                    current = EnrichedChunk(
                        kb_id=current.kb_id,
                        chunk_id=f"{current.chunk_id}+{next_chunk.chunk_id}",
                        text=merged_text,
                        score=max(current.score, next_chunk.score),
                        rerank_score=max(
                            current.rerank_score or 0,
                            next_chunk.rerank_score or 0,
                        )
                        if current.rerank_score or next_chunk.rerank_score
                        else None,
                        metadata={**current.metadata, **next_chunk.metadata},
                        enrichment_metadata={"fusion": "adjacent"},
                        embedding=current.embedding if current.score >= next_chunk.score else next_chunk.embedding,
                    )

                    fusions.append(
                        {
                            "type": "adjacent",
                            "merged_ids": [current.chunk_id, next_chunk.chunk_id],
                        }
                    )
                    continue

            fused.append(current)
            current = next_chunk

        fused.append(current)
        return fused, fusions

    def fuse(
        self,
        chunks: List[Dict[str, Any]],
        strategies: Optional[List[str]] = None,
    ) -> FusionResult:
        """Apply fusion strategies to chunks."""
        start = time.perf_counter()
        strategies = strategies or self.config.strategies

        # Convert to EnrichedChunk
        enriched = [EnrichedChunk.from_dict(c) for c in chunks]
        chunks_before = len(enriched)
        all_fusions: List[Dict[str, Any]] = []

        if "overlap" in strategies:
            enriched, fusions = self.fuse_by_overlap(enriched)
            all_fusions.extend(fusions)

        if "adjacent" in strategies:
            enriched, fusions = self.fuse_by_adjacency(enriched)
            all_fusions.extend(fusions)

        # Semantic fusion would require embeddings - skip for now

        elapsed = (time.perf_counter() - start) * 1000

        return FusionResult(
            fused_chunks=enriched,
            chunks_before=chunks_before,
            chunks_after=len(enriched),
            fusions_applied=all_fusions,
            time_ms=elapsed,
        )


# ============================================================================
# Deduplicator
# ============================================================================


class Deduplicator:
    """Remove duplicate or near-duplicate chunks."""

    def __init__(self, config: DeduplicationConfig):
        self.config = config

    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _fuzzy_similarity(self, text1: str, text2: str) -> float:
        """Calculate fuzzy similarity."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def deduplicate(
        self,
        chunks: List[Dict[str, Any]],
        method: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[List[EnrichedChunk], int, List[List[str]]]:
        """
        Remove duplicates from chunks.

        Returns:
            Tuple of (unique_chunks, duplicates_removed, duplicate_groups)
        """
        method = method or self.config.method
        threshold = threshold or self.config.similarity_threshold

        enriched = [EnrichedChunk.from_dict(c) for c in chunks]

        if method == "hash":
            return self._dedupe_by_hash(enriched)
        elif method == "fuzzy":
            return self._dedupe_by_fuzzy(enriched, threshold)
        else:  # semantic
            # v6.1.3: Use embedding-based dedup if vectors available
            has_embeddings = enriched and all(c.embedding is not None for c in enriched)
            if has_embeddings:
                logger.info(f"[DEDUP] Using embedding-based dedup ({len(enriched)} chunks, threshold={threshold})")
                return self._dedupe_by_embedding(enriched, threshold)
            logger.info(f"[DEDUP] Falling back to fuzzy dedup (no embeddings, {len(enriched)} chunks)")
            return self._dedupe_by_fuzzy(enriched, threshold)  # fallback

    def _dedupe_by_hash(
        self,
        chunks: List[EnrichedChunk],
    ) -> Tuple[List[EnrichedChunk], int, List[List[str]]]:
        """Deduplicate by exact hash."""
        seen_hashes: Dict[str, EnrichedChunk] = {}
        duplicate_groups: List[List[str]] = []

        for chunk in chunks:
            h = self._hash_text(chunk.text)
            if h in seen_hashes:
                # Find or create duplicate group
                found = False
                for group in duplicate_groups:
                    if seen_hashes[h].chunk_id in group:
                        group.append(chunk.chunk_id)
                        found = True
                        break
                if not found:
                    duplicate_groups.append([seen_hashes[h].chunk_id, chunk.chunk_id])
            else:
                seen_hashes[h] = chunk

        unique = list(seen_hashes.values())
        removed = len(chunks) - len(unique)

        return unique, removed, duplicate_groups

    def _dedupe_by_fuzzy(
        self,
        chunks: List[EnrichedChunk],
        threshold: float,
    ) -> Tuple[List[EnrichedChunk], int, List[List[str]]]:
        """Deduplicate by fuzzy similarity."""
        unique: List[EnrichedChunk] = []
        duplicate_groups: List[List[str]] = []

        for chunk in chunks:
            is_duplicate = False
            for existing in unique:
                similarity = self._fuzzy_similarity(chunk.text, existing.text)
                if similarity >= threshold:
                    # Add to duplicate group
                    found = False
                    for group in duplicate_groups:
                        if existing.chunk_id in group:
                            group.append(chunk.chunk_id)
                            found = True
                            break
                    if not found:
                        duplicate_groups.append([existing.chunk_id, chunk.chunk_id])
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(chunk)

        removed = len(chunks) - len(unique)
        return unique, removed, duplicate_groups

    def _dedupe_by_embedding(
        self,
        chunks: List[EnrichedChunk],
        threshold: float,
    ) -> Tuple[List[EnrichedChunk], int, List[List[str]]]:
        """Embedding-based dedup via cosine similarity matrix.

        v6.1.3: Uses pre-computed vectors from Qdrant (384-dim).
        BLAS-optimized matrix multiplication — ~1ms for 20 chunks.

        NOTE: threshold for cosine similarity on sentence embeddings differs from
        SequenceMatcher ratio. Recommended: 0.97-0.98 for near-duplicates,
        0.95 for tight paraphrases. Do NOT reuse fuzzy thresholds directly.
        Clamps to min 0.97 to prevent over-dedup when inheriting fuzzy config.
        """
        import numpy as np

        # Cosine on sentence embeddings needs higher threshold than SequenceMatcher
        threshold = max(threshold, 0.97)

        embeddings = np.array([c.embedding for c in chunks], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div by zero
        normed = embeddings / norms
        sim_matrix = normed @ normed.T

        to_remove: set = set()
        duplicate_groups: List[List[str]] = []
        for i in range(len(chunks)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(chunks)):
                if j in to_remove:
                    continue
                if sim_matrix[i][j] > threshold:
                    # Keep the chunk with higher RAG retrieval score
                    victim = j if chunks[i].score >= chunks[j].score else i
                    to_remove.add(victim)
                    duplicate_groups.append([chunks[i].chunk_id, chunks[j].chunk_id])

        unique = [c for idx, c in enumerate(chunks) if idx not in to_remove]
        return unique, len(to_remove), duplicate_groups


# ============================================================================
# MetadataInjector
# ============================================================================


class MetadataInjector:
    """Add enrichment metadata to chunks."""

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation."""
        return len(text) // 4

    def inject(
        self,
        chunks: List[Dict[str, Any]],
        metadata_types: List[str],
    ) -> List[EnrichedChunk]:
        """
        Inject metadata into chunks.

        Types:
        - source: Add source info
        - relevance: Add relevance scores
        - timestamp: Add processing timestamp
        - position: Add position index
        - tokens: Add token count
        """
        from datetime import datetime, timezone

        enriched: List[EnrichedChunk] = []
        now = datetime.now(timezone.utc).isoformat()

        for i, chunk in enumerate(chunks):
            ec = EnrichedChunk.from_dict(chunk)

            if "source" in metadata_types:
                ec.enrichment_metadata["source"] = {
                    "kb_id": ec.kb_id,
                    "chunk_id": ec.chunk_id,
                    "original_metadata": ec.metadata,
                }

            if "relevance" in metadata_types:
                ec.enrichment_metadata["relevance"] = {
                    "original_score": ec.score,
                    "rerank_score": ec.rerank_score,
                }

            if "timestamp" in metadata_types:
                ec.enrichment_metadata["processed_at"] = now

            if "position" in metadata_types:
                ec.enrichment_metadata["position"] = {
                    "index": i,
                    "total": len(chunks),
                }

            if "tokens" in metadata_types:
                ec.enrichment_metadata["tokens"] = self._estimate_tokens(ec.text)

            enriched.append(ec)

        return enriched


# ============================================================================
# RedisCacheProvider - Environment-Aware Caching
# ============================================================================


class RedisCacheProvider:
    """
    Redis cache for enrichment pipeline with environment isolation.

    Key format: ubp:{env}:enrichment:cache:{operation}:{hash}

    This ensures test/dev/prod environments don't share cache data.
    """

    def __init__(self, config: CacheConfig, redis_client: Optional[Any] = None):
        self.config = config
        self._redis = redis_client
        self._stats = {
            "hits": 0,
            "misses": 0,
        }

    def _generate_key(self, operation: str, *args) -> str:
        """
        Generate a cache key with environment isolation.

        Format: ubp:{env}:enrichment:cache:{operation}:{hash}
        """
        # Create hash from operation arguments
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
        """
        Invalidate cache entries.

        If pattern is None, invalidates all entries for current environment.
        """
        if not self._redis:
            return 0

        try:
            if pattern:
                full_pattern = f"{self.config.prefix}:{pattern}"
            else:
                full_pattern = f"{self.config.prefix}:*"

            # Scan and delete matching keys
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
        """
        Clear all cache entries for the current environment.

        Returns:
            Number of entries cleared.
        """
        cleared = await self.invalidate()
        self.reset_stats()
        return cleared
