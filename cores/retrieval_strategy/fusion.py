"""
retrieval_strategy/fusion.py

Score fusion algorithms for hybrid retrieval.

Implements:
- Reciprocal Rank Fusion (RRF)
- Weighted Score Fusion
- Max Fusion
- Sum Fusion
- Distribution-Based Score Fusion (DBSF)

v1.0.0: Initial release
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .providers import (
    RetrievalResult,
    FusionMethod,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Fusion Configuration
# ============================================================================


@dataclass
class RRFConfig:
    """Reciprocal Rank Fusion configuration."""
    k: int = 60  # RRF constant
    weight_bm25: float = 1.0
    weight_vector: float = 1.0


@dataclass
class WeightedConfig:
    """Weighted fusion configuration."""
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    normalize_method: str = "minmax"  # minmax, zscore, none
    combination: str = "sum"  # sum, product


@dataclass
class DBSFConfig:
    """Distribution-Based Score Fusion configuration."""
    min_score: float = 0.0
    max_score: float = 1.0


# ============================================================================
# Score Normalizers
# ============================================================================


def normalize_minmax(
    scores: List[Tuple[str, float]],
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> List[Tuple[str, float]]:
    """Normalize scores to [target_min, target_max] range using min-max scaling."""
    if not scores:
        return []
    
    values = [s for _, s in scores]
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        # All scores are the same
        mid = (target_min + target_max) / 2
        return [(doc_id, mid) for doc_id, _ in scores]
    
    scale = (target_max - target_min) / (max_val - min_val)
    
    return [
        (doc_id, target_min + (score - min_val) * scale)
        for doc_id, score in scores
    ]


def normalize_zscore(scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Normalize scores using z-score normalization."""
    if not scores:
        return []
    
    values = [s for _, s in scores]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance) if variance > 0 else 1.0
    
    return [
        (doc_id, (score - mean) / std)
        for doc_id, score in scores
    ]


def normalize_scores(
    scores: List[Tuple[str, float]],
    method: str = "minmax",
) -> List[Tuple[str, float]]:
    """Normalize scores using specified method."""
    if method == "minmax":
        return normalize_minmax(scores)
    elif method == "zscore":
        return normalize_zscore(scores)
    else:
        return scores


# ============================================================================
# Reciprocal Rank Fusion (RRF)
# ============================================================================


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[str, float]]],
    weights: Optional[List[float]] = None,
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion.
    
    RRF score = sum(weight_i / (k + rank_i))
    
    Args:
        result_lists: List of ranked results [(doc_id, score), ...]
        weights: Optional weights for each result list
        k: RRF constant (default 60)
    
    Returns:
        Fused results sorted by RRF score
    """
    if not result_lists:
        return []
    
    if weights is None:
        weights = [1.0] * len(result_lists)
    
    # Calculate RRF scores
    rrf_scores: Dict[str, float] = defaultdict(float)
    
    for list_idx, results in enumerate(result_lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0
        
        for rank, (doc_id, _) in enumerate(results, start=1):
            rrf_scores[doc_id] += weight / (k + rank)
    
    # Sort by RRF score
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    
    return sorted_results


def rrf_hybrid(
    bm25_results: List[Tuple[str, float]],
    vector_results: List[Tuple[str, float]],
    config: RRFConfig,
) -> List[Tuple[str, float]]:
    """
    RRF fusion for hybrid (BM25 + Vector) retrieval.
    
    Args:
        bm25_results: BM25 retrieval results
        vector_results: Vector retrieval results
        config: RRF configuration
    
    Returns:
        Fused results
    """
    return reciprocal_rank_fusion(
        result_lists=[bm25_results, vector_results],
        weights=[config.weight_bm25, config.weight_vector],
        k=config.k,
    )


# ============================================================================
# Weighted Score Fusion
# ============================================================================


def weighted_fusion(
    result_lists: List[List[Tuple[str, float]]],
    weights: List[float],
    normalize_method: str = "minmax",
    combination: str = "sum",
) -> List[Tuple[str, float]]:
    """
    Weighted score combination.
    
    Args:
        result_lists: List of scored results
        weights: Weights for each list (should sum to 1.0)
        normalize_method: Score normalization method
        combination: How to combine (sum, product)
    
    Returns:
        Fused results
    """
    if not result_lists:
        return []
    
    # Normalize scores in each list
    normalized_lists = [
        normalize_scores(results, normalize_method)
        for results in result_lists
    ]
    
    # Combine scores
    combined_scores: Dict[str, float] = defaultdict(float)
    doc_presence: Dict[str, int] = defaultdict(int)
    
    for list_idx, results in enumerate(normalized_lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0 / len(result_lists)
        
        for doc_id, score in results:
            if combination == "sum":
                combined_scores[doc_id] += weight * score
            elif combination == "product":
                if doc_presence[doc_id] == 0:
                    combined_scores[doc_id] = weight * score
                else:
                    combined_scores[doc_id] *= weight * score
            
            doc_presence[doc_id] += 1
    
    # Sort by combined score
    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    
    return sorted_results


def weighted_hybrid(
    bm25_results: List[Tuple[str, float]],
    vector_results: List[Tuple[str, float]],
    config: WeightedConfig,
) -> List[Tuple[str, float]]:
    """
    Weighted fusion for hybrid retrieval.
    """
    return weighted_fusion(
        result_lists=[bm25_results, vector_results],
        weights=[config.bm25_weight, config.vector_weight],
        normalize_method=config.normalize_method,
        combination=config.combination,
    )


# ============================================================================
# Max Fusion
# ============================================================================


def max_fusion(
    result_lists: List[List[Tuple[str, float]]],
    normalize: bool = True,
) -> List[Tuple[str, float]]:
    """
    Max fusion - take maximum score for each document.
    
    Args:
        result_lists: List of scored results
        normalize: Whether to normalize scores first
    
    Returns:
        Fused results with max scores
    """
    if not result_lists:
        return []
    
    # Normalize if requested
    if normalize:
        result_lists = [normalize_minmax(results) for results in result_lists]
    
    # Take max score for each doc
    max_scores: Dict[str, float] = {}
    
    for results in result_lists:
        for doc_id, score in results:
            if doc_id not in max_scores or score > max_scores[doc_id]:
                max_scores[doc_id] = score
    
    # Sort by score
    sorted_results = sorted(
        max_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    
    return sorted_results


# ============================================================================
# Sum Fusion
# ============================================================================


def sum_fusion(
    result_lists: List[List[Tuple[str, float]]],
    normalize: bool = True,
) -> List[Tuple[str, float]]:
    """
    Sum fusion - sum scores for each document.
    
    Args:
        result_lists: List of scored results
        normalize: Whether to normalize scores first
    
    Returns:
        Fused results with summed scores
    """
    if not result_lists:
        return []
    
    if normalize:
        result_lists = [normalize_minmax(results) for results in result_lists]
    
    sum_scores: Dict[str, float] = defaultdict(float)
    
    for results in result_lists:
        for doc_id, score in results:
            sum_scores[doc_id] += score
    
    sorted_results = sorted(
        sum_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    
    return sorted_results


# ============================================================================
# Distribution-Based Score Fusion (DBSF)
# ============================================================================


def dbsf_normalize(
    scores: List[Tuple[str, float]],
    global_min: float = 0.0,
    global_max: float = 1.0,
) -> List[Tuple[str, float]]:
    """
    Distribution-Based Score Fusion normalization.
    
    Maps scores to a common distribution range.
    """
    if not scores:
        return []
    
    values = [s for _, s in scores]
    
    # Calculate distribution statistics
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance) if variance > 0 else 1.0
    
    # Normalize using 3-sigma rule
    normalized = []
    for doc_id, score in scores:
        # Map to approximately [0, 1] range
        z = (score - mean) / std if std > 0 else 0
        # Use sigmoid-like transformation
        norm_score = 1 / (1 + math.exp(-z))
        # Scale to target range
        final_score = global_min + (global_max - global_min) * norm_score
        normalized.append((doc_id, final_score))
    
    return normalized


def dbsf_fusion(
    result_lists: List[List[Tuple[str, float]]],
    config: DBSFConfig,
) -> List[Tuple[str, float]]:
    """
    Distribution-Based Score Fusion.
    
    Normalizes scores based on their distributions before combining.
    """
    if not result_lists:
        return []
    
    # Normalize each list
    normalized_lists = [
        dbsf_normalize(results, config.min_score, config.max_score)
        for results in result_lists
    ]
    
    # Combine (using sum)
    combined_scores: Dict[str, float] = defaultdict(float)
    
    for results in normalized_lists:
        for doc_id, score in results:
            combined_scores[doc_id] += score
    
    # Normalize final scores
    if combined_scores:
        max_combined = max(combined_scores.values())
        if max_combined > 0:
            combined_scores = {
                doc_id: score / max_combined
                for doc_id, score in combined_scores.items()
            }
    
    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    
    return sorted_results


# ============================================================================
# Alpha Blending
# ============================================================================


def alpha_blend(
    bm25_results: List[Tuple[str, float]],
    vector_results: List[Tuple[str, float]],
    alpha: float = 0.5,
    normalize: bool = True,
) -> List[Tuple[str, float]]:
    """
    Simple alpha blending between two result sets.
    
    final_score = alpha * bm25_score + (1 - alpha) * vector_score
    
    Args:
        bm25_results: BM25 results
        vector_results: Vector results
        alpha: Weight for BM25 (0 to 1)
        normalize: Whether to normalize scores
    
    Returns:
        Blended results
    """
    if normalize:
        bm25_results = normalize_minmax(bm25_results)
        vector_results = normalize_minmax(vector_results)
    
    bm25_scores = dict(bm25_results)
    vector_scores = dict(vector_results)
    
    all_docs = set(bm25_scores.keys()) | set(vector_scores.keys())
    
    blended = []
    for doc_id in all_docs:
        bm25_score = bm25_scores.get(doc_id, 0.0)
        vector_score = vector_scores.get(doc_id, 0.0)
        final_score = alpha * bm25_score + (1 - alpha) * vector_score
        blended.append((doc_id, final_score))
    
    return sorted(blended, key=lambda x: x[1], reverse=True)


# ============================================================================
# Fusion Factory
# ============================================================================


class FusionEngine:
    """
    Factory for applying fusion methods.
    """
    
    def __init__(
        self,
        method: FusionMethod = FusionMethod.RRF,
        rrf_config: Optional[RRFConfig] = None,
        weighted_config: Optional[WeightedConfig] = None,
        dbsf_config: Optional[DBSFConfig] = None,
        alpha: float = 0.5,
    ):
        self.method = method
        self.rrf_config = rrf_config or RRFConfig()
        self.weighted_config = weighted_config or WeightedConfig()
        self.dbsf_config = dbsf_config or DBSFConfig()
        self.alpha = alpha
    
    def fuse(
        self,
        bm25_results: List[Tuple[str, float]],
        vector_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Apply fusion to BM25 and vector results.
        """
        if self.method == FusionMethod.RRF:
            return rrf_hybrid(bm25_results, vector_results, self.rrf_config)
        
        elif self.method == FusionMethod.WEIGHTED:
            return weighted_hybrid(bm25_results, vector_results, self.weighted_config)
        
        elif self.method == FusionMethod.MAX:
            return max_fusion([bm25_results, vector_results])
        
        elif self.method == FusionMethod.SUM:
            return sum_fusion([bm25_results, vector_results])
        
        elif self.method == FusionMethod.DBSF:
            return dbsf_fusion([bm25_results, vector_results], self.dbsf_config)
        
        else:
            # Default to alpha blending
            return alpha_blend(bm25_results, vector_results, self.alpha)
    
    def fuse_multi(
        self,
        result_lists: List[List[Tuple[str, float]]],
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Fuse multiple result lists.
        """
        if self.method == FusionMethod.RRF:
            return reciprocal_rank_fusion(result_lists, weights, self.rrf_config.k)
        
        elif self.method == FusionMethod.WEIGHTED:
            weights = weights or [1.0 / len(result_lists)] * len(result_lists)
            return weighted_fusion(
                result_lists,
                weights,
                self.weighted_config.normalize_method,
                self.weighted_config.combination,
            )
        
        elif self.method == FusionMethod.MAX:
            return max_fusion(result_lists)
        
        elif self.method == FusionMethod.SUM:
            return sum_fusion(result_lists)
        
        elif self.method == FusionMethod.DBSF:
            return dbsf_fusion(result_lists, self.dbsf_config)
        
        else:
            return sum_fusion(result_lists)


# ============================================================================
# Deduplication
# ============================================================================


def deduplicate_results(
    results: List[Tuple[str, float]],
    content_lookup: Dict[str, str],
    similarity_threshold: float = 0.95,
) -> List[Tuple[str, float]]:
    """
    Remove near-duplicate results based on normalized content hash.
    
    v3.7.2: Replaced O(n²) SequenceMatcher with O(n) hash-based dedup.
    Uses first 300 chars normalized (lower+strip) as fingerprint.
    
    Args:
        results: List of (doc_id, score) tuples
        content_lookup: Map from doc_id to content
        similarity_threshold: Kept for API compat (>=1.0 skips dedup)
    
    Returns:
        Deduplicated results
    """
    if similarity_threshold >= 1.0:
        return results
    
    kept = []
    seen_hashes: set = set()
    
    for doc_id, score in results:
        content = content_lookup.get(doc_id, "")
        fingerprint = content[:300].strip().lower()
        if fingerprint in seen_hashes:
            continue
        seen_hashes.add(fingerprint)
        kept.append((doc_id, score))
    
    return kept


def deduplicate_results_dict_format(
    results: List[Dict[str, Any]],
    similarity_threshold: float = 0.95,
    text_keys: Tuple[str, ...] = ("text", "content"),
) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate dict-format results based on text similarity.

    Uses SequenceMatcher for accurate near-duplicate detection.
    O(n²) over the result set — acceptable since RAG results are typically
    small (≤20 items).

    Args:
        results: List of result dicts with text/content fields
        similarity_threshold: Items with ratio >= threshold are considered
            duplicates and dropped. Pass 1.0 to disable.
        text_keys: Keys to try (in order) for extracting text content

    Returns:
        Deduplicated results preserving original dict structure
    """
    from difflib import SequenceMatcher

    if similarity_threshold >= 1.0 or len(results) <= 1:
        return results

    def _extract_text(r: Dict[str, Any]) -> str:
        for k in text_keys:
            if k in r and r[k]:
                return str(r[k])
        return ""

    kept: List[Dict[str, Any]] = []
    kept_texts: List[str] = []

    for r in results:
        text = _extract_text(r)
        if not text:
            kept.append(r)
            kept_texts.append("")
            continue

        text_norm = text.strip().lower()
        is_dup = False
        for prev in kept_texts:
            if not prev:
                continue
            if SequenceMatcher(None, text_norm, prev).ratio() >= similarity_threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(r)
            kept_texts.append(text_norm)

    return kept
