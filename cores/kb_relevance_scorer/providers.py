"""
kb_relevance_scorer/providers.py — Business logic wrapper around shared engine.

Adds Qdrant integration and config expansion for the module adapter.
Pure logic — no FastAPI, no HTTP.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("ubp.kb_relevance_scorer")

# Import shared engine (container-flat first, host-namespaced fallback)
try:
    from shared.kb_relevance import (
        DEFAULT_CONFIG,
        KBRelevanceResult,
        compute_kb_relevance,
        classify_query_type,
    )
except ModuleNotFoundError:
    from ubp_enterprise_hybrid.shared.kb_relevance import (
        DEFAULT_CONFIG,
        KBRelevanceResult,
        compute_kb_relevance,
        classify_query_type,
    )


def expand_config(raw_config: dict) -> dict[str, Any]:
    """Expand module config.json into shared engine config format.

    Handles the nested 'weights' dict and maps to flat w_* keys.
    """
    cfg = dict(DEFAULT_CONFIG)

    # Direct keys
    for key in [
        "abs_min", "avg_min", "mid_floor", "hi_floor",
        "delta12_normalizer", "std_normalizer", "max_docs_normalizer",
        "rel_threshold_default", "rel_threshold_general_knowledge",
        "rel_threshold_doc_seeking", "short_query_min_tokens",
    ]:
        if key in raw_config:
            val = raw_config[key]
            if isinstance(val, str):
                try:
                    val = float(val)
                except ValueError:
                    continue
            cfg[key] = val

    # Nested weights
    weights = raw_config.get("weights", {})
    if isinstance(weights, dict):
        for wkey in ["strength", "support", "separation", "coverage", "dispersion"]:
            if wkey in weights:
                cfg[f"w_{wkey}"] = float(weights[wkey])

    return cfg


def score_results(
    search_results: list[dict],
    query: str,
    query_type: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score KB relevance from pre-fetched search results.

    Returns a flat dict suitable for pipeline output.
    """
    result = compute_kb_relevance(
        search_results=search_results,
        query_text=query,
        query_type=query_type,
        config=config,
    )

    logger.info(
        "[KB-RELEVANCE] top1=%.3f avg=%.3f delta12=%.3f count_mid=%d "
        "rel_score=%.3f thr=%.3f gate=%s qtype=%s kb_relevant=%s",
        result.features.top1_score,
        result.features.avg_score,
        result.features.delta12,
        result.features.count_mid,
        result.rel_score,
        result.threshold_used,
        result.gate_reason or "none",
        result.query_type,
        result.kb_relevant,
    )

    return result.to_dict()


async def score_from_qdrant_provider(
    query: str,
    collection: str,
    top_k: int,
    rag_qdrant: Any,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score KB relevance by querying Qdrant directly.

    Self-contained operation — fetches results from Qdrant then scores.
    """
    if rag_qdrant is None:
        logger.warning("[KB-RELEVANCE] rag_qdrant not available, returning irrelevant")
        return compute_kb_relevance([], query).to_dict()

    try:
        result = await rag_qdrant.query_internal(
            query_text=query,
            top_k=top_k,
            collection=collection,
        )
        chunks = result.get("results", []) if result else []
    except Exception as e:
        logger.error("[KB-RELEVANCE] Qdrant query failed: %s", e)
        chunks = []

    return score_results(chunks, query, config=config)
