"""
UBP Framework Bridge for Filter RAG Context Module

Integrates the pure filter_rag_context() function with the UBP module system.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule
from .providers import FilterConfig, FilterResult, filter_rag_context

logger = logging.getLogger(__name__)


class FilterRagContextAdapter(BaseHybridModule):
    """UBP adapter for RAG context filtering."""

    def __init__(self, module_path: Path, **kwargs):
        super().__init__(module_path, **kwargs)
        self._filter_config: Optional[FilterConfig] = None

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

    async def initialize(self) -> None:
        logger.info(f"Initializing {self.manifest.name}")
        cfg = self.config or {}
        self._filter_config = FilterConfig(
            min_chars_hard=cfg.get("min_chars_hard", 30),
            min_score_hard=cfg.get("min_score_hard", 0.06),
            min_score_soft=cfg.get("min_score_soft", 0.13),
            min_chars_soft=cfg.get("min_chars_soft", 80),
            boilerplate_penalty=cfg.get("boilerplate_penalty", 0.25),
            truncation_penalty=cfg.get("truncation_penalty", 0.15),
            keyword_bonus=cfg.get("keyword_bonus", 0.20),
            relevance_floor=cfg.get("relevance_floor", 0.10),
            diversity_cap_per_source=cfg.get("diversity_cap_per_source", 3),
            max_chunks=cfg.get("max_chunks", 10),
            max_total_chars=cfg.get("max_total_chars", 12000),
            min_output_guarantee=cfg.get("min_output_guarantee", 2),
            low_confidence_threshold=cfg.get("low_confidence_threshold", 0.35),
            shadow_mode=cfg.get("shadow_mode", True),
        )
        logger.info(f"✅ {self.manifest.name} initialized (shadow_mode={self._filter_config.shadow_mode})")

    async def shutdown(self) -> None:
        logger.info(f"Shutting down {self.manifest.name}")
        self._filter_config = None

    async def health_check(self, **kwargs) -> Dict[str, Any]:
        return {
            "module": self.manifest.name,
            "version": self.manifest.version,
            "status": "healthy" if self._filter_config else "not_initialized",
            "shadow_mode": self._filter_config.shadow_mode if self._filter_config else None,
        }

    async def filter_context(
        self,
        chunks: List[Dict],
        query: str,
        config_override: Optional[Dict] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Filter RAG context chunks.

        Args:
            chunks: Retrieved chunks with text, score, source_id, chunk_id.
            query: User query.
            config_override: Optional per-request threshold overrides.
            ctx: Security context.

        Returns:
            Dict with kept, dropped, stats, fallback_triggered.
        """
        cfg = self._filter_config or FilterConfig()
        if config_override:
            cfg_dict = {
                "min_chars_hard": cfg.min_chars_hard,
                "min_score_hard": cfg.min_score_hard,
                "min_score_soft": cfg.min_score_soft,
                "min_chars_soft": cfg.min_chars_soft,
                "boilerplate_penalty": cfg.boilerplate_penalty,
                "truncation_penalty": cfg.truncation_penalty,
                "keyword_bonus": cfg.keyword_bonus,
                "relevance_floor": cfg.relevance_floor,
                "diversity_cap_per_source": cfg.diversity_cap_per_source,
                "max_chunks": cfg.max_chunks,
                "max_total_chars": cfg.max_total_chars,
                "min_output_guarantee": cfg.min_output_guarantee,
                "shadow_mode": cfg.shadow_mode,
            }
            cfg_dict.update(config_override)
            cfg = FilterConfig(**cfg_dict)

        result: FilterResult = filter_rag_context(chunks, query, cfg)

        return {
            "kept": [{"chunk_id": v.chunk_id, "action": v.action, "final_score": v.final_score, "reasons": v.reasons} for v in result.kept],
            "dropped": [{"chunk_id": v.chunk_id, "action": v.action, "final_score": v.final_score, "reasons": v.reasons} for v in result.dropped],
            "stats": {
                "input_count": result.stats.input_count,
                "output_count": result.stats.output_count,
                "dropped_by_reason": result.stats.dropped_by_reason,
                "avg_score_kept": round(result.stats.avg_score_kept, 4),
                "avg_score_dropped": round(result.stats.avg_score_dropped, 4),
                "all_low_relevance": result.stats.all_low_relevance,
                "fallback_triggered": result.stats.fallback_triggered,
                "sources_in": result.stats.sources_in,
                "sources_out": result.stats.sources_out,
            },
            "fallback_triggered": result.fallback_triggered,
        }

    async def filter(
        self,
        chunks: List[Dict],
        query: str,
        shadow_mode: Optional[bool] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Pipeline step operation: filter reranked chunks.

        In shadow mode (default): logs filter results but passes
        original chunks through unchanged.
        In active mode: returns only kept chunks.
        """
        cfg = self._filter_config or FilterConfig()

        # Per-step shadow_mode override
        effective_shadow = shadow_mode if shadow_mode is not None else cfg.shadow_mode

        result: FilterResult = filter_rag_context(chunks, query, cfg)

        logger.info(
            "[FILTER-STEP] input=%d kept=%d dropped=%d fallback=%s shadow=%s",
            result.stats.input_count, result.stats.output_count,
            len(result.dropped), result.fallback_triggered, effective_shadow,
        )

        if effective_shadow:
            # Shadow mode: log but pass original chunks through
            return {"filtered_chunks": chunks}

        # Active mode: return only kept chunks (preserve filter order)
        kept_ids = {v.chunk_id for v in result.kept}
        kept_chunks = [c for c in chunks if c.get("chunk_id") in kept_ids]
        id_order = {v.chunk_id: i for i, v in enumerate(result.kept)}
        kept_chunks.sort(key=lambda c: id_order.get(c.get("chunk_id"), 999))
        return {"filtered_chunks": kept_chunks}

    async def boost_entities(
        self,
        chunks: List[Dict],
        query: str,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Pipeline step: entity boost after retrieve, before rerank."""
        from .providers import boost_by_entity

        boosted = boost_by_entity(
            chunks=chunks,
            query=query,
            boost_factor=1.5,
            miss_penalty=0.7,
        )

        logger.info(
            "[ENTITY-BOOST] input=%d query='%s'",
            len(chunks), query[:60],
        )

        return {"boosted_chunks": boosted}

    async def search_kb_light(
        self,
        query: str,
        ctx=None,
        audit_mode: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Lightweight KB check: collection affinity + top-3 score peek.
        ROUTE-MODE-LLM tool for routing decisions.
        When audit_mode=True, runs query but output is for monitoring only
        (routing decisions use consolidated kb_signal from prefetch).
        """
        from .providers import get_collection_affinity

        # Tenant-aware: collections come from the pipeline (rag_orchestrator /
        # context_gate populate allowed_collections based on ACL). No demo
        # defaults — if the tenant has no collections, skip the peek silently.
        tenant_collections = kwargs.get("allowed_collections") or kwargs.get("available_collections")
        if not tenant_collections:
            logger.debug("[SEARCH-KB-LIGHT] no tenant collections — skipping peek")
            return {"collections_matched": [], "has_relevant_content": False, "top_matches": []}

        affinity = get_collection_affinity(
            query=query,
            available_collections=list(tenant_collections),
            user_selected=None,
        )

        top_matches: List[Dict[str, Any]] = []
        try:
            if self.di_container:
                qdrant = await self.di_container.resolve("rag_qdrant")
                if qdrant:
                    # Only query when affinity selected a tenant collection.
                    if not affinity:
                        return {"collections_matched": [], "has_relevant_content": False, "top_matches": []}
                    target_collection = affinity[0]
                    result = await qdrant.query_internal(
                        query_text=query, top_k=3,
                        collection=target_collection,
                    )
                    # QueryResult is a dataclass — access .results attribute
                    results_list = getattr(result, "results", None)
                    if results_list is None and isinstance(result, dict):
                        results_list = result.get("results", [])
                    for r in (results_list or [])[:3]:
                        if hasattr(r, "score"):
                            top_matches.append({
                                "title": (getattr(r, "metadata", {}).get("title") or getattr(r, "metadata", {}).get("filename") or "doc")[:60],
                                "score": round(r.score, 2),
                                "collection": target_collection,
                            })
                        elif isinstance(r, dict):
                            meta = r.get("metadata", {})
                            top_matches.append({
                                "title": (meta.get("title") or meta.get("filename") or "doc")[:60],
                                "score": round(r.get("score", 0), 2),
                                "collection": r.get("collection", target_collection),
                            })
                    logger.info("[SEARCH-KB-LIGHT] collection=%s matches=%d top_score=%.2f",
                                target_collection, len(top_matches),
                                top_matches[0]["score"] if top_matches else 0.0)
        except Exception as e:
            logger.warning("[SEARCH-KB-LIGHT] qdrant peek failed: %s", e)

        result = {
            "collections_matched": affinity,
            "has_relevant_content": bool(top_matches) and top_matches[0].get("score", 0) > 0.3,
            "top_matches": top_matches,
        }

        if audit_mode:
            logger.info(
                "[KB-AUDIT-DIFF] search_kb_light: has_relevant=%s top_score=%.3f collection=%s",
                result.get("has_relevant_content"),
                result["top_matches"][0]["score"] if result.get("top_matches") else 0.0,
                result["top_matches"][0].get("collection", "?") if result.get("top_matches") else "?",
            )

        return result
