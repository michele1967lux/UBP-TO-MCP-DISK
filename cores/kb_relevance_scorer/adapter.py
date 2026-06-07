"""
kb_relevance_scorer/adapter.py — UBP Framework Bridge.

Follows the UBP 3-file pattern:
- adapter.py: security, DI, lifecycle, call_operation dispatch
- providers.py: business logic (wraps shared engine + Qdrant)
- manifest.json: operation declarations

Operations:
  - score: score from pre-fetched search results (pipeline step)
  - score_from_qdrant: self-contained query + score (pipeline step)
  - health_check: module health

v1.0.0: Initial release.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("ubp.kb_relevance_scorer")


def _load_config(module_path: Path) -> dict:
    cfg_path = module_path / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


class KBRelevanceScorerAdapter:
    """Pipeline-insertable KB relevance scorer.

    Can be used as:
    1. Pipeline step via call_operation("score", ...) or call_operation("score_from_qdrant", ...)
    2. Direct import for inline use (user_router prefetch uses shared engine directly)
    """

    def __init__(self, module_path, di_container=None, event_bus=None, **kwargs):
        if isinstance(module_path, str):
            module_path = Path(module_path)
        self.module_path = module_path
        self.di_container = di_container
        self.event_bus = event_bus
        self.raw_config = _load_config(module_path)

        self._rag_qdrant = None
        self._engine_config = None
        self._initialized = False
        self._default_top_k = 5

    async def initialize(self, **kwargs):
        """Initialize module: resolve DI deps, expand config."""
        from .providers import expand_config

        self._engine_config = expand_config(self.raw_config)
        self._default_top_k = self.raw_config.get("default_top_k", 5)

        # Resolve optional rag_qdrant for score_from_qdrant operation
        if self.di_container:
            try:
                self._rag_qdrant = self.di_container.get("rag_qdrant")
            except Exception:
                logger.info("[KB-RELEVANCE] rag_qdrant not available via DI — "
                            "score_from_qdrant will require explicit rag_qdrant param")

        self._initialized = True
        logger.info(
            "[KB-RELEVANCE] Initialized v1.0.0 — thresholds: "
            "abs_min=%.2f avg_min=%.2f REL_THR=%.2f/%.2f/%.2f top_k=%d",
            self._engine_config.get("abs_min", 0.13),
            self._engine_config.get("avg_min", 0.11),
            self._engine_config.get("rel_threshold_default", 0.62),
            self._engine_config.get("rel_threshold_doc_seeking", 0.58),
            self._engine_config.get("rel_threshold_general_knowledge", 0.72),
            self._default_top_k,
        )
        return {"status": "initialized", "version": "1.0.0"}

    async def score(
        self,
        search_results: list[dict] | None = None,
        query: str = "",
        query_type: str | None = None,
        ctx: Any = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Score KB relevance from pre-fetched search results.

        Pipeline step usage:
          {"module": "kb_relevance_scorer", "operation": "score",
           "input_from": {"search_results": "retrieve.results", "query": "inputs.query"}}
        """
        from .providers import score_results

        return score_results(
            search_results=search_results or [],
            query=query,
            query_type=query_type,
            config=self._engine_config,
        )

    async def score_from_qdrant(
        self,
        query: str = "",
        collection: str = "",
        top_k: int | None = None,
        ctx: Any = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Score KB relevance by querying Qdrant directly.

        Self-contained: fetches from Qdrant, then scores.

        Pipeline step usage:
          {"module": "kb_relevance_scorer", "operation": "score_from_qdrant",
           "input_from": {"query": "inputs.query", "collection": "inputs.collection"}}
        """
        from .providers import score_from_qdrant_provider

        rag_qdrant = kwargs.get("rag_qdrant") or self._rag_qdrant
        _top_k = top_k or self._default_top_k

        return await score_from_qdrant_provider(
            query=query,
            collection=collection,
            top_k=_top_k,
            rag_qdrant=rag_qdrant,
            config=self._engine_config,
        )

    async def health_check(self, **kwargs) -> dict[str, Any]:
        """Module health check."""
        return {
            "status": "healthy",
            "module": "kb_relevance_scorer",
            "version": "1.0.0",
            "initialized": self._initialized,
            "rag_qdrant_available": self._rag_qdrant is not None,
        }

    async def call_operation(self, operation: str, **kwargs) -> Any:
        """Dispatch operation by name (ModuleLoader / pipeline step interface)."""
        if operation == "initialize":
            return await self.initialize(**kwargs)
        if operation == "score":
            return await self.score(**kwargs)
        if operation == "score_from_qdrant":
            return await self.score_from_qdrant(**kwargs)
        if operation == "health_check":
            return await self.health_check(**kwargs)
        raise ValueError(f"[kb_relevance_scorer] Unknown operation: {operation}")

    async def shutdown(self, **kwargs):
        """Graceful shutdown."""
        self._initialized = False
        logger.info("[KB-RELEVANCE] Shutdown complete")
        return {"status": "shutdown"}
