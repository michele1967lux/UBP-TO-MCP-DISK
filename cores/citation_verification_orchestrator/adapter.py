"""
citation_verification_orchestrator — adapter.py

UBP 3-file pattern bridge.  Handles lifecycle, DI resolution, event bus,
and delegates all logic to providers.VerificationOrchestrator.

LLM calls route through ``_call_llm()`` → ``pipeline_orchestrator.execute(simple_chat)``
exactly like agentic_rag (BUG-AGENT-001 pattern).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

from .providers import VerificationOrchestrator

logger = logging.getLogger("ubp.citation_verification_orchestrator")


class CitationVerificationOrchestratorAdapter:
    """UBP adapter for the Citation Verification Orchestrator module."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        module_path: Path,
        di_container: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        self.module_path = Path(module_path)
        self.di_container = di_container
        self.event_bus = event_bus
        self._initialized = False

        # Config
        self._config: Dict[str, Any] = {}
        self._enabled = True

        # Provider
        self._orchestrator: Optional[VerificationOrchestrator] = None

        # Lazy DI caches
        self._citations_verifier: Optional[Any] = None
        self._pipeline_orchestrator: Optional[Any] = None
        self._budget_manager: Optional[Any] = None
        self._redis: Optional[Any] = None
        self._current_ctx: Optional[Any] = None

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, ctx=None, **kwargs) -> Dict[str, Any]:
        if self._initialized:
            return {"status": "already_initialized"}

        self._current_ctx = ctx
        self._load_config()

        # Check global feature flag
        self._enabled = self._resolve_bool(self._config.get("enabled", True))
        if not self._enabled:
            logger.info("[CVO] Module disabled via feature flag")
            self._initialized = True
            return {"status": "disabled"}

        # Create provider
        self._orchestrator = VerificationOrchestrator(
            config=self._config,
            llm_caller=self._call_llm,
        )

        # Resolve dependencies (best-effort)
        await self._resolve_dependencies()

        # Subscribe to events
        self._subscribe_events()

        # Bootstrap trust database (async, best-effort)
        await self._bootstrap_trust_db()

        self._initialized = True
        logger.info("[CVO] Citation Verification Orchestrator initialized")
        return {
            "status": "initialized",
            "citations_verifier_available": self._citations_verifier is not None,
            "pipeline_orchestrator_available": self._pipeline_orchestrator is not None,
            "redis_available": self._redis is not None,
        }

    async def shutdown(self, ctx=None, **kwargs) -> Dict[str, Any]:
        self._initialized = False
        self._orchestrator = None
        self._citations_verifier = None
        self._pipeline_orchestrator = None
        self._redis = None
        logger.info("[CVO] Shutdown complete")
        return {"status": "shutdown"}

    async def health_check(self, ctx=None, **kwargs) -> Dict[str, Any]:
        if not self._enabled:
            return {"status": "disabled", "module": "citation_verification_orchestrator"}

        cv = await self._get_citations_verifier()
        po = await self._get_pipeline_orchestrator()
        rd = await self._get_redis()

        deps = {
            "citations_verifier": cv is not None,
            "pipeline_orchestrator": po is not None,
            "redis": rd is not None,
        }
        degraded = not cv
        return {
            "status": "degraded" if degraded else "healthy",
            "module": "citation_verification_orchestrator",
            "enabled": self._enabled,
            "dependencies": deps,
        }

    # ------------------------------------------------------------------
    # Operations (called by pipeline step executor via getattr)
    # ------------------------------------------------------------------

    async def verify_response(
        self,
        answer: str = "",
        chunks: Optional[List[Dict[str, Any]]] = None,
        query: str = "",
        ctx: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Post-generate verification step (Modalità 1 + 2)."""
        if not self._enabled or not self._orchestrator:
            return {"verified": False, "skipped": True, "skip_reason": "module disabled"}

        self._current_ctx = ctx or self._current_ctx
        chunks = chunks or []

        # Detect web sources
        web_present = any(
            c.get("url") or c.get("source_url")
            for c in chunks
        )

        # Get tightness from budget manager
        tightness = 0.0
        bm = await self._get_budget_manager()
        if bm:
            try:
                t_result = await bm.calculate_tightness(ctx=ctx)
                tightness = float(t_result.get("tightness", 0.0)) if isinstance(t_result, dict) else 0.0
            except Exception:
                pass

        # Check force flag from pipeline context
        force = kwargs.get("force_verification", False)
        if not force:
            # Check if verify_requested was set in the pipeline config
            force = kwargs.get("verify_requested", False)

        cv = await self._get_citations_verifier()

        return await self._orchestrator.verify_response(
            answer=answer,
            chunks=chunks,
            query=query,
            tightness=tightness,
            web_sources_present=web_present,
            force_verification=force,
            language=kwargs.get("language", "it"),
            citations_verifier=cv,
            min_tightness_trigger=float(kwargs.get("min_tightness_trigger", 0.7)),
            auto_verify_web_sources=bool(kwargs.get("auto_verify_web_sources", True)),
            hallucination_threshold=float(kwargs.get("hallucination_threshold", 0.3)),
            grounding_min_score=float(kwargs.get("grounding_min_score", 0.5)),
        )

    async def filter_trusted_sources(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        ctx: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Pre-generate trust filter step (Modalità 3)."""
        if not self._enabled or not self._orchestrator:
            return {"filtered_chunks": chunks or [], "removed_count": 0, "total_count": len(chunks or [])}

        self._current_ctx = ctx or self._current_ctx
        cv = await self._get_citations_verifier()
        rd = await self._get_redis()

        return await self._orchestrator.filter_trusted_sources(
            chunks=chunks or [],
            min_trust_score=float(kwargs.get("min_trust_score", 0.6)),
            internal_sources_trusted=bool(kwargs.get("internal_sources_trusted", True)),
            citations_verifier=cv,
            redis_client=rd,
        )

    async def verify_web_sources(
        self,
        urls: Optional[List[str]] = None,
        ctx: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Trust check on web URLs."""
        if not self._enabled or not self._orchestrator:
            return {"sources": [], "average_trust": 0.0}

        self._current_ctx = ctx or self._current_ctx
        cv = await self._get_citations_verifier()
        rd = await self._get_redis()

        return await self._orchestrator.verify_web_sources(
            urls=urls or [],
            domain=kwargs.get("domain", "general"),
            citations_verifier=cv,
            redis_client=rd,
        )

    async def update_trust_database(
        self,
        verification_results: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        ctx: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Auto-update trust scores after verification."""
        if not self._enabled or not self._orchestrator:
            return {"updated_domains": 0, "updates": []}

        self._current_ctx = ctx or self._current_ctx
        rd = await self._get_redis()

        return await self._orchestrator.update_trust_database(
            verification_results=verification_results or {},
            sources=sources or [],
            redis_client=rd,
        )

    # ------------------------------------------------------------------
    # Admin trust operations
    # ------------------------------------------------------------------

    async def list_trust_entries(
        self,
        ctx: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """List all trust database entries (admin)."""
        rd = await self._get_redis()
        if not self._orchestrator:
            return {"entries": [], "count": 0}
        entries = await self._orchestrator.list_trust_entries(
            redis_client=rd,
            limit=int(kwargs.get("limit", 100)),
        )
        return {"entries": entries, "count": len(entries)}

    async def set_trust_entry(
        self,
        domain: str = "",
        trust_score: float = 0.5,
        ctx: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Force-set a trust score (admin)."""
        rd = await self._get_redis()
        if not self._orchestrator or not domain:
            return {"success": False, "reason": "missing domain or not initialized"}
        # v6.4.2: strict validation — reject out-of-range scores instead of silent clamping
        if not (0.0 <= trust_score <= 1.0):
            return {"success": False, "reason": f"trust_score must be between 0.0 and 1.0, got {trust_score}"}
        return await self._orchestrator.set_trust_entry(
            domain=domain,
            trust_score=trust_score,
            redis_client=rd,
            category=kwargs.get("category", "admin_override"),
        )

    async def delete_trust_entry(
        self,
        domain: str = "",
        ctx: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Delete a trust entry (admin)."""
        rd = await self._get_redis()
        if not self._orchestrator or not domain:
            return {"success": False, "reason": "missing domain or not initialized"}
        return await self._orchestrator.delete_trust_entry(
            domain=domain, redis_client=rd,
        )

    async def get_trust_entry(
        self,
        domain: str = "",
        ctx: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get a single trust entry with decay applied."""
        rd = await self._get_redis()
        if not self._orchestrator or not domain:
            return {"domain": domain, "trust_score": None, "found": False}
        entry = await self._orchestrator.get_trust_entry(domain=domain, redis_client=rd)
        if entry is None:
            return {"domain": domain, "trust_score": None, "found": False}
        entry["found"] = True
        return entry

    async def bootstrap_trust_db(
        self,
        ctx: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Manually trigger trust DB bootstrap (admin)."""
        return await self._bootstrap_trust_db()

    # ------------------------------------------------------------------
    # call_operation — dynamic dispatch for pipeline step executor
    # ------------------------------------------------------------------

    async def call_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Dispatch operation calls to appropriate methods."""
        operation_map = {
            "initialize": self.initialize,
            "shutdown": self.shutdown,
            "health_check": self.health_check,
            "verify_response": self.verify_response,
            "filter_trusted_sources": self.filter_trusted_sources,
            "verify_web_sources": self.verify_web_sources,
            "update_trust_database": self.update_trust_database,
            "list_trust_entries": self.list_trust_entries,
            "set_trust_entry": self.set_trust_entry,
            "delete_trust_entry": self.delete_trust_entry,
            "get_trust_entry": self.get_trust_entry,
            "bootstrap_trust_db": self.bootstrap_trust_db,
        }
        method = operation_map.get(operation)
        if not method:
            raise ValueError(f"Unknown operation: {operation}")
        return await method(**kwargs)

    # ------------------------------------------------------------------
    # _call_llm — pipeline delegation (agentic_rag pattern)
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
        purpose: str = "verification",
    ) -> str:
        """Route LLM call through pipeline HA chain via simple_chat.

        Follows the same delegation pattern as agentic_rag (BUG-AGENT-001..005).
        """
        query = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        po = await self._get_pipeline_orchestrator()

        if po:
            try:
                result = await po.execute(
                    pipeline_name="simple_chat",
                    inputs={"query": query},
                    config={"max_tokens": max_tokens, "temperature": temperature},
                    ctx=self._current_ctx,
                )
                outputs = result.get("outputs", {})
                text = outputs.get("answer") or ""
                if not text and result.get("status") == "failed":
                    step_results = result.get("step_results", [{}])
                    error = step_results[0].get("error", "unknown") if step_results else "unknown"
                    raise RuntimeError(f"Pipeline failed for {purpose}: {error}")
                return text
            except Exception as e:
                logger.warning(f"[CVO] _call_llm pipeline failed: {e}")
                raise

        raise RuntimeError("[CVO] No pipeline_orchestrator available for LLM calls")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _subscribe_events(self) -> None:
        if not self.event_bus:
            return
        try:
            self.event_bus.subscribe("rag.response_ready", self._on_rag_response_ready)
            logger.info("[CVO] Subscribed to rag.response_ready")
        except Exception as e:
            logger.warning(f"[CVO] Event subscription failed: {e}")

    async def _on_rag_response_ready(self, event: Any) -> None:
        """Auto-verify on rag.response_ready if configured."""
        if not self._enabled:
            return
        try:
            payload = event if isinstance(event, dict) else getattr(event, "data", {})
            answer = payload.get("answer", "")
            chunks = payload.get("chunks", payload.get("sources", []))
            query = payload.get("query", "")

            if len(answer) < 200:
                return

            await self.verify_response(
                answer=answer, chunks=chunks, query=query,
                force_verification=False,
            )
        except Exception as e:
            logger.warning(f"[CVO] Event handler error: {e}")

    # ------------------------------------------------------------------
    # Bootstrap trust DB
    # ------------------------------------------------------------------

    async def _bootstrap_trust_db(self) -> Dict[str, Any]:
        """Load predefined trusted domains into Redis."""
        if not self._orchestrator:
            return {"bootstrapped": 0, "skipped": True}
        try:
            cv = await self._get_citations_verifier()
            rd = await self._get_redis()
            return await self._orchestrator.bootstrap_trust_database(
                redis_client=rd,
                citations_verifier=cv,
            )
        except Exception as e:
            logger.warning(f"[CVO] Trust DB bootstrap failed: {e}")
            return {"bootstrapped": 0, "error": str(e)}

    # ------------------------------------------------------------------
    # DI resolution (lazy + cached)
    # ------------------------------------------------------------------

    async def _resolve_dependencies(self) -> None:
        await self._get_citations_verifier()
        await self._get_pipeline_orchestrator()
        await self._get_budget_manager()
        await self._get_redis()

    async def _get_citations_verifier(self) -> Optional[Any]:
        if self._citations_verifier:
            return self._citations_verifier
        self._citations_verifier = await self._resolve_module("citations_verifier")
        if self._citations_verifier:
            logger.info("[CVO] citations_verifier resolved from DI")
        return self._citations_verifier

    async def _get_pipeline_orchestrator(self) -> Optional[Any]:
        if self._pipeline_orchestrator:
            return self._pipeline_orchestrator
        self._pipeline_orchestrator = await self._resolve_module("pipeline_orchestrator")
        if self._pipeline_orchestrator:
            logger.info("[CVO] pipeline_orchestrator resolved — LLM calls via HA pipeline")
        return self._pipeline_orchestrator

    async def _get_budget_manager(self) -> Optional[Any]:
        if self._budget_manager:
            return self._budget_manager
        self._budget_manager = await self._resolve_module("adaptive_budget_manager")
        return self._budget_manager

    async def _get_redis(self) -> Optional[Any]:
        if self._redis:
            return self._redis
        for name in ("system_redis_client", "redis", "redis_client"):
            self._redis = await self._resolve_module(name)
            if self._redis:
                logger.info(f"[CVO] Redis resolved as '{name}'")
                break
        return self._redis

    async def _resolve_module(self, name: str) -> Optional[Any]:
        if not self.di_container:
            return None
        try:
            if hasattr(self.di_container, "resolve"):
                module = await self.di_container.resolve(name)
            elif hasattr(self.di_container, "get"):
                module = self.di_container.get(name)
            else:
                module = None
            return module
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        config_path = self.module_path / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._config = self._resolve_env_vars(raw)
        else:
            self._config = {}

    def _resolve_env_vars(self, obj: Any) -> Any:
        """Recursively resolve ${VAR:-default} patterns in config."""
        if isinstance(obj, str):
            import re
            match = re.match(r"^\$\{([^:}]+):-([^}]*)\}$", obj)
            if match:
                var_name, default = match.group(1), match.group(2)
                return os.environ.get(var_name, default)
            return obj
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve_env_vars(v) for v in obj]
        return obj

    @staticmethod
    def _resolve_bool(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)
