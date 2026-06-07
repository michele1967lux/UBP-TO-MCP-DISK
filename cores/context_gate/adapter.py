"""
context_gate — Security gate for pipeline context resolution.

Resolves client/user settings, ACL (allowed pipelines), feature flags,
and domain preset. Injects resolved context into pipeline step outputs
so subsequent steps can use ${gate.*} and ${settings.*} variables.

SECURITY: deny-by-default.
  - Missing ctx or client_id → GateError (pipeline stops)
  - Redis/backend unreachable for auth data → GateError (pipeline stops)
  - Enrichment failures (user prefs, domain) → degrade with defaults (pipeline continues)

Env vars:
  UBP_CONTEXT_GATE__ENABLED=true   (disable gate entirely for testing)
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

logger = logging.getLogger(__name__)

# Feature-flag → pipeline exclusion map (same as user_router.py + adapter.py)
_FEATURE_PIPELINE_MAP: Dict[str, set] = {
    "web_search_enabled": {"rag_web_pipeline", "rag_web", "web_search_pipeline"},
    "report_generation_enabled": {
        "report_full", "report_research_only", "report_render_only",
        "report_with_charts", "report_interactive", "presentation_generate",
        "report_research",
    },
    "investigation_enabled": {"research_pipeline"},
    "agent_pipeline_enabled": {"agent_pipeline"},
    "hyde_enabled": {"hyde_rag"},
    "multimodal_enabled": {"multimodal_pipeline", "multimodal_vqa", "multimodal_document"},
    "media_hub_enabled": {"media_render_pipeline", "media_slot_resolve"},
}


class GateError(Exception):
    """Raised when the context gate cannot resolve authorization.

    Pipeline execution MUST stop when this is raised (deny-by-default).
    """


class ContextGateAdapter:
    """Security gate: resolves client/user context for pipeline steps.

    Designed as a pipeline step (step 0) with error_strategy: fail.
    Pure I/O — no LLM calls. Uses ClientConfigResolver (singleton, cached 60s).
    """

    def __init__(self, module_path=None, **kwargs):
        self._module_path = module_path
        self._admin_clients = None
        self._user_profile_memory = None
        self._resolver = None
        self._initialized = False
        self._resolve_count = 0
        self._error_count = 0

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

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self, **kwargs) -> Dict[str, Any]:
        """Initialize gate — resolve optional module references."""
        self._initialized = True
        logger.info("[CONTEXT-GATE] Initialized")
        return {"status": "initialized"}

    async def health_check(self, **kwargs) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "resolve_count": self._resolve_count,
            "error_count": self._error_count,
        }

    # ── Main Operation ────────────────────────────────────────────────────

    async def resolve(
        self,
        ctx: Any = None,
        pipeline_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Resolve client/user context for pipeline authorization and settings.

        Returns:
            Dict with keys: settings, allowed_pipelines, feature_flags,
            client_id, user_id, domain_preset, pipeline_overrides, default_pipeline.

        Raises:
            GateError: if ctx/client_id missing or auth data unreachable.
        """
        t0 = time.monotonic()

        # ── Auth validation (MUST succeed or raise) ───────────────────────
        if ctx is None:
            self._error_count += 1
            raise GateError("Missing security context (ctx=None)")

        user_obj = getattr(ctx, "user", None)
        if user_obj is not None:
            client_id = getattr(user_obj, "client_id", None)
            user_id = getattr(user_obj, "user_id", None)
        else:
            client_id = getattr(ctx, "client_id", None)
            user_id = getattr(ctx, "user_id", None)

        if not client_id:
            self._error_count += 1
            raise GateError("Missing client_id in security context")

        # ── Resolve auth data (MUST succeed or raise) ─────────────────────
        auth_data = await self._resolve_auth(client_id, pipeline_name)

        # ── Resolve enrichment data (CAN degrade) ─────────────────────────
        enrichment = await self._resolve_enrichment(client_id, user_id, pipeline_name)

        # ── Assemble output ───────────────────────────────────────────────
        elapsed_ms = (time.monotonic() - t0) * 1000
        self._resolve_count += 1

        result = {
            "client_id": client_id,
            "user_id": user_id or "",
            # Auth data (mandatory)
            "allowed_pipelines": auth_data["allowed_pipelines"],
            "feature_flags": auth_data["feature_flags"],
            "default_pipeline": auth_data["default_pipeline"],
            # Enrichment data (may be defaults)
            "settings": enrichment["settings"],
            "domain_preset": enrichment["domain_preset"],
            "pipeline_overrides": enrichment["pipeline_overrides"],
            "step_overrides": enrichment["step_overrides"],
            "user_preferences": enrichment["user_preferences"],
            # Routing context for downstream (LLM router, prompt builder)
            "client_identity": enrichment["client_identity"],
            "allowed_collections": enrichment["allowed_collections"],
            "dedicated_pipeline": enrichment.get("dedicated_pipeline"),
            # Meta
            "resolved_at_ms": round(elapsed_ms, 1),
            "auth_source": auth_data["source"],
            "enrichment_degraded": enrichment["degraded"],
        }

        logger.info(
            "[CONTEXT-GATE] client=%s user=%s pipelines=%d flags=%d domain=%s (%.1fms)",
            client_id,
            (user_id or "")[:8],
            len(auth_data["allowed_pipelines"]),
            len(auth_data["feature_flags"]),
            enrichment["domain_preset"],
            elapsed_ms,
        )
        return result

    # ── Auth Resolution (MUST succeed or raise) ───────────────────────────

    async def _resolve_auth(
        self, client_id: str, pipeline_name: Optional[str]
    ) -> Dict[str, Any]:
        """Resolve authorization data: allowed pipelines + feature flags.

        Raises GateError if resolution fails (deny-by-default).
        """
        resolver = self._get_resolver()

        try:
            # Load client config (cached 60s in resolver)
            allowed = await resolver.get_allowed_pipelines(client_id)
            feature_flags = await resolver.get_feature_flags(client_id)
            default_pipeline = await resolver.get_default_pipeline(client_id)

            # Apply feature-flag exclusions to allowed list
            effective_allowed = self._apply_feature_exclusions(allowed, feature_flags)

            return {
                "allowed_pipelines": effective_allowed,
                "feature_flags": feature_flags,
                "default_pipeline": default_pipeline,
                "source": "client_config",
            }
        except GateError:
            raise
        except Exception as e:
            self._error_count += 1
            raise GateError(
                f"Cannot resolve authorization for client {client_id}: {e}"
            ) from e

    def _apply_feature_exclusions(
        self, allowed: List[str], feature_flags: Dict[str, bool]
    ) -> List[str]:
        """Remove pipelines disabled by feature flags from the allowed list."""
        excluded: set = set()
        for feature, pipelines in _FEATURE_PIPELINE_MAP.items():
            # Default True for backward compat (same as ClientConfigResolver)
            if not feature_flags.get(feature, True):
                excluded.update(pipelines)

        if not excluded:
            return allowed

        if "*" in allowed:
            # Wildcard — we can't enumerate all pipelines, return ["*"] with
            # exclusion metadata. The orchestrator's execute() safety net
            # handles individual pipeline checks.
            return allowed

        return [p for p in allowed if p not in excluded]

    # ── Enrichment Resolution (CAN degrade) ───────────────────────────────

    async def _resolve_enrichment(
        self,
        client_id: str,
        user_id: Optional[str],
        pipeline_name: Optional[str],
    ) -> Dict[str, Any]:
        """Resolve non-security enrichment data. Degrades with defaults on failure."""
        degraded = False
        settings: Dict[str, Any] = {}
        domain_preset = "generic"
        pipeline_overrides: Dict[str, Any] = {}
        step_overrides: Dict[str, Dict[str, Any]] = {}
        user_preferences: Dict[str, Any] = {}

        resolver = self._get_resolver()

        # Client-level settings
        client_identity = ""
        allowed_collections: List[str] = []
        dedicated_pipeline: Optional[str] = None
        try:
            domain_config = await resolver.get_domain_config(client_id)
            domain_preset = domain_config.get("domain", "generic") if domain_config else "generic"

            if pipeline_name:
                pipeline_overrides = await resolver.resolve_pipeline_config(
                    client_id, pipeline_name
                )
                step_overrides = await resolver.get_step_overrides(
                    client_id, pipeline_name
                )

            # Build merged settings from client config
            client_config = await resolver._get_client_config(client_id)
            pc = client_config.get("pipeline_config", {})
            settings = {
                "domain_preset": domain_preset,
                **pc.get("global_settings", {}),
                **pipeline_overrides,
            }

            # Extract routing context for downstream consumers
            prompt_config = client_config.get("prompt_config", {})
            client_identity = prompt_config.get("identity", "")
            kb_config = client_config.get("kb_config", {})
            allowed_collections = kb_config.get("universal_kbs_assigned", [])

            # Dedicated pipeline resolution
            dedicated_pipeline = await resolver.get_dedicated_pipeline(client_id)
        except Exception as e:
            logger.warning(
                "[CONTEXT-GATE] Enrichment degraded (client settings): %s", e
            )
            degraded = True

        # User-level preferences (optional module)
        if user_id:
            try:
                user_prefs = await self._get_user_preferences(user_id)
                if user_prefs:
                    user_preferences = user_prefs
                    # Merge: user > client (user preferences override client settings)
                    settings = {**settings, **user_prefs}
            except Exception as e:
                logger.debug(
                    "[CONTEXT-GATE] User preferences unavailable: %s", e
                )
                # Not degraded — user prefs are fully optional

        return {
            "settings": settings,
            "domain_preset": domain_preset,
            "pipeline_overrides": pipeline_overrides,
            "step_overrides": step_overrides,
            "user_preferences": user_preferences,
            "degraded": degraded,
            "client_identity": client_identity,
            "allowed_collections": allowed_collections,
            "dedicated_pipeline": dedicated_pipeline,
        }

    async def _get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load user routing/settings preferences if user_profile_memory is available."""
        if self._user_profile_memory is None:
            return None
        try:
            result = await self._user_profile_memory.call_operation(
                "get_routing_preferences", user_id=user_id
            )
            return result if isinstance(result, dict) else None
        except Exception:
            return None

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_resolver(self):
        """Get or create the ClientConfigResolver singleton."""
        if self._resolver is None:
            try:
                from ubp_enterprise_hybrid.backend.app.core.client_config_resolver import (
                    get_client_config_resolver,
                )
            except ImportError:
                try:
                    from ubp_enterprise_hybrid.backend.app.core.client_config_resolver import (
                        get_client_config_resolver,
                    )
                except ImportError:
                    raise GateError(
                        "ClientConfigResolver not available — cannot resolve authorization"
                    )
            self._resolver = get_client_config_resolver()
        return self._resolver

    def set_admin_clients(self, module) -> None:
        """Inject admin_clients module for ClientConfigResolver."""
        self._admin_clients = module
        resolver = self._get_resolver()
        resolver.set_admin_clients(module)

    def set_user_profile_memory(self, module) -> None:
        """Inject user_profile_memory module for user preferences."""
        self._user_profile_memory = module

    # ── call_operation dispatch ────────────────────────────────────────────

    async def call_operation(self, operation: str, **kwargs):
        """Dispatch operation by name (ModuleLoader interface)."""
        if operation == "initialize":
            return await self.initialize(**kwargs)
        if operation == "resolve":
            return await self.resolve(**kwargs)
        if operation == "health_check":
            return await self.health_check(**kwargs)
        raise ValueError(f"Unknown operation: {operation}")
