"""media_hub — Adapter layer (3-file pattern).

Bridge between UBP DI container and media_hub business logic.
Resolves optional dependencies and dispatches operations.

MCP-COMPAT (ARCH-008): Added OperationContext support for dual REST/MCP compatibility.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        # Minimal fallback for import path flexibility
        from ..._shared.operation_context import OperationContext

logger = logging.getLogger("ubp.media_hub.adapter")


class MediaHubAdapter:
    """Adapter for the media_hub module.

    Resolves dependencies from DI container and dispatches operations
    to provider functions.
    """

    MODULE_NAME = "media_hub"

    def __init__(self, container: Any = None, config: Optional[Dict[str, Any]] = None):
        self._container = container
        self._config = config or self._load_config()
        self._redis = None
        self._event_bus = None
        self._chart_renderer = None
        self._diagram_renderer = None
        self._image_provider = None
        self._initialized = False

        if container:
            self._resolve_dependencies()

    def _load_config(self) -> Dict[str, Any]:
        """Load config.json."""
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _resolve_dependencies(self) -> None:
        """Resolve optional dependencies from DI container."""
        c = self._container
        if not c:
            return

        # Redis
        try:
            self._redis = getattr(c, "redis_client", None) or getattr(c, "redis", None)
        except Exception:
            pass

        # Event bus
        try:
            self._event_bus = getattr(c, "event_bus", None)
        except Exception:
            pass

        self._initialized = True
        logger.info("[MEDIA-HUB] Adapter initialized, redis=%s, event_bus=%s",
                     self._redis is not None, self._event_bus is not None)

    # ── MCP-COMPAT: OperationContext helpers (ARCH-008) ────

    def _build_context_from_di(self) -> OperationContext:
        """
        Build OperationContext from DI container — backward compatibility for REST path.
        
        MCP-COMPAT: When ctx is not provided (REST path), this method constructs
        an OperationContext from the DI container state.
        
        Returns:
            OperationContext with default values
        """
        return OperationContext(
            client_id="default",
            user_id=None,
            session_id=None,
            source="rest",
        )

    def _normalize_ctx(self, ctx: Any) -> OperationContext:
        """
        Normalize any context format to OperationContext.
        
        MCP-COMPAT: Handles both legacy security context (ctx.user.user_id) 
        and new OperationContext format for backward compatibility.
        
        Args:
            ctx: Either OperationContext, legacy security context, or None
            
        Returns:
            OperationContext instance
        """
        if ctx is None:
            return self._build_context_from_di()
        
        # Already an OperationContext
        if isinstance(ctx, OperationContext):
            return ctx
        
        # Legacy security context format (ctx.user.user_id, ctx.user.roles)
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
        
        # Fallback
        return self._build_context_from_di()

    # ── Operation Dispatch ───────────────────────────────

    async def execute(self, operation: str, **kwargs: Any) -> Dict[str, Any]:
        """Dispatch an operation by name."""
        ops = {
            "plan_media": self.plan_media,
            "render_media": self.render_media,
            "get_media": self.get_media,
            "validate_media": self.validate_media,
            "resolve_slots": self.resolve_slots,
            "health_check": self.health_check,
        }
        handler = ops.get(operation)
        if not handler:
            return {"error": f"Unknown operation: {operation}", "status": "error"}
        return await handler(**kwargs)

    # ── Operations ───────────────────────────────────────

    async def plan_media(self, request: Dict[str, Any], context: Optional[Dict] = None, **kw: Any) -> Dict[str, Any]:
        """Plan media rendering."""
        from .providers import plan_media_provider
        return await plan_media_provider(
            request_dict=request,
            config=self._config,
        )

    async def render_media(
        self,
        plan: Optional[Dict[str, Any]] = None,
        request: Optional[Dict[str, Any]] = None,
        context: Optional[Dict] = None,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Render media from plan or request."""
        from .providers import render_media_provider
        return await render_media_provider(
            plan_dict=plan,
            request_dict=request,
            config=self._config,
            redis=self._redis,
            chart_renderer=self._chart_renderer,
            diagram_renderer=self._diagram_renderer,
            image_provider=self._image_provider,
            event_bus=self._event_bus,
        )

    async def get_media(self, asset_id: str, **kw: Any) -> Dict[str, Any]:
        """Get a rendered asset."""
        from .providers import get_media_provider
        return await get_media_provider(
            asset_id=asset_id,
            config=self._config,
            redis=self._redis,
        )

    async def validate_media(
        self,
        result: Dict[str, Any],
        request: Dict[str, Any],
        **kw: Any,
    ) -> Dict[str, Any]:
        """Validate a media result."""
        from .providers import validate_media_provider
        return await validate_media_provider(
            result_dict=result,
            request_dict=request,
        )

    async def resolve_slots(
        self,
        slots: List[Dict[str, Any]],
        context: Optional[Dict] = None,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Resolve MediaSlots from requirements_collector."""
        from .providers import resolve_slots_provider
        return await resolve_slots_provider(
            slots=slots,
            config=self._config,
            redis=self._redis,
            chart_renderer=self._chart_renderer,
            diagram_renderer=self._diagram_renderer,
            image_provider=self._image_provider,
            event_bus=self._event_bus,
        )

    async def health_check(self, **kw: Any) -> Dict[str, Any]:
        """Module health check."""
        return {
            "status": "ok",
            "module": self.MODULE_NAME,
            "initialized": self._initialized,
            "providers": {
                "redis": self._redis is not None,
                "event_bus": self._event_bus is not None,
                "chart_renderer": self._chart_renderer is not None,
                "diagram_renderer": self._diagram_renderer is not None,
                "image_provider": self._image_provider is not None,
            },
            "capabilities": {
                "chart": True,  # matplotlib always available
                "diagram": self._config.get("diagram", {}).get("engine") is not None,
                "image_gen": self._config.get("image_gen", {}).get("enabled", False),
                "composite": True,
                "cache": self._config.get("cache", {}).get("enabled", True),
            },
        }

    # ── Renderer Registration ────────────────────────────

    def register_chart_renderer(self, renderer: Any) -> None:
        """Register a chart renderer (e.g., matplotlib wrapper)."""
        self._chart_renderer = renderer
        logger.info("[MEDIA-HUB] Chart renderer registered")

    def register_diagram_renderer(self, renderer: Any) -> None:
        """Register a diagram renderer (e.g., mermaid wrapper)."""
        self._diagram_renderer = renderer
        logger.info("[MEDIA-HUB] Diagram renderer registered")

    def register_image_provider(self, provider: Any) -> None:
        """Register an image generation provider."""
        self._image_provider = provider
        logger.info("[MEDIA-HUB] Image provider registered")
