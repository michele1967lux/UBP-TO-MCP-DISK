"""
Context Compression Engine — UBP Framework Adapter

Bridges the ContextCompressionProvider with the UBP module system.
Follows the 3-file pattern: adapter.py + providers.py + models.py

v1.0.0 — Initial implementation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule

from .models import (
    CompressionEngineConfig,
    CompressionMode,
    ProfileType,
)
from .providers import ContextCompressionProvider

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

logger = logging.getLogger(__name__)


class ContextCompressionEngineAdapter(BaseHybridModule):
    """
    UBP adapter for the Context Compression Engine.

    Exposes all manifest operations and manages provider lifecycle.
    """

    def __init__(self, module_path: Path, **kwargs):
        super().__init__(module_path, **kwargs)
        self.provider: Optional[ContextCompressionProvider] = None
        self._init_status: Dict[str, Any] = {"status": "not_initialized"}

    # ------------------------------------------------------------------
    # Context normalization (MCP-COMPAT)
    # ------------------------------------------------------------------

    def _build_context_from_di(self) -> OperationContext:
        return OperationContext(
            client_id="default",
            user_id=None,
            session_id=None,
            source="rest",
        )

    def _normalize_ctx(self, ctx: Any) -> OperationContext:
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

    async def initialize(self) -> None:
        """Initialize the module and its provider."""
        logger.info("[CCE-Adapter] Initializing %s", self.manifest.name)

        try:
            self.provider = ContextCompressionProvider(
                config=self.config,
                di_container=self.di_container,
            )
        except Exception as exc:
            logger.error("[CCE-Adapter] Initialization failed: %s", exc, exc_info=True)
            self._init_status = {
                "status": "failed",
                "reason": str(exc),
            }
            return

        logger.info("[CCE-Adapter] ✅ %s initialized", self.manifest.name)
        self._init_status = {
            "status": "healthy",
            "config_summary": {
                "enabled": self.config.get("enabled", True),
                "chat_trigger": self.config.get("chat_profile", {}).get(
                    "compression_trigger_threshold", 10
                ),
                "agent_trigger": self.config.get("agent_loop_profile", {}).get(
                    "compression_trigger_threshold", 8
                ),
            },
        }

    async def shutdown(self) -> None:
        """Shutdown the module."""
        logger.info("[CCE-Adapter] Shutting down %s", self.manifest.name)
        self.provider = None
        self._init_status = {"status": "shutdown"}

    async def health_check(self, **kwargs) -> Dict[str, Any]:
        """Check module health status."""
        session_count = len(self.provider.list_sessions()) if self.provider else 0
        return {
            "module": self.manifest.name,
            "status": "healthy" if self.provider else "unhealthy",
            "init_status": self._init_status,
            "active_sessions": session_count,
        }

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    async def process_turn(
        self,
        session_id: str,
        turn_number: int,
        query: str,
        response: str,
        profile: str = "chat",
        auto_compress: bool = True,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process a complete interaction turn — primary entry-point.

        Creates a sub-layer, optionally triggers compression, returns context.
        """
        if not self.provider:
            raise RuntimeError("Provider not initialized")

        profile_type = ProfileType(profile) if profile in ("chat", "agent_loop") else ProfileType.CHAT
        return await self.provider.process_turn(
            session_id=session_id,
            turn_number=turn_number,
            query=query,
            response=response,
            profile_type=profile_type,
            auto_compress=auto_compress,
        )

    async def create_sub_layer(
        self,
        session_id: str,
        turn_number: int,
        query: str,
        response: str,
        profile: str = "chat",
        manual_sub_layer: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a Layer 0 sub-layer for a turn."""
        if not self.provider:
            raise RuntimeError("Provider not initialized")

        profile_type = ProfileType(profile) if profile in ("chat", "agent_loop") else ProfileType.CHAT
        sub_layer = await self.provider.create_sub_layer(
            session_id=session_id,
            turn_number=turn_number,
            query=query,
            response=response,
            profile_type=profile_type,
            manual_sub_layer=manual_sub_layer,
        )
        return sub_layer.model_dump()

    async def compress(
        self,
        session_id: str,
        profile: str = "chat",
        mode: str = "threshold",
        force: bool = False,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Trigger compression on a session."""
        if not self.provider:
            raise RuntimeError("Provider not initialized")

        profile_type = ProfileType(profile) if profile in ("chat", "agent_loop") else ProfileType.CHAT
        compression_mode = CompressionMode.THRESHOLD
        try:
            compression_mode = CompressionMode(mode)
        except ValueError:
            pass
        result = await self.provider.compress(
            session_id=session_id,
            profile_type=profile_type,
            mode=compression_mode,
            force=force,
        )
        return result.model_dump()

    async def get_compressed_context(
        self,
        session_id: str,
        include_layer0_recent: int = 3,
        include_layer1: bool = True,
        include_layer2: bool = True,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get compressed context for LLM injection."""
        if not self.provider:
            raise RuntimeError("Provider not initialized")

        return self.provider.get_compressed_context(
            session_id=session_id,
            include_layer0_recent=include_layer0_recent,
            include_layer1=include_layer1,
            include_layer2=include_layer2,
        )

    async def get_session_state(
        self,
        session_id: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get full session compression state."""
        if not self.provider:
            raise RuntimeError("Provider not initialized")

        state = self.provider.export_state(session_id)
        if state is None:
            return {"session_id": session_id, "found": False}
        state["found"] = True
        return state

    async def delete_session(
        self,
        session_id: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Delete session compression state."""
        if not self.provider:
            raise RuntimeError("Provider not initialized")

        deleted = self.provider.delete_session(session_id)
        return {"session_id": session_id, "deleted": deleted}

    async def should_compress(
        self,
        session_id: str,
        profile: str = "chat",
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Check if compression is needed for a session."""
        if not self.provider:
            raise RuntimeError("Provider not initialized")

        profile_type = ProfileType(profile) if profile in ("chat", "agent_loop") else ProfileType.CHAT
        needs = self.provider.should_compress(session_id, profile_type)
        return {"session_id": session_id, "needs_compression": needs}

    async def get_config(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Return current configuration."""
        if not self.provider:
            raise RuntimeError("Provider not initialized")
        return self.provider.get_config()

    async def import_state(
        self,
        state_data: Dict[str, Any],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Import session state from dict."""
        if not self.provider:
            raise RuntimeError("Provider not initialized")

        session_id = self.provider.import_state(state_data)
        return {"session_id": session_id, "imported": True}


# Factory function for UBP module loading
async def create_adapter(module_path: Path, **kwargs) -> ContextCompressionEngineAdapter:
    """Create and return adapter instance."""
    return ContextCompressionEngineAdapter(module_path, **kwargs)


__all__ = ["ContextCompressionEngineAdapter", "create_adapter"]
