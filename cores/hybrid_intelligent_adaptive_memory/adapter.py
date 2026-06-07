"""
UBP adapter for HIAMS.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule

from .models import ProfileType
from .providers import HIAMSProvider

try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

logger = logging.getLogger(__name__)


class HIAMSAdapter(BaseHybridModule):
    """Framework bridge for HIAMS."""

    def __init__(self, module_path: Path, **kwargs):
        super().__init__(module_path, **kwargs)
        self.provider: HIAMSProvider | None = None
        self._init_status: Dict[str, Any] = {"status": "not_initialized"}

    def _build_context_from_di(self) -> OperationContext:
        return OperationContext(client_id="default", user_id=None, session_id=None, source="rest")

    async def initialize(self) -> None:
        logger.info("[HIAMS-Adapter] Initializing %s", self.manifest.name)
        try:
            self.provider = HIAMSProvider(config=self.config, di_container=self.di_container)
            self._init_status = {
                "status": "healthy",
                "config_summary": {
                    "chat_trigger": self.config.get("chat_profile", {}).get("compression_trigger_threshold", 6),
                    "agent_trigger": self.config.get("agent_loop_profile", {}).get("compression_trigger_threshold", 5),
                },
            }
        except Exception as exc:
            logger.error("[HIAMS-Adapter] Initialization failed: %s", exc, exc_info=True)
            self._init_status = {"status": "failed", "reason": str(exc)}

    async def shutdown(self) -> None:
        logger.info("[HIAMS-Adapter] Shutting down %s", self.manifest.name)
        self.provider = None
        self._init_status = {"status": "shutdown"}

    async def health_check(self, **kwargs) -> Dict[str, Any]:
        return {
            "module": self.manifest.name,
            "status": "healthy" if self.provider else "unhealthy",
            "init_status": self._init_status,
            "active_sessions": len(self.provider.list_sessions()) if self.provider else 0,
        }

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

    async def get_projected_context(
        self,
        session_id: str,
        query: str,
        profile: str = "chat",
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if not self.provider:
            raise RuntimeError("Provider not initialized")
        profile_type = ProfileType(profile) if profile in ("chat", "agent_loop") else ProfileType.CHAT
        return self.provider.get_projected_context(session_id, query, profile_type)

    async def compress(
        self,
        session_id: str,
        profile: str = "chat",
        force: bool = False,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if not self.provider:
            raise RuntimeError("Provider not initialized")
        profile_type = ProfileType(profile) if profile in ("chat", "agent_loop") else ProfileType.CHAT
        return (await self.provider.compress(session_id=session_id, profile_type=profile_type, force=force)).model_dump()

    async def should_compress(
        self,
        session_id: str,
        profile: str = "chat",
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if not self.provider:
            raise RuntimeError("Provider not initialized")
        profile_type = ProfileType(profile) if profile in ("chat", "agent_loop") else ProfileType.CHAT
        return self.provider.should_compress(session_id, profile_type)

    async def get_session_state(
        self,
        session_id: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
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
        if not self.provider:
            raise RuntimeError("Provider not initialized")
        deleted = self.provider.delete_session(session_id)
        return {"session_id": session_id, "deleted": deleted}

    async def get_config(self, **kwargs) -> Dict[str, Any]:
        return self.config
