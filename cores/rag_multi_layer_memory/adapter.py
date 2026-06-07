"""
Multi-Layer Memory Adapter — UBP Framework Bridge.

Main entry point for the rag_multi_layer_memory module.
Implements BaseHybridModule and exposes all operations defined in manifest.json.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ubp_enterprise_hybrid.modules.cores._shared.base_module import BaseHybridModule
from ubp_enterprise_hybrid.backend.app.infra.event_bus import EventBus
from ubp_enterprise_hybrid.backend.app.infra.di_container import DIContainer

from .compression_engine import CompressionEngine
from .layer_manager import LayerManager

logger = logging.getLogger(__name__)


class MultiLayerMemoryAdapter(BaseHybridModule):
    """
    UBP adapter for multi-layer contextual memory.

    Provides centralized, client-aware, multi-domain memory management
    with 3 progressive layers:
    - Layer 0: Working Memory (sliding window of snapshots)
    - Layer 1: Evolved Mid-term Compression (compressed blocks)
    - Layer 2: Long-term Minimal Memory (persistent critical info)
    """

    def __init__(
        self,
        module_path: Path,
        event_bus: Optional[EventBus] = None,
        di_container: Optional[DIContainer] = None,
        **kwargs,
    ):
        super().__init__(
            module_path,
            event_bus=event_bus,
            di_container=di_container,
            **kwargs,
        )

        # Build compression engine with prompts path
        prompts_path = module_path / "prompts"
        self._compression_engine = CompressionEngine(
            prompts_path=prompts_path,
            config=self.config,
            di_container=di_container,
        )

        # Build layer manager (MEM-005: pass DI for optional Redis persistence)
        self._layer_manager = LayerManager(
            config=self.config,
            compression_engine=self._compression_engine,
            di_container=di_container,
        )

        self._initialized = False

    # ------------------------------------------------------------------
    # BaseHybridModule abstract methods
    # ------------------------------------------------------------------

    async def initialize(self) -> Optional[Dict[str, Any]]:
        """Initialize the multi-layer memory system."""
        if self._initialized:
            return {
                "status": "already_initialized",
                "module": self.manifest.name,
            }

        self._initialized = True
        logger.info(
            f"[MultiLayerMemory] Initialized — "
            f"L0 max={self.config.get('layer0', {}).get('max_snapshots', 5)}, "
            f"L1 max_blocks={self.config.get('layer1', {}).get('max_blocks', 4)}, "
            f"L2 enabled={self.config.get('layer2', {}).get('enabled', True)}"
        )

        return {
            "status": "initialized",
            "module": self.manifest.name,
            "config": {
                "layer0": self.config.get("layer0", {}),
                "layer1": self.config.get("layer1", {}),
                "layer2": self.config.get("layer2", {}),
                "token_limits": self.config.get("token_limits", {}),
            },
        }

    async def shutdown(self) -> None:
        """Shutdown the module and release resources."""
        self._initialized = False
        logger.info("[MultiLayerMemory] Shutdown completed")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the memory system."""
        stats = self._layer_manager.get_stats()
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "module": self.manifest.name,
            "version": self.manifest.version,
            "active_sessions": stats.get("active_sessions", 0),
            "config": stats.get("config", {}),
        }

    # ------------------------------------------------------------------
    # Operations (exposed via manifest.json)
    # ------------------------------------------------------------------

    async def add_snapshot(
        self,
        session_id: str,
        snapshot: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add a new Sub-Layer Zero snapshot to working memory (Layer 0).

        Manages the sliding window automatically and triggers compression
        when the threshold is reached.

        Args:
            session_id: Conversation session identifier.
            snapshot: Sub-Layer Zero snapshot data.

        Returns:
            Status dict with layer0_count and compression_triggered.
        """
        result = await self._layer_manager.add_snapshot(session_id, snapshot)

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "memory.layer0.snapshot_added",
                {
                    "session_id": session_id,
                    "turn": snapshot.get("turn"),
                    "layer0_count": result["layer0_count"],
                    "compression_triggered": result["compression_triggered"],
                },
            )

            if result["compression_triggered"]:
                await self.publisher.publish(
                    "memory.layer1.compression_completed",
                    {"session_id": session_id},
                )

        return result

    async def get_memory_context(
        self,
        session_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get the full multi-layer memory context for LLM consumption.

        Returns all 3 layers formatted for injection into the LLM context.

        Args:
            session_id: Conversation session identifier.

        Returns:
            Dict with layer0, layer1, layer2, and token_estimate.
        """
        return self._layer_manager.get_memory_context(session_id)

    async def trigger_compression(
        self,
        session_id: str,
        force: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Manually trigger compression of Layer 0 → Layer 1 (→ Layer 2).

        Args:
            session_id: Session identifier.
            force: Force compression even if threshold not reached.

        Returns:
            Compression result dict.
        """
        result = await self._layer_manager.trigger_compression(session_id, force=force)

        # Publish events
        if self.publisher and result.get("status") == "completed":
            await self.publisher.publish(
                "memory.layer1.compression_completed",
                {"session_id": session_id},
            )
            if result.get("layer2_updated"):
                await self.publisher.publish(
                    "memory.layer2.updated",
                    {"session_id": session_id},
                )

        return result

    async def clear_session(
        self,
        session_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Clear all memory layers for a given session.

        Args:
            session_id: Session identifier.

        Returns:
            Status dict.
        """
        result = self._layer_manager.clear_session(session_id)

        if self.publisher:
            await self.publisher.publish(
                "memory.session_cleared",
                {"session_id": session_id},
            )

        return result

    async def get_layer(
        self,
        session_id: str,
        layer: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get a specific layer's data.

        Args:
            session_id: Session identifier.
            layer: Layer number (0, 1, or 2).

        Returns:
            Dict with layer data.
        """
        return self._layer_manager.get_layer(session_id, layer)

    async def get_stats(
        self,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get memory system statistics.

        Args:
            session_id: Optional session ID for session-specific stats.

        Returns:
            Statistics dict.
        """
        return self._layer_manager.get_stats(session_id)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_memory_message_added(self, event: Any) -> None:
        """Handle memory.message_added events."""
        data = getattr(event, "data", {}) if hasattr(event, "data") else {}
        logger.debug(
            f"[MultiLayerMemory] Received memory.message_added event: "
            f"session={data.get('session_id', 'unknown')}"
        )

    async def on_rag_chat_completed(self, event: Any) -> None:
        """Handle rag.chat.completed events."""
        data = getattr(event, "data", {}) if hasattr(event, "data") else {}
        logger.debug(
            f"[MultiLayerMemory] Received rag.chat.completed event: "
            f"session={data.get('session_id', 'unknown')}"
        )
