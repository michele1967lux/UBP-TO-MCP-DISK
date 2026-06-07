"""
Layer Manager — Orchestration of all 3 memory layers.

Coordinates Layer 0 (working memory), Layer 1 (mid-term compressed),
and Layer 2 (long-term minimal). Handles trigger logic for compression
and layer promotion.
"""

import logging
from typing import Any, Dict, List, Optional

from .compression_engine import CompressionEngine
from .models import (
    CompressionResult,
    Layer1Block,
    Layer2Memory,
    SessionMemoryState,
)
from .sub_layer_zero import SubLayerZeroManager
from ._persistence import MLMRedisSessionStore, maybe_build_mlm_store

logger = logging.getLogger(__name__)


class LayerManager:
    """
    Orchestrates all 3 memory layers for multi-layer memory management.

    Manages the lifecycle:
    - Layer 0: sliding window of snapshots (updated every turn)
    - Layer 1: compressed mid-term blocks (triggered at N-1 snapshots)
    - Layer 2: long-term minimal memory (updated during L1 compression)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        compression_engine: CompressionEngine,
        di_container: Any = None,
    ):
        """
        Initialize layer manager.

        Args:
            config: Module configuration dict.
            compression_engine: Compression engine instance.
            di_container: Optional DI container — used to resolve a Redis
                client for MEM-005 durable session persistence.
        """
        self._config = config
        self._compression = compression_engine
        self._di_container = di_container
        # MEM-005: lazy Redis store — DI resolution is async.
        self._redis_store: Optional[MLMRedisSessionStore] = None
        self._redis_store_ready: bool = False

        # Layer 0 parameters
        l0_cfg = config.get("layer0", {})
        self._max_snapshots = int(l0_cfg.get("max_snapshots", 5))
        self._compress_at = int(l0_cfg.get("compress_at", 4))

        # Layer 1 parameters
        l1_cfg = config.get("layer1", {})
        self._max_blocks = int(l1_cfg.get("max_blocks", 4))
        self._batch_size = int(l1_cfg.get("batch_size", 4))

        # Layer 2 parameters
        l2_cfg = config.get("layer2", {})
        self._layer2_enabled = bool(l2_cfg.get("enabled", True))

        # Sub-Layer Zero manager
        self._slz_manager = SubLayerZeroManager(max_snapshots=self._max_snapshots)

        # In-memory session store (acts as hot cache when MEM-005 store is on)
        self._sessions: Dict[str, SessionMemoryState] = {}

    async def _ensure_redis_store(self) -> Optional[MLMRedisSessionStore]:
        if self._redis_store_ready:
            return self._redis_store
        self._redis_store = await maybe_build_mlm_store(self._di_container)
        self._redis_store_ready = True
        return self._redis_store

    async def _persist_session(self, session: SessionMemoryState) -> None:
        store = await self._ensure_redis_store()
        if store is None:
            return
        try:
            await store.save(session.session_id, session.model_dump(mode="json"))
        except Exception as e:
            logger.warning(
                f"[MEM-005] Failed to persist multi-layer session "
                f"{session.session_id}: {e}"
            )

    async def _hydrate_session(self, session_id: str) -> Optional[SessionMemoryState]:
        store = await self._ensure_redis_store()
        if store is None:
            return None
        try:
            payload = await store.load(session_id)
        except Exception as e:
            logger.warning(f"[MEM-005] Failed to hydrate {session_id}: {e}")
            return None
        if payload is None:
            return None
        try:
            return SessionMemoryState.model_validate(payload)
        except Exception as e:
            logger.warning(
                f"[MEM-005] multi-layer session {session_id} payload invalid; "
                f"ignoring ({e})"
            )
            return None

    def _get_or_create_session(self, session_id: str) -> SessionMemoryState:
        """Sync fast-path (kept for API compatibility)."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemoryState(session_id=session_id)
        return self._sessions[session_id]

    async def _async_get_or_create_session(
        self, session_id: str
    ) -> SessionMemoryState:
        if session_id in self._sessions:
            return self._sessions[session_id]
        hydrated = await self._hydrate_session(session_id)
        if hydrated is not None:
            self._sessions[session_id] = hydrated
            return hydrated
        new_session = SessionMemoryState(session_id=session_id)
        self._sessions[session_id] = new_session
        await self._persist_session(new_session)
        return new_session

    async def add_snapshot(
        self, session_id: str, snapshot: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add a new Sub-Layer Zero snapshot.

        Updates Layer 0 sliding window and checks compression trigger.

        Args:
            session_id: Conversation session identifier.
            snapshot: Snapshot data dict.

        Returns:
            Status dict with layer0_count and compression_triggered flag.
        """
        session = await self._async_get_or_create_session(session_id)

        # Convert existing snapshots to dicts for processing
        current_dicts = [s.model_dump() for s in session.layer0]

        # Add snapshot to sliding window
        updated = self._slz_manager.add_snapshot(current_dicts, snapshot)

        # Rebuild Layer 0 from validated dicts
        from .models import SubLayerZeroSnapshot
        session.layer0 = [SubLayerZeroSnapshot(**s) for s in updated]
        session.total_turns = max(
            session.total_turns, snapshot.get("turn", session.total_turns + 1)
        )

        # Check compression trigger
        compression_triggered = False
        if self._slz_manager.should_compress(updated, self._compress_at):
            logger.info(
                f"[LayerManager] Compression trigger reached for session "
                f"{session_id} (snapshots: {len(updated)}/{self._compress_at})"
            )
            result = await self._run_compression(session)
            compression_triggered = result.new_layer1_block is not None

        # MEM-005: write-through after every mutation.
        await self._persist_session(session)

        return {
            "status": "snapshot_added",
            "layer0_count": len(session.layer0),
            "compression_triggered": compression_triggered,
            "total_turns": session.total_turns,
        }

    async def _run_compression(
        self, session: SessionMemoryState
    ) -> CompressionResult:
        """
        Execute compression: Layer 0 → Layer 1 (→ Layer 2).

        Args:
            session: The session to compress.

        Returns:
            CompressionResult.
        """
        # Get snapshots for compression
        snap_dicts = [s.model_dump() for s in session.layer0]
        to_compress = self._slz_manager.get_snapshots_for_compression(
            snap_dicts, self._batch_size
        )

        if not to_compress:
            return CompressionResult()

        # Current Layer 1 as dicts
        l1_dicts = [b.model_dump() for b in session.layer1]
        # Current Layer 2 as dict
        l2_dict = session.layer2.model_dump()

        # Run compression
        result = await self._compression.compress(to_compress, l1_dicts, l2_dict)

        # Apply results
        if result.new_layer1_block:
            session.layer1.append(result.new_layer1_block)
            session.compression_count += 1

            # Enforce Layer 1 max blocks
            if len(session.layer1) > self._max_blocks:
                self._evict_layer1_block(session)

        if result.layer2_updated and result.updated_layer2 and self._layer2_enabled:
            session.layer2 = result.updated_layer2

        return result

    def _evict_layer1_block(self, session: SessionMemoryState) -> None:
        """
        Evict the least important Layer 1 block when max_blocks is exceeded.

        Strategy: remove the block with the lowest importance score.
        Tie-breaker: oldest block (lowest last_updated_turn).
        """
        if not session.layer1:
            return

        # Find block with lowest importance (then oldest)
        min_idx = 0
        min_importance = session.layer1[0].importance
        min_turn = session.layer1[0].last_updated_turn

        for i, block in enumerate(session.layer1[1:], 1):
            if block.importance < min_importance or (
                block.importance == min_importance
                and block.last_updated_turn < min_turn
            ):
                min_idx = i
                min_importance = block.importance
                min_turn = block.last_updated_turn

        evicted = session.layer1.pop(min_idx)
        logger.info(
            f"[LayerManager] Evicted Layer 1 block: "
            f"turns={evicted.turn_range}, importance={evicted.importance}"
        )

    async def trigger_compression(
        self, session_id: str, force: bool = False
    ) -> Dict[str, Any]:
        """
        Manually trigger compression for a session.

        Args:
            session_id: Session identifier.
            force: If True, compress even if threshold not reached.

        Returns:
            Compression result dict.
        """
        session = self._get_or_create_session(session_id)

        snap_dicts = [s.model_dump() for s in session.layer0]
        if not force and not self._slz_manager.should_compress(
            snap_dicts, self._compress_at
        ):
            return {
                "status": "skipped",
                "reason": f"threshold not reached ({len(session.layer0)}/{self._compress_at})",
                "new_layer1_block": None,
                "layer2_updated": False,
            }

        result = await self._run_compression(session)

        return {
            "status": "completed",
            "new_layer1_block": (
                result.new_layer1_block.model_dump()
                if result.new_layer1_block
                else None
            ),
            "layer2_updated": result.layer2_updated,
        }

    def get_memory_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get the full multi-layer memory context for a session.

        Returns all 3 layers formatted for LLM injection.

        Args:
            session_id: Session identifier.

        Returns:
            Dict with layer0, layer1, layer2, and token_estimate.
        """
        session = self._get_or_create_session(session_id)

        from .utils import estimate_tokens_json

        l0 = [s.model_dump() for s in session.layer0]
        l1 = [b.model_dump() for b in session.layer1]
        l2 = session.layer2.model_dump()

        return {
            "layer0": l0,
            "layer1": l1,
            "layer2": l2,
            "total_turns": session.total_turns,
            "compression_count": session.compression_count,
            "token_estimate": (
                estimate_tokens_json(l0)
                + estimate_tokens_json(l1)
                + estimate_tokens_json(l2)
            ),
        }

    def get_layer(self, session_id: str, layer: int) -> Dict[str, Any]:
        """
        Get a specific layer's data for a session.

        Args:
            session_id: Session identifier.
            layer: Layer number (0, 1, or 2).

        Returns:
            Dict with layer number and data.

        Raises:
            ValueError: If layer number is invalid.
        """
        if layer not in (0, 1, 2):
            raise ValueError(f"Invalid layer number: {layer}. Must be 0, 1, or 2.")

        session = self._get_or_create_session(session_id)

        if layer == 0:
            data = [s.model_dump() for s in session.layer0]
        elif layer == 1:
            data = [b.model_dump() for b in session.layer1]
        else:
            data = session.layer2.model_dump()

        return {"layer": layer, "data": data}

    def clear_session(self, session_id: str) -> Dict[str, Any]:
        """
        Clear all memory for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Status dict.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
        return {"status": "cleared", "session_id": session_id}

    def get_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about the memory system.

        Args:
            session_id: If provided, stats for that session. Otherwise, global stats.

        Returns:
            Statistics dict.
        """
        if session_id:
            session = self._sessions.get(session_id)
            if not session:
                return {
                    "session_id": session_id,
                    "exists": False,
                }
            return {
                "session_id": session_id,
                "exists": True,
                "layer0_count": len(session.layer0),
                "layer1_count": len(session.layer1),
                "layer2_populated": bool(
                    session.layer2.critical_facts
                    or session.layer2.core_rules
                    or session.layer2.core_specifications
                ),
                "total_turns": session.total_turns,
                "compression_count": session.compression_count,
            }

        return {
            "active_sessions": len(self._sessions),
            "total_compressions": sum(
                s.compression_count for s in self._sessions.values()
            ),
            "config": {
                "max_snapshots": self._max_snapshots,
                "compress_at": self._compress_at,
                "max_blocks": self._max_blocks,
                "layer2_enabled": self._layer2_enabled,
            },
        }
