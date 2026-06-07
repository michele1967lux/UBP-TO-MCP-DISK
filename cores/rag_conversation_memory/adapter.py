"""
UBP Framework Bridge for Conversation Memory Module

Integrates RedisConversationMemoryProvider with UBP module system.
Provides session-based conversation persistence with Redis backend.

v2.0.0: Structured Memory with topic detection, compression, and decay.
        - Event-driven async compression via memory.message_added
        - Topic shift detection and memory.topic_shifted events
        - Configurable via UBP_MEMORY__* environment variables

ROADMAP v1.5.0 - FEAT-MEM-001
v2.0.0 - FEAT-MEM-002: Structured Memory

MCP-COMPAT (ARCH-008): Added OperationContext support for dual REST/MCP compatibility.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
import uuid
import asyncio
import time

import redis.asyncio as aioredis

from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule
# MCP-COMPAT (ARCH-008): Import OperationContext + canonical normalization layer
# persistence-chain-repair COMMIT-1a/1c
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import (
        OperationContext,
        extract_user_id as _extract_user_id,
        extract_client_id as _extract_client_id,
        normalize_ctx as _canonical_normalize_ctx,
        InvalidContextError,
    )
except ModuleNotFoundError:
    from _shared.operation_context import (  # type: ignore[no-redef]
        OperationContext,
        extract_user_id as _extract_user_id,
        extract_client_id as _extract_client_id,
        normalize_ctx as _canonical_normalize_ctx,
        InvalidContextError,
    )

from .providers import RedisConversationMemoryProvider
from .models import MemoryState, ContextResult, ToolUsageEntry
from .context_manager import ContextManager
from .query_rewriter import QueryRewriter, HintsBuilder

logger = logging.getLogger(__name__)


class MemoryConcurrencyError(RuntimeError):
    """Raised when a CAS-protected memory state save fails (MEM-001).

    Indicates that another process modified the session state between our
    read and write. Callers should retry the operation; do **not** ignore
    this error — silently swallowing it was the original MEM-001 bug.
    """


def _log_compression_task_error(task: asyncio.Task) -> None:
    """Callback for recompress background tasks — logs unhandled exceptions."""
    if not task.done() or task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"[MEMORY] Background recompress task failed: {exc}", exc_info=exc)


class _RecompressGuard:
    """Prevent recursive recompress spawns for the same session."""
    _active: set = set()

    @classmethod
    def try_acquire(cls, session_id: str) -> bool:
        if session_id in cls._active:
            return False
        cls._active.add(session_id)
        return True

    @classmethod
    def release(cls, session_id: str) -> None:
        cls._active.discard(session_id)


class ConversationMemoryAdapter(BaseHybridModule):
    """
    UBP adapter for conversation memory management.

    Provides session-based conversation persistence with Redis backend.
    Follows the 3-file pattern: adapter.py (this file) + providers.py + __init__.py

    v2.0.0: Structured Memory support with:
    - Topic detection and tracking
    - Progressive context compression
    - Event-driven async processing
    """

    def __init__(self, module_path: Path, **kwargs):
        super().__init__(module_path, **kwargs)
        self.provider: Optional[RedisConversationMemoryProvider] = None
        self.total_sessions_created = 0
        self.total_messages_added = 0
        self._init_status: Dict[str, Any] = {"status": "not_initialized"}

        # v2.0: Structured Memory components
        self.context_manager: Optional[ContextManager] = None
        self._structured_enabled: bool = False
        self._memory_settings: Optional[Any] = None
        self._llm_provider: Optional[Any] = None

        # v5.0: Query Rewriter (FEAT-MEM-003)
        self.query_rewriter: Optional[QueryRewriter] = None

    # ========================================================================
    # MCP-COMPAT: OperationContext helpers (ARCH-008)
    # ========================================================================

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
        """Normalize any context format to OperationContext.

        Delegates to the canonical normalize_ctx() boundary from _shared.
        persistence-chain-repair COMMIT-1c: replaced isinstance-based
        boilerplate with canonical duck-type normalization.

        See operation_context.py RFC for the full contract and rationale.
        """
        if ctx is None:
            return self._build_context_from_di()
        return _canonical_normalize_ctx(ctx)

    async def initialize(self) -> None:
        """Initialize module and provider."""
        logger.info(f"Initializing {self.manifest.name}")

        # Get Redis from DI container
        redis_client = None
        if self.di_container:
            try:
                redis_client = await self.di_container.resolve(aioredis.Redis)
                logger.info("✅ Redis client resolved from DI container")
            except Exception as e:
                logger.warning(f"Could not resolve Redis from DI: {e}")

        if not redis_client:
            logger.error("Redis client not available - conversation memory disabled")
            self._init_status = {
                "status": "degraded",
                "reason": "Redis client not available",
            }
            return

        # Initialize provider with config
        self.provider = RedisConversationMemoryProvider(
            redis_client=redis_client,
            default_ttl=self.config.get("session_ttl_seconds", 86400 * 90),
            max_messages_per_session=self.config.get("max_messages_per_session", 100),
            max_sessions_per_user=self.config.get("max_sessions_per_user", 150),
        )

        logger.info(f"✅ {self.manifest.name} initialized with Redis backend")

        # v5.0: Initialize Query Rewriter (FEAT-MEM-003)
        self.query_rewriter = QueryRewriter(redis_client=redis_client)
        logger.info("✅ QueryRewriter initialized")

        # v2.0: Initialize Structured Memory if enabled
        await self._init_structured_memory()

        self._init_status = {
            "status": "healthy",
            "provider": "redis",
            "config": {
                "session_ttl_days": self.config.get("session_ttl_seconds", 604800)
                // 86400,
                "max_messages": self.config.get("max_messages_per_session", 100),
            },
            "structured_memory": {
                "enabled": self._structured_enabled,
                "strategy": self._memory_settings.strategy if self._memory_settings else None,
            },
        }

        # v2.0: Subscribe to own message_added event for async compression
        # Note: event_bus.subscribe() is synchronous, not async
        if self._structured_enabled and self.event_bus:
            try:
                self.event_bus.subscribe(
                    "memory.message_added",
                    self._on_message_added
                )
                logger.info("✅ Subscribed to 'memory.message_added' for async compression")
            except Exception as e:
                logger.error(f"Failed to subscribe to memory events: {e}")
        elif self._structured_enabled:
            logger.warning("⚠️ Structured Memory enabled but EventBus not available!")

    async def shutdown(self) -> None:
        """Shutdown module."""
        logger.info(f"Shutting down {self.manifest.name}")
        self.provider = None
        logger.info(f"✅ {self.manifest.name} shutdown complete")

    async def call_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Dispatch operation calls to appropriate methods."""
        operation_map = {
            "create_session": self.create_session,
            "get_history": self.get_history,
            "add_message": self.add_message,
            "clear_session": self.clear_session,
            "delete_session": self.delete_session,
            "list_sessions": self.list_sessions,
            "get_context_for_llm": self.get_context_for_llm,
            "health_check": self.health_check,
            "get_structured_context": self.get_structured_context,
            "get_suggested_lane": self.get_suggested_lane,
            "set_lane_signal": self.set_lane_signal,
            "rewrite_query": self.rewrite_query,
            "set_context_pending": self.set_context_pending,
            "wait_for_context_ready": self.wait_for_context_ready,
            "record_turn_and_prepare_context": self.record_turn_and_prepare_context,
            "search_history": self.search_history,
        }
        method = operation_map.get(operation)
        if not method:
            raise ValueError(f"Unknown operation: {operation}")
        return await method(**kwargs)

    async def health_check(self, **kwargs) -> Dict[str, Any]:
        """Perform health check."""
        health = {
            "module": self.manifest.name,
            "status": "healthy" if self.provider else "unhealthy",
            "init_status": self._init_status,
            "total_sessions_created": self.total_sessions_created,
            "total_messages_added": self.total_messages_added,
        }

        if self.provider:
            health["provider"] = self.provider.health_check()

        return health

    # === OPERATIONS (mapped from manifest.json) ===

    async def create_session(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a new conversation session.

        User ID and Client ID are extracted from security context (ctx), NOT from parameters.
        This ensures client isolation (RULE-001, RULE-006).

        Args:
            metadata: Optional session metadata
            source: Session source tag (e.g. "architect", "chat")
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        # Get user_id from context (SECURITY: never from payload)
        user_id = self._get_user_id_from_ctx(ctx)
        if not user_id:
            return {"error": "User not authenticated", "request_id": request_id}

        # Get client_id from context (RULE-001: automatic propagation)
        client_id = self._get_client_id_from_ctx(ctx)

        logger.info(
            f"Creating session for user {user_id} (client: {client_id})",
            extra={"request_id": request_id},
        )

        # Merge provided metadata with security context
        session_metadata = metadata.copy() if metadata else {}
        session_metadata["client_id"] = client_id  # RULE-006: Store for filtering

        session_id = await self.provider.create_session(
            user_id=user_id, metadata=session_metadata, source=source
        )

        self.total_sessions_created += 1

        return {
            "session_id": session_id,
            "user_id": user_id,
            "client_id": client_id,
            "status": "created",
            "request_id": request_id,
        }

    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get conversation history for a session."""
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        # Verify user owns this session
        user_id = self._get_user_id_from_ctx(ctx)
        if not await self._user_owns_session(user_id, session_id, ctx):
            return {"error": "Access denied", "request_id": request_id}

        messages = await self.provider.get_history(session_id, limit)

        return {
            "session_id": session_id,
            "messages": messages,
            "count": len(messages),
            "request_id": request_id,
        }

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        emit_event: bool = True,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add a message to a conversation session."""
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        # Validate role
        if role not in ("user", "assistant"):
            return {
                "error": "Invalid role. Must be 'user' or 'assistant'",
                "request_id": request_id,
            }

        # Verify user owns this session
        user_id = self._get_user_id_from_ctx(ctx)
        if not await self._user_owns_session(user_id, session_id, ctx):
            # v4.1.0: Auto-create session if it doesn't exist yet
            # This supports the flow where rag_orchestrator creates conversation_id
            # via ConversationManager but structured memory session doesn't exist yet.
            if user_id:
                session_meta = await self.provider.get_session_metadata(session_id)
                if not session_meta:
                    # Propagate source from message metadata (e.g. "architect")
                    source = (metadata or {}).get("route", "chat")
                    await self._auto_create_session(session_id, user_id, ctx, source=source)
                else:
                    return {"error": "Access denied", "request_id": request_id}
            else:
                return {"error": "Access denied", "request_id": request_id}

        message = await self.provider.add_message(
            session_id=session_id, role=role, content=content, metadata=metadata
        )

        self.total_messages_added += 1

        # Publish event for other modules
        if emit_event and hasattr(self, "publisher") and self.publisher:
            await self.publisher.publish(
                "memory.message_added",
                {
                    "session_id": session_id,
                    "message_id": message["message_id"],
                    "role": role,
                    "content": content,
                    "user_id": user_id,
                    "request_id": request_id,
                },
            )

        return {"message": message, "status": "added", "request_id": request_id}

    async def clear_session(
        self, session_id: str, request_id: Optional[str] = None, ctx=None, **kwargs
    ) -> Dict[str, Any]:
        """Clear all messages in a session."""
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        # Verify user owns this session
        user_id = self._get_user_id_from_ctx(ctx)
        if not await self._user_owns_session(user_id, session_id, ctx):
            return {"error": "Access denied", "request_id": request_id}

        await self.provider.clear_session(session_id)

        return {"session_id": session_id, "status": "cleared", "request_id": request_id}

    async def delete_session(
        self, session_id: str, request_id: Optional[str] = None, ctx=None, **kwargs
    ) -> Dict[str, Any]:
        """Delete a session completely."""
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        # Verify user owns this session
        user_id = self._get_user_id_from_ctx(ctx)
        if not await self._user_owns_session(user_id, session_id, ctx):
            return {"error": "Access denied", "request_id": request_id}

        await self.provider.delete_session(session_id)

        return {"session_id": session_id, "status": "deleted", "request_id": request_id}

    async def list_sessions(
        self, limit: int = 20, source_filter: Optional[str] = None,
        request_id: Optional[str] = None, ctx=None, **kwargs
    ) -> Dict[str, Any]:
        """
        List sessions for the current user (or client for admins).

        Security:
        - Regular users see only their own sessions
        - Admins see all sessions within their client (RULE-006)
        - Sessions are always filtered by client_id for isolation (RULE-008)

        Args:
            limit: Max sessions to return
            source_filter: Optional filter by session source (e.g. "architect", "chat")
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        user_id = self._get_user_id_from_ctx(ctx)
        if not user_id:
            return {"error": "User not authenticated", "request_id": request_id}

        client_id = self._get_client_id_from_ctx(ctx)
        is_admin = self._is_admin(ctx)

        # Over-fetch when filtering to ensure we get enough results after filtering
        fetch_limit = limit * 3 if source_filter else limit
        sessions = await self.provider.list_sessions(user_id, fetch_limit)

        # Filter by source if specified (e.g. "architect")
        if source_filter:
            sessions = [s for s in sessions if s.get("source") == source_filter]

        # Filter by client_id for client isolation (RULE-008)
        # NOTE: list_sessions already returns full metadata via pipeline hgetall,
        # so client_id is always present if it was stored. No need to re-fetch.
        if client_id:
            sessions = [
                s for s in sessions
                if s.get("client_id") == client_id or not s.get("client_id")
            ]

        # Apply final limit
        sessions = sessions[:limit]

        return {"sessions": sessions, "count": len(sessions), "request_id": request_id}

    async def search_history(
        self,
        query: str,
        top_k: int = 5,
        time_range: Optional[str] = None,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Search user's conversation history across past sessions.

        User-scoped, client-scoped, read-only, bounded output.
        Security: user_id and client_id from ctx, never from parameters.
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        user_id = self._get_user_id_from_ctx(ctx)
        if not user_id:
            return {"error": "User not authenticated", "request_id": request_id}

        client_id = self._get_client_id_from_ctx(ctx)

        result = await self.provider.search_history(
            user_id=user_id,
            query=query,
            top_k=top_k,
            time_range=time_range,
            client_id=client_id,
        )

        result["request_id"] = request_id
        return result

    async def get_context_for_llm(
        self,
        session_id: str,
        max_turns: int = 10,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get formatted conversation context for LLM prompts."""
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"context": "", "request_id": request_id}

        # Verify user owns this session
        user_id = self._get_user_id_from_ctx(ctx)
        if not await self._user_owns_session(user_id, session_id, ctx):
            return {"context": "", "error": "Access denied", "request_id": request_id}

        context = await self.provider.get_context_for_llm(session_id, max_turns)

        # Calculate turn_count from history for budget calculation
        turn_count = 1
        try:
            history = await self.provider.get_history(session_id, limit=max_turns * 2)
            if history:
                turn_count = len(history) // 2 + 1
        except Exception as e:
            # MEM-007 v17.15: do not swallow silently. We still fall back
            # to turn_count=1 (preserving caller behaviour), but log so
            # operators can see when history retrieval starts to fail
            # systematically (Redis pressure, deserialization issues).
            logger.warning(
                "[MEM-007] get_history failed for session=%s, falling back "
                "to turn_count=1: %s",
                session_id, e,
            )

        return {
            "session_id": session_id,
            "context": context,
            "turn_count": turn_count,
            "request_id": request_id,
        }

    async def set_context_pending(
        self,
        session_id: str,
        pending: bool = True,
        ttl_seconds: int = 60,
        force: bool = False,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Set per-session context maintenance semaphore (RED/GREEN)."""
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        user_id = self._get_user_id_from_ctx(ctx)
        if not user_id:
            return {"error": "User not authenticated", "request_id": request_id}

        session_meta = await self.provider.get_session_metadata(session_id)
        if session_meta and not await self._user_owns_session(user_id, session_id, ctx):
            return {"error": "Access denied", "request_id": request_id}

        updated = await self.provider.set_compression_pending(
            session_id=session_id,
            pending=pending,
            ttl_seconds=ttl_seconds,
            force=force,
            owner_id=request_id,
        )

        return {
            "session_id": session_id,
            "pending": bool(pending),
            "updated": bool(updated),
            "request_id": request_id,
        }

    async def wait_for_context_ready(
        self,
        session_id: str,
        timeout_seconds: float = 65.0,
        poll_interval_ms: int = 100,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Wait until per-session context semaphore becomes GREEN."""
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"ready": True, "waited_ms": 0.0, "request_id": request_id}

        user_id = self._get_user_id_from_ctx(ctx)
        if not user_id:
            return {"error": "User not authenticated", "request_id": request_id}

        session_meta = await self.provider.get_session_metadata(session_id)
        if session_meta and not await self._user_owns_session(user_id, session_id, ctx):
            return {"error": "Access denied", "request_id": request_id}

        start = time.perf_counter()

        # Fast path: check immediately before subscribing
        pending = await self.provider.is_compression_pending(session_id)
        if not pending:
            waited_ms = (time.perf_counter() - start) * 1000
            return {
                "ready": True,
                "session_id": session_id,
                "waited_ms": waited_ms,
                "request_id": request_id,
            }

        # Slow path: subscribe to release channel and wait
        channel = f"ubp:memory:semaphore:release:{session_id}"
        pubsub = self.provider.redis.pubsub()

        try:
            await pubsub.subscribe(channel)

            deadline = start + timeout_seconds
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return {
                        "ready": False,
                        "session_id": session_id,
                        "waited_ms": (time.perf_counter() - start) * 1000,
                        "timeout_seconds": timeout_seconds,
                        "request_id": request_id,
                    }

                # Wait for message (max 2s per iteration as safety fallback)
                try:
                    msg = await asyncio.wait_for(
                        pubsub.get_message(
                            ignore_subscribe_messages=True,
                            timeout=min(remaining, 2.0),
                        ),
                        timeout=min(remaining, 2.5),
                    )
                except asyncio.TimeoutError:
                    msg = None

                if msg and msg.get("type") == "message":
                    # Release signal received — verify semaphore actually cleared
                    if not await self.provider.is_compression_pending(session_id):
                        waited_ms = (time.perf_counter() - start) * 1000
                        return {
                            "ready": True,
                            "session_id": session_id,
                            "waited_ms": waited_ms,
                            "request_id": request_id,
                        }

                # Fallback: re-check pending (handles race where publish < subscribe)
                if msg is None:
                    if not await self.provider.is_compression_pending(session_id):
                        waited_ms = (time.perf_counter() - start) * 1000
                        return {
                            "ready": True,
                            "session_id": session_id,
                            "waited_ms": waited_ms,
                            "request_id": request_id,
                        }
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    async def record_turn_and_prepare_context(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        lock_already_acquired: bool = False,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Atomically record user+assistant turn and prepare context for next query.
        Returns ACK only after context maintenance is complete.
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        if not session_id:
            return {"error": "Missing session_id", "request_id": request_id}

        try:
            # Record turn messages without emitting fire-and-forget compression events.
            await self.add_message(
                session_id=session_id,
                role="user",
                content=user_query,
                metadata={"route": "chat"},
                emit_event=False,
                ctx=ctx,
            )
            await self.add_message(
                session_id=session_id,
                role="assistant",
                content=assistant_answer,
                metadata={"route": "chat"},
                emit_event=False,
                ctx=ctx,
            )

            # If structured memory is disabled, release RED immediately after turn save.
            if not self._structured_enabled or not self.context_manager:
                if lock_already_acquired:
                    await self.provider.set_compression_pending(session_id, False, owner_id=request_id)
                # FIX-RECORD-TURN-READGATE-001: emetti marker uncapped HINCRBY
                # subito prima del return (after add_message ×2 completed).
                await self._mark_turn_prepared(session_id)
                return {
                    "status": "completed",
                    "session_id": session_id,
                    "structured_memory": False,
                    "request_id": request_id,
                }

            compression_result = await self._run_eager_compression(
                session_id=session_id,
                acquire_lock=not lock_already_acquired,
                release_lock=not lock_already_acquired,
                owner_id=request_id,
                tool_usage=kwargs.get("tool_usage"),
            )
        except Exception:
            if lock_already_acquired:
                await self.provider.set_compression_pending(session_id, False, owner_id=request_id)
            raise

        if compression_result.get("status") == "queued":
            # FIX-RECORD-TURN-READGATE-001: messages persistiti, compression
            # deferred-queued — comunque "prepared" per il gate (i messages
            # sono visibili e basta perché il consumer parta).
            await self._mark_turn_prepared(session_id)
            return {
                "status": "queued",
                "session_id": session_id,
                "request_id": request_id,
            }

        # FIX-RECORD-TURN-READGATE-001: post compression+hints+cache update
        # tutti i milestone completi → marker uncapped per consumer cross-process.
        await self._mark_turn_prepared(session_id)
        return {
            "status": "completed",
            "session_id": session_id,
            "compression": compression_result,
            "tool_usage_persisted": compression_result.get("tool_usage_persisted", 0),
            "request_id": request_id,
        }

    async def _mark_turn_prepared(self, session_id: str) -> None:
        """
        FIX-RECORD-TURN-READGATE-001: HINCRBY atomico del marker
        prepared_turn_count su ubp:memory:session:{sid}:metadata.
        
        Counter UNCAPPED (a differenza di ltrim sui messages) — il consumer
        cross-process (mcp-server agent_loop) attende che questo valore
        raggiunga il proprio expected_recorded_turns prima di leggere
        contesto/hints. Tollerante a over-count (gate `>=`).
        """
        try:
            meta_key = self.provider.SESSION_METADATA_KEY.format(session_id=session_id)
            await self.provider.redis.hincrby(meta_key, "prepared_turn_count", 1)
            await self.provider.redis.expire(meta_key, self.provider.default_ttl)
        except Exception as e:
            logger.warning(
                f"[MEMORY] _mark_turn_prepared failed for {session_id}: {e}"
            )

    # === HELPER METHODS ===

    async def _auto_create_session(
        self, session_id: str, user_id: str, ctx,
        source: Optional[str] = None,
    ) -> None:
        """
        v4.1.0: Auto-create a memory session for a given session_id.

        When rag_orchestrator creates conversation_id via ConversationManager,
        the structured memory session doesn't exist yet. This creates it
        so add_message() can store messages and trigger eager compression.

        Args:
            session_id: Session identifier
            user_id: User identifier
            ctx: Security context
            source: Session source tag (e.g. "architect", "chat")
        """
        client_id = self._get_client_id_from_ctx(ctx) or ""

        await self.provider.create_session_with_id(
            session_id=session_id,
            user_id=user_id,
            metadata={"client_id": client_id, "auto_created": "true"},
            source=source,
        )

        logger.info(f"[MEMORY] Auto-created session {session_id} for user {user_id} (source={source})")

    def _get_user_id_from_ctx(self, ctx) -> Optional[str]:
        """Extract user_id from any context format.

        Delegates to the canonical extract_user_id() from _shared.operation_context.
        persistence-chain-repair COMMIT-1c: replaced duck-type copy with
        canonical delegation.  This resolves the dual import identity split
        that caused O2 session-link failures on the MCP path.

        See operation_context.py RFC for the full rationale.
        """
        return _extract_user_id(ctx)

    def _get_client_id_from_ctx(self, ctx) -> Optional[str]:
        """Extract client_id from any context format.

        Delegates to the canonical extract_client_id() from _shared.operation_context.
        persistence-chain-repair COMMIT-1c: replaced duck-type copy with
        canonical delegation.  Mirrors _get_user_id_from_ctx.

        See operation_context.py RFC for the full rationale.
        """
        return _extract_client_id(ctx)

    def _resolve_memory_client_id(
        self,
        ctx,
        *,
        client_id_override: Optional[str] = None,
        op: str = "read",
    ) -> Optional[str]:
        """Resolve effective client_id, honouring sub-agent ACL policy.

        Phase 5.5/D7 closure (Phase 11): when the caller passes
        ``client_id_override`` (e.g. a parent agent reaching into a
        child sub-agent's memory), we delegate to
        :func:`subagent_memory_policy.resolve_effective_client_id` to
        enforce the parent-read-only rule.

        Backward-compat: when ``client_id_override`` is None we behave
        exactly like :meth:`_get_client_id_from_ctx`.
        """
        actor = self._get_client_id_from_ctx(ctx)
        if not client_id_override:
            return actor
        if not actor:
            return client_id_override  # unauthenticated path: no policy
        try:
            from ubp_enterprise_hybrid.mcp_runtime.core.subagent_memory_policy import (
                MemoryOp, resolve_effective_client_id,
            )
        except Exception:
            # subagent module unavailable in this build — pass-through
            return client_id_override
        op_enum = MemoryOp(op) if op in {"read", "write", "delete"} else MemoryOp.READ
        decision = resolve_effective_client_id(
            actor_client_id=actor,
            override_client_id=client_id_override,
            op=op_enum,
        )
        return decision.value

    def _is_admin(self, ctx) -> bool:
        """Check if user is admin."""
        if ctx and hasattr(ctx, "user") and ctx.user:
            # Support both callable is_admin() and boolean attribute
            is_admin_attr = getattr(ctx.user, "is_admin", None)
            if callable(is_admin_attr):
                return bool(is_admin_attr())
            return bool(is_admin_attr)
        return False

    async def _user_owns_session(
        self, user_id: Optional[str], session_id: str, ctx
    ) -> bool:
        """
        Verify user owns the session with client isolation (SECURITY).

        Security Rules Applied:
        - RULE-006: Sessions are filtered by client_id
        - RULE-008: Cross-client access denied
        - Admin can access sessions within their client only (unless system admin)
        """
        if not user_id:
            logger.debug(f"[SESSION_CHECK] Denied: no user_id provided")
            return False

        # Check session metadata
        if self.provider:
            meta = await self.provider.get_session_metadata(session_id)
            if not meta:
                logger.debug(
                    f"[SESSION_CHECK] Denied: session {session_id} metadata not found"
                )
                return False

            # Get caller's client_id
            caller_client_id = self._get_client_id_from_ctx(ctx)
            session_client_id = meta.get("client_id")
            session_user_id = meta.get("user_id")

            # DEBUG: Log session ownership check
            logger.debug(
                f"[SESSION_CHECK] session_id={session_id}, "
                f"caller_user_id={user_id}, session_user_id={session_user_id}, "
                f"caller_client_id={caller_client_id}, session_client_id={session_client_id}"
            )

            # RULE-008: Cross-client access denied (even for admin)
            # Exception: system-level admin (no client_id) can access all
            # BUG-USER-002 FIX: Also allow if session has no client_id (legacy sessions)
            if (
                caller_client_id
                and session_client_id
                and caller_client_id != session_client_id
            ):
                logger.warning(
                    f"Cross-client session access denied: user {user_id} from client {caller_client_id} "
                    f"tried to access session from client {session_client_id}",
                    extra={"session_id": session_id},
                )
                return False

            # Admin within same client can access any session in that client
            if self._is_admin(ctx):
                logger.debug(f"[SESSION_CHECK] Allowed: admin user")
                return True

            # Regular user: must own the session
            owns_session = session_user_id == user_id
            if not owns_session:
                logger.debug(
                    f"[SESSION_CHECK] Denied: user {user_id} doesn't own session "
                    f"(owner: {session_user_id})"
                )
            return owns_session

        logger.debug(f"[SESSION_CHECK] Denied: no provider available")
        return False

    # =========================================================================
    # v2.0: STRUCTURED MEMORY METHODS
    # =========================================================================

    async def _init_structured_memory(self) -> None:
        """
        Initialize Structured Memory v2.0 components.

        Loads settings, resolves LLM provider, and creates ContextManager.
        Falls back to v1.0 simple memory if disabled or unavailable.
        """
        try:
            # Load MemorySettings directly from env vars
            # NOTE: We instantiate MemorySettings() directly instead of via
            # settings.memory because build_settings() passes model_dump()
            # which overrides nested env vars with default values.
            try:
                from ubp_enterprise_hybrid.backend.app.core.config import MemorySettings
                self._memory_settings = MemorySettings()
                self._structured_enabled = self._memory_settings.structured_enabled
                logger.info(f"MemorySettings loaded: structured_enabled={self._structured_enabled}")
            except Exception as e:
                logger.warning(f"Could not load MemorySettings: {e}")
                self._structured_enabled = False

            if not self._structured_enabled:
                logger.info("Structured Memory v2.0 disabled, using simple memory v1.0")
                return

            # v2.0.1: LAZY LLM RESOLUTION
            # Don't resolve LLM provider here - pass di_container to ContextManager
            # which will resolve the LLM lazily on first compression.
            # This fixes the race condition where memory initializes before inference modules.
            llm_provider_name = self._memory_settings.llm_provider
            logger.info(f"[MEMORY] Configured LLM provider: {llm_provider_name} (will resolve lazily)")

            # Initialize ContextManager with di_container for lazy resolution
            self.context_manager = ContextManager(
                di_container=self.di_container,
                provider_name=llm_provider_name,
                settings=self._memory_settings
            )

            logger.info(
                f"✅ Structured Memory v2.0 enabled "
                f"(strategy: {self._memory_settings.strategy}, "
                f"llm: {llm_provider_name or 'none'} [lazy])"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Structured Memory: {e}")
            self._structured_enabled = False

    async def _on_message_added(self, event) -> None:
        """
        Event handler for memory.message_added.

        v4.1.0 EAGER COMPRESSION:
        - Triggers on every 'assistant' message (not just buffer overflow)
        - Always updates structured state (narrative, topics, entities)
        - Removes oldest messages if buffer overflows
        - Pre-caches formatted context for next query
        """
        payload = event.payload if hasattr(event, 'payload') else event
        logger.debug(f"[MEMORY] _on_message_added triggered with payload: {payload}")

        if not self._structured_enabled or not self.context_manager:
            return

        session_id = payload.get("session_id") if isinstance(payload, dict) else None
        if not session_id:
            logger.warning("[MEMORY] No session_id in payload")
            return

        # v4.1.0: Only compress after assistant replies (= complete turn)
        role = payload.get("role") if isinstance(payload, dict) else None
        if role != "assistant":
            logger.debug(f"[MEMORY] Skipping compression for role={role}")
            return

        await self._run_eager_compression(session_id=session_id, acquire_lock=True, release_lock=True)

    async def _run_eager_compression(
        self,
        session_id: str,
        acquire_lock: bool = True,
        release_lock: bool = True,
        owner_id: Optional[str] = None,
        tool_usage: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run structured-memory compression and cache refresh for one session."""
        if not self.provider:
            return {"status": "error", "reason": "provider_not_initialized"}
        if not self._structured_enabled or not self.context_manager:
            return {"status": "skipped", "reason": "structured_memory_disabled"}

        lock_acquired = not acquire_lock
        # MEM-006: require an explicit owner_id when acquiring a lock.
        # The previous code silently invented a uuid4 if the caller didn't
        # pass one, which made it impossible to correlate locks to
        # request/worker identity in logs and broke owner-validated
        # release. We still allow None when acquire_lock=False (callers
        # that already hold the lock).
        if acquire_lock and not owner_id:
            logger.warning(
                "[MEMORY][MEM-006] _run_eager_compression called without "
                "owner_id; falling back to anonymous uuid (caller should "
                "provide a request-scoped identifier)"
            )
        _lock_owner = owner_id or f"anon-{uuid.uuid4()}"
        try:
            if acquire_lock:
                is_pending = await self.provider.is_compression_pending(session_id)
                if is_pending:
                    await self.provider.mark_recompress_requested(session_id)
                    logger.debug(f"[MEMORY] Compression already pending for session {session_id}; marked requeue")
                    return {"status": "queued"}

                lock_acquired = await self.provider.set_compression_pending(
                    session_id, True, owner_id=_lock_owner
                )
                if not lock_acquired:
                    await self.provider.mark_recompress_requested(session_id)
                    logger.debug(f"[MEMORY] Could not acquire compression lock for session {session_id}; marked requeue")
                    return {"status": "queued"}

            # v3.7.2: Parallel fetch state + history (independent Redis calls)
            state_json, messages = await asyncio.gather(
                self.provider.get_state(session_id),
                self.provider.get_history(session_id),
            )
            current_state = MemoryState.from_json(state_json) if state_json else MemoryState()
            if not messages:
                return {"status": "completed", "session_id": session_id, "messages": 0}

            raw_buffer_size = self._memory_settings.raw_buffer_size if self._memory_settings else 10

            # Send only last complete turn (assistant + preceding user)
            last_assistant_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "assistant":
                    last_assistant_idx = i
                    break

            if last_assistant_idx == -1:
                new_messages = messages[-1:] if messages else []
            else:
                start = max(0, last_assistant_idx - 1)
                new_messages = messages[start:]

            # Eager compression with retry
            max_retries = 2
            new_state = None
            topic_shifted = False

            for attempt in range(max_retries + 1):
                try:
                    new_state, topic_shifted = await self.context_manager.check_and_compress(
                        current_state=current_state,
                        new_messages=new_messages,
                        force=True,
                    )
                    break
                except Exception as comp_err:
                    if attempt < max_retries:
                        logger.warning(
                            f"[MEMORY] Compression attempt {attempt + 1}/{max_retries + 1} "
                            f"failed for {session_id}: {comp_err}, retrying..."
                        )
                        await asyncio.sleep(0.5 * (attempt + 1))
                    else:
                        logger.error(
                            f"[MEMORY] Compression failed after {max_retries + 1} attempts "
                            f"for {session_id}: {comp_err}"
                        )
                        return {"status": "error", "reason": str(comp_err)}

            # Attach tool_usage to newest turn (post-compression, pre-save)
            tool_usage_persisted = 0
            if tool_usage and new_state and new_state.conversation_thread:
                try:
                    newest_turn = max(
                        new_state.conversation_thread, key=lambda t: t.turn_number
                    )
                    validated_entries = []
                    for raw_entry in tool_usage[:5]:
                        try:
                            validated_entries.append(ToolUsageEntry(**raw_entry))
                        except Exception as ve:
                            logger.warning(f"[MEMORY] Skipping invalid tool_usage entry: {ve}")
                    newest_turn.tool_usage = validated_entries
                    tool_usage_persisted = len(validated_entries)
                except Exception as tu_err:
                    logger.warning(f"[MEMORY] Failed to attach tool_usage: {tu_err}")

            # Save updated state (CAS-protected — MEM-001).
            # The provider returns False when version mismatch is detected,
            # i.e. another worker compressed in parallel. Previously we
            # ignored that signal and built/cached a stale context on top
            # of overwritten state. Now we raise so the caller can retry.
            saved = await self.provider.save_state(
                session_id,
                new_state.to_json(),
                expected_version=current_state.version,
            )
            if not saved:
                logger.error(
                    f"[MEMORY][MEM-001] CAS failed for session {session_id} "
                    f"(expected_version={current_state.version}); concurrent "
                    f"compression detected — aborting this run"
                )
                raise MemoryConcurrencyError(
                    f"Session {session_id} state was modified by another "
                    f"process (expected_version={current_state.version})"
                )

            # If buffer exceeds size, remove oldest messages
            if len(messages) > raw_buffer_size:
                excess = len(messages) - raw_buffer_size
                await self.provider.remove_oldest_messages(session_id, excess)
                logger.info(f"[MEMORY] Trimmed {excess} old messages from buffer")

            # Build and cache formatted context for next query
            await self._build_and_cache_context(session_id, new_state)

            # Pre-compute retrieval hints
            if self.query_rewriter:
                try:
                    hints = HintsBuilder.build_from_state(new_state)
                    await self.query_rewriter.cache_hints(
                        session_id, hints, ttl_seconds=self.provider.default_ttl
                    )
                    logger.info(f"[MEMORY] Pre-cached retrieval hints: focus={hints.current_focus}")
                except Exception as e:
                    logger.warning(f"[MEMORY] Failed to cache retrieval hints: {e}")

            # Publish topic shift event
            if topic_shifted and hasattr(self, "publisher") and self.publisher:
                await self.publisher.publish(
                    "memory.topic_shifted",
                    {
                        "session_id": session_id,
                        "old_topic": current_state.structured_context.current_topic,
                        "new_topic": new_state.structured_context.current_topic,
                    },
                )

            logger.info(
                f"[MEMORY] Eager compression done for session {session_id}: "
                f"v{new_state.version}, new_msgs={len(new_messages)}, "
                f"thread_size={len(new_state.conversation_thread)}, "
                f"focus={new_state.current_focus}"
                f"{f', tool_usage={tool_usage_persisted}' if tool_usage_persisted else ''}"
            )
            return {
                "status": "completed",
                "session_id": session_id,
                "messages": len(messages),
                "tool_usage_persisted": tool_usage_persisted,
            }

        except Exception as e:
            logger.error(f"[MEMORY] Eager compression error for session {session_id}: {e}")
            return {"status": "error", "reason": str(e)}
        finally:
            if release_lock and lock_acquired:
                # MEM-003: don't silently swallow lock-release failures.
                # If the RPC fails or returns False (e.g. owner mismatch
                # because another worker stole the lock after a TTL
                # expiry), log explicitly and attempt a best-effort
                # fallback delete on the pending key. Worst case the
                # 60s TTL on the lock will recover us.
                pending_key = f"ubp:memory:session:{session_id}:pending"
                try:
                    success = await self.provider.set_compression_pending(
                        session_id, False, owner_id=_lock_owner
                    )
                    if not success:
                        logger.error(
                            f"[MEMORY][MEM-003] Failed to release compression "
                            f"lock for {session_id} (owner_id={_lock_owner}); "
                            f"attempting fallback delete on {pending_key}"
                        )
                        try:
                            await self.provider.redis.delete(pending_key)
                            logger.info(
                                f"[MEMORY][MEM-003] Fallback delete OK for {session_id}"
                            )
                        except Exception as fb_err:
                            logger.error(
                                f"[MEMORY][MEM-003] Fallback delete failed for "
                                f"{session_id}: {fb_err} — relying on lock TTL"
                            )
                except Exception as lock_err:
                    logger.error(
                        f"[MEMORY][MEM-003] Lock release exception for "
                        f"{session_id}: {lock_err} — attempting fallback delete"
                    )
                    try:
                        await self.provider.redis.delete(pending_key)
                        logger.info(
                            f"[MEMORY][MEM-003] Exception-path fallback delete "
                            f"OK for {session_id}"
                        )
                    except Exception as fb_err:
                        logger.error(
                            f"[MEMORY][MEM-003] Exception-path fallback also "
                            f"failed for {session_id}: {fb_err} — relying on "
                            f"lock TTL"
                        )

                try:
                    if await self.provider.consume_recompress_requested(session_id):
                        if _RecompressGuard.try_acquire(session_id):
                            logger.info(f"[MEMORY] Re-running queued compression for {session_id}")

                            async def _guarded_recompress(_sid: str) -> Dict[str, Any]:
                                try:
                                    return await self._run_eager_compression(
                                        session_id=_sid,
                                        acquire_lock=True,
                                        release_lock=True,
                                    )
                                finally:
                                    _RecompressGuard.release(_sid)

                            _task = asyncio.create_task(
                                _guarded_recompress(session_id),
                                name=f"memory_recompress_{session_id[:12]}",
                            )
                            _task.add_done_callback(_log_compression_task_error)
                        else:
                            logger.debug(f"[MEMORY] Recompress already running for {session_id}, skipping")
                except Exception as e:
                    logger.warning(f"[MEMORY] Failed to process recompress queue for {session_id}: {e}")

    async def _build_and_cache_context(self, session_id: str, state: MemoryState) -> None:
        """
        Build formatted context string and cache it in Redis.
        Called after every eager compression to pre-compute context for next query.
        """
        try:
            raw_buffer_size = self._memory_settings.raw_buffer_size if self._memory_settings else 10
            raw_messages = await self.provider.get_history(session_id, limit=raw_buffer_size)

            context_result = self.context_manager.build_context_result(
                state=state,
                raw_messages=raw_messages
            )

            # Format the context string (same as get_system_message())
            cached_text = context_result.get_system_message()

            # Save to Redis
            await self.provider.save_cached_context(session_id, cached_text)

            logger.debug(f"[MEMORY] Pre-cached context for session {session_id}: {len(cached_text)} chars")

        except Exception as e:
            logger.warning(f"[MEMORY] Failed to cache context for {session_id}: {e}")

    async def get_structured_context(
        self,
        session_id: str,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get conversation context with structured memory (v2.0).

        Returns both raw recent messages and compressed context summary.
        Used by RAG orchestrator for enhanced context injection.

        Args:
            session_id: Session identifier
            request_id: Request tracking ID
            ctx: Security context

        Returns:
            Dict with raw_messages, narrative_summary, structured_context, etc.
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        # Verify access
        user_id = self._get_user_id_from_ctx(ctx)
        if not await self._user_owns_session(user_id, session_id, ctx):
            return {"error": "Access denied", "request_id": request_id}

        # Get raw buffer messages
        raw_buffer_size = (
            self._memory_settings.raw_buffer_size
            if self._memory_settings else 10
        )
        raw_messages = await self.provider.get_history(session_id, limit=raw_buffer_size)

        # If structured memory not enabled, return simple result
        if not self._structured_enabled:
            return {
                "session_id": session_id,
                "raw_messages": raw_messages,
                "has_structured_context": False,
                "request_id": request_id,
            }

        # v4.1.0: Try cached context first (pre-computed by eager compression)
        cached = await self.provider.get_cached_context(session_id)
        if cached:
            return {
                "session_id": session_id,
                "raw_messages": raw_messages,
                "has_structured_context": True,
                "system_message": cached,
                "from_cache": True,
                "request_id": request_id,
            }

        # Fallback: build from state (first turn or cache miss)
        state_json = await self.provider.get_state(session_id)
        if not state_json:
            # No state yet, return raw only
            return {
                "session_id": session_id,
                "raw_messages": raw_messages,
                "has_structured_context": False,
                "request_id": request_id,
            }

        try:
            state = MemoryState.from_json(state_json)

            # Build context result
            context_result = self.context_manager.build_context_result(
                state=state,
                raw_messages=raw_messages
            )

            return {
                "session_id": session_id,
                "raw_messages": context_result.raw_messages,
                "narrative_summary": context_result.narrative_summary,
                "structured_context": context_result.structured_context.model_dump(mode='json')
                if context_result.structured_context else None,
                "previous_topics": [t.model_dump(mode='json') for t in context_result.previous_topics],
                "has_structured_context": True,
                "topic_shifting": context_result.topic_shifting,
                "system_message": context_result.get_system_message(),
                "request_id": request_id,
            }

        except Exception as e:
            logger.error(f"Error parsing structured state for session {session_id}: {e}")
            return {
                "session_id": session_id,
                "raw_messages": raw_messages,
                "has_structured_context": False,
                "error": str(e),
                "request_id": request_id,
            }

    async def get_suggested_lane(
        self,
        session_id: str,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Read suggested_lane — micro-update (from pipeline) takes priority over state."""
        if not self.provider:
            return {"suggested_lane": None}

        # Priority 1: micro-update written by pipeline via set_lane_signal()
        try:
            lane_key = f"ubp:mem:lane:{session_id}"
            lane_data = await self.provider.redis.hgetall(lane_key)
            if lane_data:
                # Handle both str and bytes keys (depends on Redis client decode_responses)
                _raw = lane_data.get("suggested_lane") or lane_data.get(b"suggested_lane")
                _lane = _raw.decode() if isinstance(_raw, bytes) else (_raw or None)
                _raw_r = lane_data.get("lane_reason") or lane_data.get(b"lane_reason") or ""
                _reason = _raw_r.decode() if isinstance(_raw_r, bytes) else _raw_r
                if _lane:
                    logger.info("[MEMORY] lane micro-update hit: %s (session=%s)", _lane, session_id)
                    return {
                        "suggested_lane": _lane,
                        "lane_reason": _reason,
                        "source": "pipeline",
                    }
        except Exception as e:
            logger.info("[MEMORY] lane micro-update read FAILED: %s", e)

        # Priority 2: legacy — from compression state thread
        state_json = await self.provider.get_state(session_id)
        if not state_json:
            return {"suggested_lane": None}
        try:
            state = MemoryState.from_json(state_json)
            if not state.conversation_thread:
                return {"suggested_lane": None}
            latest = max(state.conversation_thread, key=lambda t: t.turn_number)
            return {
                "suggested_lane": getattr(latest, 'suggested_lane', None),
                "previous_lane": getattr(latest, 'previous_lane', None),
                "lane_reason": getattr(latest, 'lane_reason', ''),
                "source": "compression",
            }
        except Exception as e:
            logger.debug(f"[MEMORY] get_suggested_lane failed for {session_id}: {e}")
            return {"suggested_lane": None}

    async def set_lane_signal(
        self,
        session_id: str,
        lane: str,
        reason: str = "",
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Write lane_signal as atomic micro-update (no full state R/W).

        Called by user_router after pipeline execution to persist the
        lane signal derived from tools_used in reasoning_loop.
        TTL=3600s — expires if no follow-up query within 1 hour.
        """
        if not self.provider or not hasattr(self.provider, 'redis'):
            return {"status": "skipped", "reason": "no_provider"}
        try:
            lane_key = f"ubp:mem:lane:{session_id}"
            if not lane:
                await self.provider.redis.delete(lane_key)
                return {"status": "cleared"}
            await self.provider.redis.hset(lane_key, mapping={
                "suggested_lane": lane,
                "lane_reason": reason,
            })
            await self.provider.redis.expire(lane_key, 3600)
            return {"status": "updated", "lane": lane}
        except Exception as e:
            logger.debug("[MEMORY] set_lane_signal failed: %s", e)
            return {"status": "error", "reason": str(e)}

    async def rewrite_query(
        self,
        session_id: str,
        query: str,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        v5.0 FEAT-MEM-003: Rewrite a vague/continuation query using cached retrieval hints.

        Zero-latency: uses pre-computed hints from Redis (~1ms), no LLM call.

        Args:
            session_id: Session identifier
            query: Raw user query
            request_id: Request tracking ID
            ctx: Security context

        Returns:
            Dict with query, original_query, rewrite_type, hints_used, metadata
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"query": query, "original_query": query, "rewrite_type": "error",
                    "hints_used": False, "metadata": {"reason": "provider_not_initialized"},
                    "request_id": request_id}

        if not self.query_rewriter:
            return {"query": query, "original_query": query, "rewrite_type": "none",
                    "hints_used": False, "metadata": {"reason": "rewriter_not_initialized"},
                    "request_id": request_id}

        try:
            result = await self.query_rewriter.rewrite(
                session_id=session_id,
                raw_query=query,
            )
            result["request_id"] = request_id
            return result
        except Exception as e:
            logger.warning(f"[REWRITER] Query rewrite failed for session {session_id}: {e}")
            return {"query": query, "original_query": query, "rewrite_type": "error",
                    "hints_used": False, "metadata": {"reason": str(e)},
                    "request_id": request_id}
