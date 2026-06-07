"""
Conversation Memory Provider - Pure Technical Logic

Zero UBP dependencies. Can be tested standalone.
Implements Redis-based conversation persistence.

Redis Key Patterns (NAMING_POLICY.md Section 7):
- ubp:memory:session:{session_id}:messages  (List of JSON messages)
- ubp:memory:session:{session_id}:metadata  (Hash with session info)
- ubp:memory:user:{user_id}:sessions        (Sorted Set by last_active)
- ubp:memory:session:{session_id}:state     (v2.0: Structured MemoryState JSON)
- ubp:memory:session:{session_id}:pending   (v2.0: Compression lock flag)

ROADMAP v1.5.0 - FEAT-MEM-001
v2.0.0 - FEAT-MEM-002: Structured Memory with Topic Detection
"""

from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from datetime import datetime, timezone
import json
import logging
import uuid

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover — zoneinfo stdlib (py>=3.9)
    ZoneInfo = None  # type: ignore[assignment]
    ZoneInfoNotFoundError = Exception  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _convert_timestamp_for_display(
    ts_iso: Optional[str], tz_name: Optional[str] = None
) -> str:
    """fb1 (B-05 L1) — convert a stored ISO-UTC timestamp for LLM display.

    Storage stays UTC (canonical, see add_message line 322 et al.). At read
    time we localise into the user's timezone so the LLM sees timestamps
    that match the user's context. Falls back to the raw input on any
    parse / zoneinfo error to preserve backward compatibility.

    Args:
        ts_iso: ISO 8601 string (with or without offset). May be None / "".
        tz_name: IANA tz database name (e.g. "Europe/Rome"). None → UTC.

    Returns:
        Localised timestamp `YYYY-MM-DD HH:MM TZNAME`, or original input on
        any failure.
    """
    if not ts_iso:
        return ts_iso or ""
    try:
        # Accept both "...+00:00" and bare "...Z" / naive forms.
        s = ts_iso.replace("Z", "+00:00") if "Z" in ts_iso else ts_iso
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        target_tz: Any = timezone.utc
        if tz_name and tz_name.lower() != "utc" and ZoneInfo is not None:
            try:
                target_tz = ZoneInfo(tz_name)
            except ZoneInfoNotFoundError:
                logger.debug("Unknown tz_name=%r, falling back to UTC", tz_name)
        localised = dt.astimezone(target_tz)
        return localised.strftime("%Y-%m-%d %H:%M %Z").rstrip()
    except (ValueError, TypeError) as e:
        logger.debug("Could not convert timestamp %r: %s", ts_iso, e)
        return ts_iso

# Lua CAS (Compare-And-Swap) script for atomic save_state.
# Eliminates TOCTOU race condition between version check and SET.
# KEYS[1] = state_key, ARGV[1] = expected_version, ARGV[2] = state_json, ARGV[3] = ttl
LUA_CAS_SAVE_STATE = """
local current = redis.call('GET', KEYS[1])
if current then
    local ok, data = pcall(cjson.decode, current)
    if ok and tonumber(data.version) ~= tonumber(ARGV[1]) then
        return 0
    end
end
redis.call('SET', KEYS[1], ARGV[2])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
return 1
"""


@runtime_checkable
class ConversationMemoryProtocol(Protocol):
    """Interface contract for conversation memory providers."""

    async def create_session(
        self, user_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new conversation session. Returns session_id."""
        ...

    async def get_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        ...

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a message to the conversation."""
        ...

    async def clear_session(self, session_id: str) -> bool:
        """Clear all messages in a session."""
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session completely."""
        ...

    async def list_sessions(
        self, user_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """List all sessions for a user."""
        ...


class RedisConversationMemoryProvider:
    """
    Redis-based conversation memory implementation.

    Key Structure (following NAMING_POLICY.md Section 7):
    - ubp:memory:session:{session_id}:messages  (List of JSON messages)
    - ubp:memory:session:{session_id}:metadata  (Hash with session info)
    - ubp:memory:user:{user_id}:sessions        (Sorted Set by last_active)
    """

    # Key patterns following NAMING_POLICY
    KEY_PREFIX = "ubp:memory"
    SESSION_MESSAGES_KEY = f"{KEY_PREFIX}:session:{{session_id}}:messages"
    SESSION_METADATA_KEY = f"{KEY_PREFIX}:session:{{session_id}}:metadata"
    USER_SESSIONS_KEY = f"{KEY_PREFIX}:user:{{user_id}}:sessions"
    # v2.0: Structured Memory keys
    SESSION_STATE_KEY = f"{KEY_PREFIX}:session:{{session_id}}:state"
    SESSION_PENDING_KEY = f"{KEY_PREFIX}:session:{{session_id}}:pending"
    SESSION_RECOMPRESS_KEY = f"{KEY_PREFIX}:session:{{session_id}}:recompress_requested"
    # v4.1.0: Pre-cached formatted context
    SESSION_CACHED_CONTEXT_KEY = f"{KEY_PREFIX}:session:{{session_id}}:cached_context"
    # v5.0: Query rewriter retrieval hints
    SESSION_HINTS_KEY = f"{KEY_PREFIX}:session:{{session_id}}:retrieval_hints"

    def __init__(
        self,
        redis_client,
        default_ttl: int = 86400 * 90,  # 90 days safety net
        max_messages_per_session: int = 100,
        max_sessions_per_user: int = 150,
    ):
        """
        Initialize provider.

        Args:
            redis_client: Async Redis client instance
            default_ttl: Session TTL in seconds (default 90 days, safety net only)
            max_messages_per_session: Max messages to keep per session
            max_sessions_per_user: Max sessions per user (count-based eviction)
        """
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.max_messages = max_messages_per_session
        self.max_sessions_per_user = max_sessions_per_user

    async def create_session(
        self, user_id: str, metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> str:
        """Create a new conversation session with auto-generated ID."""
        session_id = str(uuid.uuid4())
        return await self.create_session_with_id(
            session_id=session_id, user_id=user_id,
            metadata=metadata, source=source,
        )

    async def create_session_with_id(
        self, session_id: str, user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> str:
        """
        Create a session with a pre-existing session_id.

        Used by auto-create flows where session_id is already assigned
        (e.g. rag_orchestrator's ConversationManager).
        """
        now = datetime.now(timezone.utc).isoformat()

        session_meta = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": now,
            "last_active": now,
            "message_count": "0",
            **(metadata or {}),
        }

        if source:
            session_meta["source"] = source

        # Convert complex types to strings
        redis_meta = {}
        for k, v in session_meta.items():
            if isinstance(v, (dict, list)):
                redis_meta[k] = json.dumps(v)
            else:
                redis_meta[k] = str(v)

        # Store metadata
        meta_key = self.SESSION_METADATA_KEY.format(session_id=session_id)
        await self.redis.hset(meta_key, mapping=redis_meta)
        await self.redis.expire(meta_key, self.default_ttl)

        # Add to user's session list (sorted by timestamp)
        # NOTE: No TTL on user sorted set - it's the session index and must persist.
        # Count-based eviction handles cleanup instead.
        user_sessions_key = self.USER_SESSIONS_KEY.format(user_id=user_id)
        await self.redis.zadd(
            user_sessions_key, {session_id: datetime.now(timezone.utc).timestamp()}
        )

        # Count-based eviction: remove oldest sessions if over limit
        evicted = await self._evict_oldest_sessions(user_id)
        if evicted:
            logger.info(f"Evicted {evicted} oldest sessions for user {user_id}")

        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id

    # MEM-002: atomic eviction via Lua. The previous implementation
    # zrange'd the oldest sessions, deleted their keys one by one, then
    # zremrangebyrank'd them — a non-atomic sequence under which a
    # concurrent eviction could observe partial state (sessions present
    # in the user index but with their per-session keys already wiped,
    # or vice-versa). One Lua eval makes the whole sweep atomic.
    _LUA_EVICT_OLDEST = """
local user_key = KEYS[1]
local max_sessions = tonumber(ARGV[1])
local prefix = ARGV[2]

local count = redis.call('zcard', user_key)
if count <= max_sessions then
    return 0
end

local excess = count - max_sessions
local oldest = redis.call('zrange', user_key, 0, excess - 1)

for _, sid in ipairs(oldest) do
    redis.call('del',
        prefix .. ':session:' .. sid .. ':messages',
        prefix .. ':session:' .. sid .. ':metadata',
        prefix .. ':session:' .. sid .. ':state',
        prefix .. ':session:' .. sid .. ':pending',
        prefix .. ':session:' .. sid .. ':recompress_requested',
        prefix .. ':session:' .. sid .. ':cached_context',
        prefix .. ':session:' .. sid .. ':retrieval_hints'
    )
end

redis.call('zremrangebyrank', user_key, 0, excess - 1)
return excess
"""

    async def _evict_oldest_sessions(self, user_id: str) -> int:
        """
        Evict oldest sessions when user exceeds max_sessions_per_user.

        Atomic Lua-backed implementation (MEM-002): the count check,
        per-session key deletion, and zset trim happen in a single
        Redis call — no observer can see a half-evicted state.

        Returns:
            Number of sessions evicted.
        """
        user_sessions_key = self.USER_SESSIONS_KEY.format(user_id=user_id)
        try:
            excess = await self.redis.eval(
                self._LUA_EVICT_OLDEST,
                1,
                user_sessions_key,
                str(self.max_sessions_per_user),
                self.KEY_PREFIX,
            )
        except Exception as e:
            # Fail-OPEN to legacy loop: never let a missing-Lua
            # environment (test fakes, redis with scripting disabled)
            # prevent eviction altogether.
            logger.warning(
                f"[MEMORY][MEM-002] Lua eviction failed for {user_id} "
                f"({e}); falling back to non-atomic loop"
            )
            count = await self.redis.zcard(user_sessions_key)
            if count <= self.max_sessions_per_user:
                return 0
            excess = count - self.max_sessions_per_user
            oldest = await self.redis.zrange(user_sessions_key, 0, excess - 1)
            for session_id_raw in oldest:
                sid = session_id_raw.decode() if isinstance(session_id_raw, bytes) else session_id_raw
                await self.redis.delete(
                    self.SESSION_MESSAGES_KEY.format(session_id=sid),
                    self.SESSION_METADATA_KEY.format(session_id=sid),
                    self.SESSION_STATE_KEY.format(session_id=sid),
                    self.SESSION_PENDING_KEY.format(session_id=sid),
                    self.SESSION_RECOMPRESS_KEY.format(session_id=sid),
                    self.SESSION_CACHED_CONTEXT_KEY.format(session_id=sid),
                    self.SESSION_HINTS_KEY.format(session_id=sid),
                )
            await self.redis.zremrangebyrank(user_sessions_key, 0, excess - 1)

        excess_int = int(excess) if excess is not None else 0
        if excess_int > 0:
            logger.info(
                f"[MEMORY] Count-based eviction: removed {excess_int} oldest "
                f"sessions for user {user_id} (limit: {self.max_sessions_per_user})"
            )
        return excess_int

    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        tz_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session.

        fb1 (B-05 L1): when ``tz_name`` is provided, the stored UTC
        ``timestamp`` of each message is localised in-place so the LLM
        sees the user's timezone. Storage remains UTC.
        """
        messages_key = self.SESSION_MESSAGES_KEY.format(session_id=session_id)

        # Get messages (newest first if limit, then reverse)
        if limit:
            # Get last N messages
            raw_messages = await self.redis.lrange(messages_key, -limit, -1)
        else:
            raw_messages = await self.redis.lrange(messages_key, 0, -1)

        messages = []
        for raw in raw_messages:
            try:
                # Handle bytes
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                msg = json.loads(raw)
                if tz_name and isinstance(msg, dict) and msg.get("timestamp"):
                    msg["timestamp"] = _convert_timestamp_for_display(
                        msg["timestamp"], tz_name
                    )
                messages.append(msg)
            except json.JSONDecodeError:
                logger.warning(f"Invalid message JSON in session {session_id}")
                continue

        return messages

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a message to the conversation."""
        message = {
            "message_id": str(uuid.uuid4()),
            "role": role,  # 'user' or 'assistant'
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        messages_key = self.SESSION_MESSAGES_KEY.format(session_id=session_id)
        meta_key = self.SESSION_METADATA_KEY.format(session_id=session_id)
        cached_key = self.SESSION_CACHED_CONTEXT_KEY.format(session_id=session_id)
        hints_key = self.SESSION_HINTS_KEY.format(session_id=session_id)

        # Add message to list
        await self.redis.rpush(messages_key, json.dumps(message))

        # Trim to max messages (keep most recent)
        await self.redis.ltrim(messages_key, -self.max_messages, -1)

        # Update TTL
        await self.redis.expire(messages_key, self.default_ttl)

        # Update metadata
        await self.redis.hset(meta_key, "last_active", datetime.now(timezone.utc).isoformat())
        await self.redis.hincrby(meta_key, "message_count", 1)

        # Invalidate derived context/hints to avoid stale reads before recompute.
        await self.redis.delete(cached_key, hints_key)

        logger.debug(f"Added {role} message to session {session_id}")
        return message

    async def clear_session(self, session_id: str) -> bool:
        """Clear all messages in a session (keep metadata)."""
        messages_key = self.SESSION_MESSAGES_KEY.format(session_id=session_id)
        meta_key = self.SESSION_METADATA_KEY.format(session_id=session_id)
        cached_key = self.SESSION_CACHED_CONTEXT_KEY.format(session_id=session_id)  # v4.1.0
        hints_key = self.SESSION_HINTS_KEY.format(session_id=session_id)  # v5.0
        pending_key = self.SESSION_PENDING_KEY.format(session_id=session_id)
        requeue_key = self.SESSION_RECOMPRESS_KEY.format(session_id=session_id)

        await self.redis.delete(messages_key, cached_key, hints_key, pending_key, requeue_key)  # v4.1.0 + v5.0
        await self.redis.hset(meta_key, "message_count", "0")

        logger.info(f"Cleared messages for session {session_id}")
        return True

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session completely."""
        messages_key = self.SESSION_MESSAGES_KEY.format(session_id=session_id)
        meta_key = self.SESSION_METADATA_KEY.format(session_id=session_id)
        cached_key = self.SESSION_CACHED_CONTEXT_KEY.format(session_id=session_id)  # v4.1.0
        hints_key = self.SESSION_HINTS_KEY.format(session_id=session_id)  # v5.0
        pending_key = self.SESSION_PENDING_KEY.format(session_id=session_id)
        requeue_key = self.SESSION_RECOMPRESS_KEY.format(session_id=session_id)

        # Get user_id to remove from user's session list
        user_id = await self.redis.hget(meta_key, "user_id")

        # Delete session data
        await self.redis.delete(
            messages_key, meta_key, cached_key, hints_key, pending_key, requeue_key
        )  # v4.1.0 + v5.0

        # Remove from user's sessions
        if user_id:
            if isinstance(user_id, bytes):
                user_id = user_id.decode("utf-8")
            user_sessions_key = self.USER_SESSIONS_KEY.format(user_id=user_id)
            await self.redis.zrem(user_sessions_key, session_id)

        logger.info(f"Deleted session {session_id}")
        return True

    async def list_sessions(
        self,
        user_id: str,
        limit: int = 20,
        tz_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all sessions for a user (most recent first).

        Uses Redis pipeline to fetch all metadata in a single round-trip
        instead of N individual hgetall calls.

        fb1 (B-05 L1): when ``tz_name`` is provided, ``created_at`` and
        ``last_active`` are localised in-place. Storage remains UTC.
        """
        user_sessions_key = self.USER_SESSIONS_KEY.format(user_id=user_id)

        # Get session IDs sorted by last activity (descending)
        session_ids = await self.redis.zrevrange(user_sessions_key, 0, limit - 1)

        if not session_ids:
            return []

        # Decode session IDs
        decoded_ids = [
            sid.decode("utf-8") if isinstance(sid, bytes) else sid
            for sid in session_ids
        ]

        # Pipeline: fetch all metadata in a single round-trip
        pipe = self.redis.pipeline(transaction=False)
        for sid in decoded_ids:
            meta_key = self.SESSION_METADATA_KEY.format(session_id=sid)
            pipe.hgetall(meta_key)
        results = await pipe.execute()

        sessions = []
        for meta in results:
            if meta:
                decoded_meta = {}
                for k, v in meta.items():
                    key = k.decode("utf-8") if isinstance(k, bytes) else k
                    val = v.decode("utf-8") if isinstance(v, bytes) else v
                    decoded_meta[key] = val
                if tz_name:
                    for _ts_field in ("created_at", "last_active"):
                        if decoded_meta.get(_ts_field):
                            decoded_meta[_ts_field] = (
                                _convert_timestamp_for_display(
                                    decoded_meta[_ts_field], tz_name
                                )
                            )
                sessions.append(decoded_meta)

        return sessions

    async def get_context_for_llm(self, session_id: str, max_turns: int = 10) -> str:
        """
        Get formatted conversation context for LLM prompt.

        Returns conversation history formatted as:
        User: message1
        Assistant: response1
        User: message2
        ...
        """
        history = await self.get_history(session_id, limit=max_turns * 2)

        if not history:
            return ""

        formatted = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted)

    async def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific session."""
        meta_key = self.SESSION_METADATA_KEY.format(session_id=session_id)
        meta = await self.redis.hgetall(meta_key)

        if not meta:
            return None

        # Decode bytes if necessary
        decoded_meta = {}
        for k, v in meta.items():
            key = k.decode("utf-8") if isinstance(k, bytes) else k
            val = v.decode("utf-8") if isinstance(v, bytes) else v
            decoded_meta[key] = val

        return decoded_meta

    # =========================================================================
    # v2.0: STRUCTURED MEMORY STATE METHODS
    # =========================================================================

    async def get_state(self, session_id: str) -> Optional[str]:
        """
        Get structured memory state JSON for a session.

        Args:
            session_id: Session identifier

        Returns:
            JSON string of MemoryState or None if not exists
        """
        state_key = self.SESSION_STATE_KEY.format(session_id=session_id)
        state_json = await self.redis.get(state_key)

        if state_json is None:
            return None

        if isinstance(state_json, bytes):
            state_json = state_json.decode("utf-8")

        return state_json

    async def save_state(
        self,
        session_id: str,
        state_json: str,
        expected_version: Optional[int] = None
    ) -> bool:
        """
        Save structured memory state to Redis.

        Supports optimistic concurrency control via expected_version.

        Args:
            session_id: Session identifier
            state_json: JSON string of MemoryState
            expected_version: If provided, only save if current version matches

        Returns:
            True if saved successfully, False if version mismatch
        """
        state_key = self.SESSION_STATE_KEY.format(session_id=session_id)

        if expected_version is not None:
            # Atomic CAS via Lua script — eliminates TOCTOU race condition
            result = await self.redis.eval(
                LUA_CAS_SAVE_STATE, 1, state_key,
                expected_version, state_json, self.default_ttl
            )
            if result == 0:
                logger.warning(
                    f"Version mismatch for session {session_id}: "
                    f"expected {expected_version} (atomic CAS rejected)"
                )
                return False
            logger.debug(f"Saved state for session {session_id} (atomic CAS)")
            return True

        await self.redis.set(state_key, state_json)
        await self.redis.expire(state_key, self.default_ttl)

        logger.debug(f"Saved state for session {session_id}")
        return True

    async def delete_state(self, session_id: str) -> bool:
        """
        Delete structured memory state for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted
        """
        state_key = self.SESSION_STATE_KEY.format(session_id=session_id)
        pending_key = self.SESSION_PENDING_KEY.format(session_id=session_id)
        cached_key = self.SESSION_CACHED_CONTEXT_KEY.format(session_id=session_id)  # v4.1.0
        hints_key = self.SESSION_HINTS_KEY.format(session_id=session_id)  # v5.0

        await self.redis.delete(state_key, pending_key, cached_key, hints_key)  # v4.1.0 + v5.0
        return True

    # =========================================================================
    # v4.1.0: PRE-CACHED CONTEXT METHODS
    # =========================================================================

    async def get_cached_context(self, session_id: str) -> Optional[str]:
        """
        Get pre-cached formatted context for a session.

        Returns:
            Cached context string or None if not cached
        """
        key = self.SESSION_CACHED_CONTEXT_KEY.format(session_id=session_id)
        cached = await self.redis.get(key)
        if cached is None:
            return None
        if isinstance(cached, bytes):
            cached = cached.decode("utf-8")
        return cached

    async def save_cached_context(self, session_id: str, context: str) -> None:
        """
        Save pre-cached formatted context for a session.

        Called after every eager compression to pre-compute context for next query.
        """
        key = self.SESSION_CACHED_CONTEXT_KEY.format(session_id=session_id)
        await self.redis.set(key, context)
        await self.redis.expire(key, self.default_ttl)

    async def is_compression_pending(self, session_id: str) -> bool:
        """
        Check if compression is currently in progress for a session.

        Used to prevent concurrent compression operations.

        Args:
            session_id: Session identifier

        Returns:
            True if compression is pending
        """
        pending_key = self.SESSION_PENDING_KEY.format(session_id=session_id)
        pending = await self.redis.get(pending_key)
        return pending is not None and pending != b"0"

    async def mark_recompress_requested(self, session_id: str, ttl_seconds: int = 120) -> None:
        """Mark that a new compression pass is required after current lock releases."""
        key = self.SESSION_RECOMPRESS_KEY.format(session_id=session_id)
        await self.redis.set(key, "1", ex=ttl_seconds)

    async def consume_recompress_requested(self, session_id: str) -> bool:
        """Atomically consume pending recompress request flag."""
        key = self.SESSION_RECOMPRESS_KEY.format(session_id=session_id)
        existed = await self.redis.get(key)
        if existed is None:
            return False
        await self.redis.delete(key)
        return True

    async def set_compression_pending(
        self,
        session_id: str,
        pending: bool,
        ttl_seconds: int = 60,
        force: bool = False,
        owner_id: Optional[str] = None,
    ) -> bool:
        """
        Set compression pending flag (lock/unlock).

        Uses TTL to auto-release lock in case of failures.
        Uses owner_id to prevent releasing another caller's lock.

        Args:
            session_id: Session identifier
            pending: True to set lock, False to release
            ttl_seconds: Lock TTL (auto-release after this time)
            force: If True, bypass NX check (use sparingly)
            owner_id: Unique caller ID for ownership verification

        Returns:
            True if lock acquired/released successfully
        """
        pending_key = self.SESSION_PENDING_KEY.format(session_id=session_id)

        if pending:
            # Try to acquire lock (NX = only if not exists), unless force=True.
            value = owner_id or "1"
            result = await self.redis.set(
                pending_key, value, nx=not force, ex=ttl_seconds
            )
            return result is not None
        else:
            # Release lock — verify ownership if owner_id provided
            if owner_id:
                current = await self.redis.get(pending_key)
                if current and current != owner_id:
                    logger.warning(
                        f"[MEMORY] Lock release denied for session {session_id}: "
                        f"owner mismatch (current={current}, caller={owner_id})"
                    )
                    return False
            await self.redis.delete(pending_key)
            # Notify waiters that semaphore is released (pub/sub, fire-and-forget)
            try:
                await self.redis.publish(
                    f"ubp:memory:semaphore:release:{session_id}", "released"
                )
            except Exception:
                pass  # Best-effort: waiters have fallback polling
            return True

    async def get_messages_for_compression(
        self,
        session_id: str,
        count: int
    ) -> List[Dict[str, Any]]:
        """
        Get oldest N messages for compression (to be archived).

        Args:
            session_id: Session identifier
            count: Number of messages to retrieve

        Returns:
            List of message dicts (oldest first)
        """
        messages_key = self.SESSION_MESSAGES_KEY.format(session_id=session_id)

        # Get oldest N messages (from the beginning of the list)
        raw_messages = await self.redis.lrange(messages_key, 0, count - 1)

        messages = []
        for raw in raw_messages:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                msg = json.loads(raw)
                messages.append(msg)
            except json.JSONDecodeError:
                logger.warning(f"Invalid message JSON in session {session_id}")
                continue

        return messages

    async def remove_oldest_messages(
        self,
        session_id: str,
        count: int
    ) -> int:
        """
        Remove oldest N messages from the session (after compression).

        Args:
            session_id: Session identifier
            count: Number of messages to remove

        Returns:
            Number of messages actually removed
        """
        messages_key = self.SESSION_MESSAGES_KEY.format(session_id=session_id)

        # LTRIM keeps elements from 'count' to end (removes first 'count' elements)
        await self.redis.ltrim(messages_key, count, -1)

        logger.debug(f"Removed {count} oldest messages from session {session_id}")
        return count

    # =========================================================================
    # v5.1: CROSS-SESSION SEARCH
    # =========================================================================

    async def search_history(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        time_range: Optional[str] = None,
        client_id: Optional[str] = None,
        tz_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search user's conversation history across past sessions.

        Scores sessions by keyword matching against conversation_thread
        (focus, key_facts, entities), topic_flow, and narrative_summary.
        User-scoped, read-only, bounded output.

        Args:
            user_id: Authenticated user (from security context)
            query: Natural language search query
            top_k: Max hits to return (clamped 1-10)
            time_range: Optional date filter (YYYY-MM-DD, range, last_7d/last_30d)
            client_id: Caller's client_id for session filtering
            tz_name: fb1 (B-05 L1) — IANA tz for localising ``date`` and
                ``last_active`` of each hit. None → UTC.
        """
        import re
        from datetime import timedelta

        top_k = max(1, min(10, top_k))
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {"hits": [], "total_sessions_searched": 0, "query_used": query}

        # Parse time_range
        date_from, date_to = self._parse_time_range(time_range)

        # Get all user sessions (most recent first, max 100)
        user_sessions_key = self.USER_SESSIONS_KEY.format(user_id=user_id)
        session_entries = await self.redis.zrevrange(
            user_sessions_key, 0, 99, withscores=True
        )
        if not session_entries:
            return {"hits": [], "total_sessions_searched": 0, "query_used": query}

        now = datetime.now(timezone.utc)
        scored_hits = []
        sessions_searched = 0

        for entry in session_entries:
            sid_raw, score_ts = entry
            sid = sid_raw.decode("utf-8") if isinstance(sid_raw, bytes) else sid_raw

            # Fetch metadata for client_id filtering
            meta_key = self.SESSION_METADATA_KEY.format(session_id=sid)
            meta = await self.redis.hgetall(meta_key)
            if not meta:
                continue

            # Decode metadata
            decoded_meta = {}
            for k, v in meta.items():
                key = k.decode("utf-8") if isinstance(k, bytes) else k
                val = v.decode("utf-8") if isinstance(v, bytes) else v
                decoded_meta[key] = val

            # Client-scoping: default-deny cross-client
            session_client = decoded_meta.get("client_id")
            if session_client and session_client != "None" and client_id:
                if session_client != client_id:
                    continue  # Skip sessions from different client

            # Time range filter
            session_date_str = decoded_meta.get("created_at", "")
            if date_from or date_to:
                try:
                    session_dt = datetime.fromisoformat(session_date_str)
                    if date_from and session_dt < date_from:
                        continue
                    if date_to and session_dt > date_to:
                        continue
                except (ValueError, TypeError):
                    pass  # Can't parse date, include session

            sessions_searched += 1

            # Load state for scoring
            state_key = self.SESSION_STATE_KEY.format(session_id=sid)
            state_json = await self.redis.get(state_key)
            if not state_json:
                # No structured state — try cached_context as fallback
                ctx_key = self.SESSION_CACHED_CONTEXT_KEY.format(session_id=sid)
                cached = await self.redis.get(ctx_key)
                if cached:
                    cached_str = cached.decode("utf-8") if isinstance(cached, bytes) else cached
                    score = self._score_text(query_tokens, cached_str) * 0.6
                    if score >= 0.30:
                        scored_hits.append(self._build_hit_from_meta(
                            sid, decoded_meta, score, "keyword_match",
                            topics=[], entities=[], summary=cached_str[:150],
                            tz_name=tz_name,
                        ))
                continue

            # Parse state
            if isinstance(state_json, bytes):
                state_json = state_json.decode("utf-8")
            try:
                state = json.loads(state_json)
            except (json.JSONDecodeError, TypeError):
                logger.debug("[SEARCH] Skipping session %s: malformed state JSON", sid[:8])
                continue

            # Score against all available data sources
            thread = state.get("conversation_thread", [])
            if not isinstance(thread, list):
                logger.debug("[SEARCH] Skipping session %s: bad conversation_thread", sid[:8])
                continue

            best_score = 0.0
            best_reason = "keyword_match"

            # Source 1: conversation_thread turns (highest weight)
            for turn in thread:
                if not isinstance(turn, dict):
                    continue
                focus = turn.get("focus", "")
                key_facts = turn.get("key_facts_full", "") or turn.get("key_facts", "")
                turn_text = f"{focus} {key_facts}"
                s = self._score_text(query_tokens, turn_text)
                if s > best_score:
                    best_score = s
                    best_reason = "topic_match" if self._score_text(query_tokens, focus) > 0.3 else "keyword_match"

            # Source 2: structured_context entities
            sc = state.get("structured_context", {})
            if isinstance(sc, dict):
                entities = sc.get("entities", {})
                if isinstance(entities, dict):
                    entity_text = " ".join(str(v) for v in entities.values() if v)
                    s = self._score_text(query_tokens, entity_text) * 0.9
                    if s > best_score:
                        best_score = s
                        best_reason = "entity_match"

            # Source 3: topic_flow
            topic_flow = state.get("topic_flow", [])
            if isinstance(topic_flow, list):
                flow_text = " ".join(
                    t.get("topic", "") if isinstance(t, dict) else str(t)
                    for t in topic_flow
                )
                s = self._score_text(query_tokens, flow_text) * 0.8
                if s > best_score:
                    best_score = s
                    best_reason = "topic_match"

            # Source 4: narrative_summary (fallback)
            narrative = state.get("narrative_summary", "")
            if narrative:
                s = self._score_text(query_tokens, narrative) * 0.6
                if s > best_score:
                    best_score = s
                    best_reason = "keyword_match"

            # Recency boost (non-dominant)
            age_days = (now - datetime.fromtimestamp(score_ts, tz=timezone.utc)).days
            recency_factor = 1.0 - (age_days / 365) * 0.3
            final_score = best_score * max(0.5, recency_factor)

            # Minimum threshold
            if final_score < 0.30:
                continue

            # Extract topics and entities for output
            topics = []
            seen_topics = set()
            for turn in thread:
                if isinstance(turn, dict):
                    f = turn.get("focus", "")
                    if f and f not in seen_topics:
                        seen_topics.add(f)
                        topics.append(f)
            topics = topics[:5]

            entities_out = []
            if isinstance(sc, dict):
                ents = sc.get("entities", {})
                if isinstance(ents, dict):
                    for v in ents.values():
                        if isinstance(v, list):
                            entities_out.extend(v)
                        elif isinstance(v, str) and v:
                            entities_out.append(v)
            entities_out = list(dict.fromkeys(entities_out))[:8]  # dedupe, max 8

            summary = (narrative or "")[:150]

            scored_hits.append(self._build_hit_from_meta(
                sid, decoded_meta, round(final_score, 3), best_reason,
                topics=topics, entities=entities_out, summary=summary,
                tz_name=tz_name,
            ))

        # Sort by score descending, take top_k
        scored_hits.sort(key=lambda h: h["relevance_score"], reverse=True)
        hits = scored_hits[:top_k]

        return {
            "hits": hits,
            "total_sessions_searched": sessions_searched,
            "query_used": query,
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenizer: lowercase, split, filter short/stop words."""
        import re
        tokens = re.findall(r"[a-zA-ZàèéìòùÀÈÉÌÒÙ0-9]+", text.lower())
        stop = {"il", "lo", "la", "le", "i", "gli", "un", "una", "di", "a", "da",
                "in", "con", "su", "per", "tra", "fra", "e", "o", "ma", "che", "è",
                "the", "a", "an", "in", "on", "of", "and", "or", "is", "to", "for"}
        return [t for t in tokens if len(t) >= 2 and t not in stop]

    @staticmethod
    def _score_text(query_tokens: List[str], text: str) -> float:
        """Score text against query tokens. Returns 0.0-1.0."""
        if not text or not query_tokens:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for t in query_tokens if t in text_lower)
        return matches / len(query_tokens)

    @staticmethod
    def _build_hit_from_meta(
        session_id: str, meta: Dict[str, str], score: float, reason: str,
        topics: List[str], entities: List[str], summary: str,
        tz_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a search hit dict from session metadata.

        fb1 (B-05 L1): when ``tz_name`` is provided, ``date`` is derived
        from ``created_at`` localised to ``tz_name`` (YYYY-MM-DD slice
        after conversion) and ``last_active`` is localised in-place.
        """
        created = meta.get("created_at", "")
        last_active = meta.get("last_active", "")
        if tz_name:
            if created:
                created_local = _convert_timestamp_for_display(created, tz_name)
                date_str = created_local[:10] if len(created_local) >= 10 else ""
            else:
                date_str = ""
            if last_active:
                last_active = _convert_timestamp_for_display(last_active, tz_name)
        else:
            date_str = created[:10] if len(created) >= 10 else ""
        msg_count = 0
        try:
            msg_count = int(meta.get("message_count", 0))
        except (ValueError, TypeError):
            pass
        return {
            "session_id": session_id,
            "date": date_str,
            "last_active": last_active,
            "topics": topics,
            "summary": summary,
            "key_entities": entities,
            "relevance_score": score,
            "match_reason": reason,
            "message_count": msg_count,
        }

    @staticmethod
    def _parse_time_range(time_range: Optional[str]):
        """Parse time_range string into (date_from, date_to) datetimes."""
        from datetime import timedelta
        if not time_range:
            return None, None
        time_range = time_range.strip()
        now = datetime.now(timezone.utc)

        if time_range == "last_7d":
            return now - timedelta(days=7), None
        if time_range == "last_30d":
            return now - timedelta(days=30), None

        # Single date: YYYY-MM-DD
        if len(time_range) == 10 and time_range[4] == "-":
            try:
                day = datetime.fromisoformat(time_range + "T00:00:00+00:00")
                return day, day + timedelta(days=1)
            except ValueError:
                return None, None

        # Range: YYYY-MM-DD/YYYY-MM-DD
        if "/" in time_range:
            parts = time_range.split("/", 1)
            try:
                d_from = datetime.fromisoformat(parts[0].strip() + "T00:00:00+00:00")
                d_to = datetime.fromisoformat(parts[1].strip() + "T23:59:59+00:00")
                return d_from, d_to
            except (ValueError, IndexError):
                return None, None

        return None, None

    def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        return {
            "status": "configured" if self.redis else "not_configured",
            "max_messages": self.max_messages,
            "default_ttl_days": self.default_ttl // 86400,
            "max_sessions_per_user": self.max_sessions_per_user,
            "structured_memory_support": True,  # v2.0
        }
