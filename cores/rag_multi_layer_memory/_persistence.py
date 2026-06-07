"""v17.15 W7 — MEM-005 Redis-backed session persistence for
``rag_multi_layer_memory``.

Mirrors the design used for ALMS (see
``mcp_agent_loop_memory/_persistence.py``) — keep the in-memory dict as
a hot cache, persist every mutation through to Redis when DI provides a
client. On worker restart, sessions hydrate on first access.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


KEY_PREFIX = "ubp:rag_mlm:session"
DEFAULT_TTL_SECONDS = 7 * 24 * 3600


class MLMRedisSessionStore:
    def __init__(self, redis_client: Any, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self._redis = redis_client
        self._ttl = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{KEY_PREFIX}:{session_id}"

    async def load(self, session_id: str) -> Optional[Any]:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"[MEM-005] Corrupted multi-layer state for {session_id} ({e})"
            )
            return None

    async def save(self, session_id: str, state_dict: Any) -> bool:
        try:
            payload = json.dumps(state_dict, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as e:
            logger.error(f"[MEM-005] Cannot serialise multi-layer state {session_id}: {e}")
            return False
        await self._redis.set(self._key(session_id), payload, ex=self._ttl)
        return True

    async def delete(self, session_id: str) -> bool:
        removed = await self._redis.delete(self._key(session_id))
        try:
            return int(removed) > 0
        except (TypeError, ValueError):
            return bool(removed)


async def maybe_build_mlm_store(di_container: Any) -> Optional[MLMRedisSessionStore]:
    if di_container is None:
        return None

    candidates = ("system_redis_client", "redis", "redis_client")
    redis_client = None
    for name in candidates:
        try:
            redis_client = await di_container.resolve(name)
            if redis_client is not None:
                break
        except Exception:
            continue

    if redis_client is None:
        try:
            import redis.asyncio as aioredis
            redis_client = await di_container.resolve(aioredis.Redis)
        except Exception:
            redis_client = None

    if redis_client is None:
        logger.warning(
            "[MEM-005] DI did not provide a Redis client; multi-layer "
            "memory sessions remain volatile (in-memory only)."
        )
        return None

    # If user passed a UBP RedisProvider, prefer its raw .client
    # attribute (a real redis.asyncio.Redis instance). Otherwise use the
    # provided client directly.
    raw = redis_client
    inner = getattr(redis_client, "client", None)
    if inner is not None and not callable(inner):
        raw = inner
    return MLMRedisSessionStore(raw)
