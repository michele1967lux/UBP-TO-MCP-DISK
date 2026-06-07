"""
Qdrant Client Wrapper - Enterprise Grade

Production-ready Qdrant client with:
- Connection pooling
- Circuit breaker pattern
- Retry with exponential backoff
- Health monitoring
- Graceful degradation
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
from functools import wraps
import uuid

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Circuit Breaker Implementation
# ============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 15.0
    half_open_max_calls: int = 3


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    half_open_calls: int = 0


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are rejected immediately
    - HALF_OPEN: Testing if service has recovered
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._stats.state

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self._stats.state.value,
            "failure_count": self._stats.failure_count,
            "success_count": self._stats.success_count,
            "total_requests": self._stats.total_requests,
            "total_failures": self._stats.total_failures,
            "total_successes": self._stats.total_successes,
        }

    async def _check_state_transition(self) -> None:
        """Check if state should transition based on timeout."""
        if self._stats.state == CircuitState.OPEN:
            time_since_failure = time.time() - (self._stats.last_failure_time or 0)
            if time_since_failure >= self.config.timeout_seconds:
                self._stats.state = CircuitState.HALF_OPEN
                self._stats.half_open_calls = 0
                self._stats.last_state_change = time.time()
                logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats.total_successes += 1
            self._stats.success_count += 1

            if self._stats.state == CircuitState.HALF_OPEN:
                if self._stats.success_count >= self.config.success_threshold:
                    self._stats.state = CircuitState.CLOSED
                    self._stats.failure_count = 0
                    self._stats.success_count = 0
                    self._stats.last_state_change = time.time()
                    logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")
            elif self._stats.state == CircuitState.CLOSED:
                self._stats.failure_count = 0

    async def record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        async with self._lock:
            self._stats.total_failures += 1
            self._stats.failure_count += 1
            self._stats.success_count = 0
            self._stats.last_failure_time = time.time()

            if self._stats.state == CircuitState.HALF_OPEN:
                self._stats.state = CircuitState.OPEN
                self._stats.last_state_change = time.time()
                logger.warning(
                    f"Circuit breaker '{self.name}' transitioned to OPEN (half-open failure)"
                )
            elif self._stats.state == CircuitState.CLOSED:
                if self._stats.failure_count >= self.config.failure_threshold:
                    self._stats.state = CircuitState.OPEN
                    self._stats.last_state_change = time.time()
                    logger.warning(
                        f"Circuit breaker '{self.name}' transitioned to OPEN"
                    )

    async def can_execute(self) -> bool:
        """Check if a call can be executed."""
        async with self._lock:
            await self._check_state_transition()
            self._stats.total_requests += 1

            if self._stats.state == CircuitState.CLOSED:
                return True
            elif self._stats.state == CircuitState.OPEN:
                return False
            else:  # HALF_OPEN
                if self._stats.half_open_calls < self.config.half_open_max_calls:
                    self._stats.half_open_calls += 1
                    return True
                return False

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function with circuit breaker protection."""
        if not await self.can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Retry after {self.config.timeout_seconds}s"
            )

        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception as e:
            await self.record_failure(e)
            raise

    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        self._stats = CircuitBreakerStats()
        logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CollectionNotFoundError(Exception):
    """Collection does not exist (HTTP 404). Not retryable, not a service outage."""

    def __init__(self, collection_name: str = "", detail: str = ""):
        self.collection_name = collection_name
        msg = f"Collection not found: '{collection_name}'"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


# ============================================================================
# Retry Configuration
# ============================================================================


@dataclass
class RetryConfig:
    """Retry configuration with exponential backoff."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    retryable_exceptions: tuple = (Exception,)


async def retry_with_backoff(
    func: Callable[..., T], config: RetryConfig, *args, **kwargs
) -> T:
    """
    Execute function with retry and exponential backoff.

    Args:
        func: Async function to execute
        config: Retry configuration
        *args, **kwargs: Function arguments

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    delay = config.initial_delay

    for attempt in range(config.max_retries):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            if _is_not_found_error(e):
                raise  # Never retry 404
            last_exception = e

            if attempt < config.max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                delay = min(delay * config.backoff_multiplier, config.max_delay)
            else:
                logger.error(f"All {config.max_retries} attempts failed: {e}")

    raise last_exception


def _is_not_found_error(exc: Exception) -> bool:
    """Check if exception is a Qdrant 404 (collection not found)."""
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse

        if isinstance(exc, UnexpectedResponse) and exc.status_code == 404:
            return True
    except ImportError:
        pass
    return False


# ============================================================================
# Qdrant Client Interface
# ============================================================================


class QdrantClientInterface(ABC):
    """Abstract interface for Qdrant client operations."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to Qdrant."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to Qdrant."""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if connected to Qdrant."""
        pass

    @abstractmethod
    async def create_collection(
        self, collection_name: str, vector_size: int, distance: str = "Cosine"
    ) -> bool:
        """Create a new collection."""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        pass

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all collections."""
        pass

    @abstractmethod
    async def upsert(
        self, collection_name: str, points: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Insert or update points."""
        pass

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete points by IDs."""
        pass

    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information."""
        pass


# ============================================================================
# Production Qdrant Client
# ============================================================================


class QdrantClient(QdrantClientInterface):
    """
    Production-ready Qdrant client wrapper.

    Features:
    - Automatic connection management
    - Circuit breaker for fault tolerance
    - Retry with exponential backoff
    - Health monitoring
    - Metrics collection
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        timeout: float = 30.0,
        api_key: Optional[str] = None,
        https: bool = False,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.timeout = timeout
        self.api_key = api_key
        self.https = https

        self._client = None
        self._connected = False
        self._lock = asyncio.Lock()

        # Fault tolerance
        self.circuit_breaker = CircuitBreaker(
            "qdrant", circuit_breaker_config or CircuitBreakerConfig()
        )
        self.retry_config = retry_config or RetryConfig()

        # Metrics
        self._metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_latency_ms": 0.0,
        }

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        avg_latency = 0.0
        if self._metrics["successful_operations"] > 0:
            avg_latency = (
                self._metrics["total_latency_ms"]
                / self._metrics["successful_operations"]
            )

        return {
            **self._metrics,
            "average_latency_ms": round(avg_latency, 2),
            "circuit_breaker": self.circuit_breaker.stats,
        }

    async def connect(self) -> None:
        """Establish connection to Qdrant."""
        async with self._lock:
            if self._connected:
                return

            try:
                from qdrant_client import QdrantClient as QC
                from qdrant_client.async_qdrant_client import AsyncQdrantClient

                # Use async client for better performance
                if self.https or self.api_key:
                    url = (
                        f"{'https' if self.https else 'http'}://{self.host}:{self.port}"
                    )
                    self._client = AsyncQdrantClient(
                        url=url,
                        api_key=self.api_key,
                        timeout=self.timeout,
                        prefer_grpc=self.prefer_grpc,
                        grpc_port=self.grpc_port if self.prefer_grpc else None,
                    )
                else:
                    self._client = AsyncQdrantClient(
                        host=self.host,
                        port=self.port,
                        timeout=self.timeout,
                        prefer_grpc=self.prefer_grpc,
                        grpc_port=self.grpc_port if self.prefer_grpc else None,
                    )

                # Test connection
                await self._client.get_collections()
                self._connected = True

                logger.info(
                    f"Connected to Qdrant at {self.host}:{self.port}",
                    extra={"grpc": self.prefer_grpc},
                )

            except ImportError:
                logger.warning(
                    "qdrant-client not installed. Using mock client. "
                    "Install with: pip install qdrant-client"
                )
                self._client = MockQdrantClientInternal()
                self._connected = True

            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise ConnectionError(f"Cannot connect to Qdrant: {e}")

    async def disconnect(self) -> None:
        """Close connection to Qdrant."""
        async with self._lock:
            if self._client and hasattr(self._client, "close"):
                await self._client.close()
            self._client = None
            self._connected = False
            logger.info("Disconnected from Qdrant")

    async def is_connected(self) -> bool:
        """Check if connected to Qdrant."""
        return self._connected and self._client is not None

    async def _execute(
        self, operation: Callable[..., T], operation_name: str, *args, **kwargs
    ) -> T:
        """Execute operation with circuit breaker and retry."""
        if not await self.is_connected():
            await self.connect()

        self._metrics["total_operations"] += 1
        start_time = time.time()

        try:
            result = await self.circuit_breaker.execute(
                retry_with_backoff, operation, self.retry_config, *args, **kwargs
            )

            latency = (time.time() - start_time) * 1000
            self._metrics["successful_operations"] += 1
            self._metrics["total_latency_ms"] += latency

            logger.debug(
                f"Qdrant {operation_name} completed",
                extra={"latency_ms": round(latency, 2)},
            )

            return result

        except CircuitBreakerOpenError:
            self._metrics["failed_operations"] += 1
            raise

        except Exception as e:
            if self._is_collection_not_found(e):
                self._metrics["failed_operations"] += 1
                await self._undo_breaker_failure()
                collection_hint = self._extract_collection_name(e)
                raise CollectionNotFoundError(
                    collection_name=collection_hint, detail=str(e)
                ) from e
            self._metrics["failed_operations"] += 1
            logger.error(
                f"Qdrant {operation_name} failed: {e}",
                extra={"error_type": type(e).__name__},
            )
            raise

    async def create_collection(
        self, collection_name: str, vector_size: int, distance: str = "Cosine"
    ) -> bool:
        """Create a new collection with specified parameters."""
        from qdrant_client.models import VectorParams, Distance

        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }

        async def _create():
            # Check if collection already exists
            collections = await self._client.get_collections()
            existing = [c.name for c in collections.collections]

            if collection_name in existing:
                logger.info(f"Collection '{collection_name}' already exists")
                return True

            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance, Distance.COSINE),
                ),
            )
            return True

        return await self._execute(_create, "create_collection")

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""

        async def _delete():
            await self._client.delete_collection(collection_name)
            return True

        return await self._execute(_delete, "delete_collection")

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""

        async def _exists():
            collections = await self._client.get_collections()
            return collection_name in [c.name for c in collections.collections]

        return await self._execute(_exists, "collection_exists")

    async def list_collections(self) -> List[str]:
        """List all collections."""

        async def _list():
            collections = await self._client.get_collections()
            return [c.name for c in collections.collections]

        return await self._execute(_list, "list_collections")

    async def upsert(
        self, collection_name: str, points: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Insert or update points in a collection.

        Args:
            collection_name: Target collection
            points: List of points with id, vector, and payload

        Returns:
            Upsert result
        """
        from qdrant_client.models import PointStruct

        async def _upsert():
            qdrant_points = [
                PointStruct(
                    id=p.get("id", str(uuid.uuid4())),
                    vector=p["vector"],
                    payload=p.get("payload", {}),
                )
                for p in points
            ]

            result = await self._client.upsert(
                collection_name=collection_name, points=qdrant_points
            )

            return {"status": "completed", "points_count": len(points)}

        return await self._execute(_upsert, "upsert")

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            collection_name: Collection to search
            query_vector: Query embedding
            limit: Max results
            score_threshold: Minimum similarity score
            filter_conditions: Qdrant filter
            with_payload: Include payload in results
            with_vectors: Include vectors in results

        Returns:
            List of search results
        """

        async def _search():
            # Use query_points API (qdrant-client >= 1.16.0)
            # The old search() method was deprecated and removed
            response = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

            # query_points returns QueryResponse with .points attribute
            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    "payload": r.payload if with_payload else None,
                    "vector": r.vector if with_vectors else None,
                }
                for r in response.points
            ]

        return await self._execute(_search, "search")

    async def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete points by IDs."""
        from qdrant_client.models import PointIdsList

        async def _delete():
            await self._client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=point_ids),
            )
            return True

        return await self._execute(_delete, "delete_points")

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get detailed collection information."""

        async def _info():
            info = await self._client.get_collection(collection_name)
            return {
                "name": collection_name,
                "status": info.status.value
                if hasattr(info.status, "value")
                else str(info.status),
                "vectors_count": getattr(
                    info, "indexed_vectors_count", getattr(info, "vectors_count", 0)
                ),
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value,
                },
            }

        return await self._execute(_info, "get_collection_info")

    async def count(self, collection_name: str) -> int:
        """Get total points count in a collection (safe, no Pydantic validation issues)."""

        async def _count():
            result = await self._client.count(collection_name=collection_name)
            return result.count

        return await self._execute(_count, "count")

    async def get_vector_dimension_safe(self, collection_name: str) -> Optional[int]:
        """
        Get vector dimension for a collection using raw HTTP API.

        This method bypasses Pydantic validation issues that can occur with
        get_collection_info() on certain Qdrant versions.

        Args:
            collection_name: Name of the collection

        Returns:
            Vector dimension (int) or None if collection doesn't exist
        """

        async def _get_dimension():
            try:
                # Try the standard method first
                info = await self._client.get_collection(collection_name)
                if hasattr(info, "config") and hasattr(info.config, "params"):
                    vectors_config = info.config.params.vectors
                    # Handle both single vector and named vectors config
                    if hasattr(vectors_config, "size"):
                        return vectors_config.size
                    elif isinstance(vectors_config, dict):
                        # Named vectors - get the default or first one
                        for v in vectors_config.values():
                            if hasattr(v, "size"):
                                return v.size
                return None
            except Exception as e:
                # Fallback: use HTTP client directly if available
                logger.warning(
                    f"Standard get_collection failed: {e}, trying HTTP fallback"
                )
                try:
                    import httpx

                    url = (
                        f"http://{self.host}:{self.port}/collections/{collection_name}"
                    )
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.get(url)
                        if response.status_code == 200:
                            data = response.json()
                            return (
                                data.get("result", {})
                                .get("config", {})
                                .get("params", {})
                                .get("vectors", {})
                                .get("size")
                            )
                except Exception as http_err:
                    logger.error(f"HTTP fallback also failed: {http_err}")
                return None

        return await self._execute(_get_dimension, "get_vector_dimension_safe")

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[Union[str, int]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Scroll through all points in a collection with pagination.

        Args:
            collection_name: Collection to scroll
            limit: Max points per page
            offset: Starting point ID (None for first page)
            with_payload: Include payload in results
            with_vectors: Include vectors in results
            filter_conditions: Optional filter

        Returns:
            Dict with 'points' list and 'next_offset' for pagination
        """

        async def _scroll():
            result = await self._client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
                scroll_filter=filter_conditions,
            )

            points, next_offset = result

            return {
                "points": [
                    {
                        "id": str(p.id),
                        "payload": p.payload if with_payload else None,
                        "vector": p.vector if with_vectors else None,
                    }
                    for p in points
                ],
                "next_offset": str(next_offset) if next_offset else None,
                "count": len(points),
            }

        return await self._execute(_scroll, "scroll")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            start = time.time()
            collections = await self.list_collections()
            latency = (time.time() - start) * 1000

            return {
                "status": "healthy",
                "connected": True,
                "collections_count": len(collections),
                "latency_ms": round(latency, 2),
                "circuit_breaker_state": self.circuit_breaker.state.value,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "circuit_breaker_state": self.circuit_breaker.state.value,
            }

    # ------------------------------------------------------------------
    # Collection-not-found helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_collection_not_found(exc: Exception) -> bool:
        """Check if exception represents a Qdrant 404."""
        try:
            from qdrant_client.http.exceptions import UnexpectedResponse

            if isinstance(exc, UnexpectedResponse) and exc.status_code == 404:
                return True
        except ImportError:
            pass
        if isinstance(exc, CollectionNotFoundError):
            return True
        return False

    @staticmethod
    def _extract_collection_name(exc: Exception) -> str:
        """Best-effort extraction of collection name from error."""
        try:
            content = getattr(exc, "content", None) or str(exc)
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            content = str(content)
            # Qdrant error format: "Collection `name` doesn't exist!"
            import re
            m = re.search(r"Collection [`'\"]?([^`'\"]+)[`'\"]?", content)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
        return ""

    async def _undo_breaker_failure(self) -> None:
        """Undo the failure count bump caused by a non-retryable 404."""
        async with self.circuit_breaker._lock:
            stats = self.circuit_breaker._stats
            if stats.failure_count > 0:
                stats.failure_count -= 1
            if stats.total_failures > 0:
                stats.total_failures -= 1


# ============================================================================
# Mock Client for Testing
# ============================================================================


class MockQdrantClientInternal:
    """Internal mock Qdrant client for testing without actual Qdrant."""

    def __init__(self):
        self._collections: Dict[str, Dict[str, Any]] = {}
        self._points: Dict[str, Dict[str, Dict[str, Any]]] = {}

    async def get_collections(self):
        """Mock get_collections."""

        class MockCollection:
            def __init__(self, name):
                self.name = name

        class MockCollections:
            def __init__(self, collections):
                self.collections = [MockCollection(n) for n in collections]

        return MockCollections(list(self._collections.keys()))

    async def create_collection(self, collection_name: str, vectors_config):
        """Mock create_collection."""
        self._collections[collection_name] = {
            "vector_size": vectors_config.size,
            "distance": vectors_config.distance,
        }
        self._points[collection_name] = {}

    async def delete_collection(self, collection_name: str):
        """Mock delete_collection."""
        self._collections.pop(collection_name, None)
        self._points.pop(collection_name, None)

    async def upsert(self, collection_name: str, points):
        """Mock upsert."""
        if collection_name not in self._points:
            self._points[collection_name] = {}

        for point in points:
            self._points[collection_name][str(point.id)] = {
                "vector": point.vector,
                "payload": point.payload,
            }

    async def search(
        self,
        collection_name: str,
        query_vector,
        limit: int = 10,
        score_threshold=None,
        query_filter=None,
        with_payload=True,
        with_vectors=False,
    ):
        """Mock search with basic similarity."""
        import math

        class MockResult:
            def __init__(self, id, score, payload, vector):
                self.id = id
                self.score = score
                self.payload = payload
                self.vector = vector

        results = []
        points = self._points.get(collection_name, {})

        for point_id, data in points.items():
            # Simple cosine similarity approximation
            score = self._cosine_similarity(query_vector, data["vector"])

            if score_threshold and score < score_threshold:
                continue

            results.append(
                MockResult(
                    id=point_id,
                    score=score,
                    payload=data["payload"] if with_payload else None,
                    vector=data["vector"] if with_vectors else None,
                )
            )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def delete(self, collection_name: str, points_selector):
        """Mock delete."""
        if collection_name in self._points:
            for point_id in points_selector.points:
                self._points[collection_name].pop(str(point_id), None)

    async def get_collection(self, collection_name: str):
        """Mock get_collection."""

        class MockVectorConfig:
            def __init__(self, size, distance):
                self.size = size
                self.distance = distance

        class MockParams:
            def __init__(self, vectors):
                self.vectors = vectors

        class MockConfig:
            def __init__(self, params):
                self.params = params

        class MockStatus:
            value = "green"

        class MockCollectionInfo:
            def __init__(self, collection_data, points_count):
                self.status = MockStatus()
                self.indexed_vectors_count = (
                    points_count  # New API (qdrant-client 1.12+)
                )
                self.vectors_count = points_count  # Keep for backwards compat
                self.points_count = points_count
                self.segments_count = 1
                self.config = MockConfig(
                    MockParams(
                        MockVectorConfig(
                            collection_data.get("vector_size", 384),
                            type(
                                "Distance",
                                (),
                                {"value": collection_data.get("distance", "Cosine")},
                            )(),
                        )
                    )
                )

        collection_data = self._collections.get(collection_name, {})
        points_count = len(self._points.get(collection_name, {}))
        return MockCollectionInfo(collection_data, points_count)

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset=None,
        with_payload: bool = True,
        with_vectors: bool = False,
        scroll_filter=None,
    ):
        """Mock scroll through points."""

        class MockPoint:
            def __init__(self, id, payload, vector):
                self.id = id
                self.payload = payload
                self.vector = vector

        points = self._points.get(collection_name, {})
        point_list = list(points.items())

        # Handle pagination via offset
        start_idx = 0
        if offset:
            for idx, (pid, _) in enumerate(point_list):
                if pid == str(offset):
                    start_idx = idx + 1
                    break

        # Slice for pagination
        page = point_list[start_idx : start_idx + limit]

        mock_points = [
            MockPoint(
                id=pid,
                payload=data["payload"] if with_payload else None,
                vector=data["vector"] if with_vectors else None,
            )
            for pid, data in page
        ]

        # Determine next offset
        next_offset = None
        if start_idx + limit < len(point_list):
            next_offset = point_list[start_idx + limit][0]

        return mock_points, next_offset

    async def close(self):
        """Mock close."""
        pass


# ============================================================================
# Factory Function
# ============================================================================


def create_qdrant_client(config: Dict[str, Any]) -> QdrantClient:
    """
    Create a QdrantClient from configuration.

    Environment variables take precedence over config file values.
    This allows dual-mode operation (localhost for dev, Docker hostname for prod).

    Environment variables (following NAMING_POLICY UBP_ prefix):
        - UBP_QDRANT_HOST: Qdrant server hostname (default: localhost)
        - UBP_QDRANT_PORT: Qdrant HTTP port (default: 6333)
        - UBP_QDRANT_GRPC_PORT: Qdrant gRPC port (default: 6334)
        - UBP_QDRANT_API_KEY: Optional API key for Qdrant Cloud

    Args:
        config: Configuration dictionary with qdrant and reliability settings

    Returns:
        Configured QdrantClient instance
    """
    import os

    qdrant_config = config.get("qdrant", {})
    reliability_config = config.get("reliability", {})

    # FIX-PORT-003 v1.8.2: Environment variables override config file (dual-mode support)
    # Priority: ENV > config.json > defaults
    # ENV var names aligned with .env file (UBP_QDRANT__ prefix with double underscore)
    host = os.getenv("UBP_QDRANT__HOST", qdrant_config.get("host", "localhost"))
    port = int(os.getenv("UBP_QDRANT__PORT", qdrant_config.get("port", 6333)))
    grpc_port = int(
        os.getenv("UBP_QDRANT__GRPC_PORT", qdrant_config.get("grpc_port", 6334))
    )
    api_key = os.getenv("UBP_QDRANT__API_KEY", qdrant_config.get("api_key"))

    logger.info(f"Qdrant client configured: host={host}, port={port}")

    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=reliability_config.get("circuit_breaker_threshold", 5),
        timeout_seconds=reliability_config.get("circuit_breaker_timeout", 60),
    )

    retry_config = RetryConfig(
        max_retries=reliability_config.get("max_retries", 3),
        initial_delay=reliability_config.get("retry_delay_seconds", 1),
        max_delay=reliability_config.get("max_retry_delay_seconds", 30),
        backoff_multiplier=reliability_config.get("retry_backoff_multiplier", 2),
    )

    return QdrantClient(
        host=host,
        port=port,
        grpc_port=grpc_port,
        prefer_grpc=qdrant_config.get("prefer_grpc", False),
        timeout=qdrant_config.get("timeout", 30),
        api_key=api_key,
        circuit_breaker_config=circuit_breaker_config,
        retry_config=retry_config,
    )
