"""
Feedback Provider - Pure Technical Logic

Zero UBP dependencies. Can be tested standalone.
Implements Redis-based feedback persistence and analytics.

Key Structure (following NAMING_POLICY.md Section 7):
- ubp:feedback:response:{response_id}      (Hash - feedback data)
- ubp:feedback:stats:daily:{YYYY-MM-DD}    (Hash - daily aggregates)
- ubp:feedback:stats:collection:{name}     (Hash - per-collection stats)
- ubp:feedback:user:{user_id}:list         (List - user's feedback IDs)
- ubp:feedback:all:list                    (List - all feedback IDs, for admin)

ROADMAP v1.5.0 - FEAT-EVAL-001
"""

from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import logging
import uuid

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback supported by the system."""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"


@dataclass
class FeedbackRecord:
    """
    Represents a feedback record.

    Attributes:
        feedback_id: Unique identifier for this feedback
        response_id: ID of the RAG response being rated
        user_id: ID of the user submitting feedback
        feedback_type: Type of feedback (thumbs_up, thumbs_down, rating)
        value: Feedback value (bool for thumbs, 1-5 for rating)
        query: Original user query (optional)
        answer_preview: Preview of the answer (truncated)
        collection: Knowledge base collection name
        created_at: Timestamp of feedback submission
        metadata: Additional metadata
    """

    feedback_id: str
    response_id: str
    user_id: str
    feedback_type: FeedbackType
    value: Any
    query: str
    answer_preview: str
    collection: str
    created_at: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feedback_id": self.feedback_id,
            "response_id": self.response_id,
            "user_id": self.user_id,
            "feedback_type": self.feedback_type.value,
            "value": self.value,
            "query": self.query,
            "answer_preview": self.answer_preview,
            "collection": self.collection,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@runtime_checkable
class FeedbackProtocol(Protocol):
    """Interface contract for feedback providers."""

    async def submit_feedback(
        self,
        response_id: str,
        user_id: str,
        feedback_type: str,
        value: Any,
        query: Optional[str] = None,
        answer: Optional[str] = None,
        collection: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit feedback for a response."""
        ...

    async def get_response_feedback(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback for a specific response."""
        ...

    async def get_feedback_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregated feedback statistics."""
        ...

    async def list_feedback(
        self, user_id: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List feedback entries."""
        ...


class RedisFeedbackProvider:
    """
    Redis-based feedback implementation.

    Provides persistent storage for user feedback on RAG responses
    with aggregated statistics and analytics support.

    Thread-safe and async-compatible.
    """

    # Key patterns following NAMING_POLICY.md Section 7
    KEY_PREFIX = "ubp:feedback"
    RESPONSE_KEY = f"{KEY_PREFIX}:response:{{response_id}}"
    DAILY_STATS_KEY = f"{KEY_PREFIX}:stats:daily:{{date}}"
    COLLECTION_STATS_KEY = f"{KEY_PREFIX}:stats:collection:{{collection}}"
    USER_LIST_KEY = f"{KEY_PREFIX}:user:{{user_id}}:list"
    ALL_LIST_KEY = f"{KEY_PREFIX}:all:list"

    # Limits
    MAX_USER_FEEDBACK_LIST = 1000
    MAX_GLOBAL_FEEDBACK_LIST = 10000
    MAX_ANSWER_PREVIEW_LENGTH = 200
    MAX_QUERY_LENGTH = 500

    def __init__(
        self,
        redis_client,
        stats_ttl: int = 86400 * 90,  # 90 days for stats
        feedback_ttl: int = 86400 * 365,  # 1 year for individual feedback
    ):
        """
        Initialize provider.

        Args:
            redis_client: Async Redis client instance
            stats_ttl: TTL for statistics keys (seconds)
            feedback_ttl: TTL for individual feedback entries (seconds)
        """
        self.redis = redis_client
        self.stats_ttl = stats_ttl
        self.feedback_ttl = feedback_ttl

        logger.debug(
            f"RedisFeedbackProvider initialized: stats_ttl={stats_ttl}s, "
            f"feedback_ttl={feedback_ttl}s"
        )

    async def submit_feedback(
        self,
        response_id: str,
        user_id: str,
        feedback_type: str,
        value: Any,
        query: Optional[str] = None,
        answer: Optional[str] = None,
        collection: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit feedback for a RAG response.

        Args:
            response_id: UUID of the RAG response
            user_id: ID of the user submitting feedback
            feedback_type: Type of feedback ('thumbs_up', 'thumbs_down', 'rating')
            value: Feedback value (bool for thumbs, 1-5 for rating)
            query: Original user query (optional)
            answer: Assistant's response (optional, will be truncated)
            collection: Knowledge base collection name (optional)
            metadata: Additional metadata (optional)

        Returns:
            Dict with feedback_id, response_id, and status

        Raises:
            ValueError: If feedback_type is invalid
        """
        # Validate feedback type
        valid_types = [ft.value for ft in FeedbackType]
        if feedback_type not in valid_types:
            raise ValueError(
                f"Invalid feedback_type: {feedback_type}. Valid types: {valid_types}"
            )

        # Validate rating value
        if feedback_type == FeedbackType.RATING.value:
            try:
                rating_value = int(value)
                if not 1 <= rating_value <= 5:
                    raise ValueError("Rating must be between 1 and 5")
                value = rating_value
            except (TypeError, ValueError):
                raise ValueError("Rating value must be an integer between 1 and 5")

        feedback_id = str(uuid.uuid4())
        now = datetime.utcnow()
        today = now.strftime("%Y-%m-%d")

        # Truncate long strings
        query_truncated = (query or "")[: self.MAX_QUERY_LENGTH]
        answer_preview = (answer or "")[: self.MAX_ANSWER_PREVIEW_LENGTH]
        collection_name = collection or "default"

        # Build feedback record
        feedback_data = {
            "feedback_id": feedback_id,
            "response_id": response_id,
            "user_id": user_id,
            "feedback_type": feedback_type,
            "value": str(value),
            "query": query_truncated,
            "answer_preview": answer_preview,
            "collection": collection_name,
            "created_at": now.isoformat(),
            "metadata": json.dumps(metadata or {}),
        }

        try:
            # Store feedback in Redis
            response_key = self.RESPONSE_KEY.format(response_id=response_id)
            await self.redis.hset(response_key, mapping=feedback_data)
            await self.redis.expire(response_key, self.feedback_ttl)

            # Add to user's feedback list
            user_list_key = self.USER_LIST_KEY.format(user_id=user_id)
            await self.redis.lpush(user_list_key, response_id)
            await self.redis.ltrim(user_list_key, 0, self.MAX_USER_FEEDBACK_LIST - 1)
            await self.redis.expire(user_list_key, self.feedback_ttl)

            # Add to global list (for admin)
            await self.redis.lpush(self.ALL_LIST_KEY, response_id)
            await self.redis.ltrim(
                self.ALL_LIST_KEY, 0, self.MAX_GLOBAL_FEEDBACK_LIST - 1
            )

            # Update statistics
            await self._update_stats(today, feedback_type, value, collection_name)

            logger.info(
                f"Feedback submitted: {feedback_id} for response {response_id} "
                f"(type={feedback_type}, user={user_id})"
            )

            return {
                "feedback_id": feedback_id,
                "response_id": response_id,
                "status": "submitted",
                "created_at": now.isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            raise

    async def _update_stats(
        self, date_str: str, feedback_type: str, value: Any, collection: str
    ) -> None:
        """
        Update aggregated statistics.

        Args:
            date_str: Date string (YYYY-MM-DD)
            feedback_type: Type of feedback
            value: Feedback value
            collection: Collection name
        """
        # Daily stats
        daily_key = self.DAILY_STATS_KEY.format(date=date_str)

        await self.redis.hincrby(daily_key, "total_feedback", 1)

        if feedback_type == FeedbackType.THUMBS_UP.value:
            await self.redis.hincrby(daily_key, "thumbs_up", 1)
        elif feedback_type == FeedbackType.THUMBS_DOWN.value:
            await self.redis.hincrby(daily_key, "thumbs_down", 1)
        elif feedback_type == FeedbackType.RATING.value:
            await self.redis.hincrby(daily_key, "ratings_count", 1)
            await self.redis.hincrbyfloat(daily_key, "ratings_sum", float(value))

        await self.redis.expire(daily_key, self.stats_ttl)

        # Collection stats
        if collection and collection != "default":
            coll_key = self.COLLECTION_STATS_KEY.format(collection=collection)
            await self.redis.hincrby(coll_key, "total_feedback", 1)

            if feedback_type == FeedbackType.THUMBS_UP.value:
                await self.redis.hincrby(coll_key, "thumbs_up", 1)
            elif feedback_type == FeedbackType.THUMBS_DOWN.value:
                await self.redis.hincrby(coll_key, "thumbs_down", 1)
            elif feedback_type == FeedbackType.RATING.value:
                await self.redis.hincrby(coll_key, "ratings_count", 1)
                await self.redis.hincrbyfloat(coll_key, "ratings_sum", float(value))

            await self.redis.expire(coll_key, self.stats_ttl)

    async def get_response_feedback(self, response_id: str) -> Optional[Dict[str, Any]]:
        """
        Get feedback for a specific response.

        Args:
            response_id: UUID of the RAG response

        Returns:
            Feedback data dict or None if not found
        """
        response_key = self.RESPONSE_KEY.format(response_id=response_id)
        data = await self.redis.hgetall(response_key)

        if not data:
            return None

        # Decode bytes if necessary (for older redis-py versions)
        decoded = {}
        for k, v in data.items():
            key = k.decode("utf-8") if isinstance(k, bytes) else k
            val = v.decode("utf-8") if isinstance(v, bytes) else v
            decoded[key] = val

        # Parse metadata JSON
        if "metadata" in decoded:
            try:
                decoded["metadata"] = json.loads(decoded["metadata"])
            except json.JSONDecodeError:
                decoded["metadata"] = {}

        return decoded

    async def get_feedback_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get aggregated feedback statistics.

        Args:
            start_date: Start date (YYYY-MM-DD), defaults to today
            end_date: End date (YYYY-MM-DD), defaults to start_date
            collection: Optional collection filter

        Returns:
            Statistics dict with counts, rates, and averages
        """
        stats = {
            "total_feedback": 0,
            "thumbs_up": 0,
            "thumbs_down": 0,
            "ratings_count": 0,
            "ratings_sum": 0.0,
            "average_rating": None,
            "satisfaction_rate": None,
            "period": {"start": start_date, "end": end_date},
        }

        if collection:
            # Get collection-specific stats
            coll_key = self.COLLECTION_STATS_KEY.format(collection=collection)
            data = await self.redis.hgetall(coll_key)

            if data:
                stats = self._parse_stats_hash(data, stats)

            stats["collection"] = collection
        else:
            # Aggregate daily stats for date range
            if not start_date:
                start_date = datetime.utcnow().strftime("%Y-%m-%d")
            if not end_date:
                end_date = start_date

            stats["period"]["start"] = start_date
            stats["period"]["end"] = end_date

            # Parse dates
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d").date()
                end = datetime.strptime(end_date, "%Y-%m-%d").date()
            except ValueError as e:
                logger.error(f"Invalid date format: {e}")
                return stats

            # Iterate through days
            current = start
            while current <= end:
                daily_key = self.DAILY_STATS_KEY.format(
                    date=current.strftime("%Y-%m-%d")
                )
                data = await self.redis.hgetall(daily_key)

                if data:
                    stats = self._parse_stats_hash(data, stats, accumulate=True)

                current = current + timedelta(days=1)

        # Calculate derived metrics
        if stats["ratings_count"] > 0:
            stats["average_rating"] = round(
                stats["ratings_sum"] / stats["ratings_count"], 2
            )

        total_thumbs = stats["thumbs_up"] + stats["thumbs_down"]
        if total_thumbs > 0:
            stats["satisfaction_rate"] = round(
                (stats["thumbs_up"] / total_thumbs) * 100, 1
            )

        return stats

    def _parse_stats_hash(
        self, data: Dict, stats: Dict[str, Any], accumulate: bool = False
    ) -> Dict[str, Any]:
        """
        Parse Redis hash data into stats dict.

        Args:
            data: Raw Redis hash data
            stats: Stats dict to update
            accumulate: Whether to add to existing values

        Returns:
            Updated stats dict
        """
        for k, v in data.items():
            key = k.decode("utf-8") if isinstance(k, bytes) else k
            val = v.decode("utf-8") if isinstance(v, bytes) else v

            if key in ["total_feedback", "thumbs_up", "thumbs_down", "ratings_count"]:
                if accumulate:
                    stats[key] += int(val)
                else:
                    stats[key] = int(val)
            elif key == "ratings_sum":
                if accumulate:
                    stats[key] += float(val)
                else:
                    stats[key] = float(val)

        return stats

    async def list_feedback(
        self, user_id: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List feedback entries.

        Args:
            user_id: Filter by user ID (None for all, admin only)
            limit: Maximum entries to return
            offset: Pagination offset

        Returns:
            List of feedback records
        """
        if user_id:
            list_key = self.USER_LIST_KEY.format(user_id=user_id)
        else:
            list_key = self.ALL_LIST_KEY

        # Get response IDs from list
        response_ids = await self.redis.lrange(list_key, offset, offset + limit - 1)

        feedback_list = []
        for response_id in response_ids:
            if isinstance(response_id, bytes):
                response_id = response_id.decode("utf-8")

            feedback = await self.get_response_feedback(response_id)
            if feedback:
                feedback_list.append(feedback)

        return feedback_list

    async def delete_feedback(self, response_id: str) -> bool:
        """
        Delete feedback for a response.

        Args:
            response_id: UUID of the response

        Returns:
            True if deleted, False if not found
        """
        response_key = self.RESPONSE_KEY.format(response_id=response_id)
        deleted = await self.redis.delete(response_key)
        return deleted > 0

    def health_check(self) -> Dict[str, Any]:
        """
        Check provider health status.

        Returns:
            Health status dict
        """
        return {
            "status": "configured" if self.redis else "not_configured",
            "stats_ttl_days": self.stats_ttl // 86400,
            "feedback_ttl_days": self.feedback_ttl // 86400,
            "max_user_feedback": self.MAX_USER_FEEDBACK_LIST,
            "max_global_feedback": self.MAX_GLOBAL_FEEDBACK_LIST,
        }
