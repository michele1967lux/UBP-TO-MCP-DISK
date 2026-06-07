"""
UBP Framework Bridge for Feedback Module

Integrates RedisFeedbackProvider with UBP module system.
Provides secure feedback collection and analytics for RAG responses.

ROADMAP v1.5.0 - FEAT-EVAL-001
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
import uuid

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

import redis.asyncio as aioredis

from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule
from .providers import RedisFeedbackProvider, FeedbackType


logger = logging.getLogger(__name__)


class FeedbackAdapter(BaseHybridModule):
    """
    UBP adapter for feedback management.

    Provides feedback collection and analytics for RAG responses.
    Implements security controls and ACL checks.

    Architecture:
    - All operations extract user_id from ctx (NEVER from payload)
    - All operations return request_id for tracing
    - Global stats require admin access
    - Users can only see their own feedback unless admin
    """

    def __init__(self, module_path: Path, **kwargs):
        """
        Initialize the adapter.

        Args:
            module_path: Path to module directory
            **kwargs: Additional arguments (event_bus, di_container)
        """
        super().__init__(module_path, **kwargs)
        self.provider: Optional[RedisFeedbackProvider] = None
        self.total_feedback_count = 0

        # Track initialization status
        self._init_status: Dict[str, Any] = {
            "status": "not_initialized",
            "redis_available": False,
            "provider_initialized": False,
            "error": None,
        }

    # MCP-COMPAT: OperationContext helpers (ARCH-008)
    def _build_context_from_di(self) -> OperationContext:
        """Build OperationContext from DI — backward compatibility for REST path."""
        return OperationContext(
            client_id="default",
            user_id=None,
            session_id=None,
            source="rest",
        )

    def _normalize_ctx(self, ctx: Any) -> OperationContext:
        """Normalize any context format to OperationContext."""
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

    async def initialize(self) -> None:
        """
        Initialize module and provider.

        Resolves Redis client from DI container.
        """
        logger.info(f"Initializing {self.manifest.name}")

        # Get Redis from DI container
        redis_client = None

        if self.di_container:
            try:
                redis_client = await self.di_container.resolve(aioredis.Redis)
                self._init_status["redis_available"] = True
                logger.info("✅ Redis client resolved from DI container")
            except Exception as e:
                logger.warning(f"Could not resolve Redis from DI: {e}")
                self._init_status["error"] = str(e)

        if not redis_client:
            self._init_status["status"] = "degraded"
            self._init_status["error"] = "Redis client not available"
            logger.error(
                f"Redis client not available - {self.manifest.name} module disabled"
            )
            return

        # Initialize provider
        self.provider = RedisFeedbackProvider(
            redis_client=redis_client,
            stats_ttl=self.config.get("stats_ttl_seconds", 86400 * 90),
            feedback_ttl=self.config.get("feedback_ttl_seconds", 86400 * 365),
        )

        self._init_status["status"] = "healthy"
        self._init_status["provider_initialized"] = True

        logger.info(f"✅ {self.manifest.name} initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown module and release resources."""
        logger.info(f"Shutting down {self.manifest.name}")
        self.provider = None
        self._init_status["provider_initialized"] = False
        self._init_status["status"] = "shutdown"
        logger.info(f"✅ {self.manifest.name} shutdown complete")

    async def health_check(self, **kwargs) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health status with module info and statistics
        """
        return {
            "module": self.manifest.name,
            "version": self.manifest.version,
            "status": "healthy" if self.provider else "unhealthy",
            "init_status": self._init_status,
            "total_feedback_count": self.total_feedback_count,
            "provider": self.provider.health_check() if self.provider else None,
        }

    # === OPERATIONS ===

    async def submit_feedback(
        self,
        response_id: str,
        feedback_type: str,
        value: Any,
        query: Optional[str] = None,
        answer: Optional[str] = None,
        collection: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Submit feedback for a RAG response.

        User ID is extracted from security context (NEVER from payload).

        Args:
            response_id: UUID of the RAG response
            feedback_type: Type of feedback ('thumbs_up', 'thumbs_down', 'rating')
            value: Feedback value (bool for thumbs, 1-5 for rating)
            query: Original user query (optional)
            answer: Assistant's response (optional)
            collection: Knowledge base collection name (optional)
            metadata: Additional metadata (optional)
            request_id: Request tracking ID (optional)
            ctx: Security context (required)

        Returns:
            Dict with feedback_id, response_id, status, and request_id
        """
        request_id = request_id or str(uuid.uuid4())

        # Validate provider
        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        # SECURITY: Get user_id from context (NEVER from payload)
        user_id = self._get_user_id_from_ctx(ctx)
        if not user_id:
            return {"error": "User not authenticated", "request_id": request_id}

        # Validate response_id
        if not response_id:
            return {"error": "response_id is required", "request_id": request_id}

        try:
            result = await self.provider.submit_feedback(
                response_id=response_id,
                user_id=user_id,
                feedback_type=feedback_type,
                value=value,
                query=query,
                answer=answer,
                collection=collection,
                metadata=metadata,
            )

            self.total_feedback_count += 1
            result["request_id"] = request_id

            # Publish event for other modules
            if hasattr(self, "publisher") and self.publisher:
                await self.publisher.publish(
                    "feedback.submitted",
                    {
                        "feedback_id": result.get("feedback_id"),
                        "response_id": response_id,
                        "feedback_type": feedback_type,
                        "user_id": user_id,
                        "collection": collection,
                        "request_id": request_id,
                    },
                )

            logger.info(
                f"Feedback submitted: {result.get('feedback_id')} "
                f"(request_id={request_id})"
            )

            return result

        except ValueError as e:
            logger.warning(f"Invalid feedback submission: {e}")
            return {"error": str(e), "request_id": request_id}
        except Exception as e:
            logger.error(f"Feedback submission failed: {e}")
            return {"error": f"Submission failed: {str(e)}", "request_id": request_id}

    async def get_response_feedback(
        self, response_id: str, request_id: Optional[str] = None, ctx=None, **kwargs
    ) -> Dict[str, Any]:
        """
        Get feedback for a specific response.

        Args:
            response_id: UUID of the RAG response
            request_id: Request tracking ID (optional)
            ctx: Security context

        Returns:
            Dict with response_id, feedback data, found status, and request_id
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        if not response_id:
            return {"error": "response_id is required", "request_id": request_id}

        try:
            feedback = await self.provider.get_response_feedback(response_id)

            return {
                "response_id": response_id,
                "feedback": feedback,
                "found": feedback is not None,
                "request_id": request_id,
            }

        except Exception as e:
            logger.error(f"Failed to get feedback: {e}")
            return {"error": str(e), "request_id": request_id}

    async def get_feedback_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        collection: Optional[str] = None,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get aggregated feedback statistics.

        Global stats (no collection filter) require admin access.

        Args:
            start_date: Start date (YYYY-MM-DD, optional)
            end_date: End date (YYYY-MM-DD, optional)
            collection: Collection filter (optional)
            request_id: Request tracking ID (optional)
            ctx: Security context

        Returns:
            Statistics dict with counts, rates, averages, and request_id
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        # ACL: Only admins can see global stats
        if not collection and not self._is_admin(ctx):
            return {
                "error": "Admin access required for global statistics",
                "request_id": request_id,
            }

        try:
            stats = await self.provider.get_feedback_stats(
                start_date=start_date, end_date=end_date, collection=collection
            )

            stats["request_id"] = request_id
            return stats

        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            return {"error": str(e), "request_id": request_id}

    async def list_feedback(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        List feedback entries.

        Users can only see their own feedback. Admins can see all or filter by user.

        Args:
            user_id: User ID filter (admin only, ignored for non-admins)
            limit: Maximum entries to return (max 100)
            offset: Pagination offset
            request_id: Request tracking ID (optional)
            ctx: Security context

        Returns:
            Dict with feedback list, count, pagination info, and request_id
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        # ACL: Non-admins can only see their own feedback
        if not self._is_admin(ctx):
            user_id = self._get_user_id_from_ctx(ctx)
            if not user_id:
                return {"error": "User not authenticated", "request_id": request_id}

        try:
            # Cap limit at 100
            safe_limit = min(limit, 100)

            feedback_list = await self.provider.list_feedback(
                user_id=user_id, limit=safe_limit, offset=offset
            )

            return {
                "feedback": feedback_list,
                "count": len(feedback_list),
                "limit": safe_limit,
                "offset": offset,
                "request_id": request_id,
            }

        except Exception as e:
            logger.error(f"Failed to list feedback: {e}")
            return {"error": str(e), "request_id": request_id}

    async def delete_feedback(
        self, response_id: str, request_id: Optional[str] = None, ctx=None, **kwargs
    ) -> Dict[str, Any]:
        """
        Delete feedback for a response.

        Users can only delete their own feedback. Admins can delete any.

        Args:
            response_id: UUID of the response
            request_id: Request tracking ID (optional)
            ctx: Security context

        Returns:
            Dict with response_id, deleted status, and request_id
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        user_id = self._get_user_id_from_ctx(ctx)
        if not user_id:
            return {"error": "User not authenticated", "request_id": request_id}

        try:
            # Check ownership before deletion (unless admin)
            if not self._is_admin(ctx):
                feedback = await self.provider.get_response_feedback(response_id)
                if feedback and feedback.get("user_id") != user_id:
                    return {
                        "error": "Access denied: can only delete own feedback",
                        "request_id": request_id,
                    }

            deleted = await self.provider.delete_feedback(response_id)

            if deleted and hasattr(self, "publisher") and self.publisher:
                await self.publisher.publish(
                    "feedback.deleted",
                    {
                        "response_id": response_id,
                        "deleted_by": user_id,
                        "request_id": request_id,
                    },
                )

            return {
                "response_id": response_id,
                "deleted": deleted,
                "request_id": request_id,
            }

        except Exception as e:
            logger.error(f"Failed to delete feedback: {e}")
            return {"error": str(e), "request_id": request_id}

    # === HELPER METHODS ===

    def _get_user_id_from_ctx(self, ctx) -> Optional[str]:
        """
        Extract user_id from security context.

        SECURITY: User ID must ALWAYS come from ctx, NEVER from payload.

        Args:
            ctx: Security context object

        Returns:
            User ID string or None if not authenticated
        """
        if ctx and hasattr(ctx, "user") and ctx.user:
            return getattr(ctx.user, "user_id", None)
        return None

    def _is_admin(self, ctx) -> bool:
        """
        Check if user has admin privileges.

        Args:
            ctx: Security context object

        Returns:
            True if user is admin, False otherwise
        """
        if ctx and hasattr(ctx, "user") and ctx.user:
            return getattr(ctx.user, "is_admin", False)
        return False
