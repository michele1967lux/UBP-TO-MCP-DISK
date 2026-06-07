"""KBManagerAdapter — UBP bridge for structured KB item management.

Follows the 3-file pattern: adapter delegates to providers.py,
validates security context, publishes events.

v1.0.0: Initial release (KB-MANAGER)

MCP-COMPAT (ARCH-008): Added OperationContext support for dual REST/MCP compatibility.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import (
        OperationContext,
        extract_user_id as _extract_user_id,
        is_operation_context_like,
    )
except ModuleNotFoundError:
    try:
        from _shared.operation_context import (
            OperationContext,
            extract_user_id as _extract_user_id,
            is_operation_context_like,
        )
    except ModuleNotFoundError:
        from ..._shared.operation_context import (
            OperationContext,
            extract_user_id as _extract_user_id,
            is_operation_context_like,
        )

logger = logging.getLogger(__name__)


def _load_config(module_path: Path) -> Dict[str, Any]:
    """Load config.json with defaults."""
    config_file = module_path / "config.json"
    if config_file.exists():
        return json.loads(config_file.read_text(encoding="utf-8"))
    return {}


class KBManagerAdapter:
    """Main adapter for kb_manager module.

    Implements all operations defined in manifest.json.
    Security checks are done here; providers.py is pure logic.
    """

    def __init__(
        self,
        module_path: Path,
        di_container: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        self.module_path = module_path
        self.di_container = di_container
        self.event_bus = event_bus
        self.config = _load_config(module_path)

        # Components (set during initialize)
        self._redis = None
        self._rag_qdrant = None
        self._provider = None
        self._schema_registry = None

        # State
        self._initialized = False

    # =================================================================
    # MCP-COMPAT: OperationContext helpers (ARCH-008)
    # =================================================================

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
        """
        Normalize any context format to OperationContext.
        
        MCP-COMPAT: Handles both legacy security context (ctx.user.user_id) 
        and new OperationContext format for backward compatibility.
        
        Args:
            ctx: Either OperationContext, legacy security context, or None
            
        Returns:
            OperationContext instance
        """
        if ctx is None:
            return self._build_context_from_di()
        
        # Already an OperationContext-like object, even across import roots.
        if is_operation_context_like(ctx):
            return ctx
        
        # Legacy security context format (ctx.user.user_id, ctx.user.roles)
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
        
        # Fallback
        return self._build_context_from_di()

    # =================================================================
    # LIFECYCLE
    # =================================================================

    async def initialize(self, ctx=None, **kwargs) -> Dict[str, Any]:
        """Initialize module: resolve dependencies, create provider."""
        start = time.perf_counter()

        # Import providers here (lazy, avoids import-time issues)
        from .providers import KBItemProvider
        from .schemas import SchemaRegistry

        # 1. Resolve Redis
        if self.di_container:
            try:
                import redis.asyncio as aioredis
                self._redis = await self.di_container.resolve(aioredis.Redis)
                logger.info("[KB-MGR] Redis client resolved")
            except Exception as e:
                logger.warning("[KB-MGR] Redis not available: %s", e)

        # 2. Resolve rag_qdrant (required)
        if self.di_container:
            try:
                self._rag_qdrant = await self.di_container.resolve("rag_qdrant")
                logger.info("[KB-MGR] rag_qdrant module resolved")
            except Exception as e:
                logger.error("[KB-MGR] rag_qdrant not available: %s", e)
                return {
                    "status": "failed",
                    "module": "kb_manager",
                    "reason": "rag_qdrant_unavailable",
                }

        if not self._rag_qdrant:
            logger.error("[KB-MGR] rag_qdrant is required but not available")
            return {
                "status": "failed",
                "module": "kb_manager",
                "reason": "rag_qdrant_unavailable",
            }

        # 3. Create schema registry
        self._schema_registry = SchemaRegistry(self._redis, self.config)

        # 4. Create provider
        self._provider = KBItemProvider(
            rag_qdrant_adapter=self._rag_qdrant,
            redis_client=self._redis,
            schema_registry=self._schema_registry,
            config=self.config,
        )

        self._initialized = True
        duration = round((time.perf_counter() - start) * 1000, 1)
        logger.info("[KB-MGR] kb_manager initialized (%.1fms)", duration)
        return {"status": "initialized", "module": "kb_manager", "duration_ms": duration}

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._initialized = False
        logger.info("[KB-MGR] kb_manager shut down")

    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        return {
            "module": "kb_manager",
            "status": "healthy" if self._initialized else "not_initialized",
            "redis_connected": self._redis is not None,
            "rag_qdrant_connected": self._rag_qdrant is not None,
        }

    # =================================================================
    # COLLECTION RESOLUTION (KB-MGR-001)
    # =================================================================

    def _resolve_collection(self, client_id: str) -> str:
        """Build dedicated collection name from client_id."""
        return f"ubp_kbm_{client_id[:8]}"

    # =================================================================
    # SECURITY HELPERS
    # =================================================================

    def _require_ctx(self, ctx: Any) -> Any:
        if not ctx:
            raise ValueError("Security context required for this operation")
        if is_operation_context_like(ctx):
            if not ctx.user_id:
                raise ValueError("Security context must contain user_id")
            return ctx
        if not hasattr(ctx, "user") or not ctx.user:
            raise ValueError("Security context required for this operation")
        if not hasattr(ctx.user, "user_id"):
            raise ValueError("Security context must contain user_id")
        return ctx

    def _is_admin(self, ctx: Any) -> bool:
        if not ctx:
            return False
        if is_operation_context_like(ctx):
            roles = getattr(ctx, "roles", [])
            return "admin" in roles if isinstance(roles, (list, set, tuple)) else False
        if not hasattr(ctx, "user"):
            return False
        roles = getattr(ctx.user, "roles", [])
        return "admin" in roles if isinstance(roles, (list, set, tuple)) else False

    def _require_admin(self, ctx: Any, operation: str = "") -> Any:
        ctx = self._require_ctx(ctx)
        if not self._is_admin(ctx):
            raise PermissionError(f"Only administrators can perform: {operation}")
        return ctx

    def _ensure_initialized(self) -> None:
        if not self._initialized or not self._provider:
            raise RuntimeError("kb_manager module not initialized")

    # =================================================================
    # OPERATIONS (public, with security)
    # =================================================================

    async def add_item(
        self,
        client_id: str,
        category: str,
        data: Dict[str, Any],
        schema_name: Optional[str] = None,
        ctx: Any = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Add a new structured item. Admin only."""
        self._require_admin(ctx, "add_item")
        self._ensure_initialized()
        collection = self._resolve_collection(client_id)

        # Validate schema if provided
        if schema_name:
            schema = await self._schema_registry.get(schema_name)
            if schema:
                self._schema_registry.validate(schema_name, data)

        result = await self._provider.add_item(
            collection=collection,
            category=category,
            data=data,
            schema_name=schema_name,
            user_id=_extract_user_id(ctx),
        )

        if self.event_bus and result.get("status") == "added":
            try:
                await self.event_bus.publish("kb_manager.item_added", {
                    "item_id": result.get("item_id"),
                    "collection": collection,
                    "category": category,
                    "user_id": _extract_user_id(ctx),
                })
            except Exception:
                pass  # Fire-and-forget

        return result

    async def update_item(
        self,
        client_id: str,
        item_id: str,
        data: Dict[str, Any],
        schema_name: Optional[str] = None,
        ctx: Any = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Update an existing item (partial merge). Admin only."""
        self._require_admin(ctx, "update_item")
        self._ensure_initialized()
        collection = self._resolve_collection(client_id)

        result = await self._provider.update_item(
            collection=collection,
            item_id=item_id,
            data=data,
            schema_name=schema_name,
            user_id=_extract_user_id(ctx),
        )

        if self.event_bus and result.get("status") == "updated":
            try:
                await self.event_bus.publish("kb_manager.item_updated", {
                    "item_id": item_id,
                    "collection": collection,
                    "version": result.get("version"),
                    "user_id": _extract_user_id(ctx),
                })
            except Exception:
                pass

        return result

    async def delete_item(
        self,
        client_id: str,
        item_id: str,
        ctx: Any = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Delete an item. Admin only."""
        self._require_admin(ctx, "delete_item")
        self._ensure_initialized()
        collection = self._resolve_collection(client_id)

        result = await self._provider.delete_item(
            collection=collection,
            item_id=item_id,
            user_id=_extract_user_id(ctx),
        )

        if self.event_bus and result.get("status") == "deleted":
            try:
                await self.event_bus.publish("kb_manager.item_deleted", {
                    "item_id": item_id,
                    "collection": collection,
                    "user_id": _extract_user_id(ctx),
                })
            except Exception:
                pass

        return result

    async def get_item(
        self,
        client_id: str,
        item_id: str,
        ctx: Any = None,
        **_: Any,
    ) -> Optional[Dict[str, Any]]:
        """Get a single item by ID. Authenticated users."""
        self._require_ctx(ctx)
        self._ensure_initialized()
        collection = self._resolve_collection(client_id)
        return await self._provider.get_item(collection=collection, item_id=item_id)

    async def search_items(
        self,
        client_id: str,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
        ctx: Any = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Semantic search for items. Authenticated users."""
        self._require_ctx(ctx)
        self._ensure_initialized()
        collection = self._resolve_collection(client_id)
        return await self._provider.search_items(
            collection=collection,
            query=query,
            category=category,
            limit=limit,
        )

    async def list_items(
        self,
        client_id: str,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        ctx: Any = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """List items with optional category filter. Authenticated users."""
        self._require_ctx(ctx)
        self._ensure_initialized()
        collection = self._resolve_collection(client_id)
        return await self._provider.list_items(
            collection=collection,
            category=category,
            limit=limit,
            offset=offset,
        )

    async def register_schema(
        self,
        schema_name: str,
        schema_def: Dict[str, Any],
        ctx: Any = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Register a domain schema. Admin only."""
        self._require_admin(ctx, "register_schema")
        self._ensure_initialized()
        await self._schema_registry.register(schema_name, schema_def)
        return {"status": "registered", "schema_name": schema_name}

    # =================================================================
    # CONFIG OPERATIONS (KB-MGR-001)
    # =================================================================

    async def save_config(
        self, client_id: str, config_data: Dict[str, Any],
        ctx: Any = None, **_: Any,
    ) -> Dict[str, Any]:
        """Save client _config.json. Admin only."""
        self._require_admin(ctx, "save_config")
        self._ensure_initialized()
        collection = self._resolve_collection(client_id)
        return await self._provider.save_config(
            collection=collection, config_data=config_data,
            user_id=_extract_user_id(ctx),
        )

    async def load_config(
        self, client_id: str, ctx: Any = None, **_: Any,
    ) -> Optional[Dict[str, Any]]:
        """Load client _config.json. Authenticated."""
        self._require_ctx(ctx)
        self._ensure_initialized()
        collection = self._resolve_collection(client_id)
        return await self._provider.load_config(collection=collection)

    async def auto_assign_collection(
        self, client_id: str, admin_clients_module: Any, ctx: Any = None,
    ) -> None:
        """Ensure dedicated collection is in client's universal_kbs_assigned."""
        collection = self._resolve_collection(client_id)
        try:
            client = await admin_clients_module.get_client_internal(client_id)
            kb_config = client.get("kb_config", {})
            assigned = kb_config.get("universal_kbs_assigned", [])
            if collection not in assigned:
                assigned.append(collection)
                kb_config["universal_kbs_assigned"] = assigned
                await admin_clients_module.update_client(
                    client_id=client_id, kb_config=kb_config, ctx=ctx,
                )
                logger.info(
                    "[KB-MGR] Auto-assigned %s to client %s",
                    collection, client_id[:8],
                )
        except Exception as e:
            logger.warning("[KB-MGR] Auto-assign failed: %s", e)

    # =================================================================
    # INTERNAL METHODS (for inter-module calls, no auth checks)
    # =================================================================

    async def load_schemas_from_preset(
        self, item_schemas: Dict[str, Dict[str, Any]]
    ) -> int:
        """Load schemas from preset domain_config. Called by apply-preset."""
        if not self._initialized or not self._schema_registry:
            return 0
        return await self._schema_registry.load_schemas_from_config(item_schemas)

    async def save_manage_history(
        self, conv_id: str, user_id: str, user_msg: str, assistant_msg: str
    ) -> None:
        """Save manage conversation turn."""
        if self._provider:
            await self._provider.save_manage_history(
                conv_id, user_id, user_msg, assistant_msg
            )

    async def load_manage_history(
        self, conv_id: str, user_id: str
    ) -> List[Dict[str, str]]:
        """Load manage conversation history."""
        if self._provider:
            return await self._provider.load_manage_history(conv_id, user_id)
        return []
