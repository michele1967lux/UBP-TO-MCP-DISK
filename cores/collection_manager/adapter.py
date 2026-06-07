"""
Collection Manager UBP Framework Bridge Layer

Integrates technical providers with UBP module system.

MCP-COMPAT (ARCH-008): Added OperationContext support for dual REST/MCP compatibility.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import uuid
import logging
from datetime import datetime, UTC

from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule
# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    from _shared.operation_context import OperationContext

from ubp_enterprise_hybrid.backend.app.infra.event_bus import Event
from .providers import MockDBClient

logger = logging.getLogger(__name__)


class CollectionManagerAdapter(BaseHybridModule):
    """UBP adapter for collection manager module."""

    def __init__(self, module_path: Path, **kwargs):
        """Initialize the collection manager."""
        super().__init__(module_path, **kwargs)

        self.db_client = None
        self.collections: Dict[str, Dict[str, Any]] = {}
        self.items: Dict[str, List[Dict[str, Any]]] = {}

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
        
        # Already an OperationContext
        if isinstance(ctx, OperationContext):
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

    async def initialize(self) -> None:
        """Initialize database connection."""
        logger.info(
            f"Initializing {self.manifest.name} module",
            extra={"mod_name": self.manifest.name},
        )

        # In production, this would connect to PostgreSQL
        # For skeleton, we use in-memory storage
        self.db_client = MockDBClient(self.config)

        logger.info(f"✅ {self.manifest.name} initialized successfully (mock DB mode)")

    async def shutdown(self) -> None:
        """Shutdown and cleanup."""
        logger.info(f"Shutting down {self.manifest.name} module")

        if self.db_client:
            await self.db_client.close()
            logger.debug("Database client closed")

        logger.info(f"✅ {self.manifest.name} shutdown successfully")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        db_status = "unknown"

        if self.db_client:
            try:
                db_status = await self.db_client.health_check()
                logger.debug(
                    "Database health check completed", extra={"status": db_status}
                )
            except Exception as e:
                db_status = "unhealthy"
                logger.warning("Database health check failed", extra={"error": str(e)})

        status = "healthy" if db_status == "healthy" else "degraded"

        return {
            "module": self.manifest.name,
            "status": status,
            "database": {
                "status": db_status,
                "collections_count": len(self.collections),
            },
        }

    async def create_collection(
        self,
        name: str,
        description: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new collection with request tracking.

        Args:
            name: Collection name
            description: Optional description
            schema: Optional JSON schema for validation
            metadata: Optional metadata
            request_id: Optional tracking ID (auto-generated if not provided)

        Returns:
            Created collection info with request_id

        Raises:
            ValueError: If name is empty or invalid
        """
        # Generate request ID for tracking
        if not request_id:
            request_id = str(uuid.uuid4())

        # Validate name is not empty
        if not name or not name.strip():
            raise ValueError("Collection name cannot be empty")

        logger.info(
            f"Creating collection '{name}' [request_id={request_id}, "
            f"has_schema={schema is not None}, has_metadata={metadata is not None}]"
        )

        collection_id = str(uuid.uuid4())

        collection = {
            "id": collection_id,
            "name": name,
            "description": description,
            "schema": schema,
            "metadata": metadata or {},
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "item_count": 0,
            "request_id": request_id,
        }

        # Store in DB (mock)
        self.collections[collection_id] = collection
        self.items[collection_id] = []

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "collection.created",
                {
                    "request_id": request_id,
                    "collection_id": collection_id,
                    "name": name,
                    "metadata": metadata,
                },
            )

        return collection
    async def list_collections(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        List all collections.

        Args:
            filter: Optional filter criteria
            limit: Maximum results to return
            offset: Offset for pagination

        Raises:
            ValueError: If limit or offset are invalid
        """
        # Validate pagination parameters
        if limit is not None and limit < 0:
            raise ValueError("Limit must be non-negative")

        if offset is not None and offset < 0:
            raise ValueError("Offset must be non-negative")

        limit = (
            int(limit)
            if limit is not None
            else int(self.config["defaults"]["default_page_size"])
        )
        offset = int(offset) if offset is not None else 0

        # Simple filtering (in production, use DB queries)
        collections = list(self.collections.values())

        # Apply filter if provided
        if filter:
            # Simple name filter for example
            if "name" in filter:
                collections = [c for c in collections if filter["name"] in c["name"]]

        # Pagination
        total = len(collections)
        _limit = (
            int(limit)
            if isinstance(limit, int)
            else int(self.config["defaults"]["default_page_size"])
        )
        _offset = int(offset) if isinstance(offset, int) else 0
        collections = collections[_offset : _offset + _limit]

        logger.info(
            "List completed",
            extra={"results": len(collections), "total": total},
        )

        return {
            "collections": collections,
            "total": total,
            "limit": _limit,
            "offset": _offset,
        }

    async def search_collections(
        self, query: str, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search collections by name or description.

        Args:
            query: Search query string
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            Search results with collections
        """
        limit = (
            int(limit)
            if limit is not None
            else int(self.config["defaults"]["default_page_size"])
        )
        offset = int(offset) if offset is not None else 0

        logger.info("Searching collections", extra={"query": query})

        # Search in name and description
        collections = [
            c
            for c in self.collections.values()
            if query.lower() in c["name"].lower()
            or (c.get("description") and query.lower() in c["description"].lower())
        ]

        # Pagination
        total = len(collections)
        collections = collections[offset : offset + limit]

        logger.info(
            "Search completed",
            extra={"query": query, "results": len(collections), "total": total},
        )

        return {
            "collections": collections,
            "total": total,
            "limit": limit,
            "offset": offset,
            "query": query,
        }

    async def get_collection(self, collection_id: str) -> Dict[str, Any]:
        """Get collection details."""
        if collection_id not in self.collections:
            raise ValueError(f"Collection not found: {collection_id}")

        return self.collections[collection_id]

    async def update_collection(
        self, collection_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update collection metadata."""
        if collection_id not in self.collections:
            raise ValueError(f"Collection not found: {collection_id}")

        collection = self.collections[collection_id]

        # Update allowed fields
        for key in ["name", "description", "metadata"]:
            if key in updates:
                collection[key] = updates[key]

        collection["updated_at"] = datetime.now(UTC).isoformat()

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "collection.updated",
                {"collection_id": collection_id, "updates": updates},
            )

        return collection

    async def delete_collection(self, collection_id: str) -> Dict[str, Any]:
        """
        Delete a collection.

        Returns status even if collection doesn't exist (graceful handling).
        """
        if collection_id not in self.collections:
            logger.warning(
                "Attempted to delete non-existent collection",
                extra={"collection_id": collection_id},
            )
            return {
                "collection_id": collection_id,
                "status": "not_found",
                "deleted": False,
            }

        # Delete collection and items
        del self.collections[collection_id]
        del self.items[collection_id]

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "collection.deleted", {"collection_id": collection_id}
            )

        return {"collection_id": collection_id, "status": "deleted", "deleted": True}

    async def add_item(
        self, collection_id: str, item_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add item to collection."""
        if collection_id not in self.collections:
            raise ValueError(f"Collection not found: {collection_id}")

        collection = self.collections[collection_id]

        # Validate against schema if enforced
        if self.config["validation"]["enforce_schema"] and collection["schema"]:
            # Simple validation (in production, use jsonschema)
            pass

        # Create item - include both flattened fields and nested 'data' for compatibility
        item_id = str(uuid.uuid4())
        item = {
            "id": item_id,
            "collection_id": collection_id,
            **item_data,
            "data": item_data,
            "created_at": datetime.now(UTC).isoformat(),
        }

        # Store item
        self.items[collection_id].append(item)

        # Update collection count
        collection["item_count"] = len(self.items[collection_id])
        collection["updated_at"] = datetime.now(UTC).isoformat()

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "collection.item.added",
                {
                    "collection_id": collection_id,
                    "item_id": item_id,
                    "data": item_data,
                },
            )

        return item

    async def get_items(
        self,
        collection_id: str,
        filter: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get items from collection."""
        if collection_id not in self.collections:
            raise ValueError(f"Collection not found: {collection_id}")

        limit = limit or self.config["defaults"]["default_page_size"]

        items = self.items[collection_id]

        # Apply filter if provided (simple implementation)
        if filter:
            # Could implement more sophisticated filtering
            pass

        # Limit results
        items = items[:limit]

        return {
            "collection_id": collection_id,
            "items": items,
            "count": len(items),
            "total": len(self.items[collection_id]),
        }

    async def list_items(
        self,
        collection_id: str,
        filter: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Alias for get_items for API consistency."""
        return await self.get_items(collection_id, filter, limit)

    async def get_item(self, collection_id: str, item_id: str) -> Dict[str, Any]:
        """
        Get a specific item from a collection.

        Args:
            collection_id: Collection identifier
            item_id: Item identifier

        Returns:
            Item data

        Raises:
            ValueError: If collection or item not found
        """
        if collection_id not in self.collections:
            raise ValueError(f"Collection not found: {collection_id}")

        # Find item in collection
        for item in self.items[collection_id]:
            if item["id"] == item_id:
                logger.info(
                    "Item retrieved",
                    extra={"collection_id": collection_id, "item_id": item_id},
                )
                return item

        raise ValueError(f"Item not found: {item_id}")

    async def update_item(
        self, collection_id: str, item_id: str, item_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update an item in a collection.

        Args:
            collection_id: Collection identifier
            item_id: Item identifier
            item_data: Updated item data

        Returns:
            Updated item data

        Raises:
            ValueError: If collection or item not found
        """
        if collection_id not in self.collections:
            raise ValueError(f"Collection not found: {collection_id}")

        # Find and update item
        for i, item in enumerate(self.items[collection_id]):
            if item["id"] == item_id:
                # Update the 'data' field while preserving id and metadata
                updated_item = {
                    "id": item_id,
                    "collection_id": collection_id,
                    **item_data,
                    "data": item_data,
                    "created_at": item.get("created_at"),
                }
                self.items[collection_id][i] = updated_item

                logger.info(
                    "Item updated",
                    extra={"collection_id": collection_id, "item_id": item_id},
                )

                # Publish event if available
                if self.publisher:
                    await self.publisher.publish(
                        "collection.item.updated",
                        {
                            "collection_id": collection_id,
                            "item_id": item_id,
                            "item": updated_item,
                        },
                    )

                return updated_item

        raise ValueError(f"Item not found: {item_id}")

    async def delete_item(self, collection_id: str, item_id: str) -> Dict[str, Any]:
        """
        Delete an item from a collection.

        Args:
            collection_id: Collection identifier
            item_id: Item identifier

        Returns:
            Deletion status

        Raises:
            ValueError: If collection or item not found
        """
        if collection_id not in self.collections:
            raise ValueError(f"Collection not found: {collection_id}")

        # Find and remove item
        for i, item in enumerate(self.items[collection_id]):
            if item["id"] == item_id:
                self.items[collection_id].pop(i)

                logger.info(
                    "Item deleted",
                    extra={"collection_id": collection_id, "item_id": item_id},
                )

                # Publish event if available
                if self.publisher:
                    await self.publisher.publish(
                        "collection.item.deleted",
                        {"collection_id": collection_id, "item_id": item_id},
                    )

                return {
                    "status": "deleted",
                    "deleted": True,
                    "collection_id": collection_id,
                    "item_id": item_id,
                }

        raise ValueError(f"Item not found: {item_id}")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get module statistics.

        Returns:
            Dictionary with module statistics
        """
        return {
            "module": self.manifest.name,
            "total_collections": len(self.collections),
            "total_items": sum(
                len(col.get("items", [])) for col in self.collections.values()
            ),
            "db_connected": self.db_client.connected if self.db_client else False,
        }

    async def get_collection_stats(self, collection_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific collection.

        Args:
            collection_id: Collection ID

        Returns:
            Dictionary with collection statistics

        Raises:
            ValueError: If collection doesn't exist
        """
        if collection_id not in self.collections:
            raise ValueError(f"Collection '{collection_id}' not found")

        collection = self.collections[collection_id]
        # Use self.items to get actual item count
        item_count = len(self.items.get(collection_id, []))

        return {
            "collection_id": collection_id,
            "name": collection["name"],
            "item_count": item_count,
            "created_at": collection.get("created_at"),
            "schema": collection.get("schema"),
        }

    # Event handlers
    async def on_collection_create(self, event: Event):
        """Handle collection.create event."""
        payload = event.payload
        await self.create_collection(
            name=payload["name"],
            description=payload.get("description"),
            schema=payload.get("schema"),
            metadata=payload.get("metadata"),
        )

    async def on_collection_item_add(self, event: Event):
        """Handle collection.item.add event."""
        payload = event.payload
        await self.add_item(
            collection_id=payload["collection_id"], item_data=payload["item_data"]
        )


# Mock DB client for skeleton
