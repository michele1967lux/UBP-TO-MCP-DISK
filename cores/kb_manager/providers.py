"""KBItemProvider — Pure CRUD logic for structured KB items.

No UBP framework dependencies. All Qdrant/Redis access is injected.

Item = 1 Qdrant point (no chunking). Items are small structured data
(menu items, drinks, products) with text representations for embedding.

Uses rag_qdrant's add_document / delete_document / query for Qdrant ops.

v1.0.0: Initial release (KB-MANAGER)
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Redis key templates
VERSION_KEY = "ubp:kb_mgr:ver:{collection}:{item_id}"
AUDIT_KEY = "ubp:kb_mgr:audit:{collection}"
HISTORY_KEY = "ubp:kb_mgr:manage_history:{conv_id}"

# Constants
DOC_ID_PREFIX = "kbm:"
MAX_AUDIT_ENTRIES = 1000
MAX_VERSIONS = 10
HISTORY_TTL = 86400  # 24h


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str) -> str:
    """Simple slugify: lowercase, replace non-alnum with hyphens, strip."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")[:64]


# _config.json constants (KB-MGR-001)
CONFIG_DOC_ID = "_config"
CONFIG_SOURCE = "kb_manager_config"
CONFIG_EMBED_TEXT = "client configuration settings preferences"
COL_EXISTS_KEY = "ubp:kb_mgr:col_exists:{collection}"


class KBItemProvider:
    """CRUD operations for structured KB items in Qdrant + Redis."""

    def __init__(
        self,
        rag_qdrant_adapter,
        redis_client,
        schema_registry,
        config: Dict[str, Any],
    ):
        self.rag = rag_qdrant_adapter
        self.redis = redis_client
        self.schema_registry = schema_registry
        self.config = config
        self._max_versions = config.get("max_versions_per_item", MAX_VERSIONS)
        self._max_audit = config.get("max_audit_entries", MAX_AUDIT_ENTRIES)
        self._collections_created: set = set()  # KB-MGR-001: lazy creation cache

    # =================================================================
    # COLLECTION MANAGEMENT (KB-MGR-001)
    # =================================================================

    async def _ensure_collection(self, collection: str) -> None:
        """Create collection if it doesn't exist (idempotent, cached)."""
        if collection in self._collections_created:
            return
        # Check Redis flag first (avoid Qdrant round-trip)
        flag_key = COL_EXISTS_KEY.format(collection=collection)
        if self.redis:
            try:
                exists = await self.redis.get(flag_key)
                if exists:
                    self._collections_created.add(collection)
                    return
            except Exception:
                pass
        # Create (idempotent)
        try:
            result = await self.rag.create_collection_internal(
                collection_name=collection,
                kb_type="client",
                description=f"KB Manager dedicated collection: {collection}",
            )
            status_str = result.get("status", "unknown") if isinstance(result, dict) else "ok"
        except Exception as e:
            logger.warning("[KB-MGR] Collection ensure failed: %s", e)
            status_str = "error"
        # Set Redis flag (TTL 24h)
        if self.redis:
            try:
                await self.redis.set(flag_key, "1", ex=86400)
            except Exception:
                pass
        self._collections_created.add(collection)
        logger.info("[KB-MGR] Collection ensured: %s (%s)", collection, status_str)

    async def save_config(
        self, collection: str, config_data: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Save/update _config.json in Qdrant collection."""
        await self._ensure_collection(collection)
        # Delete existing config point (if any)
        try:
            await self.rag.delete_document_internal(
                doc_id=CONFIG_DOC_ID, collection=collection,
            )
        except Exception:
            pass  # May not exist yet
        # Upsert new config point
        metadata = {
            "doc_id": CONFIG_DOC_ID,
            "source": CONFIG_SOURCE,
            "config_data": json.dumps(config_data, ensure_ascii=False),
            "updated_at": _now_iso(),
            "updated_by": user_id,
        }
        result = await self.rag.add_document_internal(
            doc_id=CONFIG_DOC_ID,
            text=CONFIG_EMBED_TEXT,
            metadata=metadata,
            collection=collection,
        )
        logger.info("[KB-MGR] Config saved in %s by %s", collection, user_id)
        return {"status": "saved", "collection": collection}

    async def load_config(self, collection: str) -> Optional[Dict[str, Any]]:
        """Load _config.json from Qdrant collection via scroll+filter."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        try:
            filter_obj = Filter(must=[
                FieldCondition(key="source", match=MatchValue(value=CONFIG_SOURCE))
            ])
            scroll_result = await self.rag.provider.qdrant_client.scroll(
                collection_name=collection,
                limit=1,
                filter_conditions=filter_obj,
                with_payload=True,
                with_vectors=False,
            )
            points = scroll_result.get("points", [])
            if not points:
                return None
            payload = points[0].get("payload", {})
            raw = payload.get("config_data", "{}")
            return json.loads(raw) if isinstance(raw, str) else raw
        except Exception as e:
            logger.debug("[KB-MGR] load_config from %s failed: %s", collection, e)
            return None

    # =================================================================
    # PUBLIC OPERATIONS
    # =================================================================

    async def add_item(
        self,
        collection: str,
        category: str,
        data: Dict[str, Any],
        schema_name: Optional[str] = None,
        user_id: str = "system",
    ) -> Dict[str, Any]:
        """Add a new structured item to a Qdrant collection.

        Returns: {status, item_id, collection, doc_id}
        """
        await self._ensure_collection(collection)
        start = time.perf_counter()

        # 1. Generate item_id (slugify name or UUID)
        item_id = self._generate_item_id(data)

        # 2. Check for collision
        item_id = await self._ensure_unique_id(collection, item_id)

        # 3. Build text representation for embedding
        effective_schema = schema_name or "generic"
        text_repr = self.schema_registry.build_text_repr(effective_schema, data)

        # 4. Build metadata payload
        now = _now_iso()
        metadata = {
            "item_id": item_id,
            "category": category,
            "data": json.dumps(data, ensure_ascii=False),
            "text_repr": text_repr,
            "schema_name": effective_schema,
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "created_by": user_id,
            "source": "kb_manager",
        }

        # 5. Upsert to Qdrant via rag_qdrant.add_document_internal
        doc_id = f"{DOC_ID_PREFIX}{item_id}"
        result = await self.rag.add_document_internal(
            doc_id=doc_id,
            text=text_repr,
            metadata=metadata,
            collection=collection,
        )

        if result.get("status") == "failed":
            logger.error(
                "[KB-MGR] add_item failed: %s", result.get("error"),
                extra={"item_id": item_id, "collection": collection},
            )
            return {"status": "failed", "error": result.get("error")}

        # 6. Save version in Redis
        await self._save_version(collection, item_id, data, user_id, version=1)

        # 7. Audit log
        await self._audit_log(
            collection, "add", item_id, user_id,
            summary=f"Added {data.get('name', item_id)} ({category})",
        )

        duration = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "[KB-MGR] Item added: %s in %s (%.1fms)",
            item_id, collection, duration,
        )
        return {
            "status": "added",
            "item_id": item_id,
            "doc_id": doc_id,
            "collection": collection,
            "duration_ms": duration,
        }

    async def update_item(
        self,
        collection: str,
        item_id: str,
        data: Dict[str, Any],
        schema_name: Optional[str] = None,
        user_id: str = "system",
    ) -> Dict[str, Any]:
        """Update an existing item (partial merge).

        Returns: {status, item_id, version}
        """
        await self._ensure_collection(collection)
        start = time.perf_counter()

        # 1. Get existing item
        existing = await self._get_item_by_id(collection, item_id)
        if not existing:
            return {"status": "not_found", "item_id": item_id}

        # 2. Merge data (partial update)
        old_data = json.loads(existing.get("data", "{}"))
        merged = {**old_data, **data}
        new_version = int(existing.get("version", 1)) + 1

        # 3. Delete old point
        doc_id = f"{DOC_ID_PREFIX}{item_id}"
        await self.rag.delete_document_internal(
            doc_id=doc_id, collection=collection,
        )

        # 4. Build new text repr
        effective_schema = schema_name or existing.get("schema_name", "generic")
        text_repr = self.schema_registry.build_text_repr(effective_schema, merged)

        # 5. Upsert new point
        now = _now_iso()
        metadata = {
            "item_id": item_id,
            "category": existing.get("category", ""),
            "data": json.dumps(merged, ensure_ascii=False),
            "text_repr": text_repr,
            "schema_name": effective_schema,
            "version": new_version,
            "created_at": existing.get("created_at", now),
            "updated_at": now,
            "created_by": existing.get("created_by", user_id),
            "updated_by": user_id,
            "source": "kb_manager",
        }

        result = await self.rag.add_document_internal(
            doc_id=doc_id, text=text_repr,
            metadata=metadata, collection=collection,
        )

        if result.get("status") == "failed":
            return {"status": "failed", "error": result.get("error")}

        # 6. Save version + audit
        await self._save_version(collection, item_id, merged, user_id, new_version)
        await self._audit_log(
            collection, "update", item_id, user_id,
            summary=f"Updated {merged.get('name', item_id)} to v{new_version}",
        )

        duration = round((time.perf_counter() - start) * 1000, 1)
        return {
            "status": "updated",
            "item_id": item_id,
            "version": new_version,
            "duration_ms": duration,
        }

    async def delete_item(
        self,
        collection: str,
        item_id: str,
        user_id: str = "system",
    ) -> Dict[str, Any]:
        """Delete an item from Qdrant.

        Returns: {status, item_id}
        """
        await self._ensure_collection(collection)
        doc_id = f"{DOC_ID_PREFIX}{item_id}"
        result = await self.rag.delete_document_internal(
            doc_id=doc_id, collection=collection,
        )

        await self._audit_log(
            collection, "delete", item_id, user_id,
            summary=f"Deleted {item_id}",
        )

        logger.info("[KB-MGR] Item deleted: %s from %s", item_id, collection)
        return {"status": "deleted", "item_id": item_id}

    async def get_item(
        self, collection: str, item_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a single item by item_id.

        Returns parsed item dict or None.
        """
        await self._ensure_collection(collection)
        raw = await self._get_item_by_id(collection, item_id)
        if not raw:
            return None
        return self._format_item(raw)

    async def search_items(
        self,
        collection: str,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Semantic search for items via rag_qdrant.query_internal.

        Filters results to source == "kb_manager" and optional category.
        """
        await self._ensure_collection(collection)
        result = await self.rag.query_internal(
            query_text=query, top_k=limit * 2,  # Over-fetch to account for filtering
            collection=collection,
        )

        items = []
        for r in result.get("results", []):
            meta = r.get("metadata", {})
            if meta.get("source") != "kb_manager":
                continue
            if category and meta.get("category") != category:
                continue
            items.append({
                "item_id": meta.get("item_id"),
                "category": meta.get("category"),
                "data": json.loads(meta.get("data", "{}")),
                "text_repr": meta.get("text_repr", ""),
                "score": r.get("score"),
            })
            if len(items) >= limit:
                break

        return {"items": items, "count": len(items)}

    async def list_items(
        self,
        collection: str,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List items by broad semantic search on 'menu item product'.

        Uses query_internal with a generic query, then filters by
        source == "kb_manager" in payload. For category-specific listing,
        uses the category name as query text for better relevance.
        """
        await self._ensure_collection(collection)
        query_text = category if category else "menu item product drink"
        result = await self.rag.query_internal(
            query_text=query_text,
            top_k=200,  # Fetch many to filter
            collection=collection,
        )

        items = []
        for r in result.get("results", []):
            meta = r.get("metadata", {})
            if meta.get("source") != "kb_manager":
                continue
            if category and meta.get("category") != category:
                continue
            items.append(self._format_item(meta))

        # Deduplicate by item_id (semantic search may return duplicates)
        seen = set()
        unique = []
        for item in items:
            if item["item_id"] not in seen:
                seen.add(item["item_id"])
                unique.append(item)
        items = unique

        total = len(items)
        items = items[offset:offset + limit]

        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # =================================================================
    # VERSIONING (Redis)
    # =================================================================

    async def _save_version(
        self,
        collection: str,
        item_id: str,
        data: Dict[str, Any],
        user_id: str,
        version: int,
    ) -> None:
        """Save version snapshot to Redis."""
        if not self.redis:
            return
        key = VERSION_KEY.format(collection=collection, item_id=item_id)
        try:
            raw = await self.redis.get(key)
            versions = json.loads(raw) if raw else {"versions": [], "current_version": 0}

            versions["versions"].append({
                "v": version,
                "data": data,
                "updated_by": user_id,
                "updated_at": _now_iso(),
            })
            # Keep only last N versions
            if len(versions["versions"]) > self._max_versions:
                versions["versions"] = versions["versions"][-self._max_versions:]
            versions["current_version"] = version

            await self.redis.set(key, json.dumps(versions, ensure_ascii=False))
        except Exception as e:
            logger.warning("[KB-MGR] Version save failed: %s", e)

    async def get_versions(
        self, collection: str, item_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get version history for an item."""
        if not self.redis:
            return None
        key = VERSION_KEY.format(collection=collection, item_id=item_id)
        raw = await self.redis.get(key)
        return json.loads(raw) if raw else None

    # =================================================================
    # AUDIT LOG (Redis)
    # =================================================================

    async def _audit_log(
        self,
        collection: str,
        action: str,
        item_id: str,
        user_id: str,
        summary: str = "",
    ) -> None:
        """Append audit entry to Redis list."""
        if not self.redis:
            return
        key = AUDIT_KEY.format(collection=collection)
        entry = json.dumps({
            "action": action,
            "item_id": item_id,
            "user_id": user_id,
            "timestamp": _now_iso(),
            "summary": summary,
        }, ensure_ascii=False)
        try:
            await self.redis.lpush(key, entry)
            await self.redis.ltrim(key, 0, self._max_audit - 1)
        except Exception as e:
            logger.warning("[KB-MGR] Audit log failed: %s", e)

    # =================================================================
    # MANAGE HISTORY (Redis, TTL 24h)
    # =================================================================

    async def save_manage_history(
        self, conv_id: str, user_id: str, user_msg: str, assistant_msg: str
    ) -> None:
        """Save manage conversation turn to Redis."""
        if not self.redis:
            return
        key = HISTORY_KEY.format(conv_id=conv_id)
        now = _now_iso()
        entries = [
            json.dumps({"role": "user", "content": user_msg, "timestamp": now}),
            json.dumps({"role": "assistant", "content": assistant_msg, "timestamp": now}),
        ]
        try:
            for entry in entries:
                await self.redis.rpush(key, entry)
            await self.redis.expire(key, HISTORY_TTL)
        except Exception as e:
            logger.warning("[KB-MGR] History save failed: %s", e)

    async def load_manage_history(
        self, conv_id: str, user_id: str
    ) -> List[Dict[str, str]]:
        """Load manage conversation history from Redis."""
        if not self.redis:
            return []
        key = HISTORY_KEY.format(conv_id=conv_id)
        try:
            raw_list = await self.redis.lrange(key, 0, -1)
            return [
                {"role": json.loads(r)["role"], "content": json.loads(r)["content"]}
                for r in raw_list
            ]
        except Exception as e:
            logger.warning("[KB-MGR] History load failed: %s", e)
            return []

    # =================================================================
    # INTERNAL HELPERS
    # =================================================================

    async def _get_item_by_id(
        self, collection: str, item_id: str
    ) -> Optional[Dict[str, Any]]:
        """Find item by item_id in Qdrant via query with exact doc_id match.

        Uses rag_qdrant query with filter on doc_id field.
        """
        doc_id = f"{DOC_ID_PREFIX}{item_id}"
        # Use a simple text query with the item name to narrow search,
        # then filter by doc_id in results.
        # Check both metadata (from query_internal) and top-level doc_id
        result = await self.rag.query_internal(
            query_text=item_id.replace("-", " "),
            top_k=20,
            collection=collection,
        )
        for r in result.get("results", []):
            meta = r.get("metadata", {})
            r_doc_id = r.get("doc_id") or meta.get("doc_id")
            r_item_id = meta.get("item_id")
            if r_doc_id == doc_id or r_item_id == item_id:
                return meta
        return None

    async def _ensure_unique_id(self, collection: str, item_id: str) -> str:
        """Check for ID collision and append suffix if needed."""
        existing = await self._get_item_by_id(collection, item_id)
        if not existing:
            return item_id
        # Append numeric suffix
        for i in range(2, 100):
            candidate = f"{item_id}-{i}"
            if not await self._get_item_by_id(collection, candidate):
                return candidate
        # Fallback to UUID suffix
        return f"{item_id}-{uuid.uuid4().hex[:6]}"

    def _generate_item_id(self, data: Dict[str, Any]) -> str:
        """Generate slug from name or UUID."""
        name = data.get("name", "")
        if name:
            return _slugify(name)
        return uuid.uuid4().hex[:8]

    def _format_item(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Format raw Qdrant payload into clean item dict."""
        data_raw = payload.get("data", "{}")
        if isinstance(data_raw, str):
            try:
                data = json.loads(data_raw)
            except (json.JSONDecodeError, TypeError):
                data = {}
        else:
            data = data_raw
        return {
            "item_id": payload.get("item_id", ""),
            "category": payload.get("category", ""),
            "data": data,
            "schema_name": payload.get("schema_name", "generic"),
            "version": payload.get("version", 1),
            "created_at": payload.get("created_at", ""),
            "updated_at": payload.get("updated_at", ""),
            "created_by": payload.get("created_by", ""),
            "text_repr": payload.get("text_repr", ""),
        }
