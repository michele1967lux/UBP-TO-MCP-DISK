"""rag_qdrant.adapter

UBP framework bridge layer for the rag_qdrant core module.

IRON RULES enforced:
- No business logic here
- Provider contains technical implementation
- Adapter only coordinates UBP lifecycle, DI access, ctx forwarding

Operations exposed are driven by manifest.json and invoked via:
POST /api/modules/rag_qdrant/{operation}

MCP-COMPAT (ARCH-008): Added OperationContext support for dual REST/MCP compatibility.
All public methods accept ctx: OperationContext = None as LAST parameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set, Union
import logging

# ModuleLoader may run with different sys.path setups.
# Prefer the canonical import path used across core modules, but fall back to
# a direct sibling import when only <...>/modules/cores is on sys.path.
try:
    from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:  # pragma: no cover
    from _shared import BaseHybridModule
    from _shared.operation_context import OperationContext

from .providers import RAGQdrant

logger = logging.getLogger(__name__)


class RagQdrantAdapter(BaseHybridModule):
    """Thin adapter delegating to RAGQdrant provider."""

    def __init__(self, module_path: Path, **kwargs: Any):
        super().__init__(module_path, **kwargs)
        self.kwargs = kwargs

        self.provider: Optional[RAGQdrant] = None
        self._initialized: bool = False

    async def _resolve_redis_strict(self):
        """
        Resolve Redis client from DI container (Pure DI - No Fallback).

        Resolution strategy:
        1. String key "system_redis_client" (explicit, stable)
        2. Type key aioredis.Redis (standard DI)

        Returns:
            Redis client instance or None if not available

        Note:
            This method does NOT fall back to EventBus internals.
            If Redis is required and not available, caller should handle appropriately.
        """
        if not self.di_container:
            logger.warning(
                f"[{self.manifest.name}] DI container not available - "
                "cannot resolve Redis"
            )
            return None

        # Strategy 1: String key (preferred - explicit and stable)
        try:
            client = await self.di_container.resolve("system_redis_client")
            logger.info(
                f"[{self.manifest.name}] Redis resolved via string key: {type(client)}"
            )
            return client
        except (ValueError, KeyError) as e:
            logger.warning(
                f"[{self.manifest.name}] Redis string key resolution failed: {e}"
            )
        except Exception as e:
            logger.error(
                f"[{self.manifest.name}] Redis string key resolution error: {type(e).__name__}: {e}"
            )

        # Strategy 2: Type key (fallback - standard DI)
        try:
            import redis.asyncio as aioredis

            client = await self.di_container.resolve(aioredis.Redis)
            logger.info(
                f"[{self.manifest.name}] Redis resolved via type key: {type(client)}"
            )
            return client
        except (ValueError, KeyError) as e:
            logger.warning(
                f"[{self.manifest.name}] Redis type key resolution failed: {e}"
            )
        except Exception as e:
            logger.error(
                f"[{self.manifest.name}] Redis type key resolution error: {type(e).__name__}: {e}"
            )

        logger.warning(
            f"[{self.manifest.name}] Redis not registered in DI container - using in-memory fallback"
        )
        return None

    async def initialize(self) -> Dict[str, Any]:
        """Initialize provider + dependencies (Pure DI - No Fallback)."""
        if self._initialized:
            return {
                "status": "already_initialized",
                "module": self.manifest.name,
            }

        logger.info(
            f"[{self.manifest.name}] Initializing with: "
            f"di_container={self.di_container is not None}, "
            f"event_bus={self.event_bus is not None}"
        )

        # ========================================================
        # REDIS RESOLUTION (Pure DI - No Fallback on EventBus)
        # ========================================================
        redis_client = await self._resolve_redis_strict()

        # Redis is OPTIONAL for rag_qdrant (metadata storage only)
        # But we log clearly if unavailable
        if redis_client:
            logger.info(f"[{self.manifest.name}] Redis resolved via DI container")
        else:
            logger.warning(
                f"[{self.manifest.name}] Redis NOT available via DI - "
                "collection metadata will be in-memory only! "
                "Ensure Redis is running and registered in DI container."
            )

        self.provider = RAGQdrant(
            self.module_path,
            event_bus=self.event_bus,
            publisher=self.publisher,
            redis_client=redis_client,
        )
        assert self.provider is not None

        result = await self.provider.initialize()
        self._initialized = True
        return result

    async def shutdown(self) -> None:
        if self.provider:
            await self.provider.shutdown()
        self._initialized = False
        self.provider = None

    async def health_check(self) -> Dict[str, Any]:
        if not self.provider:
            return {
                "module": self.manifest.name,
                "status": "unhealthy",
                "initialized": False,
                "components": {"provider": {"status": "not_initialized"}},
            }
        return await self.provider.health_check()

    # ------------------------------------------------------------------
    # MCP-COMPAT: OperationContext helpers (ARCH-008)
    # ------------------------------------------------------------------

    def _build_context_from_di(self) -> OperationContext:
        """
        Build OperationContext from DI container — backward compatibility for REST path.
        
        MCP-COMPAT: When ctx is not provided (REST path), this method constructs
        an OperationContext from the DI container state.
        
        Returns:
            OperationContext with default values
        """
        # In rag_qdrant, we don't have direct client_id/user_id in DI.
        # The security context is passed via the ctx parameter from the router.
        # This method provides a minimal fallback context for internal calls.
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

    # ------------------------------------------------------------------
    # Security Context Helpers
    # ------------------------------------------------------------------

    def _require_ctx(self, ctx: Any) -> Any:
        """
        Validate and return security context.

        Args:
            ctx: Security context to validate

        Returns:
            The validated ctx (for chaining)

        Raises:
            ValueError: If ctx is None or missing user info
        """
        if not ctx or not hasattr(ctx, "user") or not ctx.user:
            raise ValueError("Security context required for this operation")
        if not hasattr(ctx.user, "user_id"):
            raise ValueError("Security context must contain user_id")
        return ctx

    def _is_admin(self, ctx: Any) -> bool:
        """
        Check if the current user is an administrator.

        Args:
            ctx: Security context with user info

        Returns:
            True if user has admin role, False otherwise
        """
        if not ctx or not hasattr(ctx, "user") or not ctx.user:
            return False
        if not hasattr(ctx.user, "roles"):
            return False

        roles = ctx.user.roles
        # Ensure roles is iterable
        if not isinstance(roles, (list, set, tuple)):
            return False

        return "admin" in roles

    def _require_admin(self, ctx: Any, operation: str) -> Any:
        """
        Require admin privileges for an operation.

        Args:
            ctx: Security context
            operation: Name of the operation for error logging

        Returns:
            The validated ctx

        Raises:
            PermissionError: If user is not admin
            ValueError: If ctx is invalid
        """
        ctx = self._require_ctx(ctx)
        if not self._is_admin(ctx):
            logger.warning(
                f"Unauthorized {operation} attempt by user {ctx.user.user_id}",
                extra={"user_id": ctx.user.user_id, "operation": operation},
            )
            raise PermissionError(f"Only administrators can perform: {operation}")
        return ctx

    # ------------------------------------------------------------------
    # Manifest operations (Admin-protected)
    # ------------------------------------------------------------------

    async def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
        ctx: Any = None,
        # Parameter alias for consistency
        collection_name: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Add document to collection. Admin only."""
        self._require_admin(ctx, "add_document")
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")
        # Parameter aliasing
        resolved_collection = collection or collection_name
        return await self.provider.add_document(
            doc_id=doc_id, text=text, metadata=metadata, collection=resolved_collection
        )

    async def add_document_internal(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Internal add document - for trusted inter-module calls only.

        This method bypasses security checks because authorization is enforced
        by the CALLING module (e.g., rag_orchestrator verifies ACL permissions
        before delegating the technical indexing operation here).

        Security Note: This method should ONLY be called by modules that have
        already verified user permissions. Direct API exposure is NOT allowed.
        """
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")
        result = await self.provider.add_document(
            doc_id=doc_id, text=text, metadata=metadata, collection=collection
        )
        # KB-AWARE: Invalidate kb_has_content cache on document upload
        try:
            _redis = await self._resolve_redis_strict()
            if _redis and collection:
                # Invalidate all user caches that might reference this collection
                # Pattern: ubp:kb_check:*  — broad invalidation is safe (TTL 1h)
                async for key in _redis.scan_iter(match="ubp:kb_check:*"):
                    await _redis.delete(key)
                logger.info("[KB-CACHE-INVALIDATE] Cleared kb_check cache after document upload to %s", collection)
        except Exception as _e:
            logger.debug("[KB-CACHE-INVALIDATE] Cache invalidation failed: %s", _e)
        return result

    async def check_duplicate_internal(
        self,
        collection: str,
        content_hash: str,
        filename: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Internal: check if a document with the same content_hash exists."""
        if not self.provider:
            return None
        return await self.provider.check_duplicate(collection, content_hash, filename)

    async def load_dedup_index_internal(self, collection: str) -> Set[str]:
        """Internal: load all content_hashes for batch dedup optimization."""
        if not self.provider:
            return set()
        return await self.provider.load_dedup_index(collection)

    async def query(
        self,
        query_text: Optional[str] = None,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
        # Parameter aliases for API consistency (BUG-001, BUG-003 fix)
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Query collection for relevant documents. Admin only (users go via rag_orchestrator).

        Args:
            query_text: The search query (preferred parameter)
            top_k: Number of results to return (default: 5)
            collection: Target collection name (preferred parameter)
            filter: Metadata filters (preferred parameter)
            ctx: Security context
            collection_name: Alias for 'collection' (for API consistency)
            filters: Alias for 'filter' (for API consistency)
            query: Alias for 'query_text' (for REST/MCP consistency)
        """
        self._require_admin(ctx, "query")
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")

        # Parameter aliasing: accept both query_text and query
        resolved_query = query_text or query
        if not resolved_query:
            raise ValueError("Either 'query_text' or 'query' parameter is required")

        # Parameter aliasing: accept both collection and collection_name
        resolved_collection = collection or collection_name
        # Parameter aliasing: accept both filter and filters
        resolved_filter = filter or filters

        return await self.provider.query(
            query_text=resolved_query,
            top_k=top_k,
            collection=resolved_collection,
            filter=resolved_filter,
        )

    async def query_internal(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        # Parameter aliases for consistency
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Internal query method for module-to-module calls.

        Used by rag_orchestrator.RAGPipeline for retrieval step.
        No ctx required - caller is responsible for ACL checks before calling.
        """
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")

        # Parameter aliasing
        resolved_collection = collection or collection_name
        resolved_filter = filter or filters

        return await self.provider.query(
            query_text=query_text,
            top_k=top_k,
            collection=resolved_collection,
            filter=resolved_filter,
        )

    async def create_collection(
        self,
        collection_name: str,
        vector_size: Optional[int] = None,
        distance: Optional[str] = None,
        description: Optional[str] = None,
        embedding_model: Optional[str] = None,
        kb_type: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        ctx: Any = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Create a new collection with optional metadata. Admin only.

        Args:
            collection_name: Name of the collection to create
            vector_size: Vector dimension (optional, auto-derived from embedding_model)
            distance: Distance metric (optional, uses config default)
            description: Human-readable description of the collection (optional)
            embedding_model: Embedding model to use (e.g., 'nomic-embed-text', 'all-MiniLM-L6-v2')
            kb_type: Knowledge base type ('universal', 'client', 'personal')
            chunk_size: Chunk size for text splitting (default: 512)
            chunk_overlap: Chunk overlap for text splitting (default: 50)
            ctx: Security context (required)

        Returns:
            Dict with creation status and collection metadata
        """
        self._require_admin(ctx, "create_collection")
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")
        return await self.provider.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            distance=distance,
            description=description,
            embedding_model=embedding_model,
            kb_type=kb_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def create_collection_internal(
        self,
        collection_name: str,
        vector_size: Optional[int] = None,
        distance: Optional[str] = None,
        description: Optional[str] = None,
        embedding_model: Optional[str] = None,
        kb_type: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Internal create collection - for trusted inter-module calls only.

        This method bypasses security checks because authorization is enforced
        by the CALLING module (e.g., rag_orchestrator or admin_users verifies
        permissions before delegating the technical collection creation here).

        Security Note: This method should ONLY be called by modules that have
        already verified user permissions. Direct API exposure is NOT allowed.

        Args:
            collection_name: Name of the collection to create
            vector_size: Vector dimension (optional, auto-derived from embedding_model)
            distance: Distance metric (optional, uses config default)
            description: Human-readable description of the collection (optional)
            embedding_model: Embedding model to use
            kb_type: Knowledge base type ('universal', 'client', 'personal')
            chunk_size: Chunk size for text splitting (default: 512)
            chunk_overlap: Chunk overlap for text splitting (default: 50)

        Returns:
            Dict with creation status and collection metadata
        """
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")
        return await self.provider.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            distance=distance,
            description=description,
            embedding_model=embedding_model,
            kb_type=kb_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def delete_collection(
        self, collection_name: str, ctx: Any = None, **_: Any
    ) -> Dict[str, Any]:
        """Delete a collection. Admin only."""
        self._require_admin(ctx, "delete_collection")
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")
        return await self.provider.delete_collection(collection_name=collection_name)

    # ------------------------------------------------------------------
    # Backward-compatible helpers (Admin-protected)
    # ------------------------------------------------------------------

    async def delete_document(
        self,
        doc_id: str,
        collection: Optional[str] = None,
        ctx: Any = None,
        # Parameter alias for consistency
        collection_name: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Delete a document. Admin only."""
        self._require_admin(ctx, "delete_document")
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")
        # Parameter aliasing
        resolved_collection = collection or collection_name
        return await self.provider.delete_document(
            doc_id=doc_id, collection=resolved_collection
        )

    async def delete_document_internal(
        self,
        doc_id: str,
        collection: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Internal helper to delete a document without ACL checks."""
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")
        return await self.provider.delete_document(
            doc_id=doc_id,
            collection=collection,
        )

    async def list_collections(self, ctx: Any = None, **_: Any):
        """List all collections. Admin only."""
        self._require_admin(ctx, "list_collections")
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")
        return await self.provider.list_collections()

    async def list_documents(
        self,
        collection: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        ctx: Any = None,
        # Parameter alias for consistency
        collection_name: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """List documents in a collection with pagination.

        Access Control:
        - Admins: Can list documents from any collection
        - Users: Can only list documents from their own personal KB (personal_{user_id[:8]})

        Args:
            collection: Collection name (uses default if not specified)
            limit: Maximum number of documents to return (default 100)
            offset: Number of documents to skip (default 0)
            ctx: Security context (required)
            collection_name: Alias for 'collection' (for API consistency)

        Returns:
            Dict with 'documents' list and pagination metadata
        """
        ctx = self._require_ctx(ctx)
        is_admin = self._is_admin(ctx)

        # Parameter aliasing
        resolved_collection = collection or collection_name

        # ACL: Admin can access any collection, users only their personal KB
        if not is_admin and resolved_collection:
            user_id = ctx.user.user_id
            expected_prefix = f"personal_{user_id[:8]}"
            if not resolved_collection.startswith(expected_prefix):
                logger.warning(
                    f"[SECURITY] User {user_id} attempted to list documents from {resolved_collection}",
                    extra={"user_id": user_id, "collection": resolved_collection},
                )
                raise PermissionError(
                    f"Access denied: You can only list documents from your own Personal KB"
                )

        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")
        return await self.provider.list_documents(
            collection=resolved_collection,
            limit=limit,
            offset=offset,
        )

    async def clear(
        self, collection: Optional[str] = None, ctx: Any = None, **_: Any
    ) -> Dict[str, Any]:
        """Clear all documents from a collection. Admin only."""
        self._require_admin(ctx, "clear")
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")
        return await self.provider.clear(collection=collection)

    async def clear_internal(self, collection_name: str) -> Dict[str, Any]:
        """Internal method to clear a collection (delete all points).

        Bypasses Admin check. Trusted caller (Orchestrator) must validate ownership.
        Used for user self-service operations like clearing personal KB.

        Args:
            collection_name: Name of the collection to clear

        Returns:
            Dict with operation result
        """
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")

        result = await self.provider.clear(collection=collection_name)

        logger.info(
            f"[CLEAR_INTERNAL] Collection {collection_name} cleared",
            extra={"collection": collection_name, "result": result},
        )

        return result

    async def check_rebuild_needed(self, collection_name: str) -> Dict[str, Any]:
        """Check if a collection needs rebuild (empty or dimension mismatch).

        Internal method for inter-module use. No auth check — caller must verify.

        Returns:
            Dict with needs_rebuild, reason, allow_delete, details
        """
        if not self.provider or not self.provider.operation_handler:
            raise RuntimeError("rag_qdrant provider/operation_handler not initialized")
        return await self.provider.operation_handler.check_rebuild_needed(collection_name)

    async def get_stats(self, ctx: Any = None, **_: Any) -> Dict[str, Any]:
        """Get module statistics. Admin only."""
        self._require_admin(ctx, "get_stats")
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")
        return await self.provider.get_stats()

    async def get_collection_details(
        self,
        collection_name: str,
        include_documents: bool = False,
        uploader_filter: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
        ctx: Any = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Get unified collection view with modes A/B/C.

        TASK #83: Unified Collection View - replaces multiple endpoints

        Access Control:
        - Admins: Can access any collection
        - Users: Can only access their own personal KB (personal_{user_id[:8]})

        Modes:
        - Mode A (default): Settings + Stats only (include_documents=False)
        - Mode B: Filtered stats by uploader (uploader_filter=X)
        - Mode C: Full view with paginated documents (include_documents=True)

        Args:
            collection_name: Name of the collection to query
            include_documents: If True, include paginated document list (Mode C)
            uploader_filter: Filter by uploader_id (Mode B - affects stats)
            limit: Max documents to return (default 500, Mode C only)
            offset: Documents to skip for pagination (Mode C only)
            ctx: Security context (required)

        Returns:
            Dict with unified structure:
            - settings: {collection_name, vector_size, distance_metric, embedding_model, ...}
            - stats: {total_documents, total_chunks, unique_uploaders, by_uploader: {...}}
            - documents: [...] (only if include_documents=True)
            - pagination: {limit, offset, has_more} (only if include_documents=True)
        """
        ctx = self._require_ctx(ctx)
        is_admin = self._is_admin(ctx)

        # ACL: Admin can access any collection, users only their personal KB
        if not is_admin:
            user_id = ctx.user.user_id
            expected_prefix = f"personal_{user_id[:8]}"
            if not collection_name.startswith(expected_prefix):
                logger.warning(
                    f"[SECURITY] User {user_id} attempted to access collection {collection_name}",
                    extra={"user_id": user_id, "collection": collection_name},
                )
                raise PermissionError(
                    f"Access denied: You can only view details of your own Personal KB"
                )

        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")
        return await self.provider.get_collection_details(
            collection_name=collection_name,
            include_documents=include_documents,
            uploader_filter=uploader_filter,
            limit=limit,
            offset=offset,
        )

    async def list_embedding_models(self, ctx: Any = None, **_: Any) -> Dict[str, Any]:
        """List all available embedding models from all configured providers. Admin only.

        TASK #76: Grand Unified Registry - Aggregates models from:
        - Local (Sentence-Transformers): Always available, downloaded on-demand
        - Ollama (Container): Lists models currently installed in ubp-ollama
        - OpenAI (Cloud): Available if UBP_OPENAI_API_KEY is set
        - Cohere (Cloud): Available if UBP_COHERE_API_KEY is set

        Args:
            ctx: Security context (required)

        Returns:
            Dict with:
            - models: List of model dicts with id, name, dims, provider, available, description
            - providers: Dict of provider status {available, reason, model_count}
            - total_count: int
            - available_count: int
            - duration_ms: number
        """
        self._require_admin(ctx, "list_embedding_models")
        if not self.provider:
            raise RuntimeError("rag_qdrant provider not initialized")
        return await self.provider.list_embedding_models()
