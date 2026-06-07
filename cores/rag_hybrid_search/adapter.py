"""
UBP Framework Bridge for Hybrid Search Module
Integrates HybridSearchProvider with UBP module system.

Provides hybrid search combining dense (vector) and sparse (BM25) retrieval.

Operations:
- hybrid_search: Perform combined dense+sparse search with fusion
- sparse_search: Perform BM25-only search
- index_document: Index document for sparse search
- remove_document: Remove document from sparse index
- rebuild_index: Rebuild BM25 index from rag_qdrant
- get_index_stats: Get BM25 index statistics
- health_check: Module health status
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

from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule
from ubp_enterprise_hybrid.backend.app.infra.event_bus import Event
from .providers import HybridSearchProvider, FusionMethod, SearchResult

logger = logging.getLogger(__name__)


class HybridSearchAdapter(BaseHybridModule):
    """
    UBP adapter for hybrid search.

    Combines dense (vector) and sparse (BM25) search with score fusion.
    Follows 3-file pattern: no business logic here, only UBP integration.

    Security:
    - User ID always from ctx, never from payload
    - Admin-only operations protected
    """

    def __init__(self, module_path: Path, **kwargs):
        super().__init__(module_path, **kwargs)
        self.provider: Optional[HybridSearchProvider] = None
        self._total_searches = 0
        self._total_hybrid_searches = 0
        self._total_sparse_searches = 0

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
        """Initialize module and provider."""
        logger.info(f"Initializing {self.manifest.name}")

        # Parse fusion method from config
        fusion_str = self.config.get("fusion_method", "rrf")
        try:
            fusion_method = FusionMethod(fusion_str)
        except ValueError:
            fusion_method = FusionMethod.RRF
            logger.warning(f"Invalid fusion method '{fusion_str}', using RRF")

        # Create provider with config
        self.provider = HybridSearchProvider(
            fusion_method=fusion_method,
            dense_weight=self.config.get("dense_weight", 0.7),
            rrf_k=self.config.get("rrf_k", 60),
            bm25_k1=self.config.get("bm25_k1", 1.5),
            bm25_b=self.config.get("bm25_b", 0.75),
        )

        # Subscribe to document events for index synchronization
        # Note: EventBus.subscribe() is synchronous, not async
        if self.event_bus:
            self.event_bus.subscribe(
                "qdrant.document_indexed", self._on_document_indexed
            )
            self.event_bus.subscribe(
                "qdrant.document_deleted", self._on_document_deleted
            )
            logger.info("Subscribed to qdrant document events")

        logger.info(
            f"✅ {self.manifest.name} initialized with {fusion_method.value} fusion"
        )

    async def shutdown(self) -> None:
        """Shutdown module."""
        logger.info(f"Shutting down {self.manifest.name}")
        self.provider = None
        logger.info(f"✅ {self.manifest.name} shutdown complete")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        provider_health = self.provider.health_check() if self.provider else None

        return {
            "module": self.manifest.name,
            "version": self.manifest.version,
            "status": "healthy" if self.provider else "unhealthy",
            "total_searches": self._total_searches,
            "total_hybrid_searches": self._total_hybrid_searches,
            "total_sparse_searches": self._total_sparse_searches,
            "provider": provider_health,
        }

    # === OPERATIONS ===

    async def hybrid_search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        fusion_method: Optional[str] = None,
        dense_weight: Optional[float] = None,
        include_sparse_only: bool = False,
        include_dense_only: bool = False,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining dense and sparse retrieval.

        Args:
            query: Search query
            collection_name: Target collection
            top_k: Number of results to return
            fusion_method: Override fusion method ('rrf', 'weighted', 'max')
            dense_weight: Override dense weight for weighted fusion (0-1)
            include_sparse_only: Return only BM25 results
            include_dense_only: Return only vector results
            request_id: Optional request identifier
            ctx: Security context

        Returns:
            Dictionary with search results and metadata
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        user_id = self._get_user_id_from_ctx(ctx)
        if not user_id:
            return {"error": "User not authenticated", "request_id": request_id}

        if not query or not query.strip():
            return {"error": "Query is required", "request_id": request_id}

        if not collection_name:
            return {"error": "Collection name is required", "request_id": request_id}

        try:
            # Parse fusion method override
            fusion = None
            if fusion_method:
                try:
                    fusion = FusionMethod(fusion_method)
                except ValueError:
                    return {
                        "error": f"Invalid fusion method: {fusion_method}. Use 'rrf', 'weighted', or 'max'",
                        "request_id": request_id,
                    }

            # Get sparse results from BM25
            sparse_results: List[SearchResult] = []
            if not include_dense_only:
                sparse_results = self.provider.sparse_search(
                    collection_name=collection_name,
                    query=query,
                    top_k=top_k * 2,  # Fetch more for fusion
                )

            # Get dense results from rag_qdrant
            dense_results: List[SearchResult] = []
            if not include_sparse_only:
                qdrant_module = self._get_module("rag_qdrant")
                if qdrant_module:
                    qdrant_response = await qdrant_module.search(
                        query=query,
                        collection_name=collection_name,
                        top_k=top_k * 2,
                        ctx=ctx,
                    )

                    # Convert to SearchResult objects
                    for hit in qdrant_response.get("results", []):
                        dense_results.append(
                            SearchResult(
                                doc_id=hit.get("id", hit.get("doc_id", "")),
                                content=hit.get("content", ""),
                                score=hit.get("score", 0.0),
                                metadata=hit.get("metadata", {}),
                                source="dense",
                            )
                        )
                else:
                    logger.warning("rag_qdrant module not available for dense search")

            # Determine result type and get final results
            if include_sparse_only:
                results = sparse_results[:top_k]
                search_type = "sparse"
                self._total_sparse_searches += 1
            elif include_dense_only:
                results = dense_results[:top_k]
                search_type = "dense"
            else:
                # Perform hybrid fusion
                results = self.provider.hybrid_search(
                    query=query,
                    dense_results=dense_results,
                    sparse_results=sparse_results,
                    top_k=top_k,
                    fusion_method=fusion,
                    dense_weight=dense_weight,
                )
                search_type = "hybrid"
                self._total_hybrid_searches += 1

            self._total_searches += 1

            return {
                "query": query,
                "collection": collection_name,
                "results": [r.to_dict() for r in results],
                "count": len(results),
                "search_type": search_type,
                "fusion_method": (fusion or self.provider.fusion_method).value,
                "dense_results_count": len(dense_results),
                "sparse_results_count": len(sparse_results),
                "request_id": request_id,
            }

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            return {"error": str(e), "request_id": request_id}

    async def sparse_search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform BM25-only sparse search.

        Args:
            query: Search query
            collection_name: Target collection
            top_k: Number of results
            request_id: Optional request identifier
            ctx: Security context

        Returns:
            Dictionary with BM25 search results
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        user_id = self._get_user_id_from_ctx(ctx)
        if not user_id:
            return {"error": "User not authenticated", "request_id": request_id}

        if not query or not query.strip():
            return {"error": "Query is required", "request_id": request_id}

        try:
            results = self.provider.sparse_search(
                collection_name=collection_name,
                query=query,
                top_k=top_k,
            )

            self._total_searches += 1
            self._total_sparse_searches += 1

            return {
                "query": query,
                "collection": collection_name,
                "results": [r.to_dict() for r in results],
                "count": len(results),
                "search_type": "sparse",
                "request_id": request_id,
            }

        except Exception as e:
            logger.error(f"Sparse search failed: {e}", exc_info=True)
            return {"error": str(e), "request_id": request_id}

    async def index_document(
        self,
        collection_name: str,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Index a document for sparse search.

        This is typically called automatically via event subscription,
        but can be called directly for manual indexing.

        Args:
            collection_name: Target collection
            doc_id: Document identifier
            content: Document text content
            metadata: Optional metadata
            request_id: Optional request identifier
            ctx: Security context

        Returns:
            Indexing status
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        if not collection_name or not doc_id or not content:
            return {
                "error": "collection_name, doc_id, and content are required",
                "request_id": request_id,
            }

        try:
            self.provider.index_document(
                collection_name=collection_name,
                doc_id=doc_id,
                content=content,
                metadata=metadata,
            )

            return {
                "status": "indexed",
                "collection": collection_name,
                "doc_id": doc_id,
                "request_id": request_id,
            }

        except Exception as e:
            logger.error(f"Index document failed: {e}", exc_info=True)
            return {"error": str(e), "request_id": request_id}

    async def remove_document(
        self,
        collection_name: str,
        doc_id: str,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Remove a document from the sparse index.

        Args:
            collection_name: Target collection
            doc_id: Document identifier
            request_id: Optional request identifier
            ctx: Security context

        Returns:
            Removal status
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        try:
            removed = self.provider.remove_from_index(collection_name, doc_id)

            return {
                "status": "removed" if removed else "not_found",
                "collection": collection_name,
                "doc_id": doc_id,
                "request_id": request_id,
            }

        except Exception as e:
            logger.error(f"Remove document failed: {e}", exc_info=True)
            return {"error": str(e), "request_id": request_id}

    async def get_index_stats(
        self,
        collection_name: str,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get statistics for a collection's BM25 index.

        Args:
            collection_name: Target collection
            request_id: Optional request identifier
            ctx: Security context

        Returns:
            Index statistics or not found status
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        stats = self.provider.get_index_stats(collection_name)

        return {
            "collection": collection_name,
            "stats": stats,
            "indexed": stats is not None,
            "request_id": request_id,
        }

    async def list_indexed_collections(
        self,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        List all collections with BM25 indexes.

        Args:
            request_id: Optional request identifier
            ctx: Security context

        Returns:
            List of indexed collection names
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        collections = self.provider.list_indexed_collections()

        return {
            "collections": collections,
            "count": len(collections),
            "request_id": request_id,
        }

    async def rebuild_index(
        self,
        collection_name: str,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Rebuild BM25 index from rag_qdrant collection.

        Admin-only operation. Fetches all documents from rag_qdrant
        and rebuilds the sparse index.

        Args:
            collection_name: Target collection
            request_id: Optional request identifier
            ctx: Security context

        Returns:
            Rebuild status with document count
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        if not self._is_admin(ctx):
            return {"error": "Admin access required", "request_id": request_id}

        try:
            qdrant_module = self._get_module("rag_qdrant")
            if not qdrant_module:
                return {
                    "error": "rag_qdrant module not available",
                    "request_id": request_id,
                }

            # Get all documents from collection
            all_docs = await qdrant_module.list_documents(
                collection_name=collection_name,
                limit=self.config.get("max_indexed_docs_per_collection", 100000),
                ctx=ctx,
            )

            if "error" in all_docs:
                return {
                    "error": f"Failed to fetch documents: {all_docs['error']}",
                    "request_id": request_id,
                }

            # Clear existing index
            self.provider.clear_index(collection_name)

            # Reindex all documents
            indexed_count = 0
            for doc in all_docs.get("documents", []):
                doc_id = doc.get("id", doc.get("doc_id", ""))
                content = doc.get("content", "")

                if doc_id and content:
                    self.provider.index_document(
                        collection_name=collection_name,
                        doc_id=doc_id,
                        content=content,
                        metadata=doc.get("metadata", {}),
                    )
                    indexed_count += 1

            logger.info(
                f"Rebuilt BM25 index for {collection_name}: {indexed_count} documents"
            )

            return {
                "status": "rebuilt",
                "collection": collection_name,
                "indexed_documents": indexed_count,
                "request_id": request_id,
            }

        except Exception as e:
            logger.error(f"Index rebuild failed: {e}", exc_info=True)
            return {"error": str(e), "request_id": request_id}

    async def clear_index(
        self,
        collection_name: str,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Clear the BM25 index for a collection (admin only).

        Args:
            collection_name: Target collection
            request_id: Optional request identifier
            ctx: Security context

        Returns:
            Clear status
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        if not self._is_admin(ctx):
            return {"error": "Admin access required", "request_id": request_id}

        try:
            cleared = self.provider.clear_index(collection_name)

            return {
                "status": "cleared" if cleared else "not_found",
                "collection": collection_name,
                "request_id": request_id,
            }

        except Exception as e:
            logger.error(f"Clear index failed: {e}", exc_info=True)
            return {"error": str(e), "request_id": request_id}

    # === EVENT HANDLERS ===

    async def _on_document_indexed(self, event: Event) -> None:
        """
        Handle document indexed event from rag_qdrant.

        Automatically indexes the document in BM25 for sparse search.
        """
        if not self.provider:
            return

        try:
            payload = event.payload
            collection = payload.get("collection_name")
            doc_id = payload.get("doc_id")
            content = payload.get("content", "")
            metadata = payload.get("metadata", {})

            if collection and doc_id and content:
                self.provider.index_document(
                    collection_name=collection,
                    doc_id=doc_id,
                    content=content,
                    metadata=metadata,
                )
                logger.debug(
                    f"Auto-indexed document {doc_id} in collection {collection}"
                )

        except Exception as e:
            logger.error(f"Error handling document_indexed event: {e}")

    async def _on_document_deleted(self, event: Event) -> None:
        """
        Handle document deleted event from rag_qdrant.

        Automatically removes the document from BM25 index.
        """
        if not self.provider:
            return

        try:
            payload = event.payload
            collection = payload.get("collection_name")
            doc_id = payload.get("doc_id")

            if collection and doc_id:
                self.provider.remove_from_index(collection, doc_id)
                logger.debug(
                    f"Auto-removed document {doc_id} from collection {collection}"
                )

        except Exception as e:
            logger.error(f"Error handling document_deleted event: {e}")

    # === HELPER METHODS ===

    def _get_user_id_from_ctx(self, ctx) -> Optional[str]:
        """Extract user_id from security context."""
        if ctx and hasattr(ctx, "user") and ctx.user:
            return getattr(ctx.user, "user_id", None)
        return None

    def _is_admin(self, ctx) -> bool:
        """Check if user is admin."""
        if ctx and hasattr(ctx, "user") and ctx.user:
            return getattr(ctx.user, "is_admin", False)
        return False

    def _get_module(self, module_name: str):
        """Get another module from the system."""
        if hasattr(self, "module_manager") and self.module_manager:
            return self.module_manager.get_module(module_name)
        return None
