"""
RAG Simple Memory UBP Framework Bridge Layer

Integrates technical providers with UBP module system.

MCP-COMPAT (ARCH-008): Added OperationContext support for dual REST/MCP compatibility.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
import uuid
import asyncio

from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule
# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    from _shared.operation_context import OperationContext

from ubp_enterprise_hybrid.backend.app.infra.event_bus import Event
from .providers import SimpleVectorStore

logger = logging.getLogger(__name__)

# v6.4.0: Import unified chunker from rag_qdrant
try:
    from rag_qdrant.chunker import ChunkingManager as _MainChunkingManager
    from rag_qdrant.chunker import ChunkingConfig as _MainChunkingConfig
    _HAS_MAIN_CHUNKER = True
except ImportError:
    _HAS_MAIN_CHUNKER = False


class RAGSimpleMemoryAdapter(BaseHybridModule):
    """UBP adapter for RAG simple memory module with lazy initialization."""

    def __init__(self, module_path: Path, **kwargs):
        """Initialize the RAG module."""
        super().__init__(module_path, **kwargs)

        self.vector_store: Optional[SimpleVectorStore] = None
        self._initialization_lock = asyncio.Lock()
        self._is_initialized = False

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

    async def _ensure_initialized(self) -> None:
        """
        Ensure vector store is initialized (lazy initialization).

        This method is idempotent and thread-safe. It will only initialize once,
        even if called multiple times concurrently.
        """
        if self._is_initialized:
            return

        async with self._initialization_lock:
            # Double-check pattern to avoid race conditions
            if self._is_initialized:
                return

            logger.info(
                f"Auto-initializing {self.manifest.name} module",
                extra={"mod_name": self.manifest.name},
            )

            self.vector_store = SimpleVectorStore(self.config)
            self._is_initialized = True

            logger.info(
                f"✅ {self.manifest.name} initialized successfully",
                extra={
                    "max_documents": self.vector_store.max_documents,
                    "max_document_size": self.vector_store.max_document_size,
                    "max_memory_mb": self.vector_store.max_memory_mb,
                },
            )

    async def initialize(self) -> Dict[str, Any]:
        """
        Explicitly initialize the vector store.

        This method can be called explicitly via API or will be called automatically
        on first use (lazy initialization).

        Returns:
            Initialization status and configuration
        """
        await self._ensure_initialized()

        return {
            "status": "initialized",
            "module": self.manifest.name,
            "config": {
                "max_documents": self.vector_store.max_documents,
                "max_document_size": self.vector_store.max_document_size,
                "max_memory_mb": self.vector_store.max_memory_mb,
                "chunking_enabled": self.config["chunking"]["enabled"],
                "default_top_k": self.config["retrieval"]["default_top_k"],
            },
        }

    async def shutdown(self) -> None:
        """Shutdown and cleanup."""
        logger.info(f"Shutting down {self.manifest.name} module")

        if self.vector_store:
            stats = self.vector_store.get_stats()
            self.vector_store.clear()
            logger.debug("Vector store cleared", extra={"final_stats": stats})

        logger.info(f"✅ {self.manifest.name} shutdown successfully")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check with auto-initialization."""
        await self._ensure_initialized()

        stats = self.vector_store.get_stats()

        status = "healthy"
        # Check if approaching limits
        doc_usage = stats["total_documents"] / self.vector_store.max_documents
        if doc_usage > 0.9:
            status = "degraded"
            logger.warning(
                "Document storage approaching capacity",
                extra={
                    "usage_percent": doc_usage * 100,
                    "total_documents": stats["total_documents"],
                    "max_documents": self.vector_store.max_documents,
                },
            )

        return {
            "module": self.manifest.name,
            "status": status,
            "store": {
                "initialized": True,
                "stats": stats,
                "limits": {
                    "max_documents": self.vector_store.max_documents,
                    "max_document_size": self.vector_store.max_document_size,
                    "max_memory_mb": self.vector_store.max_memory_mb,
                },
            },
        }

    async def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a document to the RAG system with request tracking and auto-initialization.

        Args:
            doc_id: Unique document identifier
            text: Document text
            metadata: Optional metadata
            request_id: Optional tracking ID (auto-generated if not provided)

        Returns:
            Result with document info and request_id

        Raises:
            ValueError: If inputs are invalid
            MemoryError: If memory limits exceeded
        """
        # Ensure initialized (lazy initialization)
        await self._ensure_initialized()

        # Generate request ID for tracking
        if not request_id:
            request_id = str(uuid.uuid4())

        # Input validation
        if not text or not text.strip():
            raise ValueError("Document text cannot be empty")

        logger.info(
            "Adding document",
            extra={
                "request_id": request_id,
                "doc_id": doc_id,
                "text_length": len(text),
                "chunking_enabled": self.config["chunking"]["enabled"],
            },
        )

        try:
            # Optionally chunk the document
            if self.config["chunking"]["enabled"]:
                chunks = self._chunk_text(text)

                # Add each chunk as a separate document
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    chunk_metadata = {
                        **(metadata or {}),
                        "parent_doc_id": doc_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                    self.vector_store.add_document(chunk_id, chunk, chunk_metadata)

                # Publish event
                if self.publisher:
                    await self.publisher.publish(
                        "document.indexed",
                        {
                            "request_id": request_id,
                            "doc_id": doc_id,
                            "chunks": len(chunks),
                            "metadata": metadata,
                        },
                    )

                logger.info(
                    "Document added with chunks",
                    extra={
                        "request_id": request_id,
                        "doc_id": doc_id,
                        "num_chunks": len(chunks),
                    },
                )

                return {
                    "doc_id": doc_id,
                    "chunks": len(chunks),
                    "status": "indexed",
                    "request_id": request_id,
                }
            else:
                # Add whole document
                self.vector_store.add_document(doc_id, text, metadata)

                if self.publisher:
                    await self.publisher.publish(
                        "document.indexed",
                        {
                            "request_id": request_id,
                            "doc_id": doc_id,
                            "chunks": 1,
                            "metadata": metadata,
                        },
                    )

                logger.info(
                    "Document added successfully",
                    extra={"request_id": request_id, "doc_id": doc_id},
                )

                return {
                    "doc_id": doc_id,
                    "status": "indexed",
                    "request_id": request_id,
                }

        except (ValueError, MemoryError) as e:
            logger.error(
                "Failed to add document",
                extra={
                    "request_id": request_id,
                    "doc_id": doc_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into smaller pieces."""
        chunk_size = self.config["chunking"]["chunk_size"]
        chunk_overlap = self.config["chunking"]["chunk_overlap"]

        # v6.4.0: Delegate to unified chunker from rag_qdrant
        if _HAS_MAIN_CHUNKER:
            config = _MainChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=min(chunk_overlap, chunk_size - 1),
                strategy="sentence",
            )
            manager = _MainChunkingManager(config)
            return [c.text for c in manager.chunk(text)]

        # Legacy fallback with safety guard
        if chunk_overlap >= chunk_size:
            chunk_overlap = chunk_size // 4
            logger.warning(f"[MEMORY] chunk_overlap >= chunk_size, reduced to {chunk_overlap}")
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - chunk_overlap
        return chunks

    async def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG system with request tracking and auto-initialization.

        Args:
            query_text: Query text
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            request_id: Optional tracking ID (auto-generated if not provided)

        Returns:
            Query results with relevant documents and request_id

        Raises:
            ValueError: If query_text is empty or top_k is invalid
        """
        # Ensure initialized (lazy initialization)
        await self._ensure_initialized()

        # Generate request ID for tracking
        if not request_id:
            request_id = str(uuid.uuid4())

        # Input validation
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

        if top_k is None:
            top_k = self.config["retrieval"]["default_top_k"]

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if threshold is None:
            threshold = self.config["retrieval"]["min_similarity_threshold"]

        logger.info(
            "Processing query",
            extra={
                "request_id": request_id,
                "query_length": len(query_text),
                "top_k": top_k,
            },
        )

        results = self.vector_store.query(query_text, top_k, threshold)

        logger.info(
            "Query completed",
            extra={"request_id": request_id, "results_count": len(results)},
        )

        return {
            "query": query_text,
            "results": results,
            "count": len(results),
            "request_id": request_id,
        }

    async def delete_document(
        self, doc_id: str, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delete a document with request tracking and auto-initialization."""
        # Ensure initialized (lazy initialization)
        await self._ensure_initialized()

        # Generate request ID for tracking
        if not request_id:
            request_id = str(uuid.uuid4())

        logger.info(
            "Deleting document", extra={"request_id": request_id, "doc_id": doc_id}
        )

        success = self.vector_store.delete_document(doc_id)

        return {
            "doc_id": doc_id,
            "deleted": success,
            "status": "deleted" if success else "not_found",
            "request_id": request_id,
        }

    async def clear(self) -> Dict[str, Any]:
        """Clear all documents with auto-initialization."""
        # Ensure initialized (lazy initialization)
        await self._ensure_initialized()

        self.vector_store.clear()

        return {"status": "cleared"}

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics with auto-initialization."""
        # Ensure initialized (lazy initialization)
        await self._ensure_initialized()

        return self.vector_store.get_stats()

    # Event handlers
    async def on_document_added(self, event: Event):
        """Handle document.added event."""
        payload = event.payload
        await self.add_document(
            doc_id=payload["doc_id"],
            text=payload["text"],
            metadata=payload.get("metadata"),
            request_id=payload.get("request_id"),
        )

    async def on_rag_query(self, event: Event):
        """Handle rag.query event."""
        payload = event.payload
        result = await self.query(
            query_text=payload["query_text"],
            top_k=payload.get("top_k"),
            threshold=payload.get("threshold"),
            request_id=payload.get("request_id"),
        )

        # Publish completion event
        if self.publisher:
            await self.publisher.publish(
                "rag.query.completed",
                {"request_id": payload.get("request_id"), "result": result},
            )
