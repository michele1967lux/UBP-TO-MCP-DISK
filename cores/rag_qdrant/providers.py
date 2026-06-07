"""rag_qdrant.providers

Pure technical implementation (NO UBP framework dependencies).

This file hosts the *provider layer* for the rag_qdrant core module.
It intentionally contains all the operational/business logic for the module,
while the UBP adapter (adapter.py) only acts as a thin bridge.

Enterprise goals:
- Zero dependency on UBP backend/infra packages
- Explicit lifecycle (initialize/shutdown/health_check)
- Graceful degradation and robust error handling
- Testable in isolation

NOTE:
This provider is derived from the original implementation previously hosted
in rag_qdrant/__init__.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging
import json
import time

# Import technical components (pure python, local to module)
from .client import (
    QdrantClient,
    create_qdrant_client,
    CircuitBreakerConfig,
    RetryConfig,
    CircuitBreakerOpenError,
)
from .embeddings import (
    EmbeddingManager,
    EmbeddingConfig,
    create_embedding_manager,
)
from .chunker import (
    ChunkingManager,
    ChunkingConfig,
    Chunk,
    create_chunking_manager,
)
from .operations import (
    OperationHandler,
    OperationResult,
    OperationStatus,
    QueryResult,
    SearchResult,
    ValidationError,
)
from .events import (
    EventManager,
    EventPublisher,
    Event,
    EventMetadata,
    ProcessingResult,
    create_event_manager,
    create_event_publisher,
)

# FIX-PORT-001 v1.8.2: Import ManifestLoader for proper ENV expansion and type coercion
# Use absolute import to avoid "attempted relative import beyond top-level package"
from ubp_enterprise_hybrid.modules.cores._shared.manifest_loader import ManifestLoader

import os
import httpx
import hashlib
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Redis key prefix for document registry (per NAMING_POLICY.md)
REDIS_DOC_REGISTRY_PREFIX = "ubp:rag:documents"


# ============================================================================
# Embedding Model Catalogs (Grand Unified Registry)
# ============================================================================

# Local models (sentence-transformers/HuggingFace) - Always available (downloaded on demand)
LOCAL_MODELS = [
    # ── Sentence-Transformers (Classic) ──────────────────────────────────────
    {
        "id": "all-MiniLM-L6-v2",
        "name": "MiniLM L6 v2 (Fast)",
        "dims": 384,
        "provider": "sentence-transformers",
        "description": "Fast, lightweight model for general use",
        "available": True,
    },
    {
        "id": "all-mpnet-base-v2",
        "name": "MPNet Base v2 (Balanced)",
        "dims": 768,
        "provider": "sentence-transformers",
        "description": "Good balance of speed and quality",
        "available": True,
    },
    {
        "id": "paraphrase-MiniLM-L6-v2",
        "name": "Paraphrase MiniLM",
        "dims": 384,
        "provider": "sentence-transformers",
        "description": "Optimized for paraphrase detection",
        "available": True,
    },
    {
        "id": "multi-qa-MiniLM-L6-cos-v1",
        "name": "Multi-QA MiniLM",
        "dims": 384,
        "provider": "sentence-transformers",
        "description": "Optimized for question-answering",
        "available": True,
    },
    {
        "id": "all-distilroberta-v1",
        "name": "DistilRoBERTa v1",
        "dims": 768,
        "provider": "sentence-transformers",
        "description": "General purpose, good quality",
        "available": True,
    },
    # ── BGE Models (BAAI) ────────────────────────────────────────────────────
    {
        "id": "BAAI/bge-small-en-v1.5",
        "name": "BGE Small EN v1.5",
        "dims": 384,
        "provider": "sentence-transformers",
        "description": "Retrieval optimized, fast",
        "available": True,
    },
    {
        "id": "BAAI/bge-base-en-v1.5",
        "name": "BGE Base EN v1.5",
        "dims": 768,
        "provider": "sentence-transformers",
        "description": "Retrieval optimized, balanced",
        "available": True,
    },
    {
        "id": "BAAI/bge-large-en-v1.5",
        "name": "BGE Large EN v1.5",
        "dims": 1024,
        "provider": "sentence-transformers",
        "description": "Retrieval optimized, high quality",
        "available": True,
    },
    {
        "id": "BAAI/bge-m3",
        "name": "BGE M3 (Multilingual)",
        "dims": 1024,
        "provider": "sentence-transformers",
        "description": "Multi-lingual, multi-granularity",
        "available": True,
    },
    # ── E5 Models (intfloat) ─────────────────────────────────────────────────
    {
        "id": "intfloat/e5-small-v2",
        "name": "E5 Small v2",
        "dims": 384,
        "provider": "sentence-transformers",
        "description": "Multi-task, efficient",
        "available": True,
    },
    {
        "id": "intfloat/e5-base-v2",
        "name": "E5 Base v2",
        "dims": 768,
        "provider": "sentence-transformers",
        "description": "Multi-task, balanced",
        "available": True,
    },
    {
        "id": "intfloat/e5-large-v2",
        "name": "E5 Large v2",
        "dims": 1024,
        "provider": "sentence-transformers",
        "description": "Multi-task, high quality",
        "available": True,
    },
    {
        "id": "intfloat/multilingual-e5-small",
        "name": "E5 Multilingual Small",
        "dims": 384,
        "provider": "sentence-transformers",
        "description": "100+ languages, fast & lightweight",
        "available": True,
    },
    {
        "id": "intfloat/multilingual-e5-base",
        "name": "E5 Multilingual Base",
        "dims": 768,
        "provider": "sentence-transformers",
        "description": "100+ languages support",
        "available": True,
    },
    {
        "id": "intfloat/multilingual-e5-large",
        "name": "E5 Multilingual Large",
        "dims": 1024,
        "provider": "sentence-transformers",
        "description": "100+ languages, high quality",
        "available": True,
    },
    # ── GTE Models (Alibaba) ─────────────────────────────────────────────────
    {
        "id": "thenlper/gte-small",
        "name": "GTE Small",
        "dims": 384,
        "provider": "sentence-transformers",
        "description": "General text embeddings, fast",
        "available": True,
    },
    {
        "id": "thenlper/gte-base",
        "name": "GTE Base",
        "dims": 768,
        "provider": "sentence-transformers",
        "description": "General text embeddings, balanced",
        "available": True,
    },
    {
        "id": "thenlper/gte-large",
        "name": "GTE Large",
        "dims": 1024,
        "provider": "sentence-transformers",
        "description": "General text embeddings, high quality",
        "available": True,
    },
    # ── Instructor Models ────────────────────────────────────────────────────
    {
        "id": "hkunlp/instructor-base",
        "name": "Instructor Base",
        "dims": 768,
        "provider": "sentence-transformers",
        "description": "Instruction-tuned embeddings",
        "available": True,
    },
    {
        "id": "hkunlp/instructor-large",
        "name": "Instructor Large",
        "dims": 1024,
        "provider": "sentence-transformers",
        "description": "Instruction-tuned, high quality",
        "available": True,
    },
]

# OpenAI models - Available if API key is configured
OPENAI_MODELS = [
    {
        "id": "text-embedding-3-small",
        "name": "OpenAI Small (3rd Gen)",
        "dims": 1536,
        "provider": "openai",
        "description": "Cost-effective, good quality",
    },
    {
        "id": "text-embedding-3-large",
        "name": "OpenAI Large (3rd Gen)",
        "dims": 3072,
        "provider": "openai",
        "description": "Highest quality OpenAI embedding",
    },
    {
        "id": "text-embedding-ada-002",
        "name": "OpenAI Ada-002 (Legacy)",
        "dims": 1536,
        "provider": "openai",
        "description": "Legacy model, widely compatible",
    },
]

# Cohere models - Available if API key is configured
COHERE_MODELS = [
    {
        "id": "embed-english-v3.0",
        "name": "Cohere English v3",
        "dims": 1024,
        "provider": "cohere",
        "description": "Best for English text",
    },
    {
        "id": "embed-multilingual-v3.0",
        "name": "Cohere Multilingual v3",
        "dims": 1024,
        "provider": "cohere",
        "description": "Supports 100+ languages",
    },
    {
        "id": "embed-english-light-v3.0",
        "name": "Cohere English Light",
        "dims": 384,
        "provider": "cohere",
        "description": "Faster, smaller English model",
    },
]

# Ollama embedding model detection patterns
# Models matching these patterns are considered embedding models
OLLAMA_EMBEDDING_PATTERNS = [
    "embed",      # nomic-embed-text, qwen3-embedding, jina-embeddings, etc.
    "bge-m3",     # BGE M3 multilingual
    "minilm",     # all-minilm
    "snowflake-arctic-embed",
    "mxbai-embed",
]


@dataclass(frozen=True)
class ModuleManifest:
    """Minimal manifest representation for the provider layer."""

    name: str
    version: str
    description: str
    author: str
    module_type: str
    requires_redis: bool
    requires_postgres: bool
    requires_event_bus: bool
    operations: List[Dict[str, Any]]
    event_subscriptions: List[str]
    event_publications: List[str]
    config_file: str

    @classmethod
    def from_file(cls, manifest_path: Path) -> "ModuleManifest":
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            name=data.get("name", "rag_qdrant"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            module_type=data.get("module_type", "rag"),
            requires_redis=data.get("requires_redis", False),
            requires_postgres=data.get("requires_postgres", False),
            requires_event_bus=data.get("requires_event_bus", True),
            operations=data.get("operations", []) or [],
            event_subscriptions=data.get("event_subscriptions", []) or [],
            event_publications=data.get("event_publications", []) or [],
            config_file=data.get("config_file", "config.json"),
        )


class RAGQdrant:
    """Qdrant-based RAG provider (pure implementation).

    This class is intentionally framework-agnostic. It can be used:
    - directly in unit/operational tests
    - behind the UBP adapter (modules/cores/rag_qdrant/adapter.py)

     It implements operations exposed in manifest.json:
     - initialize
     - add_document
     - query
     - create_collection
     - delete_collection

     Backward-compat note:
     The previous implementation exposed extra helper APIs used by tests
     (delete_document, list_collections, clear, get_stats). We keep them here
     even if they are not present in the manifest.


    Plus additional helpful operations:
    - health_check
    - shutdown
    - get_stats
    - add_documents_batch
    - delete_document
    - list_collections
    - query_with_context

    The adapter decides what is published to the UBP API surface.
    """

    def __init__(self, module_path: Path, **kwargs: Any):
        self.module_path = Path(module_path)
        self.kwargs = kwargs

        manifest_path = self.module_path / "manifest.json"
        self.manifest = (
            ModuleManifest.from_file(manifest_path)
            if manifest_path.exists()
            else ModuleManifest(
                name="rag_qdrant",
                version="1.0.0",
                description="",
                author="",
                module_type="rag",
                requires_redis=False,
                requires_postgres=False,
                requires_event_bus=False,
                operations=[],
                event_subscriptions=[],
                event_publications=[],
                config_file="config.json",
            )
        )

        # FIX-PORT-001 v1.8.2: Use ManifestLoader for proper ENV expansion and type coercion
        # This ensures port values like "${UBP_RAG_QDRANT__PORT:-6333}" are expanded
        # and string "6333" is coerced to int 6333 (required by qdrant-client)
        config_file = getattr(self.manifest, "config_file", "config.json")
        try:
            self.config = ManifestLoader.load_config(
                module_path=self.module_path,
                config_file=config_file,
                expand_env=True,  # Enable ENV expansion and type coercion
            )
        except FileNotFoundError:
            self.config = self._default_config()

        # Event bus integration (framework-agnostic protocol)
        self.event_bus = kwargs.get("event_bus")
        self.publisher = kwargs.get("publisher")

        # Core components
        self.qdrant_client: Optional[QdrantClient] = None
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.chunking_manager: Optional[ChunkingManager] = None
        self.operation_handler: Optional[OperationHandler] = None
        self.event_manager: Optional[EventManager] = None
        self.event_publisher: Optional[EventPublisher] = None

        # Reliability settings (kept as attributes for backward-compat tests)
        reliability = self.config.get("reliability", {})
        self.max_retries = int(reliability.get("max_retries", 3))
        self.retry_delay = float(reliability.get("retry_delay_seconds", 1))
        self.retry_backoff = float(reliability.get("retry_backoff_multiplier", 2))
        self.max_retry_delay = float(reliability.get("max_retry_delay_seconds", 30))
        self.circuit_breaker_threshold = int(
            reliability.get("circuit_breaker_threshold", 5)
        )
        self.circuit_breaker_timeout = float(
            reliability.get("circuit_breaker_timeout", 60)
        )

        # State
        self._initialized: bool = False
        self._start_time: Optional[float] = None

    # ---------------------------------------------------------------------
    # Defaults
    # ---------------------------------------------------------------------

    def _default_config(self) -> Dict[str, Any]:
        return {
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "grpc_port": 6334,
                "prefer_grpc": False,
                "timeout": 30,
                "api_key": None,
            },
            "collection": {
                "default_name": "documents",
                "vector_size": 384,
                "distance": "Cosine",
                "on_disk_payload": True,
            },
            "embedding": {
                "provider": "sentence-transformers",
                "model": "all-MiniLM-L6-v2",
                "dimension": 384,
                "batch_size": 32,
            },
            "chunking": {
                "enabled": True,
                "chunk_size": 500,
                "chunk_overlap": 50,
                "split_by": "sentence",
            },
            "retrieval": {
                "default_top_k": 5,
                # FIX-BUG-004 v1.8.3: Lower default to 0.1 (was 0.7)
                # 0.7 was too aggressive for most embedding models
                "score_threshold": 0.1,
                "with_payload": True,
                "with_vectors": True,  # v6.1.3: Enable for embedding-based dedup/fusion
            },
            "reliability": {
                "max_retries": 3,
                "retry_delay_seconds": 1,
                "retry_backoff_multiplier": 2,
                "max_retry_delay_seconds": 30,
                "circuit_breaker_threshold": 5,
                "circuit_breaker_timeout": 60,
            },
        }

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the provider and all technical dependencies."""
        if self._initialized:
            logger.warning("Module already initialized")
            return {
                "status": "already_initialized",
                "module": self.manifest.name,
                "message": "Module was already initialized",
            }

        self._start_time = time.time()

        logger.info(
            "Initializing %s v%s",
            self.manifest.name,
            self.manifest.version,
            extra={"mod_name": self.manifest.name},
        )

        # Redis client is resolved by the adapter via Pure DI and passed in kwargs.
        # The provider does NOT resolve Redis itself - it uses what was injected.
        redis_client = self.kwargs.get("redis_client")

        try:
            self.qdrant_client = create_qdrant_client(self.config)
            await self.qdrant_client.connect()

            self.embedding_manager = create_embedding_manager(self.config, redis_client=redis_client)
            await self.embedding_manager.initialize()

            self.chunking_manager = create_chunking_manager(self.config)

            self.operation_handler = OperationHandler(
                client=self.qdrant_client,
                embedding_manager=self.embedding_manager,
                chunking_manager=self.chunking_manager,
                config=self.config,
                redis_client=redis_client,
            )

            # Publisher/event_manager are optional (depends on event_bus availability)
            self.event_publisher = create_event_publisher(
                event_bus=self.event_bus,
                source=self.manifest.name,
            )

            self.event_manager = create_event_manager(
                config=self.config,
                operation_handler=self.operation_handler,
                publisher=self.event_publisher,
            )

            # Ensure default collection exists
            default_collection = self.config["collection"]["default_name"]
            if not await self.qdrant_client.collection_exists(default_collection):
                await self.qdrant_client.create_collection(
                    collection_name=default_collection,
                    vector_size=self.embedding_manager.dimension,
                    distance=self.config["collection"]["distance"],
                )

            self._initialized = True
            init_time_ms = (time.time() - self._start_time) * 1000

            logger.info(
                "✅ %s initialized successfully",
                self.manifest.name,
                extra={
                    "init_time_ms": round(init_time_ms, 2),
                    "embedding_model": self.config.get("embedding", {}).get("model"),
                    "dimension": getattr(self.embedding_manager, "dimension", None),
                },
            )

            return {
                "status": "initialized",
                "module": self.manifest.name,
                "config": {
                    "qdrant_host": self.config.get("qdrant", {}).get("host"),
                    "qdrant_port": self.config.get("qdrant", {}).get("port"),
                    "embedding_model": self.config.get("embedding", {}).get("model"),
                    "dimension": getattr(self.embedding_manager, "dimension", None),
                    "default_collection": default_collection,
                },
                "initialization_time_ms": round(init_time_ms, 2),
            }

        except Exception as e:
            logger.error(
                "Failed to initialize %s: %s",
                self.manifest.name,
                e,
                extra={"error_type": type(e).__name__},
                exc_info=True,
            )
            raise

    async def shutdown(self) -> None:
        """Shutdown and cleanup."""
        logger.info("Shutting down %s", self.manifest.name)
        try:
            if self.embedding_manager:
                await self.embedding_manager.shutdown()
            if self.qdrant_client:
                await self.qdrant_client.disconnect()
        finally:
            self._initialized = False
            logger.info("✅ %s shutdown completed", self.manifest.name)

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        status = "healthy"
        components: Dict[str, Any] = {}

        if self.qdrant_client:
            qdrant_health = await self.qdrant_client.health_check()
            components["qdrant"] = qdrant_health
            if qdrant_health.get("status") != "healthy":
                status = "degraded"
        else:
            status = "unhealthy"
            components["qdrant"] = {"status": "not_initialized"}

        if self.embedding_manager:
            components["embedding"] = {
                "status": "healthy",
                "model": self.config.get("embedding", {}).get("model"),
                "dimension": getattr(self.embedding_manager, "dimension", None),
                "metrics": getattr(self.embedding_manager, "metrics", {}),
            }
        else:
            components["embedding"] = {"status": "not_initialized"}

        uptime_seconds = (time.time() - self._start_time) if self._start_time else None

        return {
            "module": self.manifest.name,
            "version": self.manifest.version,
            "status": status,
            "initialized": self._initialized,
            "uptime_seconds": round(uptime_seconds, 2) if uptime_seconds else None,
            "components": components,
        }

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                f"{self.manifest.name} not initialized. Call initialize() first."
            )

    @property
    def embedding_model(self):
        """Backward compatible alias."""
        return self.embedding_manager

    def _get_redis_client(self):
        """Get Redis client injected via DI (from kwargs).

        The adapter resolves Redis via the DI container and passes it here.
        This method does NOT fall back to EventBus internals.
        """
        return self.kwargs.get("redis_client")

    async def _save_document_to_registry(
        self,
        collection: str,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any],
        chunks_count: int,
    ) -> bool:
        """Save document metadata to Redis registry for fast listing.

        TASK #82: Document Registry for lifecycle management.
        Key format: ubp:rag:documents:{collection_name} -> HSET {doc_id} {json}
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            logger.debug("Redis client not available - document registry skipped")
            return False

        try:
            # Calculate content hash for deduplication
            content_hash = hashlib.md5(text.encode()).hexdigest()

            # Build registry entry
            registry_entry = {
                "doc_id": doc_id,
                "filename": metadata.get("filename") or metadata.get("title") or doc_id,
                "content_hash": content_hash,
                "chunk_count": chunks_count,
                "chunk_strategy": metadata.get("chunk_strategy", "default"),
                "uploader_id": metadata.get("uploader_id", "system"),
                "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                "file_size": metadata.get("file_size") or len(text.encode()),
                "mime_type": metadata.get("mime_type", "text/plain"),
                "status": "indexed",
                "title": metadata.get("title"),
                "source": metadata.get("source"),
            }

            # Save to Redis Hash
            redis_key = f"{REDIS_DOC_REGISTRY_PREFIX}:{collection}"
            await redis_client.hset(redis_key, doc_id, json.dumps(registry_entry))

            logger.info(
                "Document saved to registry: %s in %s",
                doc_id,
                collection,
                extra={"content_hash": content_hash, "chunks": chunks_count},
            )
            return True

        except Exception as e:
            logger.warning("Failed to save document to registry: %s", e)
            return False

    async def check_duplicate(
        self,
        collection: str,
        content_hash: str,
        filename: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Check if a document with the same content_hash exists in the registry.

        Returns the matching registry entry if duplicate found, None otherwise.
        Graceful degradation: returns None if Redis is unavailable.
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            return None

        try:
            redis_key = f"{REDIS_DOC_REGISTRY_PREFIX}:{collection}"
            registry_data = await redis_client.hgetall(redis_key)

            if not registry_data:
                return None

            for doc_id, json_str in registry_data.items():
                if isinstance(json_str, bytes):
                    json_str = json_str.decode()
                entry = json.loads(json_str)
                if entry.get("content_hash") == content_hash:
                    return entry

        except Exception as e:
            logger.warning("Dedup check failed (non-blocking): %s", e)

        return None

    async def load_dedup_index(self, collection: str) -> Set[str]:
        """Load all content_hashes for a collection. For batch dedup optimization.

        Returns a set of content_hash strings. Empty set if Redis unavailable.
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            return set()

        try:
            redis_key = f"{REDIS_DOC_REGISTRY_PREFIX}:{collection}"
            registry_data = await redis_client.hgetall(redis_key)

            hashes: Set[str] = set()
            for _, json_str in (registry_data or {}).items():
                if isinstance(json_str, bytes):
                    json_str = json_str.decode()
                try:
                    entry = json.loads(json_str)
                    if h := entry.get("content_hash"):
                        hashes.add(h)
                except json.JSONDecodeError:
                    continue
            return hashes

        except Exception as e:
            logger.warning("Failed to load dedup index (non-blocking): %s", e)
            return set()

    async def _delete_document_from_registry(
        self, collection: str, doc_id: str
    ) -> bool:
        """Delete document from Redis registry.

        TASK #82: Cleanup registry on document deletion.
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            return False

        try:
            redis_key = f"{REDIS_DOC_REGISTRY_PREFIX}:{collection}"
            deleted = await redis_client.hdel(redis_key, doc_id)
            if deleted:
                logger.info(
                    "Document removed from registry: %s in %s", doc_id, collection
                )
            return deleted > 0
        except Exception as e:
            logger.warning("Failed to delete document from registry: %s", e)
            return False

    async def _get_documents_from_registry(
        self, collection: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get all documents from Redis registry (fast path).

        TASK #82: Fast document listing from Redis.
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            return None

        try:
            redis_key = f"{REDIS_DOC_REGISTRY_PREFIX}:{collection}"
            registry_data = await redis_client.hgetall(redis_key)

            if not registry_data:
                return None

            documents = []
            for doc_id, json_str in registry_data.items():
                # Handle bytes from Redis
                if isinstance(doc_id, bytes):
                    doc_id = doc_id.decode()
                if isinstance(json_str, bytes):
                    json_str = json_str.decode()

                try:
                    doc_entry = json.loads(json_str)
                    documents.append(doc_entry)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in registry for doc_id: %s", doc_id)
                    continue

            return documents

        except Exception as e:
            logger.warning("Failed to read document registry: %s", e)
            return None

    # ---------------------------------------------------------------------
    # Operations (manifest-exposed)
    # ---------------------------------------------------------------------

    async def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add document to collection with Redis registry tracking.

        TASK #82: Enhanced with document lifecycle management.
        - Calculates content_hash for deduplication
        - Tracks uploader_id from metadata
        - Persists to Redis registry for fast listing
        """
        self._ensure_initialized()
        if not self.operation_handler:
            raise RuntimeError("Operation handler not initialized")

        # Ensure metadata dict exists
        metadata = metadata or {}

        # Calculate content hash before indexing
        content_hash = hashlib.md5(text.encode()).hexdigest()

        # Add content_hash to metadata for Qdrant payload
        metadata["content_hash"] = content_hash

        result = await self.operation_handler.add_document(
            doc_id=doc_id,
            text=text,
            metadata=metadata,
            collection=collection,
        )

        target_collection = collection or self.config["collection"]["default_name"]
        chunks_count = (result.data or {}).get("chunks_count", 1)

        if result.success:
            # TASK #82: Save to Redis registry
            registry_saved = await self._save_document_to_registry(
                collection=target_collection,
                doc_id=doc_id,
                text=text,
                metadata=metadata,
                chunks_count=chunks_count,
            )

            if self.event_publisher:
                await self.event_publisher.publish(
                    "document.indexed",
                    {
                        "doc_id": doc_id,
                        "collection": target_collection,
                        "chunks_count": chunks_count,
                        "content_hash": content_hash,
                    },
                )

            point_ids = (result.data or {}).get("point_ids") or []
            return {
                "status": "indexed",
                "doc_id": doc_id,
                "vector_id": point_ids[0] if point_ids else None,
                "chunks_count": chunks_count,
                "collection": target_collection,
                "content_hash": content_hash,
                "registry_saved": registry_saved,
                "duration_ms": result.duration_ms,
            }

        return {
            "status": "failed",
            "doc_id": doc_id,
            "error": result.error,
            "duration_ms": result.duration_ms,
        }

    async def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._ensure_initialized()
        if not self.operation_handler:
            raise RuntimeError("Operation handler not initialized")

        result = await self.operation_handler.query(
            query_text=query_text,
            top_k=top_k,
            collection=collection,
            filter_conditions=filter,
            score_threshold=score_threshold,
        )

        if hasattr(result, "to_dict"):
            response = result.to_dict()
            results_list = response.get("results", [])
            return {
                "results": results_list,
                "count": response.get("total_results", len(results_list)),
                "query": query_text,
                "collection": response.get("collection"),
                "filters_applied": response.get("filters_applied"),
                "duration_ms": response.get("duration_ms", 0),
            }

        # Fallback (should be rare)
        return {
            "results": [],
            "count": 0,
            "query": query_text,
            "collection": collection,
            "filters_applied": filter,
            "duration_ms": 0,
        }

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
    ) -> Dict[str, Any]:
        """Create a new collection with optional metadata.

        Args:
            collection_name: Name of the collection
            vector_size: Vector dimension (optional, auto-derived from embedding_model if not set)
            distance: Distance metric (optional)
            description: Human-readable description (optional)
            embedding_model: Embedding model to use (e.g., 'nomic-embed-text', 'all-MiniLM-L6-v2')
            kb_type: Knowledge base type ('universal', 'client', 'personal')
            chunk_size: Chunk size for text splitting (default: 512)
            chunk_overlap: Chunk overlap for text splitting (default: 50)

        Returns:
            Dict with creation status and metadata
        """
        self._ensure_initialized()
        if not self.qdrant_client:
            raise RuntimeError("Qdrant client not initialized")

        if await self.qdrant_client.collection_exists(collection_name):
            return {"status": "exists", "collection_name": collection_name}

        # TASK #70: Use passed embedding_model, derive vector_size from model if needed
        # Known models and their dimensions
        # Known models: model_name -> {dims, provider}
        # Supports both sentence-transformers (local) and ollama (container)
        # FIX-DIM-v4.1.2: Known models map for dimension derivation
        # Models are mapped to their output dimensions and providers
        known_models = {
            # Sentence-transformers (local HuggingFace models)
            "all-MiniLM-L6-v2": {"dims": 384, "provider": "sentence-transformers"},
            "all-mpnet-base-v2": {"dims": 768, "provider": "sentence-transformers"},
            "paraphrase-MiniLM-L6-v2": {
                "dims": 384,
                "provider": "sentence-transformers",
            },
            # BAAI models (HuggingFace - sentence-transformers compatible)
            "BAAI/bge-m3": {"dims": 1024, "provider": "sentence-transformers"},
            "BAAI/bge-base-en-v1.5": {"dims": 768, "provider": "sentence-transformers"},
            "BAAI/bge-large-en-v1.5": {"dims": 1024, "provider": "sentence-transformers"},
            # intfloat multilingual models
            "intfloat/multilingual-e5-large": {"dims": 1024, "provider": "sentence-transformers"},
            "intfloat/multilingual-e5-base": {"dims": 768, "provider": "sentence-transformers"},
            "intfloat/multilingual-e5-small": {"dims": 384, "provider": "sentence-transformers"},
            # Ollama embedding models (container-based)
            "nomic-embed-text": {"dims": 768, "provider": "ollama"},
            "nomic-embed-text:latest": {"dims": 768, "provider": "ollama"},
            "bge-m3": {"dims": 1024, "provider": "ollama"},
            "bge-m3:latest": {"dims": 1024, "provider": "ollama"},
            "mxbai-embed-large": {"dims": 1024, "provider": "ollama"},
            "mxbai-embed-large:latest": {"dims": 1024, "provider": "ollama"},
            "snowflake-arctic-embed:110m": {"dims": 768, "provider": "ollama"},
            "snowflake-arctic-embed:335m": {"dims": 1024, "provider": "ollama"},
            "Snowflake/snowflake-arctic-embed-l-v2.0": {"dims": 1024, "provider": "sentence-transformers"},
            "all-minilm": {"dims": 384, "provider": "ollama"},
        }

        # Determine final embedding model (prefer passed param over config)
        final_embedding_model = embedding_model or self.config.get("embedding", {}).get(
            "model", "all-MiniLM-L6-v2"
        )

        # Normalize short model names to full HuggingFace IDs
        # Prevents short names (e.g. "bge-m3") from resolving to Ollama provider
        _model_aliases = {
            "bge-m3": "BAAI/bge-m3",
            "bge-m3:latest": "BAAI/bge-m3",
        }
        final_embedding_model = _model_aliases.get(final_embedding_model, final_embedding_model)

        # Determine vector size: explicit > derived from model > config default
        if vector_size:
            final_vector_size = vector_size
        elif final_embedding_model in known_models:
            final_vector_size = known_models[final_embedding_model]["dims"]
        else:
            final_vector_size = int(
                self.config.get("collection", {}).get("vector_size", 384)
            )

        # Determine embedding provider
        if final_embedding_model in known_models:
            embedding_provider = known_models[final_embedding_model]["provider"]
        else:
            embedding_provider = self.config.get("embedding", {}).get(
                "provider", "sentence-transformers"
            )

        final_distance = distance or str(
            self.config.get("collection", {}).get("distance", "Cosine")
        )

        # Default kb_type if not provided
        final_kb_type = kb_type or "universal"

        # PHASE 2: Default chunking config
        final_chunk_size = chunk_size if chunk_size is not None else 512
        final_chunk_overlap = chunk_overlap if chunk_overlap is not None else 50

        logger.info(
            "Creating collection with user-specified params: model=%s, dims=%d, kb_type=%s, chunk_size=%d, chunk_overlap=%d",
            final_embedding_model,
            final_vector_size,
            final_kb_type,
            final_chunk_size,
            final_chunk_overlap,
        )

        await self.qdrant_client.create_collection(
            collection_name=collection_name,
            vector_size=final_vector_size,
            distance=final_distance,
        )

        # CRITICAL: Store collection metadata via CollectionMetadataManager
        # This is the SOURCE OF TRUTH for embedding model selection during ingestion
        # Metadata is ALWAYS saved (not just when description is provided)
        metadata_stored = False

        if self.operation_handler and self.operation_handler.metadata_manager:
            try:
                # TASK #70: Include kb_type in custom_metadata
                custom_meta = {}
                if description:
                    custom_meta["description"] = description
                if final_kb_type:
                    custom_meta["kb_type"] = final_kb_type

                # PHASE 2: Use dedicated chunking_config field (not custom_metadata)
                chunking_config = {
                    "chunk_size": final_chunk_size,
                    "chunk_overlap": final_chunk_overlap,
                }

                await self.operation_handler.metadata_manager.save_metadata(
                    collection_name=collection_name,
                    vector_size=final_vector_size,
                    distance_metric=final_distance,
                    embedding_model=final_embedding_model,
                    embedding_provider=embedding_provider,
                    chunking_config=chunking_config,
                    custom_metadata=custom_meta if custom_meta else None,
                )
                metadata_stored = True
                logger.info(
                    "Collection metadata saved via CollectionMetadataManager: %s",
                    collection_name,
                    extra={
                        "model": final_embedding_model,
                        "provider": embedding_provider,
                        "dimension": final_vector_size,
                        "kb_type": final_kb_type,
                    },
                )
            except Exception as e:
                logger.warning(
                    "Failed to store collection metadata: %s",
                    e,
                    extra={"collection": collection_name},
                )
        else:
            logger.warning(
                "CollectionMetadataManager not available - metadata will not be persisted for: %s",
                collection_name,
            )

        if self.event_publisher:
            await self.event_publisher.publish(
                "collection.created",
                {
                    "collection_name": collection_name,
                    "vector_size": final_vector_size,
                    "distance": final_distance,
                    "embedding_model": final_embedding_model,
                    "embedding_provider": embedding_provider,
                    "description": description,
                    "kb_type": final_kb_type,
                    "chunk_size": final_chunk_size,
                    "chunk_overlap": final_chunk_overlap,
                },
            )

        return {
            "status": "created",
            "collection_name": collection_name,
            "vector_size": final_vector_size,
            "distance": final_distance,
            "embedding_model": final_embedding_model,
            "embedding_provider": embedding_provider,
            "description": description,
            "kb_type": final_kb_type,
            "chunk_size": final_chunk_size,
            "chunk_overlap": final_chunk_overlap,
            "metadata_persisted": metadata_stored,
        }

    async def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """Delete a collection from Qdrant and clean up ALL associated Redis data.

        FIX-DELETE-SYNC-001: Complete cleanup to prevent zombie keys.

        Cleans up:
        1. Qdrant collection (vectors + payloads)
        2. Collection metadata: rag:collection:metadata:{collection_name}
        3. Collection index: rag:collections:index (SREM)
        4. Document registry: ubp:rag:documents:{collection_name} (HASH)

        Key formats per NAMING_POLICY.md:
        - Collection metadata: rag:collection:metadata:{name}
        - Collection index: rag:collections:index
        - Document registry: ubp:rag:documents:{name}
        """
        self._ensure_initialized()
        if not self.qdrant_client:
            raise RuntimeError("Qdrant client not initialized")

        # 1. Delete from Qdrant
        await self.qdrant_client.delete_collection(collection_name)
        logger.info(f"Qdrant collection deleted: {collection_name}")

        # Track cleanup status
        cleanup_status = {
            "metadata_cleaned": False,
            "index_updated": False,
            "doc_registry_cleaned": False,
        }

        # 2. Clean up collection metadata via OperationHandler's metadata_manager
        # This handles both the metadata JSON and the collections index set
        if self.operation_handler and self.operation_handler.metadata_manager:
            try:
                await self.operation_handler.metadata_manager.delete_metadata(
                    collection_name
                )
                cleanup_status["metadata_cleaned"] = True
                cleanup_status["index_updated"] = True
                logger.info(
                    f"Collection metadata cleaned from Redis: {collection_name}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to clean collection metadata via metadata_manager: {e}",
                    extra={"collection": collection_name},
                )

        # 3. Clean up document registry hash (ubp:rag:documents:{collection})
        redis_client = self.kwargs.get("redis_client")
        if redis_client:
            try:
                doc_registry_key = f"{REDIS_DOC_REGISTRY_PREFIX}:{collection_name}"
                deleted_count = await redis_client.delete(doc_registry_key)
                cleanup_status["doc_registry_cleaned"] = deleted_count > 0
                if cleanup_status["doc_registry_cleaned"]:
                    logger.info(
                        f"Document registry cleaned from Redis: {doc_registry_key}"
                    )
                else:
                    logger.debug(
                        f"No document registry found to clean: {doc_registry_key}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to clean document registry from Redis: {e}",
                    extra={"collection": collection_name},
                )

        return {
            "status": "deleted",
            "collection_name": collection_name,
            "cleanup": cleanup_status,
            "metadata_cleaned": cleanup_status["metadata_cleaned"],  # backward compat
        }

    async def get_collection_details(
        self,
        collection_name: str,
        include_documents: bool = False,
        uploader_filter: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get unified collection view with settings, stats, and documents.

        TASK #83: Unified Collection View (Modes A/B/C)
        - Mode A (include_documents=False): Settings + Stats only
        - Mode B (include_documents=False, uploader_filter=X): Filtered stats
        - Mode C (include_documents=True): Full view with paginated documents

        Args:
            collection_name: Name of the collection
            include_documents: If True, include paginated document list (Mode C)
            uploader_filter: Filter documents by uploader_id (Mode B/C)
            limit: Max documents to return (default 500)
            offset: Pagination offset (default 0)

        Returns:
            Dict with unified structure:
            {
                "settings": { collection config },
                "stats": { "total_documents": N, "total_chunks": M },
                "documents": [ ... ],  # Empty if include_documents=False
                "pagination": { "limit": N, "offset": N, "has_more": bool },
                "status": "found" | "healed" | "not_found" | "error"
            }
        """
        self._ensure_initialized()
        if not self.operation_handler:
            raise RuntimeError("Operation handler not initialized")

        start_time = time.time()
        metadata_manager = self.operation_handler.metadata_manager

        # Initialize response structure
        response = {
            "settings": {"collection_name": collection_name, "status": "not_found"},
            "stats": {
                "total_documents": 0,
                "total_chunks": 0,
            },
            "documents": [],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": False,
            },
            "status": "not_found",
        }

        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Get Settings from Redis (source of truth)
        # ─────────────────────────────────────────────────────────────────────
        metadata = await metadata_manager.get_metadata(collection_name)

        if metadata:
            custom_meta = metadata.custom_metadata or {}
            kb_type = custom_meta.get("kb_type", "universal")

            response["settings"] = {
                "collection_name": collection_name,
                "vector_size": metadata.vector_size,
                "distance_metric": metadata.distance_metric,
                "embedding_model": metadata.embedding_model,
                "embedding_provider": metadata.embedding_provider,
                "created_at": metadata.created_at,
                "updated_at": metadata.updated_at,
                "chunking_config": metadata.chunking_config,
                "custom_metadata": metadata.custom_metadata,
                "kb_type": kb_type,
            }
            response["status"] = "found"
        else:
            # Attempt healing from Qdrant
            logger.warning(
                f"No metadata for '{collection_name}' in Redis. Attempting healing..."
            )

            if self.qdrant_client:
                try:
                    dimension = await self.qdrant_client.get_vector_dimension_safe(
                        collection_name
                    )
                    if dimension:
                        known_models = {
                            384: ("all-MiniLM-L6-v2", "sentence-transformers"),
                            768: ("all-mpnet-base-v2", "sentence-transformers"),
                            1024: ("BAAI/bge-m3", "sentence-transformers"),
                            1536: ("text-embedding-ada-002", "openai"),
                            3072: ("text-embedding-3-large", "openai"),
                        }
                        model_info = known_models.get(dimension, ("unknown", "unknown"))

                        await metadata_manager.save_metadata(
                            collection_name=collection_name,
                            vector_size=dimension,
                            distance_metric="Cosine",
                            embedding_model=model_info[0],
                            embedding_provider=model_info[1],
                            custom_metadata={
                                "healed": True,
                                "healed_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )

                        response["settings"] = {
                            "collection_name": collection_name,
                            "vector_size": dimension,
                            "distance_metric": "Cosine",
                            "embedding_model": model_info[0],
                            "embedding_provider": model_info[1],
                            "created_at": None,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "chunking_config": None,
                            "custom_metadata": {"healed": True},
                            "kb_type": "universal",
                        }
                        response["status"] = "healed"
                        logger.info(
                            f"[HEALING] Collection '{collection_name}' metadata healed"
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to heal metadata for '{collection_name}': {e}"
                    )
                    response["status"] = "error"
                    response["settings"]["error"] = str(e)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Get Documents from Redis Registry
        # ─────────────────────────────────────────────────────────────────────
        all_documents: List[Dict[str, Any]] = []
        registry_docs = await self._get_documents_from_registry(collection_name)

        if registry_docs:
            all_documents = registry_docs
        else:
            # Fallback: Try to heal from Qdrant if collection exists
            if self.qdrant_client and response["status"] in ("found", "healed"):
                try:
                    qdrant_count = await self.qdrant_client.count(collection_name)
                    if qdrant_count > 0:
                        # Scroll Qdrant and heal Redis
                        scroll_result = await self.qdrant_client.scroll(
                            collection_name=collection_name,
                            limit=min(1000, limit + offset),
                            offset=None,
                            with_payload=True,
                            with_vectors=False,
                        )

                        points = scroll_result.get("points", [])
                        seen_doc_ids = set()
                        redis_client = self._get_redis_client()

                        for point in points:
                            payload = point.get("payload") or {}
                            doc_id = (
                                payload.get("doc_id")
                                or payload.get("document_id")
                                or str(point.get("id"))
                            )

                            if doc_id in seen_doc_ids:
                                continue
                            seen_doc_ids.add(doc_id)

                            doc_entry = {
                                "doc_id": doc_id,
                                "filename": payload.get("filename")
                                or payload.get("title")
                                or doc_id,
                                "content_hash": payload.get("content_hash", ""),
                                "chunk_count": payload.get("chunk_count", 1),
                                "chunk_strategy": payload.get(
                                    "chunk_strategy", "default"
                                ),
                                "uploader_id": payload.get("uploader_id", "system"),
                                "upload_timestamp": payload.get("upload_timestamp")
                                or payload.get("created_at")
                                or datetime.now(timezone.utc).isoformat(),
                                "file_size": payload.get("file_size", 0),
                                "mime_type": payload.get("mime_type", "text/plain"),
                                "status": "indexed",
                                "title": payload.get("title"),
                                "source": payload.get("source"),
                            }
                            all_documents.append(doc_entry)

                            # Auto-heal to Redis
                            if redis_client:
                                try:
                                    redis_key = (
                                        f"{REDIS_DOC_REGISTRY_PREFIX}:{collection_name}"
                                    )
                                    await redis_client.hset(
                                        redis_key, doc_id, json.dumps(doc_entry)
                                    )
                                except Exception:
                                    pass

                        if all_documents:
                            logger.info(
                                f"Auto-healed {len(all_documents)} documents to Redis registry for {collection_name}"
                            )

                except Exception as e:
                    logger.warning(
                        f"Failed to get documents from Qdrant for healing: {e}"
                    )

        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Filter by uploader_id (Mode B/C)
        # ─────────────────────────────────────────────────────────────────────
        filtered_documents = all_documents

        if uploader_filter:
            filtered_documents = [
                doc
                for doc in all_documents
                if doc.get("uploader_id") == uploader_filter
            ]

        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Calculate Stats
        # ─────────────────────────────────────────────────────────────────────
        total_documents = len(filtered_documents)
        total_chunks = sum(doc.get("chunk_count", 1) for doc in filtered_documents)

        response["stats"] = {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
        }

        # Add Qdrant point count if available (can differ from registry)
        if self.qdrant_client and response["status"] in ("found", "healed"):
            try:
                qdrant_count = await self.qdrant_client.count(collection_name)
                response["stats"]["qdrant_points"] = qdrant_count
            except Exception:
                pass

        # ─────────────────────────────────────────────────────────────────────
        # STEP 5: Paginate Documents (Mode C)
        # ─────────────────────────────────────────────────────────────────────
        if include_documents:
            # Sort by upload_timestamp descending (newest first)
            filtered_documents.sort(
                key=lambda x: x.get("upload_timestamp", ""), reverse=True
            )

            # Apply pagination
            paginated = filtered_documents[offset : offset + limit]
            has_more = (offset + len(paginated)) < total_documents

            response["documents"] = paginated
            response["pagination"] = {
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
                "total": total_documents,
            }

        # Add timing
        response["duration_ms"] = round((time.time() - start_time) * 1000, 2)

        return response

    # ---------------------------------------------------------------------
    # Additional helpers (backward compatible)
    # ---------------------------------------------------------------------

    async def delete_document(
        self, doc_id: str, collection: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delete a document (all chunks) from collection and Redis registry.

        TASK #82: Enhanced with Redis registry cleanup.
        - Deletes from Qdrant (all chunks with matching doc_id)
        - Deletes from Redis registry
        """
        self._ensure_initialized()
        if not self.operation_handler:
            raise RuntimeError("Operation handler not initialized")

        target_collection = collection or self.config.get("collection", {}).get(
            "default_name"
        )

        result = await self.operation_handler.delete_document(
            doc_id=doc_id, collection=collection
        )

        if result.success:
            # TASK #82: Also delete from Redis registry
            registry_deleted = await self._delete_document_from_registry(
                target_collection, doc_id
            )

            return {
                "status": "deleted",
                "doc_id": doc_id,
                "chunks_deleted": (result.data or {}).get("chunks_deleted", 0),
                "collection": target_collection,
                "registry_deleted": registry_deleted,
                "duration_ms": result.duration_ms,
            }

        return {
            "status": "failed",
            "doc_id": doc_id,
            "error": result.error,
            "duration_ms": result.duration_ms,
        }

    async def list_collections(self) -> List[str]:
        """List all collections in Qdrant."""
        self._ensure_initialized()
        if not self.operation_handler:
            raise RuntimeError("Operation handler not initialized")

        result = await self.operation_handler.list_collections()
        if result.success:
            return list((result.data or {}).get("collections") or [])
        # Graceful degradation: return empty list
        return []

    async def clear(self, collection: Optional[str] = None) -> Dict[str, Any]:
        """Best-effort clear by deleting & recreating a collection.

        Also clears the Redis document registry for the collection.
        """
        self._ensure_initialized()
        if not self.qdrant_client:
            raise RuntimeError("Qdrant client not initialized")
        if not self.embedding_manager:
            raise RuntimeError("Embedding manager not initialized")

        target = collection or self.config.get("collection", {}).get("default_name")
        if not target:
            raise ValueError("No collection specified and no default configured")

        try:
            # Clear Qdrant collection
            if await self.qdrant_client.collection_exists(target):
                await self.qdrant_client.delete_collection(target)
            await self.qdrant_client.create_collection(
                collection_name=target,
                vector_size=self.embedding_manager.dimension,
                distance=self.config.get("collection", {}).get("distance", "Cosine"),
            )

            # Clear Redis document registry (TASK: Personal KB clear sync)
            redis_client = self._get_redis_client()
            if redis_client:
                try:
                    redis_key = f"{REDIS_DOC_REGISTRY_PREFIX}:{target}"
                    await redis_client.delete(redis_key)
                    logger.info(
                        "Cleared Redis document registry for collection %s", target
                    )
                except Exception as redis_err:
                    logger.warning(
                        "Failed to clear Redis registry for %s: %s", target, redis_err
                    )

            return {"status": "cleared", "collection_name": target}
        except Exception as e:
            logger.warning("Failed to clear collection %s: %s", target, e)
            return {"status": "failed", "collection_name": target, "error": str(e)}

    async def get_stats(self) -> Dict[str, Any]:
        """Return operational stats."""
        self._ensure_initialized()

        collection_name = self.config.get("collection", {}).get("default_name")
        total_documents = 0

        try:
            if self.qdrant_client and collection_name:
                info = await self.qdrant_client.get_collection_info(collection_name)
                total_documents = int((info or {}).get("points_count", 0))
        except Exception as e:
            # stats must not break the module
            logger.warning("Failed to compute stats: %s", e)

        return {
            "module": self.manifest.name,
            "collection_name": collection_name,
            "total_documents": total_documents,
            "initialized": self._initialized,
        }

    async def list_documents(
        self,
        collection: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List documents in a collection with pagination.

        TASK #82: Enhanced with Redis fast path and auto-healing.
        - Step 1 (Fast Path): Read from Redis registry
        - Step 2 (Healing Path): If Redis empty, read from Qdrant and heal Redis

        Args:
            collection: Collection name (uses default if not specified)
            limit: Maximum number of documents to return (default 100)
            offset: Number of documents to skip (default 0)

        Returns:
            Dict with 'documents' list, 'total', 'limit', 'offset', 'has_more', and 'source'
        """
        self._ensure_initialized()
        if not self.qdrant_client:
            raise RuntimeError("Qdrant client not initialized")

        target_collection = collection or self.config.get("collection", {}).get(
            "default_name", "documents"
        )

        start_time = time.time()

        try:
            # ─────────────────────────────────────────────────────────────────
            # STEP 1: Try Redis Fast Path
            # ─────────────────────────────────────────────────────────────────
            registry_docs = await self._get_documents_from_registry(target_collection)

            if registry_docs:
                # Sort by upload_timestamp descending (newest first)
                registry_docs.sort(
                    key=lambda x: x.get("upload_timestamp", ""), reverse=True
                )

                # Apply pagination
                total = len(registry_docs)
                paginated = registry_docs[offset : offset + limit]

                duration_ms = (time.time() - start_time) * 1000

                return {
                    "documents": paginated,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "has_more": (offset + len(paginated)) < total,
                    "collection": target_collection,
                    "source": "redis",
                    "duration_ms": round(duration_ms, 2),
                }

            # ─────────────────────────────────────────────────────────────────
            # STEP 2: Healing Path - Read from Qdrant and heal Redis
            # ─────────────────────────────────────────────────────────────────
            logger.info(
                "Redis registry empty for %s, using Qdrant healing path",
                target_collection,
            )

            # Get total count
            total_points = await self.qdrant_client.count(target_collection)

            if total_points == 0:
                duration_ms = (time.time() - start_time) * 1000
                return {
                    "documents": [],
                    "total": 0,
                    "limit": limit,
                    "offset": offset,
                    "has_more": False,
                    "collection": target_collection,
                    "source": "qdrant",
                    "duration_ms": round(duration_ms, 2),
                }

            # Scroll through Qdrant to get documents
            scroll_result = await self.qdrant_client.scroll(
                collection_name=target_collection,
                limit=limit + offset,
                offset=None,
                with_payload=True,
                with_vectors=False,
            )

            points = scroll_result.get("points", [])
            paginated_points = (
                points[offset : offset + limit] if offset > 0 else points[:limit]
            )

            # Transform and deduplicate
            documents = []
            seen_doc_ids = set()
            docs_to_heal = []

            for point in paginated_points:
                payload = point.get("payload") or {}
                doc_id = (
                    payload.get("doc_id")
                    or payload.get("document_id")
                    or str(point.get("id"))
                )

                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)

                # Build document entry
                doc_entry = {
                    "doc_id": doc_id,
                    "filename": payload.get("filename")
                    or payload.get("title")
                    or doc_id,
                    "content_hash": payload.get("content_hash", ""),
                    "chunk_count": payload.get("chunk_count", 1),
                    "chunk_strategy": payload.get("chunk_strategy", "default"),
                    "uploader_id": payload.get("uploader_id", "system"),
                    "upload_timestamp": payload.get("upload_timestamp")
                    or payload.get("created_at")
                    or datetime.now(timezone.utc).isoformat(),
                    "file_size": payload.get("file_size", 0),
                    "mime_type": payload.get("mime_type", "text/plain"),
                    "status": "indexed",
                    "title": payload.get("title"),
                    "source": payload.get("source"),
                    # Legacy fields for backward compat
                    "text_preview": (payload.get("text") or "")[:200] + "..."
                    if payload.get("text")
                    else None,
                }

                documents.append(doc_entry)
                docs_to_heal.append(doc_entry)

            # Auto-heal: Save to Redis registry
            redis_client = self._get_redis_client()
            if redis_client and docs_to_heal:
                try:
                    redis_key = f"{REDIS_DOC_REGISTRY_PREFIX}:{target_collection}"
                    for doc in docs_to_heal:
                        await redis_client.hset(
                            redis_key, doc["doc_id"], json.dumps(doc)
                        )
                    logger.info(
                        "Auto-healed %d documents to Redis registry for %s",
                        len(docs_to_heal),
                        target_collection,
                    )
                except Exception as e:
                    logger.warning("Failed to auto-heal Redis registry: %s", e)

            duration_ms = (time.time() - start_time) * 1000

            return {
                "documents": documents,
                "total": total_points,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + len(documents)) < total_points,
                "collection": target_collection,
                "source": "qdrant+healed",
                "duration_ms": round(duration_ms, 2),
            }

        except Exception as e:
            logger.error("Failed to list documents in %s: %s", target_collection, e)
            return {
                "documents": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "has_more": False,
                "collection": target_collection,
                "error": str(e),
            }

    async def _get_ollama_embedding_dims(
        self, client: httpx.AsyncClient, base_url: str, model_name: str
    ) -> int:
        """
        Get embedding dimensions for an Ollama model by generating a test embedding.

        Args:
            client: HTTP client
            base_url: Ollama API base URL
            model_name: Name of the model

        Returns:
            Embedding dimensions, or 0 if unable to determine
        """
        try:
            response = await client.post(
                f"{base_url}/api/embeddings",
                json={"model": model_name, "prompt": "test"},
                timeout=30.0,
            )
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding", [])
                return len(embedding)
        except Exception as e:
            logger.warning(f"Failed to get embedding dims for {model_name}: {e}")
        return 0

    async def list_embedding_models(self) -> Dict[str, Any]:
        """List all available embedding models from all configured providers.

        TASK #76: Grand Unified Registry - Aggregates models from:
        - Local (Sentence-Transformers): Always available, downloaded on-demand
        - Ollama (Container): Lists models currently installed in ubp-ollama
        - OpenAI (Cloud): Available if UBP_OPENAI_API_KEY is set
        - Cohere (Cloud): Available if UBP_COHERE_API_KEY is set

        Returns:
            Dict with:
            - models: List of model dicts with id, name, dims, provider, available, description
            - providers: Dict of provider status (available: bool, reason: str)
            - total_count: int
            - available_count: int
        """
        start_time = time.time()
        available_models: List[Dict[str, Any]] = []
        providers_status: Dict[str, Dict[str, Any]] = {}

        # ─────────────────────────────────────────────────────────────────────
        # A. LOCAL (Sentence-Transformers) - Always available
        # ─────────────────────────────────────────────────────────────────────
        for model in LOCAL_MODELS:
            available_models.append({**model, "available": True})

        providers_status["sentence-transformers"] = {
            "available": True,
            "reason": "Local models, downloaded on-demand",
            "model_count": len(LOCAL_MODELS),
        }

        # ─────────────────────────────────────────────────────────────────────
        # B. OLLAMA (Container) - Dynamic discovery of embedding models
        # ─────────────────────────────────────────────────────────────────────
        ollama_models = []
        ollama_available = False
        ollama_reason = "Not checked"

        try:
            # Get Ollama base URL from environment
            ollama_base_url = os.environ.get(
                "UBP_OLLAMA_BASE_URL",
                os.environ.get("UBP_OLLAMA_API_URL", "http://ubp-ollama:11434"),
            )

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{ollama_base_url}/api/tags")

                if response.status_code == 200:
                    data = response.json()
                    installed_models = data.get("models", [])

                    # Dynamic detection: check each model for embedding capability
                    for model_info in installed_models:
                        model_name = model_info.get("name", "")
                        model_name_lower = model_name.lower()

                        # Check if model name matches embedding patterns
                        is_embedding = any(
                            pattern in model_name_lower
                            for pattern in OLLAMA_EMBEDDING_PATTERNS
                        )

                        if is_embedding:
                            # Get actual dimensions by querying the model
                            dims = await self._get_ollama_embedding_dims(
                                client, ollama_base_url, model_name
                            )

                            ollama_models.append(
                                {
                                    "id": model_name,
                                    "name": model_name.replace(":", " ").replace("-", " ").title(),
                                    "dims": dims,
                                    "provider": "ollama",
                                    "description": "Installed in Ollama container",
                                    "available": True,
                                }
                            )

                    ollama_available = True
                    ollama_reason = f"Container reachable at {ollama_base_url}"
                else:
                    ollama_reason = f"API returned status {response.status_code}"

        except httpx.ConnectError:
            ollama_reason = "Connection refused - Ollama container not running"
        except httpx.TimeoutException:
            ollama_reason = "Connection timeout - Ollama container unreachable"
        except Exception as e:
            ollama_reason = f"Error: {str(e)}"
            logger.warning("Failed to query Ollama for embedding models: %s", e)

        available_models.extend(ollama_models)
        providers_status["ollama"] = {
            "available": ollama_available,
            "reason": ollama_reason,
            "model_count": len(ollama_models),
        }

        # ─────────────────────────────────────────────────────────────────────
        # C. OPENAI (Cloud) - If API key configured
        # ─────────────────────────────────────────────────────────────────────
        openai_api_key = os.environ.get("UBP_OPENAI_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )

        if openai_api_key:
            for model in OPENAI_MODELS:
                available_models.append({**model, "available": True})

            providers_status["openai"] = {
                "available": True,
                "reason": "API key configured",
                "model_count": len(OPENAI_MODELS),
            }
        else:
            # Add models as unavailable for reference
            for model in OPENAI_MODELS:
                available_models.append({**model, "available": False})

            providers_status["openai"] = {
                "available": False,
                "reason": "No API key (set UBP_OPENAI_API_KEY)",
                "model_count": 0,
            }

        # ─────────────────────────────────────────────────────────────────────
        # D. COHERE (Cloud) - If API key configured
        # ─────────────────────────────────────────────────────────────────────
        cohere_api_key = os.environ.get("UBP_COHERE_API_KEY") or os.environ.get(
            "COHERE_API_KEY"
        )

        if cohere_api_key:
            for model in COHERE_MODELS:
                available_models.append({**model, "available": True})

            providers_status["cohere"] = {
                "available": True,
                "reason": "API key configured",
                "model_count": len(COHERE_MODELS),
            }
        else:
            # Add models as unavailable for reference
            for model in COHERE_MODELS:
                available_models.append({**model, "available": False})

            providers_status["cohere"] = {
                "available": False,
                "reason": "No API key (set UBP_COHERE_API_KEY)",
                "model_count": 0,
            }

        # ─────────────────────────────────────────────────────────────────────
        # Final aggregation
        # ─────────────────────────────────────────────────────────────────────
        duration_ms = (time.time() - start_time) * 1000
        available_count = sum(1 for m in available_models if m.get("available"))

        # Current system model (from config / .env)
        current_model = self.config.get("embedding", {}).get("model", "")
        current_dimension = self.config.get("embedding", {}).get("dimension", 384)
        current_provider = self.config.get("embedding", {}).get("provider", "sentence-transformers")

        return {
            "models": available_models,
            "providers": providers_status,
            "current_model": current_model,
            "current_dimension": current_dimension,
            "current_provider": current_provider,
            "total_count": len(available_models),
            "available_count": available_count,
            "duration_ms": round(duration_ms, 2),
        }
