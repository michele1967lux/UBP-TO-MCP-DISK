"""
RAG Operations - Enterprise Grade

Production-ready operation handlers with:
- Complete CRUD operations
- Batch processing
- Filtering and search
- Metadata management
- Validation and error handling
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic

from .client import CollectionNotFoundError
from .collection_metadata import CollectionMetadataManager

logger = logging.getLogger(__name__)


# ============================================================================
# UUID Generation Helper
# ============================================================================

# Namespace UUID for RAG Qdrant point IDs (deterministic)
RAG_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def generate_point_id(doc_id: str, chunk_index: int) -> str:
    """
    Generate deterministic UUID for Qdrant point ID.

    Uses UUID v5 (SHA-1 hash) for deterministic generation based on:
    - doc_id: Document identifier
    - chunk_index: Chunk position in document

    Returns:
        UUID string compatible with Qdrant (no hyphens for consistency)
    """
    # Create deterministic UUID from doc_id + chunk_index
    composite_key = f"{doc_id}:chunk:{chunk_index}"
    point_uuid = uuid.uuid5(RAG_NAMESPACE, composite_key)
    # Return UUID without hyphens (Qdrant accepts both formats)
    return str(point_uuid)


# ============================================================================
# Operation Results
# ============================================================================


class OperationStatus(Enum):
    """Operation status codes."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    PENDING = "pending"
    CANCELLED = "cancelled"
    NOT_FOUND = "not_found"  # Document/resource not found (idempotent success)


@dataclass
class OperationResult:
    """Result of an operation."""

    status: OperationStatus
    operation: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def success(self) -> bool:
        return self.status == OperationStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "operation": self.operation,
            "data": self.data,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp,
        }


@dataclass
class SearchResult:
    """Individual search result."""

    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: Optional[int] = None
    doc_id: Optional[str] = None
    vector: Optional[List[float]] = None  # v6.1.3: embedding for dedup/fusion

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "score": round(self.score, 4),
            "text": self.text,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "doc_id": self.doc_id,
        }
        if self.vector is not None:
            d["vector"] = self.vector
        return d


@dataclass
class QueryResult:
    """Result of a query operation."""

    query: str
    results: List[SearchResult]
    total_results: int
    duration_ms: float
    collection: str
    filters_applied: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "duration_ms": round(self.duration_ms, 2),
            "collection": self.collection,
            "filters_applied": self.filters_applied,
        }
        if self.error_code:
            d["error_code"] = self.error_code
            d["error_message"] = self.error_message
        return d


# ============================================================================
# Validation
# ============================================================================


class ValidationError(Exception):
    """Validation error with details."""

    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(message)


class OperationValidator:
    """Validates operation parameters."""

    # GAP-INGEST-001: Smart Ingestion Hard Limit
    # 20MB text = ~20 million characters (safety limit for infrastructure)
    # Documents larger than this should be pre-split by the client
    MAX_TEXT_LENGTH = 20 * 1024 * 1024  # 20MB

    @staticmethod
    def validate_doc_id(doc_id: str) -> None:
        """Validate document ID."""
        if not doc_id:
            raise ValidationError("Document ID cannot be empty", "doc_id")
        if len(doc_id) > 512:
            raise ValidationError("Document ID too long (max 512 chars)", "doc_id")

    @staticmethod
    def validate_text(
        text: str,
        min_length: int = 1,
        max_length: Optional[int] = None,
    ) -> None:
        """
        Validate text content with smart limits.

        GAP-INGEST-001: Removed 100k character blocking limit.
        Large documents are processed via chunking, not blocked.
        Only enforces a hard safety limit (20MB) for infrastructure protection.

        Args:
            text: Text content to validate
            min_length: Minimum text length (default: 1)
            max_length: Maximum text length (default: MAX_TEXT_LENGTH = 20MB)
        """
        if max_length is None:
            max_length = OperationValidator.MAX_TEXT_LENGTH

        if not text or not text.strip():
            raise ValidationError("Text cannot be empty", "text")
        if len(text) < min_length:
            raise ValidationError(f"Text too short (min {min_length} chars)", "text")
        if len(text) > max_length:
            raise ValidationError(
                f"Text exceeds safety limit ({len(text):,} chars > {max_length:,} chars). "
                f"Please split the document before ingestion.",
                "text",
            )

    @staticmethod
    def validate_collection_name(name: str) -> None:
        """Validate collection name."""
        if not name:
            raise ValidationError("Collection name cannot be empty", "collection_name")
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValidationError(
                "Collection name can only contain alphanumeric, underscore, and hyphen",
                "collection_name",
            )
        if len(name) > 128:
            raise ValidationError(
                "Collection name too long (max 128 chars)", "collection_name"
            )

    @staticmethod
    def validate_top_k(top_k: int, max_value: int = 500) -> None:
        """Validate top_k parameter."""
        if top_k < 1:
            raise ValidationError("top_k must be at least 1", "top_k")
        if top_k > max_value:
            raise ValidationError(f"top_k too large (max {max_value})", "top_k")

    @staticmethod
    def validate_score_threshold(threshold: float) -> None:
        """Validate score threshold."""
        if threshold < 0.0 or threshold > 1.0:
            raise ValidationError(
                "Score threshold must be between 0 and 1", "score_threshold"
            )

    @staticmethod
    def validate_vector_size(size: int) -> None:
        """Validate vector size."""
        if size < 1:
            raise ValidationError("Vector size must be positive", "vector_size")
        if size > 65536:
            raise ValidationError("Vector size too large (max 65536)", "vector_size")


# ============================================================================
# Filter Builder
# ============================================================================


class FilterBuilder:
    """
    Build Qdrant-compatible filters from user-friendly format.

    Supports:
    - Equality: {"field": "value"}
    - Range: {"field": {"$gt": 10, "$lt": 20}}
    - In: {"field": {"$in": ["a", "b"]}}
    - Not: {"field": {"$ne": "value"}}
    - And/Or: {"$and": [...], "$or": [...]}
    """

    @classmethod
    def build(cls, filter_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build Qdrant filter from dictionary."""
        if not filter_dict:
            return None

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

            conditions = []

            for key, value in filter_dict.items():
                if key.startswith("$"):
                    # Logical operator
                    if key == "$and":
                        sub_filters = [cls.build(f) for f in value if f]
                        if sub_filters:
                            for sf in sub_filters:
                                if sf and hasattr(sf, "must"):
                                    conditions.extend(sf.must or [])
                    elif key == "$or":
                        pass
                else:
                    condition = cls._build_field_condition(key, value)
                    if condition:
                        conditions.append(condition)

            if conditions:
                return Filter(must=conditions)
            return None

        except ImportError:
            return filter_dict

    @classmethod
    def _build_field_condition(cls, field: str, value: Any):
        """Build a single field condition."""
        try:
            from qdrant_client.models import FieldCondition, MatchValue, Range

            if isinstance(value, dict):
                if (
                    "$gt" in value
                    or "$gte" in value
                    or "$lt" in value
                    or "$lte" in value
                ):
                    return FieldCondition(
                        key=field,
                        range=Range(
                            gt=value.get("$gt"),
                            gte=value.get("$gte"),
                            lt=value.get("$lt"),
                            lte=value.get("$lte"),
                        ),
                    )
                elif "$in" in value:
                    return FieldCondition(key=field, match=MatchValue(any=value["$in"]))
                elif "$ne" in value:
                    return FieldCondition(
                        key=field, match=MatchValue(except_=value["$ne"])
                    )
            else:
                return FieldCondition(key=field, match=MatchValue(value=value))

        except ImportError:
            return None


# ============================================================================
# Operation Handlers
# ============================================================================


class OperationHandler:
    """
    Handles all RAG operations.

    Provides:
    - Document operations (add, get, update, delete)
    - Collection operations (create, delete, list)
    - Query operations (search, hybrid search)
    - Batch operations
    """

    def __init__(
        self,
        client,
        embedding_manager,
        chunking_manager,
        config: Dict[str, Any],
        redis_client: Optional[Any] = None,
    ):
        self.client = client
        self.embedding_manager = embedding_manager
        self.chunking_manager = chunking_manager
        self.config = config
        self._chunker_cache: dict = {}  # v6.4.0: (chunk_size, overlap, strategy) → ChunkingManager

        # Default collection
        self.default_collection = config.get("collection", {}).get(
            "default_name", "documents"
        )

        # Retrieval config
        retrieval_config = config.get("retrieval", {})
        self.default_top_k = retrieval_config.get("default_top_k", 5)
        # FIX-BUG-004 v1.8.3: Lower default to 0.1 (was 0.7)
        # 0.7 was too aggressive, filtering out potentially relevant results
        # Most embedding models produce scores in 0.3-0.9 range for relevant docs
        self.score_threshold = retrieval_config.get("score_threshold", 0.1)
        self.with_payload = retrieval_config.get("with_payload", True)
        self.with_vectors = retrieval_config.get("with_vectors", False)

        # Collection metadata manager
        self.metadata_manager = CollectionMetadataManager(redis_client)

        # Statistics
        self._stats = {
            "documents_added": 0,
            "documents_deleted": 0,
            "queries_executed": 0,
            "chunks_created": 0,
        }

        # Redis client for metadata operations
        self._redis_client = redis_client

        # FIX-DIM-v4.1.2: Known embedding models indexed by dimension for auto-switching
        # Order matters: first model in list is preferred fallback
        # This map is used both for:
        # 1. Fallback model selection when target model is unavailable
        # 2. Inferring model from dimension for legacy collections
        self._known_models: Dict[int, List[Dict[str, str]]] = {
            384: [
                {"model": "all-MiniLM-L6-v2", "provider": "sentence-transformers"},
                {
                    "model": "paraphrase-MiniLM-L6-v2",
                    "provider": "sentence-transformers",
                },
                {"model": "all-minilm", "provider": "ollama"},
            ],
            768: [
                {
                    "model": "nomic-embed-text",
                    "provider": "ollama",
                },  # Ollama preferred for 768
                {"model": "nomic-embed-text", "provider": "sentence-transformers"},
                {"model": "all-mpnet-base-v2", "provider": "sentence-transformers"},
                {"model": "BAAI/bge-base-en-v1.5", "provider": "sentence-transformers"},
                {"model": "intfloat/multilingual-e5-base", "provider": "sentence-transformers"},
                {"model": "snowflake-arctic-embed:110m", "provider": "ollama"},
                {
                    "model": "bert-base-nli-mean-tokens",
                    "provider": "sentence-transformers",
                },
            ],
            1024: [
                {"model": "Snowflake/snowflake-arctic-embed-l-v2.0", "provider": "sentence-transformers"},  # v6.8: Default Matryoshka 1024d
                {"model": "BAAI/bge-m3", "provider": "sentence-transformers"},
                {"model": "bge-m3", "provider": "ollama"},
                {"model": "intfloat/multilingual-e5-large", "provider": "sentence-transformers"},
                {"model": "BAAI/bge-large-en-v1.5", "provider": "sentence-transformers"},
                {"model": "mxbai-embed-large", "provider": "ollama"},
                {"model": "snowflake-arctic-embed:335m", "provider": "ollama"},
            ],
            1536: [
                {"model": "text-embedding-ada-002", "provider": "openai"},
                {"model": "text-embedding-3-small", "provider": "openai"},
            ],
            3072: [
                {"model": "text-embedding-3-large", "provider": "openai"},
            ],
        }

        # FIX-MATRYOSHKA-001: Models supporting Matryoshka (variable-dim) embeddings.
        # Native dim is max; any truncation ≤ native is valid. Same model → no switch.
        self._matryoshka_models: Dict[str, int] = {
            "Snowflake/snowflake-arctic-embed-l-v2.0": 1024,
        }

        # v6.3.2: ENSURE-COMPAT cache to avoid redundant Redis+Qdrant calls per cycle
        self._compat_cache: dict = {}  # collection_name → (timestamp, dimension)
        self._compat_cache_ttl: float = float(
            self.config.get("ensure_compat_cache_ttl", 60.0)
        )
        # FIX-MATRYOSHKA-002: Per-collection Matryoshka truncation target.
        # collection_name → target_dim when query vector must be truncated, else None.
        self._matryoshka_truncation: Dict[str, Optional[int]] = {}

    # =========================================================================
    # Collection Rebuild Safety Check
    # =========================================================================

    async def check_rebuild_needed(self, collection_name: str) -> Dict[str, Any]:
        """
        Check whether a collection needs rebuild (delete + recreate).

        Returns a dict with:
            - needs_rebuild: bool — True if KB is empty or has dimension mismatch
            - reason: str — "empty_kb", "dimension_mismatch", "not_found", or "healthy"
            - allow_delete: bool — True if deletion is safe (empty or dimension mismatch)
            - details: dict — current_chunks, current_dim, expected_dim

        Used by delete_knowledge_base to protect populated, healthy collections
        from accidental deletion. Force delete requires explicit `force=True` parameter.
        """
        result: Dict[str, Any] = {
            "needs_rebuild": False,
            "reason": "healthy",
            "allow_delete": False,
            "details": {},
        }

        try:
            # Check if collection exists in Qdrant
            qdrant_dimension = await self.client.get_vector_dimension_safe(collection_name)
            if qdrant_dimension is None:
                # Collection doesn't exist in Qdrant
                result["needs_rebuild"] = True
                result["reason"] = "not_found"
                result["allow_delete"] = True
                return result

            # Get chunk count
            points_count = await self.client.count(collection_name)

            # Get expected dimension from current embedding manager
            expected_dim = self.embedding_manager.dimension

            result["details"] = {
                "current_chunks": points_count,
                "current_dim": qdrant_dimension,
                "expected_dim": expected_dim,
            }

            # Condition 1: KB is empty
            if points_count == 0:
                result["needs_rebuild"] = True
                result["reason"] = "empty_kb"
                result["allow_delete"] = True
                return result

            # Condition 2: Dimension mismatch
            if qdrant_dimension != expected_dim:
                result["needs_rebuild"] = True
                result["reason"] = "dimension_mismatch"
                result["allow_delete"] = True
                return result

            # KB has data and dimensions match — healthy, no rebuild needed
            return result

        except Exception as e:
            logger.warning(f"[REBUILD-CHECK] Error checking collection '{collection_name}': {e}")
            # If collection can't be inspected, allow delete (likely doesn't exist)
            result["needs_rebuild"] = True
            result["reason"] = f"check_error: {e}"
            result["allow_delete"] = True
            return result

    # =========================================================================
    # Embedding Compatibility (METADATA-DRIVEN)
    # =========================================================================

    async def _ensure_compatible_embedding(self, collection_name: str) -> Optional[int]:
        """
        Ensure embedding manager is compatible with target collection.

        FIX-DIM-v4.1.2: Enhanced with coherence check between Redis metadata and Qdrant.
        FIX-MATRYOSHKA-002: Returns target truncation dimension when Matryoshka truncation
        is needed (collection_dim < model_native_dim), or None otherwise.

        This implements STRICT metadata-driven embedding selection:
        1. Read collection metadata from Redis (source of truth)
        2. VERIFY metadata coherence with actual Qdrant collection
        3. If mismatch and collection empty, auto-heal metadata
        4. If current model is incompatible, attempt to switch
        5. If no compatible model found, FAIL CLOSED

        Args:
            collection_name: Target collection name

        Returns:
            int: target_dimension if Matryoshka truncation must be applied to query vector
            None: no truncation needed (dims match or model was switched)

        Raises:
            RuntimeError: If no compatible embedding model can be found
        """
        # v6.3.2: Check in-memory cache first
        import time as _time
        _cache_entry = self._compat_cache.get(collection_name)
        if _cache_entry is not None:
            _cached_ts, _cached_dim = _cache_entry
            if (_time.time() - _cached_ts) < self._compat_cache_ttl:
                current_dimension = self.embedding_manager.dimension
                if current_dimension == _cached_dim:
                    logger.debug(
                        f"[ENSURE-COMPAT] CACHE HIT for '{collection_name}' "
                        f"(dim={_cached_dim}, age={_time.time() - _cached_ts:.1f}s)"
                    )
                    # FIX-MATRYOSHKA-002: Return cached truncation target (None if no truncation needed)
                    return self._matryoshka_truncation.get(collection_name)
            # Cache expired or dimension changed — invalidate both caches and proceed
            self._compat_cache.pop(collection_name, None)
            self._matryoshka_truncation.pop(collection_name, None)

        # DEBUG: Trace entry point
        logger.debug(f"[ENSURE-COMPAT] ENTER for collection '{collection_name}'")

        # Get ENV/config values (ground truth for system-wide settings)
        env_model = self.config.get("embedding", {}).get("model", "unknown")
        env_dimension = self.embedding_manager.dimension
        env_provider = self.config.get("embedding", {}).get("provider", "unknown")

        logger.debug(f"[ENSURE-COMPAT] Current state: env_model={env_model}, env_dim={env_dimension}")

        # Step 1: Get target dimension from metadata (Redis)
        target_dimension: Optional[int] = None
        target_model: Optional[str] = None
        target_provider: Optional[str] = None

        metadata = await self.metadata_manager.get_metadata(collection_name)

        # Step 1b: ALWAYS verify with Qdrant for coherence check
        qdrant_dimension = await self.client.get_vector_dimension_safe(collection_name)

        if metadata:
            target_dimension = metadata.vector_size
            target_model = metadata.embedding_model
            target_provider = metadata.embedding_provider

            # FIX-DIM-v4.1.2: Coherence check between Redis metadata and Qdrant
            if qdrant_dimension and qdrant_dimension != target_dimension:
                logger.error(
                    f"[COHERENCE-MISMATCH] Collection '{collection_name}': "
                    f"Redis metadata says dim={target_dimension}, but Qdrant has dim={qdrant_dimension}. "
                    f"This indicates metadata corruption. Attempting to heal..."
                )

                # Check if collection is empty - if so, we can heal
                try:
                    points_count = await self.client.count(collection_name)
                    if points_count == 0:
                        # Collection empty - heal metadata to match Qdrant
                        logger.warning(
                            f"[COHERENCE-HEAL] Collection '{collection_name}' is empty. "
                            f"Updating metadata to match Qdrant (dim={qdrant_dimension})."
                        )
                        # Infer model from Qdrant dimension
                        if qdrant_dimension in self._known_models and self._known_models[qdrant_dimension]:
                            healed_model = self._known_models[qdrant_dimension][0]["model"]
                            healed_provider = self._known_models[qdrant_dimension][0]["provider"]
                        else:
                            healed_model = env_model if env_dimension == qdrant_dimension else "unknown"
                            healed_provider = env_provider if env_dimension == qdrant_dimension else "unknown"

                        from datetime import datetime, timezone
                        await self.metadata_manager.save_metadata(
                            collection_name=collection_name,
                            vector_size=qdrant_dimension,
                            distance_metric="Cosine",
                            embedding_model=healed_model,
                            embedding_provider=healed_provider,
                            custom_metadata={
                                "coherence_healed": True,
                                "healed_at": datetime.now(timezone.utc).isoformat(),
                                "old_dimension": target_dimension,
                                "old_model": target_model,
                            },
                        )
                        self._compat_cache.pop(collection_name, None)  # v6.3.2: Invalidate after healing
                        self._matryoshka_truncation.pop(collection_name, None)
                        # Update target to healed values
                        target_dimension = qdrant_dimension
                        target_model = healed_model
                        target_provider = healed_provider
                        logger.info(
                            f"[COHERENCE-HEAL] Metadata healed for '{collection_name}': "
                            f"dim={qdrant_dimension}, model={healed_model}"
                        )
                    else:
                        # Collection has data - cannot auto-heal, use Qdrant as truth
                        logger.warning(
                            f"[COHERENCE-WARN] Collection '{collection_name}' has {points_count} points. "
                            f"Using Qdrant dimension ({qdrant_dimension}) as truth. "
                            f"Metadata says {target_dimension} - this may cause issues!"
                        )
                        target_dimension = qdrant_dimension
                except Exception as count_err:
                    logger.warning(f"[COHERENCE] Could not get point count: {count_err}")
                    # Fall back to Qdrant dimension
                    target_dimension = qdrant_dimension

            # Log coherence check result
            coherence_status = "OK" if (not qdrant_dimension or qdrant_dimension == metadata.vector_size) else "HEALED"
            logger.info(
                f"[COHERENCE-CHECK] Collection '{collection_name}': "
                f"metadata_dim={metadata.vector_size}, qdrant_dim={qdrant_dimension}, "
                f"env_dim={env_dimension}, status={coherence_status}"
            )
        else:
            # Step 1c: No metadata - query Qdrant directly (legacy collections)
            logger.warning(
                f"No metadata in Redis for '{collection_name}'. "
                f"Querying Qdrant for dimension (legacy mode)."
            )
            target_dimension = qdrant_dimension
            if target_dimension:
                # Infer the most likely model based on dimension
                inferred_model = "unknown"
                inferred_provider = "unknown"
                if (
                    target_dimension in self._known_models
                    and self._known_models[target_dimension]
                ):
                    # Use first model in the list as the most common one
                    inferred_model = self._known_models[target_dimension][0]["model"]
                    inferred_provider = self._known_models[target_dimension][0][
                        "provider"
                    ]

                logger.info(
                    f"[LEGACY] Collection '{collection_name}' has dimension {target_dimension} "
                    f"(from Qdrant). Inferred model: {inferred_model}"
                )

                # HEALING: Persist discovered metadata to Redis for future operations
                try:
                    from datetime import datetime, timezone

                    await self.metadata_manager.save_metadata(
                        collection_name=collection_name,
                        vector_size=target_dimension,
                        distance_metric="Cosine",
                        embedding_model=inferred_model,
                        embedding_provider=inferred_provider,
                        custom_metadata={
                            "healed": True,
                            "healed_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    self._compat_cache.pop(collection_name, None)  # v6.3.2: Invalidate after healing
                    self._matryoshka_truncation.pop(collection_name, None)
                    logger.warning(
                        f"[HEALING] Metadata for '{collection_name}' persisted to Redis: "
                        f"dim={target_dimension}, model={inferred_model}"
                    )
                    target_model = inferred_model
                    target_provider = inferred_provider
                except Exception as heal_err:
                    logger.error(
                        f"[HEALING] Failed to persist metadata for '{collection_name}': {heal_err}"
                    )
            else:
                logger.warning(
                    f"Could not determine dimension for '{collection_name}'. "
                    f"Collection may not exist yet. Using default model."
                )
                return  # Collection doesn't exist, will be created with current model

        # Step 2: Check current embedding manager compatibility
        current_dimension = self.embedding_manager.dimension
        current_model = self.config.get("embedding", {}).get("model", "unknown")

        # Log the compatibility check
        logger.info(
            f"[COMPAT-CHECK] Collection '{collection_name}': "
            f"target_dim={target_dimension}, target_model={target_model}, "
            f"current_dim={current_dimension}, current_model={current_model}"
        )

        if current_dimension == target_dimension:
            # Dimensions match
            if target_model and target_model != current_model:
                logger.warning(
                    f"[COMPAT] Dimension match ({current_dimension}), but models differ: "
                    f"collection uses '{target_model}', current is '{current_model}'. "
                    f"Proceeding (may affect search quality)."
                )
            else:
                logger.info(
                    f"[COMPAT-OK] Embedding compatible: dim={current_dimension}, model={current_model}"
                )
            # v6.3.2: Cache successful compatibility result
            self._matryoshka_truncation[collection_name] = None
            self._compat_cache[collection_name] = (_time.time(), current_dimension)
            return None  # Compatible

        # FIX-MATRYOSHKA-001: Matryoshka model, different dims → COMPATIBLE
        # Matryoshka models produce native_dim natively but are truncated at query time.
        # If current model is Matryoshka and both dims ≤ native_dim → no switch needed.
        # Works even when target_model is "unknown" (no Redis metadata).
        _native = self._matryoshka_models.get(current_model)
        if _native and current_dimension <= _native and (target_dimension or 0) <= _native:
            logger.info(
                f"[COMPAT-MATRYOSHKA] Model '{current_model}' is Matryoshka-capable — "
                f"current_dim={current_dimension}, collection_dim={target_dimension}, "
                f"native={_native}. No switch needed."
            )
            # FIX-MATRYOSHKA-002: Store truncation target and communicate it to callers.
            # When collection_dim < model_native_dim the query vector must be truncated
            # before sending to Qdrant — this is the truncation that was previously missing.
            _trunc_dim = target_dimension if (target_dimension and target_dimension < current_dimension) else None
            self._matryoshka_truncation[collection_name] = _trunc_dim
            self._compat_cache[collection_name] = (_time.time(), current_dimension)
            if _trunc_dim:
                logger.info(
                    f"[COMPAT-MATRYOSHKA] Truncation target stored: {_trunc_dim}d "
                    f"(native={_native}d) for collection '{collection_name}'"
                )
            return _trunc_dim

        # Step 3: Dimension mismatch - MUST switch model
        logger.error(
            f"[INCOMPATIBLE] Dimension mismatch: collection '{collection_name}' requires "
            f"{target_dimension} dims, but current model '{current_model}' produces {current_dimension} dims."
        )

        # Step 3a: Try to load the exact target model (if known)
        if target_model and target_provider:
            try:
                await self._switch_embedding_model(
                    target_model, target_provider, target_dimension
                )
                logger.info(
                    f"[SWITCH] Successfully switched to target model '{target_model}' "
                    f"(dimension: {target_dimension})"
                )
                self._matryoshka_truncation[collection_name] = None
                return None
            except Exception as e:
                logger.warning(
                    f"[SWITCH] Failed to load target model '{target_model}': {e}. "
                    f"Searching for compatible fallback..."
                )

        # Step 3b: Search for ANY model with matching dimension
        fallback_models = self._known_models.get(target_dimension, [])
        for fallback in fallback_models:
            try:
                await self._switch_embedding_model(
                    fallback["model"], fallback["provider"], target_dimension
                )
                logger.warning(
                    f"[FALLBACK] Switched to backup model '{fallback['model']}' "
                    f"(same dimension: {target_dimension}). "
                    f"Primary model was unavailable."
                )
                self._matryoshka_truncation[collection_name] = None
                return None
            except Exception as e:
                logger.debug(f"Fallback model '{fallback['model']}' not available: {e}")
                continue

        # Step 4: FAIL CLOSED - No compatible model found
        error_msg = (
            f"FATAL: No compatible embedding model found for collection '{collection_name}'. "
            f"Required dimension: {target_dimension}. "
            f"Current model '{current_model}' produces: {current_dimension}. "
            f"Available models for dimension {target_dimension}: {[m['model'] for m in fallback_models]}. "
            f"Please install a compatible model or recreate the collection."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    async def _switch_embedding_model(
        self, model_name: str, provider: str, expected_dimension: int
    ) -> None:
        """
        Switch embedding manager to a different model.

        Args:
            model_name: Model to load
            provider: Provider type (sentence-transformers, openai, etc.)
            expected_dimension: Expected output dimension

        Raises:
            RuntimeError: If model cannot be loaded or dimension doesn't match
        """
        from .embeddings import EmbeddingManager, EmbeddingConfig

        logger.info(
            f"[SWITCH] Loading embedding model '{model_name}' (provider: {provider})"
        )

        # Create new embedding config (propagate device from module config)
        new_config = EmbeddingConfig(
            provider=provider,
            model=model_name,
            dimension=expected_dimension,
            batch_size=self.config.get("embedding", {}).get("batch_size", 32),
            device=self.config.get("embedding", {}).get("device", "auto"),
        )

        # Create and initialize new manager
        new_manager = EmbeddingManager(new_config)
        await new_manager.initialize()

        # Verify dimension
        if new_manager.dimension != expected_dimension:
            await new_manager.shutdown()
            raise RuntimeError(
                f"Model '{model_name}' produces dimension {new_manager.dimension}, "
                f"expected {expected_dimension}"
            )

        # Verify model can actually embed (not just health check)
        try:
            test_result = await new_manager.embed("test", use_cache=False, is_query=False)
            if len(test_result) != expected_dimension:
                raise RuntimeError(
                    f"Test embed dimension mismatch: got {len(test_result)}, expected {expected_dimension}"
                )
        except Exception as e:
            await new_manager.shutdown()
            raise RuntimeError(
                f"Model '{model_name}' ({provider}) initialized but failed test embed: {e}"
            ) from e

        # Shutdown old manager and replace
        old_manager = self.embedding_manager
        self.embedding_manager = new_manager

        try:
            await old_manager.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down old embedding manager: {e}")

        logger.info(
            f"[SWITCH] Embedding model switched: {model_name} "
            f"(dimension: {new_manager.dimension})"
        )

    # =========================================================================
    # Collection Operations
    # =========================================================================

    async def create_collection(
        self,
        collection_name: str,
        vector_size: Optional[int] = None,
        distance: Optional[str] = None,
    ) -> OperationResult:
        """Create a new collection."""
        start_time = time.time()

        try:
            OperationValidator.validate_collection_name(collection_name)

            if vector_size is None:
                vector_size = self.embedding_manager.dimension
            else:
                OperationValidator.validate_vector_size(vector_size)

            if distance is None:
                distance = self.config.get("collection", {}).get("distance", "Cosine")

            await self.client.create_collection(
                collection_name=collection_name,
                vector_size=vector_size,
                distance=distance,
            )

            # Save collection metadata for auto-configuration
            embedding_config = self.config.get("embedding", {})
            chunking_config = self.config.get("chunking", {})

            await self.metadata_manager.save_metadata(
                collection_name=collection_name,
                vector_size=vector_size,
                distance_metric=distance,
                embedding_model=embedding_config.get("model", "unknown"),
                embedding_provider=embedding_config.get("provider", "unknown"),
                chunking_config=chunking_config
                if chunking_config.get("enabled")
                else None,
            )

            duration = (time.time() - start_time) * 1000

            logger.info(
                f"Created collection '{collection_name}' with metadata",
                extra={
                    "vector_size": vector_size,
                    "distance": distance,
                    "model": embedding_config.get("model"),
                },
            )

            return OperationResult(
                status=OperationStatus.SUCCESS,
                operation="create_collection",
                data={
                    "collection_name": collection_name,
                    "vector_size": vector_size,
                    "distance": distance,
                    "embedding_model": embedding_config.get("model"),
                    "embedding_provider": embedding_config.get("provider"),
                },
                duration_ms=duration,
            )

        except ValidationError as e:
            return OperationResult(
                status=OperationStatus.FAILED,
                operation="create_collection",
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return OperationResult(
                status=OperationStatus.FAILED,
                operation="create_collection",
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def delete_collection(self, collection_name: str) -> OperationResult:
        """Delete a collection and ALL associated data (Qdrant + Redis).

        FIX-DELETE-SYNC-001: Complete cleanup to prevent zombie keys.

        Cleans up:
        1. Qdrant collection (vectors + payloads)
        2. Collection metadata: rag:collection:metadata:{collection_name}
        3. Collection index: rag:collections:index (SREM)
        4. Document registry: ubp:rag:documents:{collection_name} (HASH)
        """
        start_time = time.time()

        try:
            OperationValidator.validate_collection_name(collection_name)

            # Track cleanup status
            cleanup_status = {
                "qdrant_deleted": False,
                "metadata_cleaned": False,
                "doc_registry_cleaned": False,
            }

            # 1. Delete from Qdrant
            await self.client.delete_collection(collection_name)
            cleanup_status["qdrant_deleted"] = True

            # 2. Delete collection metadata (JSON + index set)
            await self.metadata_manager.delete_metadata(collection_name)
            cleanup_status["metadata_cleaned"] = True

            # 3. Delete document registry hash
            if self._redis_client:
                try:
                    doc_registry_key = f"ubp:rag:documents:{collection_name}"
                    deleted_count = await self._redis_client.delete(doc_registry_key)
                    cleanup_status["doc_registry_cleaned"] = deleted_count > 0
                    if cleanup_status["doc_registry_cleaned"]:
                        logger.debug(f"Document registry cleaned: {doc_registry_key}")
                except Exception as e:
                    logger.warning(
                        f"Failed to clean document registry: {e}",
                        extra={"collection": collection_name},
                    )

            duration = (time.time() - start_time) * 1000

            logger.info(
                f"Deleted collection '{collection_name}' with full cleanup",
                extra={"cleanup": cleanup_status},
            )

            return OperationResult(
                status=OperationStatus.SUCCESS,
                operation="delete_collection",
                data={
                    "collection_name": collection_name,
                    "deleted": True,
                    "cleanup": cleanup_status,
                },
                duration_ms=duration,
            )

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return OperationResult(
                status=OperationStatus.FAILED,
                operation="delete_collection",
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def list_collections(self) -> OperationResult:
        """List all collections."""
        start_time = time.time()

        try:
            collections = await self.client.list_collections()

            duration = (time.time() - start_time) * 1000

            return OperationResult(
                status=OperationStatus.SUCCESS,
                operation="list_collections",
                data={"collections": collections, "count": len(collections)},
                duration_ms=duration,
            )

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return OperationResult(
                status=OperationStatus.FAILED,
                operation="list_collections",
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def get_collection_info(self, collection_name: str) -> OperationResult:
        """Get collection information with metadata."""
        start_time = time.time()

        try:
            OperationValidator.validate_collection_name(collection_name)

            # Get Qdrant collection info
            info = await self.client.get_collection_info(collection_name)

            # Retrieve saved metadata
            metadata = await self.metadata_manager.get_metadata(collection_name)

            # Merge Qdrant info with metadata
            if metadata:
                info["metadata"] = {
                    "embedding_model": metadata.embedding_model,
                    "embedding_provider": metadata.embedding_provider,
                    "created_at": metadata.created_at,
                    "updated_at": metadata.updated_at,
                    "chunking_config": metadata.chunking_config,
                    "custom_metadata": metadata.custom_metadata,
                }
            else:
                info["metadata"] = None
                logger.warning(
                    f"No metadata found for collection '{collection_name}'. "
                    "This collection may have been created externally or before metadata tracking was enabled."
                )

            duration = (time.time() - start_time) * 1000

            return OperationResult(
                status=OperationStatus.SUCCESS,
                operation="get_collection_info",
                data=info,
                duration_ms=duration,
            )

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return OperationResult(
                status=OperationStatus.FAILED,
                operation="get_collection_info",
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    # =========================================================================
    # Document Operations
    # =========================================================================

    async def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
        chunk_document: bool = True,
    ) -> OperationResult:
        """
        Add a document to the collection.

        Args:
            doc_id: Document identifier
            text: Document text content
            metadata: Optional metadata
            collection: Target collection
            chunk_document: Whether to chunk the document

        Returns:
            OperationResult with point IDs
        """
        start_time = time.time()
        collection = collection or self.default_collection

        try:
            # Validate inputs
            OperationValidator.validate_doc_id(doc_id)
            OperationValidator.validate_text(text)
            OperationValidator.validate_collection_name(collection)

            # CRITICAL: Ensure embedding model is compatible with target collection
            # This implements the METADATA-DRIVEN EMBEDDING protocol
            await self._ensure_compatible_embedding(collection)

            # Ensure collection exists with correct vector dimension
            if not await self.client.collection_exists(collection):
                # Create new collection with metadata
                distance_metric = self.config.get("collection", {}).get(
                    "distance", "Cosine"
                )
                await self.client.create_collection(
                    collection, self.embedding_manager.dimension, distance_metric
                )

                # Save collection metadata
                # FIX-CHUNK-DOC-003 (v1.8.5): Document-level chunking priority
                # Priority: document metadata > module config > defaults
                embedding_config = self.config.get("embedding", {})
                
                # Check if document provides custom chunking_config
                doc_chunking_config = metadata.get("chunking_config") if metadata else None
                module_chunking_config = self.config.get("chunking", {})
                
                # Use document-level config if provided, else module defaults
                effective_chunking_config = doc_chunking_config or module_chunking_config
                chunking_source = "document" if doc_chunking_config else "module"
                
                await self.metadata_manager.save_metadata(
                    collection_name=collection,
                    vector_size=self.embedding_manager.dimension,
                    distance_metric=distance_metric,
                    embedding_model=embedding_config.get("model", "unknown"),
                    embedding_provider=embedding_config.get("provider", "unknown"),
                    chunking_config=effective_chunking_config
                    if effective_chunking_config.get("enabled", True)
                    else None,
                )

                logger.info(
                    f"Created collection '{collection}' with dimension {self.embedding_manager.dimension}. "
                    f"Chunking config from {chunking_source}: chunk_size={effective_chunking_config.get('chunk_size', 500)}"
                )
            else:
                # Collection exists - verify dimension and check metadata
                try:
                    collection_info = await self.client.get_collection_info(collection)
                    # FIX-DIM-v4.1.2: vector_size is nested inside config
                    existing_dim = collection_info.get("config", {}).get("vector_size")

                    # CRITICAL: Dimension mismatch check with auto-recreate for empty collections
                    if (
                        existing_dim
                        and existing_dim != self.embedding_manager.dimension
                    ):
                        # Check if collection is empty - if so, we can safely recreate it
                        points_count = collection_info.get("points_count", 0)

                        if points_count == 0:
                            # FIX-DIM-MIGRATE-001: Empty collection with dimension mismatch
                            # Safe to delete and recreate with new embedding model dimensions
                            logger.warning(
                                f"[DIM-MIGRATE] Collection '{collection}' is EMPTY with dimension {existing_dim}, "
                                f"but embedding model requires {self.embedding_manager.dimension}. "
                                f"Recreating collection with correct dimensions..."
                            )

                            # Delete the old collection
                            try:
                                await self.client.delete_collection(collection)
                                logger.info(f"[DIM-MIGRATE] Deleted empty collection '{collection}'")
                            except Exception as del_err:
                                logger.error(f"[DIM-MIGRATE] Failed to delete collection: {del_err}")
                                # Continue anyway, create_collection might overwrite

                            # Delete old Redis metadata
                            try:
                                await self.metadata_manager.delete_metadata(collection)
                                logger.info(f"[DIM-MIGRATE] Deleted old metadata for '{collection}'")
                            except Exception as meta_err:
                                logger.warning(f"[DIM-MIGRATE] Could not delete old metadata: {meta_err}")

                            # Create new collection with correct dimensions
                            embedding_config = self.config.get("embedding", {})
                            distance_metric = self.config.get("collection", {}).get("distance", "Cosine")

                            await self.client.create_collection(
                                name=collection,
                                vector_size=self.embedding_manager.dimension,
                                distance=distance_metric,
                            )

                            # Save new metadata
                            await self.metadata_manager.save_metadata(
                                collection_name=collection,
                                vector_size=self.embedding_manager.dimension,
                                distance_metric=distance_metric,
                                embedding_model=embedding_config.get("model", "unknown"),
                                embedding_provider=embedding_config.get("provider", "unknown"),
                            )

                            logger.info(
                                f"[DIM-MIGRATE] Successfully recreated collection '{collection}' "
                                f"with dimension {self.embedding_manager.dimension} "
                                f"(model: {embedding_config.get('model', 'unknown')})"
                            )
                        else:
                            # Collection has data - cannot auto-migrate
                            error_msg = (
                                f"Collection '{collection}' exists with dimension {existing_dim} "
                                f"and contains {points_count} points, "
                                f"but embedding model generates dimension {self.embedding_manager.dimension}. "
                                f"To migrate: 1) Export data, 2) Delete collection, 3) Re-ingest with new model. "
                                f"Or use a different collection name."
                            )
                            logger.error(error_msg)
                            return OperationResult(
                                status=OperationStatus.FAILED,
                                operation="add_document",
                                error=error_msg,
                                duration_ms=(time.time() - start_time) * 1000,
                            )

                    # Retrieve collection metadata for auto-configuration info
                    coll_metadata = await self.metadata_manager.get_metadata(collection)
                    if coll_metadata:
                        current_model = self.config.get("embedding", {}).get(
                            "model", "unknown"
                        )
                        if coll_metadata.embedding_model != current_model:
                            logger.warning(
                                f"Collection '{collection}' was created with model '{coll_metadata.embedding_model}', "
                                f"but current config uses '{current_model}'. "
                                f"Dimensions match ({existing_dim}), so indexing will proceed, "
                                f"but results may be suboptimal if models are incompatible."
                            )
                    else:
                        logger.info(
                            f"No metadata found for collection '{collection}'. "
                            f"Collection may have been created externally or before metadata tracking."
                        )

                except Exception as e:
                    # If can't get collection info, log warning but continue
                    logger.warning(f"Could not verify collection dimension: {e}")

            # Prepare metadata
            base_metadata = metadata or {}
            base_metadata["doc_id"] = doc_id
            base_metadata["indexed_at"] = datetime.now(timezone.utc).isoformat()

            # ================================================================
            # FIX-CHUNK-DOC-003 (v1.8.5): Extended Chunking Priority
            # ================================================================
            # Priority: document metadata > collection metadata > module config > defaults
            # This ensures per-document chunk_size (e.g., 2000 from ingest script) 
            # takes precedence over collection defaults

            # Get collection metadata (already retrieved earlier for embedding check)
            coll_metadata = await self.metadata_manager.get_metadata(collection)

            # Determine effective chunking config with 4-level priority
            chunking_source = "defaults"
            effective_chunking_config = {}
            
            # Level 1 (Highest): Document-level chunking_config
            doc_chunking_config = base_metadata.get("chunking_config")
            if doc_chunking_config and isinstance(doc_chunking_config, dict):
                effective_chunking_config = doc_chunking_config.copy()
                chunking_source = "document"
                logger.debug(
                    f"[FIX-CHUNK-DOC-003] Using document-level chunking: {doc_chunking_config}"
                )
            # Level 2: Collection-specific chunking config (from Redis metadata)
            elif coll_metadata and coll_metadata.chunking_config:
                effective_chunking_config = coll_metadata.chunking_config.copy() if isinstance(coll_metadata.chunking_config, dict) else {}
                chunking_source = "collection"
                logger.debug(
                    f"[FIX-CHUNK-DOC-003] Using collection-specific chunking: {effective_chunking_config}"
                )
            # Level 3: Module config (from config.json)
            else:
                effective_chunking_config = self.config.get("chunking", {}).copy()
                chunking_source = "module"
                logger.debug(
                    f"[FIX-CHUNK-DOC-003] Using module default chunking: {effective_chunking_config}"
                )
            
            # Level 4 (Lowest): Apply hardcoded defaults for any missing values
            chunk_size = effective_chunking_config.get("chunk_size", 500)
            chunk_overlap = effective_chunking_config.get("chunk_overlap", 50)
            
            # v6.4.1: Bounds validation — use ChunkingConfig constants for single-source-of-truth
            MIN_CHUNK_SIZE = 50
            MAX_CHUNK_SIZE = 4000
            if chunk_size < MIN_CHUNK_SIZE:
                logger.warning(f"[FIX-CHUNK-DOC-003] chunk_size={chunk_size} too small, using minimum {MIN_CHUNK_SIZE}")
                chunk_size = MIN_CHUNK_SIZE
            elif chunk_size > MAX_CHUNK_SIZE:
                logger.warning(f"[FIX-CHUNK-DOC-003] chunk_size={chunk_size} too large, capping at {MAX_CHUNK_SIZE}")
                chunk_size = MAX_CHUNK_SIZE
            max_chunk_size = effective_chunking_config.get("max_chunk_size", chunk_size * 4)
            
            logger.info(
                f"[FIX-CHUNK-DOC-003] Chunking source: {chunking_source}, "
                f"chunk_size={chunk_size}, overlap={chunk_overlap}"
            )

            # Create a collection-specific ChunkingManager if config differs from default
            from .chunker import ChunkingManager, ChunkingConfig

            default_chunk_size = self.config.get("chunking", {}).get("chunk_size", 500)
            default_chunk_overlap = self.config.get("chunking", {}).get(
                "chunk_overlap", 50
            )

            if (
                chunk_size != default_chunk_size
                or chunk_overlap != default_chunk_overlap
            ):
                # v6.4.0: Cache ChunkingManagers per config fingerprint
                _strategy = effective_chunking_config.get("split_by", "sentence")
                _cache_key = (chunk_size, chunk_overlap, _strategy)
                collection_chunker = self._chunker_cache.get(_cache_key)
                if collection_chunker is None:
                    collection_chunking_config = ChunkingConfig(
                        strategy=_strategy,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        min_chunk_size=effective_chunking_config.get("min_chunk_size", 50),
                        max_chunk_size=effective_chunking_config.get(
                            "max_chunk_size", chunk_size * 4
                        ),
                        include_metadata=True,
                    )
                    collection_chunker = ChunkingManager(collection_chunking_config)
                    self._chunker_cache[_cache_key] = collection_chunker
                    logger.info(
                        f"[FIX-CHUNK-DOC-003] Created custom chunker for '{collection}' (source={chunking_source}): "
                        f"chunk_size={chunk_size}, overlap={chunk_overlap}"
                    )
            else:
                # Use the shared chunking manager
                collection_chunker = self.chunking_manager

            if chunk_document and len(text) > chunk_size:
                chunks = collection_chunker.chunk_with_overlap_ids(
                    text, doc_id, metadata=base_metadata
                )
            elif len(text) > max_chunk_size:
                # v6.4.1: Single doc exceeds max_chunk_size — force chunk even if < chunk_size
                chunks = collection_chunker.chunk_with_overlap_ids(
                    text, doc_id, metadata=base_metadata
                )
                logger.debug(f"[INGEST] Force-chunked single doc ({len(text)} chars > max_chunk_size={max_chunk_size})")
            else:
                # Single chunk
                from .chunker import Chunk

                chunks = [
                    Chunk(
                        text=text,
                        index=0,
                        start_char=0,
                        end_char=len(text),
                        metadata=base_metadata,
                    )
                ]

            # Generate embeddings for documents (passage/document prefix for BGE/E5)
            texts = [chunk.text for chunk in chunks]
            embeddings = await self.embedding_manager.embed_batch(texts, is_query=False)

            # Prepare points for upsert
            points = []
            point_ids = []

            # FIX-CHUNK-META-001: Get the ACTUAL strategy used from collection config
            # This ensures chunk_strategy reflects what was really used, not "default"
            used_strategy = effective_chunking_config.get("split_by", "sentence")

            skipped_chunks = 0
            for chunk, embedding in zip(chunks, embeddings):
                # Skip chunks with failed embeddings (None)
                if embedding is None:
                    skipped_chunks += 1
                    logger.warning(f"Skipping chunk {chunk.index} due to failed embedding")
                    continue

                # Generate deterministic UUID for Qdrant point ID
                point_id = generate_point_id(doc_id, chunk.index)
                point_ids.append(point_id)

                payload = {
                    "text": chunk.text,
                    "doc_id": doc_id,
                    "chunk_index": chunk.index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "chunk_strategy": used_strategy,  # FIX: Real strategy from collection
                    "chunk_size_used": chunk_size,  # FIX: Actual chunk_size used
                    **chunk.metadata,
                }

                points.append({"id": point_id, "vector": embedding, "payload": payload})

            if skipped_chunks > 0:
                logger.warning(
                    f"⚠️ Document {doc_id}: {skipped_chunks} chunks skipped due to embedding failures. "
                    f"Proceeding with {len(points)} successful chunks."
                )

            # Guard: no points means all embeddings failed - abort before Qdrant 400
            if len(points) == 0:
                raise RuntimeError(
                    f"All {len(chunks)} chunks failed embedding for document '{doc_id}' "
                    f"in collection '{collection}'. Provider: {self.embedding_manager.config.provider}, "
                    f"Model: {self.embedding_manager.config.model}. "
                    f"Check that the embedding model is available and functional."
                )

            # FIX GAP-UPSERT-001: Batch upserts to avoid Qdrant 32MB payload limit
            # Qdrant default limit is 33554432 bytes (~32MB).
            # Each point is ~5-6KB JSON (384-dim vector + text payload).
            # Safe batch size: ~1000 points per batch (~5-6MB per batch)
            UPSERT_BATCH_SIZE = 1000

            if len(points) <= UPSERT_BATCH_SIZE:
                # Small document - single upsert
                await self.client.upsert(collection, points)
            else:
                # Large document - batch upserts
                total_batches = (
                    len(points) + UPSERT_BATCH_SIZE - 1
                ) // UPSERT_BATCH_SIZE
                logger.info(
                    f"[GAP-UPSERT-001] Large document: {len(points)} points in {total_batches} batches"
                )

                for batch_idx in range(0, len(points), UPSERT_BATCH_SIZE):
                    batch = points[batch_idx : batch_idx + UPSERT_BATCH_SIZE]
                    batch_num = (batch_idx // UPSERT_BATCH_SIZE) + 1

                    logger.debug(
                        f"[GAP-UPSERT-001] Upserting batch {batch_num}/{total_batches} ({len(batch)} points)"
                    )
                    await self.client.upsert(collection, batch)

            duration = (time.time() - start_time) * 1000

            # Update stats
            self._stats["documents_added"] += 1
            self._stats["chunks_created"] += len(chunks)

            # GAP-INGEST-001: Informative message for large documents
            text_length = len(text)
            is_large_document = text_length > 100000  # 100k chars threshold for "large"

            if is_large_document:
                message = f"Large document processed: {text_length:,} chars split into {len(chunks)} chunks"
                logger.info(
                    f"✅ Large document '{doc_id}' ingested: {text_length:,} chars -> {len(chunks)} chunks",
                    extra={
                        "collection": collection,
                        "chunks": len(chunks),
                        "text_length": text_length,
                        "large_document": True,
                    },
                )
            else:
                message = f"Document indexed with {len(chunks)} chunks"
                logger.info(
                    f"Added document '{doc_id}' with {len(chunks)} chunks",
                    extra={"collection": collection, "chunks": len(chunks)},
                )

            return OperationResult(
                status=OperationStatus.SUCCESS,
                operation="add_document",
                data={
                    "doc_id": doc_id,
                    "point_ids": point_ids,
                    "chunks_count": len(chunks),
                    "collection": collection,
                    "text_length": text_length,
                    "large_document": is_large_document,
                    "message": message,
                    # GAP-CHUNKING-001: Include chunking config used for transparency
                    "chunk_size_used": chunk_size,
                    "chunk_overlap_used": chunk_overlap,
                },
                duration_ms=duration,
            )

        except ValidationError as e:
            return OperationResult(
                status=OperationStatus.FAILED,
                operation="add_document",
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return OperationResult(
                status=OperationStatus.FAILED,
                operation="add_document",
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def add_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        collection: Optional[str] = None,
        chunk_documents: bool = True,
    ) -> OperationResult:
        """Add multiple documents in batch.

        FIX-BATCH-FAIL-FAST-001: Stops immediately on fatal errors like
        dimension mismatch (when collection has data and can't auto-migrate).
        """
        start_time = time.time()

        success_count = 0
        failed_count = 0
        failed_docs = []

        for idx, doc in enumerate(documents):
            try:
                result = await self.add_document(
                    doc_id=doc["doc_id"],
                    text=doc["text"],
                    metadata=doc.get("metadata"),
                    collection=collection,
                    chunk_document=chunk_documents,
                )

                if result.success:
                    success_count += 1
                else:
                    # FIX-BATCH-FAIL-FAST-001: Check for fatal errors that should stop batch
                    error_msg = result.error or ""
                    is_fatal_error = (
                        "dimension" in error_msg.lower() and "contains" in error_msg.lower()
                    ) or (
                        "dimension" in error_msg.lower() and "mismatch" in error_msg.lower()
                    ) or (
                        "FATAL" in error_msg
                    )

                    if is_fatal_error:
                        # Stop immediately - don't process remaining documents
                        logger.error(
                            f"[BATCH-FAIL-FAST] Fatal error on document {idx+1}/{len(documents)}: {error_msg}. "
                            f"Stopping batch processing. {success_count} docs succeeded before failure."
                        )
                        return OperationResult(
                            status=OperationStatus.FAILED,
                            operation="add_documents_batch",
                            data={
                                "total": len(documents),
                                "processed": idx + 1,
                                "success": success_count,
                                "failed": 1,
                                "stopped_at_doc": doc["doc_id"],
                                "remaining_unprocessed": len(documents) - idx - 1,
                                "fatal_error": error_msg,
                            },
                            error=f"Batch stopped: {error_msg}",
                            duration_ms=(time.time() - start_time) * 1000,
                        )

                    # Non-fatal error - continue processing
                    failed_count += 1
                    failed_docs.append({"doc_id": doc["doc_id"], "error": result.error})

            except Exception as e:
                error_str = str(e)
                # Also check exceptions for fatal errors
                is_fatal_exception = (
                    "dimension" in error_str.lower() and "contains" in error_str.lower()
                ) or (
                    "dimension" in error_str.lower() and "mismatch" in error_str.lower()
                )

                if is_fatal_exception:
                    logger.error(
                        f"[BATCH-FAIL-FAST] Fatal exception on document {idx+1}/{len(documents)}: {error_str}. "
                        f"Stopping batch processing."
                    )
                    return OperationResult(
                        status=OperationStatus.FAILED,
                        operation="add_documents_batch",
                        data={
                            "total": len(documents),
                            "processed": idx + 1,
                            "success": success_count,
                            "failed": 1,
                            "stopped_at_doc": doc.get("doc_id", "unknown"),
                            "remaining_unprocessed": len(documents) - idx - 1,
                            "fatal_error": error_str,
                        },
                        error=f"Batch stopped: {error_str}",
                        duration_ms=(time.time() - start_time) * 1000,
                    )

                failed_count += 1
                failed_docs.append(
                    {"doc_id": doc.get("doc_id", "unknown"), "error": error_str}
                )

        duration = (time.time() - start_time) * 1000

        status = (
            OperationStatus.SUCCESS
            if failed_count == 0
            else (
                OperationStatus.PARTIAL if success_count > 0 else OperationStatus.FAILED
            )
        )

        return OperationResult(
            status=status,
            operation="add_documents_batch",
            data={
                "total": len(documents),
                "success": success_count,
                "failed": failed_count,
                "failed_docs": failed_docs if failed_docs else None,
            },
            duration_ms=duration,
        )

    async def delete_document(
        self, doc_id: str, collection: Optional[str] = None
    ) -> OperationResult:
        """Delete a document and all its chunks.

        TASK #82: Fixed to use scroll instead of search to avoid dimension mismatch.
        Scroll doesn't require a query vector, so it works with any collection dimension.

        TASK #83: Enhanced to handle legacy documents without doc_id in payload.
        - Step 1: Try filter by payload.doc_id
        - Step 2: Try filter by payload.document_id
        - Step 3: Try direct point ID deletion (for UUID doc_ids)
        - Step 4: Try filter by filename (for "doc_*" generated IDs)

        TASK #84: Added comprehensive logging for debugging deletion issues.

        TASK #85: Refactored for clarity and accuracy:
        - Return NOT_FOUND status when document doesn't exist (idempotent)
        - Only increment stats when chunks are actually deleted
        - Cleaner step separation without deep nesting
        """
        start_time = time.time()
        collection = collection or self.default_collection
        strategy_used = "none"

        logger.info(
            f"🗑️ DELETE_DOCUMENT START: doc_id='{doc_id}', collection='{collection}'",
            extra={"doc_id": doc_id, "collection": collection},
        )

        try:
            OperationValidator.validate_doc_id(doc_id)

            point_ids: List[str] = []

            # Step 1: Try filter by payload.doc_id
            if not point_ids:
                logger.debug(f"  Step 1: Trying filter by payload.doc_id='{doc_id}'")
                point_ids = await self._find_points_by_filter(
                    collection, {"doc_id": doc_id}
                )
                if point_ids:
                    strategy_used = "doc_id_filter"
                    logger.debug(f"  Step 1: Found {len(point_ids)} points")

            # Step 2: Try filter by payload.document_id
            if not point_ids:
                logger.debug(f"  Step 2: Trying filter by payload.document_id='{doc_id}'")
                point_ids = await self._find_points_by_filter(
                    collection, {"document_id": doc_id}
                )
                if point_ids:
                    strategy_used = "document_id_filter"
                    logger.debug(f"  Step 2: Found {len(point_ids)} points")

            # Step 3: Try direct point ID deletion (if doc_id is a valid UUID)
            if not point_ids:
                logger.debug(f"  Step 3: Checking if doc_id is a UUID for direct deletion")
                try:
                    uuid.UUID(doc_id)  # Validate UUID format
                    point_ids = [doc_id]
                    strategy_used = "direct_uuid"
                    logger.info(f"  Step 3: Using direct point ID deletion for UUID '{doc_id}'")
                except ValueError:
                    logger.debug(f"  Step 3: doc_id '{doc_id}' is not a valid UUID")

            # Step 4: Try filter by filename (for "doc_*" generated IDs)
            if not point_ids and doc_id.startswith("doc_"):
                filename = doc_id[4:]  # Remove "doc_" prefix
                logger.debug(f"  Step 4: Trying filter by payload.filename='{filename}'")
                point_ids = await self._find_points_by_filter(
                    collection, {"filename": filename}
                )
                if point_ids:
                    strategy_used = "filename_filter"
                    logger.debug(f"  Step 4: Found {len(point_ids)} points")

            # Execute deletion or return NOT_FOUND
            duration = (time.time() - start_time) * 1000

            if point_ids:
                logger.info(
                    f"  📋 Found {len(point_ids)} points to delete using strategy '{strategy_used}'",
                    extra={
                        "doc_id": doc_id,
                        "collection": collection,
                        "points_count": len(point_ids),
                        "strategy": strategy_used,
                        "sample_point_ids": point_ids[:5],
                    },
                )
                await self.client.delete_points(collection, point_ids)
                logger.info(f"  ✅ delete_points() called successfully for {len(point_ids)} points")

                # Only increment stats when chunks are actually deleted
                self._stats["documents_deleted"] += 1

                logger.info(
                    f"🗑️ DELETE_DOCUMENT SUCCESS: doc_id='{doc_id}', chunks_deleted={len(point_ids)}, "
                    f"strategy='{strategy_used}', duration={duration:.1f}ms",
                    extra={
                        "doc_id": doc_id,
                        "collection": collection,
                        "chunks_deleted": len(point_ids),
                        "strategy": strategy_used,
                        "duration_ms": duration,
                    },
                )

                return OperationResult(
                    status=OperationStatus.SUCCESS,
                    operation="delete_document",
                    data={
                        "doc_id": doc_id,
                        "chunks_deleted": len(point_ids),
                        "collection": collection,
                        "strategy_used": strategy_used,
                        "found": True,
                    },
                    duration_ms=duration,
                )
            else:
                # Document not found - return NOT_FOUND (idempotent, not an error)
                logger.warning(
                    f"  ⚠️ NO POINTS FOUND for doc_id='{doc_id}' in collection='{collection}'. "
                    f"All 4 strategies failed. Document may not exist or has different ID format.",
                    extra={
                        "doc_id": doc_id,
                        "collection": collection,
                        "strategies_tried": ["doc_id_filter", "document_id_filter", "direct_uuid", "filename_filter"],
                    },
                )

                logger.info(
                    f"🗑️ DELETE_DOCUMENT NOT_FOUND: doc_id='{doc_id}', duration={duration:.1f}ms",
                    extra={
                        "doc_id": doc_id,
                        "collection": collection,
                        "duration_ms": duration,
                    },
                )

                return OperationResult(
                    status=OperationStatus.NOT_FOUND,
                    operation="delete_document",
                    data={
                        "doc_id": doc_id,
                        "chunks_deleted": 0,
                        "collection": collection,
                        "strategy_used": strategy_used,
                        "found": False,
                        "message": "Document not found. It may have been already deleted or never existed.",
                    },
                    duration_ms=duration,
                )

        except Exception as e:
            import traceback
            duration = (time.time() - start_time) * 1000
            logger.error(
                f"🗑️ DELETE_DOCUMENT FAILED: doc_id='{doc_id}', error={e}",
                extra={
                    "doc_id": doc_id,
                    "collection": collection,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "duration_ms": duration,
                },
            )
            return OperationResult(
                status=OperationStatus.FAILED,
                operation="delete_document",
                error=str(e),
                duration_ms=duration,
            )

    async def _find_points_by_filter(
        self, collection: str, filter_dict: Dict[str, Any]
    ) -> List[str]:
        """Helper to find point IDs by filter condition.

        Args:
            collection: Collection name to search
            filter_dict: Filter conditions dict (e.g., {"doc_id": "abc123"})

        Returns:
            List of point IDs matching the filter
        """
        filter_conditions = FilterBuilder.build(filter_dict)
        scroll_result = await self.client.scroll(
            collection_name=collection,
            limit=10000,  # Increased limit to handle large documents
            offset=None,
            filter_conditions=filter_conditions,
            with_payload=False,
            with_vectors=False,
        )
        points = scroll_result.get("points", [])
        return [p.get("id") for p in points if p.get("id")]

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> QueryResult:
        """Query for similar documents."""
        start_time = time.time()
        collection = collection or self.default_collection
        top_k = top_k or self.default_top_k
        score_threshold = score_threshold or self.score_threshold

        try:
            OperationValidator.validate_text(query_text, min_length=1)
            OperationValidator.validate_top_k(top_k)
            if score_threshold is not None:
                OperationValidator.validate_score_threshold(score_threshold)

            # CRITICAL: Ensure embedding model is compatible with target collection
            # This implements the METADATA-DRIVEN EMBEDDING protocol for queries
            effective_dim = await self._ensure_compatible_embedding(collection)

            # Generate query embedding (query prefix for BGE/E5 models - critical for cross-lingual)
            query_embedding = await self.embedding_manager.embed(query_text, is_query=True)

            # FIX-MATRYOSHKA-002: Apply Matryoshka truncation when collection_dim < model_native_dim.
            # _ensure_compatible_embedding returns target_dim when truncation is needed (e.g. 384),
            # or None when dims already match or model was switched. This is the missing truncation
            # that caused "Vector dimension error: expected dim: 384, got 1024".
            if effective_dim is not None and effective_dim < len(query_embedding):
                logger.debug(
                    f"[MATRYOSHKA-TRUNC] Query vector truncated: {len(query_embedding)}d → {effective_dim}d "
                    f"for collection '{collection}'"
                )
                query_embedding = query_embedding[:effective_dim]

            # Build filter
            qdrant_filter = FilterBuilder.build(filter_conditions)

            # Search
            raw_results = await self.client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                filter_conditions=qdrant_filter,
                with_payload=self.with_payload,
                with_vectors=self.with_vectors,
            )

            # Convert to SearchResult objects
            results = []
            for r in raw_results:
                payload = r.get("payload", {})
                results.append(
                    SearchResult(
                        id=r["id"],
                        score=r["score"],
                        text=payload.get("text", ""),
                        metadata={k: v for k, v in payload.items() if k != "text"},
                        chunk_index=payload.get("chunk_index"),
                        doc_id=payload.get("doc_id"),
                        vector=r.get("vector"),  # v6.1.3: carry embedding for dedup/fusion
                    )
                )

            duration = (time.time() - start_time) * 1000

            self._stats["queries_executed"] += 1

            logger.debug(
                f"Query returned {len(results)} results",
                extra={
                    "collection": collection,
                    "top_k": top_k,
                    "duration_ms": duration,
                },
            )

            return QueryResult(
                query=query_text,
                results=results,
                total_results=len(results),
                duration_ms=duration,
                collection=collection,
                filters_applied=filter_conditions,
            )

        except CollectionNotFoundError as e:
            logger.warning(f"Query on missing collection '{e.collection_name}': {e}")
            return QueryResult(
                query=query_text,
                results=[],
                total_results=0,
                duration_ms=(time.time() - start_time) * 1000,
                collection=collection,
                filters_applied=filter_conditions,
                error_code="COLLECTION_NOT_FOUND",
                error_message=str(e),
            )

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return QueryResult(
                query=query_text,
                results=[],
                total_results=0,
                duration_ms=(time.time() - start_time) * 1000,
                collection=collection,
                filters_applied=filter_conditions,
            )

    async def query_with_context(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_neighbors: bool = True,
    ) -> Dict[str, Any]:
        """Query with expanded context (includes neighboring chunks)."""
        query_result = await self.query(
            query_text=query_text,
            top_k=top_k,
            collection=collection,
            filter_conditions=filter_conditions,
        )

        if not include_neighbors or not query_result.results:
            return query_result.to_dict()

        expanded_results = []
        seen_chunks = set()

        for result in query_result.results:
            chunk_id = result.metadata.get("chunk_id")
            if chunk_id:
                seen_chunks.add(chunk_id)

            context_text = result.text

            expanded_results.append({**result.to_dict(), "context": context_text})

        return {**query_result.to_dict(), "results": expanded_results}

    # =========================================================================
    # Metadata Migration
    # =========================================================================

    async def migrate_existing_collections_metadata(self) -> OperationResult:
        """
        Migrate metadata for existing collections that don't have metadata saved.

        This is useful for:
        - Collections created before metadata tracking was implemented
        - Collections created externally by other systems
        - System upgrades/migrations

        Returns:
            OperationResult with migration statistics
        """
        start_time = time.time()

        try:
            # List all collections in Qdrant
            collections_result = await self.list_collections()
            if not collections_result.success:
                return OperationResult(
                    status=OperationStatus.FAILED,
                    operation="migrate_metadata",
                    error="Failed to list collections",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            collections = collections_result.data.get("collections", [])
            migrated = 0
            skipped = 0
            failed = 0

            embedding_config = self.config.get("embedding", {})
            chunking_config = self.config.get("chunking", {})

            for collection_name in collections:
                try:
                    # Check if metadata already exists
                    existing_metadata = await self.metadata_manager.get_metadata(
                        collection_name
                    )

                    if existing_metadata:
                        logger.debug(
                            f"Metadata already exists for collection '{collection_name}', skipping"
                        )
                        skipped += 1
                        continue

                    # Get collection info from Qdrant
                    collection_info = await self.client.get_collection_info(
                        collection_name
                    )
                    vector_size = collection_info.get("config", {}).get("vector_size")
                    distance = collection_info.get("config", {}).get(
                        "distance", "Cosine"
                    )

                    if not vector_size:
                        logger.warning(
                            f"Could not determine vector_size for collection '{collection_name}'"
                        )
                        failed += 1
                        continue

                    # Save metadata with current config (best guess)
                    await self.metadata_manager.save_metadata(
                        collection_name=collection_name,
                        vector_size=vector_size,
                        distance_metric=distance,
                        embedding_model=embedding_config.get("model", "unknown"),
                        embedding_provider=embedding_config.get("provider", "unknown"),
                        chunking_config=chunking_config
                        if chunking_config.get("enabled")
                        else None,
                        custom_metadata={
                            "migrated": True,
                            "migration_note": "Metadata auto-generated during migration",
                        },
                    )

                    logger.info(
                        f"Migrated metadata for collection '{collection_name}'",
                        extra={
                            "vector_size": vector_size,
                            "distance": distance,
                            "model": embedding_config.get("model"),
                        },
                    )
                    migrated += 1

                except Exception as e:
                    logger.error(
                        f"Failed to migrate metadata for collection '{collection_name}': {e}"
                    )
                    failed += 1
                    continue

            duration = (time.time() - start_time) * 1000

            logger.info(
                f"Metadata migration complete: {migrated} migrated, {skipped} skipped, {failed} failed"
            )

            return OperationResult(
                status=OperationStatus.SUCCESS,
                operation="migrate_metadata",
                data={
                    "total_collections": len(collections),
                    "migrated": migrated,
                    "skipped": skipped,
                    "failed": failed,
                },
                duration_ms=duration,
            )

        except Exception as e:
            logger.error(f"Metadata migration failed: {e}")
            return OperationResult(
                status=OperationStatus.FAILED,
                operation="migrate_metadata",
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    # =========================================================================
    # Statistics
    # =========================================================================

    @property
    def stats(self) -> Dict[str, Any]:
        """Get operation statistics."""
        return {
            **self._stats,
            "embedding_stats": self.embedding_manager.metrics,
            "chunking_stats": self.chunking_manager.stats,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "documents_added": 0,
            "documents_deleted": 0,
            "queries_executed": 0,
            "chunks_created": 0,
        }
