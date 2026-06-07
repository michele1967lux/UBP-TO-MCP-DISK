"""
Collection Metadata Manager - Enterprise Grade

Manages collection-level metadata storage using Redis:
- Embedding model information (name, provider, version)
- Collection configuration (dimension, distance metric)
- Creation timestamps and versioning
- Auto-configuration support for multi-model environments
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CollectionMetadata:
    """
    Complete metadata for a Qdrant collection.

    Attributes:
        collection_name: Name of the collection
        vector_size: Embedding dimension (384, 768, 1536, etc.)
        distance_metric: Similarity metric (Cosine, Euclid, Dot)
        embedding_model: Model name (e.g., "all-MiniLM-L6-v2")
        embedding_provider: Provider (e.g., "sentence-transformers", "openai")
        created_at: ISO timestamp of collection creation
        updated_at: ISO timestamp of last metadata update
        chunking_config: Chunking settings used
        custom_metadata: User-defined metadata
    """

    collection_name: str
    vector_size: int
    distance_metric: str
    embedding_model: str
    embedding_provider: str
    created_at: str
    updated_at: str
    chunking_config: Optional[Dict[str, Any]] = None
    custom_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CollectionMetadata:
        """Create from dictionary."""
        return cls(**data)


class CollectionMetadataManager:
    """
    Manages collection metadata using Redis for persistence.

    Features:
    - Automatic metadata persistence on collection creation
    - Metadata retrieval for auto-configuration
    - Version tracking and audit trail
    - Support for multiple embedding models per system
    """

    REDIS_KEY_PREFIX = "rag:collection:metadata:"
    REDIS_KEY_INDEX = "rag:collections:index"

    def __init__(self, redis_client: Optional[Any] = None):
        """
        Initialize metadata manager.

        Args:
            redis_client: Redis client instance (optional, for in-memory fallback)
        """
        self.redis_client = redis_client
        self._in_memory_store: Dict[str, CollectionMetadata] = {}
        self._use_redis = redis_client is not None

        if not self._use_redis:
            logger.warning(
                "Redis client not provided. Collection metadata will be stored in-memory only. "
                "Metadata will be lost on restart."
            )

    async def save_metadata(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str,
        embedding_model: str,
        embedding_provider: str,
        chunking_config: Optional[Dict[str, Any]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save collection metadata.

        Args:
            collection_name: Collection name
            vector_size: Embedding dimension
            distance_metric: Similarity metric
            embedding_model: Model name
            embedding_provider: Provider name
            chunking_config: Chunking settings
            custom_metadata: Custom user metadata

        Returns:
            True if saved successfully
        """
        now = datetime.now(timezone.utc).isoformat()

        metadata = CollectionMetadata(
            collection_name=collection_name,
            vector_size=vector_size,
            distance_metric=distance_metric,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            created_at=now,
            updated_at=now,
            chunking_config=chunking_config,
            custom_metadata=custom_metadata,
        )

        if self._use_redis:
            try:
                redis_key = f"{self.REDIS_KEY_PREFIX}{collection_name}"
                await self.redis_client.set(
                    redis_key,
                    json.dumps(metadata.to_dict()),
                    ex=None,  # No expiration
                )

                # Add to index
                await self.redis_client.sadd(self.REDIS_KEY_INDEX, collection_name)

                logger.info(
                    f"Saved metadata for collection '{collection_name}'",
                    extra={
                        "model": embedding_model,
                        "provider": embedding_provider,
                        "dimension": vector_size,
                    },
                )
                return True

            except Exception as e:
                logger.error(f"Failed to save metadata to Redis: {e}")
                # Fallback to in-memory
                self._in_memory_store[collection_name] = metadata
                return False
        else:
            # In-memory storage
            self._in_memory_store[collection_name] = metadata
            logger.debug(f"Saved metadata for '{collection_name}' in-memory")
            return True

    async def get_metadata(self, collection_name: str) -> Optional[CollectionMetadata]:
        """
        Retrieve collection metadata.

        Args:
            collection_name: Collection name

        Returns:
            CollectionMetadata if found, None otherwise
        """
        if self._use_redis:
            try:
                redis_key = f"{self.REDIS_KEY_PREFIX}{collection_name}"
                data = await self.redis_client.get(redis_key)

                if data:
                    metadata_dict = json.loads(data)
                    return CollectionMetadata.from_dict(metadata_dict)

                logger.debug(
                    f"No metadata found for collection '{collection_name}' in Redis"
                )
                return None

            except Exception as e:
                logger.error(f"Failed to retrieve metadata from Redis: {e}")
                # Fallback to in-memory
                return self._in_memory_store.get(collection_name)
        else:
            # In-memory retrieval
            return self._in_memory_store.get(collection_name)

    async def update_metadata(
        self, collection_name: str, updates: Dict[str, Any]
    ) -> bool:
        """
        Update existing collection metadata.

        Args:
            collection_name: Collection name
            updates: Dictionary of fields to update

        Returns:
            True if updated successfully
        """
        metadata = await self.get_metadata(collection_name)

        if not metadata:
            logger.warning(
                f"Cannot update metadata: collection '{collection_name}' not found"
            )
            return False

        # Update fields
        metadata_dict = metadata.to_dict()
        metadata_dict.update(updates)
        metadata_dict["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Save updated metadata
        updated_metadata = CollectionMetadata.from_dict(metadata_dict)

        if self._use_redis:
            try:
                redis_key = f"{self.REDIS_KEY_PREFIX}{collection_name}"
                await self.redis_client.set(
                    redis_key, json.dumps(updated_metadata.to_dict()), ex=None
                )
                logger.info(f"Updated metadata for collection '{collection_name}'")
                return True

            except Exception as e:
                logger.error(f"Failed to update metadata in Redis: {e}")
                return False
        else:
            self._in_memory_store[collection_name] = updated_metadata
            logger.debug(f"Updated metadata for '{collection_name}' in-memory")
            return True

    async def delete_metadata(self, collection_name: str) -> bool:
        """
        Delete collection metadata.

        Args:
            collection_name: Collection name

        Returns:
            True if deleted successfully
        """
        if self._use_redis:
            try:
                redis_key = f"{self.REDIS_KEY_PREFIX}{collection_name}"
                await self.redis_client.delete(redis_key)
                await self.redis_client.srem(self.REDIS_KEY_INDEX, collection_name)
                logger.info(f"Deleted metadata for collection '{collection_name}'")
                return True

            except Exception as e:
                logger.error(f"Failed to delete metadata from Redis: {e}")
                return False
        else:
            if collection_name in self._in_memory_store:
                del self._in_memory_store[collection_name]
                logger.debug(f"Deleted metadata for '{collection_name}' from memory")
                return True
            return False

    async def list_collections(self) -> List[str]:
        """
        List all collections with metadata.

        Returns:
            List of collection names
        """
        if self._use_redis:
            try:
                collections = await self.redis_client.smembers(self.REDIS_KEY_INDEX)
                return list(collections) if collections else []

            except Exception as e:
                logger.error(f"Failed to list collections from Redis: {e}")
                return list(self._in_memory_store.keys())
        else:
            return list(self._in_memory_store.keys())

    async def get_all_metadata(self) -> Dict[str, CollectionMetadata]:
        """
        Retrieve metadata for all collections.

        Returns:
            Dictionary mapping collection names to metadata
        """
        collections = await self.list_collections()
        result = {}

        for collection_name in collections:
            metadata = await self.get_metadata(collection_name)
            if metadata:
                result[collection_name] = metadata

        return result

    async def collection_exists(self, collection_name: str) -> bool:
        """
        Check if collection metadata exists.

        Args:
            collection_name: Collection name

        Returns:
            True if metadata exists
        """
        metadata = await self.get_metadata(collection_name)
        return metadata is not None
