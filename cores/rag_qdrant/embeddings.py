"""
Embedding Providers - Enterprise Grade

Production-ready embedding generation with:
- Multiple provider support (sentence-transformers, OpenAI, Cohere, custom)
- Batch processing with adaptive sizing
- Caching layer
- Fallback chain
- Metrics and monitoring
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from functools import lru_cache
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Embedding Cache
# ============================================================================

class EmbeddingCache:
    """
    LRU cache for embeddings to avoid recomputation.
    
    Thread-safe with configurable max size and TTL.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: Optional[float] = None
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
        
        # Metrics
        self._hits = 0
        self._misses = 0
    
    def _compute_key(self, text: str, model: str, is_query: bool = True) -> str:
        """Compute cache key from text, model, and embedding type.

        Args:
            text: Input text
            model: Model identifier
            is_query: True for query embeddings (with query prefix),
                     False for passage/document embeddings (with passage prefix)

        Note: is_query is included in cache key because BGE-m3 and E5 models
        produce different embeddings for the same text depending on prefix.
        """
        # Include embedding type marker: Q=query, P=passage
        prefix_marker = "Q" if is_query else "P"
        content = f"{model}:{prefix_marker}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get(
        self,
        text: str,
        model: str,
        is_query: bool = True
    ) -> Optional[List[float]]:
        """Get embedding from cache.

        Args:
            text: Input text
            model: Model identifier
            is_query: True for query embeddings, False for passage embeddings
        """
        async with self._lock:
            key = self._compute_key(text, model, is_query)
            
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if self.ttl_seconds:
                age = time.time() - entry["timestamp"]
                if age > self.ttl_seconds:
                    del self._cache[key]
                    self._access_order.remove(key)
                    self._misses += 1
                    return None
            
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            
            self._hits += 1
            return entry["embedding"]
    
    async def set(
        self,
        text: str,
        model: str,
        embedding: List[float],
        is_query: bool = True
    ) -> None:
        """Store embedding in cache.

        Args:
            text: Input text
            model: Model identifier
            embedding: Embedding vector to cache
            is_query: True for query embeddings, False for passage embeddings
        """
        async with self._lock:
            key = self._compute_key(text, model, is_query)

            # If key already exists, remove old position from access order
            if key in self._cache:
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass

            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = self._access_order.pop(0)
                self._cache.pop(oldest_key, None)

            self._cache[key] = {
                "embedding": embedding,
                "timestamp": time.time()
            }
            self._access_order.append(key)
    
    async def clear(self) -> None:
        """Clear all cached embeddings."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._hits = 0
            self._misses = 0
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "ttl_seconds": self.ttl_seconds
        }


# ============================================================================
# Embedding Provider Interface
# ============================================================================

class EmbeddingProviderType(Enum):
    """Supported embedding providers."""
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPENAI = "openai"
    OLLAMA = "ollama"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class EmbeddingConfig:
    """Embedding provider configuration."""
    provider: str = "sentence-transformers"
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True
    cache_enabled: bool = True
    cache_max_size: int = 10000
    cache_ttl_seconds: Optional[float] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    # v4.1.4: Prefix configuration for BGE/E5 cross-lingual support
    prefix_enabled: bool = True  # Enable auto-detection and application of prefixes
    query_prefix: Optional[str] = None  # Override auto-detected query prefix (e.g., "query: ")
    passage_prefix: Optional[str] = None  # Override auto-detected passage prefix (e.g., "passage: ")
    # v4.2.9: Device configuration for GPU priority with CPU fallback
    device: str = "auto"  # "auto", "cuda", "cpu" - auto prefers GPU with CPU fallback
    # v6.4.9: Redis L2 cache for cross-container sharing
    redis_cache_ttl: int = 3600  # TTL in seconds for Redis embedding cache (0 = disabled)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._initialized = False
        
        # Metrics
        self._total_embeddings = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0
    
    @property
    def name(self) -> str:
        """Provider name."""
        return self.__class__.__name__
    
    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.config.dimension
    
    @property
    def model_name(self) -> str:
        """Model name."""
        return self.config.model
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get provider metrics."""
        avg_latency = 0.0
        if self._total_embeddings > 0:
            avg_latency = self._total_latency_ms / self._total_embeddings
        
        return {
            "provider": self.name,
            "model": self.model_name,
            "dimension": self.dimension,
            "total_embeddings": self._total_embeddings,
            "total_tokens": self._total_tokens,
            "average_latency_ms": round(avg_latency, 2)
        }
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the provider."""
        pass
    
    @abstractmethod
    async def embed_single(self, text: str, is_query: bool = True) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed
            is_query: If True, apply query prefix for models that require it (BGE-m3, E5).
                     If False, apply passage/document prefix.
                     Default: True (query behavior)
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            is_query: If True, apply query prefix for models that require it.
                     If False, apply passage/document prefix.
                     Default: False (passage/document behavior for batch indexing)
        """
        pass
    
    async def embed(
        self,
        texts: Union[str, List[str]],
        is_query: bool = True
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            is_query: True for query embeddings (uses query prefix),
                     False for passage/document embeddings (uses passage prefix)

        Returns:
            Single embedding or list of embeddings
        """
        if isinstance(texts, str):
            return await self.embed_single(texts, is_query=is_query)
        else:
            return await self.embed_batch(texts, is_query=is_query)


# ============================================================================
# Sentence Transformers Provider
# ============================================================================

class SentenceTransformersProvider(EmbeddingProvider):
    """
    Sentence Transformers embedding provider.

    Uses local models for fast, private embedding generation.

    Supports automatic prefix application for models that require it:
    - BGE models (BAAI/bge-*): Use "query: " and "passage: " prefixes
    - E5 models (intfloat/e5-*, intfloat/multilingual-e5-*): Use "query: " and "passage: " prefixes

    This is CRITICAL for cross-lingual retrieval with BGE-m3.
    """

    # Models that require query:/passage: prefixes for optimal performance
    # These patterns are matched case-insensitively against model names
    PREFIX_REQUIRED_PATTERNS = [
        "bge-",           # BAAI/bge-m3, bge-large, bge-base, etc.
        "BAAI/bge",       # Full path format
        "e5-",            # intfloat/e5-large, e5-base, etc.
        "intfloat/e5",    # Full path format
        "intfloat/multilingual-e5",  # Multilingual E5 models
    ]

    def _requires_prefix(self) -> bool:
        """Check if the current model requires query/passage prefixes.

        Respects config.prefix_enabled to allow disabling auto-detection.

        Returns:
            True if model is BGE or E5 family requiring prefixes
        """
        # Check if prefixes are disabled via config
        if not getattr(self.config, 'prefix_enabled', True):
            return False

        model_lower = self.config.model.lower()
        return any(pattern.lower() in model_lower for pattern in self.PREFIX_REQUIRED_PATTERNS)

    def _get_query_prefix(self) -> str:
        """Get the query prefix for the current model.

        Returns:
            Config override if set, "query: " for BGE/E5 models, empty string otherwise
        """
        # Check for config override first
        override = getattr(self.config, 'query_prefix', None)
        if override:
            return override

        if self._requires_prefix():
            return "query: "
        return ""

    def _get_passage_prefix(self) -> str:
        """Get the passage/document prefix for the current model.

        Returns:
            Config override if set, "passage: " for BGE/E5 models, empty string otherwise
        """
        # Check for config override first
        override = getattr(self.config, 'passage_prefix', None)
        if override:
            return override

        if self._requires_prefix():
            return "passage: "
        return ""

    def _apply_prefix(self, text: str, is_query: bool) -> str:
        """Apply appropriate prefix to text based on embedding type.

        Args:
            text: Input text
            is_query: True to apply query prefix, False for passage prefix

        Returns:
            Text with prefix applied (or unchanged if model doesn't require prefixes)
        """
        if is_query:
            prefix = self._get_query_prefix()
        else:
            prefix = self._get_passage_prefix()

        if prefix:
            return f"{prefix}{text}"
        return text

    async def initialize(self) -> None:
        """Load the sentence transformer model via SharedModelPool (GPU dedup)."""
        if self._initialized:
            return

        try:
            from ubp_enterprise_hybrid.modules.cores._shared.model_pool import SharedModelPool

            device_config = getattr(self.config, 'device', 'auto')

            logger.info(f"[EMBEDDING] Loading model via SharedModelPool: {self.config.model}")

            # Load via pool (thread-safe, dedup across modules)
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: SharedModelPool.get_sentence_transformer(
                    model_name=self.config.model,
                    device=device_config,
                    trust_remote_code=True,
                ),
            )

            # Update dimension from model
            self.config.dimension = self._model.get_sentence_embedding_dimension()

            self._initialized = True
            logger.info(
                f"[EMBEDDING] Model ready: '{self.config.model}' "
                f"dim={self.config.dimension} (via SharedModelPool)"
            )

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Release model reference. SharedModelPool owns actual lifecycle."""
        self._model = None
        self._initialized = False
        logger.info("Sentence-transformers provider shutdown (reference released)")
    
    async def embed_single(self, text: str, is_query: bool = True) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed
            is_query: If True, apply query prefix (for search queries).
                     If False, apply passage prefix (for documents).
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Apply prefix for BGE/E5 models (critical for cross-lingual retrieval)
        prefixed_text = self._apply_prefix(text, is_query)

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                prefixed_text,
                normalize_embeddings=self.config.normalize
            ).tolist()
        )

        latency = (time.time() - start_time) * 1000
        self._total_embeddings += 1
        self._total_latency_ms += latency
        self._total_tokens += len(text.split())

        # Log prefix usage for debugging (only once per session)
        if self._total_embeddings == 1 and self._requires_prefix():
            prefix_type = "query" if is_query else "passage"
            logger.info(
                f"[PREFIX] Model '{self.config.model}' uses {prefix_type} prefix for embedding"
            )

        return embedding
    
    async def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed
            is_query: If True, apply query prefix (for search queries).
                     If False, apply passage prefix (for documents/indexing).
                     Default: False (batch embedding is typically for document indexing)
        """
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        start_time = time.time()
        all_embeddings = []

        # Apply prefixes for BGE/E5 models (critical for cross-lingual retrieval)
        prefixed_texts = [self._apply_prefix(t, is_query) for t in texts]

        # Log prefix usage for batch (once at start)
        if self._requires_prefix():
            prefix_type = "query" if is_query else "passage"
            logger.debug(
                f"[PREFIX] Batch embedding {len(texts)} texts with {prefix_type} prefix "
                f"(model: {self.config.model})"
            )

        # Process in batches
        for i in range(0, len(prefixed_texts), self.config.batch_size):
            batch = prefixed_texts[i:i + self.config.batch_size]

            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                None,
                lambda b=batch: self._model.encode(
                    b,
                    normalize_embeddings=self.config.normalize,
                    batch_size=self.config.batch_size
                ).tolist()
            )

            all_embeddings.extend(batch_embeddings)

        latency = (time.time() - start_time) * 1000
        self._total_embeddings += len(texts)
        self._total_latency_ms += latency
        self._total_tokens += sum(len(t.split()) for t in texts)

        return all_embeddings


# ============================================================================
# OpenAI Provider
# ============================================================================

class OpenAIProvider(EmbeddingProvider):
    """
    OpenAI embedding provider.
    
    Uses OpenAI's embedding API.
    """
    
    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        if self._initialized:
            return
        
        try:
            import openai
            
            if not self.config.api_key:
                raise ValueError("OpenAI API key required")
            
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
            
            self._initialized = True
            logger.info(f"OpenAI provider initialized with model: {self.config.model}")
            
        except ImportError:
            raise ImportError(
                "openai not installed. Install with: pip install openai"
            )
    
    async def shutdown(self) -> None:
        """Cleanup OpenAI client."""
        if hasattr(self, '_client'):
            await self._client.close()
        self._initialized = False
        logger.info("OpenAI provider shutdown")
    
    async def embed_single(self, text: str, is_query: bool = True) -> List[float]:
        """Generate embedding using OpenAI API.

        Args:
            text: Input text
            is_query: Accepted for API compatibility but ignored (OpenAI doesn't use prefixes)
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        response = await self._client.embeddings.create(
            model=self.config.model,
            input=text
        )

        embedding = response.data[0].embedding

        latency = (time.time() - start_time) * 1000
        self._total_embeddings += 1
        self._total_latency_ms += latency
        self._total_tokens += response.usage.total_tokens

        return embedding

    async def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for batch using OpenAI API.

        Args:
            texts: List of texts
            is_query: Accepted for API compatibility but ignored (OpenAI doesn't use prefixes)
        """
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        start_time = time.time()
        all_embeddings = []

        # Process in batches (OpenAI has limits)
        batch_size = min(self.config.batch_size, 2048)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = await self._client.embeddings.create(
                model=self.config.model,
                input=batch
            )

            batch_embeddings = [d.embedding for d in response.data]
            all_embeddings.extend(batch_embeddings)

            self._total_tokens += response.usage.total_tokens

        latency = (time.time() - start_time) * 1000
        self._total_embeddings += len(texts)
        self._total_latency_ms += latency

        return all_embeddings


# ============================================================================
# Ollama Provider
# ============================================================================

class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Ollama embedding provider.

    Uses Ollama's local API for embedding generation.
    Supports models like nomic-embed-text, bge-m3, mxbai-embed-large, etc.

    Supports automatic prefix application for models that require it:
    - bge-m3: Use "query: " and "passage: " prefixes (critical for cross-lingual)

    Configuration:
        - Reads UBP_OLLAMA_API_URL from environment (default: http://ubp-ollama:11434)
        - Falls back to config.api_base if provided
    """

    # Known Ollama embedding models and their dimensions
    KNOWN_MODELS: Dict[str, int] = {
        "nomic-embed-text": 768,
        "nomic-embed-text:latest": 768,
        "bge-m3": 1024,
        "bge-m3:latest": 1024,
        "mxbai-embed-large": 1024,
        "mxbai-embed-large:latest": 1024,
        "snowflake-arctic-embed": 1024,
        "snowflake-arctic-embed:110m": 768,
        "snowflake-arctic-embed:335m": 1024,
        "snowflake-arctic-embed-l-v2.0": 1024,
        "all-minilm": 384,
        "all-minilm:latest": 384,
    }

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)

        # Determine base URL: env var > config > default
        # Check multiple env vars for compatibility:
        # - UBP_OLLAMA_BASE_URL: Container network URL (preferred)
        # - UBP_OLLAMA_API_URL: Local development URL
        self.base_url = os.environ.get(
            "UBP_OLLAMA_BASE_URL",
            os.environ.get(
                "UBP_OLLAMA_API_URL",
                config.api_base or "http://ubp-ollama:11434"
            )
        )

        # Strip trailing slash
        self.base_url = self.base_url.rstrip("/")

        self._client = None

        # Auto-detect dimension from known models
        model_lower = config.model.lower()
        if model_lower in self.KNOWN_MODELS:
            self.config.dimension = self.KNOWN_MODELS[model_lower]
        elif config.dimension == 384:  # Default wasn't explicitly set
            # Try partial match
            for known_model, dim in self.KNOWN_MODELS.items():
                if known_model in model_lower or model_lower in known_model:
                    self.config.dimension = dim
                    break

    # Models that require query:/passage: prefixes (same as SentenceTransformers)
    PREFIX_REQUIRED_PATTERNS = ["bge-", "bge_", "e5-", "e5_"]

    def _requires_prefix(self) -> bool:
        """Check if the current model requires query/passage prefixes."""
        model_lower = self.config.model.lower()
        return any(pattern in model_lower for pattern in self.PREFIX_REQUIRED_PATTERNS)

    def _apply_prefix(self, text: str, is_query: bool) -> str:
        """Apply appropriate prefix to text based on embedding type."""
        if not self._requires_prefix():
            return text
        prefix = "query: " if is_query else "passage: "
        return f"{prefix}{text}"

    async def initialize(self) -> None:
        """Initialize httpx client and verify Ollama connection."""
        if self._initialized:
            return

        try:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.config.timeout,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )

            # Verify Ollama is reachable
            try:
                response = await self._client.get("/api/tags")
                if response.status_code == 200:
                    logger.info(
                        f"Ollama embedding provider initialized: {self.base_url}",
                        extra={"model": self.config.model}
                    )
                else:
                    logger.warning(
                        f"Ollama responded with status {response.status_code}, "
                        f"embedding may fail"
                    )
            except Exception as e:
                logger.warning(
                    f"Could not verify Ollama connection: {e}. "
                    f"Proceeding anyway - embedding calls may fail."
                )

            # Try to get actual dimension from a test embedding
            try:
                test_response = await self._client.post(
                    "/api/embeddings",
                    json={"model": self.config.model, "prompt": "test"}
                )
                if test_response.status_code == 200:
                    data = test_response.json()
                    if "embedding" in data:
                        actual_dim = len(data["embedding"])
                        if actual_dim != self.config.dimension:
                            logger.info(
                                f"Updated dimension from {self.config.dimension} to "
                                f"{actual_dim} based on model response"
                            )
                            self.config.dimension = actual_dim
            except Exception as e:
                logger.debug(f"Could not auto-detect dimension: {e}")

            self._initialized = True

        except ImportError:
            raise ImportError(
                "httpx not installed. Install with: pip install httpx"
            )

    async def shutdown(self) -> None:
        """Cleanup httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False
        logger.info("Ollama embedding provider shutdown")

    async def embed_single(self, text: str, is_query: bool = True) -> List[float]:
        """Generate embedding using Ollama API.

        Args:
            text: Input text to embed
            is_query: If True, apply query prefix for BGE models.
                     If False, apply passage prefix.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Apply prefix for BGE models (critical for cross-lingual retrieval)
        prefixed_text = self._apply_prefix(text, is_query)

        try:
            response = await self._client.post(
                "/api/embeddings",
                json={
                    "model": self.config.model,
                    "prompt": prefixed_text
                }
            )

            if response.status_code != 200:
                error_text = response.text
                raise RuntimeError(
                    f"Ollama embedding failed: {response.status_code} - {error_text}"
                )

            data = response.json()

            if "embedding" not in data:
                raise RuntimeError(
                    f"Ollama response missing 'embedding' field: {data}"
                )

            embedding = data["embedding"]

            # Update dimension if different (first call may reveal actual dimension)
            if len(embedding) != self.config.dimension:
                logger.info(
                    f"Ollama model '{self.config.model}' returned dimension "
                    f"{len(embedding)}, updating from {self.config.dimension}"
                )
                self.config.dimension = len(embedding)

            latency = (time.time() - start_time) * 1000
            self._total_embeddings += 1
            self._total_latency_ms += latency
            self._total_tokens += len(text.split())

            return embedding

        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            raise

    async def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for batch.

        Args:
            texts: List of texts to embed
            is_query: If True, apply query prefix. If False, apply passage prefix.
                     Default: False (batch embedding is typically for document indexing)

        Note: Ollama API doesn't support batch embedding natively,
        so we process sequentially but could parallelize with asyncio.gather.
        """
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        start_time = time.time()

        # Process in parallel for better performance
        # Limit concurrency to avoid overwhelming Ollama
        semaphore = asyncio.Semaphore(self.config.batch_size)

        async def embed_with_semaphore(text: str) -> List[float]:
            async with semaphore:
                return await self.embed_single(text, is_query=is_query)

        # Use gather for parallel processing
        embeddings = await asyncio.gather(
            *[embed_with_semaphore(text) for text in texts],
            return_exceptions=True
        )

        # Check for exceptions
        results = []
        for i, emb in enumerate(embeddings):
            if isinstance(emb, Exception):
                logger.error(f"Failed to embed text at index {i}: {emb}")
                raise emb
            results.append(emb)

        latency = (time.time() - start_time) * 1000
        logger.debug(
            f"Ollama batch embedding: {len(texts)} texts in {latency:.2f}ms"
        )

        return results


# ============================================================================
# Cohere Provider
# ============================================================================

class CohereEmbeddingProvider(EmbeddingProvider):
    """
    Cohere embedding provider.

    Uses Cohere's API for embedding generation.
    Supports models like embed-english-v3.0, embed-multilingual-v3.0, etc.

    Configuration:
        - Reads UBP_COHERE_API_KEY from environment
        - Falls back to config.api_key if provided
    """

    # Known Cohere embedding models and their dimensions
    KNOWN_MODELS: Dict[str, int] = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
        "embed-english-v2.0": 4096,
        "embed-multilingual-v2.0": 768,
    }

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)

        # Get API key from env or config
        self.api_key = os.environ.get("UBP_COHERE_API_KEY", config.api_key)
        if not self.api_key:
            logger.warning("Cohere API key not configured")

        self.api_base = config.api_base or "https://api.cohere.ai/v1"
        self._client = None

        # Auto-detect dimension from known models
        model_lower = config.model.lower()
        if model_lower in self.KNOWN_MODELS:
            self.config.dimension = self.KNOWN_MODELS[model_lower]

    async def initialize(self) -> None:
        """Initialize Cohere client."""
        if self._initialized:
            return

        if not self.api_key:
            raise ValueError("Cohere API key required but not configured")

        try:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                timeout=self.config.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

            self._initialized = True
            logger.info(
                f"Cohere embedding provider initialized",
                extra={"model": self.config.model}
            )

        except ImportError:
            raise ImportError(
                "httpx not installed. Install with: pip install httpx"
            )

    async def shutdown(self) -> None:
        """Cleanup Cohere client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False
        logger.info("Cohere embedding provider shutdown")

    async def embed_single(self, text: str, is_query: bool = True) -> List[float]:
        """Generate embedding using Cohere API.

        Args:
            text: Input text to embed
            is_query: If True, use 'search_query' input_type.
                     If False, use 'search_document' input_type.
                     Cohere natively supports this distinction.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Cohere uses input_type to distinguish query vs document embeddings
        input_type = "search_query" if is_query else "search_document"

        try:
            response = await self._client.post(
                "/embed",
                json={
                    "texts": [text],
                    "model": self.config.model,
                    "input_type": input_type,
                    "truncate": "END",
                }
            )

            if response.status_code != 200:
                error_text = response.text
                raise RuntimeError(
                    f"Cohere embedding failed: {response.status_code} - {error_text}"
                )

            data = response.json()

            if "embeddings" not in data or not data["embeddings"]:
                raise RuntimeError(
                    f"Cohere response missing 'embeddings' field: {data}"
                )

            embedding = data["embeddings"][0]

            # Update dimension if different
            if len(embedding) != self.config.dimension:
                logger.info(
                    f"Cohere model '{self.config.model}' returned dimension "
                    f"{len(embedding)}, updating from {self.config.dimension}"
                )
                self.config.dimension = len(embedding)

            latency = (time.time() - start_time) * 1000
            self._total_embeddings += 1
            self._total_latency_ms += latency
            self._total_tokens += len(text.split())

            return embedding

        except Exception as e:
            logger.error(f"Cohere embedding error: {e}")
            raise

    async def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for batch using Cohere API (native batch support).

        Args:
            texts: List of texts to embed
            is_query: If True, use 'search_query' input_type.
                     If False, use 'search_document' input_type.
                     Default: False (batch embedding is typically for document indexing)
        """
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        start_time = time.time()

        # Cohere uses input_type to distinguish query vs document embeddings
        input_type = "search_query" if is_query else "search_document"

        try:
            # Cohere supports batch embedding natively (up to 96 texts)
            all_embeddings = []
            batch_size = min(self.config.batch_size, 96)

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                response = await self._client.post(
                    "/embed",
                    json={
                        "texts": batch,
                        "model": self.config.model,
                        "input_type": input_type,
                        "truncate": "END",
                    }
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"Cohere batch embedding failed: {response.status_code}"
                    )

                data = response.json()
                all_embeddings.extend(data["embeddings"])

            latency = (time.time() - start_time) * 1000
            self._total_embeddings += len(texts)
            self._total_latency_ms += latency

            logger.debug(
                f"Cohere batch embedding: {len(texts)} texts in {latency:.2f}ms"
            )

            return all_embeddings

        except Exception as e:
            logger.error(f"Cohere batch embedding error: {e}")
            raise


# ============================================================================
# Mock Provider for Testing
# ============================================================================

class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    async def initialize(self) -> None:
        """No-op initialization."""
        self._initialized = True
        logger.info("Mock embedding provider initialized")
    
    async def shutdown(self) -> None:
        """No-op shutdown."""
        self._initialized = False
    
    async def embed_single(self, text: str, is_query: bool = True) -> List[float]:
        """Generate deterministic mock embedding.

        Args:
            text: Input text
            is_query: Accepted for API compatibility (affects hash for testing)
        """
        import hashlib

        # Include is_query in hash to test that cache keys differ
        prefix_marker = "Q:" if is_query else "P:"
        text_hash = hashlib.md5(f"{prefix_marker}{text}".encode()).hexdigest()

        embedding = []
        for i in range(0, min(len(text_hash) * 2, self.config.dimension * 2), 2):
            if len(embedding) >= self.config.dimension:
                break
            hex_pair = text_hash[i % len(text_hash):i % len(text_hash) + 2]
            if len(hex_pair) == 2:
                value = (int(hex_pair, 16) - 128) / 128.0
                embedding.append(value)

        # Pad if needed
        while len(embedding) < self.config.dimension:
            embedding.append(0.0)

        # Normalize if configured
        if self.config.normalize:
            import math
            norm = math.sqrt(sum(x*x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]

        self._total_embeddings += 1
        return embedding

    async def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate mock embeddings for batch.

        Args:
            texts: List of texts
            is_query: Passed to embed_single for consistent behavior
        """
        return [await self.embed_single(text, is_query) for text in texts]


# ============================================================================
# Embedding Manager with Fallback Chain
# ============================================================================

def _is_permanent_embedding_error(error: Exception) -> bool:
    """Check if an embedding error is permanent (not worth retrying)."""
    error_str = str(error).lower()
    permanent_patterns = [
        "404",                    # Model not found (Ollama)
        "model not found",        # Explicit model missing
        "connection refused",     # Provider not running
        "connect call failed",    # Provider unreachable
        "no such host",           # DNS failure
        "name resolution failed", # DNS failure variant
        "cuda out of memory",     # GPU OOM - won't resolve with retry
        "out of memory",          # Generic OOM
    ]
    return any(pattern in error_str for pattern in permanent_patterns)


class EmbeddingManager:
    """
    Embedding manager with caching, fallback chain, and monitoring.

    Features:
    - Multiple provider support with fallback
    - Embedding cache layer
    - Batch processing
    - Comprehensive metrics
    """
    
    def __init__(self, config: Union[Dict[str, Any], EmbeddingConfig], redis_client=None):
        if isinstance(config, dict):
            self.config = EmbeddingConfig(**config.get("embedding", config))
        else:
            self.config = config

        self._primary_provider: Optional[EmbeddingProvider] = None
        self._fallback_providers: List[EmbeddingProvider] = []
        self._cache: Optional[EmbeddingCache] = None
        self._redis = redis_client  # L2 cache (cross-container)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize embedding manager."""
        if self._initialized:
            return
        
        # Initialize cache if enabled
        if self.config.cache_enabled:
            self._cache = EmbeddingCache(
                max_size=self.config.cache_max_size,
                ttl_seconds=self.config.cache_ttl_seconds
            )
        
        # Create primary provider
        self._primary_provider = self._create_provider(self.config)
        await self._primary_provider.initialize()
        
        # Update dimension from provider
        self.config.dimension = self._primary_provider.dimension
        
        self._initialized = True
        logger.info(
            f"Embedding manager initialized with {self.config.provider}",
            extra={
                "model": self.config.model,
                "dimension": self.config.dimension,
                "cache_enabled": self.config.cache_enabled
            }
        )
    
    async def shutdown(self) -> None:
        """Shutdown embedding manager."""
        if self._primary_provider:
            await self._primary_provider.shutdown()
        
        for provider in self._fallback_providers:
            await provider.shutdown()
        
        if self._cache:
            await self._cache.clear()
        
        self._initialized = False
        logger.info("Embedding manager shutdown")

    # --- Redis L2 cache helpers ---

    def _redis_key(self, text: str, is_query: bool) -> str:
        """Build Redis cache key (same hash as LRU)."""
        prefix_marker = "Q" if is_query else "P"
        content = f"{self.config.model}:{prefix_marker}:{text}"
        h = hashlib.sha256(content.encode()).hexdigest()
        return f"ubp:emb_cache:{h}"

    async def _redis_get(self, text: str, is_query: bool) -> Optional[List[float]]:
        """L2 cache lookup (Redis). Returns None on miss or error."""
        if not self._redis or self.config.redis_cache_ttl <= 0:
            return None
        try:
            raw = await self._redis.get(self._redis_key(text, is_query))
            if raw is not None:
                return json.loads(raw)
        except Exception:
            pass
        return None

    async def _redis_set(self, text: str, is_query: bool, embedding: List[float]) -> None:
        """L2 cache write (Redis). Best-effort, errors silenced."""
        if not self._redis or self.config.redis_cache_ttl <= 0:
            return
        try:
            await self._redis.set(
                self._redis_key(text, is_query),
                json.dumps(embedding),
                ex=self.config.redis_cache_ttl,
            )
        except Exception:
            pass

    def _create_provider(self, config: EmbeddingConfig) -> EmbeddingProvider:
        """Create embedding provider based on type."""
        provider_type = config.provider.lower()

        if provider_type == "sentence-transformers":
            return SentenceTransformersProvider(config)
        elif provider_type == "openai":
            return OpenAIProvider(config)
        elif provider_type == "ollama":
            return OllamaEmbeddingProvider(config)
        elif provider_type == "cohere":
            return CohereEmbeddingProvider(config)
        elif provider_type == "mock":
            return MockEmbeddingProvider(config)
        else:
            logger.warning(f"Unknown provider '{provider_type}', using mock")
            return MockEmbeddingProvider(config)
    
    def add_fallback(self, config: EmbeddingConfig) -> None:
        """Add a fallback provider."""
        provider = self._create_provider(config)
        self._fallback_providers.append(provider)
        logger.info(f"Added fallback provider: {provider.name}")
    
    async def embed(
        self,
        text: str,
        use_cache: bool = True,
        is_query: bool = True
    ) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Input text
            use_cache: Whether to use cache
            is_query: If True, apply query prefix for BGE/E5 models (search queries).
                     If False, apply passage prefix (document indexing).
                     Default: True (single text embedding is typically a query)

        Returns:
            Embedding vector
        """
        if not self._initialized:
            await self.initialize()

        # Check L1 cache (in-memory LRU)
        if use_cache and self._cache:
            cached = await self._cache.get(text, self.config.model, is_query)
            if cached is not None:
                return cached

        # Check L2 cache (Redis)
        if use_cache:
            redis_cached = await self._redis_get(text, is_query)
            if redis_cached is not None:
                # Promote to L1
                if self._cache:
                    await self._cache.set(text, self.config.model, redis_cached, is_query)
                return redis_cached

        # Try primary provider
        embedding = None
        last_error = None

        try:
            embedding = await self._primary_provider.embed_single(text, is_query=is_query)
        except Exception as e:
            last_error = e
            logger.warning(f"Primary provider failed: {e}")

            # Try fallbacks
            for provider in self._fallback_providers:
                try:
                    if not provider._initialized:
                        await provider.initialize()
                    embedding = await provider.embed_single(text, is_query=is_query)
                    break
                except Exception as fe:
                    last_error = fe
                    logger.warning(f"Fallback provider {provider.name} failed: {fe}")

        if embedding is None:
            raise RuntimeError(
                f"All embedding providers failed. Last error: {last_error}"
            )

        # Cache result in L1 (LRU) and L2 (Redis)
        if use_cache:
            if self._cache:
                await self._cache.set(text, self.config.model, embedding, is_query)
            await self._redis_set(text, is_query, embedding)

        return embedding
    
    async def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        is_query: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            use_cache: Whether to use cache
            is_query: If True, apply query prefix for BGE/E5 models.
                     If False, apply passage prefix (document indexing).
                     Default: False (batch embedding is typically for document indexing)

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        texts_to_embed: List[tuple] = []  # (index, text)

        # Check L1 (LRU) then L2 (Redis) cache for each text
        if use_cache:
            for i, text in enumerate(texts):
                cached = None
                if self._cache:
                    cached = await self._cache.get(text, self.config.model, is_query)
                if cached is None:
                    cached = await self._redis_get(text, is_query)
                    if cached is not None and self._cache:
                        await self._cache.set(text, self.config.model, cached, is_query)
                if cached is not None:
                    embeddings[i] = cached
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = list(enumerate(texts))

        if not texts_to_embed:
            return embeddings

        # Embed remaining texts
        indices, uncached_texts = zip(*texts_to_embed)

        try:
            new_embeddings = await self._primary_provider.embed_batch(
                list(uncached_texts),
                is_query=is_query
            )

            # Map back to original positions
            for idx, embedding in zip(indices, new_embeddings):
                embeddings[idx] = embedding

                # Cache new embeddings in L1 (LRU) and L2 (Redis)
                if use_cache:
                    if self._cache:
                        await self._cache.set(
                            texts[idx], self.config.model, embedding, is_query
                        )
                    await self._redis_set(texts[idx], is_query, embedding)

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Fall back to individual embedding with retry logic
            max_retries = 3
            failed_indices = []
            circuit_open = False  # Circuit breaker for permanent errors

            for idx, text in texts_to_embed:
                # Circuit breaker: skip remaining chunks on permanent error
                if circuit_open:
                    failed_indices.append(idx)
                    embeddings[idx] = None
                    continue

                success = False
                last_error = None

                for attempt in range(max_retries):
                    try:
                        embeddings[idx] = await self.embed(text, use_cache, is_query)
                        success = True
                        if attempt > 0:
                            logger.info(f"Embedding succeeded for index {idx} on retry {attempt + 1}")
                        break
                    except Exception as ie:
                        last_error = ie
                        # Open circuit on permanent errors (404, connection refused, etc.)
                        if _is_permanent_embedding_error(ie):
                            remaining = len(texts_to_embed) - len(failed_indices) - 1
                            logger.error(
                                f"Permanent embedding error detected at index {idx}: {ie}. "
                                f"Opening circuit breaker - skipping remaining {remaining} chunks."
                            )
                            circuit_open = True
                            break
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 0.5  # Exponential backoff: 0.5s, 1s, 2s
                            logger.warning(
                                f"Embedding failed for index {idx}, attempt {attempt + 1}/{max_retries}. "
                                f"Retrying in {wait_time}s... Error: {ie}"
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(
                                f"Embedding PERMANENTLY failed for index {idx} after {max_retries} retries: {ie}. "
                                f"Skipping this chunk."
                            )

                if not success:
                    failed_indices.append(idx)
                    embeddings[idx] = None  # Mark as failed

            if failed_indices:
                logger.warning(
                    f"⚠️ {len(failed_indices)} chunks failed embedding and were skipped: {failed_indices[:10]}..."
                    + (" (circuit breaker opened - permanent error)" if circuit_open else "")
                )

        return embeddings
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.dimension
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get embedding manager metrics."""
        provider_metrics = {}
        if self._primary_provider:
            provider_metrics["primary"] = self._primary_provider.metrics
        
        for i, provider in enumerate(self._fallback_providers):
            provider_metrics[f"fallback_{i}"] = provider.metrics
        
        cache_metrics = {}
        if self._cache:
            cache_metrics = self._cache.stats
        
        return {
            "config": {
                "provider": self.config.provider,
                "model": self.config.model,
                "dimension": self.config.dimension,
                "batch_size": self.config.batch_size
            },
            "providers": provider_metrics,
            "cache": cache_metrics
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_embedding_manager(config: Dict[str, Any], redis_client=None) -> EmbeddingManager:
    """
    Create an EmbeddingManager from configuration.

    Args:
        config: Configuration dictionary
        redis_client: Optional Redis client for L2 embedding cache

    Returns:
        Configured EmbeddingManager instance
    """
    embedding_config = config.get("embedding", {})

    return EmbeddingManager(EmbeddingConfig(
        provider=embedding_config.get("provider", "sentence-transformers"),
        model=embedding_config.get("model", "all-MiniLM-L6-v2"),
        dimension=embedding_config.get("dimension", 384),
        batch_size=embedding_config.get("batch_size", 32),
        normalize=embedding_config.get("normalize", True),
        cache_enabled=embedding_config.get("cache_enabled", True),
        cache_max_size=embedding_config.get("cache_max_size", 10000),
        cache_ttl_seconds=embedding_config.get("cache_ttl_seconds"),
        api_key=embedding_config.get("api_key"),
        api_base=embedding_config.get("api_base"),
        timeout=embedding_config.get("timeout", 30.0),
        max_retries=embedding_config.get("max_retries", 3),
        # v4.1.4: Prefix configuration for BGE/E5 cross-lingual support
        prefix_enabled=embedding_config.get("prefix_enabled", True),
        query_prefix=embedding_config.get("query_prefix"),
        passage_prefix=embedding_config.get("passage_prefix"),
        # v4.2.9: Device configuration for GPU priority with CPU fallback
        device=embedding_config.get("device", "auto"),
        # v6.4.9: Redis L2 cache TTL
        redis_cache_ttl=embedding_config.get("redis_cache_ttl", 3600),
    ), redis_client=redis_client)
