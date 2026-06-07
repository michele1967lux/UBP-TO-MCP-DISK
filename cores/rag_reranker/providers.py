"""
Reranker Provider - Pure Technical Logic
Zero UBP dependencies. Can be tested standalone.
Implements cross-encoder reranking with multiple backends.

Features:
- Local cross-encoder models (sentence-transformers)
- Cohere Rerank API integration
- Jina Rerank API integration
- Pass-through fallback for graceful degradation

Backends:
- LOCAL_CROSS_ENCODER: cross-encoder/ms-marco-MiniLM-L-6-v2
- COHERE: Cohere Rerank API
- JINA: Jina Rerank API
- NONE: Pass-through (no reranking)

v3.7.1 FIX-PERF-001:
- Wrapped model.predict() in run_in_executor to prevent blocking the event loop
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import logging
import os
import asyncio  # FIX-PERF-001 v3.7.1: Added for run_in_executor

logger = logging.getLogger(__name__)


# ============================================================================
# Reranker Model Catalogs (Grand Unified Registry)
# ============================================================================

# Sentence-Transformers CrossEncoder models (local, always available)
SENTENCE_TRANSFORMERS_RERANKERS = [
    {
        "id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "name": "MS MARCO MiniLM L6",
        "provider": "sentence-transformers",
        "type": "reranker",
        "available": True,
        "description": "Fast and efficient reranker (~80MB)",
    },
    {
        "id": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "name": "MS MARCO MiniLM L12",
        "provider": "sentence-transformers",
        "type": "reranker",
        "available": True,
        "description": "More accurate reranker (~120MB)",
    },
    {
        "id": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "name": "MS MARCO TinyBERT",
        "provider": "sentence-transformers",
        "type": "reranker",
        "available": True,
        "description": "Ultra-fast tiny reranker (~17MB)",
    },
    {
        "id": "BAAI/bge-reranker-base",
        "name": "BGE Reranker Base",
        "provider": "sentence-transformers",
        "type": "reranker",
        "available": True,
        "description": "Balanced cross-encoder reranker",
    },
    {
        "id": "BAAI/bge-reranker-large",
        "name": "BGE Reranker Large",
        "provider": "sentence-transformers",
        "type": "reranker",
        "available": True,
        "description": "High quality cross-encoder reranker",
    },
    {
        "id": "BAAI/bge-reranker-v2-m3",
        "name": "BGE Reranker v2 M3",
        "provider": "sentence-transformers",
        "type": "reranker",
        "available": True,
        "description": "Multilingual reranker v2 (2GB)",
    },
]

# Cohere Rerank API models
COHERE_RERANKERS = [
    {
        "id": "rerank-english-v2.0",
        "name": "Cohere Rerank English v2",
        "provider": "cohere",
        "type": "reranker",
        "available": False,  # Requires API key
        "description": "English reranker via Cohere API",
    },
    {
        "id": "rerank-multilingual-v2.0",
        "name": "Cohere Rerank Multilingual v2",
        "provider": "cohere",
        "type": "reranker",
        "available": False,
        "description": "Multilingual reranker via Cohere API",
    },
    {
        "id": "rerank-english-v3.0",
        "name": "Cohere Rerank English v3",
        "provider": "cohere",
        "type": "reranker",
        "available": False,
        "description": "Latest English reranker via Cohere API",
    },
]

# Jina Rerank API models
JINA_RERANKERS = [
    {
        "id": "jina-reranker-v1-base-en",
        "name": "Jina Reranker v1 Base EN",
        "provider": "jina",
        "type": "reranker",
        "available": False,  # Requires API key
        "description": "English reranker via Jina API",
    },
    {
        "id": "jina-reranker-v1-turbo-en",
        "name": "Jina Reranker v1 Turbo EN",
        "provider": "jina",
        "type": "reranker",
        "available": False,
        "description": "Fast English reranker via Jina API",
    },
    {
        "id": "jina-reranker-v2-base-multilingual",
        "name": "Jina Reranker v2 Multilingual",
        "provider": "jina",
        "type": "reranker",
        "available": False,
        "description": "Multilingual reranker via Jina API",
    },
]

# Ollama reranker patterns for dynamic discovery
OLLAMA_RERANKER_PATTERNS = [
    "reranker",
    "bge-reranker",
    "cross-encoder",
]


class RerankerBackend(str, Enum):
    """Available reranker backends."""

    LOCAL_CROSS_ENCODER = "local_cross_encoder"
    COHERE = "cohere"
    JINA = "jina"
    NONE = "none"  # Pass-through, no reranking


@dataclass
class RerankResult:
    """Represents a reranked document."""

    doc_id: str
    content: str
    original_score: float
    rerank_score: float
    original_rank: int
    new_rank: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "original_score": round(self.original_score, 6),
            "rerank_score": round(self.rerank_score, 6),
            "original_rank": self.original_rank,
            "new_rank": self.new_rank,
            "metadata": self.metadata,
        }


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of documents with 'content' field
            top_k: Number of results to return

        Returns:
            List of RerankResult sorted by rerank_score descending
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get reranker information."""
        pass


class LocalCrossEncoderReranker(BaseReranker):
    """
    Local cross-encoder reranker using sentence-transformers.

    Models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (recommended, ~80MB)
    - cross-encoder/ms-marco-TinyBERT-L-2-v2 (faster, smaller)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (more accurate)

    Cross-encoders score (query, document) pairs directly,
    providing more accurate relevance scores than bi-encoders.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/stsb-roberta-large",
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        preload: bool = False,
    ):
        """
        Initialize local cross-encoder.

        Args:
            model_name: HuggingFace model name (default: stsb-roberta-large for multilingual)
            device: 'auto', 'cpu' or 'cuda' - auto prefers GPU with CPU fallback
            batch_size: Batch size for inference
            max_length: Max sequence length (truncate longer)
            preload: If True, load model immediately instead of lazy loading
        """
        self.model_name = model_name
        self._device_config = device
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

        # TASK #86: Preload model at init to avoid blocking during operations
        if preload:
            logger.info(f"[RERANKER] Preloading model {model_name} at initialization...")
            self._load_model()
            logger.info(f"[RERANKER] Model {model_name} preloaded successfully")

    def set_shared_model(self, model) -> None:
        """Inject pre-loaded model from SharedModelPool.

        Called by adapter.py to share a single GPU model across modules.
        When set, _load_model() becomes a no-op.
        """
        self._model = model
        logger.info(f"[RERANKER] Shared model injected: {self.model_name}")

    def _resolve_device(self, device_config: str) -> str:
        """Resolve device with GPU memory awareness via centralized DeviceResolver."""
        from ubp_enterprise_hybrid.modules.cores._shared.device_resolver import DeviceResolver

        result = DeviceResolver.resolve(
            requested=device_config,
            model_name=self.model_name,
            component="reranker",
        )
        # Store precision for use in _load_model
        self._resolved_precision = result.precision
        return result.device

    def _load_model(self):
        """Lazy load — fallback if no shared model was injected."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                # v4.3.0: Precision from DeviceResolver (FP16 on GPU, FP32 on CPU)
                model_kwargs = {}
                if getattr(self, '_resolved_precision', None) == "fp16":
                    model_kwargs["torch_dtype"] = "float16"

                logger.info(f"[RERANKER] Loading CrossEncoder (standalone): {self.model_name} on {self.device.upper()}")
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=self.max_length,
                    model_kwargs=model_kwargs,
                )
                precision = "FP16" if model_kwargs.get("torch_dtype") == "float16" else "FP32"
                logger.info(f"[RERANKER] Loaded CrossEncoder: {self.model_name} on {self.device.upper()} ({precision}) OK")
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise ImportError(
                    "Please install sentence-transformers: "
                    "pip install sentence-transformers"
                )
        return self._model

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """
        Rerank documents using cross-encoder.
        
        v3.7.1 FIX-PERF-001: Wrapped model.predict() in run_in_executor to prevent
        blocking the asyncio event loop. This is CRITICAL for system stability:
        - Without executor: 19.5s blocking prevents ALL other requests (login, health, etc.)
        - With executor: Reranking runs in thread pool, event loop remains responsive
        """
        if not documents:
            return []

        model = self._load_model()

        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            content = doc.get("content", "")
            # Truncate long documents for efficiency
            if len(content) > self.max_length * 4:  # Rough char estimate
                content = content[: self.max_length * 4]
            pairs.append([query, content])

        # Get scores from cross-encoder
        # FIX-PERF-001 v3.7.1: Offload CPU-bound prediction to thread pool executor
        # This prevents blocking the main asyncio event loop during inference
        try:
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                lambda: model.predict(pairs, batch_size=self.batch_size)
            )
        except Exception as e:
            logger.error(f"Cross-encoder prediction failed: {e}")
            # Return documents unchanged on error
            return self._passthrough_results(documents, top_k)

        # Create results with scores
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(
                RerankResult(
                    doc_id=doc.get("doc_id", doc.get("id", f"doc_{i}")),
                    content=doc.get("content", ""),
                    original_score=doc.get("score", 0.0),
                    rerank_score=float(score),
                    original_rank=i + 1,
                    new_rank=0,  # Will be set after sorting
                    metadata=doc.get("metadata", {}),
                )
            )

        # Sort by rerank score descending
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Set new ranks and limit to top_k
        for i, result in enumerate(results[:top_k]):
            result.new_rank = i + 1

        return results[:top_k]

    def _passthrough_results(
        self, documents: List[Dict[str, Any]], top_k: int
    ) -> List[RerankResult]:
        """Create passthrough results on error."""
        results = []
        for i, doc in enumerate(documents[:top_k]):
            results.append(
                RerankResult(
                    doc_id=doc.get("doc_id", doc.get("id", f"doc_{i}")),
                    content=doc.get("content", ""),
                    original_score=doc.get("score", 0.0),
                    rerank_score=doc.get("score", 0.0),
                    original_rank=i + 1,
                    new_rank=i + 1,
                    metadata=doc.get("metadata", {}),
                )
            )
        return results

    def get_info(self) -> Dict[str, Any]:
        """Get reranker information."""
        return {
            "backend": RerankerBackend.LOCAL_CROSS_ENCODER.value,
            "model": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "loaded": self._model is not None,
        }


class CohereReranker(BaseReranker):
    """
    Cohere Rerank API integration.

    Models:
    - rerank-english-v2.0 (English)
    - rerank-multilingual-v2.0 (Multilingual)
    - rerank-english-v3.0 (Latest)

    Requires COHERE_API_KEY environment variable or config.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v2.0",
    ):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key
            model: Rerank model name
        """
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Get or create Cohere client."""
        if self._client is None:
            try:
                import cohere

                self._client = cohere.Client(self.api_key)
            except ImportError:
                raise ImportError("Please install cohere: pip install cohere")
        return self._client

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Rerank documents using Cohere API."""
        if not documents:
            return []

        client = self._get_client()

        # Extract document texts (max 4096 chars per doc)
        doc_texts = [doc.get("content", "")[:4096] for doc in documents]

        try:
            # Call Cohere rerank API
            response = client.rerank(
                query=query,
                documents=doc_texts,
                top_n=min(top_k, len(documents)),
                model=self.model,
            )

            # Build results
            results = []
            for i, result in enumerate(response.results):
                original_idx = result.index
                doc = documents[original_idx]

                results.append(
                    RerankResult(
                        doc_id=doc.get("doc_id", doc.get("id", f"doc_{original_idx}")),
                        content=doc.get("content", ""),
                        original_score=doc.get("score", 0.0),
                        rerank_score=result.relevance_score,
                        original_rank=original_idx + 1,
                        new_rank=i + 1,
                        metadata=doc.get("metadata", {}),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Cohere rerank failed: {e}")
            raise

    def get_info(self) -> Dict[str, Any]:
        """Get reranker information."""
        return {
            "backend": RerankerBackend.COHERE.value,
            "model": self.model,
            "api_configured": bool(self.api_key),
        }


class JinaReranker(BaseReranker):
    """
    Jina Rerank API integration.

    Models:
    - jina-reranker-v1-base-en
    - jina-reranker-v1-turbo-en (faster)
    - jina-reranker-v2-base-multilingual

    Requires JINA_API_KEY environment variable or config.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "jina-reranker-v1-base-en",
    ):
        """
        Initialize Jina reranker.

        Args:
            api_key: Jina API key
            model: Rerank model name
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.jina.ai/v1/rerank"

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Rerank documents using Jina API."""
        if not documents:
            return []

        try:
            import aiohttp
        except ImportError:
            raise ImportError("Please install aiohttp: pip install aiohttp")

        # Prepare request (max 8192 chars per doc)
        doc_texts = [doc.get("content", "")[:8192] for doc in documents]

        payload = {
            "model": self.model,
            "query": query,
            "documents": doc_texts,
            "top_n": min(top_k, len(documents)),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            # Build results
            results = []
            for i, result in enumerate(data.get("results", [])):
                original_idx = result.get("index", i)
                doc = documents[original_idx]

                results.append(
                    RerankResult(
                        doc_id=doc.get("doc_id", doc.get("id", f"doc_{original_idx}")),
                        content=doc.get("content", ""),
                        original_score=doc.get("score", 0.0),
                        rerank_score=result.get("relevance_score", 0.0),
                        original_rank=original_idx + 1,
                        new_rank=i + 1,
                        metadata=doc.get("metadata", {}),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Jina rerank failed: {e}")
            raise

    def get_info(self) -> Dict[str, Any]:
        """Get reranker information."""
        return {
            "backend": RerankerBackend.JINA.value,
            "model": self.model,
            "api_configured": bool(self.api_key),
        }


class PassThroughReranker(BaseReranker):
    """
    No-op reranker that returns documents unchanged.

    Used as fallback when no reranker is configured or available.
    """

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Return documents without reranking."""
        results = []
        for i, doc in enumerate(documents[:top_k]):
            results.append(
                RerankResult(
                    doc_id=doc.get("doc_id", doc.get("id", f"doc_{i}")),
                    content=doc.get("content", ""),
                    original_score=doc.get("score", 0.0),
                    rerank_score=doc.get("score", 0.0),
                    original_rank=i + 1,
                    new_rank=i + 1,
                    metadata=doc.get("metadata", {}),
                )
            )
        return results

    def get_info(self) -> Dict[str, Any]:
        """Get reranker information."""
        return {
            "backend": RerankerBackend.NONE.value,
            "model": "pass-through",
        }


class RerankerProvider:
    """
    Reranker provider with multiple backend support.

    Features:
    - Local cross-encoder (sentence-transformers)
    - Cohere Rerank API
    - Jina Rerank API
    - Graceful fallback to pass-through

    Usage:
        provider = RerankerProvider(
            backend=RerankerBackend.LOCAL_CROSS_ENCODER,
            device="cpu"
        )

        results = await provider.rerank(
            query="product manual",
            documents=[{"content": "...", "doc_id": "1"}],
            top_k=10
        )
    """

    def __init__(
        self,
        backend: RerankerBackend = RerankerBackend.LOCAL_CROSS_ENCODER,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
        preload: bool = True,
    ):
        """
        Initialize provider with specified backend.

        Args:
            backend: Which reranker backend to use
            model_name: Model name (backend-specific)
            api_key: API key for cloud backends
            device: Device for local models ('cpu' or 'cuda')
            batch_size: Batch size for local inference
            preload: If True, load local models immediately (avoids blocking during ops)
        """
        self.backend = backend
        self._reranker = self._create_reranker(
            backend, model_name, api_key, device, batch_size, preload
        )

    def _create_reranker(
        self,
        backend: RerankerBackend,
        model_name: Optional[str],
        api_key: Optional[str],
        device: str,
        batch_size: int,
        preload: bool = True,
    ) -> BaseReranker:
        """Create appropriate reranker instance."""

        if backend == RerankerBackend.LOCAL_CROSS_ENCODER:
            return LocalCrossEncoderReranker(
                model_name=model_name or "cross-encoder/stsb-roberta-large",
                device=device,
                batch_size=batch_size,
                preload=preload,
            )

        elif backend == RerankerBackend.COHERE:
            if not api_key:
                logger.warning(
                    "Cohere API key not provided, falling back to pass-through"
                )
                return PassThroughReranker()
            return CohereReranker(
                api_key=api_key,
                model=model_name or "rerank-english-v2.0",
            )

        elif backend == RerankerBackend.JINA:
            if not api_key:
                logger.warning(
                    "Jina API key not provided, falling back to pass-through"
                )
                return PassThroughReranker()
            return JinaReranker(
                api_key=api_key,
                model=model_name or "jina-reranker-v1-base-en",
            )

        else:
            return PassThroughReranker()

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Rerank documents using configured backend."""
        return await self._reranker.rerank(query, documents, top_k)

    def get_info(self) -> Dict[str, Any]:
        """Get reranker information."""
        return self._reranker.get_info()

    def inject_shared_model(self, model) -> None:
        """Inject a pre-loaded model from SharedModelPool into the inner reranker.

        Public facade so adapters don't access the private _reranker attribute.
        Only applicable to LOCAL_CROSS_ENCODER backend.
        """
        if hasattr(self._reranker, "set_shared_model"):
            self._reranker.set_shared_model(model)

    def get_model_name(self) -> str:
        """Return the currently loaded model name."""
        if self._reranker is None:
            return ""
        # LocalCrossEncoderReranker uses model_name, Cohere/Jina use model
        return getattr(self._reranker, "model_name", "") or getattr(self._reranker, "model", "")

    def get_api_key(self) -> Optional[str]:
        """Return the configured API key."""
        return getattr(self._reranker, "api_key", None)

    def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        info = self.get_info()
        return {
            "status": "healthy",
            **info,
        }


# ============================================================================
# Reranker Model Discovery Functions
# ============================================================================


def _scan_hf_cache_for_rerankers() -> List[Dict[str, Any]]:
    """
    Scan HuggingFace cache for downloaded reranker models.

    Uses huggingface_hub.scan_cache_dir() like vLLM does for dynamic discovery.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        logger.warning("huggingface_hub not installed, cannot scan cache")
        return []

    reranker_patterns = [
        "reranker",
        "cross-encoder",
        "bge-reranker",
        "ms-marco",
    ]

    models = []
    try:
        cache_info = scan_cache_dir()

        for repo in cache_info.repos:
            repo_id = repo.repo_id
            repo_id_lower = repo_id.lower()

            # Check if it's a reranker model
            is_reranker = any(
                pattern in repo_id_lower
                for pattern in reranker_patterns
            )

            if is_reranker:
                size_gb = round(repo.size_on_disk / (1024 ** 3), 2)
                models.append({
                    "id": repo_id,
                    "name": repo_id.split("/")[-1],
                    "provider": "sentence-transformers",
                    "type": "reranker",
                    "available": True,
                    "downloaded": True,
                    "size_gb": size_gb,
                    "description": f"Downloaded reranker ({size_gb} GB)",
                })

    except Exception as e:
        logger.warning(f"Failed to scan HF cache for rerankers: {e}")

    return models


async def list_all_reranker_models(
    ollama_base_url: str = "http://localhost:11434",
    cohere_api_key: Optional[str] = None,
    jina_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List all available reranker models across all providers.

    Discovery methods:
    - sentence-transformers: Scan HuggingFace cache for downloaded models
    - ollama: Query /api/tags for installed models with reranker patterns
    - cohere: Known API models (available if API key configured)
    - jina: Known API models (available if API key configured)

    Args:
        ollama_base_url: Ollama server URL for dynamic discovery
        cohere_api_key: Cohere API key (enables Cohere models)
        jina_api_key: Jina API key (enables Jina models)

    Returns:
        Dict with models list, providers status, and counts
    """
    import httpx

    all_models = []
    providers_status = {}

    # 1. Sentence-Transformers: Scan HuggingFace cache (DYNAMIC)
    st_models = _scan_hf_cache_for_rerankers()

    # Also add known models that can be downloaded (not yet in cache)
    downloaded_ids = {m["id"] for m in st_models}
    for known_model in SENTENCE_TRANSFORMERS_RERANKERS:
        if known_model["id"] not in downloaded_ids:
            model = dict(known_model)
            model["downloaded"] = False
            model["available"] = True  # Can be downloaded on demand
            st_models.append(model)

    all_models.extend(st_models)
    downloaded_count = sum(1 for m in st_models if m.get("downloaded", False))
    providers_status["sentence-transformers"] = {
        "available": True,
        "model_count": len(st_models),
        "downloaded_count": downloaded_count,
        "status": "ready",
    }

    # 2. Ollama reranker models (DYNAMIC via API)
    ollama_models = []
    ollama_available = False
    ollama_error = None

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ollama_base_url}/api/tags")

            if response.status_code == 200:
                data = response.json()
                installed_models = data.get("models", [])
                ollama_available = True

                for model_info in installed_models:
                    model_name = model_info.get("name", "")
                    model_name_lower = model_name.lower()

                    # Check if it's a reranker model
                    is_reranker = any(
                        pattern in model_name_lower
                        for pattern in OLLAMA_RERANKER_PATTERNS
                    )

                    if is_reranker:
                        # Get size from model info
                        size_bytes = model_info.get("size", 0)
                        size_gb = round(size_bytes / (1024 ** 3), 2) if size_bytes else None

                        ollama_models.append({
                            "id": model_name,
                            "name": model_name.split(":")[0].split("/")[-1],
                            "provider": "ollama",
                            "type": "reranker",
                            "available": True,
                            "downloaded": True,
                            "size_gb": size_gb,
                            "description": f"Ollama reranker model",
                        })
    except Exception as e:
        ollama_error = str(e)
        logger.warning(f"Failed to query Ollama for reranker models: {e}")

    all_models.extend(ollama_models)
    providers_status["ollama"] = {
        "available": ollama_available,
        "model_count": len(ollama_models),
        "downloaded_count": len(ollama_models),
        "status": "ready" if ollama_available else "unavailable",
        "error": ollama_error,
    }

    # 3. Cohere models (API-based, available if key configured)
    cohere_models = []
    for m in COHERE_RERANKERS:
        model = dict(m)
        model["available"] = bool(cohere_api_key)
        model["downloaded"] = None  # N/A for API models
        cohere_models.append(model)

    all_models.extend(cohere_models)
    providers_status["cohere"] = {
        "available": bool(cohere_api_key),
        "model_count": len(cohere_models),
        "status": "ready" if cohere_api_key else "no_api_key",
        "reason": None if cohere_api_key else "No API key configured",
    }

    # 4. Jina models (API-based, available if key configured)
    jina_models = []
    for m in JINA_RERANKERS:
        model = dict(m)
        model["available"] = bool(jina_api_key)
        model["downloaded"] = None  # N/A for API models
        jina_models.append(model)

    all_models.extend(jina_models)
    providers_status["jina"] = {
        "available": bool(jina_api_key),
        "model_count": len(jina_models),
        "status": "ready" if jina_api_key else "no_api_key",
        "reason": None if jina_api_key else "No API key configured",
    }

    # Calculate totals
    total_count = len(all_models)
    available_count = sum(1 for m in all_models if m.get("available", False))
    downloaded_total = sum(1 for m in all_models if m.get("downloaded", False))

    return {
        "models": all_models,
        "providers": providers_status,
        "total_count": total_count,
        "available_count": available_count,
        "downloaded_count": downloaded_total,
    }
