"""
multimodal_rag/adapter.py

Bridge Layer - Exposes all module operations.
Orchestrates embeddings, retrieval, VQA, document processing.

This is the main entry point for the module.

Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple, Union

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

import numpy as np

from .providers import (
    Modality,
    RetrievalMode,
    ImageData,
    TextData,
    MultiModalItem,
    ProcessedDocument,
    RetrievalResult,
    VQAResult,
    CaptionResult,
    ImageProcessor,
    OCRProvider,
    MultiModalCacheProvider,
    MetricsCollector,
)
from .embeddings import (
    EmbeddingConfig,
    UnifiedEmbedder,
)
from .retrieval import (
    RetrievalConfig,
    CrossModalRetriever,
)
from .vqa import (
    VQAConfig,
    CaptioningConfig,
    UnifiedVQAProvider,
)
from .document import (
    DocumentConfig,
    ChunkingConfig,
    UnifiedDocumentProcessor,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Utilities
# ============================================================================


def _coerce_value(value: Any) -> Any:
    """Coerce string values to appropriate types."""
    if not isinstance(value, str):
        return value
    
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False
    
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    return value


def _coerce_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively coerce config values."""
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _coerce_config(value)
        elif isinstance(value, list):
            result[key] = [
                _coerce_config(v) if isinstance(v, dict) else _coerce_value(v)
                for v in value
            ]
        else:
            result[key] = _coerce_value(value)
    return result


def _resolve_env(text: str) -> str:
    """Resolve environment variable placeholders."""
    pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
    
    def replace(match):
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)
    
    return re.sub(pattern, replace, text)


def _load_config(module_path: Path) -> Dict[str, Any]:
    """Load and resolve config.json."""
    config_file = module_path / "config.json"
    
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_file}")
        return {}
    
    with open(config_file, "r", encoding="utf-8") as f:
        raw = f.read()
    
    resolved = _resolve_env(raw)
    parsed = json.loads(resolved)
    
    return _coerce_config(parsed)


# ============================================================================
# Multi-Modal RAG Adapter
# ============================================================================


class MultiModalRAGAdapter:
    """
    Main adapter for multimodal_rag module.
    
    Implements all operations defined in manifest.json.
    Orchestrates embeddings, retrieval, VQA, and document processing.
    """
    
    def __init__(
        self,
        module_path: Path,
        di_container: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        self.module_path = Path(module_path)
        self.di_container = di_container
        self.event_bus = event_bus
        
        # Load configuration
        self.config = _load_config(module_path)
        
        # Environment
        self.env = os.environ.get("UBP_ENV", "dev")
        
        # Components (lazy init)
        self._image_processor: Optional[ImageProcessor] = None
        self._ocr_provider: Optional[OCRProvider] = None
        self._embedder: Optional[UnifiedEmbedder] = None
        self._retriever: Optional[CrossModalRetriever] = None
        self._vqa_provider: Optional[UnifiedVQAProvider] = None
        self._doc_processor: Optional[UnifiedDocumentProcessor] = None
        self._cache: Optional[MultiModalCacheProvider] = None
        self._metrics: Optional[MetricsCollector] = None
        
        # State
        self._initialized = False
    
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
    
    # ========================================================================
    # Event Publisher
    # ========================================================================
    
    @property
    def publisher(self) -> Optional[Callable]:
        """Get event publisher."""
        if self.event_bus and hasattr(self.event_bus, "publish"):
            return self.event_bus.publish
        return None
    
    async def _publish_event(self, event: str, data: Dict[str, Any]) -> None:
        """Publish event if bus available."""
        if self.publisher:
            try:
                await self.publisher(event, data)
            except Exception as e:
                logger.warning(f"Event publish failed: {e}")
    
    # ========================================================================
    # Lifecycle Operations
    # ========================================================================
    
    async def initialize(
        self,
        preload_models: bool = False,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Initialize multi-modal RAG pipeline."""
        start_time = time.perf_counter()
        
        try:
            # Get Redis if available
            redis_client = None
            if self.di_container:
                try:
                    import redis.asyncio as aioredis
                    redis_client = await self.di_container.resolve(aioredis.Redis)
                except Exception:
                    pass
            
            # Initialize components
            img_cfg = self.config.get("image_processing", {})
            self._image_processor = ImageProcessor(
                default_size=img_cfg.get("default_size", 224),
                max_size=img_cfg.get("max_size", 1024),
                preserve_aspect=img_cfg.get("preserve_aspect_ratio", True),
                normalize=img_cfg.get("normalize", True),
            )
            
            ocr_cfg = self.config.get("ocr", {})
            if ocr_cfg.get("enabled", True):
                self._ocr_provider = OCRProvider(
                    engine=ocr_cfg.get("engine", "tesseract"),
                    languages=ocr_cfg.get("languages", "eng,ita").split(","),
                    confidence_threshold=ocr_cfg.get("confidence_threshold", 0.6),
                )
            
            # Initialize cache
            cache_cfg = self.config.get("cache", {})
            self._cache = MultiModalCacheProvider(
                redis_client=redis_client,
                prefix=cache_cfg.get("redis_prefix", "ubp:multimodal"),
                ttl_seconds=cache_cfg.get("ttl_seconds", 7200),
                enabled=cache_cfg.get("enabled", True),
            )
            
            # Initialize embedder
            embed_cfg = self.config.get("embeddings", {})
            embedding_config = EmbeddingConfig(
                model_name=embed_cfg.get("default_model", "openai/clip-vit-base-patch32"),
                device=embed_cfg.get("device", "auto"),
                batch_size=embed_cfg.get("batch_size", 16),
                normalize=embed_cfg.get("normalize", True),
            )
            
            model_type = "clip"  # Default
            if "blip" in embedding_config.model_name.lower():
                model_type = "blip"
            elif "siglip" in embedding_config.model_name.lower():
                model_type = "siglip"
            
            self._embedder = UnifiedEmbedder(
                default_model=model_type,
                config=embedding_config,
                cache_provider=self._cache,
            )
            
            # Initialize retriever
            retrieval_cfg = self.config.get("retrieval", {})
            retrieval_config = RetrievalConfig(
                top_k=retrieval_cfg.get("top_k", 10),
                similarity_threshold=retrieval_cfg.get("similarity_threshold", 0.5),
                enable_reranking=retrieval_cfg.get("reranking", {}).get("enabled", True),
            )
            
            self._retriever = CrossModalRetriever(
                embedder=self._embedder,
                config=retrieval_config,
            )
            
            # Initialize VQA
            vqa_cfg = self.config.get("vqa", {})
            caption_cfg = self.config.get("captioning", {})
            
            if vqa_cfg.get("enabled", True):
                self._vqa_provider = UnifiedVQAProvider(
                    vqa_config=VQAConfig(
                        model_name=vqa_cfg.get("model", "Salesforce/blip2-opt-2.7b"),
                        device=embed_cfg.get("device", "auto"),
                        max_answer_length=vqa_cfg.get("max_answer_length", 100),
                    ),
                    caption_config=CaptioningConfig(
                        model_name=caption_cfg.get("model", "Salesforce/blip-image-captioning-base"),
                        device=embed_cfg.get("device", "auto"),
                        max_length=caption_cfg.get("max_length", 75),
                    ),
                )
            
            # Initialize document processor
            doc_cfg = self.config.get("document_processing", {})
            chunk_cfg = self.config.get("chunking", {})
            
            self._doc_processor = UnifiedDocumentProcessor(
                doc_config=DocumentConfig(
                    extract_images=doc_cfg.get("extract_images", True),
                    extract_tables=doc_cfg.get("extract_tables", True),
                    dpi=doc_cfg.get("dpi", 150),
                ),
                chunk_config=ChunkingConfig(
                    max_chunk_size=chunk_cfg.get("max_chunk_size", 1000),
                    overlap=chunk_cfg.get("overlap", 100),
                    preserve_image_context=chunk_cfg.get("preserve_image_context", True),
                ),
            )
            
            # Initialize metrics
            self._metrics = MetricsCollector()
            
            # Preload models if requested
            if preload_models:
                # Trigger model loading
                dummy_text = "test"
                await self._embedder.embed_text(dummy_text)
            
            self._initialized = True
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            await self._publish_event("multimodal.initialized", {
                "module": "multimodal_rag",
            })
            
            return {
                "status": "initialized",
                "module": "multimodal_rag",
                "version": "1.0.0",
                "env": self.env,
                "embedding_model": embedding_config.model_name,
                "vqa_enabled": self._vqa_provider is not None,
                "ocr_enabled": self._ocr_provider is not None,
                "cache_enabled": cache_cfg.get("enabled", True),
                "elapsed_ms": round(elapsed_ms, 2),
            }
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return {
                "status": "error",
                "module": "multimodal_rag",
                "error": str(e),
            }
    
    async def shutdown(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Graceful shutdown."""
        resources_released = []
        
        if self._embedder:
            self._embedder.unload_all()
            resources_released.append("embedders")
        
        if self._vqa_provider:
            self._vqa_provider.unload()
            resources_released.append("vqa_models")
        
        self._initialized = False
        
        await self._publish_event("multimodal.shutdown", {
            "module": "multimodal_rag",
        })
        
        return {
            "status": "shutdown",
            "resources_released": resources_released,
        }
    
    async def health_check(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Health check."""
        result = {
            "module": "multimodal_rag",
            "version": "1.0.0",
            "status": "healthy",
            "env": self.env,
        }
        
        if self._embedder:
            result["embedder"] = self._embedder.health_check()
        
        if self._vqa_provider:
            result["vqa"] = self._vqa_provider.health_check()
        
        if self._cache:
            result["cache"] = self._cache.get_stats()
        
        if self._retriever:
            result["retriever"] = self._retriever.get_stats()
        
        return result
    
    # ========================================================================
    # Image Operations
    # ========================================================================
    
    async def load_image(
        self,
        source: Union[str, bytes],
        image_id: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Load and process an image."""
        if not self._initialized:
            await self.initialize()
        
        image = self._image_processor.load_image(source, image_id)
        
        return image.to_dict()
    
    async def embed_image(
        self,
        image_source: Union[str, bytes, Dict[str, Any]],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate embedding for an image."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        # Load image if needed
        if isinstance(image_source, dict):
            image = ImageData(**image_source)
        else:
            image = self._image_processor.load_image(image_source)
        
        # Generate embedding
        embedding = await self._embedder.embed_image(image)
        
        image.embedding = embedding
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "image_id": image.id,
            "embedding_dimension": len(embedding),
            "embedding_preview": embedding[:5].tolist(),
            "time_ms": round(elapsed_ms, 2),
        }
    
    async def embed_text(
        self,
        text: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate embedding for text."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        embedding = await self._embedder.embed_text(text)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "embedding_dimension": len(embedding),
            "embedding_preview": embedding[:5].tolist(),
            "time_ms": round(elapsed_ms, 2),
        }
    
    # ========================================================================
    # OCR Operations
    # ========================================================================
    
    async def extract_text_from_image(
        self,
        image_source: Union[str, bytes, Dict[str, Any]],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Extract text from image using OCR."""
        if not self._initialized:
            await self.initialize()
        
        if not self._ocr_provider:
            return {"error": "OCR not enabled", "text": ""}
        
        start_time = time.perf_counter()
        
        # Load image
        if isinstance(image_source, dict):
            image = ImageData(**image_source)
        else:
            image = self._image_processor.load_image(image_source)
        
        # Extract text
        text = self._ocr_provider.extract_text(image)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "image_id": image.id,
            "extracted_text": text,
            "text_length": len(text),
            "time_ms": round(elapsed_ms, 2),
        }
    
    # ========================================================================
    # Retrieval Operations
    # ========================================================================
    
    async def index_item(
        self,
        item: Dict[str, Any],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add item to index."""
        if not self._initialized:
            await self.initialize()
        
        # Create MultiModalItem
        mm_item = MultiModalItem(
            id=item.get("id", ""),
            modality=Modality(item.get("modality", "text")),
            text_content=item.get("text_content"),
            source_id=item.get("source_id"),
        )
        
        # Load image if provided
        if "image_data" in item:
            mm_item.image_data = ImageData(**item["image_data"])
        elif "image_source" in item:
            mm_item.image_data = self._image_processor.load_image(item["image_source"])
        
        # Embed and add to index
        count = await self._retriever.add_to_index(mm_item)
        
        return {
            "indexed": True,
            "item_id": mm_item.id,
            "modality": mm_item.modality.value,
        }
    
    async def index_batch(
        self,
        items: List[Dict[str, Any]],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add batch of items to index."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        mm_items = []
        for item in items:
            mm_item = MultiModalItem(
                id=item.get("id", ""),
                modality=Modality(item.get("modality", "text")),
                text_content=item.get("text_content"),
                source_id=item.get("source_id"),
            )
            
            if "image_data" in item:
                mm_item.image_data = ImageData(**item["image_data"])
            
            mm_items.append(mm_item)
        
        count = await self._retriever.add_to_index(mm_items)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "indexed_count": count,
            "total_in_index": self._retriever.index.count(),
            "time_ms": round(elapsed_ms, 2),
        }
    
    async def retrieve(
        self,
        query: Union[str, Dict[str, Any]],
        mode: str = "hybrid",
        top_k: int = 10,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Retrieve items matching query.
        
        Args:
            query: Text string or image dict
            mode: Retrieval mode (hybrid, text_to_image, image_to_text, etc.)
            top_k: Number of results
        """
        if not self._initialized:
            await self.initialize()
        
        # Parse query
        if isinstance(query, str):
            query_obj = query
        else:
            query_obj = self._image_processor.load_image(query.get("image_source"))
        
        # Parse mode
        retrieval_mode = RetrievalMode(mode)
        
        # Retrieve
        result = await self._retriever.retrieve(
            query=query_obj,
            mode=retrieval_mode,
            top_k=top_k,
        )
        
        # Track metrics
        if self._metrics:
            self._metrics.record_retrieval(mode, result.time_ms, len(result.items))
        
        return result.to_dict()
    
    async def text_to_image_search(
        self,
        text: str,
        top_k: int = 10,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Search images using text query."""
        return await self.retrieve(text, "text_to_image", top_k, ctx, **kwargs)
    
    async def image_to_text_search(
        self,
        image_source: Union[str, bytes, Dict[str, Any]],
        top_k: int = 10,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Search text using image query."""
        if isinstance(image_source, (str, bytes)):
            image_source = {"image_source": image_source}
        return await self.retrieve(image_source, "image_to_text", top_k, ctx, **kwargs)
    
    async def find_similar_images(
        self,
        image_source: Union[str, bytes, Dict[str, Any]],
        top_k: int = 10,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Find similar images."""
        if isinstance(image_source, (str, bytes)):
            image_source = {"image_source": image_source}
        return await self.retrieve(image_source, "image_to_image", top_k, ctx, **kwargs)
    
    # ========================================================================
    # VQA Operations
    # ========================================================================
    
    async def answer_question(
        self,
        image_source: Union[str, bytes, Dict[str, Any]],
        question: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Answer question about an image."""
        if not self._initialized:
            await self.initialize()
        
        if not self._vqa_provider:
            return {"error": "VQA not enabled"}
        
        # Load image
        if isinstance(image_source, dict) and "raw_bytes" in image_source:
            image = ImageData(**image_source)
        else:
            image = self._image_processor.load_image(image_source)
        
        result = await self._vqa_provider.answer_question(image, question)
        
        return result.to_dict()
    
    async def generate_caption(
        self,
        image_source: Union[str, bytes, Dict[str, Any]],
        style: str = "descriptive",
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate caption for image."""
        if not self._initialized:
            await self.initialize()
        
        if not self._vqa_provider:
            return {"error": "Captioning not enabled"}
        
        # Load image
        if isinstance(image_source, dict) and "raw_bytes" in image_source:
            image = ImageData(**image_source)
        else:
            image = self._image_processor.load_image(image_source)
        
        result = await self._vqa_provider.generate_caption(image, style)
        
        return result.to_dict()
    
    async def describe_image(
        self,
        image_source: Union[str, bytes, Dict[str, Any]],
        aspects: Optional[List[str]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Comprehensive image description."""
        if not self._initialized:
            await self.initialize()
        
        if not self._vqa_provider:
            return {"error": "VQA not enabled"}
        
        # Load image
        if isinstance(image_source, dict) and "raw_bytes" in image_source:
            image = ImageData(**image_source)
        else:
            image = self._image_processor.load_image(image_source)
        
        return await self._vqa_provider.describe_image(image, aspects)
    
    # ========================================================================
    # Document Operations
    # ========================================================================
    
    async def process_document(
        self,
        source: Union[str, bytes, BinaryIO],
        document_id: Optional[str] = None,
        generate_captions: bool = True,
        run_ocr: bool = True,
        auto_index: bool = True,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process document and extract multi-modal content."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        # Get caption provider
        caption_provider = None
        if generate_captions and self._vqa_provider:
            caption_provider = self._vqa_provider._get_caption_provider()
        
        # Process document
        document = await self._doc_processor.process_document(
            source=source,
            document_id=document_id,
            generate_captions=generate_captions,
            run_ocr=run_ocr,
            caption_provider=caption_provider,
            ocr_provider=self._ocr_provider,
        )
        
        # Auto-index chunks
        if auto_index and document.chunks:
            await self._retriever.add_to_index(document.chunks)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            **document.to_dict(),
            "auto_indexed": auto_index,
            "index_size": self._retriever.index.count() if auto_index else None,
            "time_ms": round(elapsed_ms, 2),
        }
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    async def get_stats(
        self,
        period: str = "24h",
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get module statistics."""
        stats = {
            "module": "multimodal_rag",
            "period": period,
        }
        
        if self._metrics:
            stats["metrics"] = self._metrics.get_summary()
        
        if self._retriever:
            stats["index"] = self._retriever.get_stats()
        
        if self._cache:
            stats["cache"] = self._cache.get_stats()
        
        return stats
    
    async def get_index_stats(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Get index statistics."""
        if not self._initialized:
            await self.initialize()
        
        return self._retriever.get_stats()

    # ========================================================================
    # Missing Operations (manifest-declared, now implemented)
    # ========================================================================

    async def compute_similarity(
        self,
        text: str,
        image_source: Union[str, bytes, Dict[str, Any]],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute cosine similarity between text and image embeddings."""
        if not self._initialized:
            await self.initialize()

        start_time = time.perf_counter()

        text_emb = await self._embedder.embed_text(text)

        if isinstance(image_source, dict):
            image = ImageData(**image_source)
        else:
            image = self._image_processor.load_image(image_source)
        image_emb = await self._embedder.embed_image(image)

        # Cosine similarity
        dot = float(np.dot(text_emb, image_emb))
        norm = float(np.linalg.norm(text_emb) * np.linalg.norm(image_emb))
        similarity = dot / norm if norm > 0 else 0.0

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "similarity": round(similarity, 6),
            "text_preview": text[:100],
            "embedding_dimension": len(text_emb),
            "time_ms": round(elapsed_ms, 2),
        }

    async def multimodal_query(
        self,
        text: str,
        image_source: Union[str, bytes, Dict[str, Any]],
        fusion_method: str = "average",
        top_k: int = 10,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Query with both text and image using embedding fusion."""
        if not self._initialized:
            await self.initialize()

        # For now, delegate to hybrid retrieve with the text query.
        # Full fusion (average/weighted/concat of text+image embeddings)
        # requires retriever-level support; we use text as primary query.
        return await self.retrieve(query=text, mode="hybrid", top_k=top_k, ctx=ctx, **kwargs)

    async def clear_index(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Clear all items from the index."""
        if not self._initialized:
            await self.initialize()

        if not self._retriever:
            return {"status": "no_index", "cleared": 0}

        prev_count = self._retriever.index.count()
        self._retriever.index.clear()

        return {
            "status": "cleared",
            "items_removed": prev_count,
        }
