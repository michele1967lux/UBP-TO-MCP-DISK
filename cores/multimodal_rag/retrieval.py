"""
multimodal_rag/retrieval.py

Cross-modal retrieval strategies.

Provides:
- CrossModalRetriever: Main retrieval orchestrator
- TextToImageRetrieval: Find images from text query
- ImageToTextRetrieval: Find text from image query
- ImageToImageRetrieval: Find similar images
- HybridRetrieval: Combined retrieval
- UnifiedRetrieval: Single embedding space retrieval
- ReRanker: Multi-modal reranking

Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .providers import (
    MultiModalItem,
    ImageData,
    TextData,
    RetrievalResult,
    Modality,
    RetrievalMode,
    FusionMethod,
)
from .embeddings import UnifiedEmbedder

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    top_k: int = 10
    similarity_threshold: float = 0.5
    rerank_top_k: int = 20
    enable_reranking: bool = True
    fusion_method: FusionMethod = FusionMethod.LATE
    fusion_weights: Dict[str, float] = field(default_factory=lambda: {
        "text": 0.5,
        "image": 0.5,
    })


# ============================================================================
# Base Retriever
# ============================================================================


class BaseRetriever(ABC):
    """Base class for retrievers."""
    
    def __init__(
        self,
        embedder: UnifiedEmbedder,
        config: RetrievalConfig,
    ):
        self.embedder = embedder
        self.config = config
    
    @abstractmethod
    async def retrieve(
        self,
        query: Union[str, ImageData],
        index: Any,
        top_k: Optional[int] = None,
    ) -> List[MultiModalItem]:
        """Retrieve items matching query."""
        pass
    
    def _compute_similarity(
        self,
        query_embedding: np.ndarray,
        item_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarities."""
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Handle batch
        if len(item_embeddings.shape) == 1:
            item_embeddings = item_embeddings.reshape(1, -1)
        
        item_norms = item_embeddings / np.linalg.norm(
            item_embeddings, axis=1, keepdims=True
        )
        
        similarities = np.dot(item_norms, query_norm)
        return similarities
    
    def _filter_by_threshold(
        self,
        items: List[MultiModalItem],
        scores: np.ndarray,
        threshold: float,
    ) -> List[MultiModalItem]:
        """Filter items by score threshold."""
        filtered = []
        for item, score in zip(items, scores):
            if score >= threshold:
                item.relevance_score = float(score)
                filtered.append(item)
        return filtered


# ============================================================================
# Text to Image Retriever
# ============================================================================


class TextToImageRetriever(BaseRetriever):
    """
    Retrieve images from text query.
    
    Uses text embedding to search image embedding space.
    """
    
    async def retrieve(
        self,
        query: str,
        index: "MultiModalIndex",
        top_k: Optional[int] = None,
    ) -> List[MultiModalItem]:
        """Retrieve images matching text query."""
        top_k = top_k or self.config.top_k
        
        # Embed query
        query_embedding = await self.embedder.embed_text(query)
        
        # Search image embeddings
        results = await index.search(
            query_embedding=query_embedding,
            modality_filter=Modality.IMAGE,
            top_k=top_k,
        )
        
        # Filter by threshold
        if self.config.similarity_threshold > 0:
            results = [
                r for r in results
                if r.relevance_score >= self.config.similarity_threshold
            ]
        
        return results


# ============================================================================
# Image to Text Retriever
# ============================================================================


class ImageToTextRetriever(BaseRetriever):
    """
    Retrieve text from image query.
    
    Uses image embedding to search text embedding space.
    """
    
    async def retrieve(
        self,
        query: ImageData,
        index: "MultiModalIndex",
        top_k: Optional[int] = None,
    ) -> List[MultiModalItem]:
        """Retrieve text matching image query."""
        top_k = top_k or self.config.top_k
        
        # Embed image
        query_embedding = await self.embedder.embed_image(query)
        
        # Search text embeddings
        results = await index.search(
            query_embedding=query_embedding,
            modality_filter=Modality.TEXT,
            top_k=top_k,
        )
        
        return results


# ============================================================================
# Image to Image Retriever
# ============================================================================


class ImageToImageRetriever(BaseRetriever):
    """
    Retrieve similar images.
    
    Visual similarity search.
    """
    
    async def retrieve(
        self,
        query: ImageData,
        index: "MultiModalIndex",
        top_k: Optional[int] = None,
    ) -> List[MultiModalItem]:
        """Retrieve images similar to query image."""
        top_k = top_k or self.config.top_k
        
        # Embed image
        query_embedding = await self.embedder.embed_image(query)
        
        # Search image embeddings
        results = await index.search(
            query_embedding=query_embedding,
            modality_filter=Modality.IMAGE,
            top_k=top_k,
        )
        
        return results


# ============================================================================
# Hybrid Retriever
# ============================================================================


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining multiple modalities.
    
    Supports:
    - Text query → Text + Image results
    - Image query → Text + Image results
    - Multi-query fusion
    """
    
    async def retrieve(
        self,
        query: Union[str, ImageData],
        index: "MultiModalIndex",
        top_k: Optional[int] = None,
        include_modalities: Optional[List[Modality]] = None,
    ) -> List[MultiModalItem]:
        """Hybrid retrieval across modalities."""
        top_k = top_k or self.config.top_k
        include_modalities = include_modalities or [Modality.TEXT, Modality.IMAGE]
        
        # Embed query
        if isinstance(query, str):
            query_embedding = await self.embedder.embed_text(query)
        else:
            query_embedding = await self.embedder.embed_image(query)
        
        # Search all modalities
        all_results = []
        
        for modality in include_modalities:
            results = await index.search(
                query_embedding=query_embedding,
                modality_filter=modality,
                top_k=top_k,
            )
            all_results.extend(results)
        
        # Sort by relevance
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Deduplicate
        seen_ids = set()
        unique_results = []
        for item in all_results:
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                unique_results.append(item)
        
        return unique_results[:top_k]


# ============================================================================
# Unified Retriever
# ============================================================================


class UnifiedRetriever(BaseRetriever):
    """
    Unified embedding space retrieval.
    
    Uses combined text+image embeddings for retrieval.
    """
    
    async def retrieve(
        self,
        query: Union[str, ImageData, Tuple[str, ImageData]],
        index: "MultiModalIndex",
        top_k: Optional[int] = None,
    ) -> List[MultiModalItem]:
        """Retrieve from unified embedding space."""
        top_k = top_k or self.config.top_k
        
        # Create unified query embedding
        if isinstance(query, tuple):
            # Both text and image
            text_query, image_query = query
            text_emb = await self.embedder.embed_text(text_query)
            image_emb = await self.embedder.embed_image(image_query)
            query_embedding = (text_emb + image_emb) / 2
        elif isinstance(query, str):
            query_embedding = await self.embedder.embed_text(query)
        else:
            query_embedding = await self.embedder.embed_image(query)
        
        # Search unified embeddings
        results = await index.search(
            query_embedding=query_embedding,
            modality_filter=None,  # All modalities
            top_k=top_k,
            use_unified=True,
        )
        
        return results


# ============================================================================
# Cross-Modal Reranker
# ============================================================================


class CrossModalReranker:
    """
    Reranks results using cross-modal signals.
    
    Combines:
    - Text-text similarity
    - Image-image similarity
    - Cross-modal matching
    """
    
    def __init__(
        self,
        embedder: UnifiedEmbedder,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.embedder = embedder
        self.weights = weights or {
            "text_text": 0.3,
            "image_image": 0.3,
            "cross_modal": 0.4,
        }
    
    async def rerank(
        self,
        query_text: Optional[str],
        query_image: Optional[ImageData],
        items: List[MultiModalItem],
        top_k: int = 10,
    ) -> List[MultiModalItem]:
        """Rerank items using multi-modal signals."""
        if not items:
            return []
        
        # Get query embeddings
        query_text_emb = None
        query_image_emb = None
        
        if query_text:
            query_text_emb = await self.embedder.embed_text(query_text)
        if query_image:
            query_image_emb = await self.embedder.embed_image(query_image)
        
        # Score each item
        for item in items:
            scores = []
            
            # Text-text similarity
            if query_text_emb is not None and item.text_embedding is not None:
                tt_score = self._cosine_similarity(query_text_emb, item.text_embedding)
                scores.append(("text_text", tt_score))
            
            # Image-image similarity
            if query_image_emb is not None and item.image_embedding is not None:
                ii_score = self._cosine_similarity(query_image_emb, item.image_embedding)
                scores.append(("image_image", ii_score))
            
            # Cross-modal similarity
            if query_text_emb is not None and item.image_embedding is not None:
                ti_score = self._cosine_similarity(query_text_emb, item.image_embedding)
                scores.append(("cross_modal", ti_score))
            
            if query_image_emb is not None and item.text_embedding is not None:
                it_score = self._cosine_similarity(query_image_emb, item.text_embedding)
                scores.append(("cross_modal", it_score))
            
            # Combine scores
            if scores:
                total_weight = 0
                weighted_score = 0
                
                for score_type, score in scores:
                    weight = self.weights.get(score_type, 0.25)
                    weighted_score += weight * score
                    total_weight += weight
                
                item.relevance_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Sort by score
        items.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return items[:top_k]
    
    def _cosine_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> float:
        """Compute cosine similarity."""
        norm1 = emb1 / np.linalg.norm(emb1)
        norm2 = emb2 / np.linalg.norm(emb2)
        return float(np.dot(norm1, norm2))


# ============================================================================
# Multi-Modal Index (Abstract)
# ============================================================================


class MultiModalIndex:
    """
    Multi-modal vector index.
    
    Stores and retrieves multi-modal items.
    Supports modality-specific filtering.
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._items: Dict[str, MultiModalItem] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._modality_index: Dict[Modality, List[str]] = {
            m: [] for m in Modality
        }
    
    async def add(self, item: MultiModalItem) -> None:
        """Add item to index."""
        self._items[item.id] = item
        
        # Store embedding
        embedding = item.get_embedding()
        if embedding is not None:
            self._embeddings[item.id] = embedding
        
        # Update modality index
        self._modality_index[item.modality].append(item.id)
    
    async def add_batch(self, items: List[MultiModalItem]) -> None:
        """Add batch of items."""
        for item in items:
            await self.add(item)
    
    async def search(
        self,
        query_embedding: np.ndarray,
        modality_filter: Optional[Modality] = None,
        top_k: int = 10,
        use_unified: bool = False,
    ) -> List[MultiModalItem]:
        """Search index."""
        # Get candidate IDs
        if modality_filter:
            candidate_ids = self._modality_index.get(modality_filter, [])
        else:
            candidate_ids = list(self._items.keys())
        
        if not candidate_ids:
            return []
        
        # Get embeddings
        embeddings = []
        valid_ids = []
        
        for item_id in candidate_ids:
            if item_id in self._embeddings:
                embeddings.append(self._embeddings[item_id])
                valid_ids.append(item_id)
        
        if not embeddings:
            return []
        
        embeddings = np.array(embeddings)
        
        # Compute similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        emb_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = np.dot(emb_norms, query_norm)
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            item_id = valid_ids[idx]
            item = self._items[item_id]
            item.relevance_score = float(similarities[idx])
            results.append(item)
        
        return results
    
    def get(self, item_id: str) -> Optional[MultiModalItem]:
        """Get item by ID."""
        return self._items.get(item_id)
    
    def count(self, modality: Optional[Modality] = None) -> int:
        """Count items."""
        if modality:
            return len(self._modality_index.get(modality, []))
        return len(self._items)
    
    def clear(self) -> None:
        """Clear index."""
        self._items.clear()
        self._embeddings.clear()
        for m in Modality:
            self._modality_index[m] = []


# ============================================================================
# Cross-Modal Retriever (Main Orchestrator)
# ============================================================================


class CrossModalRetriever:
    """
    Main cross-modal retrieval orchestrator.
    
    Coordinates:
    - Multiple retrieval strategies
    - Result fusion
    - Reranking
    """
    
    def __init__(
        self,
        embedder: UnifiedEmbedder,
        config: Optional[RetrievalConfig] = None,
    ):
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        
        # Initialize retrievers
        self._text_to_image = TextToImageRetriever(embedder, self.config)
        self._image_to_text = ImageToTextRetriever(embedder, self.config)
        self._image_to_image = ImageToImageRetriever(embedder, self.config)
        self._hybrid = HybridRetriever(embedder, self.config)
        self._unified = UnifiedRetriever(embedder, self.config)
        
        # Reranker
        self._reranker = CrossModalReranker(embedder)
        
        # Index
        self._index = MultiModalIndex(embedder.dimension)
    
    @property
    def index(self) -> MultiModalIndex:
        return self._index
    
    async def add_to_index(
        self,
        items: Union[MultiModalItem, List[MultiModalItem]],
    ) -> int:
        """Add items to index."""
        if isinstance(items, MultiModalItem):
            items = [items]
        
        # Ensure embeddings
        for item in items:
            if item.get_embedding() is None:
                item = await self.embedder.embed_multimodal_item(item)
        
        await self._index.add_batch(items)
        return len(items)
    
    async def retrieve(
        self,
        query: Union[str, ImageData],
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: Optional[int] = None,
        rerank: bool = True,
    ) -> RetrievalResult:
        """
        Main retrieval method.
        
        Args:
            query: Text or image query
            mode: Retrieval mode
            top_k: Number of results
            rerank: Whether to rerank results
        
        Returns:
            RetrievalResult with matched items
        """
        start_time = time.perf_counter()
        top_k = top_k or self.config.top_k
        
        # Determine query modality
        if isinstance(query, str):
            query_modality = Modality.TEXT
            query_text = query
            query_image = None
        else:
            query_modality = Modality.IMAGE
            query_text = None
            query_image = query
        
        # Execute retrieval based on mode
        if mode == RetrievalMode.TEXT_TO_IMAGE:
            items = await self._text_to_image.retrieve(
                query, self._index, self.config.rerank_top_k if rerank else top_k
            )
        elif mode == RetrievalMode.IMAGE_TO_TEXT:
            items = await self._image_to_text.retrieve(
                query, self._index, self.config.rerank_top_k if rerank else top_k
            )
        elif mode == RetrievalMode.IMAGE_TO_IMAGE:
            items = await self._image_to_image.retrieve(
                query, self._index, self.config.rerank_top_k if rerank else top_k
            )
        elif mode == RetrievalMode.UNIFIED:
            items = await self._unified.retrieve(
                query, self._index, self.config.rerank_top_k if rerank else top_k
            )
        else:  # HYBRID
            items = await self._hybrid.retrieve(
                query, self._index, self.config.rerank_top_k if rerank else top_k
            )
        
        # Rerank if enabled
        if rerank and self.config.enable_reranking and items:
            items = await self._reranker.rerank(
                query_text=query_text,
                query_image=query_image,
                items=items,
                top_k=top_k,
            )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Build result
        return RetrievalResult(
            query=str(query) if isinstance(query, str) else f"[image:{query.id}]",
            query_modality=query_modality,
            retrieval_mode=mode,
            items=items[:top_k],
            total_candidates=self._index.count(),
            time_ms=elapsed_ms,
        )
    
    async def text_to_image(
        self,
        text: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Find images matching text."""
        return await self.retrieve(text, RetrievalMode.TEXT_TO_IMAGE, top_k)
    
    async def image_to_text(
        self,
        image: ImageData,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Find text matching image."""
        return await self.retrieve(image, RetrievalMode.IMAGE_TO_TEXT, top_k)
    
    async def find_similar_images(
        self,
        image: ImageData,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Find similar images."""
        return await self.retrieve(image, RetrievalMode.IMAGE_TO_IMAGE, top_k)
    
    async def hybrid_search(
        self,
        query: Union[str, ImageData],
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Hybrid search across modalities."""
        return await self.retrieve(query, RetrievalMode.HYBRID, top_k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "index_size": self._index.count(),
            "text_items": self._index.count(Modality.TEXT),
            "image_items": self._index.count(Modality.IMAGE),
            "config": {
                "top_k": self.config.top_k,
                "threshold": self.config.similarity_threshold,
                "reranking": self.config.enable_reranking,
            },
        }
