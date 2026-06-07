"""
multimodal_rag/embeddings.py

Multi-modal embedding providers.

Provides:
- CLIPEmbedder: OpenAI CLIP embeddings
- BLIPEmbedder: Salesforce BLIP embeddings
- SigLIPEmbedder: Google SigLIP embeddings
- OpenCLIPEmbedder: LAION OpenCLIP variants
- UnifiedEmbedder: Orchestrates multiple models
- EmbeddingConfig: Configuration

Embedding Strategies:
- Separate: Independent text/image embeddings
- Joint: Combined cross-modal embeddings
- Fusion: Multiple embeddings combined

Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .providers import ImageData, TextData, MultiModalItem, Modality

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    batch_size: int = 16
    normalize: bool = True
    cache_embeddings: bool = True
    dimension: int = 512
    max_text_length: int = 77
    image_size: int = 224


# ============================================================================
# Base Embedder
# ============================================================================


class BaseMultiModalEmbedder(ABC):
    """Base class for multi-modal embedders."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._device = None
        self._is_loaded = False
    
    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.config.dimension
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model."""
        pass
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._is_loaded:
            self._load_model()
    
    @abstractmethod
    async def embed_text(
        self,
        texts: Union[str, List[str]],
    ) -> np.ndarray:
        """Embed text(s)."""
        pass
    
    @abstractmethod
    async def embed_image(
        self,
        images: Union[ImageData, List[ImageData]],
    ) -> np.ndarray:
        """Embed image(s)."""
        pass
    
    async def embed_text_and_image(
        self,
        text: str,
        image: ImageData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Embed both text and image."""
        text_emb = await self.embed_text(text)
        image_emb = await self.embed_image(image)
        return text_emb, image_emb
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between embeddings."""
        # Normalize
        emb1 = embedding1 / np.linalg.norm(embedding1)
        emb2 = embedding2 / np.linalg.norm(embedding2)
        
        return float(np.dot(emb1, emb2))
    
    def unload(self) -> None:
        """Unload model to free memory."""
        if self._model:
            del self._model
            self._model = None
        if self._processor:
            del self._processor
            self._processor = None
        
        self._is_loaded = False
        
        # Try to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        logger.info(f"Unloaded {self.__class__.__name__}")


# ============================================================================
# CLIP Embedder
# ============================================================================


class CLIPEmbedder(BaseMultiModalEmbedder):
    """
    OpenAI CLIP embeddings.
    
    Supports:
    - ViT-B/32, ViT-B/16, ViT-L/14
    - Text and image in same embedding space
    """
    
    MODEL_CONFIGS = {
        "openai/clip-vit-base-patch32": {"dimension": 512, "image_size": 224},
        "openai/clip-vit-base-patch16": {"dimension": 512, "image_size": 224},
        "openai/clip-vit-large-patch14": {"dimension": 768, "image_size": 224},
    }
    
    def _load_model(self) -> None:
        """Load CLIP model."""
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            
            # Determine device
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device
            
            logger.info(f"Loading CLIP model: {self.config.model_name} on {self._device}")
            
            self._processor = CLIPProcessor.from_pretrained(self.config.model_name)
            self._model = CLIPModel.from_pretrained(self.config.model_name)
            self._model.to(self._device)
            self._model.eval()
            
            # Update dimension from config
            if self.config.model_name in self.MODEL_CONFIGS:
                self.config.dimension = self.MODEL_CONFIGS[self.config.model_name]["dimension"]
            
            self._is_loaded = True
            logger.info(f"CLIP model loaded, dimension: {self.config.dimension}")
            
        except ImportError as e:
            raise RuntimeError(f"Install transformers and torch: {e}")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            raise
    
    async def embed_text(
        self,
        texts: Union[str, List[str]],
    ) -> np.ndarray:
        """Embed text(s) using CLIP."""
        self._ensure_loaded()
        
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            inputs = self._processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_text_length,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
                
                if self.config.normalize:
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                embeddings.append(text_features.cpu().numpy())
        
        result = np.concatenate(embeddings, axis=0)
        
        return result[0] if len(texts) == 1 else result
    
    async def embed_image(
        self,
        images: Union[ImageData, List[ImageData]],
    ) -> np.ndarray:
        """Embed image(s) using CLIP."""
        self._ensure_loaded()
        
        import torch
        from PIL import Image
        import io
        
        if isinstance(images, ImageData):
            images = [images]
        
        embeddings = []
        
        for i in range(0, len(images), self.config.batch_size):
            batch = images[i:i + self.config.batch_size]
            
            # Load PIL images
            pil_images = []
            for img_data in batch:
                if img_data.raw_bytes:
                    pil_img = Image.open(io.BytesIO(img_data.raw_bytes)).convert("RGB")
                    pil_images.append(pil_img)
            
            if not pil_images:
                continue
            
            inputs = self._processor(
                images=pil_images,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)
                
                if self.config.normalize:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                embeddings.append(image_features.cpu().numpy())
        
        if not embeddings:
            return np.zeros((len(images), self.config.dimension))
        
        result = np.concatenate(embeddings, axis=0)
        
        return result[0] if len(images) == 1 else result


# ============================================================================
# BLIP Embedder
# ============================================================================


class BLIPEmbedder(BaseMultiModalEmbedder):
    """
    Salesforce BLIP embeddings.
    
    Supports:
    - Image-text matching
    - Image captioning
    - VQA (with BLIP-2)
    """
    
    def _load_model(self) -> None:
        """Load BLIP model."""
        try:
            import torch
            from transformers import BlipProcessor, BlipModel
            
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device
            
            logger.info(f"Loading BLIP model: {self.config.model_name}")
            
            self._processor = BlipProcessor.from_pretrained(self.config.model_name)
            self._model = BlipModel.from_pretrained(self.config.model_name)
            self._model.to(self._device)
            self._model.eval()
            
            self.config.dimension = 768
            self._is_loaded = True
            
        except ImportError as e:
            raise RuntimeError(f"Install transformers: {e}")
        except Exception as e:
            logger.error(f"Failed to load BLIP: {e}")
            raise
    
    async def embed_text(
        self,
        texts: Union[str, List[str]],
    ) -> np.ndarray:
        """Embed text using BLIP."""
        self._ensure_loaded()
        
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for text in texts:
            inputs = self._processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.get_text_features(**inputs)
                
                if self.config.normalize:
                    outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                
                embeddings.append(outputs.cpu().numpy())
        
        result = np.concatenate(embeddings, axis=0)
        return result[0] if len(texts) == 1 else result
    
    async def embed_image(
        self,
        images: Union[ImageData, List[ImageData]],
    ) -> np.ndarray:
        """Embed image using BLIP."""
        self._ensure_loaded()
        
        import torch
        from PIL import Image
        import io
        
        if isinstance(images, ImageData):
            images = [images]
        
        embeddings = []
        
        for img_data in images:
            if not img_data.raw_bytes:
                embeddings.append(np.zeros(self.config.dimension))
                continue
            
            pil_img = Image.open(io.BytesIO(img_data.raw_bytes)).convert("RGB")
            
            inputs = self._processor(
                images=pil_img,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                
                if self.config.normalize:
                    outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                
                embeddings.append(outputs.cpu().numpy().squeeze())
        
        result = np.stack(embeddings, axis=0)
        return result[0] if len(images) == 1 else result


# ============================================================================
# SigLIP Embedder
# ============================================================================


class SigLIPEmbedder(BaseMultiModalEmbedder):
    """
    Google SigLIP embeddings.
    
    Sigmoid loss for better calibration.
    """
    
    def _load_model(self) -> None:
        """Load SigLIP model."""
        try:
            import torch
            from transformers import AutoProcessor, AutoModel
            
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device
            
            logger.info(f"Loading SigLIP model: {self.config.model_name}")
            
            self._processor = AutoProcessor.from_pretrained(self.config.model_name)
            self._model = AutoModel.from_pretrained(self.config.model_name)
            self._model.to(self._device)
            self._model.eval()
            
            self.config.dimension = 768
            self._is_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load SigLIP: {e}")
            raise
    
    async def embed_text(
        self,
        texts: Union[str, List[str]],
    ) -> np.ndarray:
        """Embed text using SigLIP."""
        self._ensure_loaded()
        
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
            
            if self.config.normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        result = text_features.cpu().numpy()
        return result[0] if len(texts) == 1 else result
    
    async def embed_image(
        self,
        images: Union[ImageData, List[ImageData]],
    ) -> np.ndarray:
        """Embed image using SigLIP."""
        self._ensure_loaded()
        
        import torch
        from PIL import Image
        import io
        
        if isinstance(images, ImageData):
            images = [images]
        
        pil_images = []
        for img_data in images:
            if img_data.raw_bytes:
                pil_images.append(
                    Image.open(io.BytesIO(img_data.raw_bytes)).convert("RGB")
                )
        
        if not pil_images:
            return np.zeros((len(images), self.config.dimension))
        
        inputs = self._processor(
            images=pil_images,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            
            if self.config.normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        result = image_features.cpu().numpy()
        return result[0] if len(images) == 1 else result


# ============================================================================
# Unified Embedder
# ============================================================================


class UnifiedEmbedder:
    """
    Orchestrates multiple embedding models.
    
    Features:
    - Model selection per modality
    - Embedding fusion
    - Caching
    - Batch processing
    """
    
    EMBEDDER_CLASSES = {
        "clip": CLIPEmbedder,
        "blip": BLIPEmbedder,
        "siglip": SigLIPEmbedder,
    }
    
    def __init__(
        self,
        default_model: str = "clip",
        config: Optional[EmbeddingConfig] = None,
        cache_provider: Optional[Any] = None,
    ):
        self.default_model = default_model
        self.config = config or EmbeddingConfig()
        self._cache = cache_provider
        self._embedders: Dict[str, BaseMultiModalEmbedder] = {}
    
    def _get_embedder(self, model_type: str) -> BaseMultiModalEmbedder:
        """Get or create embedder instance."""
        if model_type not in self._embedders:
            embedder_class = self.EMBEDDER_CLASSES.get(model_type)
            
            if not embedder_class:
                # Default to CLIP
                logger.warning(f"Unknown model type '{model_type}', using CLIP")
                embedder_class = CLIPEmbedder
            
            self._embedders[model_type] = embedder_class(self.config)
        
        return self._embedders[model_type]
    
    async def embed_text(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> np.ndarray:
        """Embed text(s)."""
        model = model or self.default_model
        embedder = self._get_embedder(model)
        return await embedder.embed_text(texts)
    
    async def embed_image(
        self,
        images: Union[ImageData, List[ImageData]],
        model: Optional[str] = None,
    ) -> np.ndarray:
        """Embed image(s)."""
        model = model or self.default_model
        embedder = self._get_embedder(model)
        
        # Check cache
        if self._cache and isinstance(images, ImageData):
            cached = await self._cache.get_embedding(images.content_hash, "image")
            if cached is not None:
                return cached
        
        embedding = await embedder.embed_image(images)
        
        # Store in cache
        if self._cache and isinstance(images, ImageData):
            await self._cache.set_embedding(images.content_hash, embedding, "image")
        
        return embedding
    
    async def embed_multimodal_item(
        self,
        item: MultiModalItem,
        embed_text: bool = True,
        embed_image: bool = True,
        create_unified: bool = True,
    ) -> MultiModalItem:
        """Embed a multi-modal item."""
        embedder = self._get_embedder(self.default_model)
        
        # Embed text
        if embed_text and item.text_content:
            item.text_embedding = await embedder.embed_text(item.text_content)
        
        # Embed image
        if embed_image and item.image_data:
            item.image_embedding = await embedder.embed_image(item.image_data)
        
        # Create unified embedding
        if create_unified:
            item.unified_embedding = self._fuse_embeddings(
                item.text_embedding,
                item.image_embedding,
            )
        
        return item
    
    def _fuse_embeddings(
        self,
        text_emb: Optional[np.ndarray],
        image_emb: Optional[np.ndarray],
        method: str = "average",
    ) -> Optional[np.ndarray]:
        """Fuse text and image embeddings."""
        if text_emb is None and image_emb is None:
            return None
        
        if text_emb is None:
            return image_emb
        if image_emb is None:
            return text_emb
        
        # Ensure same dimension (pad or truncate)
        if text_emb.shape != image_emb.shape:
            min_dim = min(text_emb.shape[0], image_emb.shape[0])
            text_emb = text_emb[:min_dim]
            image_emb = image_emb[:min_dim]
        
        if method == "average":
            return (text_emb + image_emb) / 2
        elif method == "concat":
            return np.concatenate([text_emb, image_emb])
        elif method == "weighted":
            return 0.6 * text_emb + 0.4 * image_emb
        else:
            return (text_emb + image_emb) / 2
    
    def compute_cross_modal_similarity(
        self,
        text_embedding: np.ndarray,
        image_embedding: np.ndarray,
    ) -> float:
        """Compute similarity between text and image embeddings."""
        text_norm = text_embedding / np.linalg.norm(text_embedding)
        image_norm = image_embedding / np.linalg.norm(image_embedding)
        return float(np.dot(text_norm, image_norm))
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        embedder = self._get_embedder(self.default_model)
        return embedder.config.dimension
    
    def unload_all(self) -> None:
        """Unload all models."""
        for embedder in self._embedders.values():
            embedder.unload()
        self._embedders.clear()
    
    def health_check(self) -> Dict[str, Any]:
        """Check embedder health."""
        return {
            "default_model": self.default_model,
            "loaded_models": list(self._embedders.keys()),
            "dimension": self.dimension,
        }
