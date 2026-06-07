"""
multimodal_rag/providers.py

Core providers and data classes for multi-modal RAG.

Provides:
- Data classes (MultiModalItem, ImageData, DocumentChunk, etc.)
- Modality enum and types
- ImageProcessor: Image preprocessing and normalization
- OCRProvider: Text extraction from images
- DocumentProcessor: PDF/document processing
- CacheProvider: Multi-modal caching
- MetricsCollector: Performance tracking

Version: 1.0.0
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, BinaryIO

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class Modality(str, Enum):
    """Content modality types."""
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    TABLE = "table"
    DIAGRAM = "diagram"
    MIXED = "mixed"


class RetrievalMode(str, Enum):
    """Cross-modal retrieval modes."""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_TEXT = "image_to_text"
    IMAGE_TO_IMAGE = "image_to_image"
    TEXT_TO_TEXT = "text_to_text"
    HYBRID = "hybrid"
    UNIFIED = "unified"


class FusionMethod(str, Enum):
    """Result fusion methods."""
    EARLY = "early"
    LATE = "late"
    HYBRID = "hybrid"
    ATTENTION = "attention"
    WEIGHTED = "weighted"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ImageData:
    """Processed image data."""
    id: str
    raw_bytes: Optional[bytes] = None
    base64_data: Optional[str] = None
    width: int = 0
    height: int = 0
    channels: int = 3
    format: str = "RGB"
    file_path: Optional[str] = None
    url: Optional[str] = None
    
    # Processed data
    tensor: Optional[Any] = None  # torch.Tensor or np.ndarray
    embedding: Optional[np.ndarray] = None
    thumbnail: Optional[bytes] = None
    
    # Extracted content
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "has_embedding": self.embedding is not None,
            "caption": self.caption,
            "ocr_text": self.ocr_text[:200] + "..." if self.ocr_text and len(self.ocr_text) > 200 else self.ocr_text,
            "detected_objects": len(self.detected_objects),
            "metadata": self.metadata,
        }
    
    @property
    def content_hash(self) -> str:
        """Generate content hash for caching."""
        if self.raw_bytes:
            return hashlib.md5(self.raw_bytes).hexdigest()
        elif self.base64_data:
            return hashlib.md5(self.base64_data.encode()).hexdigest()
        elif self.file_path:
            return hashlib.md5(self.file_path.encode()).hexdigest()
        return hashlib.md5(self.id.encode()).hexdigest()


@dataclass
class TextData:
    """Processed text data."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    
    # Source info
    source_id: Optional[str] = None
    position: int = 0
    
    # Associated images
    associated_images: List[str] = field(default_factory=list)
    
    # Metadata
    language: str = "en"
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "has_embedding": self.embedding is not None,
            "source_id": self.source_id,
            "position": self.position,
            "associated_images": self.associated_images,
            "language": self.language,
            "token_count": self.token_count,
        }


@dataclass
class MultiModalItem:
    """Unified multi-modal item."""
    id: str
    modality: Modality
    
    # Content
    text_content: Optional[str] = None
    image_data: Optional[ImageData] = None
    
    # Embeddings
    text_embedding: Optional[np.ndarray] = None
    image_embedding: Optional[np.ndarray] = None
    unified_embedding: Optional[np.ndarray] = None
    
    # Scores
    relevance_score: float = 0.0
    text_score: float = 0.0
    image_score: float = 0.0
    
    # Source
    source_id: Optional[str] = None
    source_type: Optional[str] = None
    page_number: Optional[int] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "modality": self.modality.value,
            "text_content": self.text_content[:300] + "..." if self.text_content and len(self.text_content) > 300 else self.text_content,
            "has_image": self.image_data is not None,
            "image_info": self.image_data.to_dict() if self.image_data else None,
            "has_text_embedding": self.text_embedding is not None,
            "has_image_embedding": self.image_embedding is not None,
            "has_unified_embedding": self.unified_embedding is not None,
            "relevance_score": round(self.relevance_score, 4),
            "text_score": round(self.text_score, 4),
            "image_score": round(self.image_score, 4),
            "source_id": self.source_id,
            "page_number": self.page_number,
            "metadata": self.metadata,
        }
    
    def get_embedding(self, prefer: str = "unified") -> Optional[np.ndarray]:
        """Get embedding with preference order."""
        if prefer == "unified" and self.unified_embedding is not None:
            return self.unified_embedding
        if prefer == "text" and self.text_embedding is not None:
            return self.text_embedding
        if prefer == "image" and self.image_embedding is not None:
            return self.image_embedding
        
        # Fallback order
        return self.unified_embedding or self.text_embedding or self.image_embedding


@dataclass
class DocumentPage:
    """Document page with text and images."""
    page_number: int
    text_content: str
    images: List[ImageData] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    
    # Layout info
    width: float = 0
    height: float = 0
    
    # Image-text associations
    image_captions: Dict[str, str] = field(default_factory=dict)  # image_id -> caption
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text_length": len(self.text_content),
            "image_count": len(self.images),
            "table_count": len(self.tables),
            "dimensions": {"width": self.width, "height": self.height},
        }


@dataclass
class ProcessedDocument:
    """Fully processed document."""
    id: str
    title: str
    pages: List[DocumentPage]
    
    # Aggregated content
    full_text: str = ""
    all_images: List[ImageData] = field(default_factory=list)
    
    # Chunks
    chunks: List[MultiModalItem] = field(default_factory=list)
    
    # Metadata
    file_path: Optional[str] = None
    file_type: str = "pdf"
    page_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "page_count": self.page_count,
            "total_images": len(self.all_images),
            "total_chunks": len(self.chunks),
            "file_type": self.file_type,
            "text_length": len(self.full_text),
        }


@dataclass
class RetrievalResult:
    """Multi-modal retrieval result."""
    query: str
    query_modality: Modality
    retrieval_mode: RetrievalMode
    
    items: List[MultiModalItem]
    
    # Stats
    total_candidates: int = 0
    time_ms: float = 0
    
    # Score distributions
    text_score_range: Tuple[float, float] = (0.0, 0.0)
    image_score_range: Tuple[float, float] = (0.0, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "query_modality": self.query_modality.value,
            "retrieval_mode": self.retrieval_mode.value,
            "result_count": len(self.items),
            "items": [item.to_dict() for item in self.items[:10]],  # Limit for response
            "total_candidates": self.total_candidates,
            "time_ms": round(self.time_ms, 2),
        }


@dataclass 
class VQAResult:
    """Visual Question Answering result."""
    question: str
    answer: str
    image_id: str
    confidence: float = 0.0
    model_used: str = ""
    time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "image_id": self.image_id,
            "confidence": round(self.confidence, 3),
            "model_used": self.model_used,
            "time_ms": round(self.time_ms, 2),
        }


@dataclass
class CaptionResult:
    """Image captioning result."""
    image_id: str
    caption: str
    style: str = "descriptive"
    confidence: float = 0.0
    model_used: str = ""
    time_ms: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "caption": self.caption,
            "style": self.style,
            "confidence": round(self.confidence, 3),
            "model_used": self.model_used,
            "time_ms": round(self.time_ms, 2),
        }


# ============================================================================
# Image Processor
# ============================================================================


class ImageProcessor:
    """
    Image preprocessing and normalization.
    
    Handles:
    - Loading from various sources (file, bytes, base64, URL)
    - Resizing and normalization
    - Format conversion
    - Thumbnail generation
    """
    
    def __init__(
        self,
        default_size: int = 224,
        max_size: int = 1024,
        preserve_aspect: bool = True,
        normalize: bool = True,
    ):
        self.default_size = default_size
        self.max_size = max_size
        self.preserve_aspect = preserve_aspect
        self.normalize = normalize
        
        self._pil_available = False
        self._cv2_available = False
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check available image libraries."""
        try:
            from PIL import Image
            self._pil_available = True
        except ImportError:
            pass
        
        try:
            import cv2
            self._cv2_available = True
        except ImportError:
            pass
    
    def load_image(
        self,
        source: Union[str, bytes, Path, BinaryIO],
        image_id: Optional[str] = None,
    ) -> ImageData:
        """Load image from various sources."""
        if not self._pil_available:
            raise RuntimeError("PIL/Pillow required for image processing")
        
        from PIL import Image
        
        image_id = image_id or hashlib.md5(str(source).encode()).hexdigest()[:12]
        raw_bytes = None
        
        # Load based on source type
        if isinstance(source, bytes):
            raw_bytes = source
            img = Image.open(io.BytesIO(source))
        elif isinstance(source, str):
            if source.startswith("data:image"):
                # Base64 data URI
                base64_str = source.split(",")[1] if "," in source else source
                raw_bytes = base64.b64decode(base64_str)
                img = Image.open(io.BytesIO(raw_bytes))
            elif source.startswith("http"):
                # URL - would need requests
                raise NotImplementedError("URL loading not implemented")
            else:
                # File path
                img = Image.open(source)
                with open(source, "rb") as f:
                    raw_bytes = f.read()
        elif isinstance(source, Path):
            img = Image.open(source)
            raw_bytes = source.read_bytes()
        elif hasattr(source, "read"):
            raw_bytes = source.read()
            img = Image.open(io.BytesIO(raw_bytes))
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
        
        # Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        return ImageData(
            id=image_id,
            raw_bytes=raw_bytes,
            width=img.width,
            height=img.height,
            channels=3,
            format="RGB",
            metadata={"original_mode": img.mode},
        )
    
    def preprocess(
        self,
        image_data: ImageData,
        target_size: Optional[int] = None,
        return_tensor: bool = True,
    ) -> ImageData:
        """Preprocess image for embedding."""
        if not self._pil_available:
            raise RuntimeError("PIL/Pillow required")
        
        from PIL import Image
        
        target_size = target_size or self.default_size
        
        # Load from bytes
        if image_data.raw_bytes:
            img = Image.open(io.BytesIO(image_data.raw_bytes))
        else:
            raise ValueError("No image data available")
        
        # Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize
        if self.preserve_aspect:
            img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            # Pad to square
            new_img = Image.new("RGB", (target_size, target_size), (128, 128, 128))
            paste_x = (target_size - img.width) // 2
            paste_y = (target_size - img.height) // 2
            new_img.paste(img, (paste_x, paste_y))
            img = new_img
        else:
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize
        if self.normalize:
            img_array = img_array / 255.0
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
        
        # Update image data
        image_data.width = target_size
        image_data.height = target_size
        
        if return_tensor:
            # HWC to CHW
            img_array = img_array.transpose(2, 0, 1)
            image_data.tensor = img_array
        
        return image_data
    
    def generate_thumbnail(
        self,
        image_data: ImageData,
        size: int = 128,
    ) -> bytes:
        """Generate thumbnail for storage."""
        if not self._pil_available:
            raise RuntimeError("PIL required")
        
        from PIL import Image
        
        if not image_data.raw_bytes:
            raise ValueError("No image data")
        
        img = Image.open(io.BytesIO(image_data.raw_bytes))
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        
        thumbnail_bytes = buffer.getvalue()
        image_data.thumbnail = thumbnail_bytes
        
        return thumbnail_bytes
    
    def to_base64(self, image_data: ImageData) -> str:
        """Convert image to base64 string."""
        if image_data.raw_bytes:
            return base64.b64encode(image_data.raw_bytes).decode()
        raise ValueError("No image data")


# ============================================================================
# OCR Provider
# ============================================================================


class OCRProvider:
    """
    OCR text extraction from images.
    
    Supports:
    - Tesseract
    - EasyOCR
    - PaddleOCR
    """
    
    def __init__(
        self,
        engine: str = "tesseract",
        languages: List[str] = None,
        confidence_threshold: float = 0.6,
    ):
        self.engine = engine
        self.languages = languages or ["eng"]
        self.confidence_threshold = confidence_threshold
        
        self._tesseract_available = False
        self._easyocr_available = False
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check available OCR engines."""
        try:
            import pytesseract
            self._tesseract_available = True
        except ImportError:
            pass
        
        try:
            import easyocr
            self._easyocr_available = True
        except ImportError:
            pass
    
    def extract_text(
        self,
        image_data: ImageData,
        preprocess: bool = True,
    ) -> str:
        """Extract text from image."""
        if self.engine == "tesseract" and self._tesseract_available:
            return self._extract_tesseract(image_data, preprocess)
        elif self.engine == "easyocr" and self._easyocr_available:
            return self._extract_easyocr(image_data)
        else:
            logger.warning(f"OCR engine {self.engine} not available")
            return ""
    
    def _extract_tesseract(
        self,
        image_data: ImageData,
        preprocess: bool,
    ) -> str:
        """Extract using Tesseract."""
        import pytesseract
        from PIL import Image
        
        if not image_data.raw_bytes:
            return ""
        
        img = Image.open(io.BytesIO(image_data.raw_bytes))
        
        # Preprocess for OCR
        if preprocess:
            img = img.convert("L")  # Grayscale
        
        lang = "+".join(self.languages)
        
        try:
            text = pytesseract.image_to_string(img, lang=lang)
            return self._clean_ocr_text(text)
        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")
            return ""
    
    def _extract_easyocr(self, image_data: ImageData) -> str:
        """Extract using EasyOCR."""
        import easyocr
        
        if not image_data.raw_bytes:
            return ""
        
        # Map language codes
        lang_map = {"eng": "en", "ita": "it", "deu": "de", "fra": "fr", "spa": "es"}
        langs = [lang_map.get(l, l) for l in self.languages]
        
        reader = easyocr.Reader(langs, gpu=True)
        
        try:
            results = reader.readtext(image_data.raw_bytes)
            
            texts = []
            for bbox, text, conf in results:
                if conf >= self.confidence_threshold:
                    texts.append(text)
            
            return self._clean_ocr_text(" ".join(texts))
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
            return ""
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR output."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove isolated single characters
        text = re.sub(r'\s[a-zA-Z]\s', ' ', text)
        # Clean up
        text = text.strip()
        return text


# ============================================================================
# Cache Provider
# ============================================================================


class MultiModalCacheProvider:
    """
    Redis-based caching for multi-modal data.
    
    Caches:
    - Embeddings (text, image, unified)
    - Captions
    - OCR results
    - Retrieval results
    """
    
    def __init__(
        self,
        redis_client: Optional[Any] = None,
        prefix: str = "ubp:multimodal",
        ttl_seconds: int = 7200,
        enabled: bool = True,
    ):
        self._redis = redis_client
        self.prefix = prefix
        self.ttl = ttl_seconds
        self.enabled = enabled
        self._stats = {"hits": 0, "misses": 0}
    
    def _make_key(self, category: str, item_id: str) -> str:
        """Generate cache key."""
        return f"{self.prefix}:{category}:{item_id}"
    
    async def get_embedding(
        self,
        item_id: str,
        modality: str = "unified",
    ) -> Optional[np.ndarray]:
        """Get cached embedding."""
        if not self.enabled or not self._redis:
            return None
        
        try:
            key = self._make_key(f"embed_{modality}", item_id)
            data = await self._redis.get(key)
            
            if data:
                self._stats["hits"] += 1
                return np.frombuffer(data, dtype=np.float32)
            
            self._stats["misses"] += 1
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    async def set_embedding(
        self,
        item_id: str,
        embedding: np.ndarray,
        modality: str = "unified",
    ) -> bool:
        """Cache embedding."""
        if not self.enabled or not self._redis:
            return False
        
        try:
            key = self._make_key(f"embed_{modality}", item_id)
            await self._redis.setex(key, self.ttl, embedding.tobytes())
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    async def get_caption(self, image_id: str) -> Optional[str]:
        """Get cached caption."""
        if not self.enabled or not self._redis:
            return None
        
        try:
            key = self._make_key("caption", image_id)
            data = await self._redis.get(key)
            
            if data:
                self._stats["hits"] += 1
                return data.decode()
            
            self._stats["misses"] += 1
            return None
        except Exception:
            return None
    
    async def set_caption(self, image_id: str, caption: str) -> bool:
        """Cache caption."""
        if not self.enabled or not self._redis:
            return False
        
        try:
            key = self._make_key("caption", image_id)
            await self._redis.setex(key, self.ttl, caption.encode())
            return True
        except Exception:
            return False
    
    async def get_ocr(self, image_id: str) -> Optional[str]:
        """Get cached OCR result."""
        if not self.enabled or not self._redis:
            return None
        
        try:
            key = self._make_key("ocr", image_id)
            data = await self._redis.get(key)
            return data.decode() if data else None
        except Exception:
            return None
    
    async def set_ocr(self, image_id: str, text: str) -> bool:
        """Cache OCR result."""
        if not self.enabled or not self._redis:
            return False
        
        try:
            key = self._make_key("ocr", image_id)
            await self._redis.setex(key, self.ttl, text.encode())
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / max(total, 1)
        
        return {
            "enabled": self.enabled and self._redis is not None,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": round(hit_rate, 3),
        }


# ============================================================================
# Metrics Collector
# ============================================================================


class MetricsCollector:
    """Collects multi-modal RAG metrics."""
    
    def __init__(self):
        self._metrics = {
            "embeddings": {"text": 0, "image": 0, "unified": 0},
            "retrievals": {"text_to_image": 0, "image_to_text": 0, "hybrid": 0},
            "latencies": [],
            "modality_distribution": {},
        }
    
    def record_embedding(self, modality: str):
        """Record embedding generation."""
        if modality in self._metrics["embeddings"]:
            self._metrics["embeddings"][modality] += 1
    
    def record_retrieval(self, mode: str, latency_ms: float, result_count: int):
        """Record retrieval operation."""
        if mode in self._metrics["retrievals"]:
            self._metrics["retrievals"][mode] += 1
        
        self._metrics["latencies"].append({
            "mode": mode,
            "latency_ms": latency_ms,
            "result_count": result_count,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Keep only last 1000 entries
        if len(self._metrics["latencies"]) > 1000:
            self._metrics["latencies"] = self._metrics["latencies"][-1000:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        latencies = self._metrics["latencies"]
        
        avg_latency = 0
        if latencies:
            avg_latency = sum(l["latency_ms"] for l in latencies) / len(latencies)
        
        return {
            "embeddings": self._metrics["embeddings"],
            "retrievals": self._metrics["retrievals"],
            "avg_latency_ms": round(avg_latency, 2),
            "total_operations": sum(self._metrics["retrievals"].values()),
        }
