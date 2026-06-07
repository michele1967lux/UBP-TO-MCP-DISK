"""
multimodal_rag - Multi-Modal Retrieval Augmented Generation

Enterprise module for UBP Hybrid System.

Capabilities:
- Multi-Modal Embeddings (CLIP, BLIP, SigLIP, LLaVA)
- Cross-Modal Retrieval (text→image, image→text, image→image)
- Visual Question Answering (VQA)
- Image Captioning and Description
- OCR Integration (Tesseract, EasyOCR, PaddleOCR)
- Document Understanding (PDF with images, diagrams, tables)
- Multi-Modal Chunking (preserve image-text relationships)
- Unified Indexing (single index for all modalities)
- Result Fusion (late fusion, early fusion, hybrid)
- Image Processing (resize, normalize, augment)

Supported Models:
- CLIP (OpenAI, OpenCLIP variants)
- BLIP/BLIP-2 (Salesforce)
- SigLIP (Google)
- LLaVA (for VQA)
- Florence-2 (Microsoft)
- Qwen-VL

Architecture:
- Modality-agnostic embedding space
- Cross-modal attention mechanisms
- Hierarchical document representation
- Adaptive chunking for visual content

Version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .adapter import MultiModalRAGAdapter

__version__ = "1.0.0"
__all__ = ["create_module", "MultiModalRAGAdapter"]


def create_module(
    module_path: Path,
    di_container: Optional[Any] = None,
    event_bus: Optional[Any] = None,
) -> MultiModalRAGAdapter:
    """
    Factory function for creating the multimodal_rag adapter.
    
    This is the entry point used by ModuleLoader.
    
    Args:
        module_path: Path to the module directory
        di_container: DI container for dependency resolution
        event_bus: Event bus for publishing events
    
    Returns:
        Configured MultiModalRAGAdapter instance
    """
    return MultiModalRAGAdapter(
        module_path=module_path,
        di_container=di_container,
        event_bus=event_bus,
    )
