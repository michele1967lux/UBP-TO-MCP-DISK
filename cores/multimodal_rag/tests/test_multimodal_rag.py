"""
Standalone tests for multimodal_rag module.

Tests all 22 operations with mocked heavy dependencies (torch, transformers, CLIP, etc.).
Runs without GPU, Redis, or external services.

Usage:
    PYTHONPATH=. pytest modules/cores/multimodal_rag/tests/test_multimodal_rag.py -v
"""

import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Resolve project root so imports work standalone
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MODULE_ROOT = _HERE.parent
_CORES_ROOT = _MODULE_ROOT.parent
_PROJECT_ROOT = _CORES_ROOT.parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_CORES_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORES_ROOT))

# ---------------------------------------------------------------------------
# Import module under test
# ---------------------------------------------------------------------------
from multimodal_rag.providers import (
    Modality,
    RetrievalMode,
    FusionMethod,
    ImageData,
    TextData,
    MultiModalItem,
    RetrievalResult,
    VQAResult,
    CaptionResult,
    ImageProcessor,
    OCRProvider,
    MultiModalCacheProvider,
    MetricsCollector,
)
from multimodal_rag.adapter import MultiModalRAGAdapter, _load_config, _coerce_value


# ============================================================================
# Helpers
# ============================================================================

def _make_adapter(**overrides) -> MultiModalRAGAdapter:
    """Create adapter with module_path pointing to the real config."""
    return MultiModalRAGAdapter(
        module_path=_MODULE_ROOT,
        di_container=overrides.get("di_container"),
        event_bus=overrides.get("event_bus"),
    )


def _mock_initialized_adapter() -> MultiModalRAGAdapter:
    """Create adapter with all components mocked as initialized."""
    adapter = _make_adapter()
    adapter._initialized = True

    # Mock image processor
    mock_img = MagicMock(spec=ImageProcessor)
    mock_img.load_image.return_value = ImageData(
        id="test-img-001",
        width=224, height=224, channels=3, format="RGB",
        raw_bytes=b"\x00" * 100,
    )
    adapter._image_processor = mock_img

    # Mock OCR provider
    mock_ocr = MagicMock(spec=OCRProvider)
    mock_ocr.extract_text.return_value = "Extracted OCR text from image."
    adapter._ocr_provider = mock_ocr

    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.embed_image = AsyncMock(return_value=np.random.rand(512).astype(np.float32))
    mock_embedder.embed_text = AsyncMock(return_value=np.random.rand(512).astype(np.float32))
    mock_embedder.health_check.return_value = {"status": "healthy"}
    mock_embedder.unload_all = MagicMock()
    adapter._embedder = mock_embedder

    # Mock retriever
    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(return_value=RetrievalResult(
        query="test query",
        query_modality=Modality.TEXT,
        retrieval_mode=RetrievalMode.HYBRID,
        items=[
            MultiModalItem(id="r1", modality=Modality.TEXT, text_content="Result 1", relevance_score=0.95),
            MultiModalItem(id="r2", modality=Modality.IMAGE, relevance_score=0.87),
        ],
        time_ms=12.5,
    ))
    mock_retriever.add_to_index = AsyncMock(return_value=1)
    mock_retriever.get_stats.return_value = {"total_items": 10}
    mock_retriever.index = MagicMock()
    mock_retriever.index.count.return_value = 10
    mock_retriever.index.clear = MagicMock()
    adapter._retriever = mock_retriever

    # Mock VQA provider
    mock_vqa = MagicMock()
    mock_vqa.answer_question = AsyncMock(return_value=VQAResult(
        question="What is in the image?",
        answer="The image shows a cat.",
        image_id="test-img-001",
        confidence=0.92,
        model_used="blip2",
        time_ms=150.0,
    ))
    mock_vqa.generate_caption = AsyncMock(return_value=CaptionResult(
        image_id="test-img-001",
        caption="A fluffy cat sitting on a windowsill.",
        style="descriptive",
        model_used="blip",
        time_ms=80.0,
    ))
    mock_vqa.describe_image = AsyncMock(return_value={
        "description": "A cat on a windowsill.",
        "aspects": {"color": "orange", "scene": "indoor"},
    })
    mock_vqa.health_check.return_value = {"status": "healthy"}
    mock_vqa.unload = MagicMock()
    mock_vqa._get_caption_provider = MagicMock(return_value=MagicMock())
    adapter._vqa_provider = mock_vqa

    # Mock document processor
    from multimodal_rag.providers import ProcessedDocument
    mock_doc = MagicMock()
    mock_doc.process_document = AsyncMock(return_value=ProcessedDocument(
        id="doc-001",
        title="Test Document",
        pages=[],
        page_count=3,
        chunks=[
            MultiModalItem(id="chunk-1", modality=Modality.TEXT, text_content="Page 1 text"),
            MultiModalItem(id="chunk-2", modality=Modality.IMAGE),
        ],
    ))
    adapter._doc_processor = mock_doc

    # Mock cache
    mock_cache = MagicMock(spec=MultiModalCacheProvider)
    mock_cache.get_stats.return_value = {"hits": 100, "misses": 20}
    adapter._cache = mock_cache

    # Mock metrics
    mock_metrics = MagicMock(spec=MetricsCollector)
    mock_metrics.get_summary.return_value = {"total_requests": 50}
    mock_metrics.record_retrieval = MagicMock()
    adapter._metrics = mock_metrics

    return adapter


# ============================================================================
# TestA: Configuration Utilities
# ============================================================================

class TestA_Configuration:
    """Test config loading and coercion."""

    def test_coerce_true(self):
        for v in ("true", "True", "yes", "1", "on"):
            assert _coerce_value(v) is True

    def test_coerce_false(self):
        for v in ("false", "False", "no", "0", "off"):
            assert _coerce_value(v) is False

    def test_coerce_int(self):
        assert _coerce_value("42") == 42

    def test_coerce_float(self):
        assert _coerce_value("3.14") == 3.14

    def test_coerce_string(self):
        assert _coerce_value("hello") == "hello"

    def test_coerce_passthrough(self):
        assert _coerce_value(42) == 42
        assert _coerce_value(None) is None

    def test_load_config_exists(self):
        config = _load_config(_MODULE_ROOT)
        assert isinstance(config, dict)
        assert "embeddings" in config or "image_processing" in config


# ============================================================================
# TestB: Data Classes
# ============================================================================

class TestB_DataClasses:
    """Test core data classes."""

    def test_image_data_to_dict(self):
        img = ImageData(id="img-1", width=224, height=224)
        d = img.to_dict()
        assert d["id"] == "img-1"
        assert d["width"] == 224
        assert d["has_embedding"] is False

    def test_image_data_content_hash(self):
        img = ImageData(id="img-1", raw_bytes=b"hello")
        h = img.content_hash
        assert h == hashlib.md5(b"hello").hexdigest()

    def test_text_data_to_dict(self):
        td = TextData(id="t1", content="Hello world")
        d = td.to_dict()
        assert d["id"] == "t1"
        assert d["content"] == "Hello world"

    def test_multimodal_item_to_dict(self):
        item = MultiModalItem(id="mm-1", modality=Modality.TEXT, text_content="test")
        d = item.to_dict()
        assert d["modality"] == "text"
        assert d["has_image"] is False

    def test_retrieval_result_to_dict(self):
        rr = RetrievalResult(
            query="test", query_modality=Modality.TEXT,
            retrieval_mode=RetrievalMode.HYBRID,
            items=[MultiModalItem(id="r1", modality=Modality.TEXT)],
            time_ms=10.0,
        )
        d = rr.to_dict()
        assert d["result_count"] == 1
        assert d["retrieval_mode"] == "hybrid"

    def test_vqa_result_to_dict(self):
        vr = VQAResult(question="What?", answer="yes", image_id="img-1",
                       confidence=0.9, model_used="blip2", time_ms=50.0)
        d = vr.to_dict()
        assert d["answer"] == "yes"
        assert d["confidence"] == 0.9

    def test_caption_result_to_dict(self):
        cr = CaptionResult(image_id="img-1", caption="A cat.",
                           style="descriptive", model_used="blip", time_ms=30.0)
        d = cr.to_dict()
        assert d["caption"] == "A cat."

    def test_modality_enum(self):
        assert Modality.TEXT.value == "text"
        assert Modality.IMAGE.value == "image"
        assert Modality("document") == Modality.DOCUMENT

    def test_retrieval_mode_enum(self):
        assert RetrievalMode.HYBRID.value == "hybrid"
        assert RetrievalMode("text_to_image") == RetrievalMode.TEXT_TO_IMAGE

    def test_fusion_method_enum(self):
        assert FusionMethod.WEIGHTED.value == "weighted"


# ============================================================================
# TestC: Adapter Initialization
# ============================================================================

class TestC_Lifecycle:
    """Test adapter lifecycle operations."""

    def test_adapter_creation(self):
        adapter = _make_adapter()
        assert adapter._initialized is False
        assert adapter.module_path == _MODULE_ROOT
        assert isinstance(adapter.config, dict)

    @pytest.mark.asyncio
    async def test_health_check_uninitialized(self):
        adapter = _make_adapter()
        result = await adapter.health_check()
        assert result["module"] == "multimodal_rag"
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_shutdown(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.shutdown()
        assert result["status"] == "shutdown"
        assert adapter._initialized is False
        adapter._embedder.unload_all.assert_called_once()
        adapter._vqa_provider.unload.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_initialized(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.health_check()
        assert result["module"] == "multimodal_rag"
        assert "embedder" in result
        assert "vqa" in result
        assert "cache" in result

    def test_event_publisher_no_bus(self):
        adapter = _make_adapter()
        assert adapter.publisher is None

    def test_event_publisher_with_bus(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        adapter = _make_adapter(event_bus=bus)
        assert adapter.publisher is not None


# ============================================================================
# TestD: Image Operations
# ============================================================================

class TestD_ImageOps:
    """Test image processing operations."""

    @pytest.mark.asyncio
    async def test_load_image(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.load_image(source=b"\x89PNG\r\n", image_id="test-1")
        assert result["id"] == "test-img-001"
        adapter._image_processor.load_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_image_from_bytes(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.embed_image(image_source=b"\x89PNG\r\n")
        assert result["image_id"] == "test-img-001"
        assert result["embedding_dimension"] == 512
        assert "time_ms" in result

    @pytest.mark.asyncio
    async def test_embed_image_from_dict(self):
        adapter = _mock_initialized_adapter()
        img_dict = {"id": "img-x", "width": 100, "height": 100}
        result = await adapter.embed_image(image_source=img_dict)
        assert result["embedding_dimension"] == 512

    @pytest.mark.asyncio
    async def test_embed_text(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.embed_text(text="Hello world")
        assert result["embedding_dimension"] == 512
        assert result["text_preview"] == "Hello world"
        assert "time_ms" in result

    @pytest.mark.asyncio
    async def test_embed_text_long_preview(self):
        adapter = _mock_initialized_adapter()
        long_text = "x" * 200
        result = await adapter.embed_text(text=long_text)
        assert result["text_preview"].endswith("...")
        assert len(result["text_preview"]) == 103  # 100 + "..."


# ============================================================================
# TestE: OCR Operations
# ============================================================================

class TestE_OCR:
    """Test OCR text extraction."""

    @pytest.mark.asyncio
    async def test_extract_text_from_image(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.extract_text_from_image(image_source=b"\x89PNG")
        assert result["extracted_text"] == "Extracted OCR text from image."
        assert result["text_length"] > 0
        assert "time_ms" in result

    @pytest.mark.asyncio
    async def test_extract_text_ocr_disabled(self):
        adapter = _mock_initialized_adapter()
        adapter._ocr_provider = None
        result = await adapter.extract_text_from_image(image_source=b"\x89PNG")
        assert result["error"] == "OCR not enabled"
        assert result["text"] == ""

    @pytest.mark.asyncio
    async def test_extract_text_from_dict(self):
        adapter = _mock_initialized_adapter()
        img_dict = {"id": "ocr-img", "width": 100, "height": 100}
        result = await adapter.extract_text_from_image(image_source=img_dict)
        assert "extracted_text" in result


# ============================================================================
# TestF: Retrieval Operations
# ============================================================================

class TestF_Retrieval:
    """Test retrieval operations."""

    @pytest.mark.asyncio
    async def test_retrieve_text_query(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.retrieve(query="test query", mode="hybrid", top_k=5)
        assert result["result_count"] == 2
        assert result["retrieval_mode"] == "hybrid"
        adapter._retriever.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_to_image_search(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.text_to_image_search(text="cat", top_k=3)
        assert result["result_count"] == 2

    @pytest.mark.asyncio
    async def test_image_to_text_search(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.image_to_text_search(image_source=b"\x89PNG", top_k=5)
        assert result["result_count"] == 2

    @pytest.mark.asyncio
    async def test_find_similar_images(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.find_similar_images(image_source=b"\x89PNG", top_k=5)
        assert result["result_count"] == 2

    @pytest.mark.asyncio
    async def test_retrieve_records_metrics(self):
        adapter = _mock_initialized_adapter()
        await adapter.retrieve(query="test", mode="hybrid")
        adapter._metrics.record_retrieval.assert_called_once()


# ============================================================================
# TestG: Indexing Operations
# ============================================================================

class TestG_Indexing:
    """Test indexing operations."""

    @pytest.mark.asyncio
    async def test_index_item_text(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.index_item(item={
            "id": "item-1", "modality": "text", "text_content": "Hello world",
        })
        assert result["indexed"] is True
        assert result["item_id"] == "item-1"
        adapter._retriever.add_to_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_item_with_image(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.index_item(item={
            "id": "item-2", "modality": "image",
            "image_source": b"\x89PNG",
        })
        assert result["indexed"] is True
        adapter._image_processor.load_image.assert_called()

    @pytest.mark.asyncio
    async def test_index_batch(self):
        adapter = _mock_initialized_adapter()
        items = [
            {"id": "b1", "modality": "text", "text_content": "Text 1"},
            {"id": "b2", "modality": "text", "text_content": "Text 2"},
        ]
        result = await adapter.index_batch(items=items)
        assert result["indexed_count"] == 1  # mocked
        assert result["total_in_index"] == 10

    @pytest.mark.asyncio
    async def test_get_index_stats(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.get_index_stats()
        assert result["total_items"] == 10


# ============================================================================
# TestH: VQA and Captioning
# ============================================================================

class TestH_VQA:
    """Test VQA and captioning operations."""

    @pytest.mark.asyncio
    async def test_answer_question(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.answer_question(
            image_source=b"\x89PNG", question="What is in the image?",
        )
        assert result["answer"] == "The image shows a cat."
        assert result["confidence"] == 0.92

    @pytest.mark.asyncio
    async def test_answer_question_vqa_disabled(self):
        adapter = _mock_initialized_adapter()
        adapter._vqa_provider = None
        result = await adapter.answer_question(
            image_source=b"\x89PNG", question="What?",
        )
        assert result["error"] == "VQA not enabled"

    @pytest.mark.asyncio
    async def test_generate_caption(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.generate_caption(image_source=b"\x89PNG", style="descriptive")
        assert "fluffy cat" in result["caption"]
        assert result["style"] == "descriptive"

    @pytest.mark.asyncio
    async def test_generate_caption_disabled(self):
        adapter = _mock_initialized_adapter()
        adapter._vqa_provider = None
        result = await adapter.generate_caption(image_source=b"\x89PNG")
        assert result["error"] == "Captioning not enabled"

    @pytest.mark.asyncio
    async def test_describe_image(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.describe_image(
            image_source=b"\x89PNG", aspects=["color", "scene"],
        )
        assert "description" in result

    @pytest.mark.asyncio
    async def test_describe_image_disabled(self):
        adapter = _mock_initialized_adapter()
        adapter._vqa_provider = None
        result = await adapter.describe_image(image_source=b"\x89PNG")
        assert result["error"] == "VQA not enabled"


# ============================================================================
# TestI: Document Processing
# ============================================================================

class TestI_Document:
    """Test document processing operations."""

    @pytest.mark.asyncio
    async def test_process_document(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.process_document(
            source=b"%PDF-1.4 ...", document_id="doc-001",
            generate_captions=True, run_ocr=True, auto_index=True,
        )
        assert result["id"] == "doc-001"
        assert result["page_count"] == 3
        assert result["auto_indexed"] is True
        adapter._retriever.add_to_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_document_no_index(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.process_document(
            source=b"%PDF-1.4", auto_index=False,
        )
        assert result["auto_indexed"] is False
        adapter._retriever.add_to_index.assert_not_called()


# ============================================================================
# TestJ: Statistics
# ============================================================================

class TestJ_Stats:
    """Test statistics operations."""

    @pytest.mark.asyncio
    async def test_get_stats(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.get_stats(period="24h")
        assert result["module"] == "multimodal_rag"
        assert result["period"] == "24h"
        assert "metrics" in result
        assert "index" in result
        assert "cache" in result

    @pytest.mark.asyncio
    async def test_get_stats_no_components(self):
        adapter = _make_adapter()
        result = await adapter.get_stats()
        assert result["module"] == "multimodal_rag"
        assert "metrics" not in result


# ============================================================================
# TestK: New Operations (compute_similarity, multimodal_query, clear_index)
# ============================================================================

class TestK_NewOps:
    """Test newly implemented operations."""

    @pytest.mark.asyncio
    async def test_compute_similarity(self):
        adapter = _mock_initialized_adapter()
        # Set embeddings to known values for deterministic similarity
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        adapter._embedder.embed_text = AsyncMock(return_value=v1)
        adapter._embedder.embed_image = AsyncMock(return_value=v2)
        result = await adapter.compute_similarity(text="cat", image_source=b"\x89PNG")
        assert "similarity" in result
        assert 0.99 <= result["similarity"] <= 1.01  # cosine of identical vectors = 1.0
        assert "time_ms" in result

    @pytest.mark.asyncio
    async def test_compute_similarity_orthogonal(self):
        adapter = _mock_initialized_adapter()
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        adapter._embedder.embed_text = AsyncMock(return_value=v1)
        adapter._embedder.embed_image = AsyncMock(return_value=v2)
        result = await adapter.compute_similarity(text="a", image_source=b"\x89PNG")
        assert abs(result["similarity"]) < 0.01  # orthogonal = 0

    @pytest.mark.asyncio
    async def test_multimodal_query(self):
        adapter = _mock_initialized_adapter()
        result = await adapter.multimodal_query(
            text="find cats", image_source=b"\x89PNG",
            fusion_method="average", top_k=5,
        )
        assert result["result_count"] == 2
        assert result["retrieval_mode"] == "hybrid"

    @pytest.mark.asyncio
    async def test_clear_index(self):
        adapter = _mock_initialized_adapter()
        adapter._retriever.index = MagicMock()
        adapter._retriever.index.count = MagicMock(return_value=42)
        adapter._retriever.index.clear = MagicMock()
        result = await adapter.clear_index()
        assert result["status"] == "cleared"
        assert result["items_removed"] == 42
        adapter._retriever.index.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_index_uninitialized(self):
        adapter = _make_adapter()
        adapter._initialized = True
        adapter._retriever = None
        result = await adapter.clear_index()
        assert result["status"] == "no_index"


# ============================================================================
# TestL: Auto-initialization Guard
# ============================================================================

class TestL_AutoInit:
    """Test that operations auto-initialize when not initialized."""

    @pytest.mark.asyncio
    async def test_load_image_auto_init(self):
        adapter = _make_adapter()
        adapter.initialize = AsyncMock(return_value={"status": "initialized"})
        adapter._image_processor = MagicMock()
        adapter._image_processor.load_image.return_value = ImageData(id="auto-1")
        # Should call initialize since _initialized is False
        result = await adapter.load_image(source=b"\x89PNG")
        adapter.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_auto_init(self):
        adapter = _make_adapter()
        adapter.initialize = AsyncMock(return_value={"status": "initialized"})
        adapter._retriever = MagicMock()
        adapter._retriever.retrieve = AsyncMock(return_value=RetrievalResult(
            query="q", query_modality=Modality.TEXT,
            retrieval_mode=RetrievalMode.HYBRID, items=[], time_ms=1.0,
        ))
        adapter._metrics = MagicMock()
        result = await adapter.retrieve(query="test")
        adapter.initialize.assert_called_once()


# ============================================================================
# TestM: Error Handling
# ============================================================================

class TestM_Errors:
    """Test error handling paths."""

    @pytest.mark.asyncio
    async def test_embed_image_error(self):
        adapter = _mock_initialized_adapter()
        adapter._embedder.embed_image = AsyncMock(side_effect=RuntimeError("GPU OOM"))
        with pytest.raises(RuntimeError, match="GPU OOM"):
            await adapter.embed_image(image_source=b"\x89PNG")

    @pytest.mark.asyncio
    async def test_retrieve_invalid_mode(self):
        adapter = _mock_initialized_adapter()
        with pytest.raises(ValueError):
            await adapter.retrieve(query="test", mode="invalid_mode")

    @pytest.mark.asyncio
    async def test_event_publish_failure_silent(self):
        bus = MagicMock()
        bus.publish = AsyncMock(side_effect=Exception("bus down"))
        adapter = _make_adapter(event_bus=bus)
        adapter._initialized = True
        adapter._embedder = MagicMock()
        adapter._embedder.unload_all = MagicMock()
        adapter._vqa_provider = None
        # shutdown publishes event — should not raise
        result = await adapter.shutdown()
        assert result["status"] == "shutdown"
