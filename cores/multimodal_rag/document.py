"""
multimodal_rag/document.py

Document processing with image extraction.

Provides:
- PDFProcessor: Extract text and images from PDF
- DocumentChunker: Multi-modal aware chunking
- ImageTextAssociator: Link images to nearby text
- TableExtractor: Extract tables as structured data

Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from .providers import (
    ImageData,
    TextData,
    MultiModalItem,
    DocumentPage,
    ProcessedDocument,
    Modality,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DocumentConfig:
    """Document processing configuration."""
    extract_images: bool = True
    extract_tables: bool = True
    ocr_scanned_pages: bool = True
    dpi: int = 150
    min_image_size: int = 50
    max_images_per_page: int = 10


@dataclass
class ChunkingConfig:
    """Chunking configuration."""
    max_chunk_size: int = 1000
    chunk_size: int = 750  # v6.4.1: Target size for merge threshold
    overlap: int = 100
    preserve_image_context: bool = True
    include_caption_in_chunk: bool = True
    include_ocr_in_chunk: bool = True
    max_images_per_chunk: int = 3


# ============================================================================
# PDF Processor
# ============================================================================


class PDFProcessor:
    """
    Process PDF documents extracting text and images.
    
    Features:
    - Text extraction with layout preservation
    - Image extraction with metadata
    - Table detection
    - Scanned page OCR
    """
    
    def __init__(self, config: DocumentConfig):
        self.config = config
        self._pymupdf_available = False
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check available PDF libraries."""
        try:
            import fitz  # PyMuPDF
            self._pymupdf_available = True
        except ImportError:
            logger.warning("PyMuPDF not available, PDF processing limited")
    
    def process(
        self,
        source: Union[str, Path, bytes, BinaryIO],
        document_id: Optional[str] = None,
    ) -> ProcessedDocument:
        """Process PDF document."""
        if not self._pymupdf_available:
            raise RuntimeError("PyMuPDF required for PDF processing")
        
        import fitz
        
        # Open document
        if isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
            doc_id = document_id or hashlib.md5(source[:1000]).hexdigest()[:12]
        elif isinstance(source, (str, Path)):
            doc = fitz.open(source)
            doc_id = document_id or Path(source).stem
        else:
            data = source.read()
            doc = fitz.open(stream=data, filetype="pdf")
            doc_id = document_id or hashlib.md5(data[:1000]).hexdigest()[:12]
        
        # Extract pages
        pages = []
        all_images = []
        all_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text("text")
            all_text.append(text)
            
            # Extract images
            images = []
            if self.config.extract_images:
                images = self._extract_page_images(page, page_num, doc_id)
                all_images.extend(images)
            
            # Extract tables (basic)
            tables = []
            if self.config.extract_tables:
                tables = self._extract_tables(page)
            
            pages.append(DocumentPage(
                page_number=page_num + 1,
                text_content=text,
                images=images,
                tables=tables,
                width=page.rect.width,
                height=page.rect.height,
            ))
        
        doc.close()
        
        return ProcessedDocument(
            id=doc_id,
            title=doc_id,
            pages=pages,
            full_text="\n\n".join(all_text),
            all_images=all_images,
            page_count=len(pages),
            file_type="pdf",
            metadata={"source": str(source) if isinstance(source, (str, Path)) else "bytes"},
        )
    
    def _extract_page_images(
        self,
        page: Any,
        page_num: int,
        doc_id: str,
    ) -> List[ImageData]:
        """Extract images from PDF page."""
        images = []
        
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list[:self.config.max_images_per_page]):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                
                if not base_image:
                    continue
                
                image_bytes = base_image["image"]
                
                # Check minimum size
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                
                if width < self.config.min_image_size or height < self.config.min_image_size:
                    continue
                
                image_id = f"{doc_id}_p{page_num + 1}_img{img_index + 1}"
                
                images.append(ImageData(
                    id=image_id,
                    raw_bytes=image_bytes,
                    width=width,
                    height=height,
                    format=base_image.get("ext", "png").upper(),
                    metadata={
                        "page": page_num + 1,
                        "index": img_index,
                        "xref": xref,
                    },
                ))
                
            except Exception as e:
                logger.warning(f"Failed to extract image: {e}")
        
        return images
    
    def _extract_tables(self, page: Any) -> List[Dict[str, Any]]:
        """Extract tables from page (basic implementation)."""
        tables = []
        
        # Try to find table-like structures in text
        text = page.get_text("text")
        lines = text.split("\n")
        
        # Simple heuristic: look for lines with consistent separators
        current_table = []
        
        for line in lines:
            # Check for table-like patterns
            if "|" in line or "\t" in line:
                cells = re.split(r'\||\t', line)
                cells = [c.strip() for c in cells if c.strip()]
                if len(cells) >= 2:
                    current_table.append(cells)
            elif current_table:
                if len(current_table) >= 2:
                    tables.append({
                        "rows": current_table,
                        "num_rows": len(current_table),
                        "num_cols": max(len(row) for row in current_table),
                    })
                current_table = []
        
        return tables
    
    def render_page_as_image(
        self,
        source: Union[str, Path, bytes],
        page_num: int,
        dpi: Optional[int] = None,
    ) -> ImageData:
        """Render PDF page as image."""
        if not self._pymupdf_available:
            raise RuntimeError("PyMuPDF required")
        
        import fitz
        
        dpi = dpi or self.config.dpi
        
        if isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            doc = fitz.open(source)
        
        page = doc[page_num]
        
        # Render
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to bytes
        img_bytes = pix.tobytes("png")
        
        doc.close()
        
        return ImageData(
            id=f"page_{page_num + 1}_render",
            raw_bytes=img_bytes,
            width=pix.width,
            height=pix.height,
            format="PNG",
            metadata={"page": page_num + 1, "dpi": dpi},
        )


# ============================================================================
# Document Chunker
# ============================================================================


class DocumentChunker:
    """
    Multi-modal aware document chunking.
    
    Creates chunks that preserve image-text relationships.
    """
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
    
    def chunk_document(
        self,
        document: ProcessedDocument,
        image_captions: Optional[Dict[str, str]] = None,
        ocr_results: Optional[Dict[str, str]] = None,
    ) -> List[MultiModalItem]:
        """Chunk document into multi-modal items."""
        chunks = []
        image_captions = image_captions or {}
        ocr_results = ocr_results or {}
        
        for page in document.pages:
            page_chunks = self._chunk_page(
                page,
                document.id,
                image_captions,
                ocr_results,
            )
            chunks.extend(page_chunks)
        
        document.chunks = chunks
        return chunks
    
    def _chunk_page(
        self,
        page: DocumentPage,
        doc_id: str,
        image_captions: Dict[str, str],
        ocr_results: Dict[str, str],
    ) -> List[MultiModalItem]:
        """Chunk a single page."""
        chunks = []
        
        text = page.text_content
        images = page.images
        
        # Simple chunking by size
        text_chunks = self._split_text(text)
        
        # Associate images with chunks
        images_per_chunk = self._distribute_images(images, len(text_chunks))
        
        for i, text_chunk in enumerate(text_chunks):
            chunk_id = f"{doc_id}_p{page.page_number}_c{i + 1}"
            
            # Get associated images
            chunk_images = images_per_chunk.get(i, [])
            
            # Build text content
            content_parts = [text_chunk]
            
            # Add captions
            for img in chunk_images:
                if img.id in image_captions and self.config.include_caption_in_chunk:
                    content_parts.append(f"[Image: {image_captions[img.id]}]")
                if img.id in ocr_results and self.config.include_ocr_in_chunk:
                    ocr_text = ocr_results[img.id]
                    if ocr_text:
                        content_parts.append(f"[Text in image: {ocr_text[:200]}]")
            
            full_content = "\n".join(content_parts)
            
            # Determine modality
            if chunk_images and text_chunk.strip():
                modality = Modality.MIXED
            elif chunk_images:
                modality = Modality.IMAGE
            else:
                modality = Modality.TEXT
            
            # Create item
            item = MultiModalItem(
                id=chunk_id,
                modality=modality,
                text_content=full_content,
                image_data=chunk_images[0] if chunk_images else None,
                source_id=doc_id,
                page_number=page.page_number,
                metadata={
                    "chunk_index": i,
                    "image_count": len(chunk_images),
                    "text_length": len(text_chunk),
                },
            )
            
            chunks.append(item)
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if not text or not text.strip():
            return [""]
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # v6.4.1: Use chunk_size (target) not max_chunk_size (hard limit)
            if current_size + para_size <= self.config.chunk_size:
                current_chunk.append(para)
                current_size += para_size
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                
                # Start new chunk
                if para_size <= self.config.max_chunk_size:
                    current_chunk = [para]
                    current_size = para_size
                else:
                    # Split large paragraph
                    sub_chunks = self._split_large_text(para)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = [sub_chunks[-1]] if sub_chunks else []
                    current_size = len(current_chunk[0]) if current_chunk else 0
        
        # Add remaining
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks if chunks else [""]
    
    def _split_large_text(self, text: str) -> List[str]:
        """Split large text by sentences with overlap."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current = []
        current_size = 0
        overlap = self.config.overlap  # v6.4.1: Use configured overlap
        
        for sentence in sentences:
            if current_size + len(sentence) <= self.config.max_chunk_size:
                current.append(sentence)
                current_size += len(sentence)
            else:
                if current:
                    chunks.append(" ".join(current))
                    # v6.4.1: Carry overlap sentences to next chunk
                    overlap_sentences = []
                    overlap_size = 0
                    for s in reversed(current):
                        if overlap_size + len(s) <= overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += len(s)
                        else:
                            break
                    current = overlap_sentences + [sentence]
                    current_size = overlap_size + len(sentence)
                else:
                    current = [sentence]
                    current_size = len(sentence)
        
        if current:
            chunks.append(" ".join(current))
        
        return chunks
    
    def _distribute_images(
        self,
        images: List[ImageData],
        num_chunks: int,
    ) -> Dict[int, List[ImageData]]:
        """Distribute images across chunks."""
        if not images or num_chunks == 0:
            return {}
        
        distribution = {}
        images_per_chunk = max(1, len(images) // max(num_chunks, 1))
        
        for i, img in enumerate(images):
            chunk_idx = min(i // max(images_per_chunk, 1), num_chunks - 1)
            
            if chunk_idx not in distribution:
                distribution[chunk_idx] = []
            
            if len(distribution[chunk_idx]) < self.config.max_images_per_chunk:
                distribution[chunk_idx].append(img)
        
        return distribution


# ============================================================================
# Image-Text Associator
# ============================================================================


class ImageTextAssociator:
    """
    Associates images with nearby text.
    
    Uses:
    - Spatial proximity
    - Caption detection
    - Reference matching
    """
    
    def __init__(self, proximity_threshold: int = 100):
        self.proximity_threshold = proximity_threshold
    
    def associate(
        self,
        page: DocumentPage,
    ) -> Dict[str, str]:
        """Associate images with text on a page."""
        associations = {}
        
        text = page.text_content
        
        for image in page.images:
            # Try to find caption
            caption = self._find_caption(image, text)
            
            if caption:
                associations[image.id] = caption
            else:
                # Use surrounding text
                surrounding = self._get_surrounding_text(image, text)
                if surrounding:
                    associations[image.id] = surrounding
        
        return associations
    
    def _find_caption(
        self,
        image: ImageData,
        text: str,
    ) -> Optional[str]:
        """Find caption for image."""
        # Look for common caption patterns
        patterns = [
            r'(?:Figure|Fig\.?)\s*\d+[:.]\s*([^\n]+)',
            r'(?:Image|Img\.?)\s*\d+[:.]\s*([^\n]+)',
            r'(?:Photo|Picture)\s*\d*[:.]\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _get_surrounding_text(
        self,
        image: ImageData,
        text: str,
        max_length: int = 200,
    ) -> Optional[str]:
        """Get text surrounding image location."""
        # Simple: return first N characters
        if text:
            return text[:max_length].strip()
        return None


# ============================================================================
# Unified Document Processor
# ============================================================================


class UnifiedDocumentProcessor:
    """
    Unified document processing pipeline.
    
    Combines:
    - PDF extraction
    - Image processing
    - OCR
    - Chunking
    - Association
    """
    
    def __init__(
        self,
        doc_config: Optional[DocumentConfig] = None,
        chunk_config: Optional[ChunkingConfig] = None,
    ):
        self.doc_config = doc_config or DocumentConfig()
        self.chunk_config = chunk_config or ChunkingConfig()
        
        self._pdf_processor = PDFProcessor(self.doc_config)
        self._chunker = DocumentChunker(self.chunk_config)
        self._associator = ImageTextAssociator()
    
    async def process_document(
        self,
        source: Union[str, Path, bytes, BinaryIO],
        document_id: Optional[str] = None,
        generate_captions: bool = True,
        run_ocr: bool = True,
        caption_provider: Optional[Any] = None,
        ocr_provider: Optional[Any] = None,
    ) -> ProcessedDocument:
        """
        Process document through full pipeline.
        
        Steps:
        1. Extract text and images from PDF
        2. Associate images with text
        3. Generate captions (optional)
        4. Run OCR on images (optional)
        5. Chunk into multi-modal items
        """
        # Extract from PDF
        document = self._pdf_processor.process(source, document_id)
        logger.info(f"Extracted {document.page_count} pages, {len(document.all_images)} images")
        
        # Associate images with text
        all_captions = {}
        for page in document.pages:
            page_associations = self._associator.associate(page)
            all_captions.update(page_associations)
        
        # Generate captions
        if generate_captions and caption_provider:
            for image in document.all_images:
                if image.id not in all_captions:
                    try:
                        result = await caption_provider.caption(image)
                        all_captions[image.id] = result.caption
                        image.caption = result.caption
                    except Exception as e:
                        logger.warning(f"Caption generation failed: {e}")
        
        # Run OCR
        ocr_results = {}
        if run_ocr and ocr_provider:
            for image in document.all_images:
                try:
                    ocr_text = ocr_provider.extract_text(image)
                    if ocr_text:
                        ocr_results[image.id] = ocr_text
                        image.ocr_text = ocr_text
                except Exception as e:
                    logger.warning(f"OCR failed: {e}")
        
        # Chunk document
        chunks = self._chunker.chunk_document(document, all_captions, ocr_results)
        logger.info(f"Created {len(chunks)} chunks")
        
        return document
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "pdf_available": self._pdf_processor._pymupdf_available,
            "config": {
                "extract_images": self.doc_config.extract_images,
                "extract_tables": self.doc_config.extract_tables,
                "max_chunk_size": self.chunk_config.max_chunk_size,
            },
        }
