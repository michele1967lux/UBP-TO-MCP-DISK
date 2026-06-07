# Multimodal RAG

**Multi-Modal Retrieval Augmented Generation** - Image, text, document understanding with cross-modal retrieval, VQA, and unified embedding space.

Version: 1.0.0 | Architecture: Enterprise | Module Type: retrieval

---

## Overview

`multimodal_rag` provides a complete multi-modal RAG solution that combines text and visual understanding in a unified system. It enables searching images with text, finding text from images, visual question answering, document processing with embedded images, and intelligent cross-modal retrieval.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Cross-Modal Retrieval** | Text→Image, Image→Text, Image→Image search |
| **Unified Embedding Space** | CLIP/BLIP embeddings for text and images |
| **Visual QA (VQA)** | Answer questions about images |
| **Image Captioning** | Generate descriptions for images |
| **OCR Integration** | Extract text from images |
| **Document Processing** | PDF with embedded images |
| **Multi-Modal Chunking** | Preserve image-text relationships |
| **Result Fusion** | Combine multi-modal search results |

---

## Architecture

```
multimodal_rag/
├── __init__.py        # Module factory
├── adapter.py         # Bridge layer - all operations
├── providers.py       # Data classes, ImageProcessor, OCR, Cache
├── embeddings.py      # CLIP, BLIP, SigLIP embedders
├── retrieval.py       # Cross-modal retrieval strategies
├── vqa.py             # VQA and captioning providers
├── document.py        # PDF/document processing
├── config.json        # Configuration (100+ options)
├── manifest.json      # 22 operations defined
└── README.md          # This file
```

### Component Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                      multimodal_rag                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Image     │    │    Text     │    │  Document   │         │
│  │  Processor  │    │  Processor  │    │  Processor  │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌──────────────────────────────────────────────────┐          │
│  │           Unified Embedder (CLIP/BLIP)           │          │
│  │  ┌────────┐  ┌────────┐  ┌────────┐             │          │
│  │  │  CLIP  │  │  BLIP  │  │ SigLIP │             │          │
│  │  └────────┘  └────────┘  └────────┘             │          │
│  └──────────────────────────────────────────────────┘          │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────┐          │
│  │            Multi-Modal Index                      │          │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐         │          │
│  │  │  Text    │ │  Image   │ │  Mixed   │         │          │
│  │  │ Vectors  │ │ Vectors  │ │ Vectors  │         │          │
│  │  └──────────┘ └──────────┘ └──────────┘         │          │
│  └──────────────────────────────────────────────────┘          │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────┐          │
│  │          Cross-Modal Retriever                    │          │
│  │  • Text→Image  • Image→Text  • Hybrid            │          │
│  │  • Image→Image • Unified     • Reranking         │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │  VQA Engine  │    │  Captioning  │                          │
│  │   (BLIP-2)   │    │    (BLIP)    │                          │
│  └──────────────┘    └──────────────┘                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Basic Usage

```python
from multimodal_rag import create_module

# Initialize
mmrag = create_module(module_path)
await mmrag.initialize(preload_models=True)

# Index content
await mmrag.index_item({
    "id": "doc1_img1",
    "modality": "image",
    "image_source": "/path/to/image.jpg",
    "text_content": "A sunset over mountains"
})

# Search images with text
results = await mmrag.text_to_image_search(
    text="mountain landscape at sunset",
    top_k=5
)

# Answer questions about images
answer = await mmrag.answer_question(
    image_source="/path/to/image.jpg",
    question="What time of day is shown in this image?"
)
# → "The image shows sunset, with warm orange and red colors in the sky"

# Process PDF with images
doc = await mmrag.process_document(
    source="/path/to/document.pdf",
    generate_captions=True,
    run_ocr=True,
    auto_index=True
)
```

---

## Retrieval Modes

### 1. Text to Image (`text_to_image`)

Find images matching a text description.

```python
# Find images of cats
results = await mmrag.text_to_image_search(
    text="orange tabby cat sleeping on a couch",
    top_k=10
)

for item in results["items"]:
    print(f"Image: {item['id']}, Score: {item['relevance_score']}")
```

### 2. Image to Text (`image_to_text`)

Find text descriptions or documents related to an image.

```python
# Find text about this image
results = await mmrag.image_to_text_search(
    image_source="/path/to/query_image.jpg",
    top_k=10
)

for item in results["items"]:
    print(f"Text: {item['text_content'][:100]}...")
```

### 3. Image to Image (`image_to_image`)

Find visually similar images.

```python
# Find similar images
results = await mmrag.find_similar_images(
    image_source="/path/to/reference.jpg",
    top_k=5
)
```

### 4. Hybrid Search (`hybrid`)

Search across all modalities simultaneously.

```python
# Hybrid search - returns both images and text
results = await mmrag.retrieve(
    query="technical diagram of neural network architecture",
    mode="hybrid",
    top_k=10
)

# Results may include:
# - Images of neural network diagrams
# - Text explaining neural network architecture
# - Documents with embedded architecture diagrams
```

### 5. Multi-Modal Query

Query with both text AND image for enhanced retrieval.

```python
# Use both text and image as query
results = await mmrag.multimodal_query(
    text="similar style but different color",
    image_source="/path/to/reference.jpg",
    fusion_method="weighted",
    top_k=10
)
```

---

## Visual Question Answering (VQA)

Ask questions about images and get natural language answers.

```python
# Basic VQA
result = await mmrag.answer_question(
    image_source="chart.png",
    question="What is the trend shown in this chart?"
)
print(result["answer"])
# → "The chart shows an upward trend with revenue increasing from Q1 to Q4"

# Multiple questions
questions = [
    "What type of chart is this?",
    "What are the axis labels?",
    "What is the maximum value shown?"
]

for q in questions:
    result = await mmrag.answer_question(image_source="chart.png", question=q)
    print(f"Q: {q}\nA: {result['answer']}\n")
```

### Comprehensive Image Description

```python
# Get detailed description with multiple aspects
description = await mmrag.describe_image(
    image_source="photo.jpg",
    aspects=["content", "objects", "colors", "scene", "mood"]
)

print(f"Caption: {description['caption']}")
print(f"Main content: {description['aspects']['content']}")
print(f"Objects: {description['aspects']['objects']}")
print(f"Colors: {description['aspects']['colors']}")
print(f"Scene type: {description['aspects']['scene']}")
print(f"Mood: {description['aspects']['mood']}")
```

---

## Image Captioning

Generate captions with different styles.

```python
# Descriptive caption (default)
caption = await mmrag.generate_caption(
    image_source="photo.jpg",
    style="descriptive"
)
# → "A golden retriever playing fetch in a sunny park"

# Detailed caption
caption = await mmrag.generate_caption(
    image_source="photo.jpg",
    style="detailed"
)
# → "A golden retriever with light fur is caught mid-jump while playing fetch in a green park on a sunny afternoon, with trees visible in the background"

# Concise caption
caption = await mmrag.generate_caption(
    image_source="photo.jpg",
    style="concise"
)
# → "Dog playing in park"

# Technical caption
caption = await mmrag.generate_caption(
    image_source="diagram.png",
    style="technical"
)
# → "System architecture diagram showing microservices communication via REST APIs"
```

---

## OCR - Text Extraction from Images

Extract text from images using Tesseract or EasyOCR.

```python
# Extract text from image
result = await mmrag.extract_text_from_image(
    image_source="scanned_document.png"
)

print(f"Extracted text ({result['text_length']} chars):")
print(result["extracted_text"])
```

### Supported OCR Engines

| Engine | Languages | GPU | Notes |
|--------|-----------|-----|-------|
| Tesseract | 100+ | ❌ | Fast, widely available |
| EasyOCR | 80+ | ✅ | Better accuracy, slower |
| PaddleOCR | 80+ | ✅ | Best for Asian languages |

---

## Document Processing

Process PDFs extracting text, images, and their relationships.

```python
# Process PDF with full pipeline
doc = await mmrag.process_document(
    source="report.pdf",
    document_id="q4_report",
    generate_captions=True,   # Caption extracted images
    run_ocr=True,             # OCR on image text
    auto_index=True           # Automatically index chunks
)

print(f"Document: {doc['title']}")
print(f"Pages: {doc['page_count']}")
print(f"Images extracted: {doc['total_images']}")
print(f"Chunks created: {doc['total_chunks']}")
print(f"Processing time: {doc['time_ms']}ms")
```

### Multi-Modal Chunking

Documents are chunked preserving image-text relationships:

```
┌─────────────────────────────────────────┐
│ Chunk 1 (Mixed)                         │
│ ┌─────────────────────────────────────┐ │
│ │ Text: "Figure 1 shows the system    │ │
│ │ architecture. The main components   │ │
│ │ include..."                         │ │
│ │                                     │ │
│ │ [Image: System architecture diagram]│ │
│ │ Caption: "Microservices architecture│ │
│ │ with API gateway"                   │ │
│ │                                     │ │
│ │ OCR Text: "API Gateway, Service A,  │ │
│ │ Service B, Database"                │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

---

## Embedding Models

### CLIP (OpenAI)

Default choice for balanced performance.

```python
# CLIP ViT-B/32 (default)
# Dimension: 512, Fast, Good accuracy

# CLIP ViT-L/14
# Dimension: 768, Slower, Better accuracy
```

### BLIP (Salesforce)

Better for captioning and VQA.

```python
# BLIP-base
# Good for captioning

# BLIP-2
# State-of-the-art VQA and captioning
```

### SigLIP (Google)

Sigmoid loss, better calibrated similarities.

```python
# SigLIP-base
# Dimension: 768, Well-calibrated scores
```

### Model Comparison

| Model | Dimension | Speed | Text→Image | VQA | Captioning |
|-------|-----------|-------|------------|-----|------------|
| CLIP-B/32 | 512 | ⚡⚡⚡ | ⭐⭐⭐ | ❌ | ❌ |
| CLIP-L/14 | 768 | ⚡⚡ | ⭐⭐⭐⭐ | ❌ | ❌ |
| BLIP | 768 | ⚡⚡ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| BLIP-2 | 768 | ⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| SigLIP | 768 | ⚡⚡ | ⭐⭐⭐⭐ | ❌ | ❌ |

---

## Cross-Modal Reranking

Results are reranked using multiple signals:

```python
# Reranking combines:
# 1. Text-Text similarity (query text ↔ item text)
# 2. Image-Image similarity (query image ↔ item image)  
# 3. Cross-Modal similarity (text ↔ image)

# Weights can be configured:
reranking_weights = {
    "text_text": 0.3,
    "image_image": 0.3,
    "cross_modal": 0.4
}
```

---

## Configuration

### Key Environment Variables

```bash
# Embedding Model
UBP_MMRAG__EMBED_MODEL=openai/clip-vit-base-patch32
UBP_MMRAG__DEVICE=auto  # auto, cuda, cpu
UBP_MMRAG__EMBED_BATCH_SIZE=16

# Image Processing
UBP_MMRAG__IMG_SIZE=224
UBP_MMRAG__IMG_MAX_SIZE=1024
UBP_MMRAG__IMG_PRESERVE_ASPECT=true

# OCR
UBP_MMRAG__OCR_ENABLED=true
UBP_MMRAG__OCR_ENGINE=tesseract  # tesseract, easyocr, paddleocr
UBP_MMRAG__OCR_LANGUAGES=eng,ita
UBP_MMRAG__OCR_CONFIDENCE=0.6

# VQA
UBP_MMRAG__VQA_ENABLED=true
UBP_MMRAG__VQA_MODEL=blip2
UBP_MMRAG__VQA_MAX_LENGTH=100

# Captioning
UBP_MMRAG__CAPTION_ENABLED=true
UBP_MMRAG__CAPTION_MODEL=blip
UBP_MMRAG__CAPTION_MAX_LENGTH=75

# Retrieval
UBP_MMRAG__RETRIEVAL_STRATEGY=hybrid
UBP_MMRAG__RETRIEVAL_TOP_K=10
UBP_MMRAG__RETRIEVAL_THRESHOLD=0.5
UBP_MMRAG__RERANK_ENABLED=true

# Document Processing
UBP_MMRAG__DOC_EXTRACT_IMAGES=true
UBP_MMRAG__DOC_EXTRACT_TABLES=true
UBP_MMRAG__DOC_DPI=150

# Cache
UBP_MMRAG__CACHE_ENABLED=true
UBP_MMRAG__CACHE_TTL=7200
```

---

## Integration Examples

### With RAG Pipeline

```python
# 1. Process user query with potential image
query_text = "Find documents about this product"
query_image = user_uploaded_image

# 2. Multi-modal retrieval
results = await mmrag.retrieve(
    query=query_text,
    mode="hybrid",
    top_k=20
)

# 3. If query includes image, use multi-modal query
if query_image:
    results = await mmrag.multimodal_query(
        text=query_text,
        image_source=query_image,
        fusion_method="weighted",
        top_k=20
    )

# 4. Enrich with VQA if needed
for item in results["items"]:
    if item["modality"] == "image":
        context = await mmrag.answer_question(
            image_source=item["image_data"],
            question=f"How does this relate to: {query_text}"
        )
        item["vqa_context"] = context["answer"]

# 5. Send to LLM for final response
```

### With Agentic RAG

```python
# Agent can decide to use visual search
agent_tools = {
    "text_search": retriever.search,
    "image_search": mmrag.text_to_image_search,
    "visual_qa": mmrag.answer_question,
    "describe_image": mmrag.describe_image,
}

# Agent reasoning:
# "User is asking about a chart in the document.
#  I should use visual_qa to analyze the chart."
```

---

## Performance Optimization

### GPU Memory Management

```python
# Lazy loading (default) - load models on demand
await mmrag.initialize(preload_models=False)

# Preload for faster first query
await mmrag.initialize(preload_models=True)

# Unload to free memory
await mmrag.shutdown()
```

### Batch Processing

```python
# Index many items efficiently
items = [{"id": f"img_{i}", "image_source": f"image_{i}.jpg"} 
         for i in range(1000)]

result = await mmrag.index_batch(items)
print(f"Indexed {result['indexed_count']} items in {result['time_ms']}ms")
```

### Caching

```python
# Embeddings are cached in Redis
# - Image embeddings: keyed by content hash
# - Captions: keyed by image ID
# - OCR results: keyed by image ID

# Cache stats
stats = await mmrag.get_stats()
print(f"Cache hit rate: {stats['cache']['hit_rate']}")
```

---

## API Reference

### Core Operations

| Operation | Description | Returns |
|-----------|-------------|---------|
| `initialize` | Start module, load models | status |
| `shutdown` | Stop module, free resources | status |
| `health_check` | Check all components | health info |

### Image Operations

| Operation | Description | Returns |
|-----------|-------------|---------|
| `load_image` | Load and preprocess image | ImageData |
| `embed_image` | Generate CLIP embedding | embedding |
| `embed_text` | Generate text embedding | embedding |
| `extract_text_from_image` | OCR | extracted text |

### Retrieval Operations

| Operation | Description | Returns |
|-----------|-------------|---------|
| `retrieve` | Main retrieval (any mode) | items |
| `text_to_image_search` | Find images from text | images |
| `image_to_text_search` | Find text from image | text items |
| `find_similar_images` | Visual similarity | images |
| `multimodal_query` | Combined query | items |

### VQA Operations

| Operation | Description | Returns |
|-----------|-------------|---------|
| `answer_question` | VQA | answer |
| `generate_caption` | Image caption | caption |
| `describe_image` | Multi-aspect description | description |

### Document Operations

| Operation | Description | Returns |
|-----------|-------------|---------|
| `process_document` | Full PDF processing | document info |

### Index Operations

| Operation | Description | Returns |
|-----------|-------------|---------|
| `index_item` | Add single item | status |
| `index_batch` | Add multiple items | count |
| `get_index_stats` | Index statistics | stats |
| `clear_index` | Clear all items | status |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8+ GB |
| RAM | 8 GB | 16+ GB |
| Storage | 5 GB | 20 GB |
| CPU | 4 cores | 8+ cores |

**Note:** CPU-only mode is supported but significantly slower for embedding generation.

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size
UBP_MMRAG__EMBED_BATCH_SIZE=8

# Use smaller model
UBP_MMRAG__EMBED_MODEL=openai/clip-vit-base-patch32

# Force CPU
UBP_MMRAG__DEVICE=cpu
```

**2. OCR Not Working**
```bash
# Install Tesseract
apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-ita

# Or use EasyOCR (no system deps)
pip install easyocr
UBP_MMRAG__OCR_ENGINE=easyocr
```

**3. PDF Processing Fails**
```bash
# Install PyMuPDF
pip install PyMuPDF

# Or pdf2image + poppler
apt-get install poppler-utils
pip install pdf2image
```

---

## Changelog

### v1.0.0 (2025-01)
- Initial release
- CLIP, BLIP, SigLIP embedding support
- Cross-modal retrieval (5 modes)
- VQA with BLIP-2
- Image captioning
- OCR integration (Tesseract, EasyOCR)
- PDF processing with image extraction
- Multi-modal chunking
- Redis caching
- Cross-modal reranking

---

## License

Enterprise License - UBP Hybrid System
