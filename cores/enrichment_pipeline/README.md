# enrichment_pipeline

> Advanced RAG enrichment pipeline with reranking, query expansion, HyDE, compression, and chunk fusion.

**Version:** 1.0.0  
**Module Type:** enrichment  
**Architecture:** 3-file-pattern  

---

## Overview

`enrichment_pipeline` is a core module for UBP Enterprise Hybrid that provides advanced RAG enrichment capabilities:

- **Reranking** with BGE cross-encoder models (SOTA multilingual)
- **Query Expansion** via LLM-generated semantic variants
- **HyDE** (Hypothetical Document Embedding) generation
- **Context Compression** (extractive + abstractive)
- **Chunk Fusion** (overlap, semantic, adjacent merging)
- **Deduplication** (hash, fuzzy, semantic)
- **Metadata Injection** for enhanced context

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  enrichment_pipeline                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  adapter.py ──────────► pipeline.py (Orchestrator)                      │
│       │                      │                                          │
│       │                      ├──► providers.py                          │
│       │                      │    ├── RerankerProvider (BGE)            │
│       │                      │    ├── ContextCompressor                 │
│       │                      │    ├── ChunkFusion                       │
│       │                      │    ├── Deduplicator                      │
│       │                      │    └── MetadataInjector                  │
│       │                      │                                          │
│       │                      └──► delegation.py                         │
│       │                           └── LLMDelegator → inference_vllm     │
│       │                                                                 │
│       └──────────────► Individual operation handlers                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Query Expansion (LLM)                                       │
│     Generate semantic variants of the query                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. HyDE Generation (LLM)                                       │
│     Generate hypothetical document for enhanced matching        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Reranking (BGE Cross-Encoder)                               │
│     Score and reorder chunks by relevance                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Chunk Fusion                                                │
│     Merge overlapping/adjacent/similar chunks                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Deduplication                                               │
│     Remove duplicate/near-duplicate chunks                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. Compression                                                 │
│     Reduce token count while preserving information             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. Metadata Injection                                          │
│     Add source, relevance, position, token count                │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
modules/cores/enrichment_pipeline/
├── manifest.json      # API contract (operations, events)
├── config.json        # Configuration with ${ENV_VAR} placeholders
├── __init__.py        # Factory: create_module()
├── adapter.py         # Bridge Layer - all operations
├── providers.py       # Core providers - ZERO UBP dependencies
├── pipeline.py        # Pipeline orchestrator
├── delegation.py      # LLM delegation to inference_vllm
└── README.md          # This file
```

---

## Operations

### Pipeline Operations

| Operation | Description |
|-----------|-------------|
| `enrich_context` | Full pipeline execution (configurable) |

### Individual Operations

| Operation | Description |
|-----------|-------------|
| `rerank` | Rerank chunks with cross-encoder |
| `expand_query` | Generate query variants via LLM |
| `generate_hyde` | Generate hypothetical document |
| `compress_context` | Compress chunks (extractive/abstractive) |
| `fuse_chunks` | Merge similar/overlapping chunks |
| `deduplicate` | Remove duplicate chunks |
| `inject_metadata` | Add enrichment metadata |

### Configuration Operations

| Operation | Description |
|-----------|-------------|
| `get_pipeline_config` | Get current pipeline configuration |
| `set_pipeline_config` | Update pipeline (admin only) |

### Admin Operations

| Operation | Description |
|-----------|-------------|
| `initialize` | Initialize and load models |
| `shutdown` | Unload models and cleanup |
| `health_check` | Check reranker and delegation health |
| `get_stats` | Get enrichment metrics |

---

## Configuration

> **FIX-012 v1.8.2: SOURCE OF TRUTH**
>
> This module uses `config.json` as its runtime configuration source.
> The backend also has `EnrichmentSettings` in `backend/app/core/config.py`
> for Pydantic validation and ENV variable support.
>
> **Priority (highest to lowest):**
> 1. Environment variables (`UBP_ENRICHMENT__*`)
> 2. `config.json` in this directory
> 3. `EnrichmentSettings` defaults in backend
>
> **Important:** Keep both files in sync when modifying defaults.

### Environment Variables

```bash
# Reranker
UBP_ENRICHMENT_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
UBP_ENRICHMENT_RERANKER_DEVICE=cuda
UBP_ENRICHMENT_RERANKER_BATCH_SIZE=32
UBP_ENRICHMENT_RERANKER_MAX_LENGTH=512

# Query Expansion
UBP_ENRICHMENT_EXPANSION_ENABLED=true
UBP_ENRICHMENT_EXPANSION_VARIANTS=3
UBP_ENRICHMENT_LLM_MODULE=inference_vllm

# HyDE
UBP_ENRICHMENT_HYDE_ENABLED=true
UBP_ENRICHMENT_HYDE_MAX_TOKENS=300

# Compression
UBP_ENRICHMENT_COMPRESSION_ENABLED=true
UBP_ENRICHMENT_COMPRESSION_RATIO=0.5
UBP_ENRICHMENT_COMPRESSION_METHOD=extractive

# Fusion
UBP_ENRICHMENT_FUSION_ENABLED=true
UBP_ENRICHMENT_FUSION_OVERLAP=0.3
UBP_ENRICHMENT_FUSION_SEMANTIC=0.85

# Deduplication
UBP_ENRICHMENT_DEDUP_ENABLED=true
UBP_ENRICHMENT_DEDUP_THRESHOLD=0.95
UBP_ENRICHMENT_DEDUP_METHOD=semantic

# Pipeline
UBP_ENRICHMENT_PIPELINE_TIMEOUT=60
UBP_ENRICHMENT_FAIL_FAST=false
```

---

## Reranker Models

| Model | VRAM | Multilingual | Quality | Speed |
|-------|------|--------------|---------|-------|
| **BAAI/bge-reranker-v2-m3** | ~2GB | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| BAAI/bge-reranker-base | ~400MB | ❌ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| cross-encoder/ms-marco | ~400MB | ❌ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Recommended:** BGE-reranker-v2-m3 for multilingual support and SOTA quality.

---

## Usage Examples

### Full Pipeline

```python
result = await module.enrich_context(
    query="What is our refund policy?",
    chunks=retrieved_chunks,
    pipeline_config={
        "steps": [
            {"step": "query_expansion", "enabled": True},
            {"step": "rerank", "enabled": True},
            {"step": "fusion", "enabled": True},
            {"step": "compression", "enabled": True, "config": {"compression_ratio": 0.6}},
            {"step": "metadata", "enabled": True},
        ]
    },
    top_k=5,
)

print(f"Enriched query: {result['enriched_query']}")
print(f"Steps applied: {result['enrichment_applied']}")
print(f"Time: {result['stats']['total_time_ms']}ms")
```

### Rerank Only

```python
result = await module.rerank(
    query="machine learning tutorial",
    chunks=chunks,
    top_k=10,
)

for chunk in result["reranked_chunks"]:
    print(f"[{chunk['rerank_score']:.3f}] {chunk['text'][:100]}...")
```

### Query Expansion

```python
result = await module.expand_query(
    query="how to deploy docker containers",
    num_variants=3,
    expansion_type="semantic",
)

print(f"Original: {result['original_query']}")
print(f"Variants: {result['expanded_queries']}")
print(f"Combined: {result['combined_query']}")
```

### HyDE Generation

```python
result = await module.generate_hyde(
    query="What causes climate change?",
    document_type="technical",
)

print(f"Hypothetical document: {result['hypothetical_document']}")
```

---

## Dependencies

### Required

- `inference_vllm` - For LLM operations (query expansion, HyDE, abstractive compression)

### Python Packages

```
sentence-transformers>=2.2.0
torch>=2.0.0
FlagEmbedding>=1.2.0
```

### Infrastructure

- CUDA-capable GPU (recommended, 2GB+ VRAM for reranker)
- CPU fallback available

---

## Integration with inference_vllm

This module delegates LLM operations to `inference_vllm`:

```
enrichment_pipeline                    inference_vllm
      │                                     │
      │  expand_query() ───────────────────►│ generate()
      │                                     │
      │  generate_hyde() ──────────────────►│ generate()
      │                                     │
      │  compress_abstractive() ───────────►│ generate()
      │                                     │
```

Configure the delegation target:

```bash
UBP_ENRICHMENT_LLM_MODULE=inference_vllm
```

---

## Event Publications

```
enrichment.pipeline.started
enrichment.pipeline.completed
enrichment.pipeline.failed
enrichment.rerank.completed
enrichment.query_expansion.completed
enrichment.hyde.generated
enrichment.compression.completed
enrichment.fusion.completed
enrichment.model.loaded
enrichment.model.unloaded
enrichment.health.degraded
```

---

## Redis Keys (Cache)

```
ubp:enrichment:cache:rerank:{hash}
ubp:enrichment:cache:hyde:{hash}
ubp:enrichment:stats:{metric}
```

---

## Performance Tips

1. **Batch Size**: Increase `UBP_ENRICHMENT_RERANKER_BATCH_SIZE` for throughput
2. **Disable Unused Steps**: Set `enabled: false` for steps you don't need
3. **Cache**: Enable caching for repeated queries
4. **GPU**: Use CUDA for reranking (10x faster than CPU)
5. **Compression First**: If token limits are tight, run compression earlier

---

## Checklist

- [x] Pattern 3-file enterprise
- [x] `providers.py` ZERO import da backend.app
- [x] Naming policy compliant
- [x] Operations with `ctx=None, **kwargs`
- [x] Config with `${ENV_VAR}` placeholders
- [x] `initialize()`, `shutdown()`, `health_check()`
- [x] Event publications declared
- [x] Dependencies on `inference_vllm`

---

**Author:** UBP Team  
**License:** Commercial  
**Last Updated:** January 2025
