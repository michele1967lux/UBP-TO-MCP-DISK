# HyDE Pipeline

**Enterprise Hypothetical Document Embedding Engine** for UBP Enterprise Hybrid

Version: 1.0.0 | Architecture: 3-file-pattern | Module Type: core

---

## Overview

`hyde_pipeline` generates hypothetical documents that would answer a user's query, enabling embedding-based retrieval that outperforms raw query matching. Instead of searching with the original query, HyDE creates rich, contextual documents that better match the semantic space of your knowledge base.

### Why HyDE?

Traditional RAG struggles with:
- Abstract or conceptual queries
- Poorly-formed questions
- Vocabulary mismatch between query and documents

HyDE solves this by generating documents in the same "language" as your knowledge base, dramatically improving retrieval accuracy.

### Key Innovations

- **7 Document Formats**: answer, technical_doc, faq, code_snippet, tutorial, troubleshooting, article
- **7 Domain Adapters**: AI/ML, DevOps, API, Database, Security, Cloud, General
- **Ensemble Generation**: Multiple documents with diversity scoring
- **Iterative Refinement**: Auto-improve low-quality documents
- **Hallucination Detection**: Identify fabricated content
- **Semantic Chunking**: Optimal splitting for embedding
- **Cross-Lingual Support**: EN/IT with auto-detection

---

## Architecture

```
hyde_pipeline/
├── __init__.py          # Module factory for ModuleLoader
├── adapter.py           # Bridge layer - exposes 20+ operations
├── providers.py         # Core logic - zero backend dependencies
├── delegation.py        # LLM delegation with multi-format support
├── pipeline.py          # 8-step configurable orchestrator
├── prompts.py           # Domain-specific prompt templates
├── config.json          # 150+ environment variables
├── manifest.json        # Operation definitions and metadata
└── README.md            # This file
```

### Layer Responsibilities

| Layer | File | Purpose |
|-------|------|---------|
| **Entry Point** | `__init__.py` | Factory function for ModuleLoader |
| **Bridge** | `adapter.py` | Exposes operations, manages lifecycle, DI resolution |
| **Logic** | `providers.py` | Core algorithms (QA, chunking, fusion, workers) |
| **Delegation** | `delegation.py` | LLM communication with format selection |
| **Orchestration** | `pipeline.py` | Multi-step pipeline with configuration |
| **Templates** | `prompts.py` | Domain and format-specific prompts |

---

## Document Formats

| Format | Description | Temperature | Use Case |
|--------|-------------|-------------|----------|
| `answer` | Direct knowledge base response | 0.5 | General queries |
| `technical_doc` | Documentation with sections | 0.3 | Technical details |
| `faq` | Q&A format entry | 0.4 | Common questions |
| `code_snippet` | Code-focused with examples | 0.2 | Implementation |
| `tutorial` | Step-by-step guide | 0.4 | How-to queries |
| `troubleshooting` | Problem-solution format | 0.3 | Debugging |
| `article` | Narrative article excerpt | 0.6 | Conceptual topics |

---

## Domain Adaptation

| Domain | Description | Preferred Formats |
|--------|-------------|-------------------|
| `ai_ml` | ML, Deep Learning, LLMs, RAG | technical_doc, code_snippet, answer |
| `devops` | Docker, K8s, CI/CD, IaC | tutorial, troubleshooting, code_snippet |
| `api_integration` | REST, GraphQL, Auth, Webhooks | code_snippet, technical_doc, faq |
| `database` | SQL, NoSQL, Query Optimization | code_snippet, technical_doc, troubleshooting |
| `security` | Auth, Encryption, Compliance | technical_doc, troubleshooting, faq |
| `cloud` | AWS, Azure, GCP, Serverless | tutorial, technical_doc, troubleshooting |
| `general` | General technical queries | answer, faq, article |

Domains are auto-detected based on query keywords with confidence scoring.

---

## Pipeline Steps

The full HyDE pipeline executes 8 configurable steps:

```
1. classify_query      → Detect domain and language
2. select_format       → Choose optimal document format
3. generate_document   → Create HyDE document via LLM
4. quality_assurance   → Multi-dimensional quality scoring
5. hallucination_check → Detect fabricated content
6. refinement          → Iterative improvement (if needed)
7. chunking            → Semantic splitting for embedding
8. format_output       → Prepare final result
```

Each step can be enabled/disabled:

```json
{
  "pipeline": {
    "steps": {
      "classify_query": { "enabled": true, "timeout": 5 },
      "hallucination_check": { "enabled": true, "timeout": 5 },
      "refinement": { "enabled": false }
    }
  }
}
```

---

## Operations

### Core Operations

| Operation | Description | Admin |
|-----------|-------------|-------|
| `generate_hyde` | Full pipeline execution | No |
| `generate_document` | Direct generation (no pipeline) | No |
| `generate_ensemble` | Multi-document with diversity | No |
| `classify_query` | Domain and language detection | No |
| `assess_quality` | Multi-dimensional scoring | No |
| `check_hallucination` | Detect fabricated content | No |
| `refine_document` | Iterative improvement | No |
| `chunk_document` | Semantic chunking | No |
| `fuse_documents` | Combine multiple documents | No |

### Session Operations

| Operation | Description | Admin |
|-----------|-------------|-------|
| `get_session` | Retrieve session state | No |
| `delete_session` | Delete session | No |

### Admin Operations

| Operation | Description | Admin |
|-----------|-------------|-------|
| `get_stats` | Metrics and statistics | Yes |
| `set_pipeline_config` | Update pipeline steps | Yes |
| `reload_config` | Hot-reload from config.json | Yes |

### Lifecycle Operations

| Operation | Description |
|-----------|-------------|
| `initialize` | Initialize components and worker pool |
| `shutdown` | Graceful shutdown |
| `health_check` | Component health status |

---

## Usage Examples

### Basic HyDE Generation

```python
# Full pipeline with auto-detection
result = await hyde.generate_hyde(
    query="How do I implement vector similarity search?",
)

# Returns:
# {
#   "session_id": "uuid",
#   "document": {...},
#   "chunks": [...],
#   "classification": {"domain": "ai_ml", "confidence": 0.85},
#   "quality_assessment": {"overall_score": 7.8},
#   "time_ms": 1234.56
# }
```

### Format Override

```python
# Force code-focused output
result = await hyde.generate_hyde(
    query="How to connect to PostgreSQL in Python?",
    format_type="code_snippet",
    domain="database",
)
```

### Ensemble Generation

```python
# Generate 3 diverse documents and fuse
result = await hyde.generate_ensemble(
    query="Explain transformer architecture",
    count=3,
    formats=["technical_doc", "answer", "article"],
    fuse_results=True,
)

# Returns multiple documents plus fused result
# diversity_score indicates how different the documents are
```

### Direct Generation (No Pipeline)

```python
# Skip pipeline for speed
result = await hyde.generate_document(
    query="What is a Docker container?",
    format_type="faq",
    max_length=300,
)
```

### Quality Assessment

```python
# Score an existing document
assessment = await hyde.assess_quality(
    document=document_dict,
    original_query="How does RAG work?",
)

# Returns:
# {
#   "overall_score": 7.2,
#   "relevance_score": 8.1,
#   "coherence_score": 6.8,
#   "quality_level": "good",
#   "suggestions": ["Add more technical terms"]
# }
```

### Document Refinement

```python
# Improve a low-quality document
refined = await hyde.refine_document(
    document=document_dict,
    strategy="technical",  # expand, focus, technical, simplify
    quality_score=5.2,
    issues=["Low terminology score"],
)
```

---

## Quality Assurance

### Scoring Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Relevance | 35% | Keyword overlap with query |
| Coherence | 25% | Sentence structure and flow |
| Informativeness | 20% | Information density |
| Format Adherence | 10% | Matches expected format |
| Terminology | 10% | Domain-appropriate terms |

### Quality Levels

| Level | Score | Description |
|-------|-------|-------------|
| EXCELLENT | ≥ 8.0 | Ready for retrieval |
| GOOD | ≥ 6.0 | Acceptable quality |
| ACCEPTABLE | ≥ 4.0 | Minimum threshold |
| POOR | < 4.0 | Triggers refinement |

---

## Hallucination Detection

HyDE documents are generated by LLMs which may hallucinate. The detector checks for:

| Check | Description |
|-------|-------------|
| Invented APIs | Suspicious endpoint patterns |
| Fake Versions | Unusually high version numbers |
| Unknown Terms | Potentially made-up CamelCase terms |

Results include:
- `hallucination_detected`: Boolean flag
- `confidence`: Document confidence (0-1)
- `suspicious_elements`: List of flagged content
- `recommendation`: accept / review / reject

---

## Chunking Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `semantic` | Split on paragraphs/sections | Default, most contexts |
| `sentence` | Split on sentence boundaries | Dense content |
| `paragraph` | One chunk per paragraph | Structured docs |
| `fixed` | Fixed size with overlap | Consistent chunks |

```python
chunks = await hyde.chunk_document(
    document=doc,
    strategy="semantic",
    chunk_size=256,
)
```

---

## Ensemble & Fusion

### Fusion Strategies

| Strategy | Description |
|----------|-------------|
| `weighted_concat` | Concatenate weighted by quality (default) |
| `best_selection` | Select highest quality document |
| `merge_unique` | Merge unique sentences from each |

### Diversity Scoring

Ensemble generation includes a `diversity_score` (0-1):
- 1.0 = Completely different documents
- 0.0 = Identical documents

Higher diversity improves retrieval coverage.

---

## Configuration

### Environment Variables

All settings can be overridden with prefix `UBP_HYDE__`:

```bash
# Core settings
UBP_HYDE__ENABLED=true
UBP_HYDE__DEFAULT_FORMAT=answer
UBP_HYDE__DEFAULT_DOMAIN=auto
UBP_HYDE__DEFAULT_LENGTH=300
UBP_HYDE__TEMPERATURE=0.5
UBP_HYDE__MAX_TOKENS=600
UBP_HYDE__TIMEOUT_SECONDS=30

# Quality Assurance
UBP_HYDE__QA_ENABLED=true
UBP_HYDE__QA_WEIGHT_RELEVANCE=0.35
UBP_HYDE__QA_THRESHOLD_EXCELLENT=8.0

# Ensemble
UBP_HYDE__ENSEMBLE_ENABLED=true
UBP_HYDE__ENSEMBLE_COUNT=3
UBP_HYDE__ENSEMBLE_FUSION=weighted_concat
UBP_HYDE__ENSEMBLE_DIVERSITY_PENALTY=0.1

# Refinement
UBP_HYDE__REFINEMENT_ENABLED=true
UBP_HYDE__REFINEMENT_MAX_ITER=2
UBP_HYDE__REFINEMENT_QUALITY_THRESHOLD=6.0

# Hallucination Detection
UBP_HYDE__HALLUCINATION_DETECTION_ENABLED=true
UBP_HYDE__HALLUCINATION_CHECK_APIS=true

# Chunking
UBP_HYDE__CHUNKING_ENABLED=true
UBP_HYDE__CHUNKING_STRATEGY=semantic
UBP_HYDE__CHUNK_SIZE=256
UBP_HYDE__CHUNK_OVERLAP=50

# Cache
UBP_HYDE__CACHE_ENABLED=true
UBP_HYDE__CACHE_TTL=3600
UBP_HYDE__CACHE_SEMANTIC_MATCHING=true

# Session
UBP_HYDE__SESSION_ENABLED=true
UBP_HYDE__SESSION_TTL=3600

# Worker Pool
UBP_HYDE__WORKERS_ENABLED=true
UBP_HYDE__WORKER_POOL_SIZE=4

# Debug
UBP_HYDE__DEBUG_ENABLED=false
UBP_HYDE__DEBUG_LOG_PROMPTS=false
UBP_HYDE__DEBUG_LOG_QA_SCORES=true

# LLM Delegation
UBP_HYDE__LLM_MODULE=inference_ollama_grok
UBP_HYDE__FALLBACK_ENABLED=true
```

---

## Redis Keys

Environment-isolated key patterns:

```
ubp:{env}:hyde:cache:hyde:{hash}
ubp:{env}:hyde:cache:qa:{hash}
ubp:{env}:hyde:cache:chunks:{hash}
ubp:{env}:hyde:session:{session_id}
ubp:{env}:hyde:stats:*
```

Where `{env}` is `dev`, `test`, or `prod` from `UBP_ENV`.

---

## Events Published

| Event | Description |
|-------|-------------|
| `hyde.initialized` | Module initialized |
| `hyde.pipeline.started` | Pipeline execution began |
| `hyde.pipeline.completed` | Pipeline finished |
| `hyde.pipeline.failed` | Pipeline failed |
| `hyde.generation.completed` | Document generated |
| `hyde.ensemble.completed` | Ensemble generated |
| `hyde.refinement.completed` | Refinement done |
| `hyde.hallucination.detected` | Potential hallucination |
| `hyde.session.created` | New session |
| `hyde.session.updated` | Session updated |
| `hyde.health.degraded` | Component degraded |
| `hyde.shutdown` | Module shutdown |

---

## Metrics

Available via `get_stats()` (admin only):

```python
stats = await hyde.get_stats(period="24h")
# {
#   "metrics": {
#     "total_generations": 1234,
#     "format_distribution": {"answer": 500, "technical_doc": 300, ...},
#     "domain_distribution": {"ai_ml": 400, ...},
#     "qa_scores": {"avg": 7.2, "min": 4.1, "max": 9.8},
#     "execution_times": {"avg_ms": 1200, "min_ms": 800, "max_ms": 3200},
#     "hallucination_rate": 0.05,
#     "refinement_rate": 0.15,
#     "ensemble_rate": 0.20
#   },
#   "cache": {"hits": 890, "misses": 344, "hit_rate": 0.72},
#   "worker_pool": {"active_workers": 4, "pending_tasks": 2}
# }
```

---

## Deployment

### 1. Copy Module

```bash
cp -r hyde_pipeline/ modules/cores/hyde_pipeline/
```

### 2. Configure Environment

```bash
# .env or environment
UBP_ENV=prod
UBP_HYDE__ENABLED=true
UBP_HYDE__LLM_MODULE=inference_vllm
```

### 3. Initialize via ModuleLoader

```python
from modules.cores.hyde_pipeline import create_module

module = create_module(
    module_path=Path("modules/cores/hyde_pipeline"),
    di_container=container,
    event_bus=event_bus,
)

await module.initialize()
```

### 4. Health Check

```python
health = await module.health_check()
# {
#   "module": "hyde_pipeline",
#   "status": "healthy",
#   "llm_delegation": {"status": "available"},
#   "worker_pool": {"active_workers": 4},
#   "cache": {"hit_rate": 0.85}
# }
```

---

## Integration with RAG

HyDE is designed to enhance RAG retrieval:

```python
# 1. Generate HyDE document
hyde_result = await hyde.generate_hyde(query=user_query)

# 2. Use HyDE document for embedding
hyde_embedding = await embedder.embed(hyde_result["document"]["content"])

# 3. Also embed chunks for multi-vector search
chunk_embeddings = [
    await embedder.embed(chunk["content"])
    for chunk in hyde_result["chunks"]
]

# 4. Search with HyDE embedding (better than raw query)
results = await vector_db.search(hyde_embedding, top_k=10)
```

---

## Troubleshooting

### Low Quality Scores

```
Adjust:
- UBP_HYDE__TEMPERATURE (lower = more focused)
- UBP_HYDE__DEFAULT_LENGTH (longer = more context)
- UBP_HYDE__REFINEMENT_ENABLED=true
- UBP_HYDE__REFINEMENT_QUALITY_THRESHOLD (lower threshold)
```

### Hallucinations Detected

```
Options:
1. Enable stricter detection: UBP_HYDE__HALLUCINATION_CHECK_APIS=true
2. Use lower temperature: UBP_HYDE__TEMPERATURE=0.3
3. Try different format: format_type="faq" (more constrained)
```

### High Latency

```
Optimize:
- UBP_HYDE__CACHE_ENABLED=true
- UBP_HYDE__WORKER_POOL_SIZE (increase workers)
- Disable unnecessary steps in pipeline_config
- Use generate_document instead of generate_hyde for simple cases
```

### LLM Not Available

```
Check:
1. UBP_HYDE__LLM_MODULE points to valid module
2. Inference module is initialized
3. ProviderMapper has "hyde" role configured
```

---

## Dependencies

### Required
- Python 3.10+
- Redis (for caching and sessions)

### Optional
- `inference_vllm` - vLLM inference
- `inference_ollama_grok` - Ollama inference
- `inference_openai_anthropic` - Cloud APIs

### Python Packages
- Standard library only for core logic

---

## License

Internal use - UBP Enterprise Hybrid

---

## Changelog

### v1.0.0 (2025-01)
- Initial release
- 7 document formats
- 7 domain adapters
- Ensemble generation with fusion
- Iterative refinement
- Hallucination detection
- Semantic chunking
- Quality assurance scoring
- Cross-lingual EN/IT
- Redis caching
- Session management
- Worker pool parallelism
- Full observability
