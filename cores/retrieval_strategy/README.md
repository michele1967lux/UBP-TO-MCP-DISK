# Retrieval Strategy

**Advanced Retrieval Strategies Engine** for UBP Enterprise Hybrid

Version: 1.0.0 | Architecture: 3-file-pattern | Module Type: core

---

## Overview

`retrieval_strategy` implements multiple advanced retrieval strategies for production RAG systems:

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Hybrid** | BM25 + Vector with fusion | Most queries (default) |
| **Hierarchical** | Multi-level document retrieval | Complex analytical queries |
| **Multi-Index** | Search across specialized indexes | Comparative queries |
| **Router** | LLM-based dynamic selection | Unknown query types |
| **BM25** | Pure keyword retrieval | Technical terms, exact matches |

---

## Architecture

```
retrieval_strategy/
├── __init__.py          # Module factory for ModuleLoader
├── adapter.py           # Bridge layer - exposes 15+ operations
├── providers.py         # Core: BM25Index, HierarchicalIndex, data classes
├── strategies.py        # Strategy implementations
├── fusion.py            # Score fusion algorithms (RRF, weighted, etc.)
├── router.py            # LLM-based query routing
├── config.json          # 200+ environment variables
├── manifest.json        # Operation definitions
└── README.md            # This file
```

---

## 1. Hybrid Retrieval (BM25 + Vector)

The de-facto standard for robust RAG, combining:
- **BM25**: Keyword/lexical matching (exact terms, technical jargon)
- **Vector**: Semantic matching (synonyms, paraphrases, concepts)

### Fusion Methods

```
┌─────────────┐     ┌─────────────┐
│   Query     │     │   Query     │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│    BM25     │     │   Vector    │
│  Retriever  │     │  Retriever  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └────────┬──────────┘
                │
         ┌──────▼──────┐
         │   Fusion    │
         │  (RRF/etc)  │
         └──────┬──────┘
                │
                ▼
         ┌─────────────┐
         │   Results   │
         └─────────────┘
```

### Fusion Algorithms

| Method | Formula | Best For |
|--------|---------|----------|
| **RRF** | `1/(k+rank)` | Balanced combination |
| **Weighted** | `α·BM25 + (1-α)·Vector` | Tunable blending |
| **Max** | `max(score_bm25, score_vector)` | High recall |
| **Sum** | `norm(bm25) + norm(vector)` | Simple aggregation |
| **DBSF** | Distribution-based normalization | Different score scales |

### Usage

```python
# Hybrid retrieval with RRF fusion
result = await retrieval.hybrid_retrieve(
    query="How does transformer attention work?",
    top_k=10,
    fusion_method="rrf",
    alpha=0.5,  # BM25 weight for weighted fusion
)

# Results include both keyword and semantic matches
for r in result["results"]:
    print(f"{r['score']:.3f}: {r['content'][:100]}...")
```

---

## 2. Hierarchical Retrieval

Multi-level retrieval that reduces noise and improves grounding:

```
┌────────────────────────────────────────┐
│           Document Level               │  ← Coarse context (4000 tokens)
│  ┌──────────────────────────────────┐  │
│  │        Section Level             │  │  ← Medium context (1000 tokens)
│  │  ┌────────────────────────────┐  │  │
│  │  │    Paragraph Level         │  │  │  ← Fine-grained (300 tokens)
│  │  │  ┌──────────────────────┐  │  │  │
│  │  │  │   Answer Spans       │  │  │  │
│  │  │  └──────────────────────┘  │  │  │
│  │  └────────────────────────────┘  │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

### Features

- **Parent-Child Linking**: Navigate chunk hierarchy
- **Context Expansion**: Include surrounding paragraphs
- **Weighted Combination**: Configurable level weights
- **Overlap Handling**: Merge adjacent chunks

### Configuration

| Level | Chunk Size | Overlap | Weight | Top K |
|-------|------------|---------|--------|-------|
| Document | 4000 | 200 | 0.2 | 5 |
| Section | 1000 | 100 | 0.3 | 10 |
| Paragraph | 300 | 50 | 0.5 | 20 |

### Usage

```python
# Add documents with automatic hierarchical chunking
await retrieval.add_hierarchical_documents(
    documents=[
        {"id": "doc1", "content": long_document, "metadata": {...}},
    ]
)

# Hierarchical retrieval
result = await retrieval.hierarchical_retrieve(
    query="What are the key findings?",
    top_k=10,
    levels=["section", "paragraph"],  # Skip document level
    expand_context=True,  # Include surrounding paragraphs
)
```

---

## 3. Router-based Retrieval

LLM-powered query analysis to select the optimal strategy:

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│     Query Router        │
│   (LLM Classification)  │
│                         │
│  - Query type?          │
│  - Which index?         │
│  - Skip retrieval?      │
└──────┬──────────────────┘
       │
       ├──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐
│   Hybrid    │ │Hierarchical │ │ Multi-Index │ │  Skip   │
│  Strategy   │ │  Strategy   │ │  Strategy   │ │(no RAG) │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────┘
```

### Query Classification

| Class | Description | Default Strategy |
|-------|-------------|------------------|
| `factual` | Simple factual questions | Hybrid |
| `analytical` | Complex analysis | Hierarchical |
| `comparative` | Comparison questions | Multi-Index |
| `keyword` | Technical terms | BM25 |
| `semantic` | Conceptual queries | Vector |

### Skip Retrieval Detection

Router can skip retrieval for:
- Greetings: "Hi", "Hello", "Ciao"
- Clarifications: "What do you mean?"
- Follow-ups: "Tell me more"
- Acknowledgments: "Thanks", "OK"

### Usage

```python
# Let router decide the best strategy
result = await retrieval.router_retrieve(
    query="Compare the performance of GPT-4 vs Claude",
    top_k=10,
    language="en",
)

# Check what router decided
decision = result["metadata"]["routing_decision"]
print(f"Query class: {decision['query_class']}")
print(f"Strategy: {decision['selected_strategy']}")
print(f"Confidence: {decision['confidence']}")
```

---

## 4. Multi-Index Retrieval

Search across multiple specialized indexes:

```python
# Create specialized indexes
await retrieval.create_index("technical", "bm25")
await retrieval.create_index("legal", "bm25")
await retrieval.create_index("financial", "bm25")

# Add documents to specific indexes
await retrieval.add_documents(
    documents=technical_docs,
    index_name="technical",
)

# Search across indexes
result = await retrieval.multi_index_retrieve(
    query="API rate limiting regulations",
    top_k=10,
    index_names=["technical", "legal"],  # Or None for all
)
```

### Index Weighting

```python
# Configure index weights in config.json
"multi_index": {
    "index_weights": {
        "default": 1.0,
        "technical": 1.2,  # Boost technical results
        "legal": 1.0,
        "financial": 1.0
    }
}
```

---

## BM25 Implementation

Pure Python BM25 with no external dependencies:

### Algorithm

```
score(D,Q) = Σ IDF(qi) · (f(qi,D) · (k1 + 1)) / (f(qi,D) + k1 · (1 - b + b · |D|/avgdl))
```

Where:
- `IDF(qi)` = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
- `f(qi,D)` = term frequency in document
- `|D|` = document length
- `avgdl` = average document length
- `k1` = 1.5 (term saturation)
- `b` = 0.75 (length normalization)

### Features

- **Stopword removal** (EN/IT)
- **Configurable tokenization**
- **In-memory indexing**
- **Fast lookups**

---

## Operations Reference

### Retrieval Operations

| Operation | Description |
|-----------|-------------|
| `retrieve` | Execute retrieval with any strategy |
| `hybrid_retrieve` | Force hybrid strategy |
| `hierarchical_retrieve` | Force hierarchical strategy |
| `multi_index_retrieve` | Search multiple indexes |
| `router_retrieve` | LLM-routed retrieval |
| `bm25_search` | BM25-only search |

### Index Operations

| Operation | Description |
|-----------|-------------|
| `add_documents` | Add to BM25 index |
| `add_hierarchical_documents` | Add with chunking |
| `create_index` | Create named index |
| `get_indexes` | List indexes |
| `clear_index` | Clear an index |

### Utility Operations

| Operation | Description |
|-----------|-------------|
| `classify_query` | Classify without retrieval |
| `get_available_strategies` | List strategies |
| `get_stats` | Get metrics |
| `health_check` | Component health |

---

## Configuration

### Key Environment Variables

```bash
# Core
UBP_RETRIEVAL__DEFAULT_STRATEGY=hybrid
UBP_RETRIEVAL__MAX_RESULTS=20

# Hybrid
UBP_RETRIEVAL__HYBRID_FUSION=rrf
UBP_RETRIEVAL__HYBRID_ALPHA=0.5
UBP_RETRIEVAL__HYBRID_BM25_WEIGHT=0.4
UBP_RETRIEVAL__HYBRID_VECTOR_WEIGHT=0.6

# BM25
UBP_RETRIEVAL__BM25_K1=1.5
UBP_RETRIEVAL__BM25_B=0.75
UBP_RETRIEVAL__BM25_STOPWORDS=true

# Hierarchical
UBP_RETRIEVAL__HIER_DOC_CHUNK_SIZE=4000
UBP_RETRIEVAL__HIER_SEC_CHUNK_SIZE=1000
UBP_RETRIEVAL__HIER_PARA_CHUNK_SIZE=300

# Router
UBP_RETRIEVAL__ROUTER_LLM_MODULE=inference_ollama_grok
UBP_RETRIEVAL__ROUTER_FALLBACK=hybrid

# RRF Fusion
UBP_RETRIEVAL__RRF_K=60

# Cache
UBP_RETRIEVAL__CACHE_ENABLED=true
UBP_RETRIEVAL__CACHE_TTL=1800
```

---

## Integration Examples

### With retrieval_pipeline

```python
# retrieval_pipeline delegates to retrieval_strategy
result = await retrieval_pipeline.retrieve(
    query=user_query,
    strategy="hybrid",  # Uses retrieval_strategy
)
```

### With reasoning_rag

```python
# Get context with retrieval_strategy
context = await retrieval.hybrid_retrieve(
    query=user_query,
    top_k=10,
)

# Pass to reasoning
result = await reasoning.chain_of_thought(
    query=user_query,
    context=[r["content"] for r in context["results"]],
)
```

### With graph_rag

```python
# Hybrid: vector + graph
vector_results = await retrieval.hybrid_retrieve(query)
graph_results = await graph.get_subgraph(entity_ids)

# Combine for answer
combined_context = vector_results + graph_results
```

---

## Deployment

### 1. Copy Module

```bash
cp -r retrieval_strategy/ modules/cores/retrieval_strategy/
```

### 2. Configure

```bash
# .env
UBP_RETRIEVAL__HYBRID_FUSION=rrf
UBP_RETRIEVAL__ROUTER_ENABLED=true
```

### 3. Initialize

```python
from modules.cores.retrieval_strategy import create_module

module = create_module(
    module_path=Path("modules/cores/retrieval_strategy"),
    di_container=container,
    event_bus=event_bus,
)

await module.initialize()
```

---

## Performance Tips

### 1. Index Size
- BM25 is memory-bound (~1GB for 1M documents)
- Use hierarchical for large documents

### 2. Fusion Method
- RRF: Best balance, position-based
- Weighted: When you know optimal α
- DBSF: When score distributions differ greatly

### 3. Router
- Cache router decisions (default: enabled)
- Use fallback for reliability
- Monitor router metrics

### 4. Caching
- Enable semantic caching for similar queries
- TTL based on data freshness requirements

---

## Metrics

```python
stats = await retrieval.get_stats()

# Example output:
{
    "metrics": {
        "total_queries": 1500,
        "strategy_distribution": {
            "hybrid": 1200,
            "hierarchical": 200,
            "multi_index": 100
        },
        "latency_stats": {
            "avg_ms": 45.2,
            "p50_ms": 38.0,
            "p99_ms": 120.5
        },
        "cache_hit_rate": 0.35
    }
}
```

---

## Troubleshooting

### Low Recall
- Increase `top_k`
- Try different fusion method
- Check BM25 stopwords settings

### High Latency
- Enable caching
- Reduce `top_k` for fusion input
- Use BM25-only for simple queries

### Poor Routing
- Check LLM module availability
- Lower router temperature
- Review fallback strategy

### Memory Issues
- Clear unused indexes
- Reduce hierarchical chunk overlap
- Use Redis for caching

---

## Dependencies

### Required
- Python 3.10+

### Optional
- Redis (for distributed caching)
- embedding_service (for vector search)
- qdrant_store (for vector storage)
- inference_ollama_grok (for router LLM)

---

## Changelog

### v1.0.0 (2025-01)
- Initial release
- Hybrid retrieval (BM25 + Vector)
- Hierarchical multi-level retrieval
- Multi-index support
- LLM-based routing
- 5 fusion algorithms
- Cross-lingual support (EN/IT)
- Comprehensive caching
- Query classification
- Skip retrieval detection
