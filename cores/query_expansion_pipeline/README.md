# Query Expansion Pipeline

**Advanced Query Expansion for RAG** - Multi-strategy expansion, intent detection, entity extraction

Version: 1.0.0 | Architecture: 3-file-pattern | Module Type: enrichment

---

## Overview

`query_expansion_pipeline` provides sophisticated query expansion capabilities for RAG systems, improving retrieval quality by generating multiple query variants.

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Strategy** | 7 expansion strategies |
| **Intent Detection** | Classify query intent |
| **Entity Extraction** | Extract named entities |
| **Query Decomposition** | Break complex queries |
| **Synonym Expansion** | Dictionary-based synonyms |
| **Contextual** | Chat history aware |
| **Quality Scoring** | Filter low-quality expansions |
| **Hybrid Mode** | Combine multiple strategies |

---

## Strategies

### 1. Semantic (`semantic`)
LLM-based semantic variation generation.

```python
result = await qexp.expand_semantic(
    query="How to implement RAG?",
    num_variants=3,
)
# Returns: ["RAG implementation guide", "Building RAG systems", ...]
```

### 2. Synonym (`synonym`)
Dictionary-based synonym expansion.

```python
result = await qexp.expand_synonyms(
    query="create a fast API",
    max_synonyms_per_term=3,
)
# Returns: ["build a quick API", "make a rapid API", ...]
```

### 3. Decompose (`decompose`)
Break complex queries into sub-queries.

```python
result = await qexp.decompose_query(
    query="What is RAG and how does it improve LLM accuracy?",
    max_subqueries=3,
)
# Returns: ["What is RAG?", "How does RAG work?", "How does RAG improve accuracy?"]
```

### 4. Reformulate (`reformulate`)
Rephrase as different question types.

```python
result = await qexp.reformulate_query(
    query="machine learning basics",
)
# Returns: ["What is machine learning?", "How does ML work?", "ML examples", ...]
```

### 5. Keywords (`keywords`)
Extract and expand key terms.

```python
result = await qexp.extract_keywords(
    query="How to deploy FastAPI on Kubernetes?",
    max_keywords=5,
)
# Returns: ["FastAPI Kubernetes deployment", "deploy FastAPI", ...]
```

### 6. Contextual (`contextual`)
Use conversation history for expansion.

```python
result = await qexp.expand_contextual(
    query="How do I fix it?",
    chat_history=[
        {"role": "user", "content": "My Python code has a bug"},
        {"role": "assistant", "content": "What error are you seeing?"},
    ],
)
# Returns: ["How do I fix Python bug?", ...]
```

### 7. Hybrid (`hybrid`)
Combine multiple strategies.

```python
result = await qexp.expand_hybrid(
    query="machine learning",
    strategies=["semantic", "synonym", "keywords"],
    weights={"semantic": 0.5, "synonym": 0.3, "keywords": 0.2},
)
```

---

## Architecture

```
query_expansion_pipeline/
├── __init__.py        # Module factory
├── adapter.py         # Bridge layer - all operations
├── providers.py       # Core data classes and utilities
├── strategies.py      # Expansion strategies
├── delegation.py      # LLM delegation
├── prompts.py         # Prompt templates (EN/IT)
├── config.json        # Configuration
├── manifest.json      # Operation definitions
└── README.md          # This file
```

---

## Main Operation

### `expand` - Full Pipeline

```python
result = await qexp.expand(
    query="How to build a chatbot?",
    strategy="semantic",           # Strategy to use
    num_expansions=5,              # Max expansions
    chat_history=None,             # For contextual
    include_original=True,         # Include original in combined
    detect_intent=True,            # Detect query intent
    extract_entities=True,         # Extract named entities
    filter_quality=True,           # Quality filter
)
```

**Response:**

```json
{
  "original_query": "How to build a chatbot?",
  "expanded_queries": [
    {"text": "Chatbot development guide", "strategy": "semantic", "score": 0.85},
    {"text": "Building conversational AI", "strategy": "semantic", "score": 0.82},
    {"text": "Create a chatbot tutorial", "strategy": "semantic", "score": 0.78}
  ],
  "combined_query": "How to build a chatbot? | Chatbot development guide | Building conversational AI",
  "strategy_used": "semantic",
  "intent": {"intent": "procedural", "confidence": 0.85},
  "entities": [{"text": "chatbot", "type": "TECH"}],
  "language": "en",
  "time_ms": 245.3,
  "expansion_count": 3
}
```

---

## Analysis Operations

### Intent Detection

```python
result = await qexp.detect_intent(query="What is machine learning?")
# {"intent": "definition", "confidence": 0.9, "signals": ["matched: ^what is"]}
```

**Intent Types:**
- `informational` - Seeking information
- `definition` - Asking what something is
- `procedural` - How to do something
- `comparison` - Comparing things
- `factual` - Specific facts
- `opinion` - Seeking recommendations
- `navigational` - Finding a specific resource

### Entity Extraction

```python
result = await qexp.extract_entities(query="How to use PyTorch with NVIDIA GPUs?")
# {
#   "entities": [
#     {"text": "PyTorch", "type": "TECH"},
#     {"text": "NVIDIA", "type": "ORG"}
#   ]
# }
```

**Entity Types:**
- `PERSON`, `ORG`, `PRODUCT`, `TECH`, `LOCATION`, `DATE`

### Query Normalization

```python
result = await qexp.normalize_query(
    query="  What's   AI ???  ",
    options={"lowercase": True, "remove_punctuation": True},
)
# {"original": "  What's   AI ???  ", "normalized": "what's ai", "changed": True}
```

---

## Configuration

### Key Environment Variables

```bash
# Default strategy
UBP_QEXP__DEFAULT_STRATEGY=semantic
UBP_QEXP__MAX_EXPANSIONS=10

# Semantic strategy
UBP_QEXP__SEMANTIC_ENABLED=true
UBP_QEXP__SEMANTIC_VARIANTS=3
UBP_QEXP__SEMANTIC_TEMPERATURE=0.7

# Synonym strategy
UBP_QEXP__SYNONYM_ENABLED=true
UBP_QEXP__SYNONYM_MAX=3

# Decomposition
UBP_QEXP__DECOMPOSE_ENABLED=true
UBP_QEXP__DECOMPOSE_MAX=5

# Intent detection
UBP_QEXP__INTENT_ENABLED=true
UBP_QEXP__INTENT_ADAPT=true

# Entity extraction
UBP_QEXP__ENTITY_ENABLED=true

# Quality filtering
UBP_QEXP__QUALITY_SCORING=true
UBP_QEXP__QUALITY_THRESHOLD=0.3
UBP_QEXP__QUALITY_SIM_THRESHOLD=0.9

# LLM
UBP_QEXP__LLM_MODULE=inference_ollama_grok
UBP_QEXP__LLM_TIMEOUT=30

# Cache
UBP_QEXP__CACHE_ENABLED=true
UBP_QEXP__CACHE_TTL=3600
```

---

## Integration

### With Retrieval

```python
# 1. Expand query
expansion = await qexp.expand(query=user_query, strategy="hybrid")

# 2. Use combined query for retrieval
results = await retriever.search(
    query=expansion["combined_query"],
    top_k=10,
)

# 3. Or use individual expansions
for exp in expansion["expanded_queries"]:
    partial = await retriever.search(query=exp["text"], top_k=5)
    # Merge results...
```

### With enrichment_pipeline

```python
# enrichment_pipeline can delegate to query_expansion_pipeline
result = await enrichment.enrich_context(
    query=query,
    chunks=chunks,
    pipeline_config={
        "steps": [
            {"step": "query_expansion", "enabled": True, "config": {
                "strategy": "semantic",
                "num_variants": 3,
            }},
            {"step": "rerank", "enabled": True},
        ]
    },
)
```

---

## Quality Scoring

Expansions are scored on:

1. **Length** (0.2 weight) - Appropriate length
2. **Relevance** (0.4 weight) - Word overlap with original
3. **Diversity** (0.4 weight) - Different from other expansions

```python
# Filter only high-quality expansions
result = await qexp.expand(
    query="...",
    filter_quality=True,  # Enable quality filtering
)
```

Configure threshold:
```bash
UBP_QEXP__QUALITY_THRESHOLD=0.3  # Minimum score (0-1)
```

---

## Caching

Results are cached in Redis to avoid redundant LLM calls.

```python
# First call - generates expansions
result1 = await qexp.expand(query="machine learning")  # ~250ms

# Second call - from cache
result2 = await qexp.expand(query="machine learning")  # ~1ms
result2["from_cache"]  # True
```

Configure:
```bash
UBP_QEXP__CACHE_ENABLED=true
UBP_QEXP__CACHE_TTL=3600  # 1 hour
```

---

## Language Support

Automatic language detection with prompts in:
- **English** (en) - Default
- **Italian** (it)

```python
# Auto-detect language
result = await qexp.expand(query="Come funziona il machine learning?")
result["language"]  # "it"
```

---

## Hybrid Strategy Details

Combine strategies with configurable weights:

```python
result = await qexp.expand_hybrid(
    query="machine learning tutorial",
    strategies=["semantic", "synonym", "keywords"],
    weights={
        "semantic": 0.5,    # 50% weight
        "synonym": 0.3,     # 30% weight  
        "keywords": 0.2,    # 20% weight
    },
)
```

**Combination Methods:**
- `weighted` (default) - Score-based ranking
- `voting` - Frequency-based (expansions appearing in multiple strategies rank higher)

---

## Example: Complete RAG Pipeline

```python
# 1. Initialize
qexp = await query_expansion_pipeline.initialize()
retriever = await retrieval_strategy.initialize()
enrichment = await enrichment_pipeline.initialize()

# 2. User query
user_query = "How do I deploy a FastAPI app?"

# 3. Expand query
expansion = await qexp.expand(
    query=user_query,
    strategy="hybrid",
    num_expansions=5,
)

# 4. Retrieve using expanded query
retrieved = await retriever.retrieve(
    query=expansion["combined_query"],
    top_k=20,
)

# 5. Enrich results
enriched = await enrichment.enrich_context(
    query=user_query,
    chunks=retrieved["chunks"],
    top_k=5,
)

# 6. Generate response using enriched chunks
```

---

## Operations Summary

| Operation | Description | LLM Required |
|-----------|-------------|--------------|
| `expand` | Main expansion with any strategy | Depends |
| `expand_semantic` | LLM semantic variations | Yes |
| `expand_synonyms` | Dictionary synonyms | No |
| `decompose_query` | Split complex queries | Yes (recommended) |
| `reformulate_query` | Different question forms | No |
| `extract_keywords` | Keyword extraction | No |
| `expand_contextual` | Chat history aware | Yes (recommended) |
| `expand_hybrid` | Multi-strategy combo | Depends |
| `detect_intent` | Classify query intent | No |
| `extract_entities` | Extract named entities | No |
| `normalize_query` | Clean and normalize | No |
| `detect_language` | Detect language | No |

---

## Changelog

### v1.0.0 (2025-01)
- Initial release
- 7 expansion strategies
- Intent detection
- Entity extraction
- Quality scoring
- Redis caching
- Multi-language support (EN/IT)
