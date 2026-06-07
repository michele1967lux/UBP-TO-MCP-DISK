# RAG Reranker Module v1.0

Document reranking with cross-encoder models for improved RAG retrieval quality in UBP Enterprise Hybrid.

## Architecture Overview

This module implements the **3-File Pattern Architecture**:

```
rag_reranker/
├── __init__.py          # Entry point with create_module() factory
├── adapter.py           # RerankerAdapter (UBP integration layer)
├── providers.py         # RerankerProvider + backend implementations
├── config.json          # Configuration
├── manifest.json        # Module metadata and operations
└── README.md            # This file
```

## Features

### Reranking Backends
- **Cross-Encoder (Local)**: sentence-transformers cross-encoder models
- **Cohere Rerank**: Cohere's rerank API (requires API key)
- **OpenAI**: Custom reranking via GPT models
- **Simple**: TF-IDF based fallback (no dependencies)

### Two-Stage Retrieval
1. **Initial Retrieval**: Get top-K candidates (50-100 docs)
2. **Reranking**: Score each candidate against query
3. **Final Selection**: Return top-N reranked results

### Production Features
- **Backend Fallback**: Graceful degradation to simple backend
- **Batch Processing**: Efficient batch scoring
- **Score Normalization**: Consistent 0-1 score range

## Operations

| Operation | Description | Auth Required |
|-----------|-------------|---------------|
| `rerank` | Rerank provided documents | Yes |
| `rerank_search_results` | Search + rerank in one call | Yes |
| `get_reranker_info` | Get backend information | Yes |
| `compare_backends` | Compare results across backends (admin) | Admin |
| `health_check` | Check module health | No |

## Configuration

### config.json

```json
{
  "enabled": true,
  "backend": "cross-encoder",
  "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "default_top_k": 10,
  "batch_size": 32,
  "cohere_api_key": null,
  "openai_model": "gpt-3.5-turbo"
}
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | true | Enable/disable module |
| `backend` | string | "cross-encoder" | Reranking backend |
| `model` | string | "ms-marco-MiniLM-L-6-v2" | Cross-encoder model |
| `default_top_k` | integer | 10 | Default results after reranking |
| `batch_size` | integer | 32 | Batch size for scoring |
| `cohere_api_key` | string | null | Cohere API key (if using cohere) |

### Backend Options

| Backend | Model | Pros | Cons |
|---------|-------|------|------|
| `cross-encoder` | ms-marco-* | Fast, local, free | Requires GPU for speed |
| `cohere` | rerank-english-v2.0 | High quality | API costs |
| `openai` | gpt-3.5-turbo | Flexible | Slow, expensive |
| `simple` | TF-IDF | No dependencies | Lower quality |

## API Usage Examples

### Rerank Documents
```bash
curl -X POST "http://localhost:8000/api/modules/rag_reranker/rerank" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to optimize neural networks?",
    "documents": [
      {"content": "Neural network optimization involves..."},
      {"content": "Database optimization techniques..."},
      {"content": "Gradient descent is used to..."}
    ],
    "top_k": 2
  }'
```

### Search + Rerank (One Call)
```bash
curl -X POST "http://localhost:8000/api/modules/rag_reranker/rerank_search_results" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning optimization",
    "collection_name": "ml_docs",
    "initial_top_k": 50,
    "final_top_k": 10,
    "search_type": "hybrid"
  }'
```

### Get Reranker Info
```bash
curl -X POST "http://localhost:8000/api/modules/rag_reranker/get_reranker_info" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json"
```

## Response Example

```json
{
  "query": "How to optimize neural networks?",
  "results": [
    {
      "content": "Gradient descent is used to...",
      "score": 0.95,
      "original_rank": 3,
      "reranked_rank": 1,
      "metadata": {}
    },
    {
      "content": "Neural network optimization involves...",
      "score": 0.87,
      "original_rank": 1,
      "reranked_rank": 2,
      "metadata": {}
    }
  ],
  "count": 2,
  "original_count": 3,
  "backend": "cross-encoder",
  "request_id": "req-uuid"
}
```

## How Reranking Works

### Cross-Encoder vs Bi-Encoder

**Bi-Encoder (Initial Retrieval)**:
- Encodes query and documents separately
- Fast: O(1) per query (precomputed doc embeddings)
- Lower accuracy: No query-document interaction

**Cross-Encoder (Reranking)**:
- Encodes query + document together
- Slow: O(n) per query (must score each document)
- Higher accuracy: Full query-document attention

### Why Two-Stage?
```
All Documents (10,000)
    │
    ▼ Bi-Encoder (fast)
Top 50 Candidates
    │
    ▼ Cross-Encoder (accurate)
Top 10 Final Results
```

## Integration with Hybrid Search

```python
# In rag_orchestrator:
# 1. Hybrid search gets initial candidates
results = await hybrid_search.hybrid_search(query, collection, top_k=50)

# 2. Reranker refines the results
reranked = await reranker.rerank(query, results, top_k=10)
```

## Dependencies

- **Required**: rag_qdrant (for search integration)
- **Optional**: rag_hybrid_search (for hybrid+rerank)
- **Optional**: sentence-transformers (for cross-encoder backend)
- **Optional**: cohere (for Cohere backend)

## Performance Considerations

- **Latency**: Cross-encoder adds 50-200ms for 50 documents
- **GPU Acceleration**: Strongly recommended for production
- **Batch Size**: Tune based on available memory
- **Caching**: Consider caching for frequent queries

## Testing

```bash
# Run module tests
cd ubp_enterprise_hybrid
pytest tests/integration/test_rag_v150_modules.py -k "reranker" -v
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-31 | Initial release with ROADMAP v1.5.0 |

---

**Module Type:** search  
**Architecture:** 3-file-pattern  
**Production Ready:** Yes
