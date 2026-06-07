# RAG Hybrid Search Module v1.0

Hybrid search combining dense (vector) and sparse (BM25) retrieval with score fusion for UBP Enterprise Hybrid.

## Architecture Overview

This module implements the **3-File Pattern Architecture**:

```
rag_hybrid_search/
├── __init__.py          # Entry point with create_module() factory
├── adapter.py           # HybridSearchAdapter (UBP integration layer)
├── providers.py         # HybridSearchProvider + BM25Index (business logic)
├── config.json          # Configuration
├── manifest.json        # Module metadata and operations
└── README.md            # This file
```

## Features

### Search Modes
- **Hybrid Search**: Combines dense and sparse results with score fusion
- **Sparse Search**: BM25-only retrieval for keyword matching
- **Dense Search**: Vector-only retrieval via rag_qdrant integration

### Score Fusion Methods
- **RRF (Reciprocal Rank Fusion)**: Default, robust for diverse results
- **Weighted**: Configurable dense/sparse weight balance
- **Max**: Takes maximum score from either method

### BM25 Index
- **In-Memory Index**: Fast sparse retrieval
- **Auto-Indexing**: Subscribes to qdrant.document_indexed events
- **Per-Collection**: Separate index for each collection

## Operations

| Operation | Description | Auth Required |
|-----------|-------------|---------------|
| `hybrid_search` | Combined dense + sparse search | Yes |
| `sparse_search` | BM25-only search | Yes |
| `index_document` | Add document to BM25 index | Yes |
| `remove_document` | Remove from BM25 index | Yes |
| `get_index_stats` | Get BM25 index statistics | Yes |
| `list_indexed_collections` | List indexed collections | Yes |
| `rebuild_index` | Rebuild index from Qdrant (admin) | Admin |
| `clear_index` | Clear collection index (admin) | Admin |
| `health_check` | Check module health | No |

## Configuration

### config.json

```json
{
  "enabled": true,
  "default_fusion_method": "rrf",
  "default_dense_weight": 0.7,
  "rrf_k": 60,
  "bm25_k1": 1.5,
  "bm25_b": 0.75,
  "default_top_k": 10
}
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | true | Enable/disable module |
| `default_fusion_method` | string | "rrf" | Default fusion (rrf/weighted/max) |
| `default_dense_weight` | float | 0.7 | Dense weight for weighted fusion |
| `rrf_k` | integer | 60 | RRF constant (higher = more uniform) |
| `bm25_k1` | float | 1.5 | BM25 term frequency saturation |
| `bm25_b` | float | 0.75 | BM25 length normalization |
| `default_top_k` | integer | 10 | Default results to return |

## Score Fusion Explained

### Reciprocal Rank Fusion (RRF)
```
RRF_score(d) = sum(1 / (k + rank_i(d)))
```
- Combines rankings, not raw scores
- Robust to score scale differences
- `k=60` is the standard constant

### Weighted Fusion
```
final_score = dense_weight * dense_score + (1 - dense_weight) * sparse_score
```
- Direct score combination
- Requires normalized scores
- Best when both methods produce similar scales

## API Usage Examples

### Hybrid Search
```bash
curl -X POST "http://localhost:8000/api/modules/rag_hybrid_search/hybrid_search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning optimization",
    "collection_name": "ml_docs",
    "top_k": 10,
    "fusion_method": "rrf"
  }'
```

### Sparse-Only Search
```bash
curl -X POST "http://localhost:8000/api/modules/rag_hybrid_search/sparse_search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "gradient descent",
    "collection_name": "ml_docs",
    "top_k": 5
  }'
```

### Get Index Stats
```bash
curl -X POST "http://localhost:8000/api/modules/rag_hybrid_search/get_index_stats" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "ml_docs"}'
```

## Response Example

```json
{
  "query": "machine learning optimization",
  "collection": "ml_docs",
  "results": [
    {
      "doc_id": "doc-001",
      "content": "Gradient descent is an optimization...",
      "score": 0.89,
      "dense_score": 0.85,
      "sparse_score": 0.92,
      "metadata": {"source": "textbook"}
    }
  ],
  "count": 10,
  "search_type": "hybrid",
  "fusion_method": "rrf",
  "dense_results_count": 10,
  "sparse_results_count": 8,
  "request_id": "req-uuid"
}
```

## Event Integration

### Subscribed Events
- `qdrant.document_indexed` - Auto-index new documents
- `qdrant.document_deleted` - Remove from BM25 index

This allows the BM25 index to stay synchronized with the Qdrant vector store automatically.

## Dependencies

- **Required**: rag_qdrant (for dense search and document sync)
- **Optional**: Event Bus (for auto-indexing)

## Performance Considerations

- **Memory Usage**: BM25 index is in-memory; monitor for large collections
- **Index Rebuild**: Can be slow for large collections; run during low traffic
- **Fusion Overhead**: Minimal; RRF adds ~1-2ms per search

## Testing

```bash
# Run module tests
cd ubp_enterprise_hybrid
pytest tests/integration/test_rag_v150_modules.py -k "hybrid_search" -v
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-31 | Initial release with ROADMAP v1.5.0 |

---

**Module Type:** search  
**Architecture:** 3-file-pattern  
**Production Ready:** Yes
