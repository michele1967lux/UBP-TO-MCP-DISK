# Graph RAG

**Enterprise Knowledge Graph RAG Engine** for UBP Enterprise Hybrid

Version: 1.0.0 | Architecture: 3-file-pattern | Module Type: core

---

## Overview

`graph_rag` implements Graph-based RAG with knowledge graph construction and traversal. Unlike traditional vector-based RAG, Graph-RAG captures structured relationships between entities, enabling:

- **Multi-hop reasoning**: Follow paths through connected entities
- **Relationship-aware retrieval**: Filter by specific relation types
- **Contextual answers**: Include supporting paths and evidence
- **Explainable results**: Show how entities are connected

---

## Architecture

```
graph_rag/
├── __init__.py          # Module factory for ModuleLoader
├── adapter.py           # Bridge layer - exposes 20+ operations
├── providers.py         # Core logic - KnowledgeGraph, entities, relations
├── delegation.py        # LLM delegation for extraction
├── pipeline.py          # Graph-RAG orchestrator
├── prompts.py           # Extraction prompt templates
├── config.json          # 200+ environment variables
├── manifest.json        # Operation definitions and metadata
└── README.md            # This file
```

---

## Core Concepts

### Knowledge Graph Structure

```
┌─────────────┐          ┌─────────────┐
│   Entity    │──────────│   Entity    │
│  (Node)     │ Relation │   (Node)    │
│             │  (Edge)  │             │
│ - id        │          │ - id        │
│ - text      │          │ - text      │
│ - type      │──────────│ - type      │
│ - properties│          │ - properties│
└─────────────┘          └─────────────┘
```

### Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| `person` | People, individuals | "John Smith", "Dr. Johnson" |
| `organization` | Companies, institutions | "Google", "MIT" |
| `location` | Places, addresses | "New York", "Building A" |
| `technology` | Technologies, tools | "Python", "Kubernetes" |
| `concept` | Abstract concepts | "Machine Learning", "Agile" |
| `product` | Products, services | "iPhone", "AWS Lambda" |
| `event` | Events, occurrences | "World War II", "IPO 2023" |
| `date` | Dates, time periods | "2023", "Q3 2024" |

### Relation Types

| Type | Description | Example |
|------|-------------|---------|
| `works_for` | Employment | (John) --[works_for]--> (Google) |
| `located_in` | Location | (Google) --[located_in]--> (California) |
| `created_by` | Creation | (Python) --[created_by]--> (Guido van Rossum) |
| `part_of` | Part-whole | (GPU) --[part_of]--> (Server) |
| `depends_on` | Dependency | (TensorFlow) --[depends_on]--> (Python) |
| `causes` | Causation | (Bug) --[causes]--> (Crash) |
| `related_to` | General | (AI) --[related_to]--> (Machine Learning) |

---

## Pipeline Flow

```
Query: "Who created Python and where do they work?"
                │
                ▼
┌───────────────────────────────────────────┐
│  1. Parse Query                           │
│     - Detect language                     │
│     - Analyze structure                   │
└───────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
│  2. Extract Query Entities                │
│     - "Python" (technology)               │
│     - Intent: find creator and workplace  │
└───────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
│  3. Graph Retrieval                       │
│     - Match "Python" in graph             │
│     - Find: entity_id="python_001"        │
└───────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
│  4. Subgraph Extraction                   │
│     (Python) ──[created_by]──> (Guido)    │
│     (Guido) ──[works_for]──> (Google)     │
│     (Google) ──[located_in]──> (California)│
└───────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
│  5. Context Aggregation                   │
│     - Relevant paths identified           │
│     - Key facts extracted                 │
└───────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
│  6. Generate Answer                       │
│     "Python was created by Guido van      │
│      Rossum, who works at Google."        │
└───────────────────────────────────────────┘
```

---

## Operations

### Graph Building Operations

| Operation | Description |
|-----------|-------------|
| `build_graph` | Build graph from documents |
| `extract_entities` | Extract entities from text |
| `extract_relations` | Extract relations between entities |
| `add_entity` | Add single entity manually |
| `add_relation` | Add single relation manually |
| `clear_graph` | Clear entire graph |

### Query Operations

| Operation | Description |
|-----------|-------------|
| `query` | Full Graph-RAG pipeline |
| `search_entities` | Search entities by name |
| `get_subgraph` | Extract subgraph around entities |
| `find_paths` | Find paths between entities |

### Admin Operations

| Operation | Description |
|-----------|-------------|
| `get_graph_stats` | Graph statistics |
| `export_graph` | Export as triples/JSON |
| `get_stats` | Metrics and statistics |
| `reload_config` | Hot-reload configuration |

---

## Usage Examples

### Build Knowledge Graph from Documents

```python
result = await graph.build_graph(
    texts=[
        "Python was created by Guido van Rossum in 1991. "
        "Guido worked at Google from 2005 to 2012.",
        
        "TensorFlow is an ML framework developed by Google. "
        "It is written in Python and C++.",
    ],
    language="en",
)

print(f"Entities: {result['entities_added']}")  # 7
print(f"Relations: {result['relations_added']}")  # 5
```

### Query the Graph

```python
result = await graph.query(
    query="What is the relationship between Python and Google?",
    max_hops=3,
    retrieval_strategy="hybrid",
)

print(f"Answer: {result['answer']}")
print(f"Supporting facts: {result['supporting_fact_count']}")
print(f"Confidence: {result['confidence']}")
```

### Search Entities

```python
results = await graph.search_entities(
    query="Python",
    top_k=5,
    entity_types=["technology", "product"],
    fuzzy=True,
)

for item in results['results']:
    entity = item['entity']
    score = item['score']
    print(f"{entity['normalized']} ({entity['entity_type']}): {score}")
```

### Extract Subgraph

```python
subgraph = await graph.get_subgraph(
    entity_ids=["python_001", "google_001"],
    max_depth=2,
    max_nodes=50,
)

print(f"Nodes: {subgraph['node_count']}")
print(f"Edges: {subgraph['edge_count']}")

# Get as triples
for triple in subgraph['triples']:
    print(f"{triple['subject']} --[{triple['predicate']}]--> {triple['object']}")
```

### Find Paths

```python
paths = await graph.find_paths(
    source_entity_id="python_001",
    target_entity_id="google_001",
    max_length=4,
    max_paths=5,
)

for path in paths['paths']:
    print(f"Path (length {path['length']}): {' -> '.join(path['nodes'])}")
```

### Manual Graph Construction

```python
# Add entities
await graph.add_entity(
    text="Anthropic",
    normalized="Anthropic",
    entity_type="organization",
    properties={"founded": 2021, "hq": "San Francisco"},
)

await graph.add_entity(
    text="Claude",
    normalized="Claude",
    entity_type="product",
    properties={"type": "AI Assistant"},
)

# Add relation
await graph.add_relation(
    source_id="claude_001",
    target_id="anthropic_001",
    relation_type="created_by",
    evidence="Claude is an AI assistant created by Anthropic.",
)
```

---

## Retrieval Strategies

| Strategy | Use Case | Description |
|----------|----------|-------------|
| `entity_centric` | Entity lookup | Start from matched entities, expand neighborhood |
| `relation_guided` | Specific relations | Follow only certain relation types |
| `path_based` | Connection finding | Find paths between query entities |
| `subgraph` | Local context | Extract dense subgraph around seeds |
| `hybrid` | General queries | Combine graph + vector search (default) |

---

## Graph Algorithms

### PageRank
Compute entity importance based on graph structure.

```python
pagerank = knowledge_graph.compute_pagerank(
    damping=0.85,
    max_iterations=100,
)
```

### Degree Centrality
Find most connected entities.

```python
centrality = knowledge_graph.compute_degree_centrality()
top_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
```

### Path Finding (BFS)
Find shortest paths between entities.

```python
paths = knowledge_graph.find_paths(
    source_id="entity_a",
    target_id="entity_b",
    max_length=5,
)
```

---

## Configuration

### Key Environment Variables

```bash
# Core Settings
UBP_GRAPH__ENABLED=true
UBP_GRAPH__DEFAULT_BACKEND=memory
UBP_GRAPH__MAX_SIZE=100000

# Entity Extraction
UBP_GRAPH__ENTITY_ENABLED=true
UBP_GRAPH__ENTITY_MIN_CONFIDENCE=0.6
UBP_GRAPH__ENTITY_MAX_PER_CHUNK=20

# Relation Extraction
UBP_GRAPH__RELATION_ENABLED=true
UBP_GRAPH__RELATION_MIN_CONFIDENCE=0.5
UBP_GRAPH__RELATION_EVIDENCE_SPANS=true

# Graph Construction
UBP_GRAPH__MERGE_STRATEGY=smart
UBP_GRAPH__MAX_NODES=50000
UBP_GRAPH__MAX_EDGES=200000

# Retrieval
UBP_GRAPH__RETRIEVAL_STRATEGY=hybrid
UBP_GRAPH__RETRIEVAL_MAX_HOPS=3
UBP_GRAPH__RETRIEVAL_MAX_SUBGRAPH=100

# LLM Delegation
UBP_GRAPH__LLM_MODULE=inference_ollama_grok
UBP_GRAPH__LLM_TIMEOUT=30

# Cache
UBP_GRAPH__CACHE_ENABLED=true
UBP_GRAPH__CACHE_TTL=3600
```

---

## Export Formats

### Triples Format

```json
{
  "format": "triples",
  "count": 150,
  "triples": [
    {
      "subject": "Python",
      "predicate": "created_by",
      "object": "Guido van Rossum",
      "confidence": 0.95
    }
  ]
}
```

### JSON Format

```json
{
  "format": "json",
  "stats": {"node_count": 100, "edge_count": 150},
  "entities": [...],
  "relations": [...]
}
```

---

## Integration with reasoning_rag

Graph-RAG can be combined with reasoning_rag for enhanced capabilities:

```python
# 1. Build knowledge graph
await graph.build_graph(texts=documents)

# 2. Use graph for context retrieval
subgraph = await graph.get_subgraph(entity_ids=query_entities)

# 3. Pass to reasoning for answer generation
result = await reasoning.chain_of_thought(
    query=user_query,
    context=subgraph.get_triples(),
)
```

---

## Deployment

### 1. Copy Module

```bash
cp -r graph_rag/ modules/cores/graph_rag/
```

### 2. Configure Environment

```bash
# .env
UBP_GRAPH__ENABLED=true
UBP_GRAPH__LLM_MODULE=inference_vllm
UBP_GRAPH__MAX_NODES=100000
```

### 3. Initialize

```python
from modules.cores.graph_rag import create_module

module = create_module(
    module_path=Path("modules/cores/graph_rag"),
    di_container=container,
    event_bus=event_bus,
)

await module.initialize()
```

---

## Troubleshooting

### Low Entity Extraction

- Lower `min_confidence` threshold
- Increase `max_entities_per_chunk`
- Check LLM availability

### Missing Relations

- Lower `relation_min_confidence`
- Enable `bidirectional_detection`
- Verify entities are extracted first

### Slow Graph Queries

- Reduce `max_hops`
- Reduce `max_subgraph_nodes`
- Enable caching
- Use `entity_centric` strategy for simple lookups

### Memory Issues

- Reduce `max_nodes` and `max_edges`
- Enable graph persistence
- Consider Redis backend

---

## Dependencies

### Required
- Python 3.10+

### Optional
- Redis (for caching and persistence)
- Neo4j (for production graph storage)
- `inference_vllm` / `inference_ollama_grok` (for LLM extraction)

---

## License

Internal use - UBP Enterprise Hybrid

---

## Changelog

### v1.0.0 (2025-01)
- Initial release
- Entity extraction with LLM
- Relation extraction with evidence
- In-memory knowledge graph
- Graph-based retrieval
- Multi-hop path finding
- Subgraph extraction
- PageRank and centrality
- Cross-lingual EN/IT
- Redis caching
- Session management
- Export formats (triples, JSON)
