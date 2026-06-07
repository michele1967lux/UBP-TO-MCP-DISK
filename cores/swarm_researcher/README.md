# swarm_researcher

**Multi-Source Parallel Research Module**

Version: 1.0.0 | Architecture: 3-file-pattern | Pipeline-native

---

## Overview

The `swarm_researcher` module provides parallel research execution with:

- **Multi-Source Retrieval**: RAG (knowledge base) + Web search
- **Parallel Execution**: Concurrent queries with adaptive worker scaling
- **Citation Tracking**: Automatic citation extraction and bibliography generation
- **Result Aggregation**: RRF, weighted, and other fusion strategies
- **Quality Analysis**: Source scoring and coverage reporting
- **Dependency Management**: Task scheduling respecting dependencies

## Features

| Feature | Description |
|---------|-------------|
| **Parallel Execution** | Execute multiple queries concurrently |
| **Fallback Logic** | Automatic fallback between sources |
| **RRF Aggregation** | Reciprocal Rank Fusion for result merging |
| **Deduplication** | Remove duplicate content across sources |
| **Citation Styles** | APA, MLA, Chicago, IEEE, Harvard |
| **Quality Scoring** | Evaluate source relevance and authority |
| **Progress Tracking** | Real-time execution progress |
| **Retry Logic** | Intelligent retry with exponential backoff |

## Quick Start

```python
from swarm_researcher import create_module
from pathlib import Path

# Create and initialize
researcher = create_module(Path("./swarm_researcher"), di_container)
await researcher.initialize(max_workers=5)

# Single query research
result = await researcher.research_single(
    query="machine learning best practices",
    source_config={"preference": "rag_first", "collections": ["docs"]}
)

# Parallel research
result = await researcher.research_parallel(
    queries=["topic 1", "topic 2", "topic 3"],
    source_config={
        "preference": "mixed",
        "collections": ["docs"],
        "web_enabled": True
    },
    extract_citations=True
)

print(f"Found {len(result['documents'])} documents")
print(f"Generated {len(result['citations'])} citations")
```

## Operations

### Research Operations

#### `research_single`

Research a single query.

```python
result = await researcher.research_single(
    query="What are the best practices for API design?",
    source_config={
        "preference": "rag_first",
        "collections": ["technical_docs"],
        "top_k": 10,
        "rerank": True
    }
)
```

#### `research_parallel`

Research multiple queries in parallel.

```python
result = await researcher.research_parallel(
    queries=[
        "API authentication methods",
        "REST vs GraphQL",
        "API versioning strategies"
    ],
    source_config={
        "preference": "mixed",
        "collections": ["docs"],
        "max_parallel": 5
    },
    deduplicate=True,
    extract_citations=True
)

# Result contains:
# - results: Individual results per query
# - aggregated: Merged and deduplicated
# - documents: Final document list
# - citations: Extracted citations
# - stats: Execution statistics
```

#### `research_section`

Research optimized for a document section.

```python
result = await researcher.research_section(
    section_id="methodology",
    queries=[
        "research methodology types",
        "data collection methods",
        "analysis techniques"
    ],
    max_documents=10
)
```

#### `research_with_dependencies`

Research with task dependencies.

```python
result = await researcher.research_with_dependencies(
    task_graph=[
        {
            "id": "background",
            "suggested_queries": ["topic background", "history"],
            "source_preference": "rag_first",
            "depends_on": []
        },
        {
            "id": "analysis",
            "suggested_queries": ["detailed analysis", "comparison"],
            "source_preference": "mixed",
            "depends_on": ["background"]
        },
        {
            "id": "conclusions",
            "suggested_queries": ["conclusions", "recommendations"],
            "source_preference": "llm_reasoning",
            "depends_on": ["analysis"]
        }
    ]
)

# Results organized by section
for section_id, section_data in result["by_section"].items():
    print(f"Section {section_id}: {section_data['documents_count']} docs")
```

### Aggregation Operations

#### `aggregate_results`

Aggregate multiple research results.

```python
result = await researcher.aggregate_results(
    results=previous_results,
    strategy="rrf",  # rrf, weighted, interleave, union
    deduplicate=True
)
```

#### `deduplicate`

Remove duplicate documents.

```python
result = await researcher.deduplicate(
    documents=documents,
    similarity_threshold=0.85
)
print(f"Removed {result['duplicates_removed']} duplicates")
```

### Citation Operations

#### `track_citations`

Track citations for documents.

```python
result = await researcher.track_citations(
    documents=documents,
    style="apa"
)
```

#### `generate_bibliography`

Generate formatted bibliography.

```python
result = await researcher.generate_bibliography(
    style="apa",
    numbered=True
)

print(result["formatted_text"])
# [1] Author, A. (2024). Title. Retrieved from https://...
# [2] Author, B. (2023). Another Title.
```

### Quality Operations

#### `score_quality`

Score document quality.

```python
result = await researcher.score_quality(documents=documents)

for score in result["scores"]:
    print(f"Doc {score['document_id']}: {score['overall']:.2f}")
    print(f"  Relevance: {score['relevance']:.2f}")
    print(f"  Authority: {score['authority']:.2f}")
```

#### `analyze_coverage`

Analyze research coverage.

```python
result = await researcher.analyze_coverage(
    queries=original_queries,
    results=research_results,
    required_topics=["authentication", "authorization", "encryption"]
)

coverage = result["coverage"]
print(f"Coverage: {coverage['coverage_percentage']:.1f}%")
print(f"Missing: {coverage['topics_missing']}")
```

## Source Configuration

```python
source_config = {
    # Source preference
    "preference": "rag_first",  # rag_only, web_only, rag_first, web_first, mixed, adaptive
    
    # RAG settings
    "collections": ["docs", "knowledge_base"],
    "top_k": 10,
    "min_score": 0.5,
    "rerank": True,
    "rerank_top_k": 5,
    "enable_hyde": False,
    "metadata_filters": {"category": "technical"},
    
    # Web settings
    "web_enabled": True,
    "web_max_results": 5,
    
    # Fallback
    "fallback_enabled": True,
    
    # Execution
    "max_parallel": 5,
    "timeout": 60,
    
    # Processing
    "deduplicate": True,
    "extract_citations": True,
    "citation_style": "apa"
}
```

## Aggregation Strategies

| Strategy | Description |
|----------|-------------|
| `rrf` | Reciprocal Rank Fusion - combines rankings |
| `weighted` | Apply weights to sources (rag: 0.6, web: 0.4) |
| `interleave` | Round-robin from each source |
| `union` | Simple union of all documents |
| `concat` | Concatenate in order |

## Pipeline Integration

```yaml
name: research_pipeline
steps:
  - id: research
    module: swarm_researcher
    operation: research_parallel
    params:
      extract_citations: true
      deduplicate: true
    input_from:
      queries: plan.sections[*].suggested_queries
      source_config:
        preference: ${config.source_preference|default:rag_first}
        collections: inputs.collections
    output_as: research_data
    enabled: true
    timeout: 300

  - id: aggregate
    module: swarm_researcher
    operation: aggregate_results
    params:
      strategy: rrf
    input_from:
      results: research_data.results
    output_as: aggregated

  - id: bibliography
    module: swarm_researcher
    operation: generate_bibliography
    params:
      style: ${config.citation_style|default:apa}
    input_from:
      citations: aggregated.citations
    output_as: bibliography
```

## Citation Styles

| Style | Example |
|-------|---------|
| APA | Author, A. (2024). Title. Retrieved from URL |
| MLA | Author. "Title." Date. URL. |
| Chicago | Author. "Title." Date. URL. |
| IEEE | Author, "Title," Date. |
| Numeric | [1], [2], [3] inline |

## Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    research_parallel                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Create Tasks                                             │
│     queries → ResearchTask[]                                 │
│                                                              │
│  2. Parallel Execution (with semaphore)                      │
│     ┌─────────┐ ┌─────────┐ ┌─────────┐                     │
│     │ Task 1  │ │ Task 2  │ │ Task 3  │  ...                │
│     └────┬────┘ └────┬────┘ └────┬────┘                     │
│          │           │           │                           │
│          ▼           ▼           ▼                           │
│     ┌─────────────────────────────────┐                     │
│     │        Source Router            │                     │
│     │  RAG → Web → Fallback → Merge   │                     │
│     └─────────────────────────────────┘                     │
│                                                              │
│  3. Aggregate Results                                        │
│     RRF fusion + Deduplication                               │
│                                                              │
│  4. Track Citations                                          │
│     Extract + Format bibliography                            │
│                                                              │
│  5. Return                                                   │
│     {documents, citations, stats}                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Error Handling

```python
result = await researcher.research_parallel(queries)

# Check individual query success
for r in result["results"]:
    if not r["success"]:
        print(f"Query failed: {r['query']}, Error: {r.get('error')}")

# Check overall stats
stats = result["stats"]
print(f"Success rate: {stats['total_queries'] - stats.get('failed', 0)}/{stats['total_queries']}")
```

## Configuration

Environment variables:

```bash
SWARM_RESEARCHER__MAX_WORKERS=5
SWARM_RESEARCHER__TIMEOUT=60
SWARM_RESEARCHER__PREFERENCE=rag_first
SWARM_RESEARCHER__RAG_TOP_K=10
SWARM_RESEARCHER__RERANK=true
SWARM_RESEARCHER__WEB_ENABLED=true
SWARM_RESEARCHER__CITATION_STYLE=apa
SWARM_RESEARCHER__AGG_STRATEGY=rrf
```

---

**Module**: swarm_researcher v1.0.0 | **Architecture**: Pipeline-native | **Status**: Production Ready
