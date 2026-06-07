# Investigation Pipeline

**Enterprise RAG Investigation Engine** for UBP Enterprise Hybrid

Version: 1.0.0 | Architecture: 3-file-pattern | Module Type: investigation

---

## Overview

`investigation_pipeline` is an enterprise-grade module for generating high-quality investigative questions from user queries. Instead of generating hypothetical answers (like HyDE), it decomposes queries into targeted search questions that improve RAG retrieval accuracy.

### Key Capabilities

- **Multi-Strategy Generation**: 5 strategies optimized for different query types
- **Adaptive Strategy Selection**: Auto-detects optimal strategy via query classification
- **Quality Assurance System**: Multi-dimensional scoring with auto-retry on low quality
- **Parallel Worker Pool**: Async task execution with priority queue and backoff
- **Session Management**: Persistent sessions with history tracking
- **Redis Caching**: Environment-isolated caching for questions and QA results
- **Fallback Chains**: Graceful degradation when primary strategy fails
- **Comprehensive Observability**: Metrics, logging, and debug tracing

---

## Architecture

```
investigation_pipeline/
├── __init__.py          # Module factory for ModuleLoader
├── adapter.py           # Bridge layer - exposes all operations
├── providers.py         # Core logic - zero backend dependencies
├── delegation.py        # LLM delegation with multi-strategy support
├── pipeline.py          # 8-step pipeline orchestrator
├── prompts.py           # Strategy templates for 6 query categories
├── config.json          # 100+ environment variables
├── manifest.json        # Operation definitions and metadata
└── README.md            # This file
```

### Layer Responsibilities

| Layer | File | Purpose |
|-------|------|---------|
| **Entry Point** | `__init__.py` | Factory function for ModuleLoader integration |
| **Bridge** | `adapter.py` | Exposes operations, manages lifecycle, handles DI |
| **Logic** | `providers.py` | Core algorithms (QA, dedup, workers, sessions, cache) |
| **Delegation** | `delegation.py` | LLM communication with strategy selection |
| **Orchestration** | `pipeline.py` | Configurable multi-step pipeline execution |
| **Templates** | `prompts.py` | Category-specific prompt templates |

---

## Strategies

### Available Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `adaptive` | Auto-selects based on query classification | General use (default) |
| `decomposition` | Breaks into aspects: definition, components, purpose, implementation | Complex technical queries |
| `chain_of_thought` | Logical reasoning steps | Problem-solving, troubleshooting |
| `semantic_expansion` | Synonyms, related concepts, alternatives | Conceptual queries |
| `cross_reference` | Prerequisites, dependencies, related features | Integration, architecture |

### Strategy Selection (Adaptive Mode)

```
Query Category        → Recommended Strategy
─────────────────────────────────────────────
ai_ml                 → decomposition
troubleshooting       → chain_of_thought
api_integration       → cross_reference
system_admin          → decomposition
technical             → decomposition
conceptual            → semantic_expansion
```

---

## Pipeline Steps

The full investigation pipeline executes 8 configurable steps:

```
1. classify_query      → Detect category and language
2. select_strategy     → Choose optimal strategy (or use override)
3. generate_questions  → LLM generates investigation questions
4. quality_assurance   → Score and validate questions
5. cross_reference     → (Optional) Add related questions
6. deduplicate         → Remove duplicate/similar questions
7. rank_questions      → Sort by quality score
8. format_output       → Prepare final result
```

Each step can be enabled/disabled via configuration:

```json
{
  "pipeline": {
    "steps": {
      "classify_query": { "enabled": true, "timeout": 5 },
      "quality_assurance": { "enabled": true, "timeout": 10 },
      "cross_reference": { "enabled": false },
      "deduplicate": { "enabled": true }
    }
  }
}
```

---

## Operations

### Core Operations

| Operation | Description | Admin |
|-----------|-------------|-------|
| `investigate` | Full pipeline execution | No |
| `generate_questions` | Direct generation (no pipeline) | No |
| `generate_multi_strategy` | Parallel multi-strategy generation | No |
| `classify_query` | Query category detection | No |
| `assess_quality` | QA validation for questions | No |
| `deduplicate_questions` | Remove duplicates | No |

### Session Operations

| Operation | Description | Admin |
|-----------|-------------|-------|
| `get_session` | Retrieve session state | No |
| `delete_session` | Delete session | No |

### Admin Operations

| Operation | Description | Admin |
|-----------|-------------|-------|
| `get_stats` | Metrics and statistics | Yes |
| `get_pipeline_config` | Current configuration | No |
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

### Basic Investigation

```python
# Full pipeline with adaptive strategy
result = await investigation.investigate(
    query="How do I implement RAG with Qdrant?",
    num_questions=5,
)

# Returns:
# {
#   "session_id": "uuid",
#   "questions": [...],
#   "strategy_used": "decomposition",
#   "category_detected": "ai_ml",
#   "quality_assessment": {...},
#   "time_ms": 1234.56
# }
```

### Strategy Override

```python
# Force chain-of-thought for troubleshooting
result = await investigation.investigate(
    query="Why is my Docker container crashing?",
    num_questions=5,
    strategy="chain_of_thought",
)
```

### Multi-Strategy Generation

```python
# Generate with multiple strategies in parallel
results = await investigation.generate_multi_strategy(
    query="Explain transformer architecture",
    num_questions=3,
    strategies=["decomposition", "semantic_expansion", "chain_of_thought"],
)

# Returns dict with results from each strategy
```

### Direct Generation (No Pipeline)

```python
# Skip pipeline, direct LLM call
result = await investigation.generate_questions(
    query="What is vector similarity search?",
    num_questions=5,
    strategy="semantic_expansion",
    category="ai_ml",
)
```

### Session Continuity

```python
# First investigation
result1 = await investigation.investigate(
    query="How does embedding work?",
    num_questions=3,
)
session_id = result1["session_id"]

# Continue in same session
result2 = await investigation.investigate(
    query="How do I choose embedding dimensions?",
    num_questions=3,
    session_id=session_id,
)
```

---

## Quality Assurance

### Scoring Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Relevance | 40% | Keyword overlap with original query |
| Specificity | 25% | Technical terms and precision |
| Length | 20% | Optimal character count (50-150) |
| Structure | 15% | Proper question format |

### Quality Levels

| Level | Score | Description |
|-------|-------|-------------|
| EXCELLENT | ≥ 8.0 | High-quality, ready for retrieval |
| GOOD | ≥ 6.0 | Acceptable quality |
| ACCEPTABLE | ≥ 4.0 | Minimum acceptable |
| POOR | < 4.0 | Triggers retry if enabled |

### Auto-Retry

When `auto_retry_on_low_quality` is enabled:
1. Questions scoring below `min_acceptable_score` trigger retry
2. Up to `max_qa_retries` attempts
3. Falls back through strategy chain if needed

---

## Configuration

### Environment Variables

All settings can be overridden via environment variables with prefix `UBP_INVESTIGATION__`:

```bash
# Core settings
UBP_INVESTIGATION__ENABLED=true
UBP_INVESTIGATION__DEFAULT_NUM_QUESTIONS=5
UBP_INVESTIGATION__DEFAULT_STRATEGY=adaptive
UBP_INVESTIGATION__TEMPERATURE=0.7
UBP_INVESTIGATION__MAX_TOKENS=500
UBP_INVESTIGATION__TIMEOUT_SECONDS=30

# Quality Assurance
UBP_INVESTIGATION__QA_ENABLED=true
UBP_INVESTIGATION__QA_MIN_SCORE=4.0
UBP_INVESTIGATION__QA_AUTO_RETRY=true
UBP_INVESTIGATION__QA_MAX_RETRIES=2

# Worker Pool
UBP_INVESTIGATION__WORKERS_ENABLED=true
UBP_INVESTIGATION__WORKER_POOL_SIZE=4
UBP_INVESTIGATION__WORKER_MAX_POOL_SIZE=8
UBP_INVESTIGATION__WORKER_TASK_TIMEOUT=30

# Cache
UBP_INVESTIGATION__CACHE_ENABLED=true
UBP_INVESTIGATION__CACHE_TTL=3600

# Deduplication
UBP_INVESTIGATION__DEDUP_ENABLED=true
UBP_INVESTIGATION__DEDUP_THRESHOLD=0.85
UBP_INVESTIGATION__DEDUP_METHOD=fuzzy

# Session
UBP_INVESTIGATION__SESSION_TTL=3600
UBP_INVESTIGATION__SESSION_MAX_HISTORY=50

# Debug
UBP_INVESTIGATION__DEBUG_ENABLED=false
UBP_INVESTIGATION__DEBUG_LOG_PROMPTS=false
UBP_INVESTIGATION__DEBUG_LOG_RESPONSES=false
UBP_INVESTIGATION__DEBUG_LOG_QA_SCORES=true
UBP_INVESTIGATION__DEBUG_LOG_FALLBACKS=true

# LLM Delegation
UBP_INVESTIGATION__LLM_MODULE=inference_ollama_grok
UBP_INVESTIGATION__LLM_OPERATION=generate
UBP_INVESTIGATION__FALLBACK_ENABLED=true
```

### LLM Provider Configuration

The module supports multiple LLM backends via ProviderMapper:

```bash
# Direct module specification
UBP_INVESTIGATION__LLM_MODULE=inference_vllm

# Or via role-based configuration (recommended)
# Configure in ProviderMapper for role "investigation"
```

**Supported modules:**
- `inference_vllm` - vLLM backend
- `inference_ollama_grok` - Ollama with Grok models
- `inference_openai_anthropic` - OpenAI/Anthropic APIs

---

## Fallback Chain

When primary generation fails, the module falls back through a chain:

```
Primary Strategy
      ↓ (fail)
decomposition
      ↓ (fail)
chain_of_thought
      ↓ (fail)
simple (basic prompt)
```

Configure via:
```bash
UBP_INVESTIGATION__FALLBACK_ENABLED=true
UBP_INVESTIGATION__FALLBACK_CHAIN=decomposition,chain_of_thought,simple
```

---

## Redis Keys

Environment-isolated key patterns:

```
ubp:{env}:investigation:cache:questions:{hash}
ubp:{env}:investigation:cache:qa:{hash}
ubp:{env}:investigation:session:{session_id}
ubp:{env}:investigation:stats:*
```

Where `{env}` is `dev`, `test`, or `prod` from `UBP_ENV`.

---

## Events Published

| Event | Description |
|-------|-------------|
| `investigation.pipeline.started` | Pipeline execution began |
| `investigation.pipeline.completed` | Pipeline finished successfully |
| `investigation.pipeline.failed` | Pipeline failed |
| `investigation.generation.completed` | Questions generated |
| `investigation.generation.failed` | Generation failed |
| `investigation.fallback.succeeded` | Fallback strategy worked |
| `investigation.fallback.exhausted` | All fallbacks failed |
| `investigation.qa.completed` | Quality assessment done |
| `investigation.session.created` | New session created |
| `investigation.session.updated` | Session updated |
| `investigation.health.degraded` | Component health degraded |

---

## Deployment

### 1. Copy Module

```bash
cp -r investigation_pipeline/ modules/cores/investigation_pipeline/
```

### 2. Configure Environment

```bash
# .env or environment
UBP_ENV=prod
UBP_INVESTIGATION__ENABLED=true
UBP_INVESTIGATION__LLM_MODULE=inference_vllm
```

### 3. Initialize via ModuleLoader

```python
from modules.cores.investigation_pipeline import create_module

module = create_module(
    module_path=Path("modules/cores/investigation_pipeline"),
    di_container=container,
    event_bus=event_bus,
)

await module.initialize()
```

### 4. Health Check

```python
health = await module.health_check()
# {
#   "module": "investigation_pipeline",
#   "status": "healthy",
#   "llm_delegation": {"status": "available"},
#   "worker_pool": {"active_workers": 4},
#   "cache": {"hit_rate": 0.85}
# }
```

---

## Metrics

Available via `get_stats()` (admin only):

```python
stats = await module.get_stats(period="24h")
# {
#   "total_investigations": 1234,
#   "questions_generated": 6170,
#   "strategy_distribution": {
#     "adaptive": 800,
#     "decomposition": 300,
#     "chain_of_thought": 134
#   },
#   "category_distribution": {
#     "ai_ml": 450,
#     "troubleshooting": 320,
#     ...
#   },
#   "qa_scores": {
#     "avg": 7.2,
#     "min": 4.1,
#     "max": 9.8
#   },
#   "execution_times": {
#     "avg_ms": 1850,
#     "p50_ms": 1650,
#     "p95_ms": 3200
#   },
#   "fallback_triggers": 42,
#   "cache": {
#     "hits": 890,
#     "misses": 344,
#     "hit_rate": 0.72
#   }
# }
```

---

## Troubleshooting

### LLM Delegation Not Available

```
Check:
1. UBP_INVESTIGATION__LLM_MODULE points to valid module
2. Target inference module is initialized
3. ProviderMapper has "investigation" role configured
```

### Low Quality Scores

```
Adjust:
- UBP_INVESTIGATION__TEMPERATURE (lower = more focused)
- UBP_INVESTIGATION__QA_MIN_SCORE (threshold)
- UBP_INVESTIGATION__QA_AUTO_RETRY=true
```

### High Latency

```
Check:
- UBP_INVESTIGATION__WORKER_POOL_SIZE (increase workers)
- UBP_INVESTIGATION__CACHE_ENABLED=true
- UBP_INVESTIGATION__TIMEOUT_SECONDS (adjust if needed)
```

### Fallback Loop

```
If constantly falling back:
1. Check LLM module health
2. Review debug logs: UBP_INVESTIGATION__DEBUG_LOG_FALLBACKS=true
3. Simplify fallback chain
```

---

## Dependencies

### Required
- Python 3.10+
- Redis (for caching and sessions)
- Event Bus (for observability)

### Optional
- `inference_vllm` - vLLM inference
- `inference_ollama_grok` - Ollama inference
- `inference_openai_anthropic` - Cloud APIs

### Python Packages
- `aioredis` - Async Redis client
- Standard library only for core logic

---

## License

Internal use - UBP Enterprise Hybrid

---

## Changelog

### v1.0.0 (2025-01)
- Initial release
- 5 investigation strategies
- Multi-dimensional QA system
- Parallel worker pool
- Session management
- Redis caching with env isolation
- Fallback chains
- Full observability
