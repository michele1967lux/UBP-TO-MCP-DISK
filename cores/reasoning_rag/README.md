# Reasoning RAG

**Enterprise Reasoning-Aware RAG Engine** for UBP Enterprise Hybrid

Version: 1.0.0 | Architecture: 3-file-pattern | Module Type: core

---

## Overview

`reasoning_rag` implements advanced RAG strategies that combine reasoning with retrieval, enabling sophisticated question answering beyond simple lookup.

### The Problem with Traditional RAG

Standard RAG retrieves documents and generates answers in a single pass. This fails for:
- **Multi-hop queries**: Questions requiring information from multiple sources
- **Complex reasoning**: Questions needing step-by-step logic
- **Verification needs**: Claims that must be fact-checked
- **Citation requirements**: Answers that must be traceable to sources

### Our Solution: Reasoning-Aware RAG

This module provides four advanced strategies:

1. **Self-Ask RAG**: Iteratively decomposes complex queries into simpler sub-questions
2. **Chain-of-Thought RAG**: Interleaves reasoning steps with targeted retrieval
3. **Evidence Attribution**: Generates answers with inline citations and source tracking
4. **Verification**: Multi-source fact checking with contradiction detection

---

## Architecture

```
reasoning_rag/
├── __init__.py          # Module factory for ModuleLoader
├── adapter.py           # Bridge layer - exposes 15+ operations
├── providers.py         # Core logic - zero backend dependencies
├── delegation.py        # LLM and retrieval delegation
├── pipeline.py          # Multi-strategy orchestrator
├── prompts.py           # Strategy-specific prompt templates
├── config.json          # 180+ environment variables
├── manifest.json        # Operation definitions and metadata
└── README.md            # This file
```

---

## Strategies

### 1. Self-Ask RAG

**Paradigm**: Iterative sub-question decomposition

```
Query: "How does RAG compare to fine-tuning for improving LLM accuracy?"
    │
    ▼
┌─────────────────────────────────────────────┐
│  Iteration 1                                │
│  Sub-questions:                             │
│  1. What is RAG and how does it work?       │
│  2. What is fine-tuning for LLMs?           │
│  3. What metrics measure LLM accuracy?      │
│     │                                       │
│     ▼ Retrieve & Answer each                │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Iteration 2                                │
│  Sub-questions:                             │
│  1. What are RAG's advantages?              │
│  2. What are fine-tuning's advantages?      │
│     │                                       │
│     ▼ Retrieve & Answer each                │
└─────────────────────────────────────────────┘
    │
    ▼
Integration → Final Answer
```

**Best for**:
- Multi-part questions
- Comparative queries
- Questions requiring multiple facts

**Usage**:
```python
result = await reasoning.self_ask(
    query="Compare Python and JavaScript for backend development",
    max_iterations=5,
)
# Returns sub_questions, iteration_count, integrated answer
```

### 2. Chain-of-Thought RAG (Interleaved RAG)

**Paradigm**: Reasoning interleaved with retrieval

```
Query: "Why did the 2008 financial crisis happen?"
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 1: "I need to understand the housing  │
│           market conditions in 2008..."     │
│           → RETRIEVAL needed                │
│           → Fetch documents about housing   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 2: "Based on the retrieved info,      │
│           subprime mortgages were key..."   │
│           → Continue reasoning              │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 3: "I need to know about CDOs and     │
│           how they spread the risk..."      │
│           → RETRIEVAL needed                │
│           → Fetch documents about CDOs      │
└─────────────────────────────────────────────┘
    │
    ▼
... more steps ...
    │
    ▼
Synthesis → Final Answer
```

**Best for**:
- Explanatory questions (how/why)
- Causal reasoning
- Step-by-step procedures

**Usage**:
```python
result = await reasoning.chain_of_thought(
    query="Explain how neural networks learn",
    max_steps=8,
)
# Returns reasoning_steps, intermediate conclusions, final answer
```

### 3. Evidence Attribution

**Paradigm**: Every claim is traced and cited

```
Query: "What causes climate change?"
    │
    ▼
┌─────────────────────────────────────────────┐
│  Retrieve relevant documents                │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Generate answer with inline citations:     │
│                                             │
│  "Climate change is primarily caused by     │
│   greenhouse gas emissions [1], particularly│
│   CO2 from burning fossil fuels [2].        │
│   Deforestation contributes by reducing     │
│   carbon absorption [1][3]..."              │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Extract and attribute claims:              │
│  - Claim: "GHG causes climate change"       │
│    Source: [1], Confidence: 0.95            │
│  - Claim: "CO2 from fossil fuels"           │
│    Source: [2], Confidence: 0.92            │
└─────────────────────────────────────────────┘
```

**Best for**:
- Research questions
- Compliance-required contexts
- Academic/professional use

**Usage**:
```python
result = await reasoning.evidence_attribution(
    query="What are the health benefits of meditation?",
)
# Returns answer_with_citations, claims, evidence, citations
```

### 4. Verification

**Paradigm**: Multi-source fact checking

```
Claims to verify:
  1. "Einstein won the Nobel Prize for relativity"
  2. "The moon is 384,400 km from Earth"
    │
    ▼
┌─────────────────────────────────────────────┐
│  For each claim:                            │
│  1. Retrieve from multiple sources          │
│  2. Check for support or contradiction      │
│  3. Detect conflicts between sources        │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Results:                                   │
│  1. CONTRADICTED - Nobel was for            │
│     photoelectric effect, not relativity    │
│  2. VERIFIED - Multiple sources confirm     │
│     average distance is 384,400 km          │
└─────────────────────────────────────────────┘
```

**Best for**:
- Fact-checking
- Claim verification
- Critical information validation

**Usage**:
```python
result = await reasoning.verify_claims(
    query="Verify these claims about Python",
    claims=["Python was created in 1991", "Python is the fastest language"],
)
# Returns verifications, contradictions, confidence scores
```

---

## Automatic Strategy Selection

The module automatically selects the best strategy based on query analysis:

| Query Type | Detected Intent | Recommended Strategy |
|------------|-----------------|---------------------|
| "Compare X and Y" | comparative | self_ask |
| "How does X work?" | explanatory | chain_of_thought |
| "Why did X happen?" | causal | chain_of_thought |
| "What is X?" | definitional | evidence_attribution |
| "Is it true that X?" | factual | verification |
| Simple factual | factual | direct |

```python
# Auto-select strategy
result = await reasoning.reason(
    query="Your question here",
    strategy="auto",  # or omit for auto
)
```

---

## Operations

### Core Reasoning Operations

| Operation | Description |
|-----------|-------------|
| `reason` | Full pipeline with auto strategy selection |
| `self_ask` | Iterative sub-question decomposition |
| `chain_of_thought` | Interleaved reasoning and retrieval |
| `evidence_attribution` | Answer with citations |
| `verify_claims` | Multi-source fact checking |

### Analysis Operations

| Operation | Description |
|-----------|-------------|
| `analyze_query` | Complexity, intent, and strategy recommendation |
| `extract_claims` | Extract factual claims from text |

### Session Operations

| Operation | Description |
|-----------|-------------|
| `get_session` | Retrieve session with reasoning history |
| `delete_session` | Delete session |

### Admin Operations

| Operation | Description |
|-----------|-------------|
| `get_stats` | Metrics and statistics |
| `set_pipeline_config` | Update pipeline steps |
| `reload_config` | Hot-reload configuration |

---

## Usage Examples

### Full Pipeline with Auto-Selection

```python
result = await reasoning.reason(
    query="How did the invention of the printing press impact literacy rates in Europe?",
)

print(f"Strategy used: {result['strategy_used']}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Reasoning steps: {len(result['reasoning_steps'])}")
```

### Self-Ask for Complex Queries

```python
result = await reasoning.self_ask(
    query="What are the key differences between REST and GraphQL APIs, and when should I use each?",
    max_iterations=4,
)

print("Sub-questions explored:")
for sq in result['sub_questions']:
    print(f"  Q: {sq['question']}")
    print(f"  A: {sq['answer'][:100]}...")
```

### Evidence Attribution for Research

```python
result = await reasoning.evidence_attribution(
    query="What are the proven benefits of intermittent fasting?",
)

print(f"Answer: {result['answer']}")
print("\nCitations:")
for cit in result['citations']:
    print(f"  [{cit['citation_id']}] {cit['source_title']}")
```

### Claim Verification

```python
result = await reasoning.verify_claims(
    query="Verify health claims",
    claims=[
        "Drinking 8 glasses of water daily is necessary",
        "Vitamin C prevents colds",
        "Sugar causes hyperactivity in children",
    ],
)

for v in result['verifications']:
    print(f"{v['claim_text']}: {v['status']} (confidence: {v['confidence']:.2f})")

if result['contradictions']:
    print("\nContradictions found:")
    for c in result['contradictions']:
        print(f"  {c['claim']}: {c['severity']} severity")
```

### Query Analysis

```python
analysis = await reasoning.analyze_query(
    query="Why do some programming languages perform better than others for machine learning?"
)

print(f"Complexity: {analysis['complexity']}")
print(f"Intent: {analysis['intent']}")
print(f"Recommended strategy: {analysis['recommended_strategy']}")
print(f"Reason: {analysis['strategy_reason']}")
print(f"Estimated steps: {analysis['estimated_steps']}")
```

---

## Configuration

### Key Environment Variables

```bash
# Core Settings
UBP_REASONING__ENABLED=true
UBP_REASONING__DEFAULT_STRATEGY=auto
UBP_REASONING__MAX_DEPTH=5
UBP_REASONING__TEMPERATURE=0.3
UBP_REASONING__TIMEOUT_SECONDS=60

# Self-Ask Settings
UBP_REASONING__SELF_ASK_MAX_ITER=5
UBP_REASONING__SELF_ASK_MAX_SUB_Q=3
UBP_REASONING__SELF_ASK_CONVERGENCE=0.85
UBP_REASONING__SELF_ASK_CONFIDENCE=0.8

# Chain-of-Thought Settings
UBP_REASONING__COT_MAX_STEPS=8
UBP_REASONING__COT_INTERLEAVE_MODE=adaptive
UBP_REASONING__COT_AUTO_RETRIEVAL_THRESHOLD=0.6

# Evidence Attribution
UBP_REASONING__EVIDENCE_ENABLED=true
UBP_REASONING__EVIDENCE_MIN_CONFIDENCE=0.5
UBP_REASONING__EVIDENCE_CITATION_FORMAT=inline

# Verification
UBP_REASONING__VERIFICATION_ENABLED=true
UBP_REASONING__VERIFICATION_MIN_SOURCES=2
UBP_REASONING__VERIFICATION_CONTRADICTION=true

# Retrieval
UBP_REASONING__RETRIEVAL_MODULE=retrieval_strategy
UBP_REASONING__RETRIEVAL_TOP_K=5
UBP_REASONING__RETRIEVAL_RERANK=true

# LLM Delegation
UBP_REASONING__LLM_MODULE=inference_ollama_grok
UBP_REASONING__LLM_TIMEOUT=30

# Cache
UBP_REASONING__CACHE_ENABLED=true
UBP_REASONING__CACHE_TTL=3600

# Session
UBP_REASONING__SESSION_ENABLED=true
UBP_REASONING__SESSION_MAX_HISTORY=50

# Debug
UBP_REASONING__DEBUG_ENABLED=false
UBP_REASONING__DEBUG_LOG_STEPS=true
```

---

## Pipeline Steps

The reasoning pipeline executes these configurable steps:

```
1. analyze_query     → Detect complexity and intent
2. select_strategy   → Choose optimal strategy
3. execute_reasoning → Run selected strategy
4. gather_evidence   → Collect additional evidence
5. verify_claims     → Multi-source verification
6. synthesize_answer → Final synthesis
7. format_output     → Prepare result
```

Each step can be enabled/disabled:

```json
{
  "pipeline": {
    "steps": {
      "analyze_query": { "enabled": true, "timeout": 5 },
      "verify_claims": { "enabled": false }
    }
  }
}
```

---

## Reasoning Trace

Every reasoning operation produces a complete trace:

```python
trace = result['reasoning_trace']

print(f"Trace ID: {trace['trace_id']}")
print(f"Strategy: {trace['strategy']}")
print(f"Total duration: {trace['total_duration_ms']}ms")

for entry in trace['entries']:
    print(f"  [{entry['entry_type']}] {entry['timestamp']}")
    print(f"    Duration: {entry['duration_ms']}ms")
    print(f"    Content: {entry['content']}")
```

---

## Metrics

Available via `get_stats()`:

```python
stats = await reasoning.get_stats(period="24h")

# {
#   "metrics": {
#     "total_queries": 1234,
#     "strategy_distribution": {
#       "self_ask": 300,
#       "chain_of_thought": 500,
#       "evidence_attribution": 200,
#       "verification": 100,
#       "direct": 134
#     },
#     "iterations": {"avg": 2.5, "max": 5},
#     "steps": {"avg": 4.2, "max": 8},
#     "confidence": {"avg": 0.78, "min": 0.45, "max": 0.98},
#     "verification": {
#       "verified": 450,
#       "partially": 120,
#       "unverified": 80,
#       "contradicted": 50
#     },
#     "execution_times": {"avg_ms": 2500, "min_ms": 800, "max_ms": 8000}
#   }
# }
```

---

## Cross-Lingual Support

The module supports English and Italian:

```python
# Auto-detection
result = await reasoning.reason(
    query="Perché il cielo è blu?",  # Italian detected
    language="auto",
)

# All prompts and responses in Italian
```

---

## Integration with Other Modules

### Retrieval Module

```python
# Configure retrieval backend
{
  "retrieval": {
    "module": "retrieval_strategy",
    "operation": "retrieve",
    "default_top_k": 5
  }
}
```

### LLM Module

```python
# Configure LLM backend
{
  "delegation": {
    "llm_module": "inference_vllm",
    "llm_operation": "generate"
  }
}
```

---

## Deployment

### 1. Copy Module

```bash
cp -r reasoning_rag/ modules/cores/reasoning_rag/
```

### 2. Configure Environment

```bash
# .env
UBP_ENV=prod
UBP_REASONING__ENABLED=true
UBP_REASONING__LLM_MODULE=inference_vllm
UBP_REASONING__RETRIEVAL_MODULE=retrieval_strategy
```

### 3. Initialize

```python
from modules.cores.reasoning_rag import create_module

module = create_module(
    module_path=Path("modules/cores/reasoning_rag"),
    di_container=container,
    event_bus=event_bus,
)

await module.initialize()
```

### 4. Health Check

```python
health = await module.health_check()
# {"status": "healthy", "strategies": {...}}
```

---

## Troubleshooting

### Low Confidence Scores

- Increase retrieval `top_k`
- Enable verification step
- Use `self_ask` for complex queries

### Slow Response Times

- Reduce `max_iterations` or `max_steps`
- Enable caching
- Disable unnecessary pipeline steps

### Missing Citations

- Ensure `evidence_attribution` is enabled
- Check retrieval module is available
- Lower `min_confidence` threshold

### Contradictions Not Detected

- Increase `min_sources_for_verification`
- Lower `contradiction_threshold`
- Enable `multi_source_check`

---

## Dependencies

### Required
- Python 3.10+
- Redis (for caching and sessions)

### Optional
- `retrieval_strategy` - Document retrieval
- `inference_vllm` / `inference_ollama_grok` - LLM backend

---

## License

Internal use - UBP Enterprise Hybrid

---

## Changelog

### v1.0.0 (2025-01)
- Initial release
- Self-Ask RAG strategy
- Chain-of-Thought RAG strategy
- Evidence Attribution strategy
- Verification strategy
- Automatic strategy selection
- Query complexity analysis
- Cross-lingual EN/IT
- Complete reasoning trace
- Redis caching
- Session management
- Comprehensive metrics
