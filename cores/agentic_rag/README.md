# Agentic RAG

**Autonomous RAG Engine** with tool use, planning, and parallel execution

Version: 1.0.0 | Architecture: 3-file-pattern | Module Type: core

---

## Overview

`agentic_rag` implements an autonomous agent for RAG that can:
- Reason step-by-step using ReAct loops
- Plan and decompose complex queries
- Execute tasks in parallel
- Use multiple tools to gather information
- Self-correct and replan on failures

---

## Architecture

```
agentic_rag/
├── __init__.py          # Module factory for ModuleLoader
├── adapter.py           # Bridge layer - exposes 15+ operations
├── providers.py         # Core data classes, state, memory
├── tools.py             # Tool registry and execution
├── planner.py           # Query decomposition and planning
├── executor.py          # Parallel execution engine
├── prompts.py           # ReAct prompt templates (EN/IT)
├── config.json          # 200+ environment variables
├── manifest.json        # Operation definitions
└── README.md            # This file
```

---

## Execution Modes

### 1️⃣ ReAct Mode (Thought-Action-Observation)

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Thought   │──────────────────────┐
│ "I need to  │                      │
│  search..." │                      │
└──────┬──────┘                      │
       │                             │
       ▼                             │
┌─────────────┐                      │
│   Action    │                      │
│ retrieval() │                      │
└──────┬──────┘                      │
       │                             │
       ▼                             │
┌─────────────┐                      │
│ Observation │                      │
│ "Found: ..." │                     │
└──────┬──────┘                      │
       │                             │
       └────────────────┐            │
                        ▼            │
                  ┌───────────┐      │
                  │ Continue? ├──Yes─┘
                  └─────┬─────┘
                        │ No
                        ▼
                  ┌───────────┐
                  │  Answer   │
                  └───────────┘
```

### 2️⃣ Plan-Execute Mode

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│           Planner               │
│  - Decompose into tasks         │
│  - Identify dependencies        │
│  - Create parallel batches      │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│     Parallel Executor           │
│                                 │
│  Batch 1: [Task1, Task2, Task3] │  ← Execute in parallel
│              ↓                  │
│  Batch 2: [Task4]               │  ← Depends on Batch 1
│              ↓                  │
│  Batch 3: [Task5]               │  ← Synthesis
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────┐
│   Answer    │
└─────────────┘
```

### 3️⃣ Parallel Mode (Maximum Concurrency)

Forces parallel execution for all independent tasks.

---

## Parallel Execution Engine

### Worker Pool

```python
# Configurable worker pool
worker_pool_size: 8        # Total workers
max_concurrent: 5          # Max parallel tasks
batch_size: 3              # Tasks per batch
task_timeout: 30           # Per-task timeout
```

### Dependency-Aware Scheduling

```python
# Tasks are scheduled based on dependencies
Task A: []           # No deps → Batch 1
Task B: []           # No deps → Batch 1
Task C: [A]          # Depends on A → Batch 2
Task D: [A, B]       # Depends on A,B → Batch 2
Task E: [C, D]       # Depends on C,D → Batch 3

# Execution:
# Batch 1: [A, B]    ← parallel
# Batch 2: [C, D]    ← parallel (after Batch 1)
# Batch 3: [E]       ← after Batch 2
```

### Configuration

```bash
# Parallel execution config
UBP_AGENTIC__PARALLEL_ENABLED=true
UBP_AGENTIC__PARALLEL_MAX_CONCURRENT=5
UBP_AGENTIC__WORKER_POOL_SIZE=8
UBP_AGENTIC__PARALLEL_BATCH_SIZE=3
UBP_AGENTIC__DEPENDENCY_AWARE=true
UBP_AGENTIC__FAIL_FAST=false
UBP_AGENTIC__RETRY_FAILED=true
UBP_AGENTIC__MAX_RETRIES=2
```

---

## Built-in Tools

| Tool | Description | Category |
|------|-------------|----------|
| `retrieval` | Search knowledge base | retrieval |
| `calculator` | Math operations | computation |
| `summarizer` | Text summarization | text_processing |
| `graph_query` | Knowledge graph queries | knowledge_graph |
| `web_search` | Web search (optional) | web |

### Tool Schema

```python
@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: List[ToolParameter]
    returns: str
    category: str
    timeout_seconds: int = 30
    max_retries: int = 2
```

### Register Custom Tool

```python
await agent.register_tool(
    name="my_tool",
    description="Does something useful",
    parameters=[
        {"name": "input", "type": "string", "required": True},
    ],
    handler_module="my_module",
    handler_operation="my_operation",
)
```

---

## Usage Examples

### Basic Query (ReAct)

```python
result = await agent.query(
    query="What is the capital of France and its population?",
    mode="react",
    max_iterations=5,
)

print(result["answer"])
# "Paris is the capital of France with a population of approximately 2.1 million..."

print(result["total_iterations"])
# 3

print(result["total_tool_calls"])
# 2
```

### Parallel Query

```python
result = await agent.parallel_query(
    query="Compare GDP growth of USA, China, and Germany in 2023",
    max_concurrent=5,
)

print(result["metadata"]["parallel_batches"])
# 2 (retrieval in parallel, then synthesis)
```

### Plan-Execute

```python
# Create plan first
plan = await agent.create_plan(
    query="Analyze the impact of AI on healthcare",
)

print(plan["execution_order"])
# [['task_1', 'task_2', 'task_3'], ['task_4'], ['task_5']]

# Execute plan
result = await agent.execute_plan(
    plan_dict=plan,
    enable_parallel=True,
)
```

### Direct Tool Call

```python
result = await agent.call_tool(
    tool_name="calculator",
    arguments={"expression": "sqrt(144) + 5^2"},
)

print(result["result"])
# 37.0
```

---

## ReAct Prompt Format

```
Question: What are the main features of Python 3.10?

Thought: I need to search for information about Python 3.10 features.
Action: retrieval
Action Input: {"query": "Python 3.10 new features"}

Observation: Python 3.10 introduced structural pattern matching (match/case), 
better error messages, parenthesized context managers...

Thought: I have enough information to answer the question.
Final Answer: Python 3.10 introduced several key features:
1. Structural Pattern Matching (match/case statements)
2. Better error messages with precise locations
3. Parenthesized context managers
...
```

---

## State Management

### Agent State

```python
@dataclass
class AgentState:
    state_id: str
    session_id: str
    query: str
    mode: ExecutionMode
    plan: Optional[AgentPlan]
    steps: List[AgentStep]
    current_iteration: int
    status: TaskStatus
    final_answer: Optional[str]
    working_memory: Dict[str, Any]
```

### Memory Types

1. **Working Memory**: Short-term, decaying relevance
2. **Episodic Memory**: Long-term successful patterns

```python
# Working memory - recent context
self._working_memory.add("search_result_1", data, relevance=0.9)

# Episodic memory - past successes
self._episodic_memory.add_episode(
    query="...",
    answer="...",
    success=True,
    tool_sequence=["retrieval", "calculator", "summarizer"],
)
```

---

## Operations Reference

### Core Operations

| Operation | Description |
|-----------|-------------|
| `query` | Execute agentic query with any mode |
| `react` | ReAct-style reasoning |
| `plan_execute` | Plan then execute |
| `parallel_query` | Maximum parallelism |

### Planning Operations

| Operation | Description |
|-----------|-------------|
| `create_plan` | Create execution plan |
| `execute_plan` | Execute existing plan |

### Tool Operations

| Operation | Description |
|-----------|-------------|
| `call_tool` | Direct tool invocation |
| `register_tool` | Register external tool |
| `list_tools` | Get available tools |

### Admin Operations

| Operation | Description |
|-----------|-------------|
| `get_state` | Get session state |
| `get_stats` | Get metrics |
| `reload_config` | Hot-reload |
| `health_check` | Component health |

---

## Configuration

### Key Environment Variables

```bash
# Core
UBP_AGENTIC__DEFAULT_MODE=react
UBP_AGENTIC__MAX_ITERATIONS=10
UBP_AGENTIC__TIMEOUT=120

# ReAct Loop
UBP_AGENTIC__REACT_MAX_ITER=8
UBP_AGENTIC__REFLECTION_ENABLED=true
UBP_AGENTIC__EARLY_STOP_CONFIDENCE=0.9

# Planning
UBP_AGENTIC__PLANNING_ENABLED=true
UBP_AGENTIC__MAX_SUBTASKS=8
UBP_AGENTIC__REPLAN_ON_FAILURE=true

# Parallel Execution
UBP_AGENTIC__PARALLEL_ENABLED=true
UBP_AGENTIC__PARALLEL_MAX_CONCURRENT=5
UBP_AGENTIC__WORKER_POOL_SIZE=8
UBP_AGENTIC__FAIL_FAST=false

# Tools
UBP_AGENTIC__TOOL_RETRIEVAL=true
UBP_AGENTIC__TOOL_CALCULATOR=true
UBP_AGENTIC__TOOL_SUMMARIZER=true
UBP_AGENTIC__TOOL_GRAPH=true

# Safety
UBP_AGENTIC__MAX_TOOL_CALLS=20
UBP_AGENTIC__RATE_LIMIT_QPM=30
```

---

## Metrics

```python
stats = await agent.get_stats()

# Example output:
{
    "metrics": {
        "total_executions": 150,
        "success_rate": 0.94,
        "mode_distribution": {
            "react": 100,
            "plan_execute": 40,
            "parallel": 10
        },
        "latency_stats": {
            "avg_ms": 2500,
            "min_ms": 500,
            "max_ms": 8000
        },
        "iteration_stats": {
            "avg": 3.2,
            "max": 8
        },
        "avg_parallel_batches": 2.1
    },
    "executor": {
        "worker_pool": {
            "pool_size": 8,
            "active_workers": 0,
            "completed_tasks": 450
        }
    }
}
```

---

## Error Handling

### Retry Strategy

```python
# Exponential backoff
retry_strategy: exponential
base_delay_ms: 500
max_delay_ms: 5000
max_retries: 3
```

### Fail-Fast vs Continue-On-Error

```python
# Fail-fast: stop on first error
fail_fast: true

# Continue-on-error: complete what we can
fail_fast: false
continue_on_error: true
```

### Fallback

```python
# If agent fails completely
fallback_enabled: true
fallback_strategy: simple_retrieval
```

---

## Integration Examples

### With retrieval_strategy

```python
# Agentic uses retrieval_strategy as a tool
result = await agent.call_tool(
    tool_name="retrieval",
    arguments={
        "query": "machine learning basics",
        "strategy": "hybrid",
        "top_k": 5,
    },
)
```

### With graph_rag

```python
# Use graph_query tool
result = await agent.call_tool(
    tool_name="graph_query",
    arguments={
        "query": "Find connections between Python and Machine Learning",
        "max_hops": 2,
    },
)
```

---

## Deployment

### 1. Copy Module

```bash
cp -r agentic_rag/ modules/cores/agentic_rag/
```

### 2. Configure

```bash
# .env
UBP_AGENTIC__PARALLEL_ENABLED=true
UBP_AGENTIC__MAX_CONCURRENT=5
```

### 3. Initialize

```python
from modules.cores.agentic_rag import create_module

agent = create_module(
    module_path=Path("modules/cores/agentic_rag"),
    di_container=container,
    event_bus=event_bus,
)

await agent.initialize()
```

---

## Best Practices

### 1. Choose Right Mode

| Query Type | Recommended Mode |
|------------|------------------|
| Simple factual | `react` (few iterations) |
| Multi-part questions | `parallel_query` |
| Complex analysis | `plan_execute` |
| Exploratory | `react` |

### 2. Tune Parallelism

```python
# For independent sub-queries
max_concurrent=5
batch_size=5

# For dependent tasks
dependency_aware=True
batch_size=2
```

### 3. Monitor Iterations

```python
# Set reasonable limits
max_iterations=8
early_stop_confidence=0.9
```

---

## Troubleshooting

### Agent Loops Forever
- Reduce `max_iterations`
- Enable `early_stopping`
- Check LLM response quality

### Parallel Tasks Failing
- Increase `task_timeout`
- Enable `retry_failed`
- Check tool availability

### Memory Issues
- Reduce `working_memory_max`
- Clear old sessions
- Limit `episodic_memory_max`

---

## Dependencies

### Required
- Python 3.10+

### Optional
- inference_ollama_grok (LLM for reasoning)
- retrieval_strategy (document retrieval)
- graph_rag (knowledge graph)

---

## Changelog

### v1.0.0 (2025-01)
- Initial release
- ReAct reasoning loops
- Plan-then-execute mode
- Parallel execution with worker pool
- Dependency-aware scheduling
- 5 built-in tools
- State and memory management
- Cross-lingual support (EN/IT)
- Comprehensive metrics
