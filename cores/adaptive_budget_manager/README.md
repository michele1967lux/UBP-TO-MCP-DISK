# Adaptive Budget Manager Module

## Overview

**Modulo universale** per la gestione adattiva del budget di token in contesti di chat, principalmente (ma non esclusivamente) utilizzato nei pipeline RAG.

Contiene la classe **AdaptiveBudgetManager** che gestisce intelligentemente l'allocazione dei token.

## Features

- **Universale**: Funziona con qualsiasi tipo di chat (RAG, Pure LLM, conversazioni semplici)
- **Dynamic Memory Allocation**: Automatically allocates 20-40% of context window to conversation memory based on "tightness"
- **Tightness Factor**: Calculates context pressure based on token usage and conversation turns (0=ample, 1=very tight)
- **Similarity Threshold Scaling**: (Solo per RAG) Adjusts retrieval threshold from 0.4 to 0.7 based on available space
- **Context Compression**: Automatically compresses memory and documents when context is tight
- **Model-Agnostic**: Adapts to different context windows (Ollama 4096, Grok 131072, etc.)

## Use Cases

### 1. RAG Pipeline (Caso Principale)
- Gestione dinamica del budget tra documenti recuperati e memoria conversazione
- Scaling del similarity threshold basato su spazio disponibile
- Compressione intelligente quando necessario

### 2. Pure LLM Chat (Senza RAG)
- Gestione del budget conversazionale
- Compressione automatica della history quando il context window è pieno
- Prioritizzazione dei messaggi recenti

### 3. Multi-Turn Conversations
- Adattamento progressivo al crescere della conversazione
- Decay intelligente dei messaggi vecchi
- Mantenimento del contesto rilevante

## How It Works

### Tightness Calculation

```
tightness = (current_usage / total_window) + (turn_count * penalty_factor)
```

- **Low tightness (0.0-0.3)**: Ample space - expand context
- **Medium tightness (0.3-0.7)**: Moderate - balance memory and docs
- **High tightness (0.7-1.0)**: Very tight - compress and filter aggressively

### Memory Allocation

```
memory_fraction = min_fraction + (tightness * (max_fraction - min_fraction))
memory_tokens = total_window * memory_fraction
```

Example:
- **Ollama (4096 tokens, turn 5)**: ~35% memory = 1434 tokens
- **Grok (131072 tokens, turn 1)**: ~20% memory = 26214 tokens

### Similarity Threshold Scaling

```
threshold = base_score + (tightness * (max_score - base_score))
```

Example:
- **Low tightness**: 0.4 (accept more docs)
- **High tightness**: 0.7 (only highly relevant docs)

## Configuration

```json
{
  "enabled": true,
  "base_min_score": 0.4,
  "max_threshold": 0.7,
  "min_memory_fraction": 0.2,
  "max_memory_fraction": 0.4,
  "turn_penalty_factor": 0.05,
  "compression_enabled": true,
  "compression_threshold": 0.5,
  "support_llm_provider": "ollama",
  "support_llm_model": "llama3.2"
}
```

### Parameters

- `base_min_score`: Base similarity threshold (default: 0.4)
- `max_threshold`: Maximum threshold when tight (default: 0.7)
- `min_memory_fraction`: Minimum memory allocation (default: 20%)
- `max_memory_fraction`: Maximum memory allocation (default: 40%)
- `turn_penalty_factor`: Penalty per turn (default: 0.05)
- `compression_enabled`: Enable automatic compression (default: true)
- `compression_threshold`: Tightness to trigger compression (default: 0.5)
- `support_llm_provider`: Provider for summarization (default: "ollama")
- `support_llm_model`: Model for summarization (default: "llama3.2")

## Usage

### As a Module

```python
from modules.cores.adaptive_budget_manager import AdaptiveBudgetManager

# Initialize
manager = AdaptiveBudgetManager(config)

# Adjust budget for RAG
result = await manager.adjust(
    query="User question",
    conversation_context="Previous chat history...",
    retrieved_docs=[...],
    turn_count=5,
    rag_config={...}
)

# Adjust budget for Pure Chat
result = await manager.adjust_for_pure_chat(
    query="User question",
    conversation_context="Previous chat history...",
    turn_count=5,
    chat_config={...}
)

# Access results
filtered_docs = result["filtered_docs"]
memory_tokens = result["memory_tokens"]
tightness = result["tightness"]
```

### Via API

```bash
curl -X POST http://localhost:8000/api/modules/adaptive_budget_manager/adjust_budget \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the status?",
    "retrieved_docs": [...],
    "turn_count": 3,
    "config": {...}
  }'
```

## Integration with RAG Orchestrator

The module integrates into the RAG pipeline between retrieval and augmentation:

```
Retrieve → [ADAPTIVE BUDGET] → Augment → Generate
```

Adaptive budget adjusts:
1. **Memory allocation**: Compresses conversation context if needed
2. **Document filtering**: Scales similarity threshold
3. **Document budget**: Calculates remaining space for docs

## Examples

### Example 1: Small Window, High Turns (Ollama, Turn 10)

```
Context Window: 4096 tokens
Turn Count: 10
Tightness: 0.8 (very tight)

Result:
- Memory: 38% = 1556 tokens (compressed)
- Threshold: 0.64 (strict filtering)
- Doc Budget: ~1200 tokens (only 2-3 relevant docs)
```

### Example 2: Large Window, Low Turns (Grok, Turn 1)

```
Context Window: 131072 tokens
Turn Count: 1
Tightness: 0.15 (ample)

Result:
- Memory: 23% = 30147 tokens (full history)
- Threshold: 0.45 (relaxed filtering)
- Doc Budget: ~80000 tokens (many docs)
```

## Benefits

- **Automatic**: No manual tuning required
- **Intelligent**: Adapts based on actual usage
- **Resilient**: Handles different model sizes
- **Efficient**: Compresses only when necessary
- **Quality**: Prioritizes memory continuity

## Dependencies

- `modules.cores._shared.token_limits`: Token counting and validation
- Pydantic: Configuration validation
- Optional: LLM adapter for summarization

## Module Name

Il modulo si chiama **adaptive_budget_manager** in coerenza con la classe principale **AdaptiveBudgetManager**.

## Version History

- **v1.0.0**: Initial implementation with tightness calculation, memory allocation, threshold scaling, and compression
