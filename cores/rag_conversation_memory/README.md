# RAG Conversation Memory Module v5.0.0

Redis-based conversation memory with **Thread-Based Structured Summary**, Smart Promote, Gentle Decay, and **Memory-Aware Query Rewriting** for multi-turn RAG sessions in UBP Enterprise Hybrid.

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 5.0.0 | 2026-02-08 | FEAT-MEM-003: Memory-aware query rewriting — zero-latency pre-computed hints, QueryRewriter + HintsBuilder, `rewrite_query` operation, `SESSION_HINTS_KEY` lifecycle |
| 4.2.0 | 2026-02-02 | Thread-based structured summary, importance-weighted fading, smart merge/promote, dual current (HOLD), explicit reset detection, archival |
| 4.1.0 | 2026-01-31 | Eager compression after every assistant reply, pre-cached context, summary-first context injection, dynamic response_budget_ratio |
| 2.0.0 | 2026-01-28 | Structured Memory: topic detection, LLM compression, decay system, event-driven async |
| 1.0.0 | 2025-12-31 | Initial release with ROADMAP v1.5.0 (simple raw message buffer) |

---

## Architecture Overview

### 3-File Pattern

```
rag_conversation_memory/
├── __init__.py          # Entry point with create_module() factory
├── adapter.py           # ConversationMemoryAdapter (UBP integration layer)
├── providers.py         # RedisConversationMemoryProvider (Redis persistence)
├── models.py            # Data models: ConversationTurn, MemoryState, ContextResult
├── context_manager.py   # Compression engine: fading, merge, promote, hold
├── query_rewriter.py    # v5.0: QueryRewriter + HintsBuilder (zero-latency query expansion)
├── config.json          # Module configuration
├── manifest.json        # Module metadata and operations
└── README.md            # This file
```

### Core Principle (v4.2.0)

> **LLM does content extraction, code does lifecycle.**

- The LLM extracts `focus`, `key_facts`, `importance`, and `entities` from new messages.
- Deterministic Python code handles fading, merging, promoting, holding, truncating, and archiving.

---

## Features

### v5.0.0 — Memory-Aware Query Rewriting (FEAT-MEM-003)

| Feature | Description |
|---------|-------------|
| **Pre-Computed Retrieval Hints** | After each eager compression in `_on_message_added`, `HintsBuilder.build_from_state()` extracts keywords/entities and builds 3 pre-expanded query variants (continuation, deepdive, primary). Cached in Redis. |
| **Zero-Latency Query Rewriting** | `QueryRewriter` does a Redis GET (~1ms), classifies the query with regex, returns expanded query. No LLM call at query-time. |
| **Query Classification** | 6 types: `none` (specific query, no rewrite), `continuation` ("spiega meglio"), `deepdive` ("approfondisci"), `referential` (pronouns), `enriched` (short but on-topic), `fallback` (no hints available). |
| **retrieval_query Separation** | Rewritten query used only for Qdrant retrieval. Original `query` preserved for memory saves, LLM prompt (`_original_user_query`), and compression. |
| **Prompt Separation** | Memory context gets explicit `=== CONTESTO CONVERSAZIONE PRECEDENTE ===` headers with "NON trattare come documentazione" instruction, preventing LLM from treating memory as documentation. |
| **Debug Panel** | `QueryRewriteDebugPanel` in both ArchitectTab.tsx and UserChatTab.tsx shows rewrite type, original vs expanded query, hints metadata. |

#### Query Rewrite Flow

```
User sends "spiega meglio"
    ↓
rag_orchestrator.rag_chat() / ask_architect()
    ↓
memory_module.rewrite_query(session_id, query)
    ↓
QueryRewriter.rewrite():
    1. GET Redis hints (~1ms)
    2. Classify query (continuation/deepdive/referential/specific/...)
    3. Return expanded query from pre-cached variants
    ↓
retrieval_query = expanded query (for Qdrant/pipeline)
query = original "spiega meglio" (for memory saves, LLM prompt)
```

#### Hint Pre-Computation Flow

```
_on_message_added() (after eager compression)
    ↓
HintsBuilder.build_from_state(new_state)
    ↓
Extracts: current_focus, key_facts, entities, last_query
    ↓
Builds 3 variants:
    - continuation: focus + key_facts keywords
    - deepdive: focus + "approfondimento" + entities
    - primary: focus + last_query keywords
    ↓
QueryRewriter.cache_hints(session_id, hints, ttl)
    ↓
Redis SET with session TTL (90d)
```

### v4.2.0 — Thread-Based Structured Summary

| Feature | Description |
|---------|-------------|
| **Conversation Thread** | Chronologically ordered list of `ConversationTurn` entries, each representing a sub-topic |
| **Importance-Weighted Fading** | Topics with higher importance fade slower: `turns_absent_eff = turns_absent / (1 + importance / 5)` |
| **Smart Merge (Adjacent)** | Consecutive turns on the same topic are merged into one entry (`merge_count` increments) |
| **Smart Merge (Non-Adjacent)** | Same topic within `soft_merge_window` turns (default 4) gets merged without creating a new entry |
| **Smart Promote** | Old topic beyond merge window gets promoted: detail level restored, `reactivation_boost` applied, anchor sentence generated |
| **Dual Current (HOLD)** | Two active topic pointers: `[CURRENT]` for the active topic, `[HOLD]` for the previously active topic (supports topic ping-pong) |
| **Explicit Reset Detection** | IT/EN patterns ("cambiamo argomento", "new topic") create a new topic without promoting old ones |
| **Archival** | Turns faded beyond `background` level are moved to `archived_turns` for future rehydration |
| **Detail Levels** | 6 levels: `full` > `high` > `recent` > `fading` > `background` > `archived` |

### v4.1.0 — Eager Compression & Pre-Caching

| Feature | Description |
|---------|-------------|
| **Eager Compression** | Triggers on every assistant reply (not just buffer overflow) via `memory.message_added` event |
| **Pre-Cached Context** | Formatted `system_message` is pre-computed and cached in Redis after each compression |
| **Summary-First Injection** | Structured context is injected before raw messages for optimal LLM attention |
| **Dynamic Last-Turn Detection** | Finds the last complete turn (user+assistant) dynamically instead of relying on buffer indices |

### Session Management (v1.0+)

- **Create Sessions**: Start new conversation sessions per user
- **List Sessions**: View all sessions for the current user (client-isolated)
- **Delete Sessions**: Remove sessions and all associated messages
- **Clear Sessions**: Remove messages but keep session metadata

### Security

- **User Isolation**: Sessions scoped to authenticated users
- **Client Isolation**: RULE-006/RULE-008 — cross-client access denied
- **Admin Access**: Same-client admin can access any session in their client
- **Auto-Create**: v4.1.0 — Sessions auto-created when `rag_orchestrator` sends messages before explicit session creation

---

## Data Model

### ConversationTurn (v4.2.0)

Each entry in the conversation thread:

```python
class ConversationTurn(BaseModel):
    turn_number: int        # Turn number (updated on merge/resume)
    focus: str              # Sub-topic label (e.g. "Redis override system")
    key_facts: str          # Dense facts, truncated by detail_level
    key_facts_full: str     # Original facts pre-truncation (restore on promote)
    detail_level: str       # full/high/recent/fading/background
    importance: int         # 0-10, assigned by LLM, modifies fading speed
    query: str              # Original user query
    reactivation_boost: int # Resistance to fading (residual turns)
    anchor_sentence: str    # Bridge sentence for reactivated topics
    is_resumed: bool        # True if topic was reactivated
    merge_count: int        # How many turns merged into this entry
    timestamp: datetime     # When this entry was created/updated
```

### MemoryState

Complete memory state stored in Redis per session:

```python
class MemoryState(BaseModel):
    # Versioning & metadata
    version: int
    created_at: datetime
    last_updated: datetime
    token_count: int
    turn_counter: int
    compression_history: List[Dict]

    # v4.2.0: Thread-based memory
    conversation_thread: List[ConversationTurn]   # Chronological thread
    current_focus: Optional[str]                   # Active topic pointer
    hold_focus: Optional[str]                      # Paused topic (ping-pong)
    hold_since_turn: int                           # When hold became active
    topic_flow: List[Dict]                         # Topic progression log
    topic_progression: str                         # Flat string for LLM
    archived_turns: List[Dict]                     # Archived for rehydration

    # Legacy (backward compat, derived from thread)
    narrative_summary: str                         # Concatenated key_facts
    structured_context: StructuredContext           # current_topic, intent, entities
    previous_topics: List[Topic]                   # Derived from thread
```

### ContextResult

Returned by `get_structured_context` and consumed by the RAG orchestrator:

```python
class ContextResult(BaseModel):
    raw_messages: List[Dict]
    narrative_summary: str
    structured_context: Optional[StructuredContext]
    previous_topics: List[Topic]
    has_structured_context: bool
    topic_shifting: bool
    # v4.2.0
    conversation_thread: List[ConversationTurn]
    current_focus: Optional[str]
    hold_focus: Optional[str]
    topic_progression: str
```

---

## Detail Levels & Fading

### Fading Table

| Turns Absent (effective) | Detail Level | Max key_facts chars | Boost to Resist |
|--------------------------|-------------|--------------------|--------------------|
| 0 | `full` | 500 | -- |
| 1 | `high` | 300 | -- |
| 2 | `recent` | 200 | -- |
| 3-5 | `fading` | 80 | boost >= 2 |
| 6-10 | `background` | 30 | boost >= 4 |
| 11+ | `archived` | 0 (moved to archive) | boost >= 6 |

### Importance-Weighted Decay

```
turns_absent_effective = turns_absent / (1 + importance / 5.0)
```

Examples:
- `importance=8`: effective fading is ~3.6x slower
- `importance=4` (minimum floor): effective fading is ~1.8x slower
- `importance=2`: almost linear fading

### Importance Hardening

- **Floor**: LLM importance values are clamped to a minimum of 4 (never below)
- **Critical keyword boost**: If focus or query contains security-related keywords (`auth`, `security`, `error`, `crash`, `bug`, `password`, `token`, `permission`, `vulnerability`), importance is boosted to at least 7

---

## Merge / Promote Logic

4-step decision tree executed on every new turn:

### Step 1: Adjacent Merge

If the last turn in the thread has the same focus AND is within 2 turns:
- Merge `key_facts` with `|` separator
- Increment `merge_count`
- Reset `detail_level` to `full`
- Apply `reactivation_boost = max(current, 2)`

### Step 2: Soft Merge (Non-Adjacent, Within Window)

If a matching focus exists in the thread within `soft_merge_window` turns (default 4):
- Same as adjacent merge, but for non-consecutive same-topic entries
- Prevents fragmentation of oscillating topics

### Step 3: Smart Promote (Beyond Window)

If a matching focus exists but beyond the merge window:
- Merge `key_facts_full` with new facts
- Promote `detail_level` one step up (e.g. `fading` -> `recent`)
- Set `is_resumed = True`
- Set `anchor_sentence` (generated by LLM as a bridge to old context)
- Apply `reactivation_boost = 5` (configurable)
- Update `turn_number` to current

### Step 4: New Topic

If no match found:
- Create new `ConversationTurn` entry
- Append to thread

### Explicit Reset

If query matches reset patterns (IT: "cambiamo argomento", "dimentica", "parliamo d'altro"; EN: "new topic", "forget", "start over"):
- Skip all merge/promote logic
- Create new topic entry directly
- No anchor sentence generated

---

## Dual Current: CURRENT + HOLD

Two simultaneous topic pointers for topic ping-pong:

| Pointer | Description |
|---------|-------------|
| `current_focus` | The topic actively being discussed |
| `hold_focus` | The previously active topic (max `hold_max_turns` before expiry) |

### Hold Behavior

1. **Topic changes**: old `current_focus` moves to `hold_focus`
2. **Return to hold**: if user returns to `hold_focus`, instant swap (no merge overhead)
3. **Hold expiry**: after `hold_max_turns` (default 3), `hold_focus` is cleared and the topic decays normally
4. **Explicit reset**: hold is not set (clean break)

---

## Output Format (system_message)

The structured context injected into the LLM system prompt:

```
=== CONVERSATION MEMORY ===
[CURRENT] Redis override mechanism -- Redis override, dynamic, SettingsManager singleton, OverrideCache, admin-only API
[HOLD] Docker containers -- docker-compose profiles, infra/backend/ai, Nginx gateway
[RECENT] Settings propagation -- .env SSOT, Redis > ENV > defaults, hot-reload
[FADING] Core modules -- 33+ modules, 3-file pattern
[BACKGROUND] UBP overview -- Enterprise hybrid RAG system

TOPIC FLOW: UBP overview -> core modules -> settings -> containers -> override system
ENTITIES: Docker, Nginx, OverrideCache, Qdrant, Redis, SettingsManager, vLLM
INTENT: progressive_deepdive
```

### With Smart Promote (resumed topic):

```
=== CONVERSATION MEMORY ===
[CURRENT] Core modules (resumed) -- 33+ modules, 3-file pattern, DI/event bus | event bus pub/sub, dead-letter
  ↳ Riprendendo i moduli core discussi prima: pipeline_orchestrator, rag_qdrant, enrichment_pipeline.
[HOLD] Override system -- Redis dynamic overrides, SettingsManager, OverrideCache
[FADING] Container startup -- docker-compose profiles
[BACKGROUND] UBP overview -- Enterprise hybrid RAG system

TOPIC FLOW: UBP overview -> core modules -> settings -> containers -> override -> core modules (resumed)
ENTITIES: Docker, EventBus, OverrideCache, Redis, SettingsManager
INTENT: progressive_deepdive
```

### With Topic Shift Detection:

When a topic shift occurs, the suffix `[TOPIC SHIFT DETECTED]` is appended to signal the RAG orchestrator.

---

## Event-Driven Architecture

### Event Flow

```
User sends message
    ↓
add_message() stores in Redis
    ↓
Publishes "memory.message_added" {session_id, role, message_id}
    ↓
_on_message_added() event handler (async)
    ↓ (only on role="assistant")
    ├── Get current MemoryState from Redis
    ├── Get message buffer from Redis
    ├── Dynamic last-turn detection (find last assistant + preceding user)
    ├── Call LLM for content extraction (focus, key_facts, importance, entities)
    ├── Apply deterministic lifecycle (fading → merge/promote → hold → trim)
    ├── Save new MemoryState to Redis
    ├── Trim buffer if > raw_buffer_size
    ├── Pre-cache formatted context
    ├── v5.0: Pre-compute retrieval hints (HintsBuilder → Redis cache)
    └── Publish "memory.topic_shifted" if topic changed
```

### Published Events

| Event | Payload | When |
|-------|---------|------|
| `memory.session_created` | `{session_id, user_id}` | New session created |
| `memory.message_added` | `{session_id, message_id, role}` | Message added |
| `memory.session_cleared` | `{session_id}` | Session messages cleared |
| `memory.session_deleted` | `{session_id}` | Session deleted |
| `memory.topic_shifted` | `{session_id, old_topic, new_topic}` | Topic shift detected |
| `memory.context_compressed` | `{session_id}` | Compression completed |

### Subscribed Events

| Event | Handler | Purpose |
|-------|---------|---------|
| `memory.message_added` | `_on_message_added()` | Trigger eager compression |
| `rag.chat.completed` | (legacy) | Auto-save RAG responses |
| `user.session.ended` | (legacy) | Cleanup on logout |

---

## Configuration

### Environment Variables (MemorySettings)

All configurable via `UBP_MEMORY__*` environment variables.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `UBP_MEMORY__STRUCTURED_ENABLED` | bool | `false` | **Master switch** for structured memory |
| `UBP_MEMORY__STRATEGY` | str | `token_budget` | Compression strategy: `token_budget`, `message_count`, `hybrid` |
| `UBP_MEMORY__RAW_BUFFER_SIZE` | int | `10` | Max raw messages kept in buffer |
| `UBP_MEMORY__MAX_CONTEXT_TOKENS` | int | `4000` | Max token budget for context |
| `UBP_MEMORY__COMPRESSION_THRESHOLD` | float | `0.8` | Trigger compression at this % of budget |
| `UBP_MEMORY__TOPIC_DECAY_TURNS` | int | `5` | Legacy decay counter (backward compat) |
| `UBP_MEMORY__MAX_PREVIOUS_TOPICS` | int | `3` | Max legacy previous_topics kept |
| `UBP_MEMORY__SUMMARY_MAX_TOKENS` | int | `1500` | Max tokens for narrative summary |
| `UBP_MEMORY__LLM_PROVIDER` | str | `vllm` | LLM provider alias for compression |
| `UBP_MEMORY__LLM_MODEL` | str | `""` | Specific model name (empty = provider default) |
| `UBP_MEMORY__LLM_TIMEOUT_SECONDS` | int | `30` | LLM call timeout |
| `UBP_MEMORY__MAX_THREAD_TURNS` | int | `20` | Max active turns in conversation thread |
| `UBP_MEMORY__MAX_ARCHIVED_TURNS` | int | `30` | Max archived turns for rehydration |
| `UBP_MEMORY__FADING_KEY_FACTS_CHARS` | int | `80` | Max chars for key_facts at fading level |
| `UBP_MEMORY__BACKGROUND_KEY_FACTS_CHARS` | int | `30` | Max chars at background level |
| `UBP_MEMORY__REACTIVATION_BOOST_TURNS` | int | `5` | Boost after smart promote |
| `UBP_MEMORY__HOLD_MAX_TURNS` | int | `3` | Turns before hold expires |
| `UBP_MEMORY__SOFT_MERGE_WINDOW` | int | `4` | Turn window for non-adjacent soft merge |

### config.json (Module Config)

```json
{
  "enabled": true,
  "session_ttl_seconds": 604800,
  "max_messages_per_session": 100,
  "auto_save_responses": true,
  "context_max_turns": 10
}
```

### Tuning Examples

```bash
# Disable HOLD (no dual current)
UBP_MEMORY__HOLD_MAX_TURNS=0

# Aggressive merge (7-turn window)
UBP_MEMORY__SOFT_MERGE_WINDOW=7

# Conservative merge (2-turn window)
UBP_MEMORY__SOFT_MERGE_WINDOW=2

# Longer boost persistence after promote
UBP_MEMORY__REACTIVATION_BOOST_TURNS=8

# Smaller thread (faster fading)
UBP_MEMORY__MAX_THREAD_TURNS=10

# Use Grok for compression instead of vLLM
UBP_MEMORY__LLM_PROVIDER=grok
UBP_MEMORY__LLM_MODEL=grok-4-fast-reasoning
```

---

## Redis Key Schema

Following NAMING_POLICY.md Section 7:

```
ubp:{env}:memory:session:{session_id}            # Session metadata (hash)
ubp:{env}:memory:session:{session_id}:messages   # Message list (sorted set)
ubp:{env}:memory:session:{session_id}:state      # MemoryState JSON (string)
ubp:{env}:memory:session:{session_id}:ctx_cache  # Pre-cached context (string)
ubp:{env}:memory:session:{session_id}:comp_lock  # Compression lock (string, TTL)
ubp:{env}:memory:session:{session_id}:retrieval_hints  # v5.0: Pre-computed query rewrite hints (string, JSON)
ubp:{env}:memory:user:{user_id}:sessions         # User's session index (sorted set)
```

---

## Operations

| Operation | Description | Auth Required |
|-----------|-------------|---------------|
| `create_session` | Create a new conversation session | Yes |
| `get_history` | Get messages for a session | Yes |
| `add_message` | Add a message to a session | Yes |
| `clear_session` | Clear messages (keep metadata) | Yes |
| `delete_session` | Delete session completely | Yes |
| `list_sessions` | List user's sessions | Yes |
| `get_context_for_llm` | Get formatted LLM context (v1.0 simple) | Yes |
| `get_structured_context` | Get structured context with thread (v2.0+) | Yes |
| `rewrite_query` | Rewrite vague/continuation query using pre-cached hints (v5.0) | Yes |
| `health_check` | Check module health | No |

---

## API Usage Examples

### Create Session

```bash
curl -X POST "http://localhost:8000/api/modules/rag_conversation_memory/create_session" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"topic": "Python Help"}}'
```

### Add Message

```bash
curl -X POST "http://localhost:8000/api/modules/rag_conversation_memory/add_message" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "uuid-here",
    "role": "user",
    "content": "Come funziona il sistema di override Redis?"
  }'
```

### Get Structured Context (v4.2.0)

```bash
curl -X POST "http://localhost:8000/api/modules/rag_conversation_memory/get_structured_context" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "uuid-here"}'
```

Response:

```json
{
  "session_id": "uuid-here",
  "has_structured_context": true,
  "from_cache": true,
  "system_message": "=== CONVERSATION MEMORY ===\n[CURRENT] Redis override mechanism -- ...\n[RECENT] Settings propagation -- ...\n\nTOPIC FLOW: UBP overview -> settings -> override\nENTITIES: Redis, SettingsManager, OverrideCache\nINTENT: question",
  "raw_messages": [...]
}
```

### Rewrite Query (v5.0.0)

```bash
curl -X POST "http://localhost:8000/api/modules/rag_conversation_memory/rewrite_query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "uuid-here", "query": "spiega meglio"}'
```

Response:

```json
{
  "query": "Redis override mechanism approfondimento: override, dynamic, SettingsManager",
  "original_query": "spiega meglio",
  "rewrite_type": "continuation",
  "hints_used": true,
  "metadata": {
    "current_focus": "Redis override mechanism",
    "variant_used": "continuation",
    "hints_age_seconds": 12.5
  },
  "request_id": "uuid"
}
```

Rewrite types: `none` (specific query), `continuation` ("spiega meglio"), `deepdive` ("approfondisci"), `referential` (pronouns), `enriched` (short on-topic), `fallback` (no hints), `error`.

### Get History

```bash
curl -X POST "http://localhost:8000/api/modules/rag_conversation_memory/get_history" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "uuid-here", "limit": 20}'
```

---

## Integration with RAG Orchestrator

The `rag_orchestrator` module integrates with conversation memory for context injection and query rewriting:

```python
# In rag_orchestrator pipeline:

# 0. v5.0: Memory-Aware Query Rewriting (before retrieval)
rewrite_result = None
retrieval_query = query  # default: no rewrite
if conversation_id and self.memory_module and hasattr(self.memory_module, 'rewrite_query'):
    rewrite_result = await self.memory_module.rewrite_query(
        session_id=conversation_id, query=query, ctx=ctx
    )
    if rewrite_result.get("rewrite_type") not in ("none", "error"):
        retrieval_query = rewrite_result["query"]

# 1. Get structured context (pre-cached)
ctx = await memory_module.get_structured_context(session_id)

# 2. Extract system_message for injection
if ctx.get("has_structured_context"):
    system_message = ctx.get("system_message", "")
    # Injected BEFORE raw messages in the LLM prompt

# 3. Use retrieval_query for Qdrant retrieval, original query for LLM prompt
#    Config key "_original_user_query" preserves original for _generate()
if retrieval_query != query:
    rag_config["_original_user_query"] = query
result = await pipeline.chat(query=retrieval_query, ...)

# 4. After generating response, add ORIGINAL query to memory (not rewritten)
await memory_module.add_message(session_id, "assistant", response)
```

### Query Rewrite Integration Points (v5.0)

The rewriter is integrated in **all routes** of `rag_chat()` and `ask_architect()`:

| Route | Uses `retrieval_query` for | Preserves `query` for |
|-------|---------------------------|-----------------------|
| CHAT | LLM query (harmless: no retrieval) | Memory saves |
| WEB | Web search query | Memory saves |
| RAG | Qdrant retrieval + reranking | Memory saves, LLM prompt via `_original_user_query` |
| ARCHITECT | Pipeline retrieval | Memory saves, LLM prompt via `_original_user_query` |

### Prompt Separation (v5.0)

Memory context is injected with explicit headers to prevent LLM confusion:

```
=== CONTESTO CONVERSAZIONE PRECEDENTE ===
Usa per continuita'. NON trattare come documentazione da analizzare.

[memory context here]

=== FINE CONTESTO CONVERSAZIONE ===

[RAG chunks from Qdrant here under "DOCUMENTAZIONE UFFICIALE RECUPERATA"]
```

---

## Compression Engine (context_manager.py)

### LLM Prompt

The compression prompt asks the LLM to extract:

```json
{
  "focus": "specific sub-topic label",
  "key_facts": "dense comma-separated facts",
  "importance": 5,
  "matched_existing_topic": null,
  "anchor_sentence": "",
  "entities": {},
  "intent": "question|request|information|exploration|progressive_deepdive",
  "confidence": 0.8
}
```

### Lifecycle Pipeline (per turn)

```
1. _apply_fading()           # Importance-weighted decay on all existing entries
2. _check_explicit_reset()   # IT/EN reset pattern detection
3. _handle_merge_or_promote()  # 4-step: adjacent merge → soft merge → promote → new
4. _manage_hold()            # Dual current pointer management
5. Trim thread               # Enforce max_thread_turns
6. Update topic_flow         # Append to progression log
7. Sync legacy fields        # narrative_summary, previous_topics
8. Build new MemoryState     # Assemble and return
```

### Fallback Compression

If LLM is unavailable:
- Uses first 50 chars of user message as `focus`
- Uses first 200 chars as `key_facts`
- Sets `importance = 5`
- Appends to thread with `detail_level = "full"`
- Concatenates to `narrative_summary`

### LLM Provider Resolution

- **Lazy resolution**: LLM provider is resolved on first compression, not at module init
- **Avoids race condition**: Memory module initializes before inference modules
- **Fallback chain**: Configured provider -> `inference_ollama_grok` -> fallback compression
- **Configurable**: `UBP_MEMORY__LLM_PROVIDER=grok` (recommended for cloud speed)

---

## Adapter Layer (adapter.py)

### Key Methods

| Method | Description |
|--------|-------------|
| `_on_message_added()` | Async event handler: eager compression on every assistant reply |
| `_build_and_cache_context()` | Pre-computes and caches formatted system_message in Redis |
| `get_structured_context()` | Returns cached or freshly-built context result |
| `_auto_create_session()` | v4.1.0: Auto-creates memory session for orchestrator-initiated conversations |
| `_init_structured_memory()` | Loads MemorySettings, creates ContextManager with lazy LLM |
| `rewrite_query()` | v5.0: Delegates to `QueryRewriter.rewrite()`, returns expanded query + metadata |

### Dynamic Last-Turn Detection (v4.2.0 Fix)

The adapter finds the last complete turn dynamically instead of using `turn_counter` as a buffer index (which broke when the buffer was trimmed):

```python
# Find last assistant message and its preceding user message
last_assistant_idx = -1
for i in range(len(messages) - 1, -1, -1):
    if messages[i].get("role") == "assistant":
        last_assistant_idx = i
        break

if last_assistant_idx == -1:
    new_messages = messages[-1:] if messages else []
else:
    start = max(0, last_assistant_idx - 1)
    new_messages = messages[start:]
```

This ensures only the latest turn (user + assistant) is sent to compression, preventing double-counting of already-compressed messages.

---

## Explicit Reset Patterns

### Italian

```
nuovo argomento, cambiamo topic, dimentica, ripartiamo,
lasciamo perdere, non mi interessa piu, parliamo d'altro,
cambiamo discorso
```

### English

```
new topic, forget, start over, let's move on,
different subject, change topic, never mind, moving on
```

When detected:
- No merge/promote logic applied
- New topic entry created directly
- HOLD pointer is not set
- Clean break from previous context

---

## Backward Compatibility

| Component | Strategy |
|-----------|----------|
| `narrative_summary` | Derived from `conversation_thread` via `derive_narrative_summary()` |
| `previous_topics` | Derived from thread entries via `_sync_previous_topics()` |
| `structured_context` | Populated with `current_topic`, `intent`, `entities` from LLM extraction |
| Empty thread (old states) | Falls back to legacy pipe-delimited format (`CONTEXT SUMMARY | CURRENT TOPIC | ...`) |
| `get_context_for_llm()` | Still works, returns simple formatted context (v1.0) |
| `get_structured_context()` | Returns full v4.2.0 format with thread fields |

---

## Dependencies

- **Required**: Redis (session storage, state persistence, context caching)
- **Required**: Event Bus (async compression trigger via `memory.message_added`)
- **Required**: LLM Provider (content extraction; configurable via `UBP_MEMORY__LLM_PROVIDER`)
- **Optional**: rag_orchestrator (consumes structured context for prompt injection)

---

## Logging

All memory lifecycle decisions are logged with the `[MEMORY-THREAD]` prefix:

```
[MEMORY-THREAD] action=new_topic focus="Redis override system" importance=6 turn=4
[MEMORY-THREAD] action=merge_adjacent focus="Redis override system" merge_count=2 importance=7 turn=5
[MEMORY-THREAD] action=fading focus="UBP overview" detail_level=full->recent boost=0 importance=5 turn=7
[MEMORY-THREAD] action=smart_promote focus="Core modules" detail_level=fading->recent boost=5 importance=6 turn=8
[MEMORY-THREAD] action=hold_set current="Core modules" hold="Redis override system"
[MEMORY-THREAD] action=hold_swap current="Redis override"->current hold="Core modules"->hold
[MEMORY-THREAD] action=hold_expired hold="Docker containers" after 4 turns
[MEMORY-THREAD] action=archive focus="UBP overview" detail_level=background->archived importance=4 turn=15
[MEMORY-THREAD] action=new_topic_reset focus="Artifact system" importance=5 turn=9
[MEMORY-THREAD] action=trim_to_max focus="Settings" turn=2
```

**Note**: Logger effective level must be INFO or lower to see `[MEMORY-THREAD]` logs. Module loggers default to WARNING in production. Set `UBP_LOG_LEVEL=INFO` or configure per-module logging.

---

## Testing

### Direct Memory Test (Recommended)

Bypasses chat endpoint, tests memory module directly via `add_message` + `get_structured_context`:

```bash
python test_memory_v42_direct.py
```

Expected output (6/6 PASS):

```
SCORE: 6/6
  [PASS] MEMORY header
  [PASS] [CURRENT]
  [PASS] TOPIC FLOW
  [PASS] ENTITIES
  [PASS] INTENT
  [PASS] Detail levels
```

### Chat-Based Test (Slow)

Tests through full RAG pipeline (requires vLLM inference):

```bash
python test_memory_v42_thread.py
```

### Unit Tests

```bash
cd ubp_enterprise_hybrid
pytest tests/integration/test_rag_v150_modules.py -k "conversation_memory" -v
```

---

## Deployment

### Quick Deploy (docker cp + restart)

```bash
# Copy modified files
docker cp ubp_enterprise_hybrid/modules/cores/rag_conversation_memory/. \
  ubp-backend:/app/modules/cores/rag_conversation_memory/
docker cp ubp_enterprise_hybrid/backend/app/core/config.py \
  ubp-backend:/app/ubp_enterprise_hybrid/backend/app/core/config.py

# Restart
docker compose --profile backend restart backend

# Verify
docker compose logs -f backend --tail=50
curl http://localhost:8000/health
```

**Container path**: Files are at `/app/modules/cores/rag_conversation_memory/` inside the container (NOT `/app/ubp_enterprise_hybrid/modules/cores/`).

### Full Rebuild (Production)

```bash
docker compose --profile infra --profile backend build backend
docker compose --profile infra --profile backend up -d
```

---

## Known Limitations & Future Work (v4.3.0)

| Item | Status | Notes |
|------|--------|-------|
| Entity-based HOLD decay | Planned | Decay hold when entity overlap < 30% |
| Periodic consolidation | Planned | Background job every 8-10 turns to distill pinned_facts |
| Semantic similarity for merge | Planned | Use embeddings instead of exact/LLM-reported match |
| Archival rehydration | Planned | Recall archived turns via semantic search |
| Multi-language reset patterns | Partial | IT and EN supported; DE/FR/ES not yet |

---

**Module Type:** memory
**Architecture:** 3-file-pattern
**Production Ready:** Yes
**Current Version:** 5.0.0
