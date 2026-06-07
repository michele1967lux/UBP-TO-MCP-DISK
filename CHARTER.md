# Charter: UBP-TO-MCP-DISK — Portable Module Conversion

## 1. Obiettivo

Rendere i 39 moduli in `cores/` **portable, plug-and-play**, utilizzabili sia dal sistema `mcp-disk` che da qualsiasi altro sistema, mantenendo **totale compatibilità** con i moduli orchestratori esistenti.

---

## 2. Finding — Stato Attuale

### 2.1 Architettura Moduli UBP (3-file pattern)

| Componente | Ruolo |
|---|---|
| `adapter.py` | Bridge framework: lifecycle, DI, security context, event publishing |
| `providers.py` | Business logic pura (no framework deps) |
| `__init__.py` | Factory `create_module(module_path, **kwargs)` |
| `manifest.json` | Dichiarazione operazioni, ACL, metadata |
| `config.json` | Configurazione default (env-overridable) |

### 2.2 Coupling con `ubp_enterprise_hybrid`

| Dipendenza | Moduli affetti | Gravità |
|---|---|---|
| `BaseHybridModule` | 17 moduli | ALTA — ereditarietà diretta |
| `OperationContext` | **Tutti i 38 adapter** | ALTA — parametro universale |
| `ProviderMapper` | 8 moduli (LLM delegation) | MEDIA — lazy import |
| `SharedModelPool` | 3 moduli (GPU dedup) | MEDIA |
| `EventBus` / `Event` | 6 moduli | MEDIA |
| `DIContainer` | Multipli (via BaseHybridModule) | MEDIA |
| `manifest_loader.coerce_config_types` | 2 moduli | BASSA |
| `settings_manager` (admin routes) | 2 moduli | ALTA — FastAPI coupling |
| `DKISettings` / `MemorySettings` | 2 moduli | ALTA — config coupling |
| `subagent_memory_policy` | 1 modulo | MEDIA |
| `pipeline_orchestrator.PIPELINE_TEMPLATES` | 1 modulo | MEDIA |

### 2.3 Cross-Module Dependencies (via DI)

| Provider | Consumer |
|---|---|
| `rag_qdrant` | 9 moduli |
| `inference_ollama_grok` | 12 moduli |
| `inference_vllm` | 8 moduli |
| `inference_openai_anthropic` | 5 moduli |
| `pipeline_orchestrator` | 3 moduli |
| `adaptive_budget_manager` | 3 moduli |
| `enrichment_pipeline` | 1 modulo |
| `rag_reranker` | 1 modulo |
| `citations_verifier` | 1 modulo |
| `web_search` | 3 moduli |
| `retrieval_strategy` | 4 moduli |

### 2.4 Sistema Target: mcp-disk

| Concetto | Implementazione |
|---|---|
| Registrazione tool | `@core_tool(name, desc, schema)` decorator |
| Invocazione | `call_tool(name, args)` gateway |
| Catalogo | YAML/JSON cards su disco (`scope="tools"`) |
| Contesto | `RunContext` via `contextvars` (`current_context()`) |
| Schema | `obj_schema(props, required)` + `prop(type, desc, **extra)` |
| Runtime | `build_runtime_registry(ctx)` in `runtime.py` |

---

## 3. Strategia di Conversione

### 3.1 Principi

1. **Zero modifiche a `providers.py`** — la business logic resta intatta
2. **Adapter layer astratto** — nuova interfaccia portable che sostituisce `BaseHybridModule` e `OperationContext`
3. **Bridge mcp-disk** — wrapper `@core_tool` che mappa le operazioni del modulo al sistema mcp-disk
4. **DI agnostica** — dependency injection via interfaccia, non via package specifico
5. **Backward compat** — i moduli orchestratori continuano a funzionare con il sistema UBP originale

### 3.2 Architettura Target

```
cores/<module>/
├── adapter.py          # Adapter portable (NO ubp_enterprise_hybrid imports)
├── providers.py        # INTATTO — business logic pura
├── __init__.py         # Factory create_module() + portable factory
├── manifest.json       # Invariato
├── config.json         # Invariato
├── bridge_mcp_disk.py  # NEW — @core_tool wrappers per mcp-disk
└── _portable/          # NEW — shared portable abstractions
    ├── context.py      # PortableContext (replaces OperationContext)
    ├── base.py         # PortableModule (replaces BaseHybridModule)
    ├── di.py           # Portable DI interface
    └── events.py       # Portable event interface
```

### 3.3 Portable Abstractions

```python
# _portable/context.py
class PortableContext:
    user_id: str
    client_id: str | None
    session_id: str | None
    metadata: dict
    # Methods: extract_user_id(), extract_client_id(), normalize_ctx()

# _portable/base.py
class PortableModule:
    # Replaces BaseHybridModule
    # Provides: config loading, manifest parsing, DI access
    # NO ubp_enterprise_hybrid imports

# _portable/di.py
class DIResolver(Protocol):
    def resolve(self, name: str) -> Any: ...

# _portable/events.py
class EventPublisher(Protocol):
    async def publish(self, event_type: str, payload: dict) -> None: ...
```

---

## 4. Checklist di Conversione per Modulo

### Tier 1 — Standalone (basso coupling, nessun BaseHybridModule)

| # | Modulo | Operazioni | Dipendenze critiche | Status |
|---|---|---|---|---|
| 1 | `citation_manager` | 19 | Nessuna | [ ] |
| 2 | `content_planner` | 16 | LLM via env var | [ ] |
| 3 | `document_renderer` | 12 | Nessuna (solo libs rendering) | [ ] |
| 4 | `filter_rag_context` | 2 | Nessuna (pure heuristics) | [ ] |
| 5 | `kb_relevance_scorer` | 3 | rag_qdrant (optional) | [ ] |
| 6 | `tool_loop_compression` | 3 | tool_analysis, tool_synopsis | [ ] |
| 7 | `citation_verification_orchestrator` | 10 | citations_verifier, pipeline_orch | [ ] |
| 8 | `citations_verifier` | 7 | web_search, adaptive_budget | [ ] |
| 9 | `agentic_rag` | 15 | pipeline_orch, retrieval_strategy | [ ] |
| 10 | `context_gate` | 3 | admin_clients (optional) | [ ] |
| 11 | `graph_rag` | 21 | inference modules (optional) | [ ] |
| 12 | `hyde_pipeline` | 19 | inference modules (optional) | [ ] |
| 13 | `investigation_pipeline` | 15 | inference modules (optional) | [ ] |
| 14 | `query_expansion_pipeline` | 18 | inference modules (optional) | [ ] |
| 15 | `reasoning_rag` | 17 | inference modules (optional) | [ ] |
| 16 | `streaming_rag` | 17 | inference modules (optional) | [ ] |
| 17 | `swarm_researcher` | 13 | rag_qdrant, web_search (optional) | [ ] |
| 18 | `multimodal_rag` | 22 | inference modules (optional) | [ ] |
| 19 | `media_hub` | 6 | inference modules (optional) | [ ] |
| 20 | `kb_manager` | 10 | rag_qdrant (required) | [ ] |
| 21 | `enrichment_pipeline` | 18 | inference modules (optional) | [ ] |
| 22 | `embedding_prefilter` | 11 | rag_qdrant (required), SharedModelPool | [ ] |

### Tier 2 — BaseHybridModule (medio coupling)

| # | Modulo | Operazioni | Dipendenze critiche | Status |
|---|---|---|---|---|
| 23 | `adaptive_budget_manager` | 6 | DI container (LLM) | [ ] |
| 24 | `context_compression_engine` | 9 | DI container (LLM) | [ ] |
| 25 | `hybrid_intelligent_adaptive_memory` | 6 | DI container | [ ] |
| 26 | `rag_multi_layer_memory` | 9 | DI, EventBus, prompts/ | [ ] |
| 27 | `collection_manager` | 7 | EventBus | [ ] |
| 28 | `rag_qdrant` | 10 | Nessuna (è dependency per altri) | [ ] |
| 29 | `rag_simple_memory` | 6 | EventBus, rag_qdrant.chunker | [ ] |
| 30 | `rag_feedback` | 6 | Redis | [ ] |
| 31 | `rag_hybrid_search` | 9 | rag_qdrant, EventBus | [ ] |
| 32 | `rag_reranker` | 6 | rag_qdrant, settings_manager | [ ] |
| 33 | `rag_web_crawler` | 6 | rag_qdrant (required) | [ ] |
| 34 | `rag_conversation_memory` | 14 | Redis, MemorySettings, subagent_memory_policy | [ ] |

### Tier 3 — Orchestratori (alto coupling, massima attenzione)

| # | Modulo | Operazioni | Dipendenze critiche | Status |
|---|---|---|---|---|
| 35 | `admin_clients` | 19 | EventBus, Redis, PLATFORM_ADMIN | [ ] |
| 36 | `admin_users` | 12 | EventBus, Redis | [ ] |
| 37 | `rag_orchestrator` | 37 | **TUTTO** — hub centrale | [ ] |
| 38 | `retrieval_strategy` | 18 | SYSTEM_COLLECTIONS, multipli | [ ] |

### Tier 4 — Speciale

| # | Modulo | Note | Status |
|---|---|---|---|
| 39 | `goal_orchestration` | Manifest-only, no adapter | [ ] |

---

## 5. Ordine di Esecuzione

1. **Creare `_portable/` shared abstractions** (context, base, di, events)
2. **Tier 1** — moduli standalone (partire dai più semplici, zero deps)
3. **Tier 2** — moduli BaseHybridModule (sostituire base class)
4. **Tier 3** — orchestratori (massima cautela, backward compat)
5. **Bridge mcp-disk** — wrapper `@core_tool` per ogni modulo
6. **Test** — verifica funzionalità per ogni tier

---

## 6. Vincoli

- **NON modificare `providers.py`** — la business logic è sacra
- **NON rompere i moduli orchestratori** — compatibilità totale
- **Ogni modulo modificato = 1 commit + push**
- **Solo scritture nella cartella di lavoro corrente** (`UBP-TO-MCP-DISK/`)
- **Git repo con stesso nome della cartella**

---

## 7. Risk Assessment

| Rischio | Probabilità | Mitigazione |
|---|---|---|
| Breaking change orchestratori | Media | Adapter pattern con fallback |
| Perdita funzionalità DI | Media | DIResolver protocol con mock |
| Incompatibilità schema mcp-disk | Bassa | obj_schema mapping automatico |
| Cross-module deps circular | Bassa | Lazy resolution via protocol |
| EventBus non disponibile in mcp-disk | Alta | NoopEventPublisher fallback |
