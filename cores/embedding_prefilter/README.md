# Embedding Prefilter — Module Manual

> **Version:** 2.0.0 (ARCH-007 Phase 2)  
> **Status:** Production  
> **Last updated:** 2026-02-22

## Overview

The Embedding Prefilter is a 4-layer meta-routing brain that classifies user queries
into routing lanes (RAG, WEB, REPORT, FAST) **without calling an LLM**. It uses
cosine similarity between the query embedding and pre-computed cluster centroids
to make sub-30ms routing decisions, replacing the ~3000ms LLM Router call in
the majority of cases.

```
Query → [L1 Embedding] → [L2 Scoring] → [L3 Evidence*] → [L4 Decision] → Route
                                                                        ↘ DEFER → LLM Router
```

*L2 produces normalized [0,1] scores. Softmax applied ONCE after L3 evidence.

---

## Architecture

### Files

| File | LOC | Purpose |
|------|-----|---------|
| `adapter.py` | ~400 | DI bridge, R1 safe mode, dedicated embedder lifecycle |
| `providers.py` | ~930 | Core engine: 4 layers, 7 reinforcements, exemplars |
| `schemas.py` | ~150 | Pydantic models (PrefilterResult, RoutingProfile) |
| `config.json` | — | All tunable parameters (zero code changes for tuning) |
| `__init__.py` | ~56 | Module registration |

### Routing Lanes

| Lane | Route String | Description |
|------|-------------|-------------|
| RAG | `RAG` | Knowledge base retrieval (medical, legal, technical) |
| WEB | `WEB` | Real-time web search (news, current events) |
| REPORT | `REPORT` | Structured document generation (reports, analysis) |
| FAST | `FAST` | Direct chat (greetings, simple questions) |
| DEFER | `LLM_ROUTER` | Ambiguous → fallback to LLM Router for classification |

### Pipeline Flow

```
user_router.py
  → embedding_prefilter.pre_route(query, user_id)
    → L1: Embed query (Arctic 128d) + cosine sim vs centroids
    → L2: Min-max normalize → [0,1] scores (no softmax)
    → L3: Evidence enrichment (midband: delta<0.30 on normalized)
    → L4: Decision engine (thresholds + R7 severity + B2 delta guard)
  → result: {route, confidence, scores, decision_path}
```

---

## Embedding Model

### Current: Snowflake Arctic Embed L v2.0

- **Full dimension:** 1024d
- **Matryoshka routing dimension:** 128d (first 128 floats truncated)
- **VRAM:** ~480MB (dedicated instance)
- **Warm latency:** ~12ms per embed call

### Dedicated Embedder (Soluzione A)

The prefilter loads its **own** SentenceTransformer instance, independent from
`rag_qdrant`'s EmbeddingManager. This prevents model-swap ping-pong where
rag_qdrant switches to e5-small for 384d collection retrieval, causing the
prefilter's next Arctic call to trigger a GPU reload (>500ms → R1 timeout).

```
GPU Memory Layout (RTX 3080 10GB):
  rag_qdrant Arctic:   ~3.4 GB (full retrieval)
  prefilter Arctic:    ~480 MB (routing only, 128d truncated)
  Free:                ~1.2 GB
```

### Configuration

```json
"layer1_embedding": {
    "dedicated_model": true,
    "model": null,
    "device": "auto",
    "matryoshka_dim": 128,
    "full_dim": 1024
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `dedicated_model` | `true` | Load standalone SentenceTransformer (recommended) |
| `model` | `null` | Auto-detect from rag_qdrant `.env` config |
| `device` | `"auto"` | `auto` = CUDA if available, else CPU |
| `matryoshka_dim` | `128` | Truncation dimension for routing (128, 384, or 1024) |
| `full_dim` | `1024` | Full model output dimension |
| `prototype_collection` | `"routing_prototypes"` | Qdrant collection for exemplar-based centroids |
| `prototype_fallback` | `true` | Fall back to hardcoded `INTENT_EXEMPLARS` if Qdrant unavailable |

### Environment Variables

```bash
# In .env — shared with rag_qdrant
UBP_RAG_QDRANT__EMBEDDING_MODEL=Snowflake/snowflake-arctic-embed-l-v2.0
UBP_RAG_QDRANT__EMBEDDING_DIM=1024
UBP_RAG_QDRANT__EMBEDDING_DEVICE=cuda
UBP_RAG_QDRANT__EMBEDDING_BATCH=64
```

---

## Layer Details

### Layer 1 — Embedding Similarity

Computes cosine similarity between the query embedding (128d) and each
cluster centroid.

**Centroids** are loaded at `initialize()` with a two-tier strategy:

1. **Primary (Qdrant):** Load pre-computed 128d vectors from `routing_prototypes`
   collection. Groups by cluster, computes weighted mean, L2-normalizes.
   Startup time: ~38ms. Supports hot-reload via `reload_centroids` operation.

2. **Fallback (Hardcoded):** Embed all 127 exemplars from `INTENT_EXEMPLARS`,
   truncate to 128d, average per-cluster, L2-normalize. Startup: ~1700ms.
   Used when `prototype_collection` is not set or Qdrant is unavailable.

Hot-reload: `call_operation("reload_centroids")` recomputes centroids from
Qdrant without restart. Atomic swap — zero downtime.

**Output:** `raw_scores = {RAG: 0.82, WEB: 0.45, REPORT: 0.38, FAST: 0.30}`

### Layer 2 — Semantic Scoring

Transforms raw cosine scores into normalized [0, 1] range:

1. **Min-max normalization:** scales to [0, 1] range (NO softmax — softmax applied once after L3)
2. **Discrimination check:** if `max - min < min_discrimination_range` (0.05),
   all scores are too similar → returns raw scores (softmax will produce near-uniform → defer)

**Output:** `normalized_scores = {RAG: 1.000, REPORT: 0.189, WEB: 0.000, FAST: 0.173}`

### Layer 3 — Evidence Enrichment (ACTIVE)

Activates when Layer 2 produces midband results (uncertain decisions).
Collects evidence from 3 sources and adjusts normalized scores before softmax.

**Trigger:** `normalized_delta_top2 < 0.30` (on min-max normalized [0,1] scores, before softmax)

**Evidence Sources:**

| Source | Method | Latency | Effect |
|--------|--------|---------|--------|
| RAG Preview | Search user KB collections (1024d) | ~10ms | Boosts RAG if docs found, WEB if none |
| Web Signal | Similarity vs WEB prototypes (128d) | ~5ms | Boosts WEB if similar to WEB prototypes |
| Context Memory | Recent route history from session | ~1ms | Boosts dominant route if ≥2/3 recent |

**Budget:** 20ms hard timeout. If evidence collection exceeds budget, L3 is skipped.

**Behavior:** L3 adds small boosts/penalties (±0.02 to ±0.10) to normalized scores.
No renormalization — softmax applied once after L3 handles the distribution naturally.

**Zero overhead for clear decisions:** When normalized delta ≥ 0.30,
L3 is completely bypassed (no evidence collection, no score adjustment).

### Layer 4 — Decision Engine

Takes ranked scores and applies thresholds + reinforcements:

```
if top_score >= high_confidence (0.80) → ACCEPT (HIGH confidence)
if top_score >= effective_threshold    → ACCEPT (MIDBAND confidence)
if top_score <  effective_threshold    → DEFER to LLM Router
```

**Active guards in Layer 4:**

| Guard | Config Key | Effect |
|-------|-----------|--------|
| R7 Severity | `r7_severity.severity_penalties` | Raises threshold per-route |
| B2 Delta Guard | `min_top2_delta: 0.15` | Defers when top-1 ≈ top-2 |

---

## Reinforcements (R1–R7)

### R1 — Safe Mode (ACTIVE)

Wraps entire `pre_route()` in a timeout. If embedding/scoring exceeds
`timeout_ms` (500ms), falls back to LLM Router. Protects against cold-start
model loading, GPU OOM, or deadlocks.

```json
"r1_safe_mode": { "enabled": true, "timeout_ms": 500 }
```

### R2 — Dynamic Interaction (ACTIVE)

When prefilter is uncertain (confidence < 0.70 **AND** top-2 delta < 0.15), returns
`DYNAMIC_INTERACTION` with clickable options instead of blind LLM Router deferral.
Both conditions must be true (AND, not OR) to prevent false triggers on clear decisions.

```json
"r2_dynamic_interaction": {
    "enabled": true,
    "confidence_threshold": 0.70,
    "delta_threshold": 0.15,
    "max_per_session": 3,
    "excluded_routes": ["FAST"],
    "fast_exclusion_min_confidence": 0.60
}
```

**FAST exclusion:** FAST route is excluded from interaction only when confidence > 0.60
(clearly chat). Low-confidence FAST (e.g., "ciao dimmi il prezzo bitcoin") can trigger
interaction if both AND conditions are met.

**Response format:**
```json
{
    "type": "route_choice",
    "options": [
        {"route": "REPORT", "label": "Genera un Report", "icon": "📊", "confidence": 0.647},
        {"route": "WEB", "label": "Cerca sul Web", "icon": "🌐", "confidence": 0.300}
    ],
    "decision_id": "d468c8eb",
    "fallback_route": "REPORT"
}
```

**Callback:** `POST /api/user/chat/route-choice` with `{decision_id, chosen_route}`.
Pending choices stored in Redis with 5-minute TTL. Atomic get-and-delete prevents
double execution and replay attacks.

**Non-interactive clients:** If `interaction_ui_supported = false` in client config,
auto-routes to top confidence route immediately without waiting.

**Priority:** R2 fires BEFORE R7/B2 — asking the user is preferred over blind LLM deferral.

### R3 — Route Stability Guard (DISABLED)

Prevents route flapping (RAG→FAST→RAG→FAST). Planned for Phase 2c (C1).

### R4 — Cold-Start Strategy (DISABLED)

Reduces confidence for new users (< 10 lifetime queries) to force more
deferral. Planned for Phase 2c (C2).

### R5 — Prototype Drift Protection (DISABLED)

Anti-prototypes that penalize known misrouting patterns.
Planned for Phase 2g (D3).

### R6 — Softmax Masking (DISABLED)

Pre-softmax baseline gating. Currently unused.

### R7 — Severity-Adjusted Thresholds (ACTIVE)

Raises the defer threshold for safety-critical routes:

```json
"r7_severity": {
    "enabled": true,
    "base_threshold": 0.55,
    "severity_penalties": {
        "REPORT": 0.12,
        "WEB": 0.08,
        "HYBRID": 0.05,
        "RAG": 0.0,
        "FAST": 0.0
    }
}
```

**Effective thresholds:** REPORT=0.67, WEB=0.63, HYBRID=0.60, RAG/FAST=0.55

A REPORT query with confidence 0.65 is deferred to LLM Router for confirmation,
while a RAG query at 0.65 is accepted directly.

---

## Configuration Reference

### `config.json` — Complete

```json
{
  "layer1_embedding": {
    "dedicated_model": true,
    "model": null,
    "device": "auto",
    "ood_threshold": 0.55,
    "context_weight": 0.25,
    "matryoshka_dim": 128,
    "full_dim": 1024,
    "prototype_collection": "routing_prototypes",
    "prototype_fallback": true
  },
  "layer2_semantic_scoring": {
    "enabled": true,
    "softmax_temperature": 0.3,
    "min_discrimination_range": 0.05,
    "baseline_raw_threshold": 0.45,
    "global_bias": 0.0
  },
  "layer3_evidence_enrichment": {
    "enabled": true,
    "delta_threshold": 0.30,
    "timeout_ms": 20,
    "sources": {
      "rag_preview": {"enabled": true, "boost_strong": 0.08, "boost_weak": 0.04},
      "web_signal": {"enabled": true},
      "context_memory": {"enabled": true, "lookback": 3, "boost": 0.06}
    }
  },
  "layer4_decision_engine": {
    "high_confidence_threshold": 0.80,
    "defer_to_llm_router_threshold": 0.55,
    "entropy_uncertainty_threshold": 0.85,
    "min_top2_delta": 0.15
  },
  "reinforcements": { "r1..r7": "see sections above" }
}
```

### Key Tuning Parameters

| Parameter | Current | Effect of Increase | Effect of Decrease |
|-----------|---------|-------------------|-------------------|
| `softmax_temperature` | 0.3 | Flatter distribution, more deferral | Sharper, fewer deferrals |
| `min_discrimination_range` | 0.05 | More deferral on ambiguous queries | Less deferral |
| `high_confidence_threshold` | 0.80 | Fewer HIGH decisions | More HIGH (risky) |
| `defer_to_llm_router_threshold` | 0.55 | More deferral | Less deferral (risky) |
| `min_top2_delta` | 0.15 | More deferral on close top-2 | Less delta guard protection |
| `r7 severity_penalties` | varies | Higher threshold per-route | Lower per-route threshold |

---

## Operational Notes

### Logs

All prefilter decisions are logged with structured tags:

```
[PREFILTER-RAW]  Raw cosine similarities (L1 output)
[PREFILTER-L2]   Normalized + softmax scores (L2 output)
[PREFILTER-CAL]  Calibration details (min-max range, discrimination)
[PREFILTER]      Final decision (route, confidence, decision_path)
```

**Example log line:**
```
[PREFILTER] Route=RAG confidence=0.858 decision=HIGH path=L1→L2→L4 time=19ms
```

### Health Check

```bash
curl http://localhost:8000/api/system/health | jq '.modules.embedding_prefilter'
```

Returns: `{status, centroids_loaded, model_name, matryoshka_dim, embed_time_ms}`

### Shutdown

On container shutdown, the dedicated embedder releases GPU memory:
```python
self._dedicated_model = None
torch.cuda.empty_cache()
```

### Cold Start

First request after container start may trigger R1 timeout (500ms) while
the SentenceTransformer model loads to GPU (~260ms). Subsequent requests
are warm (~12ms). This is by design — R1 safely defers to LLM Router
during model loading.

---

## Routing Prototypes (Qdrant)

The module stores routing exemplars in the `routing_prototypes` Qdrant collection.
This replaces the hardcoded `INTENT_EXEMPLARS` dict with a persistent, hot-reloadable store.

### Collection Schema

| Field | Type | Description |
|-------|------|-------------|
| vector | float[128] | Pre-computed 128d Matryoshka embedding |
| cluster | string | Route cluster: `chat`, `rag`, `report`, `web_search` |
| text | string | Original exemplar text |
| weight | float | Centroid contribution weight (default 1.0) |
| language | string | Language code: `it`, `en`, `mixed` |
| domain | string | Domain category: `general`, `medical`, `legal` |
| source | string | Origin: `initial_migration`, `admin`, `feedback` |
| active | bool | Only active exemplars contribute to centroids |
| created_at | string | ISO timestamp |

### Current State

- 127 points migrated from `INTENT_EXEMPLARS`
- Distribution: chat=36, rag=32, report=30, web_search=29
- Init from Qdrant: ~38ms (vs ~1700ms embedding from hardcoded)

### Migration Tool

```bash
docker exec ubp-backend python tools/migrate_exemplars_to_qdrant.py --device cpu
```

### Hot-Reload

```python
# Reload centroids after adding/removing prototypes
await prefilter.call_operation("reload_centroids")
# Returns: {"status": "reloaded", "clusters": {...}, "elapsed_ms": 38}
```

---

## Exemplars

127 exemplars distributed across 4 clusters, multilingual (IT/EN),
with short and long variants:

| Cluster | Count | Examples |
|---------|-------|----------|
| RAG | ~35 | "effetti collaterali del paracetamolo", "drug interactions warfarin" |
| WEB | ~30 | "cerca online le ultime notizie", "latest news about..." |
| REPORT | ~32 | "genera un report dettagliato", "create analysis report..." |
| FAST | ~30 | "ciao come stai", "hello", "grazie" |

Exemplars are defined in `providers.py` (`INTENT_EXEMPLARS` dict) and
can be overridden via config: `"exemplars": { "RAG": [...], ... }`.

---

## Integration Points

| System | Direction | Description |
|--------|-----------|-------------|
| `user_router.py` | Caller | Calls `pre_route()` before LLM Router |
| `rag_qdrant` | Reads `.env` | Auto-detects embedding model from env vars |
| `Qdrant` | Storage | `routing_prototypes` collection for exemplar vectors |
| `LLM Router` | Fallback | Receives deferred queries for full classification |
| `user_profile_manager` | Future | Will provide routing preferences (C1/C2/C3) |
| `event_bus` | Publisher | Emits `embedding_prefilter.decision` events |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 2.4.0 | 2026-02-22 | Critical fixes: single softmax (L4), R2 AND logic, RAG Preview on real KB, Web Signal via prototypes, drift monitoring, atomic callback, non-interactive client fallback |
| 2.3.0 | 2026-02-22 | Layer 3 Evidence Enrichment: RAG preview, web signal, context memory |
| 2.2.0 | 2026-02-22 | R2 Dynamic Interaction, route-choice callback, InteractionOptions schema |
| 2.1.0 | 2026-02-22 | Qdrant prototype DB, hot-reload centroids, 38ms init |
| 2.0.0 | 2026-02-22 | Arctic Embed L v2.0, dedicated embedder, Matryoshka 128d |
| 1.1.0 | 2026-02-22 | R7 severity thresholds, B2 top-2 delta guard |
| 1.0.0 | 2026-02-21 | Initial release: 4-layer pipeline, e5-small 384d, R1 safe mode |
