# Citation Verification Orchestrator

**Module ID**: `citation_verification_orchestrator`  
**Version**: 1.0.0  
**Category**: Core  

## Overview

Orchestrates citation verification, grounding check, hallucination detection, and trusted source filtering across RAG pipelines. Delegates to `citations_verifier` for verification logic and uses `_call_llm` via `simple_chat` for LLM calls (agentic_rag delegation pattern).

## Architecture

```
User Query → Pipeline
                ├─ rerank
                ├─ filter_trusted (Modalità 3: pre-generate trust filter)
                ├─ deduplicate
                ├─ generate
                └─ verify_citations (Modalità 1+2: post-generate verification)
                        │
                        ├─ citations_verifier.verify_document() [primary]
                        └─ LLM grounding check via _call_llm → simple_chat [fallback]
```

## Operations

| Operation | Description | Pipeline Step |
|-----------|-------------|---------------|
| `verify_response` | Full post-generate verification: claim extraction → grounding → hallucination detection | `verify_citations` |
| `filter_trusted_sources` | Pre-generate trust filter: removes chunks from untrusted sources | `filter_trusted` |
| `verify_web_sources` | Trust check on web URLs | On-demand |
| `update_trust_database` | Auto-update trust scores after verification | Post-verification |
| `list_trust_entries` | List all trust DB entries (admin) | Admin API |
| `set_trust_entry` | Force-set domain trust score (admin) | Admin API |
| `delete_trust_entry` | Delete trust entry (admin) | Admin API |
| `bootstrap_trust_db` | Bootstrap trust DB from predefined lists | Admin API / Init |

## Activation Modes

### Modalità 1: Automatic (Confidence-Based)
- Triggers when `tightness >= 0.7` OR web sources detected
- Runs `verify_citations` step post-generate
- Config: `verify_citations_enabled` (default: false)

### Modalità 2: Explicit (Keyword-Triggered)
- User keywords: "verifica fonti", "verify sources", "fact check", etc.
- Sets `verify_requested=True` → forces verification
- Config: `verify_citations_enabled=True` via merged_config

### Modalità 3: Trust Filter (Pre-Generate)
- User keywords: "fonti affidabili", "trusted sources", etc.
- Sets `trust_filter_requested=True` → activates `filter_trusted` step
- Removes chunks from untrusted sources before generation

## Trust Database (Redis)

| Key | Format | TTL |
|-----|--------|-----|
| `ubp:trust:domain:{domain}` | JSON | 30 days |

### Auto-Learn
- Grounding > 0.8 → trust_score += 0.02
- Grounding < 0.3 → trust_score -= 0.05
- Decay: -0.001/day when not verified

### Bootstrap
On init, loads predefined domains from `citations_verifier/config.json` (medical, legal, technical, financial, academic, government, news categories).

## Admin Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/trust-database` | List all trust entries |
| POST | `/api/admin/trust-database` | Set domain trust score |
| DELETE | `/api/admin/trust-database/{domain}` | Delete trust entry |
| POST | `/api/admin/trust-database/bootstrap` | Trigger bootstrap |

## Dependencies

| Module | Required | Purpose |
|--------|----------|---------|
| `citations_verifier` | ✅ | Verification logic |
| `pipeline_orchestrator` | Optional | LLM calls via simple_chat |
| `adaptive_budget_manager` | Optional | Tightness calculation |
| Redis | Optional | Trust database persistence |

## Pipeline Integration

Added to 6 RAG pipelines:
- `rag_chat_standard`
- `rag_chat_enhanced`
- `rag_chat_conversational`
- `rag_chat_personalized`
- `rag_deepdive`
- `report_research`

## Configuration

See `config.json` for all settings. Key environment variables:
- `UBP_CITATION_VERIFICATION__ENABLED` — Global on/off
- `UBP_CITATION_VERIFICATION__AUTO_TIGHTNESS` — Auto-verify threshold
- `UBP_CITATION_VERIFICATION__TRUST_MIN` — Minimum trust score for filter
- `UBP_CITATION_VERIFICATION__AUTO_LEARN` — Enable auto-learn
