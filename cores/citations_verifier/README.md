# citations_verifier v1.0.0

**Citation verification, trust list management, and trusted source filtering for web search and RAG validation.**

Tre modalità operative:

- **VERIFY** (post-generation): Estrae affermazioni verificabili dal testo generato, le verifica contro chunk RAG e/o fonti web, produce un trust score.
- **FILTER** (pre-generation): Fornisce liste di siti affidabili ad altri moduli come filtro nelle ricerche. Modalità `exclusive` = SOLO siti trusted.
- **MANAGE** (lifecycle): CRUD su trust list per dominio. Auto-discovery di nuove fonti via analisi web. Persistenza Redis + JSON backup.

---

## Architettura

```
┌──────────────────────────────────────────────────────────────────┐
│                       citations_verifier                         │
│                                                                  │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ ClaimExtractor │  │  RAGVerifier │  │    WebVerifier      │  │
│  │ (LLM+heuristic)│  │(keyword+LLM) │  │ (trusted+generic)  │  │
│  └───────┬────────┘  └──────┬───────┘  └─────────┬───────────┘  │
│          │                  │                     │              │
│  ┌───────┴──────────────────┴─────────────────────┴───────────┐  │
│  │              VerificationOrchestrator                       │  │
│  │        (parallel claims, merge evidence, trust score)       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ TrustListManager│  │TrustListDiscovery│  │SearchFilter    │  │
│  │ (Redis+JSON)    │  │(web analysis)    │  │Builder         │  │
│  └─────────────────┘  └──────────────────┘  └────────────────┘  │
│                                                                  │
│  Dependencies (via DI):                                          │
│  ├── web_search (required)                                       │
│  ├── inference_vllm (optional) — LLM per claim extraction        │
│  └── Redis (optional) — trust list persistence                   │
└──────────────────────────────────────────────────────────────────┘
```

### Pattern UBP 3-file

| File | Righe | Ruolo |
|------|-------|-------|
| `manifest.json` | 7 operazioni, 4 eventi, dipendenze |
| `config.json` | 4 sezioni config + 9 trust lists predefinite |
| `__init__.py` | Factory `create_module()` |
| `providers.py` | ~850 righe — 7 componenti puri, zero UBP |
| `adapter.py` | ~500 righe — DI, EventBus, Redis, lifecycle |

---

## Trust Lists predefinite

| Dominio | N. fonti | Esempi top score |
|---------|----------|------------------|
| `medical` | 12 | PubMed (0.98), WHO (0.97), ISS (0.96), Cochrane (0.96) |
| `legal_it` | 9 | Normattiva (0.99), Gazzetta Ufficiale (0.99), Corte Cost. (0.98) |
| `legal_eu` | 5 | EUR-Lex (0.99), CJEU (0.98), ECHR (0.98) |
| `technical` | 12 | Python docs (0.99), MDN (0.97), IEEE (0.96), arXiv (0.95) |
| `financial` | 10 | CONSOB (0.98), Banca d'Italia (0.98), ECB (0.98), SEC (0.97) |
| `academic` | 9 | Nature (0.97), Science (0.97), JSTOR (0.96), Scopus (0.95) |
| `government_it` | 8 | governo.it (0.98), ISTAT (0.98), INPS (0.97), INAIL (0.97) |
| `news_it` | 5 | ANSA (0.90), Adnkronos (0.88), Sole 24 Ore (0.85) |
| `news_intl` | 6 | Reuters (0.93), AP (0.93), BBC (0.90), Economist (0.90) |

Caricate automaticamente al primo avvio se Redis è vuoto. Possibilità di creare liste custom.

---

## Operazioni

### `verify_document` — Verifica completa

```python
result = await citations.verify_document(
    text="Il diabete di tipo 2 colpisce il 6.2% della popolazione...",
    rag_chunks=chunks_from_rag,
    domain="medical",
    verification_depth="standard",
    language="it",
)
# {
#     "claims_total": 8,
#     "claims_verified": 5,  "claims_partial": 2,
#     "claims_unverified": 0, "claims_contradicted": 1,
#     "trust_score": 0.72,
#     "claims": [...per-claim detail with evidence...],
#     "sources_used": ["pubmed.ncbi.nlm.nih.gov", "iss.it", ...]
# }
```

**Depth levels:**

| Depth | Cosa fa | Uso |
|-------|---------|-----|
| `quick` | Solo RAG chunks, no web | Verifiche rapide in-pipeline |
| `standard` | RAG + web su siti trusted | Default — balance qualità/velocità |
| `deep` | RAG + web trusted + generica | Report critici, legale, medico |

### `get_trusted_sources` — Fonte per altri moduli

Interfaccia primaria per `web_search` e altri moduli:

```python
# Lista URL
sources = await citations.get_trusted_sources("medical", format="domains")
# {"sources": ["pubmed.ncbi.nlm.nih.gov", "who.int", ...]}

# Filtro per SearXNG
filter = await citations.get_trusted_sources("medical", format="searxng", exclusive=True)
# {"search_filter": "site:pubmed... OR site:who.int...", "exclusive": true}

# Metadati completi
full = await citations.get_trusted_sources("legal_it", format="full")
# {"sources": [{"url": "normattiva.it", "trust_score": 0.99, ...}, ...]}
```

### `get_search_filter` — Filtro rapido inline

Per chiamata durante costruzione query:

```python
filter = await citations.get_search_filter("medical", exclusive=True)
query = f"diabete tipo 2 ({filter['filter_query']})"
results = await web_search.search(query=query, max_results=10)
```

### `manage_trust_list` — CRUD

```python
# Aggiungere
await citations.manage_trust_list(action="add", domain="medical",
    entries=[{"url": "clinicaltrials.gov", "trust_score": 0.93}])

# Rimuovere
await citations.manage_trust_list(action="remove", domain="medical", target_url="webmd.com")

# Creare lista custom
await citations.manage_trust_list(action="create_list", domain="food_safety")

# Lista tutti i domini disponibili
await citations.manage_trust_list(action="list_all")
```

### `discover_trusted_sources` — Auto-discovery

```python
result = await citations.discover_trusted_sources(
    domain="medical",
    auto_add=True,
    min_score_to_add=0.85,
    max_discoveries=20,
)
# {
#     "discovered": [
#         {"url": "clinicaltrials.gov", "trust_score": 0.91, "status": "auto_added"},
#         {"url": "drugs.com", "trust_score": 0.72, "status": "proposed"},
#     ],
#     "auto_added": 3, "proposed": 12, "already_known": 5,
# }
```

---

## Integrazione con web_search

### Modalità `prioritize` (default)

Trust list come boost nel ranking — risultati trusted prima, poi generici:

```python
trusted = await citations.get_trusted_sources("technical", format="domains")
results = await web_search.search(query=query, max_results=10)
# Post-rank: boost risultati da trusted_domains
```

### Modalità `exclusive`

SOLO siti trusted, zero generici:

```python
filter = await citations.get_search_filter("medical", exclusive=True)
results = await web_search.search(
    query=f"{user_query} ({filter['filter_query']})",
    max_results=10,
)
```

### Configurazione globale

```env
# Forza exclusive per certi domini
UBP_CITATIONS__EXCLUSIVE_DOMAINS=medical,legal_it

# Default per altri domini
UBP_CITATIONS__FILTER_MODE=prioritize
```

---

## Come funziona la verifica

### 1. Estrazione claims

**Heuristic (veloce):** Split frasi → pattern matching (%, date, legge, secondo, causa) → score verificabilità.

**LLM (qualità):** Prompt strutturato → claim + tipo. Fallback automatico a heuristic.

Tipi: `factual | statistical | citation | definition | legal`

### 2. Verifica RAG

**Keyword overlap:** Estrae keyword → calcola copertura claim vs chunk → score.

**LLM semantic:** Invia claim+chunk → chiede SUPPORTED/PARTIAL/UNSUPPORTED/CONTRADICTED.

Se RAG conferma con confidence ≥ 0.8 → done, no web search.

### 3. Verifica Web

1. Cerca su siti trusted (filtro `site:`)
2. Se insufficiente, cerca genericamente
3. Pesatura: trust_score × snippet_relevance
4. Trusted web pesa 2x rispetto a generico

### 4. Trust Score

```
trust_score = (verified × 1.0 + partial × 0.6 - contradicted × 0.5) / total_checked
```

| Score | Significato |
|-------|-------------|
| > 0.8 | Documento affidabile |
| 0.5-0.8 | Parzialmente verificato, attenzione |
| < 0.5 | Verifiche insufficienti o contraddizioni |

---

## Configurazione completa

### Verification

| ENV | Default | Descrizione |
|-----|---------|-------------|
| `UBP_CITATIONS__ENABLED` | `true` | Abilita modulo |
| `UBP_CITATIONS__DEFAULT_DEPTH` | `standard` | Profondità verifica |
| `UBP_CITATIONS__MAX_CLAIMS` | `50` | Max claim per documento |
| `UBP_CITATIONS__MAX_PARALLEL` | `5` | Verifiche parallele |
| `UBP_CITATIONS__CLAIM_MIN_LEN` | `15` | Lunghezza minima claim |
| `UBP_CITATIONS__CONF_SUPPORTED` | `0.75` | Soglia SUPPORTED |
| `UBP_CITATIONS__CONF_PARTIAL` | `0.45` | Soglia PARTIAL |
| `UBP_CITATIONS__SEARCHES_PER_CLAIM` | `2` | Ricerche web per claim |
| `UBP_CITATIONS__CLAIM_TIMEOUT` | `30` | Timeout per claim (s) |

### Trust Lists

| ENV | Default | Descrizione |
|-----|---------|-------------|
| `UBP_CITATIONS__STORAGE` | `redis` | Backend: redis / json |
| `UBP_CITATIONS__REDIS_PREFIX` | `ubp:trust_list` | Prefix chiavi Redis |
| `UBP_CITATIONS__JSON_BACKUP` | `data/trust_lists` | Path backup JSON |
| `UBP_CITATIONS__AUTO_BACKUP` | `true` | Backup automatico su modifica |
| `UBP_CITATIONS__DEFAULT_SCORE` | `0.7` | Score default nuove fonti |
| `UBP_CITATIONS__SCORE_DECAY_DAYS` | `90` | Giorni prima del decay score |
| `UBP_CITATIONS__MAX_DOMAINS` | `200` | Max domini per lista |

### Discovery

| ENV | Default | Descrizione |
|-----|---------|-------------|
| `UBP_CITATIONS__DISCOVERY_ENABLED` | `true` | Abilita auto-discovery |
| `UBP_CITATIONS__DISC_SEARCHES` | `5` | Ricerche per dominio |
| `UBP_CITATIONS__DISC_MIN_APPEARANCES` | `3` | Apparizioni minime per considerare |
| `UBP_CITATIONS__DISC_SIGNALS` | `tld,https,...` | Segnali di autorità da valutare |
| `UBP_CITATIONS__DISC_AUTO_ADD` | `0.85` | Soglia auto-add |
| `UBP_CITATIONS__DISC_COOLDOWN` | `24` | Ore tra discovery successive |

### Search Integration

| ENV | Default | Descrizione |
|-----|---------|-------------|
| `UBP_CITATIONS__FILTER_MODE` | `prioritize` | prioritize / exclusive / disabled |
| `UBP_CITATIONS__MAX_FILTER_SITES` | `10` | Max siti nel filtro query |
| `UBP_CITATIONS__FALLBACK_GENERIC` | `true` | Se trusted insufficienti, cerca generico |
| `UBP_CITATIONS__EXCLUSIVE_DOMAINS` | (vuoto) | Domini sempre in modalità exclusive (CSV) |

---

## EventBus

### Eventi pubblicati

| Evento | Trigger | Payload |
|--------|---------|---------|
| `citations.verification_completed` | Dopo verify_document | trust_score, claims_total, contradicted |
| `citations.claim_contradicted` | Claim contraddetto | claim detail con evidence |
| `citations.trust_list_updated` | Modifica trust list | domain, action, affected count |
| `citations.sources_discovered` | Dopo discovery | domain, total, auto_added, proposed |

### Subscription automatiche

| Evento | Azione |
|--------|--------|
| `report.generated` | Auto-verifica il report (depth=standard) |
| `rag.response_ready` | Auto-verifica risposta RAG (depth=quick) |

---

## Test

### Unit Test — ClaimExtractor

```python
from citations_verifier.providers import ClaimExtractor, VerificationConfig

def test_extract_statistical_claims():
    extractor = ClaimExtractor(VerificationConfig())
    claims = extractor.extract_heuristic(
        "Il 42% degli italiani soffre di ipertensione. "
        "La pressione arteriosa normale è sotto 120/80 mmHg. "
        "Questo è un fatto generico senza numeri."
    )
    assert len(claims) >= 2  # Le prime due frasi hanno indicatori
    types = [c.claim_type for c in claims]
    assert "statistical" in types

def test_extract_legal_claims():
    extractor = ClaimExtractor(VerificationConfig())
    claims = extractor.extract_heuristic(
        "L'art. 2043 del Codice Civile stabilisce l'obbligo di risarcimento. "
        "Il D.Lgs. 81/2008 regola la sicurezza sul lavoro."
    )
    assert len(claims) >= 2
    assert all(c.claim_type == "legal" for c in claims)
```

### Unit Test — TrustListManager

```python
import pytest
from citations_verifier.providers import TrustListManager, TrustListConfig, TrustedSource

@pytest.mark.asyncio
async def test_trust_list_crud():
    mgr = TrustListManager(TrustListConfig())
    await mgr.initialize(redis=None, predefined_lists={
        "test": [{"url": "example.com", "score": 0.9, "notes": "Test"}]
    })

    # Read
    sources = await mgr.get_list("test")
    assert len(sources) == 1
    assert sources[0].url == "example.com"

    # Add
    added = await mgr.add_entries("test", [
        TrustedSource(url="new.org", trust_score=0.85)
    ])
    assert added == 1
    assert len(await mgr.get_list("test")) == 2

    # Remove
    removed = await mgr.remove_entry("test", "example.com")
    assert removed
    assert len(await mgr.get_list("test")) == 1

    # Filter
    filter_str = mgr.build_search_filter("test", min_score=0.7)
    assert "site:new.org" in filter_str
```

### Unit Test — RAGVerifier

```python
from citations_verifier.providers import RAGVerifier, VerificationConfig, Claim, ClaimStatus

def test_rag_supported():
    verifier = RAGVerifier(VerificationConfig())
    claim = Claim(text="Il diabete di tipo 2 colpisce il 6% della popolazione")
    chunks = [{"text": "Il diabete mellito di tipo 2 è una patologia che colpisce circa il 6% della popolazione italiana secondo i dati ISTAT"}]

    result = verifier.verify_against_chunks(claim, chunks)
    assert result.status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIAL)
    assert result.confidence > 0.4

def test_rag_unsupported():
    verifier = RAGVerifier(VerificationConfig())
    claim = Claim(text="La capitale della Francia è Parigi")
    chunks = [{"text": "Il trattamento del diabete prevede dieta ed esercizio fisico"}]

    result = verifier.verify_against_chunks(claim, chunks)
    assert result.status == ClaimStatus.UNSUPPORTED
```

### Integration Test

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

@pytest.mark.asyncio
async def test_full_verification_pipeline():
    from citations_verifier.adapter import CitationsVerifierAdapter

    # Mock DI
    container = AsyncMock()
    mock_ws = AsyncMock()
    mock_ws.search = AsyncMock(return_value=[
        {"title": "PubMed result", "url": "https://pubmed.ncbi.nlm.nih.gov/123", "snippet": "Diabete tipo 2 colpisce 6%"}
    ])

    async def resolve(name):
        if name == "web_search": return mock_ws
        return None
    container.resolve = resolve

    adapter = CitationsVerifierAdapter(
        module_path=Path("."), di_container=container,
    )
    await adapter.initialize()

    result = await adapter.verify_document(
        text="Il diabete di tipo 2 colpisce il 6% della popolazione italiana secondo l'ISS.",
        domain="medical",
        verification_depth="standard",
    )

    assert result["claims_total"] >= 1
    assert "trust_score" in result
    assert 0 <= result["trust_score"] <= 1
```

### Esecuzione

```bash
python -m pytest modules/cores/citations_verifier/tests/ -v
python -m pytest modules/cores/citations_verifier/tests/ -v --cov
```

---

## Troubleshooting

### Il modulo non verifica via web

**Causa**: web_search non disponibile da DI. Controllare `health_check` → `web_search_available`.

### Trust list vuote al restart

**Causa**: Redis non connesso e le predefined lists non si caricano.
**Fix**: Le predefined si caricano comunque in-memory. Controllare `redis_connected` in health_check.

### Discovery non trova nuove fonti

**Causa**: `min_appearances` troppo alto o poche query seed.
**Fix**: Ridurre `UBP_CITATIONS__DISC_MIN_APPEARANCES` (3→2) o fornire `seed_queries` custom.

### Trust score sempre basso

**Causa**: Claims estratti troppo generici o threshold troppo alti.
**Fix**: Alzare `claim_min_length` (15→25) o abbassare `confidence_threshold_supported` (0.75→0.65).

### Exclusive mode blocca tutte le ricerche

**Causa**: Trust list per il dominio è vuota o con score troppo bassi.
**Fix**: Verificare la lista con `get_trusted_sources(domain, min_trust_score=0.5)` e controllare che ci siano fonti.
