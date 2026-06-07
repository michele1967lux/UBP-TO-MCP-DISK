# Report Ecosystem v2.0 - Architettura Modulare

**Proposta Architetturale per Sistema Report Enterprise-Grade**

Version: 2.0.0 | Status: PROPOSAL | Data: 2025-01-23

---

## 📋 Executive Summary

Sistema di generazione report completamente modulare, pipeline-native, con:
- 6 moduli standalone riusabili
- PDF nativo + DOCX + PPTX + Markdown
- Grafici, tabelle avanzate, immagini (predisposizione)
- Citation tracking completo
- Template personalizzabili con microprompt
- Interattività estrema per costruzione report
- Storage multi-backend (filesystem + S3)
- Risultato professionale + user-friendly

---

## 🏗️ Architettura Complessiva

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           REPORT ECOSYSTEM v2.0                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        ORCHESTRATION LAYER                               │   │
│   │                    (pipeline_orchestrator esistente)                     │   │
│   └───────────────────────────────┬─────────────────────────────────────────┘   │
│                                   │                                              │
│   ┌───────────────────────────────┼─────────────────────────────────────────┐   │
│   │                          CORE MODULES                                    │   │
│   │                                                                          │   │
│   │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │   │
│   │  │  content_planner │  │ multi_source_    │  │ document_composer│       │   │
│   │  │                  │  │   researcher     │  │                  │       │   │
│   │  │ • Plan structure │  │                  │  │ • Section build  │       │   │
│   │  │ • Template match │  │ • Parallel fetch │  │ • Content merge  │       │   │
│   │  │ • Microprompts   │  │ • Citations      │  │ • Quality check  │       │   │
│   │  │ • Token budget   │  │ • Aggregation    │  │ • Iterative edit │       │   │
│   │  │ • Validation     │  │ • Deduplication  │  │ • Version track  │       │   │
│   │  └──────────────────┘  └──────────────────┘  └──────────────────┘       │   │
│   │                                                                          │   │
│   │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │   │
│   │  │ document_renderer│  │  chart_generator │  │ artifact_storage │       │   │
│   │  │                  │  │                  │  │                  │       │   │
│   │  │ • PDF (native)   │  │ • Bar/Line/Pie   │  │ • Filesystem     │       │   │
│   │  │ • DOCX           │  │ • Tables adv.    │  │ • S3/Cloud       │       │   │
│   │  │ • PPTX           │  │ • Data viz       │  │ • Versioning     │       │   │
│   │  │ • Markdown       │  │ • Export PNG/SVG │  │ • TTL/Cleanup    │       │   │
│   │  │ • Custom templates│ │ • Embed ready    │  │ • Streaming      │       │   │
│   │  └──────────────────┘  └──────────────────┘  └──────────────────┘       │   │
│   │                                                                          │   │
│   └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                        INTERACTION LAYER                                  │  │
│   │                                                                           │  │
│   │   Pipeline Templates:                                                     │  │
│   │   • report_quick.yaml        - Generazione veloce, no interazione        │  │
│   │   • report_interactive.yaml  - Step-by-step con approvazioni             │  │
│   │   • report_professional.yaml - Full workflow con review                  │  │
│   │   • presentation_builder.yaml - Solo presentazioni                       │  │
│   │   • document_export.yaml     - Solo export multi-formato                 │  │
│   │                                                                           │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Moduli Dettagliati

### 1️⃣ content_planner

**Responsabilità:** Pianificazione struttura report, gestione template, microprompt per sezione.

```
content_planner/
├── __init__.py
├── adapter.py           # ~400 linee
├── providers/
│   ├── __init__.py
│   ├── plan_generator.py    # ~300 linee - LLM-based planning
│   ├── template_manager.py  # ~250 linee - Template loading/matching
│   ├── microprompt_engine.py # ~200 linee - Per-section prompts
│   └── validators.py        # ~150 linee - Plan validation
├── templates/
│   └── default_templates.yaml
├── config.json
├── manifest.json
└── README.md

Totale: ~1300 linee
```

#### Operazioni

| Operazione | Descrizione | Input | Output |
|------------|-------------|-------|--------|
| `plan_structure` | Genera piano strutturato da query | query, constraints, template_hints | StructuredPlan |
| `match_template` | Trova template più adatto | query, available_templates | TemplateMatch |
| `adapt_template` | Adatta template al contesto | template_id, context, customizations | AdaptedTemplate |
| `generate_microprompts` | Genera prompt specifici per sezione | section_plan, context | List[Microprompt] |
| `validate_plan` | Valida piano strutturale | plan | ValidationResult |
| `estimate_resources` | Stima token/tempo/costi | plan | ResourceEstimate |
| `modify_section` | Modifica sezione in piano | plan, section_id, changes | UpdatedPlan |
| `reorder_sections` | Riordina sezioni | plan, new_order | UpdatedPlan |
| `add_section` | Aggiunge sezione | plan, section_spec | UpdatedPlan |
| `remove_section` | Rimuove sezione | plan, section_id | UpdatedPlan |

#### Data Classes

```python
@dataclass
class StructuredPlan:
    """Piano strutturato del report."""
    id: str
    title: str
    description: str
    template_id: Optional[str]
    sections: List[SectionPlan]
    metadata: PlanMetadata
    constraints: PlanConstraints
    version: int = 1
    created_at: datetime
    updated_at: datetime

@dataclass
class SectionPlan:
    """Piano di una singola sezione."""
    id: str
    title: str
    description: str
    order: int
    
    # Content guidance
    microprompt: str                    # Prompt specifico per questa sezione
    content_type: ContentType           # narrative, table, chart, mixed
    target_tokens: int
    
    # Source configuration
    source_preference: SourcePreference
    suggested_queries: List[str]
    required_data_types: List[str]      # es: ["prices", "specs", "reviews"]
    
    # Dependencies
    depends_on: List[str]               # Section IDs
    provides_context_for: List[str]     # Section IDs che useranno output
    
    # Customization
    custom_template: Optional[str]      # Template DOCX/section override
    style_hints: Dict[str, Any]         # font, spacing, etc.
    
    # Flags
    required: bool = True
    interactive_review: bool = False    # Richiede approvazione utente
    enabled: bool = True                # Per skip dinamico

@dataclass
class Microprompt:
    """Prompt specifico per generazione contenuto sezione."""
    section_id: str
    system_prompt: str
    user_prompt_template: str
    output_format: str                  # markdown, json, structured
    quality_criteria: List[str]
    example_output: Optional[str]
    
@dataclass
class ReportTemplate:
    """Template report personalizzabile."""
    id: str
    name: str
    description: str
    category: str                       # business, technical, academic, etc.
    
    # Structure
    default_sections: List[SectionTemplate]
    required_sections: List[str]        # IDs delle sezioni obbligatorie
    optional_sections: List[str]
    
    # Styling
    docx_template_path: Optional[str]   # Template DOCX custom
    pptx_template_path: Optional[str]
    style_config: StyleConfig
    
    # Selection rules
    keywords: List[str]
    selection_patterns: List[str]       # Regex patterns
    domain_hints: List[str]
    
    # Microprompts defaults
    section_microprompts: Dict[str, str]

@dataclass
class SectionTemplate:
    """Template per sezione specifica."""
    id: str
    title_template: str                 # Con placeholder: "Analysis of {topic}"
    microprompt: str
    content_type: ContentType
    default_tokens: int
    source_preference: SourcePreference
    style_hints: Dict[str, Any]
```

#### Template YAML Example

```yaml
# templates/computo_metrico.yaml
id: computo_metrico
name: "Computo Metrico"
description: "Computo metrico per lavori edili"
category: technical
keywords:
  - computo
  - metrico
  - preventivo
  - lavori
  - ristrutturazione

default_sections:
  - id: intestazione
    title: "Intestazione"
    microprompt: |
      Genera l'intestazione formale del computo metrico includendo:
      - Oggetto dei lavori
      - Ubicazione
      - Committente (se specificato)
      - Data
      Usa un tono formale e professionale.
    content_type: narrative
    default_tokens: 200
    source_preference: llm_reasoning
    
  - id: descrizione_lavori
    title: "Descrizione Lavori"
    microprompt: |
      Descrivi dettagliatamente i lavori da eseguire basandoti sulle informazioni
      raccolte. Includi:
      - Tipologia di intervento
      - Materiali principali
      - Metodologie di esecuzione
      Sii preciso e tecnico.
    content_type: narrative
    default_tokens: 500
    source_preference: mixed
    
  - id: voci_computo
    title: "Voci di Computo"
    microprompt: |
      Genera una tabella dettagliata delle voci di computo con:
      - Numero progressivo
      - Descrizione voce
      - Unità di misura
      - Quantità
      - Prezzo unitario (€)
      - Importo totale (€)
      Usa i prezzi dal listino se disponibili, altrimenti stima ragionevole.
    content_type: table
    default_tokens: 1000
    source_preference: rag_first
    required_data_types:
      - prices
      - quantities
      
  - id: riepilogo
    title: "Riepilogo Economico"
    microprompt: |
      Genera il riepilogo economico con:
      - Totale lavori
      - Spese generali (15%)
      - Utile impresa (10%)
      - Totale complessivo
      - IVA (se applicabile)
      Formatta come tabella di riepilogo.
    content_type: table
    default_tokens: 300
    source_preference: llm_reasoning
    depends_on:
      - voci_computo

style_config:
  font_family: "Times New Roman"
  font_size: 11
  header_style: "formal"
  table_style: "professional_bordered"
```

---

### 2️⃣ multi_source_researcher

**Responsabilità:** Ricerca parallela da multiple fonti con citation tracking.

```
multi_source_researcher/
├── __init__.py
├── adapter.py               # ~400 linee
├── providers/
│   ├── __init__.py
│   ├── parallel_executor.py # ~350 linee - Swarm execution
│   ├── source_router.py     # ~200 linee - Source selection
│   ├── citation_tracker.py  # ~250 linee - Citation management
│   ├── aggregator.py        # ~200 linee - Result aggregation
│   └── quality_scorer.py    # ~150 linee - Source quality
├── config.json
├── manifest.json
└── README.md

Totale: ~1550 linee
```

#### Operazioni

| Operazione | Descrizione | Input | Output |
|------------|-------------|-------|--------|
| `research_single` | Ricerca per singola query | query, source_config | ResearchResult |
| `research_parallel` | Ricerca parallela multi-query | queries[], source_config | List[ResearchResult] |
| `research_section` | Ricerca ottimizzata per sezione | section_plan, context | SectionResearchResult |
| `research_with_deps` | Ricerca rispettando dipendenze | task_graph | OrderedResults |
| `aggregate_results` | Aggrega risultati multi-fonte | results[], strategy | AggregatedResult |
| `deduplicate` | Rimuove duplicati | results[] | DeduplicatedResults |
| `score_sources` | Valuta qualità fonti | sources[] | ScoredSources |
| `extract_citations` | Estrae citazioni strutturate | content, sources | Citations |
| `format_bibliography` | Genera bibliografia | citations, style | FormattedBibliography |

#### Data Classes

```python
@dataclass
class ResearchResult:
    """Risultato di una singola ricerca."""
    query: str
    source_type: SourceType             # rag, web, hybrid
    
    # Results
    documents: List[RetrievedDocument]
    total_found: int
    
    # Quality metrics
    relevance_scores: List[float]
    coverage_score: float               # Quanto copre la query
    
    # Citations
    citations: List[Citation]
    
    # Metadata
    execution_time_ms: int
    source_breakdown: Dict[str, int]    # {rag: 5, web: 3}

@dataclass
class Citation:
    """Citazione strutturata."""
    id: str
    source_type: SourceType
    
    # Content
    text: str                           # Testo citato
    context: str                        # Contesto circostante
    
    # Source info
    title: str
    author: Optional[str]
    date: Optional[str]
    url: Optional[str]
    page: Optional[int]
    
    # RAG specific
    collection: Optional[str]
    chunk_id: Optional[str]
    document_id: Optional[str]
    
    # Quality
    relevance_score: float
    confidence: float
    
    # Formatting
    def to_footnote(self, style: str = "chicago") -> str: ...
    def to_bibliography(self, style: str = "chicago") -> str: ...
    def to_inline(self, style: str = "author_date") -> str: ...

@dataclass
class SectionResearchResult:
    """Risultato ricerca per sezione report."""
    section_id: str
    
    # Research results
    primary_results: ResearchResult
    fallback_results: Optional[ResearchResult]
    
    # Aggregated data
    relevant_content: List[ContentChunk]
    data_tables: List[DataTable]        # Dati strutturati trovati
    
    # Citations for this section
    citations: List[Citation]
    
    # Coverage analysis
    queries_executed: List[str]
    coverage_gaps: List[str]            # Cosa non è stato trovato
    suggestions: List[str]              # Query aggiuntive suggerite
    
    # Quality
    overall_quality: float
    confidence: float
```

#### Source Configuration

```python
@dataclass
class SourceConfig:
    """Configurazione fonti per ricerca."""
    
    # RAG settings
    rag_enabled: bool = True
    rag_collections: List[str] = field(default_factory=list)
    rag_top_k: int = 10
    rag_min_score: float = 0.5
    rag_rerank: bool = True
    
    # Web settings
    web_enabled: bool = True
    web_max_results: int = 5
    web_domains_whitelist: List[str] = field(default_factory=list)
    web_domains_blacklist: List[str] = field(default_factory=list)
    
    # Hybrid settings
    source_preference: SourcePreference = SourcePreference.RAG_FIRST
    fallback_enabled: bool = True
    fallback_threshold: int = 2         # Min docs before fallback
    
    # Parallel execution
    max_parallel_queries: int = 5
    timeout_seconds: int = 30
    
    # Citation
    citation_style: str = "chicago"     # chicago, apa, mla, ieee
    extract_citations: bool = True
```

---

### 3️⃣ document_composer

**Responsabilità:** Composizione contenuto sezioni, editing iterativo, version tracking.

```
document_composer/
├── __init__.py
├── adapter.py               # ~400 linee
├── providers/
│   ├── __init__.py
│   ├── section_generator.py # ~350 linee - Content generation
│   ├── content_merger.py    # ~200 linee - Merge multi-source
│   ├── quality_checker.py   # ~250 linee - Quality validation
│   ├── version_tracker.py   # ~200 linee - Version management
│   └── interactive_editor.py # ~300 linee - Interactive editing
├── config.json
├── manifest.json
└── README.md

Totale: ~1700 linee
```

#### Operazioni

| Operazione | Descrizione | Input | Output |
|------------|-------------|-------|--------|
| `generate_section` | Genera contenuto sezione | section_plan, research_data, microprompt | SectionContent |
| `generate_all_sections` | Genera tutte le sezioni | plan, research_results | List[SectionContent] |
| `merge_content` | Unisce contenuti multi-fonte | contents[], strategy | MergedContent |
| `check_quality` | Verifica qualità contenuto | content, criteria | QualityReport |
| `improve_section` | Migliora sezione con feedback | section, feedback, criteria | ImprovedSection |
| `get_version` | Recupera versione specifica | section_id, version | SectionContent |
| `list_versions` | Lista versioni sezione | section_id | List[VersionInfo] |
| `compare_versions` | Confronta due versioni | section_id, v1, v2 | DiffResult |
| `revert_version` | Ripristina versione | section_id, version | SectionContent |
| `compose_document` | Assembla documento finale | sections[], metadata | ComposedDocument |

#### Data Classes

```python
@dataclass
class SectionContent:
    """Contenuto generato per una sezione."""
    section_id: str
    title: str
    
    # Content
    content_markdown: str               # Contenuto in markdown
    content_html: Optional[str]         # Pre-rendered HTML
    content_structured: Optional[Dict]  # Per tabelle/dati
    
    # Embedded elements
    tables: List[TableData]
    charts: List[ChartSpec]
    images: List[ImageRef]
    
    # Citations
    citations: List[Citation]
    footnotes: List[str]
    
    # Metadata
    token_count: int
    generation_model: str
    generation_time_ms: int
    
    # Version
    version: int
    previous_version_id: Optional[str]
    created_at: datetime
    
    # Quality
    quality_score: float
    quality_issues: List[QualityIssue]

@dataclass
class ComposedDocument:
    """Documento composto pronto per rendering."""
    id: str
    title: str
    
    # Structure
    sections: List[SectionContent]
    table_of_contents: TableOfContents
    
    # Bibliography
    citations: List[Citation]
    bibliography: str
    
    # Metadata
    metadata: DocumentMetadata
    
    # Quality
    overall_quality: float
    total_tokens: int
    
    # Versions
    version: int
    version_history: List[VersionInfo]

@dataclass
class QualityReport:
    """Report qualità contenuto."""
    overall_score: float                # 0-1
    
    # Detailed scores
    relevance: float                    # Pertinenza alla query
    completeness: float                 # Copertura argomenti
    coherence: float                    # Coerenza interna
    accuracy: float                     # Accuratezza (vs fonti)
    readability: float                  # Leggibilità
    
    # Issues found
    issues: List[QualityIssue]
    
    # Suggestions
    improvements: List[str]
    
    # Verification
    claims_verified: int
    claims_unverified: int
    potential_hallucinations: List[str]

@dataclass 
class QualityIssue:
    """Problema di qualità identificato."""
    type: QualityIssueType              # factual, coherence, style, citation
    severity: Severity                  # low, medium, high
    location: str                       # Section/paragraph reference
    description: str
    suggestion: Optional[str]
```

---

### 4️⃣ document_renderer

**Responsabilità:** Rendering multi-formato con template personalizzabili.

```
document_renderer/
├── __init__.py
├── adapter.py               # ~300 linee
├── providers/
│   ├── __init__.py
│   ├── base_renderer.py     # ~200 linee - Abstract base
│   ├── docx_renderer.py     # ~500 linee - Word
│   ├── pdf_renderer.py      # ~450 linee - PDF nativo
│   ├── pptx_renderer.py     # ~500 linee - PowerPoint
│   ├── markdown_renderer.py # ~150 linee - Markdown
│   ├── html_renderer.py     # ~200 linee - HTML
│   └── template_engine.py   # ~250 linee - Template processing
├── templates/
│   ├── default.docx
│   ├── professional.docx
│   ├── default.pptx
│   └── styles/
│       └── default_styles.yaml
├── config.json
├── manifest.json
└── README.md

Totale: ~2550 linee
```

#### Operazioni

| Operazione | Descrizione | Input | Output |
|------------|-------------|-------|--------|
| `render_docx` | Genera documento Word | composed_doc, template, options | bytes |
| `render_pdf` | Genera PDF nativo | composed_doc, options | bytes |
| `render_pptx` | Genera PowerPoint | composed_doc/slides, template, options | bytes |
| `render_markdown` | Genera Markdown | composed_doc, options | str |
| `render_html` | Genera HTML | composed_doc, options | str |
| `render_multi` | Genera multipli formati | composed_doc, formats[] | Dict[format, bytes] |
| `list_templates` | Lista template disponibili | format | List[TemplateInfo] |
| `upload_template` | Carica template custom | template_bytes, format, name | TemplateInfo |
| `preview` | Genera preview (prima pagina) | composed_doc, format | bytes (image) |

#### Rendering Options

```python
@dataclass
class DocxOptions:
    """Opzioni rendering DOCX."""
    template_id: Optional[str] = None   # Template custom
    template_path: Optional[str] = None # Path diretto
    
    # Page setup
    page_size: str = "A4"               # A4, Letter, Legal
    orientation: str = "portrait"       # portrait, landscape
    margins: Margins = field(default_factory=lambda: Margins(2.5, 2.5, 2.5, 2.5))
    
    # Typography
    font_family: str = "Calibri"
    font_size: int = 11
    line_spacing: float = 1.15
    
    # Sections
    include_toc: bool = True
    include_cover: bool = True
    include_bibliography: bool = True
    include_page_numbers: bool = True
    
    # Headers/Footers
    header_text: Optional[str] = None
    footer_text: Optional[str] = None
    
    # Tables
    table_style: str = "professional"   # professional, simple, bordered
    
    # Charts (embedded)
    embed_charts: bool = True
    chart_width_inches: float = 6.0

@dataclass
class PdfOptions:
    """Opzioni rendering PDF nativo."""
    template_id: Optional[str] = None
    
    # Page setup
    page_size: str = "A4"
    orientation: str = "portrait"
    margins: Margins = field(default_factory=lambda: Margins(2.5, 2.5, 2.5, 2.5))
    
    # Typography
    font_family: str = "Helvetica"
    font_size: int = 11
    
    # Features
    include_toc: bool = True
    include_cover: bool = True
    include_bibliography: bool = True
    include_page_numbers: bool = True
    include_bookmarks: bool = True      # PDF bookmarks/outline
    
    # Security
    encrypt: bool = False
    password: Optional[str] = None
    allow_printing: bool = True
    allow_copying: bool = True
    
    # Watermark
    watermark_text: Optional[str] = None
    watermark_opacity: float = 0.1

@dataclass
class PptxOptions:
    """Opzioni rendering PowerPoint."""
    template_id: Optional[str] = None
    template_path: Optional[str] = None
    
    # Slide setup
    width_inches: float = 13.333        # 16:9 default
    height_inches: float = 7.5
    
    # Content distribution
    max_slides: int = 50
    content_per_slide: str = "auto"     # auto, one_section, custom
    
    # Styling
    theme: str = "professional"         # professional, minimal, colorful
    font_title: str = "Calibri"
    font_body: str = "Calibri"
    
    # Features
    include_title_slide: bool = True
    include_toc_slide: bool = True
    include_summary_slide: bool = True
    include_speaker_notes: bool = True
    
    # Charts
    embed_charts: bool = True
    animate_charts: bool = False
```

---

### 5️⃣ chart_generator

**Responsabilità:** Generazione grafici, tabelle avanzate, visualizzazioni.

```
chart_generator/
├── __init__.py
├── adapter.py               # ~300 linee
├── providers/
│   ├── __init__.py
│   ├── chart_builder.py     # ~400 linee - Chart creation
│   ├── table_builder.py     # ~350 linee - Advanced tables
│   ├── data_processor.py    # ~200 linee - Data transformation
│   └── export_engine.py     # ~200 linee - Export PNG/SVG
├── config.json
├── manifest.json
└── README.md

Totale: ~1450 linee
```

#### Operazioni

| Operazione | Descrizione | Input | Output |
|------------|-------------|-------|--------|
| `create_chart` | Crea grafico | data, chart_type, options | ChartResult |
| `create_table` | Crea tabella avanzata | data, options | TableResult |
| `transform_data` | Trasforma dati per visualizzazione | raw_data, transformations | TransformedData |
| `export_png` | Esporta grafico come PNG | chart, resolution | bytes |
| `export_svg` | Esporta grafico come SVG | chart | str |
| `embed_in_docx` | Prepara per embedding DOCX | chart | DocxEmbeddable |
| `embed_in_pptx` | Prepara per embedding PPTX | chart | PptxEmbeddable |
| `suggest_visualization` | Suggerisce tipo grafico | data | VisualizationSuggestion |

#### Chart Types

```python
class ChartType(str, Enum):
    """Tipi di grafico supportati."""
    # Basic
    BAR = "bar"
    BAR_HORIZONTAL = "bar_horizontal"
    BAR_STACKED = "bar_stacked"
    LINE = "line"
    LINE_AREA = "line_area"
    PIE = "pie"
    DONUT = "donut"
    
    # Advanced
    SCATTER = "scatter"
    BUBBLE = "bubble"
    RADAR = "radar"
    WATERFALL = "waterfall"
    GANTT = "gantt"
    
    # Tables as charts
    HEATMAP = "heatmap"
    TREEMAP = "treemap"

@dataclass
class ChartSpec:
    """Specifica per creazione grafico."""
    chart_type: ChartType
    title: str
    
    # Data
    data: ChartData
    
    # Axes
    x_axis: Optional[AxisConfig] = None
    y_axis: Optional[AxisConfig] = None
    
    # Styling
    colors: Optional[List[str]] = None
    theme: str = "professional"
    
    # Size
    width: int = 800
    height: int = 600
    
    # Labels
    show_legend: bool = True
    show_data_labels: bool = False
    
    # Export
    export_format: str = "png"          # png, svg, both

@dataclass
class TableSpec:
    """Specifica per tabella avanzata."""
    data: List[List[Any]]
    headers: List[str]
    
    # Styling
    style: str = "professional"         # professional, minimal, striped
    header_style: str = "bold"
    
    # Features
    merge_cells: List[MergeSpec] = field(default_factory=list)
    column_widths: Optional[List[float]] = None
    row_heights: Optional[List[float]] = None
    
    # Formatting
    number_format: Optional[str] = None # es: "#,##0.00"
    currency_symbol: Optional[str] = None
    
    # Colors
    header_bg_color: str = "#4472C4"
    header_text_color: str = "#FFFFFF"
    alternate_row_color: Optional[str] = "#F2F2F2"
    
    # Totals
    show_totals_row: bool = False
    totals_columns: List[int] = field(default_factory=list)
```

---

### 6️⃣ artifact_storage

**Responsabilità:** Storage multi-backend, versioning, lifecycle management.

```
artifact_storage/
├── __init__.py
├── adapter.py               # ~350 linee
├── providers/
│   ├── __init__.py
│   ├── filesystem_backend.py # ~250 linee
│   ├── s3_backend.py        # ~300 linee
│   ├── version_manager.py   # ~200 linee
│   └── cleanup_manager.py   # ~150 linee
├── config.json
├── manifest.json
└── README.md

Totale: ~1250 linee
```

#### Operazioni

| Operazione | Descrizione | Input | Output |
|------------|-------------|-------|--------|
| `store` | Salva artifact | content, metadata | ArtifactInfo |
| `store_batch` | Salva multipli | artifacts[] | List[ArtifactInfo] |
| `retrieve` | Recupera artifact | artifact_id | bytes |
| `retrieve_stream` | Recupera in streaming | artifact_id | AsyncIterator[bytes] |
| `get_metadata` | Recupera metadata | artifact_id | ArtifactMetadata |
| `list_artifacts` | Lista per utente/sessione | filters | List[ArtifactInfo] |
| `delete` | Elimina artifact | artifact_id | bool |
| `get_versions` | Lista versioni | artifact_id | List[VersionInfo] |
| `get_version` | Recupera versione specifica | artifact_id, version | bytes |
| `cleanup_expired` | Pulizia scaduti | - | CleanupReport |
| `get_download_url` | Genera URL download | artifact_id, expires_in | str |

#### Storage Configuration

```python
@dataclass
class StorageConfig:
    """Configurazione storage."""
    
    # Backend selection
    primary_backend: str = "filesystem"  # filesystem, s3
    fallback_backend: Optional[str] = None
    
    # Filesystem
    base_path: str = "/app/artifacts"
    
    # S3
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None
    s3_prefix: str = "artifacts/"
    s3_access_key: Optional[str] = None  # or use IAM role
    s3_secret_key: Optional[str] = None
    
    # Versioning
    versioning_enabled: bool = True
    max_versions: int = 10
    
    # TTL
    default_ttl_hours: int = 168         # 7 days
    max_ttl_hours: int = 720             # 30 days
    
    # Limits
    max_file_size_mb: int = 100
    max_storage_per_user_mb: int = 1000
    
    # Security
    encrypt_at_rest: bool = False
    encryption_key: Optional[str] = None

@dataclass
class ArtifactMetadata:
    """Metadata artifact."""
    artifact_id: str
    
    # Ownership
    user_id: str
    session_id: Optional[str]
    
    # File info
    filename: str
    format: str                          # docx, pdf, pptx, etc.
    mime_type: str
    size_bytes: int
    checksum: str                        # SHA256
    
    # Content info
    title: str
    description: Optional[str]
    document_type: str                   # report, presentation, export
    
    # Version
    version: int
    is_latest: bool
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    
    # Storage
    storage_backend: str
    storage_path: str
    
    # Access
    download_url: str
    download_count: int
```

---

## 🔄 Pipeline Templates

### Pipeline: Report Quick (No Interazione)

```yaml
name: report_quick
version: "1.0.0"
description: "Generazione report veloce senza interazione"

variables:
  formats: ${config.output_formats|default:["pdf", "docx"]}
  quality_threshold: ${config.quality_threshold|default:0.7}

steps:
  # 1. Planning
  - id: plan
    module: content_planner
    operation: plan_structure
    params:
      auto_approve: true
    input_from:
      query: inputs.query
      template_hints: inputs.template_hints
    output_as: plan
    enabled: true

  # 2. Research (parallel per sezione)
  - id: research
    module: multi_source_researcher
    operation: research_parallel
    params:
      extract_citations: true
    input_from:
      queries: plan.sections[*].suggested_queries
      source_config: inputs.source_config
    output_as: research_data
    enabled: true
    timeout: 120

  # 3. Generate Content
  - id: generate
    module: document_composer
    operation: generate_all_sections
    input_from:
      plan: plan
      research_results: research_data
    output_as: sections
    enabled: true
    timeout: 180

  # 4. Quality Check
  - id: quality_check
    module: document_composer
    operation: check_quality
    input_from:
      content: sections
    output_as: quality
    enabled: ${config.quality_check_enabled|default:true}

  # 5. Compose Document
  - id: compose
    module: document_composer
    operation: compose_document
    input_from:
      sections: sections
      citations: research_data.citations
    output_as: document
    enabled: true

  # 6. Generate Charts (if any)
  - id: charts
    module: chart_generator
    operation: create_charts_batch
    input_from:
      chart_specs: document.charts
    output_as: rendered_charts
    enabled: ${document.has_charts}
    error_strategy: skip

  # 7. Render Formats (parallel)
  - id: render_pdf
    module: document_renderer
    operation: render_pdf
    input_from:
      composed_doc: document
      charts: rendered_charts
    output_as: pdf_bytes
    parallel_group: render
    enabled: ${"pdf" in formats}

  - id: render_docx
    module: document_renderer
    operation: render_docx
    input_from:
      composed_doc: document
      charts: rendered_charts
    output_as: docx_bytes
    parallel_group: render
    enabled: ${"docx" in formats}

  # 8. Store Artifacts
  - id: store
    module: artifact_storage
    operation: store_batch
    input_from:
      artifacts:
        - content: pdf_bytes
          format: pdf
          title: document.title
        - content: docx_bytes
          format: docx
          title: document.title
    output_as: stored

output_mapping:
  artifacts: stored.artifacts
  quality_report: quality
  document_id: document.id

error_strategy: fail
timeout: 600
```

### Pipeline: Report Interactive (Con Approvazioni)

```yaml
name: report_interactive
version: "1.0.0"
description: "Generazione report interattiva con step di approvazione"

steps:
  # 1. Planning
  - id: plan
    module: content_planner
    operation: plan_structure
    input_from:
      query: inputs.query
    output_as: plan
    
  # 2. CHECKPOINT: Approvazione Piano
  - id: plan_approval
    module: _builtin
    operation: await_approval
    params:
      approval_type: plan
      timeout_minutes: 60
      allow_modifications: true
    input_from:
      data: plan
    output_as: approved_plan
    
  # 3. Research
  - id: research
    module: multi_source_researcher
    operation: research_parallel
    input_from:
      queries: approved_plan.sections[*].suggested_queries
    output_as: research_data

  # 4. Generate Sections (one by one with optional review)
  - id: generate_section
    module: document_composer
    operation: generate_section
    params:
      await_review: ${section.interactive_review}
    input_from:
      section_plan: approved_plan.sections
      research_data: research_data
    output_as: sections
    loop: approved_plan.sections
    
  # 5. CHECKPOINT: Review Draft
  - id: draft_review
    module: _builtin
    operation: await_approval
    params:
      approval_type: draft
      allow_edit: true
      allow_regenerate: true
    input_from:
      sections: sections
    output_as: reviewed_sections

  # 6. Improvement (if feedback provided)
  - id: improve
    module: document_composer
    operation: improve_sections
    input_from:
      sections: reviewed_sections.sections
      feedback: reviewed_sections.feedback
    output_as: final_sections
    condition: reviewed_sections.has_feedback

  # 7. Compose
  - id: compose
    module: document_composer
    operation: compose_document
    input_from:
      sections: final_sections
    output_as: document

  # 8. Format Selection
  - id: format_selection
    module: _builtin
    operation: await_input
    params:
      prompt: "Seleziona i formati di export"
      options: ["pdf", "docx", "pptx", "all"]
    output_as: selected_formats

  # 9. Render & Store
  - id: render_store
    module: document_renderer
    operation: render_multi
    input_from:
      document: document
      formats: selected_formats.formats
    output_as: rendered

  - id: store
    module: artifact_storage
    operation: store_batch
    input_from:
      artifacts: rendered.files
    output_as: stored

output_mapping:
  artifacts: stored.artifacts
  document: document
```

### Pipeline: Presentation Builder

```yaml
name: presentation_builder
version: "1.0.0"
description: "Creazione presentazione PowerPoint"

steps:
  # 1. Plan Slides
  - id: plan
    module: content_planner
    operation: plan_structure
    params:
      content_type: presentation
      max_sections: 20  # slides
    input_from:
      query: inputs.query
      template_hints: {type: "presentation"}
    output_as: slide_plan

  # 2. Research
  - id: research
    module: multi_source_researcher
    operation: research_parallel
    input_from:
      queries: slide_plan.sections[*].suggested_queries
    output_as: research_data

  # 3. Generate Slide Content
  - id: generate
    module: document_composer
    operation: generate_all_sections
    params:
      content_format: slide
      max_tokens_per_section: 200
    input_from:
      plan: slide_plan
      research_data: research_data
    output_as: slides

  # 4. Generate Charts for Data Slides
  - id: charts
    module: chart_generator
    operation: create_charts_batch
    input_from:
      chart_specs: slides.chart_specs
    output_as: charts
    enabled: ${slides.has_charts}

  # 5. Render PPTX
  - id: render
    module: document_renderer
    operation: render_pptx
    params:
      include_speaker_notes: true
    input_from:
      slides: slides
      charts: charts
      template: inputs.pptx_template
    output_as: pptx_bytes

  # 6. Store
  - id: store
    module: artifact_storage
    operation: store
    input_from:
      content: pptx_bytes
      metadata:
        format: pptx
        title: slide_plan.title
    output_as: artifact

output_mapping:
  artifact: artifact
  slide_count: slides.count
```

---

## 📊 Riepilogo Moduli

| Modulo | Linee Stimate | Riusabile | Priorità |
|--------|---------------|-----------|----------|
| `content_planner` | ~1,300 | ✅ Qualsiasi contenuto strutturato | 🔴 ALTA |
| `multi_source_researcher` | ~1,550 | ✅ Qualsiasi ricerca multi-fonte | 🔴 ALTA |
| `document_composer` | ~1,700 | ✅ Qualsiasi composizione contenuti | 🔴 ALTA |
| `document_renderer` | ~2,550 | ✅ Qualsiasi export documento | 🔴 ALTA |
| `chart_generator` | ~1,450 | ✅ Qualsiasi visualizzazione dati | 🟡 MEDIA |
| `artifact_storage` | ~1,250 | ✅ Qualsiasi file storage | 🔴 ALTA |
| **TOTALE** | **~9,800** | | |

---

## 🔧 Builtin Operations Aggiuntive

Per supportare interattività, servono nuove operazioni builtin:

```python
# Da aggiungere a pipeline_orchestrator/builtin_operations.py

INTERACTIVE_OPERATIONS = {
    "await_approval": {
        "description": "Attende approvazione utente prima di continuare",
        "parameters": {
            "approval_type": {"type": "string", "required": True},
            "data": {"type": "any", "required": True},
            "timeout_minutes": {"type": "integer", "default": 60},
            "allow_modifications": {"type": "boolean", "default": False}
        },
        "returns": {
            "approved": "boolean",
            "data": "any (possibly modified)",
            "modifications": "list of changes"
        }
    },
    
    "await_input": {
        "description": "Richiede input utente",
        "parameters": {
            "prompt": {"type": "string", "required": True},
            "input_type": {"type": "string", "default": "text"},
            "options": {"type": "array", "required": False},
            "validation": {"type": "object", "required": False}
        },
        "returns": {
            "value": "any",
            "timestamp": "datetime"
        }
    },
    
    "checkpoint": {
        "description": "Salva stato per possibile resume",
        "parameters": {
            "checkpoint_id": {"type": "string", "required": True},
            "data": {"type": "any", "required": True}
        },
        "returns": {
            "saved": "boolean",
            "checkpoint_id": "string"
        }
    }
}
```

---

## 📋 Dipendenze Python

```txt
# document_renderer
python-docx>=0.8.11
reportlab>=4.0.0          # PDF nativo
python-pptx>=0.6.21
Pillow>=9.0.0
svglib>=1.5.0

# chart_generator
matplotlib>=3.7.0
plotly>=5.14.0
pandas>=2.0.0

# artifact_storage
boto3>=1.26.0             # S3
aiofiles>=23.0.0

# general
pydantic>=2.0.0
aioredis>=2.0.0
PyYAML>=6.0
```

---

## ❓ Conferma per Procedere

Prima di implementare, conferma:

1. **Ordine implementazione:** 
   - Fase 1: `artifact_storage` + `document_renderer` (fondamentali)
   - Fase 2: `content_planner` + `multi_source_researcher`
   - Fase 3: `document_composer` + `chart_generator`
   
   Va bene questa sequenza?

2. **Interattività:** Le operazioni builtin `await_approval`, `await_input` come le vuoi gestire? 
   - A) Callback HTTP/WebSocket
   - B) Polling endpoint
   - C) Entrambi

3. **PDF nativo:** Confermi ReportLab come libreria? Alternative: WeasyPrint, xhtml2pdf

4. **Charts:** Confermi Matplotlib + Plotly? Alternative: Altair, Bokeh

5. **Immagini:** Per ora solo predisposizione (`ImageRef` nei data class) o vuoi già supporto base?

---

**Attendo conferma per iniziare implementazione.** 🚀
