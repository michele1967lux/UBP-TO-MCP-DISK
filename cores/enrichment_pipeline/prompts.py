"""
enrichment_pipeline/prompts.py

Enterprise-grade prompt templates for Advanced Retrieval (v2.2).

FEAT-HYDE-001: Hypothetical Document Embeddings
FEAT-EXPAND-001: Query Expansion with variants
FEAT-INVEST-001: Investigative Query Decomposition (v2.2.2)
FEAT-CLASSIFY-001: Unified Query Classification System (v2.2.3)

These prompts are optimized for RAG retrieval improvement.
"""

from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Unified Query Classification System (v2.2.3 - FEAT-CLASSIFY-001)
# =============================================================================
# Centralized keyword definitions for query classification.
# Used by both HyDE document type detection and Investigative type detection.
# Single source of truth - eliminates duplication and ensures consistency.

QUERY_CATEGORIES: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # AI/ML - Artificial Intelligence and Machine Learning
    # -------------------------------------------------------------------------
    "ai_ml": {
        "keywords": [
            "machine learning", "deep learning", "neural network", "artificial intelligence",
            "llm", "large language model", "transformer", "bert", "gpt", "embedding",
            "rag", "retrieval augmented", "vector", "similarity", "cosine",
            "training", "inference", "fine-tuning", "fine tuning", "prompt engineering",
            "token", "attention mechanism", "model", "algorithm", "dataset",
            "accuracy", "loss function", "gradient", "backpropagation",
            "classification", "regression", "clustering", "nlp", "natural language",
            "computer vision", "reinforcement learning", "supervised", "unsupervised",
        ],
        "exclusive": ["neural network", "deep learning", "machine learning", "llm", "rag"],
        "weight": 1.3,  # High priority for AI-specific queries
        "description": "AI/ML systems, models, training, and inference",
        "hyde_type": "ai_ml",
        "investigative_type": "ai_ml",
    },

    # -------------------------------------------------------------------------
    # API Integration - REST, GraphQL, Webhooks, Authentication
    # -------------------------------------------------------------------------
    "api_integration": {
        "keywords": [
            "rest api", "graphql", "webhook", "integration", "oauth", "oauth2",
            "jwt", "bearer token", "authentication", "authorization",
            "cors", "csrf", "rate limit", "throttle", "http method",
            "get request", "post request", "put request", "delete request",
            "request body", "response body", "http header", "payload",
            "json schema", "xml", "soap", "grpc", "websocket",
            "api key", "access token", "refresh token", "openapi", "swagger",
        ],
        "exclusive": ["rest api", "graphql", "webhook", "oauth", "jwt", "swagger", "openapi"],
        "shared": ["api", "endpoint", "request", "response"],  # Shared with system_admin
        "weight": 1.2,
        "description": "API design, integration, and authentication",
        "hyde_type": "technical",  # Maps to technical for HyDE
        "investigative_type": "api_integration",
    },

    # -------------------------------------------------------------------------
    # System Administration - Infrastructure, DevOps, Deployment
    # -------------------------------------------------------------------------
    "system_admin": {
        "keywords": [
            "install", "configure", "configuration", "setup", "deploy", "deployment",
            "server", "database", "service", "container", "docker", "kubernetes", "k8s",
            "cloud", "aws", "azure", "gcp", "monitoring", "logging", "log",
            "backup", "restore", "recovery", "security", "firewall", "network",
            "cluster", "scaling", "autoscaling", "load balancer", "infrastructure",
            "devops", "ci/cd", "pipeline", "ansible", "terraform", "helm",
            "nginx", "apache", "redis", "postgresql", "mysql", "mongodb",
            "environment variable", "env", "systemd", "cron", "daemon",
        ],
        "exclusive": ["docker", "kubernetes", "k8s", "terraform", "ansible", "devops"],
        "shared": ["api", "endpoint", "server", "database", "service"],
        "weight": 1.1,
        "description": "System administration, infrastructure, and DevOps",
        "hyde_type": "system_admin",
        "investigative_type": "system_admin",
    },

    # -------------------------------------------------------------------------
    # Troubleshooting - Error diagnosis, debugging, problem resolution
    # -------------------------------------------------------------------------
    "troubleshooting": {
        "keywords": [
            "error", "errore", "problem", "problema", "issue", "fail", "failed",
            "not working", "non funziona", "broken", "crash", "exception",
            "bug", "debug", "debugging", "fix", "resolve", "troubleshoot",
            "timeout", "connection error", "connection refused", "permission denied",
            "access denied", "unauthorized", "forbidden", "404", "500", "503",
            "stack trace", "traceback", "log error", "warning", "critical",
            "memory leak", "out of memory", "oom", "segfault", "core dump",
        ],
        "exclusive": ["stack trace", "traceback", "debug", "troubleshoot", "core dump"],
        "weight": 1.4,  # Highest priority - users need immediate help
        "description": "Error diagnosis, debugging, and problem resolution",
        "hyde_type": "troubleshooting",
        "investigative_type": "troubleshooting",
    },

    # -------------------------------------------------------------------------
    # Technical - Architecture, implementation, design patterns
    # -------------------------------------------------------------------------
    "technical": {
        "keywords": [
            "architecture", "implementation", "design pattern", "pattern",
            "protocol", "data structure", "performance", "optimization",
            "scalability", "concurrency", "threading", "async", "asynchronous",
            "synchronization", "mutex", "semaphore", "lock", "deadlock",
            "memory management", "garbage collection", "caching", "cache",
            "microservice", "monolith", "event driven", "message queue",
            "pub/sub", "kafka", "rabbitmq", "dependency injection",
            "solid principles", "clean architecture", "hexagonal",
        ],
        "exclusive": ["design pattern", "architecture", "microservice", "clean architecture"],
        "shared": ["api", "function", "method", "class", "parameter", "config"],
        "weight": 1.0,
        "description": "Software architecture, design patterns, and implementation",
        "hyde_type": "technical",
        "investigative_type": "technical",
    },

    # -------------------------------------------------------------------------
    # Conceptual - Theory, definitions, explanations
    # -------------------------------------------------------------------------
    "conceptual": {
        "keywords": [
            "what is", "cos'è", "che cos'è", "cosa significa",
            "how does", "come funziona", "why does", "perché",
            "explain", "spiegare", "spiega", "concept", "concetto",
            "theory", "teoria", "principle", "principio", "foundation",
            "overview", "panoramica", "introduction", "introduzione",
            "definition", "definizione", "meaning", "significato",
            "purpose", "scopo", "goal", "obiettivo", "objective",
            "difference between", "differenza tra", "compare", "confronta",
            "pros and cons", "vantaggi e svantaggi", "when to use",
        ],
        "exclusive": ["what is", "cos'è", "how does", "come funziona", "explain"],
        "weight": 0.9,  # Lower weight - often combined with other categories
        "description": "Conceptual explanations, definitions, and theory",
        "hyde_type": "article",  # Maps to article for HyDE
        "investigative_type": "conceptual",
    },

    # -------------------------------------------------------------------------
    # FAQ - Frequently asked questions patterns
    # -------------------------------------------------------------------------
    "faq": {
        "keywords": [
            "how do i", "come faccio", "how to", "come si fa",
            "can i", "posso", "is it possible", "è possibile",
            "what should", "cosa dovrei", "where can", "dove posso",
            "when should", "quando dovrei", "why should", "perché dovrei",
            "best way to", "modo migliore per", "recommended", "consigliato",
            "step by step", "passo passo", "tutorial", "guide", "guida",
            "example", "esempio", "sample", "demo",
        ],
        "exclusive": ["how do i", "come faccio", "step by step", "tutorial"],
        "weight": 1.0,
        "description": "How-to questions and practical guidance",
        "hyde_type": "faq",
        "investigative_type": "default",  # Maps to default for investigative
    },
}

# Type mappings for HyDE (which categories map to which HyDE template)
HYDE_TYPE_MAPPING: Dict[str, str] = {
    category: config["hyde_type"]
    for category, config in QUERY_CATEGORIES.items()
}

# Type mappings for Investigative (which categories map to which template)
INVESTIGATIVE_TYPE_MAPPING: Dict[str, str] = {
    category: config["investigative_type"]
    for category, config in QUERY_CATEGORIES.items()
}

# Default fallbacks
HYDE_DEFAULT_TYPE = "answer"
INVESTIGATIVE_DEFAULT_TYPE = "default"


def classify_query(
    query: str,
    categories: Optional[List[str]] = None,
    default_type: str = "default",
    use_weights: bool = True,
) -> Tuple[str, Dict[str, int]]:
    """
    Classify a query into one of the defined categories.

    This is the unified classification function used by both HyDE and
    Investigative Query Decomposition for consistent query understanding.

    Args:
        query: The user's query text
        categories: List of category names to consider (None = all)
        default_type: Default type to return if no matches
        use_weights: Whether to apply category weights

    Returns:
        Tuple of (winning_category, scores_dict)

    Example:
        >>> category, scores = classify_query("How do I configure Docker?")
        >>> print(category)  # "system_admin"
        >>> print(scores)    # {"system_admin": 2, "faq": 1, ...}
    """
    query_lower = query.lower()

    # Determine which categories to check
    if categories is None:
        categories_to_check = list(QUERY_CATEGORIES.keys())
    else:
        categories_to_check = [c for c in categories if c in QUERY_CATEGORIES]

    if not categories_to_check:
        return default_type, {}

    scores: Dict[str, float] = {}
    exclusive_winner: Optional[str] = None

    for category_name in categories_to_check:
        config = QUERY_CATEGORIES[category_name]
        keywords = config.get("keywords", [])
        exclusive = config.get("exclusive", [])
        weight = config.get("weight", 1.0) if use_weights else 1.0

        # Count keyword matches
        match_count = sum(1 for kw in keywords if kw in query_lower)

        # Check for exclusive keywords (auto-win)
        for excl_kw in exclusive:
            if excl_kw in query_lower:
                exclusive_winner = category_name
                break

        # Apply weight to score
        scores[category_name] = match_count * weight

    # If exclusive keyword found, that category wins
    if exclusive_winner and scores.get(exclusive_winner, 0) > 0:
        return exclusive_winner, {k: int(v) for k, v in scores.items()}

    # Find category with highest score
    if not scores or max(scores.values()) == 0:
        return default_type, {k: int(v) for k, v in scores.items()}

    # Get winner (in case of tie, first in order wins)
    winner = max(scores.keys(), key=lambda k: scores[k])

    return winner, {k: int(v) for k, v in scores.items()}


def get_hyde_category(query: str) -> str:
    """
    Get the HyDE document type for a query.

    Uses unified classification and maps to HyDE-specific template names.

    Args:
        query: User's query

    Returns:
        HyDE template type (e.g., "answer", "technical", "faq")
    """
    # Categories relevant for HyDE
    hyde_categories = ["ai_ml", "system_admin", "troubleshooting", "technical", "faq", "conceptual"]

    category, _ = classify_query(
        query,
        categories=hyde_categories,
        default_type="answer"
    )

    # Map to HyDE template type
    return HYDE_TYPE_MAPPING.get(category, HYDE_DEFAULT_TYPE)


def get_investigative_category(query: str) -> str:
    """
    Get the Investigative template type for a query.

    Uses unified classification and maps to Investigative-specific template names.

    Args:
        query: User's query

    Returns:
        Investigative template type (e.g., "default", "ai_ml", "api_integration")
    """
    # Categories relevant for Investigative
    investigative_categories = [
        "ai_ml", "api_integration", "system_admin",
        "troubleshooting", "technical", "conceptual"
    ]

    category, _ = classify_query(
        query,
        categories=investigative_categories,
        default_type="default"
    )

    # Map to Investigative template type
    return INVESTIGATIVE_TYPE_MAPPING.get(category, INVESTIGATIVE_DEFAULT_TYPE)


# =============================================================================
# HyDE (Hypothetical Document Embeddings) Prompts
# =============================================================================

HYDE_SYSTEM_PROMPT = """You are an expert AI assistant specialized in generating hypothetical documents for retrieval-augmented generation (RAG) systems.

IMPORTANT INSTRUCTIONS:
- Generate content that would naturally appear in a real knowledge base document
- Focus on factual, informative content that directly addresses the query
- Use technical terminology appropriate to the domain
- Structure the content as it would appear in official documentation
- Keep responses concise but comprehensive (2-4 paragraphs)
- Always generate in the SAME LANGUAGE as the user's query
- Avoid generic introductions or meta-commentary

Your goal is to create document content that, when embedded and searched, will help retrieve relevant real documents."""

HYDE_TEMPLATES: Dict[str, str] = {
    # Default: General answer document - Enhanced for RAG retrieval
    # ENTERPRISE: Language-agnostic - responds in query's language
    "answer": """Write a comprehensive hypothetical document that would perfectly answer this question. Structure it as content from a technical knowledge base or documentation.

CRITICAL: You MUST write your entire response in the EXACT SAME LANGUAGE as the user's query below. If the query is in Italian, respond in Italian. If in French, respond in French. If in German, respond in German. Match the query language precisely.

Focus on:
- Key concepts and definitions
- Step-by-step explanations
- Technical details and specifications
- Practical examples or use cases

Question: {query}

Hypothetical Knowledge Base Document:""",
    # Technical documentation style - Enhanced precision
    "technical": """Generate a technical documentation excerpt that provides authoritative information about this topic.

CRITICAL: You MUST write your entire response in the EXACT SAME LANGUAGE as the user's query below. Match the query language precisely.

Include:
- Precise technical definitions
- System architecture details
- Configuration parameters
- Implementation specifics
- API references or code examples where relevant

Technical Query: {query}

Official Documentation Excerpt:""",
    # FAQ style - More structured
    "faq": """Create a professional FAQ entry that provides a complete answer to this question.

CRITICAL: You MUST write your entire response in the EXACT SAME LANGUAGE as the user's query below. Match the query language precisely.

Structure it as:
Q: [Restate the question clearly in the same language]
A: [Comprehensive answer with details, examples, and additional context]

Question: {query}

Professional FAQ Entry:""",
    # Article/knowledge base style - Enhanced structure
    "article": """Write an informative knowledge base article excerpt that thoroughly covers this topic.

CRITICAL: You MUST write your entire response in the EXACT SAME LANGUAGE as the user's query below. Match the query language precisely.

Include:
- Overview and context
- Key components or features
- How it works (mechanisms, processes)
- Benefits and use cases
- Best practices or recommendations

Topic: {query}

Knowledge Base Article Excerpt:""",
    # Troubleshooting guide style - More actionable
    "troubleshooting": """Create a detailed troubleshooting guide section for this issue.

CRITICAL: You MUST write your entire response in the EXACT SAME LANGUAGE as the user's query below. Match the query language precisely.

Structure it with:
PROBLEM: [Clear description]
CAUSES: [List potential causes]
SOLUTIONS: [Step-by-step resolution steps]
PREVENTION: [Best practices to avoid]

Issue: {query}

Troubleshooting Guide:""",
    # AI/ML specific - New template for technical AI queries
    "ai_ml": """Write a technical document excerpt about this AI/ML concept, framework, or system.

CRITICAL: You MUST write your entire response in the EXACT SAME LANGUAGE as the user's query below. Match the query language precisely.

Include:
- Core concepts and theory
- Architecture and components
- Training/deployment details
- Performance characteristics
- Integration patterns

AI/ML Topic: {query}

Technical AI Documentation:""",
    # System administration - New template for infrastructure queries
    "system_admin": """Generate system administration documentation that addresses this operational or infrastructure question.

CRITICAL: You MUST write your entire response in the EXACT SAME LANGUAGE as the user's query below. Match the query language precisely.

Cover:
- System requirements and prerequisites
- Installation and configuration steps
- Monitoring and maintenance procedures
- Security considerations
- Troubleshooting common issues

System Query: {query}

System Administration Guide:""",
    # Italian language support - Enhanced
    "answer_it": """Scrivi un documento ipotetico completo che risponda perfettamente a questa domanda. Strutturalo come contenuto estratto da una knowledge base tecnica.

Concentrati su:
- Concetti chiave e definizioni
- Spiegazioni passo-passo
- Dettagli tecnici e specifiche
- Esempi pratici o casi d'uso

Domanda: {query}

Documento della Knowledge Base:""",
    # Italian technical - New
    "technical_it": """Genera un estratto di documentazione tecnica che fornisca informazioni autorevoli su questo argomento. Includi:

- Definizioni tecniche precise
- Dettagli sull'architettura di sistema
- Parametri di configurazione
- Specifiche di implementazione
- Riferimenti API o esempi di codice se pertinenti

Query Tecnica: {query}

Estratto Documentazione Ufficiale:""",
    # -------------------------------------------------------------------------
    # Additional Italian Templates (v2.2.3 - FEAT-CLASSIFY-001)
    # -------------------------------------------------------------------------
    # System administration - Italian
    "system_admin_it": """Genera documentazione di amministrazione di sistema che affronti questa domanda operativa o infrastrutturale. Copri:

- Requisiti di sistema e prerequisiti
- Passaggi di installazione e configurazione
- Procedure di monitoraggio e manutenzione
- Considerazioni sulla sicurezza
- Risoluzione dei problemi comuni

Query Sistema: {query}

Guida Amministrazione Sistema:""",
    # Troubleshooting - Italian
    "troubleshooting_it": """Crea una sezione dettagliata della guida alla risoluzione dei problemi per questo issue. Strutturala con:

PROBLEMA: [Descrizione chiara]
CAUSE: [Lista delle possibili cause]
SOLUZIONI: [Passaggi di risoluzione step-by-step]
PREVENZIONE: [Best practice per evitare il problema]

Issue: {query}

Guida alla Risoluzione:""",
    # AI/ML - Italian
    "ai_ml_it": """Scrivi un estratto di documento tecnico su questo concetto, framework o sistema AI/ML. Includi:

- Concetti e teoria di base
- Architettura e componenti
- Dettagli di training/deployment
- Caratteristiche di performance
- Pattern di integrazione

Argomento AI/ML: {query}

Documentazione Tecnica AI:""",
    # FAQ - Italian
    "faq_it": """Crea una voce FAQ professionale che fornisca una risposta completa a questa domanda. Strutturala come:

D: [Riformula la domanda chiaramente]
R: [Risposta completa con dettagli, esempi e contesto aggiuntivo]

Domanda: {query}

Voce FAQ Professionale:""",
    # Article - Italian
    "article_it": """Scrivi un estratto di articolo della knowledge base che copra a fondo questo argomento. Includi:

- Panoramica e contesto
- Componenti o caratteristiche principali
- Come funziona (meccanismi, processi)
- Benefici e casi d'uso
- Best practice o raccomandazioni

Argomento: {query}

Estratto Articolo Knowledge Base:""",
}


# =============================================================================
# Query Expansion Prompts
# =============================================================================

EXPANSION_SYSTEM_PROMPT = """You are an AI search assistant. Generate different search queries to find information about the user's question. Focus on technical terms, synonyms, and related concepts.

IMPORTANT: Always generate the expanded queries in the SAME LANGUAGE as the user's original query. If the query is in Italian, generate Italian variants. If in English, generate English variants.

Return ONLY a JSON list of strings."""

EXPANSION_TEMPLATES: Dict[str, str] = {
    # Semantic expansion (synonyms, related terms)
    "semantic": """Generate {n} different search queries to find information about the user's question. Focus on:
- Synonyms and alternative phrasings
- Technical terms related to the topic
- Related concepts that might contain the answer

User question: {query}

Return ONLY a JSON array of strings, no explanation:""",
    # Keyword extraction expansion
    "keywords": """Extract {n} different keyword-focused search queries from the user's question. Focus on:
- Key technical terms
- Entity names
- Specific concepts mentioned or implied

User question: {query}

Return ONLY a JSON array of strings:""",
    # Reformulation expansion
    "reformulate": """Reformulate the user's question in {n} different ways that might help find relevant documents:
- Rephrase as a statement
- Ask from different angles
- Use more specific or more general terms

User question: {query}

Return ONLY a JSON array of strings:""",
    # Italian language support
    "semantic_it": """Genera {n} diverse query di ricerca per trovare informazioni sulla domanda dell'utente. Concentrati su:
- Sinonimi e formulazioni alternative
- Termini tecnici relativi all'argomento
- Concetti correlati che potrebbero contenere la risposta

Domanda utente: {query}

Restituisci SOLO un array JSON di stringhe, senza spiegazioni:""",
}


# =============================================================================
# Investigative Query Decomposition (v2.2.2 - FEAT-INVEST-001)
# =============================================================================
# Alternative to HyDE: Instead of generating a hypothetical answer,
# generate specific search questions that would lead to the answer.
# This avoids hallucination on unknown terms and works domain-agnostically.

INVESTIGATIVE_SYSTEM_PROMPT = """You are an expert AI research assistant specialized in investigative query decomposition for enterprise knowledge bases. Your goal is to break down complex queries into specific, targeted search questions that will efficiently retrieve relevant information from technical documentation.

IMPORTANT PRINCIPLES:
- Generate questions that are specific and actionable
- Focus on factual, technical aspects that can be found in documentation
- Avoid questions that require opinion or speculation
- Always generate questions in the SAME LANGUAGE as the user's query
- Ensure questions are diverse but related to the core topic
- Prioritize questions that would appear in technical manuals or API docs

Return ONLY a valid JSON array of strings."""

INVESTIGATIVE_TEMPLATES: Dict[str, str] = {
    # Default investigative decomposition - Enhanced structure
    # ENTERPRISE: Language-agnostic - generates questions in query's language
    "default": """You are an expert research assistant analyzing: '{query}'

CRITICAL: You MUST generate ALL questions in the EXACT SAME LANGUAGE as the user's query above. If the query is in Italian, generate Italian questions. If in French, generate French questions. Match the query language precisely.

Break this down into {n} specific investigative questions that would help locate precise information in technical documentation. Focus on:

FOUNDATIONAL QUESTIONS:
- What is [core concept]? (definition and scope)
- What are the main components of [subject]?
- What is the primary purpose/function of [subject]?

TECHNICAL QUESTIONS:
- How is [subject] implemented/configured?
- What are the key parameters/settings for [subject]?
- What are the requirements/prerequisites for [subject]?

OPERATIONAL QUESTIONS:
- How do you use [subject] in practice?
- What are common issues/troubleshooting for [subject]?
- What are best practices for [subject]?

Return ONLY a JSON array of strings with specific, searchable questions:""",
    # AI/ML specific investigation - New template
    "ai_ml": """You are investigating an AI/ML system or concept: '{query}'

CRITICAL: You MUST generate ALL questions in the EXACT SAME LANGUAGE as the user's query above. Match the query language precisely.

Generate {n} targeted questions to find technical details in ML/AI documentation:

MODEL & ARCHITECTURE:
- What is the architecture of [AI/ML system]?
- What algorithms/models does [system] use?
- What are the training requirements for [system]?

DATA & PERFORMANCE:
- What data formats does [system] support?
- What are the performance metrics for [system]?
- What are the computational requirements for [system]?

INTEGRATION & USAGE:
- How do you integrate [system] with existing workflows?
- What APIs/endpoints does [system] provide?
- How do you monitor and maintain [system]?

Return ONLY a JSON array of strings:""",
    # System administration investigation - New template
    "system_admin": """You are investigating a system administration or infrastructure topic: '{query}'

CRITICAL: You MUST generate ALL questions in the EXACT SAME LANGUAGE as the user's query above. Match the query language precisely.

Generate {n} specific questions to find operational and configuration details:

DEPLOYMENT & SETUP:
- How do you install/configure [system/component]?
- What are the system requirements for [system/component]?
- How do you initialize/start [system/component]?

MONITORING & MAINTENANCE:
- How do you monitor the health/status of [system/component]?
- What are the backup/recovery procedures for [system/component]?
- How do you troubleshoot common issues with [system/component]?

SECURITY & COMPLIANCE:
- What security measures are implemented in [system/component]?
- How do you configure access controls for [system/component]?
- What compliance standards does [system/component] meet?

Return ONLY a JSON array of strings:""",
    # API/Integration investigation - New template
    "api_integration": """You are investigating API or integration functionality: '{query}'

CRITICAL: You MUST generate ALL questions in the EXACT SAME LANGUAGE as the user's query above. Match the query language precisely.

Generate {n} specific questions about integration and API usage:

API ENDPOINTS & METHODS:
- What API endpoints are available for [functionality]?
- What HTTP methods does [API] support?
- What authentication mechanisms does [API] use?

DATA FORMATS & SCHEMAS:
- What data formats does [API] accept/return?
- What are the request/response schemas for [API]?
- How do you handle errors in [API] calls?

RATE LIMITS & PERFORMANCE:
- What are the rate limits for [API]?
- How do you optimize performance with [API]?
- What are the SLA/guarantees for [API]?

Return ONLY a JSON array of strings:""",
    # Technical investigation - Enhanced
    "technical": """You are a technical researcher investigating: '{query}'

CRITICAL: You MUST generate ALL questions in the EXACT SAME LANGUAGE as the user's query above. Match the query language precisely.

Generate {n} precise technical questions to find detailed specifications:

ARCHITECTURAL QUESTIONS:
- What is the detailed architecture of [system/component]?
- How do the components of [system] interact?
- What design patterns does [system] follow?

IMPLEMENTATION QUESTIONS:
- How is [system/component] implemented internally?
- What protocols/standards does [system] use?
- What are the performance characteristics of [system]?

CONFIGURATION QUESTIONS:
- How do you configure [system/component] for production?
- What are the tuning parameters for [system]?
- How do you scale [system/component]?

Return ONLY a JSON array of strings:""",
    # Conceptual investigation - Enhanced
    "conceptual": """You are researching a conceptual or theoretical topic: '{query}'

CRITICAL: You MUST generate ALL questions in the EXACT SAME LANGUAGE as the user's query above. Match the query language precisely.

Generate {n} questions to build comprehensive understanding:

DEFINITION & SCOPE:
- What is the formal definition of [concept]?
- What is the scope and boundaries of [concept]?
- How does [concept] relate to other similar concepts?

THEORY & PRINCIPLES:
- What are the theoretical foundations of [concept]?
- What are the key principles behind [concept]?
- How has [concept] evolved over time?

PRACTICAL APPLICATIONS:
- What are real-world applications of [concept]?
- How do you measure/evaluate [concept]?
- What are the limitations of [concept]?

Return ONLY a JSON array of strings:""",
    # Troubleshooting investigation - New template
    "troubleshooting": """You are diagnosing issues with: '{query}'

CRITICAL: You MUST generate ALL questions in the EXACT SAME LANGUAGE as the user's query above. Match the query language precisely.

Generate {n} specific questions to identify and resolve problems:

DIAGNOSIS QUESTIONS:
- What are common symptoms of issues with [system/component]?
- How do you identify the root cause of [problem]?
- What diagnostic tools/logs are available for [system]?

RESOLUTION QUESTIONS:
- What are the standard fixes for [common problem]?
- How do you recover from [failure scenario]?
- What preventive measures exist for [problem]?

ESCALATION QUESTIONS:
- When should you escalate [issue] to specialists?
- What additional information is needed to resolve [problem]?
- How do you document [issue] for future reference?

Return ONLY a JSON array of strings:""",
    # Italian language support - Enhanced
    "default_it": """Sei un esperto assistente di ricerca che analizza: '{query}'

Suddividi questo argomento in {n} domande investigative specifiche che aiuterebbero a localizzare informazioni precise nella documentazione tecnica. Concentrati su:

DOMANDE FONDAMENTALI:
- Che cos'è [concetto principale]? (definizione e ambito)
- Quali sono i componenti principali di [oggetto]?
- Qual è lo scopo/funzione principale di [oggetto]?

DOMANDE TECNICHE:
- Come viene implementato/configurato [oggetto]?
- Quali sono i parametri/impostazioni chiave per [oggetto]?
- Quali sono i requisiti/prerequisiti per [oggetto]?

DOMANDE OPERATIVE:
- Come si usa [oggetto] nella pratica?
- Quali sono i problemi comuni/risoluzione problemi per [oggetto]?
- Quali sono le best practice per [oggetto]?

Restituisci SOLO un array JSON di stringhe con domande specifiche e ricercabili:""",
    # -------------------------------------------------------------------------
    # Italian Templates (v2.2.3 - FEAT-CLASSIFY-001)
    # -------------------------------------------------------------------------
    # Technical investigation - Italian
    "technical_it": """Sei un ricercatore tecnico che indaga: '{query}'

Genera {n} domande tecniche precise per trovare specifiche dettagliate:

DOMANDE ARCHITETTURALI:
- Qual è l'architettura dettagliata di [sistema/componente]?
- Come interagiscono i componenti di [sistema]?
- Quali design pattern segue [sistema]?

DOMANDE IMPLEMENTATIVE:
- Come è implementato internamente [sistema/componente]?
- Quali protocolli/standard utilizza [sistema]?
- Quali sono le caratteristiche di performance di [sistema]?

DOMANDE DI CONFIGURAZIONE:
- Come si configura [sistema/componente] per la produzione?
- Quali sono i parametri di tuning per [sistema]?
- Come si scala [sistema/componente]?

Restituisci SOLO un array JSON di stringhe:""",
    # System administration - Italian
    "system_admin_it": """Stai investigando un argomento di amministrazione di sistema o infrastruttura: '{query}'

Genera {n} domande specifiche per trovare dettagli operativi e di configurazione:

DEPLOYMENT E SETUP:
- Come si installa/configura [sistema/componente]?
- Quali sono i requisiti di sistema per [sistema/componente]?
- Come si inizializza/avvia [sistema/componente]?

MONITORAGGIO E MANUTENZIONE:
- Come si monitora lo stato/salute di [sistema/componente]?
- Quali sono le procedure di backup/recovery per [sistema/componente]?
- Come si risolvono i problemi comuni con [sistema/componente]?

SICUREZZA E COMPLIANCE:
- Quali misure di sicurezza sono implementate in [sistema/componente]?
- Come si configurano i controlli di accesso per [sistema/componente]?
- Quali standard di compliance rispetta [sistema/componente]?

Restituisci SOLO un array JSON di stringhe:""",
    # Troubleshooting - Italian
    "troubleshooting_it": """Stai diagnosticando problemi con: '{query}'

Genera {n} domande specifiche per identificare e risolvere i problemi:

DOMANDE DI DIAGNOSI:
- Quali sono i sintomi comuni dei problemi con [sistema/componente]?
- Come si identifica la causa principale di [problema]?
- Quali strumenti diagnostici/log sono disponibili per [sistema]?

DOMANDE DI RISOLUZIONE:
- Quali sono le soluzioni standard per [problema comune]?
- Come si recupera da [scenario di errore]?
- Quali misure preventive esistono per [problema]?

DOMANDE DI ESCALATION:
- Quando si deve escalare [problema] agli specialisti?
- Quali informazioni aggiuntive servono per risolvere [problema]?
- Come si documenta [problema] per riferimento futuro?

Restituisci SOLO un array JSON di stringhe:""",
    # AI/ML - Italian
    "ai_ml_it": """Stai investigando un sistema o concetto AI/ML: '{query}'

Genera {n} domande mirate per trovare dettagli tecnici nella documentazione ML/AI:

MODELLO E ARCHITETTURA:
- Qual è l'architettura di [sistema AI/ML]?
- Quali algoritmi/modelli utilizza [sistema]?
- Quali sono i requisiti di training per [sistema]?

DATI E PERFORMANCE:
- Quali formati di dati supporta [sistema]?
- Quali sono le metriche di performance per [sistema]?
- Quali sono i requisiti computazionali per [sistema]?

INTEGRAZIONE E UTILIZZO:
- Come si integra [sistema] con workflow esistenti?
- Quali API/endpoint fornisce [sistema]?
- Come si monitora e mantiene [sistema]?

Restituisci SOLO un array JSON di stringhe:""",
    # API Integration - Italian
    "api_integration_it": """Stai investigando funzionalità API o di integrazione: '{query}'

Genera {n} domande specifiche sull'integrazione e l'uso delle API:

ENDPOINT E METODI API:
- Quali endpoint API sono disponibili per [funzionalità]?
- Quali metodi HTTP supporta [API]?
- Quali meccanismi di autenticazione usa [API]?

FORMATI DATI E SCHEMA:
- Quali formati dati accetta/restituisce [API]?
- Quali sono gli schema request/response per [API]?
- Come si gestiscono gli errori nelle chiamate [API]?

RATE LIMIT E PERFORMANCE:
- Quali sono i rate limit per [API]?
- Come si ottimizza la performance con [API]?
- Quali sono le SLA/garanzie per [API]?

Restituisci SOLO un array JSON di stringhe:""",
    # Conceptual - Italian
    "conceptual_it": """Stai ricercando un argomento concettuale o teorico: '{query}'

Genera {n} domande per costruire una comprensione completa:

DEFINIZIONE E AMBITO:
- Qual è la definizione formale di [concetto]?
- Qual è l'ambito e i limiti di [concetto]?
- Come si relaziona [concetto] con altri concetti simili?

TEORIA E PRINCIPI:
- Quali sono le fondamenta teoriche di [concetto]?
- Quali sono i principi chiave dietro [concetto]?
- Come si è evoluto [concetto] nel tempo?

APPLICAZIONI PRATICHE:
- Quali sono le applicazioni reali di [concetto]?
- Come si misura/valuta [concetto]?
- Quali sono i limiti di [concetto]?

Restituisci SOLO un array JSON di stringhe:""",
}


# =============================================================================
# Combined Optimization Prompt
# =============================================================================

OPTIMIZE_QUERY_PROMPT = """You are an expert search query optimizer. Given a user's question, generate optimized search queries that will help find relevant documents in a knowledge base.

User question: {query}

Generate:
1. A hypothetical answer document (2-3 sentences) that would perfectly answer this question
2. {n} alternative search queries using different terminology

Return as JSON:
{{
    "hyde_document": "hypothetical answer text",
    "expanded_queries": ["query1", "query2", "query3"]
}}

Response:"""


# =============================================================================
# Natural Language Filter Extraction Prompt (Q2F)
# =============================================================================

FILTER_EXTRACTION_PROMPT = """You are a metadata filter extraction assistant for an enterprise knowledge base. Convert the user's natural language request into structured filters that can be applied to document metadata.

Only use the following allowed fields: {allowed_fields}.
Supported operators: equals, in. Always return strict JSON with this schema:
{{
    "filters": [
        {{"field": "filename", "operator": "equals", "value": "MANUAL_03_ARCH.md"}}
    ],
    "entities": {{"filename": ["MANUAL_03_ARCH.md"]}},
    "confidence": 0.92,
    "reason": "User explicitly requested that filename"
}}

Rules:
- If the query mentions specific files ("nel file MANUAL"), map to filename equals.
- If multiple files are mentioned, use operator "in" with a list value.
- If no concrete constraint is found, return an empty list for "filters".
- Never hallucinate filenames or knowledge bases.
- CRITICAL: If a field name appears as the SUBJECT of the question (what the user is asking ABOUT), do NOT create a filter on that field. Only filter when the user provides a concrete VALUE to match against.
  Example: "come viene propagato kb_id?" → NO filter (kb_id is the topic, not a value).
  Example: "cerca nei documenti con tag security" → filter tags=security (concrete value).
  Example: "cosa fa il campo filename?" → NO filter (filename is the topic).

User query: {query}

Response:"""


# =============================================================================
# Utility Functions
# =============================================================================


def get_hyde_prompt(query: str, document_type: str = "answer") -> str:
    """
    Get formatted HyDE prompt for the given query and document type.

    Args:
        query: User's original query
        document_type: Type of hypothetical document to generate

    Returns:
        Formatted prompt string
    """
    template = HYDE_TEMPLATES.get(document_type, HYDE_TEMPLATES["answer"])
    return template.format(query=query)


def get_expansion_prompt(
    query: str, n: int = 3, expansion_type: str = "semantic"
) -> str:
    """
    Get formatted query expansion prompt.

    Args:
        query: User's original query
        n: Number of variants to generate
        expansion_type: Type of expansion (semantic, keywords, reformulate)

    Returns:
        Formatted prompt string
    """
    template = EXPANSION_TEMPLATES.get(expansion_type, EXPANSION_TEMPLATES["semantic"])
    return template.format(query=query, n=n)


def get_optimize_prompt(query: str, n: int = 3) -> str:
    """
    Get combined optimization prompt for HyDE + Expansion.

    Args:
        query: User's original query
        n: Number of expansion variants

    Returns:
        Formatted prompt string
    """
    return OPTIMIZE_QUERY_PROMPT.format(query=query, n=n)


def get_filter_prompt(query: str, allowed_fields: List[str]) -> str:
    """Get formatted prompt for natural language → metadata filters."""
    allowed = ", ".join(allowed_fields)
    return FILTER_EXTRACTION_PROMPT.format(query=query, allowed_fields=allowed)


def get_investigative_prompt(
    query: str, n: int = 5, investigation_type: str = "default"
) -> str:
    """
    Get formatted investigative query decomposition prompt (v2.2.2).

    Instead of generating a hypothetical answer (HyDE), this generates
    specific search questions that would lead to finding the answer.
    This approach avoids hallucination on unknown terms.

    Args:
        query: User's original query
        n: Number of investigative questions to generate (default: 5)
        investigation_type: Type of investigation (default, technical, conceptual)

    Returns:
        Formatted prompt string
    """
    # Auto-detect Italian
    lang_suffix = detect_language_suffix(query)
    type_key = (
        f"{investigation_type}{lang_suffix}" if lang_suffix else investigation_type
    )

    template = INVESTIGATIVE_TEMPLATES.get(type_key, INVESTIGATIVE_TEMPLATES["default"])
    return template.format(query=query, n=n)


# =============================================================================
# Language Detection Helper
# =============================================================================


def detect_language_suffix(query: str) -> str:
    """
    Simple language detection for prompt selection.
    Returns '_it' for Italian, '' for English/default.

    Args:
        query: User's query text

    Returns:
        Language suffix for prompt templates
    """
    # Simple Italian detection based on common words
    italian_markers = [
        "come",
        "cosa",
        "perché",
        "quando",
        "dove",
        "chi",
        "non",
        "sono",
        "essere",
        "fare",
        "avere",
        "questo",
        "della",
        "delle",
        "degli",
        "nella",
        "nelle",
    ]

    query_lower = query.lower()
    italian_count = sum(1 for marker in italian_markers if marker in query_lower)

    return "_it" if italian_count >= 2 else ""
