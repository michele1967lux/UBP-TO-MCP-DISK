"""
investigation_pipeline/prompts.py

Enterprise-grade prompt templates for RAG Investigation Engine (v1.0).

FEAT-INVEST-001: Multi-Strategy Investigation Generation
FEAT-QA-001: Quality Assurance Validation Prompts
FEAT-CLASSIFY-001: Query Classification System

Strategies:
- Decomposition: Break query into aspects
- Chain-of-Thought: Logical reasoning steps
- Semantic Expansion: Synonyms and related concepts
- Cross-Reference: Dependencies and related features
- Adaptive: Auto-select based on classification

Multi-language support: EN, IT
"""

from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Query Classification System (FEAT-CLASSIFY-001)
# =============================================================================
# Centralized keyword definitions for query classification.
# Single source of truth for strategy selection.

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
            "langchain", "llamaindex", "huggingface", "ollama", "vllm",
        ],
        "exclusive": ["neural network", "deep learning", "machine learning", "llm", "rag", "embedding"],
        "weight": 1.3,
        "description": "AI/ML systems, models, training, and inference",
        "preferred_strategy": "decomposition",
        "secondary_strategy": "semantic_expansion",
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
            "why", "perché", "doesn't work", "non funziona",
        ],
        "exclusive": ["stack trace", "traceback", "debug", "troubleshoot", "core dump", "error"],
        "weight": 1.4,
        "description": "Error diagnosis, debugging, and problem resolution",
        "preferred_strategy": "chain_of_thought",
        "secondary_strategy": "decomposition",
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
            "endpoint", "route", "middleware",
        ],
        "exclusive": ["rest api", "graphql", "webhook", "oauth", "jwt", "swagger", "openapi"],
        "weight": 1.2,
        "description": "API design, integration, and authentication",
        "preferred_strategy": "decomposition",
        "secondary_strategy": "cross_reference",
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
        "weight": 1.1,
        "description": "System administration, infrastructure, and DevOps",
        "preferred_strategy": "decomposition",
        "secondary_strategy": "chain_of_thought",
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
            "repository", "factory", "singleton", "observer",
        ],
        "exclusive": ["design pattern", "architecture", "microservice", "clean architecture"],
        "weight": 1.0,
        "description": "Software architecture, design patterns, and implementation",
        "preferred_strategy": "decomposition",
        "secondary_strategy": "semantic_expansion",
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
        "exclusive": ["what is", "cos'è", "how does", "come funziona", "explain", "definition"],
        "weight": 0.9,
        "description": "Conceptual explanations, definitions, and theory",
        "preferred_strategy": "semantic_expansion",
        "secondary_strategy": "decomposition",
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
        "exclusive": ["how do i", "come faccio", "step by step", "tutorial", "how to"],
        "weight": 1.0,
        "description": "How-to questions and practical guidance",
        "preferred_strategy": "chain_of_thought",
        "secondary_strategy": "decomposition",
    },
}

# Default category when no match
DEFAULT_CATEGORY = "technical"
DEFAULT_STRATEGY = "decomposition"


# =============================================================================
# Decomposition Strategy Templates (FEAT-INVEST-001)
# =============================================================================
# Breaks the query into multiple aspects for comprehensive coverage.

DECOMPOSITION_TEMPLATES: Dict[str, str] = {
    "default": """You are an expert research assistant specializing in investigative query decomposition.

Given a user query, generate {n} highly specific investigative search questions that will help find comprehensive information in a knowledge base.

DECOMPOSITION ASPECTS:
1. DEFINITION: What is [concept]? How is it defined?
2. COMPONENTS: What are the main parts/elements of [concept]?
3. PURPOSE: Why is [concept] used? What problems does it solve?
4. IMPLEMENTATION: How is [concept] implemented/configured?
5. EXAMPLES: What are real-world examples of [concept]?
6. ALTERNATIVES: What alternatives exist to [concept]?
7. BEST PRACTICES: What are recommended practices for [concept]?

RULES:
- Each question must end with '?'
- Questions must be specific and searchable
- Avoid vague or overly broad questions
- Focus on information that would be in documentation
- Do NOT generate hypothetical answers

User Query: {query}

Generate exactly {n} investigative questions as a JSON array:""",

    "ai_ml": """You are an AI/ML expert assistant specializing in machine learning system investigation.

Given a user query about AI/ML systems, generate {n} highly specific investigative search questions.

ML-SPECIFIC DECOMPOSITION:
1. MODEL ARCHITECTURE: What architecture/model type is used?
2. TRAINING: How is the model trained? What data is needed?
3. INFERENCE: How is the model deployed for inference?
4. PARAMETERS: What are the key hyperparameters/configurations?
5. INTEGRATION: How does it integrate with other systems?
6. PERFORMANCE: How is performance measured/optimized?
7. COMMON ISSUES: What are typical problems and solutions?

RULES:
- Focus on technical ML terminology
- Include questions about model behavior
- Consider data pipeline aspects
- Each question must end with '?'

User Query: {query}

Generate exactly {n} investigative questions as a JSON array:""",

    "api_integration": """You are an API integration expert assistant.

Given a user query about API integration, generate {n} highly specific investigative search questions.

API-SPECIFIC DECOMPOSITION:
1. ENDPOINTS: What endpoints are available? What do they do?
2. AUTHENTICATION: How is authentication/authorization handled?
3. REQUEST FORMAT: What is the request format/schema?
4. RESPONSE FORMAT: What is the response format/schema?
5. ERROR HANDLING: What errors can occur? How to handle them?
6. RATE LIMITS: What are the rate limits and quotas?
7. EXAMPLES: What are working code examples?

RULES:
- Focus on API-specific terminology
- Include questions about request/response formats
- Consider error scenarios
- Each question must end with '?'

User Query: {query}

Generate exactly {n} investigative questions as a JSON array:""",

    "system_admin": """You are a DevOps/SysAdmin expert assistant.

Given a user query about system administration, generate {n} highly specific investigative search questions.

SYSADMIN-SPECIFIC DECOMPOSITION:
1. INSTALLATION: How to install/deploy?
2. CONFIGURATION: What configuration options exist?
3. PREREQUISITES: What dependencies/requirements are needed?
4. MONITORING: How to monitor health and performance?
5. TROUBLESHOOTING: What are common issues and fixes?
6. SECURITY: What security considerations apply?
7. SCALING: How to scale and optimize?

RULES:
- Focus on operational aspects
- Include infrastructure considerations
- Consider security implications
- Each question must end with '?'

User Query: {query}

Generate exactly {n} investigative questions as a JSON array:""",

    "troubleshooting": """You are an expert troubleshooter and debugger.

Given a user query about an error or problem, generate {n} highly specific investigative search questions to diagnose the issue.

TROUBLESHOOTING DECOMPOSITION:
1. ERROR MESSAGE: What does the exact error message mean?
2. ROOT CAUSE: What typically causes this error?
3. PREREQUISITES: What conditions must be met?
4. CONFIGURATION: What configuration might be wrong?
5. LOGS: Where are relevant logs? What to look for?
6. SIMILAR ISSUES: What similar issues exist and their solutions?
7. WORKAROUNDS: What workarounds or fixes exist?

RULES:
- Focus on diagnostic questions
- Include log/debug information queries
- Consider environmental factors
- Each question must end with '?'

User Query: {query}

Generate exactly {n} investigative questions as a JSON array:""",

    "conceptual": """You are an expert educator explaining technical concepts.

Given a user query about a concept, generate {n} highly specific investigative search questions for comprehensive understanding.

CONCEPTUAL DECOMPOSITION:
1. DEFINITION: What is the precise definition?
2. HISTORY: How did this concept evolve?
3. PRINCIPLES: What are the underlying principles?
4. APPLICATIONS: What are practical applications?
5. COMPARISONS: How does it compare to alternatives?
6. LIMITATIONS: What are the limitations?
7. RESOURCES: Where to learn more?

RULES:
- Focus on foundational understanding
- Include comparative questions
- Consider learning progression
- Each question must end with '?'

User Query: {query}

Generate exactly {n} investigative questions as a JSON array:""",

    # Italian variants
    "default_it": """Sei un assistente di ricerca esperto specializzato nella decomposizione investigativa delle query.

Data una query utente, genera {n} domande di ricerca investigative altamente specifiche che aiuteranno a trovare informazioni complete in una knowledge base.

ASPETTI DI DECOMPOSIZIONE:
1. DEFINIZIONE: Cos'è [concetto]? Come viene definito?
2. COMPONENTI: Quali sono le parti/elementi principali di [concetto]?
3. SCOPO: Perché viene usato [concetto]? Quali problemi risolve?
4. IMPLEMENTAZIONE: Come viene implementato/configurato [concetto]?
5. ESEMPI: Quali sono esempi reali di [concetto]?
6. ALTERNATIVE: Quali alternative esistono a [concetto]?
7. BEST PRACTICES: Quali sono le pratiche raccomandate per [concetto]?

REGOLE:
- Ogni domanda deve terminare con '?'
- Le domande devono essere specifiche e ricercabili
- Evita domande vaghe o troppo ampie
- Concentrati su informazioni presenti nella documentazione
- NON generare risposte ipotetiche

Query Utente: {query}

Genera esattamente {n} domande investigative come array JSON:""",

    "ai_ml_it": """Sei un esperto assistente AI/ML specializzato nell'investigazione di sistemi di machine learning.

Data una query utente su sistemi AI/ML, genera {n} domande di ricerca investigative altamente specifiche.

DECOMPOSIZIONE SPECIFICA ML:
1. ARCHITETTURA MODELLO: Quale architettura/tipo di modello viene usato?
2. TRAINING: Come viene addestrato il modello? Quali dati servono?
3. INFERENCE: Come viene deployato il modello per l'inference?
4. PARAMETRI: Quali sono gli iperparametri/configurazioni chiave?
5. INTEGRAZIONE: Come si integra con altri sistemi?
6. PERFORMANCE: Come vengono misurate/ottimizzate le performance?
7. PROBLEMI COMUNI: Quali sono i problemi tipici e le soluzioni?

REGOLE:
- Concentrati sulla terminologia tecnica ML
- Includi domande sul comportamento del modello
- Considera aspetti della data pipeline
- Ogni domanda deve terminare con '?'

Query Utente: {query}

Genera esattamente {n} domande investigative come array JSON:""",

    "troubleshooting_it": """Sei un esperto troubleshooter e debugger.

Data una query utente su un errore o problema, genera {n} domande di ricerca investigative altamente specifiche per diagnosticare il problema.

DECOMPOSIZIONE TROUBLESHOOTING:
1. MESSAGGIO ERRORE: Cosa significa esattamente il messaggio di errore?
2. CAUSA ROOT: Cosa causa tipicamente questo errore?
3. PREREQUISITI: Quali condizioni devono essere soddisfatte?
4. CONFIGURAZIONE: Quale configurazione potrebbe essere sbagliata?
5. LOG: Dove sono i log rilevanti? Cosa cercare?
6. PROBLEMI SIMILI: Quali problemi simili esistono e quali sono le soluzioni?
7. WORKAROUND: Quali workaround o fix esistono?

REGOLE:
- Concentrati su domande diagnostiche
- Includi query su informazioni di log/debug
- Considera fattori ambientali
- Ogni domanda deve terminare con '?'

Query Utente: {query}

Genera esattamente {n} domande investigative come array JSON:""",
}


# =============================================================================
# Chain-of-Thought Strategy Templates
# =============================================================================
# Uses logical reasoning steps to generate questions.

CHAIN_OF_THOUGHT_TEMPLATES: Dict[str, str] = {
    "default": """You are an expert analyst using chain-of-thought reasoning to investigate a query.

Given a user query, think through the problem step-by-step and generate {n} investigative questions that follow a logical progression.

REASONING CHAIN:
Step 1: What is the core concept/problem being asked about?
Step 2: What foundational knowledge is needed to understand it?
Step 3: What are the key components or factors involved?
Step 4: What interactions or dependencies exist?
Step 5: What are the practical implications or applications?

RULES:
- Questions should build upon each other logically
- Start with foundational questions, progress to advanced
- Each question must end with '?'
- Focus on information in documentation

User Query: {query}

Think step-by-step and generate exactly {n} investigative questions as a JSON array:""",

    "troubleshooting": """You are an expert debugger using systematic reasoning to diagnose an issue.

Given a user query about an error/problem, think through the diagnosis step-by-step and generate {n} investigative questions.

DIAGNOSTIC REASONING CHAIN:
Step 1: What is the exact symptom or error?
Step 2: When/where does this occur?
Step 3: What changed recently that might cause this?
Step 4: What are the dependencies involved?
Step 5: What are the most common causes?
Step 6: How can each cause be verified or ruled out?

RULES:
- Follow systematic debugging methodology
- Questions should help isolate the root cause
- Each question must end with '?'
- Progress from general to specific

User Query: {query}

Think step-by-step and generate exactly {n} diagnostic questions as a JSON array:""",

    "default_it": """Sei un analista esperto che usa ragionamento chain-of-thought per investigare una query.

Data una query utente, ragiona passo-passo attraverso il problema e genera {n} domande investigative che seguono una progressione logica.

CATENA DI RAGIONAMENTO:
Passo 1: Qual è il concetto/problema principale su cui viene chiesto?
Passo 2: Quale conoscenza fondamentale è necessaria per capirlo?
Passo 3: Quali sono i componenti o fattori chiave coinvolti?
Passo 4: Quali interazioni o dipendenze esistono?
Passo 5: Quali sono le implicazioni pratiche o applicazioni?

REGOLE:
- Le domande devono costruirsi l'una sull'altra logicamente
- Inizia con domande fondamentali, progredisci verso le avanzate
- Ogni domanda deve terminare con '?'
- Concentrati su informazioni nella documentazione

Query Utente: {query}

Ragiona passo-passo e genera esattamente {n} domande investigative come array JSON:""",
}


# =============================================================================
# Semantic Expansion Strategy Templates
# =============================================================================
# Generates synonyms, related concepts, and technical alternatives.

SEMANTIC_EXPANSION_TEMPLATES: Dict[str, str] = {
    "default": """You are a semantic analysis expert generating search variations.

Given a user query, generate {n} investigative questions that explore related concepts, synonyms, and alternative terminology.

EXPANSION DIMENSIONS:
1. SYNONYMS: Alternative terms for the same concept
2. RELATED CONCEPTS: Closely related topics
3. TECHNICAL VARIANTS: Different technical approaches
4. BROADER CONTEXT: Parent concepts or categories
5. NARROWER FOCUS: Specific subtopics
6. CROSS-DOMAIN: Applications in other domains

RULES:
- Each question should use different terminology
- Explore the semantic neighborhood of the query
- Questions must be searchable in documentation
- Each question must end with '?'

User Query: {query}

Generate exactly {n} semantically diverse questions as a JSON array:""",

    "ai_ml": """You are an AI/ML terminology expert generating search variations.

Given a user query about AI/ML, generate {n} investigative questions exploring related ML concepts and alternative terminology.

ML SEMANTIC EXPANSION:
1. MODEL SYNONYMS: Alternative model names (e.g., encoder → transformer)
2. TECHNIQUE VARIANTS: Different approaches to same problem
3. FRAMEWORK SPECIFIC: Framework-specific terminology
4. ACADEMIC vs INDUSTRY: Different naming conventions
5. VERSION DIFFERENCES: Terminology changes across versions
6. IMPLEMENTATION DETAILS: Low-level component names

User Query: {query}

Generate exactly {n} semantically diverse ML questions as a JSON array:""",

    "default_it": """Sei un esperto di analisi semantica che genera variazioni di ricerca.

Data una query utente, genera {n} domande investigative che esplorano concetti correlati, sinonimi e terminologia alternativa.

DIMENSIONI DI ESPANSIONE:
1. SINONIMI: Termini alternativi per lo stesso concetto
2. CONCETTI CORRELATI: Argomenti strettamente correlati
3. VARIANTI TECNICHE: Approcci tecnici diversi
4. CONTESTO PIÙ AMPIO: Concetti o categorie parent
5. FOCUS PIÙ STRETTO: Sottotemi specifici
6. CROSS-DOMAIN: Applicazioni in altri domini

REGOLE:
- Ogni domanda deve usare terminologia diversa
- Esplora il vicinato semantico della query
- Le domande devono essere ricercabili nella documentazione
- Ogni domanda deve terminare con '?'

Query Utente: {query}

Genera esattamente {n} domande semanticamente diverse come array JSON:""",
}


# =============================================================================
# Cross-Reference Strategy Templates
# =============================================================================
# Explores prerequisites, dependencies, and related features.

CROSS_REFERENCE_TEMPLATES: Dict[str, str] = {
    "default": """You are a documentation expert generating cross-reference questions.

Given a user query, generate {n} investigative questions that explore prerequisites, dependencies, and related documentation.

CROSS-REFERENCE ASPECTS:
1. PREREQUISITES: What must be understood/configured first?
2. DEPENDENCIES: What does this depend on?
3. DEPENDENTS: What depends on this?
4. RELATED FEATURES: What features work together with this?
5. MIGRATION: How to migrate from/to this?
6. COMPATIBILITY: What is compatible/incompatible?
7. INTEGRATION: How does it integrate with other systems?

RULES:
- Focus on relationships between concepts
- Questions should help navigate documentation
- Each question must end with '?'

User Query: {query}

Generate exactly {n} cross-reference questions as a JSON array:""",

    "default_it": """Sei un esperto di documentazione che genera domande di cross-reference.

Data una query utente, genera {n} domande investigative che esplorano prerequisiti, dipendenze e documentazione correlata.

ASPETTI DI CROSS-REFERENCE:
1. PREREQUISITI: Cosa deve essere capito/configurato prima?
2. DIPENDENZE: Da cosa dipende questo?
3. DIPENDENTI: Cosa dipende da questo?
4. FEATURE CORRELATE: Quali feature funzionano insieme a questa?
5. MIGRAZIONE: Come migrare da/verso questo?
6. COMPATIBILITÀ: Cosa è compatibile/incompatibile?
7. INTEGRAZIONE: Come si integra con altri sistemi?

REGOLE:
- Concentrati sulle relazioni tra concetti
- Le domande dovrebbero aiutare a navigare la documentazione
- Ogni domanda deve terminare con '?'

Query Utente: {query}

Genera esattamente {n} domande di cross-reference come array JSON:""",
}


# =============================================================================
# Quality Assurance Validation Prompt
# =============================================================================

QA_VALIDATION_PROMPT = """You are a quality assurance expert evaluating investigative questions.

Evaluate the following questions for quality and relevance to the original query.

Original Query: {query}

Questions to evaluate:
{questions}

For each question, score 1-10 on:
1. RELEVANCE: How relevant is it to the original query?
2. SPECIFICITY: How specific and searchable is it?
3. QUALITY: Is it well-formed and clear?

Return a JSON object:
{{
  "scores": [
    {{"question": "...", "relevance": 8, "specificity": 7, "quality": 9, "average": 8.0}},
    ...
  ],
  "overall_score": 7.5,
  "recommendations": ["...", "..."]
}}"""


# =============================================================================
# Simple Fallback Template
# =============================================================================

SIMPLE_FALLBACK_TEMPLATE = """Generate {n} simple search questions about: {query}

Return only a JSON array of questions. Each must end with '?'."""


# =============================================================================
# Utility Functions
# =============================================================================


def classify_query(query: str) -> Tuple[str, float, List[str]]:
    """
    Classify query into a category with confidence and matched keywords.
    
    Args:
        query: User's query text
        
    Returns:
        Tuple of (category, confidence, matched_keywords)
    """
    query_lower = query.lower()
    
    category_scores: Dict[str, Tuple[float, List[str]]] = {}
    
    for category, config in QUERY_CATEGORIES.items():
        keywords = config.get("keywords", [])
        exclusive = config.get("exclusive", [])
        weight = config.get("weight", 1.0)
        
        matched = []
        score = 0.0
        
        # Check exclusive keywords first (higher weight)
        for kw in exclusive:
            if kw in query_lower:
                matched.append(kw)
                score += 3.0  # Exclusive keywords have higher impact
        
        # Check regular keywords
        for kw in keywords:
            if kw in query_lower and kw not in exclusive:
                matched.append(kw)
                score += 1.0
        
        # Apply category weight
        final_score = score * weight
        
        if final_score > 0:
            category_scores[category] = (final_score, matched)
    
    if not category_scores:
        return DEFAULT_CATEGORY, 0.5, []
    
    # Get best category
    best_category = max(category_scores.items(), key=lambda x: x[1][0])
    category_name = best_category[0]
    score, matched = best_category[1]
    
    # Normalize confidence (0.5 - 1.0 range)
    max_possible = 10.0 * QUERY_CATEGORIES[category_name].get("weight", 1.0)
    confidence = min(0.5 + (score / max_possible) * 0.5, 1.0)
    
    return category_name, confidence, matched


def get_preferred_strategy(category: str) -> str:
    """Get preferred strategy for a category."""
    config = QUERY_CATEGORIES.get(category, {})
    return config.get("preferred_strategy", DEFAULT_STRATEGY)


def get_secondary_strategy(category: str) -> str:
    """Get secondary/fallback strategy for a category."""
    config = QUERY_CATEGORIES.get(category, {})
    return config.get("secondary_strategy", "decomposition")


def detect_language(query: str) -> str:
    """
    Detect query language.
    
    Returns:
        'it' for Italian, 'en' for English/default
    """
    italian_markers = [
        "come", "cosa", "perché", "quando", "dove", "chi",
        "non", "sono", "essere", "fare", "avere", "questo",
        "della", "delle", "degli", "nella", "nelle", "qual",
        "quali", "quanto", "quanti", "devo", "posso", "vorrei",
    ]
    
    query_lower = query.lower()
    italian_count = sum(1 for marker in italian_markers if f" {marker} " in f" {query_lower} " or query_lower.startswith(marker + " ") or query_lower.endswith(" " + marker))
    
    return "it" if italian_count >= 2 else "en"


def get_investigation_prompt(
    query: str,
    n: int = 5,
    strategy: str = "decomposition",
    category: Optional[str] = None,
) -> str:
    """
    Get formatted investigation prompt based on strategy and category.
    
    Args:
        query: User's original query
        n: Number of questions to generate
        strategy: Investigation strategy
        category: Query category (auto-detected if None)
        
    Returns:
        Formatted prompt string
    """
    # Auto-detect category if not provided
    if category is None:
        category, _, _ = classify_query(query)
    
    # Detect language
    lang = detect_language(query)
    lang_suffix = "_it" if lang == "it" else ""
    
    # Select template based on strategy
    if strategy == "decomposition":
        templates = DECOMPOSITION_TEMPLATES
        template_key = f"{category}{lang_suffix}"
        if template_key not in templates:
            template_key = f"default{lang_suffix}"
        if template_key not in templates:
            template_key = "default"
    elif strategy == "chain_of_thought":
        templates = CHAIN_OF_THOUGHT_TEMPLATES
        template_key = f"default{lang_suffix}" if lang_suffix else "default"
        if category == "troubleshooting":
            template_key = f"troubleshooting{lang_suffix}" if f"troubleshooting{lang_suffix}" in templates else "troubleshooting"
        if template_key not in templates:
            template_key = "default"
    elif strategy == "semantic_expansion":
        templates = SEMANTIC_EXPANSION_TEMPLATES
        template_key = f"{category}{lang_suffix}"
        if template_key not in templates:
            template_key = f"default{lang_suffix}"
        if template_key not in templates:
            template_key = "default"
    elif strategy == "cross_reference":
        templates = CROSS_REFERENCE_TEMPLATES
        template_key = f"default{lang_suffix}" if lang_suffix else "default"
        if template_key not in templates:
            template_key = "default"
    else:
        # Fallback to decomposition
        templates = DECOMPOSITION_TEMPLATES
        template_key = "default"
    
    template = templates.get(template_key, templates.get("default", SIMPLE_FALLBACK_TEMPLATE))
    return template.format(query=query, n=n)


def get_simple_fallback_prompt(
    query: str,
    n: int = 5,
    language: Optional[str] = None,
) -> str:
    """Get simple fallback prompt for error recovery."""
    base_prompt = SIMPLE_FALLBACK_TEMPLATE.format(query=query, n=n)
    if language and language.lower() not in ("en", "english"):
        base_prompt += f"\n\nIMPORTANT: Generate all questions in {language}."
    return base_prompt


def get_chain_of_thought_prompt(
    query: str,
    n: int = 5,
    language: Optional[str] = None,
    category: Optional[str] = None,
) -> str:
    """
    Get chain-of-thought investigation prompt.

    Uses logical reasoning steps to generate investigative questions.
    Language is auto-detected if not specified.
    """
    # Language parameter accepted for API compatibility but detection is automatic
    return get_investigation_prompt(query, n, "chain_of_thought", category)


def get_semantic_expansion_prompt(
    query: str,
    n: int = 5,
    language: Optional[str] = None,
    category: Optional[str] = None,
) -> str:
    """
    Get semantic expansion investigation prompt.

    Generates questions exploring synonyms, related concepts, and alternatives.
    Language is auto-detected if not specified.
    """
    return get_investigation_prompt(query, n, "semantic_expansion", category)


def get_cross_reference_prompt(
    query: str,
    n: int = 5,
    category: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """
    Get cross-reference investigation prompt.

    Explores prerequisites, dependencies, and related documentation.
    Language is auto-detected if not specified.
    """
    return get_investigation_prompt(query, n, "cross_reference", category)


def get_qa_validation_prompt(query: str, questions: List[str]) -> str:
    """Get QA validation prompt."""
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return QA_VALIDATION_PROMPT.format(query=query, questions=questions_text)


# =============================================================================
# Strategy Registry
# =============================================================================

STRATEGY_TEMPLATES: Dict[str, Dict[str, str]] = {
    "decomposition": DECOMPOSITION_TEMPLATES,
    "chain_of_thought": CHAIN_OF_THOUGHT_TEMPLATES,
    "semantic_expansion": SEMANTIC_EXPANSION_TEMPLATES,
    "cross_reference": CROSS_REFERENCE_TEMPLATES,
}

AVAILABLE_STRATEGIES = list(STRATEGY_TEMPLATES.keys()) + ["adaptive", "simple"]
