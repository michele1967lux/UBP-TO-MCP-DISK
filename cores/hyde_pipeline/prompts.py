"""
hyde_pipeline/prompts.py

Domain-specific and format-specific prompt templates for HyDE generation.

Features:
- 7 Document Formats: answer, technical_doc, faq, code_snippet, tutorial, troubleshooting, article
- 7 Domains: ai_ml, devops, api_integration, database, security, cloud, general
- Cross-lingual support: EN/IT
- Refinement prompts for iterative improvement
- Quality assessment prompts
- Hallucination detection prompts

v1.0.0: Initial release
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# Domain Definitions
# ============================================================================

DOMAINS: Dict[str, Dict[str, Any]] = {
    "ai_ml": {
        "name": "AI/ML",
        "description": "Machine Learning, Deep Learning, LLMs, RAG, Embeddings",
        "keywords": [
            "machine learning", "deep learning", "neural network", "llm", "large language model",
            "rag", "retrieval", "embedding", "transformer", "attention", "training", "inference",
            "model", "pytorch", "tensorflow", "fine-tuning", "prompt", "vector", "similarity",
            "classification", "regression", "clustering", "nlp", "computer vision", "generative",
        ],
        "terminology": [
            "epoch", "batch size", "learning rate", "loss function", "gradient descent",
            "backpropagation", "activation function", "dropout", "regularization",
            "overfitting", "underfitting", "hyperparameter", "feature extraction",
            "tokenization", "embedding dimension", "cosine similarity", "semantic search",
        ],
        "preferred_formats": ["technical_doc", "code_snippet", "answer"],
    },
    "devops": {
        "name": "DevOps",
        "description": "Docker, Kubernetes, CI/CD, Infrastructure as Code",
        "keywords": [
            "docker", "kubernetes", "k8s", "container", "ci/cd", "pipeline", "deploy",
            "helm", "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
            "infrastructure", "automation", "monitoring", "logging", "scaling",
            "microservices", "service mesh", "istio", "prometheus", "grafana",
        ],
        "terminology": [
            "pod", "deployment", "service", "ingress", "configmap", "secret",
            "namespace", "replica", "node", "cluster", "dockerfile", "image",
            "registry", "volume", "persistent volume", "statefulset", "daemonset",
        ],
        "preferred_formats": ["tutorial", "troubleshooting", "code_snippet"],
    },
    "api_integration": {
        "name": "API Integration",
        "description": "REST APIs, GraphQL, Authentication, Webhooks",
        "keywords": [
            "rest api", "restful", "graphql", "webhook", "oauth", "oauth2", "jwt",
            "endpoint", "request", "response", "http", "https", "authentication",
            "authorization", "bearer token", "api key", "rate limit", "pagination",
            "swagger", "openapi", "postman", "curl", "fetch", "axios",
        ],
        "terminology": [
            "GET", "POST", "PUT", "DELETE", "PATCH", "header", "body", "query parameter",
            "path parameter", "status code", "200", "201", "400", "401", "403", "404", "500",
            "content-type", "application/json", "multipart", "cors", "preflight",
        ],
        "preferred_formats": ["code_snippet", "technical_doc", "faq"],
    },
    "database": {
        "name": "Database",
        "description": "SQL, NoSQL, Query Optimization, Data Modeling",
        "keywords": [
            "sql", "nosql", "postgresql", "postgres", "mysql", "mongodb", "redis",
            "elasticsearch", "qdrant", "pinecone", "query", "index", "transaction",
            "schema", "migration", "orm", "sqlalchemy", "prisma", "vector database",
            "full-text search", "aggregation", "join", "normalization",
        ],
        "terminology": [
            "SELECT", "INSERT", "UPDATE", "DELETE", "WHERE", "JOIN", "LEFT JOIN",
            "INNER JOIN", "GROUP BY", "ORDER BY", "HAVING", "INDEX", "PRIMARY KEY",
            "FOREIGN KEY", "CONSTRAINT", "TRIGGER", "STORED PROCEDURE", "VIEW",
            "ACID", "CAP theorem", "sharding", "replication", "partition",
        ],
        "preferred_formats": ["code_snippet", "technical_doc", "troubleshooting"],
    },
    "security": {
        "name": "Security",
        "description": "Authentication, Encryption, Vulnerabilities, Compliance",
        "keywords": [
            "authentication", "authorization", "encryption", "ssl", "tls", "certificate",
            "vulnerability", "penetration", "firewall", "rbac", "role-based",
            "access control", "xss", "csrf", "sql injection", "security audit",
            "compliance", "gdpr", "hipaa", "soc2", "zero trust", "mfa", "2fa",
        ],
        "terminology": [
            "hash", "salt", "bcrypt", "argon2", "aes", "rsa", "public key", "private key",
            "certificate authority", "csr", "crl", "ocsp", "tls handshake",
            "cipher suite", "perfect forward secrecy", "hsts", "csp", "cors policy",
        ],
        "preferred_formats": ["technical_doc", "troubleshooting", "faq"],
    },
    "cloud": {
        "name": "Cloud",
        "description": "AWS, Azure, GCP, Serverless, Cloud Architecture",
        "keywords": [
            "aws", "amazon", "azure", "microsoft", "gcp", "google cloud", "cloud",
            "serverless", "lambda", "s3", "ec2", "rds", "vpc", "load balancer",
            "auto scaling", "cdn", "cloudflare", "route53", "iam", "cloudformation",
            "arm template", "cloud run", "app engine", "cloud functions",
        ],
        "terminology": [
            "region", "availability zone", "vpc", "subnet", "security group",
            "nacl", "igw", "nat gateway", "elastic ip", "arn", "bucket policy",
            "iam role", "iam policy", "cross-account", "peering", "transit gateway",
        ],
        "preferred_formats": ["tutorial", "technical_doc", "troubleshooting"],
    },
    "general": {
        "name": "General",
        "description": "General technical and conceptual queries",
        "keywords": [],
        "terminology": [],
        "preferred_formats": ["answer", "faq", "article"],
    },
}


# ============================================================================
# Format Templates - English
# ============================================================================

FORMAT_TEMPLATES_EN: Dict[str, str] = {
    # ENTERPRISE: All templates are language-agnostic - they respond in the query's language
    "answer": """You are a knowledgeable technical expert. Generate a direct, informative answer that would appear in a high-quality knowledge base.

CRITICAL: You MUST write your entire response in the EXACT SAME LANGUAGE as the user's query below. If the query is in Italian, respond in Italian. If in French, respond in French. If in German, respond in German. Match the query language precisely.

Query: {query}
{domain_context}

Write a hypothetical document that directly answers this query. The document should:
- Be factual and informative
- Include specific technical details
- Use appropriate terminology
- Be self-contained (readable without additional context)
- Be between {min_length} and {max_length} characters

Hypothetical Document:""",

    "technical_doc": """You are a technical documentation writer. Generate a documentation excerpt that would answer the user's query.

CRITICAL: You MUST write your entire response in the EXACT SAME LANGUAGE as the user's query below. Match the query language precisely.

Query: {query}
{domain_context}

Write a hypothetical technical documentation section that addresses this query. Include:
- Clear explanations of concepts
- Technical specifications where relevant
- Proper terminology
- Examples if applicable
- Be between {min_length} and {max_length} characters

Do NOT use markdown headers. Write in flowing prose with clear paragraphs.

Hypothetical Documentation:""",

    "faq": """You are creating content for a technical FAQ. Generate an FAQ entry that addresses the user's query.

CRITICAL: You MUST write your entire response in the EXACT SAME LANGUAGE as the user's query below. Match the query language precisely.

Query: {query}
{domain_context}

Write a hypothetical FAQ entry in Q&A format:
- Restate the question clearly in the same language
- Provide a comprehensive answer
- Include practical guidance
- Address common follow-up concerns
- Be between {min_length} and {max_length} characters

FAQ Entry:""",

    "code_snippet": """You are a senior developer writing documentation with code examples.

CRITICAL: You MUST write all explanatory text in the EXACT SAME LANGUAGE as the user's query below. Code can remain in English but comments and descriptions must match the query language.

Query: {query}
{domain_context}

Generate a hypothetical code-focused documentation excerpt that:
- Shows relevant code examples
- Includes explanatory comments
- Provides context for the code
- Explains key concepts
- Be between {min_length} and {max_length} characters

Code Documentation:""",

    "tutorial": """You are writing a step-by-step tutorial for developers.

CRITICAL: You MUST write your entire response in the EXACT SAME LANGUAGE as the user's query below. Match the query language precisely.

Query: {query}
{domain_context}

Generate a hypothetical tutorial excerpt that:
- Provides clear, numbered steps
- Explains the 'why' behind each step
- Includes practical tips
- Anticipates common issues
- Be between {min_length} and {max_length} characters

Do NOT use markdown formatting. Write in clear prose with numbered steps inline.

Tutorial Excerpt:""",

    "troubleshooting": """You are a support engineer writing troubleshooting documentation.

CRITICAL: You MUST write your entire response in the EXACT SAME LANGUAGE as the user's query below. Match the query language precisely.

Query: {query}
{domain_context}

Generate a hypothetical troubleshooting guide that:
- Identifies the likely problem
- Lists diagnostic steps
- Provides solutions
- Explains root causes
- Be between {min_length} and {max_length} characters

Troubleshooting Guide:""",

    "article": """You are a technical writer creating an educational article.

CRITICAL: You MUST write your entire response in the EXACT SAME LANGUAGE as the user's query below. Match the query language precisely.

Query: {query}
{domain_context}

Generate a hypothetical article excerpt that:
- Introduces the topic clearly
- Explains concepts with examples
- Provides practical insights
- Maintains engaging narrative flow
- Be between {min_length} and {max_length} characters

Article Excerpt:""",
}


# ============================================================================
# Format Templates - Italian
# ============================================================================

FORMAT_TEMPLATES_IT: Dict[str, str] = {
    "answer": """Sei un esperto tecnico competente. Genera una risposta diretta e informativa che apparirebbe in una knowledge base di alta qualità.

Query: {query}
{domain_context}

Scrivi un documento ipotetico che risponda direttamente a questa query. Il documento deve:
- Essere fattuale e informativo
- Includere dettagli tecnici specifici
- Usare terminologia appropriata
- Essere autosufficiente (leggibile senza contesto aggiuntivo)
- Essere tra {min_length} e {max_length} caratteri

Documento Ipotetico:""",

    "technical_doc": """Sei un redattore di documentazione tecnica. Genera un estratto di documentazione che risponda alla query dell'utente.

Query: {query}
{domain_context}

Scrivi una sezione di documentazione tecnica ipotetica che affronti questa query. Includi:
- Spiegazioni chiare dei concetti
- Specifiche tecniche dove rilevante
- Terminologia appropriata
- Esempi se applicabile
- Essere tra {min_length} e {max_length} caratteri

NON usare intestazioni markdown. Scrivi in prosa fluida con paragrafi chiari.

Documentazione Ipotetica:""",

    "faq": """Stai creando contenuti per una FAQ tecnica. Genera una voce FAQ che affronti la query dell'utente.

Query: {query}
{domain_context}

Scrivi una voce FAQ ipotetica in formato Q&A:
- Riformula la domanda chiaramente
- Fornisci una risposta completa
- Includi indicazioni pratiche
- Affronta dubbi comuni di follow-up
- Essere tra {min_length} e {max_length} caratteri

Voce FAQ:""",

    "code_snippet": """Sei uno sviluppatore senior che scrive documentazione con esempi di codice.

Query: {query}
{domain_context}

Genera un estratto di documentazione ipotetico focalizzato sul codice che:
- Mostri esempi di codice rilevanti
- Includa commenti esplicativi
- Fornisca contesto per il codice
- Spieghi i concetti chiave
- Essere tra {min_length} e {max_length} caratteri

Documentazione Codice:""",

    "tutorial": """Stai scrivendo un tutorial passo-passo per sviluppatori.

Query: {query}
{domain_context}

Genera un estratto di tutorial ipotetico che:
- Fornisca passi chiari e numerati
- Spieghi il 'perché' dietro ogni passo
- Includa suggerimenti pratici
- Anticipi problemi comuni
- Essere tra {min_length} e {max_length} caratteri

NON usare formattazione markdown. Scrivi in prosa chiara con passi numerati inline.

Estratto Tutorial:""",

    "troubleshooting": """Sei un ingegnere di supporto che scrive documentazione di troubleshooting.

Query: {query}
{domain_context}

Genera una guida di troubleshooting ipotetica che:
- Identifichi il probabile problema
- Elenchi i passi diagnostici
- Fornisca soluzioni
- Spieghi le cause radice
- Essere tra {min_length} e {max_length} caratteri

Guida Troubleshooting:""",

    "article": """Sei un redattore tecnico che crea un articolo educativo.

Query: {query}
{domain_context}

Genera un estratto di articolo ipotetico che:
- Introduca l'argomento chiaramente
- Spieghi i concetti con esempi
- Fornisca intuizioni pratiche
- Mantenga un flusso narrativo coinvolgente
- Essere tra {min_length} e {max_length} caratteri

Estratto Articolo:""",
}


# ============================================================================
# Domain Context Templates
# ============================================================================

DOMAIN_CONTEXT_TEMPLATES: Dict[str, str] = {
    "ai_ml": """
Domain: AI/Machine Learning
Context: This query relates to machine learning, deep learning, neural networks, or AI systems.
Use appropriate ML terminology: epochs, batch size, learning rate, loss functions, model architecture, embeddings, etc.
Reference relevant frameworks (PyTorch, TensorFlow, Hugging Face) where applicable.""",

    "devops": """
Domain: DevOps/Infrastructure
Context: This query relates to containerization, orchestration, CI/CD, or infrastructure management.
Use appropriate DevOps terminology: containers, pods, deployments, pipelines, images, volumes, etc.
Reference relevant tools (Docker, Kubernetes, Terraform, Ansible) where applicable.""",

    "api_integration": """
Domain: API Integration
Context: This query relates to REST APIs, authentication, or system integration.
Use appropriate API terminology: endpoints, requests, responses, authentication, rate limiting, etc.
Include HTTP methods, status codes, and headers where relevant.""",

    "database": """
Domain: Database
Context: This query relates to databases, queries, or data management.
Use appropriate database terminology: tables, indexes, queries, transactions, schemas, etc.
Reference SQL syntax or NoSQL concepts where applicable.""",

    "security": """
Domain: Security
Context: This query relates to security, authentication, or data protection.
Use appropriate security terminology: encryption, authentication, authorization, vulnerabilities, etc.
Consider compliance requirements (GDPR, SOC2) where relevant.""",

    "cloud": """
Domain: Cloud Computing
Context: This query relates to cloud services, architecture, or serverless computing.
Use appropriate cloud terminology: regions, availability zones, services, scaling, etc.
Reference cloud providers (AWS, Azure, GCP) where applicable.""",

    "general": """
Domain: General Technical
Context: This is a general technical query.
Provide clear, well-structured information with appropriate technical depth.""",
}


# ============================================================================
# Refinement Prompts
# ============================================================================

REFINEMENT_PROMPTS: Dict[str, str] = {
    "expand": """The following hypothetical document needs expansion with more detail.

Original Query: {query}
Current Document:
{document}

Quality Feedback:
- Score: {score}/10
- Issues: {issues}

Expand this document by:
- Adding more specific technical details
- Including additional examples
- Elaborating on key concepts
- Maintaining the same format and style

Expanded Document:""",

    "focus": """The following hypothetical document is too broad and needs to be more focused.

Original Query: {query}
Current Document:
{document}

Quality Feedback:
- Score: {score}/10
- Issues: {issues}

Make this document more focused by:
- Concentrating on the core question
- Removing tangential information
- Being more direct and specific
- Maintaining technical accuracy

Focused Document:""",

    "technical": """The following hypothetical document needs more technical depth.

Original Query: {query}
Current Document:
{document}

Quality Feedback:
- Score: {score}/10
- Issues: {issues}

Enhance technical depth by:
- Adding specific technical terminology
- Including implementation details
- Referencing specific tools/technologies
- Adding code examples if appropriate

Technical Document:""",

    "simplify": """The following hypothetical document is too complex and needs simplification.

Original Query: {query}
Current Document:
{document}

Quality Feedback:
- Score: {score}/10
- Issues: {issues}

Simplify this document by:
- Using clearer language
- Breaking down complex concepts
- Removing jargon where unnecessary
- Making it more accessible

Simplified Document:""",
}


# ============================================================================
# Quality Assessment Prompts
# ============================================================================

QUALITY_ASSESSMENT_PROMPT = """Evaluate the quality of this hypothetical document for RAG retrieval.

Original Query: {query}
Document Format: {format}
Document:
{document}

Rate from 1-10 on each dimension:
1. Relevance: Does it directly address the query?
2. Coherence: Is it well-structured and logical?
3. Informativeness: Does it provide useful, specific information?
4. Format Adherence: Does it match the expected format ({format})?
5. Terminology: Does it use appropriate technical terms?

Respond in JSON format:
{{
    "relevance": <score>,
    "coherence": <score>,
    "informativeness": <score>,
    "format_adherence": <score>,
    "terminology": <score>,
    "issues": ["<issue1>", "<issue2>"],
    "suggestions": ["<suggestion1>", "<suggestion2>"]
}}"""


# ============================================================================
# Hallucination Detection Prompts
# ============================================================================

HALLUCINATION_CHECK_PROMPT = """Analyze this document for potential hallucinations or fabricated information.

Original Query: {query}
Document:
{document}

Check for:
1. Invented API endpoints or methods that don't exist
2. Fake version numbers for software/libraries
3. Non-existent configuration options
4. Made-up technical terms
5. Incorrect technical facts

Respond in JSON format:
{{
    "hallucination_detected": <true/false>,
    "confidence": <0.0-1.0>,
    "suspicious_elements": [
        {{"text": "<suspicious text>", "reason": "<why suspicious>", "severity": "low|medium|high"}}
    ],
    "recommendation": "accept|review|reject"
}}"""


# ============================================================================
# Ensemble Fusion Prompts
# ============================================================================

ENSEMBLE_FUSION_PROMPT = """You have multiple hypothetical documents generated for the same query. Create an optimal combined document.

Original Query: {query}

Documents:
{documents}

Create a single, unified document that:
1. Combines the best elements from each source
2. Eliminates redundancy
3. Maintains coherent flow
4. Preserves technical accuracy
5. Stays within {max_length} characters

Fused Document:"""


# ============================================================================
# Cross-Lingual Templates
# ============================================================================

LANGUAGE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "en": FORMAT_TEMPLATES_EN,
    "it": FORMAT_TEMPLATES_IT,
}


# ============================================================================
# Utility Functions
# ============================================================================

def get_format_template(
    format_type: str,
    language: str = "en",
) -> str:
    """Get the appropriate format template for the given format and language."""
    templates = LANGUAGE_TEMPLATES.get(language, FORMAT_TEMPLATES_EN)
    return templates.get(format_type, templates.get("answer", ""))


def get_domain_context(domain: str) -> str:
    """Get the domain context for prompt enrichment."""
    return DOMAIN_CONTEXT_TEMPLATES.get(domain, DOMAIN_CONTEXT_TEMPLATES["general"])


def get_domain_info(domain: str) -> Dict[str, Any]:
    """Get full domain information."""
    return DOMAINS.get(domain, DOMAINS["general"])


def get_refinement_prompt(strategy: str) -> str:
    """Get refinement prompt for the given strategy."""
    return REFINEMENT_PROMPTS.get(strategy, REFINEMENT_PROMPTS["expand"])


def detect_domain(query: str) -> tuple[str, float]:
    """
    Detect the most likely domain for a query based on keyword matching.
    
    Returns:
        Tuple of (domain_name, confidence_score)
    """
    query_lower = query.lower()
    best_domain = "general"
    best_score = 0.0
    
    for domain_name, domain_info in DOMAINS.items():
        if domain_name == "general":
            continue
            
        keywords = domain_info.get("keywords", [])
        terminology = domain_info.get("terminology", [])
        all_terms = keywords + terminology
        
        if not all_terms:
            continue
        
        matches = sum(1 for term in all_terms if term.lower() in query_lower)
        weight = domain_info.get("weight", 1.0) if isinstance(domain_info.get("weight"), (int, float)) else 1.0
        score = (matches / len(all_terms)) * weight
        
        if score > best_score:
            best_score = score
            best_domain = domain_name
    
    # Normalize confidence to 0-1 range
    confidence = min(best_score * 2, 1.0)  # Scale up since matches are usually partial
    
    return best_domain, confidence


def detect_language(text: str) -> str:
    """
    Simple language detection based on common words.
    
    Returns:
        Language code: 'en' or 'it'
    """
    italian_markers = [
        "come", "cosa", "perché", "quando", "dove", "chi", "quale",
        "il", "la", "lo", "gli", "le", "un", "una", "uno",
        "è", "sono", "sei", "siamo", "essere", "avere", "fare",
        "non", "che", "per", "con", "su", "da", "in", "di",
    ]
    
    text_lower = text.lower()
    words = text_lower.split()
    
    italian_count = sum(1 for word in words if word in italian_markers)
    italian_ratio = italian_count / max(len(words), 1)
    
    return "it" if italian_ratio > 0.15 else "en"


def build_hyde_prompt(
    query: str,
    format_type: str = "answer",
    domain: str = "auto",
    language: str = "auto",
    min_length: int = 100,
    max_length: int = 400,
) -> str:
    """
    Build a complete HyDE prompt with all context.
    
    Args:
        query: User's query
        format_type: Document format (answer, technical_doc, etc.)
        domain: Domain (ai_ml, devops, etc.) or 'auto' for detection
        language: Language code or 'auto' for detection
        min_length: Minimum document length
        max_length: Maximum document length
    
    Returns:
        Complete prompt string
    """
    # Auto-detect language if needed
    if language == "auto":
        language = detect_language(query)
    
    # Auto-detect domain if needed
    if domain == "auto":
        domain, _ = detect_domain(query)
    
    # Get template and context
    template = get_format_template(format_type, language)
    domain_context = get_domain_context(domain)
    
    # Build prompt
    prompt = template.format(
        query=query,
        domain_context=domain_context,
        min_length=min_length,
        max_length=max_length,
    )
    
    return prompt


# ============================================================================
# Data Classes for Structured Access
# ============================================================================

@dataclass
class FormatConfig:
    """Configuration for a document format."""
    name: str
    enabled: bool = True
    weight: float = 1.0
    temperature: float = 0.5
    max_length: int = 400
    description: str = ""


@dataclass
class DomainConfig:
    """Configuration for a domain."""
    name: str
    enabled: bool = True
    weight: float = 1.0
    keywords: List[str] = field(default_factory=list)
    terminology: List[str] = field(default_factory=list)
    preferred_formats: List[str] = field(default_factory=list)
    terminology_boost: bool = True


def get_all_formats() -> List[str]:
    """Get list of all available formats."""
    return list(FORMAT_TEMPLATES_EN.keys())


def get_all_domains() -> List[str]:
    """Get list of all available domains."""
    return list(DOMAINS.keys())
