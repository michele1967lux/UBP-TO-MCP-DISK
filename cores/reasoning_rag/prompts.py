"""
reasoning_rag/prompts.py

Strategy-specific prompt templates for Reasoning-Aware RAG.

Strategies:
- Self-Ask RAG: Iterative sub-question decomposition
- Chain-of-Thought RAG: Interleaved reasoning and retrieval
- Evidence Attribution: Citation and source tracking
- Verification: Multi-source fact checking

Features:
- Multi-language support (EN/IT)
- Query complexity analysis
- Claim extraction
- Contradiction detection

v1.0.0: Initial release
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


# ============================================================================
# Enums
# ============================================================================


class ReasoningStrategy(Enum):
    """Available reasoning strategies."""
    SELF_ASK = "self_ask"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    EVIDENCE_ATTRIBUTION = "evidence_attribution"
    VERIFICATION = "verification"
    DIRECT = "direct"
    HYBRID = "hybrid"


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"


class QueryIntent(Enum):
    """Query intent types."""
    FACTUAL = "factual"
    EXPLANATORY = "explanatory"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    CAUSAL = "causal"
    DEFINITIONAL = "definitional"
    EVALUATIVE = "evaluative"


# ============================================================================
# Self-Ask RAG Prompts
# ============================================================================


SELF_ASK_DECOMPOSITION_EN = """You are an expert at breaking down complex questions into simpler sub-questions.

Original Question: {query}

Context gathered so far:
{context}

Current iteration: {iteration} of {max_iterations}

Your task:
1. Analyze what information is still needed to fully answer the original question
2. Generate {max_sub_questions} focused sub-questions that will help gather missing information
3. Each sub-question should be specific and searchable

Rules:
- Do NOT repeat sub-questions already asked: {previous_questions}
- Each sub-question should address a different aspect
- Sub-questions should be answerable with factual information
- If the original question can now be fully answered, respond with "NO_MORE_QUESTIONS"

Respond in JSON format:
{{
    "analysis": "<what information is still missing>",
    "sub_questions": ["<sub-question 1>", "<sub-question 2>", ...],
    "can_answer_now": <true/false>,
    "confidence": <0.0-1.0>
}}"""


SELF_ASK_DECOMPOSITION_IT = """Sei un esperto nel scomporre domande complesse in sotto-domande più semplici.

Domanda Originale: {query}

Contesto raccolto finora:
{context}

Iterazione corrente: {iteration} di {max_iterations}

Il tuo compito:
1. Analizza quali informazioni mancano ancora per rispondere completamente alla domanda originale
2. Genera {max_sub_questions} sotto-domande focalizzate che aiuteranno a raccogliere le informazioni mancanti
3. Ogni sotto-domanda deve essere specifica e ricercabile

Regole:
- NON ripetere sotto-domande già poste: {previous_questions}
- Ogni sotto-domanda deve affrontare un aspetto diverso
- Le sotto-domande devono essere rispondibili con informazioni fattuali
- Se la domanda originale può ora essere completamente risposta, rispondi con "NO_MORE_QUESTIONS"

Rispondi in formato JSON:
{{
    "analysis": "<quali informazioni mancano ancora>",
    "sub_questions": ["<sotto-domanda 1>", "<sotto-domanda 2>", ...],
    "can_answer_now": <true/false>,
    "confidence": <0.0-1.0>
}}"""


SELF_ASK_INTEGRATION_EN = """You are synthesizing information from multiple sub-questions to answer an original question.

Original Question: {query}

Sub-questions and their answers:
{sub_qa_pairs}

Your task:
1. Integrate all the gathered information
2. Provide a comprehensive answer to the original question
3. Identify any gaps or uncertainties

Respond in JSON format:
{{
    "answer": "<comprehensive answer to the original question>",
    "key_points": ["<point 1>", "<point 2>", ...],
    "confidence": <0.0-1.0>,
    "gaps": ["<any remaining uncertainties>"],
    "sources_used": [<indices of sub-questions that contributed>]
}}"""


SELF_ASK_INTEGRATION_IT = """Stai sintetizzando informazioni da multiple sotto-domande per rispondere alla domanda originale.

Domanda Originale: {query}

Sotto-domande e relative risposte:
{sub_qa_pairs}

Il tuo compito:
1. Integra tutte le informazioni raccolte
2. Fornisci una risposta completa alla domanda originale
3. Identifica eventuali lacune o incertezze

Rispondi in formato JSON:
{{
    "answer": "<risposta completa alla domanda originale>",
    "key_points": ["<punto 1>", "<punto 2>", ...],
    "confidence": <0.0-1.0>,
    "gaps": ["<eventuali incertezze rimanenti>"],
    "sources_used": [<indici delle sotto-domande che hanno contribuito>]
}}"""


# ============================================================================
# Chain-of-Thought RAG Prompts
# ============================================================================


COT_REASONING_EN = """You are a reasoning assistant that thinks step by step, retrieving information when needed.

Question: {query}

Available context from previous retrievals:
{context}

Current reasoning step: {step} of {max_steps}
Previous thoughts: {previous_thoughts}

Instructions:
1. Think about what you need to reason about next
2. If you need more information, specify what to search for
3. If you can make a conclusion, state it clearly
4. Build upon previous thoughts

Respond in JSON format:
{{
    "thought": "<your current reasoning step>",
    "needs_retrieval": <true/false>,
    "retrieval_query": "<what to search for, if needed>",
    "intermediate_conclusion": "<any conclusion from this step>",
    "confidence": <0.0-1.0>,
    "ready_for_final_answer": <true/false>
}}"""


COT_REASONING_IT = """Sei un assistente di ragionamento che pensa passo dopo passo, recuperando informazioni quando necessario.

Domanda: {query}

Contesto disponibile dai recuperi precedenti:
{context}

Passo di ragionamento corrente: {step} di {max_steps}
Pensieri precedenti: {previous_thoughts}

Istruzioni:
1. Pensa a cosa devi ragionare successivamente
2. Se hai bisogno di più informazioni, specifica cosa cercare
3. Se puoi trarre una conclusione, dichiarala chiaramente
4. Costruisci sui pensieri precedenti

Rispondi in formato JSON:
{{
    "thought": "<il tuo passo di ragionamento corrente>",
    "needs_retrieval": <true/false>,
    "retrieval_query": "<cosa cercare, se necessario>",
    "intermediate_conclusion": "<eventuale conclusione da questo passo>",
    "confidence": <0.0-1.0>,
    "ready_for_final_answer": <true/false>
}}"""


COT_SYNTHESIS_EN = """You have completed a chain of reasoning. Now synthesize the final answer.

Original Question: {query}

Complete reasoning chain:
{reasoning_chain}

All retrieved context:
{all_context}

Provide the final answer based on your reasoning:

Respond in JSON format:
{{
    "answer": "<final comprehensive answer>",
    "reasoning_summary": "<brief summary of how you arrived at this answer>",
    "key_insights": ["<insight 1>", "<insight 2>", ...],
    "confidence": <0.0-1.0>,
    "supporting_evidence": ["<evidence 1>", "<evidence 2>", ...]
}}"""


COT_SYNTHESIS_IT = """Hai completato una catena di ragionamento. Ora sintetizza la risposta finale.

Domanda Originale: {query}

Catena di ragionamento completa:
{reasoning_chain}

Tutto il contesto recuperato:
{all_context}

Fornisci la risposta finale basata sul tuo ragionamento:

Rispondi in formato JSON:
{{
    "answer": "<risposta finale completa>",
    "reasoning_summary": "<breve riassunto di come sei arrivato a questa risposta>",
    "key_insights": ["<intuizione 1>", "<intuizione 2>", ...],
    "confidence": <0.0-1.0>,
    "supporting_evidence": ["<evidenza 1>", "<evidenza 2>", ...]
}}"""


# ============================================================================
# Evidence Attribution Prompts
# ============================================================================


EVIDENCE_CLAIM_EXTRACTION_EN = """Extract factual claims from the following text that need to be attributed to sources.

Text: {text}

Instructions:
1. Identify each distinct factual claim
2. Mark claims that are opinions vs facts
3. Note which claims are most important to verify

Respond in JSON format:
{{
    "claims": [
        {{
            "claim": "<the factual claim>",
            "type": "fact|opinion|inference",
            "importance": "high|medium|low",
            "needs_verification": <true/false>
        }}
    ],
    "total_claims": <number>
}}"""


EVIDENCE_CLAIM_EXTRACTION_IT = """Estrai le affermazioni fattuali dal seguente testo che devono essere attribuite alle fonti.

Testo: {text}

Istruzioni:
1. Identifica ogni affermazione fattuale distinta
2. Distingui tra opinioni e fatti
3. Nota quali affermazioni sono più importanti da verificare

Rispondi in formato JSON:
{{
    "claims": [
        {{
            "claim": "<l'affermazione fattuale>",
            "type": "fact|opinion|inference",
            "importance": "high|medium|low",
            "needs_verification": <true/false>
        }}
    ],
    "total_claims": <numero>
}}"""


EVIDENCE_ATTRIBUTION_EN = """Attribute the following claims to their sources and provide citations.

Claims to attribute:
{claims}

Available sources:
{sources}

Instructions:
1. Match each claim to the most relevant source(s)
2. Extract the exact text span that supports the claim
3. Assign a confidence score based on how well the source supports the claim
4. Note if a claim cannot be attributed

Respond in JSON format:
{{
    "attributions": [
        {{
            "claim": "<the claim>",
            "source_ids": [<list of source indices>],
            "supporting_text": "<exact text from source>",
            "confidence": <0.0-1.0>,
            "attribution_type": "direct|inferred|partial|none"
        }}
    ],
    "unattributed_claims": ["<claims without sources>"],
    "overall_grounding_score": <0.0-1.0>
}}"""


EVIDENCE_ATTRIBUTION_IT = """Attribuisci le seguenti affermazioni alle loro fonti e fornisci citazioni.

Affermazioni da attribuire:
{claims}

Fonti disponibili:
{sources}

Istruzioni:
1. Abbina ogni affermazione alla fonte o alle fonti più rilevanti
2. Estrai il testo esatto che supporta l'affermazione
3. Assegna un punteggio di confidenza basato su quanto bene la fonte supporta l'affermazione
4. Nota se un'affermazione non può essere attribuita

Rispondi in formato JSON:
{{
    "attributions": [
        {{
            "claim": "<l'affermazione>",
            "source_ids": [<lista degli indici delle fonti>],
            "supporting_text": "<testo esatto dalla fonte>",
            "confidence": <0.0-1.0>,
            "attribution_type": "direct|inferred|partial|none"
        }}
    ],
    "unattributed_claims": ["<affermazioni senza fonti>"],
    "overall_grounding_score": <0.0-1.0>
}}"""


EVIDENCE_ANSWER_WITH_CITATIONS_EN = """Generate an answer with inline citations for the following question.

Question: {query}

Sources:
{sources}

Instructions:
1. Answer the question comprehensively
2. Include inline citations [1], [2], etc. for each factual claim
3. Only include information that can be attributed to sources
4. Note any aspects that cannot be answered from the sources

Respond in JSON format:
{{
    "answer_with_citations": "<answer with [n] citations inline>",
    "citations": [
        {{
            "id": <number>,
            "source_id": <source index>,
            "text": "<cited text>",
            "page": "<page number if available>"
        }}
    ],
    "unanswered_aspects": ["<aspects not covered by sources>"],
    "confidence": <0.0-1.0>
}}"""


EVIDENCE_ANSWER_WITH_CITATIONS_IT = """Genera una risposta con citazioni inline per la seguente domanda.

Domanda: {query}

Fonti:
{sources}

Istruzioni:
1. Rispondi alla domanda in modo completo
2. Includi citazioni inline [1], [2], ecc. per ogni affermazione fattuale
3. Includi solo informazioni che possono essere attribuite alle fonti
4. Nota eventuali aspetti che non possono essere risposti dalle fonti

Rispondi in formato JSON:
{{
    "answer_with_citations": "<risposta con citazioni [n] inline>",
    "citations": [
        {{
            "id": <numero>,
            "source_id": <indice fonte>,
            "text": "<testo citato>",
            "page": "<numero pagina se disponibile>"
        }}
    ],
    "unanswered_aspects": ["<aspetti non coperti dalle fonti>"],
    "confidence": <0.0-1.0>
}}"""


# ============================================================================
# Verification Prompts
# ============================================================================


VERIFICATION_FACT_CHECK_EN = """Verify the following claims against multiple sources.

Claims to verify:
{claims}

Sources:
{sources}

Instructions:
1. Check each claim against all available sources
2. Identify supporting, contradicting, or neutral evidence
3. Detect any contradictions between sources
4. Provide an overall verification status

Respond in JSON format:
{{
    "verifications": [
        {{
            "claim": "<the claim>",
            "status": "verified|partially_verified|unverified|contradicted",
            "supporting_sources": [<source indices>],
            "contradicting_sources": [<source indices>],
            "confidence": <0.0-1.0>,
            "notes": "<any relevant notes>"
        }}
    ],
    "contradictions_found": [
        {{
            "claim": "<claim with contradiction>",
            "source_a": "<what source A says>",
            "source_b": "<what source B says>",
            "severity": "high|medium|low"
        }}
    ],
    "overall_verification_score": <0.0-1.0>
}}"""


VERIFICATION_FACT_CHECK_IT = """Verifica le seguenti affermazioni contro multiple fonti.

Affermazioni da verificare:
{claims}

Fonti:
{sources}

Istruzioni:
1. Controlla ogni affermazione contro tutte le fonti disponibili
2. Identifica evidenze a supporto, contraddittorie o neutre
3. Rileva eventuali contraddizioni tra le fonti
4. Fornisci uno stato di verifica complessivo

Rispondi in formato JSON:
{{
    "verifications": [
        {{
            "claim": "<l'affermazione>",
            "status": "verified|partially_verified|unverified|contradicted",
            "supporting_sources": [<indici fonti>],
            "contradicting_sources": [<indici fonti>],
            "confidence": <0.0-1.0>,
            "notes": "<eventuali note rilevanti>"
        }}
    ],
    "contradictions_found": [
        {{
            "claim": "<affermazione con contraddizione>",
            "source_a": "<cosa dice la fonte A>",
            "source_b": "<cosa dice la fonte B>",
            "severity": "high|medium|low"
        }}
    ],
    "overall_verification_score": <0.0-1.0>
}}"""


VERIFICATION_GROUNDING_CHECK_EN = """Check if the generated answer is properly grounded in the provided sources.

Generated Answer: {answer}

Sources used:
{sources}

Instructions:
1. Check each statement in the answer against the sources
2. Identify any hallucinated or unsupported statements
3. Verify that citations are accurate
4. Assess overall grounding quality

Respond in JSON format:
{{
    "grounding_analysis": [
        {{
            "statement": "<statement from answer>",
            "grounded": <true/false>,
            "source_support": "<which source supports this>",
            "hallucination_risk": "none|low|medium|high"
        }}
    ],
    "hallucinated_content": ["<statements not supported by sources>"],
    "citation_accuracy": <0.0-1.0>,
    "overall_grounding_score": <0.0-1.0>,
    "recommendation": "accept|revise|reject"
}}"""


VERIFICATION_GROUNDING_CHECK_IT = """Verifica se la risposta generata è correttamente fondata nelle fonti fornite.

Risposta Generata: {answer}

Fonti utilizzate:
{sources}

Istruzioni:
1. Controlla ogni affermazione nella risposta contro le fonti
2. Identifica affermazioni alucinate o non supportate
3. Verifica che le citazioni siano accurate
4. Valuta la qualità complessiva del grounding

Rispondi in formato JSON:
{{
    "grounding_analysis": [
        {{
            "statement": "<affermazione dalla risposta>",
            "grounded": <true/false>,
            "source_support": "<quale fonte supporta questo>",
            "hallucination_risk": "none|low|medium|high"
        }}
    ],
    "hallucinated_content": ["<affermazioni non supportate dalle fonti>"],
    "citation_accuracy": <0.0-1.0>,
    "overall_grounding_score": <0.0-1.0>,
    "recommendation": "accept|revise|reject"
}}"""


# ============================================================================
# Query Analysis Prompts
# ============================================================================


QUERY_ANALYSIS_EN = """Analyze the following query to determine the best reasoning strategy.

Query: {query}

Analyze:
1. Query complexity (simple, moderate, complex, multi-hop)
2. Query intent (factual, explanatory, comparative, procedural, causal)
3. Whether it requires multiple pieces of information
4. Whether it needs step-by-step reasoning
5. Whether verification is important

Respond in JSON format:
{{
    "complexity": "simple|moderate|complex|multi_hop",
    "intent": "factual|explanatory|comparative|procedural|causal|definitional|evaluative",
    "requires_multi_source": <true/false>,
    "requires_reasoning": <true/false>,
    "requires_verification": <true/false>,
    "recommended_strategy": "self_ask|chain_of_thought|evidence_attribution|verification|direct",
    "strategy_reason": "<why this strategy>",
    "estimated_steps": <number>,
    "key_entities": ["<entity 1>", "<entity 2>"],
    "language": "en|it"
}}"""


QUERY_ANALYSIS_IT = """Analizza la seguente query per determinare la migliore strategia di ragionamento.

Query: {query}

Analizza:
1. Complessità della query (simple, moderate, complex, multi-hop)
2. Intento della query (factual, explanatory, comparative, procedural, causal)
3. Se richiede multiple informazioni
4. Se necessita di ragionamento passo-passo
5. Se la verifica è importante

Rispondi in formato JSON:
{{
    "complexity": "simple|moderate|complex|multi_hop",
    "intent": "factual|explanatory|comparative|procedural|causal|definitional|evaluative",
    "requires_multi_source": <true/false>,
    "requires_reasoning": <true/false>,
    "requires_verification": <true/false>,
    "recommended_strategy": "self_ask|chain_of_thought|evidence_attribution|verification|direct",
    "strategy_reason": "<perché questa strategia>",
    "estimated_steps": <numero>,
    "key_entities": ["<entità 1>", "<entità 2>"],
    "language": "en|it"
}}"""


# ============================================================================
# Direct Answer Prompt
# ============================================================================


DIRECT_ANSWER_EN = """Answer the following question directly using the provided context.

Question: {query}

Context:
{context}

Provide a direct, concise answer:

Respond in JSON format:
{{
    "answer": "<direct answer>",
    "confidence": <0.0-1.0>,
    "sources_used": [<indices of context items used>]
}}"""


DIRECT_ANSWER_IT = """Rispondi direttamente alla seguente domanda usando il contesto fornito.

Domanda: {query}

Contesto:
{context}

Fornisci una risposta diretta e concisa:

Rispondi in formato JSON:
{{
    "answer": "<risposta diretta>",
    "confidence": <0.0-1.0>,
    "sources_used": [<indici degli elementi di contesto usati>]
}}"""


# ============================================================================
# Template Registry
# ============================================================================


TEMPLATES = {
    "en": {
        "self_ask_decomposition": SELF_ASK_DECOMPOSITION_EN,
        "self_ask_integration": SELF_ASK_INTEGRATION_EN,
        "cot_reasoning": COT_REASONING_EN,
        "cot_synthesis": COT_SYNTHESIS_EN,
        "evidence_claim_extraction": EVIDENCE_CLAIM_EXTRACTION_EN,
        "evidence_attribution": EVIDENCE_ATTRIBUTION_EN,
        "evidence_answer_with_citations": EVIDENCE_ANSWER_WITH_CITATIONS_EN,
        "verification_fact_check": VERIFICATION_FACT_CHECK_EN,
        "verification_grounding": VERIFICATION_GROUNDING_CHECK_EN,
        "query_analysis": QUERY_ANALYSIS_EN,
        "direct_answer": DIRECT_ANSWER_EN,
    },
    "it": {
        "self_ask_decomposition": SELF_ASK_DECOMPOSITION_IT,
        "self_ask_integration": SELF_ASK_INTEGRATION_IT,
        "cot_reasoning": COT_REASONING_IT,
        "cot_synthesis": COT_SYNTHESIS_IT,
        "evidence_claim_extraction": EVIDENCE_CLAIM_EXTRACTION_IT,
        "evidence_attribution": EVIDENCE_ATTRIBUTION_IT,
        "evidence_answer_with_citations": EVIDENCE_ANSWER_WITH_CITATIONS_IT,
        "verification_fact_check": VERIFICATION_FACT_CHECK_IT,
        "verification_grounding": VERIFICATION_GROUNDING_CHECK_IT,
        "query_analysis": QUERY_ANALYSIS_IT,
        "direct_answer": DIRECT_ANSWER_IT,
    },
}


# ============================================================================
# Utility Functions
# ============================================================================


def get_template(template_name: str, language: str = "en") -> str:
    """Get a template by name and language."""
    lang_templates = TEMPLATES.get(language, TEMPLATES["en"])
    return lang_templates.get(template_name, TEMPLATES["en"].get(template_name, ""))


def detect_language(text: str) -> str:
    """Simple language detection based on common words."""
    italian_markers = {
        "come", "cosa", "perché", "quando", "dove", "chi", "quale",
        "il", "la", "lo", "gli", "le", "un", "una", "uno",
        "è", "sono", "sei", "siamo", "essere", "avere", "fare",
        "non", "che", "per", "con", "su", "da", "in", "di",
    }
    
    words = set(text.lower().split())
    italian_count = len(words & italian_markers)
    italian_ratio = italian_count / max(len(words), 1)
    
    return "it" if italian_ratio > 0.15 else "en"


def analyze_query_complexity(query: str) -> Tuple[QueryComplexity, float]:
    """
    Analyze query complexity based on heuristics.
    
    Returns:
        Tuple of (complexity level, confidence)
    """
    query_lower = query.lower()
    words = query_lower.split()
    
    # Multi-hop indicators
    multi_hop_indicators = ["and", "also", "additionally", "furthermore", "compare", "versus", "vs", "difference"]
    multi_hop_count = sum(1 for w in words if w in multi_hop_indicators)
    
    # Complexity indicators
    complex_indicators = ["why", "how", "explain", "analyze", "evaluate", "implications", "consequences"]
    complex_count = sum(1 for w in words if w in complex_indicators)
    
    # Question count (multiple questions)
    question_count = query.count("?")
    
    # Length-based complexity
    word_count = len(words)
    
    # Calculate complexity score
    score = 0.0
    score += multi_hop_count * 0.2
    score += complex_count * 0.15
    score += (question_count - 1) * 0.25 if question_count > 1 else 0
    score += min(word_count / 50, 0.3)  # Longer queries tend to be more complex
    
    # Determine complexity level
    if score >= 0.7:
        return QueryComplexity.MULTI_HOP, min(score, 1.0)
    elif score >= 0.5:
        return QueryComplexity.COMPLEX, score
    elif score >= 0.25:
        return QueryComplexity.MODERATE, score
    else:
        return QueryComplexity.SIMPLE, 1.0 - score


def detect_query_intent(query: str) -> QueryIntent:
    """Detect the primary intent of a query."""
    query_lower = query.lower()
    
    # Intent patterns
    intent_patterns = {
        QueryIntent.DEFINITIONAL: ["what is", "what are", "define", "definition", "meaning of", "cos'è", "cosa sono"],
        QueryIntent.EXPLANATORY: ["explain", "how does", "how do", "why does", "spiega", "come funziona"],
        QueryIntent.COMPARATIVE: ["compare", "difference", "versus", "vs", "better", "confronta", "differenza"],
        QueryIntent.PROCEDURAL: ["how to", "steps to", "process", "procedure", "come fare", "procedura"],
        QueryIntent.CAUSAL: ["why", "cause", "reason", "because", "effect", "perché", "causa"],
        QueryIntent.EVALUATIVE: ["should", "best", "recommend", "evaluate", "assessment", "consiglia", "migliore"],
    }
    
    for intent, patterns in intent_patterns.items():
        for pattern in patterns:
            if pattern in query_lower:
                return intent
    
    return QueryIntent.FACTUAL


def recommend_strategy(
    complexity: QueryComplexity,
    intent: QueryIntent,
    requires_verification: bool = False,
) -> ReasoningStrategy:
    """
    Recommend the best reasoning strategy based on query analysis.
    
    Args:
        complexity: Query complexity level
        intent: Query intent type
        requires_verification: Whether verification is important
        
    Returns:
        Recommended reasoning strategy
    """
    # Verification takes precedence if required
    if requires_verification:
        return ReasoningStrategy.VERIFICATION
    
    # Simple queries don't need complex reasoning
    if complexity == QueryComplexity.SIMPLE:
        if intent == QueryIntent.FACTUAL:
            return ReasoningStrategy.EVIDENCE_ATTRIBUTION
        return ReasoningStrategy.DIRECT
    
    # Multi-hop queries benefit from Self-Ask
    if complexity == QueryComplexity.MULTI_HOP:
        return ReasoningStrategy.SELF_ASK
    
    # Complex queries with explanatory/causal intent use CoT
    if complexity in (QueryComplexity.COMPLEX, QueryComplexity.MODERATE):
        if intent in (QueryIntent.EXPLANATORY, QueryIntent.CAUSAL, QueryIntent.PROCEDURAL):
            return ReasoningStrategy.CHAIN_OF_THOUGHT
        elif intent == QueryIntent.COMPARATIVE:
            return ReasoningStrategy.SELF_ASK
    
    # Default to CoT for moderate complexity
    return ReasoningStrategy.CHAIN_OF_THOUGHT


def get_strategy_config(strategy: ReasoningStrategy) -> Dict[str, Any]:
    """Get default configuration for a strategy."""
    configs = {
        ReasoningStrategy.SELF_ASK: {
            "max_iterations": 5,
            "max_sub_questions": 3,
            "temperature": 0.4,
            "convergence_threshold": 0.85,
        },
        ReasoningStrategy.CHAIN_OF_THOUGHT: {
            "max_steps": 8,
            "temperature": 0.3,
            "interleave_mode": "adaptive",
        },
        ReasoningStrategy.EVIDENCE_ATTRIBUTION: {
            "min_confidence": 0.5,
            "citation_format": "inline",
            "track_spans": True,
        },
        ReasoningStrategy.VERIFICATION: {
            "min_sources": 2,
            "contradiction_threshold": 0.7,
            "fact_check_temperature": 0.1,
        },
        ReasoningStrategy.DIRECT: {
            "temperature": 0.3,
            "max_tokens": 500,
        },
    }
    return configs.get(strategy, configs[ReasoningStrategy.DIRECT])
