"""
query_expansion_pipeline/prompts.py

Prompt templates for LLM-based query expansion.

Templates:
- Semantic expansion
- Query decomposition
- Contextual expansion
- Intent-aware expansion
- Cross-lingual expansion

Supports: English (en), Italian (it)

Version: 1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class PromptTemplates:
    """Prompt templates for query expansion."""
    
    # ========================================================================
    # Semantic Expansion
    # ========================================================================
    
    SEMANTIC_EXPANSION_EN = """Generate {num_variants} semantic variations of this search query.

Requirements for each variation:
- Preserve the original intent and meaning
- Use different phrasing, synonyms, or sentence structure
- Be a valid, standalone search query
- Be diverse from other variations

Original query: {query}

Return ONLY the variations, one per line, without numbering or explanations:"""

    SEMANTIC_EXPANSION_IT = """Genera {num_variants} variazioni semantiche di questa query di ricerca.

Requisiti per ogni variazione:
- Preservare l'intento e il significato originale
- Usare frasi diverse, sinonimi o struttura diversa
- Essere una query di ricerca valida e autonoma
- Essere diversa dalle altre variazioni

Query originale: {query}

Restituisci SOLO le variazioni, una per riga, senza numerazione o spiegazioni:"""

    # ========================================================================
    # Query Decomposition
    # ========================================================================
    
    DECOMPOSITION_EN = """Break down this complex query into simpler, independent sub-questions.

Requirements:
- Each sub-question should be answerable independently
- Cover all aspects of the original query
- Keep the core intent of each part
- Be specific and focused

Complex query: {query}

Generate up to {max_subqueries} sub-questions, one per line:"""

    DECOMPOSITION_IT = """Scomponi questa query complessa in sotto-domande più semplici e indipendenti.

Requisiti:
- Ogni sotto-domanda deve essere rispondibile indipendentemente
- Coprire tutti gli aspetti della query originale
- Mantenere l'intento principale di ogni parte
- Essere specifico e focalizzato

Query complessa: {query}

Genera fino a {max_subqueries} sotto-domande, una per riga:"""

    # ========================================================================
    # Contextual Expansion
    # ========================================================================
    
    CONTEXTUAL_EN = """Given this conversation context, generate a more specific and complete version of the current query.

Previous messages:
{chat_history}

Current query: {query}

Rewrite the query to:
- Be more specific and self-contained
- Incorporate relevant context from the conversation
- Resolve any pronouns or references
- Be a complete, standalone search query

Rewritten query:"""

    CONTEXTUAL_IT = """Dato questo contesto di conversazione, genera una versione più specifica e completa della query attuale.

Messaggi precedenti:
{chat_history}

Query attuale: {query}

Riscrivi la query per:
- Essere più specifica e autonoma
- Incorporare il contesto rilevante dalla conversazione
- Risolvere pronomi o riferimenti
- Essere una query di ricerca completa e autonoma

Query riscritta:"""

    # ========================================================================
    # Intent-Aware Expansion
    # ========================================================================
    
    INTENT_AWARE_EN = """Generate search query variations tailored to this specific intent type.

Original query: {query}
Detected intent: {intent}

For {intent} queries, generate variations that:
{intent_guidance}

Return {num_variants} variations, one per line:"""

    INTENT_GUIDANCE = {
        "informational": "- Focus on gathering comprehensive information\n- Include variations asking for explanations, details, and context",
        "definition": "- Focus on understanding meaning and concepts\n- Include variations asking what something is, means, or signifies",
        "procedural": "- Focus on steps, processes, and how-to\n- Include variations asking for instructions, guides, and tutorials",
        "comparison": "- Focus on differences, similarities, trade-offs\n- Include variations comparing features, pros/cons, advantages",
        "factual": "- Focus on specific facts and data\n- Include variations asking for numbers, dates, names, statistics",
        "opinion": "- Focus on recommendations and evaluations\n- Include variations asking for reviews, best practices, suggestions",
    }

    # ========================================================================
    # Keyword Extraction
    # ========================================================================
    
    KEYWORD_EXTRACTION_EN = """Extract the most important keywords and key phrases from this query.

Query: {query}

Requirements:
- Identify core concepts and named entities
- Include technical terms and domain-specific vocabulary
- Exclude common stopwords and filler words
- Order by importance

Return up to {max_keywords} keywords/phrases, one per line:"""

    KEYWORD_EXTRACTION_IT = """Estrai le parole chiave e le frasi chiave più importanti da questa query.

Query: {query}

Requisiti:
- Identificare concetti principali ed entità nominate
- Includere termini tecnici e vocabolario specifico del dominio
- Escludere stopwords comuni e parole di riempimento
- Ordinare per importanza

Restituisci fino a {max_keywords} parole/frasi chiave, una per riga:"""

    # ========================================================================
    # Reformulation
    # ========================================================================
    
    REFORMULATION_EN = """Reformulate this query in different ways while preserving the core meaning.

Original query: {query}

Generate variations including:
1. As a "what" question
2. As a "how" question  
3. As a statement seeking information
4. Using different vocabulary

Return 4 reformulations, one per line:"""

    REFORMULATION_IT = """Riformula questa query in modi diversi mantenendo il significato principale.

Query originale: {query}

Genera variazioni includendo:
1. Come domanda "cosa/che cosa"
2. Come domanda "come"
3. Come affermazione che cerca informazioni
4. Usando vocabolario diverso

Restituisci 4 riformulazioni, una per riga:"""

    # ========================================================================
    # Template Selection
    # ========================================================================
    
    def semantic_expansion(
        self,
        query: str,
        num_variants: int = 3,
        language: str = "en",
    ) -> str:
        """Get semantic expansion prompt."""
        template = self.SEMANTIC_EXPANSION_IT if language == "it" else self.SEMANTIC_EXPANSION_EN
        return template.format(query=query, num_variants=num_variants)
    
    def decomposition(
        self,
        query: str,
        max_subqueries: int = 5,
        language: str = "en",
    ) -> str:
        """Get decomposition prompt."""
        template = self.DECOMPOSITION_IT if language == "it" else self.DECOMPOSITION_EN
        return template.format(query=query, max_subqueries=max_subqueries)
    
    def contextual(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        language: str = "en",
    ) -> str:
        """Get contextual expansion prompt."""
        template = self.CONTEXTUAL_IT if language == "it" else self.CONTEXTUAL_EN
        
        history_text = self._format_history(chat_history)
        
        return template.format(query=query, chat_history=history_text)
    
    def intent_aware(
        self,
        query: str,
        intent: str,
        num_variants: int = 3,
        language: str = "en",
    ) -> str:
        """Get intent-aware expansion prompt."""
        guidance = self.INTENT_GUIDANCE.get(intent, self.INTENT_GUIDANCE["informational"])
        
        return self.INTENT_AWARE_EN.format(
            query=query,
            intent=intent,
            intent_guidance=guidance,
            num_variants=num_variants,
        )
    
    def keyword_extraction(
        self,
        query: str,
        max_keywords: int = 5,
        language: str = "en",
    ) -> str:
        """Get keyword extraction prompt."""
        template = self.KEYWORD_EXTRACTION_IT if language == "it" else self.KEYWORD_EXTRACTION_EN
        return template.format(query=query, max_keywords=max_keywords)
    
    def reformulation(
        self,
        query: str,
        language: str = "en",
    ) -> str:
        """Get reformulation prompt."""
        template = self.REFORMULATION_IT if language == "it" else self.REFORMULATION_EN
        return template.format(query=query)
    
    def _format_history(
        self,
        history: List[Dict[str, str]],
        max_messages: int = 5,
    ) -> str:
        """Format chat history for prompt."""
        recent = history[-max_messages:] if len(history) > max_messages else history
        
        lines = []
        for msg in recent:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def detect_language(self, text: str) -> str:
        """Simple language detection."""
        italian_indicators = {
            "il", "la", "di", "che", "è", "un", "per", "con", "non",
            "sono", "come", "cosa", "questo", "quello", "anche",
        }
        
        words = set(text.lower().split())
        italian_count = len(words & italian_indicators)
        
        return "it" if italian_count >= 2 else "en"
