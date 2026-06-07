"""
graph_rag/prompts.py

Prompt templates for Knowledge Graph RAG operations.
Multi-language support: EN/IT

v1.0.0: Initial release
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
from enum import Enum


class EntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    PRODUCT = "product"
    EVENT = "event"
    QUANTITY = "quantity"
    CUSTOM = "custom"


class RelationType(Enum):
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    CREATED_BY = "created_by"
    USED_BY = "used_by"
    RELATED_TO = "related_to"
    DEPENDS_ON = "depends_on"
    CAUSES = "causes"
    HAS_PROPERTY = "has_property"
    INSTANCE_OF = "instance_of"
    SUBCLASS_OF = "subclass_of"
    CUSTOM = "custom"


# Entity Extraction
ENTITY_EXTRACTION_EN = """Extract all named entities from the text.

Text:
{text}

For each entity provide: text, type (person/organization/location/date/technology/concept/product/event/quantity), normalized form, confidence (0-1).

Respond in JSON:
{{"entities": [{{"text": "...", "type": "...", "normalized": "...", "confidence": 0.9}}], "entity_count": N}}"""

ENTITY_EXTRACTION_IT = """Estrai tutte le entità nominate dal testo.

Testo:
{text}

Per ogni entità fornisci: testo, tipo (person/organization/location/date/technology/concept/product/event/quantity), forma normalizzata, confidenza (0-1).

Rispondi in JSON:
{{"entities": [{{"text": "...", "type": "...", "normalized": "...", "confidence": 0.9}}], "entity_count": N}}"""


# Relation Extraction
RELATION_EXTRACTION_EN = """Extract relations between entities.

Text:
{text}

Entities:
{entities}

For each relation provide: source, target, relation_type (works_for/located_in/part_of/created_by/used_by/related_to/depends_on/causes/has_property/instance_of/subclass_of), evidence, confidence.

Respond in JSON:
{{"relations": [{{"source": "...", "target": "...", "relation_type": "...", "evidence": "...", "confidence": 0.8}}], "relation_count": N}}"""

RELATION_EXTRACTION_IT = """Estrai le relazioni tra entità.

Testo:
{text}

Entità:
{entities}

Per ogni relazione fornisci: sorgente, target, tipo_relazione (works_for/located_in/part_of/created_by/used_by/related_to/depends_on/causes/has_property/instance_of/subclass_of), evidenza, confidenza.

Rispondi in JSON:
{{"relations": [{{"source": "...", "target": "...", "relation_type": "...", "evidence": "...", "confidence": 0.8}}], "relation_count": N}}"""


# Combined Extraction
COMBINED_EXTRACTION_EN = """Extract all entities and relations from the text in one pass.

Text:
{text}

Entity types: person, organization, location, date, technology, concept, product, event, quantity
Relation types: works_for, located_in, part_of, created_by, used_by, related_to, depends_on, causes, has_property, instance_of, subclass_of

Respond in JSON:
{{
    "entities": [{{"id": "e1", "text": "...", "type": "...", "normalized": "...", "confidence": 0.9}}],
    "relations": [{{"source_id": "e1", "target_id": "e2", "relation_type": "...", "evidence": "...", "confidence": 0.8}}],
    "summary": {{"entity_count": N, "relation_count": M, "main_topics": ["..."]}}
}}"""

COMBINED_EXTRACTION_IT = """Estrai tutte le entità e relazioni dal testo in un singolo passaggio.

Testo:
{text}

Tipi entità: person, organization, location, date, technology, concept, product, event, quantity
Tipi relazione: works_for, located_in, part_of, created_by, used_by, related_to, depends_on, causes, has_property, instance_of, subclass_of

Rispondi in JSON:
{{
    "entities": [{{"id": "e1", "text": "...", "type": "...", "normalized": "...", "confidence": 0.9}}],
    "relations": [{{"source_id": "e1", "target_id": "e2", "relation_type": "...", "evidence": "...", "confidence": 0.8}}],
    "summary": {{"entity_count": N, "relation_count": M, "main_topics": ["..."]}}
}}"""


# Query Entity Recognition
QUERY_ENTITY_RECOGNITION_EN = """Identify entities in this query for knowledge graph search.

Query: {query}

Respond in JSON:
{{
    "query_entities": [{{"text": "...", "type": "...", "normalized": "...", "is_main_subject": true}}],
    "query_intent": "what the user wants to know",
    "relation_hints": ["possible relation types"],
    "expansion_terms": ["related search terms"]
}}"""

QUERY_ENTITY_RECOGNITION_IT = """Identifica le entità in questa query per la ricerca nel knowledge graph.

Query: {query}

Rispondi in JSON:
{{
    "query_entities": [{{"text": "...", "type": "...", "normalized": "...", "is_main_subject": true}}],
    "query_intent": "cosa l'utente vuole sapere",
    "relation_hints": ["possibili tipi di relazione"],
    "expansion_terms": ["termini di ricerca correlati"]
}}"""


# Subgraph Reasoning
SUBGRAPH_REASONING_EN = """Analyze this knowledge graph subgraph to answer the query.

Query: {query}

Subgraph triples:
{triples}

Analyze paths, identify relevant facts, and determine if the query can be answered.

Respond in JSON:
{{
    "relevant_paths": [{{"path": ["entity1", "relation", "entity2"], "relevance_score": 0.9, "explanation": "..."}}],
    "key_facts": ["fact1", "fact2"],
    "knowledge_gaps": ["missing info"],
    "confidence": 0.8,
    "can_answer": true
}}"""

SUBGRAPH_REASONING_IT = """Analizza questo sottografo del knowledge graph per rispondere alla query.

Query: {query}

Triple del sottografo:
{triples}

Analizza i percorsi, identifica i fatti rilevanti e determina se la query può essere risposta.

Rispondi in JSON:
{{
    "relevant_paths": [{{"path": ["entità1", "relazione", "entità2"], "relevance_score": 0.9, "explanation": "..."}}],
    "key_facts": ["fatto1", "fatto2"],
    "knowledge_gaps": ["info mancante"],
    "confidence": 0.8,
    "can_answer": true
}}"""


# Answer Generation
ANSWER_GENERATION_EN = """Generate an answer using the knowledge graph context.

Query: {query}

Graph Context:
{context}

Key Facts:
{facts}

Relevant Paths:
{paths}

Use ONLY information from the graph. Cite entities and relations. Acknowledge gaps.

Respond in JSON:
{{
    "answer": "comprehensive answer",
    "supporting_entities": ["entity1", "entity2"],
    "supporting_relations": ["relation1"],
    "confidence": 0.8,
    "completeness": "complete|partial|insufficient",
    "caveats": ["limitations"]
}}"""

ANSWER_GENERATION_IT = """Genera una risposta usando il contesto del knowledge graph.

Query: {query}

Contesto del Grafo:
{context}

Fatti Chiave:
{facts}

Percorsi Rilevanti:
{paths}

Usa SOLO informazioni dal grafo. Cita entità e relazioni. Riconosci le lacune.

Rispondi in JSON:
{{
    "answer": "risposta completa",
    "supporting_entities": ["entità1", "entità2"],
    "supporting_relations": ["relazione1"],
    "confidence": 0.8,
    "completeness": "complete|partial|insufficient",
    "caveats": ["limitazioni"]
}}"""


# Template Registry
TEMPLATES = {
    "en": {
        "entity_extraction": ENTITY_EXTRACTION_EN,
        "relation_extraction": RELATION_EXTRACTION_EN,
        "combined_extraction": COMBINED_EXTRACTION_EN,
        "query_entity_recognition": QUERY_ENTITY_RECOGNITION_EN,
        "subgraph_reasoning": SUBGRAPH_REASONING_EN,
        "answer_generation": ANSWER_GENERATION_EN,
    },
    "it": {
        "entity_extraction": ENTITY_EXTRACTION_IT,
        "relation_extraction": RELATION_EXTRACTION_IT,
        "combined_extraction": COMBINED_EXTRACTION_IT,
        "query_entity_recognition": QUERY_ENTITY_RECOGNITION_IT,
        "subgraph_reasoning": SUBGRAPH_REASONING_IT,
        "answer_generation": ANSWER_GENERATION_IT,
    },
}


def get_template(template_name: str, language: str = "en") -> str:
    """Get a template by name and language."""
    lang_templates = TEMPLATES.get(language, TEMPLATES["en"])
    return lang_templates.get(template_name, TEMPLATES["en"].get(template_name, ""))


def detect_language(text: str) -> str:
    """Simple language detection."""
    italian_markers = {
        "come", "cosa", "perché", "quando", "dove", "chi", "quale",
        "il", "la", "lo", "gli", "le", "un", "una", "uno",
        "è", "sono", "sei", "siamo", "essere", "avere", "fare",
        "non", "che", "per", "con", "su", "da", "in", "di",
    }
    words = set(text.lower().split())
    italian_ratio = len(words & italian_markers) / max(len(words), 1)
    return "it" if italian_ratio > 0.15 else "en"


def get_entity_types() -> List[str]:
    return [e.value for e in EntityType]


def get_relation_types() -> List[str]:
    return [r.value for r in RelationType]


def format_entities_for_relation_extraction(entities: List[Dict[str, Any]]) -> str:
    lines = []
    for e in entities:
        lines.append(f"- {e.get('normalized', e.get('text'))}: {e.get('type', 'unknown')}")
    return "\n".join(lines)


def format_triples_for_reasoning(triples: List[Tuple[str, str, str]]) -> str:
    lines = []
    for s, r, o in triples:
        lines.append(f"({s}) --[{r}]--> ({o})")
    return "\n".join(lines)
