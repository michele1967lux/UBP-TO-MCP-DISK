"""
graph_rag/delegation.py

Delegation layer for LLM-based entity and relation extraction.
Handles extraction, graph building, and answer generation.

Features:
- Entity extraction with coreference resolution
- Relation extraction with evidence spans
- Query entity recognition
- Subgraph reasoning
- Answer generation from graph context

v1.0.0: Initial release
"""

from __future__ import annotations

# WARN-CV-001 fix: shared LLM response normalizer
try:
    from ubp_enterprise_hybrid.modules.cores._shared.utils import extract_llm_text as _extract_llm_text
except ImportError:
    _extract_llm_text = None  # type: ignore[assignment]

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .prompts import (
    get_template,
    detect_language,
    EntityType,
    RelationType,
    format_entities_for_relation_extraction,
    format_triples_for_reasoning,
)
from .providers import (
    Entity,
    Relation,
    Triple,
    Subgraph,
    ExtractionResult,
    GraphPath,
    GraphRAGResult,
    KnowledgeGraph,
    EntityExtractionConfig,
    RelationExtractionConfig,
    DebugConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class IModuleRegistry(Protocol):
    """Protocol for module registry."""
    def get_module(self, module_name: str) -> Optional[Any]: ...


class IEventPublisher(Protocol):
    """Protocol for event publishing."""
    async def publish(self, event_type: str, payload: Dict[str, Any]) -> None: ...


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class LLMDelegationConfig:
    """Configuration for LLM delegation."""
    llm_module: str = "inference_ollama_grok"
    llm_operation: str = "generate"
    timeout_seconds: int = 30
    max_retries: int = 2


# ============================================================================
# Graph Delegator
# ============================================================================


class GraphDelegator:
    """
    Handles LLM delegation for graph operations.
    
    Features:
    - Entity extraction
    - Relation extraction
    - Combined extraction
    - Query analysis
    - Subgraph reasoning
    - Answer generation
    """
    
    def __init__(
        self,
        llm_config: LLMDelegationConfig,
        entity_config: EntityExtractionConfig,
        relation_config: RelationExtractionConfig,
        module_registry: IModuleRegistry,
        event_publisher: Optional[IEventPublisher] = None,
        debug_config: Optional[DebugConfig] = None,
    ):
        self.llm_config = llm_config
        self.entity_config = entity_config
        self.relation_config = relation_config
        self._module_registry = module_registry
        self._event_publisher = event_publisher
        self._debug = debug_config or DebugConfig()
        
        self._llm_module: Optional[Any] = None
    
    def is_available(self) -> bool:
        """Check if LLM module is available."""
        module = self._module_registry.get_module(self.llm_config.llm_module)
        return module is not None
    
    async def _get_llm_module(self) -> Optional[Any]:
        """Get or resolve the LLM module."""
        if self._llm_module:
            return self._llm_module

        # Try cache first, then async resolution
        module = self._module_registry.get_module(self.llm_config.llm_module)
        if not module and hasattr(self._module_registry, "resolve_module"):
            module = await self._module_registry.resolve_module(self.llm_config.llm_module)
        if module:
            self._llm_module = module
        return module
    
    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        """Call the LLM module with ProviderMapper fallback chain."""
        # Try primary module first (cached)
        module = await self._get_llm_module()
        if module:
            try:
                return await self._execute_llm_call(module, prompt, temperature, max_tokens)
            except Exception as e:
                logger.warning(f"[GRAPH] Primary LLM failed: {e}, trying fallback chain")
                self._llm_module = None  # Clear cache to force re-resolution

        # Fallback: walk ProviderMapper chain
        try:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper
            chain = ProviderMapper.resolve_chain("rag")
            for module_name, provider_name in chain:
                if module_name == self.llm_config.llm_module:
                    continue  # Skip already-failed primary
                fallback_module = self._module_registry.get_module(module_name)
                if not fallback_module and hasattr(self._module_registry, "resolve_module"):
                    fallback_module = await self._module_registry.resolve_module(module_name)
                if fallback_module:
                    try:
                        result = await self._execute_llm_call(fallback_module, prompt, temperature, max_tokens)
                        logger.info(f"[GRAPH] Fallback succeeded with {module_name}")
                        return result
                    except Exception as fb_err:
                        logger.warning(f"[GRAPH] Fallback {module_name} also failed: {fb_err}")
        except Exception as ie:
            logger.warning(
                f"[GRAPH] ProviderMapper NOT AVAILABLE during fallback chain walk. "
                f"Cannot attempt alternative providers. Cause: {ie}"
            )

        raise RuntimeError(f"LLM module '{self.llm_config.llm_module}' not available (all fallbacks exhausted)")

    async def _execute_llm_call(self, module: Any, prompt: str, temperature: float, max_tokens: int) -> str:
        """Execute a single LLM call on given module."""
        if self._debug.log_extractions:
            logger.debug(f"[GRAPH] Prompt:\n{prompt[:500]}...")

        operation = getattr(module, self.llm_config.llm_operation, None)
        if not operation:
            raise RuntimeError(f"Operation '{self.llm_config.llm_operation}' not found")

        result = await asyncio.wait_for(
            operation(prompt=prompt, temperature=temperature, max_tokens=max_tokens),
            timeout=self.llm_config.timeout_seconds,
        )

        # WARN-CV-001: shared normalizer
        if _extract_llm_text is not None:
            response = _extract_llm_text(result)
        elif isinstance(result, dict):
            response = result.get("text") or result.get("response") or result.get("content", "")
        else:
            response = str(result)

        if self._debug.log_extractions:
            logger.debug(f"[GRAPH] Response:\n{response[:500]}...")

        return response
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to extract JSON block
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try full response
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {"raw_response": response}
    
    # ========================================================================
    # Entity Extraction
    # ========================================================================
    
    async def extract_entities(
        self,
        text: str,
        doc_id: str,
        language: str = "auto",
    ) -> List[Entity]:
        """Extract entities from text."""
        if language == "auto":
            language = detect_language(text)
        
        start_time = time.perf_counter()
        
        prompt = get_template("entity_extraction", language).format(text=text)
        response = await self._call_llm(prompt, temperature=self.entity_config.temperature)
        data = self._parse_json_response(response)
        
        entities = []
        for item in data.get("entities", [])[:self.entity_config.max_entities_per_chunk]:
            confidence = float(item.get("confidence", 0.5))
            if confidence < self.entity_config.min_confidence:
                continue
            
            entity_type_str = item.get("type", "custom").lower()
            try:
                entity_type = EntityType(entity_type_str)
            except ValueError:
                entity_type = EntityType.CUSTOM
            
            entity = Entity(
                entity_id=str(uuid.uuid4()),
                text=item.get("text", ""),
                normalized=item.get("normalized", item.get("text", "")),
                entity_type=entity_type,
                confidence=confidence,
                source_docs={doc_id},
            )
            entities.append(entity)
        
        if self._debug.log_extractions:
            logger.info(f"[GRAPH] Extracted {len(entities)} entities from doc {doc_id}")
        
        return entities
    
    # ========================================================================
    # Relation Extraction
    # ========================================================================
    
    async def extract_relations(
        self,
        text: str,
        entities: List[Entity],
        doc_id: str,
        language: str = "auto",
    ) -> List[Relation]:
        """Extract relations between entities."""
        if not entities:
            return []
        
        if language == "auto":
            language = detect_language(text)
        
        # Format entities for prompt
        entities_text = format_entities_for_relation_extraction([e.to_dict() for e in entities])
        
        prompt = get_template("relation_extraction", language).format(
            text=text,
            entities=entities_text,
        )
        
        response = await self._call_llm(prompt, temperature=self.relation_config.temperature)
        data = self._parse_json_response(response)
        
        # Build entity lookup
        entity_lookup = {e.normalized.lower(): e for e in entities}
        
        relations = []
        for item in data.get("relations", [])[:self.relation_config.max_relations_per_chunk]:
            confidence = float(item.get("confidence", 0.5))
            if confidence < self.relation_config.min_confidence:
                continue
            
            source_name = item.get("source", "").lower()
            target_name = item.get("target", "").lower()
            
            source_entity = entity_lookup.get(source_name)
            target_entity = entity_lookup.get(target_name)
            
            if not source_entity or not target_entity:
                continue
            
            relation_type_str = item.get("relation_type", "related_to").lower()
            try:
                relation_type = RelationType(relation_type_str)
            except ValueError:
                relation_type = RelationType.RELATED_TO
            
            relation = Relation(
                relation_id=str(uuid.uuid4()),
                source_id=source_entity.entity_id,
                target_id=target_entity.entity_id,
                relation_type=relation_type,
                confidence=confidence,
                evidence=item.get("evidence", ""),
                bidirectional=item.get("bidirectional", False),
                source_docs={doc_id},
            )
            relations.append(relation)
        
        if self._debug.log_extractions:
            logger.info(f"[GRAPH] Extracted {len(relations)} relations from doc {doc_id}")
        
        return relations
    
    # ========================================================================
    # Combined Extraction
    # ========================================================================
    
    async def extract_combined(
        self,
        text: str,
        doc_id: str,
        language: str = "auto",
    ) -> ExtractionResult:
        """Extract entities and relations in a single pass."""
        if language == "auto":
            language = detect_language(text)
        
        start_time = time.perf_counter()
        
        prompt = get_template("combined_extraction", language).format(text=text)
        response = await self._call_llm(prompt, temperature=0.1, max_tokens=3000)
        data = self._parse_json_response(response)
        
        # Parse entities
        entities = []
        entity_id_map = {}  # prompt_id -> entity
        
        for item in data.get("entities", [])[:self.entity_config.max_entities_per_chunk]:
            confidence = float(item.get("confidence", 0.5))
            if confidence < self.entity_config.min_confidence:
                continue
            
            entity_type_str = item.get("type", "custom").lower()
            try:
                entity_type = EntityType(entity_type_str)
            except ValueError:
                entity_type = EntityType.CUSTOM
            
            entity = Entity(
                entity_id=str(uuid.uuid4()),
                text=item.get("text", ""),
                normalized=item.get("normalized", item.get("text", "")),
                entity_type=entity_type,
                confidence=confidence,
                source_docs={doc_id},
            )
            entities.append(entity)
            
            prompt_id = item.get("id", entity.normalized)
            entity_id_map[prompt_id] = entity
        
        # Parse relations
        relations = []
        triples = []
        
        for item in data.get("relations", [])[:self.relation_config.max_relations_per_chunk]:
            confidence = float(item.get("confidence", 0.5))
            if confidence < self.relation_config.min_confidence:
                continue
            
            source_id = item.get("source_id", "")
            target_id = item.get("target_id", "")
            
            source_entity = entity_id_map.get(source_id)
            target_entity = entity_id_map.get(target_id)
            
            if not source_entity or not target_entity:
                continue
            
            relation_type_str = item.get("relation_type", "related_to").lower()
            try:
                relation_type = RelationType(relation_type_str)
            except ValueError:
                relation_type = RelationType.RELATED_TO
            
            relation = Relation(
                relation_id=str(uuid.uuid4()),
                source_id=source_entity.entity_id,
                target_id=target_entity.entity_id,
                relation_type=relation_type,
                confidence=confidence,
                evidence=item.get("evidence", ""),
                source_docs={doc_id},
            )
            relations.append(relation)
            
            triples.append(Triple(
                subject=source_entity.normalized,
                predicate=relation_type.value,
                object=target_entity.normalized,
                confidence=confidence,
                evidence=item.get("evidence", ""),
            ))
        
        extraction_time = (time.perf_counter() - start_time) * 1000
        
        main_topics = data.get("summary", {}).get("main_topics", [])
        
        if self._debug.log_extractions:
            logger.info(
                f"[GRAPH] Combined extraction: {len(entities)} entities, "
                f"{len(relations)} relations in {extraction_time:.2f}ms"
            )
        
        return ExtractionResult(
            doc_id=doc_id,
            entities=entities,
            relations=relations,
            triples=triples,
            main_topics=main_topics,
            language=language,
            extraction_time_ms=extraction_time,
        )
    
    # ========================================================================
    # Query Entity Recognition
    # ========================================================================
    
    async def recognize_query_entities(
        self,
        query: str,
        language: str = "auto",
    ) -> Dict[str, Any]:
        """Recognize entities and intent in a query."""
        if language == "auto":
            language = detect_language(query)
        
        prompt = get_template("query_entity_recognition", language).format(query=query)
        response = await self._call_llm(prompt, temperature=0.1)
        data = self._parse_json_response(response)
        
        # Parse query entities
        query_entities = []
        for item in data.get("query_entities", []):
            entity_type_str = item.get("type", "concept").lower()
            try:
                entity_type = EntityType(entity_type_str)
            except ValueError:
                entity_type = EntityType.CONCEPT
            
            query_entities.append({
                "text": item.get("text", ""),
                "type": entity_type,
                "normalized": item.get("normalized", item.get("text", "")),
                "is_main_subject": item.get("is_main_subject", False),
            })
        
        return {
            "query_entities": query_entities,
            "query_intent": data.get("query_intent", ""),
            "relation_hints": data.get("relation_hints", []),
            "expansion_terms": data.get("expansion_terms", []),
        }
    
    # ========================================================================
    # Subgraph Reasoning
    # ========================================================================
    
    async def reason_over_subgraph(
        self,
        query: str,
        subgraph: Subgraph,
        language: str = "auto",
    ) -> Dict[str, Any]:
        """Reason over a subgraph to extract relevant facts."""
        if language == "auto":
            language = detect_language(query)
        
        triples = subgraph.get_triples()
        triples_text = format_triples_for_reasoning([t.to_tuple() for t in triples])
        
        prompt = get_template("subgraph_reasoning", language).format(
            query=query,
            triples=triples_text,
        )
        
        response = await self._call_llm(prompt, temperature=0.2)
        data = self._parse_json_response(response)
        
        return {
            "relevant_paths": data.get("relevant_paths", []),
            "key_facts": data.get("key_facts", []),
            "knowledge_gaps": data.get("knowledge_gaps", []),
            "confidence": float(data.get("confidence", 0.5)),
            "can_answer": data.get("can_answer", True),
        }
    
    # ========================================================================
    # Answer Generation
    # ========================================================================
    
    async def generate_answer(
        self,
        query: str,
        subgraph: Subgraph,
        reasoning_result: Dict[str, Any],
        language: str = "auto",
    ) -> Dict[str, Any]:
        """Generate answer from graph context."""
        if language == "auto":
            language = detect_language(query)
        
        # Format context
        triples = subgraph.get_triples()
        context_lines = []
        for triple in triples[:50]:
            context_lines.append(f"- {triple.subject} --[{triple.predicate}]--> {triple.object}")
        context_text = "\n".join(context_lines)
        
        facts_text = "\n".join([f"- {fact}" for fact in reasoning_result.get("key_facts", [])])
        
        paths_text = ""
        for path_info in reasoning_result.get("relevant_paths", [])[:5]:
            path = path_info.get("path", [])
            paths_text += f"- {' -> '.join(path)}\n"
        
        prompt = get_template("answer_generation", language).format(
            query=query,
            context=context_text,
            facts=facts_text or "No specific facts extracted.",
            paths=paths_text or "No specific paths identified.",
        )
        
        response = await self._call_llm(prompt, temperature=0.3)
        data = self._parse_json_response(response)
        
        return {
            "answer": data.get("answer", "Unable to generate answer from graph context."),
            "supporting_entities": data.get("supporting_entities", []),
            "supporting_relations": data.get("supporting_relations", []),
            "confidence": float(data.get("confidence", 0.5)),
            "completeness": data.get("completeness", "partial"),
            "caveats": data.get("caveats", []),
        }
    
    # ========================================================================
    # Health Check
    # ========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of delegation."""
        try:
            llm_module = await self._get_llm_module()
            
            return {
                "status": "available" if llm_module else "degraded",
                "llm_module": self.llm_config.llm_module,
                "llm_available": llm_module is not None,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
