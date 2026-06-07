"""
enrichment_pipeline/delegation.py

Delegation layer for LLM operations.
Handles query expansion, HyDE generation, investigative decomposition,
and abstractive compression by delegating to inference_vllm module.

ZERO direct imports from other modules - uses DI for resolution.

v2.2.2: Added Investigative Query Decomposition (FEAT-INVEST-001)
v2.2.3: Unified Query Classification System (FEAT-CLASSIFY-001)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from .prompts import (
    get_filter_prompt,
    get_hyde_category,
    get_investigative_category,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class IModuleRegistry(Protocol):
    """Protocol for module registry."""

    def get_module(self, module_name: str) -> Optional[Any]: ...
    def is_module_loaded(self, module_name: str) -> bool: ...


class IEventPublisher(Protocol):
    """Protocol for event publishing."""

    async def publish(self, event_type: str, payload: Dict[str, Any]) -> None: ...


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class LLMDelegationConfig:
    """Configuration for LLM delegation.

    v3.6.1: Added provider field to ensure explicit provider passing to LLM module.
    v6.0.1: Removed model field — provider-only resolution.
    """

    llm_module: str = "inference_vllm"
    llm_operation: str = "generate"
    timeout_seconds: int = 30
    max_retries: int = 2
    provider: Optional[str] = None  # v3.6.1: Explicit provider for generate calls


@dataclass
class QueryExpansionConfig:
    """Configuration for query expansion."""

    num_variants: int = 3
    temperature: float = 0.7
    max_tokens: int = 200
    prompt_template: str = """Generate {num_variants} semantic variations of this search query. Each variation should express the same information need differently. Return only the variations, one per line, without numbering.

Original query: {query}

Variations:"""


@dataclass
class HyDEConfig:
    """Configuration for HyDE generation."""

    temperature: float = 0.5
    max_tokens: int = 300
    prompt_templates: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.prompt_templates is None:
            # Import templates from prompts.py to avoid duplication
            from .prompts import HYDE_TEMPLATES

            self.prompt_templates = HYDE_TEMPLATES.copy()


@dataclass
class InvestigativeConfig:
    """Configuration for investigative query decomposition (v2.2.2)."""

    num_questions: int = 5
    temperature: float = 0.7
    max_tokens: int = 400
    prompt_templates: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.prompt_templates is None:
            # Import templates from prompts.py to avoid duplication
            from .prompts import INVESTIGATIVE_TEMPLATES

            self.prompt_templates = INVESTIGATIVE_TEMPLATES.copy()


@dataclass
class AbstractiveCompressionConfig:
    """Configuration for abstractive compression."""

    temperature: float = 0.3
    max_tokens: int = 200
    prompt_template: str = """Summarize the following text concisely while preserving all key information relevant to the query. Keep only the most important facts and details.

Query: {query}

Text to summarize:
{text}

Concise summary:"""


@dataclass
class QueryExpansionResult:
    """Result from query expansion."""

    original_query: str
    expanded_queries: List[str]
    combined_query: str
    time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "expanded_queries": self.expanded_queries,
            "combined_query": self.combined_query,
            "time_ms": self.time_ms,
        }


@dataclass
class HyDEResult:
    """Result from HyDE generation."""

    hypothetical_document: str
    query: str
    document_type: str
    time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothetical_document": self.hypothetical_document,
            "query": self.query,
            "document_type": self.document_type,
            "time_ms": self.time_ms,
        }


@dataclass
class InvestigativeResult:
    """Result from investigative query decomposition (v2.2.2)."""

    investigative_questions: List[str]
    original_query: str
    time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "investigative_questions": self.investigative_questions,
            "original_query": self.original_query,
            "time_ms": self.time_ms,
        }


@dataclass
class FilterExtractionConfig:
    """Configuration for natural language → metadata filters."""

    enabled: bool = True
    temperature: float = 0.0
    max_tokens: int = 200
    allowed_fields: List[str] = field(
        default_factory=lambda: ["filename", "doc_id", "kb_id", "uploader_id", "tags"]
    )


@dataclass
class FilterExtractionResult:
    """Result from natural language filter extraction."""

    filters: Optional[Dict[str, Any]]
    entities: Optional[Dict[str, Any]]
    applied_fields: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filters": self.filters,
            "entities": self.entities,
            "applied_fields": self.applied_fields,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
        }


# ============================================================================
# Exceptions
# ============================================================================


class DelegationError(Exception):
    """Base exception for delegation errors."""

    pass


class ModuleNotFoundError(DelegationError):
    """Target module not found."""

    pass


class LLMGenerationError(DelegationError):
    """LLM generation failed."""

    pass


# ============================================================================
# LLMDelegator
# ============================================================================


class LLMDelegator:
    """
    Delegates LLM operations to inference module.

    Handles:
    - Query expansion
    - HyDE generation
    - Abstractive compression

    Uses DI container to resolve modules dynamically.
    """

    def __init__(
        self,
        config: LLMDelegationConfig,
        module_registry: IModuleRegistry,
        event_publisher: Optional[IEventPublisher] = None,
    ):
        self.config = config
        self.module_registry = module_registry
        self.event_publisher = event_publisher

        # Sub-configs
        self.expansion_config = QueryExpansionConfig()
        self.hyde_config = HyDEConfig()
        self.investigative_config = InvestigativeConfig()  # v2.2.2
        self.compression_config = AbstractiveCompressionConfig()
        self.filter_config = FilterExtractionConfig()

    async def _publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Publish event if publisher available."""
        if self.event_publisher:
            try:
                await self.event_publisher.publish(event_type, payload)
            except Exception as e:
                logger.warning(f"Failed to publish event {event_type}: {e}")

    def _get_llm_module(self) -> Any:
        """Get LLM module from registry."""
        if not self.module_registry.is_module_loaded(self.config.llm_module):
            raise ModuleNotFoundError(
                f"LLM module '{self.config.llm_module}' not loaded"
            )

        module = self.module_registry.get_module(self.config.llm_module)
        if module is None:
            raise ModuleNotFoundError(
                f"LLM module '{self.config.llm_module}' not found"
            )

        return module

    def _clean_llm_response(self, text: str) -> str:
        """
        Clean LLM response by removing special tokens and artifacts.

        Handles tokens from various LLM backends:
        - vLLM/Llama: <|assistant|>, <|user|>, <|system|>, <|end|>, <|eot_id|>
        - Grok: <|assistant|>
        - Generic: [INST], [/INST], <<SYS>>, <</SYS>>
        """
        if not text:
            return text

        # Common special tokens to remove (order matters - longer patterns first)
        special_tokens = [
            r"<\|assistant\|>",
            r"<\|user\|>",
            r"<\|system\|>",
            r"<\|end\|>",
            r"<\|eot_id\|>",
            r"<\|im_start\|>",
            r"<\|im_end\|>",
            r"<\|endoftext\|>",
            r"\[INST\]",
            r"\[/INST\]",
            r"<<SYS>>",
            r"<</SYS>>",
            r"###\s*(Assistant|User|System):?",
            r"(Assistant|User|System):",  # Only at start of line
        ]

        cleaned = text
        for token_pattern in special_tokens:
            cleaned = re.sub(token_pattern, "", cleaned, flags=re.IGNORECASE)

        # Remove leading/trailing whitespace and normalize multiple newlines
        cleaned = cleaned.strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned

    async def _generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 200,
    ) -> str:
        """Call LLM generate operation.

        v3.6.1: Now passes explicit provider from config to ensure correct
        provider is used even when module instance is shared across callers.
        v6.0.1: Removed model passing — provider-only resolution.
        v6.1.3: Use chat() instead of generate() when available.
               Reason: vLLM /v1/completions ignores chat_template_kwargs,
               so enable_thinking=false has no effect there.
               Only /v1/chat/completions processes chat_template_kwargs,
               which is the only way to disable Qwen3 thinking/reasoning.
               This eliminates reasoning traces ("Okay, the user wants...")
               from HyDE documents and all enrichment LLM outputs.
        """
        module = self._get_llm_module()

        # v6.1.3: Prefer chat() over generate() for thinking control.
        # vLLM's /v1/completions endpoint ignores chat_template_kwargs,
        # so enable_thinking=false only works via /v1/chat/completions (chat method).
        use_chat = hasattr(module, "chat")

        try:
            if use_chat:
                # Route through chat endpoint where enable_thinking=false is effective
                chat_kwargs = {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if self.config.provider:
                    chat_kwargs["provider"] = self.config.provider
                    logger.debug(
                        f"[LLMDelegator] Calling chat (for thinking control) with provider: {self.config.provider}"
                    )

                result = await module.chat(**chat_kwargs)

                # chat() returns {"message": {"content": "..."}} format
                msg = result.get("message", {})
                raw_text = msg.get("content", "") if isinstance(msg, dict) else ""

            else:
                # Fallback: module has no chat method, use generate
                if not hasattr(module, "generate"):
                    raise LLMGenerationError(
                        f"Module {self.config.llm_module} has no 'generate' or 'chat' method"
                    )

                generate_kwargs = {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if self.config.provider:
                    generate_kwargs["provider"] = self.config.provider

                result = await module.generate(**generate_kwargs)
                raw_text = result.get("text", "")

            # Clean special tokens from LLM response
            return self._clean_llm_response(raw_text)

        except Exception as e:
            raise LLMGenerationError(f"LLM generation failed: {e}") from e

    def _parse_query_variants(self, text: str) -> List[str]:
        """Parse query variants from LLM output."""
        # Split by newlines and clean up
        lines = text.strip().split("\n")
        variants = []

        for line in lines:
            # Remove numbering, bullets, etc.
            cleaned = re.sub(r"^[\d\.\)\-\*\•]+\s*", "", line.strip())
            if cleaned and len(cleaned) > 3:
                variants.append(cleaned)

        return variants

    def _parse_filter_payload(self, text: str) -> FilterExtractionResult:
        """Parse JSON payload returned by the LLM for filter extraction."""
        raw = (text or "").strip()
        if not raw:
            return FilterExtractionResult(filters=None, entities=None, raw_response=raw)

        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            payload = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return FilterExtractionResult(filters=None, entities=None, raw_response=raw)

        filters_data = payload.get("filters") or payload.get("filter") or []
        if isinstance(filters_data, dict):
            filters_iter = [filters_data]
        elif isinstance(filters_data, list):
            filters_iter = filters_data
        else:
            filters_iter = []

        conditions: List[Dict[str, Any]] = []
        applied_fields: List[str] = []

        for entry in filters_iter:
            if not isinstance(entry, dict):
                continue
            field = (entry.get("field") or entry.get("key") or "").strip()
            if not field or field not in self.filter_config.allowed_fields:
                continue
            operator = (entry.get("operator") or "equals").lower()
            value = entry.get("value")
            values = entry.get("values")
            condition = self._build_filter_condition(field, operator, value, values)
            if condition:
                applied_fields.append(field)
                conditions.append(condition)

        if not conditions:
            return FilterExtractionResult(
                filters=None,
                entities=payload.get("entities"),
                applied_fields=[],
                confidence=payload.get("confidence"),
                raw_response=raw,
            )

        filter_dict: Dict[str, Any]
        if len(conditions) == 1:
            filter_dict = conditions[0]
        else:
            filter_dict = {"$and": conditions}

        return FilterExtractionResult(
            filters=filter_dict,
            entities=payload.get("entities"),
            applied_fields=applied_fields,
            confidence=payload.get("confidence"),
            raw_response=raw,
        )

    def _build_filter_condition(
        self,
        field: str,
        operator: str,
        value: Optional[Any],
        values: Optional[Any],
    ) -> Optional[Dict[str, Any]]:
        """Build a filter condition supported by FilterBuilder."""
        raw_values = values if values is not None else value
        if raw_values is None:
            return None

        if isinstance(raw_values, list):
            normalized_values = [
                self._normalize_filter_value(v) for v in raw_values if v is not None
            ]
        else:
            normalized = self._normalize_filter_value(raw_values)
            normalized_values = [normalized] if normalized is not None else []

        normalized_values = [v for v in normalized_values if v not in (None, "")]
        if not normalized_values:
            return None

        if operator in ("in", "any") or len(normalized_values) > 1:
            return {field: {"$in": normalized_values}}

        return {field: normalized_values[0]}

    def _normalize_filter_value(self, value: Any) -> Optional[Any]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return value
        return str(value).strip()

    async def expand_query(
        self,
        query: str,
        num_variants: Optional[int] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        expansion_type: str = "semantic",
    ) -> QueryExpansionResult:
        """
        Generate query variations using LLM.

        Args:
            query: Original search query
            num_variants: Number of variants to generate
            chat_history: Previous messages for context
            expansion_type: Type of expansion (semantic, keywords, reformulate)

        Returns:
            QueryExpansionResult with variants
        """
        start = time.perf_counter()
        num_variants = num_variants or self.expansion_config.num_variants

        # Build prompt
        prompt = self.expansion_config.prompt_template.format(
            query=query,
            num_variants=num_variants,
        )

        # Add chat history context if available
        if chat_history:
            history_text = "\n".join(
                [
                    f"{m['role']}: {m['content']}"
                    for m in chat_history[-3:]  # Last 3 messages
                ]
            )
            prompt = f"Previous conversation:\n{history_text}\n\n{prompt}"

        try:
            response = await self._generate(
                prompt=prompt,
                temperature=self.expansion_config.temperature,
                max_tokens=self.expansion_config.max_tokens,
            )

            variants = self._parse_query_variants(response)

            # Ensure we have at least the original query
            if not variants:
                variants = [query]

            # Limit to requested number
            variants = variants[:num_variants]

            # Create combined query for retrieval
            all_queries = [query] + variants
            combined = " | ".join(all_queries)

            await self._publish_event(
                "enrichment.query_expansion.completed",
                {"query": query, "variants_count": len(variants)},
            )

            elapsed = (time.perf_counter() - start) * 1000

            return QueryExpansionResult(
                original_query=query,
                expanded_queries=variants,
                combined_query=combined,
                time_ms=elapsed,
            )

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            elapsed = (time.perf_counter() - start) * 1000

            # Return original query on failure
            return QueryExpansionResult(
                original_query=query,
                expanded_queries=[],
                combined_query=query,
                time_ms=elapsed,
            )

    def _detect_document_type(self, query: str) -> str:
        """
        Intelligently detect the best document type based on query content.

        Analyzes the query to determine the most appropriate HyDE template.

        v2.2.3: Now uses unified classification system from prompts.py
        (FEAT-CLASSIFY-001) for consistent categorization across all features.
        """
        return get_hyde_category(query)

    async def generate_hyde(
        self,
        query: str,
        document_type: str = "auto",
        max_length: int = 300,
    ) -> HyDEResult:
        """
        Generate Hypothetical Document Embedding (HyDE).

        Creates a hypothetical document that would answer the query,
        which can then be used for retrieval.

        Args:
            query: Search query
            document_type: Type of document to generate
            max_length: Max tokens for generated document

        Returns:
            HyDEResult with hypothetical document
        """
        start = time.perf_counter()

        # Auto-detect document type if requested
        if document_type == "auto":
            detected_type = self._detect_document_type(query)
            logger.info(
                f"HyDE: Auto-detected document type '{detected_type}' for query: {query[:50]}..."
            )
            document_type = detected_type

        # Language detection for template selection
        lang_suffix = self._detect_language_suffix(query)
        if lang_suffix and self.hyde_config.prompt_templates:
            # Try language-specific template first (e.g., "answer_it" for Italian)
            lang_specific_type = f"{document_type}{lang_suffix}"
            if lang_specific_type in self.hyde_config.prompt_templates:
                document_type = lang_specific_type
                logger.info(
                    f"HyDE: Using language-specific template '{lang_specific_type}'"
                )

        # Get prompt template for document type
        if self.hyde_config.prompt_templates is None:
            # Fallback to default if not initialized
            template = f"""Write a detailed answer to this question as if you were an expert.

Question: {{query}}

Answer:"""
        else:
            template = self.hyde_config.prompt_templates.get(
                document_type,
                self.hyde_config.prompt_templates.get("answer", ""),
            )

            # If template is empty, use default
            if not template:
                template = self.hyde_config.prompt_templates.get("answer", "")

        prompt = template.format(query=query)

        try:
            response = await self._generate(
                prompt=prompt,
                temperature=self.hyde_config.temperature,
                max_tokens=max_length,
            )

            await self._publish_event(
                "enrichment.hyde.generated",
                {"query": query, "document_type": document_type},
            )

            elapsed = (time.perf_counter() - start) * 1000

            # Validate and clean the response
            cleaned_response = response.strip()
            validation_result = self._validate_hyde_quality(
                cleaned_response, query, document_type
            )

            if not validation_result["is_valid"]:
                logger.warning(
                    f"HyDE quality validation failed: {validation_result['reason']}"
                )
                # Try to regenerate with different parameters if quality is poor
                if validation_result["can_retry"]:
                    logger.info(
                        "Attempting HyDE regeneration with adjusted parameters..."
                    )
                    try:
                        # Retry with lower temperature for more focused response
                        retry_response = await self._generate(
                            prompt=prompt,
                            temperature=max(0.1, self.hyde_config.temperature - 0.2),
                            max_tokens=max_length,
                        )
                        retry_cleaned = retry_response.strip()
                        retry_validation = self._validate_hyde_quality(
                            retry_cleaned, query, document_type
                        )

                        if retry_validation["is_valid"]:
                            cleaned_response = retry_cleaned
                            logger.info("HyDE regeneration successful")
                        else:
                            logger.warning(
                                f"HyDE regeneration also failed: {retry_validation['reason']}"
                            )
                    except Exception as retry_error:
                        logger.error(f"HyDE regeneration failed: {retry_error}")

            return HyDEResult(
                hypothetical_document=cleaned_response,
                query=query,
                document_type=document_type,
                time_ms=elapsed,
            )

        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            elapsed = (time.perf_counter() - start) * 1000

            # Return empty document on failure
            return HyDEResult(
                hypothetical_document="",
                query=query,
                document_type=document_type,
                time_ms=elapsed,
            )

    def _validate_hyde_quality(
        self, document: str, query: str, document_type: str
    ) -> Dict[str, Any]:
        """
        Validate the quality of generated HyDE document.

        Returns dict with:
        - is_valid: bool
        - reason: str (if invalid)
        - can_retry: bool
        """
        if not document or len(document.strip()) < 50:
            return {
                "is_valid": False,
                "reason": "Document too short or empty",
                "can_retry": True,
            }

        # Extract key terms from query for relevance checking
        query_lower = query.lower()
        key_terms = []

        # Add AI/ML terms if relevant
        if any(term in query_lower for term in ["ai", "ml", "rag", "llm", "neural"]):
            key_terms.extend(
                [
                    "artificial intelligence",
                    "machine learning",
                    "neural network",
                    "language model",
                ]
            )

        # Add system terms if relevant
        if any(term in query_lower for term in ["system", "server", "database", "api"]):
            key_terms.extend(["system", "server", "database", "api", "configuration"])

        # Check for off-topic content (like traffic lights when asking about RAG)
        off_topic_indicators = [
            "traffic light",
            "semaphore",
            "red light",
            "green light",
            "stop light",
            "vehicle",
            "car",
            "road",
            "intersection",
            "pedestrian",
        ]

        document_lower = document.lower()
        off_topic_matches = [
            term for term in off_topic_indicators if term in document_lower
        ]

        if off_topic_matches:
            return {
                "is_valid": False,
                "reason": f"Document appears off-topic (mentions: {', '.join(off_topic_matches[:3])})",
                "can_retry": True,
            }

        # Check for generic/uninformative content
        generic_phrases = [
            "this is a",
            "it is a",
            "the system is",
            "this feature",
            "i don't know",
            "not sure",
            "unclear",
            "unknown",
        ]

        generic_count = sum(1 for phrase in generic_phrases if phrase in document_lower)
        if generic_count > len(document.split()) * 0.1:  # More than 10% generic content
            return {
                "is_valid": False,
                "reason": "Document contains too much generic/uninformative content",
                "can_retry": True,
            }

        # Check for minimum information density
        words = document.split()
        if len(words) < 100 and not any(term in document_lower for term in key_terms):
            return {
                "is_valid": False,
                "reason": "Document lacks sufficient detail and relevant terms",
                "can_retry": True,
            }

        return {
            "is_valid": True,
            "reason": "Document passed quality validation",
            "can_retry": False,
        }

    def _detect_language_suffix(self, query: str) -> str:
        """
        Simple language detection for template selection.
        Returns '_it' for Italian, '' for English/default.

        Args:
            query: User's query text

        Returns:
            Language suffix for prompt templates
        """
        # Italian detection based on common words
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

    def _detect_investigative_type(self, query: str) -> str:
        """
        Intelligently detect the best investigative template based on query content.

        Analyzes the query to determine the most appropriate investigative approach.

        v2.2.3: Now uses unified classification system from prompts.py
        (FEAT-CLASSIFY-001) for consistent categorization across all features.
        """
        return get_investigative_category(query)

    async def generate_investigative(
        self,
        query: str,
        num_questions: Optional[int] = None,
        investigation_type: str = "auto",
    ) -> InvestigativeResult:
        """
        Generate investigative search questions (v2.2.2 - FEAT-INVEST-001).

        Instead of generating a hypothetical answer (HyDE), this generates
        specific search questions that would lead to finding the answer.
        This approach avoids hallucination on unknown terms and works
        domain-agnostically.

        Args:
            query: Original user query
            num_questions: Number of questions to generate (default: 5)

        Returns:
            InvestigativeResult with list of search questions
        """
        start = time.perf_counter()
        num_questions = num_questions or self.investigative_config.num_questions

        # Auto-detect investigation type if requested
        if investigation_type == "auto":
            detected_type = self._detect_investigative_type(query)
            logger.info(
                f"Investigative: Auto-detected type '{detected_type}' for query: {query[:50]}..."
            )
            investigation_type = detected_type

        # Language detection for template selection
        lang_suffix = self._detect_language_suffix(query)
        if lang_suffix and self.investigative_config.prompt_templates:
            # Try language-specific template first (e.g., "default_it" for Italian)
            lang_specific_type = f"{investigation_type}{lang_suffix}"
            if lang_specific_type in self.investigative_config.prompt_templates:
                investigation_type = lang_specific_type
                logger.info(
                    f"Investigative: Using language-specific template '{lang_specific_type}'"
                )

        # Get template for investigation type
        if self.investigative_config.prompt_templates is None:
            # Fallback to default if not initialized
            template = f"""You are an AI research assistant. The user asked: '{{query}}'.

You do not know the answer yet.
Generate {{n}} specific search questions that, if answered by a document, would allow you to answer the user.
Focus on definitions, components, architecture, and purpose.

Return ONLY a JSON array of strings, no explanation:"""
        else:
            template = self.investigative_config.prompt_templates.get(
                investigation_type,
                self.investigative_config.prompt_templates.get("default", ""),
            )

        prompt = template.format(query=query, n=num_questions)

        try:
            response = await self._generate(
                prompt=prompt,
                temperature=self.investigative_config.temperature,
                max_tokens=self.investigative_config.max_tokens,
            )

            # Parse investigative questions from response
            questions = self._parse_investigative_questions(response)

            # Validate and filter questions
            questions = [
                q for q in questions if isinstance(q, str) and len(q.strip()) > 5
            ]

            # Quality validation
            validation_result = self._validate_investigative_quality(
                questions, query, investigation_type
            )
            if not validation_result["is_valid"]:
                logger.warning(
                    f"Investigative quality validation failed: {validation_result['reason']}"
                )
                # Try to regenerate with different parameters if quality is poor
                if validation_result["can_retry"] and len(questions) < num_questions:
                    logger.info(
                        "Attempting investigative regeneration with adjusted parameters..."
                    )
                    try:
                        # Retry with lower temperature for more focused questions
                        retry_response = await self._generate(
                            prompt=prompt,
                            temperature=max(
                                0.1, self.investigative_config.temperature - 0.2
                            ),
                            max_tokens=self.investigative_config.max_tokens,
                        )
                        retry_questions = self._parse_investigative_questions(
                            retry_response
                        )
                        retry_questions = [
                            q
                            for q in retry_questions
                            if isinstance(q, str) and len(q.strip()) > 5
                        ]

                        if len(retry_questions) > len(questions):
                            questions = retry_questions
                            logger.info(
                                f"Investigative regeneration successful: {len(questions)} questions"
                            )
                        else:
                            logger.warning(
                                "Investigative regeneration did not improve results"
                            )
                    except Exception as retry_error:
                        logger.error(
                            f"Investigative regeneration failed: {retry_error}"
                        )

            # Limit to requested number
            questions = questions[:num_questions]

            await self._publish_event(
                "enrichment.investigative.completed",
                {"query": query, "questions_count": len(questions)},
            )

            elapsed = (time.perf_counter() - start) * 1000

            logger.info(
                f"[ENRICHMENT] Generated {len(questions)} investigative questions",
                extra={"query": query[:50], "questions": len(questions)},
            )

            return InvestigativeResult(
                investigative_questions=questions,
                original_query=query,
                time_ms=elapsed,
            )

        except Exception as e:
            logger.error(f"Investigative generation failed: {e}")
            elapsed = (time.perf_counter() - start) * 1000

            # Return empty list on failure
            return InvestigativeResult(
                investigative_questions=[],
                original_query=query,
                time_ms=elapsed,
            )

    def _parse_investigative_questions(self, response: str) -> List[str]:
        """
        Parse investigative questions from LLM response.
        Enhanced version with better JSON and fallback parsing.

        v4.2.9-FIX: Added robust cleaning for malformed LLM outputs:
        - Strips extra quotes from JSON strings
        - Handles {"question": "..."} format
        - Cleans trailing commas and quotes
        """
        questions = []
        try:
            # Try to extract JSON array from response
            response_clean = response.strip()
            # Find JSON array in response
            start_idx = response_clean.find("[")
            end_idx = response_clean.rfind("]") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_clean[start_idx:end_idx]
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    questions = [str(q) for q in parsed if q]
            else:
                # Fallback: parse line by line
                questions = self._parse_query_variants(response)
        except json.JSONDecodeError:
            # Fallback: parse line by line
            questions = self._parse_query_variants(response)

        # v4.2.9-FIX: Clean malformed questions from inconsistent LLM outputs
        # Handles: "\"question text\","  ->  "question text?"
        cleaned_questions = []
        for q in questions:
            q_clean = str(q).strip()

            # Skip non-question content
            if q_clean.startswith("[") or q_clean.startswith("{"):
                continue
            if "JSON" in q_clean or "array" in q_clean.lower():
                continue
            if q_clean.startswith('"question"'):
                # Handle {"question": "text"} format - extract the text
                if '": "' in q_clean:
                    q_clean = q_clean.split('": "', 1)[-1]

            # Strip surrounding quotes and trailing punctuation artifacts
            q_clean = q_clean.strip('"').strip("'").rstrip(",").strip()

            # Ensure it ends with ? if it looks like a question
            if q_clean and len(q_clean) > 15 and not q_clean.endswith("?"):
                # Check if it's a question-like structure
                question_starters = ["what", "how", "why", "when", "where", "which", "who",
                                     "cosa", "come", "perché", "quando", "dove", "quale", "chi"]
                if any(q_clean.lower().startswith(starter) for starter in question_starters):
                    q_clean = q_clean.rstrip(".") + "?"

            # Only keep valid questions
            if q_clean and len(q_clean) > 15:
                cleaned_questions.append(q_clean)

        return cleaned_questions if cleaned_questions else questions

    def _validate_investigative_quality(
        self, questions: List[str], query: str, investigation_type: str
    ) -> Dict[str, Any]:
        """
        Validate the quality of generated investigative questions.

        Returns dict with:
        - is_valid: bool
        - reason: str (if invalid)
        - can_retry: bool

        v4.2.9-FIX: Relaxed validation to prevent retry loops that cause 120s+ latency.
        - Reduced minimum questions from 3 to 1
        - Reduced valid question threshold from 60% to 40%
        - DISABLED retry (can_retry always False) - retry doesn't improve quality
        """
        # v4.2.9-FIX: Accept any output with at least 1 question
        if not questions or len(questions) == 0:
            return {
                "is_valid": False,
                "reason": "No questions generated",
                "can_retry": False,  # v4.2.9: Disabled retry - doesn't help
            }

        # v4.2.9-FIX: Reduced from 3 to 1 - even 1 question is useful
        if len(questions) < 1:
            return {
                "is_valid": False,
                "reason": "Too few questions generated",
                "can_retry": False,  # v4.2.9: Disabled retry
            }

        # Check for duplicate or very similar questions (keep this check but no retry)
        unique_questions = set(q.lower().strip() for q in questions)
        if len(unique_questions) < len(questions) * 0.5:  # Relaxed from 70% to 50%
            logger.warning(
                f"Investigative validation: {len(unique_questions)}/{len(questions)} unique questions"
            )
            # v4.2.9: Log warning but don't fail - duplicates are still usable

        # Check question quality (length, structure) - relaxed thresholds
        valid_questions = 0
        for q in questions:
            q_clean = q.strip()
            if len(q_clean) < 10:  # Too short
                continue
            # v4.2.9-FIX: Relaxed - don't require question mark (LLM sometimes omits it)
            # if not q_clean.endswith("?"):
            #     continue
            if len(q_clean.split()) < 3:  # Relaxed from 4 to 3 words
                continue

            valid_questions += 1

        # v4.2.9-FIX: Relaxed from 60% to 40% - accept "good enough" output
        if valid_questions < len(questions) * 0.4:
            logger.warning(
                f"Investigative validation: only {valid_questions}/{len(questions)} valid questions"
            )
            # v4.2.9: Log warning but still accept if we have at least 1 valid
            if valid_questions >= 1:
                return {
                    "is_valid": True,
                    "reason": f"Accepted {valid_questions} valid questions (relaxed mode)",
                    "can_retry": False,
                }
            return {
                "is_valid": False,
                "reason": "Too many low-quality or irrelevant questions",
                "can_retry": False,  # v4.2.9: Disabled retry
            }

        # Check for off-topic content (keep but don't retry)
        off_topic_indicators = [
            "what is the meaning of life",
            "how to make friends",
            "favorite color",
        ]

        for question in questions:
            question_lower = question.lower()
            if any(indicator in question_lower for indicator in off_topic_indicators):
                logger.warning(f"Investigative validation: off-topic question detected")
                return {
                    "is_valid": False,
                    "reason": "Questions appear off-topic or irrelevant",
                    "can_retry": False,  # v4.2.9: Disabled retry
                }

        return {
            "is_valid": True,
            "reason": "Questions passed quality validation",
            "can_retry": False,
        }

    async def extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract Qdrant-compatible filters from a natural language query."""
        if not self.filter_config.enabled:
            return FilterExtractionResult(
                filters=None, entities=None, raw_response=None
            ).to_dict()

        if not self.is_available():
            logger.debug("LLM delegator unavailable for filter extraction")
            return FilterExtractionResult(
                filters=None, entities=None, raw_response=None
            ).to_dict()

        prompt = get_filter_prompt(
            query=query, allowed_fields=self.filter_config.allowed_fields
        )

        try:
            response = await self._generate(
                prompt=prompt,
                temperature=self.filter_config.temperature,
                max_tokens=self.filter_config.max_tokens,
            )
            result = self._parse_filter_payload(response)
            if result.filters:
                await self._publish_event(
                    "enrichment.filters.extracted",
                    {
                        "fields": result.applied_fields,
                        "confidence": result.confidence,
                    },
                )
            return result.to_dict()
        except Exception as e:
            logger.warning(f"Filter extraction failed: {e}")
            return FilterExtractionResult(
                filters=None,
                entities=None,
                raw_response=str(e),
            ).to_dict()

    async def compress_abstractive(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        target_ratio: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Compress chunks using abstractive summarization.

        Args:
            query: Original query for relevance
            chunks: Chunks to compress
            target_ratio: Target compression ratio

        Returns:
            Compressed chunks
        """
        compressed = []

        for chunk in chunks:
            text = chunk.get("text", "")

            # Skip if already short
            if len(text) < 200:
                compressed.append(chunk)
                continue

            prompt = self.compression_config.prompt_template.format(
                query=query,
                text=text,
            )

            target_tokens = int(len(text) // 4 * target_ratio)

            try:
                summary = await self._generate(
                    prompt=prompt,
                    temperature=self.compression_config.temperature,
                    max_tokens=target_tokens,
                )

                compressed_chunk = {
                    **chunk,
                    "text": summary.strip(),
                    "enrichment_metadata": {
                        **chunk.get("enrichment_metadata", {}),
                        "compression": "abstractive",
                        "original_length": len(text),
                    },
                }
                compressed.append(compressed_chunk)

            except Exception as e:
                logger.warning(f"Abstractive compression failed for chunk: {e}")
                compressed.append(chunk)

        await self._publish_event(
            "enrichment.compression.completed",
            {"method": "abstractive", "chunks_count": len(compressed)},
        )

        return compressed

    def is_available(self) -> bool:
        """Check if LLM delegation is available."""
        try:
            self._get_llm_module()
            return True
        except ModuleNotFoundError:
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check delegation health."""
        try:
            module = self._get_llm_module()
            return {
                "status": "available",
                "module": self.config.llm_module,
            }
        except ModuleNotFoundError:
            return {
                "status": "unavailable",
                "module": self.config.llm_module,
            }
