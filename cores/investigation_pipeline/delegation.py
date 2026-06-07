"""
investigation_pipeline/delegation.py

Delegation layer for LLM operations.
Handles multi-strategy investigation generation by delegating to inference modules.

ZERO direct imports from other modules - uses DI for resolution.

v1.0.0: Multi-Strategy Investigation Generation
- Decomposition: Break query into aspects
- Chain-of-Thought: Logical reasoning steps
- Semantic Expansion: Synonyms and related concepts
- Cross-Reference: Dependencies and related features
- Adaptive: Auto-select based on classification

Supports fallback chains for robustness.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .prompts import (
    get_investigation_prompt,
    get_chain_of_thought_prompt,
    get_semantic_expansion_prompt,
    get_cross_reference_prompt,
    get_simple_fallback_prompt,
    QUERY_CATEGORIES,
)
from .providers import (
    InvestigationStrategy,
    QueryClassification,
    QueryClassifier,
    InvestigationQuestion,
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
    """Configuration for LLM delegation."""
    llm_module: str = "inference_ollama_grok"
    llm_operation: str = "generate"
    provider: str = "grok"
    timeout_seconds: int = 30
    max_retries: int = 2
    fallback_enabled: bool = True
    fallback_chain: List[str] = field(
        default_factory=lambda: ["decomposition", "chain_of_thought", "simple"]
    )


@dataclass
class InvestigationGenerationConfig:
    """Configuration for investigation generation."""
    num_questions: int = 5
    temperature: float = 0.7
    max_tokens: int = 500
    strategy: str = "adaptive"


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    questions: List[str]
    strategy_used: str
    raw_response: str
    time_ms: float
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "questions": self.questions,
            "strategy_used": self.strategy_used,
            "raw_response": self.raw_response,
            "time_ms": self.time_ms,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "retries": self.retries,
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


class ParseError(DelegationError):
    """Failed to parse LLM response."""
    pass


# ============================================================================
# InvestigationDelegator
# ============================================================================


class InvestigationDelegator:
    """
    Delegates investigation generation to LLM inference modules.
    
    Supports multiple strategies:
    - Decomposition: Breaks query into aspects
    - Chain-of-Thought: Logical reasoning steps
    - Semantic Expansion: Synonyms and related concepts
    - Cross-Reference: Dependencies and related features
    - Adaptive: Auto-selects based on classification
    
    Implements fallback chains for robustness.
    """

    def __init__(
        self,
        config: LLMDelegationConfig,
        module_registry: IModuleRegistry,
        event_publisher: Optional[IEventPublisher] = None,
        debug_config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.module_registry = module_registry
        self.event_publisher = event_publisher
        self.debug_config = debug_config or {}
        
        # Initialize classifier
        self.classifier = QueryClassifier(
            categories=QUERY_CATEGORIES,
            default_category="technical",
        )
        
        # Strategy handlers
        self._strategy_handlers = {
            InvestigationStrategy.DECOMPOSITION.value: self._generate_decomposition,
            InvestigationStrategy.CHAIN_OF_THOUGHT.value: self._generate_chain_of_thought,
            InvestigationStrategy.SEMANTIC_EXPANSION.value: self._generate_semantic_expansion,
            InvestigationStrategy.CROSS_REFERENCE.value: self._generate_cross_reference,
            InvestigationStrategy.ADAPTIVE.value: self._generate_adaptive,
        }

    async def _get_llm_module(self) -> Any:
        """Get or resolve the LLM module from registry (async with lazy resolution)."""
        if not self.module_registry:
            raise ModuleNotFoundError("Module registry not available")

        # Try sync cache first
        module = self.module_registry.get_module(self.config.llm_module)
        if module:
            return module

        # Fallback: async resolution via DI container (like hyde_pipeline)
        if hasattr(self.module_registry, "resolve_module"):
            module = await self.module_registry.resolve_module(self.config.llm_module)
            if module:
                return module

        raise ModuleNotFoundError(f"Module '{self.config.llm_module}' not found")

    async def _generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Generate text using LLM module."""
        module = await self._get_llm_module()
        
        # Log prompt if debug enabled
        if self.debug_config.get("log_prompts", False):
            logger.debug(f"[INVESTIGATION] Prompt:\n{prompt[:500]}...")
        
        try:
            # Call the generate operation
            result = await module.generate(
                prompt=prompt,
                provider=self.config.provider,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract text from result
            if isinstance(result, dict):
                response = result.get("text", result.get("response", str(result)))
            else:
                response = str(result)
            
            # Log response if debug enabled
            if self.debug_config.get("log_responses", False):
                logger.debug(f"[INVESTIGATION] Response:\n{response[:500]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMGenerationError(f"Generation failed: {e}") from e

    async def _publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Publish event if publisher available."""
        if self.event_publisher:
            try:
                await self.event_publisher.publish(event_type, payload)
            except Exception as e:
                logger.warning(f"Failed to publish event: {e}")

    async def generate_investigation(
        self,
        query: str,
        num_questions: int = 5,
        strategy: str = "adaptive",
        category: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate investigation questions using specified strategy.
        
        Args:
            query: User's original query
            num_questions: Number of questions to generate
            strategy: Strategy to use (adaptive, decomposition, chain_of_thought, etc.)
            category: Pre-classified category (optional)
            
        Returns:
            GenerationResult with generated questions
        """
        start_time = time.perf_counter()
        
        # Classify query if not provided
        classification = self.classifier.classify(query)
        effective_category = category or classification.category
        
        # Log strategy selection if debug enabled
        if self.debug_config.get("log_strategy_selection", True):
            logger.info(
                f"[INVESTIGATION] Strategy: {strategy}, Category: {effective_category}, "
                f"Language: {classification.language}"
            )
        
        # Get handler for strategy
        handler = self._strategy_handlers.get(strategy)
        if not handler:
            logger.warning(f"Unknown strategy '{strategy}', falling back to decomposition")
            handler = self._strategy_handlers[InvestigationStrategy.DECOMPOSITION.value]
        
        # Try primary generation
        try:
            questions = await handler(
                query=query,
                num_questions=num_questions,
                category=effective_category,
                language=classification.language,
            )
            
            # Validate result
            if questions and len(questions) >= 1:
                elapsed = (time.perf_counter() - start_time) * 1000
                
                await self._publish_event(
                    "investigation.generation.completed",
                    {
                        "query": query[:100],
                        "strategy": strategy,
                        "questions_count": len(questions),
                        "time_ms": elapsed,
                    },
                )
                
                return GenerationResult(
                    questions=questions,
                    strategy_used=strategy,
                    raw_response="",
                    time_ms=elapsed,
                )
            else:
                raise LLMGenerationError("No valid questions generated")
            
        except Exception as e:
            logger.warning(f"Primary generation failed: {e}")
            
            # Try fallback chain
            if self.config.fallback_enabled:
                return await self._execute_fallback_chain(
                    query=query,
                    num_questions=num_questions,
                    category=effective_category,
                    language=classification.language,
                    original_error=str(e),
                    start_time=start_time,
                )
            
            # No fallback, return empty result
            elapsed = (time.perf_counter() - start_time) * 1000
            return GenerationResult(
                questions=[],
                strategy_used=strategy,
                raw_response="",
                time_ms=elapsed,
                used_fallback=False,
                fallback_reason=str(e),
            )

    async def _execute_fallback_chain(
        self,
        query: str,
        num_questions: int,
        category: str,
        language: str,
        original_error: str,
        start_time: float,
    ) -> GenerationResult:
        """Execute fallback chain until success or exhaustion."""
        
        if self.debug_config.get("log_fallback_triggers", True):
            logger.info(f"[INVESTIGATION] Triggering fallback chain: {self.config.fallback_chain}")
        
        for fallback_strategy in self.config.fallback_chain:
            try:
                logger.info(f"[INVESTIGATION] Trying fallback strategy: {fallback_strategy}")
                
                if fallback_strategy == "simple":
                    questions = await self._generate_simple_fallback(
                        query=query,
                        num_questions=num_questions,
                        language=language,
                    )
                else:
                    handler = self._strategy_handlers.get(fallback_strategy)
                    if handler:
                        questions = await handler(
                            query=query,
                            num_questions=num_questions,
                            category=category,
                            language=language,
                        )
                    else:
                        continue
                
                if questions and len(questions) >= 1:
                    elapsed = (time.perf_counter() - start_time) * 1000
                    
                    if self.debug_config.get("log_fallback_triggers", True):
                        logger.info(
                            f"[INVESTIGATION] Fallback succeeded with {fallback_strategy}: "
                            f"{len(questions)} questions"
                        )
                    
                    await self._publish_event(
                        "investigation.fallback.succeeded",
                        {
                            "query": query[:100],
                            "strategy": fallback_strategy,
                            "questions_count": len(questions),
                            "original_error": original_error,
                        },
                    )
                    
                    return GenerationResult(
                        questions=questions,
                        strategy_used=fallback_strategy,
                        raw_response="",
                        time_ms=elapsed,
                        used_fallback=True,
                        fallback_reason=original_error,
                    )
                    
            except Exception as e:
                logger.warning(f"Fallback {fallback_strategy} failed: {e}")
                continue
        
        # All fallbacks exhausted
        elapsed = (time.perf_counter() - start_time) * 1000
        
        logger.error("[INVESTIGATION] All fallbacks exhausted, returning empty result")
        
        await self._publish_event(
            "investigation.fallback.exhausted",
            {
                "query": query[:100],
                "original_error": original_error,
            },
        )
        
        return GenerationResult(
            questions=[],
            strategy_used="none",
            raw_response="",
            time_ms=elapsed,
            used_fallback=True,
            fallback_reason="All fallbacks exhausted",
        )

    # ========================================================================
    # Strategy Handlers
    # ========================================================================

    async def _generate_decomposition(
        self,
        query: str,
        num_questions: int,
        category: str,
        language: str,
    ) -> List[str]:
        """Generate questions using decomposition strategy."""
        prompt = get_investigation_prompt(
            query=query,
            n=num_questions,
            strategy="decomposition",
            category=category,
        )
        
        response = await self._generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=500,
        )
        
        return self._parse_questions(response)

    async def _generate_chain_of_thought(
        self,
        query: str,
        num_questions: int,
        category: str,
        language: str,
    ) -> List[str]:
        """Generate questions using chain-of-thought reasoning."""
        prompt = get_chain_of_thought_prompt(
            query=query,
            n=num_questions,
            language=language,
        )
        
        response = await self._generate(
            prompt=prompt,
            temperature=0.8,
            max_tokens=600,
        )
        
        return self._parse_questions(response)

    async def _generate_semantic_expansion(
        self,
        query: str,
        num_questions: int,
        category: str,
        language: str,
    ) -> List[str]:
        """Generate questions using semantic expansion."""
        prompt = get_semantic_expansion_prompt(
            query=query,
            n=num_questions,
            language=language,
        )
        
        response = await self._generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=500,
        )
        
        return self._parse_questions(response)

    async def _generate_cross_reference(
        self,
        query: str,
        num_questions: int,
        category: str,
        language: str,
    ) -> List[str]:
        """Generate questions using cross-reference strategy."""
        prompt = get_cross_reference_prompt(
            query=query,
            n=num_questions,
            category=category,
            language=language,
        )
        
        response = await self._generate(
            prompt=prompt,
            temperature=0.6,
            max_tokens=500,
        )
        
        return self._parse_questions(response)

    async def _generate_adaptive(
        self,
        query: str,
        num_questions: int,
        category: str,
        language: str,
    ) -> List[str]:
        """
        Generate questions using adaptive strategy selection.
        
        Uses classification to pick the best strategy for the query.
        """
        # Get preferred strategy from category config
        cat_config = QUERY_CATEGORIES.get(category, {})
        preferred = cat_config.get("preferred_strategy", "decomposition")
        
        if self.debug_config.get("log_strategy_selection", True):
            logger.info(
                f"[INVESTIGATION] Adaptive: category={category}, "
                f"preferred_strategy={preferred}"
            )
        
        # Map to handler
        handler = self._strategy_handlers.get(preferred)
        if handler and handler != self._generate_adaptive:  # Avoid recursion
            return await handler(
                query=query,
                num_questions=num_questions,
                category=category,
                language=language,
            )
        
        # Default to decomposition
        return await self._generate_decomposition(
            query=query,
            num_questions=num_questions,
            category=category,
            language=language,
        )

    async def _generate_simple_fallback(
        self,
        query: str,
        num_questions: int,
        language: str,
    ) -> List[str]:
        """Generate questions using simple fallback template."""
        prompt = get_simple_fallback_prompt(
            query=query,
            n=num_questions,
            language=language,
        )
        
        response = await self._generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=400,
        )
        
        return self._parse_questions(response)

    # ========================================================================
    # Parsing
    # ========================================================================

    def _parse_questions(self, response: str) -> List[str]:
        """
        Parse questions from LLM response.
        
        Tries multiple parsing strategies:
        1. JSON array
        2. Numbered list
        3. Bullet points
        4. Line by line
        """
        questions: List[str] = []
        response = response.strip()
        
        # Try JSON array first
        try:
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                if isinstance(parsed, list):
                    questions = [str(q).strip() for q in parsed if q]
                    if questions:
                        return self._clean_questions(questions)
        except json.JSONDecodeError:
            pass
        
        # Try numbered list (1. 2. 3.)
        numbered_pattern = r'^\s*(?:\d+[\.\)]\s*|[-•*]\s*)(.+)$'
        for line in response.split('\n'):
            match = re.match(numbered_pattern, line.strip())
            if match:
                q = match.group(1).strip()
                if q:
                    questions.append(q)
        
        if questions:
            return self._clean_questions(questions)
        
        # Fall back to line by line
        for line in response.split('\n'):
            line = line.strip()
            if line and len(line) > 10 and '?' in line:
                questions.append(line)
        
        return self._clean_questions(questions)

    def _clean_questions(self, questions: List[str]) -> List[str]:
        """Clean and validate questions."""
        cleaned: List[str] = []
        
        for q in questions:
            # Remove common prefixes
            q = re.sub(r'^[\d\.\)\-\•\*]+\s*', '', q)
            q = re.sub(r'^(Question|Q|Domanda)[\s\d:]*:?\s*', '', q, flags=re.IGNORECASE)
            q = q.strip()
            
            # Validate
            if len(q) < 10:
                continue
            if len(q) > 500:
                q = q[:500]
            
            # Ensure ends with question mark
            if not q.endswith('?'):
                q += '?'
            
            # Capitalize first letter
            if q and q[0].islower():
                q = q[0].upper() + q[1:]
            
            cleaned.append(q)
        
        return cleaned

    # ========================================================================
    # Multi-Strategy Parallel Execution
    # ========================================================================

    async def generate_multi_strategy(
        self,
        query: str,
        num_questions: int = 5,
        strategies: Optional[List[str]] = None,
    ) -> Dict[str, GenerationResult]:
        """
        Generate questions using multiple strategies in parallel.
        
        Args:
            query: User's query
            num_questions: Questions per strategy
            strategies: List of strategies to use (defaults to all)
            
        Returns:
            Dict mapping strategy name to GenerationResult
        """
        import asyncio
        
        if strategies is None:
            strategies = [
                InvestigationStrategy.DECOMPOSITION.value,
                InvestigationStrategy.CHAIN_OF_THOUGHT.value,
                InvestigationStrategy.SEMANTIC_EXPANSION.value,
            ]
        
        # Classify query once
        classification = self.classifier.classify(query)
        
        # Create tasks
        tasks = {}
        for strategy in strategies:
            tasks[strategy] = self.generate_investigation(
                query=query,
                num_questions=num_questions,
                strategy=strategy,
                category=classification.category,
            )
        
        # Execute in parallel
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Map results
        output: Dict[str, GenerationResult] = {}
        for strategy, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Strategy {strategy} failed: {result}")
                output[strategy] = GenerationResult(
                    questions=[],
                    strategy_used=strategy,
                    raw_response="",
                    time_ms=0,
                    used_fallback=False,
                    fallback_reason=str(result),
                )
            else:
                output[strategy] = result
        
        return output

    # ========================================================================
    # Health Check
    # ========================================================================

    async def is_available_async(self) -> bool:
        """Check if LLM delegation is available (async with lazy resolution)."""
        try:
            await self._get_llm_module()
            return True
        except ModuleNotFoundError:
            return False

    def is_available(self) -> bool:
        """Check if LLM delegation is available (sync, cache-only)."""
        if not self.module_registry:
            return False
        module = self.module_registry.get_module(self.config.llm_module)
        return module is not None

    async def health_check(self) -> Dict[str, Any]:
        """Check delegation health."""
        try:
            module = await self._get_llm_module()
            return {
                "status": "available",
                "module": self.config.llm_module,
                "fallback_enabled": self.config.fallback_enabled,
                "fallback_chain": self.config.fallback_chain,
            }
        except ModuleNotFoundError:
            return {
                "status": "unavailable",
                "module": self.config.llm_module,
                "fallback_enabled": self.config.fallback_enabled,
            }
