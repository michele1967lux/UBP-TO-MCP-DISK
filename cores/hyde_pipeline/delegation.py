"""
hyde_pipeline/delegation.py

Delegation layer for LLM operations.
Handles document generation, refinement, and quality assessment
by delegating to inference modules.

ZERO direct imports from other modules - uses DI for resolution.

Features:
- Multi-format document generation
- Adaptive temperature based on format
- Iterative refinement
- Ensemble generation with diversity
- Fallback chain support
- Debug logging

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
from typing import Any, Callable, Dict, List, Optional, Protocol

from .prompts import (
    build_hyde_prompt,
    get_format_template,
    get_domain_context,
    get_refinement_prompt,
    detect_domain,
    detect_language,
    QUALITY_ASSESSMENT_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
    ENSEMBLE_FUSION_PROMPT,
)
from .providers import (
    HyDEDocument,
    DocumentFormat,
    Domain,
    QualityLevel,
    RefinementStrategy,
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
# Configuration
# ============================================================================


@dataclass
class LLMDelegationConfig:
    """Configuration for LLM delegation.
    v6.0.1: Removed model field — provider-only resolution.
    """
    llm_module: str = "inference_ollama_grok"
    llm_operation: str = "generate"
    timeout_seconds: int = 30
    max_retries: int = 2
    fallback_enabled: bool = True
    fallback_chain: List[str] = field(default_factory=lambda: ["answer", "technical_doc", "faq"])
    provider: Optional[str] = None


@dataclass
class FormatGenerationConfig:
    """Per-format generation configuration."""
    temperature: float = 0.5
    max_tokens: int = 600
    max_length: int = 400


# ============================================================================
# Result Classes
# ============================================================================


@dataclass
class GenerationResult:
    """Result from document generation."""
    document: HyDEDocument
    raw_response: str
    format_type: str
    domain: str
    language: str
    time_ms: float
    fallback_used: bool = False
    fallback_format: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "format_type": self.format_type,
            "domain": self.domain,
            "language": self.language,
            "time_ms": round(self.time_ms, 2),
            "fallback_used": self.fallback_used,
            "fallback_format": self.fallback_format,
        }


@dataclass
class RefinementResult:
    """Result from document refinement."""
    original_document: HyDEDocument
    refined_document: HyDEDocument
    strategy: str
    iterations: int
    score_before: float
    score_after: float
    time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_document": self.original_document.to_dict(),
            "refined_document": self.refined_document.to_dict(),
            "strategy": self.strategy,
            "iterations": self.iterations,
            "score_before": round(self.score_before, 2),
            "score_after": round(self.score_after, 2),
            "improvement": round(self.score_after - self.score_before, 2),
            "time_ms": round(self.time_ms, 2),
        }


@dataclass
class EnsembleGenerationResult:
    """Result from ensemble generation."""
    documents: List[HyDEDocument]
    formats_used: List[str]
    temperatures_used: List[float]
    diversity_score: float
    time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": [d.to_dict() for d in self.documents],
            "formats_used": self.formats_used,
            "temperatures_used": self.temperatures_used,
            "diversity_score": round(self.diversity_score, 3),
            "document_count": len(self.documents),
            "time_ms": round(self.time_ms, 2),
        }


# ============================================================================
# Format Configuration
# ============================================================================

# Per-format generation parameters
FORMAT_CONFIGS: Dict[str, FormatGenerationConfig] = {
    "answer": FormatGenerationConfig(temperature=0.5, max_tokens=500, max_length=400),
    "technical_doc": FormatGenerationConfig(temperature=0.3, max_tokens=700, max_length=600),
    "faq": FormatGenerationConfig(temperature=0.4, max_tokens=450, max_length=350),
    "code_snippet": FormatGenerationConfig(temperature=0.2, max_tokens=600, max_length=500),
    "tutorial": FormatGenerationConfig(temperature=0.4, max_tokens=800, max_length=700),
    "troubleshooting": FormatGenerationConfig(temperature=0.3, max_tokens=600, max_length=500),
    "article": FormatGenerationConfig(temperature=0.6, max_tokens=900, max_length=800),
}


# ============================================================================
# HyDE Delegator
# ============================================================================


class HyDEDelegator:
    """
    Handles LLM delegation for HyDE document generation.
    
    Features:
    - Multi-format generation with adaptive parameters
    - Iterative refinement
    - Ensemble generation with diversity
    - Fallback chain support
    - Comprehensive logging
    """
    
    def __init__(
        self,
        config: LLMDelegationConfig,
        module_registry: IModuleRegistry,
        event_publisher: Optional[IEventPublisher] = None,
        debug_config: Optional[Dict[str, bool]] = None,
    ):
        self.config = config
        self._module_registry = module_registry
        self._event_publisher = event_publisher
        self._debug = debug_config or {}
        
        # Cached module reference
        self._llm_module: Optional[Any] = None
    
    def is_available(self) -> bool:
        """Check if LLM module is available."""
        if self._llm_module:
            return True
        
        module = self._module_registry.get_module(self.config.llm_module)
        return module is not None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of LLM delegation."""
        try:
            module = await self._get_llm_module()
            if module:
                return {
                    "status": "available",
                    "module": self.config.llm_module,
                    "operation": self.config.llm_operation,
                }
            return {
                "status": "unavailable",
                "module": self.config.llm_module,
                "error": "Module not loaded",
            }
        except Exception as e:
            return {
                "status": "error",
                "module": self.config.llm_module,
                "error": str(e),
            }
    
    async def _get_llm_module(self) -> Optional[Any]:
        """Get or resolve the LLM module."""
        if self._llm_module:
            return self._llm_module
        
        # Try to get from registry
        module = self._module_registry.get_module(self.config.llm_module)
        if module:
            self._llm_module = module
            return module
        
        # Try async resolution if available
        if hasattr(self._module_registry, "resolve_module"):
            module = await self._module_registry.resolve_module(self.config.llm_module)
            if module:
                self._llm_module = module
                return module
        
        logger.warning(f"LLM module '{self.config.llm_module}' not available")
        return None
    
    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 600,
    ) -> str:
        """Call the LLM module."""
        module = await self._get_llm_module()
        if not module:
            raise RuntimeError(f"LLM module '{self.config.llm_module}' not available")
        
        # Log prompt if debug enabled
        if self._debug.get("log_prompts"):
            logger.debug(f"[HYDE] Prompt:\n{prompt[:500]}...")
        
        # Call the LLM
        operation = getattr(module, self.config.llm_operation, None)
        if not operation:
            raise RuntimeError(f"Operation '{self.config.llm_operation}' not found")
        
        try:
            # Build kwargs — v6.0.1: provider-only, no model
            call_kwargs = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if self.config.provider:
                call_kwargs["provider"] = self.config.provider

            result = await asyncio.wait_for(
                operation(**call_kwargs),
                timeout=self.config.timeout_seconds,
            )
            
            # Extract text from result (WARN-CV-001: shared normalizer)
            if _extract_llm_text is not None:
                response = _extract_llm_text(result)
            elif isinstance(result, dict):
                response = result.get("text") or result.get("response") or result.get("content", "")
            else:
                response = str(result)
            
            # Log response if debug enabled
            if self._debug.get("log_responses"):
                logger.debug(f"[HYDE] Response:\n{response[:500]}...")
            
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"[HYDE] LLM call timeout after {self.config.timeout_seconds}s")
            raise
        except Exception as e:
            logger.error(f"[HYDE] LLM call failed: {e}")
            raise
    
    # ========================================================================
    # Document Generation
    # ========================================================================
    
    async def generate_document(
        self,
        query: str,
        format_type: str = "answer",
        domain: str = "auto",
        language: str = "auto",
        min_length: int = 100,
        max_length: int = 400,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        """
        Generate a HyDE document.
        
        Args:
            query: User's query
            format_type: Document format (answer, technical_doc, etc.)
            domain: Domain (ai_ml, devops, etc.) or 'auto'
            language: Language code or 'auto'
            min_length: Minimum document length
            max_length: Maximum document length
            temperature: Temperature override
            
        Returns:
            GenerationResult with generated document
        """
        start_time = time.perf_counter()
        
        # Auto-detect if needed
        if language == "auto":
            language = detect_language(query)
        if domain == "auto":
            domain, _ = detect_domain(query)
        
        # Get format-specific config
        format_config = FORMAT_CONFIGS.get(format_type, FORMAT_CONFIGS["answer"])
        temp = temperature if temperature is not None else format_config.temperature
        max_len = max_length or format_config.max_length
        
        # Build prompt
        prompt = build_hyde_prompt(
            query=query,
            format_type=format_type,
            domain=domain,
            language=language,
            min_length=min_length,
            max_length=max_len,
        )
        
        # Generate
        fallback_used = False
        fallback_format = None
        
        try:
            response = await self._call_llm(
                prompt=prompt,
                temperature=temp,
                max_tokens=format_config.max_tokens,
            )
            
            # Clean response
            content = self._clean_response(response)
            
        except Exception as e:
            # Try fallback chain
            if self.config.fallback_enabled:
                content, fallback_format = await self._try_fallback(
                    query=query,
                    domain=domain,
                    language=language,
                    min_length=min_length,
                    max_length=max_len,
                    error=str(e),
                )
                fallback_used = True
            else:
                raise
        
        # Create document
        document = HyDEDocument(
            document_id=str(uuid.uuid4()),
            content=content,
            query=query,
            format_type=fallback_format or format_type,
            domain=domain,
            language=language,
            metadata={
                "temperature": temp,
                "max_length": max_len,
            },
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Publish event
        if self._event_publisher:
            await self._event_publisher.publish(
                "hyde.generation.completed",
                {
                    "document_id": document.document_id,
                    "format_type": format_type,
                    "domain": domain,
                    "time_ms": elapsed_ms,
                },
            )
        
        if self._debug.get("log_strategy_selection"):
            logger.info(
                f"[HYDE] Generated {format_type} document for domain '{domain}' "
                f"({len(content)} chars, {elapsed_ms:.1f}ms)"
            )
        
        return GenerationResult(
            document=document,
            raw_response=response if not fallback_used else content,
            format_type=fallback_format or format_type,
            domain=domain,
            language=language,
            time_ms=elapsed_ms,
            fallback_used=fallback_used,
            fallback_format=fallback_format,
        )
    
    async def _try_fallback(
        self,
        query: str,
        domain: str,
        language: str,
        min_length: int,
        max_length: int,
        error: str,
    ) -> tuple[str, str]:
        """Try fallback formats on failure."""
        if self._debug.get("log_fallback_triggers"):
            logger.warning(f"[HYDE] Primary generation failed: {error}, trying fallback")

        fallback_errors: List[str] = []

        for fallback_format in self.config.fallback_chain:
            try:
                format_config = FORMAT_CONFIGS.get(fallback_format, FORMAT_CONFIGS["answer"])
                prompt = build_hyde_prompt(
                    query=query,
                    format_type=fallback_format,
                    domain=domain,
                    language=language,
                    min_length=min_length,
                    max_length=max_length,
                )

                response = await self._call_llm(
                    prompt=prompt,
                    temperature=format_config.temperature,
                    max_tokens=format_config.max_tokens,
                )

                content = self._clean_response(response)

                if self._debug.get("log_fallback_triggers"):
                    logger.info(f"[HYDE] Fallback to '{fallback_format}' succeeded")

                return content, fallback_format

            except Exception as e:
                fallback_errors.append(f"{fallback_format}: {e}")
                if self._debug.get("log_fallback_triggers"):
                    logger.warning(f"[HYDE] Fallback '{fallback_format}' failed: {e}")
                continue

        # All fallbacks failed - include all errors in message
        errors_detail = "; ".join(fallback_errors)
        raise RuntimeError(
            f"All fallback formats failed. Original error: {error}. "
            f"Fallback errors: [{errors_detail}]"
        )
    
    def _clean_response(self, response: str) -> str:
        """Clean and validate LLM response."""
        # Remove common artifacts
        content = response.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            if len(lines) > 2:
                content = "\n".join(lines[1:-1])
        
        # Remove leading labels
        prefixes = [
            "Hypothetical Document:",
            "Document:",
            "Answer:",
            "Technical Documentation:",
            "FAQ Entry:",
            "Documento Ipotetico:",
        ]
        for prefix in prefixes:
            if content.startswith(prefix):
                content = content[len(prefix):].strip()
        
        return content
    
    # ========================================================================
    # Ensemble Generation
    # ========================================================================
    
    async def generate_ensemble(
        self,
        query: str,
        count: int = 3,
        formats: Optional[List[str]] = None,
        domain: str = "auto",
        language: str = "auto",
        temperature_spread: float = 0.2,
        parallel: bool = True,
    ) -> EnsembleGenerationResult:
        """
        Generate multiple documents with diversity.
        
        Args:
            query: User's query
            count: Number of documents to generate
            formats: Specific formats to use (or auto-select)
            domain: Domain or 'auto'
            language: Language or 'auto'
            temperature_spread: Temperature variation
            parallel: Generate in parallel
            
        Returns:
            EnsembleGenerationResult with all documents
        """
        start_time = time.perf_counter()
        
        # Auto-detect
        if language == "auto":
            language = detect_language(query)
        if domain == "auto":
            domain, _ = detect_domain(query)
        
        # Select formats
        if formats:
            selected_formats = formats[:count]
        else:
            # Auto-select based on domain and diversity
            all_formats = list(FORMAT_CONFIGS.keys())
            selected_formats = all_formats[:count]
        
        # Ensure we have enough formats
        while len(selected_formats) < count:
            selected_formats.append(selected_formats[len(selected_formats) % len(FORMAT_CONFIGS)])
        
        # Calculate temperatures with spread
        base_temps = [FORMAT_CONFIGS.get(f, FORMAT_CONFIGS["answer"]).temperature for f in selected_formats]
        temperatures = []
        for i, base in enumerate(base_temps):
            # Vary temperature around base
            offset = (i / max(count - 1, 1) - 0.5) * temperature_spread
            temp = max(0.1, min(1.0, base + offset))
            temperatures.append(temp)
        
        # Generate documents
        documents = []
        
        if parallel:
            # Parallel generation
            tasks = []
            for i, (fmt, temp) in enumerate(zip(selected_formats, temperatures)):
                task = self.generate_document(
                    query=query,
                    format_type=fmt,
                    domain=domain,
                    language=language,
                    temperature=temp,
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, GenerationResult):
                    documents.append(result.document)
                elif isinstance(result, Exception):
                    logger.warning(f"[HYDE] Ensemble generation error: {result}")
        else:
            # Sequential generation
            for fmt, temp in zip(selected_formats, temperatures):
                try:
                    result = await self.generate_document(
                        query=query,
                        format_type=fmt,
                        domain=domain,
                        language=language,
                        temperature=temp,
                    )
                    documents.append(result.document)
                except Exception as e:
                    logger.warning(f"[HYDE] Ensemble generation error: {e}")
        
        # Calculate diversity
        diversity = self._calculate_diversity(documents)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        if self._debug.get("log_ensemble_details"):
            logger.info(
                f"[HYDE] Ensemble generated {len(documents)} documents, "
                f"diversity={diversity:.3f}, {elapsed_ms:.1f}ms"
            )
        
        return EnsembleGenerationResult(
            documents=documents,
            formats_used=selected_formats[:len(documents)],
            temperatures_used=temperatures[:len(documents)],
            diversity_score=diversity,
            time_ms=elapsed_ms,
        )
    
    def _calculate_diversity(self, documents: List[HyDEDocument]) -> float:
        """Calculate diversity score for documents."""
        if len(documents) < 2:
            return 1.0
        
        from difflib import SequenceMatcher
        
        similarities = []
        for i, doc1 in enumerate(documents):
            for doc2 in documents[i + 1:]:
                ratio = SequenceMatcher(
                    None,
                    doc1.content.lower(),
                    doc2.content.lower(),
                ).ratio()
                similarities.append(ratio)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        return 1.0 - avg_similarity
    
    # ========================================================================
    # Document Refinement
    # ========================================================================
    
    async def refine_document(
        self,
        document: HyDEDocument,
        strategy: str = "expand",
        quality_score: float = 0.0,
        issues: List[str] = None,
    ) -> RefinementResult:
        """
        Refine a document using iterative improvement.
        
        Args:
            document: Document to refine
            strategy: Refinement strategy (expand, focus, technical, simplify)
            quality_score: Current quality score
            issues: Identified quality issues
            
        Returns:
            RefinementResult with original and refined documents
        """
        start_time = time.perf_counter()
        issues = issues or []
        
        # Get refinement prompt
        refinement_prompt = get_refinement_prompt(strategy)
        
        prompt = refinement_prompt.format(
            query=document.query,
            document=document.content,
            score=quality_score,
            issues=", ".join(issues) if issues else "None identified",
        )
        
        # Get appropriate temperature for refinement
        temp_map = {
            "expand": 0.5,
            "focus": 0.3,
            "technical": 0.3,
            "simplify": 0.4,
        }
        temperature = temp_map.get(strategy, 0.4)
        
        # Generate refined version
        response = await self._call_llm(
            prompt=prompt,
            temperature=temperature,
            max_tokens=800,
        )
        
        content = self._clean_response(response)
        
        # Create refined document
        refined = HyDEDocument(
            document_id=str(uuid.uuid4()),
            content=content,
            query=document.query,
            format_type=document.format_type,
            domain=document.domain,
            language=document.language,
            metadata={
                **document.metadata,
                "refined": True,
                "refinement_strategy": strategy,
                "original_document_id": document.document_id,
            },
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        if self._debug.get("log_refinement_steps"):
            logger.info(
                f"[HYDE] Refined document using '{strategy}' strategy "
                f"({len(document.content)} -> {len(content)} chars, {elapsed_ms:.1f}ms)"
            )
        
        return RefinementResult(
            original_document=document,
            refined_document=refined,
            strategy=strategy,
            iterations=1,
            score_before=quality_score,
            score_after=0.0,  # Will be updated by QA
            time_ms=elapsed_ms,
        )
    
    # ========================================================================
    # LLM-Based Quality Assessment
    # ========================================================================
    
    async def assess_quality_llm(
        self,
        document: HyDEDocument,
    ) -> Dict[str, Any]:
        """
        Use LLM to assess document quality.
        
        This provides a second opinion on quality beyond rule-based assessment.
        
        Args:
            document: Document to assess
            
        Returns:
            Quality assessment dict from LLM
        """
        prompt = QUALITY_ASSESSMENT_PROMPT.format(
            query=document.query,
            format=document.format_type,
            document=document.content,
        )
        
        try:
            response = await self._call_llm(
                prompt=prompt,
                temperature=0.1,  # Low for consistent scoring
                max_tokens=500,
            )
            
            # Parse JSON response
            return self._parse_json_response(response)
            
        except Exception as e:
            logger.warning(f"[HYDE] LLM quality assessment failed: {e}")
            return {"error": str(e)}
    
    async def check_hallucination_llm(
        self,
        document: HyDEDocument,
    ) -> Dict[str, Any]:
        """
        Use LLM to check for hallucinations.
        
        Args:
            document: Document to check
            
        Returns:
            Hallucination check result from LLM
        """
        prompt = HALLUCINATION_CHECK_PROMPT.format(
            query=document.query,
            document=document.content,
        )
        
        try:
            response = await self._call_llm(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500,
            )
            
            return self._parse_json_response(response)
            
        except Exception as e:
            logger.warning(f"[HYDE] LLM hallucination check failed: {e}")
            return {"error": str(e)}
    
    async def fuse_documents_llm(
        self,
        documents: List[HyDEDocument],
        max_length: int = 600,
    ) -> HyDEDocument:
        """
        Use LLM to intelligently fuse multiple documents.
        
        Args:
            documents: Documents to fuse
            max_length: Maximum fused document length
            
        Returns:
            Fused document
        """
        if len(documents) == 1:
            return documents[0]
        
        # Build documents string
        docs_str = "\n\n---\n\n".join([
            f"Document {i+1} (Format: {d.format_type}, Score: {d.quality_score:.1f}):\n{d.content}"
            for i, d in enumerate(documents)
        ])
        
        prompt = ENSEMBLE_FUSION_PROMPT.format(
            query=documents[0].query,
            documents=docs_str,
            max_length=max_length,
        )
        
        response = await self._call_llm(
            prompt=prompt,
            temperature=0.4,
            max_tokens=800,
        )
        
        content = self._clean_response(response)
        
        # Use best document's properties as base
        best = max(documents, key=lambda d: d.quality_score)
        
        return HyDEDocument(
            document_id=str(uuid.uuid4()),
            content=content,
            query=best.query,
            format_type="fused",
            domain=best.domain,
            language=best.language,
            quality_score=sum(d.quality_score for d in documents) / len(documents),
            confidence=min(d.confidence for d in documents),
            metadata={
                "fused": True,
                "source_count": len(documents),
                "source_ids": [d.document_id for d in documents],
            },
        )
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
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
    # Convenience Methods
    # ========================================================================
    
    async def generate_for_domain(
        self,
        query: str,
        domain: str,
    ) -> GenerationResult:
        """
        Generate document with optimal format for domain.
        
        Args:
            query: User's query
            domain: Target domain
            
        Returns:
            GenerationResult optimized for domain
        """
        from .prompts import DOMAINS
        
        domain_info = DOMAINS.get(domain, DOMAINS["general"])
        preferred = domain_info.get("preferred_formats", ["answer"])
        format_type = preferred[0] if preferred else "answer"
        
        return await self.generate_document(
            query=query,
            format_type=format_type,
            domain=domain,
        )
    
    async def generate_multi_format(
        self,
        query: str,
        formats: List[str],
        domain: str = "auto",
    ) -> Dict[str, GenerationResult]:
        """
        Generate documents in multiple formats.
        
        Args:
            query: User's query
            formats: List of formats to generate
            domain: Domain or 'auto'
            
        Returns:
            Dict mapping format to result
        """
        results = {}
        
        tasks = []
        for fmt in formats:
            task = self.generate_document(
                query=query,
                format_type=fmt,
                domain=domain,
            )
            tasks.append((fmt, task))
        
        for fmt, task in tasks:
            try:
                result = await task
                results[fmt] = result
            except Exception as e:
                logger.warning(f"[HYDE] Multi-format generation failed for {fmt}: {e}")
        
        return results
