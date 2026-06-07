"""
hyde_pipeline/pipeline.py

Pipeline orchestrator for HyDE operations.
Executes configurable multi-step processing.

Steps:
1. classify_query - Detect domain and language
2. select_format - Choose optimal format
3. generate_document - Create HyDE document
4. quality_assurance - Score document quality
5. hallucination_check - Detect fabricated content
6. refinement - Iterative improvement (optional)
7. chunking - Semantic chunking for embedding
8. format_output - Prepare final result

v1.0.0: Initial release with full pipeline support
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .providers import (
    HyDEDocument,
    HyDEResult,
    DocumentChunk,
    DomainClassification,
    QualityAssessment,
    HallucinationCheck,
    EnsembleResult,
    RefinementResult,
    QualityLevel,
    DomainClassifier,
    QualityAssuranceProvider,
    HallucinationDetector,
    DocumentChunker,
    EnsembleFusion,
    DebugConfig,
)
from .delegation import HyDEDelegator, GenerationResult

logger = logging.getLogger(__name__)


# ============================================================================
# Pipeline Configuration
# ============================================================================


@dataclass
class StepConfig:
    """Configuration for a single pipeline step."""
    enabled: bool = True
    timeout: int = 30
    required: bool = False  # If True, step failure stops pipeline
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "timeout": self.timeout,
            "required": self.required,
        }


@dataclass
class PipelineConfig:
    """Configuration for the HyDE pipeline."""
    default_timeout_seconds: int = 60
    fail_fast: bool = False
    steps: Dict[str, StepConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure all steps have config
        default_steps = {
            "classify_query": StepConfig(enabled=True, timeout=5),
            "select_format": StepConfig(enabled=True, timeout=2),
            "generate_document": StepConfig(enabled=True, timeout=30, required=True),
            "quality_assurance": StepConfig(enabled=True, timeout=10),
            "hallucination_check": StepConfig(enabled=True, timeout=5),
            "refinement": StepConfig(enabled=True, timeout=20),
            "chunking": StepConfig(enabled=True, timeout=5),
            "format_output": StepConfig(enabled=True, timeout=2),
        }
        for step_name, default_config in default_steps.items():
            if step_name not in self.steps:
                self.steps[step_name] = default_config
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_timeout_seconds": self.default_timeout_seconds,
            "fail_fast": self.fail_fast,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        steps = {}
        steps_data = data.get("steps", {})
        for step_name, step_data in steps_data.items():
            if isinstance(step_data, dict):
                steps[step_name] = StepConfig(
                    enabled=step_data.get("enabled", True),
                    timeout=int(step_data.get("timeout", 30)),
                    required=step_data.get("required", False),
                )
        
        return cls(
            default_timeout_seconds=int(data.get("default_timeout_seconds", 60)),
            fail_fast=data.get("fail_fast", False),
            steps=steps,
        )


# ============================================================================
# Pipeline Context
# ============================================================================


@dataclass
class PipelineContext:
    """Context passed through pipeline steps."""
    query: str
    session_id: str
    
    # Input parameters
    format_type: Optional[str] = None
    domain: Optional[str] = None
    language: Optional[str] = None
    num_documents: int = 1
    enable_refinement: bool = True
    enable_ensemble: bool = False
    max_length: int = 400
    
    # Step results
    classification: Optional[DomainClassification] = None
    selected_format: Optional[str] = None
    generation_result: Optional[GenerationResult] = None
    document: Optional[HyDEDocument] = None
    ensemble_result: Optional[EnsembleResult] = None
    quality_assessment: Optional[QualityAssessment] = None
    hallucination_check: Optional[HallucinationCheck] = None
    refinement_result: Optional[RefinementResult] = None
    chunks: List[DocumentChunk] = field(default_factory=list)
    
    # Execution metadata
    step_times: Dict[str, float] = field(default_factory=dict)
    step_errors: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "session_id": self.session_id,
            "format_type": self.format_type,
            "domain": self.domain,
            "language": self.language,
            "classification": self.classification.to_dict() if self.classification else None,
            "selected_format": self.selected_format,
            "document": self.document.to_dict() if self.document else None,
            "quality_assessment": self.quality_assessment.to_dict() if self.quality_assessment else None,
            "hallucination_check": self.hallucination_check.to_dict() if self.hallucination_check else None,
            "chunk_count": len(self.chunks),
            "step_times": self.step_times,
            "step_errors": self.step_errors,
        }


# ============================================================================
# HyDE Pipeline
# ============================================================================


class HyDEPipeline:
    """
    Configurable multi-step pipeline for HyDE document generation.
    
    Orchestrates:
    - Query classification
    - Format selection
    - Document generation (single or ensemble)
    - Quality assessment
    - Hallucination detection
    - Iterative refinement
    - Semantic chunking
    """
    
    AVAILABLE_STEPS = [
        "classify_query",
        "select_format",
        "generate_document",
        "quality_assurance",
        "hallucination_check",
        "refinement",
        "chunking",
        "format_output",
    ]
    
    def __init__(
        self,
        delegator: HyDEDelegator,
        classifier: DomainClassifier,
        qa_provider: QualityAssuranceProvider,
        hallucination_detector: HallucinationDetector,
        chunker: DocumentChunker,
        ensemble_fusion: EnsembleFusion,
        config: PipelineConfig,
        debug_config: Optional[DebugConfig] = None,
    ):
        self.delegator = delegator
        self.classifier = classifier
        self.qa_provider = qa_provider
        self.hallucination_detector = hallucination_detector
        self.chunker = chunker
        self.ensemble_fusion = ensemble_fusion
        self.config = config
        self.debug_config = debug_config or DebugConfig()
        
        # Step handlers
        self._step_handlers: Dict[str, Callable] = {
            "classify_query": self._step_classify_query,
            "select_format": self._step_select_format,
            "generate_document": self._step_generate_document,
            "quality_assurance": self._step_quality_assurance,
            "hallucination_check": self._step_hallucination_check,
            "refinement": self._step_refinement,
            "chunking": self._step_chunking,
            "format_output": self._step_format_output,
        }
    
    async def execute(
        self,
        query: str,
        session_id: str,
        format_type: Optional[str] = None,
        domain: Optional[str] = None,
        language: Optional[str] = None,
        num_documents: int = 1,
        enable_refinement: bool = True,
        enable_ensemble: bool = False,
        max_length: int = 400,
        pipeline_config: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
    ) -> HyDEResult:
        """
        Execute the HyDE pipeline.
        
        Args:
            query: User's query
            session_id: Session identifier
            format_type: Document format override
            domain: Domain override
            language: Language override
            num_documents: Documents to generate (ensemble)
            enable_refinement: Enable iterative refinement
            enable_ensemble: Enable ensemble generation
            max_length: Maximum document length
            pipeline_config: Step configuration override
            ctx: Security context
            
        Returns:
            HyDEResult with document, chunks, and metadata
        """
        start_time = time.perf_counter()
        
        # Apply config overrides
        if pipeline_config:
            self._apply_config_override(pipeline_config)
        
        # Create context
        context = PipelineContext(
            query=query,
            session_id=session_id,
            format_type=format_type,
            domain=domain,
            language=language,
            num_documents=num_documents,
            enable_refinement=enable_refinement,
            enable_ensemble=enable_ensemble,
            max_length=max_length,
        )
        
        # Execute steps
        for step_name in self.AVAILABLE_STEPS:
            step_config = self.config.steps.get(step_name, StepConfig())
            
            if not step_config.enabled:
                continue
            
            handler = self._step_handlers.get(step_name)
            if not handler:
                continue
            
            step_start = time.perf_counter()
            
            try:
                # Execute with timeout
                await asyncio.wait_for(
                    handler(context),
                    timeout=step_config.timeout,
                )
                
            except asyncio.TimeoutError:
                error_msg = f"Step '{step_name}' timed out after {step_config.timeout}s"
                context.step_errors[step_name] = error_msg
                logger.warning(f"[HYDE] {error_msg}")
                
                if step_config.required or self.config.fail_fast:
                    raise RuntimeError(error_msg)
                    
            except Exception as e:
                error_msg = f"Step '{step_name}' failed: {e}"
                context.step_errors[step_name] = error_msg
                logger.error(f"[HYDE] {error_msg}")
                
                if step_config.required or self.config.fail_fast:
                    raise
            
            context.step_times[step_name] = (time.perf_counter() - step_start) * 1000
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Build result
        if not context.document:
            raise RuntimeError("Pipeline failed to generate document")
        
        return HyDEResult(
            session_id=session_id,
            query=query,
            document=context.document,
            chunks=context.chunks,
            classification=context.classification or DomainClassification(
                query=query,
                domain=context.domain or "general",
                confidence=1.0,
                language=context.language or "en",
            ),
            quality_assessment=context.quality_assessment,
            hallucination_check=context.hallucination_check,
            refinement_applied=context.refinement_result is not None,
            time_ms=total_time,
            step_times=context.step_times,
            step_errors=context.step_errors,
        )
    
    def _apply_config_override(self, config_override: Dict[str, Any]) -> None:
        """Apply runtime configuration override."""
        for step_name, step_config in config_override.items():
            if step_name in self.config.steps:
                if isinstance(step_config, dict):
                    if "enabled" in step_config:
                        self.config.steps[step_name].enabled = step_config["enabled"]
                    if "timeout" in step_config:
                        self.config.steps[step_name].timeout = step_config["timeout"]
    
    # ========================================================================
    # Step Handlers
    # ========================================================================
    
    async def _step_classify_query(self, ctx: PipelineContext) -> None:
        """Classify query for domain and language."""
        classification = self.classifier.classify(ctx.query)
        ctx.classification = classification
        
        # Set detected values if not overridden
        if not ctx.domain:
            ctx.domain = classification.domain
        if not ctx.language:
            ctx.language = classification.language
        
        if self.debug_config.trace_execution:
            logger.debug(
                f"[HYDE] Classified: domain={classification.domain}, "
                f"language={classification.language}, confidence={classification.confidence:.3f}"
            )
    
    async def _step_select_format(self, ctx: PipelineContext) -> None:
        """Select optimal document format."""
        if ctx.format_type:
            # Use provided format
            ctx.selected_format = ctx.format_type
        elif ctx.classification and ctx.classification.preferred_formats:
            # Use domain's preferred format
            ctx.selected_format = ctx.classification.preferred_formats[0]
        else:
            # Default to answer
            ctx.selected_format = "answer"
        
        if self.debug_config.trace_execution:
            logger.debug(f"[HYDE] Selected format: {ctx.selected_format}")
    
    async def _step_generate_document(self, ctx: PipelineContext) -> None:
        """Generate HyDE document(s)."""
        if ctx.enable_ensemble and ctx.num_documents > 1:
            # Ensemble generation
            ensemble = await self.delegator.generate_ensemble(
                query=ctx.query,
                count=ctx.num_documents,
                domain=ctx.domain,
                language=ctx.language,
            )
            ctx.ensemble_result = EnsembleResult(
                documents=ensemble.documents,
                fused_document=None,
                strategy="ensemble",
                diversity_score=ensemble.diversity_score,
                generation_time_ms=ensemble.time_ms,
            )
            
            # Fuse documents
            if ensemble.documents:
                fused = self.ensemble_fusion.fuse(ensemble.documents)
                ctx.ensemble_result.fused_document = fused
                ctx.document = fused
        else:
            # Single document generation
            result = await self.delegator.generate_document(
                query=ctx.query,
                format_type=ctx.selected_format or "answer",
                domain=ctx.domain,
                language=ctx.language,
                max_length=ctx.max_length,
            )
            ctx.generation_result = result
            ctx.document = result.document
        
        if self.debug_config.trace_execution:
            logger.debug(
                f"[HYDE] Generated document: {len(ctx.document.content)} chars"
            )
    
    async def _step_quality_assurance(self, ctx: PipelineContext) -> None:
        """Assess document quality."""
        if not ctx.document:
            return
        
        assessment = self.qa_provider.assess(
            document=ctx.document,
            original_query=ctx.query,
        )
        ctx.quality_assessment = assessment
        ctx.document.quality_score = assessment.overall_score
        
        if self.debug_config.log_qa_scores:
            logger.info(
                f"[HYDE] Quality: {assessment.overall_score:.2f} "
                f"({assessment.quality_level.value})"
            )
    
    async def _step_hallucination_check(self, ctx: PipelineContext) -> None:
        """Check for hallucinated content."""
        if not ctx.document:
            return
        
        check = self.hallucination_detector.check(ctx.document)
        ctx.hallucination_check = check
        
        # Apply confidence penalty
        if check.hallucination_detected:
            ctx.document.confidence *= (1.0 - 0.2 * len(check.suspicious_elements))
            ctx.document.confidence = max(ctx.document.confidence, 0.0)
        
        if self.debug_config.log_hallucination_checks:
            logger.info(
                f"[HYDE] Hallucination check: {check.recommendation}, "
                f"confidence={check.confidence:.3f}"
            )
    
    async def _step_refinement(self, ctx: PipelineContext) -> None:
        """Refine document if quality is low."""
        if not ctx.document or not ctx.enable_refinement:
            return
        
        # Check if refinement needed
        if ctx.quality_assessment:
            if ctx.quality_assessment.overall_score >= 6.0:
                return  # Good enough, skip refinement
            
            # Select strategy based on issues
            issues = ctx.quality_assessment.issues
            if "Low relevance" in str(issues):
                strategy = "focus"
            elif "Low information" in str(issues):
                strategy = "expand"
            elif "Insufficient terminology" in str(issues):
                strategy = "technical"
            else:
                strategy = "expand"
            
            # Refine
            result = await self.delegator.refine_document(
                document=ctx.document,
                strategy=strategy,
                quality_score=ctx.quality_assessment.overall_score,
                issues=issues,
            )
            
            ctx.refinement_result = RefinementResult(
                original_document=ctx.document,
                refined_document=result.refined_document,
                iterations=1,
                score_improvement=0,  # Will be updated by re-assessment
                strategies_applied=[strategy],
                refinement_time_ms=result.time_ms,
            )
            
            # Use refined document
            ctx.document = result.refined_document
            
            if self.debug_config.log_refinement_steps:
                logger.info(
                    f"[HYDE] Refined with '{strategy}': "
                    f"{len(result.original_document.content)} -> {len(result.refined_document.content)} chars"
                )
    
    async def _step_chunking(self, ctx: PipelineContext) -> None:
        """Chunk document for embedding."""
        if not ctx.document:
            return
        
        chunks = self.chunker.chunk(ctx.document)
        ctx.chunks = chunks
        
        if self.debug_config.log_chunking:
            logger.debug(f"[HYDE] Created {len(chunks)} chunks")
    
    async def _step_format_output(self, ctx: PipelineContext) -> None:
        """Format final output."""
        # Ensure document has all metadata
        if ctx.document:
            if not ctx.document.metadata.get("pipeline_processed"):
                ctx.document.metadata["pipeline_processed"] = True
                ctx.document.metadata["session_id"] = ctx.session_id
    
    # ========================================================================
    # Configuration Management
    # ========================================================================
    
    def get_config(self) -> Dict[str, Any]:
        """Get current pipeline configuration."""
        return {
            "default_timeout_seconds": self.config.default_timeout_seconds,
            "fail_fast": self.config.fail_fast,
            "steps": {
                name: {
                    "enabled": step.enabled,
                    "timeout": step.timeout,
                    "required": step.required,
                }
                for name, step in self.config.steps.items()
            },
            "available_steps": self.AVAILABLE_STEPS,
        }
    
    def update_config(self, steps_config: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update pipeline configuration."""
        for step_update in steps_config:
            step_name = step_update.get("step")
            if step_name and step_name in self.config.steps:
                if "enabled" in step_update:
                    self.config.steps[step_name].enabled = step_update["enabled"]
                if "timeout" in step_update:
                    self.config.steps[step_name].timeout = step_update["timeout"]
                if "required" in step_update:
                    self.config.steps[step_name].required = step_update["required"]
        
        return self.get_config()
