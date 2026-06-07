"""
investigation_pipeline/pipeline.py

Pipeline orchestrator for investigation operations.
Chains multiple investigation steps in configurable order.

Pipeline Steps:
1. classify_query: Detect category and keywords
2. select_strategy: Choose optimal strategy
3. generate_questions: Execute generation
4. quality_assurance: Validate and score
5. cross_reference: Add related questions (optional)
6. deduplicate: Remove duplicates
7. rank_questions: Sort by score
8. format_output: Final formatting
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Awaitable

from .providers import (
    InvestigationConfig,
    QualityAssuranceConfig,
    DeduplicationConfig,
    DebugConfig,
    InvestigationQuestion,
    InvestigationResult,
    QueryClassification,
    QualityAssessment,
    QualityAssuranceProvider,
    InvestigationDeduplicator,
    QueryClassifier,
    InvestigationStrategy,
)
from .delegation import (
    InvestigationDelegator,
    LLMDelegationConfig,
    GenerationResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class PipelineStepConfig:
    """Configuration for a single pipeline step."""
    step: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    steps: List[PipelineStepConfig] = field(default_factory=list)
    timeout_seconds: int = 60
    fail_fast: bool = False
    
    @classmethod
    def default(cls) -> "PipelineConfig":
        """Create default pipeline configuration."""
        return cls(
            steps=[
                PipelineStepConfig(step="classify_query", enabled=True),
                PipelineStepConfig(step="select_strategy", enabled=True),
                PipelineStepConfig(step="generate_questions", enabled=True),
                PipelineStepConfig(step="quality_assurance", enabled=True),
                PipelineStepConfig(step="cross_reference", enabled=False),
                PipelineStepConfig(step="deduplicate", enabled=True),
                PipelineStepConfig(step="rank_questions", enabled=True),
                PipelineStepConfig(step="format_output", enabled=True),
            ]
        )
    
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PipelineConfig":
        """Create from dict, merging with defaults."""
        if data is None:
            return cls.default()

        def to_bool(value: Any, default: bool = True) -> bool:
            """Convert string/bool to boolean."""
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return default

        steps = []
        for step_data in data.get("steps", data.get("default_steps", [])):
            if isinstance(step_data, dict):
                steps.append(PipelineStepConfig(
                    step=step_data.get("step", ""),
                    enabled=to_bool(step_data.get("enabled", True)),
                    config=step_data.get("config", {}),
                ))

        return cls(
            steps=steps if steps else cls.default().steps,
            timeout_seconds=int(data.get("timeout_seconds", 60)),
            fail_fast=to_bool(data.get("fail_fast", False), default=False),
        )


@dataclass
class StepResult:
    """Result from a single pipeline step."""
    step_name: str
    success: bool
    time_ms: float
    output: Any = None
    error: Optional[str] = None


@dataclass
class PipelineContext:
    """Context passed through pipeline steps."""
    original_query: str
    session_id: str
    num_questions: int
    strategy: Optional[str] = None
    classification: Optional[QueryClassification] = None
    questions: List[InvestigationQuestion] = field(default_factory=list)
    raw_questions: List[str] = field(default_factory=list)
    quality_assessment: Optional[QualityAssessment] = None
    generation_result: Optional[GenerationResult] = None
    step_times: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# InvestigationPipeline
# ============================================================================


class InvestigationPipeline:
    """
    Orchestrates investigation pipeline execution.
    
    Steps executed in order:
    1. classify_query: Detect category and keywords
    2. select_strategy: Choose optimal strategy
    3. generate_questions: Execute LLM generation
    4. quality_assurance: Validate and score questions
    5. cross_reference: Add related questions (optional)
    6. deduplicate: Remove duplicate questions
    7. rank_questions: Sort by quality score
    8. format_output: Final formatting
    """
    
    AVAILABLE_STEPS = [
        "classify_query",
        "select_strategy",
        "generate_questions",
        "quality_assurance",
        "cross_reference",
        "deduplicate",
        "rank_questions",
        "format_output",
    ]
    
    def __init__(
        self,
        delegator: InvestigationDelegator,
        qa_provider: QualityAssuranceProvider,
        deduplicator: InvestigationDeduplicator,
        classifier: QueryClassifier,
        config: Optional[PipelineConfig] = None,
        debug_config: Optional[DebugConfig] = None,
    ):
        self.delegator = delegator
        self.qa_provider = qa_provider
        self.deduplicator = deduplicator
        self.classifier = classifier
        self.config = config or PipelineConfig.default()
        self.debug_config = debug_config or DebugConfig()
        
        # Step handlers
        self._step_handlers: Dict[str, Callable] = {
            "classify_query": self._step_classify_query,
            "select_strategy": self._step_select_strategy,
            "generate_questions": self._step_generate_questions,
            "quality_assurance": self._step_quality_assurance,
            "cross_reference": self._step_cross_reference,
            "deduplicate": self._step_deduplicate,
            "rank_questions": self._step_rank_questions,
            "format_output": self._step_format_output,
        }
    
    async def execute(
        self,
        query: str,
        session_id: str,
        num_questions: int = 5,
        strategy: Optional[str] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
        ctx: Optional[Any] = None,
    ) -> InvestigationResult:
        """
        Execute full investigation pipeline.
        
        Args:
            query: Original user query
            session_id: Session identifier
            num_questions: Number of questions to generate
            strategy: Strategy override (optional)
            pipeline_config: Override pipeline configuration
            ctx: Security context
        
        Returns:
            InvestigationResult with generated questions and stats
        """
        start_time = time.perf_counter()
        
        # Parse config override
        config = PipelineConfig.from_dict(pipeline_config) if pipeline_config else self.config
        
        # Initialize context
        context = PipelineContext(
            original_query=query,
            session_id=session_id,
            num_questions=num_questions,
            strategy=strategy,
        )
        
        # Track steps executed
        steps_executed: List[str] = []
        deduplication_stats: Dict[str, Any] = {}
        
        # Execute each enabled step
        for step_config in config.steps:
            if not step_config.enabled:
                continue
            
            step_name = step_config.step
            if step_name not in self._step_handlers:
                logger.warning(f"Unknown pipeline step: {step_name}")
                continue
            
            handler = self._step_handlers[step_name]
            
            try:
                step_start = time.perf_counter()
                
                result = await handler(
                    context=context,
                    config=step_config.config,
                )
                
                step_time = (time.perf_counter() - step_start) * 1000
                context.step_times[step_name] = step_time
                
                # Update context with step results
                if result:
                    if step_name == "classify_query":
                        context.classification = result.get("classification")
                    elif step_name == "select_strategy":
                        context.strategy = result.get("strategy")
                    elif step_name == "generate_questions":
                        context.raw_questions = result.get("questions", [])
                        context.generation_result = result.get("generation_result")
                    elif step_name == "quality_assurance":
                        context.questions = result.get("questions", [])
                        context.quality_assessment = result.get("assessment")
                    elif step_name == "deduplicate":
                        context.questions = result.get("questions", [])
                        deduplication_stats = result.get("stats", {})
                    elif step_name == "rank_questions":
                        context.questions = result.get("questions", [])
                
                steps_executed.append(step_name)
                
                if self.debug_config.trace_execution:
                    logger.debug(
                        f"[PIPELINE] Step '{step_name}' completed in {step_time:.2f}ms"
                    )
                
            except Exception as e:
                logger.error(f"Pipeline step '{step_name}' failed: {e}")
                context.step_times[step_name] = (time.perf_counter() - step_start) * 1000
                
                if config.fail_fast:
                    raise
        
        # Build final result
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Ensure we have a classification
        if not context.classification:
            context.classification = QueryClassification(
                category="unknown",
                confidence=0.0,
                keywords_matched=[],
                preferred_strategy="decomposition",
                secondary_strategy=None,
                all_matches={},
            )
        
        return InvestigationResult(
            session_id=session_id,
            original_query=query,
            questions=context.questions,
            strategy_used=context.strategy or "adaptive",
            category_detected=context.classification.category,
            classification=context.classification,
            quality_assessment=context.quality_assessment,
            deduplication_stats=deduplication_stats,
            pipeline_stats={
                "total_time_ms": round(total_time, 2),
                "step_times": {k: round(v, 2) for k, v in context.step_times.items()},
                "steps_executed": steps_executed,
                "questions_generated": len(context.questions),
                "questions_requested": num_questions,
            },
            time_ms=total_time,
        )
    
    # ========================================================================
    # Step Handlers
    # ========================================================================
    
    async def _step_classify_query(
        self,
        context: PipelineContext,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Classify query to detect category and keywords."""
        classification = self.classifier.classify(context.original_query)
        
        if self.debug_config.log_strategy_selection:
            logger.info(
                f"[PIPELINE] Classification: category={classification.category}, "
                f"confidence={classification.confidence:.2f}, "
                f"keywords={classification.keywords_matched}"
            )
        
        return {"classification": classification}
    
    async def _step_select_strategy(
        self,
        context: PipelineContext,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Select optimal strategy based on classification."""
        # If strategy already specified, use it
        if context.strategy:
            return {"strategy": context.strategy}
        
        # Get from classification
        if context.classification:
            strategy = context.classification.preferred_strategy
        else:
            strategy = config.get("default_strategy", "adaptive")
        
        if self.debug_config.log_strategy_selection:
            logger.info(f"[PIPELINE] Selected strategy: {strategy}")
        
        return {"strategy": strategy}
    
    async def _step_generate_questions(
        self,
        context: PipelineContext,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate questions using delegator."""
        category = context.classification.category if context.classification else "technical"
        
        result = await self.delegator.generate_investigation(
            query=context.original_query,
            num_questions=context.num_questions,
            strategy=context.strategy or "adaptive",
            category=category,
        )
        
        return {
            "questions": result.questions,
            "generation_result": result,
        }
    
    async def _step_quality_assurance(
        self,
        context: PipelineContext,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess quality of generated questions."""
        if not context.raw_questions:
            return {"questions": [], "assessment": None}
        
        strategy = context.strategy or "unknown"
        
        questions, assessment = self.qa_provider.assess_questions(
            questions=context.raw_questions,
            original_query=context.original_query,
            strategy=strategy,
        )
        
        if self.debug_config.log_qa_scores:
            logger.info(
                f"[PIPELINE] QA Assessment: score={assessment.overall_score:.2f}, "
                f"level={assessment.quality_level.value}, "
                f"valid={len([q for q in questions if q.is_valid])}/{len(questions)}"
            )
        
        # Filter to valid questions if configured
        if config.get("filter_invalid", True):
            questions = [q for q in questions if q.is_valid]
        
        return {
            "questions": questions,
            "assessment": assessment,
        }
    
    async def _step_cross_reference(
        self,
        context: PipelineContext,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add cross-reference questions (optional step)."""
        if not context.questions:
            return {"questions": context.questions}
        
        # This step can generate additional related questions
        # For now, pass through - can be extended later
        return {"questions": context.questions}
    
    async def _step_deduplicate(
        self,
        context: PipelineContext,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Remove duplicate questions."""
        if not context.questions:
            return {"questions": [], "stats": {}}
        
        unique, stats = self.deduplicator.deduplicate(context.questions)
        
        if self.debug_config.trace_execution:
            logger.debug(
                f"[PIPELINE] Deduplication: {stats.get('original_count')} -> "
                f"{stats.get('unique_count')} ({stats.get('removed')} removed)"
            )
        
        return {
            "questions": unique,
            "stats": stats,
        }
    
    async def _step_rank_questions(
        self,
        context: PipelineContext,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Rank questions by quality score."""
        if not context.questions:
            return {"questions": []}
        
        # Sort by score descending
        ranked = sorted(
            context.questions,
            key=lambda q: q.score,
            reverse=True,
        )
        
        # Limit to requested number
        if len(ranked) > context.num_questions:
            ranked = ranked[:context.num_questions]
        
        return {"questions": ranked}
    
    async def _step_format_output(
        self,
        context: PipelineContext,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Final output formatting."""
        # Update category in questions
        category = context.classification.category if context.classification else "unknown"
        
        for q in context.questions:
            q.category = category
            q.metadata["session_id"] = context.session_id
            q.metadata["original_query"] = context.original_query[:100]
        
        return {"questions": context.questions}
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    def get_config(self) -> Dict[str, Any]:
        """Get current pipeline configuration."""
        return {
            "default_pipeline": [
                {"step": s.step, "enabled": s.enabled, "config": s.config}
                for s in self.config.steps
            ],
            "available_steps": self.AVAILABLE_STEPS,
            "timeout_seconds": self.config.timeout_seconds,
            "fail_fast": self.config.fail_fast,
        }
    
    def update_config(self, pipeline_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update pipeline configuration."""
        self.config = PipelineConfig.from_dict({"steps": pipeline_steps})
        return self.get_config()
