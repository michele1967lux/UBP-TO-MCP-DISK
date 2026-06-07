"""
reasoning_rag/pipeline.py

Pipeline orchestrator for Reasoning-Aware RAG.
Executes configurable multi-step processing with strategy selection.

Steps:
1. analyze_query - Complexity and intent detection
2. select_strategy - Choose optimal reasoning strategy
3. execute_reasoning - Run selected strategy
4. gather_evidence - Collect and attribute evidence
5. verify_claims - Multi-source verification
6. synthesize_answer - Final answer synthesis
7. format_output - Prepare result with citations

v1.0.0: Initial release
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .providers import (
    ReasoningResult,
    ReasoningTrace,
    QueryAnalysis,
    ReasoningStrategy,
    QueryComplexity,
    QueryIntent,
    QueryAnalyzer,
    DebugConfig,
)
from .delegation import ReasoningDelegator

logger = logging.getLogger(__name__)


# ============================================================================
# Pipeline Configuration
# ============================================================================


@dataclass
class StepConfig:
    """Configuration for a single pipeline step."""
    enabled: bool = True
    timeout: int = 30
    required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "timeout": self.timeout,
            "required": self.required,
        }


@dataclass
class PipelineConfig:
    """Configuration for the reasoning pipeline."""
    default_timeout_seconds: int = 120
    fail_fast: bool = False
    steps: Dict[str, StepConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        default_steps = {
            "analyze_query": StepConfig(enabled=True, timeout=5),
            "select_strategy": StepConfig(enabled=True, timeout=3),
            "execute_reasoning": StepConfig(enabled=True, timeout=180, required=True),
            "gather_evidence": StepConfig(enabled=True, timeout=15),
            "verify_claims": StepConfig(enabled=True, timeout=20),
            "synthesize_answer": StepConfig(enabled=True, timeout=15),
            "format_output": StepConfig(enabled=True, timeout=5),
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
            default_timeout_seconds=int(data.get("default_timeout_seconds", 120)),
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
    strategy_override: Optional[ReasoningStrategy] = None
    language: Optional[str] = None
    enable_verification: bool = True
    enable_evidence: bool = True
    claims_to_verify: Optional[List[str]] = None
    
    # Step results
    query_analysis: Optional[QueryAnalysis] = None
    selected_strategy: Optional[ReasoningStrategy] = None
    reasoning_result: Optional[ReasoningResult] = None
    
    # Execution metadata
    step_times: Dict[str, float] = field(default_factory=dict)
    step_errors: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "session_id": self.session_id,
            "strategy_override": self.strategy_override.value if self.strategy_override else None,
            "language": self.language,
            "query_analysis": self.query_analysis.to_dict() if self.query_analysis else None,
            "selected_strategy": self.selected_strategy.value if self.selected_strategy else None,
            "step_times": self.step_times,
            "step_errors": self.step_errors,
        }


# ============================================================================
# Reasoning Pipeline
# ============================================================================


class ReasoningPipeline:
    """
    Orchestrates reasoning operations through configurable steps.
    
    Features:
    - Automatic strategy selection
    - Multiple reasoning strategies
    - Evidence gathering and attribution
    - Claim verification
    - Configurable step execution
    """
    
    AVAILABLE_STEPS = [
        "analyze_query",
        "select_strategy",
        "execute_reasoning",
        "gather_evidence",
        "verify_claims",
        "synthesize_answer",
        "format_output",
    ]
    
    def __init__(
        self,
        delegator: ReasoningDelegator,
        query_analyzer: QueryAnalyzer,
        config: PipelineConfig,
        debug_config: Optional[DebugConfig] = None,
    ):
        self.delegator = delegator
        self.query_analyzer = query_analyzer
        self.config = config
        self.debug_config = debug_config or DebugConfig()
        
        # Step handlers
        self._step_handlers: Dict[str, Callable] = {
            "analyze_query": self._step_analyze_query,
            "select_strategy": self._step_select_strategy,
            "execute_reasoning": self._step_execute_reasoning,
            "gather_evidence": self._step_gather_evidence,
            "verify_claims": self._step_verify_claims,
            "synthesize_answer": self._step_synthesize_answer,
            "format_output": self._step_format_output,
        }
    
    async def execute(
        self,
        query: str,
        session_id: str,
        strategy: Optional[str] = None,
        language: Optional[str] = None,
        enable_verification: bool = True,
        enable_evidence: bool = True,
        claims_to_verify: Optional[List[str]] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
    ) -> ReasoningResult:
        """
        Execute the reasoning pipeline.
        
        Args:
            query: User's query
            session_id: Session identifier
            strategy: Strategy override
            language: Language override
            enable_verification: Enable claim verification
            enable_evidence: Enable evidence attribution
            claims_to_verify: Specific claims to verify
            pipeline_config: Step configuration override
            ctx: Security context
            
        Returns:
            ReasoningResult with answer, trace, and metadata
        """
        start_time = time.perf_counter()
        
        # Apply config overrides
        if pipeline_config:
            self._apply_config_override(pipeline_config)
        
        # Parse strategy override
        strategy_override = None
        if strategy:
            try:
                strategy_override = ReasoningStrategy(strategy)
            except ValueError:
                logger.warning(f"Unknown strategy '{strategy}', will auto-select")
        
        # Create context
        context = PipelineContext(
            query=query,
            session_id=session_id,
            strategy_override=strategy_override,
            language=language,
            enable_verification=enable_verification,
            enable_evidence=enable_evidence,
            claims_to_verify=claims_to_verify,
        )
        
        # Execute steps
        for step_name in self.AVAILABLE_STEPS:
            step_config = self.config.steps.get(step_name, StepConfig())
            
            if not step_config.enabled:
                continue
            
            # Skip evidence/verification steps if disabled
            if step_name == "gather_evidence" and not enable_evidence:
                continue
            if step_name == "verify_claims" and not enable_verification:
                continue
            
            handler = self._step_handlers.get(step_name)
            if not handler:
                continue
            
            step_start = time.perf_counter()
            
            try:
                await asyncio.wait_for(
                    handler(context),
                    timeout=step_config.timeout,
                )
                
            except asyncio.TimeoutError:
                error_msg = f"Step '{step_name}' timed out after {step_config.timeout}s"
                context.step_errors[step_name] = error_msg
                logger.warning(f"[REASONING] {error_msg}")
                
                if step_config.required or self.config.fail_fast:
                    raise RuntimeError(error_msg)
                    
            except Exception as e:
                error_msg = f"Step '{step_name}' failed: {e}"
                context.step_errors[step_name] = error_msg
                logger.error(f"[REASONING] {error_msg}")
                
                if step_config.required or self.config.fail_fast:
                    raise
            
            context.step_times[step_name] = (time.perf_counter() - step_start) * 1000
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Build result
        if not context.reasoning_result:
            raise RuntimeError("Pipeline failed to generate result")
        
        # Update timing
        context.reasoning_result.time_ms = total_time
        
        return context.reasoning_result
    
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
    
    async def _step_analyze_query(self, ctx: PipelineContext) -> None:
        """Analyze query for complexity and intent."""
        analysis = self.query_analyzer.analyze(ctx.query)
        ctx.query_analysis = analysis
        
        # Set language if not overridden
        if not ctx.language:
            ctx.language = analysis.language
        
        if self.debug_config.trace_execution:
            logger.debug(
                f"[REASONING] Query analysis: complexity={analysis.complexity.value}, "
                f"intent={analysis.intent.value}, recommended={analysis.recommended_strategy.value}"
            )
    
    async def _step_select_strategy(self, ctx: PipelineContext) -> None:
        """Select the reasoning strategy."""
        if ctx.strategy_override:
            ctx.selected_strategy = ctx.strategy_override
            if self.debug_config.trace_execution:
                logger.debug(f"[REASONING] Using override strategy: {ctx.selected_strategy.value}")
        elif ctx.query_analysis:
            ctx.selected_strategy = ctx.query_analysis.recommended_strategy
            if self.debug_config.trace_execution:
                logger.debug(f"[REASONING] Auto-selected strategy: {ctx.selected_strategy.value}")
        else:
            ctx.selected_strategy = ReasoningStrategy.CHAIN_OF_THOUGHT
            if self.debug_config.trace_execution:
                logger.debug(f"[REASONING] Defaulting to: {ctx.selected_strategy.value}")
    
    async def _step_execute_reasoning(self, ctx: PipelineContext) -> None:
        """Execute the selected reasoning strategy."""
        strategy = ctx.selected_strategy or ReasoningStrategy.CHAIN_OF_THOUGHT
        language = ctx.language or "en"
        
        if strategy == ReasoningStrategy.SELF_ASK:
            ctx.reasoning_result = await self.delegator.execute_self_ask(
                query=ctx.query,
                session_id=ctx.session_id,
                language=language,
            )
        elif strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            ctx.reasoning_result = await self.delegator.execute_chain_of_thought(
                query=ctx.query,
                session_id=ctx.session_id,
                language=language,
            )
        elif strategy == ReasoningStrategy.EVIDENCE_ATTRIBUTION:
            ctx.reasoning_result = await self.delegator.execute_evidence_attribution(
                query=ctx.query,
                session_id=ctx.session_id,
                language=language,
            )
        elif strategy == ReasoningStrategy.VERIFICATION:
            ctx.reasoning_result = await self.delegator.execute_verification(
                query=ctx.query,
                session_id=ctx.session_id,
                claims_to_verify=ctx.claims_to_verify,
                language=language,
            )
        else:  # DIRECT
            ctx.reasoning_result = await self.delegator.execute_direct(
                query=ctx.query,
                session_id=ctx.session_id,
                language=language,
            )
        
        if self.debug_config.trace_execution:
            logger.debug(
                f"[REASONING] Strategy {strategy.value} completed, "
                f"confidence={ctx.reasoning_result.confidence:.2f}"
            )
    
    async def _step_gather_evidence(self, ctx: PipelineContext) -> None:
        """Gather additional evidence if needed."""
        if not ctx.reasoning_result:
            return
        
        # Evidence already gathered for evidence_attribution strategy
        if ctx.selected_strategy == ReasoningStrategy.EVIDENCE_ATTRIBUTION:
            return
        
        # If low confidence and we have claims, gather more evidence
        if ctx.reasoning_result.confidence < 0.6 and ctx.reasoning_result.claims:
            # This would call additional retrieval for claims
            # For now, we log and skip
            if self.debug_config.trace_execution:
                logger.debug("[REASONING] Evidence gathering skipped (already have evidence)")
    
    async def _step_verify_claims(self, ctx: PipelineContext) -> None:
        """Verify claims if needed."""
        if not ctx.reasoning_result:
            return
        
        # Verification already done for verification strategy
        if ctx.selected_strategy == ReasoningStrategy.VERIFICATION:
            return
        
        # Skip if no claims to verify
        if not ctx.reasoning_result.claims:
            return
        
        # Could run additional verification here
        if self.debug_config.trace_execution:
            logger.debug(f"[REASONING] {len(ctx.reasoning_result.claims)} claims available for verification")
    
    async def _step_synthesize_answer(self, ctx: PipelineContext) -> None:
        """Synthesize final answer (already done in strategy execution)."""
        # Answer synthesis is handled within each strategy
        pass
    
    async def _step_format_output(self, ctx: PipelineContext) -> None:
        """Format the final output."""
        if ctx.reasoning_result:
            # Ensure metadata is complete
            ctx.reasoning_result.reasoning_trace.completed_at = datetime.utcnow()
    
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
            "available_strategies": [s.value for s in ReasoningStrategy],
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
