"""
reasoning_rag/adapter.py

Bridge layer that exposes all Reasoning-Aware RAG operations to the UBP system.
Handles initialization, configuration, DI resolution, and operation routing.

Operations:
- initialize: Start components and worker pool
- reason: Full reasoning pipeline with strategy selection
- self_ask: Iterative sub-question decomposition
- chain_of_thought: Interleaved reasoning and retrieval
- evidence_attribution: Answer with citations
- verify_claims: Multi-source fact checking
- analyze_query: Query complexity and intent analysis
- extract_claims: Extract claims from text
- get_session / delete_session: Session management
- get_stats: Metrics and statistics (admin)
- get_pipeline_config / set_pipeline_config: Configuration (admin)
- reload_config: Hot-reload configuration (admin)
- shutdown: Graceful shutdown
- health_check: Component health status

v1.0.0: Initial release with full enterprise features
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from .providers import (
    # Data classes
    ReasoningResult,
    ReasoningTrace,
    SubQuestion,
    ReasoningStep,
    Claim,
    Evidence,
    Citation,
    Verification,
    Contradiction,
    RetrievedDocument,
    QueryAnalysis,
    ReasoningSession,
    WorkerTask,
    WorkerStats,
    # Enums
    ReasoningStrategy,
    QueryComplexity,
    QueryIntent,
    VerificationStatus,
    AttributionType,
    TaskStatus,
    TaskPriority,
    # Configs
    ReasoningConfig,
    SelfAskConfig,
    ChainOfThoughtConfig,
    EvidenceConfig,
    VerificationConfig,
    RetrievalConfig,
    CacheConfig,
    SessionConfig,
    WorkerPoolConfig,
    MetricsConfig,
    DebugConfig,
    # Providers
    QueryAnalyzer,
    ReasoningCacheProvider,
    ReasoningSessionManager,
    ReasoningWorkerPool,
    ReasoningMetricsCollector,
)
from .delegation import (
    ReasoningDelegator,
    LLMDelegationConfig,
)
from .pipeline import (
    ReasoningPipeline,
    PipelineConfig,
    StepConfig,
    PipelineContext,
)
from .prompts import detect_language

import sys
from pathlib import Path as _Path
_portable_path = str(_Path(__file__).resolve().parent.parent.parent)
if _portable_path not in sys.path:
    sys.path.insert(0, _portable_path)

from _portable.context import PortableContext

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
# DI Container Module Registry Wrapper
# ============================================================================


class DIContainerModuleRegistry:
    """Wraps DI container to provide module registry interface."""

    def __init__(self, di_container: Optional[Any] = None):
        self._container = di_container
        self._cached_modules: Dict[str, Any] = {}

    def get_module(self, module_name: str) -> Optional[Any]:
        """Get a module by name (sync - cache only)."""
        if module_name in self._cached_modules:
            return self._cached_modules[module_name]
        return None

    async def resolve_module(self, module_name: str) -> Optional[Any]:
        """Async module resolution via DI container."""
        if module_name in self._cached_modules:
            return self._cached_modules[module_name]

        if not self._container:
            return None

        # DI container.resolve() is async - must be awaited
        if hasattr(self._container, "resolve"):
            try:
                module = await self._container.resolve(module_name)
                if module:
                    self._cached_modules[module_name] = module
                    return module
            except Exception as e:
                logger.warning(f"Failed to resolve module '{module_name}': {e}")

        return None

    def is_module_loaded(self, module_name: str) -> bool:
        """Check if module is loaded."""
        return module_name in self._cached_modules


# ============================================================================
# Configuration Utilities
# ============================================================================


def resolve_env_value(value: Any) -> Any:
    """Resolve environment variable placeholders."""
    if not isinstance(value, str):
        return value
    
    pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
    
    def replace(match):
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)
    
    return re.sub(pattern, replace, value)


def coerce_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively coerce configuration values to appropriate types."""
    result = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = coerce_config_types(value)
        elif isinstance(value, list):
            result[key] = [
                coerce_config_types(v) if isinstance(v, dict) else _coerce_value(v)
                for v in value
            ]
        else:
            result[key] = _coerce_value(value)
    
    return result


def _coerce_value(value: Any) -> Any:
    """Coerce a single value to appropriate type."""
    if not isinstance(value, str):
        return value
    
    value = resolve_env_value(value)
    
    if not isinstance(value, str):
        return value
    
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False
    
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    return value


# ============================================================================
# Reasoning RAG Adapter
# ============================================================================


class ReasoningRAGAdapter:
    """
    Adapter that exposes Reasoning-Aware RAG operations to the UBP system.
    
    Strategies:
    - Self-Ask: Iterative sub-question decomposition
    - Chain-of-Thought: Interleaved reasoning and retrieval
    - Evidence Attribution: Answer with citations
    - Verification: Multi-source fact checking
    """
    
    def __init__(
        self,
        module_path: Path,
        di_container: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        self.module_path = Path(module_path)
        self._di_container = di_container
        self._event_bus = event_bus
        
        # Module registry wrapper
        self._module_registry = DIContainerModuleRegistry(di_container)
        
        # Configuration
        self._config: Dict[str, Any] = {}
        self._reasoning_config: Optional[ReasoningConfig] = None
        self._self_ask_config: Optional[SelfAskConfig] = None
        self._cot_config: Optional[ChainOfThoughtConfig] = None
        self._evidence_config: Optional[EvidenceConfig] = None
        self._verification_config: Optional[VerificationConfig] = None
        self._retrieval_config: Optional[RetrievalConfig] = None
        self._cache_config: Optional[CacheConfig] = None
        self._session_config: Optional[SessionConfig] = None
        self._worker_config: Optional[WorkerPoolConfig] = None
        self._metrics_config: Optional[MetricsConfig] = None
        self._debug_config: Optional[DebugConfig] = None
        self._llm_config: Optional[LLMDelegationConfig] = None
        self._pipeline_config: Optional[PipelineConfig] = None
        
        # Components
        self._query_analyzer: Optional[QueryAnalyzer] = None
        self._cache: Optional[ReasoningCacheProvider] = None
        self._session_manager: Optional[ReasoningSessionManager] = None
        self._worker_pool: Optional[ReasoningWorkerPool] = None
        self._metrics: Optional[ReasoningMetricsCollector] = None
        self._delegator: Optional[ReasoningDelegator] = None
        self._pipeline: Optional[ReasoningPipeline] = None
        
        # State
        self._initialized = False
        self._redis_client: Optional[Any] = None
    
    def _build_context_from_di(self) -> PortableContext:
        return PortableContext(
            client_id="default",
            user_id=None,
            session_id=None,
            source="portable",
        )

    def _normalize_ctx(self, ctx: Any) -> PortableContext:
        return PortableContext.normalize(ctx)
    
    # ========================================================================
    # Configuration Loading
    # ========================================================================
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json."""
        config_path = self.module_path / "config.json"
        
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            return {}
        
        with open(config_path, "r") as f:
            raw_config = json.load(f)
        
        return coerce_config_types(raw_config)
    
    def _build_reasoning_config(self) -> ReasoningConfig:
        """Build core reasoning config."""
        cfg = self._config.get("reasoning_rag", {})
        return ReasoningConfig(
            enabled=cfg.get("enabled", True),
            default_strategy=cfg.get("default_strategy", "auto"),
            max_reasoning_depth=cfg.get("max_reasoning_depth", 5),
            temperature=cfg.get("temperature", 0.3),
            max_tokens=cfg.get("max_tokens", 800),
            timeout_seconds=cfg.get("timeout_seconds", 60),
            retry_enabled=cfg.get("retry_enabled", True),
            max_retries=cfg.get("max_retries", 2),
        )
    
    def _build_self_ask_config(self) -> SelfAskConfig:
        """Build Self-Ask config."""
        cfg = self._config.get("self_ask", {})
        return SelfAskConfig(
            enabled=cfg.get("enabled", True),
            max_iterations=cfg.get("max_iterations", 5),
            min_iterations=cfg.get("min_iterations", 1),
            convergence_threshold=cfg.get("convergence_threshold", 0.85),
            sub_question_temperature=cfg.get("sub_question_temperature", 0.4),
            integration_temperature=cfg.get("integration_temperature", 0.2),
            max_sub_questions_per_iteration=cfg.get("max_sub_questions_per_iteration", 3),
            retrieval_top_k=cfg.get("retrieval_top_k", 5),
            early_stop_on_confidence=cfg.get("early_stop_on_confidence", True),
            confidence_threshold=cfg.get("confidence_threshold", 0.8),
        )
    
    def _build_cot_config(self) -> ChainOfThoughtConfig:
        """Build Chain-of-Thought config."""
        cfg = self._config.get("chain_of_thought", {})
        return ChainOfThoughtConfig(
            enabled=cfg.get("enabled", True),
            max_reasoning_steps=cfg.get("max_reasoning_steps", 8),
            min_reasoning_steps=cfg.get("min_reasoning_steps", 2),
            auto_retrieval_threshold=cfg.get("auto_retrieval_threshold", 0.6),
            reasoning_temperature=cfg.get("reasoning_temperature", 0.3),
            synthesis_temperature=cfg.get("synthesis_temperature", 0.2),
            retrieval_top_k=cfg.get("retrieval_top_k", 3),
            interleave_mode=cfg.get("interleave_mode", "adaptive"),
            thought_verification=cfg.get("thought_verification", True),
            step_by_step_logging=cfg.get("step_by_step_logging", True),
        )
    
    def _build_evidence_config(self) -> EvidenceConfig:
        """Build Evidence Attribution config."""
        cfg = self._config.get("evidence_attribution", {})
        return EvidenceConfig(
            enabled=cfg.get("enabled", True),
            min_confidence=cfg.get("min_confidence", 0.5),
            citation_format=cfg.get("citation_format", "inline"),
            track_spans=cfg.get("track_spans", True),
            source_deduplication=cfg.get("source_deduplication", True),
            claim_extraction_enabled=cfg.get("claim_extraction_enabled", True),
            verification_level=cfg.get("verification_level", "standard"),
        )
    
    def _build_verification_config(self) -> VerificationConfig:
        """Build Verification config."""
        cfg = self._config.get("verification", {})
        return VerificationConfig(
            enabled=cfg.get("enabled", True),
            multi_source_check=cfg.get("multi_source_check", True),
            min_sources_for_verification=cfg.get("min_sources_for_verification", 2),
            contradiction_detection=cfg.get("contradiction_detection", True),
            contradiction_threshold=cfg.get("contradiction_threshold", 0.7),
            confidence_aggregation=cfg.get("confidence_aggregation", "weighted_average"),
            grounding_validation=cfg.get("grounding_validation", True),
            hallucination_check=cfg.get("hallucination_check", True),
            fact_check_temperature=cfg.get("fact_check_temperature", 0.1),
        )
    
    def _build_retrieval_config(self) -> RetrievalConfig:
        """Build Retrieval config."""
        cfg = self._config.get("retrieval", {})
        return RetrievalConfig(
            module=cfg.get("module", "retrieval_strategy"),
            operation=cfg.get("operation", "retrieve"),
            default_top_k=cfg.get("default_top_k", 5),
            rerank_enabled=cfg.get("rerank_enabled", True),
            rerank_top_k=cfg.get("rerank_top_k", 3),
            min_relevance_score=cfg.get("min_relevance_score", 0.5),
            hybrid_search=cfg.get("hybrid_search", True),
            timeout_seconds=cfg.get("timeout_seconds", 10),
            fallback_to_llm=cfg.get("fallback_to_llm", True),
        )
    
    def _build_cache_config(self) -> CacheConfig:
        """Build Cache config."""
        cfg = self._config.get("cache", {})
        env = os.environ.get("UBP_ENV", "dev")
        return CacheConfig(
            enabled=cfg.get("enabled", True),
            ttl_seconds=cfg.get("ttl_seconds", 3600),
            base_prefix="ubp",
            env=env,
            cache_sub_questions=cfg.get("cache_sub_questions", True),
            cache_retrievals=cfg.get("cache_retrievals", True),
            cache_reasoning_steps=cfg.get("cache_reasoning_steps", False),
            cache_final_answers=cfg.get("cache_final_answers", True),
            semantic_matching=cfg.get("semantic_matching", True),
            semantic_threshold=cfg.get("semantic_threshold", 0.92),
        )
    
    def _build_session_config(self) -> SessionConfig:
        """Build Session config."""
        cfg = self._config.get("session_management", {})
        return SessionConfig(
            enabled=cfg.get("enabled", True),
            ttl_seconds=cfg.get("ttl_seconds", 3600),
            max_history_size=cfg.get("max_history_size", 50),
            persist_reasoning_trace=cfg.get("persist_reasoning_trace", True),
            persist_retrievals=cfg.get("persist_retrievals", True),
            auto_cleanup=cfg.get("auto_cleanup", True),
        )
    
    def _build_worker_config(self) -> WorkerPoolConfig:
        """Build Worker Pool config."""
        cfg = self._config.get("worker_pool", {})
        return WorkerPoolConfig(
            enabled=cfg.get("enabled", True),
            pool_size=cfg.get("pool_size", 4),
            max_pool_size=cfg.get("max_pool_size", 8),
            task_timeout_seconds=cfg.get("task_timeout_seconds", 30),
            queue_max_size=cfg.get("queue_max_size", 100),
            retry_on_failure=cfg.get("retry_on_failure", True),
            max_task_retries=cfg.get("max_task_retries", 2),
            backoff_multiplier=cfg.get("backoff_multiplier", 1.5),
        )
    
    def _build_metrics_config(self) -> MetricsConfig:
        """Build Metrics config."""
        cfg = self._config.get("metrics", {})
        return MetricsConfig(
            enabled=cfg.get("enabled", True),
            collect_timings=cfg.get("collect_timings", True),
            collect_strategy_distribution=cfg.get("collect_strategy_distribution", True),
            collect_iteration_counts=cfg.get("collect_iteration_counts", True),
            collect_retrieval_stats=cfg.get("collect_retrieval_stats", True),
            collect_confidence_scores=cfg.get("collect_confidence_scores", True),
            collect_verification_stats=cfg.get("collect_verification_stats", True),
            retention_hours=cfg.get("retention_hours", 24),
        )
    
    def _build_debug_config(self) -> DebugConfig:
        """Build Debug config."""
        cfg = self._config.get("debug", {})
        return DebugConfig(
            enabled=cfg.get("enabled", False),
            log_prompts=cfg.get("log_prompts", False),
            log_responses=cfg.get("log_responses", False),
            log_sub_questions=cfg.get("log_sub_questions", True),
            log_reasoning_steps=cfg.get("log_reasoning_steps", True),
            log_retrievals=cfg.get("log_retrievals", True),
            log_evidence=cfg.get("log_evidence", True),
            log_verification=cfg.get("log_verification", True),
            trace_execution=cfg.get("trace_execution", False),
        )
    
    def _resolve_llm_module_name(self) -> str:
        """Resolve LLM module name from ProviderMapper chain (role=rag)."""
        try:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper
            chain = ProviderMapper.resolve_chain("rag")
            if chain:
                module_name = chain[0][0]  # First in chain = primary
                logger.info(f"[REASONING] LLM resolved via ProviderMapper: {module_name}")
                return module_name
        except Exception as e:
            logger.warning(
                f"[REASONING] ProviderMapper NOT AVAILABLE - using hardcoded fallback "
                f"'inference_ollama_grok'. Centralized provider config (UBP_ROLES__RAG_PROVIDER) "
                f"is IGNORED for this module. Cause: {e}"
            )
        return "inference_ollama_grok"

    def _build_llm_config(self) -> LLMDelegationConfig:
        """Build LLM Delegation config."""
        cfg = self._config.get("delegation", {})
        # v3.7: Use ProviderMapper for centralized resolution, cfg override still honored
        default_module = self._resolve_llm_module_name()
        return LLMDelegationConfig(
            llm_module=cfg.get("llm_module", default_module),
            llm_operation=cfg.get("llm_operation", "generate"),
            timeout_seconds=cfg.get("timeout_seconds", 30),
            max_retries=cfg.get("max_retries", 2),
            fallback_enabled=cfg.get("fallback_enabled", True),
            fallback_chain=cfg.get("fallback_chain", ["chain_of_thought", "self_ask", "direct"]),
        )
    
    def _build_pipeline_config(self) -> PipelineConfig:
        """Build Pipeline config."""
        cfg = self._config.get("pipeline", {})
        steps_cfg = cfg.get("steps", {})
        
        steps = {}
        for step_name, step_data in steps_cfg.items():
            if isinstance(step_data, dict):
                steps[step_name] = StepConfig(
                    enabled=step_data.get("enabled", True),
                    timeout=step_data.get("timeout", 30),
                    required=step_data.get("required", False),
                )
        
        return PipelineConfig(
            default_timeout_seconds=cfg.get("default_timeout_seconds", 120),
            fail_fast=cfg.get("fail_fast", False),
            steps=steps,
        )
    
    # ========================================================================
    # Operations
    # ========================================================================
    
    async def initialize(self, ctx: Any = None) -> Dict[str, Any]:
        """Initialize all Reasoning RAG components."""
        if self._initialized:
            return {"status": "already_initialized"}
        
        try:
            # Load configuration
            self._config = self._load_config()
            
            # Build configs
            self._reasoning_config = self._build_reasoning_config()
            self._self_ask_config = self._build_self_ask_config()
            self._cot_config = self._build_cot_config()
            self._evidence_config = self._build_evidence_config()
            self._verification_config = self._build_verification_config()
            self._retrieval_config = self._build_retrieval_config()
            self._cache_config = self._build_cache_config()
            self._session_config = self._build_session_config()
            self._worker_config = self._build_worker_config()
            self._metrics_config = self._build_metrics_config()
            self._debug_config = self._build_debug_config()
            self._llm_config = self._build_llm_config()
            self._pipeline_config = self._build_pipeline_config()
            
            # Initialize components
            complexity_thresholds = self._config.get("query_analysis", {}).get("complexity_thresholds", {})
            self._query_analyzer = QueryAnalyzer(complexity_thresholds=complexity_thresholds)
            
            # Try to get Redis client
            if self._di_container:
                self._redis_client = getattr(self._di_container, "redis", None)
            
            self._cache = ReasoningCacheProvider(
                config=self._cache_config,
                redis_client=self._redis_client,
            )
            
            self._session_manager = ReasoningSessionManager(self._session_config)
            
            self._worker_pool = ReasoningWorkerPool(self._worker_config)
            if self._worker_config.enabled:
                await self._worker_pool.start()
            
            self._metrics = ReasoningMetricsCollector(self._metrics_config)
            
            # Initialize delegator
            self._delegator = ReasoningDelegator(
                llm_config=self._llm_config,
                retrieval_config=self._retrieval_config,
                self_ask_config=self._self_ask_config,
                cot_config=self._cot_config,
                evidence_config=self._evidence_config,
                verification_config=self._verification_config,
                module_registry=self._module_registry,
                event_publisher=self._event_bus,
                debug_config=self._debug_config,
            )
            
            # Initialize pipeline
            self._pipeline = ReasoningPipeline(
                delegator=self._delegator,
                query_analyzer=self._query_analyzer,
                config=self._pipeline_config,
                debug_config=self._debug_config,
            )
            
            self._initialized = True
            
            logger.info("Reasoning RAG pipeline initialized successfully")
            
            # Publish event
            if self._event_bus:
                await self._event_bus.publish(
                    "reasoning.initialized",
                    {"module": "reasoning_rag", "status": "success"},
                )
            
            return {
                "status": "initialized",
                "components": {
                    "query_analyzer": True,
                    "cache": self._cache_config.enabled,
                    "session_manager": self._session_config.enabled,
                    "worker_pool": self._worker_config.enabled,
                    "metrics": self._metrics_config.enabled,
                    "delegator": True,
                    "pipeline": True,
                },
                "strategies": ["self_ask", "chain_of_thought", "evidence_attribution", "verification", "direct"],
            }
            
        except Exception as e:
            logger.error(f"Reasoning RAG initialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def reason(
        self,
        query: str,
        strategy: Optional[str] = None,
        language: Optional[str] = None,
        enable_verification: bool = True,
        enable_evidence: bool = True,
        claims_to_verify: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Full reasoning pipeline with automatic strategy selection.
        
        Args:
            query: User's query
            strategy: Strategy override (self_ask, chain_of_thought, evidence_attribution, verification, direct)
            language: Language override (en, it)
            enable_verification: Enable claim verification step
            enable_evidence: Enable evidence attribution step
            claims_to_verify: Specific claims to verify
            session_id: Existing session ID
            pipeline_config: Step configuration override
            ctx: Security context
            
        Returns:
            Reasoning result with answer, trace, and metadata
        """
        if not self._initialized:
            await self.initialize(ctx)
        
        # Get or create session
        if session_id:
            session = await self._session_manager.get_session(session_id)
        else:
            session = await self._session_manager.create_session()
        
        if not session:
            session = await self._session_manager.create_session()
        
        # Check cache
        cache_key = f"{query}:{strategy}:{language}"
        if self._cache_config.enabled and self._cache_config.cache_final_answers:
            cached = await self._cache.get("reasoning", cache_key)
            if cached:
                return {**cached, "cached": True}
        
        # Execute pipeline
        result = await self._pipeline.execute(
            query=query,
            session_id=session.session_id,
            strategy=strategy,
            language=language,
            enable_verification=enable_verification,
            enable_evidence=enable_evidence,
            claims_to_verify=claims_to_verify,
            pipeline_config=pipeline_config,
            ctx=ctx,
        )
        
        # Update session
        await self._session_manager.update_session(
            session_id=session.session_id,
            query=query,
            result=result,
        )
        
        # Record metrics
        if self._metrics_config.enabled:
            verification_counts = {
                "verified": sum(1 for v in result.verifications if v.status.value == "verified"),
                "partially": sum(1 for v in result.verifications if v.status.value == "partially_verified"),
                "unverified": sum(1 for v in result.verifications if v.status.value == "unverified"),
                "contradicted": sum(1 for v in result.verifications if v.status.value == "contradicted"),
            }
            
            await self._metrics.record_reasoning(
                strategy=result.strategy_used,
                iterations=result.iteration_count,
                steps=result.step_count,
                retrievals=len(result.retrieved_docs),
                confidence=result.confidence,
                verifications=verification_counts,
                execution_time_ms=result.time_ms,
            )
        
        response = result.to_dict()
        
        # Cache result
        if self._cache_config.enabled and self._cache_config.cache_final_answers:
            await self._cache.set("reasoning", cache_key, response)
        
        # Publish event
        if self._event_bus:
            await self._event_bus.publish(
                "reasoning.completed",
                {
                    "session_id": session.session_id,
                    "strategy": result.strategy_used.value,
                    "confidence": result.confidence,
                    "time_ms": result.time_ms,
                },
            )
        
        return response
    
    async def self_ask(
        self,
        query: str,
        max_iterations: Optional[int] = None,
        language: str = "auto",
        session_id: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Execute Self-Ask reasoning strategy.
        
        Iteratively decomposes query into sub-questions.
        """
        if not self._initialized:
            await self.initialize(ctx)
        
        # Detect language
        if language == "auto":
            language = detect_language(query)
        
        # Get or create session
        session = await self._session_manager.create_session() if not session_id else \
                  await self._session_manager.get_session(session_id) or await self._session_manager.create_session()
        
        # Apply iteration override
        if max_iterations:
            original_max = self._self_ask_config.max_iterations
            self._self_ask_config.max_iterations = max_iterations
        
        result = await self._delegator.execute_self_ask(
            query=query,
            session_id=session.session_id,
            language=language,
        )
        
        # Restore config
        if max_iterations:
            self._self_ask_config.max_iterations = original_max
        
        return result.to_dict()
    
    async def chain_of_thought(
        self,
        query: str,
        max_steps: Optional[int] = None,
        language: str = "auto",
        session_id: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Execute Chain-of-Thought reasoning with interleaved retrieval.
        """
        if not self._initialized:
            await self.initialize(ctx)
        
        if language == "auto":
            language = detect_language(query)
        
        session = await self._session_manager.create_session() if not session_id else \
                  await self._session_manager.get_session(session_id) or await self._session_manager.create_session()
        
        if max_steps:
            original_max = self._cot_config.max_reasoning_steps
            self._cot_config.max_reasoning_steps = max_steps
        
        result = await self._delegator.execute_chain_of_thought(
            query=query,
            session_id=session.session_id,
            language=language,
        )
        
        if max_steps:
            self._cot_config.max_reasoning_steps = original_max
        
        return result.to_dict()
    
    async def evidence_attribution(
        self,
        query: str,
        language: str = "auto",
        session_id: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Generate answer with inline citations and evidence attribution.
        """
        if not self._initialized:
            await self.initialize(ctx)
        
        if language == "auto":
            language = detect_language(query)
        
        session = await self._session_manager.create_session() if not session_id else \
                  await self._session_manager.get_session(session_id) or await self._session_manager.create_session()
        
        result = await self._delegator.execute_evidence_attribution(
            query=query,
            session_id=session.session_id,
            language=language,
        )
        
        return result.to_dict()
    
    async def verify_claims(
        self,
        query: str,
        claims: Optional[List[str]] = None,
        language: str = "auto",
        session_id: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Multi-source fact checking and verification.
        """
        if not self._initialized:
            await self.initialize(ctx)
        
        if language == "auto":
            language = detect_language(query)
        
        session = await self._session_manager.create_session() if not session_id else \
                  await self._session_manager.get_session(session_id) or await self._session_manager.create_session()
        
        result = await self._delegator.execute_verification(
            query=query,
            session_id=session.session_id,
            claims_to_verify=claims,
            language=language,
        )
        
        return result.to_dict()
    
    async def analyze_query(
        self,
        query: str,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Analyze query for complexity, intent, and recommended strategy.
        """
        if not self._initialized:
            await self.initialize(ctx)
        
        analysis = self._query_analyzer.analyze(query)
        return analysis.to_dict()
    
    async def extract_claims(
        self,
        text: str,
        language: str = "auto",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Extract factual claims from text.
        """
        if not self._initialized:
            await self.initialize(ctx)
        
        if language == "auto":
            language = detect_language(text)
        
        from .prompts import get_template
        
        prompt = get_template("evidence_claim_extraction", language).format(text=text)
        
        # Call LLM directly
        response = await self._delegator._call_llm(prompt, temperature=0.1)
        data = self._delegator._parse_json_response(response)
        
        return {
            "claims": data.get("claims", []),
            "total_claims": data.get("total_claims", len(data.get("claims", []))),
        }
    
    async def get_session(self, session_id: str, ctx: Any = None) -> Dict[str, Any]:
        """Get session state."""
        if not self._initialized:
            await self.initialize(ctx)
        
        session = await self._session_manager.get_session(session_id)
        if session:
            return session.to_dict()
        return {"error": "Session not found", "session_id": session_id}
    
    async def delete_session(self, session_id: str, ctx: Any = None) -> Dict[str, Any]:
        """Delete a session."""
        if not self._initialized:
            await self.initialize(ctx)
        
        deleted = await self._session_manager.delete_session(session_id)
        return {"deleted": deleted, "session_id": session_id}
    
    async def get_stats(self, period: str = "24h", ctx: Any = None) -> Dict[str, Any]:
        """Get metrics and statistics (admin only)."""
        if ctx and hasattr(ctx, "user") and hasattr(ctx.user, "is_admin"):
            if not ctx.user.is_admin:
                return {"error": "Admin access required"}
        
        if not self._initialized:
            await self.initialize(ctx)
        
        metrics = self._metrics.get_metrics() if self._metrics else {}
        cache_stats = self._cache.get_stats() if self._cache else {}
        worker_stats = self._worker_pool.get_stats().to_dict() if self._worker_pool else {}
        
        return {
            "period": period,
            "metrics": metrics,
            "cache": cache_stats,
            "worker_pool": worker_stats,
        }
    
    async def get_pipeline_config(self, ctx: Any = None) -> Dict[str, Any]:
        """Get pipeline configuration."""
        if not self._initialized:
            await self.initialize(ctx)
        
        return self._pipeline.get_config() if self._pipeline else {}
    
    async def set_pipeline_config(
        self,
        steps: List[Dict[str, Any]],
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Update pipeline configuration (admin only)."""
        if ctx and hasattr(ctx, "user") and hasattr(ctx.user, "is_admin"):
            if not ctx.user.is_admin:
                return {"error": "Admin access required"}
        
        if not self._initialized:
            await self.initialize(ctx)
        
        return self._pipeline.update_config(steps) if self._pipeline else {}
    
    async def reload_config(self, ctx: Any = None) -> Dict[str, Any]:
        """Hot-reload configuration (admin only)."""
        if ctx and hasattr(ctx, "user") and hasattr(ctx.user, "is_admin"):
            if not ctx.user.is_admin:
                return {"error": "Admin access required"}
        
        try:
            self._config = self._load_config()
            
            self._reasoning_config = self._build_reasoning_config()
            self._self_ask_config = self._build_self_ask_config()
            self._cot_config = self._build_cot_config()
            self._debug_config = self._build_debug_config()
            self._pipeline_config = self._build_pipeline_config()
            
            if self._pipeline:
                self._pipeline.config = self._pipeline_config
            
            return {"status": "reloaded", "timestamp": __import__("datetime").datetime.utcnow().isoformat()}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def shutdown(self, ctx: Any = None) -> Dict[str, Any]:
        """Graceful shutdown."""
        try:
            if self._worker_pool:
                await self._worker_pool.stop()
            
            if self._cache:
                await self._cache.clear()
            
            self._initialized = False
            
            if self._event_bus:
                await self._event_bus.publish(
                    "reasoning.shutdown",
                    {"module": "reasoning_rag"},
                )
            
            logger.info("Reasoning RAG pipeline shut down")
            return {"status": "shutdown"}
            
        except Exception as e:
            logger.error(f"Reasoning RAG shutdown error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def health_check(self, ctx: Any = None) -> Dict[str, Any]:
        """Check component health."""
        if not self._initialized:
            return {
                "module": "reasoning_rag",
                "status": "not_initialized",
            }
        
        llm_health = await self._delegator.health_check()
        worker_stats = self._worker_pool.get_stats() if self._worker_pool else None
        cache_stats = self._cache.get_stats() if self._cache else None
        
        status = "healthy"
        if llm_health.get("status") != "available":
            status = "degraded"
        
        return {
            "module": "reasoning_rag",
            "status": status,
            "initialized": self._initialized,
            "llm_delegation": llm_health,
            "worker_pool": worker_stats.to_dict() if worker_stats else None,
            "cache": cache_stats,
            "strategies": {
                "self_ask": self._self_ask_config.enabled if self._self_ask_config else False,
                "chain_of_thought": self._cot_config.enabled if self._cot_config else False,
                "evidence_attribution": self._evidence_config.enabled if self._evidence_config else False,
                "verification": self._verification_config.enabled if self._verification_config else False,
            },
        }
    
    async def get_available_strategies(self, ctx: Any = None) -> Dict[str, Any]:
        """Get list of available reasoning strategies."""
        return {
            "strategies": [
                {
                    "name": "self_ask",
                    "description": "Iterative sub-question decomposition",
                    "best_for": "Multi-hop queries, complex questions",
                },
                {
                    "name": "chain_of_thought",
                    "description": "Step-by-step reasoning with interleaved retrieval",
                    "best_for": "Explanatory, causal, procedural queries",
                },
                {
                    "name": "evidence_attribution",
                    "description": "Answer with inline citations and source tracking",
                    "best_for": "Factual queries requiring verification",
                },
                {
                    "name": "verification",
                    "description": "Multi-source fact checking and contradiction detection",
                    "best_for": "Verifying claims, fact-checking",
                },
                {
                    "name": "direct",
                    "description": "Simple retrieval and answer",
                    "best_for": "Simple factual queries",
                },
            ],
            "auto_selection": True,
        }
