"""
investigation_pipeline/adapter.py

Bridge Layer - Exposes all module operations.
Orchestrates providers, delegation, and pipeline execution.

This is the main entry point for the module.

Operations exposed:
- investigate: Full pipeline execution
- generate_questions: Direct generation
- generate_multi_strategy: Parallel multi-strategy
- classify_query: Query classification
- assess_quality: QA validation
- deduplicate_questions: Remove duplicates
- get_session / delete_session: Session management
- get_stats: Metrics and statistics
- get_pipeline_config / set_pipeline_config: Configuration
- reload_config: Hot-reload configuration
- initialize / shutdown: Lifecycle
- health_check: Health status
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

from .providers import (
    InvestigationConfig,
    QualityAssuranceConfig,
    WorkerPoolConfig,
    SessionConfig,
    CacheConfig,
    DeduplicationConfig,
    MetricsConfig,
    DebugConfig,
    InvestigationQuestion,
    InvestigationResult,
    QueryClassification,
    QualityAssessment,
    QualityAssuranceProvider,
    InvestigationDeduplicator,
    InvestigationWorkerPool,
    InvestigationSessionManager,
    RedisCacheProvider,
    MetricsCollector,
    QueryClassifier,
    InvestigationStrategy,
)
from .delegation import (
    InvestigationDelegator,
    LLMDelegationConfig,
    GenerationResult,
)
from .pipeline import (
    InvestigationPipeline,
    PipelineConfig,
)
from .prompts import QUERY_CATEGORIES

# v1.10.0: Role-Based Configuration Engine
try:
    from ubp_enterprise_hybrid.modules.cores._shared import ProviderMapper, ProviderConfigurationError
    PROVIDER_MAPPER_AVAILABLE = True
except ImportError:
    PROVIDER_MAPPER_AVAILABLE = False
    ProviderMapper = None
    ProviderConfigurationError = Exception

# FIX-TYPE-001 v2.2.4: Import type coercion for config safety
try:
    from ubp_enterprise_hybrid.modules.cores._shared.manifest_loader import coerce_config_types
except ImportError:
    def coerce_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback coercion function."""
        def _coerce(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_coerce(v) for v in obj]
            elif isinstance(obj, str):
                if obj.lower() == "true":
                    return True
                elif obj.lower() == "false":
                    return False
                try:
                    return int(obj)
                except ValueError:
                    try:
                        return float(obj)
                    except ValueError:
                        return obj
            return obj
        return _coerce(config)

logger = logging.getLogger(__name__)


# ============================================================================
# DI Container Adapter for Module Registry
# ============================================================================


class DIContainerModuleRegistry:
    """
    Adapter that wraps the DI container to provide IModuleRegistry interface.
    
    This allows investigation_pipeline to resolve LLM modules via the standard
    DI container without requiring backend.app.infra.interfaces.
    """

    def __init__(self, di_container: Any):
        self._di_container = di_container
        self._module_cache: Dict[str, Any] = {}

    def is_module_loaded(self, module_name: str) -> bool:
        """Check if a module is registered in the DI container."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return module_name in self._module_cache
            return True
        except Exception:
            return False

    def get_module(self, module_name: str) -> Optional[Any]:
        """Get a module from the DI container (sync wrapper)."""
        if module_name in self._module_cache:
            return self._module_cache[module_name]
        return None

    async def resolve_module(self, module_name: str) -> Optional[Any]:
        """Async method to resolve and cache a module."""
        try:
            module = await self._di_container.resolve(module_name)
            self._module_cache[module_name] = module
            return module
        except Exception as e:
            logger.warning(f"Failed to resolve module '{module_name}': {e}")
            return None


# ============================================================================
# Configuration Loading
# ============================================================================


def _load_config(module_path: Path) -> Dict[str, Any]:
    """Load and resolve config.json with environment variables."""
    config_file = module_path / "config.json"

    if not config_file.exists():
        logger.warning(f"Config file not found: {config_file}")
        return {}

    with open(config_file, "r", encoding="utf-8") as f:
        raw_config = f.read()

    # Resolve ${VAR:-default} patterns
    def resolve_env(match: re.Match) -> str:
        var_expr = match.group(1)
        if ":-" in var_expr:
            var_name, default = var_expr.split(":-", 1)
        else:
            var_name, default = var_expr, ""
        return os.environ.get(var_name, default)

    resolved = re.sub(r"\$\{([^}]+)\}", resolve_env, raw_config)

    # FIX-TYPE-001 v2.2.4: Coerce string "true"/"false" to bool after env resolution
    parsed = json.loads(resolved)
    return coerce_config_types(parsed)


def _build_investigation_config(config: Dict[str, Any]) -> InvestigationConfig:
    """Build InvestigationConfig from config dict."""
    inv_cfg = config.get("investigation", {})
    return InvestigationConfig(
        enabled=inv_cfg.get("enabled", True),
        default_num_questions=int(inv_cfg.get("default_num_questions", 5)),
        min_questions=int(inv_cfg.get("min_questions", 3)),
        max_questions=int(inv_cfg.get("max_questions", 10)),
        default_strategy=inv_cfg.get("default_strategy", "adaptive"),
        temperature=float(inv_cfg.get("temperature", 0.7)),
        max_tokens=int(inv_cfg.get("max_tokens", 500)),
        timeout_seconds=int(inv_cfg.get("timeout_seconds", 30)),
        retry_enabled=inv_cfg.get("retry_enabled", True),
        max_retries=int(inv_cfg.get("max_retries", 2)),
        retry_delay_seconds=float(inv_cfg.get("retry_delay_seconds", 1.0)),
    )


def _build_qa_config(config: Dict[str, Any]) -> QualityAssuranceConfig:
    """Build QualityAssuranceConfig from config dict."""
    qa_cfg = config.get("quality_assurance", {})
    scoring = qa_cfg.get("scoring", {})
    weights = scoring.get("weights", {})
    thresholds = scoring.get("thresholds", {})
    validation = qa_cfg.get("validation", {})
    
    return QualityAssuranceConfig(
        enabled=qa_cfg.get("enabled", True),
        auto_retry_on_low_quality=qa_cfg.get("auto_retry_on_low_quality", True),
        max_qa_retries=int(qa_cfg.get("max_qa_retries", 2)),
        min_acceptable_score=float(qa_cfg.get("min_acceptable_score", 4.0)),
        weight_relevance=float(weights.get("relevance", 0.40)),
        weight_specificity=float(weights.get("specificity", 0.25)),
        weight_length=float(weights.get("length", 0.20)),
        weight_structure=float(weights.get("structure", 0.15)),
        threshold_excellent=float(thresholds.get("excellent", 8.0)),
        threshold_good=float(thresholds.get("good", 6.0)),
        threshold_acceptable=float(thresholds.get("acceptable", 4.0)),
        length_optimal_min=int(scoring.get("length_optimal_min", 50)),
        length_optimal_max=int(scoring.get("length_optimal_max", 150)),
        min_words=int(scoring.get("min_words", 4)),
        min_length=int(scoring.get("min_length", 10)),
        max_length=int(scoring.get("max_length", 500)),
        require_question_mark=validation.get("require_question_mark", True),
        require_capitalization=validation.get("require_capitalization", False),
        min_keyword_overlap=float(validation.get("min_keyword_overlap", 0.1)),
        check_off_topic=validation.get("check_off_topic", True),
    )


def _build_worker_pool_config(config: Dict[str, Any]) -> WorkerPoolConfig:
    """Build WorkerPoolConfig from config dict."""
    wp_cfg = config.get("worker_pool", {})
    return WorkerPoolConfig(
        enabled=wp_cfg.get("enabled", True),
        pool_size=int(wp_cfg.get("pool_size", 4)),
        max_pool_size=int(wp_cfg.get("max_pool_size", 8)),
        task_timeout_seconds=int(wp_cfg.get("task_timeout_seconds", 30)),
        queue_max_size=int(wp_cfg.get("queue_max_size", 100)),
        retry_on_failure=wp_cfg.get("retry_on_failure", True),
        max_task_retries=int(wp_cfg.get("max_task_retries", 2)),
        backoff_multiplier=float(wp_cfg.get("backoff_multiplier", 1.5)),
        enable_priorities=wp_cfg.get("enable_priorities", True),
    )


def _build_session_config(config: Dict[str, Any]) -> SessionConfig:
    """Build SessionConfig from config dict."""
    sess_cfg = config.get("session_management", {})
    return SessionConfig(
        enabled=sess_cfg.get("enabled", True),
        ttl_seconds=int(sess_cfg.get("ttl_seconds", 3600)),
        max_history_size=int(sess_cfg.get("max_history_size", 50)),
        persist_results=sess_cfg.get("persist_results", True),
        auto_cleanup=sess_cfg.get("auto_cleanup", True),
    )


def _build_cache_config(config: Dict[str, Any], env: str) -> CacheConfig:
    """Build CacheConfig from config dict with environment isolation."""
    cache_cfg = config.get("cache", {})
    return CacheConfig(
        enabled=cache_cfg.get("enabled", True),
        ttl_seconds=int(cache_cfg.get("ttl_seconds", 3600)),
        base_prefix="ubp",
        env=env,
        cache_questions=cache_cfg.get("cache_questions", True),
        cache_qa_results=cache_cfg.get("cache_qa_results", True),
        cache_sessions=cache_cfg.get("cache_sessions", True),
    )


def _build_dedup_config(config: Dict[str, Any]) -> DeduplicationConfig:
    """Build DeduplicationConfig from config dict."""
    dedup_cfg = config.get("deduplication", {})
    return DeduplicationConfig(
        enabled=dedup_cfg.get("enabled", True),
        similarity_threshold=float(dedup_cfg.get("similarity_threshold", 0.85)),
        method=dedup_cfg.get("method", "fuzzy"),
        min_unique_ratio=float(dedup_cfg.get("min_unique_ratio", 0.70)),
    )


def _build_metrics_config(config: Dict[str, Any]) -> MetricsConfig:
    """Build MetricsConfig from config dict."""
    metrics_cfg = config.get("metrics", {})
    return MetricsConfig(
        enabled=metrics_cfg.get("enabled", True),
        collect_timings=metrics_cfg.get("collect_timings", True),
        collect_strategy_distribution=metrics_cfg.get("collect_strategy_distribution", True),
        collect_qa_scores=metrics_cfg.get("collect_qa_scores", True),
        retention_hours=int(metrics_cfg.get("retention_hours", 24)),
    )


def _build_debug_config(config: Dict[str, Any]) -> DebugConfig:
    """Build DebugConfig from config dict."""
    debug_cfg = config.get("debug", {})
    return DebugConfig(
        enabled=debug_cfg.get("enabled", False),
        log_prompts=debug_cfg.get("log_prompts", False),
        log_responses=debug_cfg.get("log_responses", False),
        log_qa_scores=debug_cfg.get("log_qa_scores", True),
        log_fallback_triggers=debug_cfg.get("log_fallback_triggers", True),
        log_strategy_selection=debug_cfg.get("log_strategy_selection", True),
        log_worker_stats=debug_cfg.get("log_worker_stats", True),
        trace_execution=debug_cfg.get("trace_execution", False),
    )


def _build_delegation_config(config: Dict[str, Any]) -> LLMDelegationConfig:
    """Build LLMDelegationConfig from config dict."""
    del_cfg = config.get("delegation", {})
    return LLMDelegationConfig(
        llm_module=del_cfg.get("llm_module", "inference_ollama_grok"),
        llm_operation=del_cfg.get("llm_operation", "generate"),
        provider=del_cfg.get("provider", "grok"),
        timeout_seconds=int(del_cfg.get("timeout_seconds", 30)),
        max_retries=int(del_cfg.get("max_retries", 2)),
        fallback_enabled=del_cfg.get("fallback_enabled", True),
        fallback_chain=del_cfg.get("fallback_chain", ["decomposition", "chain_of_thought", "simple"]),
    )


def _build_pipeline_config(config: Dict[str, Any]) -> PipelineConfig:
    """Build PipelineConfig from config dict."""
    pipeline_cfg = config.get("pipeline", {})
    return PipelineConfig.from_dict(pipeline_cfg)


# ============================================================================
# InvestigationPipelineAdapter
# ============================================================================


class InvestigationPipelineAdapter:
    """
    Main adapter for investigation_pipeline module.
    
    Implements all operations defined in manifest.json.
    Uses dependency injection for external services.
    """

    def __init__(
        self,
        module_path: Path,
        di_container: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        self.module_path = module_path
        self.di_container = di_container
        self.event_bus = event_bus

        # Load configuration
        self.config = _load_config(module_path)

        # Get environment for cache isolation (dev/test/prod)
        self.env = os.environ.get("UBP_ENV", "dev")

        # Build component configs
        self.investigation_config = _build_investigation_config(self.config)
        self.qa_config = _build_qa_config(self.config)
        self.worker_pool_config = _build_worker_pool_config(self.config)
        self.session_config = _build_session_config(self.config)
        self.cache_config = _build_cache_config(self.config, self.env)
        self.dedup_config = _build_dedup_config(self.config)
        self.metrics_config = _build_metrics_config(self.config)
        self.debug_config = _build_debug_config(self.config)
        self.delegation_config = _build_delegation_config(self.config)
        self.pipeline_config = _build_pipeline_config(self.config)

        # Components (initialized in initialize())
        self._qa_provider: Optional[QualityAssuranceProvider] = None
        self._deduplicator: Optional[InvestigationDeduplicator] = None
        self._worker_pool: Optional[InvestigationWorkerPool] = None
        self._session_manager: Optional[InvestigationSessionManager] = None
        self._cache: Optional[RedisCacheProvider] = None
        self._metrics: Optional[MetricsCollector] = None
        self._classifier: Optional[QueryClassifier] = None
        self._delegator: Optional[InvestigationDelegator] = None
        self._pipeline: Optional[InvestigationPipeline] = None

        # State
        self._initialized = False
        self._module_registry: Optional[Any] = None

    # MCP-COMPAT: OperationContext helpers (ARCH-008)
    def _build_context_from_di(self) -> OperationContext:
        """Build OperationContext from DI — backward compatibility for REST path."""
        return OperationContext(
            client_id="default",
            user_id=None,
            session_id=None,
            source="rest",
        )

    def _normalize_ctx(self, ctx: Any) -> OperationContext:
        """Normalize any context format to OperationContext."""
        if ctx is None:
            return self._build_context_from_di()
        if isinstance(ctx, OperationContext):
            return ctx
        if hasattr(ctx, "user") and ctx.user:
            user_id = getattr(ctx.user, "user_id", None)
            roles = getattr(ctx.user, "roles", [])
            client_id = getattr(ctx.user, "client_id", "default")
            if not isinstance(roles, (list, tuple)):
                roles = []
            return OperationContext(
                client_id=str(client_id) if client_id else "default",
                user_id=str(user_id) if user_id else None,
                roles=list(roles),
                source="rest",
            )
        return self._build_context_from_di()

    # ========================================================================
    # Lifecycle Operations
    # ========================================================================

    async def initialize(
        self,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Initialize investigation pipeline."""
        start_time = time.perf_counter()

        try:
            # Resolve module registry and Redis from DI
            redis_client = None
            if self.di_container:
                self._module_registry = DIContainerModuleRegistry(self.di_container)

                # Try to get Redis client for caching
                try:
                    import redis.asyncio as aioredis
                    redis_client = await self.di_container.resolve(aioredis.Redis)
                except Exception as e:
                    logger.info(f"Redis not available for caching: {e}")

            # Initialize cache provider with environment isolation
            self._cache = RedisCacheProvider(self.cache_config, redis_client)
            logger.info(f"Cache initialized with prefix: {self.cache_config.prefix}")

            # Initialize providers
            self._qa_provider = QualityAssuranceProvider(self.qa_config)
            self._deduplicator = InvestigationDeduplicator(self.dedup_config)
            self._classifier = QueryClassifier(QUERY_CATEGORIES, "technical")
            self._session_manager = InvestigationSessionManager(self.session_config)
            self._metrics = MetricsCollector(self.metrics_config)

            # Initialize worker pool
            if self.worker_pool_config.enabled:
                self._worker_pool = InvestigationWorkerPool(self.worker_pool_config)
                await self._worker_pool.start()

            # Initialize delegator via _get_delegator() which handles
            # ProviderMapper fallback (BUG-001 fix)
            if self._module_registry:
                await self._get_delegator()

            # Initialize pipeline
            if self._delegator:
                self._pipeline = InvestigationPipeline(
                    delegator=self._delegator,
                    qa_provider=self._qa_provider,
                    deduplicator=self._deduplicator,
                    classifier=self._classifier,
                    config=self.pipeline_config,
                    debug_config=self.debug_config,
                )

            self._initialized = True
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return {
                "status": "initialized",
                "module": "investigation_pipeline",
                "env": self.env,
                "default_strategy": self.investigation_config.default_strategy,
                "pipeline_steps_available": InvestigationPipeline.AVAILABLE_STEPS,
                "strategies_available": [s.value for s in InvestigationStrategy],
                "llm_delegation_available": self._delegator.is_available() if self._delegator else False,
                "worker_pool_enabled": self.worker_pool_config.enabled,
                "cache_enabled": self.cache_config.enabled,
                "cache_prefix": self.cache_config.prefix,
                "elapsed_ms": round(elapsed_ms, 2),
            }

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return {
                "status": "error",
                "module": "investigation_pipeline",
                "error": str(e),
            }

    async def shutdown(self, ctx=None, **kwargs) -> Dict[str, Any]:
        """Graceful shutdown."""
        resources_released = []

        if self._worker_pool:
            await self._worker_pool.stop()
            resources_released.append("worker_pool")

        self._initialized = False

        return {
            "status": "shutdown",
            "resources_released": resources_released,
        }

    async def health_check(self, ctx=None, **kwargs) -> Dict[str, Any]:
        """Health check for investigation module."""
        result: Dict[str, Any] = {
            "module": "investigation_pipeline",
            "status": "healthy",
            "env": self.env,
        }

        # Check LLM delegation
        if not self._delegator:
            await self._get_delegator()

        if self._delegator:
            result["llm_delegation"] = await self._delegator.health_check()
        else:
            result["llm_delegation"] = {"status": "not_configured"}

        # Check worker pool
        if self._worker_pool:
            result["worker_pool"] = self._worker_pool.get_stats().to_dict()
        else:
            result["worker_pool"] = {"status": "disabled"}

        # Cache stats
        if self._cache:
            result["cache"] = self._cache.get_stats()
        else:
            result["cache"] = {"status": "disabled"}

        return result

    async def _get_delegator(self) -> Optional[InvestigationDelegator]:
        """Lazily resolve the delegator to avoid DI race conditions."""
        if self._delegator and self._delegator.is_available():
            return self._delegator

        if not self.di_container:
            logger.warning("Delegator requested but DI container is not available")
            return None

        if not self._module_registry:
            self._module_registry = DIContainerModuleRegistry(self.di_container)

        # Try ProviderMapper if available
        if PROVIDER_MAPPER_AVAILABLE and ProviderMapper:
            try:
                # "investigation" is NOT a valid role in ProviderMapper.
                # Use "enrichment" (same as hyde_pipeline) for correct LLM routing.
                provider_chain = ProviderMapper.resolve_chain("enrichment")
            except ProviderConfigurationError as exc:
                logger.error(f"[INVESTIGATION] Provider configuration error: {exc}")
                provider_chain = None

            if provider_chain:
                for module_name, provider_name in provider_chain:
                    resolved_llm = await self._module_registry.resolve_module(module_name)
                    if not resolved_llm:
                        logger.warning(
                            f"[INVESTIGATION] FALLBACK: module '{module_name}' "
                            f"(provider '{provider_name}') not ready, trying next"
                        )
                        continue
                    if hasattr(resolved_llm, "set_default_provider"):
                        try:
                            resolved_llm.set_default_provider(provider_name)
                        except Exception as exc:
                            logger.warning(
                                f"[INVESTIGATION] FALLBACK: set_default_provider('{provider_name}') "
                                f"failed on '{module_name}': {exc}, trying next"
                            )
                            continue

                    self.delegation_config = LLMDelegationConfig(
                        llm_module=module_name,
                        llm_operation=self.delegation_config.llm_operation,
                        provider=provider_name,  # FIX BUG-001: was self.delegation_config.provider
                        timeout_seconds=self.delegation_config.timeout_seconds,
                        max_retries=self.delegation_config.max_retries,
                        fallback_enabled=self.delegation_config.fallback_enabled,
                        fallback_chain=self.delegation_config.fallback_chain,
                    )
                    logger.info(
                        f"[INVESTIGATION] LLM linked to module '{module_name}' "
                        f"with provider '{provider_name}'"
                    )
                    break
                else:
                    logger.error(
                        "[INVESTIGATION] FALLBACK EXHAUSTED: no LLM module resolved "
                        "from provider chain"
                    )

        # Create delegator
        self._delegator = InvestigationDelegator(
            config=self.delegation_config,
            module_registry=self._module_registry,
            event_publisher=self.event_bus,
            debug_config={
                "log_prompts": self.debug_config.log_prompts,
                "log_responses": self.debug_config.log_responses,
                "log_strategy_selection": self.debug_config.log_strategy_selection,
                "log_fallback_triggers": self.debug_config.log_fallback_triggers,
            },
        )

        # Update pipeline if exists
        if self._pipeline:
            self._pipeline.delegator = self._delegator

        return self._delegator

    # ========================================================================
    # Main Operations
    # ========================================================================

    async def investigate(
        self,
        query: str,
        num_questions: int = 5,
        strategy: Optional[str] = None,
        session_id: Optional[str] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Full investigation pipeline execution.
        
        Args:
            query: User's original query
            num_questions: Number of questions to generate
            strategy: Strategy override (adaptive, decomposition, chain_of_thought, etc.)
            session_id: Optional session ID for continuity
            pipeline_config: Override pipeline steps
            ctx: Security context
            
        Returns:
            Complete investigation result with questions and stats
        """
        if not self._pipeline:
            raise RuntimeError("Module not initialized")

        # Ensure delegator is linked
        await self._get_delegator()

        # Create or use session
        if not session_id:
            session = await self._session_manager.create_session()
            session_id = session.session_id

        # Execute pipeline
        result = await self._pipeline.execute(
            query=query,
            session_id=session_id,
            num_questions=num_questions,
            strategy=strategy,
            pipeline_config=pipeline_config,
            ctx=ctx,
        )

        # Update session
        await self._session_manager.update_session(
            session_id=session_id,
            query=query,
            questions=result.questions,
            strategy=result.strategy_used,
            category=result.category_detected,
        )

        # Record metrics
        if self._metrics:
            await self._metrics.record_investigation(
                questions_count=len(result.questions),
                strategy=result.strategy_used,
                category=result.category_detected,
                qa_score=result.quality_assessment.overall_score if result.quality_assessment else 0,
                execution_time_ms=result.time_ms,
                used_fallback=False,  # TODO: track from generation result
            )

        return result.to_dict()

    async def generate_questions(
        self,
        query: str,
        num_questions: int = 5,
        strategy: str = "adaptive",
        category: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Direct question generation without full pipeline.
        
        Args:
            query: User's query
            num_questions: Number of questions
            strategy: Generation strategy
            category: Pre-classified category
            ctx: Security context
            
        Returns:
            Generated questions with metadata
        """
        delegator = await self._get_delegator()
        if not delegator:
            raise RuntimeError("LLM delegation not configured")

        result = await delegator.generate_investigation(
            query=query,
            num_questions=num_questions,
            strategy=strategy,
            category=category,
        )

        return result.to_dict()

    async def generate_multi_strategy(
        self,
        query: str,
        num_questions: int = 5,
        strategies: Optional[List[str]] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate questions using multiple strategies in parallel.
        
        Args:
            query: User's query
            num_questions: Questions per strategy
            strategies: List of strategies to use
            ctx: Security context
            
        Returns:
            Results from each strategy
        """
        delegator = await self._get_delegator()
        if not delegator:
            raise RuntimeError("LLM delegation not configured")

        results = await delegator.generate_multi_strategy(
            query=query,
            num_questions=num_questions,
            strategies=strategies,
        )

        return {
            strategy: result.to_dict()
            for strategy, result in results.items()
        }

    async def classify_query(
        self,
        query: str,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Classify query to detect category and optimal strategy.
        
        Args:
            query: User's query
            ctx: Security context
            
        Returns:
            Classification result with category and strategy
        """
        if not self._classifier:
            raise RuntimeError("Module not initialized")

        classification = self._classifier.classify(query)
        return classification.to_dict()

    async def assess_quality(
        self,
        questions: List[str],
        original_query: str,
        strategy: str = "unknown",
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Assess quality of investigation questions.
        
        Args:
            questions: List of question texts
            original_query: Original user query
            strategy: Strategy used
            ctx: Security context
            
        Returns:
            Quality assessment with scores
        """
        if not self._qa_provider:
            raise RuntimeError("Module not initialized")

        assessed_questions, overall = self._qa_provider.assess_questions(
            questions=questions,
            original_query=original_query,
            strategy=strategy,
        )

        return {
            "questions": [q.to_dict() for q in assessed_questions],
            "overall_assessment": overall.to_dict(),
        }

    async def deduplicate_questions(
        self,
        questions: List[Dict[str, Any]],
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Remove duplicate questions.
        
        Args:
            questions: List of question dicts
            ctx: Security context
            
        Returns:
            Unique questions and stats
        """
        if not self._deduplicator:
            raise RuntimeError("Module not initialized")

        # Convert to InvestigationQuestion
        question_objects = [
            InvestigationQuestion.from_dict(q) if isinstance(q, dict) else q
            for q in questions
        ]

        unique, stats = self._deduplicator.deduplicate(question_objects)

        return {
            "unique_questions": [q.to_dict() for q in unique],
            "stats": stats,
        }

    # ========================================================================
    # Session Operations
    # ========================================================================

    async def get_session(
        self,
        session_id: str,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get session state."""
        if not self._session_manager:
            raise RuntimeError("Module not initialized")

        session = await self._session_manager.get_session(session_id)
        if session:
            return session.to_dict()
        return {"error": "Session not found", "session_id": session_id}

    async def delete_session(
        self,
        session_id: str,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Delete session."""
        if not self._session_manager:
            raise RuntimeError("Module not initialized")

        deleted = await self._session_manager.delete_session(session_id)
        return {
            "deleted": deleted,
            "session_id": session_id,
        }

    # ========================================================================
    # Statistics Operations
    # ========================================================================

    async def get_stats(
        self,
        period: str = "24h",
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get investigation statistics."""
        stats: Dict[str, Any] = {
            "module": "investigation_pipeline",
            "period": period,
        }

        if self._metrics:
            stats["metrics"] = self._metrics.get_metrics()

        if self._worker_pool:
            stats["worker_pool"] = self._worker_pool.get_stats().to_dict()

        if self._cache:
            stats["cache"] = self._cache.get_stats()

        return stats

    # ========================================================================
    # Configuration Operations
    # ========================================================================

    async def get_pipeline_config(self, ctx=None, **kwargs) -> Dict[str, Any]:
        """Get current pipeline configuration."""
        if not self._pipeline:
            raise RuntimeError("Module not initialized")

        return self._pipeline.get_config()

    async def set_pipeline_config(
        self,
        pipeline_steps: List[Dict[str, Any]],
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Update pipeline configuration."""
        if not self._pipeline:
            raise RuntimeError("Module not initialized")

        # Check admin permission
        if ctx and hasattr(ctx, "user"):
            if not getattr(ctx.user, "is_admin", False):
                raise PermissionError("Admin privileges required")

        new_config = self._pipeline.update_config(pipeline_steps)

        return {
            "updated": True,
            "new_config": new_config,
        }

    async def reload_config(
        self,
        clear_cache: bool = True,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Hot-reload configuration from config.json without restart.
        
        Security: Admin only operation.
        """
        # Check admin permission
        if ctx and hasattr(ctx, "user"):
            if not getattr(ctx.user, "is_admin", False):
                raise PermissionError("Admin privileges required")

        reload_results = {
            "config_reloaded": False,
            "cache_cleared": False,
            "components_updated": [],
            "errors": [],
        }

        try:
            # 1. Reload config.json
            self.config = _load_config(self.module_path)
            reload_results["config_reloaded"] = True
            logger.info("Config.json reloaded")

            # 2. Rebuild component configs
            self.investigation_config = _build_investigation_config(self.config)
            reload_results["components_updated"].append("investigation_config")

            self.qa_config = _build_qa_config(self.config)
            reload_results["components_updated"].append("qa_config")

            self.dedup_config = _build_dedup_config(self.config)
            reload_results["components_updated"].append("dedup_config")

            self.delegation_config = _build_delegation_config(self.config)
            reload_results["components_updated"].append("delegation_config")

            self.pipeline_config = _build_pipeline_config(self.config)
            reload_results["components_updated"].append("pipeline_config")

            self.cache_config = _build_cache_config(self.config, self.env)
            reload_results["components_updated"].append("cache_config")

            self.debug_config = _build_debug_config(self.config)
            reload_results["components_updated"].append("debug_config")

            # 3. Update live component instances if initialized
            if self._qa_provider:
                self._qa_provider.config = self.qa_config
            if self._deduplicator:
                self._deduplicator.config = self.dedup_config
            if self._pipeline:
                self._pipeline.config = self.pipeline_config

            logger.info(f"Updated {len(reload_results['components_updated'])} component configs")

        except Exception as e:
            reload_results["errors"].append(f"Config reload failed: {str(e)}")
            logger.error(f"Failed to reload config: {e}")

        # 4. Clear cache if requested
        if clear_cache and self._cache:
            try:
                cleared = await self._cache.clear()
                reload_results["cache_cleared"] = True
                reload_results["cache_entries_cleared"] = cleared
                logger.info(f"Cache cleared: {cleared} entries")
            except Exception as e:
                reload_results["errors"].append(f"Cache clear failed: {str(e)}")
                logger.error(f"Failed to clear cache: {e}")

        # Get current pipeline config
        current_config = None
        if self._pipeline:
            current_config = self._pipeline.get_config()

        return {
            "status": "success" if not reload_results["errors"] else "partial",
            "module": "investigation_pipeline",
            "operation": "reload_config",
            "current_pipeline_config": current_config,
            "default_strategy": self.investigation_config.default_strategy,
            **reload_results,
        }
