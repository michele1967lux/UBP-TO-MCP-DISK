"""
hyde_pipeline/adapter.py

Bridge layer that exposes all HyDE operations to the UBP system.
Handles initialization, configuration, DI resolution, and operation routing.

Follows the 3-file pattern from enrichment_pipeline.

Operations:
- initialize: Start components and worker pool
- generate_hyde: Full pipeline with session management
- generate_document: Direct document generation
- generate_ensemble: Multi-document ensemble generation
- classify_query: Domain and language detection
- assess_quality: Document quality scoring
- check_hallucination: Detect fabricated content
- refine_document: Iterative document improvement
- chunk_document: Semantic chunking
- fuse_documents: Combine multiple documents
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

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

from .providers import (
    # Data classes
    HyDEDocument,
    DocumentChunk,
    QualityAssessment,
    HallucinationCheck,
    DomainClassification,
    EnsembleResult,
    RefinementResult,
    HyDESession,
    HyDEResult,
    WorkerTask,
    WorkerStats,
    # Enums
    DocumentFormat,
    Domain,
    QualityLevel,
    RefinementStrategy,
    ChunkingStrategy,
    TaskStatus,
    TaskPriority,
    # Configs
    HyDEConfig,
    QualityAssuranceConfig,
    EnsembleConfig,
    RefinementConfig,
    HallucinationConfig,
    ChunkingConfig,
    CacheConfig,
    SessionConfig,
    WorkerPoolConfig,
    MetricsConfig,
    DebugConfig,
    # Providers
    DomainClassifier,
    QualityAssuranceProvider,
    HallucinationDetector,
    DocumentChunker,
    EnsembleFusion,
    RedisCacheProvider,
    HyDESessionManager,
    HyDEWorkerPool,
    MetricsCollector,
)
from .delegation import (
    HyDEDelegator,
    LLMDelegationConfig,
    GenerationResult,
)
from .pipeline import (
    HyDEPipeline,
    PipelineConfig,
    StepConfig,
    PipelineContext,
)
from .prompts import DOMAINS

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


class IProviderMapper(Protocol):
    """Protocol for provider mapping."""
    def get_provider_chain(self, role: str) -> List[str]: ...


# ============================================================================
# DI Container Module Registry Wrapper
# ============================================================================


class DIContainerModuleRegistry:
    """
    Wraps DI container to provide module registry interface.
    Handles lazy resolution of modules.
    """

    def __init__(self, di_container: Optional[Any] = None):
        self._container = di_container
        self._cached_modules: Dict[str, Any] = {}

    def get_module(self, module_name: str) -> Optional[Any]:
        """Get a module by name (sync - cache only)."""
        if module_name in self._cached_modules:
            return self._cached_modules[module_name]
        return None

    def is_module_loaded(self, module_name: str) -> bool:
        """Check if module is loaded."""
        return module_name in self._cached_modules

    async def resolve_module(self, module_name: str) -> Optional[Any]:
        """Async module resolution via DI container."""
        # Return cached if available
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


# ============================================================================
# Configuration Utilities
# ============================================================================


def resolve_env_value(value: Any) -> Any:
    """
    Resolve environment variable placeholders.
    Format: ${VAR_NAME:-default_value}
    """
    if not isinstance(value, str):
        return value
    
    pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
    
    def replace(match):
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)
    
    return re.sub(pattern, replace, value)


def coerce_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively coerce configuration values to appropriate types.
    Handles booleans, integers, floats.
    """
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
    
    # Resolve env vars first
    value = resolve_env_value(value)
    
    if not isinstance(value, str):
        return value
    
    # Boolean
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False
    
    # Number
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    return value


# ============================================================================
# Query Validation & Cache Utilities
# ============================================================================


def _validate_query(query: str, min_length: int = 3, max_length: int = 2000) -> None:
    """
    Validate query input.

    Raises ValueError for invalid queries so the API layer returns 400.
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string")

    stripped = query.strip()
    if not stripped:
        raise ValueError("Query cannot be empty or whitespace-only")

    if len(stripped) < min_length:
        raise ValueError(
            f"Query too short: {len(stripped)} chars (minimum {min_length})"
        )

    if len(stripped) > max_length:
        raise ValueError(
            f"Query too long: {len(stripped)} chars (maximum {max_length})"
        )


def _coerce_document_input(document: Any) -> HyDEDocument:
    """
    Coerce document input to HyDEDocument.

    Accepts:
    - dict: passed to HyDEDocument.from_dict()
    - str: wrapped as {"content": string, "query": ""}
    - HyDEDocument: returned as-is

    Raises ValueError for unsupported types.
    """
    if isinstance(document, HyDEDocument):
        return document
    if isinstance(document, str):
        document = document.strip()
        if not document:
            raise ValueError("Document content cannot be empty")
        return HyDEDocument.from_dict({"content": document})
    if isinstance(document, dict):
        return HyDEDocument.from_dict(document)
    raise ValueError(
        f"Document must be a string, dict, or HyDEDocument, got {type(document).__name__}"
    )


def _normalize_cache_key(raw_key: str) -> str:
    """
    Normalize a cache key for better hit rates.

    - lowercase
    - strip whitespace
    - collapse multiple spaces to single space
    """
    import re as _re
    normalized = raw_key.lower().strip()
    normalized = _re.sub(r'\s+', ' ', normalized)
    return normalized


# ============================================================================
# HyDE Pipeline Adapter
# ============================================================================


class HyDEPipelineAdapter:
    """
    Adapter that exposes HyDE operations to the UBP system.
    
    Responsibilities:
    - Load configuration from config.json
    - Initialize all components
    - Expose operations for ModuleLoader
    - Handle session and cache management
    - Provide metrics and health checks
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
        self._hyde_config: Optional[HyDEConfig] = None
        self._qa_config: Optional[QualityAssuranceConfig] = None
        self._ensemble_config: Optional[EnsembleConfig] = None
        self._refinement_config: Optional[RefinementConfig] = None
        self._hallucination_config: Optional[HallucinationConfig] = None
        self._chunking_config: Optional[ChunkingConfig] = None
        self._cache_config: Optional[CacheConfig] = None
        self._session_config: Optional[SessionConfig] = None
        self._worker_config: Optional[WorkerPoolConfig] = None
        self._metrics_config: Optional[MetricsConfig] = None
        self._debug_config: Optional[DebugConfig] = None
        self._delegation_config: Optional[LLMDelegationConfig] = None
        self._pipeline_config: Optional[PipelineConfig] = None
        
        # Components
        self._classifier: Optional[DomainClassifier] = None
        self._qa_provider: Optional[QualityAssuranceProvider] = None
        self._hallucination_detector: Optional[HallucinationDetector] = None
        self._chunker: Optional[DocumentChunker] = None
        self._ensemble_fusion: Optional[EnsembleFusion] = None
        self._cache: Optional[RedisCacheProvider] = None
        self._session_manager: Optional[HyDESessionManager] = None
        self._worker_pool: Optional[HyDEWorkerPool] = None
        self._metrics: Optional[MetricsCollector] = None
        self._delegator: Optional[HyDEDelegator] = None
        self._pipeline: Optional[HyDEPipeline] = None
        
        # State
        self._initialized = False
        self._redis_client: Optional[Any] = None
    
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
        
        # Resolve environment variables and coerce types
        return coerce_config_types(raw_config)
    
    def _build_hyde_config(self) -> HyDEConfig:
        """Build HyDE config from loaded configuration."""
        cfg = self._config.get("hyde", {})
        return HyDEConfig(
            enabled=cfg.get("enabled", True),
            default_format=cfg.get("default_format", "answer"),
            default_domain=cfg.get("default_domain", "auto"),
            default_length=cfg.get("default_length", 300),
            min_length=cfg.get("min_length", 100),
            max_length=cfg.get("max_length", 1000),
            temperature=cfg.get("temperature", 0.5),
            max_tokens=cfg.get("max_tokens", 600),
            timeout_seconds=cfg.get("timeout_seconds", 30),
            retry_enabled=cfg.get("retry_enabled", True),
            max_retries=cfg.get("max_retries", 2),
            retry_delay_seconds=cfg.get("retry_delay_seconds", 1.0),
        )
    
    def _build_qa_config(self) -> QualityAssuranceConfig:
        """Build QA config."""
        cfg = self._config.get("quality_assurance", {})
        weights = cfg.get("scoring", {}).get("weights", {})
        thresholds = cfg.get("scoring", {}).get("thresholds", {})
        
        return QualityAssuranceConfig(
            enabled=cfg.get("enabled", True),
            weight_relevance=weights.get("relevance", 0.35),
            weight_coherence=weights.get("coherence", 0.25),
            weight_informativeness=weights.get("informativeness", 0.20),
            weight_format_adherence=weights.get("format_adherence", 0.10),
            weight_terminology=weights.get("terminology", 0.10),
            threshold_excellent=thresholds.get("excellent", 8.0),
            threshold_good=thresholds.get("good", 6.0),
            threshold_acceptable=thresholds.get("acceptable", 4.0),
        )
    
    def _build_ensemble_config(self) -> EnsembleConfig:
        """Build ensemble config."""
        cfg = self._config.get("ensemble", {})
        return EnsembleConfig(
            enabled=cfg.get("enabled", True),
            default_count=cfg.get("default_count", 3),
            max_count=cfg.get("max_count", 5),
            fusion_strategy=cfg.get("fusion_strategy", "weighted_concat"),
            diversity_penalty=cfg.get("diversity_penalty", 0.1),
            parallel_generation=cfg.get("parallel_generation", True),
            temperature_spread=cfg.get("temperature_spread", 0.2),
            format_diversity=cfg.get("format_diversity", True),
        )
    
    def _build_refinement_config(self) -> RefinementConfig:
        """Build refinement config."""
        cfg = self._config.get("refinement", {})
        strategies = cfg.get("strategies", {})
        
        return RefinementConfig(
            enabled=cfg.get("enabled", True),
            max_iterations=cfg.get("max_iterations", 2),
            quality_threshold=cfg.get("quality_threshold", 6.0),
            improvement_min=cfg.get("improvement_min", 0.5),
            strategies_enabled={
                "expand": strategies.get("expand", True),
                "focus": strategies.get("focus", True),
                "technical": strategies.get("technical", True),
                "simplify": strategies.get("simplify", True),
            },
        )
    
    def _build_hallucination_config(self) -> HallucinationConfig:
        """Build hallucination detection config."""
        cfg = self._config.get("hallucination_detection", {})
        return HallucinationConfig(
            enabled=cfg.get("enabled", True),
            check_unknown_terms=cfg.get("check_unknown_terms", True),
            check_invented_apis=cfg.get("check_invented_apis", True),
            check_fake_versions=cfg.get("check_fake_versions", True),
            confidence_penalty=cfg.get("confidence_penalty", 0.2),
            max_unknown_ratio=cfg.get("max_unknown_ratio", 0.15),
        )
    
    def _build_chunking_config(self) -> ChunkingConfig:
        """Build chunking config."""
        cfg = self._config.get("chunking", {})
        return ChunkingConfig(
            enabled=cfg.get("enabled", True),
            strategy=cfg.get("strategy", "semantic"),
            chunk_size=cfg.get("chunk_size", 256),
            chunk_overlap=cfg.get("chunk_overlap", 50),
            min_chunk_size=cfg.get("min_chunk_size", 100),
            preserve_sentences=cfg.get("preserve_sentences", True),
            preserve_code_blocks=cfg.get("preserve_code_blocks", True),
        )
    
    def _build_cache_config(self) -> CacheConfig:
        """Build cache config."""
        cfg = self._config.get("cache", {})
        env = os.environ.get("UBP_ENV", "dev")
        
        return CacheConfig(
            enabled=cfg.get("enabled", True),
            ttl_seconds=cfg.get("ttl_seconds", 3600),
            base_prefix="ubp",
            env=env,
            cache_documents=cfg.get("cache_documents", True),
            cache_chunks=cfg.get("cache_chunks", True),
            cache_qa_results=cfg.get("cache_qa_results", True),
            semantic_matching=cfg.get("semantic_matching", True),
            semantic_threshold=cfg.get("semantic_threshold", 0.92),
        )
    
    def _build_session_config(self) -> SessionConfig:
        """Build session config."""
        cfg = self._config.get("session_management", {})
        return SessionConfig(
            enabled=cfg.get("enabled", True),
            ttl_seconds=cfg.get("ttl_seconds", 3600),
            max_history_size=cfg.get("max_history_size", 30),
            persist_documents=cfg.get("persist_documents", True),
            auto_cleanup=cfg.get("auto_cleanup", True),
        )
    
    def _build_worker_config(self) -> WorkerPoolConfig:
        """Build worker pool config."""
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
            enable_priorities=cfg.get("enable_priorities", True),
        )
    
    def _build_metrics_config(self) -> MetricsConfig:
        """Build metrics config."""
        cfg = self._config.get("metrics", {})
        return MetricsConfig(
            enabled=cfg.get("enabled", True),
            collect_timings=cfg.get("collect_timings", True),
            collect_format_distribution=cfg.get("collect_format_distribution", True),
            collect_domain_distribution=cfg.get("collect_domain_distribution", True),
            collect_qa_scores=cfg.get("collect_qa_scores", True),
            collect_hallucination_rates=cfg.get("collect_hallucination_rates", True),
            collect_refinement_stats=cfg.get("collect_refinement_stats", True),
            retention_hours=cfg.get("retention_hours", 24),
        )
    
    def _build_debug_config(self) -> DebugConfig:
        """Build debug config."""
        cfg = self._config.get("debug", {})
        return DebugConfig(
            enabled=cfg.get("enabled", False),
            log_prompts=cfg.get("log_prompts", False),
            log_responses=cfg.get("log_responses", False),
            log_qa_scores=cfg.get("log_qa_scores", True),
            log_hallucination_checks=cfg.get("log_hallucination_checks", True),
            log_refinement_steps=cfg.get("log_refinement_steps", True),
            log_ensemble_details=cfg.get("log_ensemble_details", True),
            log_chunking=cfg.get("log_chunking", False),
            trace_execution=cfg.get("trace_execution", False),
        )
    
    def _build_delegation_config(self) -> LLMDelegationConfig:
        """Build LLM delegation config."""
        cfg = self._config.get("delegation", {})
        provider = cfg.get("provider", "") or None  # Convert empty string to None
        return LLMDelegationConfig(
            llm_module=cfg.get("llm_module", "inference_ollama_grok"),
            llm_operation=cfg.get("llm_operation", "generate"),
            timeout_seconds=cfg.get("timeout_seconds", 30),
            max_retries=cfg.get("max_retries", 2),
            fallback_enabled=cfg.get("fallback_enabled", True),
            fallback_chain=cfg.get("fallback_chain", ["answer", "technical_doc", "faq"]),
            provider=provider,
        )
    
    def _build_pipeline_config(self) -> PipelineConfig:
        """Build pipeline config."""
        cfg = self._config.get("pipeline", {})
        steps_cfg = cfg.get("steps", {})
        
        steps = {}
        for step_name, step_data in steps_cfg.items():
            if isinstance(step_data, dict):
                steps[step_name] = StepConfig(
                    enabled=step_data.get("enabled", True),
                    timeout=step_data.get("timeout", 30),
                )
        
        return PipelineConfig(
            default_timeout_seconds=cfg.get("default_timeout_seconds", 60),
            fail_fast=cfg.get("fail_fast", False),
            steps=steps,
        )
    
    # ========================================================================
    # LLM Delegator Resolution
    # ========================================================================
    
    def _get_delegator(self) -> HyDEDelegator:
        """Get or create the LLM delegator (lazy resolution)."""
        if self._delegator:
            return self._delegator

        delegation_config = self._delegation_config or self._build_delegation_config()

        # Use ProviderMapper with "enrichment" role (same as enrichment_pipeline)
        # "hyde" is NOT a valid role - must use "enrichment" for correct routing
        try:
            from ubp_enterprise_hybrid.modules.cores._shared import ProviderMapper, ProviderConfigurationError
            provider_chain = ProviderMapper.resolve_chain("enrichment")
            if provider_chain:
                module_name, provider_name = provider_chain[0]
                delegation_config.llm_module = module_name
                delegation_config.provider = provider_name

                # v6.0.1: Model resolution removed — inference module resolves from ProviderInventory
                logger.info(
                    f"[HYDE] ProviderMapper resolved: module='{module_name}', "
                    f"provider='{provider_name}'"
                )
        except Exception as e:
            logger.warning(
                f"[HYDE] ProviderMapper resolution failed, using config defaults: {e}"
            )

        self._delegator = HyDEDelegator(
            config=delegation_config,
            module_registry=self._module_registry,
            event_publisher=self._event_bus,
            debug_config={
                "log_prompts": self._debug_config.log_prompts if self._debug_config else False,
                "log_responses": self._debug_config.log_responses if self._debug_config else False,
                "log_strategy_selection": self._debug_config.trace_execution if self._debug_config else False,
                "log_fallback_triggers": self._debug_config.log_refinement_steps if self._debug_config else True,
            },
        )

        return self._delegator
    
    # ========================================================================
    # Operations
    # ========================================================================
    
    async def initialize(self, ctx: Any = None) -> Dict[str, Any]:
        """Initialize all HyDE components."""
        if self._initialized:
            return {"status": "already_initialized"}
        
        try:
            # Load configuration
            self._config = self._load_config()
            
            # Build configs
            self._hyde_config = self._build_hyde_config()
            self._qa_config = self._build_qa_config()
            self._ensemble_config = self._build_ensemble_config()
            self._refinement_config = self._build_refinement_config()
            self._hallucination_config = self._build_hallucination_config()
            self._chunking_config = self._build_chunking_config()
            self._cache_config = self._build_cache_config()
            self._session_config = self._build_session_config()
            self._worker_config = self._build_worker_config()
            self._metrics_config = self._build_metrics_config()
            self._debug_config = self._build_debug_config()
            self._delegation_config = self._build_delegation_config()
            self._pipeline_config = self._build_pipeline_config()
            
            # Get domains config
            domains_config = self._config.get("domains", DOMAINS)
            
            # Initialize components
            self._classifier = DomainClassifier(
                domains_config=domains_config,
                default_domain=self._hyde_config.default_domain,
            )
            
            self._qa_provider = QualityAssuranceProvider(self._qa_config)
            self._hallucination_detector = HallucinationDetector(self._hallucination_config)
            self._chunker = DocumentChunker(self._chunking_config)
            self._ensemble_fusion = EnsembleFusion(self._ensemble_config)
            
            # Try to get Redis client
            if self._di_container:
                self._redis_client = getattr(self._di_container, "redis", None)
            
            self._cache = RedisCacheProvider(
                config=self._cache_config,
                redis_client=self._redis_client,
            )
            
            self._session_manager = HyDESessionManager(self._session_config)
            
            self._worker_pool = HyDEWorkerPool(self._worker_config)
            if self._worker_config.enabled:
                await self._worker_pool.start()
            
            self._metrics = MetricsCollector(self._metrics_config)
            
            # Initialize pipeline
            delegator = self._get_delegator()
            self._pipeline = HyDEPipeline(
                delegator=delegator,
                classifier=self._classifier,
                qa_provider=self._qa_provider,
                hallucination_detector=self._hallucination_detector,
                chunker=self._chunker,
                ensemble_fusion=self._ensemble_fusion,
                config=self._pipeline_config,
                debug_config=self._debug_config,
            )
            
            self._initialized = True
            
            logger.info("HyDE pipeline initialized successfully")
            
            # Publish event
            if self._event_bus:
                await self._event_bus.publish(
                    "hyde.initialized",
                    {"module": "hyde_pipeline", "status": "success"},
                )
            
            return {
                "status": "initialized",
                "components": {
                    "classifier": True,
                    "qa_provider": True,
                    "hallucination_detector": True,
                    "chunker": True,
                    "ensemble_fusion": True,
                    "cache": self._cache_config.enabled,
                    "session_manager": self._session_config.enabled,
                    "worker_pool": self._worker_config.enabled,
                    "metrics": self._metrics_config.enabled,
                    "pipeline": True,
                },
            }
            
        except Exception as e:
            logger.error(f"HyDE initialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def generate_hyde(
        self,
        query: str,
        format_type: Optional[str] = None,
        domain: Optional[str] = None,
        language: Optional[str] = None,
        num_documents: int = 1,
        enable_refinement: bool = True,
        enable_ensemble: bool = False,
        max_length: Optional[int] = None,
        session_id: Optional[str] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Full HyDE pipeline execution."""
        if not self._initialized:
            await self.initialize(ctx)

        # Validate query
        _validate_query(query)

        import time
        start_time = time.perf_counter()

        # Get or create session
        if session_id:
            session = await self._session_manager.get_session(session_id)
        else:
            session = await self._session_manager.create_session()

        if not session:
            session = await self._session_manager.create_session()

        # Check cache (with normalized key)
        cache_key = _normalize_cache_key(f"{query}:{format_type}:{domain}:{max_length}")
        if self._cache_config.enabled and self._cache_config.cache_documents:
            cached = await self._cache.get("hyde", cache_key)
            if cached:
                logger.debug("[HYDE-CACHE] HIT for key: %s", cache_key[:80])
                return {**cached, "cached": True}
            else:
                logger.debug("[HYDE-CACHE] MISS for key: %s", cache_key[:80])

        # Execute pipeline
        result = await self._pipeline.execute(
            query=query,
            session_id=session.session_id,
            format_type=format_type,
            domain=domain,
            language=language,
            num_documents=num_documents,
            enable_refinement=enable_refinement,
            enable_ensemble=enable_ensemble,
            max_length=max_length or self._hyde_config.default_length,
            pipeline_config=pipeline_config,
            ctx=ctx,
        )

        # Update session
        await self._session_manager.update_session(
            session_id=session.session_id,
            query=query,
            document=result.document,
            format_type=result.document.format_type,
            domain=result.document.domain,
        )

        # Record metrics
        if self._metrics_config.enabled:
            await self._metrics.record_generation(
                format_type=result.document.format_type,
                domain=result.document.domain,
                quality_score=result.document.quality_score,
                execution_time_ms=result.time_ms,
                hallucination_detected=result.hallucination_check.hallucination_detected if result.hallucination_check else False,
                refinement_applied=result.refinement_applied,
                ensemble_used=enable_ensemble,
            )

        response = result.to_dict()
        # Add flat string for compatibility with enrichment_pipeline consumers
        response["hypothetical_document"] = result.document.content

        # Cache result
        if self._cache_config.enabled and self._cache_config.cache_documents:
            await self._cache.set("hyde", cache_key, response)

        # Publish event
        if self._event_bus:
            await self._event_bus.publish(
                "hyde.pipeline.completed",
                {
                    "session_id": session.session_id,
                    "document_id": result.document.document_id,
                    "format_type": result.document.format_type,
                    "quality_score": result.document.quality_score,
                    "time_ms": result.time_ms,
                },
            )

        return response
    
    async def generate_document(
        self,
        query: str,
        format_type: str = "answer",
        domain: str = "auto",
        language: str = "auto",
        min_length: int = 100,
        max_length: int = 400,
        temperature: Optional[float] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Direct document generation (no pipeline)."""
        if not self._initialized:
            await self.initialize(ctx)

        # Validate query
        _validate_query(query)

        delegator = self._get_delegator()
        result = await delegator.generate_document(
            query=query,
            format_type=format_type,
            domain=domain,
            language=language,
            min_length=min_length,
            max_length=max_length,
            temperature=temperature,
        )

        response = result.to_dict()
        # Add flat string for compatibility with enrichment_pipeline consumers
        response["hypothetical_document"] = result.document.content
        return response
    
    async def generate_ensemble(
        self,
        query: str,
        count: int = 3,
        formats: Optional[List[str]] = None,
        domain: str = "auto",
        language: str = "auto",
        fuse_results: bool = True,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Ensemble document generation."""
        if not self._initialized:
            await self.initialize(ctx)

        # Validate query
        _validate_query(query)

        delegator = self._get_delegator()
        result = await delegator.generate_ensemble(
            query=query,
            count=min(count, self._ensemble_config.max_count),
            formats=formats,
            domain=domain,
            language=language,
        )
        
        response = result.to_dict()
        
        # Fuse if requested
        if fuse_results and len(result.documents) > 1:
            fused = self._ensemble_fusion.fuse(result.documents)
            if fused:
                response["fused_document"] = fused.to_dict()
        
        return response
    
    async def classify_query(self, query: str, ctx: Any = None) -> Dict[str, Any]:
        """Classify query for domain and language."""
        if not self._initialized:
            await self.initialize(ctx)

        # Validate query
        _validate_query(query)

        classification = self._classifier.classify(query)
        return classification.to_dict()
    
    async def assess_quality(
        self,
        document: Union[Dict[str, Any], str],
        original_query: str,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Assess document quality."""
        if not self._initialized:
            await self.initialize(ctx)

        doc = _coerce_document_input(document)

        assessment = self._qa_provider.assess(doc, original_query)
        return assessment.to_dict()
    
    async def check_hallucination(
        self,
        document: Union[Dict[str, Any], str],
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Check document for hallucinations."""
        if not self._initialized:
            await self.initialize(ctx)

        doc = _coerce_document_input(document)

        check = self._hallucination_detector.check(doc)
        return check.to_dict()
    
    async def refine_document(
        self,
        document: Union[Dict[str, Any], str],
        strategy: str = "expand",
        quality_score: float = 0.0,
        issues: Optional[List[str]] = None,
        query: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Refine a document.

        Args:
            document: Document dict or string to refine
            strategy: Refinement strategy
            quality_score: Current quality score
            issues: Identified quality issues
            query: Original query (required if not present in document)
        """
        if not self._initialized:
            await self.initialize(ctx)

        doc = _coerce_document_input(document)

        # Set query from parameter if document.query is empty
        if query and not doc.query:
            doc.query = query

        # Validate that query is present (needed for refinement prompt)
        if not doc.query:
            raise ValueError(
                "Query is required for document refinement. Provide it as a "
                "'query' parameter or include it in the document dict."
            )

        delegator = self._get_delegator()
        result = await delegator.refine_document(
            document=doc,
            strategy=strategy,
            quality_score=quality_score,
            issues=issues or [],
        )

        return result.to_dict()
    
    async def chunk_document(
        self,
        document: Union[Dict[str, Any], str],
        strategy: Optional[str] = None,
        chunk_size: Optional[int] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Chunk a document for embedding."""
        if not self._initialized:
            await self.initialize(ctx)

        doc = _coerce_document_input(document)
        
        # Apply overrides
        if strategy:
            self._chunker.config.strategy = strategy
        if chunk_size:
            self._chunker.config.chunk_size = chunk_size
        
        chunks = self._chunker.chunk(doc)
        
        return {
            "document_id": doc.document_id,
            "chunks": [c.to_dict() for c in chunks],
            "chunk_count": len(chunks),
            "strategy": self._chunker.config.strategy,
        }
    
    async def fuse_documents(
        self,
        documents: List[Union[Dict[str, Any], str]],
        strategy: str = "weighted_concat",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Fuse multiple documents."""
        if not self._initialized:
            await self.initialize(ctx)

        docs = [_coerce_document_input(d) for d in documents]
        fused = self._ensemble_fusion.fuse(docs, strategy=strategy)
        
        if fused:
            return fused.to_dict()
        return {"error": "No documents to fuse"}
    
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
        # Check admin permission
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
        # Check admin permission
        if ctx and hasattr(ctx, "user") and hasattr(ctx.user, "is_admin"):
            if not ctx.user.is_admin:
                return {"error": "Admin access required"}
        
        if not self._initialized:
            await self.initialize(ctx)
        
        return self._pipeline.update_config(steps) if self._pipeline else {}
    
    async def reload_config(self, ctx: Any = None) -> Dict[str, Any]:
        """Hot-reload configuration (admin only)."""
        # Check admin permission
        if ctx and hasattr(ctx, "user") and hasattr(ctx.user, "is_admin"):
            if not ctx.user.is_admin:
                return {"error": "Admin access required"}
        
        try:
            self._config = self._load_config()
            
            # Rebuild configs
            self._hyde_config = self._build_hyde_config()
            self._qa_config = self._build_qa_config()
            self._ensemble_config = self._build_ensemble_config()
            self._debug_config = self._build_debug_config()
            self._pipeline_config = self._build_pipeline_config()
            
            # Update providers
            if self._qa_provider:
                self._qa_provider.config = self._qa_config
            
            if self._pipeline:
                self._pipeline.config = self._pipeline_config
            
            return {"status": "reloaded", "timestamp": __import__("datetime").datetime.utcnow().isoformat()}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def shutdown(self, ctx: Any = None) -> Dict[str, Any]:
        """Graceful shutdown."""
        try:
            # Stop worker pool
            if self._worker_pool:
                await self._worker_pool.stop()
            
            # Clear cache
            if self._cache:
                await self._cache.clear()
            
            self._initialized = False
            
            # Publish event
            if self._event_bus:
                await self._event_bus.publish(
                    "hyde.shutdown",
                    {"module": "hyde_pipeline"},
                )
            
            logger.info("HyDE pipeline shut down")
            return {"status": "shutdown"}
            
        except Exception as e:
            logger.error(f"HyDE shutdown error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def health_check(self, ctx: Any = None) -> Dict[str, Any]:
        """Check component health."""
        if not self._initialized:
            return {
                "module": "hyde_pipeline",
                "status": "not_initialized",
            }
        
        # Check LLM delegation
        delegator = self._get_delegator()
        llm_health = await delegator.health_check()
        
        # Check worker pool
        worker_stats = self._worker_pool.get_stats() if self._worker_pool else None
        
        # Check cache
        cache_stats = self._cache.get_stats() if self._cache else None
        
        # Determine overall status
        status = "healthy"
        if llm_health.get("status") != "available":
            status = "degraded"
        
        return {
            "module": "hyde_pipeline",
            "status": status,
            "initialized": self._initialized,
            "llm_delegation": llm_health,
            "worker_pool": worker_stats.to_dict() if worker_stats else None,
            "cache": cache_stats,
            "components": {
                "classifier": self._classifier is not None,
                "qa_provider": self._qa_provider is not None,
                "hallucination_detector": self._hallucination_detector is not None,
                "chunker": self._chunker is not None,
                "ensemble_fusion": self._ensemble_fusion is not None,
                "session_manager": self._session_manager is not None,
                "metrics": self._metrics is not None,
                "pipeline": self._pipeline is not None,
            },
        }
    
    async def get_available_formats(self, ctx: Any = None) -> Dict[str, Any]:
        """Get list of available document formats."""
        from .prompts import get_all_formats
        formats = get_all_formats()
        return {"formats": formats, "count": len(formats)}
    
    async def get_available_domains(self, ctx: Any = None) -> Dict[str, Any]:
        """Get list of available domains."""
        from .prompts import get_all_domains, DOMAINS
        domains = get_all_domains()
        return {
            "domains": domains,
            "domain_info": {
                name: {
                    "description": info.get("description", ""),
                    "preferred_formats": info.get("preferred_formats", []),
                }
                for name, info in DOMAINS.items()
            },
        }
