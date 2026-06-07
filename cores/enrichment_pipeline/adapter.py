"""
enrichment_pipeline/adapter.py

Bridge Layer - Exposes all module operations.
Orchestrates providers, delegation, and pipeline execution.

This is the main entry point for the module.

v3.7.1 FIX-PERF-1:
- Parallelized pre-retrieval LLM calls (HyDE, expansion, investigative)
- Reduces sequential wait time from 15-30s to max(individual calls)

MCP-COMPAT (ARCH-008): Added OperationContext support for dual REST/MCP compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import asyncio  # FIX-PERF-1 v3.7.1: Added for parallel execution
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .providers import (
    RerankerProvider,
    RerankerConfig,
    MedicalRerankerConfig,
    ContextCompressor,
    CompressionConfig,
    ChunkFusion,
    FusionConfig,
    Deduplicator,
    DeduplicationConfig,
    MetadataInjector,
    EnrichedChunk,
    RedisCacheProvider,
    CacheConfig,
)
from .delegation import (
    LLMDelegator,
    LLMDelegationConfig,
)
from .pipeline import (
    EnrichmentPipeline,
    PipelineConfig,
    PipelineResult,
)

# v1.10.0: Role-Based Configuration Engine
from ubp_enterprise_hybrid.modules.cores._shared import ProviderMapper, ProviderConfigurationError

# FIX-TYPE-001 v2.2.4: Import type coercion for config safety
from ubp_enterprise_hybrid.modules.cores._shared.manifest_loader import coerce_config_types

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    from _shared.operation_context import OperationContext

logger = logging.getLogger(__name__)


# ============================================================================
# DI Container Adapter for Module Registry
# ============================================================================


class DIContainerModuleRegistry:
    """
    Adapter that wraps the DI container to provide IModuleRegistry interface.

    This allows enrichment_pipeline to resolve LLM modules via the standard
    DI container without requiring backend.app.infra.interfaces.
    """

    def __init__(self, di_container: Any):
        self._di_container = di_container
        self._module_cache: Dict[str, Any] = {}

    def is_module_loaded(self, module_name: str) -> bool:
        """Check if a module is registered in the DI container."""
        try:
            # Try to resolve - if it works, the module is loaded
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, can't call sync resolve
                # Check cache or assume loaded if we resolved it before
                return module_name in self._module_cache
            return True  # Assume loaded, will fail on get_module if not
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


def _build_reranker_config(config: Dict[str, Any]) -> RerankerConfig:
    """Build RerankerConfig from config dict."""
    reranker_cfg = config.get("reranker", {})
    return RerankerConfig(
        model=reranker_cfg.get("model", "BAAI/bge-reranker-v2-m3"),
        device=reranker_cfg.get("device", "cuda"),
        batch_size=int(reranker_cfg.get("batch_size", 32)),
        max_length=int(reranker_cfg.get("max_length", 512)),
        normalize_scores=str(reranker_cfg.get("normalize_scores", "true")).lower()
        == "true",
        cache_model=str(reranker_cfg.get("cache_model", "true")).lower() == "true",
    )


def _build_medical_reranker_config(config: Dict[str, Any]) -> RerankerConfig:
    """Build RerankerConfig for medical reranker from config dict."""
    med_cfg = config.get("medical_reranker", {})
    return RerankerConfig(
        model=med_cfg.get("model", "ncbi/MedCPT-Cross-Encoder"),
        device=med_cfg.get("device", "auto"),
        batch_size=int(med_cfg.get("batch_size", 64)),
        max_length=int(med_cfg.get("max_length", 512)),
        normalize_scores=str(med_cfg.get("normalize_scores", "true")).lower()
        == "true",
        cache_model=str(med_cfg.get("cache_model", "true")).lower() == "true",
    )


def _build_compression_config(config: Dict[str, Any]) -> CompressionConfig:
    """Build CompressionConfig from config dict."""
    comp_cfg = config.get("compression", {})
    return CompressionConfig(
        default_ratio=float(comp_cfg.get("default_ratio", 0.5)),
        method=comp_cfg.get("method", "extractive"),
        min_chunk_length=int(comp_cfg.get("min_chunk_length", 50)),
        preserve_sentences=str(comp_cfg.get("preserve_sentences", "true")).lower()
        == "true",
    )


def _build_fusion_config(config: Dict[str, Any]) -> FusionConfig:
    """Build FusionConfig from config dict."""
    fusion_cfg = config.get("fusion", {})
    return FusionConfig(
        overlap_threshold=float(fusion_cfg.get("overlap_threshold", 0.3)),
        semantic_threshold=float(fusion_cfg.get("semantic_threshold", 0.93)),
        max_fused_length=int(fusion_cfg.get("max_fused_length", 1000)),
        strategies=fusion_cfg.get("strategies", ["overlap", "adjacent", "semantic"]),
    )


def _build_dedup_config(config: Dict[str, Any]) -> DeduplicationConfig:
    """Build DeduplicationConfig from config dict."""
    dedup_cfg = config.get("deduplication", {})
    return DeduplicationConfig(
        similarity_threshold=float(dedup_cfg.get("similarity_threshold", 0.95)),
        method=dedup_cfg.get("method", "semantic"),
    )


def _build_delegation_config(config: Dict[str, Any]) -> LLMDelegationConfig:
    """Build LLMDelegationConfig from config dict."""
    del_cfg = config.get("delegation", {})
    return LLMDelegationConfig(
        llm_module=del_cfg.get("llm_module", "inference_vllm"),
        timeout_seconds=int(del_cfg.get("timeout_seconds", 30)),
    )


def _build_pipeline_config(config: Dict[str, Any]) -> PipelineConfig:
    """Build PipelineConfig from config dict."""
    pipeline_cfg = config.get("pipeline", {})
    return PipelineConfig.from_dict(pipeline_cfg)


def _build_cache_config(config: Dict[str, Any], env: str) -> CacheConfig:
    """Build CacheConfig from config dict with environment isolation."""
    cache_cfg = config.get("cache", {})
    return CacheConfig(
        enabled=str(cache_cfg.get("enabled", "true")).lower() == "true",
        ttl_seconds=int(cache_cfg.get("ttl_seconds", 3600)),
        base_prefix="ubp",
        env=env,  # Environment for isolation (dev/test/prod)
        cache_rerank=str(cache_cfg.get("cache_rerank", "true")).lower() == "true",
        cache_hyde=str(cache_cfg.get("cache_hyde", "true")).lower() == "true",
    )


# ============================================================================
# EnrichmentPipelineAdapter
# ============================================================================


class EnrichmentPipelineAdapter:
    """
    Main adapter for enrichment_pipeline module.

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
        self.publisher = event_bus  # Alias for event publishing

        # Load configuration
        self.config = _load_config(module_path)

        # Get environment for cache isolation (dev/test/prod)
        self.env = os.environ.get("UBP_ENV", "dev")

        # Build component configs
        self.reranker_config = _build_reranker_config(self.config)
        self.medical_reranker_config = _build_medical_reranker_config(self.config)
        self.compression_config = _build_compression_config(self.config)
        self.fusion_config = _build_fusion_config(self.config)
        self.dedup_config = _build_dedup_config(self.config)
        self.delegation_config = _build_delegation_config(self.config)
        self.pipeline_config = _build_pipeline_config(self.config)
        self.cache_config = _build_cache_config(self.config, self.env)

        # Components (initialized in initialize())
        self._reranker: Optional[RerankerProvider] = None
        self._medical_reranker: Optional[RerankerProvider] = None
        self._compressor: Optional[ContextCompressor] = None
        self._fusion: Optional[ChunkFusion] = None
        self._deduplicator: Optional[Deduplicator] = None
        self._metadata_injector: Optional[MetadataInjector] = None
        self._llm_delegator: Optional[LLMDelegator] = None
        self._pipeline: Optional[EnrichmentPipeline] = None
        self._cache: Optional[RedisCacheProvider] = None

        # State
        self._initialized = False
        self._module_registry: Optional[Any] = None

    # ========================================================================
    # MCP-COMPAT: OperationContext helpers (ARCH-008)
    # ========================================================================

    def _build_context_from_di(self) -> OperationContext:
        """
        Build OperationContext from DI container — backward compatibility for REST path.
        
        MCP-COMPAT: When ctx is not provided (REST path), this method constructs
        an OperationContext from the DI container state.
        
        Returns:
            OperationContext with default values
        """
        return OperationContext(
            client_id="default",
            user_id=None,
            session_id=None,
            source="rest",
        )

    def _normalize_ctx(self, ctx: Any) -> OperationContext:
        """
        Normalize any context format to OperationContext.
        
        MCP-COMPAT: Handles both legacy security context (ctx.user.user_id) 
        and new OperationContext format for backward compatibility.
        
        Args:
            ctx: Either OperationContext, legacy security context, or None
            
        Returns:
            OperationContext instance
        """
        if ctx is None:
            return self._build_context_from_di()
        
        # Already an OperationContext
        if isinstance(ctx, OperationContext):
            return ctx
        
        # Legacy security context format (ctx.user.user_id, ctx.user.roles)
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
        
        # Fallback
        return self._build_context_from_di()

    # ========================================================================
    # Lifecycle Operations
    # ========================================================================

    async def initialize(
        self,
        preload_models: bool = True,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Initialize enrichment pipeline and load models."""
        start_time = time.perf_counter()

        try:
            # Resolve module registry and Redis from DI
            redis_client = None
            if self.di_container:
                # Create adapter wrapper for module registry (LLM resolved lazily)
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

            # Initialize reranker — model loaded via SharedModelPool for GPU dedup
            self._reranker = RerankerProvider(self.reranker_config)
            if preload_models:
                from ubp_enterprise_hybrid.modules.cores._shared.model_pool import SharedModelPool
                reranker_model = SharedModelPool.get_cross_encoder(
                    model_name=self.reranker_config.model,
                    device=self.reranker_config.device,
                    max_length=self.reranker_config.max_length,
                )
                self._reranker.set_shared_model(reranker_model)

            # Initialize medical reranker if enabled
            medical_cfg = self.config.get("medical_reranker", {})
            if str(medical_cfg.get("enabled", "false")).lower() == "true":
                self._medical_reranker = RerankerProvider(self.medical_reranker_config)
                if preload_models:
                    from ubp_enterprise_hybrid.modules.cores._shared.model_pool import SharedModelPool
                    med_model = SharedModelPool.get_cross_encoder(
                        model_name=self.medical_reranker_config.model,
                        device=self.medical_reranker_config.device,
                        max_length=self.medical_reranker_config.max_length,
                    )
                    self._medical_reranker.set_shared_model(med_model)
                logger.info(f"Medical reranker loaded: {self.medical_reranker_config.model}")
            else:
                logger.info("Medical reranker disabled (UBP_ENRICHMENT__MEDICAL_RERANK_ENABLED=false)")

            # Initialize other providers
            self._compressor = ContextCompressor(self.compression_config)
            self._fusion = ChunkFusion(self.fusion_config)
            self._deduplicator = Deduplicator(self.dedup_config)
            self._metadata_injector = MetadataInjector()

            # Initialize LLM delegator if module registry available
            if self._module_registry:
                self._llm_delegator = LLMDelegator(
                    self.delegation_config,
                    self._module_registry,
                    event_publisher=self.publisher,
                )

            # Initialize pipeline
            self._pipeline = EnrichmentPipeline(
                reranker=self._reranker,
                compressor=self._compressor,
                fusion=self._fusion,
                deduplicator=self._deduplicator,
                metadata_injector=self._metadata_injector,
                llm_delegator=self._llm_delegator,
                config=self.pipeline_config,
            )

            self._initialized = True
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return {
                "status": "initialized",
                "module": "enrichment_pipeline",
                "env": self.env,
                "reranker_loaded": self._reranker.is_loaded
                if self._reranker
                else False,
                "reranker_model": self.reranker_config.model,
                "device": self.reranker_config.device,
                "medical_reranker_loaded": self._medical_reranker.is_loaded
                if self._medical_reranker
                else False,
                "medical_reranker_model": self.medical_reranker_config.model
                if self._medical_reranker
                else None,
                "pipeline_steps_available": EnrichmentPipeline.AVAILABLE_STEPS,
                "llm_delegation_available": self._llm_delegator.is_available()
                if self._llm_delegator
                else False,
                "cache_enabled": self.cache_config.enabled,
                "cache_prefix": self.cache_config.prefix,
                "elapsed_ms": round(elapsed_ms, 2),
            }

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return {
                "status": "error",
                "module": "enrichment_pipeline",
                "error": str(e),
            }

    async def shutdown(self, ctx=None, **kwargs) -> Dict[str, Any]:
        """Graceful shutdown and model unload."""
        resources_released = []

        if self._reranker and self._reranker.is_loaded:
            self._reranker.unload_model()
            resources_released.append("reranker_model")

        if self._medical_reranker and self._medical_reranker.is_loaded:
            self._medical_reranker.unload_model()
            resources_released.append("medical_reranker_model")

        self._initialized = False

        return {
            "status": "shutdown",
            "resources_released": resources_released,
        }

    async def health_check(self, ctx=None, **kwargs) -> Dict[str, Any]:
        """Health check for enrichment module."""
        result: Dict[str, Any] = {
            "module": "enrichment_pipeline",
            "status": "healthy",
            "env": self.env,
        }

        # Check reranker
        if self._reranker:
            result["reranker"] = self._reranker.health_check()
            if result["reranker"].get("status") != "healthy":
                result["status"] = "degraded"
        else:
            result["reranker"] = {"status": "not_initialized"}
            result["status"] = "unhealthy"

        # Check medical reranker
        if self._medical_reranker:
            result["medical_reranker"] = self._medical_reranker.health_check()
        else:
            result["medical_reranker"] = {"status": "disabled"}

        # Check LLM delegation
        if not self._llm_delegator:
            await self._get_llm_delegator()

        if self._llm_delegator:
            result["llm_delegation"] = await self._llm_delegator.health_check()
        else:
            result["llm_delegation"] = {"status": "not_configured"}

        # Cache stats with environment info
        if self._cache:
            result["cache"] = self._cache.get_stats()
        else:
            result["cache"] = {"status": "disabled", "hit_rate": 0}

        return result

    async def _get_llm_delegator(self) -> Optional[LLMDelegator]:
        """
        Lazily resolve the LLM delegator to avoid DI race conditions.

        v3.4.1: Added provider change detection to support hot-reload of
        enrichment provider override. Checks if configured provider has changed
        before returning cached delegator.

        TODO v3.5.0: Migrate to event-driven invalidation (Option B) for better
        performance. Current approach (Option A) has minimal overhead but checks
        provider on every call. Event-driven approach would invalidate cache only
        when settings.override.changed event is published.
        """
        # v3.4.1: Check if provider has changed (supports hot-reload override)
        # v3.6.1: Now also checks internal provider change (e.g., ollama -> grok)
        try:
            current_chain = ProviderMapper.resolve_chain("enrichment")
            expected_module = current_chain[0][0] if current_chain else None
            expected_provider = current_chain[0][1] if current_chain else None
        except ProviderConfigurationError:
            expected_module = None
            expected_provider = None

        # Fast path: only if delegator exists AND module+provider have NOT changed
        if self._llm_delegator and self._llm_delegator.is_available():
            current_module = self._llm_delegator.config.llm_module
            current_provider = self._llm_delegator.config.provider

            # v3.6.1: Check both module AND provider for changes
            if current_module == expected_module and current_provider == expected_provider:
                return self._llm_delegator
            else:
                # Provider changed via override - invalidate and recreate
                logger.warning(
                    f"[ENRICHMENT] Provider changed via override: "
                    f"{current_module}/{current_provider} → {expected_module}/{expected_provider}. "
                    f"Recreating delegator."
                )
                self._llm_delegator = None

        if not self.di_container:
            logger.warning("LLM delegation requested but DI container is not available")
            return None

        if not self._module_registry:
            self._module_registry = DIContainerModuleRegistry(self.di_container)

        try:
            provider_chain = ProviderMapper.resolve_chain("enrichment")
        except ProviderConfigurationError as exc:
            logger.error(f"[ENRICHMENT] Provider configuration error: {exc}")
            return None

        if not provider_chain:
            logger.warning("[ENRICHMENT] No providers configured for role 'enrichment'")
            return None

        resolved_module_name: Optional[str] = None
        resolved_provider_name: Optional[str] = None

        for module_name, provider_name in provider_chain:
            resolved_llm = await self._module_registry.resolve_module(module_name)
            if not resolved_llm:
                logger.warning(
                    f"[ENRICHMENT] Module '{module_name}' not ready yet, trying next provider"
                )
                continue

            if hasattr(resolved_llm, "set_default_provider"):
                try:
                    resolved_llm.set_default_provider(provider_name)
                except Exception as exc:
                    logger.warning(
                        f"[ENRICHMENT] Failed to configure provider '{provider_name}' for '{module_name}': {exc}"
                    )

            resolved_module_name = module_name
            resolved_provider_name = provider_name
            break

        if not resolved_module_name:
            logger.error(
                "[ENRICHMENT] Unable to resolve any LLM module for enrichment role"
            )
            return None

        # Preserve existing delegation config while updating module binding
        # v3.6.1: Now includes explicit provider to ensure correct provider is used
        # even when module instance is shared across multiple callers
        # v6.0.1: Model resolution removed — inference module resolves from ProviderInventory
        current_delegate_cfg = self.delegation_config
        self.delegation_config = LLMDelegationConfig(
            llm_module=resolved_module_name,
            llm_operation=current_delegate_cfg.llm_operation,
            timeout_seconds=current_delegate_cfg.timeout_seconds,
            max_retries=current_delegate_cfg.max_retries,
            provider=resolved_provider_name,  # v3.6.1: Explicit provider
        )

        self._llm_delegator = LLMDelegator(
            self.delegation_config,
            self._module_registry,
            event_publisher=self.publisher,
        )

        if self._pipeline:
            self._pipeline.llm_delegator = self._llm_delegator

        logger.info(
            f"✅ Enrichment LLM linked to module '{resolved_module_name}' "
            f"with explicit provider '{resolved_provider_name}'"
        )

        return self._llm_delegator

    async def get_stats(
        self,
        period: str = "24h",
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get enrichment statistics."""
        # TODO: Implement actual stats collection
        return {
            "module": "enrichment_pipeline",
            "period": period,
            "requests": {"total": 0, "by_operation": {}},
            "latency": {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0},
            "rerank_stats": {"total": 0, "avg_chunks": 0, "avg_time_ms": 0},
            "cache_stats": {"hits": 0, "misses": 0, "hit_rate": 0},
        }

    # ========================================================================
    # Input Normalization
    # ========================================================================

    @staticmethod
    def _normalize_chunks(chunks: List) -> List[Dict[str, Any]]:
        """Normalize chunks to list[dict]. Accepts list[str] or list[dict]."""
        return [
            chunk if isinstance(chunk, dict) else {"text": str(chunk)}
            for chunk in chunks
        ]

    # ========================================================================
    # Main Pipeline Operation
    # ========================================================================

    async def enrich_context(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        pipeline_config: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Full enrichment pipeline execution."""
        chunks = self._normalize_chunks(chunks)
        if not self._pipeline:
            raise RuntimeError("Module not initialized")

        # Ensure LLM delegator is linked before executing steps that depend on it
        await self._get_llm_delegator()

        result = await self._pipeline.execute(
            query=query,
            chunks=chunks,
            pipeline_config=pipeline_config,
            chat_history=chat_history,
            top_k=top_k,
            ctx=ctx,
        )

        return result.to_dict()

    # ========================================================================
    # Individual Operations
    # ========================================================================

    async def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 10,
        return_scores: bool = True,
        reranker_type: str = "primary",
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Rerank chunks using cross-encoder model.

        Args:
            reranker_type: "primary" (bge-reranker), "medical" (MedCPT),
                          or "cascade" (primary 2x top_k, then medical re-rank)
        """
        chunks = self._normalize_chunks(chunks)
        if not self._reranker:
            raise RuntimeError("Module not initialized")

        if reranker_type == "medical":
            # Use medical reranker, fallback to primary if not loaded
            reranker = self._medical_reranker if self._medical_reranker else self._reranker
            if reranker != self._medical_reranker:
                logger.warning("[RERANK] Medical reranker not loaded, falling back to primary")
            result = reranker.rerank(query=query, chunks=chunks, top_k=top_k)

        elif reranker_type == "cascade":
            # Stage 1: primary reranker with 2x top_k
            cascade_top_k = min(top_k * 2, len(chunks))
            stage1 = self._reranker.rerank(query=query, chunks=chunks, top_k=cascade_top_k)

            # Stage 2: medical reranker on stage1 results
            if self._medical_reranker and stage1.reranked_chunks:
                stage1_dicts = [c.to_dict() for c in stage1.reranked_chunks]
                result = self._medical_reranker.rerank(
                    query=query, chunks=stage1_dicts, top_k=top_k,
                )
                # Annotate that cascade was used
                result_dict = result.to_dict()
                result_dict["reranker_type"] = "cascade"
                result_dict["cascade_stage1_model"] = self._reranker.config.model
                result_dict["cascade_stage2_model"] = self._medical_reranker.config.model
                return result_dict
            else:
                # No medical reranker, just use primary result with top_k
                if len(stage1.reranked_chunks) > top_k:
                    stage1.reranked_chunks = stage1.reranked_chunks[:top_k]
                result = stage1

        else:
            # Default: primary reranker
            result = self._reranker.rerank(query=query, chunks=chunks, top_k=top_k)

        result_dict = result.to_dict()
        result_dict["reranker_type"] = reranker_type
        return result_dict

    async def expand_query(
        self,
        query: str,
        num_variants: int = 3,
        chat_history: Optional[List[Dict[str, str]]] = None,
        expansion_type: str = "semantic",
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate query variations using LLM."""
        delegator = await self._get_llm_delegator()
        if not delegator:
            raise RuntimeError("LLM delegation not configured")

        result = await delegator.expand_query(
            query=query,
            num_variants=num_variants,
            chat_history=chat_history,
            expansion_type=expansion_type,
        )

        return result.to_dict()

    async def generate_hyde(
        self,
        query: str,
        document_type: str = "auto",
        format_type: Optional[str] = None,
        max_length: int = 300,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate Hypothetical Document Embedding (HyDE).

        Args:
            query: User's query
            document_type: Document type (backward-compatible)
            format_type: Alias for document_type (hyde_pipeline convention)
            max_length: Maximum document length
        """
        delegator = await self._get_llm_delegator()
        if not delegator:
            raise RuntimeError("LLM delegation not configured")

        # format_type takes precedence as alias for document_type
        effective_type = format_type or document_type

        result = await delegator.generate_hyde(
            query=query,
            document_type=effective_type,
            max_length=max_length,
        )

        response = result.to_dict()
        # Include both naming conventions in response
        response["format_type"] = effective_type
        return response

    async def generate_investigative(
        self,
        query: str,
        num_questions: int = 5,
        investigation_type: str = "auto",
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate investigative search questions (v2.2.2 - FEAT-INVEST-001).

        Instead of generating a hypothetical answer (HyDE), this generates
        specific search questions that would lead to finding the answer.
        This approach avoids hallucination on unknown terms.

        Args:
            query: User's original query
            num_questions: Number of questions to generate (default: 5)
            ctx: Security context

        Returns:
            Dict with:
                - investigative_questions: List[str] - Generated search questions
                - original_query: str - The original query
                - time_ms: float - Processing time
        """
        delegator = await self._get_llm_delegator()
        if not delegator:
            raise RuntimeError("LLM delegation not configured")

        result = await delegator.generate_investigative(
            query=query,
            num_questions=num_questions,
            investigation_type=investigation_type,
        )

        return result.to_dict()

    async def optimize_query(
        self,
        query: str,
        hyde_enabled: bool = True,
        expansion_enabled: bool = True,
        investigative_enabled: bool = False,
        num_variants: int = 3,
        num_questions: int = 5,
        hyde_document_type: str = "answer",
        rewrite_focus: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Optimize query for retrieval using HyDE, Query Expansion, and/or Investigative Decomposition.

        FEAT-ARCH-2.2: Advanced Retrieval Query Optimization
        FEAT-INVEST-001 (v2.2.2): Investigative Query Decomposition

        This method generates a list of optimized search queries that can be
        used for multi-retrieval, improving recall for vague or ambiguous queries.

        Args:
            query: Original user query
            hyde_enabled: Generate hypothetical document (HyDE)
            expansion_enabled: Generate query variants
            investigative_enabled: Generate investigative search questions (v2.2.2)
            num_variants: Number of expansion variants to generate
            num_questions: Number of investigative questions to generate
            hyde_document_type: Type of HyDE document (answer, technical, faq, etc.)
            ctx: Security context

        Returns:
            Dict with:
                - search_queries: List[str] - All queries for retrieval
                - original_query: str - The original query
                - hyde_document: str | None - Generated HyDE document
                - expanded_queries: List[str] - Expansion variants
                - investigative_questions: List[str] - Investigative questions (v2.2.2)
                - optimization_applied: List[str] - Which optimizations were used
                - time_ms: float - Total processing time
        """
        import time

        start_time = time.perf_counter()

        # LOG RECEIVED PARAMETERS - Critical for debugging configuration propagation
        logger.info(
            f"[ENRICHMENT] optimize_query called with flags: "
            f"hyde_enabled={hyde_enabled}, expansion_enabled={expansion_enabled}, "
            f"investigative_enabled={investigative_enabled}",
            extra={
                "query": query[:100],
                "hyde_enabled": hyde_enabled,
                "expansion_enabled": expansion_enabled,
                "investigative_enabled": investigative_enabled,
                "num_variants": num_variants,
                "num_questions": num_questions,
            },
        )

        delegator = await self._get_llm_delegator()

        search_queries = [query]  # Always include original
        hyde_document = None
        expanded_queries = []
        investigative_questions = []  # v2.2.2
        optimization_applied = []
        # v6.0.2: Per-task timing for debug panels
        hyde_time_ms: Optional[float] = None
        expansion_time_ms: Optional[float] = None
        investigative_time_ms: Optional[float] = None

        # FIX-PERF-1 v3.7.1: Parallelize pre-retrieval LLM calls
        # Sequential execution was taking 15-30s (hyde + expansion + investigative)
        # Parallel execution reduces to max(individual call times) ~10s with Grok
        # or ~4-5s with vLLM
        
        if delegator and delegator.is_available():
            tasks = []
            task_names = []  # Positional tracking (same order as tasks)

            # 1. HyDE Generation (parallel)
            if hyde_enabled:
                tasks.append(delegator.generate_hyde(
                    query=query,
                    document_type=hyde_document_type,
                ))
                task_names.append("hyde")

            # 2. Query Expansion (parallel)
            if expansion_enabled:
                tasks.append(delegator.expand_query(
                    query=query,
                    num_variants=num_variants,
                ))
                task_names.append("expansion")

            # 3. Investigative Decomposition (parallel, v2.2.2 - FEAT-INVEST-001)
            # v5.0: Use rewrite_focus (current_focus from memory) for investigation
            # when query was rewritten. For vague queries like "entra nei dettagli",
            # neither the raw query nor the keyword-stuffed retrieval_query are useful.
            # The current_focus ("Redis override mechanism") gives investigation
            # the topic context it needs to generate meaningful sub-questions.
            if investigative_enabled:
                investigation_query = rewrite_focus if rewrite_focus else query
                tasks.append(delegator.generate_investigative(
                    query=investigation_query,
                    num_questions=num_questions,
                ))
                task_names.append("investigative")

            # Execute all tasks in parallel
            if tasks:
                logger.info(f"[ENRICHMENT-PARALLEL] Executing {len(tasks)} LLM tasks in parallel: {task_names}")
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results (positional: tasks and results have same order)
                for task_type, result in zip(task_names, results):
                    
                    if isinstance(result, Exception):
                        logger.warning(f"[ENRICHMENT] {task_type} failed: {result}")
                        continue
                    
                    # Handle HyDE result
                    if task_type == "hyde" and result:
                        hyde_time_ms = getattr(result, 'time_ms', None)
                        hyde_doc = getattr(result, 'hypothetical_document', None)
                        if hyde_doc and len(hyde_doc) > 20:  # Validate non-empty
                            hyde_document = hyde_doc
                            search_queries.append(hyde_doc)
                            optimization_applied.append("hyde")
                            logger.info(
                                f"[ENRICHMENT] Generated HyDE document ({len(hyde_doc)} chars)",
                                extra={"query": query[:50], "hyde_length": len(hyde_doc)},
                            )

                    # Handle Expansion result
                    elif task_type == "expansion" and result:
                        expansion_time_ms = getattr(result, 'time_ms', None)
                        variants = getattr(result, 'expanded_queries', [])
                        if variants:
                            expanded_queries = variants
                            search_queries.extend(variants)
                            optimization_applied.append("expansion")
                            logger.info(
                                f"[ENRICHMENT] Generated {len(variants)} query variants",
                                extra={"query": query[:50], "variants": len(variants)},
                            )

                    # Handle Investigative result
                    elif task_type == "investigative" and result:
                        investigative_time_ms = getattr(result, 'time_ms', None)
                        questions = getattr(result, 'investigative_questions', [])
                        if questions:
                            investigative_questions = questions
                            search_queries.extend(questions)
                            optimization_applied.append("investigative")
                            logger.info(
                                f"[ENRICHMENT] Generated {len(questions)} investigative questions",
                                extra={"query": query[:50], "questions": len(questions)},
                            )
        else:
            # Log warnings if delegator not available
            if hyde_enabled:
                logger.warning(
                    "[ENRICHMENT] HyDE requested but LLM delegator not available",
                    extra={
                        "hyde_enabled": hyde_enabled,
                        "delegator_available": delegator is not None,
                    },
                )
            if expansion_enabled:
                logger.warning(
                    "[ENRICHMENT] Query expansion requested but LLM delegator not available",
                    extra={
                        "expansion_enabled": expansion_enabled,
                        "delegator_available": delegator is not None,
                    },
                )
            if investigative_enabled:
                logger.warning(
                    "[ENRICHMENT] Investigative requested but LLM delegator not available",
                    extra={
                        "investigative_enabled": investigative_enabled,
                        "delegator_available": delegator is not None,
                    },
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Deduplicate search queries while preserving order
        seen = set()
        unique_queries = []
        for q in search_queries:
            q_normalized = q.strip().lower()[:100]  # Normalize for comparison
            if q_normalized not in seen:
                seen.add(q_normalized)
                unique_queries.append(q)

        logger.info(
            f"[ENRICHMENT] Query optimization complete: {len(unique_queries)} search queries",
            extra={
                "original": query[:50],
                "queries_count": len(unique_queries),
                "optimizations": optimization_applied,
                "time_ms": round(elapsed_ms, 2),
            },
        )

        return {
            "search_queries": unique_queries,
            "original_query": query,
            "hyde_document": hyde_document,
            "expanded_queries": expanded_queries,
            "investigative_questions": investigative_questions,  # v2.2.2
            "optimization_applied": optimization_applied,
            "queries_count": len(unique_queries),
            "time_ms": round(elapsed_ms, 2),
            # v6.0.2: Per-task timing for debug panels
            "hyde_time_ms": hyde_time_ms,
            "expansion_time_ms": expansion_time_ms,
            "investigative_time_ms": investigative_time_ms,
        }

    async def decompose_query(
        self,
        query: str,
        max_subqueries: int = 5,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Decompose a complex query into simpler sub-queries for investigative retrieval.

        Delegates to LLM investigative generation when available.
        Falls back to returning the original query as single sub-query.

        Args:
            query: The complex query to decompose
            max_subqueries: Maximum number of sub-queries to generate
            ctx: Security context

        Returns:
            Dict with sub_queries list and original_query
        """
        delegator = await self._get_llm_delegator()
        if not delegator or not delegator.is_available():
            logger.debug(
                "[ENRICHMENT] decompose_query: LLM delegator not available, returning original query"
            )
            return {"sub_queries": [query], "original_query": query}

        try:
            result = await delegator.generate_investigative(
                query=query,
                num_questions=max_subqueries,
            )
            questions = (
                result.investigative_questions
                if hasattr(result, "investigative_questions")
                else []
            )
            if questions:
                return {
                    "sub_queries": questions,
                    "original_query": query,
                }
            return {"sub_queries": [query], "original_query": query}
        except Exception as e:
            logger.warning(f"[ENRICHMENT] decompose_query failed: {e}")
            return {"sub_queries": [query], "original_query": query}

    async def extract_filters(
        self,
        query: str,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Extract metadata filters from natural language query."""
        delegator = await self._get_llm_delegator()
        if not delegator:
            logger.debug("Filter extraction skipped (LLM unavailable)")
            return {"filters": None}

        try:
            return await delegator.extract_filters(query=query)
        except Exception as e:
            logger.warning(f"[ENRICHMENT] Filter extraction failed: {e}")
            return {"filters": None, "error": str(e)}

    async def compress_context(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        compression_ratio: float = 0.5,
        method: str = "extractive",
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compress chunks to reduce token count."""
        chunks = self._normalize_chunks(chunks)
        if not self._compressor:
            raise RuntimeError("Module not initialized")

        if method == "abstractive":
            delegator = await self._get_llm_delegator()
            if delegator and delegator.is_available():
                # Use LLM for abstractive compression
                compressed = await delegator.compress_abstractive(
                    query=query,
                    chunks=chunks,
                    target_ratio=compression_ratio,
                )

                original_tokens = sum(len(c.get("text", "")) // 4 for c in chunks)
                compressed_tokens = sum(len(c.get("text", "")) // 4 for c in compressed)

                return {
                    "compressed_chunks": compressed,
                    "original_tokens": original_tokens,
                    "compressed_tokens": compressed_tokens,
                    "actual_ratio": compressed_tokens / original_tokens
                    if original_tokens > 0
                    else 1.0,
                    "method_used": "abstractive",
                    "time_ms": 0,
                }

        result = self._compressor.compress(
            query=query,
            chunks=chunks,
            target_ratio=compression_ratio,
            method="extractive",
        )

        return result.to_dict()

    async def fuse_chunks(
        self,
        chunks: List[Dict[str, Any]],
        strategy: str = "all",
        overlap_threshold: float = 0.3,
        semantic_threshold: float = 0.93,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Merge overlapping or semantically similar chunks."""
        chunks = self._normalize_chunks(chunks)
        if not self._fusion:
            raise RuntimeError("Module not initialized")

        strategies = None
        if strategy != "all":
            strategies = [strategy]

        result = self._fusion.fuse(
            chunks=chunks,
            strategies=strategies,
        )

        return result.to_dict()

    async def deduplicate(
        self,
        chunks: List[Dict[str, Any]],
        similarity_threshold: float = 0.95,
        method: str = "semantic",
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Remove duplicate or near-duplicate chunks."""
        chunks = self._normalize_chunks(chunks)
        if not self._deduplicator:
            raise RuntimeError("Module not initialized")

        unique, removed, groups = self._deduplicator.deduplicate(
            chunks=chunks,
            method=method,
            threshold=similarity_threshold,
        )

        return {
            "unique_chunks": [c.to_dict() for c in unique],
            "duplicates_removed": removed,
            "duplicate_groups": groups,
        }

    async def inject_metadata(
        self,
        chunks: List[Dict[str, Any]],
        metadata_types: Optional[List[str]] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add enrichment metadata to chunks."""
        chunks = self._normalize_chunks(chunks)
        if not self._metadata_injector:
            raise RuntimeError("Module not initialized")

        metadata_types = metadata_types or ["source", "relevance", "position", "tokens"]

        enriched = self._metadata_injector.inject(
            chunks=chunks,
            metadata_types=metadata_types,
        )

        return {
            "enriched_chunks": [c.to_dict() for c in enriched],
            "metadata_added": metadata_types,
        }

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

    async def reload_pipeline_config(
        self,
        clear_cache: bool = True,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Hot-reload pipeline configuration from config.json without restart.

        Reloads:
        1. config.json with environment variable resolution
        2. All component configs (reranker, compression, fusion, etc.)
        3. Pipeline step configuration
        4. Optionally clears the enrichment cache

        Security: Admin only operation.

        Args:
            clear_cache: Whether to clear enrichment cache (default True)
            ctx: Security context

        Returns:
            Dict with reload status and updated configuration
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
            old_config = self.config.copy() if self.config else {}
            self.config = _load_config(self.module_path)

            reload_results["config_reloaded"] = True
            logger.info("Config.json reloaded")

            # 2. Rebuild component configs
            old_reranker_model = self.reranker_config.model
            self.reranker_config = _build_reranker_config(self.config)
            if old_reranker_model != self.reranker_config.model:
                reload_results["reranker_model_changed"] = {
                    "before": old_reranker_model,
                    "after": self.reranker_config.model,
                    "note": "Reranker model change requires shutdown() + initialize() to take effect",
                }
            reload_results["components_updated"].append("reranker_config")

            self.medical_reranker_config = _build_medical_reranker_config(self.config)
            reload_results["components_updated"].append("medical_reranker_config")

            self.compression_config = _build_compression_config(self.config)
            reload_results["components_updated"].append("compression_config")

            self.fusion_config = _build_fusion_config(self.config)
            reload_results["components_updated"].append("fusion_config")

            self.dedup_config = _build_dedup_config(self.config)
            reload_results["components_updated"].append("dedup_config")

            self.delegation_config = _build_delegation_config(self.config)
            reload_results["components_updated"].append("delegation_config")

            self.pipeline_config = _build_pipeline_config(self.config)
            reload_results["components_updated"].append("pipeline_config")

            self.cache_config = _build_cache_config(self.config, self.env)
            reload_results["components_updated"].append("cache_config")

            # 3. Update live component instances if initialized
            if self._compressor:
                self._compressor.config = self.compression_config
            if self._fusion:
                self._fusion.config = self.fusion_config
            if self._deduplicator:
                self._deduplicator.config = self.dedup_config
            if self._pipeline:
                self._pipeline.config = self.pipeline_config

            logger.info(
                f"Updated {len(reload_results['components_updated'])} component configs"
            )

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
            "module": "enrichment_pipeline",
            "operation": "reload_pipeline_config",
            "current_pipeline_config": current_config,
            "reranker_model": self.reranker_config.model,
            "device": self.reranker_config.device,
            **reload_results,
        }
