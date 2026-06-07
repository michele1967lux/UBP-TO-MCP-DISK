"""
enrichment_pipeline/pipeline.py

Pipeline orchestrator for enrichment operations.
Chains multiple enrichment steps in configurable order.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Awaitable

from .providers import (
    RerankerProvider,
    RerankerConfig,
    ContextCompressor,
    CompressionConfig,
    ChunkFusion,
    FusionConfig,
    Deduplicator,
    DeduplicationConfig,
    MetadataInjector,
    EnrichedChunk,
)
from .delegation import (
    LLMDelegator,
    LLMDelegationConfig,
    QueryExpansionResult,
    HyDEResult,
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
    enable_parallel_execution: bool = True

    @classmethod
    def default(cls) -> "PipelineConfig":
        """Create default pipeline configuration."""
        return cls(
            steps=[
                PipelineStepConfig(step="query_expansion", enabled=True),
                PipelineStepConfig(step="hyde", enabled=True),
                PipelineStepConfig(step="rerank", enabled=True),
                PipelineStepConfig(step="fusion", enabled=True),
                PipelineStepConfig(step="deduplication", enabled=True),
                PipelineStepConfig(step="compression", enabled=True),
                PipelineStepConfig(step="metadata", enabled=True),
            ],
            enable_parallel_execution=True,
        )
    
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PipelineConfig":
        """Create from dict, merging with defaults."""
        if data is None:
            return cls.default()

        def to_bool(value: Any, default: bool = True) -> bool:
            """Convert string/bool to boolean (handles env var expansion)."""
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
            enable_parallel_execution=to_bool(data.get("enable_parallel_execution", True), default=True),
        )


@dataclass
class PipelineResult:
    """Result from full pipeline execution."""
    enriched_query: Optional[str]
    enriched_chunks: List[EnrichedChunk]
    metadata_added: Dict[str, Any]
    enrichment_applied: List[str]
    stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enriched_query": self.enriched_query,
            "enriched_chunks": [c.to_dict() for c in self.enriched_chunks],
            "metadata_added": self.metadata_added,
            "enrichment_applied": self.enrichment_applied,
            "stats": self.stats,
        }


@dataclass
class StepResult:
    """Result from a single pipeline step."""
    step_name: str
    success: bool
    time_ms: float
    output: Any = None
    error: Optional[str] = None


# ============================================================================
# EnrichmentPipeline
# ============================================================================

class EnrichmentPipeline:
    """
    Orchestrates enrichment pipeline execution.
    
    Steps executed in order:
    1. Query Expansion (LLM)
    2. HyDE Generation (LLM)
    3. Reranking (BGE)
    4. Chunk Fusion
    5. Deduplication
    6. Compression
    7. Metadata Injection
    """
    
    AVAILABLE_STEPS = [
        "query_expansion",
        "hyde",
        "rerank",
        "fusion",
        "deduplication",
        "compression",
        "metadata",
    ]
    
    def __init__(
        self,
        reranker: RerankerProvider,
        compressor: ContextCompressor,
        fusion: ChunkFusion,
        deduplicator: Deduplicator,
        metadata_injector: MetadataInjector,
        llm_delegator: Optional[LLMDelegator] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.reranker = reranker
        self.compressor = compressor
        self.fusion = fusion
        self.deduplicator = deduplicator
        self.metadata_injector = metadata_injector
        self.llm_delegator = llm_delegator
        self.config = config or PipelineConfig.default()
        
        # Step handlers
        self._step_handlers: Dict[str, Callable] = {
            "query_expansion": self._step_query_expansion,
            "hyde": self._step_hyde,
            "rerank": self._step_rerank,
            "fusion": self._step_fusion,
            "deduplication": self._step_deduplication,
            "compression": self._step_compression,
            "metadata": self._step_metadata,
        }

        # Steps that can run in parallel (both operate on original query, don't modify chunks)
        self._parallel_eligible_steps = {"query_expansion", "hyde"}

    def _group_parallel_steps(
        self, steps: List[PipelineStepConfig], enable_parallel: bool
    ) -> List[List[PipelineStepConfig]]:
        """
        Group pipeline steps into batches for parallel/sequential execution.

        Steps 'query_expansion' and 'hyde' can run in parallel because:
        - Both operate on the original query (not intermediate state)
        - Neither modifies chunks
        - Both produce metadata that can be merged

        All other steps must run sequentially as they modify chunks.

        Args:
            steps: List of step configs to group
            enable_parallel: Whether parallel execution is enabled

        Returns:
            List of batches, each batch is a list of steps to run together
        """
        if not enable_parallel:
            # Each step in its own batch (sequential execution)
            return [[step] for step in steps if step.enabled]

        batches: List[List[PipelineStepConfig]] = []
        current_parallel_batch: List[PipelineStepConfig] = []

        for step in steps:
            if not step.enabled:
                continue

            if step.step in self._parallel_eligible_steps:
                # Accumulate parallel-eligible steps
                current_parallel_batch.append(step)
            else:
                # Flush any accumulated parallel steps first
                if current_parallel_batch:
                    batches.append(current_parallel_batch)
                    current_parallel_batch = []
                # Sequential step in its own batch
                batches.append([step])

        # Flush remaining parallel steps
        if current_parallel_batch:
            batches.append(current_parallel_batch)

        return batches
    
    async def execute(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        pipeline_config: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        ctx: Optional[Any] = None,
    ) -> PipelineResult:
        """
        Execute full enrichment pipeline.
        
        Args:
            query: Original search query
            chunks: Retrieved chunks to enrich
            pipeline_config: Override pipeline configuration
            chat_history: Previous messages for context
            top_k: Number of chunks to return
            ctx: Security context
        
        Returns:
            PipelineResult with enriched chunks and stats
        """
        start_time = time.perf_counter()
        
        # Parse config override
        config = PipelineConfig.from_dict(pipeline_config) if pipeline_config else self.config
        
        # Initialize state
        current_query = query
        current_chunks = chunks
        enriched_query: Optional[str] = None
        enrichment_applied: List[str] = []
        step_times: Dict[str, float] = {}
        metadata_added: Dict[str, Any] = {}
        
        # Context for steps
        context = {
            "original_query": query,
            "chat_history": chat_history,
            "top_k": top_k,
            "ctx": ctx,
        }

        # Group steps into parallel/sequential batches
        batches = self._group_parallel_steps(config.steps, config.enable_parallel_execution)

        # Execute batches
        for batch in batches:
            if len(batch) == 1:
                # Single step - sequential execution
                step_config = batch[0]
                step_name = step_config.step

                if step_name not in self._step_handlers:
                    logger.warning(f"Unknown pipeline step: {step_name}")
                    continue

                handler = self._step_handlers[step_name]

                try:
                    step_start = time.perf_counter()

                    result = await handler(
                        query=current_query,
                        chunks=current_chunks,
                        config=step_config.config,
                        context=context,
                    )

                    step_time = (time.perf_counter() - step_start) * 1000
                    step_times[step_name] = step_time

                    # Update state from result
                    if result.get("query"):
                        current_query = result["query"]
                        enriched_query = result["query"]

                    if result.get("chunks"):
                        current_chunks = result["chunks"]

                    if result.get("metadata"):
                        metadata_added.update(result["metadata"])

                    enrichment_applied.append(step_name)
                    logger.debug(f"Step '{step_name}' completed in {step_time:.1f}ms")

                except Exception as e:
                    logger.error(f"Step '{step_name}' failed: {e}")
                    step_times[step_name] = (time.perf_counter() - step_start) * 1000

                    if config.fail_fast:
                        raise

            else:
                # Multiple steps - parallel execution
                batch_start = time.perf_counter()
                step_names = [s.step for s in batch]
                logger.debug(f"Executing parallel batch: {step_names}")

                # Create coroutines for all steps in batch
                async def run_step(step_cfg: PipelineStepConfig) -> tuple[str, float, Dict[str, Any], Optional[str]]:
                    """Run a single step and return (name, time_ms, result, error)."""
                    sname = step_cfg.step
                    if sname not in self._step_handlers:
                        return (sname, 0.0, {}, f"Unknown step: {sname}")

                    handler = self._step_handlers[sname]
                    sstart = time.perf_counter()
                    try:
                        res = await handler(
                            query=current_query,
                            chunks=current_chunks,
                            config=step_cfg.config,
                            context=context,
                        )
                        stime = (time.perf_counter() - sstart) * 1000
                        return (sname, stime, res, None)
                    except Exception as ex:
                        stime = (time.perf_counter() - sstart) * 1000
                        return (sname, stime, {}, str(ex))

                # Execute all steps in parallel
                tasks = [run_step(sc) for sc in batch]
                results = await asyncio.gather(*tasks)

                batch_time = (time.perf_counter() - batch_start) * 1000
                logger.info(f"Parallel batch {step_names} completed in {batch_time:.1f}ms")

                # Merge results from parallel steps
                # Parallel steps (query_expansion, hyde) don't modify chunks,
                # they only add metadata and possibly modify query
                for sname, stime, result, error in results:
                    step_times[sname] = stime

                    if error:
                        logger.error(f"Step '{sname}' failed: {error}")
                        if config.fail_fast:
                            raise RuntimeError(f"Step '{sname}' failed: {error}")
                        continue

                    # Update query if provided (query_expansion provides combined_query)
                    if result.get("query"):
                        current_query = result["query"]
                        enriched_query = result["query"]

                    # Chunks are not modified by parallel-eligible steps
                    # but include in case future steps do
                    if result.get("chunks"):
                        current_chunks = result["chunks"]

                    # Merge metadata from all parallel steps
                    if result.get("metadata"):
                        metadata_added.update(result["metadata"])

                    enrichment_applied.append(sname)
                    logger.debug(f"Step '{sname}' completed in {stime:.1f}ms (parallel)")
        
        # Convert final chunks to EnrichedChunk
        final_chunks = [
            EnrichedChunk.from_dict(c) if isinstance(c, dict) else c
            for c in current_chunks
        ]
        
        # Apply top_k limit
        if len(final_chunks) > top_k:
            final_chunks = final_chunks[:top_k]
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        return PipelineResult(
            enriched_query=enriched_query,
            enriched_chunks=final_chunks,
            metadata_added=metadata_added,
            enrichment_applied=enrichment_applied,
            stats={
                "total_time_ms": round(total_time, 2),
                "step_times": {k: round(v, 2) for k, v in step_times.items()},
                "chunks_before": len(chunks),
                "chunks_after": len(final_chunks),
            },
        )
    
    # ========================================================================
    # Step Handlers
    # ========================================================================
    
    async def _step_query_expansion(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Query expansion step."""
        if not self.llm_delegator or not self.llm_delegator.is_available():
            logger.warning("LLM delegator not available, skipping query expansion")
            return {"query": query, "chunks": chunks}
        
        result = await self.llm_delegator.expand_query(
            query=query,
            num_variants=config.get("num_variants", 3),
            chat_history=context.get("chat_history"),
            expansion_type=config.get("expansion_type", "semantic"),
        )
        
        return {
            "query": result.combined_query,
            "chunks": chunks,
            "metadata": {
                "query_expansion": {
                    "original": query,
                    "variants": result.expanded_queries,
                },
            },
        }
    
    async def _step_hyde(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """HyDE generation step."""
        if not self.llm_delegator or not self.llm_delegator.is_available():
            logger.warning("LLM delegator not available, skipping HyDE")
            return {"chunks": chunks}
        
        result = await self.llm_delegator.generate_hyde(
            query=context.get("original_query", query),
            document_type=config.get("document_type", "answer"),
            max_length=config.get("max_length", 200),
        )
        
        # HyDE document can be used to enhance retrieval
        # For now, add it as metadata
        return {
            "chunks": chunks,
            "metadata": {
                "hyde": {
                    "hypothetical_document": result.hypothetical_document,
                    "document_type": result.document_type,
                },
            },
        }
    
    async def _step_rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Reranking step."""
        # Use original query for reranking (more precise)
        rerank_query = context.get("original_query", query)
        
        result = self.reranker.rerank(
            query=rerank_query,
            chunks=chunks,
            top_k=config.get("top_k", context.get("top_k")),
        )
        
        return {
            "chunks": [c.to_dict() for c in result.reranked_chunks],
            "metadata": {
                "rerank": {
                    "model": result.model_used,
                    "time_ms": result.time_ms,
                },
            },
        }
    
    async def _step_fusion(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Chunk fusion step."""
        result = self.fusion.fuse(
            chunks=chunks,
            strategies=config.get("strategies"),
        )
        
        return {
            "chunks": [c.to_dict() for c in result.fused_chunks],
            "metadata": {
                "fusion": {
                    "chunks_before": result.chunks_before,
                    "chunks_after": result.chunks_after,
                    "fusions_applied": result.fusions_applied,
                },
            },
        }
    
    async def _step_deduplication(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deduplication step."""
        unique, removed, groups = self.deduplicator.deduplicate(
            chunks=chunks,
            method=config.get("method"),
            threshold=config.get("similarity_threshold"),
        )
        
        return {
            "chunks": [c.to_dict() for c in unique],
            "metadata": {
                "deduplication": {
                    "duplicates_removed": removed,
                    "duplicate_groups": groups,
                },
            },
        }
    
    async def _step_compression(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compression step."""
        method = config.get("method", "extractive")
        original_query = context.get("original_query", query)
        
        if method == "abstractive" and self.llm_delegator and self.llm_delegator.is_available():
            # Use LLM for abstractive compression
            compressed = await self.llm_delegator.compress_abstractive(
                query=original_query,
                chunks=chunks,
                target_ratio=config.get("compression_ratio", 0.5),
            )
            result_chunks = compressed
            actual_method = "abstractive"
        else:
            # Use extractive compression
            result = self.compressor.compress(
                query=original_query,
                chunks=chunks,
                target_ratio=config.get("compression_ratio", 0.5),
                method="extractive",
            )
            result_chunks = [c.to_dict() for c in result.compressed_chunks]
            actual_method = "extractive"
        
        return {
            "chunks": result_chunks,
            "metadata": {
                "compression": {
                    "method": actual_method,
                },
            },
        }
    
    async def _step_metadata(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Metadata injection step."""
        metadata_types = config.get("metadata_types", ["source", "relevance", "position", "tokens"])
        
        enriched = self.metadata_injector.inject(
            chunks=chunks,
            metadata_types=metadata_types,
        )
        
        return {
            "chunks": [c.to_dict() for c in enriched],
            "metadata": {
                "metadata_injected": metadata_types,
            },
        }
    
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
            "reranker_model": self.reranker.config.model,
            "device": self.reranker.config.device,
        }
    
    def update_config(self, pipeline_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update pipeline configuration."""
        self.config = PipelineConfig.from_dict({"steps": pipeline_steps})
        return self.get_config()
