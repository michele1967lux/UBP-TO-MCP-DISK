"""
RAG Orchestrator Providers - Technical Logic Layer (v3.7.0)

Pure technical logic for RAG pipeline orchestration.
Can be tested independently without UBP framework.

Pipeline: Retrieve → Augment → Generate

v3.7.0 additions:
- Context Governor integration via get_execution_plan()
- Dynamic system prompt injection based on tightness
- ExecutionPlan-driven RAG parameters (top_k, threshold)
- Feature flag: UBP_CONTEXT_GOVERNOR__ENABLED

Redis Key Prefixes:
All Redis keys are managed via RedisKeyManager to ensure proper isolation
between production and test environments. See NAMING_POLICY.md Section 7.

Key patterns used by this module:
- ACL permissions:  ubp:{env}:rag:acl:{entity_type}:{entity_id}:{collection_id}
- RAG config:       ubp:{env}:rag:config:{entity_type}:{entity_id}
- Default config:   ubp:{env}:rag:config:default
- History index:    ubp:{env}:rag:history:{user_id}:index (SET of conversation_ids)
- History data:     ubp:{env}:rag:history:{user_id}:{conversation_id} (HASH)
- KB Keywords:      ubp:{env}:rag:keywords:{collection_name} (SET) [v1.8.1 DKI]

Where {env} is:
- Production: "" (empty, results in ubp:...)
- Test:       "test:{suite}:" (e.g., ubp:test:integration:...)
- Development: "dev:" (e.g., ubp:dev:...)
"""

import asyncio
import logging
import json
import os
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING, Set
import uuid
import re

# FIX-008 v1.8.5: Import sanitizer for context cleanup
from ubp_enterprise_hybrid.modules.cores._shared.utils import sanitize_for_prompt

# FIX-TYPE-001 v2.2.4: Import type coercion for Redis config safety
from ubp_enterprise_hybrid.modules.cores._shared.manifest_loader import coerce_config_types
from .tool_definitions import get_tool_definitions
from .tool_executor import ArchitectToolExecutor

# v3.7.0: Type hints for Context Governor integration
if TYPE_CHECKING:
    from ubp_enterprise_hybrid.modules.cores.adaptive_budget_manager.models import ExecutionPlan

# Import Redis key manager for environment-aware key prefixing
try:
    from ubp_enterprise_hybrid.backend.app.infra.redis_keys import get_key_manager

    REDIS_KEYS_AVAILABLE = True
except ImportError:
    REDIS_KEYS_AVAILABLE = False
    get_key_manager = None

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Pure RAG pipeline implementation.

    Orchestrates the three-stage RAG process:
    1. Retrieve: Query Qdrant for relevant documents
    2. Augment: Build context from retrieved documents
    3. Generate: Use LLM to generate answer with context

    Optional: GPU Reranking via enrichment_pipeline (v1.7.1)
    """

    def __init__(self, qdrant_module, llm_module, enrichment_resolver=None, reranker_resolver=None, adaptive_memory_resolver=None, tool_llm_resolver=None):
        """
        Initialize RAG pipeline with dependencies.

        Args:
            qdrant_module: Module instance with query() method
            llm_module: Module instance with generate() method
            enrichment_resolver: Optional async callable that returns enrichment_pipeline
                                 module (v1.7.1 - lazy resolution for GPU reranking)
            reranker_resolver: Optional async callable that returns rag_reranker
                              module (v3.7.0 - dedicated reranking)
            adaptive_memory_resolver: Optional async callable that returns adaptive_budget_manager
                                     module (v4.0.0 - adaptive token budget management)
            tool_llm_resolver: Optional async callable that returns a separate LLM module
                              for tool calling (v6.8.0 - dual-LLM orchestration)
        """
        self.qdrant = qdrant_module
        self.llm = llm_module
        self._enrichment_resolver = enrichment_resolver  # v1.7.1 Lazy resolver
        self._enrichment = None  # Cached enrichment module
        self._enrichment_resolved = False  # Track if we've tried resolving
        self._enrichment_failed = False  # Track if resolution permanently failed (v3.7.1 FIX-RC1)
        self._reranker_resolver = reranker_resolver  # v3.7.0 Lazy resolver
        self._reranker = None  # Cached reranker module
        self._reranker_resolved = False  # Track if we've tried resolving
        self._adaptive_memory_resolver = adaptive_memory_resolver  # v4.0.0 Lazy resolver
        self._adaptive_memory = None  # Cached adaptive memory module
        self._adaptive_memory_resolved = False  # Track if we've tried resolving
        self._tool_llm_resolver = tool_llm_resolver  # v6.8.0 Lazy resolver for tool-calling LLM
        self._tool_llm = None  # Cached tool LLM module
        self._tool_llm_resolved = False

    async def _get_enrichment(self):
        """
        Lazy resolve enrichment module on first use (v1.7.1).
        
        v3.7.1 FIX-RC1: Added retry mechanism with visible diagnostics.
        If first resolution fails, retry once. If still fails, log WARNING 
        visible in Docker and mark as permanently failed.
        """
        if self._enrichment_resolved:
            return self._enrichment

        self._enrichment_resolved = True
        if self._enrichment_resolver is not None:
            try:
                self._enrichment = await self._enrichment_resolver()
                if self._enrichment:
                    logger.info("enrichment_pipeline resolved - GPU Reranking enabled")
                else:
                    logger.debug(
                        "enrichment_pipeline not available - Using vector scores only"
                    )
            except Exception as e:
                logger.warning(
                    f"Could not resolve enrichment_pipeline: {e} - Using vector scores only"
                )
                self._enrichment = None
        
        # FIX-RC1 v3.7.1: Retry mechanism if first resolution returned None
        if self._enrichment is None and not self._enrichment_failed and self._enrichment_resolver is not None:
            logger.warning("[ENRICHMENT-GUARD] First resolution returned None, retrying once...")
            self._enrichment_resolved = False  # Allow one retry
            try:
                self._enrichment = await self._enrichment_resolver()
                self._enrichment_resolved = True
                if self._enrichment:
                    logger.info("[ENRICHMENT-GUARD] Retry successful - enrichment_pipeline now available")
                else:
                    logger.warning("[ENRICHMENT-GUARD] Retry failed - marking as permanently unavailable")
                    self._enrichment_failed = True
            except Exception as e:
                logger.error(f"[ENRICHMENT-GUARD] Retry failed with exception: {e}")
                self._enrichment_resolved = True
                self._enrichment_failed = True
        
        # FIX-RC1 v3.7.1: Visible warning if enrichment permanently unavailable
        if self._enrichment is None and self._enrichment_failed:
            logger.warning("[ENRICHMENT-GUARD] enrichment_pipeline NOT resolved - ALL enrichment features disabled!")
            
        return self._enrichment
    
    async def _get_reranker(self):
        """Lazy resolve rag_reranker module on first use (v3.7.0)."""
        if self._reranker_resolved:
            return self._reranker
        
        self._reranker_resolved = True
        if self._reranker_resolver is not None:
            try:
                self._reranker = await self._reranker_resolver()
                if self._reranker:
                    logger.info("rag_reranker resolved - Dedicated reranking available")
                else:
                    logger.debug("rag_reranker not available")
            except Exception as e:
                logger.warning(f"Could not resolve rag_reranker: {e}")
                self._reranker = None
        return self._reranker
    
    async def _get_adaptive_memory(self):
        """Lazy resolve adaptive_budget_manager module on first use (v4.0.0)."""
        if self._adaptive_memory_resolved:
            return self._adaptive_memory
        
        self._adaptive_memory_resolved = True
        if self._adaptive_memory_resolver is not None:
            try:
                self._adaptive_memory = await self._adaptive_memory_resolver()
                if self._adaptive_memory:
                    logger.info("adaptive_budget_manager resolved - Adaptive budget management enabled")
                else:
                    logger.debug("adaptive_budget_manager not available - Using static budget")
            except Exception as e:
                logger.warning(f"Could not resolve adaptive_budget_manager: {e} - Using static budget")
                self._adaptive_memory = None
        return self._adaptive_memory

    async def _get_tool_llm(self):
        """Lazy resolve tool-calling LLM module on first use (v6.8.0).
        
        Enables dual-LLM orchestration: a separate LLM can execute tool calls
        (e.g. vllm_remote for search_knowledge_base) while the main LLM (self.llm)
        generates the final answer.
        """
        if self._tool_llm_resolved:
            return self._tool_llm

        self._tool_llm_resolved = True
        if self._tool_llm_resolver is not None:
            try:
                self._tool_llm = await self._tool_llm_resolver()
                if self._tool_llm:
                    logger.info("[TOOL-LLM] Dedicated tool-calling LLM resolved for dual-LLM orchestration")
                else:
                    logger.debug("[TOOL-LLM] Tool LLM not available — tool calls will use main LLM")
            except Exception as e:
                logger.warning(f"[TOOL-LLM] Could not resolve tool LLM: {e} — tool calls will use main LLM")
                self._tool_llm = None
        return self._tool_llm

    def _build_context_governor_constraints(
        self,
        config: Dict[str, Any],
        pipeline_config: Optional[Dict[str, Any]] = None,
        collections: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        v3.7.1: Build 4-layer constraints for Context Governor.
        
        Extracts constraints from:
        1. User override_config (highest priority)
        2. Collection metadata (from pipeline_config._layers_debug)
        3. Client policy (from pipeline_config or capabilities)
        4. System defaults (lowest priority)
        
        Args:
            config: Merged RAG configuration (includes user overrides)
            pipeline_config: Permission-aware pipeline config with _layers_debug
            collections: Target collections for metadata lookup
            
        Returns:
            Dict with constraint parameters for Context Governor:
            - max_doc_budget_tokens: Hard cap on document context budget
            - max_memory_budget_tokens: Hard cap on conversation memory
            - max_top_k: Maximum chunks to retrieve
            - min_similarity_threshold: Minimum quality floor
            - allowed_strategies: List of permitted context strategies
            - source: Origin of constraints for debugging
        """
        constraints = {}
        sources = []
        
        pipeline_config = pipeline_config or {}
        layers_debug = pipeline_config.get("_layers_debug", {})
        
        # === Priority 1: User override_config (highest) ===
        # These come from the API request's override_config field
        if config.get("max_doc_budget_tokens"):
            constraints["max_doc_budget_tokens"] = int(config["max_doc_budget_tokens"])
            sources.append("user")
        
        if config.get("max_memory_budget_tokens"):
            constraints["max_memory_budget_tokens"] = int(config["max_memory_budget_tokens"])
            if "user" not in sources:
                sources.append("user")
        
        if config.get("max_top_k"):
            constraints["max_top_k"] = int(config["max_top_k"])
            if "user" not in sources:
                sources.append("user")
        
        if config.get("min_similarity_threshold"):
            constraints["min_similarity_threshold"] = float(config["min_similarity_threshold"])
            if "user" not in sources:
                sources.append("user")
        
        if config.get("allowed_strategies"):
            allowed = config["allowed_strategies"]
            if isinstance(allowed, str):
                allowed = [s.strip() for s in allowed.split(",")]
            constraints["allowed_strategies"] = allowed
            if "user" not in sources:
                sources.append("user")
        
        # === Priority 2: Collection metadata ===
        # Check target_collections from pipeline_config for collection-level constraints
        target_collections = layers_debug.get("target_collections", [])
        if target_collections and isinstance(target_collections, list):
            # For multiple collections, use the most restrictive constraint
            for coll_info in target_collections:
                if not isinstance(coll_info, dict):
                    continue
                    
                # Collection may specify constraints in metadata
                coll_constraints = coll_info.get("constraints", {})
                
                if coll_constraints.get("max_doc_budget_tokens") and "max_doc_budget_tokens" not in constraints:
                    constraints["max_doc_budget_tokens"] = int(coll_constraints["max_doc_budget_tokens"])
                    sources.append("collection")
                
                if coll_constraints.get("max_top_k") and "max_top_k" not in constraints:
                    constraints["max_top_k"] = int(coll_constraints["max_top_k"])
                    if "collection" not in sources:
                        sources.append("collection")
                
                if coll_constraints.get("min_similarity_threshold") and "min_similarity_threshold" not in constraints:
                    constraints["min_similarity_threshold"] = float(coll_constraints["min_similarity_threshold"])
                    if "collection" not in sources:
                        sources.append("collection")
        
        # === Priority 3: Client policy ===
        # Check client capabilities from pipeline_config
        client_capabilities = layers_debug.get("client_capabilities", {})
        if client_capabilities:
            cap_constraints = client_capabilities.get("context_governor_constraints", {})
            
            if cap_constraints.get("max_doc_budget_tokens") and "max_doc_budget_tokens" not in constraints:
                constraints["max_doc_budget_tokens"] = int(cap_constraints["max_doc_budget_tokens"])
                sources.append("client")
            
            if cap_constraints.get("max_memory_budget_tokens") and "max_memory_budget_tokens" not in constraints:
                constraints["max_memory_budget_tokens"] = int(cap_constraints["max_memory_budget_tokens"])
                if "client" not in sources:
                    sources.append("client")
            
            if cap_constraints.get("allowed_strategies") and "allowed_strategies" not in constraints:
                allowed = cap_constraints["allowed_strategies"]
                if isinstance(allowed, str):
                    allowed = [s.strip() for s in allowed.split(",")]
                constraints["allowed_strategies"] = allowed
                if "client" not in sources:
                    sources.append("client")
        
        # Set source metadata
        if sources:
            constraints["source"] = "+".join(sources)
        else:
            constraints["source"] = "none"
        
        # Log constraint building
        if constraints and constraints.get("source") != "none":
            logger.info(
                "[CONTEXT-GOVERNOR] 4-layer constraints built",
                extra={
                    "constraints": constraints,
                    "sources": sources,
                    "collections": collections,
                }
            )
        
        return constraints

    async def chat(
        self,
        query: str,
        collections: List[str],
        config: Dict[str, Any],
        return_debug: bool = False,
        web_context: Optional[str] = None,
        conversation_context: Optional[str] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
        turn_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Execute RAG chat pipeline.

        Args:
            query: User question
            collections: List of collection names to query (empty list = Pure LLM mode)
            config: RAG configuration (model, temperature, top_k, system_prompt, etc.)
                    v1.10.2: Can include context_limit_tokens for dynamic budget calculation
                    v4.0.0: Can include adaptive_memory_enabled for intelligent budget management
            return_debug: If True, include debug info with retrieved chunks
            web_context: Optional pre-formatted web search results to augment context
                         (ROADMAP v1.5.0 - FEAT-WEB-001)
            conversation_context: Optional formatted conversation history for multi-turn
                                  (ROADMAP v1.5.0 - FEAT-MEM-001)
            pipeline_config: Optional permission-aware enrichment configuration
            turn_count: Conversation turn number (for adaptive memory, v4.0.0)

        Returns:
            Dict with answer, sources, config_used, and optionally debug info
        """
        # v6.8.1: Phase timing instrumentation for p95 latency RCA
        _t_e2e_start = time.perf_counter()
        _phase_timings: Dict[str, Any] = {}

        # Check for Pure LLM mode (empty collections list)
        pure_llm_mode = len(collections) == 0

        logger.info(
            "Executing RAG pipeline"
            if not pure_llm_mode
            else "Executing Pure LLM mode (no retrieval)",
            extra={
                "query_length": len(query),
                "collections": collections,
                "pure_llm_mode": pure_llm_mode,
                "top_k": config.get("top_k", 5),
                "web_context_provided": web_context is not None,
                "conversation_context_provided": conversation_context is not None,
            },
        )

        # v2.2.3: Initialize all variables before conditional blocks (FIX-SCOPE-001)
        retrieved_docs = []
        retrieve_result = {}  # Must be initialized to avoid UnboundLocalError
        context = ""
        hyde_document = None
        execution_plan: Optional["ExecutionPlan"] = None  # v3.7.0: Context Governor
        _adaptive_budget_debug: Optional[Dict[str, Any]] = None  # v4.1.0: FIX-SCOPE-002

        if not pure_llm_mode:
            # Step 1: Retrieve - Query Qdrant for relevant documents
            # v2.2.2: _retrieve now returns dict with docs and hyde_document
            _t_retrieval_start = time.perf_counter()
            retrieve_result = await self._retrieve(
                query, collections, config, pipeline_config=pipeline_config
            )
            retrieved_docs = retrieve_result.get("docs", [])
            hyde_document = retrieve_result.get("hyde_document")
            _phase_timings["t_retrieval_ms"] = round((time.perf_counter() - _t_retrieval_start) * 1000, 2)

            if not retrieved_docs:
                logger.warning("No documents retrieved from Qdrant")
                # PRE-GENERATE-GUARD: Instead of returning "I don't have enough info",
                # fall back to web context or general knowledge.
                _capabilities = config.get("_capabilities", {})
                _pipeline_name = config.get("_pipeline_name", "unknown")

                # If web_context was provided (pre-fetched), use it
                if web_context:
                    logger.warning(
                        "[PRE-GENERATE-GUARD] pipeline=%s | action=fallback_web_context | "
                        "reason=empty_retrieval",
                        _pipeline_name,
                    )
                    # Continue with web context as the source
                    context = web_context
                    # Skip retrieval-dependent steps, jump to generate
                    _phase_timings["t_retrieval_ms"] = round(
                        (time.perf_counter() - _t_retrieval_start) * 1000, 2
                    )
                    _phase_timings["guard_activated"] = "fallback_web_context"
                elif _capabilities.get("web_effective_enabled", False):
                    logger.warning(
                        "[PRE-GENERATE-GUARD] pipeline=%s | action=fallback_knowledge | "
                        "reason=empty_retrieval_web_not_prefetched",
                        _pipeline_name,
                    )
                    # Web is enabled but no pre-fetched context — generate from knowledge
                    context = ""
                    _phase_timings["t_retrieval_ms"] = round(
                        (time.perf_counter() - _t_retrieval_start) * 1000, 2
                    )
                    _phase_timings["guard_activated"] = "fallback_knowledge"
                    # Modify system prompt to allow general knowledge
                    if "system_prompt" in config:
                        config["system_prompt"] = (
                            "Rispondi usando la tua conoscenza generale. "
                            "Segnala brevemente che non stai usando documenti del cliente.\n\n"
                            + config["system_prompt"]
                        )
                else:
                    logger.warning(
                        "[PRE-GENERATE-GUARD] pipeline=%s | action=fallback_knowledge_only | "
                        "reason=empty_retrieval_no_web",
                        _pipeline_name,
                    )
                    context = ""
                    _phase_timings["t_retrieval_ms"] = round(
                        (time.perf_counter() - _t_retrieval_start) * 1000, 2
                    )
                    _phase_timings["guard_activated"] = "fallback_knowledge_only"
                    if "system_prompt" in config:
                        config["system_prompt"] = (
                            "Rispondi usando la tua conoscenza generale. "
                            "Segnala brevemente che non stai usando documenti del cliente.\n\n"
                            + config["system_prompt"]
                        )

                # If guard activated with context (web or empty), skip to generate
                if "guard_activated" in _phase_timings:
                    _t_generate_start = time.perf_counter()
                    generate_result = await self._generate(
                        query, context, config, execution_plan=None,
                    )
                    _phase_timings["t_generate_ms"] = round(
                        (time.perf_counter() - _t_generate_start) * 1000, 2
                    )
                    _answer = generate_result.get("answer", "")
                    # Add interaction suffix
                    _interaction_suffix = (
                        "\n\n*Ho risposto con le mie conoscenze generali, "
                        "non dalla tua documentazione. "
                        "Vuoi che cerchi in modo più specifico?*"
                    )
                    _answer = _answer + _interaction_suffix
                    _phase_timings["t_e2e_ms"] = round(
                        (time.perf_counter() - _t_e2e_start) * 1000, 2
                    )
                    return {
                        "answer": _answer,
                        "sources": [],
                        "config_used": {k: v for k, v in config.items() if not k.startswith("_")},
                        "debug": {"retrieved_chunks": [], "guard": _phase_timings.get("guard_activated")} if return_debug else None,
                        "metadata": {
                            "guard_activated": _phase_timings.get("guard_activated"),
                            "phase_timings": _phase_timings,
                        },
                    }

            # v4.0.0: Initialize adaptive budget debug capture (populated if adjust_budget runs)
            _adaptive_budget_debug: Optional[Dict[str, Any]] = None

            # v6.8.1: Start context packing timer (budget + filter + augment)
            _t_context_packing_start = time.perf_counter()

            # Step 1.5: Context Governor (v3.7.0) OR Adaptive Budget Management (v4.0.0)
            # Context Governor is the evolution of Adaptive Memory with ExecutionPlan output
            context_governor_enabled = config.get("context_governor_enabled", False) or \
                os.environ.get("UBP_CONTEXT_GOVERNOR__ENABLED", "false").lower() == "true"
            adaptive_memory_enabled = config.get("adaptive_memory_enabled", False)
            adaptive_memory = await self._get_adaptive_memory()

            # v4.1.2 DEBUG: Log path decision
            print(f"[RAG-PATH-DEBUG] context_governor={context_governor_enabled}, adaptive_memory_enabled={adaptive_memory_enabled}, adaptive_memory={adaptive_memory is not None}", flush=True)
            logger.warning(
                f"[RAG-PATH-DEBUG] context_governor={context_governor_enabled}, "
                f"adaptive_memory_enabled={adaptive_memory_enabled}, "
                f"adaptive_memory_available={adaptive_memory is not None}, "
                f"config.keys={list(config.keys())}"
            )
            
            # === CONTEXT GOVERNOR PATH (v3.7.0) ===
            _budget_t0 = time.perf_counter()
            if context_governor_enabled and adaptive_memory and hasattr(adaptive_memory, "get_execution_plan"):
                try:
                    logger.info("[CONTEXT-GOVERNOR] Generating ExecutionPlan for request")

                    # v4.1.2: Resolve provider via ProviderMapper (centralized fallback)
                    from ubp_enterprise_hybrid.modules.cores._shared.token_limits import TokenCounter
                    from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper
                    resolved_provider = config.get("provider") or ProviderMapper.get_provider_for_role_safe("rag")

                    # v6.8.0: FIX-BUDGET-003 — Budget uses the GENERATION provider, not the tool provider.
                    # Previous FIX-ARCH-001 overwrote resolved_provider with _tool_settings["provider"]
                    # (e.g. "vllm_remote" with 20K ctx) even when the generation LLM is "grok" (2M ctx).
                    # This caused tightness≈0.87, threshold spike, and 0/60 docs injected.
                    # The tool provider is irrelevant for budget: it executes search_knowledge_base(),
                    # not the final answer generation. Budget must reflect the answer LLM's context window.
                    logger.info(
                        f"[BUDGET] resolved_provider='{resolved_provider}' "
                        f"(generation LLM for budget calculation)",
                    )

                    # Calculate current memory tokens if conversation context exists
                    current_memory_tokens = 0
                    if conversation_context:
                        current_memory_tokens = TokenCounter.count_tokens(
                            conversation_context,
                            config.get("model"),
                            resolved_provider
                        )

                    # v3.7.1: Build 4-layer constraints from config and pipeline_config
                    constraints = self._build_context_governor_constraints(
                        config=config,
                        pipeline_config=pipeline_config,
                        collections=collections,
                    )

                    # v6.3.1: FIX-ARCH-002 — Pass chunk token info to budget manager.
                    # Without this, budget manager sees chunk_tokens=0 → tightness=0.05
                    # and allows all docs through, causing context overflow.
                    #
                    # retrieve_result is always a dict here (assigned at line 382 by _retrieve()).
                    # compression_stats is None when enrichment compression is disabled (most cases).
                    # In that case, estimate total_chunk_tokens from chunk texts directly.
                    total_chunks = len(retrieved_docs)
                    comp_stats = retrieve_result.get("compression_stats")
                    if comp_stats and isinstance(comp_stats, dict):
                        total_chunk_tokens = comp_stats.get("compressed_tokens", 0)
                    else:
                        # Compression disabled or not run: sum actual chunk token counts
                        total_chunk_tokens = sum(
                            TokenCounter.count_tokens(
                                doc.get("text", ""), config.get("model"), resolved_provider
                            )
                            for doc in retrieved_docs
                        )
                        logger.info(
                            f"[BUDGET] compression_stats unavailable, estimated chunk_tokens={total_chunk_tokens} "
                            f"from {total_chunks} docs via TokenCounter",
                        )

                    # Get ExecutionPlan - the unified output of Context Governor
                    execution_plan = adaptive_memory.get_execution_plan(
                        query=query,
                        turn_count=turn_count,
                        provider=resolved_provider,
                        model=config.get("model"),
                        task_profile=config.get("task_profile", "chat"),
                        current_memory_tokens=current_memory_tokens,
                        rag_config=config,
                        constraints=constraints,  # v3.7.1: Pass 4-layer constraints
                        total_chunk_tokens=total_chunk_tokens,  # v6.3.1: FIX-ARCH-002
                        total_chunks=total_chunks,              # v6.3.1: FIX-ARCH-002
                    )
                    
                    # Apply ExecutionPlan parameters
                    # 1. Use execution_plan.doc_budget_tokens for context budget
                    doc_budget_tokens = execution_plan.doc_budget_tokens
                    max_context_chars = int(doc_budget_tokens * config.get("chars_per_token", 3.5))
                    
                    # 2. Filter documents by adjusted similarity threshold
                    # Prefer rerank_score (cross-encoder) over raw cosine score
                    similarity_threshold = execution_plan.similarity_threshold
                    original_doc_count = len(retrieved_docs)
                    _pre_filter_docs = list(retrieved_docs)  # FIX-ARCHITECT-001: backup for safety net
                    retrieved_docs = [
                        doc for doc in retrieved_docs
                        if (doc.get("rerank_score") or doc.get("score", 0)) >= similarity_threshold
                    ]

                    # FIX-ARCHITECT-001: Safety net — if threshold dropped ALL docs,
                    # fall back to top-N by score. Reranker scores (cross-encoder)
                    # use a different scale than cosine similarity, so threshold
                    # may be too aggressive after reranking.
                    if not retrieved_docs and _pre_filter_docs:
                        _fallback_k = min(10, len(_pre_filter_docs))
                        retrieved_docs = sorted(
                            _pre_filter_docs,
                            key=lambda d: d.get("rerank_score") or d.get("score", 0),
                            reverse=True,
                        )[:_fallback_k]
                        logger.warning(
                            "[RAG-SAFETY-NET] threshold=%.3f dropped ALL %d docs "
                            "— fallback to top-%d by score (max=%.3f, min=%.3f)",
                            similarity_threshold,
                            original_doc_count,
                            _fallback_k,
                            max((d.get("rerank_score") or d.get("score", 0) for d in retrieved_docs), default=0),
                            min((d.get("rerank_score") or d.get("score", 0) for d in retrieved_docs), default=0),
                        )

                    # Low-relevance detection — if all remaining docs score
                    # poorly, they're likely noise. Clear them so the LLM responds
                    # from general knowledge instead of hallucinating over garbage context.
                    if retrieved_docs:
                        _scores = [
                            doc.get("rerank_score") or doc.get("score", 0)
                            for doc in retrieved_docs
                        ]
                        _avg_score = sum(_scores) / len(_scores)
                        _max_score = max(_scores)
                        if _avg_score < 0.45 and _max_score < 0.55:
                            logger.warning(
                                "[RAG-QUALITY] Low relevance detected: avg=%.3f max=%.3f "
                                "docs=%d — clearing context for general knowledge fallback",
                                _avg_score, _max_score, len(retrieved_docs),
                            )
                            retrieved_docs = []

                    # 3. Limit to execution_plan.rag_top_k
                    if len(retrieved_docs) > execution_plan.rag_top_k:
                        retrieved_docs = retrieved_docs[:execution_plan.rag_top_k]
                    
                    # Safely extract enum values (may be string or enum depending on serialization path)
                    strategy_value = execution_plan.context_strategy.value if hasattr(execution_plan.context_strategy, 'value') else str(execution_plan.context_strategy)
                    response_style_value = execution_plan.response_style.value if hasattr(execution_plan.response_style, 'value') else str(execution_plan.response_style)
                    
                    logger.info(
                        f"[CONTEXT-GOVERNOR] ExecutionPlan applied: tightness={execution_plan.tightness:.2f}, "
                        f"strategy={strategy_value}, "
                        f"docs={len(retrieved_docs)}/{original_doc_count}, budget={max_context_chars} chars",
                        extra={
                            "tightness": execution_plan.tightness,
                            "context_strategy": strategy_value,
                            "filtered_doc_count": len(retrieved_docs),
                            "original_doc_count": original_doc_count,
                            "doc_budget_tokens": doc_budget_tokens,
                            "rag_top_k": execution_plan.rag_top_k,
                            "similarity_threshold": similarity_threshold,
                            "system_modifier": execution_plan.system_instruction_modifier is not None,
                            "response_style": response_style_value,
                        }
                    )
                    
                    # Skip legacy adaptive memory path
                    adaptive_memory_enabled = False

                    # Build context with ExecutionPlan budget
                    context = self._augment(retrieved_docs, max_context_chars)

                    # v4.3.0: Populate _adaptive_budget_debug from ExecutionPlan
                    # so frontend always receives consistent debug data
                    _adaptive_budget_time_ms = round((time.perf_counter() - _budget_t0) * 1000, 2)
                    _adaptive_budget_debug = {
                        "mode": "rag",
                        "tightness": execution_plan.tightness,
                        "context_window": execution_plan.context_window,
                        "memory_tokens": execution_plan.memory_budget_tokens,
                        "doc_budget_tokens": doc_budget_tokens,
                        "response_budget_tokens": execution_plan.reserved_output_tokens,
                        "fixed_overhead_tokens": execution_plan.query_overhead_tokens,
                        "adjusted_min_score": similarity_threshold,
                        "original_doc_count": original_doc_count,
                        "filtered_doc_count": len(retrieved_docs),
                        "compression_applied": False,
                        "doc_compression_needed": execution_plan.compression_recommended,
                        "max_context_chars": max_context_chars,
                        "time_ms": _adaptive_budget_time_ms,
                    }

                except Exception as e:
                    logger.error(f"[CONTEXT-GOVERNOR] ExecutionPlan generation failed: {e}", exc_info=True)
                    # Fallback to legacy adaptive memory or standard calculation
                    execution_plan = None
                    context_governor_enabled = False
            
            # === LEGACY ADAPTIVE MEMORY PATH (v4.0.0 - deprecated, use Context Governor) ===
            # FIX-BUDGET-004: Restored correct method name adjust_budget (adapter interface)
            # The adapter.adjust_budget() internally extracts model/provider from config
            # and calls provider.adjust() with correct parameters
            if adaptive_memory_enabled and adaptive_memory and hasattr(adaptive_memory, "adjust_budget"):
                _budget_t0 = time.perf_counter()  # may re-assign if CG path skipped
                try:
                    logger.info("[ADAPTIVE_MEMORY] Adjusting budget dynamically (legacy path)")

                    # FIX-BUDGET-004: adapter.adjust_budget() expects config dict with model/provider inside
                    # The adapter extracts model/provider from config internally
                    adjustment_result = await adaptive_memory.adjust_budget(
                        query=query,
                        conversation_context=conversation_context,
                        retrieved_docs=retrieved_docs,
                        turn_count=turn_count,
                        config=config
                    )
                    
                    # Apply adjustments
                    # 1. Update conversation context (potentially compressed)
                    conversation_context = adjustment_result.get("conversation_context")
                    
                    # 2. Use filtered documents (by adjusted similarity threshold)
                    retrieved_docs = adjustment_result.get("filtered_docs", retrieved_docs)
                    
                    # 3. Use adjusted document budget
                    doc_budget_tokens = adjustment_result.get("doc_budget_tokens", 0)
                    max_context_chars = int(doc_budget_tokens * config.get("chars_per_token", 3.5))
                    
                    # v4.0.0: Capture full adjustment result for debug panel
                    _adaptive_budget_time_ms = round((time.perf_counter() - _budget_t0) * 1000, 2)
                    _adaptive_budget_debug = {
                        "mode": "rag",
                        "tightness": adjustment_result.get("tightness"),
                        "context_window": adjustment_result.get("context_window"),
                        "memory_tokens": adjustment_result.get("memory_tokens"),
                        "doc_budget_tokens": doc_budget_tokens,
                        "fixed_overhead_tokens": adjustment_result.get("fixed_overhead_tokens"),
                        "adjusted_min_score": adjustment_result.get("adjusted_min_score"),
                        "original_doc_count": adjustment_result.get("original_doc_count"),
                        "filtered_doc_count": adjustment_result.get("filtered_doc_count"),
                        "compression_applied": adjustment_result.get("compression_applied", False),
                        "doc_compression_needed": adjustment_result.get("doc_compression_needed", False),
                        "max_context_chars": max_context_chars,
                        "time_ms": _adaptive_budget_time_ms,
                    }

                    logger.info(
                        f"[ADAPTIVE_MEMORY] Budget adjusted: tightness={adjustment_result.get('tightness', 0):.2f}, "
                        f"docs={len(retrieved_docs)}, budget={max_context_chars} chars",
                        extra={
                            "tightness": adjustment_result.get("tightness"),
                            "filtered_doc_count": len(retrieved_docs),
                            "doc_budget_tokens": doc_budget_tokens,
                            "compression_applied": adjustment_result.get("compression_applied", False),
                        }
                    )

                    # Skip the old budget calculation since adaptive memory handled it
                    context = self._augment(retrieved_docs, max_context_chars)

                except Exception as e:
                    logger.error(f"Adaptive memory adjustment failed: {e}", exc_info=True)
                    # Fallback to standard budget calculation
                    adaptive_memory_enabled = False
            
            # Step 2: Standard Augment (if neither Context Governor nor adaptive memory used)
            if not context_governor_enabled and not adaptive_memory_enabled:
                # v1.10.2: Dynamic budget calculation based on provider context window
                context_limit_tokens = config.get("context_limit_tokens")
                if context_limit_tokens:
                    # Use dynamic budget calculation
                    max_context_chars = self._calculate_document_budget(
                        context_limit_tokens=context_limit_tokens,
                        query=query,
                        config=config,
                        web_context=web_context,
                        conversation_context=conversation_context,
                    )
                    logger.info(
                        f"Dynamic context budget: {max_context_chars} chars "
                        f"(from {context_limit_tokens} tokens)",
                        extra={
                            "context_limit_tokens": context_limit_tokens,
                            "max_context_chars": max_context_chars,
                        },
                    )
                else:
                    # Fallback to static config (FIX-006 v1.8.2)
                    max_context_chars = config.get("max_context_chars", 8000)

                context = self._augment(retrieved_docs, max_context_chars)

            # Step 2.5: Add web context if provided (ROADMAP v1.5.0 - FEAT-WEB-001)
            if web_context:
                context = self._augment_with_web_context(context, web_context)

            # v6.8.1: End context packing timer
            _phase_timings["t_context_packing_ms"] = round((time.perf_counter() - _t_context_packing_start) * 1000, 2)
        else:
            # Pure LLM mode: No retrieval, but can still use web context
            logger.info("Pure LLM mode: skipping retrieval step")

            # Step 1.5 (Pure LLM): Adaptive Budget Management (v4.0.0)
            # Apply intelligent memory management even without documents
            adaptive_memory_enabled = config.get("adaptive_memory_enabled", False)
            adaptive_memory = await self._get_adaptive_memory()
            
            if adaptive_memory_enabled and adaptive_memory and hasattr(adaptive_memory, "adjust_for_pure_chat"):
                try:
                    logger.info("[ADAPTIVE_MEMORY] Adjusting budget for Pure LLM mode")
                    
                    # Call adaptive budget manager for pure chat
                    adjustment_result = await adaptive_memory.adjust_for_pure_chat(
                        query=query,
                        conversation_context=conversation_context,
                        turn_count=turn_count,
                        chat_config=config
                    )
                    
                    # Apply adjustments
                    # Update conversation context (potentially compressed)
                    conversation_context = adjustment_result.get("conversation_context")
                    
                    # v4.0.0: Capture full adjustment result for debug panel
                    _adaptive_budget_debug = {
                        "mode": "pure_chat",
                        "tightness": adjustment_result.get("tightness"),
                        "context_window": adjustment_result.get("context_window"),
                        "memory_tokens": adjustment_result.get("memory_tokens"),
                        "response_budget_tokens": adjustment_result.get("response_budget_tokens"),
                        "fixed_overhead_tokens": adjustment_result.get("fixed_overhead_tokens"),
                        "compression_applied": adjustment_result.get("compression_applied", False),
                        "utilization_pct": adjustment_result.get("utilization_pct", 0),
                    }

                    logger.info(
                        f"[ADAPTIVE_MEMORY] Pure LLM budget adjusted: tightness={adjustment_result.get('tightness', 0):.2f}, "
                        f"memory={adjustment_result.get('memory_tokens', 0)} tokens",
                        extra={
                            "tightness": adjustment_result.get("tightness"),
                            "memory_tokens": adjustment_result.get("memory_tokens"),
                            "compression_applied": adjustment_result.get("compression_applied", False),
                            "utilization_pct": adjustment_result.get("utilization_pct", 0),
                        }
                    )
                    
                except Exception as e:
                    logger.error(f"Adaptive memory adjustment failed in Pure LLM mode: {e}", exc_info=True)
            
            # Add web context if provided
            if web_context:
                context = f"Web search results:\n{web_context}"
            else:
                context = ""

        # v4.0.0: Snapshot for Context Debug Panel - save RAG-only context before conversation augmentation
        _rag_context_only = context
        _conversation_context_raw = conversation_context

        # Step 2.6: Add conversation context if provided (ROADMAP v1.5.0 - FEAT-MEM-001)
        if conversation_context:
            context = self._augment_with_conversation_context(
                context, conversation_context
            )

        # v4.0.0: Combined context after RAG + conversation merge
        _combined_context = context

        # Step 3: Generate - Use LLM to generate answer
        # In Pure LLM mode, context may be empty but we still call LLM
        # v3.7.0: Pass execution_plan for Context Governor system prompt injection
        tool_settings = config.get("_tool_settings")
        seen_chunk_ids = None
        if tool_settings and tool_settings.get("enabled") and retrieved_docs:
            seen_chunk_ids = {
                doc.get("metadata", {}).get("chunk_id")
                for doc in retrieved_docs
                if doc.get("metadata", {}).get("chunk_id")
            }
        _t_generation_start = time.perf_counter()
        if tool_settings and tool_settings.get("enabled") and collections:
            generate_result = await self._generate_with_tools(
                query,
                context,
                config,
                tool_settings,
                execution_plan=execution_plan,
                seen_chunk_ids=seen_chunk_ids,
            )
        else:
            generate_result = await self._generate(
                query, context, config, execution_plan=execution_plan
            )
        _phase_timings["t_generation_total_ms"] = round((time.perf_counter() - _t_generation_start) * 1000, 2)
        # v6.8.1: Merge sub-phase timings from generate functions
        _phase_timings.update(generate_result.get("_phase_timings", {}))
        answer = generate_result["text"]
        prompt_debug = generate_result.get("prompt_debug", {})

        # Extract sources metadata (empty in Pure LLM mode)
        sources = self._extract_sources(retrieved_docs) if retrieved_docs else []

        # Strip internal keys (non-serializable modules, etc.) from config_used
        config_used = {k: v for k, v in config.items() if not k.startswith("_")}
        result = {"answer": answer, "sources": sources, "config_used": config_used}

        if return_debug:
            result["debug"] = {
                "retrieved_chunks": [
                    {
                        "text": doc.get("text", ""),
                        "score": doc.get("score", 0.0),
                        "metadata": doc.get("metadata", {}),
                        "collection": doc.get("collection", "unknown"),
                        "query_source": doc.get("query_source"),
                    }
                    for doc in retrieved_docs
                ],
                "has_conversation_context": conversation_context is not None,
                # v2.2.2: Include HyDE document for debug visibility
                "hyde_document": hyde_document,
                "search_queries": retrieve_result.get("search_queries"),
                "expanded_queries": retrieve_result.get("expanded_queries"),
                "investigative_questions": retrieve_result.get("investigative_questions"),
                "optimization_applied": retrieve_result.get("optimization_applied"),
                "optimization_time_ms": retrieve_result.get("optimization_time_ms"),
                "filters_applied": retrieve_result.get("filters_applied"),
                "filter_entities": retrieve_result.get("filter_entities"),
                "filter_confidence": retrieve_result.get("filter_confidence"),
                "filter_raw_response": retrieve_result.get("filter_raw_response"),
                # v4.0.0: Enrichment step stats
                "rerank_stats": retrieve_result.get("rerank_stats"),
                "fusion_stats": retrieve_result.get("fusion_stats"),
                "dedup_stats": retrieve_result.get("dedup_stats"),
                "compression_stats": retrieve_result.get("compression_stats"),
                "enrichment_flags": retrieve_result.get("enrichment_flags"),
                # v6.0.2: Per-step timing
                "hyde_time_ms": retrieve_result.get("hyde_time_ms"),
                "expansion_time_ms": retrieve_result.get("expansion_time_ms"),
                "investigative_time_ms": retrieve_result.get("investigative_time_ms"),
                "filter_time_ms": retrieve_result.get("filter_time_ms"),
                "metadata_time_ms": retrieve_result.get("metadata_time_ms"),
                "adaptive_budget_time_ms": retrieve_result.get("adaptive_budget_time_ms"),
                # v3.7.0: Include Context Governor ExecutionPlan info
                "context_governor": {
                    "enabled": execution_plan is not None,
                    "tightness": execution_plan.tightness if execution_plan else None,
                    "context_strategy": (execution_plan.context_strategy.value if hasattr(execution_plan.context_strategy, 'value') else str(execution_plan.context_strategy)) if execution_plan else None,
                    "rag_top_k": execution_plan.rag_top_k if execution_plan else None,
                    "similarity_threshold": execution_plan.similarity_threshold if execution_plan else None,
                    "doc_budget_tokens": execution_plan.doc_budget_tokens if execution_plan else None,
                    "memory_budget_tokens": execution_plan.memory_budget_tokens if execution_plan else None,
                    "system_modifier_injected": bool(execution_plan and execution_plan.system_instruction_modifier),
                    "response_style": (execution_plan.response_style.value if hasattr(execution_plan.response_style, 'value') else str(execution_plan.response_style)) if execution_plan else None,
                    "suggested_temperature": execution_plan.suggested_temperature if execution_plan else None,
                    # v6.3.1: Provider and context_window for observability
                    "provider_name": execution_plan.provider_name if execution_plan else None,
                    "context_window": execution_plan.context_window if execution_plan else None,
                    "reserved_output_tokens": execution_plan.reserved_output_tokens if execution_plan else None,
                    # v3.7.1: Include 4-layer constraint info
                    "constraints_applied": execution_plan.constraints_applied if execution_plan else [],
                    "constraints_source": execution_plan.constraints_source if execution_plan else None,
                } if execution_plan else None,
                # v4.0.0: Adaptive Budget Manager debug
                "adaptive_budget_debug": _adaptive_budget_debug,
                "tool_calls": generate_result.get("tool_calls_debug"),
                # v4.0.0: Context Debug Panel - prompt debug info
                "prompt_debug": {
                    **prompt_debug,
                    "rag_context_only": _rag_context_only,
                    "conversation_context": _conversation_context_raw,
                    "combined_context": _combined_context,
                    "query_as_sent": query,
                },
                "context_chars": len(context),
            }

        # v6.8.1: Finalize e2e timing and emit structured phase breakdown
        _phase_timings["t_total_e2e_ms"] = round((time.perf_counter() - _t_e2e_start) * 1000, 2)
        logger.info(
            "[PHASE-TIMING] RAG pipeline phase breakdown",
            extra=_phase_timings,
        )
        if return_debug and "debug" in result:
            result["debug"]["phase_timings"] = _phase_timings

        logger.info(
            "RAG pipeline completed",
            extra={"answer_length": len(answer), "sources_count": len(sources)},
        )

        return result

    async def _retrieve(
        self,
        query: str,
        collections: List[str],
        config: Dict[str, Any],
        pipeline_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve with full enrichment pipeline support (v1.8.0).

        Pipeline order:
        1. Query Expansion (pre-retrieval, LLM-based)
        2. HyDE Generation (pre-retrieval, LLM-based)
        3. Vector Retrieval (Qdrant)
        4. Reranking (post-retrieval, GPU cross-encoder)
        5. Chunk Fusion (post-retrieval)
        6. Deduplication (post-retrieval)
        7. Compression (post-retrieval)
        8. Metadata Injection (post-retrieval)

        Args:
            query: User question
            collections: List of collection names to query
            config: RAG configuration with top_k and enrichment settings
            pipeline_config: Permission-aware enrichment overrides

        Returns:
            List of retrieved documents with text, score, and metadata
        """
        top_k = config.get("top_k", 5)
        enrichment = await self._get_enrichment()
        enrichment_config = config.get("enrichment", {}) or {}
        pipeline_config = pipeline_config or {}

        pipeline_steps: Dict[str, Optional[bool]] = {}
        pipeline_enabled_override = (
            pipeline_config.get("enabled") if pipeline_config else None
        )
        for step in pipeline_config.get("steps", []) or []:
            name = step.get("step") if isinstance(step, dict) else None
            if not name:
                continue
            pipeline_steps[name] = step.get("enabled")

        base_enrichment_enabled = enrichment_config.get("enabled", True)
        if pipeline_enabled_override is not None:
            base_enrichment_enabled = base_enrichment_enabled and bool(
                pipeline_enabled_override
            )
        
        # FIX-RC1 v3.7.1: Enhanced guard with retry mechanism
        # If enrichment is None, attempt one more resolution before giving up
        if enrichment is None and not self._enrichment_failed:
            logger.warning("[ENRICHMENT-GUARD] enrichment is None at _retrieve, attempting retry...")
            enrichment = await self._get_enrichment()
        
        enrichment_enabled = enrichment is not None and base_enrichment_enabled
        
        # FIX-RC1 v3.7.1: Explicit warning if enrichment unavailable affects pipeline
        if enrichment is None and base_enrichment_enabled:
            logger.warning(
                f"[ENRICHMENT-GUARD] enrichment_pipeline is None but config requires enrichment - "
                f"ALL enrichment steps will be disabled! (pipeline_enabled_override={pipeline_enabled_override})"
            )

        def _apply_step_flag(step_name: str, config_key: str, default_value: bool) -> bool:
            """
            Determine if a step is enabled.

            Priority:
            1. pipeline_steps (from 4-layer merged pipeline_config) - if present, use it
            2. enrichment_config (from rag_config) - fallback
            3. default_value - final fallback
            """
            if not enrichment_enabled:
                return False
            # pipeline_steps contains the 4-layer merged result from adapter._build_enrichment_pipeline_config
            override = pipeline_steps.get(step_name)
            if override is not None:
                return bool(override)
            # Fallback to enrichment_config
            return enrichment_config.get(config_key, default_value)

        hyde_enabled = _apply_step_flag("hyde", "hyde_enabled", False)
        expansion_enabled = _apply_step_flag("query_expansion", "query_expansion_enabled", False)
        investigative_enabled = _apply_step_flag("investigative", "investigative_enabled", False)
        filters_enabled = _apply_step_flag("query_filters", "query_filters_enabled", False)
        rerank_enabled = _apply_step_flag("rerank", "rerank_enabled", True)
        fusion_enabled = _apply_step_flag("fusion", "fusion_enabled", False)
        dedup_enabled = _apply_step_flag("dedup", "dedup_enabled", False)
        compression_enabled = _apply_step_flag("compression", "compression_enabled", False)

        # LOG RESOLVED FLAGS - Critical for debugging configuration propagation
        logger.info(
            f"[RAG_PIPELINE] Resolved enrichment flags: "
            f"hyde={hyde_enabled}, expansion={expansion_enabled}, "
            f"investigative={investigative_enabled}, rerank={rerank_enabled}",
            extra={
                "enrichment_enabled": enrichment_enabled,
                "hyde_enabled": hyde_enabled,
                "expansion_enabled": expansion_enabled,
                "investigative_enabled": investigative_enabled,
                "rerank_enabled": rerank_enabled,
                "fusion_enabled": fusion_enabled,
                "dedup_enabled": dedup_enabled,
                "compression_enabled": compression_enabled,
                "pipeline_steps_count": len(pipeline_steps),
            },
        )

        # === PRE-RETRIEVAL OPTIMIZATION & FILTERS ===

        hyde_document_generated = None
        expanded_queries: List[str] = []
        investigative_questions: List[str] = []
        optimization_applied: List[str] = []
        optimization_time_ms: Optional[float] = None

        # v4.0.0: Enrichment step stats for debug panels
        _rerank_stats: Optional[Dict[str, Any]] = None
        _fusion_stats: Optional[Dict[str, Any]] = None
        _dedup_stats: Optional[Dict[str, Any]] = None
        _compression_stats: Optional[Dict[str, Any]] = None
        # v6.0.2: Per-step timing for debug panels
        _hyde_time_ms: Optional[float] = None
        _expansion_time_ms: Optional[float] = None
        _investigative_time_ms: Optional[float] = None
        _filter_time_ms: Optional[float] = None
        _metadata_time_ms: Optional[float] = None
        _adaptive_budget_time_ms: Optional[float] = None

        search_queries = [query]
        if enrichment_enabled and hasattr(enrichment, "optimize_query"):
            try:
                logger.info(
                    f"[RAG_PIPELINE] Calling enrichment.optimize_query with: "
                    f"hyde_enabled={hyde_enabled}, expansion_enabled={expansion_enabled}, "
                    f"investigative_enabled={investigative_enabled}",
                    extra={
                        "hyde_enabled": hyde_enabled,
                        "expansion_enabled": expansion_enabled,
                        "investigative_enabled": investigative_enabled,
                    },
                )
                # v5.0: Pass rewrite_focus for investigation decomposition.
                # For vague queries ("entra nei dettagli"), investigation needs the
                # topic context (current_focus), not the raw query or keyword-stuffed
                # retrieval_query. HyDE and expansion correctly use retrieval_query.
                rewrite_focus = config.get("_rewrite_focus")
                optimization_result = await enrichment.optimize_query(
                    query=query,
                    hyde_enabled=hyde_enabled,
                    expansion_enabled=expansion_enabled,
                    investigative_enabled=investigative_enabled,
                    num_variants=enrichment_config.get("query_expansion_variants", 3),
                    num_questions=enrichment_config.get("investigative_num_questions", 5),
                    hyde_document_type=enrichment_config.get("hyde_document_type", "answer"),
                    rewrite_focus=rewrite_focus,
                )
                search_queries = optimization_result.get("search_queries") or [query]
                hyde_document_generated = optimization_result.get("hyde_document")
                expanded_queries = optimization_result.get("expanded_queries", [])
                investigative_questions = optimization_result.get("investigative_questions", [])
                optimization_applied = optimization_result.get("optimization_applied", [])
                optimization_time_ms = optimization_result.get("time_ms")
                # v6.0.2: Per-task timing from enrichment pipeline
                _hyde_time_ms = optimization_result.get("hyde_time_ms")
                _expansion_time_ms = optimization_result.get("expansion_time_ms")
                _investigative_time_ms = optimization_result.get("investigative_time_ms")
                logger.info(
                    "[RAG] Query optimization applied",
                    extra={
                        "optimizations": optimization_applied,
                        "queries_count": len(search_queries),
                        "time_ms": optimization_time_ms,
                    },
                )
            except Exception as e:
                logger.warning(
                    f"Query optimization failed, using original query only: {e}",
                    extra={"error_type": type(e).__name__},
                )
                search_queries = [query]
        else:
            search_queries = [query]

        # Deduplicate search queries while preserving order
        deduped_queries: List[str] = []
        seen_queries = set()
        for sq in search_queries:
            normalized = (sq or "").strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            deduped_queries.append(sq)
        if deduped_queries:
            search_queries = deduped_queries

        filters_result: Optional[Dict[str, Any]] = None
        qdrant_filters: Optional[Dict[str, Any]] = None
        if filters_enabled and hasattr(enrichment, "extract_filters"):
            try:
                _filter_t0 = time.perf_counter()
                filters_result = await enrichment.extract_filters(query=query)
                _filter_time_ms = round((time.perf_counter() - _filter_t0) * 1000, 2)
                qdrant_filters = filters_result.get("filters")
                if qdrant_filters:
                    logger.info(
                        "[RAG] Applying natural language filters",
                        extra={"filters": qdrant_filters},
                    )
            except Exception as e:
                logger.warning(
                    f"Natural language filter extraction failed: {e}",
                    extra={"error_type": type(e).__name__},
                )
                filters_result = None
                qdrant_filters = None

        # === VECTOR RETRIEVAL ===

        use_reranking = rerank_enabled
        
        # FIX-PERF-3 v3.7.1: Adaptive oversample factor
        # With many enrichment queries (10+), oversample_factor=4 creates too many candidates
        # (e.g., 10 queries × 4 = 40 per collection = 400-1200 total candidates to rerank)
        # Reduce oversample when many queries are active
        base_oversample_factor = enrichment_config.get("oversample_factor", 4)
        num_queries = len(search_queries)
        
        if num_queries <= 2:
            effective_oversample = base_oversample_factor  # Full oversample for few queries
        elif num_queries <= 5:
            effective_oversample = max(2, base_oversample_factor // 2)  # Half oversample
        else:
            effective_oversample = max(2, base_oversample_factor // 3)  # Minimal oversample
        
        fetch_k = top_k * effective_oversample if use_reranking else top_k
        
        logger.info(f"[RETRIEVAL-CONFIG] queries={num_queries}, base_oversample={base_oversample_factor}, "
                   f"effective_oversample={effective_oversample}, fetch_k={fetch_k}")

        all_docs = []
        seen_texts = set()  # Track unique texts across queries

        # FIX-PERF-2 v3.7.1: Parallelize Qdrant retrieval
        # Sequential N×M queries were taking 10-20s for 10 queries × 1-3 collections
        # Parallel execution reduces to ~1-2s (queries are independent)
        
        # Build all retrieval tasks
        retrieval_tasks = []
        task_metadata = []  # Track (query, collection) for each task
        
        for eq in search_queries:
            for collection in collections:
                task = self.qdrant.query_internal(
                    query_text=eq,
                    collection=collection,
                    top_k=fetch_k,
                    filters=qdrant_filters,
                )
                retrieval_tasks.append(task)
                task_metadata.append({
                    "query": eq,
                    "collection": collection,
                    "query_source": (
                        "original" if eq == query
                        else "hyde" if hyde_document_generated and eq == hyde_document_generated
                        else "investigative" if eq in investigative_questions
                        else "variant"
                    )
                })
        
        # Execute all retrieval tasks in parallel
        if retrieval_tasks:
            logger.info(f"[RETRIEVAL-PARALLEL] Executing {len(retrieval_tasks)} Qdrant queries in parallel "
                       f"({len(search_queries)} queries × {len(collections)} collections)")
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
            
            # Process results
            for metadata, result in zip(task_metadata, retrieval_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Error retrieving from collection {metadata['collection']}",
                        extra={"error": str(result), "error_type": type(result).__name__},
                    )
                    continue
                
                results_list = result.get("results", [])
                for item in results_list:
                    text = item.get("text", "")
                    # Avoid duplicates from multiple query variants
                    text_hash = hash(text[:200]) if text else 0
                    if text_hash in seen_texts:
                        continue
                    seen_texts.add(text_hash)
                    all_docs.append(
                        {
                            "collection": metadata["collection"],
                            "text": text,
                            "score": item.get("score", 0.0),
                            "metadata": item.get("metadata", {}),
                            "query_source": metadata["query_source"],
                            "embedding": item.get("vector"),  # v6.1.3: carry for dedup/fusion
                        }
                    )

        # FIX-FILTER-001: Zero-result fallback without NL filters
        # If NL filters produced 0 results, retry with original query only (no filters)
        # This prevents bad filter extraction from killing retrieval entirely
        if not all_docs and qdrant_filters:
            logger.warning(
                "[RAG] NL filters produced 0 results, retrying without filters",
                extra={"filters_applied": qdrant_filters, "queries_tried": len(search_queries)},
            )
            fallback_result = await self.qdrant.query_internal(
                query_text=query,
                collection=collections[0] if collections else "",
                top_k=fetch_k,
            )
            for item in fallback_result.get("results", []):
                text = item.get("text", "")
                text_hash = hash(text[:200]) if text else 0
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    all_docs.append({
                        "collection": collections[0] if collections else "",
                        "text": text,
                        "score": item.get("score", 0.0),
                        "metadata": item.get("metadata", {}),
                        "query_source": "filter_fallback",
                    })
            if all_docs:
                logger.info(f"[RAG] Filter fallback recovered {len(all_docs)} candidates")

        # Sort by vector score (descending)
        all_docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # FIX-PERF-4 v3.7.1: Cap maximum candidates before reranking
        # Prevents excessive reranking time when many queries × collections generate 600+ candidates
        # Cap at top_k * 8 to maintain quality while limiting computational cost
        MAX_CANDIDATES_FOR_RERANK = top_k * 8
        if use_reranking and len(all_docs) > MAX_CANDIDATES_FOR_RERANK:
            logger.info(f"[CANDIDATE-CAP] Reducing candidates from {len(all_docs)} to {MAX_CANDIDATES_FOR_RERANK} "
                       f"before reranking (top_k={top_k})")
            all_docs = all_docs[:MAX_CANDIDATES_FOR_RERANK]

        # v4.0.0: Build mappings to restore 'collection' after enrichment steps
        # Rerankers/fusion create new dicts without 'collection' field
        _collection_map_text = {}      # text[:200] → collection
        _collection_map_hash = {}      # content_hash → collection
        _collection_map_chunkid = {}   # chunk_id → collection
        for doc in all_docs:
            if "collection" not in doc:
                continue
            coll = doc["collection"]
            text_key = doc.get("text", "")[:200]
            if text_key:
                _collection_map_text[text_key] = coll
            meta = doc.get("metadata", {})
            if meta.get("content_hash"):
                _collection_map_hash[meta["content_hash"]] = coll
            if meta.get("chunk_id"):
                _collection_map_chunkid[meta["chunk_id"]] = coll

        logger.info(
            f"Retrieved {len(all_docs)} candidates from {len(collections)} collections",
            extra={
                "queries_used": len(search_queries),
                "collections": len(collections),
                "candidates": len(all_docs),
            },
        )

        # === POST-RETRIEVAL ENRICHMENT ===

        # 3. Reranking (GPU cross-encoder) - v3.7.0: Use rag_reranker if available
        _candidates_before_rerank = len(all_docs)
        use_reranking = rerank_enabled and all_docs
        if use_reranking:
            try:
                logger.info(
                    f"Reranking: {len(all_docs)} candidates",
                    extra={"query_length": len(query), "candidates": len(all_docs)},
                )

                # v3.7.0: Try rag_reranker first, fall back to enrichment's internal reranker
                reranker = await self._get_reranker()
                if reranker:
                    # Use dedicated rag_reranker module
                    logger.debug("Using dedicated rag_reranker module")
                    rerank_result = await reranker.rerank_internal(
                        query=query,
                        chunks=all_docs,
                        top_k=top_k * 2,  # Keep extra for subsequent steps
                        return_scores=True,
                    )
                elif enrichment:
                    # Fall back to enrichment's internal reranker
                    logger.debug("Using enrichment_pipeline's internal reranker")
                    rerank_result = await enrichment.rerank(
                        query=query,
                        chunks=all_docs,
                        top_k=top_k * 2,
                        return_scores=True,
                    )
                else:
                    # No reranker available
                    logger.warning("Reranking enabled but no reranker module available")
                    rerank_result = None

                if rerank_result:
                    reranked = rerank_result.get("reranked_chunks", [])
                    if reranked:
                        # v6.1.3: Carry embeddings through reranking for dedup/fusion
                        _emb_lookup = {d.get("text", "")[:200]: d.get("embedding") for d in all_docs if d.get("embedding") is not None}
                        if _emb_lookup:
                            for doc in reranked:
                                key = doc.get("text", doc.get("content", ""))[:200]
                                if key in _emb_lookup:
                                    doc["embedding"] = _emb_lookup[key]
                        # Add rerank metadata
                        for doc in reranked:
                            if "rerank_score" in doc:
                                doc["metadata"]["rerank_score"] = doc["rerank_score"]
                        all_docs = reranked
                        # v4.0.0: Capture rerank stats
                        _rerank_stats = {
                            "candidates_before": _candidates_before_rerank,
                            "candidates_after": len(reranked),
                            "model_used": rerank_result.get("model_used", "unknown"),
                            "time_ms": rerank_result.get("time_ms", 0),
                            "top_scores": [
                                round(doc.get("rerank_score", doc.get("metadata", {}).get("rerank_score", 0)), 4)
                                for doc in reranked[:5]
                            ],
                        }
                        logger.info(
                            f"Reranked: {len(reranked)} chunks selected",
                            extra={
                            "model": rerank_result.get("model_used", "unknown"),
                            "time_ms": rerank_result.get("time_ms", 0),
                        },
                    )
            except Exception as e:
                logger.warning(
                    f"Reranking failed, using vector scores: {e}",
                    extra={"error_type": type(e).__name__},
                )

        # 4. Chunk Fusion
        if fusion_enabled and all_docs:
            try:
                _fusion_t0 = time.perf_counter()
                fusion_result = await enrichment.fuse_chunks(
                    chunks=all_docs,
                    strategy="all",
                    overlap_threshold=enrichment_config.get(
                        "fusion_overlap_threshold", 0.3
                    ),
                    semantic_threshold=enrichment_config.get(
                        "fusion_semantic_threshold", 0.93
                    ),
                )
                _fusion_elapsed = round((time.perf_counter() - _fusion_t0) * 1000, 2)
                fused = fusion_result.get("fused_chunks", [])
                if fused:
                    chunks_before = fusion_result.get("chunks_before", len(all_docs))
                    all_docs = fused
                    # v4.0.0: Capture fusion stats
                    _fusion_stats = {
                        "chunks_before": chunks_before,
                        "chunks_after": len(fused),
                        "time_ms": _fusion_elapsed,
                    }
                    logger.info(
                        f"Fusion: {chunks_before} → {len(fused)} chunks",
                        extra={"before": chunks_before, "after": len(fused)},
                    )
            except Exception as e:
                logger.warning(
                    f"Chunk fusion failed: {e}",
                    extra={"error_type": type(e).__name__},
                )

        # 5. Deduplication
        if dedup_enabled and all_docs:
            try:
                _dedup_t0 = time.perf_counter()
                dedup_result = await enrichment.deduplicate(
                    chunks=all_docs,
                    method=enrichment_config.get("dedup_method", "semantic"),
                    similarity_threshold=enrichment_config.get("dedup_threshold", 0.95),
                )
                _dedup_elapsed = round((time.perf_counter() - _dedup_t0) * 1000, 2)
                unique = dedup_result.get("unique_chunks", [])
                if unique:
                    removed = dedup_result.get("duplicates_removed", 0)
                    all_docs = unique
                    # v4.0.0: Capture dedup stats
                    _dedup_stats = {
                        "duplicates_removed": removed,
                        "remaining": len(unique),
                        "method": enrichment_config.get("dedup_method", "semantic"),
                        "time_ms": _dedup_elapsed,
                    }
                    logger.info(
                        f"Deduplicated: removed {removed} duplicates",
                        extra={"removed": removed, "remaining": len(unique)},
                    )
            except Exception as e:
                logger.warning(
                    f"Deduplication failed: {e}",
                    extra={"error_type": type(e).__name__},
                )

        # Limit to top_k before compression
        all_docs = all_docs[:top_k]

        # 6. Compression
        if compression_enabled and all_docs:
            try:
                _compress_t0 = time.perf_counter()
                compress_result = await enrichment.compress_context(
                    query=query,
                    chunks=all_docs,
                    compression_ratio=enrichment_config.get("compression_ratio", 0.5),
                    method=enrichment_config.get("compression_method", "extractive"),
                )
                _compress_elapsed = round((time.perf_counter() - _compress_t0) * 1000, 2)
                compressed = compress_result.get("compressed_chunks", [])
                if compressed:
                    all_docs = compressed
                    # v4.0.0: Capture compression stats
                    _compression_stats = {
                        "original_tokens": compress_result.get("original_tokens", 0),
                        "compressed_tokens": compress_result.get("compressed_tokens", 0),
                        "method": enrichment_config.get("compression_method", "extractive"),
                        "ratio": enrichment_config.get("compression_ratio", 0.5),
                        "time_ms": _compress_elapsed,
                    }
                    logger.info(
                        f"Compressed: {compress_result.get('original_tokens', 0)} → "
                        f"{compress_result.get('compressed_tokens', 0)} tokens",
                        extra={
                            "original": compress_result.get("original_tokens", 0),
                            "compressed": compress_result.get("compressed_tokens", 0),
                        },
                    )
            except Exception as e:
                logger.warning(
                    f"Compression failed: {e}",
                    extra={"error_type": type(e).__name__},
                )

        # 7. Metadata Injection
        if (
            enrichment_enabled
            and enrichment_config.get("metadata_enabled", True)
            and all_docs
        ):
            try:
                # Parse metadata_types if it's a comma-separated string
                metadata_types = enrichment_config.get(
                    "metadata_types", "source,relevance,position,tokens"
                )
                if isinstance(metadata_types, str):
                    metadata_types = [t.strip() for t in metadata_types.split(",")]

                _meta_t0 = time.perf_counter()
                meta_result = await enrichment.inject_metadata(
                    chunks=all_docs,
                    metadata_types=metadata_types,
                )
                _metadata_time_ms = round((time.perf_counter() - _meta_t0) * 1000, 2)
                enriched = meta_result.get("enriched_chunks", [])
                if enriched:
                    all_docs = enriched
                    logger.debug(
                        f"Metadata injected: {metadata_types}",
                        extra={"types": metadata_types},
                    )
            except Exception as e:
                logger.warning(
                    f"Metadata injection failed: {e}",
                    extra={"error_type": type(e).__name__},
                )

        # v4.0.0: Restore 'collection' field lost during enrichment steps
        # (rerankers/fusion create new dicts without it)
        for doc in all_docs:
            if "collection" not in doc:
                meta = doc.get("metadata", {})
                # Try text prefix match first, then content_hash, then chunk_id
                text_key = doc.get("text", "")[:200]
                coll = (
                    _collection_map_text.get(text_key)
                    or _collection_map_hash.get(meta.get("content_hash", ""))
                    or _collection_map_chunkid.get(meta.get("chunk_id", ""))
                )
                if coll:
                    doc["collection"] = coll

        # v2.2.2+: Return dict with docs, optimization metadata, and filters for debug
        return {
            "docs": all_docs,
            "hyde_document": hyde_document_generated,
            "search_queries": search_queries,
            "expanded_queries": expanded_queries,
            "investigative_questions": investigative_questions,
            "optimization_applied": optimization_applied,
            "optimization_time_ms": optimization_time_ms,
            "filters_applied": qdrant_filters,
            "filter_entities": filters_result.get("entities") if filters_result else None,
            "filter_confidence": filters_result.get("confidence") if filters_result else None,
            "filter_raw_response": filters_result.get("raw_response") if filters_result else None,
            # v4.0.0: Enrichment step stats for debug panels
            "rerank_stats": _rerank_stats,
            "fusion_stats": _fusion_stats,
            "dedup_stats": _dedup_stats,
            "compression_stats": _compression_stats,
            # v4.0.0: Enrichment flags resolved
            "enrichment_flags": {
                "hyde_enabled": hyde_enabled,
                "expansion_enabled": expansion_enabled,
                "investigative_enabled": investigative_enabled,
                "rerank_enabled": rerank_enabled,
                "fusion_enabled": fusion_enabled,
                "dedup_enabled": dedup_enabled,
                "compression_enabled": compression_enabled,
                "filters_enabled": filters_enabled,
            },
            # v6.0.2: Per-step timing for debug panels
            "hyde_time_ms": _hyde_time_ms,
            "expansion_time_ms": _expansion_time_ms,
            "investigative_time_ms": _investigative_time_ms,
            "filter_time_ms": _filter_time_ms,
            "metadata_time_ms": _metadata_time_ms,
            "adaptive_budget_time_ms": _adaptive_budget_time_ms,
        }

    def _calculate_document_budget(
        self,
        context_limit_tokens: int,
        query: str,
        config: Dict[str, Any],
        web_context: Optional[str] = None,
        conversation_context: Optional[str] = None,
    ) -> int:
        """
        Calculate available character budget for document context.

        v1.10.2: Dynamic budget calculation based on provider context window.
        v1.8.6: chars_per_token and safety_margin are now configurable via .env
                (UBP_RAG__CHARS_PER_TOKEN, UBP_RAG__SAFETY_MARGIN)

        Subtracts overhead from system prompt, query, web/conversation context,
        and a safety margin to determine how much space is available for documents.

        Args:
            context_limit_tokens: Provider's context window in tokens
            query: User query
            config: RAG configuration with system_prompt template,
                    chars_per_token (default 3.0), safety_margin (default 1000)
            web_context: Optional web search results (already formatted)
            conversation_context: Optional conversation history (already formatted)

        Returns:
            Available character budget for document context
        """
        # v1.8.6: Read tuning parameters from config (externalized to .env)
        # - chars_per_token: 3.0 conservativo per IT/Code, 4.0 per EN
        # - safety_margin: buffer per tokenization variance
        chars_per_token = float(config.get("chars_per_token", 3.0))
        safety_margin = int(config.get("safety_margin", 1000))

        # Convert tokens to chars using configurable factor
        total_budget_chars = int(context_limit_tokens * chars_per_token)

        # Get system prompt template and calculate overhead
        system_prompt_template = config.get(
            "system_prompt",
            "Use the following context to answer the question:\n\n{context}",
        )
        # Template overhead = template length minus {context} placeholder
        template_overhead = len(system_prompt_template) - len("{context}")

        # Query overhead: "\n\nQuestion: {query}\n\nAnswer:"
        query_overhead = len("\n\nQuestion: ") + len(query) + len("\n\nAnswer:")

        # Web context overhead (with separator)
        web_overhead = 0
        if web_context:
            web_overhead = len("\n\n--- Web Search Results ---\n\n") + len(web_context)

        # Conversation context overhead (with separator)
        conversation_overhead = 0
        if conversation_context:
            conversation_overhead = (
                len("--- Previous Conversation ---\n\n")
                + len(conversation_context)
                + len("\n\n--- Current Context ---\n\n")
            )

        # Response budget (reserve space for LLM response)
        # v4.1.0 FIX-BUDGET-RATIO: Dynamic max_tokens based on provider context window.
        # effective_max_tokens = min(UBP_RAG__MAX_TOKENS, context_window * ratio)
        # This prevents reserving more output tokens than the provider can handle.
        # With ratio=0.30 and vLLM (8K): min(6000, 8192*0.30) = 2457
        # With ratio=0.30 and Grok (2M): min(6000, 2M*0.30)   = 6000 (cap prevails)
        raw_max_tokens = config.get("max_tokens", 1000)
        response_ratio = float(config.get("response_budget_ratio", 1.0))
        if context_limit_tokens and response_ratio < 1.0:
            max_response_tokens = min(
                int(raw_max_tokens),
                int(context_limit_tokens * response_ratio)
            )
        else:
            max_response_tokens = int(raw_max_tokens)

        # v6.3.1: FIX-ARCH-003 — Cap max_tokens to execution_plan.reserved_output_tokens.
        if execution_plan and hasattr(execution_plan, 'reserved_output_tokens'):
            max_response_tokens = min(max_response_tokens, execution_plan.reserved_output_tokens)

        response_budget = int(
            max_response_tokens * 4
        )  # ~4 chars per token for response

        # Calculate available budget for documents
        total_overhead = (
            template_overhead
            + query_overhead
            + web_overhead
            + conversation_overhead
            + response_budget
            + safety_margin
        )

        available_budget = total_budget_chars - total_overhead

        logger.debug(
            f"Context budget calculation: {total_budget_chars} total - {total_overhead} overhead = {available_budget} available",
            extra={
                "context_limit_tokens": context_limit_tokens,
                "chars_per_token": chars_per_token,
                "total_budget_chars": total_budget_chars,
                "template_overhead": template_overhead,
                "query_overhead": query_overhead,
                "web_overhead": web_overhead,
                "conversation_overhead": conversation_overhead,
                "response_budget": response_budget,
                "safety_margin": safety_margin,
                "available_budget": available_budget,
            },
        )

        # Ensure minimum budget (at least 1000 chars for at least 1 document)
        return max(available_budget, 1000)

    def _augment(
        self, retrieved_docs: List[Dict[str, Any]], max_context_chars: int = 8000
    ) -> str:
        """
        Build context string from retrieved documents with length limit.

        FIX-006 v1.8.2: Added max_context_chars to prevent prompt overflow.
        v1.10.2: max_context_chars should be calculated via _calculate_document_budget()
                 for provider-aware dynamic limits.
        v1.10.4: Enhanced logging for context saturation with dropped doc tracking.
        FIX-008 v1.8.5: Context Sanitization - normalize box-drawing chars to ASCII.
        v2.1.0: FIX-GROUNDING - Include source filename in chunk header for LLM grounding.
                Format changed from "[N] text" to "[N | FILENAME.md] text"

        Args:
            retrieved_docs: List of retrieved documents
            max_context_chars: Maximum context length in characters (default 8000)
                               Use _calculate_document_budget() for dynamic calculation.

        Returns:
            Context string with numbered documents and source files, truncated if necessary
        """
        if not retrieved_docs:
            return ""

        context_parts = []
        current_length = 0
        docs_included = 0
        docs_dropped = 0
        total_docs = len(retrieved_docs)

        for idx, doc in enumerate(retrieved_docs, 1):
            raw_text = doc.get("text", "").strip()
            if not raw_text:
                continue

            # FIX-008 v1.8.5: Sanitize text to normalize box-drawing chars and remove control chars
            text = sanitize_for_prompt(raw_text)

            # v2.1.0 FIX-GROUNDING: Extract source filename from metadata for LLM grounding
            # Priority: filename > doc_id > "unknown"
            # This enables the LLM to cite correct sources in [RIFERIMENTI] section
            metadata = doc.get("metadata", {})
            source_file = (
                metadata.get("filename") or metadata.get("doc_id") or "unknown"
            )

            # Format: [N | FILENAME.md] text...
            # This format allows the system prompt to instruct the LLM to only cite
            # files that appear in this format, preventing hallucination of file names
            formatted = f"[{idx} | {source_file}] {text}"
            formatted_length = len(formatted) + 2  # +2 for \n\n separator

            # Check if adding this chunk would exceed limit
            if current_length + formatted_length > max_context_chars:
                remaining = max_context_chars - current_length
                if remaining > 200:  # Minimum useful content
                    # Truncate the text to fit
                    header = f"[{idx} | {source_file}] "
                    truncate_at = (
                        remaining - len(header) - 50
                    )  # Reserve space for "..."
                    if truncate_at > 0:
                        truncated_text = text[:truncate_at] + "... [truncated]"
                        context_parts.append(f"{header}{truncated_text}")
                        docs_included += 1

                # Count remaining docs as dropped
                docs_dropped = total_docs - docs_included

                # LOG CONTEXT SATURATION (v1.10.4 SAFETY CHECK)
                logger.warning(
                    f"CONTEXT SATURATION: Dropped {docs_dropped} docs due to budget limit",
                    extra={
                        "max_chars": max_context_chars,
                        "docs_included": docs_included,
                        "docs_dropped": docs_dropped,
                        "total_docs": total_docs,
                        "saturation_point": idx,
                        "current_length": current_length,
                    },
                )
                break

            context_parts.append(formatted)
            current_length += formatted_length
            docs_included += 1

        # Log success case (all docs fit)
        if docs_dropped == 0 and docs_included > 0:
            logger.debug(
                f"Context built: {docs_included}/{total_docs} docs, {current_length} chars",
                extra={
                    "docs_included": docs_included,
                    "total_docs": total_docs,
                    "context_length": current_length,
                    "max_chars": max_context_chars,
                    "utilization_pct": round(
                        current_length / max_context_chars * 100, 1
                    ),
                },
            )

        return "\n\n".join(context_parts)

    def _augment_with_web_context(self, kb_context: str, web_context: str) -> str:
        """
        Augment knowledge base context with web search results.

        ROADMAP v1.5.0 - FEAT-WEB-001

        Args:
            kb_context: Context from knowledge base retrieval
            web_context: Pre-formatted web search results

        Returns:
            Combined context with both KB and web information
        """
        if not web_context:
            return kb_context

        combined = kb_context
        if kb_context:
            combined += "\n\n--- Web Search Results ---\n\n"
        combined += web_context

        logger.info(
            "Augmented context with web search results",
            extra={
                "kb_context_length": len(kb_context),
                "web_context_length": len(web_context),
                "combined_length": len(combined),
            },
        )

        return combined

    def _augment_with_conversation_context(
        self, current_context: str, conversation_context: str
    ) -> str:
        """
        Augment current context with conversation history.

        ROADMAP v1.5.0 - FEAT-MEM-001

        Args:
            current_context: Current context from KB + web
            conversation_context: Formatted conversation history

        Returns:
            Combined context with conversation history prepended
        """
        if not conversation_context:
            return current_context

        # Prepend conversation history so LLM has context of prior turns
        # v5.0: Explicit separation to prevent LLM treating memory as documentation
        combined = "=== CONTESTO CONVERSAZIONE PRECEDENTE ===\n"
        combined += "Usa per continuita'. NON trattare come documentazione da analizzare.\n\n"
        combined += conversation_context
        combined += "\n\n=== FINE CONTESTO CONVERSAZIONE ===\n\n"
        combined += current_context

        logger.info(
            "Augmented context with conversation history",
            extra={
                "current_context_length": len(current_context),
                "conversation_context_length": len(conversation_context),
                "combined_length": len(combined),
            },
        )

        return combined

    async def _generate(
        self,
        query: str,
        context: str,
        config: Dict[str, Any],
        execution_plan: Optional["ExecutionPlan"] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using LLM with augmented context.

        v3.7.0: Added execution_plan for Context Governor integration.
        v4.0.0: Returns dict with text + prompt_debug for Context Debug Panel.

        Args:
            query: User question
            context: Augmented context from retrieved documents
            config: RAG configuration with model, temperature, system_prompt
            execution_plan: Optional ExecutionPlan from Context Governor (v3.7.0)

        Returns:
            Dict with "text" (answer) and "prompt_debug" (debug metadata)
        """
        # Build system prompt with context
        system_prompt_template = config.get(
            "system_prompt",
            "Use the following context to answer the question:\n\n{context}",
        )

        system_prompt = system_prompt_template.replace("{context}", context)
        
        # v3.7.0: Inject Context Governor system_instruction_modifier
        # This guides the LLM to adjust response style based on tightness
        if execution_plan and execution_plan.system_instruction_modifier:
            # Prepend the modifier directive to guide response style
            modifier = execution_plan.system_instruction_modifier
            system_prompt = f"{modifier}\n\n{system_prompt}"
            
            logger.info(
                "[CONTEXT-GOVERNOR] Injected system_instruction_modifier into prompt",
                extra={
                    "modifier": modifier[:100] + "..." if len(modifier) > 100 else modifier,
                    "tightness": execution_plan.tightness,
                    "response_style": execution_plan.response_style.value if hasattr(execution_plan.response_style, 'value') else str(execution_plan.response_style),
                }
            )

        # Build full prompt
        # v5.0: Use original user query for LLM prompt (natural language),
        # while retrieval used the expanded query
        user_query = config.get("_original_user_query", query)
        full_prompt = f"{system_prompt}\n\nQuestion: {user_query}\n\nAnswer:"

        # v3.7.0: Use execution_plan suggested temperature if available
        temperature = config.get("temperature", 0.7)
        if execution_plan and execution_plan.suggested_temperature is not None:
            temperature = execution_plan.suggested_temperature
            logger.debug(f"[CONTEXT-GOVERNOR] Using suggested temperature: {temperature}")

        # v4.0.0: Build prompt debug metadata for Context Debug Panel
        # v4.1.0: Added effective_max_tokens and response_budget_ratio
        _prompt_debug = {
            "full_prompt_sent": full_prompt,
            "rag_context_length_chars": len(context),
            "full_prompt_length_chars": len(full_prompt),
            "temperature_used": temperature,
            "model_override": config.get("model"),
            "max_tokens_raw": config.get("max_tokens", 1000),
            "response_budget_ratio": config.get("response_budget_ratio"),
            "context_limit_tokens": config.get("context_limit_tokens"),
        }

        try:
            # Generate using LLM
            # FIX-MODEL v1.8.2: Let each LLM module use its own default model
            # Only pass model if explicitly configured via UBP_RAG__MODEL
            # This allows each provider to use its canonical default:
            #   - Ollama: UBP_PROVIDER_OLLAMA__DEFAULT_MODEL
            #   - vLLM: UBP_PROVIDER_VLLM__DEFAULT_MODEL
            #   - Grok: UBP_PROVIDER_GROK__DEFAULT_MODEL
            model_override = config.get("model")

            # v4.1.0 FIX-BUDGET-RATIO: Cap max_tokens to provider context window.
            # Same logic as _calculate_document_budget() to stay consistent.
            raw_max_tokens = int(config.get("max_tokens", 1000))
            response_ratio = float(config.get("response_budget_ratio", 1.0))
            context_limit = config.get("context_limit_tokens", 0)
            if context_limit and response_ratio < 1.0:
                effective_max_tokens = min(raw_max_tokens, int(context_limit * response_ratio))
            else:
                effective_max_tokens = raw_max_tokens

            _prompt_debug["max_tokens_effective"] = effective_max_tokens

            generate_kwargs = {
                "prompt": full_prompt,
                "temperature": temperature,
                "max_tokens": effective_max_tokens,
            }

            # Only pass model if explicitly set (not empty/None)
            if model_override:
                generate_kwargs["model"] = model_override

            _t_llm_start = time.perf_counter()
            result = await self.llm.generate(**generate_kwargs)
            _t_llm_ms = round((time.perf_counter() - _t_llm_start) * 1000, 2)

            # inference_ollama_grok.generate() returns:
            # {"text": "...", "model": "...", "provider": "...", "metadata": {...}}
            # OR on error: {"error": "...", "model": "...", ...}
            # v4.1.0 FIX-HA-001: Re-raise LLM errors so _chat_with_fallback
            # can catch them and try the fallback provider (Grok).
            # Previously these were swallowed, returning a soft error string
            # that looked like a "successful" response to the HA layer.
            # P1 defensive guard: a non-conforming provider returning a bare
            # string would cause AttributeError on .get(). Normalise here so
            # the HA fallback path receives a proper RuntimeError instead of a
            # confusing AttributeError that hides the real provider failure.
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"LLM generate() returned unexpected type "
                    f"{type(result).__name__!r}: {result!r}"
                )
            if result.get("error"):
                error_msg = result.get("error")
                logger.error(f"LLM generation failed: {error_msg}")
                raise RuntimeError(f"LLM generation error: {error_msg}")

            generated_text = result.get("text", "")
            _gen_timings = {"t_generation_llm_ms": _t_llm_ms}
            if generated_text:
                return {"text": generated_text, "prompt_debug": _prompt_debug, "_phase_timings": _gen_timings}
            else:
                logger.warning("LLM returned empty response")
                return {"text": "I couldn't generate an answer.", "prompt_debug": _prompt_debug, "_phase_timings": _gen_timings}

        except Exception as e:
            logger.error(
                "Error generating answer with LLM",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            # v4.1.0 FIX-HA-001: Re-raise to let _chat_with_fallback trigger
            # the fallback provider instead of returning a soft error.
            raise

    async def _generate_with_tools(
        self,
        query: str,
        context: str,
        config: Dict[str, Any],
        tool_settings: Dict[str, Any],
        execution_plan: Optional["ExecutionPlan"] = None,
        seen_chunk_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer using LLM with tool calling support.

        FEAT-TOOL-001: Allows the Architect agent to expand context via tools.
        """
        system_prompt_template = config.get(
            "system_prompt",
            "Use the following context to answer the question:\n\n{context}",
        )

        system_prompt = system_prompt_template.replace("{context}", context)

        # FEAT-TOOL-001: inject tool instructions for prompts that lack them
        # (e.g. RAG chat). Architect already has its own section → skip.
        if "search_knowledge_base" not in system_prompt:
            from .tool_definitions import build_tool_prompt_section

            tool_section = build_tool_prompt_section(
                int(tool_settings.get("max_iterations", 2)),
                has_web_search=bool(tool_settings.get("web_search_available")),
            )
            system_prompt = f"{system_prompt}\n{tool_section}"

        if execution_plan and execution_plan.system_instruction_modifier:
            modifier = execution_plan.system_instruction_modifier
            system_prompt = f"{modifier}\n\n{system_prompt}"
            logger.info(
                "[CONTEXT-GOVERNOR] Injected system_instruction_modifier into prompt",
                extra={
                    "modifier": modifier[:100] + "..." if len(modifier) > 100 else modifier,
                    "tightness": execution_plan.tightness,
                    "response_style": execution_plan.response_style.value if hasattr(execution_plan.response_style, 'value') else str(execution_plan.response_style),
                },
            )

        user_query = config.get("_original_user_query", query)
        full_prompt = f"{system_prompt}\n\nQuestion: {user_query}\n\nAnswer:"

        temperature = config.get("temperature", 0.7)
        if execution_plan and execution_plan.suggested_temperature is not None:
            temperature = execution_plan.suggested_temperature
            logger.debug(f"[CONTEXT-GOVERNOR] Using suggested temperature: {temperature}")

        raw_max_tokens = int(config.get("max_tokens", 1000))
        response_ratio = float(config.get("response_budget_ratio", 1.0))
        context_limit = config.get("context_limit_tokens", 0)
        if context_limit and response_ratio < 1.0:
            effective_max_tokens = min(raw_max_tokens, int(context_limit * response_ratio))
        else:
            effective_max_tokens = raw_max_tokens

        # v6.3.1: FIX-ARCH-003 — Cap max_tokens to execution_plan.reserved_output_tokens.
        if execution_plan and hasattr(execution_plan, 'reserved_output_tokens'):
            effective_max_tokens = min(effective_max_tokens, execution_plan.reserved_output_tokens)

        _prompt_debug = {
            "full_prompt_sent": full_prompt,
            "rag_context_length_chars": len(context),
            "full_prompt_length_chars": len(full_prompt),
            "temperature_used": temperature,
            "model_override": config.get("model"),
            "max_tokens_raw": raw_max_tokens,
            "response_budget_ratio": config.get("response_budget_ratio"),
            "context_limit_tokens": config.get("context_limit_tokens"),
            "max_tokens_effective": effective_max_tokens,
        }

        tools = get_tool_definitions(tool_settings)
        if not tools:
            return await self._generate(query, context, config, execution_plan=execution_plan)

        # v6.8.1: Phase timing for tool-calling path
        _t_tool_resolve_start = time.perf_counter()

        executor = ArchitectToolExecutor(
            qdrant_module=self.qdrant,
            settings=tool_settings,
            collections=config.get("_tool_collections", []),
            web_search_module=config.get("_web_module"),
        )

        max_iterations = int(tool_settings.get("max_iterations", 1))
        max_context_expansion_kb = float(tool_settings.get("max_context_expansion_kb", 8))
        seen_ids = set(seen_chunk_ids or [])
        tool_calls_debug: List[Dict[str, Any]] = []
        expansion_kb = 0.0

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        answer = ""
        tool_choice = "auto"
        _t_tool_llm_chat_total = 0.0
        _t_tool_execution_total = 0.0
        _tool_roundtrips = 0
        for iteration in range(max_iterations):
            if iteration == max_iterations - 1:
                tool_choice = "none"

            chat_kwargs: Dict[str, Any] = {
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": effective_max_tokens,
                "tool_choice": tool_choice,
            }

            # v6.8.0: Dual-LLM orchestration for tool calling.
            # Strategy: Try dedicated tool_llm (e.g. inference_vllm for vllm_remote)
            # with the configured tool_provider. If unavailable or unsupported,
            # fall back to self.llm (main generation LLM) with config["provider"].
            generation_provider = config.get("provider")
            tool_provider = tool_settings.get("provider")
            tool_llm = await self._get_tool_llm()
            _t_resolve_tool_llm_ms = round((time.perf_counter() - _t_tool_resolve_start) * 1000, 2)

            effective_llm = None
            effective_provider = None
            fallback_reason = None

            if tool_provider and tool_llm:
                # Dual-LLM path: use dedicated tool LLM with its native provider
                chat_kwargs["provider"] = tool_provider
                effective_llm = tool_llm
                effective_provider = tool_provider
                logger.info(
                    f"[TOOL-ROUTING] dual-LLM: tool_provider_requested={tool_provider}, "
                    f"tool_provider_effective={tool_provider}, using dedicated tool_llm"
                )
            elif generation_provider:
                # Single-LLM path: use main LLM with generation provider
                chat_kwargs["provider"] = generation_provider
                effective_llm = self.llm
                effective_provider = generation_provider
                fallback_reason = "no_dedicated_tool_llm" if tool_provider else "no_tool_provider_configured"
                if tool_provider and tool_provider != generation_provider:
                    logger.info(
                        f"[TOOL-ROUTING] tool_provider_requested={tool_provider}, "
                        f"tool_provider_effective={generation_provider}, "
                        f"reason={fallback_reason}"
                    )
            else:
                effective_llm = self.llm
                fallback_reason = "no_provider_configured"

            _t_chat_call_start = time.perf_counter()
            result = await effective_llm.chat_with_tools(**chat_kwargs)
            _t_tool_llm_chat_total += time.perf_counter() - _t_chat_call_start
            _tool_roundtrips += 1

            if result.get("tools_supported") is False:
                # If tool_llm was tried and failed, retry with main LLM
                if effective_llm is not self.llm and generation_provider:
                    logger.info(
                        f"[TOOL-ROUTING] tool_provider_requested={tool_provider}, "
                        f"tool_provider_effective={generation_provider}, "
                        f"reason=cross_module_unsupported (falling back to main LLM)"
                    )
                    chat_kwargs["provider"] = generation_provider
                    _t_fallback_start = time.perf_counter()
                    result = await self.llm.chat_with_tools(**chat_kwargs)
                    _t_tool_llm_chat_total += time.perf_counter() - _t_fallback_start
                    _tool_roundtrips += 1
                    if result.get("tools_supported") is False:
                        return await self._generate(query, context, config, execution_plan=execution_plan)
                else:
                    return await self._generate(query, context, config, execution_plan=execution_plan)

            if result.get("text"):
                answer = result["text"]
                break

            tool_calls = result.get("tool_calls") or []
            if not tool_calls:
                answer = "I couldn't generate an answer."
                break

            force_text_next = False
            for tool_call in tool_calls:
                tool_name = (
                    tool_call.get("function", {}).get("name")
                    if isinstance(tool_call, dict)
                    else None
                ) or (tool_call.get("name") if isinstance(tool_call, dict) else None)
                if not tool_name:
                    raise ValueError("Tool call missing name")

                arguments_raw = (
                    tool_call.get("function", {}).get("arguments")
                    if isinstance(tool_call, dict)
                    else None
                ) or (tool_call.get("arguments") if isinstance(tool_call, dict) else None)
                if isinstance(arguments_raw, str):
                    try:
                        arguments = json.loads(arguments_raw)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Tool call arguments not valid JSON",
                            extra={"error": str(e), "tool": tool_name},
                        )
                        arguments = {}
                elif isinstance(arguments_raw, dict):
                    arguments = arguments_raw
                else:
                    arguments = {}

                tool_call_id = (
                    tool_call.get("id") if isinstance(tool_call, dict) else None
                ) or str(uuid.uuid4())

                try:
                    _t_exec_start = time.perf_counter()
                    tool_result = await executor.execute_tool_call(
                        tool_name=tool_name,
                        arguments=arguments,
                        seen_chunk_ids=seen_ids,
                    )
                    _t_tool_execution_total += time.perf_counter() - _t_exec_start
                except Exception as e:
                    logger.warning(
                        "[TOOL] execute_tool_call failed, injecting error result",
                        extra={"tool": tool_name, "error": str(e), "query": arguments.get("query", "")},
                    )
                    tool_result = {
                        "tool_name": tool_name,
                        "query": arguments.get("query", ""),
                        "reason": arguments.get("reason", ""),
                        "chunks_found": 0,
                        "chunk_ids": [],
                        "chunks": [],
                        "latency_ms": 0,
                        "content": f"[TOOL_RESULT | {tool_name}]\nError: {e}\n[END_TOOL_RESULT]",
                        "error": str(e),
                    }
                tool_result["tool_call_id"] = tool_call_id
                tool_calls_debug.append(tool_result)

                expansion_kb += len(tool_result.get("content", "")) / 1024
                if expansion_kb > max_context_expansion_kb:
                    tool_choice = "none"
                    force_text_next = True

                tool_call_payload = dict(tool_call) if isinstance(tool_call, dict) else {}
                tool_call_payload.setdefault("id", tool_call_id)
                messages.append(
                    {"role": "assistant", "content": None, "tool_calls": [tool_call_payload]}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result.get("content", ""),
                    }
                )
                seen_ids.update(tool_result.get("chunk_ids", []))

                if force_text_next:
                    break

        if not answer:
            answer = "I couldn't generate an answer."

        return {
            "text": answer,
            "prompt_debug": _prompt_debug,
            "tool_calls_debug": tool_calls_debug,
            "_phase_timings": {
                "t_resolve_tool_llm_ms": _t_resolve_tool_llm_ms,
                "t_tool_llm_chat_total_ms": round(_t_tool_llm_chat_total * 1000, 2),
                "t_tool_execution_total_ms": round(_t_tool_execution_total * 1000, 2),
                "tool_roundtrips": _tool_roundtrips,
            },
        }

    def _extract_sources(
        self, retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract source metadata from retrieved documents.

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            List of source objects with collection, score, metadata
        """
        sources = []

        for doc in retrieved_docs:
            sources.append(
                {
                    "collection": doc.get("collection", "unknown"),
                    "score": round(doc.get("score", 0.0), 4),
                    "metadata": doc.get("metadata", {}),
                    "preview": doc.get("text", "")[:200] + "..."
                    if len(doc.get("text", "")) > 200
                    else doc.get("text", ""),
                }
            )

        return sources


class ACLManager:
    """
    Access Control List (ACL) Manager.

    Manages permissions for users and clients on RAG collections using Redis.
    Pure technical logic with no UBP dependencies.

    Redis keys follow NAMING_POLICY.md Section 7:
    - Pattern: ubp:{env}:rag:acl:{entity_type}:{entity_id}:{collection_id}
    """

    def __init__(self, redis_client, config: dict):
        """
        Initialize ACL manager.

        Args:
            redis_client: Redis client instance
            config: ACL configuration dict with default_access
        """
        self.redis = redis_client
        self.config = config
        self.default_access = config.get("default_access", "none")

        # Use RedisKeyManager for environment-aware key prefixes
        if REDIS_KEYS_AVAILABLE and get_key_manager is not None:
            self._key_manager = get_key_manager()
        else:
            self._key_manager = None

        # Legacy prefix for backward compatibility (used if key manager unavailable)
        # Schema compliant default: ubp:rag:acl (empty env = production)
        self._legacy_prefix = config.get("redis_key_prefix", "ubp:rag:acl")

    def _build_acl_key(
        self, entity_type: str, entity_id: str, collection_id: str
    ) -> str:
        """Build a Redis key for ACL permission with proper environment prefix."""
        if self._key_manager:
            return self._key_manager.key(
                "rag", "acl", entity_type, entity_id, collection_id
            )
        return f"{self._legacy_prefix}:{entity_type}:{entity_id}:{collection_id}"

    def _build_acl_pattern(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        collection_id: Optional[str] = None,
    ) -> str:
        """Build a Redis KEYS pattern for ACL with proper environment prefix."""
        if self._key_manager:
            parts = ["rag", "acl"]
            if entity_type:
                parts.append(entity_type)
                if entity_id:
                    parts.append(entity_id)
                    if collection_id:
                        parts.append(collection_id)
                    else:
                        parts.append("*")
                else:
                    parts.append("*")
            else:
                parts.append("*")
            return self._key_manager.key(*parts)
        # Legacy pattern
        if entity_type and entity_id and collection_id:
            return f"{self._legacy_prefix}:{entity_type}:{entity_id}:{collection_id}"
        elif entity_type and entity_id:
            return f"{self._legacy_prefix}:{entity_type}:{entity_id}:*"
        elif entity_type:
            return f"{self._legacy_prefix}:{entity_type}:*"
        return f"{self._legacy_prefix}:*"

    def _parse_acl_key(self, key_str: str) -> Optional[Dict[str, str]]:
        """
        Parse ACL key to extract entity_type, entity_id, collection_id.

        Works with both legacy (rag:acl:...) and new (ubp:...:rag:acl:...) formats.

        Returns:
            Dict with entity_type, entity_id, collection_id or None if invalid
        """
        # Find the 'rag:acl' marker and extract parts after it
        if ":rag:acl:" in key_str:
            # New format: ubp:...:rag:acl:entity_type:entity_id:collection_id
            parts = key_str.split(":rag:acl:")[1].split(":")
        elif key_str.startswith("rag:acl:"):
            # Legacy format: rag:acl:entity_type:entity_id:collection_id
            parts = key_str[8:].split(":")  # Skip 'rag:acl:'
        else:
            return None

        if len(parts) >= 3:
            return {
                "entity_type": parts[0],
                "entity_id": parts[1],
                "collection_id": parts[2],
            }
        return None

    async def set_permission(
        self, entity_type: str, entity_id: str, collection_id: str, access_level: str
    ) -> Dict[str, Any]:
        """
        Set access permission for an entity on a collection.

        Args:
            entity_type: 'user' or 'client'
            entity_id: user_id or client_id
            collection_id: Target collection
            access_level: 'read', 'write', or 'none'

        Returns:
            Status dict with success/error
        """
        # Validate inputs
        if entity_type not in ["user", "client"]:
            return {
                "status": "error",
                "message": "entity_type must be 'user' or 'client'",
            }

        if access_level not in ["read", "write", "none"]:
            return {
                "status": "error",
                "message": "access_level must be 'read', 'write', or 'none'",
            }

        try:
            acl_key = self._build_acl_key(entity_type, entity_id, collection_id)

            if access_level == "none":
                # Remove permission
                await self.redis.delete(acl_key)
                message = f"Permission removed for {entity_type} {entity_id} on {collection_id}"
            else:
                # Set permission
                await self.redis.set(acl_key, access_level)
                message = f"Permission set to '{access_level}' for {entity_type} {entity_id} on {collection_id}"

            return {"status": "success", "message": message}

        except Exception as e:
            return {"status": "error", "message": f"Error setting permission: {str(e)}"}

    async def get_permissions(
        self,
        collection_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get permissions (optionally filtered).

        Args:
            collection_id: Filter by collection
            entity_type: Filter by entity type ('user' or 'client')
            entity_id: Filter by entity ID

        Returns:
            Dict with permissions array and count
        """
        try:
            # Build search pattern using helper
            pattern = self._build_acl_pattern(entity_type, entity_id, collection_id)

            # Scan Redis for matching keys
            permissions = []
            async for key in self.redis.scan_iter(match=pattern):
                access_level = await self.redis.get(key)

                # Parse key using helper (handles both legacy and new formats)
                key_str = key if isinstance(key, str) else key.decode("utf-8")
                parsed = self._parse_acl_key(key_str)

                if parsed:
                    permissions.append(
                        {
                            "entity_type": parsed["entity_type"],
                            "entity_id": parsed["entity_id"],
                            "collection_id": parsed["collection_id"],
                            "access_level": access_level
                            if isinstance(access_level, str)
                            else access_level.decode("utf-8"),
                        }
                    )

            return {"permissions": permissions, "count": len(permissions)}

        except Exception as e:
            logger.error(f"Error getting permissions: {e}")
            return {"permissions": [], "count": 0, "error": str(e)}

    async def check_access(
        self, user_id: str, client_id: Optional[str], collection_id: str
    ) -> bool:
        """
        Check if user/client has read access to collection.

        Args:
            user_id: User ID
            client_id: Optional client ID
            collection_id: Collection to check

        Returns:
            True if access granted, False otherwise
        """
        # SECURITY FIX P0: Personal KB owner always has access to their own KB
        # Personal KB naming convention: personal_{user_id[:8]}
        if collection_id and collection_id.startswith("personal_"):
            # Extract the user_id prefix from collection name
            kb_user_prefix = collection_id.replace("personal_", "")
            user_prefix = user_id[:8] if user_id else ""
            if kb_user_prefix == user_prefix:
                logger.debug(
                    f"Personal KB access granted: {user_id} owns {collection_id}"
                )
                return True

        # Check user permission
        user_key = self._build_acl_key("user", user_id, collection_id)
        user_access = await self.redis.get(user_key)

        if user_access:
            access_str = (
                user_access.decode("utf-8")
                if isinstance(user_access, bytes)
                else user_access
            )
            if access_str in ["read", "write"]:
                return True

        # Check client permission if provided
        if client_id:
            client_key = self._build_acl_key("client", client_id, collection_id)
            client_access = await self.redis.get(client_key)

            if client_access:
                # BUG-004 FIX: Handle both bytes and str from Redis
                client_access_str = (
                    client_access.decode("utf-8")
                    if isinstance(client_access, bytes)
                    else client_access
                )
                if client_access_str in ["read", "write"]:
                    return True

        # Default: use configured default access
        return self.default_access in ["read", "write"]

    async def check_write_access(
        self, user_id: str, client_id: Optional[str], collection_id: str,
        *, roles: Optional[list] = None,
    ) -> bool:
        """
        Check if user/client has write access to collection.

        Args:
            user_id: User ID
            client_id: Optional client ID
            collection_id: Collection to check
            roles: Optional list of user roles (used for admin verification
                   when client_id is None — VULN-007 remediation)

        Returns:
            True if write access granted, False otherwise
        """
        # RULE 1: Personal KB ownership (FIRST — applies to ALL users including admin)
        if collection_id and collection_id.startswith("personal_"):
            kb_user_prefix = collection_id.replace("personal_", "")
            user_prefix = user_id[:8] if user_id else ""
            if kb_user_prefix == user_prefix:
                logger.debug(
                    f"Personal KB write access granted: {user_id} owns {collection_id}"
                )
                return True
            # Admin CANNOT write to personal KBs that aren't theirs
            if client_id is None:
                logger.warning(
                    f"Admin {user_id} denied write to personal KB {collection_id} (owner: {kb_user_prefix})"
                )
                return False

        # RULE 2: Admin can write to system and client KBs (NOT personal — handled above)
        # VULN-007 remediation: require explicit admin role verification, not just
        # missing client_id as a proxy for "admin user".
        if client_id is None:
            _is_admin = bool(
                roles and ("admin" in roles or "system" in roles)
            )
            if _is_admin:
                logger.debug(
                    "Admin write access granted to non-personal KB (role verified)",
                    extra={"user_id": user_id, "collection": collection_id},
                )
                return True
            logger.warning(
                "Write access denied: client_id=None without admin role for %s (VULN-007 enforcement)",
                collection_id,
                extra={"user_id": user_id},
            )
            return False

        # RULE 3: Client KB prefix match (client_id is NOT None here)
        if collection_id and collection_id.startswith("client_"):
            client_prefix = f"client_{client_id[:8]}"
            if collection_id.startswith(client_prefix):
                logger.debug(
                    f"Client KB write access granted: client {client_id} owns {collection_id}"
                )
                return True

        # RULE 4: Check user ACL key
        user_key = self._build_acl_key("user", user_id, collection_id)
        user_access = await self.redis.get(user_key)

        if user_access:
            access_str = (
                user_access.decode("utf-8")
                if isinstance(user_access, bytes)
                else user_access
            )
            if access_str == "write":
                return True

        # RULE 5: Check client ACL key
        if client_id:
            client_key = self._build_acl_key("client", client_id, collection_id)
            client_access = await self.redis.get(client_key)

            if client_access:
                client_access_str = (
                    client_access.decode("utf-8")
                    if isinstance(client_access, bytes)
                    else client_access
                )
                if client_access_str == "write":
                    return True

        # Default: deny write access
        return False

    async def get_accessible_collections(
        self, user_id: str, client_id: Optional[str]
    ) -> List[str]:
        """
        Get list of collections accessible by user/client.

        Args:
            user_id: User ID
            client_id: Optional client ID

        Returns:
            List of collection IDs
        """
        try:
            collections = []

            # SECURITY FIX P0: Always include user's personal KB if it exists
            # Personal KB naming convention: personal_{user_id[:8]}
            if user_id:
                personal_kb_name = f"personal_{user_id[:8]}"
                # Check if personal KB exists (metadata stored in Redis)
                kb_metadata_key = f"rag:kb:{personal_kb_name}:metadata"
                kb_exists = await self.redis.exists(kb_metadata_key)
                if kb_exists:
                    collections.append(personal_kb_name)
                    logger.debug(f"Added personal KB to accessible: {personal_kb_name}")

            # Get all user permissions
            user_perms = await self.get_permissions(
                entity_type="user", entity_id=user_id
            )
            collections.extend(
                [
                    p["collection_id"]
                    for p in user_perms.get("permissions", [])
                    if p.get("access_level") in ["read", "write"]
                ]
            )

            # Add client permissions if provided
            if client_id:
                client_perms = await self.get_permissions(
                    entity_type="client", entity_id=client_id
                )
                collections.extend(
                    [
                        p["collection_id"]
                        for p in client_perms.get("permissions", [])
                        if p.get("access_level") in ["read", "write"]
                    ]
                )

            # Remove duplicates
            return list(set(collections))

        except Exception as e:
            logger.error(f"Error getting accessible collections: {e}")
            return []


class ConfigManager:
    """
    RAG Configuration Manager.

    Manages per-user/client RAG configurations using Redis.
    Pure technical logic with no UBP dependencies.

    Redis keys follow NAMING_POLICY.md Section 7:
    - Pattern: ubp:{env}:rag:config:{entity_type}:{entity_id}
    - Default: ubp:{env}:rag:config:default
    """

    def __init__(self, redis_client, config: dict):
        """
        Initialize config manager.

        Args:
            redis_client: Redis client instance
            config: Config storage settings
        """
        self.redis = redis_client
        self.config = config

        # Use RedisKeyManager for environment-aware key prefixes
        if REDIS_KEYS_AVAILABLE and get_key_manager is not None:
            self._key_manager = get_key_manager()
        else:
            self._key_manager = None

        # Legacy prefix for backward compatibility (used if key manager unavailable)
        self._legacy_prefix = config.get("redis_key_prefix", "rag:config")

    def _build_config_key(
        self, entity_type: str, entity_id: Optional[str] = None
    ) -> str:
        """Build a Redis key for RAG config with proper environment prefix."""
        if self._key_manager:
            if entity_type == "default":
                return self._key_manager.key("rag", "config", "default")
            # entity_id is required for non-default entity types
            if entity_id is None:
                raise ValueError(
                    f"entity_id is required for entity_type '{entity_type}'"
                )
            return self._key_manager.key("rag", "config", entity_type, entity_id)
        # Legacy format
        if entity_type == "default":
            return f"{self._legacy_prefix}:default"
        return f"{self._legacy_prefix}:{entity_type}:{entity_id}"

    async def set_rag_config(
        self, entity_type: str, entity_id: Optional[str], config_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Set RAG configuration for an entity.

        Args:
            entity_type: 'user', 'client', or 'default'
            entity_id: user_id or client_id (null for default)
            config_json: RAG configuration

        Returns:
            Status dict with success/error
        """
        if entity_type not in ["user", "client", "default"]:
            return {
                "status": "error",
                "message": "entity_type must be 'user', 'client', or 'default'",
            }

        try:
            config_key = self._build_config_key(entity_type, entity_id)

            await self.redis.set(config_key, json.dumps(config_json))

            return {
                "status": "success",
                "message": f"RAG config set for {entity_type} {entity_id or 'default'}",
            }

        except Exception as e:
            logger.error(f"Error setting RAG config: {e}")
            return {"status": "error", "message": f"Error setting RAG config: {str(e)}"}

    async def get_rag_config(
        self, entity_type: str, entity_id: Optional[str], default_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get RAG configuration for an entity (with fallback to default).

        Args:
            entity_type: 'user', 'client', or 'default'
            entity_id: user_id or client_id
            default_config: Fallback config from file

        Returns:
            Dict with config and source
        """
        try:
            # Try entity-specific config
            if entity_type != "default" and entity_id:
                config_key = self._build_config_key(entity_type, entity_id)
                config_str = await self.redis.get(config_key)

                if config_str:
                    config = json.loads(
                        config_str
                        if isinstance(config_str, str)
                        else config_str.decode("utf-8")
                    )
                    # FIX-TYPE-001 v2.2.4: Coerce string "true"/"false" to bool
                    config = coerce_config_types(config)
                    return {"config": config, "source": f"{entity_type}_specific"}

            # Fallback to default config in Redis
            default_key = self._build_config_key("default")
            default_str = await self.redis.get(default_key)

            if default_str:
                config = json.loads(
                    default_str
                    if isinstance(default_str, str)
                    else default_str.decode("utf-8")
                )
                # FIX-TYPE-001 v2.2.4: Coerce string "true"/"false" to bool
                config = coerce_config_types(config)
                return {"config": config, "source": "default"}

            # Fallback to config file default
            return {"config": default_config, "source": "config_file_default"}

        except Exception as e:
            logger.error(f"Error getting RAG config: {e}")
            return {
                "config": default_config,
                "source": "config_file_default",
                "error": str(e),
            }


class ConversationManager:
    """
    Legacy conversation transcript store for rag_orchestrator.

    This surface is not the canonical authority for memory-session lifecycle,
    turn recording, structured context preparation, or query rewrite. Those
    responsibilities belong to `rag_conversation_memory`.

    ConversationManager remains the rag_orchestrator-owned CRUD/read-model
    surface for the `rag:history` namespace and supports legacy or fallback
    conversation endpoints that still expect transcript-oriented history,
    titles, and per-message metadata stored alongside the transcript.

    ROADMAP v1.5.0 - FEAT-MEM-001 (Task #15)

    Redis keys follow NAMING_POLICY.md Section 7:
    - Index:   ubp:{env}:rag:history:{user_id}:index (SET of conversation_ids)
    - History: ubp:{env}:rag:history:{user_id}:{conversation_id} (HASH with messages)
    """

    def __init__(self, redis_client, config: dict):
        """
        Initialize conversation manager.

        Args:
            redis_client: Redis client instance
            config: Conversation storage settings
        """
        self.redis = redis_client
        self.config = config
        self.default_ttl = (
            config.get("conversation_ttl_days", 30) * 86400
        )  # days to seconds
        self.max_messages_per_conversation = config.get(
            "max_messages_per_conversation", 100
        )

        # Use RedisKeyManager for environment-aware key prefixes
        if REDIS_KEYS_AVAILABLE and get_key_manager is not None:
            self._key_manager = get_key_manager()
        else:
            self._key_manager = None

        # Legacy prefix for backward compatibility
        self._legacy_prefix = config.get("redis_key_prefix", "rag:history")

    def _build_history_key(self, user_id: str, conversation_id: str) -> str:
        """Build a Redis key for conversation history with proper environment prefix."""
        if self._key_manager:
            return self._key_manager.key("rag", "history", user_id, conversation_id)
        return f"{self._legacy_prefix}:{user_id}:{conversation_id}"

    def _build_index_key(self, user_id: str) -> str:
        """Build a Redis key for user's conversation index."""
        if self._key_manager:
            return self._key_manager.key("rag", "history", user_id, "index")
        return f"{self._legacy_prefix}:{user_id}:index"

    async def create_conversation(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new conversation.

        Args:
            user_id: User ID (owner of the conversation)
            conversation_id: Optional conversation ID (generated if not provided)
            title: Optional conversation title

        Returns:
            Dict with conversation_id, title, created_at
        """
        from datetime import datetime

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        now = datetime.utcnow().isoformat() + "Z"
        title = title or f"Conversation {now[:10]}"

        try:
            history_key = self._build_history_key(user_id, conversation_id)
            index_key = self._build_index_key(user_id)

            # Create conversation hash
            conversation_data = {
                "id": conversation_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
                "messages": json.dumps([]),
            }

            await self.redis.hset(history_key, mapping=conversation_data)
            await self.redis.expire(history_key, self.default_ttl)

            # Add to user's conversation index
            await self.redis.sadd(index_key, conversation_id)
            await self.redis.expire(index_key, self.default_ttl)

            logger.info(
                f"Created conversation {conversation_id} for user {user_id}",
                extra={"user_id": user_id, "conversation_id": conversation_id},
            )

            return {
                "conversation_id": conversation_id,
                "title": title,
                "created_at": now,
                "status": "created",
            }

        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            return {
                "conversation_id": conversation_id,
                "status": "error",
                "message": str(e),
            }

    async def add_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a message to a conversation.

        Args:
            user_id: User ID (owner)
            conversation_id: Conversation ID
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata (sources, model, etc.)

        Returns:
            Dict with status and message_id
        """
        from datetime import datetime

        if role not in ["user", "assistant", "system"]:
            return {
                "status": "error",
                "message": "role must be 'user', 'assistant', or 'system'",
            }

        try:
            history_key = self._build_history_key(user_id, conversation_id)

            # Check if conversation exists
            exists = await self.redis.exists(history_key)
            if not exists:
                # Auto-create conversation if it doesn't exist
                await self.create_conversation(user_id, conversation_id)

            # Get current messages
            messages_json = await self.redis.hget(history_key, "messages")
            if messages_json:
                messages = json.loads(
                    messages_json
                    if isinstance(messages_json, str)
                    else messages_json.decode("utf-8")
                )
            else:
                messages = []

            # Check max messages limit
            if len(messages) >= self.max_messages_per_conversation:
                # Remove oldest messages (keep last N-1 to make room)
                messages = messages[-(self.max_messages_per_conversation - 1) :]

            # Create message
            message_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat() + "Z"

            message = {
                "id": message_id,
                "role": role,
                "content": content,
                "timestamp": now,
                "metadata": metadata or {},
            }

            messages.append(message)

            # Update conversation
            await self.redis.hset(history_key, "messages", json.dumps(messages))
            await self.redis.hset(history_key, "updated_at", now)

            # Update title from first user message if title is default
            if role == "user" and len(messages) == 1:
                title_preview = content[:50] + ("..." if len(content) > 50 else "")
                await self.redis.hset(history_key, "title", title_preview)

            # Refresh TTL
            await self.redis.expire(history_key, self.default_ttl)

            logger.debug(
                f"Added message to conversation {conversation_id}",
                extra={"conversation_id": conversation_id, "role": role},
            )

            return {
                "status": "success",
                "message_id": message_id,
                "conversation_id": conversation_id,
            }

        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return {"status": "error", "message": str(e)}

    async def get_conversation(
        self, user_id: str, conversation_id: str
    ) -> Dict[str, Any]:
        """
        Get a conversation with all messages.

        Args:
            user_id: User ID (owner)
            conversation_id: Conversation ID

        Returns:
            Dict with conversation data including messages
        """
        try:
            history_key = self._build_history_key(user_id, conversation_id)

            # Get all conversation data
            data = await self.redis.hgetall(history_key)

            if not data:
                return {
                    "status": "error",
                    "message": "Conversation not found",
                    "conversation_id": conversation_id,
                }

            # Decode bytes if necessary
            decoded = {}
            for k, v in data.items():
                key = k if isinstance(k, str) else k.decode("utf-8")
                value = v if isinstance(v, str) else v.decode("utf-8")
                decoded[key] = value

            # Parse messages JSON
            messages = json.loads(decoded.get("messages", "[]"))

            return {
                "status": "success",
                "conversation": {
                    "id": decoded.get("id", conversation_id),
                    "title": decoded.get("title", ""),
                    "created_at": decoded.get("created_at", ""),
                    "updated_at": decoded.get("updated_at", ""),
                    "messages": messages,
                    "message_count": len(messages),
                },
            }

        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return {"status": "error", "message": str(e)}

    async def list_conversations(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """
        List all conversations for a user.

        Args:
            user_id: User ID
            limit: Maximum conversations to return
            offset: Pagination offset

        Returns:
            Dict with conversations array and count
        """
        try:
            index_key = self._build_index_key(user_id)

            # Get all conversation IDs from index
            conversation_ids = await self.redis.smembers(index_key)

            if not conversation_ids:
                return {"conversations": [], "count": 0, "total": 0}

            # Decode bytes if necessary
            conv_ids = [
                cid if isinstance(cid, str) else cid.decode("utf-8")
                for cid in conversation_ids
            ]

            # Get metadata for each conversation
            conversations = []
            for conv_id in conv_ids:
                history_key = self._build_history_key(user_id, conv_id)
                data = await self.redis.hmget(
                    history_key, "id", "title", "created_at", "updated_at", "messages"
                )

                if data[0]:  # If conversation exists
                    messages = (
                        json.loads(
                            data[4]
                            if isinstance(data[4], str)
                            else data[4].decode("utf-8")
                        )
                        if data[4]
                        else []
                    )

                    conversations.append(
                        {
                            "id": data[0]
                            if isinstance(data[0], str)
                            else data[0].decode("utf-8"),
                            "title": data[1]
                            if isinstance(data[1], str)
                            else data[1].decode("utf-8")
                            if data[1]
                            else "",
                            "created_at": data[2]
                            if isinstance(data[2], str)
                            else data[2].decode("utf-8")
                            if data[2]
                            else "",
                            "updated_at": data[3]
                            if isinstance(data[3], str)
                            else data[3].decode("utf-8")
                            if data[3]
                            else "",
                            "message_count": len(messages),
                        }
                    )

            # Sort by updated_at descending
            conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

            # Apply pagination
            total = len(conversations)
            paginated = conversations[offset : offset + limit]

            return {
                "conversations": paginated,
                "count": len(paginated),
                "total": total,
            }

        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return {"conversations": [], "count": 0, "total": 0, "error": str(e)}

    async def delete_conversation(
        self, user_id: str, conversation_id: str
    ) -> Dict[str, Any]:
        """
        Delete a conversation.

        Args:
            user_id: User ID (owner)
            conversation_id: Conversation ID to delete

        Returns:
            Dict with status
        """
        try:
            history_key = self._build_history_key(user_id, conversation_id)
            index_key = self._build_index_key(user_id)

            # Check if conversation exists
            exists = await self.redis.exists(history_key)
            if not exists:
                return {
                    "status": "error",
                    "message": "Conversation not found",
                    "conversation_id": conversation_id,
                }

            # Delete conversation hash
            await self.redis.delete(history_key)

            # Remove from index
            await self.redis.srem(index_key, conversation_id)

            logger.info(
                f"Deleted conversation {conversation_id} for user {user_id}",
                extra={"user_id": user_id, "conversation_id": conversation_id},
            )

            return {
                "status": "success",
                "message": f"Conversation {conversation_id} deleted",
                "conversation_id": conversation_id,
            }

        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return {"status": "error", "message": str(e)}

    async def get_context_for_llm(
        self,
        user_id: str,
        conversation_id: str,
        max_turns: int = 10,
    ) -> Dict[str, Any]:
        """
        Get transcript-formatted context for legacy/fallback LLM consumers.

        This reads ConversationManager's own `rag:history` transcript store. It
        does not replace the canonical structured-context preparation performed
        by `rag_conversation_memory`.

        Args:
            user_id: User ID
            conversation_id: Conversation ID
            max_turns: Maximum message pairs to include

        Returns:
            Dict with formatted context string
        """
        try:
            result = await self.get_conversation(user_id, conversation_id)

            if result.get("status") == "error":
                return {"context": "", "turns": 0}

            messages = result.get("conversation", {}).get("messages", [])

            if not messages:
                return {"context": "", "turns": 0}

            # Get last N*2 messages (N turns = N user + N assistant)
            recent = messages[-(max_turns * 2) :]

            # Format for LLM
            context_parts = []
            for msg in recent:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                context_parts.append(f"{role}: {content}")

            return {
                "context": "\n\n".join(context_parts),
                "turns": len(recent) // 2,
                "message_count": len(recent),
            }

        except Exception as e:
            logger.error(f"Error getting context for LLM: {e}")
            return {"context": "", "turns": 0, "error": str(e)}

    async def update_conversation_title(
        self, user_id: str, conversation_id: str, title: str
    ) -> Dict[str, Any]:
        """
        Update conversation title.

        Args:
            user_id: User ID
            conversation_id: Conversation ID
            title: New title

        Returns:
            Dict with status
        """
        try:
            history_key = self._build_history_key(user_id, conversation_id)

            exists = await self.redis.exists(history_key)
            if not exists:
                return {"status": "error", "message": "Conversation not found"}

            await self.redis.hset(history_key, "title", title)

            return {
                "status": "success",
                "message": "Title updated",
                "conversation_id": conversation_id,
            }

        except Exception as e:
            logger.error(f"Error updating title: {e}")
            return {"status": "error", "message": str(e)}


class DocumentChunker:
    """
    Utility class for chunking documents before ingestion.
    """

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text
            chunk_size: Target chunk size in characters
            overlap: Number of overlapping characters

        Returns:
            List of text chunks
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size

            # Extract chunk
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence end markers
                last_period = chunk.rfind(". ")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > chunk_size // 2:  # Only break if we're past halfway
                    chunk = chunk[: break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())

            # Move start with overlap
            start = end - overlap if end < text_length else text_length

        return [c for c in chunks if c]  # Filter empty chunks


class KeywordManager:
    """
    Dynamic Keyword Manager for Knowledge Base Collections.

    ROADMAP v1.8.1 - FEAT-DKI-001 (Dynamic Knowledge Injection)

    Manages automatically extracted keywords per collection for intelligent routing.
    When documents are ingested, keywords are extracted via LLM and stored in Redis Sets.
    The Semantic Router uses these keywords to boost RAG routing for short/ambiguous queries.

    Redis keys follow NAMING_POLICY.md Section 7:
    - Pattern: ubp:{env}:rag:keywords:{collection_name} (SET of keywords)

    Flow:
    1. INGESTION: extract_and_store_keywords() called after text extraction
    2. ROUTING: get_keywords_for_collections() returns all keywords for user's accessible KBs
    3. MATCHING: Router checks if query contains any keywords -> boost RAG confidence
    """

    def __init__(self, redis_client, llm_module=None, config=None):
        """
        Initialize keyword manager.

        Args:
            redis_client: Redis client instance
            llm_module: Optional LLM module for keyword extraction (lazy resolution ok)
            config: Optional DKISettings for configurable extraction (v2.0)
        """
        self.redis = redis_client
        self.llm_module = llm_module

        # Use RedisKeyManager for environment-aware key prefixes
        if REDIS_KEYS_AVAILABLE and get_key_manager is not None:
            self._key_manager = get_key_manager()
        else:
            self._key_manager = None

        # Legacy prefix for backward compatibility
        self._legacy_prefix = "ubp:rag:keywords"

        # FEAT-DKI-001 v2.0: Configurable extraction parameters
        self.enabled = config.enabled if config else True
        self.batch_size = config.batch_size if config else 1
        self.max_keywords_per_doc = config.max_keywords_per_doc if config else 15
        self.sample_chars = config.sample_chars if config else 4000
        self.temperature = config.temperature if config else 0.1
        self.max_tokens = config.max_tokens if config else 400
        self.summarize_enabled = config.summarize_enabled if config else False
        self.summarize_max_chars = config.summarize_max_chars if config else 8000
        self.summarize_max_tokens = config.summarize_max_tokens if config else 1000

    def _build_keywords_key(self, collection_name: str) -> str:
        """Build Redis key for collection keywords with proper environment prefix."""
        if self._key_manager:
            return self._key_manager.key("rag", "keywords", collection_name)
        return f"{self._legacy_prefix}:{collection_name}"

    async def extract_keywords_via_llm(self, text: str) -> List[str]:
        """
        Extract keywords from text using LLM.

        Sends first N characters to LLM with extraction prompt.
        Returns list of unique technical terms, product names, entities.

        Args:
            text: Full document text

        Returns:
            List of extracted keywords (lowercase, deduplicated)
        """
        if not self.llm_module:
            logger.warning(
                "KeywordManager: LLM module not available, skipping extraction"
            )
            return []

        # Sample first N chars (summary/intro usually has key terms)
        sample = text[: self.sample_chars] if len(text) > self.sample_chars else text

        extraction_prompt = f"""Extract the top {self.max_keywords_per_doc} unique technical terms, product names, proper nouns, or specific entities from this text.

RULES:
- Include product names, system names, technical terms, acronyms
- Include proper nouns (company names, project names, technologies)
- Prefer multi-word phrases for specificity (e.g., "UBP Enterprise" not just "Enterprise")
- Exclude common words, articles, prepositions
- Return ONLY a JSON array of strings

TEXT:
{sample}

RESPONSE (JSON array only):"""

        try:
            # Use the LLM module for extraction (fire-and-forget friendly)
            response = await self.llm_module.generate(
                prompt=extraction_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            # Parse JSON response - support both 'text' and 'response' keys for compatibility
            # inference_ollama_grok returns 'text', other modules may return 'response'
            response_text = (
                response.get("text") or response.get("response") or ""
            ).strip()

            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = (
                    response_text.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            # Find JSON array in response
            json_match = re.search(r"\[.*?\]", response_text, re.DOTALL)
            if json_match:
                keywords = json.loads(json_match.group())
            else:
                keywords = json.loads(response_text)

            # Normalize: lowercase, strip, dedupe
            keywords = list(
                set(
                    kw.lower().strip()
                    for kw in keywords
                    if isinstance(kw, str) and len(kw) > 1
                )
            )

            logger.info(
                f"KeywordManager: Extracted {len(keywords)} keywords",
                extra={"keywords": keywords[:5]},  # Log first 5
            )
            return keywords[: self.max_keywords_per_doc]

        except json.JSONDecodeError as e:
            logger.warning(f"KeywordManager: Failed to parse LLM response as JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"KeywordManager: Error extracting keywords: {e}")
            return []

    async def store_keywords(self, collection_name: str, keywords: List[str]) -> int:
        """
        Store keywords for a collection in Redis Set.

        Args:
            collection_name: Target collection
            keywords: List of keywords to add

        Returns:
            Number of new keywords added
        """
        if not keywords:
            return 0

        key = self._build_keywords_key(collection_name)
        # DEBUG: Log key and keywords being stored
        print(
            f"[KeywordManager DEBUG] Storing keywords - Key: {key}, Count: {len(keywords)}"
        )
        try:
            # SADD returns count of NEW members added
            added = await self.redis.sadd(key, *keywords)
            print(f"[KeywordManager DEBUG] SADD returned: {added}")
            logger.info(
                f"KeywordManager: Stored {added} new keywords for {collection_name}",
                extra={
                    "total_keywords": len(keywords),
                    "new_added": added,
                    "redis_key": key,
                },
            )
            return added
        except Exception as e:
            logger.error(f"KeywordManager: Error storing keywords: {e}")
            return 0

    async def extract_and_store_keywords(
        self, collection_name: str, text: str
    ) -> Dict[str, Any]:
        """
        Full extraction + storage pipeline for ingestion.

        Called during document ingestion after text extraction.

        Args:
            collection_name: Target collection
            text: Full document text

        Returns:
            Status dict with extracted keywords count
        """
        try:
            keywords = await self.extract_keywords_via_llm(text)
            if not keywords:
                return {
                    "status": "skipped",
                    "message": "No keywords extracted",
                    "keywords_count": 0,
                }

            added = await self.store_keywords(collection_name, keywords)

            return {
                "status": "success",
                "keywords_extracted": len(keywords),
                "keywords_added": added,
                "sample_keywords": keywords[:5],
            }
        except Exception as e:
            logger.error(f"KeywordManager: extract_and_store_keywords failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "keywords_count": 0,
            }

    async def get_keywords(self, collection_name: str) -> List[str]:
        """
        Get all keywords for a collection.

        Args:
            collection_name: Collection name

        Returns:
            List of keywords
        """
        key = self._build_keywords_key(collection_name)
        try:
            keywords = await self.redis.smembers(key)
            # Redis returns bytes or strings depending on decode_responses
            return [kw.decode() if isinstance(kw, bytes) else kw for kw in keywords]
        except Exception as e:
            logger.error(f"KeywordManager: Error getting keywords: {e}")
            return []

    async def get_keywords_for_collections(
        self, collection_names: List[str]
    ) -> Dict[str, List[str]]:
        """
        Get keywords for multiple collections (for user's accessible KBs).

        Used by Router to build dynamic keyword matching set.

        Args:
            collection_names: List of collection names

        Returns:
            Dict mapping collection_name -> list of keywords
        """
        tasks = [self.get_keywords(coll) for coll in collection_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            coll: kw if not isinstance(kw, Exception) else []
            for coll, kw in zip(collection_names, results)
        }

    async def get_all_keywords_flat(self, collection_names: List[str]) -> set:
        """
        Get flattened set of all keywords from multiple collections.

        Optimized for fast membership testing during routing.

        Args:
            collection_names: List of collection names

        Returns:
            Set of all unique keywords
        """
        tasks = [self.get_keywords(coll) for coll in collection_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_keywords = set()
        for kw in results:
            if not isinstance(kw, Exception):
                all_keywords.update(kw)
        return all_keywords

    async def remove_keywords(self, collection_name: str, keywords: List[str]) -> int:
        """
        Remove specific keywords from a collection.

        Args:
            collection_name: Collection name
            keywords: Keywords to remove

        Returns:
            Number of keywords removed
        """
        if not keywords:
            return 0

        key = self._build_keywords_key(collection_name)
        try:
            removed = await self.redis.srem(key, *keywords)
            return removed
        except Exception as e:
            logger.error(f"KeywordManager: Error removing keywords: {e}")
            return 0

    async def clear_keywords(self, collection_name: str) -> bool:
        """
        Clear all keywords for a collection.

        Args:
            collection_name: Collection name

        Returns:
            True if successful
        """
        key = self._build_keywords_key(collection_name)
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"KeywordManager: Error clearing keywords: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored keywords.

        Returns:
            Dict with stats per collection
        """
        try:
            # Scan for all keyword keys
            if self._key_manager:
                pattern = self._key_manager.key("rag", "keywords", "*")
            else:
                pattern = f"{self._legacy_prefix}:*"

            stats = {}
            async for key in self.redis.scan_iter(match=pattern):
                key_str = key.decode() if isinstance(key, bytes) else key
                # Extract collection name from key
                collection_name = key_str.split(":")[-1]
                count = await self.redis.scard(key)
                stats[collection_name] = count

            return {"collections": stats, "total_collections": len(stats)}
        except Exception as e:
            logger.error(f"KeywordManager: Error getting stats: {e}")
            return {"error": str(e)}

    # === FEAT-DKI-001 v2.0: Batch Extraction & Summarize ===

    async def extract_keywords_batch(
        self, collection_name: str, texts: List[str]
    ) -> Dict[str, Any]:
        """
        Extract keywords from multiple documents in a single LLM call.

        Combines text samples from N documents and sends one prompt.

        Args:
            collection_name: Target collection
            texts: List of document texts

        Returns:
            Status dict with extraction results
        """
        if not self.enabled or not self.llm_module:
            return {"status": "disabled", "keywords_count": 0}

        # Combine: first sample_chars per doc, separated by ---
        combined_samples = []
        for text in texts:
            sample = text[: self.sample_chars] if len(text) > self.sample_chars else text
            combined_samples.append(sample)
        combined = "\n---\n".join(combined_samples)

        # Use same extraction logic with combined text
        keywords = await self.extract_keywords_via_llm(combined)
        if not keywords:
            return {"status": "skipped", "keywords_count": 0}

        added = await self.store_keywords(collection_name, keywords)
        return {
            "status": "success",
            "keywords_extracted": len(keywords),
            "keywords_added": added,
            "documents_in_batch": len(texts),
        }

    def _build_summary_key(self, collection_name: str) -> str:
        """Build Redis key for collection summary with proper environment prefix."""
        if self._key_manager:
            return self._key_manager.key("rag", "summary", collection_name)
        return f"ubp:rag:summary:{collection_name}"

    async def get_summary(self, collection_name: str) -> Optional[str]:
        """
        Get stored summary for a collection.

        Args:
            collection_name: Collection name

        Returns:
            Summary text or None if not available
        """
        key = self._build_summary_key(collection_name)
        try:
            summary = await self.redis.get(key)
            if summary is None:
                return None
            return summary.decode() if isinstance(summary, bytes) else summary
        except Exception as e:
            logger.error(f"KeywordManager: Error getting summary: {e}")
            return None

    async def summarize_collection(
        self, collection_name: str, texts: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a global summary of ingested documents. (Future development)

        Combines document texts up to summarize_max_chars and generates
        a structured summary via LLM. Stored in Redis with 7-day TTL.

        Args:
            collection_name: Target collection
            texts: List of document texts

        Returns:
            Status dict with summary result
        """
        if not self.summarize_enabled or not self.llm_module:
            return {"status": "disabled"}

        # Combine text respecting the limit
        combined = ""
        for text in texts:
            if len(combined) + len(text) + 5 > self.summarize_max_chars:
                remaining = self.summarize_max_chars - len(combined)
                if remaining > 100:
                    combined += "\n---\n" + text[:remaining]
                break
            combined += "\n---\n" + text if combined else text

        prompt = (
            f'Generate a concise summary of the following documents from knowledge base "{collection_name}".\n'
            "Highlight: main topics, key entities, document types, and domain coverage.\n"
            "Return a structured summary in 3-5 paragraphs.\n\n"
            f"DOCUMENTS:\n{combined}\n\nSUMMARY:"
        )

        try:
            response = await self.llm_module.generate(
                prompt=prompt,
                max_tokens=self.summarize_max_tokens,
                temperature=0.3,
            )
            summary_text = (
                response.get("text") or response.get("response") or ""
            ).strip()

            if not summary_text:
                return {"status": "skipped", "message": "Empty summary from LLM"}

            # Store in Redis with 7-day TTL
            summary_key = self._build_summary_key(collection_name)
            await self.redis.set(summary_key, summary_text, ex=604800)

            logger.info(
                f"KeywordManager: Summary stored for {collection_name} "
                f"({len(summary_text)} chars, {len(texts)} docs)"
            )
            return {
                "status": "success",
                "summary": summary_text,
                "chars_processed": len(combined),
            }
        except Exception as e:
            logger.error(f"KeywordManager: Summarize failed: {e}")
            return {"status": "error", "message": str(e)}
