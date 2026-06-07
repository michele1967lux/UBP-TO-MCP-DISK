"""
UBP Framework Bridge for Adaptive Budget Manager Module

Integrates AdaptiveBudgetManager with UBP module system.
Provides adaptive token budget management for chat scenarios (RAG, Pure LLM, etc.).
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging

from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule
from .providers import AdaptiveBudgetManager
from .models import (
    AdaptiveMemoryConfig,
    BudgetAdjustmentResult,
    TightnessResult,
    SummarizationResult,
    ExecutionPlan,  # v3.7.0: Context Governor
)

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

logger = logging.getLogger(__name__)


class AdaptiveBudgetManagerAdapter(BaseHybridModule):
    """
    UBP adapter for adaptive budget management.
    
    Provides adaptive token budget allocation for any chat scenario.
    Follows the 3-file pattern: adapter.py + providers.py + __init__.py
    """
    
    def __init__(self, module_path: Path, **kwargs):
        super().__init__(module_path, **kwargs)
        self.provider: Optional[AdaptiveBudgetManager] = None
        self._init_status: Dict[str, Any] = {"status": "not_initialized"}
        
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
        
    async def initialize(self) -> None:
        """Initialize module and provider."""
        logger.info(f"Initializing {self.manifest.name}")
        
        # Validate configuration
        try:
            adaptive_config = AdaptiveMemoryConfig(**self.config)
        except Exception as e:
            logger.error(f"Invalid configuration: {e}")
            self._init_status = {
                "status": "failed",
                "reason": f"Invalid configuration: {e}",
            }
            return
        
        # Initialize provider with DI container for lazy LLM resolution
        self.provider = AdaptiveBudgetManager(
            config=self.config,
            di_container=self.di_container,
        )
        
        logger.info(f"✅ {self.manifest.name} initialized")
        self._init_status = {
            "status": "healthy",
            "config": {
                "base_min_score": self.config.get("base_min_score"),
                "max_threshold": self.config.get("max_threshold"),
                "compression_enabled": self.config.get("compression_enabled"),
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown module and release resources."""
        logger.info(f"Shutting down {self.manifest.name}")
        self.provider = None
        self._init_status = {"status": "shutdown"}

    async def adjust_for_pure_chat(
        self,
        query: str,
        conversation_context: Optional[str],
        turn_count: int,
        chat_config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Adjust memory budget for pure chat scenarios (without RAG).
        
        Args:
            query: User query
            conversation_context: Conversation history
            turn_count: Conversation turn number
            chat_config: Chat configuration
            **kwargs: Optional MCP-side overhead (system_prompt_tokens,
                      tool_overhead_tokens, memory_tokens_estimate)
            
        Returns:
            Dict with adjusted context and metadata
        """
        if not self.provider:
            raise RuntimeError("Provider not initialized")
        
        # Extract model/provider from config
        model = chat_config.get("model")
        provider = chat_config.get("provider")
        max_tokens_client = kwargs.get("max_tokens_client") or chat_config.get("max_tokens_client")
        
        # FIX 2: Forward MCP-side overhead to provider
        system_prompt_tokens = kwargs.get("system_prompt_tokens", 0)
        tool_overhead_tokens = kwargs.get("tool_overhead_tokens", 0)
        memory_tokens_estimate = kwargs.get("memory_tokens_estimate", 0)
        
        # Call provider
        result = await self.provider.adjust_for_pure_chat(
            query=query,
            conversation_context=conversation_context,
            turn_count=turn_count,
            chat_config=chat_config,
            model=model,
            provider=provider,
            max_tokens_client=max_tokens_client,
            system_prompt_tokens=system_prompt_tokens,
            tool_overhead_tokens=tool_overhead_tokens,
            memory_tokens_estimate=memory_tokens_estimate,
        )
        
        return result
    
    async def adjust_budget(
        self,
        query: str,
        conversation_context: Optional[str],
        retrieved_docs: List[Dict[str, Any]],
        turn_count: int,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Adjust token budget dynamically.
        
        Args:
            query: User query
            conversation_context: Conversation history
            retrieved_docs: Documents from retrieval
            turn_count: Conversation turn number
            config: RAG configuration
            
        Returns:
            BudgetAdjustmentResult as dict
        """
        if not self.provider:
            raise RuntimeError("Provider not initialized")
        
        # Extract model/provider from config
        model = config.get("model")
        provider = config.get("provider")
        max_tokens_client = kwargs.get("max_tokens_client") or config.get("max_tokens_client")
        
        # Call provider
        result = await self.provider.adjust(
            query=query,
            conversation_context=conversation_context,
            retrieved_docs=retrieved_docs,
            turn_count=turn_count,
            rag_config=config,
            model=model,
            provider=provider,
            max_tokens_client=max_tokens_client
        )
        
        # Validate result
        adjustment_result = BudgetAdjustmentResult(
            conversation_context=result.get("conversation_context"),
            filtered_docs=result.get("filtered_docs", []),
            doc_budget_tokens=result.get("doc_budget_tokens", 0),
            tightness=result.get("tightness", 0.0),
            adjusted_min_score=result.get("adjusted_min_score", 0.4),
            memory_tokens=result.get("memory_tokens", 0),
            compression_applied=result.get("compression_applied", False),
            original_doc_count=result.get("original_doc_count", 0),
            filtered_doc_count=result.get("filtered_doc_count", 0),
        )
        
        return adjustment_result.model_dump()
    
    async def calculate_tightness(
        self,
        total_tokens: int,
        current_overhead_tokens: int,
        turn_count: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate tightness factor.
        
        Args:
            total_tokens: Model context window
            current_overhead_tokens: Current token usage
            turn_count: Conversation turn number
            
        Returns:
            TightnessResult as dict
        """
        if not self.provider:
            raise RuntimeError("Provider not initialized")
        
        result = self.provider.calculate_tightness(
            total_tokens, current_overhead_tokens, turn_count
        )
        
        tightness_result = TightnessResult(**result)
        return tightness_result.model_dump()
    
    async def recalculate_after_delta(self, ctx=None, **kwargs) -> Dict[str, Any]:
        """DCBL: ricalcola budget dopo ogni tool result nel loop agent_loop.

        Dispatch target — riceve conteggi numerici pre-calcolati dal MCP server.
        Restituisce BudgetState.to_dict() per serializzazione HTTP.
        """
        result = await self.provider.recalculate_after_delta(**kwargs)
        return result.to_dict() if hasattr(result, "to_dict") else result

    async def summarize_context(
        self,
        text: str,
        target_tokens: int,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compress context using LLM-based summarization.
        
        Args:
            text: Text to summarize
            target_tokens: Target token count
            config: Optional summarization config
            
        Returns:
            SummarizationResult as dict
        """
        if not self.provider:
            raise RuntimeError("Provider not initialized")
        
        model = config.get("model") if config else None
        provider = config.get("provider") if config else None
        
        summary = await self.provider.summarize_context(
            text, target_tokens, model, provider
        )
        
        # Calculate metrics
        from ubp_enterprise_hybrid.modules.cores._shared.token_limits import TokenCounter
        original_tokens = TokenCounter.count_tokens(text, model, provider)
        summary_tokens = TokenCounter.count_tokens(summary, model, provider)
        
        result = SummarizationResult(
            summary=summary,
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            compression_ratio=summary_tokens / original_tokens if original_tokens > 0 else 0.0
        )
        
        return result.model_dump()
    
    def get_execution_plan(
        self,
        query: str,
        turn_count: int = 1,
        provider: str = "grok",
        model: Optional[str] = None,
        task_profile: str = "chat",
        current_memory_tokens: int = 0,
        rag_config: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "ExecutionPlan":
        """
        v3.7.0 Context Governor: Get unified ExecutionPlan for request.
        
        This method is a proxy to the provider's get_execution_plan() method,
        exposed at the adapter level for RAG orchestrator integration.
        
        v3.7.1: Added constraints parameter for 4-layer policy enforcement.
        
        Args:
            query: User query text
            turn_count: Current conversation turn number
            provider: LLM provider name (grok, ollama, openai, etc.)
            model: Optional specific model override
            task_profile: Task type (chat, analysis, creative, coding)
            current_memory_tokens: Current conversation memory token count
            rag_config: Optional RAG-specific configuration
            constraints: Optional 4-layer constraints dict with:
                - max_doc_budget_tokens: Hard cap on document budget
                - max_memory_budget_tokens: Hard cap on memory budget
                - max_top_k: Maximum chunks to retrieve
                - min_similarity_threshold: Floor for similarity
                - allowed_strategies: List of allowed ContextStrategy values
                - source: str indicating constraint origin
            
        Returns:
            ExecutionPlan with unified parameters for the request
        """
        if not self.provider:
            raise RuntimeError("Provider not initialized")
        
        return self.provider.get_execution_plan(
            query=query,
            turn_count=turn_count,
            provider=provider,
            model=model,
            task_profile=task_profile,
            current_memory_tokens=current_memory_tokens,
            rag_config=rag_config,
            constraints=constraints,
            # v6.3.0: forward overflow strategy parameters
            total_chunk_tokens=kwargs.get("total_chunk_tokens", 0),
            total_chunks=kwargs.get("total_chunks", 0),
            modules_available=kwargs.get("modules_available"),
            user_preferences=kwargs.get("user_preferences"),
        )
    
    async def health_check(self, **kwargs) -> Dict[str, Any]:
        """
        Check module health status.
        
        Returns:
            Health status dict
        """
        return {
            "module": self.manifest.name,
            "status": "healthy" if self.provider else "unhealthy",
            "init_status": self._init_status,
            "config": {
                "base_min_score": self.config.get("base_min_score"),
                "max_threshold": self.config.get("max_threshold"),
                "compression_enabled": self.config.get("compression_enabled"),
            },
            "llm_resolution": "available" if (self.provider and self.provider._llm_provider) else "pending (lazy)",
            "support_llm_provider": self.config.get("support_llm_provider") or "vllm",
            # v6.0.1: support_llm_model removed — model resolved by inference module
        }


# Factory function for UBP module loading
async def create_adapter(module_path: Path, **kwargs) -> AdaptiveBudgetManagerAdapter:
    """Create and return adapter instance."""
    return AdaptiveBudgetManagerAdapter(module_path, **kwargs)


__all__ = ["AdaptiveBudgetManagerAdapter", "create_adapter"]
