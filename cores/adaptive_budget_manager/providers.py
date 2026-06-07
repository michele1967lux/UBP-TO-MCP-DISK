"""
RAG Adaptive Memory Providers - Technical Logic Layer (v3.7.0)

Implements adaptive token budget management for RAG pipeline with:
- Dynamic memory allocation based on context window
- Tightness factor calculation
- Similarity threshold scaling
- Context compression/summarization
- ExecutionPlan generation (v3.7.0)
- Dynamic system prompt injection (v3.7.0)

Can be tested independently without UBP framework.
"""

import logging
import re
import time as _time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Optional
from ubp_enterprise_hybrid.modules.cores._shared.token_limits import TokenCounter, ContextValidator
from ubp_enterprise_hybrid.modules.cores._shared.token_limits import get_provider_limits_from_env
from ubp_enterprise_hybrid.modules.cores.adaptive_budget_manager.models import UserPreferences
from ubp_enterprise_hybrid.mcp_runtime.core.budget_calc import (
    calculate_pure_chat_budget,
    recalculate_after_delta as pure_recalculate_after_delta,
)


# ---------------------------------------------------------------------------
# DCBL: BudgetState — contratto ricco tra Budget Manager e agent_loop
# ---------------------------------------------------------------------------

@dataclass(frozen=False)
class BudgetState:
    """Stato completo del budget dopo ogni ricalcolo DCBL.

    Restituito da recalculate_after_delta(). Serializzato via to_dict()
    per il dispatch HTTP. L'agent_loop usa i campi per decidere compression,
    preflight e max_tokens per la re-call.
    """
    # Contesto
    context_window: int = 40960
    model: str = "unknown"
    provider: str = "unknown"
    # Token accounting (token-first)
    fixed_overhead_tokens: int = 0
    structured_memory_tokens: int = 0
    history_tokens: int = 0
    delta_tokens: int = 0
    total_estimated_tokens: int = 0
    # Metriche di decisione
    tightness: float = 0.0
    response_budget_tokens: int = 4096
    # Compression — NB: flag di RICHIESTA, non di esecuzione
    compression_needed: bool = False
    compression_level: int = 0  # 0=none, 1=selective (keep last raw), 2=full (all synopsis)
    compression_mode: Literal["none", "level_1_selective", "level_2_full", "delta_guard", "budget_pressure"] = "none"
    compression_target: int = 0
    # Safety
    safety_margin_used: bool = False
    is_critical: bool = False
    # Metadati debug
    round_count: int = 0
    session_id: str = ""
    original_budget_max_tokens: int = 0
    timestamp: float = field(default_factory=_time.time)

    def needs_compression(self) -> bool:
        """Alias — usa compression_needed flag calcolato dal budget manager."""
        return self.compression_needed

    def is_safe(self) -> bool:
        return self.total_estimated_tokens <= self.context_window * 0.92

    def get_summary(self) -> str:
        return (
            f"tightness={self.tightness:.3f} resp_budget={self.response_budget_tokens} "
            f"total_est={self.total_estimated_tokens} compression={self.compression_mode} "
            f"delta={self.delta_tokens}"
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

logger = logging.getLogger(__name__)


class AdaptiveBudgetManager:
    """
    Adaptive token budget manager for RAG pipeline.
    
    Automatically adjusts memory allocation and retrieval parameters based on:
    - Model context window size
    - Current token usage
    - Conversation turn count
    
    Implements intelligent scaling to:
    - Expand when space available (low turns, large window)
    - Restrict when tight (high turns, small window)
    - Prioritize memory over documents
    - Compress when necessary
    """
    
    def __init__(self, config: Dict[str, Any], llm_adapter=None, di_container=None):
        """
        Initialize adaptive budget manager.

        Args:
            config: Configuration dict with adaptive memory settings
            llm_adapter: Optional LLM adapter for summarization (legacy, deprecated)
            di_container: DI container for lazy LLM resolution
        """
        self.config = config
        self.llm_adapter = llm_adapter
        self._di_container = di_container

        # LLM resolution state (v7.2.2: chain-based with retry)
        self._llm_chain: List = []  # [(module, provider_name), ...]
        self._llm_provider = llm_adapter  # primary (first in chain)
        self._llm_resolved = llm_adapter is not None
        self._llm_resolve_failed_at: Optional[float] = None
        _LLM_RETRY_INTERVAL = 60.0  # seconds before retrying failed resolution
        self._llm_retry_interval = _LLM_RETRY_INTERVAL

        # Load configuration
        self.base_min_score = float(config.get("base_min_score", 0.4))
        self.max_threshold = float(config.get("max_threshold", 0.7))
        self.quality_floor = float(config.get("quality_floor", 0.35))
        self.min_memory_fraction = float(config.get("min_memory_fraction", 0.2))
        self.max_memory_fraction = float(config.get("max_memory_fraction", 0.4))
        self.turn_penalty_factor = float(config.get("turn_penalty_factor", 0.05))
        self.compression_enabled = config.get("compression_enabled", True)
        self.compression_threshold = float(config.get("compression_threshold", 0.5))
        
        # v6.2.4: Window split recommendation settings
        self.split_enabled = config.get("split_enabled", True)
        self.split_tightness_threshold = float(config.get("split_tightness_threshold", 0.70))
        
        # v6.3.1: Global override for min_response_tokens (0 = use per-profile defaults)
        self.min_response_tokens_override = int(config.get("min_response_tokens_override", 0))

        # v6.3.0: Overflow strategy settings
        overflow_cfg = config.get("overflow_strategy", {})
        self.overflow_strategy_enabled = self._bool(overflow_cfg.get("enabled", True))
        self.overflow_selective_threshold = float(overflow_cfg.get("selective_threshold", 1.3))
        self.overflow_compressed_threshold = float(overflow_cfg.get("compressed_threshold", 2.0))
        self.overflow_split_threshold = float(overflow_cfg.get("split_threshold", 2.0))
        self.overflow_summarize_threshold = float(overflow_cfg.get("summarize_threshold", 4.0))
        _prefer_tasks = overflow_cfg.get("prefer_split_for_tasks", "reasoning,report,analysis")
        self.overflow_prefer_split_tasks = (
            [t.strip() for t in _prefer_tasks.split(",")]
            if isinstance(_prefer_tasks, str) else list(_prefer_tasks)
        )
        self.overflow_fallback_no_splitter = overflow_cfg.get("fallback_without_splitter", "compressed")
        self.overflow_include_chunks_in_tightness = self._bool(
            overflow_cfg.get("include_chunks_in_tightness", True)
        )
        
        logger.info(
            "AdaptiveBudgetManager initialized",
            extra={
                "base_min_score": self.base_min_score,
                "max_threshold": self.max_threshold,
                "compression_enabled": self.compression_enabled,
            }
        )
    
    @staticmethod
    def _bool(val) -> bool:
        """Convert config value to bool (handles string 'false'/'true')."""
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() not in ("false", "0", "no", "")
        return bool(val)
    
    async def _get_llm(self):
        """
        Lazily resolve LLM provider via ProviderMapper.resolve_chain().

        v7.2.2: Uses the centralized chain pattern (same as rag_orchestrator,
        enrichment_pipeline, etc.) with health filtering and N-level fallback.
        On failure, retries after _llm_retry_interval seconds (no permanent latch).
        """
        # Fast path: already resolved successfully
        if self._llm_resolved and self._llm_provider is not None:
            return self._llm_provider

        # Retry gate: don't re-resolve too frequently after failure
        if self._llm_resolve_failed_at is not None:
            import time
            elapsed = time.time() - self._llm_resolve_failed_at
            if elapsed < self._llm_retry_interval:
                return self._llm_provider  # still None — wait for retry interval

        if not self._di_container:
            logger.warning("[BUDGET] No DI container available for LLM resolution")
            return None

        try:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper

            # Use resolve_chain with "enrichment" role (summarization is enrichment-tier)
            chain = ProviderMapper.resolve_chain("enrichment")
            if not chain:
                logger.warning("[BUDGET] resolve_chain('enrichment') returned empty chain")
                import time
                self._llm_resolve_failed_at = time.time()
                return None

            # Iterate chain: try each provider until one resolves from DI
            self._llm_chain = []
            primary_module = None

            for idx, (module_name, provider_name) in enumerate(chain):
                role_label = "Primary" if idx == 0 else f"Fallback-{idx}"
                try:
                    module = await self._di_container.resolve(module_name)
                    if module is None:
                        logger.warning(
                            f"[BUDGET] [{role_label}] {module_name} not in DI container — skipping"
                        )
                        continue

                    if hasattr(module, "set_default_provider"):
                        module.set_default_provider(provider_name)

                    self._llm_chain.append((module, provider_name))
                    if primary_module is None:
                        primary_module = module
                        logger.info(
                            f"[BUDGET] [{role_label}] LLM resolved: {module_name} ({provider_name})"
                        )
                except Exception as resolve_err:
                    logger.warning(
                        f"[BUDGET] [{role_label}] Could not resolve {module_name}: {resolve_err}"
                    )

            if primary_module:
                self._llm_provider = primary_module
                self._llm_resolved = True
                self._llm_resolve_failed_at = None
                logger.info(
                    f"[BUDGET] LLM chain resolved: {len(self._llm_chain)} provider(s) available"
                )
                return self._llm_provider
            else:
                logger.warning("[BUDGET] All providers in chain failed to resolve")
                import time
                self._llm_resolve_failed_at = time.time()
                return None

        except Exception as e:
            logger.warning(f"[BUDGET] Chain resolution failed: {e}")
            import time
            self._llm_resolve_failed_at = time.time()
            return None

    def calculate_tightness(
        self,
        total_tokens: int,
        current_overhead_tokens: int,
        turn_count: int
    ) -> Dict[str, float]:
        """
        Calculate tightness factor based on current token usage and conversation turns.
        
        Tightness ranges from 0 (ample space) to 1 (very tight).
        
        Formula:
        - used_fraction = current_overhead / total_tokens
        - turn_penalty = min(turn_count * turn_penalty_factor, 0.5)
        - tightness = min(max(used_fraction + turn_penalty, 0), 1)
        
        Args:
            total_tokens: Total context window size
            current_overhead_tokens: Current token overhead (query, template, etc.)
            turn_count: Current conversation turn number
            
        Returns:
            Dict with tightness, used_fraction, and turn_penalty
        """
        # Calculate fraction of context window already used
        used_fraction = current_overhead_tokens / total_tokens if total_tokens > 0 else 0
        
        # Penalize based on conversation turns (memory grows over time)
        # Cap at 0.5 to avoid over-penalization
        turn_penalty = min(turn_count * self.turn_penalty_factor, 0.5)
        
        # Combined tightness factor (0-1 scale)
        tightness = min(max(used_fraction + turn_penalty, 0), 1)
        
        logger.debug(
            f"Tightness calculated: {tightness:.2f}",
            extra={
                "total_tokens": total_tokens,
                "current_overhead": current_overhead_tokens,
                "turn_count": turn_count,
                "used_fraction": used_fraction,
                "turn_penalty": turn_penalty,
                "tightness": tightness,
            }
        )
        
        return {
            "tightness": tightness,
            "used_fraction": used_fraction,
            "turn_penalty": turn_penalty,
        }
    
    def calculate_memory_allocation(
        self,
        total_tokens: int,
        tightness: float
    ) -> int:
        """
        Calculate dynamic memory token allocation based on tightness.
        
        Allocation scales from min_memory_fraction (when ample space)
        to max_memory_fraction (when tight) of total context window.
        
        Args:
            total_tokens: Total context window size
            tightness: Current tightness factor (0-1)
            
        Returns:
            Tokens allocated for memory
        """
        # Scale memory fraction based on tightness
        # Low tightness (0.0) -> min_memory_fraction (e.g., 20%)
        # High tightness (1.0) -> max_memory_fraction (e.g., 40%)
        memory_fraction = self.min_memory_fraction + (
            tightness * (self.max_memory_fraction - self.min_memory_fraction)
        )
        
        memory_tokens = int(total_tokens * memory_fraction)
        
        logger.debug(
            f"Memory allocation: {memory_tokens} tokens ({memory_fraction:.1%})",
            extra={
                "total_tokens": total_tokens,
                "tightness": tightness,
                "memory_fraction": memory_fraction,
                "memory_tokens": memory_tokens,
            }
        )
        
        return memory_tokens
    
    def calculate_adjusted_threshold(self, tightness: float) -> float:
        """
        Calculate adjusted similarity threshold based on tightness.
        
        Scales threshold from base_min_score (when ample space)
        to max_threshold (when very tight), with a quality_floor
        minimum to prevent irrelevant chunks with large context windows.
        
        Args:
            tightness: Current tightness factor (0-1)
            
        Returns:
            Adjusted similarity threshold
        """
        # Scale threshold based on tightness
        # Low tightness -> base_min_score (e.g., 0.15)
        # High tightness -> max_threshold (e.g., 0.50)
        adjusted_score = self.base_min_score + (
            tightness * (self.max_threshold - self.base_min_score)
        )
        
        # Ensure within bounds
        adjusted_score = min(adjusted_score, self.max_threshold)

        # Quality floor — minimum threshold regardless of tightness.
        # With large context windows (Grok 2M), tightness~0 → threshold~base_min_score
        # which lets irrelevant chunks through. quality_floor prevents this.
        if adjusted_score < self.quality_floor:
            adjusted_score = self.quality_floor

        logger.debug(
            f"Threshold adjusted: {adjusted_score:.2f} (floor={self.quality_floor:.2f})",
            extra={
                "tightness": tightness,
                "base_min_score": self.base_min_score,
                "quality_floor": self.quality_floor,
                "adjusted_score": adjusted_score,
            }
        )

        return adjusted_score

    # Turn-boundary regex: line-anchored, case-insensitive, optional space before colon
    TURN_SPLIT_PATTERN = re.compile(
        r'(?m)(?=^(?:User|Assistant|Utente|Assistente)\s*:)',
        re.IGNORECASE
    )

    def _smart_truncation(
        self,
        text: str,
        target_tokens: int,
        model: str = None,
        provider: str = None,
    ) -> str:
        """
        Smart truncation fallback: keep last N turns intact (token-capped),
        prepend truncated old-turns header. Works with plain text conversation_context.
        """
        min_recent_turns = int(self.config.get("min_recent_turns", 4))
        recent_token_ratio = 0.40  # recent turns get at most 40% of target

        turns = self.TURN_SPLIT_PATTERN.split(text)
        turns = [t.strip() for t in turns if t.strip()]

        if len(turns) <= min_recent_turns + 2:
            # Context already small — just hard-truncate to fit
            chars_estimate = int(target_tokens * 3.5)
            return text[:chars_estimate]

        # Token-based recent window: start from last turns,
        # expand until hitting token budget or running out of turns
        max_recent_tokens = int(target_tokens * recent_token_ratio)
        recent_turns = []
        recent_tokens = 0
        for turn in reversed(turns):
            turn_tokens = TokenCounter.count_tokens(turn, model, provider)
            if recent_tokens + turn_tokens > max_recent_tokens and len(recent_turns) >= min_recent_turns:
                break
            recent_turns.insert(0, turn)
            recent_tokens += turn_tokens

        old_turns = turns[:len(turns) - len(recent_turns)]
        recent_text = "\n".join(recent_turns)

        # Budget for old summary = target minus recent minus separator overhead
        old_budget = max(target_tokens - recent_tokens - 50, int(target_tokens * 0.2))
        old_text = "\n".join(old_turns)
        old_chars = int(old_budget * 3.5)
        old_summary = old_text[:old_chars]

        result = f"[Conversazione precedente riassunta]\n{old_summary}\n[/Riassunto]\n\n{recent_text}"

        # Final length check
        result_tokens = TokenCounter.count_tokens(result, model, provider)
        if result_tokens > target_tokens * 1.08:
            chars_limit = int(target_tokens * 3.5)
            result = result[:chars_limit]

        return result
    
    async def summarize_context(
        self,
        text: str,
        target_tokens: int,
        model: str = None,
        provider: str = None,
        strategy: str = "hierarchical"
    ) -> str:
        """
        Compress context using LLM-based summarization with reinforced prompt.
        
        Args:
            text: Text to summarize
            target_tokens: Target token count for summary
            model: Model to use for summarization (optional)
            provider: Provider to use for summarization (optional)
            strategy: 'hierarchical' (default) or 'recent' (preserves last turns)
            
        Returns:
            Compressed text (or original if summarization fails)
        """
        if not self.compression_enabled:
            logger.warning("[BUDGET] Compression disabled in config")
            return text

        # Count original tokens
        original_tokens = TokenCounter.count_tokens(text, model, provider)

        # If already within target, no need to compress
        if original_tokens <= target_tokens:
            logger.debug(
                f"[BUDGET] Text already within target ({original_tokens} <= {target_tokens})"
            )
            return text

        try:
            # --- Reinforced prompt (v7.2.1) ---
            min_target = int(target_tokens * 0.85)
            max_target = int(target_tokens * 1.10)

            prompt = (
                f"Sei un assistente specializzato nella compressione fedele di conversazioni lunghe.\n"
                f"Obiettivo: produrre un riassunto che contenga tra {min_target} e {max_target} token.\n\n"
                f"Regole obbligatorie:\n"
                f"1. Mantieni TUTTI i fatti importanti, nomi propri, numeri, date, decisioni, richieste esplicite.\n"
                f"2. Preserva l'ordine cronologico degli eventi principali.\n"
                f"3. Mantieni il significato e il tono originale della conversazione.\n"
                f"4. NON aggiungere commenti meta come 'riassunto compresso', 'ecco il riassunto', 'in sintesi'.\n"
                f"5. Scrivi in modo naturale e fluido, ma conciso.\n"
                f"6. Il testo finale DEVE essere lungo tra {min_target} e {max_target} token (contali con precisione).\n"
                f"7. Se necessario, espandi leggermente i punti chiave per raggiungere la lunghezza minima.\n\n"
                f"Contesto da comprimere:\n{text}\n\n"
                f"Ora scrivi il riassunto rispettando esattamente le regole sopra."
            )

            if strategy == "recent":
                prompt += "\nPriorità: ultimi 6-8 turni devono rimanere quasi integri."

            # Lazy resolve LLM via DI container (v7.2.2: chain-based)
            llm = await self._get_llm()
            if not llm:
                truncated = text[:target_tokens * 4] + "... [truncated - no LLM]"
                truncated_tokens = TokenCounter.count_tokens(truncated, model, provider)
                logger.warning(
                    "[BUDGET] TRUNCATION MODE: LLM provider unavailable, context truncated instead of compressed",
                    extra={
                        "reason": "no_llm_provider",
                        "original_tokens": original_tokens,
                        "target_tokens": target_tokens,
                        "truncated_tokens": truncated_tokens,
                    }
                )
                return truncated

            # v7.2.2: Try each provider in chain until one succeeds
            llm_chain = self._llm_chain if self._llm_chain else [(llm, "primary")]
            summary = ""
            last_error = None

            for chain_idx, (llm_module, prov_name) in enumerate(llm_chain):
                if not hasattr(llm_module, 'generate'):
                    logger.warning(
                        f"[BUDGET] Chain[{chain_idx}] '{prov_name}' has no 'generate' — skipping"
                    )
                    continue
                try:
                    result = await llm_module.generate(
                        prompt=prompt,
                        max_tokens=target_tokens + 120,
                        temperature=0.1,
                        provider=prov_name,
                    )
                    summary = result.get('text', result.get('content', ''))
                    if summary and len(summary.strip()) >= 10:
                        if chain_idx > 0:
                            logger.info(
                                f"[BUDGET] Chain fallback succeeded: '{prov_name}' (attempt {chain_idx + 1})"
                            )
                        break  # success
                    else:
                        logger.warning(
                            f"[BUDGET] Chain[{chain_idx}] '{prov_name}' returned empty summary — trying next"
                        )
                        summary = ""
                except Exception as gen_err:
                    last_error = gen_err
                    logger.warning(
                        f"[BUDGET] Chain[{chain_idx}] '{prov_name}' generate() failed: {gen_err} — trying next"
                    )

            if not summary or len(summary.strip()) < 10:
                truncated = text[:target_tokens * 4] + "... [truncated - empty]"
                truncated_tokens = TokenCounter.count_tokens(truncated, model, provider)
                logger.warning(
                    "[BUDGET] TRUNCATION MODE: All LLM providers in chain failed, context truncated",
                    extra={
                        "reason": "all_chain_failed",
                        "original_tokens": original_tokens,
                        "target_tokens": target_tokens,
                        "truncated_tokens": truncated_tokens,
                        "chain_length": len(llm_chain),
                        "last_error": str(last_error) if last_error else None,
                    }
                )
                return truncated

            summary_tokens = TokenCounter.count_tokens(summary, model, provider)

            # --- Post-LLM validation (v7.2.1) ---
            MIN_ACCEPTABLE_RATIO = 0.70
            MAX_ACCEPTABLE_RATIO = 1.20

            if summary_tokens < target_tokens * MIN_ACCEPTABLE_RATIO:
                logger.warning(
                    f"[BUDGET] Summary too short: {summary_tokens} tok vs ~{target_tokens} "
                    f"(ratio {summary_tokens / target_tokens:.2f}). Fallback: smart truncation"
                )
                summary = self._smart_truncation(text, target_tokens, model, provider)
                summary_tokens = TokenCounter.count_tokens(summary, model, provider)
            elif summary_tokens > target_tokens * MAX_ACCEPTABLE_RATIO:
                logger.warning(
                    f"[BUDGET] Summary too long: {summary_tokens} tok vs ~{target_tokens}. Hard truncating."
                )
                chars_estimate = int(target_tokens * 3.5)
                summary = summary[:chars_estimate]
                summary_tokens = TokenCounter.count_tokens(summary, model, provider)

            logger.info(
                "[BUDGET] Context compressed via LLM",
                extra={
                    "original_tokens": original_tokens,
                    "summary_tokens": summary_tokens,
                    "compression_ratio": summary_tokens / original_tokens if original_tokens > 0 else 0,
                    "chain_length": len(self._llm_chain),
                }
            )
            return summary

        except Exception as e:
            truncated = text[:target_tokens * 4] + "... [truncated - error]"
            truncated_tokens = TokenCounter.count_tokens(truncated, model, provider)
            logger.error(
                f"[BUDGET] TRUNCATION MODE: LLM compression failed with exception, context truncated — {e}",
                extra={
                    "reason": "llm_exception",
                    "original_tokens": original_tokens,
                    "target_tokens": target_tokens,
                    "truncated_tokens": truncated_tokens,
                    "error": str(e),
                },
                exc_info=True,
            )
            return truncated
    
    async def adjust_for_pure_chat(
        self,
        query: str,
        conversation_context: Optional[str],
        turn_count: int,
        chat_config: Dict[str, Any],
        model: str = None,
        provider: str = None,
        max_tokens_client: Optional[int] = None,
        system_prompt_tokens: int = 0,
        tool_overhead_tokens: int = 0,
        memory_tokens_estimate: int = 0,
    ) -> Dict[str, Any]:
        """
        Adjust memory budget for pure chat (non-RAG) scenarios.
        
        Delegates numeric calculation to calculate_pure_chat_budget() (pure function),
        then handles LLM-based compression if needed.
        """
        # v4.1.2: Resolve provider via ProviderMapper if not provided
        if not provider:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper
            provider = ProviderMapper.get_provider_for_role_safe("chat")
            logger.debug(f"[PURE_CHAT] Provider resolved via ProviderMapper: {provider}")

        # Get context window from model/provider
        limits = TokenCounter.get_model_limits(model, provider)
        total_tokens = limits.context_window

        # Calculate token counts for pure function
        if system_prompt_tokens > 0:
            template_tokens = system_prompt_tokens
        else:
            system_prompt_template = chat_config.get(
                "system_prompt",
                "You are a helpful assistant."
            )
            template_tokens = TokenCounter.count_tokens(system_prompt_template, model, provider)
        query_format = f"\n\nUser: {query}\n\nAssistant:"
        query_tokens = TokenCounter.count_tokens(query_format, model, provider)

        current_conversation_tokens = 0
        if conversation_context:
            current_conversation_tokens = TokenCounter.count_tokens(
                conversation_context, model, provider
            )

        response_budget_init = max_tokens_client or chat_config.get("max_tokens", 1000)
        mcp_overhead = tool_overhead_tokens + memory_tokens_estimate

        # Pure function call — no HTTP, no side effects
        budget = calculate_pure_chat_budget(
            system_prompt_tokens=template_tokens,
            query_tokens=query_tokens,
            conversation_tokens=current_conversation_tokens,
            current_turn=turn_count,
            context_window=total_tokens,
            max_tokens_client=response_budget_init,
            mcp_overhead_tokens=mcp_overhead,
            turn_penalty_factor=self.turn_penalty_factor,
            compression_threshold=self.compression_threshold,
            compression_enabled=self.compression_enabled,
            min_compression_tokens=int(self.config.get("min_compression_tokens", 800)),
            safety_hard_limit=0.92,
            min_memory_fraction=self.min_memory_fraction,
        )

        # LLM compression if needed (async, requires DI)
        compression_applied = False
        compression_mode = budget["compression_mode"]
        pre_compression_tokens = current_conversation_tokens

        if budget["compression_needed"] and conversation_context is not None:
            compression_target = budget["compression_target"]
            logger.info(
                f"[PURE_CHAT] Compressing ({compression_mode}): "
                f"{current_conversation_tokens} → {compression_target} tokens "
                f"(tightness={budget['tightness']:.2f})",
                extra={
                    "original_tokens": current_conversation_tokens,
                    "target_tokens": compression_target,
                    "tightness": budget["tightness"],
                    "compression_mode": compression_mode,
                }
            )
            conversation_context = await self.summarize_context(
                conversation_context, compression_target, model, provider
            )
            current_conversation_tokens = TokenCounter.count_tokens(
                conversation_context, model, provider
            )
            compression_applied = True

            # Recalculate budget post-compression with updated token count
            budget = calculate_pure_chat_budget(
                system_prompt_tokens=template_tokens,
                query_tokens=query_tokens,
                conversation_tokens=current_conversation_tokens,
                current_turn=turn_count,
                context_window=total_tokens,
                max_tokens_client=response_budget_init,
                mcp_overhead_tokens=mcp_overhead,
                turn_penalty_factor=self.turn_penalty_factor,
                compression_threshold=self.compression_threshold,
                compression_enabled=self.compression_enabled,
                min_compression_tokens=int(self.config.get("min_compression_tokens", 800)),
                safety_hard_limit=0.92,
                min_memory_fraction=self.min_memory_fraction,
            )

        # Build result dict (backward compatible with existing callers)
        result = {
            "conversation_context": conversation_context,
            "memory_tokens": budget["memory_tokens"],
            "response_budget_tokens": budget["response_budget_tokens"],
            "tightness": budget["tightness"],
            "compression_applied": compression_applied,
            "compression_mode": compression_mode if compression_applied else None,
            "context_window": total_tokens,
            "fixed_overhead_tokens": budget["fixed_overhead_tokens"],
            "utilization_pct": budget["utilization_pct"],
        }

        if compression_applied:
            result["pre_compression_tokens"] = pre_compression_tokens
            result["compression_ratio"] = round(
                current_conversation_tokens / pre_compression_tokens, 2
            ) if pre_compression_tokens > 0 else None

        logger.info(
            "[PURE_CHAT] Budget adjustment completed",
            extra={
                "tightness": f"{budget['tightness']:.2f}",
                "memory_tokens": budget["memory_tokens"],
                "response_budget_tokens": budget["response_budget_tokens"],
                "compression_applied": compression_applied,
                "compression_mode": compression_mode if compression_applied else None,
                "utilization_pct": budget["utilization_pct"],
                "system_prompt_tokens": system_prompt_tokens,
                "tool_overhead_tokens": tool_overhead_tokens,
                "memory_tokens_estimate": memory_tokens_estimate,
                "mcp_overhead": mcp_overhead,
            }
        )

        return result
    
    async def adjust(
        self,
        query: str,
        conversation_context: Optional[str],
        retrieved_docs: List[Dict[str, Any]],
        turn_count: int,
        rag_config: Dict[str, Any],
        model: str = None,
        provider: str = None,
        max_tokens_client: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main adjustment method - orchestrates adaptive budget management.
        
        Process:
        1. Calculate tightness factor
        2. Allocate memory tokens dynamically
        3. Compress conversation context if needed
        4. Adjust similarity threshold
        5. Filter documents by adjusted threshold
        6. Calculate remaining budget for documents
        7. Compress documents if still tight
        
        Args:
            query: User query
            conversation_context: Conversation history (formatted)
            retrieved_docs: Documents from retrieval
            turn_count: Current conversation turn
            rag_config: RAG configuration with template, etc.
            model: Model name (for token counting)
            provider: Provider name (for token counting)
            
        Returns:
            Dict with adjusted context, filtered docs, and metadata
        """
        # v4.1.2: Resolve provider via ProviderMapper if not provided
        logger.info(f"[BUDGET] adjust() called: model={model}, provider={provider}")
        if not provider:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper
            provider = ProviderMapper.get_provider_for_role_safe("rag")
            logger.info(f"[BUDGET] Provider resolved via ProviderMapper: {provider}")

        # Get context window from model/provider
        limits = TokenCounter.get_model_limits(model, provider)
        total_tokens = limits.context_window

        # v6.8.0: FIX-BUDGET-004 — Respect explicit context_limit_tokens as cap (same as get_execution_plan)
        explicit_limit = rag_config.get("context_limit_tokens")
        if explicit_limit and int(explicit_limit) > 0 and int(explicit_limit) < total_tokens:
            logger.info(
                f"[BUDGET] adjust() context_limit_tokens={explicit_limit} caps "
                f"provider context_window={total_tokens} → using {explicit_limit}",
            )
            total_tokens = int(explicit_limit)

        logger.info(f"[BUDGET] Context window: {total_tokens} tokens (provider={provider})")

        # Calculate fixed overhead (template, query, response budget)
        # Calculate fixed overhead in token space (v7.2.0: Token-First)
        system_prompt_template = rag_config.get(
            "system_prompt",
            "Use the following context to answer the question:\n\n{context}"
        )
        # Exclude {context} placeholder from template overhead
        template_text = system_prompt_template.replace("{context}", "")
        template_tokens = TokenCounter.count_tokens(template_text, model, provider)
        query_format = f"\n\nQuestion: {query}\n\nAnswer:"
        query_tokens = TokenCounter.count_tokens(query_format, model, provider)
        
        response_budget_tokens = max_tokens_client or rag_config.get("max_tokens", 1000)
        safety_tokens = 300  # Fixed safety margin in tokens (higher for RAG)
        
        fixed_overhead_tokens = template_tokens + query_tokens + response_budget_tokens + safety_tokens
        
        # Calculate current overhead including conversation context
        current_conversation_tokens = 0
        if conversation_context:
            current_conversation_tokens = TokenCounter.count_tokens(
                conversation_context, model, provider
            )
        
        current_overhead_tokens = fixed_overhead_tokens + current_conversation_tokens
        
        # Step 1: Calculate tightness
        tightness_result = self.calculate_tightness(
            total_tokens, current_overhead_tokens, turn_count
        )
        tightness = tightness_result["tightness"]
        
        # Step 2: Calculate memory allocation
        min_memory_tokens = self.calculate_memory_allocation(total_tokens, tightness)
        
        # Step 3: Dual-gate compression (v7.2.0)
        MIN_COMPRESSION_TOKENS = int(self.config.get("min_compression_tokens", 800))
        compression_applied = False
        compression_mode = None
        pre_compression_tokens = current_conversation_tokens

        should_compress = (
            conversation_context is not None
            and self.compression_enabled
            and (
                current_conversation_tokens > min_memory_tokens
                or (tightness >= self.compression_threshold
                    and current_conversation_tokens >= MIN_COMPRESSION_TOKENS)
            )
        )

        if should_compress:
            if current_conversation_tokens > min_memory_tokens:
                compression_target = min_memory_tokens
                compression_mode = "reactive"
            else:
                t_range = max(1.0 - self.compression_threshold, 0.01)
                t_ratio = min((tightness - self.compression_threshold) / t_range, 1.0)
                keep_ratio = max(0.5, 0.8 - 0.3 * t_ratio)
                compression_target = int(current_conversation_tokens * keep_ratio)
                compression_mode = "proactive"

            logger.info(
                f"[BUDGET] Compressing conversation ({compression_mode}): "
                f"{current_conversation_tokens} → {compression_target} tokens "
                f"(tightness={tightness:.2f})",
                extra={
                    "original_tokens": current_conversation_tokens,
                    "target_tokens": compression_target,
                    "tightness": tightness,
                    "compression_mode": compression_mode,
                }
            )
            conversation_context = await self.summarize_context(
                conversation_context, compression_target, model, provider
            )
            current_conversation_tokens = TokenCounter.count_tokens(
                conversation_context, model, provider
            )
            compression_applied = True
        
        # Step 4: Adjust similarity threshold
        adjusted_min_score = self.calculate_adjusted_threshold(tightness)
        
        # Step 5: Filter documents by adjusted threshold
        # FIX-BUDGET-004: Use rerank_score when available (post-reranking quality score)
        # Rerank scores (0-1, typically 0.5-0.9) are more reliable than raw cosine similarity
        # (typically 0.1-0.3 for high-dimensional embeddings). Fall back to raw score.
        original_doc_count = len(retrieved_docs)
        filtered_docs = [
            doc for doc in retrieved_docs
            if (doc.get("metadata", {}).get("rerank_score") or doc.get("score", 0.0)) >= adjusted_min_score
        ]
        filtered_doc_count = len(filtered_docs)
        
        logger.info(
            f"Filtered documents: {original_doc_count} -> {filtered_doc_count} (threshold: {adjusted_min_score:.2f})",
            extra={
                "original_count": original_doc_count,
                "filtered_count": filtered_doc_count,
                "adjusted_threshold": adjusted_min_score,
                "tightness": tightness,
            }
        )
        
        # Step 6: Calculate remaining budget for documents
        memory_tokens = current_conversation_tokens
        doc_budget_tokens = total_tokens - (fixed_overhead_tokens + memory_tokens)
        
        # Ensure minimum budget
        doc_budget_tokens = max(doc_budget_tokens, int(total_tokens * 0.1))
        
        # v7.2.1: Safety hard limit — cap doc budget when committed tokens
        # (overhead + memory) exceed 92% of context. See pure_chat fix comment.
        SAFETY_HARD_LIMIT_RATIO = 0.92
        committed_tokens = fixed_overhead_tokens + memory_tokens
        if total_tokens > 0 and committed_tokens > total_tokens * SAFETY_HARD_LIMIT_RATIO:
            doc_budget_tokens = int(total_tokens * SAFETY_HARD_LIMIT_RATIO) - fixed_overhead_tokens - memory_tokens
            doc_budget_tokens = max(doc_budget_tokens, int(total_tokens * 0.05))
            logger.warning(
                f"[BUDGET] SAFETY: Committed tokens {committed_tokens} exceeds "
                f"{SAFETY_HARD_LIMIT_RATIO*100:.0f}% of {total_tokens}. "
                f"Doc budget reduced to {doc_budget_tokens}",
            )
        
        # Step 7: Document compression (if still very tight)
        # This would be implemented in the augment phase of RAG pipeline
        # For now, we just flag it
        doc_compression_needed = doc_budget_tokens < total_tokens * 0.1
        
        if doc_compression_needed and self.compression_enabled:
            logger.warning(
                f"Document budget very tight ({doc_budget_tokens} tokens), compression recommended",
                extra={
                    "doc_budget_tokens": doc_budget_tokens,
                    "tightness": tightness,
                    "filtered_doc_count": filtered_doc_count,
                }
            )
        
        result = {
            "conversation_context": conversation_context,
            "filtered_docs": filtered_docs,
            "doc_budget_tokens": doc_budget_tokens,
            "tightness": tightness,
            "adjusted_min_score": adjusted_min_score,
            "memory_tokens": memory_tokens,
            "compression_applied": compression_applied,
            "compression_mode": compression_mode,
            "original_doc_count": original_doc_count,
            "filtered_doc_count": filtered_doc_count,
            "doc_compression_needed": doc_compression_needed,
            "context_window": total_tokens,
            "fixed_overhead_tokens": fixed_overhead_tokens,
        }
        
        if compression_applied:
            result["pre_compression_tokens"] = pre_compression_tokens
            result["compression_ratio"] = round(
                current_conversation_tokens / pre_compression_tokens, 2
            ) if pre_compression_tokens > 0 else None
        
        logger.info(
            "Budget adjustment completed",
            extra={
                "tightness": f"{tightness:.2f}",
                "memory_tokens": memory_tokens,
                "doc_budget_tokens": doc_budget_tokens,
                "filtered_docs": filtered_doc_count,
                "compression_applied": compression_applied,
                "compression_mode": compression_mode,
            }
        )
        
        return result

    # =========================================================================
    # CONTEXT GOVERNOR v3.7.0 - NEW METHODS
    # =========================================================================

    def _get_tightness_thresholds(self) -> Dict[str, float]:
        """
        Get tightness thresholds from config.
        
        Returns dict with: emergency, critical, tight, comfortable
        """
        thresholds = self.config.get("tightness_thresholds", {})
        return {
            "emergency": float(thresholds.get("emergency", 0.95)),
            "critical": float(thresholds.get("critical", 0.85)),
            "tight": float(thresholds.get("tight", 0.7)),
            "comfortable": float(thresholds.get("comfortable", 0.5)),
        }

    def _determine_context_strategy(self, tightness: float) -> str:
        """
        Determine context handling strategy based on tightness.
        
        Args:
            tightness: Current tightness factor (0-1)
            
        Returns:
            Strategy name: full, compressed, metadata_only, emergency
        """
        thresholds = self._get_tightness_thresholds()
        
        if tightness >= thresholds["emergency"]:
            return "emergency"
        elif tightness >= thresholds["critical"]:
            return "metadata_only"
        elif tightness >= thresholds["tight"]:
            return "compressed"
        else:
            return "full"

    def suggest_system_instruction(self, tightness: float, base_instruction: Optional[str] = None) -> str:
        """
        Generate dynamic system instruction based on tightness.
        
        Injects context-aware directives to guide LLM behavior when
        context space is limited.
        
        Args:
            tightness: Current tightness factor (0-1)
            base_instruction: Optional base system prompt to enhance
            
        Returns:
            Enhanced system instruction with tightness-based directives
        """
        # Check if dynamic prompts are enabled
        context_governor_config = self.config.get("context_governor", {})
        if not context_governor_config.get("dynamic_prompt", False):
            return base_instruction or ""
        
        thresholds = self._get_tightness_thresholds()
        
        # Build directive based on tightness level
        directive = ""
        
        if tightness >= thresholds["emergency"]:
            directive = (
                "IMPORTANTE: Lo spazio di contesto è ESTREMAMENTE LIMITATO. "
                "DEVI rispondere in modo TELEGRAFICO: "
                "- Usa solo parole chiave e elenchi puntati "
                "- Nessuna introduzione o conclusione "
                "- Massimo 2-3 frasi per concetto "
                "- Evita formule di cortesia"
            )
        elif tightness >= thresholds["critical"]:
            directive = (
                "AVVISO: Spazio di contesto limitato. "
                "Rispondi in modo CONCISO: "
                "- Vai dritto al punto "
                "- Usa frasi brevi "
                "- Evita ripetizioni e ridondanze"
            )
        elif tightness >= thresholds["tight"]:
            directive = (
                "Nota: Sii moderatamente conciso nella risposta. "
                "Evita digressioni non strettamente necessarie."
            )
        # Below tight threshold: no directive needed
        
        if base_instruction and directive:
            return f"{base_instruction}\n\n{directive}"
        elif directive:
            return directive
        else:
            return base_instruction or ""

    def _compute_overflow_strategy(
        self,
        doc_budget_tokens: int,
        total_chunk_tokens: int,
        total_chunks: int,
        context_window: int,
        task_profile: str,
        tightness: float,
        modules_available: Dict[str, bool],
        user_preferences: Optional[UserPreferences] = None,
    ) -> Dict[str, Any]:
        """
        v6.3.0: Decide overflow strategy based on chunk/budget ratio.

        The budget manager is the SINGLE decision maker for overflow handling.
        Returns a dict of fields to merge into ExecutionPlan.

        Called ONLY when total_chunk_tokens > 0 (new callers).
        Legacy callers (total_chunk_tokens=0) skip this entirely.

        v6.3.0 Fase 4: When user_preferences is provided and
        overflow_preference != "auto", the preference directly selects
        the strategy (detailed→split, focused→selective, overview→summarize).
        expertise_level == "expert" lowers the split threshold by 25%.
        """
        result: Dict[str, Any] = {
            "overflow_ratio": 0.0,
            "chunk_tokens_available": doc_budget_tokens,
            "chunks_that_fit": total_chunks,
            "chunks_dropped": 0,
            "strategy_details": {},
            "split_recommended": False,
            "split_sections": 0,
            "split_reason": "",
            "split_chunks_per_section": [],
        }

        if total_chunk_tokens <= 0 or doc_budget_tokens <= 0:
            result["context_strategy"] = "full"
            return result

        avg_chunk_tokens = total_chunk_tokens / max(total_chunks, 1)
        overflow_ratio = total_chunk_tokens / doc_budget_tokens
        chunks_that_fit = int(doc_budget_tokens / avg_chunk_tokens) if avg_chunk_tokens > 0 else 0
        chunks_dropped = max(0, total_chunks - chunks_that_fit)

        result["overflow_ratio"] = round(overflow_ratio, 3)
        result["chunk_tokens_available"] = doc_budget_tokens
        result["chunks_that_fit"] = chunks_that_fit
        result["chunks_dropped"] = chunks_dropped

        has_splitter = modules_available.get("window_split_merge", False)
        is_complex_task = task_profile in self.overflow_prefer_split_tasks

        # --- STRATEGY SELECTION ---

        # 1. Everything fits
        if overflow_ratio <= 1.0:
            result["context_strategy"] = "full"
            result["strategy_details"] = {
                "action": "none",
                "reason": f"No overflow ({overflow_ratio:.2f}x), all {total_chunks} chunks fit",
            }
            return result

        # 1.5 v6.3.0 Fase 4: Explicit user preference overrides auto-selection
        if (user_preferences
                and user_preferences.overflow_preference != "auto"
                and overflow_ratio > 1.0):
            pref = user_preferences.overflow_preference
            if pref == "detailed":
                if has_splitter:
                    sections = max(2, min(int(overflow_ratio + 0.5), 6))
                    result["context_strategy"] = "split"
                    result["split_recommended"] = True
                    result["split_sections"] = sections
                    result["split_reason"] = (
                        f"User preference: detailed (overflow {overflow_ratio:.2f}x)"
                    )
                    result["split_chunks_per_section"] = self._distribute_count(
                        total_chunks, sections
                    )
                    result["strategy_details"] = {
                        "action": "split",
                        "reason": result["split_reason"],
                        "params": {"sections": sections, "source": "user_preference"},
                    }
                else:
                    result["context_strategy"] = "compressed"
                    result["strategy_details"] = {
                        "action": "priority_truncate",
                        "reason": (
                            f"User preference: detailed, but splitter unavailable "
                            f"(overflow {overflow_ratio:.2f}x), fallback to compressed"
                        ),
                        "params": {"method": "priority_truncate", "source": "user_preference"},
                    }
                return result
            elif pref == "focused":
                result["context_strategy"] = "selective"
                result["strategy_details"] = {
                    "action": "rerank",
                    "reason": (
                        f"User preference: focused (overflow {overflow_ratio:.2f}x), "
                        f"keep top {chunks_that_fit} of {total_chunks}"
                    ),
                    "params": {"keep_top": chunks_that_fit, "method": "relevance_drop",
                               "source": "user_preference"},
                }
                return result
            elif pref == "overview":
                result["context_strategy"] = "summarize"
                result["strategy_details"] = {
                    "action": "pre_summarize",
                    "reason": (
                        f"User preference: overview (overflow {overflow_ratio:.2f}x)"
                    ),
                    "params": {"target_ratio": 0.3, "method": "extractive",
                               "source": "user_preference"},
                }
                return result

        # 2. Slight overflow → selective (rerank + drop low relevance)
        if overflow_ratio <= self.overflow_selective_threshold:
            result["context_strategy"] = "selective"
            result["strategy_details"] = {
                "action": "rerank",
                "reason": (
                    f"Slight overflow {overflow_ratio:.2f}x, "
                    f"drop {chunks_dropped} of {total_chunks} least relevant chunks"
                ),
                "params": {"keep_top": chunks_that_fit, "method": "relevance_drop"},
            }
            return result

        # 3. Moderate overflow → depends on task_profile and user expertise
        if overflow_ratio <= self.overflow_compressed_threshold:
            # v6.3.0 Fase 4: expert users get split even for non-complex tasks
            expert_boost = (user_preferences is not None
                            and user_preferences.expertise_level == "expert")
            if (is_complex_task or expert_boost) and has_splitter:
                sections = max(2, min(int(overflow_ratio + 0.5), 4))
                result["context_strategy"] = "split"
                result["split_recommended"] = True
                result["split_sections"] = sections
                result["split_reason"] = (
                    f"Overflow {overflow_ratio:.2f}x, task={task_profile}, "
                    f"split to preserve detail"
                )
                result["split_chunks_per_section"] = self._distribute_count(
                    total_chunks, sections
                )
                result["strategy_details"] = {
                    "action": "split",
                    "reason": result["split_reason"],
                    "params": {"sections": sections},
                }
            else:
                result["context_strategy"] = "compressed"
                action = "priority_truncate" if is_complex_task else "tail_truncate"
                result["strategy_details"] = {
                    "action": action,
                    "reason": (
                        f"Overflow {overflow_ratio:.2f}x, task={task_profile}, "
                        f"{'splitter unavailable, ' if is_complex_task else ''}"
                        f"truncate to fit"
                    ),
                    "params": {"method": action},
                }
            return result

        # 4. Significant overflow → split if available
        if overflow_ratio <= self.overflow_summarize_threshold:
            if has_splitter:
                sections = max(2, min(int(overflow_ratio + 0.5), 6))
                result["context_strategy"] = "split"
                result["split_recommended"] = True
                result["split_sections"] = sections
                result["split_reason"] = (
                    f"Overflow {overflow_ratio:.2f}x, split into {sections} sections"
                )
                result["split_chunks_per_section"] = self._distribute_count(
                    total_chunks, sections
                )
                result["strategy_details"] = {
                    "action": "split",
                    "reason": result["split_reason"],
                    "params": {"sections": sections},
                }
            else:
                # Fallback: hybrid rerank + truncate
                result["context_strategy"] = self.overflow_fallback_no_splitter
                result["strategy_details"] = {
                    "action": "hybrid",
                    "reason": (
                        f"Overflow {overflow_ratio:.2f}x, splitter unavailable, "
                        f"hybrid fallback"
                    ),
                    "params": {
                        "steps": [
                            {"action": "rerank", "keep_top": chunks_that_fit + 2},
                            {"action": "tail_truncate"},
                        ]
                    },
                }
            return result

        # 5. Extreme overflow (> summarize_threshold)
        if has_splitter:
            sections = min(int(overflow_ratio), 6)
            result["context_strategy"] = "split"
            result["split_recommended"] = True
            result["split_sections"] = sections
            result["split_reason"] = (
                f"Extreme overflow {overflow_ratio:.2f}x, split into {sections} sections"
            )
            result["split_chunks_per_section"] = self._distribute_count(
                total_chunks, sections
            )
            result["strategy_details"] = {
                "action": "split",
                "reason": result["split_reason"],
                "params": {"sections": sections},
            }
        else:
            result["context_strategy"] = "summarize"
            result["strategy_details"] = {
                "action": "pre_summarize",
                "reason": (
                    f"Extreme overflow {overflow_ratio:.2f}x, "
                    f"pre-summarize chunks to fit"
                ),
                "params": {"target_ratio": 0.25, "method": "extractive"},
            }
        return result

    @staticmethod
    def _distribute_count(total: int, sections: int) -> List[int]:
        """Evenly distribute total items across sections."""
        if sections <= 0:
            return []
        base = total // sections
        remainder = total % sections
        return [base + (1 if i < remainder else 0) for i in range(sections)]

    def get_execution_plan(
        self,
        query: str,
        turn_count: int,
        provider: str = "grok",
        model: str = None,
        task_profile: str = "chat",
        current_memory_tokens: int = 0,
        rag_config: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        # --- v6.3.0: Overflow strategy parameters (all with defaults) ---
        total_chunk_tokens: int = 0,
        total_chunks: int = 0,
        modules_available: Optional[Dict[str, bool]] = None,
        user_preferences: Optional["UserPreferences"] = None,
    ) -> "ExecutionPlan":
        """
        Generate unified ExecutionPlan for a request.
        
        This is the MAIN OUTPUT of Context Governor - a single source of truth
        for all RAG pipeline parameters.
        
        v3.7.1: Now supports 4-layer constraints from collection metadata and client policy.
        v6.3.0: Accepts total_chunk_tokens/total_chunks for overflow strategy decision.
                 When total_chunk_tokens > 0, computes overflow_ratio and selects one of
                 5 strategies: full, selective, compressed, split, summarize.
                 When total_chunk_tokens == 0 (legacy), behavior is IDENTICAL to before.
        
        Args:
            query: User query
            turn_count: Current conversation turn number
            provider: Target provider name (grok, ollama, etc.)
            model: Target model name (optional)
            task_profile: Task type (reasoning, chat, extraction)
            current_memory_tokens: Tokens already used by conversation memory
            rag_config: Optional RAG configuration dict
            constraints: Optional 4-layer constraints dict with:
                - max_doc_budget_tokens: Hard cap on document budget (from collection/client)
                - max_memory_budget_tokens: Hard cap on memory budget
                - max_top_k: Maximum chunks to retrieve (from collection policy)
                - min_similarity_threshold: Floor for similarity (from collection policy)
                - allowed_strategies: List of allowed ContextStrategy values
                - source: str indicating constraint origin ("collection", "client", "user")
            total_chunk_tokens: v6.3.0 — Total tokens across all retrieved chunks (0 = legacy)
            total_chunks: v6.3.0 — Number of retrieved chunks (0 = legacy)
            modules_available: v6.3.0 — Dict of available modules {"window_split_merge": True, ...}
            
        Returns:
            ExecutionPlan with all calculated parameters, respecting constraints
        """
        from .models import (
            ExecutionPlan,
            ContextStrategy,
            ResponseStyle,
            TASK_PROFILES,
        )
        
        rag_config = rag_config or {}
        
        # Get context window from ENV-aware resolution
        limits = get_provider_limits_from_env(provider)
        context_window = limits.context_window

        # v6.8.0: FIX-BUDGET-004 — Respect explicit context_limit_tokens as cap.
        # Callers like Architect set context_limit_tokens (e.g. 131072) to cap budget
        # even when the provider has a much larger window (Grok=2M). Without this,
        # context_limit_tokens is a dead setting when Context Governor is active.
        explicit_limit = rag_config.get("context_limit_tokens")
        if explicit_limit and int(explicit_limit) > 0 and int(explicit_limit) < context_window:
            logger.info(
                f"[BUDGET] context_limit_tokens={explicit_limit} caps "
                f"provider context_window={context_window} → using {explicit_limit}",
            )
            context_window = int(explicit_limit)
        
        # Get task profile configuration
        profile = TASK_PROFILES.get(task_profile, TASK_PROFILES["chat"])
        
        # Calculate reserved output tokens based on profile
        reserved_output_tokens = int(context_window * profile.output_ratio)
        
        # Calculate query overhead
        chars_per_token = rag_config.get("chars_per_token", 3.5)
        query_tokens = int(len(query) / chars_per_token)
        safety_margin = rag_config.get("safety_margin", 500)
        query_overhead_tokens = query_tokens + safety_margin
        
        # Calculate current overhead
        current_overhead_tokens = query_overhead_tokens + current_memory_tokens
        
        # v6.3.0: Add chunk_pressure to tightness when chunk info is available
        chunk_pressure = 0
        if (total_chunk_tokens > 0
                and self.overflow_strategy_enabled
                and self.overflow_include_chunks_in_tightness):
            # Pressure = min(chunk_tokens, doc_budget estimate) — can't exceed what fits
            # Use a preliminary doc_budget estimate before final calculation
            preliminary_doc_budget = max(
                (context_window - reserved_output_tokens)
                - current_memory_tokens - query_overhead_tokens,
                int(context_window * 0.05),
            )
            chunk_pressure = min(total_chunk_tokens, preliminary_doc_budget)

        # Calculate tightness (with chunk_pressure added to overhead)
        tightness_result = self.calculate_tightness(
            context_window, current_overhead_tokens + chunk_pressure, turn_count
        )
        tightness = tightness_result["tightness"]

        # Determine context strategy
        context_strategy = self._determine_context_strategy(tightness)
        
        # Calculate memory allocation
        memory_budget_tokens = self.calculate_memory_allocation(context_window, tightness)
        
        # Calculate max input tokens (context_window - reserved_output)
        max_input_tokens = context_window - reserved_output_tokens
        
        # Calculate doc budget (remaining after memory and query overhead)
        doc_budget_tokens = max_input_tokens - memory_budget_tokens - query_overhead_tokens
        doc_budget_tokens = max(doc_budget_tokens, int(context_window * 0.05))  # Minimum 5%

        # v6.3.1: SAFETY NET — Ensure minimum response tokens (non-negotiable).
        # If actual chunks would leave insufficient response space, reduce doc_budget.
        min_resp = self.min_response_tokens_override if self.min_response_tokens_override > 0 else profile.min_response_tokens
        if total_chunk_tokens > 0:
            actual_doc_tokens = min(total_chunk_tokens, doc_budget_tokens)
            real_response_space = context_window - memory_budget_tokens - query_overhead_tokens - actual_doc_tokens
            if real_response_space < min_resp:
                doc_budget_tokens = context_window - memory_budget_tokens - query_overhead_tokens - min_resp
                doc_budget_tokens = max(doc_budget_tokens, int(context_window * 0.05))
                logger.warning(
                    f"[BUDGET] Response space insufficient ({real_response_space} < {min_resp}), "
                    f"reduced doc_budget to {doc_budget_tokens} to protect min_response_tokens",
                    extra={
                        "real_response_space": real_response_space,
                        "min_response_tokens": min_resp,
                        "doc_budget_tokens": doc_budget_tokens,
                        "total_chunk_tokens": total_chunk_tokens,
                        "task_profile": task_profile,
                    },
                )
            else:
                logger.info(
                    f"[BUDGET] Response space OK: {real_response_space} tokens available "
                    f"(min_required={min_resp}, task={task_profile})",
                )
        else:
            allocated_response = context_window - memory_budget_tokens - query_overhead_tokens - doc_budget_tokens
            if allocated_response < min_resp:
                doc_budget_tokens = context_window - memory_budget_tokens - query_overhead_tokens - min_resp
                doc_budget_tokens = max(doc_budget_tokens, int(context_window * 0.05))
                logger.warning(
                    f"[BUDGET] Allocated response space insufficient ({allocated_response} < {min_resp}), "
                    f"reduced doc_budget to {doc_budget_tokens}",
                )

        # Update reserved_output_tokens to reflect actual available response space
        reserved_output_tokens = context_window - memory_budget_tokens - query_overhead_tokens - doc_budget_tokens

        # Calculate adjusted similarity threshold
        similarity_threshold = self.calculate_adjusted_threshold(tightness)
        
        # Calculate RAG parameters based on tightness
        base_top_k = rag_config.get("top_k", 10)
        if tightness >= 0.85:  # Critical
            rag_top_k = max(3, base_top_k // 3)
            oversample_factor = 2.0
        elif tightness >= 0.7:  # Tight
            rag_top_k = max(5, base_top_k // 2)
            oversample_factor = 3.0
        else:  # Comfortable
            rag_top_k = base_top_k
            oversample_factor = 4.0
        
        # Generate dynamic system instruction
        system_instruction_modifier = self.suggest_system_instruction(
            tightness, 
            base_instruction=None
        )
        
        # Determine response style based on tightness
        if tightness >= 0.85:
            response_style = ResponseStyle.TELEGRAPHIC
        elif tightness >= 0.7:
            response_style = ResponseStyle.CONCISE
        elif tightness >= 0.5:
            response_style = ResponseStyle.STANDARD
        else:
            response_style = profile.response_style
        
        # Check if compression is recommended
        compression_recommended = (
            tightness >= self.compression_threshold and
            self.compression_enabled
        )
        
        # v6.2.4: Check if window splitting is recommended
        # Split is recommended when tightness exceeds threshold AND
        # the doc budget is insufficient for a reasonable response.
        # This is provider-agnostic — any provider at the limit triggers it.
        split_recommended = False
        if self.split_enabled and tightness >= self.split_tightness_threshold:
            # Estimate overflow: if reserved output eats >50% of remaining budget,
            # the generate call will likely overflow or produce degraded output
            remaining_for_docs = max_input_tokens - query_overhead_tokens - memory_budget_tokens
            if remaining_for_docs < reserved_output_tokens:
                split_recommended = True
            elif tightness >= 0.95:
                # Emergency: always recommend split
                split_recommended = True
        
        if split_recommended:
            logger.info(
                "[CONTEXT-GOVERNOR] Window split recommended",
                extra={
                    "tightness": f"{tightness:.2f}",
                    "threshold": f"{self.split_tightness_threshold:.2f}",
                    "provider": provider,
                    "context_window": context_window,
                    "doc_budget": doc_budget_tokens,
                    "reserved_output": reserved_output_tokens,
                }
            )
        
        # === v3.7.1: Apply 4-layer constraints from collection/client ===
        constraints = constraints or {}
        constraints_applied = []
        
        # Apply max_doc_budget_tokens constraint (from collection or client policy)
        if constraints.get("max_doc_budget_tokens"):
            max_allowed = constraints["max_doc_budget_tokens"]
            if doc_budget_tokens > max_allowed:
                doc_budget_tokens = max_allowed
                constraints_applied.append(f"doc_budget capped to {max_allowed}")
        
        # Apply max_memory_budget_tokens constraint
        if constraints.get("max_memory_budget_tokens"):
            max_allowed = constraints["max_memory_budget_tokens"]
            if memory_budget_tokens > max_allowed:
                memory_budget_tokens = max_allowed
                constraints_applied.append(f"memory_budget capped to {max_allowed}")
        
        # Apply max_top_k constraint (from collection policy)
        if constraints.get("max_top_k"):
            max_allowed = constraints["max_top_k"]
            if rag_top_k > max_allowed:
                rag_top_k = max_allowed
                constraints_applied.append(f"top_k capped to {max_allowed}")
        
        # Apply min_similarity_threshold constraint (collection may require higher quality)
        if constraints.get("min_similarity_threshold"):
            min_required = constraints["min_similarity_threshold"]
            if similarity_threshold < min_required:
                similarity_threshold = min_required
                constraints_applied.append(f"similarity raised to {min_required}")
        
        # Apply allowed_strategies constraint (client may restrict compression)
        if constraints.get("allowed_strategies"):
            allowed = constraints["allowed_strategies"]
            if context_strategy not in allowed:
                # Fall back to most permissive allowed strategy
                if "full" in allowed:
                    context_strategy = "full"
                elif "compressed" in allowed:
                    context_strategy = "compressed"
                elif "metadata_only" in allowed:
                    context_strategy = "metadata_only"
                # emergency is last resort
                constraints_applied.append(f"strategy restricted to {context_strategy}")
        
        # Log constraint application
        if constraints_applied:
            logger.info(
                "[CONTEXT-GOVERNOR] 4-layer constraints applied",
                extra={
                    "constraints_source": constraints.get("source", "unknown"),
                    "applied": constraints_applied,
                    "original_constraints": constraints,
                }
            )
        
        # === v6.3.0: Compute overflow strategy when chunk info is available ===
        overflow_fields = {}
        if (total_chunk_tokens > 0
                and self.overflow_strategy_enabled):
            _modules = modules_available or {}
            overflow_fields = self._compute_overflow_strategy(
                doc_budget_tokens=doc_budget_tokens,
                total_chunk_tokens=total_chunk_tokens,
                total_chunks=total_chunks,
                context_window=context_window,
                task_profile=task_profile,
                tightness=tightness,
                modules_available=_modules,
                user_preferences=user_preferences,
            )
            # Overflow strategy may override context_strategy and split_recommended
            if "context_strategy" in overflow_fields:
                context_strategy = overflow_fields.pop("context_strategy")
            if overflow_fields.get("split_recommended"):
                split_recommended = True
        
        logger.info(
            f"[CONTEXT-GOVERNOR] ExecutionPlan generated: "
            f"provider={provider} ctx={context_window} "
            f"tightness={tightness:.3f} doc_budget={doc_budget_tokens} "
            f"reserved_output={reserved_output_tokens} "
            f"overflow_ratio={overflow_fields.get('overflow_ratio', 0):.2f} "
            f"context_strategy={context_strategy} "
            f"overflow_action={overflow_fields.get('strategy_details', {}).get('action', 'none')} "
            f"split_rec={split_recommended} chunk_tokens={total_chunk_tokens}",
            extra={
                "provider": provider,
                "context_window": context_window,
                "tightness": f"{tightness:.2f}",
                "strategy": context_strategy,
                "profile": task_profile,
                "max_input_tokens": max_input_tokens,
                "doc_budget": doc_budget_tokens,
                "memory_budget": memory_budget_tokens,
                "rag_top_k": rag_top_k,
                "similarity_threshold": f"{similarity_threshold:.2f}",
                "response_style": response_style.value if hasattr(response_style, 'value') else response_style,
                "split_recommended": split_recommended,
                "constraints_applied": constraints_applied,  # v3.7.1
                "overflow_ratio": overflow_fields.get("overflow_ratio", 0),  # v6.3.0
                "overflow_strategy": overflow_fields.get("strategy_details", {}).get("action", "none"),  # v6.3.0
            }
        )
        
        return ExecutionPlan(
            # Token budget
            max_input_tokens=max_input_tokens,
            reserved_output_tokens=reserved_output_tokens,
            doc_budget_tokens=doc_budget_tokens,
            memory_budget_tokens=memory_budget_tokens,
            query_overhead_tokens=query_overhead_tokens,
            
            # Context state
            tightness=tightness,
            context_strategy=ContextStrategy(context_strategy),
            
            # RAG parameters
            rag_top_k=rag_top_k,
            similarity_threshold=similarity_threshold,
            oversample_factor=oversample_factor,
            reranking_enabled=rag_config.get("reranking_enabled", True),
            
            # Response guidance
            system_instruction_modifier=system_instruction_modifier if system_instruction_modifier else None,
            response_style=response_style,
            suggested_temperature=profile.temperature,
            
            # Task context
            task_profile=task_profile,
            provider_name=provider,
            model_name=model,
            context_window=context_window,
            
            # Metadata
            turn_count=turn_count,
            compression_recommended=compression_recommended,
            split_recommended=split_recommended,
            split_tightness_threshold=self.split_tightness_threshold,
            constraints_applied=constraints_applied,  # v3.7.1
            constraints_source=constraints.get("source") if constraints else None,  # v3.7.1
            
            # v6.3.0: Overflow strategy fields
            overflow_ratio=overflow_fields.get("overflow_ratio", 0.0),
            chunk_tokens_available=overflow_fields.get("chunk_tokens_available", 0),
            chunks_that_fit=overflow_fields.get("chunks_that_fit", 0),
            chunks_dropped=overflow_fields.get("chunks_dropped", 0),
            strategy_details=overflow_fields.get("strategy_details", {}),
            split_sections=overflow_fields.get("split_sections", 0),
            split_reason=overflow_fields.get("split_reason", ""),
            split_chunks_per_section=overflow_fields.get("split_chunks_per_section", []),
        )


# =============================================================================
# EXPORTS
    # =========================================================================
    # DCBL: recalculate_after_delta — ricalcolo budget dopo ogni tool result
    # =========================================================================

    async def recalculate_after_delta(
        self,
        cumulative_history_tokens: int = 0,
        delta_tokens: int = 0,
        message_count: int = 0,
        synopsis_tokens: int = 0,
        previous_budget: Optional[Dict[str, Any]] = None,
        model: str = None,
        provider: str = None,
        system_prompt_tokens: int = 0,
        tool_overhead_tokens: int = 0,
        structured_memory_tokens: int = 0,
        original_budget_max_tokens: int = 0,
        is_post_compression_recalc: bool = False,
        **_: Any,
    ) -> BudgetState:
        """DCBL module wrapper over the canonical pure runtime authority.

        The public module operation stays stable for dispatch/HTTP callers and
        still returns ``BudgetState``. Numeric DCBL logic is delegated to
        ``mcp_runtime.core.budget_calc.recalculate_after_delta`` to avoid drift.
        """
        result = pure_recalculate_after_delta(
            cumulative_history_tokens=cumulative_history_tokens,
            delta_tokens=delta_tokens,
            synopsis_tokens=synopsis_tokens,
            previous_budget=previous_budget,
            system_prompt_tokens=system_prompt_tokens,
            tool_overhead_tokens=tool_overhead_tokens,
            structured_memory_tokens=structured_memory_tokens,
            original_budget_max_tokens=original_budget_max_tokens,
            is_post_compression_recalc=is_post_compression_recalc,
            compression_threshold=self.compression_threshold,
            compression_enabled=self.compression_enabled,
        )

        logger.info(
            "[DCBL] module wrapper: history=%d delta=%d budget=%d "
            "compress=%s level=%d post_compress=%s",
            cumulative_history_tokens,
            delta_tokens,
            result["response_budget_tokens"],
            result["compression_mode"],
            result["compression_level"],
            is_post_compression_recalc,
        )

        return BudgetState(
            context_window=result["context_window"],
            model=model or "unknown",
            provider=provider or "unknown",
            fixed_overhead_tokens=result["fixed_overhead_tokens"],
            structured_memory_tokens=result["structured_memory_tokens"],
            history_tokens=result["history_tokens"],
            delta_tokens=result["delta_tokens"],
            total_estimated_tokens=result["total_estimated_tokens"],
            tightness=result["tightness"],
            response_budget_tokens=result["response_budget_tokens"],
            compression_needed=result["compression_needed"],
            compression_level=result["compression_level"],
            compression_mode=result["compression_mode"],
            compression_target=result["compression_target"],
            safety_margin_used=result["safety_margin_used"],
            is_critical=result["is_critical"],
            round_count=result["round_count"],
            session_id=result["session_id"],
            original_budget_max_tokens=result["original_budget_max_tokens"],
        )


# =============================================================================

__all__ = [
    "AdaptiveBudgetManager",
    "BudgetState",
]
