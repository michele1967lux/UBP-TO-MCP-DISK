"""
Pydantic models for RAG Adaptive Memory module (v3.7.0).

Defines configuration and data models for adaptive token budget management
and Context Governor execution planning.

v3.7.0 additions:
- ExecutionPlan: Unified output from Context Governor
- TaskProfile: Task-specific output ratios and styles
- ContextStrategy: Enum for context handling strategies
- TightnessThresholds: Configuration for tightness levels

v6.3.0 additions:
- ContextStrategy: +SELECTIVE, +SPLIT, +SUMMARIZE for overflow strategies
- ExecutionPlan: +overflow_ratio, +chunks_that_fit, +chunks_dropped,
  +chunk_tokens_available, +strategy_details, +split_chunks_per_section,
  +split_sections, +split_reason
- UserPreferences: user preference model for overflow strategy hints
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, List, Literal
from enum import Enum


# =============================================================================
# CONTEXT GOVERNOR v3.7.0 - NEW MODELS
# =============================================================================

class ContextStrategy(str, Enum):
    """
    Context handling strategy based on tightness level.
    
    Determines how documents and memory are processed
    before being sent to the LLM.
    
    v6.3.0: Added SELECTIVE, SPLIT, SUMMARIZE for overflow strategies.
    """
    FULL = "full"                    # All content, no compression
    SELECTIVE = "selective"          # v6.3.0: Rerank + drop low-relevance chunks
    COMPRESSED = "compressed"        # LLM-based summarization / truncation
    SPLIT = "split"                  # v6.3.0: Distribute chunks across N sections
    SUMMARIZE = "summarize"          # v6.3.0: Pre-summarize chunks before generation
    METADATA_ONLY = "metadata_only"  # Only titles/headers, no content
    EMERGENCY = "emergency"          # Minimal context, drop documents


class ResponseStyle(str, Enum):
    """
    Suggested response style based on context tightness.
    
    Injected into system prompt to guide LLM behavior.
    """
    VERBOSE = "verbose"        # Detailed explanations, examples
    STANDARD = "standard"      # Normal response length
    CONCISE = "concise"        # Shorter, to-the-point
    TELEGRAPHIC = "telegraphic"  # Minimal, bullet points


class TaskProfile(BaseModel):
    """
    Task-specific configuration for output token allocation.
    
    Different tasks require different output/input ratios:
    - Reasoning: More output (40%) for chain-of-thought
    - Chat: Less output (10%) for conversational responses
    - Extraction: Moderate output (20%) for structured data
    """
    name: str = Field(description="Profile name: reasoning, chat, extraction, etc.")
    output_ratio: float = Field(
        default=0.15,
        ge=0.05,
        le=0.5,
        description="Fraction of context window reserved for output (0.1 = 10%)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Suggested temperature for this task type"
    )
    response_style: ResponseStyle = Field(
        default=ResponseStyle.STANDARD,
        description="Suggested response style for this task"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Suggested top_p for this task type"
    )
    min_response_tokens: int = Field(
        default=512,
        ge=100,
        le=8192,
        description="Minimum tokens guaranteed for LLM response (non-negotiable safety net)"
    )


# Pre-defined task profiles
TASK_PROFILES: Dict[str, TaskProfile] = {
    "reasoning": TaskProfile(
        name="reasoning",
        output_ratio=0.4,
        temperature=0.2,
        response_style=ResponseStyle.VERBOSE,
        top_p=0.95,
        min_response_tokens=1536,
    ),
    "chat": TaskProfile(
        name="chat",
        output_ratio=0.1,
        temperature=0.7,
        response_style=ResponseStyle.STANDARD,
        top_p=0.9,
        min_response_tokens=512,
    ),
    "rag": TaskProfile(
        name="rag",
        output_ratio=0.15,
        temperature=0.5,
        response_style=ResponseStyle.STANDARD,
        top_p=0.9,
        min_response_tokens=1024,
    ),
    "extraction": TaskProfile(
        name="extraction",
        output_ratio=0.2,
        temperature=0.0,
        response_style=ResponseStyle.TELEGRAPHIC,
        top_p=1.0,
        min_response_tokens=512,
    ),
    "summarization": TaskProfile(
        name="summarization",
        output_ratio=0.3,
        temperature=0.3,
        response_style=ResponseStyle.CONCISE,
        top_p=0.9,
        min_response_tokens=1024,
    ),
    "code": TaskProfile(
        name="code",
        output_ratio=0.5,
        temperature=0.1,
        response_style=ResponseStyle.VERBOSE,
        top_p=0.95,
        min_response_tokens=1536,
    ),
    "report": TaskProfile(
        name="report",
        output_ratio=0.35,
        temperature=0.3,
        response_style=ResponseStyle.VERBOSE,
        top_p=0.9,
        min_response_tokens=2048,
    ),
    "analysis": TaskProfile(
        name="analysis",
        output_ratio=0.3,
        temperature=0.3,
        response_style=ResponseStyle.VERBOSE,
        top_p=0.9,
        min_response_tokens=1536,
    ),
}


class TightnessThresholds(BaseModel):
    """
    Configuration for tightness level thresholds.
    
    Determines which ContextStrategy to use based on tightness value.
    """
    emergency: float = Field(
        default=0.95,
        ge=0.8,
        le=1.0,
        description="Threshold for EMERGENCY strategy (drop documents)"
    )
    critical: float = Field(
        default=0.85,
        ge=0.7,
        le=0.95,
        description="Threshold for METADATA_ONLY strategy"
    )
    tight: float = Field(
        default=0.7,
        ge=0.5,
        le=0.85,
        description="Threshold for COMPRESSED strategy"
    )
    comfortable: float = Field(
        default=0.5,
        ge=0.2,
        le=0.7,
        description="Below this = FULL strategy (no compression)"
    )


class ExecutionPlan(BaseModel):
    """
    Unified output from Context Governor for a single request.
    
    Contains all calculated parameters needed for RAG pipeline execution,
    replacing scattered calculations across multiple modules.
    
    This is the SINGLE SOURCE OF TRUTH for:
    - Token budget allocation (input, output, docs, memory)
    - RAG retrieval parameters (top_k, threshold, oversample)
    - Context strategy (full, compressed, metadata_only, emergency)
    - Response guidance (system prompt modifier, style)
    """
    model_config = ConfigDict(protected_namespaces=(), use_enum_values=True)
    # === TOKEN BUDGET ALLOCATION ===
    max_input_tokens: int = Field(
        description="Maximum tokens available for input (docs + memory + query)"
    )
    reserved_output_tokens: int = Field(
        description="Tokens reserved for LLM output"
    )
    doc_budget_tokens: int = Field(
        description="Tokens allocated for retrieved documents"
    )
    memory_budget_tokens: int = Field(
        description="Tokens allocated for conversation memory"
    )
    query_overhead_tokens: int = Field(
        default=500,
        description="Reserved tokens for query, template, safety margin"
    )
    
    # === CONTEXT STATE ===
    tightness: float = Field(
        ge=0.0,
        le=1.0,
        description="Current context tightness (0=ample, 1=critical)"
    )
    context_strategy: ContextStrategy = Field(
        description="How to handle context (full, compressed, metadata_only, emergency)"
    )
    
    # === RAG RETRIEVAL PARAMETERS ===
    rag_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of chunks to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieved docs"
    )
    oversample_factor: float = Field(
        default=4.0,
        ge=1.0,
        le=10.0,
        description="Oversample multiplier for reranking (retrieves top_k * oversample)"
    )
    reranking_enabled: bool = Field(
        default=True,
        description="Whether to apply reranking to results"
    )
    
    # === RESPONSE GUIDANCE ===
    system_instruction_modifier: Optional[str] = Field(
        default=None,
        description="Additional instruction to inject based on tightness (e.g., 'Sii conciso')"
    )
    response_style: ResponseStyle = Field(
        default=ResponseStyle.STANDARD,
        description="Suggested response style"
    )
    suggested_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Suggested temperature based on task profile"
    )
    
    # === TASK CONTEXT ===
    task_profile: str = Field(
        default="chat",
        description="Task profile used: reasoning, chat, extraction, etc."
    )
    provider_name: str = Field(
        default="unknown",
        description="Target provider for context window calculations"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Target model (if specified)"
    )
    context_window: int = Field(
        default=4096,
        description="Provider's total context window size"
    )
    
    # === METADATA ===
    turn_count: int = Field(
        default=0,
        ge=0,
        description="Current conversation turn number"
    )
    compression_recommended: bool = Field(
        default=False,
        description="Whether compression is recommended based on tightness"
    )
    split_recommended: bool = Field(
        default=False,
        description=(
            "v6.2.4: Whether window splitting is recommended. "
            "True when tightness >= split_tightness_threshold AND "
            "doc_budget overflow is detected. Consumers (pipeline, endpoint) "
            "use this flag to activate window_split_merge module."
        ),
    )
    split_tightness_threshold: float = Field(
        default=0.70,
        ge=0.3,
        le=1.0,
        description="v6.2.4: Tightness threshold that triggered split recommendation",
    )
    
    # === v6.3.0: OVERFLOW STRATEGY FIELDS ===
    overflow_ratio: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "v6.3.0: Ratio of total_chunk_tokens / doc_budget_tokens. "
            "0.0 = legacy caller (no chunk info), >1.0 = overflow detected."
        ),
    )
    chunk_tokens_available: int = Field(
        default=0,
        ge=0,
        description="v6.3.0: How many chunk tokens fit in doc_budget",
    )
    chunks_that_fit: int = Field(
        default=0,
        ge=0,
        description="v6.3.0: How many whole chunks fit in doc_budget",
    )
    chunks_dropped: int = Field(
        default=0,
        ge=0,
        description="v6.3.0: How many chunks would be dropped without overflow strategy",
    )
    strategy_details: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            'v6.3.0: Overflow strategy details. '
            '{"action": "split|truncate|rerank|hybrid", "reason": "...", "params": {...}}'
        ),
    )
    split_sections: int = Field(
        default=0,
        ge=0,
        description="v6.3.0: Recommended number of split sections (0 = no split)",
    )
    split_reason: str = Field(
        default="",
        description="v6.3.0: Human-readable reason for the chosen overflow strategy",
    )
    split_chunks_per_section: List[int] = Field(
        default_factory=list,
        description="v6.3.0: Distribution of chunks per section when split",
    )
    constraints_applied: List[str] = Field(
        default_factory=list,
        description="v3.7.1: List of 4-layer constraints that were applied (from collection/client policy)"
    )
    constraints_source: Optional[str] = Field(
        default=None,
        description="v3.7.1: Source of applied constraints (collection, client, user, merged)"
    )
    
    


# =============================================================================
# LEGACY MODELS (v3.5.0) - Kept for backward compatibility
# =============================================================================


class AdaptiveMemoryConfig(BaseModel):
    """Configuration for adaptive memory management."""
    
    enabled: bool = Field(default=True, description="Enable adaptive memory management")
    base_min_score: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Base similarity threshold for RAG retrieval"
    )
    max_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Maximum similarity threshold when context is tight"
    )
    min_memory_fraction: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Minimum fraction of context window for memory (20%)"
    )
    max_memory_fraction: float = Field(
        default=0.4,
        ge=0.2,
        le=0.6,
        description="Maximum fraction of context window for memory when tight (40%)"
    )
    turn_penalty_factor: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Penalty factor per turn for tightness calculation"
    )
    compression_enabled: bool = Field(
        default=True,
        description="Enable automatic context compression when needed"
    )
    compression_threshold: float = Field(
        default=0.5,
        ge=0.3,
        le=0.8,
        description="Tightness threshold to trigger compression (0.5 = 50%)"
    )
    support_llm_provider: str = Field(
        default="vllm",
        description="Provider for summarization LLM (fallback when env not set)"
    )
    # v6.0.1: support_llm_model removed — model resolved by inference module from provider


class BudgetAdjustmentResult(BaseModel):
    """Result of budget adjustment operation."""
    
    conversation_context: Optional[str] = Field(
        default=None,
        description="Potentially compressed conversation context"
    )
    filtered_docs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Documents filtered by adjusted threshold"
    )
    doc_budget_tokens: int = Field(
        description="Available tokens for documents"
    )
    tightness: float = Field(
        ge=0.0,
        le=1.0,
        description="Measure of context pressure (0=ample, 1=very tight)"
    )
    adjusted_min_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Adapted similarity threshold"
    )
    memory_tokens: int = Field(
        description="Tokens allocated to memory"
    )
    compression_applied: bool = Field(
        default=False,
        description="Whether compression was applied"
    )
    original_doc_count: int = Field(
        description="Original number of documents before filtering"
    )
    filtered_doc_count: int = Field(
        description="Number of documents after filtering"
    )


class TightnessResult(BaseModel):
    """Result of tightness calculation."""
    
    tightness: float = Field(
        ge=0.0,
        le=1.0,
        description="Tightness factor (0=ample space, 1=very tight)"
    )
    used_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of context window currently used"
    )
    turn_penalty: float = Field(
        ge=0.0,
        le=1.0,
        description="Penalty based on conversation turns"
    )


class SummarizationResult(BaseModel):
    """Result of context summarization."""
    
    summary: str = Field(description="Compressed text")
    original_tokens: int = Field(description="Original token count")
    summary_tokens: int = Field(description="Summary token count")
    compression_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Compression ratio (summary_tokens / original_tokens)"
    )


# =============================================================================
# v6.3.0 - USER PREFERENCES FOR OVERFLOW STRATEGY
# =============================================================================

class UserPreferences(BaseModel):
    """
    Resolved user preferences that influence overflow strategy selection.

    Built by UserPreferenceResolver from 4 sources (priority order):
    1. Explicit Redis preferences (user chose in past ask_user interaction)
    2. User profile (from user_profile_memory module)
    3. Session history (inferred from behavior)
    4. System defaults

    The budget manager reads these to bias strategy selection:
    - overflow_preference != "auto" → force a specific strategy
    - expertise_level == "expert" → lower split threshold (more detail)
    - interaction_willingness == "low" → never trigger ask_user
    """
    overflow_preference: str = Field(
        default="auto",
        description="Preferred overflow handling: detailed (→split), "
                    "focused (→selective), overview (→summarize), auto (→budget decides)"
    )
    detail_level: str = Field(
        default="auto",
        description="Desired detail level: high, medium, low, auto"
    )
    language: str = Field(
        default="auto",
        description="User language for localized prompts: it, en, auto"
    )
    expertise_level: str = Field(
        default="auto",
        description="User expertise: expert, intermediate, beginner, auto"
    )
    interaction_willingness: str = Field(
        default="auto",
        description="Willingness for interactive prompts: high, low, auto"
    )
    source: str = Field(
        default="default",
        description="Where these preferences came from: explicit, profile, history, default"
    )


__all__ = [
    # v3.7.0 - Context Governor models
    "ContextStrategy",
    "ResponseStyle",
    "TaskProfile",
    "TightnessThresholds",
    "ExecutionPlan",
    "TASK_PROFILES",
    # v6.3.0 - User preferences
    "UserPreferences",
    
    # Legacy models (v3.5.0)
    "AdaptiveMemoryConfig",
    "BudgetAdjustmentResult",
    "TightnessResult",
    "SummarizationResult",
]
