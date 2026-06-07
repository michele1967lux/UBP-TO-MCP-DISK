"""
Pydantic models for RAG Multi-Layer Memory.

Defines validated structures for Layer 0 (snapshots), Layer 1 (compressed blocks),
and Layer 2 (long-term minimal memory).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Layer 0 — Sub-Layer Zero (Snapshot)
# =============================================================================

class PreferencesModel(BaseModel):
    """User preferences — explicit and inferred."""
    explicit: List[str] = Field(default_factory=list)
    inferred: List[str] = Field(default_factory=list)


class SubLayerZeroSnapshot(BaseModel):
    """
    Single snapshot in Layer 0 (Working Memory).

    Represents the contextual state at a specific turn.
    Contains both fixed fields and a dynamic_context section
    that is completely client-aware and domain-specific.
    """
    turn: int = Field(..., ge=0, description="Turn number")
    focus: str = Field(..., description="Short, precise focus string")
    intent: str = Field("", description="User's main intent")
    key_facts: List[str] = Field(default_factory=list, description="Key facts from this turn")
    preferences: PreferencesModel = Field(default_factory=PreferencesModel)
    state_change: Optional[str] = Field(None, description="State change description or null")
    entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generic entities (names, dates, numbers, recurring objects)"
    )
    dynamic_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Completely dynamic, client-aware context section"
    )
    pending: List[str] = Field(
        default_factory=list,
        description="Open questions and points to clarify"
    )


# =============================================================================
# Layer 1 — Compressed Block
# =============================================================================

class Layer1Block(BaseModel):
    """
    Single compressed block in Layer 1 (Evolved Mid-term Compression).

    Each block summarizes a range of turns, capturing evolution,
    user choices, rules, and specifications.
    """
    turn_range: str = Field(..., description="Turn range covered (e.g., '18-22')")
    focus: str = Field(..., description="Main focus of this block")
    evolution_summary: str = Field(
        "", description="Brief description of the evolution in this range"
    )
    user_choices: List[str] = Field(default_factory=list, description="Important user choices")
    user_rules: List[str] = Field(default_factory=list, description="Rules set by the user")
    specifications: List[str] = Field(
        default_factory=list, description="Technical specifications requested"
    )
    key_facts: List[str] = Field(default_factory=list, description="Essential facts")
    preferences: PreferencesModel = Field(default_factory=PreferencesModel)
    dynamic_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Completely dynamic, client-aware context"
    )
    importance: int = Field(5, ge=1, le=10, description="Importance score 1-10")
    last_updated_turn: int = Field(0, ge=0, description="Last turn that updated this block")


# =============================================================================
# Layer 2 — Long-term Minimal Memory
# =============================================================================

class Layer2Memory(BaseModel):
    """
    Long-term Minimal Memory (Layer 2).

    Extremely minimal and conservative — updated only when compression
    of Layer 1 identifies truly important, persistent information.
    """
    critical_facts: List[str] = Field(default_factory=list, description="Critical persistent facts")
    stable_preferences: PreferencesModel = Field(default_factory=PreferencesModel)
    core_rules: List[str] = Field(default_factory=list, description="Strong/fixed rules")
    core_specifications: List[str] = Field(
        default_factory=list, description="Persistent specifications"
    )
    dynamic_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Only truly stable, cross-cutting keys"
    )
    last_updated_turn: int = Field(0, ge=0, description="Last turn that updated Layer 2")


# =============================================================================
# Session State — Aggregated memory for one conversation
# =============================================================================

class SessionMemoryState(BaseModel):
    """
    Complete memory state for a single conversation session.

    Holds Layer 0 (snapshots), Layer 1 (compressed blocks),
    and Layer 2 (long-term) together.
    """
    session_id: str = Field(..., description="Conversation session identifier")
    layer0: List[SubLayerZeroSnapshot] = Field(
        default_factory=list, description="Working memory snapshots (sliding window)"
    )
    layer1: List[Layer1Block] = Field(
        default_factory=list, description="Compressed mid-term blocks"
    )
    layer2: Layer2Memory = Field(
        default_factory=Layer2Memory, description="Long-term minimal memory"
    )
    total_turns: int = Field(0, description="Total turns processed")
    compression_count: int = Field(0, description="Number of compressions performed")


# =============================================================================
# Compression Result — Output from compression engine
# =============================================================================

class CompressionResult(BaseModel):
    """Result of a compression operation."""
    new_layer1_block: Optional[Layer1Block] = Field(
        None, description="New Layer 1 block (if produced)"
    )
    layer2_updated: bool = Field(False, description="Whether Layer 2 was updated")
    updated_layer2: Optional[Layer2Memory] = Field(
        None, description="Updated Layer 2 (if changed)"
    )
