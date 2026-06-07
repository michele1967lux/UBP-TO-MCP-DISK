"""
Pydantic models for HIAMS.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProfileType(str, Enum):
    """Profile type for parameter selection."""

    CHAT = "chat"
    AGENT_LOOP = "agent_loop"


class CoverageClass(str, Enum):
    """Coverage class used for retention and eviction."""

    ORDER = "order"
    DIETARY = "dietary"
    EVENT = "event"
    LOGISTICS = "logistics"
    GENERAL = "general"


class ProjectionIntent(str, Enum):
    """High-level projection intent."""

    ORDER = "order"
    DIETARY = "dietary"
    EVENT = "event"
    LOGISTICS = "logistics"
    CROSS_THREAD = "cross_thread"
    GENERAL = "general"


class OrderItem(BaseModel):
    """Structured order item."""

    name: str
    quantity: int = Field(default=1, ge=1)
    unit_price: Optional[float] = None
    status: str = "active"
    notes: List[str] = Field(default_factory=list)


class OrderState(BaseModel):
    """Protected slot for the current order."""

    items: List[OrderItem] = Field(default_factory=list)
    removed_items: List[str] = Field(default_factory=list)
    substitutions: Dict[str, str] = Field(default_factory=dict)
    current_total: Optional[float] = None
    currency: str = "EUR"
    total_history: List[float] = Field(default_factory=list)


class DietaryProfile(BaseModel):
    """Protected slot for dietary constraints."""

    constraints: List[str] = Field(default_factory=list)
    allergens: List[str] = Field(default_factory=list)
    contamination_level: Optional[str] = None
    safe_items: List[str] = Field(default_factory=list)


class EventPlan(BaseModel):
    """Protected slot for event planning."""

    name: Optional[str] = None
    location: Optional[str] = None
    time: Optional[str] = None
    status: str = "unknown"
    notes: List[str] = Field(default_factory=list)


class LogisticsState(BaseModel):
    """Protected slot for logistics and service mode."""

    directions: List[str] = Field(default_factory=list)
    service_mode: Optional[str] = None
    supplement: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class KeyEntities(BaseModel):
    """Protected slot for critical named entities."""

    products: List[str] = Field(default_factory=list)
    people: List[str] = Field(default_factory=list)
    places: List[str] = Field(default_factory=list)
    events: List[str] = Field(default_factory=list)
    prices: List[str] = Field(default_factory=list)
    times: List[str] = Field(default_factory=list)


class StructuredSlots(BaseModel):
    """Protected memory slots - never compressed away."""

    order_state: OrderState = Field(default_factory=OrderState)
    dietary_profile: DietaryProfile = Field(default_factory=DietaryProfile)
    event_plan: EventPlan = Field(default_factory=EventPlan)
    logistics: LogisticsState = Field(default_factory=LogisticsState)
    negative_decisions: List[str] = Field(default_factory=list)
    key_entities: KeyEntities = Field(default_factory=KeyEntities)


class SlotUpdateResult(BaseModel):
    """Result of structured slot update."""

    slots: StructuredSlots
    changed_slots: List[str] = Field(default_factory=list)
    new_facts: List[str] = Field(default_factory=list)
    info_gain_score: float = 0.0


class AdaptiveSnapshot(BaseModel):
    """Delta-oriented Layer 0 snapshot."""

    snapshot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    turn_number: int = Field(..., ge=0)
    timestamp: float = Field(default_factory=time.time)
    focus: str
    intent: str = ""
    query_summary: str
    response_summary: str = ""
    coverage_classes: List[CoverageClass] = Field(default_factory=list)
    key_facts: List[str] = Field(default_factory=list)
    structured_delta: Dict[str, Any] = Field(default_factory=dict)
    referenced_slots: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    user_salience: float = 0.0
    info_gain_score: float = 0.0
    absorbed: bool = False


class AdaptiveLayer1Block(BaseModel):
    """Compressed mid-term block."""

    block_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    topic: str
    turn_range: List[int] = Field(default_factory=list)
    coverage_classes: List[CoverageClass] = Field(default_factory=list)
    conversation_focus: str = ""
    key_facts: List[str] = Field(default_factory=list)
    active_threads: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    entity_index: List[str] = Field(default_factory=list)
    numeric_facts: List[str] = Field(default_factory=list)
    directions: List[str] = Field(default_factory=list)
    event_refs: List[str] = Field(default_factory=list)
    order_refs: List[str] = Field(default_factory=list)
    negative_decisions: List[str] = Field(default_factory=list)
    quality_score: float = 0.0
    info_gain_score: float = 0.0
    last_updated_turn: int = 0
    covered_turns: List[int] = Field(default_factory=list)


class AdaptiveLayer2Memory(BaseModel):
    """Long-term persistent memory."""

    version: int = 1
    stable_facts: List[str] = Field(default_factory=list)
    slot_facts: List[str] = Field(default_factory=list)
    promoted_items: List[str] = Field(default_factory=list)
    event_timeline: List[str] = Field(default_factory=list)
    order_summary: List[str] = Field(default_factory=list)
    dietary_summary: List[str] = Field(default_factory=list)
    logistics_summary: List[str] = Field(default_factory=list)
    unresolved_threads: List[str] = Field(default_factory=list)
    coverage_index: Dict[str, List[str]] = Field(default_factory=dict)
    last_updated_turn: int = 0


class ProjectionResult(BaseModel):
    """Projected context returned to the caller."""

    intent: ProjectionIntent
    rendered_context: str
    token_estimate: int = 0
    budget_tokens: int = 0
    included_components: List[str] = Field(default_factory=list)
    selected_slot_fields: List[str] = Field(default_factory=list)
    included_layer0_turns: List[int] = Field(default_factory=list)
    included_layer1_blocks: List[str] = Field(default_factory=list)
    included_layer2: bool = False


class CompressionResult(BaseModel):
    """Adaptive compression result."""

    success: bool
    compression_triggered: bool = False
    layer1_block: Optional[AdaptiveLayer1Block] = None
    layer2: Optional[AdaptiveLayer2Memory] = None
    consumed_snapshots: int = 0
    reason: str = ""
    info_gain_score: float = 0.0
    latency_ms: float = 0.0


class HIAMSSessionState(BaseModel):
    """Full state for one HIAMS session."""

    session_id: str
    structured_slots: StructuredSlots = Field(default_factory=StructuredSlots)
    layer0: List[AdaptiveSnapshot] = Field(default_factory=list)
    layer1_blocks: List[AdaptiveLayer1Block] = Field(default_factory=list)
    layer2: AdaptiveLayer2Memory = Field(default_factory=AdaptiveLayer2Memory)
    total_turns: int = 0
    total_compressions: int = 0
    last_compression_turn: int = -1
    query_history: List[str] = Field(default_factory=list)
    projection_history: List[ProjectionIntent] = Field(default_factory=list)


class HIAMSProfileSettings(BaseModel):
    """Per-profile settings."""

    compression_trigger_threshold: int = Field(default=6, ge=2)
    min_turns_between_compressions: int = Field(default=3, ge=1)
    min_delta_info_score: float = Field(default=4.0, ge=0.0)
    layer0_base_window: int = Field(default=8, ge=3)
    layer0_recall_window: int = Field(default=12, ge=4)
    max_layer1_blocks: int = Field(default=5, ge=2)
    projection_budget_tokens: int = Field(default=2800, ge=400)
    projection_budget_tokens_cross_thread: int = Field(default=4200, ge=800)
    enable_layer2: bool = True


def _default_agent_loop_profile() -> HIAMSProfileSettings:
    return HIAMSProfileSettings(
        compression_trigger_threshold=5,
        min_turns_between_compressions=2,
        min_delta_info_score=3.0,
        layer0_base_window=6,
        layer0_recall_window=10,
        max_layer1_blocks=4,
        projection_budget_tokens=2200,
        projection_budget_tokens_cross_thread=3200,
    )


class HIAMSConfig(BaseModel):
    """Top-level configuration."""

    enabled: bool = True
    log_level: str = "INFO"
    chat_profile: HIAMSProfileSettings = Field(default_factory=HIAMSProfileSettings)
    agent_loop_profile: HIAMSProfileSettings = Field(default_factory=_default_agent_loop_profile)
    compression_provider_override: Optional[str] = None
    compression_model_override: Optional[str] = None

    def get_profile(self, profile_type: ProfileType) -> HIAMSProfileSettings:
        if profile_type == ProfileType.AGENT_LOOP:
            return self.agent_loop_profile
        return self.chat_profile


class ProcessingResult(BaseModel):
    """Result of processing a turn."""

    structured_slots: StructuredSlots
    slot_update: SlotUpdateResult
    snapshot: AdaptiveSnapshot
    compression_result: Optional[CompressionResult] = None
    projected_context: ProjectionResult
