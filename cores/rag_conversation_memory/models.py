"""
Structured Memory v4.2.0 - Data Models

Thread-Based Summary with Smart Promote & Gentle Decay.

Models:
- ConversationTurn: A single topic entry in the conversation thread
- Topic: Legacy topic with decay tracking (backward compat)
- StructuredContext: Current conversation state (topic, intent, entities)
- MemoryState: Complete memory state including thread and history
- ContextResult: Result of context retrieval for LLM consumption

v4.2.0: Thread-based structured summary with importance-weighted fading
v2.0.0: Initial implementation for FEAT-MEM-002
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator, model_validator
import json


# --- Tool Memory models (v4.2.1) ---
# Turn-level synopsis of tool executions — bounded, never raw payload.

class ToolOutputSynopsis(BaseModel):
    """Bounded synopsis of tool output — never raw payload."""
    kind: str = Field(
        ...,
        description="retrieval | mutation | status | error | generic"
    )
    summary: str = Field(
        ...,
        max_length=200,
        description="Bounded summary, hard-capped at 200 chars"
    )
    result_count: Optional[int] = Field(
        default=None, ge=0,
        description="Number of results (if applicable)"
    )
    best_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Best relevance score (if retrieval)"
    )

    @model_validator(mode="after")
    def enforce_generic_summary_cap(self) -> "ToolOutputSynopsis":
        """kind=generic → summary hard-capped at 50 chars."""
        if self.kind == "generic" and len(self.summary) > 50:
            self.summary = self.summary[:50]
        return self


class ToolInputSynopsis(BaseModel):
    """Whitelist-derived input summary — never raw arg dump."""
    fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Max 5 whitelisted fields, each value ≤40 chars"
    )

    @field_validator("fields")
    @classmethod
    def enforce_field_caps(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Max 5 fields, each key ≤40 chars, each value ≤40 chars."""
        capped: Dict[str, str] = {}
        for key, val in list(v.items())[:5]:
            capped[str(key)[:40]] = str(val)[:40]
        return capped


class ToolUsageEntry(BaseModel):
    """Single tool execution synopsis for turn memory (not knowledge store)."""
    tool_name: str = Field(
        ..., max_length=100,
        description="Fully qualified tool name"
    )
    input_synopsis: ToolInputSynopsis = Field(
        default_factory=ToolInputSynopsis,
        description="Whitelist-derived input summary"
    )
    output_synopsis: ToolOutputSynopsis = Field(
        ..., description="Typed output synopsis with kind/summary"
    )
    source_refs: List[str] = Field(
        default_factory=list,
        description="Max 5 canonical type:id refs, each ≤100 chars"
    )
    success: bool = Field(
        default=True,
        description="Whether tool execution succeeded"
    )

    @field_validator("source_refs")
    @classmethod
    def enforce_ref_caps(cls, v: List[str]) -> List[str]:
        """Max 5 refs, each ≤100 chars, must contain ':' (type:id format)."""
        validated: List[str] = []
        for ref in v[:5]:
            ref_str = str(ref)[:100]
            if ":" in ref_str:
                validated.append(ref_str)
        return validated


class ConversationTurn(BaseModel):
    """
    A single topic entry in the conversation thread.

    Each turn represents a sub-topic discussed in the conversation.
    Turns can be merged (adjacent same-topic), promoted (resumed old topic),
    or archived (faded beyond threshold).
    """

    turn_number: int = Field(
        ..., description="Turn number (updated on merge/resume)"
    )
    focus: str = Field(
        ..., description="Sub-topic label"
    )
    key_facts: str = Field(
        default="", description="Key facts (truncated by code for detail_level)"
    )
    key_facts_full: str = Field(
        default="", description="Original key facts pre-truncation (restore on promote)"
    )
    detail_level: str = Field(
        default="full",
        description="Detail level: full/high/recent/fading/background"
    )
    importance: int = Field(
        default=5, ge=0, le=10,
        description="0-10 importance score assigned by LLM, modifies fading speed"
    )
    query: str = Field(
        default="", description="Original user query"
    )
    reactivation_boost: int = Field(
        default=0, ge=0,
        description="Resistance to fading (residual turns)"
    )
    anchor_sentence: str = Field(
        default="", description="Bridge sentence for reactivated topics"
    )
    is_resumed: bool = Field(
        default=False, description="True if topic was reactivated"
    )
    merge_count: int = Field(
        default=1, ge=1,
        description="How many turns merged into this entry"
    )
    # ROUTE-MODE-LLM: Pipeline lane suggestion for next turn
    suggested_lane: Optional[str] = Field(
        default=None,
        description="Pipeline suggested for next query based on conversation flow"
    )
    previous_lane: Optional[str] = Field(
        default=None,
        description="Pipeline that handled current query"
    )
    lane_reason: str = Field(
        default="",
        description="Brief reason for lane suggestion"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # Tool Memory v4.2.1: turn-level tool synopsis (NOT knowledge store)
    tool_usage: List[ToolUsageEntry] = Field(
        default_factory=list,
        description="Tool execution synopses for this turn, max 5 entries"
    )

    @field_validator("tool_usage")
    @classmethod
    def enforce_tool_usage_cap(cls, v: List) -> List:
        """Hard cap at 5 entries — defense-in-depth, MCP already caps."""
        return v[:5]


class Topic(BaseModel):
    """
    Represents a conversation topic with status and decay tracking.
    Retained for backward compatibility with v2.0 states.
    """

    topic: str = Field(..., description="Topic name/label")
    status: str = Field(
        default="open",
        description="Topic status: open, closed, abandoned, stale"
    )
    key_info: str = Field(
        default="",
        description="Essential information to preserve about this topic"
    )
    decay_remaining: int = Field(
        default=5,
        description="Turns remaining before topic is removed (decay counter)"
    )
    last_mentioned_turn: int = Field(
        default=0,
        description="Turn number when topic was last mentioned"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When topic was first detected"
    )

    def decrement_decay(self) -> bool:
        """Decrement decay counter. Return True if topic should be removed."""
        self.decay_remaining -= 1
        if self.decay_remaining <= 0:
            self.status = "stale"
            return True
        return False

    def reset_decay(self, decay_turns: int, current_turn: int) -> None:
        """Reset decay counter when topic is mentioned again."""
        self.decay_remaining = decay_turns
        self.last_mentioned_turn = current_turn
        if self.status == "abandoned":
            self.status = "open"


class StructuredContext(BaseModel):
    """Current structured state of the conversation."""

    current_topic: str = Field(default="", description="Current main topic")
    topic_status: str = Field(default="open", description="open or shifting")
    intent: str = Field(default="general", description="Detected user intent")
    entities: Dict[str, Any] = Field(
        default_factory=dict, description="Named entities"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    def is_shifting(self) -> bool:
        return self.topic_status == "shifting"


def _render_tool_line(turn: "ConversationTurn") -> Optional[str]:
    """Render tool usage for a turn, respecting detail_level.

    detail_level provenance: assigned by compression pipeline in
    context_manager.py (_apply_thread_update / _apply_fading).
    Values: full → high → recent → fading → background.
    Newer turns start at "full"; older turns decay over compression cycles.

    Allowed fields ONLY: tool_name, kind, summary, source_refs.
    Returns None if no tool_usage, or detail_level is fading/background.
    """
    if not turn.tool_usage:
        return None
    dl = turn.detail_level
    if dl in ("fading", "background"):
        return None

    if dl in ("full", "high"):
        # Detail: tool_name [kind]: summary
        parts: List[str] = []
        for entry in turn.tool_usage:
            parts.append(
                f"{entry.tool_name} [{entry.output_synopsis.kind}]: "
                f"{entry.output_synopsis.summary}"
            )
        return f"  TOOLS: {', '.join(parts)}"

    # recent or unknown: compressed count per abbreviated name
    counts: Dict[str, int] = {}
    for entry in turn.tool_usage:
        short = (
            entry.tool_name.rsplit("__", 1)[-1]
            if "__" in entry.tool_name
            else entry.tool_name
        )
        counts[short] = counts.get(short, 0) + 1
    parts_c = [f"{name} \u00d7{count}" for name, count in counts.items()]
    return f"  TOOLS: {', '.join(parts_c)}"


def render_thread_context(
    conversation_thread: List["ConversationTurn"],
    current_focus: Optional[str],
    hold_focus: Optional[str],
    topic_progression: str,
    structured_context: Optional["StructuredContext"] = None,
    topic_shifting: bool = False,
) -> str:
    """
    Render conversation thread into formatted context string for LLM.

    Single source of truth for memory rendering, used by both
    MemoryState.get_context_summary_for_llm() and
    ContextResult._thread_system_message().
    """
    lines = ["=== CONVERSATION MEMORY ==="]

    # First pass: identify CURRENT and HOLD entries
    current_entry = None
    hold_entry = None
    other_entries = []

    for turn in conversation_thread:
        if current_focus and turn.focus == current_focus:
            current_entry = turn
        elif hold_focus and turn.focus == hold_focus:
            hold_entry = turn
        else:
            other_entries.append(turn)

    # Safety net: if no current_entry found, use most recent turn
    if current_entry is None and conversation_thread:
        most_recent = max(conversation_thread, key=lambda t: t.turn_number)
        current_entry = most_recent
        other_entries = [t for t in other_entries if t is not most_recent]

    # Render CURRENT
    if current_entry:
        lines.append(f"[CURRENT] {current_entry.focus} — {current_entry.key_facts}")
        if current_entry.query:
            lines.append(f"  LAST QUERY: {current_entry.query}")
        if current_entry.is_resumed and current_entry.anchor_sentence:
            lines.append(f"  ↳ {current_entry.anchor_sentence}")
        tool_line = _render_tool_line(current_entry)
        if tool_line:
            lines.append(tool_line)

    # Render HOLD
    if hold_entry:
        lines.append(f"[HOLD] {hold_entry.focus} — {hold_entry.key_facts}")
        tool_line = _render_tool_line(hold_entry)
        if tool_line:
            lines.append(tool_line)

    # Render others in reverse chronological order
    sorted_others = sorted(other_entries, key=lambda t: t.turn_number, reverse=True)
    for turn in sorted_others:
        dl = turn.detail_level
        if dl in ("full", "high", "recent"):
            tag = "RECENT"
        elif dl == "fading":
            tag = "FADING"
        elif dl == "background":
            tag = "BACKGROUND"
        else:
            tag = "FADING"
        lines.append(f"[{tag}] {turn.focus} — {turn.key_facts}")
        if turn.is_resumed and turn.anchor_sentence:
            lines.append(f"  ↳ {turn.anchor_sentence}")
        tool_line = _render_tool_line(turn)
        if tool_line:
            lines.append(tool_line)

    # Topic flow
    if topic_progression:
        lines.append("")
        lines.append(f"TOPIC FLOW: {topic_progression}")

    # Entities
    if structured_context and structured_context.entities:
        entities = set()
        for k, v in structured_context.entities.items():
            if isinstance(v, list):
                entities.update(str(e) for e in v)
            else:
                entities.add(str(v))
        if entities:
            lines.append(f"ENTITIES: {', '.join(sorted(entities))}")

    # Intent
    if structured_context and structured_context.intent and structured_context.intent != "general":
        lines.append(f"INTENT: {structured_context.intent}")

    # Routing lane suggestion
    if conversation_thread:
        latest = max(conversation_thread, key=lambda t: t.turn_number)
        if getattr(latest, 'suggested_lane', None):
            lines.append(
                f"[ROUTING] Suggested: {latest.suggested_lane}"
                f" | Previous: {getattr(latest, 'previous_lane', 'simple_chat')}"
                f" | {getattr(latest, 'lane_reason', '')}"
            )

    # Topic shifting flag
    if topic_shifting:
        lines.append("[TOPIC SHIFT DETECTED]")

    return "\n".join(lines)


class MemoryState(BaseModel):
    """
    Complete memory state for a conversation session.

    v4.2.0: Thread-based structured summary.
    Contains conversation_thread (ordered list of ConversationTurn),
    current_focus/hold_focus pointers, topic_flow, and archived_turns.
    Backward compatible: if conversation_thread is empty, falls back
    to narrative_summary + previous_topics (v2.0 format).
    """

    # Version for optimistic concurrency control
    version: int = Field(default=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = Field(default=0)

    # Legacy v2.0 fields (backward compat)
    narrative_summary: str = Field(default="")
    structured_context: StructuredContext = Field(default_factory=StructuredContext)
    previous_topics: List[Topic] = Field(default_factory=list)

    # Conversation tracking
    turn_counter: int = Field(default=0)
    compression_history: List[Dict[str, Any]] = Field(default_factory=list)

    # === v4.2.0: Thread-based memory ===
    conversation_thread: List[ConversationTurn] = Field(
        default_factory=list,
        description="Chronologically ordered conversation thread"
    )
    current_focus: Optional[str] = Field(
        default=None, description="Active topic pointer"
    )
    hold_focus: Optional[str] = Field(
        default=None, description="Paused topic for ping-pong"
    )
    hold_since_turn: int = Field(
        default=0, description="Turn when hold became active"
    )
    topic_flow: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Topic progression log"
    )
    topic_progression: str = Field(
        default="", description="Flat string derived for LLM"
    )
    archived_turns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Archived turns for future rehydration"
    )

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "MemoryState":
        return cls.model_validate_json(json_str)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryState":
        return cls.model_validate(data)

    def increment_version(self) -> None:
        self.version += 1
        self.last_updated = datetime.now(timezone.utc)

    def add_compression_event(
        self,
        messages_compressed: int,
        tokens_saved: int,
        trigger: str = "threshold"
    ) -> None:
        self.compression_history.append({
            "turn": self.turn_counter,
            "messages_compressed": messages_compressed,
            "tokens_saved": tokens_saved,
            "trigger": trigger,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        if len(self.compression_history) > 10:
            self.compression_history = self.compression_history[-10:]

    def move_current_to_previous(self, max_previous: int = 3) -> Optional[Topic]:
        if not self.structured_context.current_topic:
            return None
        old_topic = Topic(
            topic=self.structured_context.current_topic,
            status="abandoned",
            key_info="",
            last_mentioned_turn=self.turn_counter
        )
        self.previous_topics.insert(0, old_topic)
        if len(self.previous_topics) > max_previous:
            self.previous_topics = self.previous_topics[:max_previous]
        return old_topic

    def apply_decay(self, decay_turns: int) -> List[Topic]:
        removed = []
        remaining = []
        for topic in self.previous_topics:
            if topic.decrement_decay():
                removed.append(topic)
            else:
                remaining.append(topic)
        self.previous_topics = remaining
        return removed

    def derive_narrative_summary(self) -> str:
        """
        Derive narrative_summary from conversation_thread for backward compat.
        Concatenates key_facts from all active turns.
        """
        if not self.conversation_thread:
            return self.narrative_summary

        parts = []
        for turn in self.conversation_thread:
            if turn.key_facts:
                parts.append(f"{turn.focus}: {turn.key_facts}")
        return " | ".join(parts) if parts else self.narrative_summary

    def get_context_summary_for_llm(self) -> str:
        """
        Generate formatted context summary for injection into LLM prompt.

        v4.2.0: Thread-based structured output with detail levels.
        Falls back to legacy format if conversation_thread is empty.
        v4.3.0: Delegates to render_thread_context() shared function.
        """
        if not self.conversation_thread:
            return self._legacy_context_summary()

        return render_thread_context(
            conversation_thread=self.conversation_thread,
            current_focus=self.current_focus,
            hold_focus=self.hold_focus,
            topic_progression=self.topic_progression,
            structured_context=self.structured_context,
        )

    def _legacy_context_summary(self) -> str:
        """Legacy v2.0 format for backward compat."""
        parts = []
        if self.narrative_summary:
            parts.append(f"CONTEXT SUMMARY: {self.narrative_summary}")
        if self.structured_context.current_topic:
            status = self.structured_context.topic_status
            parts.append(f"CURRENT TOPIC: {self.structured_context.current_topic} ({status})")
        if self.structured_context.intent and self.structured_context.intent != "general":
            parts.append(f"USER INTENT: {self.structured_context.intent}")
        if self.previous_topics:
            prev_topics_str = ", ".join([
                f"{t.topic} ({t.status}, decay:{t.decay_remaining})"
                for t in self.previous_topics
            ])
            parts.append(f"PREVIOUS TOPICS: {prev_topics_str}")
        return " | ".join(parts) if parts else ""


class ContextResult(BaseModel):
    """
    Result of context retrieval for LLM consumption.
    Combines raw messages with structured memory state.
    """

    raw_messages: List[Dict[str, Any]] = Field(default_factory=list)
    narrative_summary: str = Field(default="")
    structured_context: Optional[StructuredContext] = Field(default=None)
    previous_topics: List[Topic] = Field(default_factory=list)
    has_structured_context: bool = Field(default=False)
    topic_shifting: bool = Field(default=False)

    # v4.2.0: Thread-based fields
    conversation_thread: List[ConversationTurn] = Field(default_factory=list)
    current_focus: Optional[str] = Field(default=None)
    hold_focus: Optional[str] = Field(default=None)
    topic_progression: str = Field(default="")

    def get_system_message(self) -> Optional[str]:
        """
        Generate system message for LLM.

        v4.2.0: If conversation_thread is present, produce structured output.
        Otherwise fallback to legacy pipe-delimited format.
        """
        if not self.has_structured_context:
            return None

        # v4.2.0: Thread-based format
        if self.conversation_thread:
            return self._thread_system_message()

        # Legacy format
        parts = []
        if self.narrative_summary:
            parts.append(f"CONTEXT SUMMARY: {self.narrative_summary}")
        if self.structured_context and self.structured_context.current_topic:
            parts.append(f"CURRENT TOPIC: {self.structured_context.current_topic}")
            if self.structured_context.intent != "general":
                parts.append(f"INTENT: {self.structured_context.intent}")
        if self.previous_topics:
            prev = [f"{t.topic}" for t in self.previous_topics[:3]]
            parts.append(f"PREVIOUS TOPICS: {', '.join(prev)}")
        if self.topic_shifting:
            parts.append("[TOPIC SHIFT DETECTED]")
        return " | ".join(parts) if parts else None

    def _thread_system_message(self) -> str:
        """Build system message from conversation_thread.
        v4.3.0: Delegates to render_thread_context() shared function.
        """
        return render_thread_context(
            conversation_thread=self.conversation_thread,
            current_focus=self.current_focus,
            hold_focus=self.hold_focus,
            topic_progression=self.topic_progression,
            structured_context=self.structured_context,
            topic_shifting=self.topic_shifting,
        )
