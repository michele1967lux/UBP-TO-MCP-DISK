"""
Heuristic Engine Models - Data structures for pattern-based signal extraction.

This module defines the core data structures for the multi-layered heuristic
classification system. The system extracts signals (not intents) that downstream
components use for routing decisions.

Design principles:
- Deterministic, explainable, testable
- Signals over intents
- Trace everything for enterprise audit

v2.0 CHANGES:
- Added explicit_source_type to HardFlags for web/internal distinction
- Enhanced get_suggested_route with commercial intent boost
- Added WEB tiebreaker for unknown queries (Smart Fallback)
- Improved confidence calculation for commercial queries

FIX-008 v1.8.2:
- Externalized magic numbers to router_weights.yaml
- Added RoutingWeights singleton for configurable scoring

ROADMAP v1.7.x - FEAT-ROUTER-002 / FEAT-ROUTER-003
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


class RoutingWeights:
    """
    FIX-008 v1.8.2: Externalized routing weights configuration.

    Singleton that loads weights from router_weights.yaml.
    All magic numbers for scoring and confidence are now configurable.
    """

    _instance: Optional["RoutingWeights"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> "RoutingWeights":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load configuration from router_weights.yaml."""
        config_path = Path(__file__).parent / "config" / "router_weights.yaml"
        try:
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"Loaded router weights from {config_path}")
            else:
                logger.warning(f"Router weights file not found: {config_path}, using defaults")
                self._config = {}
        except Exception as e:
            logger.error(f"Failed to load router weights: {e}, using defaults")
            self._config = {}

    def reload(self) -> None:
        """Reload configuration from disk."""
        self._load_config()

    # === WEIGHTS ===
    @property
    def commercial_fresh_boost(self) -> float:
        return self._config.get("weights", {}).get("commercial_fresh_boost", 0.8)

    @property
    def commercial_standalone_boost(self) -> float:
        return self._config.get("weights", {}).get("commercial_standalone_boost", 0.5)

    @property
    def technical_depth_rag_boost(self) -> float:
        return self._config.get("weights", {}).get("technical_depth_rag_boost", 0.3)

    # === THRESHOLDS ===
    @property
    def unknown_query_fallback(self) -> float:
        return self._config.get("thresholds", {}).get("unknown_query_fallback", 0.3)

    @property
    def rag_web_tiebreaker(self) -> float:
        return self._config.get("thresholds", {}).get("rag_web_tiebreaker", 0.2)

    @property
    def high_fresh_threshold(self) -> float:
        return self._config.get("thresholds", {}).get("high_fresh_threshold", 0.8)

    @property
    def high_commercial_with_fresh(self) -> float:
        return self._config.get("thresholds", {}).get("high_commercial_with_fresh", 0.5)

    @property
    def high_commercial_threshold(self) -> float:
        return self._config.get("thresholds", {}).get("high_commercial_threshold", 0.8)

    @property
    def strong_signal_threshold(self) -> float:
        return self._config.get("thresholds", {}).get("strong_signal_threshold", 0.7)

    # === CONFIDENCE VALUES ===
    @property
    def hard_flag_confidence(self) -> float:
        return self._config.get("confidence", {}).get("hard_flag_confidence", 0.95)

    @property
    def shopping_query_confidence(self) -> float:
        return self._config.get("confidence", {}).get("shopping_query_confidence", 0.88)

    @property
    def high_capability_confidence(self) -> float:
        return self._config.get("confidence", {}).get("high_capability_confidence", 0.85)

    @property
    def commercial_only_confidence(self) -> float:
        return self._config.get("confidence", {}).get("commercial_only_confidence", 0.82)

    @property
    def uncertain_confidence(self) -> float:
        return self._config.get("confidence", {}).get("uncertain_confidence", 0.5)

    @property
    def base_start(self) -> float:
        return self._config.get("confidence", {}).get("base_start", 0.5)

    @property
    def gap_weight(self) -> float:
        return self._config.get("confidence", {}).get("gap_weight", 0.3)

    @property
    def max_score_weight(self) -> float:
        return self._config.get("confidence", {}).get("max_score_weight", 0.4)

    @property
    def strong_signal_bonus(self) -> float:
        return self._config.get("confidence", {}).get("strong_signal_bonus", 0.1)

    @property
    def ambiguity_penalty_factor(self) -> float:
        return self._config.get("confidence", {}).get("ambiguity_penalty_factor", 0.3)


# Singleton instance
routing_weights = RoutingWeights()


class SignalScope(str, Enum):
    """Capability scopes for pattern classification."""

    FRESHNESS = "freshness"  # Needs real-time/current info (web)
    INTERNAL = "internal"  # Internal knowledge/documents (rag)
    CONVERSATIONAL = "conversational"  # Pure chat/greeting
    COMMERCIAL = "commercial"  # Pricing, products
    TECHNICAL = "technical"  # Technical questions
    SAFETY = "safety"  # Safety/policy sensitive
    TEMPORAL = "temporal"  # Time-bound queries


@dataclass
class TraceEntry:
    """
    Single trace entry for audit trail.

    Every pattern match or score modification is recorded
    for enterprise-grade explainability.
    """

    pattern: str
    effect: float
    layer: str  # "A", "B", "C", "D"
    scope: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "effect": round(self.effect, 3),
            "layer": self.layer,
            "scope": self.scope,
            "reason": self.reason,
        }


@dataclass
class HardFlags:
    """
    Layer A - Binary flags from hard patterns.

    These are non-negotiable signals that have absolute priority.
    They don't contribute to scoring - they set hard constraints.

    v2.0: Added explicit_source_type to distinguish web vs internal requests.
    """

    temporal_query: bool = False  # Contains explicit date/time reference
    internal_reference: bool = False  # References internal docs/systems
    policy_sensitive: bool = False  # Touches compliance/legal
    safety_triggered: bool = False  # Contains safety keywords
    explicit_source: bool = False  # User explicitly requested source type
    explicit_source_type: str = (
        ""  # v2.0: "web" or "internal" when explicit_source=True
    )
    greeting_detected: bool = False  # Clear greeting pattern
    farewell_detected: bool = False  # Clear goodbye pattern

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temporal_query": self.temporal_query,
            "internal_reference": self.internal_reference,
            "policy_sensitive": self.policy_sensitive,
            "safety_triggered": self.safety_triggered,
            "explicit_source": self.explicit_source,
            "explicit_source_type": self.explicit_source_type,
            "greeting_detected": self.greeting_detected,
            "farewell_detected": self.farewell_detected,
        }

    def has_any_flag(self) -> bool:
        """Check if any hard flag is set."""
        return any(
            [
                self.temporal_query,
                self.internal_reference,
                self.policy_sensitive,
                self.safety_triggered,
                self.explicit_source,
                self.greeting_detected,
                self.farewell_detected,
            ]
        )


@dataclass
class CapabilityScores:
    """
    Layer B output - Weighted capability scores.

    These scores indicate what capabilities the query might need.
    Higher score = stronger signal for that capability.
    All scores are normalized to 0.0 - 1.0 range.
    """

    needs_fresh_info: float = 0.0  # Web search capability
    internal_knowledge: float = 0.0  # RAG/document retrieval
    pure_conversation: float = 0.0  # Chat/small talk
    commercial_info: float = 0.0  # Pricing/products
    technical_depth: float = 0.0  # Technical explanation

    def to_dict(self) -> Dict[str, float]:
        return {
            "needs_fresh_info": round(self.needs_fresh_info, 3),
            "internal_knowledge": round(self.internal_knowledge, 3),
            "pure_conversation": round(self.pure_conversation, 3),
            "commercial_info": round(self.commercial_info, 3),
            "technical_depth": round(self.technical_depth, 3),
        }

    def get_dominant_capability(self) -> tuple[str, float]:
        """Return the capability with highest score."""
        scores = self.to_dict()
        if not scores:
            return ("none", 0.0)
        dominant = max(scores.items(), key=lambda x: x[1])
        return dominant

    def normalize(self, max_value: float = 1.0) -> None:
        """Normalize all scores to max_value ceiling."""
        self.needs_fresh_info = min(self.needs_fresh_info, max_value)
        self.internal_knowledge = min(self.internal_knowledge, max_value)
        self.pure_conversation = min(self.pure_conversation, max_value)
        self.commercial_info = min(self.commercial_info, max_value)
        self.technical_depth = min(self.technical_depth, max_value)


@dataclass
class PatternSignals:
    """
    Complete output of the heuristic engine.

    This is the contract between the heuristic engine and the router.
    It contains all signals extracted from the query, NOT a final decision.

    The router (or downstream component) uses these signals to make
    the final routing decision, potentially combining with other factors.
    """

    # Layer A - Hard flags (binary, absolute priority)
    flags: HardFlags = field(default_factory=HardFlags)

    # Layer B - Capability scores (weighted, 0.0-1.0)
    capabilities: CapabilityScores = field(default_factory=CapabilityScores)

    # Ambiguity indicator (0.0 = very clear, 1.0 = very ambiguous)
    ambiguity: float = 0.5

    # Full trace for audit/debug
    trace: List[TraceEntry] = field(default_factory=list)

    # Negations detected (Layer D effects)
    negations_applied: List[str] = field(default_factory=list)

    # Original query (for reference)
    query: str = ""

    # Processing metadata
    tokens_analyzed: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flags": self.flags.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "ambiguity": round(self.ambiguity, 3),
            "trace": [t.to_dict() for t in self.trace],
            "negations_applied": self.negations_applied,
            "tokens_analyzed": self.tokens_analyzed,
            "processing_time_ms": round(self.processing_time_ms, 3),
        }

    def add_trace(
        self,
        pattern: str,
        effect: float,
        layer: str,
        scope: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Add a trace entry."""
        self.trace.append(
            TraceEntry(
                pattern=pattern,
                effect=effect,
                layer=layer,
                scope=scope,
                reason=reason,
            )
        )

    def get_suggested_route(self) -> str:
        """
        Get suggested route based on signals.

        This is a helper method - the final decision should be made
        by the router considering all factors.

        v2.1 CHANGES (EXPLICIT WEB MODE):
        - WEB routing ONLY with explicit trigger ("cerca online", etc.)
        - Removed automatic WEB tiebreaker for unknown queries
        - Temporal queries alone no longer trigger WEB (need explicit request)
        - Default for unknown/ambiguous queries is RAG (not WEB)

        Returns: "chat", "rag", or "web"
        """
        # =======================================================================
        # PRIORITY 1: Hard flags (absolute priority)
        # =======================================================================

        # Greeting/Farewell → CHAT
        if self.flags.greeting_detected or self.flags.farewell_detected:
            return "chat"

        # v2.1: EXPLICIT WEB MODE - WEB ONLY with explicit source request
        # This is the ONLY way to trigger WEB routing
        if self.flags.explicit_source:
            if self.flags.explicit_source_type == "web":
                return "web"
            elif self.flags.explicit_source_type == "internal":
                return "rag"
            # Fallback for explicit_source without type → RAG (conservative)
            return "rag"

        # Internal reference or policy sensitive → RAG
        if self.flags.internal_reference or self.flags.policy_sensitive:
            return "rag"

        # v2.1: Temporal query alone does NOT trigger WEB anymore
        # User must explicitly request web search with "cerca online" etc.
        # Temporal queries go to RAG by default (might be in internal docs)
        if self.flags.temporal_query:
            return "rag"

        # =======================================================================
        # PRIORITY 2: Capability scores (NO automatic WEB routing)
        # =======================================================================
        cap = self.capabilities

        # v2.1: WEB score only matters if explicit_source_type == "web"
        # Since we already checked that above, we don't route to WEB here

        # RAG score: internal knowledge + technical depth
        rag_score = cap.internal_knowledge
        if cap.technical_depth > 0:
            rag_score += cap.technical_depth * routing_weights.technical_depth_rag_boost

        # CHAT score: pure conversation
        chat_score = cap.pure_conversation

        # v2.1: Simplified scoring - no automatic WEB
        scores = {
            "chat": chat_score,
            "rag": rag_score,
        }

        # Find the winning route between CHAT and RAG only
        max_route, max_score = max(scores.items(), key=lambda x: x[1])

        # =======================================================================
        # PRIORITY 3: Default for unknown/low-signal queries → RAG
        # =======================================================================

        # v2.1: Unknown queries default to RAG (not WEB)
        # Rationale: Without explicit "cerca online", assume user wants
        # to search internal knowledge base first
        if max_score < routing_weights.unknown_query_fallback:
            return "rag"

        return max_route

    def get_confidence(self) -> float:
        """
        Calculate confidence in the suggested route.

        Based on:
        - Presence of hard flags (high confidence)
        - Dominance of one capability over others
        - Absolute score of dominant capability
        - Low ambiguity score

        v2.0: Enhanced for commercial/shopping queries with strong signals.
        FIX-008 v1.8.2: All thresholds now configurable via router_weights.yaml
        """
        # Hard flags = high confidence
        if self.flags.has_any_flag():
            return routing_weights.hard_flag_confidence

        # Calculate from capability spread
        cap_dict = self.capabilities.to_dict()
        values = list(cap_dict.values())

        if not values or max(values) == 0:
            return routing_weights.uncertain_confidence  # No signals = uncertain

        max_score = max(values)
        second_max = sorted(values, reverse=True)[1] if len(values) > 1 else 0

        cap = self.capabilities

        # v2.0: Strong commercial + fresh signals = high confidence for WEB
        # FIX-008: Using externalized thresholds
        if (cap.needs_fresh_info >= routing_weights.high_fresh_threshold and
                cap.commercial_info >= routing_weights.high_commercial_with_fresh):
            return routing_weights.shopping_query_confidence  # Shopping/price queries

        if cap.needs_fresh_info >= 1.0:
            return routing_weights.high_capability_confidence

        # v2.0: Strong commercial signal alone
        if cap.commercial_info >= routing_weights.high_commercial_threshold:
            return routing_weights.commercial_only_confidence

        # Direct RAG indicators
        if cap.internal_knowledge >= 1.0:
            return routing_weights.high_capability_confidence

        # Direct CHAT indicators
        if cap.pure_conversation >= 1.0:
            return routing_weights.high_capability_confidence

        # Confidence based on gap between top two scores
        gap = max_score - second_max

        # Improved formula: weight absolute score more heavily
        # FIX-008: Using externalized formula weights
        base_confidence = (routing_weights.base_start +
                          (gap * routing_weights.gap_weight) +
                          (max_score * routing_weights.max_score_weight))

        # Bonus for strong signals in primary capabilities
        if max_score >= routing_weights.strong_signal_threshold:
            base_confidence += routing_weights.strong_signal_bonus

        # Reduce by ambiguity
        confidence = base_confidence * (1 - self.ambiguity * routing_weights.ambiguity_penalty_factor)

        return min(max(confidence, 0.0), 1.0)


@dataclass
class PatternUnit:
    """
    Layer B - Single pattern unit for keyword scoring.

    This is the building block of the scoring system.
    Loaded from patterns.yaml configuration.
    """

    tokens: List[str]  # Keywords that trigger this pattern
    weight: float = 1.0  # Base weight contribution
    scope: List[str] = field(default_factory=list)  # Which capabilities affected
    negations: List[str] = field(default_factory=list)  # Words that negate this pattern
    boosts: List[str] = field(default_factory=list)  # Words that boost this pattern
    boost_multiplier: float = 1.3  # How much boosts multiply

    # Pattern metadata
    id: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tokens": self.tokens,
            "weight": self.weight,
            "scope": self.scope,
            "negations": self.negations,
            "boosts": self.boosts,
            "boost_multiplier": self.boost_multiplier,
            "description": self.description,
        }


@dataclass
class FuzzyMatch:
    """
    Layer C - Fuzzy match result.

    Used only for whitelisted short keywords with controlled
    edit distance tolerance.
    """

    original: str  # Original token in query
    matched: str  # Pattern token it matched
    distance: int  # Levenshtein distance
    confidence: float  # Match confidence (1.0 - distance/len)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "matched": self.matched,
            "distance": self.distance,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class NegationRule:
    """
    Layer D - Negation rule definition.

    Negations can:
    - Subtract from capability scores
    - Zero out specific capabilities
    - Set explicit constraints
    """

    triggers: List[str]  # Phrases that trigger this negation
    affects: List[str]  # Capabilities affected
    effect: str = "subtract"  # "subtract", "zero", "constraint"
    value: float = 0.5  # How much to subtract (if subtract)
    constraint: Optional[str] = None  # Constraint type if effect="constraint"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggers": self.triggers,
            "affects": self.affects,
            "effect": self.effect,
            "value": self.value,
            "constraint": self.constraint,
        }


# Type aliases for clarity
PatternUnits = List[PatternUnit]
NegationRules = List[NegationRule]
FuzzyWhitelist = Dict[str, List[str]]  # category -> allowed fuzzy words


__all__ = [
    "SignalScope",
    "TraceEntry",
    "HardFlags",
    "CapabilityScores",
    "PatternSignals",
    "PatternUnit",
    "FuzzyMatch",
    "NegationRule",
    "PatternUnits",
    "NegationRules",
    "FuzzyWhitelist",
]
