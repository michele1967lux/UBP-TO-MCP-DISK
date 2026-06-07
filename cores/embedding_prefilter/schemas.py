"""
embedding_prefilter/schemas.py — Dataclass e tipi per il modulo Embedding Prefilter.

Definisce:
- PreRouteDecision: output principale del prefilter
- StabilityInfo: R3 per-user routing stability (preparato, non attivo Phase 1)
- RouteScoreBreakdown: dettaglio per-route scoring
- RoutingProfile: R3 per-user Redis profile (preparato, non attivo Phase 1)
- Costanti: ROUTE_SEVERITY, SEVERITY_BASE_PENALTY
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# Route Severity Mapping (R7)
# ============================================================================
# Misrouting severity: higher severity = stricter threshold required
# Used with base_threshold + severity_penalty formula (Correzione #4)

ROUTE_SEVERITY: Dict[str, str] = {
    "REPORT": "high",
    "WEB": "medium-high",
    "RAG": "low",
    "FAST": "low",
}

# Severity penalties added to base_threshold when R7 is active
# Effective threshold = base_threshold + penalty
# e.g., REPORT: 0.55 + 0.12 = 0.67 (NOT 0.82)
SEVERITY_BASE_PENALTY: Dict[str, float] = {
    "REPORT": 0.12,
    "WEB": 0.08,
    "RAG": 0.0,
    "FAST": 0.0,
}

# Valid routes from the prefilter
VALID_ROUTES = {"FAST", "RAG", "WEB", "REPORT", "LLM_ROUTER", "DYNAMIC_INTERACTION"}

# Mapping from embedding cluster name to route.
CLUSTER_TO_ROUTE: Dict[str, str] = {
    "chat": "FAST",
    "rag": "RAG",
    "web_search": "WEB",
    "report": "REPORT",
}

# Reverse: route to intent (for classification_result compatibility)
ROUTE_TO_INTENT: Dict[str, str] = {
    "FAST": "chat",
    "RAG": "rag",
    "WEB": "web_search",
    "REPORT": "report",
    "LLM_ROUTER": "rag",  # safe default when deferring
    "DYNAMIC_INTERACTION": "rag",  # safe default; actual route chosen by user
}


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class StabilityInfo:
    """R3: Route Stability Guard data (structure present, not used in Phase 1).

    Tracks per-user route flapping to prevent oscillation.
    """
    flap_count: int = 0
    last_routes: List[str] = field(default_factory=list)
    stabilized_route: Optional[str] = None


@dataclass
class InteractionOption:
    """Single option in a DYNAMIC_INTERACTION response."""
    route: str           # RAG, WEB, REPORT, FAST
    label: str           # "Cerca nella Knowledge Base"
    icon: str            # "📚"
    confidence: float    # prefilter score for this route


@dataclass
class InteractionOptions:
    """R2: Options for user route choice when prefilter is uncertain."""
    query: str
    options: List[InteractionOption]
    decision_id: str
    fallback_route: str  # top route if user doesn't choose
    expires_in_seconds: int = 300


@dataclass
class PreRouteDecision:
    """Output principale del prefilter.

    Contiene la decisione di routing con tutti i metadati necessari
    per correlazione log, debugging, e override policy.
    """
    decision_id: str                          # uuid4[:8] per correlazione log cross-stage
    route: str                                # FAST|RAG|WEB|REPORT|LLM_ROUTER|DYNAMIC_INTERACTION
    confidence: float                         # calibrated confidence (= raw in Phase 1, R6 disabled)
    raw_confidence: float                     # pre-calibration (sempre populated)
    reasoning: str                            # human-readable reasoning
    layer_trace: List[str] = field(default_factory=list)  # ["L1:embedding", "L2:semantic_scoring", ...]
    scores: Dict[str, float] = field(default_factory=dict)  # softmax-normalized per-route scores
    raw_scores: Dict[str, float] = field(default_factory=dict)  # pre-softmax cosine similarities
    evidence: Dict[str, Any] = field(default_factory=dict)  # enrichment from Layer 3 (empty in Phase 1)
    stability_info: Optional[StabilityInfo] = None  # R3 data (None in Phase 1)
    freshness_signal: float = 0.0             # L2.5: 0.0-1.0, higher = query needs fresh data
    severity_level: str = "low"               # R7: "high"|"medium-high"|"medium"|"low" (informative only in Phase 1)
    time_ms: float = 0.0
    deferred_to_llm_router: bool = False
    interaction_options: Optional[InteractionOptions] = None  # R2: populated when route=DYNAMIC_INTERACTION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for event bus and API responses."""
        return {
            "decision_id": self.decision_id,
            "route": self.route,
            "confidence": self.confidence,
            "raw_confidence": self.raw_confidence,
            "reasoning": self.reasoning,
            "layer_trace": self.layer_trace,
            "scores": self.scores,
            "raw_scores": self.raw_scores,
            "evidence": self.evidence,
            "stability_info": {
                "flap_count": self.stability_info.flap_count,
                "last_routes": self.stability_info.last_routes,
                "stabilized_route": self.stability_info.stabilized_route,
            } if self.stability_info else None,
            "freshness_signal": self.freshness_signal,
            "severity_level": self.severity_level,
            "time_ms": self.time_ms,
            "deferred_to_llm_router": self.deferred_to_llm_router,
            "interaction_options": {
                "query": self.interaction_options.query,
                "options": [
                    {"route": o.route, "label": o.label, "icon": o.icon, "confidence": o.confidence}
                    for o in self.interaction_options.options
                ],
                "decision_id": self.interaction_options.decision_id,
                "fallback_route": self.interaction_options.fallback_route,
                "expires_in_seconds": self.interaction_options.expires_in_seconds,
            } if self.interaction_options else None,
        }


@dataclass
class RouteScoreBreakdown:
    """Dettaglio scoring per singola route.

    Usato per debug e logging dei punteggi per-route.
    """
    route: str
    cosine_sim: float                         # raw cosine similarity
    softmax_prob: float                       # softmax-normalized probability
    severity_penalty: float = 0.0             # R7 penalty (0.0 in Phase 1)
    calibrated_prob: float = 0.0              # R6 calibrated (= softmax_prob in Phase 1)
    masked: bool = False                      # R6: below threshold (always False in Phase 1)


@dataclass
class RoutingProfile:
    """R3: Per-user Redis routing profile (structure present, not used in Phase 1).

    Tracks user routing history for stability guard and cold-start detection.
    """
    user_id: str
    recent_routes: List[str] = field(default_factory=list)  # last N routes
    route_counts: Dict[str, int] = field(default_factory=dict)
    total_queries: int = 0                    # lifetime total (per R4 cold-start)
    last_updated: float = 0.0
    dynamic_interaction_count: int = 0        # R2: session counter
