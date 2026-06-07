"""
Prometheus Metrics for Heuristic Router

FIX-011 v1.8.2: Added comprehensive Prometheus metrics for router monitoring.

Metrics exposed:
- router_queries_total: Counter of queries by route type
- router_confidence_histogram: Distribution of routing confidence
- router_layer_hits_total: Counter of pattern matches by layer
- router_processing_seconds: Histogram of processing time
- router_ambiguity_histogram: Distribution of ambiguity scores
"""

from prometheus_client import Counter, Histogram, Gauge, REGISTRY
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SAFE METRIC CREATION (handles re-import in tests)
# =============================================================================

def _get_or_create_counter(name: str, description: str, labels: list):
    """Get existing counter or create new one."""
    try:
        return Counter(name, description, labels)
    except ValueError:
        # Already registered, get from registry
        return REGISTRY._names_to_collectors.get(name)


def _get_or_create_histogram(name: str, description: str, labels: list = None, buckets=None):
    """Get existing histogram or create new one."""
    try:
        if labels:
            return Histogram(name, description, labels, buckets=buckets)
        return Histogram(name, description, buckets=buckets)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


def _get_or_create_gauge(name: str, description: str, labels: list):
    """Get existing gauge or create new one."""
    try:
        return Gauge(name, description, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


# =============================================================================
# COUNTERS
# =============================================================================

# Total queries by route type
ROUTER_QUERIES_TOTAL = _get_or_create_counter(
    "router_queries_total",
    "Total number of queries processed by route type",
    ["route_type"]  # chat, rag, web
)

# Layer hits by layer
ROUTER_LAYER_HITS_TOTAL = _get_or_create_counter(
    "router_layer_hits_total",
    "Total pattern matches by layer",
    ["layer"]  # A, B, C, D
)

# Explicit source requests
ROUTER_EXPLICIT_SOURCE_TOTAL = _get_or_create_counter(
    "router_explicit_source_total",
    "Queries with explicit source flag",
    ["source_type"]  # web, internal
)


# =============================================================================
# HISTOGRAMS
# =============================================================================

# Routing confidence distribution
ROUTER_CONFIDENCE_HISTOGRAM = _get_or_create_histogram(
    "router_confidence",
    "Distribution of routing confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)

# Processing time distribution
ROUTER_PROCESSING_SECONDS = _get_or_create_histogram(
    "router_processing_seconds",
    "Time spent processing queries",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Ambiguity distribution
ROUTER_AMBIGUITY_HISTOGRAM = _get_or_create_histogram(
    "router_ambiguity",
    "Distribution of query ambiguity scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Capability score distributions
ROUTER_CAPABILITY_HISTOGRAM = _get_or_create_histogram(
    "router_capability_score",
    "Distribution of capability scores by type",
    ["capability"],  # needs_fresh_info, internal_knowledge, etc.
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)


# =============================================================================
# GAUGES
# =============================================================================

# Current pattern counts (for monitoring config size)
ROUTER_PATTERN_COUNT = _get_or_create_gauge(
    "router_pattern_count",
    "Number of loaded patterns by type",
    ["pattern_type"]  # hard, soft, negation, fuzzy
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def record_query_metrics(signals) -> None:
    """
    Record metrics from a PatternSignals result.

    Args:
        signals: PatternSignals object from heuristic engine
    """
    try:
        # Route type
        route = signals.get_suggested_route()
        ROUTER_QUERIES_TOTAL.labels(route_type=route).inc()

        # Confidence
        confidence = signals.get_confidence()
        ROUTER_CONFIDENCE_HISTOGRAM.observe(confidence)

        # Processing time (convert ms to seconds)
        processing_time = signals.processing_time_ms / 1000.0
        ROUTER_PROCESSING_SECONDS.observe(processing_time)

        # Ambiguity
        ROUTER_AMBIGUITY_HISTOGRAM.observe(signals.ambiguity)

        # Capability scores
        caps = signals.capabilities
        ROUTER_CAPABILITY_HISTOGRAM.labels(capability="needs_fresh_info").observe(caps.needs_fresh_info)
        ROUTER_CAPABILITY_HISTOGRAM.labels(capability="internal_knowledge").observe(caps.internal_knowledge)
        ROUTER_CAPABILITY_HISTOGRAM.labels(capability="pure_conversation").observe(caps.pure_conversation)
        ROUTER_CAPABILITY_HISTOGRAM.labels(capability="commercial_info").observe(caps.commercial_info)
        ROUTER_CAPABILITY_HISTOGRAM.labels(capability="technical_depth").observe(caps.technical_depth)

        # Explicit source
        if signals.flags.explicit_source:
            source_type = signals.flags.explicit_source_type or "unknown"
            ROUTER_EXPLICIT_SOURCE_TOTAL.labels(source_type=source_type).inc()

    except Exception as e:
        logger.warning(f"Failed to record router metrics: {e}")


def record_layer_hits(layer_hits: dict) -> None:
    """
    Record layer hit counts.

    Args:
        layer_hits: Dict of layer -> hit count
    """
    try:
        for layer, count in layer_hits.items():
            # Record each hit individually for accurate counting
            for _ in range(count):
                ROUTER_LAYER_HITS_TOTAL.labels(layer=layer).inc()
    except Exception as e:
        logger.warning(f"Failed to record layer metrics: {e}")


def set_pattern_counts(hard: int, soft: int, negation: int, fuzzy: int) -> None:
    """
    Set current pattern counts for monitoring.

    Args:
        hard: Number of hard patterns
        soft: Number of soft patterns
        negation: Number of negation rules
        fuzzy: Number of fuzzy whitelist entries
    """
    try:
        ROUTER_PATTERN_COUNT.labels(pattern_type="hard").set(hard)
        ROUTER_PATTERN_COUNT.labels(pattern_type="soft").set(soft)
        ROUTER_PATTERN_COUNT.labels(pattern_type="negation").set(negation)
        ROUTER_PATTERN_COUNT.labels(pattern_type="fuzzy").set(fuzzy)
    except Exception as e:
        logger.warning(f"Failed to set pattern counts: {e}")
