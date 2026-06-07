"""
Prometheus Metrics for Report Generation Pipeline

v5.0.4: Comprehensive metrics for report lifecycle monitoring.

Metrics exposed:
- report_sessions_total: Counter of sessions by status
- report_planning_seconds: Histogram of planning latency
- report_research_seconds: Histogram of research phase latency
- report_drafting_seconds: Histogram of drafting phase latency
- report_total_seconds: Histogram of end-to-end latency
- report_sections_total: Counter of sections by status
- report_active_sessions: Gauge of currently active sessions
"""

from prometheus_client import Counter, Histogram, Gauge, REGISTRY
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# =============================================================================
# SAFE METRIC CREATION (handles re-import in tests/hot-reload)
# =============================================================================

def _get_or_create_counter(name: str, description: str, labels: list):
    """Get existing counter or create new one."""
    try:
        return Counter(name, description, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


def _get_or_create_histogram(name: str, description: str, labels: list = None, buckets=None):
    """Get existing histogram or create new one."""
    try:
        kwargs = {}
        if buckets:
            kwargs["buckets"] = buckets
        if labels:
            return Histogram(name, description, labels, **kwargs)
        return Histogram(name, description, **kwargs)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


def _get_or_create_gauge(name: str, description: str):
    """Get existing gauge or create new one."""
    try:
        return Gauge(name, description)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


# =============================================================================
# COUNTERS
# =============================================================================

REPORT_SESSIONS_TOTAL = _get_or_create_counter(
    "ubp_report_sessions_total",
    "Total report sessions by status",
    ["status"],  # started, completed, partial, error, cancelled
)

REPORT_SECTIONS_TOTAL = _get_or_create_counter(
    "ubp_report_sections_total",
    "Total report sections by status",
    ["status"],  # success, error, timeout
)

REPORT_PLANNING_TYPE_TOTAL = _get_or_create_counter(
    "ubp_report_planning_type_total",
    "Planning method used",
    ["type"],  # dynamic, static, fallback
)


# =============================================================================
# HISTOGRAMS
# =============================================================================

REPORT_PLANNING_SECONDS = _get_or_create_histogram(
    "ubp_report_planning_seconds",
    "Time to generate report plan",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

REPORT_RESEARCH_SECONDS = _get_or_create_histogram(
    "ubp_report_research_seconds",
    "Time for research phase (all sections)",
    buckets=[5, 10, 20, 30, 60, 120, 180],
)

REPORT_DRAFTING_SECONDS = _get_or_create_histogram(
    "ubp_report_drafting_seconds",
    "Time for drafting phase (all sections)",
    buckets=[5, 10, 20, 40, 60, 120, 300],
)

REPORT_TOTAL_SECONDS = _get_or_create_histogram(
    "ubp_report_total_seconds",
    "End-to-end report generation time",
    buckets=[10, 30, 60, 90, 120, 180, 300, 600],
)

REPORT_SECTION_SECONDS = _get_or_create_histogram(
    "ubp_report_section_seconds",
    "Time per individual section (research + draft)",
    buckets=[2, 5, 10, 20, 30, 60, 120],
)


# =============================================================================
# GAUGES
# =============================================================================

REPORT_ACTIVE_SESSIONS = _get_or_create_gauge(
    "ubp_report_active_sessions",
    "Number of currently active report sessions",
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def record_session_started():
    """Record a new report session started."""
    try:
        REPORT_SESSIONS_TOTAL.labels(status="started").inc()
        REPORT_ACTIVE_SESSIONS.inc()
    except Exception as e:
        logger.warning(f"Failed to record session start metric: {e}")


def record_session_completed(status: str = "completed"):
    """Record a report session completed."""
    try:
        REPORT_SESSIONS_TOTAL.labels(status=status).inc()
        REPORT_ACTIVE_SESSIONS.dec()
    except Exception as e:
        logger.warning(f"Failed to record session complete metric: {e}")


def record_planning(duration_seconds: float, planning_type: str = "dynamic"):
    """Record planning phase metrics."""
    try:
        REPORT_PLANNING_SECONDS.observe(duration_seconds)
        REPORT_PLANNING_TYPE_TOTAL.labels(type=planning_type).inc()
    except Exception as e:
        logger.warning(f"Failed to record planning metric: {e}")


def record_swarm_result(total_time_ms: float, sections_succeeded: int, sections_failed: int):
    """Record swarm execution result metrics."""
    try:
        REPORT_TOTAL_SECONDS.observe(total_time_ms / 1000.0)
        for _ in range(sections_succeeded):
            REPORT_SECTIONS_TOTAL.labels(status="success").inc()
        for _ in range(sections_failed):
            REPORT_SECTIONS_TOTAL.labels(status="error").inc()
    except Exception as e:
        logger.warning(f"Failed to record swarm result metric: {e}")


def record_section_time(duration_seconds: float):
    """Record individual section generation time."""
    try:
        REPORT_SECTION_SECONDS.observe(duration_seconds)
    except Exception as e:
        logger.warning(f"Failed to record section time metric: {e}")
