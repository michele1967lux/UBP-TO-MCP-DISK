"""
Router Package - Intelligent Query Routing System

This package provides multi-layered heuristic signal extraction and
query routing for the RAG Orchestrator.

Components:
- HeuristicEngine: Multi-layer signal extraction (A, B, C, D)
- PatternSignals: Signal output structure
- FuzzyMatcher: Controlled fuzzy matching

ROADMAP v1.7.x - FEAT-ROUTER-002
"""

from .models import (
    SignalScope,
    TraceEntry,
    HardFlags,
    CapabilityScores,
    PatternSignals,
    PatternUnit,
    FuzzyMatch,
    NegationRule,
)

from .heuristic_engine import (
    HeuristicEngine,
    EngineConfig,
    get_heuristic_engine,
)

from .fuzzy import (
    FuzzyMatcher,
    levenshtein_distance,
    get_fuzzy_matcher,
)


__all__ = [
    # Models
    "SignalScope",
    "TraceEntry",
    "HardFlags",
    "CapabilityScores",
    "PatternSignals",
    "PatternUnit",
    "FuzzyMatch",
    "NegationRule",
    # Engine
    "HeuristicEngine",
    "EngineConfig",
    "get_heuristic_engine",
    # Fuzzy
    "FuzzyMatcher",
    "levenshtein_distance",
    "get_fuzzy_matcher",
]
