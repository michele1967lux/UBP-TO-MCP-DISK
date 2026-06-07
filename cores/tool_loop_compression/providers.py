"""tool_loop_compression/providers.py — DCBL/retrieval-cap constants (Wave C)."""
from __future__ import annotations

SCORE_THRESHOLD: float = 0.4
DEFAULT_CONTEXT_WINDOW: int = 40960
DEFAULT_RESPONSE_RESERVE: int = 4096

__all__ = ["SCORE_THRESHOLD", "DEFAULT_CONTEXT_WINDOW", "DEFAULT_RESPONSE_RESERVE"]
