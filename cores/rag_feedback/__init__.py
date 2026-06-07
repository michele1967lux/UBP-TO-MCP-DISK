"""
RAG Feedback Module - User Feedback Collection and Analytics

Provides feedback collection and analytics for RAG responses.

Architecture:
- __init__.py: Entry point (this file)
- adapter.py: UBP framework bridge
- providers.py: Pure technical logic (zero UBP dependencies)

Redis Key Patterns (NAMING_POLICY.md Section 7):
- ubp:feedback:response:{response_id}      (Hash - feedback data)
- ubp:feedback:stats:daily:{YYYY-MM-DD}    (Hash - daily aggregates)
- ubp:feedback:stats:collection:{name}     (Hash - per-collection stats)
- ubp:feedback:user:{user_id}:list         (List - user's feedback IDs)
- ubp:feedback:all:list                    (List - all feedback IDs, for admin)

ROADMAP v1.5.0 - FEAT-EVAL-001
"""

from pathlib import Path
from .adapter import FeedbackAdapter

# Module version
__version__ = "1.0.0"

# Public API
__all__ = ["FeedbackAdapter", "create_module"]


def create_module(module_path: Path, **kwargs) -> FeedbackAdapter:
    """
    Factory function to create module instance.

    Args:
        module_path: Path to module directory
        **kwargs: Additional arguments (event_bus, di_container)

    Returns:
        Initialized FeedbackAdapter instance
    """
    return FeedbackAdapter(module_path, **kwargs)
