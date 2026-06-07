"""
Conversation Memory Module - Redis-based Session Persistence

Provides multi-turn conversation memory for RAG sessions with Redis backend.

Architecture:
- __init__.py: Entry point (this file)
- adapter.py: UBP framework bridge
- providers.py: Pure technical logic (zero UBP dependencies)

Redis Key Patterns (NAMING_POLICY.md Section 7):
- ubp:memory:session:{session_id}:messages  (List of JSON messages)
- ubp:memory:session:{session_id}:metadata  (Hash with session info)
- ubp:memory:user:{user_id}:sessions        (Sorted Set by last_active)

ROADMAP v1.5.0 - FEAT-MEM-001
"""

from pathlib import Path
from .adapter import ConversationMemoryAdapter

# Module version
__version__ = "1.0.0"

# Public API
__all__ = ["ConversationMemoryAdapter", "create_module"]


def create_module(module_path: Path, **kwargs) -> ConversationMemoryAdapter:
    """
    Factory function to create module instance.

    Args:
        module_path: Path to module directory
        **kwargs: Additional arguments (event_bus, di_container)

    Returns:
        Initialized module instance
    """
    return ConversationMemoryAdapter(module_path, **kwargs)
