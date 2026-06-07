"""
Utility functions for RAG Multi-Layer Memory.

Token counting, JSON merge, and snapshot helpers.
"""

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Average characters per token (English text heuristic)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.

    Uses a simple heuristic: ~CHARS_PER_TOKEN characters per token.
    For JSON structures, this provides a reasonable approximation.

    Args:
        text: Input text or JSON string.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_tokens_json(obj: Any) -> int:
    """
    Estimate token count for a JSON-serializable object.

    Args:
        obj: Any JSON-serializable object (dict, list, etc.)

    Returns:
        Estimated token count.
    """
    try:
        return estimate_tokens(json.dumps(obj, ensure_ascii=False))
    except (TypeError, ValueError):
        return 0


def format_snapshots_for_prompt(snapshots: List[Dict[str, Any]]) -> str:
    """
    Format Layer 0 snapshots into a readable string for LLM prompts.

    Args:
        snapshots: List of snapshot dictionaries.

    Returns:
        Formatted string representation.
    """
    if not snapshots:
        return "No snapshots available."

    parts = []
    for i, snap in enumerate(snapshots):
        parts.append(f"--- Snapshot {i + 1} (Turn {snap.get('turn', '?')}) ---")
        parts.append(json.dumps(snap, ensure_ascii=False, indent=2))
    return "\n".join(parts)


def format_layer1_for_prompt(blocks: List[Dict[str, Any]]) -> str:
    """
    Format Layer 1 blocks into a readable string for LLM prompts.

    Args:
        blocks: List of Layer 1 block dictionaries.

    Returns:
        Formatted string representation.
    """
    if not blocks:
        return "No Layer 1 blocks available."

    parts = []
    for i, block in enumerate(blocks):
        parts.append(f"--- Block {i + 1} (Turns {block.get('turn_range', '?')}) ---")
        parts.append(json.dumps(block, ensure_ascii=False, indent=2))
    return "\n".join(parts)


def format_layer2_for_prompt(layer2: Dict[str, Any]) -> str:
    """
    Format Layer 2 memory into a readable string for LLM prompts.

    Args:
        layer2: Layer 2 memory dictionary.

    Returns:
        Formatted string representation.
    """
    if not layer2 or all(not v for v in layer2.values() if isinstance(v, (list, dict))):
        return "No long-term memory available."
    return json.dumps(layer2, ensure_ascii=False, indent=2)


def build_turn_range(snapshots: List[Dict[str, Any]]) -> str:
    """
    Build a turn range string from a list of snapshots.

    Args:
        snapshots: List of snapshot dictionaries with 'turn' field.

    Returns:
        Range string like '18-22'.
    """
    if not snapshots:
        return "0-0"

    turns = [s.get("turn", 0) for s in snapshots]
    return f"{min(turns)}-{max(turns)}"
