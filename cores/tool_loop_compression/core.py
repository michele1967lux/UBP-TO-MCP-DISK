"""
tool_loop_compression/core.py — pure tool-loop context compression (DCBL) +
budget-aware retrieval cap.

Extracted from agent_loop.py (Wave C, Compression & Synopsis Cluster, 2026-06-04).
Was AgentLoop._compress_tool_loop_context + _cap_retrieval_result.

C0 decision: NEW utility module (not consolidated into context_compression_engine,
which is a stateful session/memory module). This is stateless, per-turn, hot-path.

Imports nucleus (estimate_message_tokens from tool_analysis) and reasoning synopsis
(extract_reasoning_synopsis from tool_synopsis). `emergency_trim` is passed as a
callback (the only residual AgentLoop dependency — not extracted in this wave).
`prune_old_tool_outputs` imported lazily from mcp_runtime.core.compaction_pruning
(leaf, no cycle).

Dependency surface: inputs + nucleus + synopsis + 1 callback + leaf prune util;
no Redis/Memory/Gov/ACL/Persistence state; sole side effect diagnostic logging.
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any, Callable, Dict, List, Optional

from ubp_enterprise_hybrid.modules.cores.tool_analysis.core import estimate_message_tokens
from ubp_enterprise_hybrid.modules.cores.tool_synopsis.core import extract_reasoning_synopsis
from .providers import SCORE_THRESHOLD, DEFAULT_CONTEXT_WINDOW, DEFAULT_RESPONSE_RESERVE

logger = logging.getLogger(__name__)


def compress_tool_loop_context(
    messages: List[Dict[str, Any]],
    tool_usage_entries: List[Dict[str, Any]],
    compression_level: int,
    emergency_trim: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """DCBL v2: compress tool results in the tool-calling loop.

    Level 1 (selective): replace PREVIOUS-round tool results with synopsis; last
    round raw. Level 2 (full): replace ALL tool results + trim old history (via
    the ``emergency_trim`` callback). Safe fallback: leave a message raw if no
    matching synopsis entry (never empty, never crash).
    """
    tool_msg_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if not tool_msg_indices:
        return messages

    # Phase 5.2 pre-pass: prune verbose old tool outputs (idempotent, gated).
    try:
        from ubp_enterprise_hybrid.mcp_runtime.core.compaction_pruning import prune_old_tool_outputs
        pre_total = sum(estimate_message_tokens(m) for m in messages)
        pruned_msgs, saved = prune_old_tool_outputs(messages, total_estimated_tokens=pre_total)
        if saved > 0:
            logger.info("[DCBL] Phase5.2 prune: saved=%d tokens (pre=%d)", saved, pre_total)
            messages = pruned_msgs
            tool_msg_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    except Exception as prune_err:
        logger.debug("[DCBL] Phase5.2 prune skipped: %s", prune_err)

    if compression_level == 1:
        indices_to_compress = tool_msg_indices[:-1]
    else:
        indices_to_compress = tool_msg_indices

    if len(tool_msg_indices) != len(tool_usage_entries):
        logger.warning(
            "[DCBL] Tool message/synopsis count mismatch: %d tool messages vs "
            "%d synopsis entries — some messages will not be compressed",
            len(tool_msg_indices), len(tool_usage_entries),
        )
    compressed = list(messages)
    for pos, msg_idx in enumerate(tool_msg_indices):
        if msg_idx not in indices_to_compress:
            continue
        if pos >= len(tool_usage_entries):
            continue
        entry = tool_usage_entries[pos]
        reasoning_raw = entry.get("reasoning_raw")
        if not reasoning_raw:
            continue
        original_tokens = entry.get("original_tokens", 0)
        original_chars = original_tokens * 4
        if compression_level == 1:
            max_chars = max(200, int(original_chars * 0.25))
        else:
            max_chars = max(200, int(original_chars * 0.15))
        synopsis_text = extract_reasoning_synopsis(reasoning_raw, max_chars)
        if synopsis_text:
            tool_name = entry.get("tool_name", "unknown")
            compressed[msg_idx] = {
                "role": "tool",
                "tool_call_id": messages[msg_idx].get("tool_call_id", ""),
                "content": f"[SYNOPSIS:{tool_name}] {synopsis_text}",
            }

    if compression_level >= 2 and emergency_trim is not None:
        compressed = emergency_trim(compressed)

    old_tokens = sum(estimate_message_tokens(m) for m in messages)
    new_tokens = sum(estimate_message_tokens(m) for m in compressed)
    logger.info(
        "[DCBL] compress_tool_loop_context: level=%d msgs=%d→%d tokens~%d→%d",
        compression_level, len(messages), len(compressed), old_tokens, new_tokens,
    )
    return compressed


def cap_retrieval_result(
    result: Dict[str, Any],
    budget_max_tokens: Optional[int],
    budget_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Budget-aware cap for retrieval tool results. Keeps score-ordered chunks
    that fit the available token budget; strips chunks below SCORE_THRESHOLD."""
    ctx_window = budget_state.get("context_window", DEFAULT_CONTEXT_WINDOW) if budget_state else DEFAULT_CONTEXT_WINDOW
    committed = (
        budget_state.get("fixed_overhead_tokens", 0) + budget_state.get("history_tokens", 0)
    ) if budget_state else 0
    response_reserve = budget_max_tokens or DEFAULT_RESPONSE_RESERVE
    safety = int(ctx_window * 0.07)
    available = max(1024, ctx_window - committed - response_reserve - safety)

    for key in ("results", "chunks", "documents"):
        items = result.get(key)
        if not isinstance(items, list) or not items:
            continue
        scored = [c for c in items if c.get("score", 1.0) >= SCORE_THRESHOLD]
        scored.sort(key=lambda c: c.get("score", 0), reverse=True)
        selected = []
        tokens_used = 0
        for chunk in scored:
            text = chunk.get("text", chunk.get("content", ""))
            chunk_tokens = estimate_message_tokens({"content": text})
            if tokens_used + chunk_tokens > available:
                break
            selected.append(chunk)
            tokens_used += chunk_tokens
        logger.info(
            "[DCBL] Retrieval cap: %d→%d chunks, %d tokens (available=%d, score>=%.1f)",
            len(items), len(selected), tokens_used, available, SCORE_THRESHOLD,
        )
        result = {**result, key: selected}
        break
    return result
