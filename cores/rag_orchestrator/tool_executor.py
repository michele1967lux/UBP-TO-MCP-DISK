"""Tool execution for Architect tool calling (FEAT-TOOL-001)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

from ubp_enterprise_hybrid.modules.cores._shared.utils import sanitize_for_prompt

logger = logging.getLogger(__name__)


class ArchitectToolExecutor:
    """Execute tool calls for the Architect agent."""

    def __init__(
        self,
        qdrant_module: Any,
        settings: Dict[str, Any],
        collections: List[str],
        web_search_module: Optional[Any] = None,
    ) -> None:
        self.qdrant = qdrant_module
        self.settings = settings
        self.collections = collections
        self.web_search_module = web_search_module

    async def execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        seen_chunk_ids: Set[str],
    ) -> Dict[str, Any]:
        if tool_name == "search_knowledge_base":
            return await self._execute_kb_search(
                query=str(arguments.get("query", "")).strip(),
                reason=str(arguments.get("reason", "")).strip(),
                seen_chunk_ids=seen_chunk_ids,
            )
        if tool_name == "search_web":
            return await self._execute_web_search(
                query=str(arguments.get("query", "")).strip(),
                reason=str(arguments.get("reason", "")).strip(),
            )
        raise ValueError(f"Unknown tool '{tool_name}'")

    async def _execute_kb_search(
        self,
        query: str,
        reason: str,
        seen_chunk_ids: Set[str],
    ) -> Dict[str, Any]:
        if not query:
            raise ValueError("search_knowledge_base requires a non-empty query")
        if not reason:
            logger.warning("[TOOL] search_knowledge_base missing reason")
            reason = "not provided"
        if not self.collections:
            raise ValueError("No collections configured for tool search")

        top_k = int(self.settings.get("top_k", 2))
        similarity_threshold = self.settings.get("similarity_threshold")
        timeout_ms = int(self.settings.get("timeout_ms", 2000))
        effective_top_k = top_k + len(seen_chunk_ids)
        target_collection = self.collections[0]

        start_time = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.qdrant.query_internal(
                    query_text=query,
                    top_k=effective_top_k,
                    collection=target_collection,
                ),
                timeout=timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "[TOOL] search_knowledge_base timed out",
                extra={"query": query, "timeout_ms": timeout_ms, "latency_ms": round(latency_ms, 2)},
            )
            return {
                "tool_name": "search_knowledge_base",
                "query": query,
                "reason": reason,
                "chunks_found": 0,
                "chunk_ids": [],
                "chunks": [],
                "latency_ms": round(latency_ms, 2),
                "content": f'[TOOL_RESULT | search_knowledge_base]\nQuery: "{query}"\nError: Search timed out after {timeout_ms}ms\n[END_TOOL_RESULT]',
                "error": f"timeout ({timeout_ms}ms)",
            }
        latency_ms = (time.perf_counter() - start_time) * 1000

        results_list = result.get("results", []) or []
        if similarity_threshold is not None:
            results_list = [
                item
                for item in results_list
                if float(item.get("score") or 0.0) >= float(similarity_threshold)
            ]
        filtered_results = []
        new_chunk_ids: List[str] = []

        for item in results_list:
            metadata = item.get("metadata", {}) or {}
            chunk_id = metadata.get("chunk_id")
            if chunk_id and chunk_id in seen_chunk_ids:
                continue
            filtered_results.append(item)
            if chunk_id:
                new_chunk_ids.append(chunk_id)
            if len(filtered_results) >= top_k:
                break

        formatted_lines = [
            "[TOOL_RESULT | search_knowledge_base]",
            f"Query: \"{query}\"",
            f"Results: {len(filtered_results)} chunk(s) found",
        ]

        chunks: List[Dict[str, Any]] = []
        for idx, item in enumerate(filtered_results, 1):
            metadata = item.get("metadata", {}) or {}
            source_file = (
                metadata.get("filename") or metadata.get("doc_id") or "unknown"
            )
            raw_text = (item.get("text") or "").strip()
            text = sanitize_for_prompt(raw_text)
            score_val = float(item.get("score") or 0.0)
            formatted_lines.append(
                f"[T{idx} | {source_file}] (score: {score_val:.2f}) {text}"
            )
            chunks.append(
                {
                    "text": item.get("text", ""),
                    "score": item.get("score", 0.0),
                    "metadata": metadata,
                    "collection": item.get("collection") or result.get("collection"),
                }
            )

        formatted_lines.append("[END_TOOL_RESULT]")
        formatted_text = "\n".join(formatted_lines)

        return {
            "tool_name": "search_knowledge_base",
            "query": query,
            "reason": reason,
            "chunks_found": len(filtered_results),
            "chunk_ids": new_chunk_ids,
            "chunks": chunks,
            "latency_ms": round(latency_ms, 2),
            "content": formatted_text,
        }

    async def _execute_web_search(
        self,
        query: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Execute a web search via the web_search module."""
        if not query:
            raise ValueError("search_web requires a non-empty query")
        if not self.web_search_module:
            raise ValueError("Web search module not available")
        if not reason:
            logger.warning("[TOOL] search_web missing reason")
            reason = "not provided"

        timeout_ms = int(self.settings.get("timeout_ms", 3000))
        start_time = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                self.web_search_module.search(query=query, max_results=3),
                timeout=timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "[TOOL] search_web timed out",
                extra={"query": query, "timeout_ms": timeout_ms, "latency_ms": round(latency_ms, 2)},
            )
            return {
                "tool_name": "search_web",
                "query": query,
                "reason": reason,
                "chunks_found": 0,
                "chunk_ids": [],
                "chunks": [],
                "latency_ms": round(latency_ms, 2),
                "content": f'[TOOL_RESULT | search_web]\nQuery: "{query}"\nError: timed out after {timeout_ms}ms\n[END_TOOL_RESULT]',
                "error": f"timeout ({timeout_ms}ms)",
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "[TOOL] search_web failed",
                extra={"query": query, "error": str(e), "latency_ms": round(latency_ms, 2)},
            )
            return {
                "tool_name": "search_web",
                "query": query,
                "reason": reason,
                "chunks_found": 0,
                "chunk_ids": [],
                "chunks": [],
                "latency_ms": round(latency_ms, 2),
                "content": f'[TOOL_RESULT | search_web]\nQuery: "{query}"\nError: {str(e)[:200]}\n[END_TOOL_RESULT]',
                "error": str(e)[:200],
            }

        latency_ms = (time.perf_counter() - start_time) * 1000
        results_list = result.get("results", []) or []

        formatted_lines = [
            "[TOOL_RESULT | search_web]",
            f'Query: "{query}"',
            f"Results: {len(results_list)} result(s)",
        ]
        for idx, item in enumerate(results_list[:5], 1):
            title = item.get("title", "N/A")
            url = item.get("href") or item.get("url", "")
            snippet = item.get("body") or item.get("snippet") or item.get("description", "")
            if len(snippet) > 500:
                snippet = snippet[:500] + "..."
            formatted_lines.append(f"[W{idx}] {title}\n    URL: {url}\n    {snippet}")
        formatted_lines.append("[END_TOOL_RESULT]")

        return {
            "tool_name": "search_web",
            "query": query,
            "reason": reason,
            "chunks_found": len(results_list),
            "chunk_ids": [],
            "chunks": [],
            "latency_ms": round(latency_ms, 2),
            "content": "\n".join(formatted_lines),
        }
