"""tool_loop_compression/adapter.py — module interface (BaseHybridModule). Wave C.
Hot path imports core directly; adapter for 3-file conformance (utility, no MCP)."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict

from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule
from .core import compress_tool_loop_context, cap_retrieval_result

logger = logging.getLogger(__name__)


class ToolLoopCompressionAdapter(BaseHybridModule):
    """Framework bridge for tool-loop compression (utility module)."""

    def __init__(self, module_path: Path, **kwargs):
        super().__init__(module_path, **kwargs)
        self._init_status: Dict[str, Any] = {"status": "not_initialized"}

    async def initialize(self) -> None:
        self._init_status = {"status": "healthy"}
        logger.info("[ToolLoopCompression-Adapter] initialized")

    async def shutdown(self) -> None:
        self._init_status = {"status": "shutdown"}

    async def health_check(self, **kwargs) -> Dict[str, Any]:
        return {"module": self.manifest.name, "status": "healthy", "init_status": self._init_status}

    def compress(self, messages, tool_usage_entries, compression_level, emergency_trim=None, **kwargs):
        return compress_tool_loop_context(messages, tool_usage_entries, compression_level, emergency_trim)

    def cap_retrieval(self, result, budget_max_tokens, budget_state, **kwargs):
        return cap_retrieval_result(result, budget_max_tokens, budget_state)
