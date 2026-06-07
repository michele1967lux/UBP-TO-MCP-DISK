"""
citations_verifier — Citation verification, trust list management, trusted source filtering.

Three operational modes:

1. VERIFY (post-generation):
   Extract claims from generated text → verify against RAG chunks and/or web sources
   → produce verification report with trust score per claim.

2. FILTER (pre-generation / during search):
   Provide trust lists to web_search module as site filters.
   Modes: prioritize (trusted first, then generic) | exclusive (ONLY trusted sites).
   Called by pipeline before or during search to ensure source quality.

3. MANAGE (lifecycle):
   CRUD on trust lists. Auto-discovery of new trusted sources via web analysis.
   Persistence: Redis primary + JSON backup.

Predefined trust list domains:
- medical, legal_it, legal_eu, technical, financial
- academic, government_it, news_it, news_intl

Dependencies:
- web_search (required): for web verification and source discovery
- adaptive_budget_manager (optional): for LLM budget in claim extraction
- inference_vllm (optional): LLM for claim extraction and semantic comparison
- rag_orchestrator (optional): for RAG chunk access

v1.0.0 — 2026-02-15
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .adapter import CitationsVerifierAdapter

__version__ = "1.0.0"
__all__ = ["create_module", "CitationsVerifierAdapter"]


def create_module(
    module_path: Path,
    di_container: Optional[Any] = None,
    event_bus: Optional[Any] = None,
) -> CitationsVerifierAdapter:
    """Factory function for UBP Module Loader auto-discovery."""
    return CitationsVerifierAdapter(
        module_path=module_path,
        di_container=di_container,
        event_bus=event_bus,
    )
