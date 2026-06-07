"""
citation_verification_orchestrator — Orchestrates citation verification,
grounding check, hallucination detection, and trusted source filtering.

Delegates to citations_verifier for verification logic.
LLM calls route through _call_llm → pipeline_orchestrator.execute(simple_chat).

Operations:
- verify_response:         Post-generate full verification pipeline
- filter_trusted_sources:  Pre-generate trust filter (removes low-trust chunks)
- verify_web_sources:      Trust check on web URLs
- update_trust_database:   Auto-update trust scores after verification

Dependencies:
- citations_verifier (required): Core verification + trust list management
- pipeline_orchestrator (optional): For _call_llm delegation via simple_chat
- adaptive_budget_manager (optional): For tightness trigger
- redis (optional): For trust database persistence

v1.0.0 — 2026-02-16
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .adapter import CitationVerificationOrchestratorAdapter

__version__ = "1.0.0"
__all__ = ["create_module", "CitationVerificationOrchestratorAdapter"]


def create_module(
    module_path: Path,
    di_container: Optional[Any] = None,
    event_bus: Optional[Any] = None,
) -> CitationVerificationOrchestratorAdapter:
    """Factory function for UBP Module Loader auto-discovery."""
    return CitationVerificationOrchestratorAdapter(
        module_path=module_path,
        di_container=di_container,
        event_bus=event_bus,
    )
