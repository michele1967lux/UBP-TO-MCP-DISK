"""Tool schema definitions for Architect tool calling (FEAT-TOOL-001)."""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Generic tool-instruction block injected into system prompts that do NOT
# already contain tool instructions (e.g. RAG chat).  Architect has its own
# richer section in prompts/architect.py, so injection is skipped there.
# ---------------------------------------------------------------------------

# Template when BOTH KB search and web search are available
TOOL_SECTION_TEMPLATE = """
================================================================================
STRUMENTI DISPONIBILI:
================================================================================
Hai accesso ai seguenti tool:

1. `search_knowledge_base` - Cerca informazioni nella knowledge base del progetto
2. `search_web` - Cerca informazioni aggiornate sul web

QUANDO USARE search_knowledge_base:
- Quando serve un dettaglio tecnico preciso non presente nei chunk
- Quando la domanda tocca piu' argomenti e il contesto ne copre solo alcuni

QUANDO USARE search_web:
- Quando servono informazioni aggiornate non presenti nella knowledge base
- Per documentazione esterna, eventi recenti, o argomenti non coperti dalla KB

QUANDO NON USARE I TOOL:
- Quando il contesto fornito e' sufficiente per rispondere
- Per domande generiche o concettuali
- Per informazioni deducibili dal contesto

REGOLE:
- Usa query precise e specifiche
- Massimo {max_iterations} ricerche totali per risposta
- Se i tool non trovano risultati, rispondi con le info disponibili
"""

# Template when only KB search is available (no web search module)
TOOL_SECTION_KB_ONLY_TEMPLATE = """
================================================================================
STRUMENTI DISPONIBILI:
================================================================================
Hai accesso al tool `search_knowledge_base` per cercare informazioni
specifiche nella knowledge base.

QUANDO USARE IL TOOL:
- Quando il contesto fornito NON copre un aspetto specifico della domanda
- Quando serve un dettaglio tecnico preciso non presente nei chunk
- Quando la domanda tocca piu' argomenti e il contesto ne copre solo alcuni

QUANDO NON USARE IL TOOL:
- Quando il contesto fornito e' sufficiente per rispondere
- Per domande generiche o concettuali
- Per informazioni deducibili dal contesto

REGOLE:
- Usa query precise e specifiche
- Massimo {max_iterations} ricerche per risposta
- Se il tool non trova risultati, rispondi con le info disponibili
"""


def build_tool_prompt_section(max_iterations: int, has_web_search: bool = False) -> str:
    """Return the tool-instruction block with *max_iterations* filled in."""
    template = TOOL_SECTION_TEMPLATE if has_web_search else TOOL_SECTION_KB_ONLY_TEMPLATE
    return template.format(max_iterations=max_iterations)

SEARCH_KB_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the project knowledge base for specific technical information. "
            "Use when the provided context does not cover a specific aspect of the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Specific search query. Be precise: "
                        "'ProviderMapper fallback chain configuration' not 'provider config'."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reason why this search is needed.",
                },
            },
            "required": ["query", "reason"],
        },
    },
}

WEB_SEARCH_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web for current information not available in the knowledge base. "
            "Use for recent events, external documentation, or topics not covered by KB."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Web search query. Be specific and concise.",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reason why web search is needed.",
                },
            },
            "required": ["query", "reason"],
        },
    },
}


def get_tool_definitions(settings: Any) -> List[Dict[str, Any]]:
    """Return the list of active tool schemas."""
    if not settings:
        return []
    if isinstance(settings, dict):
        enabled = bool(settings.get("enabled", False))
    else:
        enabled = bool(getattr(settings, "enabled", False))
    if not enabled:
        return []
    tools = [SEARCH_KB_TOOL_SCHEMA]
    if isinstance(settings, dict) and settings.get("web_search_available"):
        tools.append(WEB_SEARCH_TOOL_SCHEMA)
    return tools
