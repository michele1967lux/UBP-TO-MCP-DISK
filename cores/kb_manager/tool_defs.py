"""OpenAI-compatible tool definitions for KB management via LLM.

Used by the /api/client/manage endpoint for tool-calling chat.
Each tool maps to a kb_manager adapter operation.

v1.0.0: Initial release (KB-MANAGER)
"""

from typing import Dict, List

KB_ADD_ITEM_TOOL: Dict = {
    "type": "function",
    "function": {
        "name": "kb_add_item",
        "description": (
            "Add a new item to the knowledge base "
            "(menu item, drink, product, etc.)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Item category (e.g. 'antipasto', 'primo', "
                        "'cocktail', 'vino', 'dolce')"
                    ),
                },
                "data": {
                    "type": "object",
                    "description": "Structured item data",
                    "properties": {
                        "name": {"type": "string", "description": "Item name"},
                        "price": {"type": "number", "description": "Price in EUR"},
                        "description": {
                            "type": "string",
                            "description": "Brief description",
                        },
                        "ingredients": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of ingredients",
                        },
                        "allergens": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Allergens present (Reg. UE 1169/2011)"
                            ),
                        },
                    },
                    "required": ["name"],
                },
            },
            "required": ["category", "data"],
        },
    },
}

KB_UPDATE_ITEM_TOOL: Dict = {
    "type": "function",
    "function": {
        "name": "kb_update_item",
        "description": (
            "Update an existing item in the knowledge base. "
            "Only the fields provided will be changed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": (
                        "The item identifier (slug, e.g. 'gin-tonic', "
                        "'bruschetta-pomodoro')"
                    ),
                },
                "data": {
                    "type": "object",
                    "description": "Fields to update",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number"},
                        "description": {"type": "string"},
                        "ingredients": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "allergens": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["item_id", "data"],
        },
    },
}

KB_DELETE_ITEM_TOOL: Dict = {
    "type": "function",
    "function": {
        "name": "kb_delete_item",
        "description": (
            "Remove an item from the knowledge base. "
            "Ask for confirmation before deleting."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": "The item identifier to delete",
                },
            },
            "required": ["item_id"],
        },
    },
}

KB_SEARCH_ITEMS_TOOL: Dict = {
    "type": "function",
    "function": {
        "name": "kb_search_items",
        "description": (
            "Search items in the knowledge base by text. "
            "Returns matching items with relevance scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text (e.g. 'pomodoro', 'cocktail gin')",
                },
                "category": {
                    "type": "string",
                    "description": "Optional: filter by category",
                },
            },
            "required": ["query"],
        },
    },
}

KB_LIST_ITEMS_TOOL: Dict = {
    "type": "function",
    "function": {
        "name": "kb_list_items",
        "description": (
            "List all items in the knowledge base, "
            "optionally filtered by category."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional: filter by category",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max items to return (default 20)",
                    "default": 20,
                },
            },
        },
    },
}

WEB_ENRICH_TOOL: Dict = {
    "type": "function",
    "function": {
        "name": "web_enrich",
        "description": (
            "Search the web for information to enrich an item's description. "
            "Returns web results that can be used to update the item."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query (e.g. 'carbonara ricetta originale ingredienti')"
                    ),
                },
                "item_id": {
                    "type": "string",
                    "description": (
                        "Optional: item to enrich with the search results"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


def get_kb_manager_tools(domain: str = "generic") -> List[Dict]:
    """Return tools appropriate for the domain."""
    return [
        KB_ADD_ITEM_TOOL,
        KB_UPDATE_ITEM_TOOL,
        KB_DELETE_ITEM_TOOL,
        KB_SEARCH_ITEMS_TOOL,
        KB_LIST_ITEMS_TOOL,
        WEB_ENRICH_TOOL,
    ]


TOOL_PROMPT_SECTION = """
================================================================================
STRUMENTI DI GESTIONE KB:
================================================================================
Hai accesso ai seguenti tool per gestire la knowledge base del locale:

1. `kb_add_item` — Aggiungi un nuovo elemento (piatto, drink, prodotto)
2. `kb_update_item` — Modifica un elemento esistente (prezzo, ingredienti, descrizione)
3. `kb_delete_item` — Rimuovi un elemento
4. `kb_search_items` — Cerca elementi per testo
5. `kb_list_items` — Elenca elementi per categoria
6. `web_enrich` — Cerca info sul web per arricchire la scheda di un elemento

REGOLE:
- Usa kb_add_item quando l'utente vuole aggiungere qualcosa al menu/catalogo
- Usa kb_update_item quando vuole modificare prezzo, ingredienti, descrizione
- Usa kb_delete_item quando vuole rimuovere un elemento
- Usa kb_search_items per verificare se un elemento esiste prima di aggiungerlo
- Usa web_enrich quando l'utente chiede di arricchire una scheda con info dal web
- Conferma SEMPRE prima di eliminare: "Vuoi che rimuova [nome]?"
- Dopo ogni operazione, conferma il risultato all'utente
- Estrai SEMPRE gli allergeni se disponibili
- Prezzo: usa numeri (9.00, non "nove euro")

REGOLA CRITICA — item_id:
- Prima di QUALSIASI kb_update_item o kb_delete_item, DEVI chiamare kb_search_items
  per ottenere l'item_id ESATTO dall'elenco risultati
- Usa SEMPRE l'item_id restituito dalla search (es: "bruschetta-al-pomodoro"),
  MAI inventarlo o dedurlo dal nome dell'utente
- L'item_id e' uno slug tecnico (es: "gin-tonic", "bruschetta-al-pomodoro"),
  NON il nome leggibile ("Gin Tonic", "Bruschetta al pomodoro")
"""
