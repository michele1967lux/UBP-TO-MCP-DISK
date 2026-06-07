"""
Adaptive Budget Manager Module

**Modulo universale** per la gestione adattiva del budget di token in contesti di chat.

Classe principale: **AdaptiveBudgetManager**

Applicabile a:
- Pipeline RAG (caso principale)
- Chat Pure LLM (senza retrieval)
- Conversazioni multi-turn generiche
- Qualsiasi scenario dove serve gestione intelligente del context window

Features:
- Dynamic memory allocation (20-40% of context window)
- Tightness factor calculation based on usage and turns
- Similarity threshold scaling (0.4-0.7) per RAG
- Context compression/summarization when needed
- Automatic adaptation to different model sizes

v1.0.0 - Initial implementation
v3.7.0 - Context Governor with ExecutionPlan
"""

from .adapter import AdaptiveBudgetManagerAdapter, create_adapter
from .providers import AdaptiveBudgetManager
from .models import (
    AdaptiveMemoryConfig,
    BudgetAdjustmentResult,
    TightnessResult,
    SummarizationResult,
    ExecutionPlan,  # v3.7.0: Context Governor
    ContextStrategy,  # v3.7.0: Context Governor
    TaskProfile,  # v3.7.0: Context Governor
)

__version__ = "3.7.0"
__all__ = [
    "AdaptiveBudgetManagerAdapter",
    "create_adapter",
    "AdaptiveBudgetManager",
    "AdaptiveMemoryConfig",
    "BudgetAdjustmentResult",
    "TightnessResult",
    "SummarizationResult",
    "ExecutionPlan",
    "ContextStrategy",
    "TaskProfile",
]
