"""
agentic_rag - Autonomous RAG Engine with Parallel Execution

Core module for UBP Enterprise Hybrid.

Implements autonomous retrieval-augmented generation with:

1. Tool-based Retrieval:
   - Dynamic tool selection
   - Multi-source retrieval
   - Tool chaining
   - Result aggregation

2. Planning & Reasoning:
   - Query decomposition
   - Sub-task planning
   - Execution strategy
   - Adaptive replanning

3. Parallel Execution:
   - Concurrent tool calls
   - Async task orchestration
   - Worker pool management
   - Dependency-aware scheduling

4. Multi-step Workflows:
   - Sequential execution
   - Parallel branches
   - Conditional routing
   - Loop handling

5. State Management:
   - Execution context
   - Intermediate results
   - Memory across steps
   - Checkpoint/resume

6. Observation & Control:
   - Step-by-step tracing
   - Timeout handling
   - Error recovery
   - Human-in-the-loop

Features:
- ReAct-style reasoning loops
- Parallel tool execution
- Configurable execution strategies
- Tool registry with schemas
- Comprehensive observability
- Cross-lingual support (EN/IT)

v1.0.0: Initial release with full enterprise features

Architecture:
- adapter.py: Bridge layer exposing all operations
- providers.py: Core data classes and state management
- planner.py: Query decomposition and planning
- executor.py: Parallel execution engine
- tools.py: Tool registry and execution
- prompts.py: Agent prompt templates
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .adapter import AgenticRAGAdapter

__version__ = "1.0.0"
__all__ = ["create_module", "AgenticRAGAdapter"]


def create_module(
    module_path: Path,
    di_container: Optional[Any] = None,
    event_bus: Optional[Any] = None,
) -> AgenticRAGAdapter:
    """
    Factory function for creating the agentic_rag adapter.
    
    This is the entry point used by ModuleLoader.
    
    Args:
        module_path: Path to the module directory
        di_container: DI container for dependency resolution
        event_bus: Event bus for publishing events
    
    Returns:
        Configured AgenticRAGAdapter instance
    """
    return AgenticRAGAdapter(
        module_path=module_path,
        di_container=di_container,
        event_bus=event_bus,
    )
