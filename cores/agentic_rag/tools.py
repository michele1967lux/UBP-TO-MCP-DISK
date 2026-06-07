"""
agentic_rag/tools.py

Tool registry and execution for agentic RAG.

Provides:
- ToolSchema: Tool definition schema
- ToolRegistry: Registry of available tools
- ToolExecutor: Tool execution with retry/timeout
- Built-in tools: retrieval, calculator, summarizer, etc.

v1.0.0: Initial release
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from .providers import ToolCall, ToolResult, TaskStatus

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class IModuleRegistry(Protocol):
    """Protocol for module registry."""
    def get_module(self, module_name: str) -> Optional[Any]: ...


# ============================================================================
# Tool Schema
# ============================================================================


@dataclass
class ToolParameter:
    """Single tool parameter."""
    name: str
    param_type: str  # string, integer, float, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class ToolSchema:
    """Tool definition schema."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    returns: str = "any"
    category: str = "general"
    requires_confirmation: bool = False
    timeout_seconds: int = 30
    max_retries: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.param_type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in self.parameters
            ],
            "returns": self.returns,
            "category": self.category,
        }
    
    def to_function_spec(self) -> Dict[str, Any]:
        """Convert to OpenAI-style function specification."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.param_type,
                "description": param.description,
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


# ============================================================================
# Base Tool
# ============================================================================


class BaseTool(ABC):
    """Base class for tools."""
    
    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Return tool schema."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool."""
        pass
    
    async def validate_args(self, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate arguments against schema."""
        for param in self.schema.parameters:
            if param.required and param.name not in args:
                return False, f"Missing required parameter: {param.name}"
        return True, None


# ============================================================================
# Built-in Tools
# ============================================================================


class RetrievalTool(BaseTool):
    """Tool for document retrieval."""
    
    def __init__(self, module_registry: IModuleRegistry, config: Dict[str, Any]):
        self._registry = module_registry
        self._config = config
        self._module_name = config.get("module", "retrieval_strategy")
        self._default_strategy = config.get("default_strategy", "hybrid")
        self._default_top_k = config.get("default_top_k", 5)
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="retrieval",
            description="Search and retrieve relevant documents from the knowledge base",
            parameters=[
                ToolParameter("query", "string", "Search query", required=True),
                ToolParameter("top_k", "integer", "Number of results to return", required=False, default=5),
                ToolParameter("strategy", "string", "Retrieval strategy", required=False, default="hybrid",
                             enum=["hybrid", "bm25", "vector", "hierarchical"]),
                ToolParameter("filters", "object", "Metadata filters", required=False),
            ],
            returns="array of documents",
            category="retrieval",
            timeout_seconds=15,
        )
    
    async def execute(
        self,
        query: str,
        top_k: int = None,
        strategy: str = None,
        filters: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """Execute retrieval."""
        top_k = top_k or self._default_top_k
        strategy = strategy or self._default_strategy
        
        module = self._registry.get_module(self._module_name)
        if not module:
            return [{"error": f"Module {self._module_name} not available"}]
        
        try:
            result = await module.retrieve(
                query=query,
                strategy=strategy,
                top_k=top_k,
                filters=filters,
            )
            
            if isinstance(result, dict):
                return result.get("results", [])
            return []
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return [{"error": str(e)}]


class CalculatorTool(BaseTool):
    """Safe calculator tool."""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._safe_eval = config.get("safe_eval", True)
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="calculator",
            description="Perform mathematical calculations. Supports basic arithmetic, powers, roots, and common functions.",
            parameters=[
                ToolParameter("expression", "string", "Mathematical expression to evaluate", required=True),
            ],
            returns="number",
            category="computation",
            timeout_seconds=5,
        )
    
    async def execute(self, expression: str) -> Union[float, str]:
        """Evaluate mathematical expression safely."""
        try:
            # Clean expression
            expr = expression.strip()
            
            # Allowed operations
            allowed_names = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "log10": math.log10,
                "exp": math.exp,
                "pi": math.pi,
                "e": math.e,
            }
            
            # Only allow safe characters
            if self._safe_eval:
                safe_pattern = r'^[\d\s\+\-\*\/\.\(\)\,\%\^]+$|^[\w\d\s\+\-\*\/\.\(\)\,\%\^]+$'
                if not re.match(safe_pattern, expr):
                    return f"Error: Expression contains unsafe characters"
            
            # Replace ^ with **
            expr = expr.replace("^", "**")
            
            # Evaluate
            result = eval(expr, {"__builtins__": {}}, allowed_names)
            
            if isinstance(result, float):
                return round(result, 10)
            return result
            
        except Exception as e:
            return f"Error: {str(e)}"


class SummarizerTool(BaseTool):
    """Tool for text summarization."""
    
    def __init__(self, module_registry: IModuleRegistry, config: Dict[str, Any]):
        self._registry = module_registry
        self._config = config
        self._llm_module = config.get("llm_module", "inference_ollama_grok")
        self._max_input = config.get("max_input_tokens", 4000)
        self._target_length = config.get("target_length", 200)
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="summarizer",
            description="Summarize long text into a concise summary",
            parameters=[
                ToolParameter("text", "string", "Text to summarize", required=True),
                ToolParameter("max_length", "integer", "Target summary length in words", required=False, default=200),
                ToolParameter("style", "string", "Summary style", required=False, default="concise",
                             enum=["concise", "detailed", "bullet_points"]),
            ],
            returns="string",
            category="text_processing",
            timeout_seconds=30,
        )
    
    async def execute(
        self,
        text: str,
        max_length: int = None,
        style: str = "concise",
    ) -> str:
        """Summarize text."""
        max_length = max_length or self._target_length
        
        # Truncate if too long
        if len(text) > self._max_input * 4:  # Rough char estimate
            text = text[:self._max_input * 4]
        
        module = self._registry.get_module(self._llm_module)
        if not module:
            # Fallback: simple extractive summary
            sentences = text.split(". ")
            return ". ".join(sentences[:3]) + "."
        
        try:
            prompt = f"""Summarize the following text in approximately {max_length} words.
Style: {style}

Text:
{text}

Summary:"""
            
            result = await module.generate(prompt=prompt, max_tokens=max_length * 2)
            
            if isinstance(result, dict):
                return result.get("text", result.get("response", ""))
            return str(result)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return f"Error: {str(e)}"


class GraphQueryTool(BaseTool):
    """Tool for knowledge graph queries."""
    
    def __init__(self, module_registry: IModuleRegistry, config: Dict[str, Any]):
        self._registry = module_registry
        self._config = config
        self._module_name = config.get("module", "graph_rag")
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="graph_query",
            description="Query the knowledge graph for entity relationships and paths",
            parameters=[
                ToolParameter("query", "string", "Natural language query about entities/relations", required=True),
                ToolParameter("entity_names", "array", "Specific entities to search for", required=False),
                ToolParameter("max_hops", "integer", "Maximum relationship hops", required=False, default=2),
            ],
            returns="object with entities and relations",
            category="knowledge_graph",
            timeout_seconds=20,
        )
    
    async def execute(
        self,
        query: str,
        entity_names: List[str] = None,
        max_hops: int = 2,
    ) -> Dict[str, Any]:
        """Query knowledge graph."""
        module = self._registry.get_module(self._module_name)
        if not module:
            return {"error": f"Module {self._module_name} not available"}
        
        try:
            if entity_names:
                result = await module.get_subgraph(
                    entity_names=entity_names,
                    max_depth=max_hops,
                )
            else:
                result = await module.query(query=query)
            
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return {"error": str(e)}


class WebSearchTool(BaseTool):
    """Tool for web search (placeholder)."""
    
    def __init__(self, module_registry: IModuleRegistry, config: Dict[str, Any]):
        self._registry = module_registry
        self._config = config
        self._module_name = config.get("module", "web_search")
        self._max_results = config.get("max_results", 5)
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_search",
            description="Search the web for current information",
            parameters=[
                ToolParameter("query", "string", "Search query", required=True),
                ToolParameter("max_results", "integer", "Maximum results", required=False, default=5),
            ],
            returns="array of search results",
            category="web",
            timeout_seconds=15,
        )
    
    async def execute(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Execute web search."""
        max_results = max_results or self._max_results
        
        module = self._registry.get_module(self._module_name)
        if not module:
            return [{"error": f"Web search module not available"}]
        
        try:
            result = await module.search(query=query, max_results=max_results)
            if isinstance(result, dict):
                return result.get("results", [])
            return []
        except Exception as e:
            return [{"error": str(e)}]


# ============================================================================
# Tool Registry
# ============================================================================


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self, module_registry: IModuleRegistry):
        self._module_registry = module_registry
        self._tools: Dict[str, BaseTool] = {}
        self._schemas: Dict[str, ToolSchema] = {}
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.schema.name] = tool
        self._schemas[tool.schema.name] = tool.schema
        logger.debug(f"Registered tool: {tool.schema.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            del self._schemas[name]
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name."""
        return self._tools.get(name)
    
    def get_schema(self, name: str) -> Optional[ToolSchema]:
        """Get tool schema by name."""
        return self._schemas.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_all_schemas(self) -> List[ToolSchema]:
        """Get all tool schemas."""
        return list(self._schemas.values())
    
    def get_function_specs(self) -> List[Dict[str, Any]]:
        """Get all tools as function specifications."""
        return [schema.to_function_spec() for schema in self._schemas.values()]
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get tools by category."""
        return [
            tool for tool in self._tools.values()
            if tool.schema.category == category
        ]
    
    def register_builtin_tools(self, config: Dict[str, Any]) -> None:
        """Register built-in tools based on configuration."""
        builtin_config = config.get("tools", {}).get("builtin", {})
        
        # Retrieval tool
        if builtin_config.get("retrieval", {}).get("enabled", True):
            self.register(RetrievalTool(
                self._module_registry,
                builtin_config.get("retrieval", {}),
            ))
        
        # Calculator tool
        if builtin_config.get("calculator", {}).get("enabled", True):
            self.register(CalculatorTool(
                builtin_config.get("calculator", {}),
            ))
        
        # Summarizer tool
        if builtin_config.get("summarizer", {}).get("enabled", True):
            self.register(SummarizerTool(
                self._module_registry,
                builtin_config.get("summarizer", {}),
            ))
        
        # Graph query tool
        if builtin_config.get("graph_query", {}).get("enabled", True):
            self.register(GraphQueryTool(
                self._module_registry,
                builtin_config.get("graph_query", {}),
            ))
        
        # Web search tool
        if builtin_config.get("web_search", {}).get("enabled", False):
            self.register(WebSearchTool(
                self._module_registry,
                builtin_config.get("web_search", {}),
            ))


# ============================================================================
# Tool Executor
# ============================================================================


class ToolExecutor:
    """Executes tools with retry, timeout, and error handling."""
    
    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout: int = 30,
        max_retries: int = 2,
    ):
        self.registry = registry
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self._execution_counts: Dict[str, int] = {}
    
    async def execute(
        self,
        tool_call: ToolCall,
        timeout: Optional[int] = None,
    ) -> ToolResult:
        """Execute a tool call."""
        start_time = time.perf_counter()
        
        tool = self.registry.get_tool(tool_call.tool_name)
        if not tool:
            return ToolResult(
                tool_id=tool_call.tool_id,
                tool_name=tool_call.tool_name,
                success=False,
                error=f"Tool '{tool_call.tool_name}' not found",
            )
        
        # Validate arguments
        valid, error = await tool.validate_args(tool_call.arguments)
        if not valid:
            return ToolResult(
                tool_id=tool_call.tool_id,
                tool_name=tool_call.tool_name,
                success=False,
                error=error,
            )
        
        # Determine timeout
        tool_timeout = timeout or tool.schema.timeout_seconds or self.default_timeout
        
        # Execute with retry
        last_error = None
        max_retries = min(tool.schema.max_retries, self.max_retries)
        
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    tool.execute(**tool_call.arguments),
                    timeout=tool_timeout,
                )
                
                execution_time = (time.perf_counter() - start_time) * 1000
                self._execution_counts[tool_call.tool_name] = \
                    self._execution_counts.get(tool_call.tool_name, 0) + 1
                
                return ToolResult(
                    tool_id=tool_call.tool_id,
                    tool_name=tool_call.tool_name,
                    success=True,
                    result=result,
                    execution_time_ms=execution_time,
                    metadata={"attempt": attempt + 1},
                )
                
            except asyncio.TimeoutError:
                last_error = f"Timeout after {tool_timeout}s"
                logger.warning(f"Tool {tool_call.tool_name} timeout (attempt {attempt + 1})")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Tool {tool_call.tool_name} error: {e} (attempt {attempt + 1})")
            
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))  # Backoff
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return ToolResult(
            tool_id=tool_call.tool_id,
            tool_name=tool_call.tool_name,
            success=False,
            error=last_error,
            execution_time_ms=execution_time,
            metadata={"attempts": max_retries + 1},
        )
    
    async def execute_parallel(
        self,
        tool_calls: List[ToolCall],
        max_concurrent: int = 5,
    ) -> List[ToolResult]:
        """Execute multiple tool calls in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(call: ToolCall) -> ToolResult:
            async with semaphore:
                return await self.execute(call)
        
        results = await asyncio.gather(
            *[execute_with_semaphore(call) for call in tool_calls],
            return_exceptions=True,
        )
        
        # Convert exceptions to ToolResults
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ToolResult(
                    tool_id=tool_calls[i].tool_id,
                    tool_name=tool_calls[i].tool_name,
                    success=False,
                    error=str(result),
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def get_execution_stats(self) -> Dict[str, int]:
        """Get tool execution statistics."""
        return dict(self._execution_counts)
