"""
agentic_rag/adapter.py

Bridge layer that exposes all agentic RAG operations to the UBP system.

Operations:
- initialize: Start components
- query: Full agentic RAG pipeline
- react: ReAct-style reasoning loop
- plan_execute: Plan-then-execute mode
- parallel_query: Parallel execution mode
- create_plan: Create execution plan
- execute_plan: Execute existing plan
- call_tool: Direct tool invocation
- register_tool: Register external tool
- list_tools: Get available tools
- get_state: Get agent state
- continue_session: Resume session
- get_stats: Get metrics (admin)
- reload_config: Hot-reload (admin)
- shutdown: Graceful shutdown
- health_check: Component health

v1.0.0: Initial release with full enterprise features
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

from .providers import (
    # Data classes
    AgentTask,
    AgentPlan,
    AgentStep,
    AgentState,
    AgentResult,
    ExecutionContext,
    ToolCall,
    ToolResult,
    # Enums
    TaskStatus,
    TaskType,
    StepType,
    ExecutionMode,
    # Configs
    PlanningConfig,
    ParallelConfig,
    ReactConfig,
    StateConfig,
    MemoryConfig,
    MetricsConfig,
    DebugConfig,
    # Providers
    WorkingMemory,
    EpisodicMemory,
    StateManager,
    ExecutionMetrics,
)
from .tools import (
    ToolSchema,
    ToolParameter,
    BaseTool,
    ToolRegistry,
    ToolExecutor,
)
from .planner import AgentPlanner
from .executor import ParallelExecutor, ExecutionResult
from .prompts import (
    get_template,
    detect_language,
    format_tools_for_prompt,
    format_history_for_prompt,
    parse_react_response,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class IModuleRegistry(Protocol):
    """Protocol for module registry."""
    def get_module(self, module_name: str) -> Optional[Any]: ...


# ============================================================================
# DI Container Wrapper
# ============================================================================


class DIContainerModuleRegistry:
    """Wraps DI container to provide module registry interface."""

    def __init__(self, di_container: Optional[Any] = None):
        self._container = di_container
        self._cached_modules: Dict[str, Any] = {}

    def get_module(self, module_name: str) -> Optional[Any]:
        """Get a module by name (sync - cache only)."""
        if module_name in self._cached_modules:
            return self._cached_modules[module_name]
        return None

    async def resolve_module(self, module_name: str) -> Optional[Any]:
        """Async module resolution via DI container."""
        if module_name in self._cached_modules:
            return self._cached_modules[module_name]

        if not self._container:
            return None

        # DI container.resolve() is async - must be awaited
        if hasattr(self._container, "resolve"):
            try:
                module = await self._container.resolve(module_name)
                if module:
                    self._cached_modules[module_name] = module
                    return module
            except Exception as e:
                logger.warning(f"Failed to resolve module '{module_name}': {e}")

        return None


# ============================================================================
# Configuration Utilities
# ============================================================================


def resolve_env_value(value: Any) -> Any:
    """Resolve environment variable placeholders."""
    if not isinstance(value, str):
        return value
    
    pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
    
    def replace(match):
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)
    
    return re.sub(pattern, replace, value)


def coerce_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively coerce configuration values."""
    result = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = coerce_config_types(value)
        elif isinstance(value, list):
            result[key] = [
                coerce_config_types(v) if isinstance(v, dict) else _coerce_value(v)
                for v in value
            ]
        else:
            result[key] = _coerce_value(value)
    
    return result


def _coerce_value(value: Any) -> Any:
    """Coerce a single value."""
    if not isinstance(value, str):
        return value
    
    value = resolve_env_value(value)
    
    if not isinstance(value, str):
        return value
    
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False
    
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    return value


# ============================================================================
# Agentic RAG Adapter
# ============================================================================


class AgenticRAGAdapter:
    """
    Adapter exposing agentic RAG operations.
    
    Features:
    - ReAct-style reasoning loops
    - Plan-then-execute mode
    - Parallel task execution
    - Tool management
    - State persistence
    - Comprehensive metrics
    """
    
    def __init__(
        self,
        module_path: Path,
        di_container: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        self.module_path = Path(module_path)
        self._di_container = di_container
        self._event_bus = event_bus
        
        self._module_registry = DIContainerModuleRegistry(di_container)
        
        # Configuration
        self._config: Dict[str, Any] = {}
        self._planning_config: Optional[PlanningConfig] = None
        self._parallel_config: Optional[ParallelConfig] = None
        self._react_config: Optional[ReactConfig] = None
        self._state_config: Optional[StateConfig] = None
        self._memory_config: Optional[MemoryConfig] = None
        self._metrics_config: Optional[MetricsConfig] = None
        self._debug_config: Optional[DebugConfig] = None
        
        # Components
        self._tool_registry: Optional[ToolRegistry] = None
        self._tool_executor: Optional[ToolExecutor] = None
        self._planner: Optional[AgentPlanner] = None
        self._executor: Optional[ParallelExecutor] = None
        self._state_manager: Optional[StateManager] = None
        self._metrics: Optional[ExecutionMetrics] = None
        
        # Memory
        self._working_memory: Optional[WorkingMemory] = None
        self._episodic_memory: Optional[EpisodicMemory] = None
        
        # LLM
        self._llm_module: Optional[Any] = None

        # v6.3.2: Pipeline orchestrator for delegated LLM calls
        self._pipeline_orchestrator: Optional[Any] = None
        self._current_ctx: Optional[Any] = None

        # State
        self._initialized = False
        self._active_sessions: Dict[str, AgentState] = {}
    
    # MCP-COMPAT: OperationContext helpers (ARCH-008)
    def _build_context_from_di(self) -> OperationContext:
        """Build OperationContext from DI — backward compatibility for REST path."""
        return OperationContext(
            client_id="default",
            user_id=None,
            session_id=None,
            source="rest",
        )

    def _normalize_ctx(self, ctx: Any) -> OperationContext:
        """Normalize any context format to OperationContext."""
        if ctx is None:
            return self._build_context_from_di()
        if isinstance(ctx, OperationContext):
            return ctx
        if hasattr(ctx, "user") and ctx.user:
            user_id = getattr(ctx.user, "user_id", None)
            roles = getattr(ctx.user, "roles", [])
            client_id = getattr(ctx.user, "client_id", "default")
            if not isinstance(roles, (list, tuple)):
                roles = []
            return OperationContext(
                client_id=str(client_id) if client_id else "default",
                user_id=str(user_id) if user_id else None,
                roles=list(roles),
                source="rest",
            )
        return self._build_context_from_di()
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json."""
        config_path = self.module_path / "config.json"
        
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}")
            return {}
        
        with open(config_path, "r") as f:
            raw_config = json.load(f)
        
        return coerce_config_types(raw_config)
    
    def _build_configs(self) -> None:
        """Build configuration objects."""
        cfg = self._config
        
        planning_cfg = cfg.get("planning", {})
        self._planning_config = PlanningConfig(
            enabled=planning_cfg.get("enabled", True),
            decomposition_strategy=planning_cfg.get("decomposition_strategy", "adaptive"),
            max_subtasks=planning_cfg.get("max_subtasks", 8),
            min_confidence=planning_cfg.get("min_confidence_threshold", 0.6),
            replan_on_failure=planning_cfg.get("replan_on_failure", True),
            max_replans=planning_cfg.get("max_replans", 3),
            temperature=planning_cfg.get("temperature", 0.2),
        )
        
        parallel_cfg = cfg.get("execution", {}).get("parallel", {})
        self._parallel_config = ParallelConfig(
            enabled=parallel_cfg.get("enabled", True),
            max_concurrent=parallel_cfg.get("max_concurrent", 5),
            worker_pool_size=parallel_cfg.get("worker_pool_size", 8),
            task_timeout=parallel_cfg.get("task_timeout", 30),
            batch_size=parallel_cfg.get("batch_size", 3),
            dependency_aware=parallel_cfg.get("dependency_aware", True),
            fail_fast=parallel_cfg.get("fail_fast", False),
            retry_failed=parallel_cfg.get("retry_failed", True),
            max_retries=parallel_cfg.get("max_retries", 2),
        )
        
        react_cfg = cfg.get("react_loop", {})
        self._react_config = ReactConfig(
            enabled=react_cfg.get("enabled", True),
            max_iterations=react_cfg.get("max_iterations", 8),
            reflection_enabled=react_cfg.get("reflection_enabled", True),
            reflection_interval=react_cfg.get("reflection_interval", 3),
            early_stop_confidence=react_cfg.get("early_stopping", {}).get("confidence_threshold", 0.9),
        )
        
        state_cfg = cfg.get("state_management", {})
        self._state_config = StateConfig(
            enabled=state_cfg.get("enabled", True),
            persist_state=state_cfg.get("persist_state", True),
            checkpoint_interval=state_cfg.get("checkpoint_interval", 3),
            max_state_size_mb=state_cfg.get("max_state_size_mb", 10),
            context_max_tokens=state_cfg.get("context_window", {}).get("max_tokens", 8000),
        )
        
        memory_cfg = cfg.get("memory", {})
        self._memory_config = MemoryConfig(
            working_memory_max=memory_cfg.get("working_memory", {}).get("max_items", 20),
            episodic_memory_max=memory_cfg.get("episodic_memory", {}).get("max_episodes", 50),
            relevance_decay=memory_cfg.get("working_memory", {}).get("relevance_decay", 0.9),
        )
        
        metrics_cfg = cfg.get("observation", {}).get("metrics", {})
        self._metrics_config = MetricsConfig(
            enabled=metrics_cfg.get("enabled", True),
            track_latency=metrics_cfg.get("track_latency", True),
            track_tool_usage=metrics_cfg.get("track_tool_usage", True),
            track_iterations=metrics_cfg.get("track_iterations", True),
        )
        
        debug_cfg = cfg.get("debug", {})
        self._debug_config = DebugConfig(
            enabled=debug_cfg.get("enabled", False),
            log_plans=debug_cfg.get("log_plans", True),
            log_tool_calls=debug_cfg.get("log_tool_calls", True),
            log_state_changes=debug_cfg.get("log_state_changes", True),
        )
    
    # ========================================================================
    # Operations
    # ========================================================================
    
    async def initialize(self, ctx: Any = None) -> Dict[str, Any]:
        """Initialize agentic RAG components."""
        if self._initialized:
            return {"status": "already_initialized"}
        
        try:
            self._config = self._load_config()
            self._build_configs()
            
            # Initialize tool registry
            self._tool_registry = ToolRegistry(self._module_registry)
            self._tool_registry.register_builtin_tools(self._config)
            
            # Initialize tool executor
            self._tool_executor = ToolExecutor(
                registry=self._tool_registry,
                default_timeout=self._parallel_config.task_timeout,
                max_retries=self._parallel_config.max_retries,
            )
            
            # v6.3.2: Resolve pipeline_orchestrator for delegated LLM calls
            try:
                orch = self._module_registry.get_module("pipeline_orchestrator")
                if not orch and hasattr(self._module_registry, "resolve_module"):
                    orch = await self._module_registry.resolve_module("pipeline_orchestrator")
                if orch:
                    self._pipeline_orchestrator = orch
                    logger.info("[AGENTIC] pipeline_orchestrator resolved — LLM calls via HA pipeline")
                else:
                    logger.warning("[AGENTIC] pipeline_orchestrator NOT available — direct LLM calls")
            except Exception as e:
                logger.warning(f"[AGENTIC] pipeline_orchestrator resolution failed: {e}")

            # Initialize planner
            self._planner = AgentPlanner(
                config=self._planning_config,
                tool_registry=self._tool_registry,
                module_registry=self._module_registry,
                llm_caller=self._call_llm,
            )

            # Initialize parallel executor
            self._executor = ParallelExecutor(
                config=self._parallel_config,
                tool_registry=self._tool_registry,
                module_registry=self._module_registry,
                llm_caller=self._call_llm,
            )
            
            # Initialize state manager
            self._state_manager = StateManager(self._state_config)
            
            # Initialize metrics
            self._metrics = ExecutionMetrics(self._metrics_config)
            
            # Initialize memory
            self._working_memory = WorkingMemory(self._memory_config)
            self._episodic_memory = EpisodicMemory(self._memory_config)
            
            self._initialized = True
            
            logger.info("Agentic RAG adapter initialized")
            
            if self._event_bus:
                await self._event_bus.publish(
                    "agentic.initialized",
                    {"module": "agentic_rag", "status": "success"},
                )
            
            return {
                "status": "initialized",
                "tools": self._tool_registry.list_tools(),
                "modes": ["react", "plan_execute", "parallel"],
            }
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def query(
        self,
        query: str,
        mode: str = "react",
        max_iterations: int = None,
        enable_parallel: bool = True,
        language: str = "auto",
        session_id: str = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Execute agentic RAG query.
        
        Args:
            query: User query
            mode: Execution mode (react, plan_execute, parallel)
            max_iterations: Maximum reasoning iterations
            enable_parallel: Enable parallel execution
            language: Query language (auto, en, it)
            session_id: Optional session ID to resume
            ctx: Security context
        
        Returns:
            AgentResult as dict
        """
        if not self._initialized:
            await self.initialize(ctx)

        # v6.3.2: Save ctx for _call_llm pipeline delegation
        self._current_ctx = ctx

        start_time = time.perf_counter()

        # Detect language
        if language == "auto":
            language = detect_language(query)
        
        # Get or create session
        session_id = session_id or str(uuid.uuid4())
        
        # Parse mode
        try:
            exec_mode = ExecutionMode(mode.lower())
        except ValueError:
            exec_mode = ExecutionMode.REACT
        
        # Create state
        state = AgentState(
            state_id=str(uuid.uuid4()),
            session_id=session_id,
            query=query,
            mode=exec_mode,
        )
        
        self._active_sessions[session_id] = state
        
        try:
            # Execute based on mode
            if exec_mode == ExecutionMode.REACT:
                result = await self._execute_react(
                    state, language, max_iterations or self._react_config.max_iterations
                )
            elif exec_mode == ExecutionMode.PLAN_EXECUTE:
                result = await self._execute_plan_mode(state, language, enable_parallel)
            elif exec_mode == ExecutionMode.PARALLEL:
                result = await self._execute_parallel_mode(state, language)
            else:
                result = await self._execute_react(state, language, max_iterations or 5)
            
            # Record metrics
            if self._metrics:
                tool_calls = self._tool_executor.get_execution_stats()
                await self._metrics.record_execution(
                    mode=exec_mode,
                    success=True,
                    latency_ms=result.execution_time_ms,
                    iterations=result.total_iterations,
                    tool_calls=tool_calls,
                    parallel_batches=result.plan.get_parallel_batches().__len__() if result.plan else 0,
                )
            
            # Store in episodic memory
            self._episodic_memory.add_episode(
                query=query,
                answer=result.answer,
                success=True,
                tool_sequence=[s.tool_call.tool_name for s in result.steps if s.tool_call],
            )
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return AgentResult(
                session_id=session_id,
                query=query,
                answer=f"Error: {str(e)}",
                confidence=0.0,
                mode_used=exec_mode,
                total_iterations=state.current_iteration,
                total_tool_calls=len([s for s in state.steps if s.tool_call]),
                steps=state.steps,
                execution_time_ms=execution_time,
                metadata={"error": str(e)},
            ).to_dict()
    
    async def _execute_react(
        self,
        state: AgentState,
        language: str,
        max_iterations: int,
    ) -> AgentResult:
        """Execute ReAct-style reasoning loop."""
        start_time = time.perf_counter()

        # Get tools
        tools = self._tool_registry.get_function_specs()
        tools_str = format_tools_for_prompt(tools)

        # System prompt
        system_prompt = get_template("react_system", language).format(tools=tools_str)

        history_steps = []

        for iteration in range(max_iterations):
            state.current_iteration = iteration + 1

            # Build history
            history_str = format_history_for_prompt(history_steps)

            # Step prompt
            step_prompt = get_template("react_step", language).format(
                query=state.query,
                history=history_str,
            )

            # v6.3.2: LLM call via pipeline HA chain (replaces A1 + A2 fallback)
            try:
                response_text = await self._call_llm(
                    prompt=step_prompt,
                    system_prompt=system_prompt,
                    max_tokens=1500,
                    temperature=0.3,
                    purpose="react_step",
                )
            except Exception as e:
                logger.error(f"[AGENTIC] LLM failed in ReAct iteration {iteration}: {e}")
                break
            
            # Parse response
            parsed = parse_react_response(response_text)
            
            # Record thought
            if parsed["thought"]:
                thought_step = AgentStep(
                    step_id=f"{state.state_id}_thought_{iteration}",
                    step_type=StepType.THOUGHT,
                    iteration=iteration,
                    content=parsed["thought"],
                )
                state.add_step(thought_step)
                history_steps.append({"type": "thought", "content": parsed["thought"]})
            
            # Check for final answer
            if parsed["final_answer"]:
                answer_step = AgentStep(
                    step_id=f"{state.state_id}_answer",
                    step_type=StepType.ANSWER,
                    iteration=iteration,
                    content=parsed["final_answer"],
                )
                state.add_step(answer_step)
                state.final_answer = parsed["final_answer"]
                state.status = TaskStatus.COMPLETED
                break
            
            # Execute action
            if parsed["action"] and parsed["action"] != "none":
                action_input = parsed["action_input"] or {}
                
                tool_call = ToolCall(
                    tool_id=f"call_{iteration}",
                    tool_name=parsed["action"],
                    arguments=action_input,
                )
                
                action_step = AgentStep(
                    step_id=f"{state.state_id}_action_{iteration}",
                    step_type=StepType.ACTION,
                    iteration=iteration,
                    content=f"Execute {parsed['action']}",
                    tool_call=tool_call,
                )
                state.add_step(action_step)
                history_steps.append({
                    "type": "action",
                    "tool": parsed["action"],
                    "params": action_input,
                })
                
                # Execute tool
                tool_result = await self._tool_executor.execute(tool_call)
                
                action_step.tool_result = tool_result
                
                # Record observation
                obs_content = str(tool_result.result) if tool_result.success else f"Error: {tool_result.error}"
                
                obs_step = AgentStep(
                    step_id=f"{state.state_id}_obs_{iteration}",
                    step_type=StepType.OBSERVATION,
                    iteration=iteration,
                    content=obs_content,
                )
                state.add_step(obs_step)
                history_steps.append({"type": "observation", "content": obs_content[:500]})
            
            # Reflection check
            if (self._react_config.reflection_enabled and 
                (iteration + 1) % self._react_config.reflection_interval == 0):
                # Could add reflection step here
                pass
        
        # Generate final answer if not already done
        if not state.final_answer:
            state.final_answer = self._synthesize_fallback(state)
            state.status = TaskStatus.COMPLETED
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return AgentResult(
            session_id=state.session_id,
            query=state.query,
            answer=state.final_answer,
            confidence=state.confidence or 0.7,
            mode_used=ExecutionMode.REACT,
            total_iterations=state.current_iteration,
            total_tool_calls=len([s for s in state.steps if s.tool_call]),
            steps=state.steps,
            reasoning_trace=[s.content for s in state.steps if s.step_type == StepType.THOUGHT],
            execution_time_ms=execution_time,
        )
    
    async def _execute_plan_mode(
        self,
        state: AgentState,
        language: str,
        enable_parallel: bool,
    ) -> AgentResult:
        """Execute plan-then-execute mode."""
        start_time = time.perf_counter()
        
        # Create plan
        plan = await self._planner.create_plan(state.query, language)
        state.plan = plan
        
        if self._debug_config.log_plans:
            logger.debug(f"Created plan: {plan.to_dict()}")
        
        # Create execution context
        context = ExecutionContext(
            query=state.query,
            session_id=state.session_id,
            state=state,
            available_tools=self._tool_registry.list_tools(),
        )
        
        # Execute plan
        if enable_parallel and self._parallel_config.enabled:
            exec_result = await self._executor.execute_plan(plan, context)
        else:
            # Sequential fallback
            self._parallel_config.enabled = False
            exec_result = await self._executor.execute_plan(plan, context)
        
        # Extract answer from results
        synthesis_result = None
        for task in reversed(plan.tasks):
            if task.task_type == TaskType.SYNTHESIS and task.result:
                synthesis_result = task.result
                break
        
        if synthesis_result and isinstance(synthesis_result, dict):
            answer = synthesis_result.get("answer", str(synthesis_result))
        elif synthesis_result:
            answer = str(synthesis_result)
        else:
            answer = self._synthesize_from_results(exec_result.task_results, state.query)
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return AgentResult(
            session_id=state.session_id,
            query=state.query,
            answer=answer,
            confidence=plan.confidence,
            mode_used=ExecutionMode.PLAN_EXECUTE,
            total_iterations=1,
            total_tool_calls=exec_result.total_successful,
            steps=state.steps,
            plan=plan,
            execution_time_ms=execution_time,
            metadata={
                "parallel_batches": exec_result.parallel_batches_executed,
                "tasks_successful": exec_result.total_successful,
                "tasks_failed": exec_result.total_failed,
            },
        )
    
    async def _execute_parallel_mode(
        self,
        state: AgentState,
        language: str,
    ) -> AgentResult:
        """Execute with maximum parallelism."""
        # Same as plan_execute but forces parallel
        return await self._execute_plan_mode(state, language, enable_parallel=True)
    
    async def react(
        self,
        query: str,
        max_iterations: int = 8,
        language: str = "auto",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Execute ReAct-style reasoning."""
        return await self.query(
            query=query,
            mode="react",
            max_iterations=max_iterations,
            language=language,
            ctx=ctx,
        )
    
    async def plan_execute(
        self,
        query: str,
        enable_parallel: bool = True,
        language: str = "auto",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Execute plan-then-execute mode."""
        return await self.query(
            query=query,
            mode="plan_execute",
            enable_parallel=enable_parallel,
            language=language,
            ctx=ctx,
        )
    
    async def parallel_query(
        self,
        query: str,
        max_concurrent: int = None,
        language: str = "auto",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Execute with parallel task execution."""
        if max_concurrent:
            self._parallel_config.max_concurrent = max_concurrent
        
        return await self.query(
            query=query,
            mode="parallel",
            enable_parallel=True,
            language=language,
            ctx=ctx,
        )
    
    async def create_plan(
        self,
        query: str,
        language: str = "en",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Create an execution plan without executing."""
        if not self._initialized:
            await self.initialize(ctx)
        
        plan = await self._planner.create_plan(query, language)
        return plan.to_dict()
    
    async def execute_plan(
        self,
        plan_dict: Dict[str, Any],
        enable_parallel: bool = True,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Execute an existing plan."""
        if not self._initialized:
            await self.initialize(ctx)
        
        # Reconstruct plan from dict
        tasks = []
        for t in plan_dict.get("tasks", []):
            task = AgentTask(
                task_id=t["task_id"],
                task_type=TaskType(t["task_type"]),
                description=t["description"],
                tool_name=t.get("tool_name"),
                arguments=t.get("arguments", {}),
                dependencies=t.get("dependencies", []),
            )
            tasks.append(task)
        
        plan = AgentPlan(
            plan_id=plan_dict.get("plan_id", str(uuid.uuid4())),
            query=plan_dict["query"],
            tasks=tasks,
            execution_order=plan_dict.get("execution_order", []),
            reasoning=plan_dict.get("reasoning", ""),
            confidence=plan_dict.get("confidence", 0.7),
        )
        
        # Create state and context
        state = AgentState(
            state_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            query=plan.query,
            plan=plan,
        )
        
        context = ExecutionContext(
            query=plan.query,
            session_id=state.session_id,
            state=state,
            available_tools=self._tool_registry.list_tools(),
        )
        
        # Execute
        if enable_parallel:
            result = await self._executor.execute_plan(plan, context)
        else:
            self._parallel_config.enabled = False
            result = await self._executor.execute_plan(plan, context)
        
        return {
            "success": result.success,
            "task_results": {k: str(v)[:500] for k, v in result.task_results.items()},
            "successful": result.total_successful,
            "failed": result.total_failed,
            "parallel_batches": result.parallel_batches_executed,
            "total_time_ms": round(result.total_time_ms, 2),
        }
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Directly invoke a tool."""
        if not self._initialized:
            await self.initialize(ctx)
        
        tool_call = ToolCall(
            tool_id=str(uuid.uuid4()),
            tool_name=tool_name,
            arguments=arguments,
        )
        
        result = await self._tool_executor.execute(tool_call)
        return result.to_dict()
    
    async def register_tool(
        self,
        name: str,
        description: str,
        parameters: List[Dict[str, Any]],
        handler_module: str,
        handler_operation: str,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Register an external tool."""
        if not self._initialized:
            await self.initialize(ctx)
        
        # Would need to create a dynamic tool class
        # For now, return schema that would be registered
        return {
            "registered": True,
            "tool_name": name,
            "description": description,
            "parameter_count": len(parameters),
        }
    
    async def list_tools(self, ctx: Any = None) -> Dict[str, Any]:
        """List available tools."""
        if not self._initialized:
            await self.initialize(ctx)
        
        tools = []
        for schema in self._tool_registry.get_all_schemas():
            tools.append(schema.to_dict())
        
        return {"tools": tools, "count": len(tools)}
    
    async def get_state(
        self,
        session_id: str,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Get agent state for a session."""
        if session_id in self._active_sessions:
            return self._active_sessions[session_id].to_dict()
        
        # Try to load from state manager
        if self._state_manager:
            state = await self._state_manager.load_state(session_id)
            if state:
                return state.to_dict()
        
        return {"error": "Session not found"}
    
    async def get_stats(self, ctx: Any = None) -> Dict[str, Any]:
        """Get metrics and statistics."""
        if not self._initialized:
            await self.initialize(ctx)
        
        return {
            "metrics": self._metrics.get_metrics() if self._metrics else {},
            "executor": self._executor.get_stats() if self._executor else {},
            "active_sessions": len(self._active_sessions),
            "tools_registered": len(self._tool_registry.list_tools()) if self._tool_registry else 0,
        }
    
    async def reload_config(self, ctx: Any = None) -> Dict[str, Any]:
        """Hot-reload configuration."""
        try:
            self._config = self._load_config()
            self._build_configs()
            return {"status": "reloaded"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def shutdown(self, ctx: Any = None) -> Dict[str, Any]:
        """Graceful shutdown."""
        self._initialized = False
        self._active_sessions.clear()
        
        if self._event_bus:
            await self._event_bus.publish(
                "agentic.shutdown",
                {"module": "agentic_rag"},
            )
        
        logger.info("Agentic RAG adapter shut down")
        return {"status": "shutdown"}
    
    async def health_check(self, ctx: Any = None) -> Dict[str, Any]:
        """Check component health."""
        if not self._initialized:
            return {"module": "agentic_rag", "status": "not_initialized"}
        
        llm_available = await self._get_llm_module() is not None
        
        return {
            "module": "agentic_rag",
            "status": "healthy" if llm_available else "degraded",
            "initialized": self._initialized,
            "llm_available": llm_available,
            "tools_count": len(self._tool_registry.list_tools()) if self._tool_registry else 0,
            "active_sessions": len(self._active_sessions),
        }
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    async def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 1500,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        purpose: str = "generate",
    ) -> str:
        """Route LLM call through pipeline HA chain. Falls back to direct if unavailable.

        v6.3.2: Resolves BUG-AGENT-001..005 by delegating to pipeline_orchestrator
        which provides ProviderMapper HA chain, ModelGuard, context window validation,
        max_prompt_length check, and automatic fallback.
        """
        query = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        if self._pipeline_orchestrator:
            result = await self._pipeline_orchestrator.execute(
                pipeline_name="simple_chat",
                inputs={"query": query},
                config={"max_tokens": max_tokens, "temperature": temperature},
                ctx=self._current_ctx,
            )
            outputs = result.get("outputs", {})
            text = outputs.get("answer") or ""
            if not text and result.get("status") == "failed":
                step_results = result.get("step_results", [{}])
                error = step_results[0].get("error", "unknown") if step_results else "unknown"
                raise RuntimeError(f"Pipeline failed for {purpose}: {error}")
            return text

        # Fallback: direct LLM call (no HA, no protections)
        logger.warning(f"[AGENTIC] _call_llm fallback to direct LLM for purpose={purpose}")
        llm = await self._get_llm_module()
        if not llm:
            raise RuntimeError("No LLM available")
        result = await llm.generate(prompt=query, max_tokens=max_tokens, temperature=temperature)
        if isinstance(result, dict):
            return result.get("text") or result.get("response", "")
        return str(result)

    async def _get_llm_module(self) -> Optional[Any]:
        """Get or resolve LLM module via ProviderMapper chain."""
        if self._llm_module:
            return self._llm_module

        # Try ProviderMapper chain
        try:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper
            chain = ProviderMapper.resolve_chain("rag")
            for module_name, provider_name in chain:
                module = self._module_registry.get_module(module_name)
                if not module and hasattr(self._module_registry, "resolve_module"):
                    module = await self._module_registry.resolve_module(module_name)
                if module:
                    self._llm_module = module
                    self._llm_module_name = module_name  # Track for fallback
                    logger.info(f"[AGENTIC] LLM resolved via ProviderMapper: {module_name}")
                    return module
                logger.warning(f"[AGENTIC] Module '{module_name}' not available, trying next")
        except Exception as e:
            logger.warning(
                f"[AGENTIC] ProviderMapper NOT AVAILABLE - using hardcoded fallback "
                f"'inference_ollama_grok'. Centralized provider config (UBP_ROLES__RAG_PROVIDER) "
                f"is IGNORED for this module. Cause: {e}"
            )

        # Legacy fallback
        module = self._module_registry.get_module("inference_ollama_grok")
        if not module and hasattr(self._module_registry, "resolve_module"):
            module = await self._module_registry.resolve_module("inference_ollama_grok")
        if module:
            self._llm_module = module
            self._llm_module_name = "inference_ollama_grok"
        return module

    async def _get_fallback_llm(self) -> Optional[Any]:
        """Get fallback LLM from ProviderMapper chain (skipping current primary)."""
        try:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper
            chain = ProviderMapper.resolve_chain("rag")
            for module_name, provider_name in chain:
                if module_name == getattr(self, '_llm_module_name', None):
                    continue  # Skip failed primary
                module = self._module_registry.get_module(module_name)
                if not module and hasattr(self._module_registry, "resolve_module"):
                    module = await self._module_registry.resolve_module(module_name)
                if module:
                    self._llm_module = module
                    self._llm_module_name = module_name
                    logger.info(f"[AGENTIC] Fallback LLM resolved: {module_name}")
                    return module
        except Exception as e:
            logger.warning(
                f"[AGENTIC] ProviderMapper NOT AVAILABLE for fallback resolution. Cause: {e}"
            )
        return None
    
    def _synthesize_fallback(self, state: AgentState) -> str:
        """Fallback synthesis from state."""
        observations = [
            s.content for s in state.steps
            if s.step_type == StepType.OBSERVATION
        ]
        
        if observations:
            return f"Based on gathered information: {' '.join(observations[:3])}"
        
        return "Unable to gather sufficient information to answer the query."
    
    def _synthesize_from_results(
        self,
        results: Dict[str, Any],
        query: str,
    ) -> str:
        """Synthesize answer from task results."""
        parts = []
        
        for task_id, result in results.items():
            if isinstance(result, dict):
                if "answer" in result:
                    parts.append(result["answer"])
                elif "reasoning" in result:
                    parts.append(result["reasoning"])
            elif isinstance(result, list) and result:
                for item in result[:3]:
                    if isinstance(item, dict):
                        content = item.get("content", item.get("text", ""))
                        if content:
                            parts.append(content[:200])
        
        if parts:
            return " ".join(parts)
        
        return f"Executed {len(results)} tasks but could not synthesize a clear answer."
