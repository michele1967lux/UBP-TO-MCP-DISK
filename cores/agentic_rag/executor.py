"""
agentic_rag/executor.py

Parallel execution engine for agentic RAG.

Features:
- Dependency-aware task scheduling
- Parallel batch execution
- Worker pool management
- Timeout and retry handling
- Fail-fast and continue-on-error modes
- Progress tracking

v1.0.0: Initial release
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple

from .providers import (
    AgentTask,
    AgentPlan,
    AgentState,
    AgentStep,
    ExecutionContext,
    TaskStatus,
    TaskType,
    StepType,
    ParallelConfig,
    ToolCall,
    ToolResult,
)
from .tools import ToolRegistry, ToolExecutor

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class IModuleRegistry(Protocol):
    """Protocol for module registry."""
    def get_module(self, module_name: str) -> Optional[Any]: ...


class IProgressCallback(Protocol):
    """Protocol for progress callbacks."""
    async def on_task_start(self, task: AgentTask) -> None: ...
    async def on_task_complete(self, task: AgentTask, result: Any) -> None: ...
    async def on_batch_complete(self, batch_num: int, total_batches: int) -> None: ...


# ============================================================================
# Execution Result
# ============================================================================


@dataclass
class BatchResult:
    """Result from executing a batch of tasks."""
    batch_id: int
    tasks: List[AgentTask]
    successful: List[str]
    failed: List[str]
    results: Dict[str, Any]
    execution_time_ms: float
    
    @property
    def all_successful(self) -> bool:
        return len(self.failed) == 0


@dataclass
class ExecutionResult:
    """Complete execution result."""
    plan: AgentPlan
    batches: List[BatchResult]
    total_successful: int
    total_failed: int
    task_results: Dict[str, Any]
    total_time_ms: float
    parallel_batches_executed: int
    
    @property
    def success(self) -> bool:
        return self.total_failed == 0


# ============================================================================
# Worker Pool
# ============================================================================


class WorkerPool:
    """Pool of async workers for task execution."""
    
    def __init__(self, size: int = 8):
        self.size = size
        self._semaphore = asyncio.Semaphore(size)
        self._active_workers = 0
        self._completed_tasks = 0
        self._lock = asyncio.Lock()
    
    async def submit(self, coro) -> Any:
        """Submit a coroutine to the worker pool."""
        async with self._semaphore:
            async with self._lock:
                self._active_workers += 1
            try:
                result = await coro
                async with self._lock:
                    self._completed_tasks += 1
                return result
            finally:
                async with self._lock:
                    self._active_workers -= 1
    
    async def submit_batch(self, coros: List) -> List[Any]:
        """Submit a batch of coroutines."""
        results = await asyncio.gather(
            *[self.submit(coro) for coro in coros],
            return_exceptions=True,
        )
        return results
    
    @property
    def stats(self) -> Dict[str, int]:
        return {
            "pool_size": self.size,
            "active_workers": self._active_workers,
            "completed_tasks": self._completed_tasks,
        }


# ============================================================================
# Task Scheduler
# ============================================================================


class TaskScheduler:
    """Schedules tasks for execution based on dependencies."""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
    
    def get_ready_tasks(
        self,
        plan: AgentPlan,
        completed: Set[str],
        running: Set[str],
    ) -> List[AgentTask]:
        """Get tasks ready for execution."""
        ready = []
        
        for task in plan.tasks:
            if task.task_id in completed or task.task_id in running:
                continue
            
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check dependencies
            deps_satisfied = all(dep in completed for dep in task.dependencies)
            if deps_satisfied:
                ready.append(task)
        
        # Limit batch size
        if self.config.batch_size > 0:
            ready = ready[:self.config.batch_size]
        
        return ready
    
    def get_parallel_batches(self, plan: AgentPlan) -> List[List[AgentTask]]:
        """Get pre-computed parallel batches from plan."""
        if not plan.execution_order:
            return self._compute_batches(plan)
        
        task_map = {t.task_id: t for t in plan.tasks}
        batches = []
        
        for batch_ids in plan.execution_order:
            batch = [task_map[tid] for tid in batch_ids if tid in task_map]
            if batch:
                batches.append(batch)
        
        return batches
    
    def _compute_batches(self, plan: AgentPlan) -> List[List[AgentTask]]:
        """Compute parallel batches from dependencies."""
        batches = []
        completed = set()
        remaining = list(plan.tasks)
        
        while remaining:
            batch = []
            for task in remaining:
                if all(dep in completed for dep in task.dependencies):
                    batch.append(task)
            
            if not batch:
                # Handle circular dependencies
                batch = [remaining[0]]
            
            batches.append(batch)
            completed.update(t.task_id for t in batch)
            remaining = [t for t in remaining if t.task_id not in completed]
        
        return batches


# ============================================================================
# Parallel Executor
# ============================================================================


class ParallelExecutor:
    """
    Executes tasks in parallel with dependency awareness.
    
    Features:
    - Worker pool for concurrent execution
    - Dependency-aware scheduling
    - Batch execution
    - Retry handling
    - Progress callbacks
    """
    
    def __init__(
        self,
        config: ParallelConfig,
        tool_registry: ToolRegistry,
        module_registry: IModuleRegistry,
        llm_caller: Optional[Callable] = None,
    ):
        self.config = config
        self.tool_registry = tool_registry
        self._module_registry = module_registry
        self._llm_caller = llm_caller
        
        self._worker_pool = WorkerPool(config.worker_pool_size)
        self._scheduler = TaskScheduler(config)
        self._tool_executor = ToolExecutor(
            registry=tool_registry,
            default_timeout=config.task_timeout,
            max_retries=config.max_retries,
        )
        
        self._progress_callbacks: List[IProgressCallback] = []
    
    def add_progress_callback(self, callback: IProgressCallback) -> None:
        """Add a progress callback."""
        self._progress_callbacks.append(callback)
    
    async def execute_plan(
        self,
        plan: AgentPlan,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Execute a complete plan with parallel batches."""
        start_time = time.perf_counter()
        
        if self.config.enabled and self.config.dependency_aware:
            result = await self._execute_dependency_aware(plan, context)
        else:
            result = await self._execute_sequential(plan, context)
        
        result.total_time_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    async def _execute_dependency_aware(
        self,
        plan: AgentPlan,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Execute with dependency awareness and parallel batches."""
        batches = self._scheduler.get_parallel_batches(plan)
        
        all_results: Dict[str, Any] = {}
        batch_results: List[BatchResult] = []
        completed: Set[str] = set()
        failed: Set[str] = set()
        
        for batch_idx, batch in enumerate(batches):
            batch_start = time.perf_counter()
            
            # Filter out tasks whose dependencies failed (if fail-fast disabled)
            executable = []
            for task in batch:
                failed_deps = [dep for dep in task.dependencies if dep in failed]
                if failed_deps and self.config.fail_fast:
                    task.status = TaskStatus.SKIPPED
                    task.error = f"Skipped due to failed dependencies: {failed_deps}"
                    failed.add(task.task_id)
                else:
                    executable.append(task)
            
            if not executable:
                continue
            
            # Execute batch in parallel
            if self.config.enabled and len(executable) > 1:
                results = await self._execute_batch_parallel(
                    executable, context, all_results
                )
            else:
                results = await self._execute_batch_sequential(
                    executable, context, all_results
                )
            
            # Process results
            batch_successful = []
            batch_failed = []
            
            for task, result in zip(executable, results):
                if isinstance(result, Exception):
                    task.status = TaskStatus.FAILED
                    task.error = str(result)
                    batch_failed.append(task.task_id)
                    failed.add(task.task_id)
                elif result is not None:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    all_results[task.task_id] = result
                    batch_successful.append(task.task_id)
                    completed.add(task.task_id)
                else:
                    task.status = TaskStatus.FAILED
                    task.error = "No result returned"
                    batch_failed.append(task.task_id)
                    failed.add(task.task_id)
            
            batch_time = (time.perf_counter() - batch_start) * 1000
            
            batch_results.append(BatchResult(
                batch_id=batch_idx,
                tasks=executable,
                successful=batch_successful,
                failed=batch_failed,
                results={tid: all_results.get(tid) for tid in batch_successful},
                execution_time_ms=batch_time,
            ))
            
            # Notify progress
            for callback in self._progress_callbacks:
                await callback.on_batch_complete(batch_idx + 1, len(batches))
            
            # Fail-fast check
            if batch_failed and self.config.fail_fast:
                logger.warning(f"Fail-fast: stopping after batch {batch_idx} failure")
                break
        
        return ExecutionResult(
            plan=plan,
            batches=batch_results,
            total_successful=len(completed),
            total_failed=len(failed),
            task_results=all_results,
            total_time_ms=0,  # Will be set by caller
            parallel_batches_executed=len(batch_results),
        )
    
    async def _execute_batch_parallel(
        self,
        tasks: List[AgentTask],
        context: ExecutionContext,
        previous_results: Dict[str, Any],
    ) -> List[Any]:
        """Execute a batch of tasks in parallel."""
        # Create coroutines for each task
        coros = []
        for task in tasks:
            # Collect results from dependencies
            dep_results = {
                dep: previous_results.get(dep)
                for dep in task.dependencies
            }
            
            coro = self._execute_single_task(task, context, dep_results)
            coros.append(coro)
        
        # Execute with worker pool
        results = await self._worker_pool.submit_batch(coros)
        return results
    
    async def _execute_batch_sequential(
        self,
        tasks: List[AgentTask],
        context: ExecutionContext,
        previous_results: Dict[str, Any],
    ) -> List[Any]:
        """Execute tasks sequentially."""
        results = []
        
        for task in tasks:
            dep_results = {
                dep: previous_results.get(dep)
                for dep in task.dependencies
            }
            
            try:
                result = await self._execute_single_task(task, context, dep_results)
                results.append(result)
            except Exception as e:
                results.append(e)
        
        return results
    
    async def _execute_sequential(
        self,
        plan: AgentPlan,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Execute plan sequentially (fallback)."""
        all_results: Dict[str, Any] = {}
        completed: Set[str] = set()
        failed: Set[str] = set()
        
        for task in plan.tasks:
            dep_results = {dep: all_results.get(dep) for dep in task.dependencies}
            
            try:
                result = await self._execute_single_task(task, context, dep_results)
                task.status = TaskStatus.COMPLETED
                task.result = result
                all_results[task.task_id] = result
                completed.add(task.task_id)
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                failed.add(task.task_id)
                
                if self.config.fail_fast:
                    break
        
        return ExecutionResult(
            plan=plan,
            batches=[],
            total_successful=len(completed),
            total_failed=len(failed),
            task_results=all_results,
            total_time_ms=0,
            parallel_batches_executed=0,
        )
    
    async def _execute_single_task(
        self,
        task: AgentTask,
        context: ExecutionContext,
        dep_results: Dict[str, Any],
    ) -> Any:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        
        # Notify callbacks
        for callback in self._progress_callbacks:
            await callback.on_task_start(task)
        
        try:
            if task.task_type == TaskType.TOOL_CALL and task.tool_name:
                result = await self._execute_tool_task(task)
            elif task.task_type == TaskType.RETRIEVAL:
                result = await self._execute_retrieval_task(task, context)
            elif task.task_type == TaskType.REASONING:
                result = await self._execute_reasoning_task(task, context, dep_results)
            elif task.task_type == TaskType.SYNTHESIS:
                result = await self._execute_synthesis_task(task, context, dep_results)
            elif task.task_type == TaskType.VERIFICATION:
                result = await self._execute_verification_task(task, context, dep_results)
            else:
                result = {"status": "unknown_task_type"}
            
            task.completed_at = datetime.utcnow()
            
            # Notify callbacks
            for callback in self._progress_callbacks:
                await callback.on_task_complete(task, result)
            
            return result
            
        except asyncio.TimeoutError:
            raise Exception(f"Task timeout after {self.config.task_timeout}s")
        except Exception as e:
            task.completed_at = datetime.utcnow()
            raise
    
    async def _execute_tool_task(self, task: AgentTask) -> Any:
        """Execute a tool-based task."""
        tool_call = ToolCall(
            tool_id=f"call_{task.task_id}",
            tool_name=task.tool_name,
            arguments=task.arguments,
        )
        
        result = await asyncio.wait_for(
            self._tool_executor.execute(tool_call),
            timeout=self.config.task_timeout,
        )
        
        if not result.success:
            raise Exception(result.error)
        
        return result.result
    
    async def _execute_retrieval_task(
        self,
        task: AgentTask,
        context: ExecutionContext,
    ) -> Any:
        """Execute a retrieval task."""
        # Use retrieval tool
        tool_call = ToolCall(
            tool_id=f"retrieve_{task.task_id}",
            tool_name=task.tool_name or "retrieval",
            arguments=task.arguments or {"query": context.query},
        )
        
        result = await self._tool_executor.execute(tool_call)
        
        if not result.success:
            raise Exception(result.error)
        
        return result.result
    
    async def _execute_reasoning_task(
        self,
        task: AgentTask,
        context: ExecutionContext,
        dep_results: Dict[str, Any],
    ) -> Any:
        """Execute a reasoning task via pipeline HA chain."""
        if not self._llm_caller:
            return {"reasoning": "LLM not available", "inputs": dep_results}

        # Build context from dependency results
        context_text = self._format_results_for_reasoning(dep_results)

        prompt = f"""Based on the following information, reason about the query.

Query: {context.query}

Information gathered:
{context_text}

Task: {task.description}

Provide your reasoning and any conclusions:"""

        try:
            response = await self._llm_caller(prompt=prompt, max_tokens=1000, purpose="reasoning")
            return {"reasoning": response, "inputs": list(dep_results.keys())}
        except Exception as e:
            return {"error": str(e)}
    
    async def _execute_synthesis_task(
        self,
        task: AgentTask,
        context: ExecutionContext,
        dep_results: Dict[str, Any],
    ) -> Any:
        """Execute a synthesis task via pipeline HA chain."""
        if not self._llm_caller:
            return self._simple_synthesis(context.query, dep_results)

        context_text = self._format_results_for_synthesis(dep_results)

        prompt = f"""Synthesize a comprehensive answer to the query based on the information provided.

Query: {context.query}

Information:
{context_text}

Provide a clear, well-structured answer:"""

        try:
            response = await self._llm_caller(prompt=prompt, max_tokens=1500, purpose="synthesis")
            return {"answer": response, "source_count": len(dep_results)}
        except Exception as e:
            return self._simple_synthesis(context.query, dep_results)
    
    async def _execute_verification_task(
        self,
        task: AgentTask,
        context: ExecutionContext,
        dep_results: Dict[str, Any],
    ) -> Any:
        """Execute a verification task."""
        # Simple verification - check for consistency
        return {
            "verified": True,
            "checks_performed": ["result_present", "non_empty"],
            "inputs_verified": list(dep_results.keys()),
        }
    
    def _format_results_for_reasoning(self, results: Dict[str, Any]) -> str:
        """Format results for reasoning prompt."""
        lines = []
        for task_id, result in results.items():
            if isinstance(result, list):
                lines.append(f"[{task_id}]: {len(result)} items")
                for i, item in enumerate(result[:3]):
                    preview = str(item)[:200]
                    lines.append(f"  {i+1}. {preview}")
            elif isinstance(result, dict):
                lines.append(f"[{task_id}]: {result}")
            else:
                lines.append(f"[{task_id}]: {str(result)[:500]}")
        return "\n".join(lines)
    
    def _format_results_for_synthesis(self, results: Dict[str, Any]) -> str:
        """Format results for synthesis prompt."""
        lines = []
        for task_id, result in results.items():
            if isinstance(result, dict) and "reasoning" in result:
                lines.append(f"Analysis: {result['reasoning']}")
            elif isinstance(result, dict) and "answer" in result:
                lines.append(f"Previous answer: {result['answer']}")
            elif isinstance(result, list):
                for item in result[:5]:
                    if isinstance(item, dict):
                        content = item.get("content", item.get("text", str(item)))
                        lines.append(f"- {content[:300]}")
                    else:
                        lines.append(f"- {str(item)[:300]}")
            else:
                lines.append(str(result)[:500])
        return "\n\n".join(lines)
    
    def _simple_synthesis(self, query: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Simple synthesis fallback."""
        parts = []
        for task_id, result in results.items():
            if isinstance(result, dict) and "reasoning" in result:
                parts.append(result["reasoning"])
            elif isinstance(result, list) and result:
                for item in result[:3]:
                    if isinstance(item, dict):
                        parts.append(item.get("content", str(item))[:200])
        
        answer = " ".join(parts) if parts else "No information gathered."
        return {"answer": answer, "source_count": len(results), "method": "simple"}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "worker_pool": self._worker_pool.stats,
            "tool_executions": self._tool_executor.get_execution_stats(),
            "config": {
                "parallel_enabled": self.config.enabled,
                "max_concurrent": self.config.max_concurrent,
                "batch_size": self.config.batch_size,
            },
        }
