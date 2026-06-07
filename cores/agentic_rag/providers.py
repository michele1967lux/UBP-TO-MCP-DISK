"""
agentic_rag/providers.py

Core data classes and state management for agentic RAG.
ZERO dependencies from backend.app - must be testable standalone.

Provides:
- AgentTask: Single task representation
- AgentPlan: Execution plan with tasks
- ToolCall: Tool invocation
- ToolResult: Tool execution result
- AgentStep: Single reasoning step
- AgentState: Complete agent state
- ExecutionContext: Context for execution
- AgentResult: Final agent result
- WorkingMemory: Short-term memory
- EpisodicMemory: Long-term episode memory
- StateManager: State persistence
- ExecutionMetrics: Performance metrics

v1.0.0: Initial release
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Callable, Dict, FrozenSet, Generator, Iterable,
    List, Optional, Protocol, Set, Tuple, Union,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ExecutionMode(Enum):
    """Agent execution mode."""
    REACT = "react"  # ReAct loop (Thought-Action-Observation)
    PLAN_EXECUTE = "plan_execute"  # Plan then execute
    ITERATIVE = "iterative"  # Iterative refinement
    PARALLEL = "parallel"  # Parallel execution


class TaskType(Enum):
    """Types of agent tasks."""
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"
    REFLECTION = "reflection"


class StepType(Enum):
    """Types of agent steps."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    ANSWER = "answer"


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class PlanningConfig:
    """Planning configuration."""
    enabled: bool = True
    decomposition_strategy: str = "adaptive"
    max_subtasks: int = 8
    min_confidence: float = 0.6
    replan_on_failure: bool = True
    max_replans: int = 3
    temperature: float = 0.2


@dataclass
class ParallelConfig:
    """Parallel execution configuration."""
    enabled: bool = True
    max_concurrent: int = 5
    worker_pool_size: int = 8
    task_timeout: int = 30
    batch_size: int = 3
    dependency_aware: bool = True
    fail_fast: bool = False
    retry_failed: bool = True
    max_retries: int = 2


@dataclass
class ReactConfig:
    """ReAct loop configuration."""
    enabled: bool = True
    max_iterations: int = 8
    reflection_enabled: bool = True
    reflection_interval: int = 3
    early_stop_confidence: float = 0.9


@dataclass
class StateConfig:
    """State management configuration."""
    enabled: bool = True
    persist_state: bool = True
    checkpoint_interval: int = 3
    max_state_size_mb: int = 10
    context_max_tokens: int = 8000


@dataclass
class MemoryConfig:
    """Memory configuration."""
    working_memory_max: int = 20
    episodic_memory_max: int = 50
    relevance_decay: float = 0.9


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    enabled: bool = True
    track_latency: bool = True
    track_tool_usage: bool = True
    track_iterations: bool = True


@dataclass
class DebugConfig:
    """Debug configuration."""
    enabled: bool = False
    log_plans: bool = True
    log_tool_calls: bool = True
    log_state_changes: bool = True


# ============================================================================
# Core Data Classes
# ============================================================================


@dataclass
class ToolCall:
    """Represents a tool invocation."""
    tool_id: str
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "result_preview": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


@dataclass
class AgentTask:
    """Single task in an execution plan."""
    task_id: str
    task_type: TaskType
    description: str
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # task_ids this depends on
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    priority: int = 0
    retries: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def execution_time_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0
    
    @property
    def can_execute(self) -> bool:
        """Check if task can be executed (no pending dependencies)."""
        return self.status == TaskStatus.PENDING and len(self.dependencies) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "description": self.description,
            "tool_name": self.tool_name,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "priority": self.priority,
            "result_preview": str(self.result)[:200] if self.result else None,
            "error": self.error,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


@dataclass
class AgentPlan:
    """Execution plan with tasks."""
    plan_id: str
    query: str
    tasks: List[AgentTask] = field(default_factory=list)
    execution_order: List[List[str]] = field(default_factory=list)  # Batches of parallel tasks
    reasoning: str = ""
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_ready_tasks(self, completed_ids: Set[str]) -> List[AgentTask]:
        """Get tasks ready to execute (dependencies satisfied)."""
        ready = []
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in task.dependencies):
                ready.append(task)
        return sorted(ready, key=lambda t: t.priority, reverse=True)
    
    def get_parallel_batches(self) -> List[List[AgentTask]]:
        """Get tasks organized into parallel execution batches."""
        if self.execution_order:
            batches = []
            task_map = {t.task_id: t for t in self.tasks}
            for batch_ids in self.execution_order:
                batch = [task_map[tid] for tid in batch_ids if tid in task_map]
                if batch:
                    batches.append(batch)
            return batches
        
        # Compute from dependencies
        return self._compute_parallel_batches()
    
    def _compute_parallel_batches(self) -> List[List[AgentTask]]:
        """Compute parallel batches from task dependencies."""
        batches = []
        completed = set()
        remaining = list(self.tasks)
        
        while remaining:
            batch = []
            for task in remaining:
                if all(dep in completed for dep in task.dependencies):
                    batch.append(task)
            
            if not batch:
                # Circular dependency or error
                break
            
            batches.append(batch)
            completed.update(t.task_id for t in batch)
            remaining = [t for t in remaining if t.task_id not in completed]
        
        return batches
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "query": self.query,
            "task_count": len(self.tasks),
            "tasks": [t.to_dict() for t in self.tasks],
            "execution_order": self.execution_order,
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class AgentStep:
    """Single reasoning step."""
    step_id: str
    step_type: StepType
    iteration: int
    content: str
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "iteration": self.iteration,
            "content": self.content[:500] if len(self.content) > 500 else self.content,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "tool_result": self.tool_result.to_dict() if self.tool_result else None,
        }


@dataclass
class AgentState:
    """Complete agent state for an execution."""
    state_id: str
    session_id: str
    query: str
    mode: ExecutionMode = ExecutionMode.REACT
    plan: Optional[AgentPlan] = None
    steps: List[AgentStep] = field(default_factory=list)
    current_iteration: int = 0
    status: TaskStatus = TaskStatus.PENDING
    final_answer: Optional[str] = None
    confidence: float = 0.0
    working_memory: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_step(self, step: AgentStep) -> None:
        """Add a step to the state."""
        self.steps.append(step)
        self.updated_at = datetime.utcnow()
    
    def get_last_steps(self, n: int = 5) -> List[AgentStep]:
        """Get last N steps."""
        return self.steps[-n:] if self.steps else []
    
    def get_tool_results(self) -> List[ToolResult]:
        """Get all tool results."""
        return [s.tool_result for s in self.steps if s.tool_result]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "session_id": self.session_id,
            "query": self.query,
            "mode": self.mode.value,
            "plan": self.plan.to_dict() if self.plan else None,
            "step_count": len(self.steps),
            "current_iteration": self.current_iteration,
            "status": self.status.value,
            "final_answer": self.final_answer,
            "confidence": round(self.confidence, 3),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ExecutionContext:
    """Context for task execution."""
    query: str
    session_id: str
    state: AgentState
    available_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_results: Dict[str, Any] = field(default_factory=dict)  # Results from dependencies
    max_retries: int = 2
    timeout: int = 30


@dataclass
class AgentResult:
    """Final result from agent execution."""
    session_id: str
    query: str
    answer: str
    confidence: float
    mode_used: ExecutionMode
    total_iterations: int
    total_tool_calls: int
    steps: List[AgentStep] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    plan: Optional[AgentPlan] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "mode_used": self.mode_used.value,
            "total_iterations": self.total_iterations,
            "total_tool_calls": self.total_tool_calls,
            "step_count": len(self.steps),
            "source_count": len(self.sources),
            "reasoning_trace": self.reasoning_trace[-5:],  # Last 5
            "execution_time_ms": round(self.execution_time_ms, 2),
            "plan": self.plan.to_dict() if self.plan else None,
        }


# ============================================================================
# Memory Classes
# ============================================================================


@dataclass
class MemoryItem:
    """Single memory item."""
    item_id: str
    content: Any
    relevance: float = 1.0
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)


class WorkingMemory:
    """Short-term working memory for agent."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._items: Dict[str, MemoryItem] = {}
        self._access_order: deque = deque()
    
    def add(self, key: str, content: Any, relevance: float = 1.0) -> None:
        """Add item to working memory."""
        item = MemoryItem(
            item_id=key,
            content=content,
            relevance=relevance,
        )
        self._items[key] = item
        self._access_order.append(key)
        
        # Enforce max size
        while len(self._items) > self.config.working_memory_max:
            oldest = self._access_order.popleft()
            if oldest in self._items:
                del self._items[oldest]
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from memory."""
        item = self._items.get(key)
        if item:
            item.access_count += 1
            item.last_accessed = datetime.utcnow()
            return item.content
        return None
    
    def get_relevant(self, n: int = 5) -> List[Tuple[str, Any]]:
        """Get most relevant items."""
        sorted_items = sorted(
            self._items.values(),
            key=lambda x: x.relevance * (self.config.relevance_decay ** x.access_count),
            reverse=True,
        )
        return [(i.item_id, i.content) for i in sorted_items[:n]]
    
    def decay(self) -> None:
        """Apply relevance decay to all items."""
        for item in self._items.values():
            item.relevance *= self.config.relevance_decay
    
    def clear(self) -> None:
        """Clear working memory."""
        self._items.clear()
        self._access_order.clear()


@dataclass
class Episode:
    """Single episode in episodic memory."""
    episode_id: str
    query: str
    answer: str
    success: bool
    tool_sequence: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class EpisodicMemory:
    """Long-term episodic memory."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._episodes: Dict[str, Episode] = {}
    
    def add_episode(
        self,
        query: str,
        answer: str,
        success: bool,
        tool_sequence: List[str],
    ) -> Episode:
        """Add a new episode."""
        episode = Episode(
            episode_id=str(uuid.uuid4()),
            query=query,
            answer=answer,
            success=success,
            tool_sequence=tool_sequence,
        )
        self._episodes[episode.episode_id] = episode
        
        # Enforce max size
        if len(self._episodes) > self.config.episodic_memory_max:
            oldest_id = min(
                self._episodes.keys(),
                key=lambda k: self._episodes[k].created_at,
            )
            del self._episodes[oldest_id]
        
        return episode
    
    def get_similar(self, query: str, n: int = 3) -> List[Episode]:
        """Get similar past episodes (simple keyword matching)."""
        query_words = set(query.lower().split())
        
        scored = []
        for episode in self._episodes.values():
            episode_words = set(episode.query.lower().split())
            overlap = len(query_words & episode_words)
            if overlap > 0:
                scored.append((episode, overlap))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:n]]
    
    def get_successful_patterns(self) -> Dict[str, int]:
        """Get tool sequences from successful episodes."""
        patterns = defaultdict(int)
        for episode in self._episodes.values():
            if episode.success and episode.tool_sequence:
                pattern = "->".join(episode.tool_sequence)
                patterns[pattern] += 1
        return dict(patterns)


# ============================================================================
# State Manager
# ============================================================================


class StateManager:
    """Manages agent state persistence."""
    
    def __init__(self, config: StateConfig, redis_client: Optional[Any] = None):
        self.config = config
        self._redis = redis_client
        self._local_states: Dict[str, AgentState] = {}
        self._checkpoints: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    async def save_state(self, state: AgentState) -> bool:
        """Save agent state."""
        if not self.config.enabled:
            return False
        
        state.updated_at = datetime.utcnow()
        
        # Local storage
        self._local_states[state.state_id] = state
        
        # Redis storage
        if self._redis and self.config.persist_state:
            try:
                key = f"ubp:agentic:state:{state.state_id}"
                await self._redis.set(
                    key,
                    json.dumps(state.to_dict()),
                    ex=3600,  # 1 hour TTL
                )
            except Exception as e:
                logger.warning(f"Failed to save state to Redis: {e}")
        
        return True
    
    async def load_state(self, state_id: str) -> Optional[AgentState]:
        """Load agent state."""
        # Check local
        if state_id in self._local_states:
            return self._local_states[state_id]
        
        # Check Redis
        if self._redis:
            try:
                key = f"ubp:agentic:state:{state_id}"
                data = await self._redis.get(key)
                if data:
                    # Would need to reconstruct AgentState from dict
                    pass
            except Exception:
                pass
        
        return None
    
    async def checkpoint(self, state: AgentState) -> None:
        """Create a checkpoint of current state."""
        checkpoint_data = {
            "iteration": state.current_iteration,
            "step_count": len(state.steps),
            "timestamp": datetime.utcnow().isoformat(),
            "working_memory_keys": list(state.working_memory.keys()),
        }
        self._checkpoints[state.state_id].append(checkpoint_data)
    
    async def delete_state(self, state_id: str) -> bool:
        """Delete agent state."""
        if state_id in self._local_states:
            del self._local_states[state_id]
        
        if state_id in self._checkpoints:
            del self._checkpoints[state_id]
        
        if self._redis:
            try:
                await self._redis.delete(f"ubp:agentic:state:{state_id}")
            except Exception:
                pass
        
        return True


# ============================================================================
# Execution Metrics
# ============================================================================


class ExecutionMetrics:
    """Collects execution metrics."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self._metrics = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "tool_calls": defaultdict(int),
            "mode_usage": defaultdict(int),
            "latencies": [],
            "iterations": [],
            "parallel_batches": [],
        }
    
    async def record_execution(
        self,
        mode: ExecutionMode,
        success: bool,
        latency_ms: float,
        iterations: int,
        tool_calls: Dict[str, int],
        parallel_batches: int = 0,
    ) -> None:
        """Record execution metrics."""
        if not self.config.enabled:
            return
        
        self._metrics["total_executions"] += 1
        self._metrics["mode_usage"][mode.value] += 1
        
        if success:
            self._metrics["successful"] += 1
        else:
            self._metrics["failed"] += 1
        
        if self.config.track_latency:
            self._metrics["latencies"].append(latency_ms)
            if len(self._metrics["latencies"]) > 1000:
                self._metrics["latencies"] = self._metrics["latencies"][-1000:]
        
        if self.config.track_iterations:
            self._metrics["iterations"].append(iterations)
            self._metrics["parallel_batches"].append(parallel_batches)
        
        if self.config.track_tool_usage:
            for tool, count in tool_calls.items():
                self._metrics["tool_calls"][tool] += count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        latencies = self._metrics["latencies"]
        iterations = self._metrics["iterations"]
        
        return {
            "total_executions": self._metrics["total_executions"],
            "success_rate": (
                self._metrics["successful"] / max(self._metrics["total_executions"], 1)
            ),
            "mode_distribution": dict(self._metrics["mode_usage"]),
            "tool_usage": dict(self._metrics["tool_calls"]),
            "latency_stats": {
                "avg_ms": sum(latencies) / len(latencies) if latencies else 0,
                "min_ms": min(latencies) if latencies else 0,
                "max_ms": max(latencies) if latencies else 0,
            },
            "iteration_stats": {
                "avg": sum(iterations) / len(iterations) if iterations else 0,
                "max": max(iterations) if iterations else 0,
            },
            "avg_parallel_batches": (
                sum(self._metrics["parallel_batches"]) / 
                len(self._metrics["parallel_batches"])
                if self._metrics["parallel_batches"] else 0
            ),
        }
