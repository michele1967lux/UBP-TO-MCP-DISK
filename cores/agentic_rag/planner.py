"""
agentic_rag/planner.py

Query decomposition and execution planning.

Features:
- Query complexity analysis
- Task decomposition
- Dependency graph construction
- Parallel batch scheduling
- Adaptive replanning

v1.0.0: Initial release
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple

from .providers import (
    AgentTask,
    AgentPlan,
    TaskType,
    TaskStatus,
    PlanningConfig,
)
from .tools import ToolSchema, ToolRegistry

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class IModuleRegistry(Protocol):
    """Protocol for module registry."""
    def get_module(self, module_name: str) -> Optional[Any]: ...


# ============================================================================
# Planning Prompts
# ============================================================================


DECOMPOSITION_PROMPT_EN = """Analyze this query and create an execution plan.

Query: {query}

Available tools:
{tools}

Instructions:
1. Break down the query into subtasks
2. Identify which tools are needed for each subtask
3. Determine dependencies between tasks
4. Tasks with no dependencies can run in PARALLEL
5. Return a structured plan

Respond in JSON format:
{{
    "reasoning": "<your analysis of the query>",
    "tasks": [
        {{
            "id": "task_1",
            "type": "<retrieval|reasoning|tool_call|synthesis|verification>",
            "description": "<what this task does>",
            "tool": "<tool name or null>",
            "arguments": {{}},
            "dependencies": [],
            "can_parallelize": true
        }}
    ],
    "parallel_batches": [
        ["task_1", "task_2"],
        ["task_3"]
    ],
    "confidence": <0.0-1.0>
}}"""


DECOMPOSITION_PROMPT_IT = """Analizza questa query e crea un piano di esecuzione.

Query: {query}

Tool disponibili:
{tools}

Istruzioni:
1. Scomponi la query in sottotask
2. Identifica quali tool sono necessari per ogni sottotask
3. Determina le dipendenze tra i task
4. I task senza dipendenze possono essere eseguiti in PARALLELO
5. Restituisci un piano strutturato

Rispondi in formato JSON:
{{
    "reasoning": "<la tua analisi della query>",
    "tasks": [
        {{
            "id": "task_1",
            "type": "<retrieval|reasoning|tool_call|synthesis|verification>",
            "description": "<cosa fa questo task>",
            "tool": "<nome tool o null>",
            "arguments": {{}},
            "dependencies": [],
            "can_parallelize": true
        }}
    ],
    "parallel_batches": [
        ["task_1", "task_2"],
        ["task_3"]
    ],
    "confidence": <0.0-1.0>
}}"""


REPLAN_PROMPT = """The previous plan failed. Create a revised plan.

Original query: {query}
Previous plan reasoning: {previous_reasoning}
Failed tasks: {failed_tasks}
Successful results so far: {successful_results}

Create a new plan that:
1. Avoids the same failures
2. Uses alternative approaches if available
3. Builds on successful results

Respond in the same JSON format as before."""


# ============================================================================
# Query Analyzer
# ============================================================================


class QueryAnalyzer:
    """Analyzes query complexity and characteristics."""
    
    def __init__(self):
        self._complexity_patterns = {
            "simple": [
                r"^what is\s+",
                r"^who is\s+",
                r"^when\s+",
                r"^where\s+",
                r"^define\s+",
            ],
            "medium": [
                r"\band\b.*\band\b",
                r"compare|versus|vs\.",
                r"explain.*how",
                r"list.*all",
                r"summarize",
            ],
            "complex": [
                r"analyze.*impact",
                r"what.*if",
                r"how.*would.*change",
                r"compare.*and.*contrast",
                r"evaluate.*and.*recommend",
                r"step.by.step",
            ],
        }
        
        self._tool_indicators = {
            "retrieval": ["find", "search", "look up", "what does", "information about"],
            "calculator": ["calculate", "compute", "how much", "sum", "average", "percentage"],
            "summarizer": ["summarize", "summarise", "brief", "tldr", "main points"],
            "graph_query": ["related to", "connection between", "relationship", "linked to"],
            "web_search": ["latest", "current", "recent news", "today"],
        }
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics."""
        query_lower = query.lower()
        
        # Determine complexity
        complexity = "simple"
        for level in ["complex", "medium"]:
            patterns = self._complexity_patterns[level]
            if any(re.search(p, query_lower) for p in patterns):
                complexity = level
                break
        
        # Identify likely tools
        suggested_tools = []
        for tool, indicators in self._tool_indicators.items():
            if any(ind in query_lower for ind in indicators):
                suggested_tools.append(tool)
        
        if not suggested_tools:
            suggested_tools = ["retrieval"]  # Default
        
        # Estimate subtask count
        word_count = len(query.split())
        question_count = query.count("?")
        conjunction_count = query_lower.count(" and ") + query_lower.count(" or ")
        
        estimated_subtasks = max(1, min(8, 
            1 + question_count + conjunction_count + (1 if word_count > 20 else 0)
        ))
        
        # Can parallelize?
        can_parallelize = complexity != "simple" and estimated_subtasks > 1
        
        return {
            "complexity": complexity,
            "suggested_tools": suggested_tools,
            "estimated_subtasks": estimated_subtasks,
            "can_parallelize": can_parallelize,
            "word_count": word_count,
            "question_count": question_count,
        }


# ============================================================================
# Task Graph Builder
# ============================================================================


class TaskGraphBuilder:
    """Builds and optimizes task dependency graphs."""
    
    def build_graph(self, tasks: List[AgentTask]) -> Dict[str, Set[str]]:
        """Build dependency graph from tasks."""
        graph = {task.task_id: set(task.dependencies) for task in tasks}
        return graph
    
    def topological_sort(self, tasks: List[AgentTask]) -> List[List[str]]:
        """Sort tasks into parallel execution batches."""
        graph = self.build_graph(tasks)
        task_map = {t.task_id: t for t in tasks}
        
        batches = []
        completed = set()
        remaining = set(graph.keys())
        
        while remaining:
            # Find tasks with all dependencies satisfied
            ready = []
            for task_id in remaining:
                deps = graph[task_id]
                if deps.issubset(completed):
                    ready.append(task_id)
            
            if not ready:
                # Circular dependency - break by taking one
                logger.warning("Circular dependency detected in task graph")
                ready = [list(remaining)[0]]
            
            batches.append(ready)
            completed.update(ready)
            remaining -= set(ready)
        
        return batches
    
    def optimize_batches(
        self,
        batches: List[List[str]],
        tasks: List[AgentTask],
        max_batch_size: int = 5,
    ) -> List[List[str]]:
        """Optimize batch sizes for parallel execution."""
        task_map = {t.task_id: t for t in tasks}
        optimized = []
        
        for batch in batches:
            # Split large batches
            if len(batch) <= max_batch_size:
                optimized.append(batch)
            else:
                for i in range(0, len(batch), max_batch_size):
                    optimized.append(batch[i:i + max_batch_size])
        
        return optimized


# ============================================================================
# Planner
# ============================================================================


class AgentPlanner:
    """Plans execution strategy for queries."""
    
    def __init__(
        self,
        config: PlanningConfig,
        tool_registry: ToolRegistry,
        module_registry: IModuleRegistry,
        llm_caller: Optional[Callable] = None,
    ):
        self.config = config
        self.tool_registry = tool_registry
        self._module_registry = module_registry
        self._llm_caller = llm_caller
        self._analyzer = QueryAnalyzer()
        self._graph_builder = TaskGraphBuilder()
        self._plan_cache: Dict[str, AgentPlan] = {}
    
    async def create_plan(
        self,
        query: str,
        language: str = "en",
        force_simple: bool = False,
    ) -> AgentPlan:
        """Create an execution plan for the query."""
        plan_id = str(uuid.uuid4())
        
        # Analyze query
        analysis = self._analyzer.analyze(query)
        
        # Simple queries don't need decomposition
        if force_simple or analysis["complexity"] == "simple":
            return self._create_simple_plan(plan_id, query, analysis)
        
        # Try LLM decomposition
        if self.config.enabled:
            try:
                plan = await self._llm_decompose(plan_id, query, language, analysis)
                if plan and plan.tasks:
                    return plan
            except Exception as e:
                logger.warning(f"LLM decomposition failed: {e}")
        
        # Fallback to heuristic decomposition
        return self._heuristic_decompose(plan_id, query, analysis)
    
    def _create_simple_plan(
        self,
        plan_id: str,
        query: str,
        analysis: Dict[str, Any],
    ) -> AgentPlan:
        """Create a simple single-task plan."""
        tool = analysis["suggested_tools"][0] if analysis["suggested_tools"] else "retrieval"
        
        task = AgentTask(
            task_id=f"{plan_id}_task_1",
            task_type=TaskType.RETRIEVAL if tool == "retrieval" else TaskType.TOOL_CALL,
            description=f"Execute {tool} for: {query}",
            tool_name=tool,
            arguments={"query": query},
        )
        
        synthesis_task = AgentTask(
            task_id=f"{plan_id}_task_2",
            task_type=TaskType.SYNTHESIS,
            description="Synthesize answer from results",
            dependencies=[task.task_id],
        )
        
        return AgentPlan(
            plan_id=plan_id,
            query=query,
            tasks=[task, synthesis_task],
            execution_order=[[task.task_id], [synthesis_task.task_id]],
            reasoning="Simple query - single retrieval + synthesis",
            confidence=0.9,
        )
    
    async def _llm_decompose(
        self,
        plan_id: str,
        query: str,
        language: str,
        analysis: Dict[str, Any],
    ) -> Optional[AgentPlan]:
        """Use LLM for task decomposition via pipeline HA chain."""
        if not self._llm_caller:
            return None

        # Format tools
        tools_desc = self._format_tools_for_prompt()

        # Select prompt
        prompt_template = (
            DECOMPOSITION_PROMPT_IT if language == "it"
            else DECOMPOSITION_PROMPT_EN
        )

        prompt = prompt_template.format(query=query, tools=tools_desc)

        try:
            response = await asyncio.wait_for(
                self._llm_caller(
                    prompt=prompt,
                    max_tokens=2000,
                    temperature=self.config.temperature,
                    purpose="decomposition",
                ),
                timeout=30,
            )
            return self._parse_plan_response(plan_id, query, response)
        except Exception as e:
            logger.error(f"LLM decomposition error: {e}")
            return None
    
    def _parse_plan_response(
        self,
        plan_id: str,
        query: str,
        response: str,
    ) -> Optional[AgentPlan]:
        """Parse LLM response into AgentPlan."""
        try:
            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return None
            
            data = json.loads(json_match.group())
            
            tasks = []
            for t in data.get("tasks", []):
                task_type_str = t.get("type", "tool_call").lower()
                try:
                    task_type = TaskType(task_type_str)
                except ValueError:
                    task_type = TaskType.TOOL_CALL
                
                task = AgentTask(
                    task_id=t.get("id", f"{plan_id}_{len(tasks)}"),
                    task_type=task_type,
                    description=t.get("description", ""),
                    tool_name=t.get("tool"),
                    arguments=t.get("arguments", {}),
                    dependencies=t.get("dependencies", []),
                )
                tasks.append(task)
            
            # Get or compute parallel batches
            parallel_batches = data.get("parallel_batches", [])
            if not parallel_batches:
                parallel_batches = self._graph_builder.topological_sort(tasks)
            
            return AgentPlan(
                plan_id=plan_id,
                query=query,
                tasks=tasks,
                execution_order=parallel_batches,
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.7)),
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            return None
    
    def _heuristic_decompose(
        self,
        plan_id: str,
        query: str,
        analysis: Dict[str, Any],
    ) -> AgentPlan:
        """Heuristic-based task decomposition."""
        tasks = []
        
        # Create retrieval tasks for each suggested tool
        parallel_tasks = []
        for i, tool in enumerate(analysis["suggested_tools"][:3]):  # Max 3 parallel
            task = AgentTask(
                task_id=f"{plan_id}_retrieve_{i}",
                task_type=TaskType.RETRIEVAL if tool == "retrieval" else TaskType.TOOL_CALL,
                description=f"Use {tool} to gather information",
                tool_name=tool,
                arguments={"query": query},
                priority=len(analysis["suggested_tools"]) - i,
            )
            tasks.append(task)
            parallel_tasks.append(task.task_id)
        
        # Reasoning task
        reasoning_task = AgentTask(
            task_id=f"{plan_id}_reason",
            task_type=TaskType.REASONING,
            description="Analyze gathered information",
            dependencies=parallel_tasks,
        )
        tasks.append(reasoning_task)
        
        # Synthesis task
        synthesis_task = AgentTask(
            task_id=f"{plan_id}_synthesize",
            task_type=TaskType.SYNTHESIS,
            description="Generate final answer",
            dependencies=[reasoning_task.task_id],
        )
        tasks.append(synthesis_task)
        
        # Build execution order
        execution_order = [
            parallel_tasks,  # Parallel retrieval
            [reasoning_task.task_id],  # Sequential reasoning
            [synthesis_task.task_id],  # Sequential synthesis
        ]
        
        return AgentPlan(
            plan_id=plan_id,
            query=query,
            tasks=tasks,
            execution_order=execution_order,
            reasoning=f"Heuristic plan: {len(parallel_tasks)} parallel retrievals, then reasoning and synthesis",
            confidence=0.7,
        )
    
    async def replan(
        self,
        original_plan: AgentPlan,
        failed_tasks: List[AgentTask],
        successful_results: Dict[str, Any],
        language: str = "en",
    ) -> Optional[AgentPlan]:
        """Create a revised plan after failures via pipeline HA chain."""
        if not self.config.replan_on_failure:
            return None

        if not self._llm_caller:
            return self._heuristic_replan(original_plan, failed_tasks)

        failed_info = [
            {"id": t.task_id, "description": t.description, "error": t.error}
            for t in failed_tasks
        ]

        prompt = REPLAN_PROMPT.format(
            query=original_plan.query,
            previous_reasoning=original_plan.reasoning,
            failed_tasks=json.dumps(failed_info),
            successful_results=json.dumps(successful_results),
        )

        try:
            response = await self._llm_caller(
                prompt=prompt, max_tokens=2000, temperature=0.3, purpose="replan",
            )
            new_plan = self._parse_plan_response(
                f"replan_{original_plan.plan_id}",
                original_plan.query,
                response,
            )
            return new_plan
        except Exception as e:
            logger.error(f"Replan failed: {e}")
            return self._heuristic_replan(original_plan, failed_tasks)
    
    def _heuristic_replan(
        self,
        original_plan: AgentPlan,
        failed_tasks: List[AgentTask],
    ) -> AgentPlan:
        """Simple heuristic replan."""
        failed_ids = {t.task_id for t in failed_tasks}
        
        # Keep successful tasks, replace failed with simpler versions
        new_tasks = []
        for task in original_plan.tasks:
            if task.task_id not in failed_ids:
                new_tasks.append(task)
            else:
                # Replace with simpler retrieval
                new_task = AgentTask(
                    task_id=f"retry_{task.task_id}",
                    task_type=TaskType.RETRIEVAL,
                    description=f"Simplified retry: {task.description}",
                    tool_name="retrieval",
                    arguments={"query": original_plan.query},
                    dependencies=[],
                )
                new_tasks.append(new_task)
        
        return AgentPlan(
            plan_id=f"replan_{original_plan.plan_id}",
            query=original_plan.query,
            tasks=new_tasks,
            execution_order=self._graph_builder.topological_sort(new_tasks),
            reasoning="Heuristic replan after failure",
            confidence=0.5,
        )
    
    def _format_tools_for_prompt(self) -> str:
        """Format tools for inclusion in prompt."""
        tools = self.tool_registry.get_all_schemas()
        lines = []
        for tool in tools:
            params = ", ".join([
                f"{p.name}: {p.param_type}"
                for p in tool.parameters
                if p.required
            ])
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)
    
