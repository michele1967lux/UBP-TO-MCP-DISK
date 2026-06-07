"""
graph_rag/pipeline.py

Pipeline orchestrator for Graph-RAG operations.
Executes configurable multi-step graph processing.

Steps:
1. parse_query - Parse and analyze query
2. extract_query_entities - Extract entities from query
3. graph_retrieval - Retrieve from knowledge graph
4. subgraph_extraction - Extract relevant subgraph
5. context_aggregation - Aggregate context from subgraph
6. generate_answer - Generate final answer

v1.0.0: Initial release
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .providers import (
    Entity,
    Relation,
    Subgraph,
    GraphPath,
    GraphRAGResult,
    KnowledgeGraph,
    RetrievalStrategy,
    DebugConfig,
)
from .delegation import GraphDelegator
from .prompts import detect_language, EntityType

logger = logging.getLogger(__name__)


# ============================================================================
# Pipeline Configuration
# ============================================================================


@dataclass
class StepConfig:
    """Configuration for a single pipeline step."""
    enabled: bool = True
    timeout: int = 30
    required: bool = False


@dataclass
class PipelineConfig:
    """Configuration for the graph pipeline."""
    default_timeout_seconds: int = 120
    fail_fast: bool = False
    steps: Dict[str, StepConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        default_steps = {
            "parse_query": StepConfig(enabled=True, timeout=5),
            "extract_query_entities": StepConfig(enabled=True, timeout=10),
            "graph_retrieval": StepConfig(enabled=True, timeout=30, required=True),
            "subgraph_extraction": StepConfig(enabled=True, timeout=20),
            "context_aggregation": StepConfig(enabled=True, timeout=10),
            "generate_answer": StepConfig(enabled=True, timeout=30, required=True),
        }
        for step_name, default_config in default_steps.items():
            if step_name not in self.steps:
                self.steps[step_name] = default_config


# ============================================================================
# Pipeline Context
# ============================================================================


@dataclass
class PipelineContext:
    """Context passed through pipeline steps."""
    query: str
    session_id: str
    
    # Input parameters
    language: Optional[str] = None
    max_hops: int = 2
    max_subgraph_nodes: int = 100
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    
    # Step results
    query_analysis: Optional[Dict[str, Any]] = None
    query_entities: List[Entity] = field(default_factory=list)
    matched_graph_entities: List[Entity] = field(default_factory=list)
    subgraph: Optional[Subgraph] = None
    paths: List[GraphPath] = field(default_factory=list)
    reasoning_result: Optional[Dict[str, Any]] = None
    answer_data: Optional[Dict[str, Any]] = None
    
    # Execution metadata
    step_times: Dict[str, float] = field(default_factory=dict)
    step_errors: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# Graph RAG Pipeline
# ============================================================================


class GraphRAGPipeline:
    """
    Orchestrates Graph-RAG operations through configurable steps.
    
    Features:
    - Query entity extraction
    - Graph-based retrieval
    - Subgraph extraction
    - Path-based reasoning
    - Context aggregation
    - Answer generation
    """
    
    AVAILABLE_STEPS = [
        "parse_query",
        "extract_query_entities",
        "graph_retrieval",
        "subgraph_extraction",
        "context_aggregation",
        "generate_answer",
    ]
    
    def __init__(
        self,
        delegator: GraphDelegator,
        knowledge_graph: KnowledgeGraph,
        config: PipelineConfig,
        debug_config: Optional[DebugConfig] = None,
    ):
        self.delegator = delegator
        self.knowledge_graph = knowledge_graph
        self.config = config
        self.debug_config = debug_config or DebugConfig()
        
        self._step_handlers: Dict[str, Callable] = {
            "parse_query": self._step_parse_query,
            "extract_query_entities": self._step_extract_query_entities,
            "graph_retrieval": self._step_graph_retrieval,
            "subgraph_extraction": self._step_subgraph_extraction,
            "context_aggregation": self._step_context_aggregation,
            "generate_answer": self._step_generate_answer,
        }
    
    async def execute(
        self,
        query: str,
        session_id: str,
        language: Optional[str] = None,
        max_hops: int = 2,
        max_subgraph_nodes: int = 100,
        retrieval_strategy: Optional[str] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
    ) -> GraphRAGResult:
        """
        Execute the Graph-RAG pipeline.
        
        Args:
            query: User's query
            session_id: Session identifier
            language: Language override
            max_hops: Maximum graph traversal hops
            max_subgraph_nodes: Maximum nodes in subgraph
            retrieval_strategy: Retrieval strategy override
            pipeline_config: Step configuration override
            ctx: Security context
            
        Returns:
            GraphRAGResult with answer and graph context
        """
        start_time = time.perf_counter()
        
        # Apply config overrides
        if pipeline_config:
            self._apply_config_override(pipeline_config)
        
        # Parse retrieval strategy
        strategy = RetrievalStrategy.HYBRID
        if retrieval_strategy:
            try:
                strategy = RetrievalStrategy(retrieval_strategy)
            except ValueError:
                pass
        
        # Create context
        context = PipelineContext(
            query=query,
            session_id=session_id,
            language=language,
            max_hops=max_hops,
            max_subgraph_nodes=max_subgraph_nodes,
            retrieval_strategy=strategy,
        )
        
        # Execute steps
        for step_name in self.AVAILABLE_STEPS:
            step_config = self.config.steps.get(step_name, StepConfig())
            
            if not step_config.enabled:
                continue
            
            handler = self._step_handlers.get(step_name)
            if not handler:
                continue
            
            step_start = time.perf_counter()
            
            try:
                await asyncio.wait_for(
                    handler(context),
                    timeout=step_config.timeout,
                )
                
            except asyncio.TimeoutError:
                error_msg = f"Step '{step_name}' timed out after {step_config.timeout}s"
                context.step_errors[step_name] = error_msg
                logger.warning(f"[GRAPH] {error_msg}")
                
                if step_config.required or self.config.fail_fast:
                    raise RuntimeError(error_msg)
                    
            except Exception as e:
                error_msg = f"Step '{step_name}' failed: {e}"
                context.step_errors[step_name] = error_msg
                logger.error(f"[GRAPH] {error_msg}")
                
                if step_config.required or self.config.fail_fast:
                    raise
            
            context.step_times[step_name] = (time.perf_counter() - step_start) * 1000
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Build result
        answer_data = context.answer_data or {}
        
        return GraphRAGResult(
            session_id=session_id,
            query=query,
            answer=answer_data.get("answer", "Unable to generate answer."),
            confidence=float(answer_data.get("confidence", 0.0)),
            query_entities=context.query_entities,
            graph_context=context.subgraph,
            supporting_facts=context.subgraph.get_triples() if context.subgraph else [],
            paths_used=context.paths,
            completeness=answer_data.get("completeness", "insufficient"),
            caveats=answer_data.get("caveats", []),
            time_ms=total_time,
        )
    
    def _apply_config_override(self, config_override: Dict[str, Any]) -> None:
        """Apply runtime configuration override."""
        for step_name, step_config in config_override.items():
            if step_name in self.config.steps:
                if isinstance(step_config, dict):
                    if "enabled" in step_config:
                        self.config.steps[step_name].enabled = step_config["enabled"]
                    if "timeout" in step_config:
                        self.config.steps[step_name].timeout = step_config["timeout"]
    
    # ========================================================================
    # Step Handlers
    # ========================================================================
    
    async def _step_parse_query(self, ctx: PipelineContext) -> None:
        """Parse and analyze the query."""
        if not ctx.language:
            ctx.language = detect_language(ctx.query)
        
        if self.debug_config.trace_execution:
            logger.debug(f"[GRAPH] Query language: {ctx.language}")
    
    async def _step_extract_query_entities(self, ctx: PipelineContext) -> None:
        """Extract entities from the query."""
        result = await self.delegator.recognize_query_entities(
            query=ctx.query,
            language=ctx.language or "en",
        )
        
        ctx.query_analysis = result
        
        # Convert to Entity objects
        for item in result.get("query_entities", []):
            entity = Entity(
                entity_id=str(uuid.uuid4()),
                text=item.get("text", ""),
                normalized=item.get("normalized", item.get("text", "")),
                entity_type=item.get("type", EntityType.CONCEPT),
                confidence=1.0 if item.get("is_main_subject") else 0.8,
            )
            ctx.query_entities.append(entity)
        
        if self.debug_config.trace_execution:
            logger.debug(f"[GRAPH] Extracted {len(ctx.query_entities)} query entities")
    
    async def _step_graph_retrieval(self, ctx: PipelineContext) -> None:
        """Retrieve matching entities from knowledge graph."""
        matched = []
        
        for query_entity in ctx.query_entities:
            # Search in graph
            results = self.knowledge_graph.search_entities(
                query=query_entity.normalized,
                top_k=10,
                fuzzy=True,
                fuzzy_threshold=0.7,
            )
            
            for entity, score in results:
                if entity not in matched:
                    matched.append(entity)
        
        # Also search by query terms
        query_terms = ctx.query.split()
        for term in query_terms:
            if len(term) > 3:
                results = self.knowledge_graph.search_entities(
                    query=term,
                    top_k=5,
                    fuzzy=True,
                    fuzzy_threshold=0.8,
                )
                for entity, score in results[:3]:
                    if entity not in matched:
                        matched.append(entity)
        
        ctx.matched_graph_entities = matched[:20]  # Limit matches
        
        if self.debug_config.trace_execution:
            logger.debug(f"[GRAPH] Matched {len(ctx.matched_graph_entities)} graph entities")
    
    async def _step_subgraph_extraction(self, ctx: PipelineContext) -> None:
        """Extract relevant subgraph."""
        if not ctx.matched_graph_entities:
            ctx.subgraph = Subgraph(subgraph_id=str(uuid.uuid4()))
            return
        
        seed_ids = [e.entity_id for e in ctx.matched_graph_entities[:5]]
        
        ctx.subgraph = self.knowledge_graph.extract_subgraph(
            seed_entity_ids=seed_ids,
            max_depth=ctx.max_hops,
            max_nodes=ctx.max_subgraph_nodes,
        )
        
        # Find paths between main entities
        if len(seed_ids) >= 2:
            for i, source_id in enumerate(seed_ids[:3]):
                for target_id in seed_ids[i+1:4]:
                    paths = self.knowledge_graph.find_paths(
                        source_id=source_id,
                        target_id=target_id,
                        max_length=ctx.max_hops + 1,
                        max_paths=3,
                    )
                    ctx.paths.extend(paths)
        
        if self.debug_config.trace_execution:
            logger.debug(
                f"[GRAPH] Subgraph: {ctx.subgraph.node_count} nodes, "
                f"{ctx.subgraph.edge_count} edges, {len(ctx.paths)} paths"
            )
    
    async def _step_context_aggregation(self, ctx: PipelineContext) -> None:
        """Aggregate context from subgraph using reasoning."""
        if not ctx.subgraph or ctx.subgraph.node_count == 0:
            ctx.reasoning_result = {
                "relevant_paths": [],
                "key_facts": [],
                "knowledge_gaps": ["No relevant information found in knowledge graph"],
                "confidence": 0.0,
                "can_answer": False,
            }
            return
        
        ctx.reasoning_result = await self.delegator.reason_over_subgraph(
            query=ctx.query,
            subgraph=ctx.subgraph,
            language=ctx.language or "en",
        )
        
        if self.debug_config.trace_execution:
            logger.debug(
                f"[GRAPH] Reasoning: {len(ctx.reasoning_result.get('key_facts', []))} facts, "
                f"confidence={ctx.reasoning_result.get('confidence', 0):.2f}"
            )
    
    async def _step_generate_answer(self, ctx: PipelineContext) -> None:
        """Generate final answer from graph context."""
        if not ctx.subgraph or not ctx.reasoning_result:
            ctx.answer_data = {
                "answer": "I couldn't find relevant information in the knowledge graph to answer this question.",
                "confidence": 0.0,
                "completeness": "insufficient",
                "caveats": ["No relevant graph context found"],
            }
            return
        
        ctx.answer_data = await self.delegator.generate_answer(
            query=ctx.query,
            subgraph=ctx.subgraph,
            reasoning_result=ctx.reasoning_result,
            language=ctx.language or "en",
        )
    
    # ========================================================================
    # Configuration Management
    # ========================================================================
    
    def get_config(self) -> Dict[str, Any]:
        """Get current pipeline configuration."""
        return {
            "default_timeout_seconds": self.config.default_timeout_seconds,
            "fail_fast": self.config.fail_fast,
            "steps": {
                name: {
                    "enabled": step.enabled,
                    "timeout": step.timeout,
                    "required": step.required,
                }
                for name, step in self.config.steps.items()
            },
            "available_steps": self.AVAILABLE_STEPS,
        }
    
    def update_config(self, steps_config: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update pipeline configuration."""
        for step_update in steps_config:
            step_name = step_update.get("step")
            if step_name and step_name in self.config.steps:
                if "enabled" in step_update:
                    self.config.steps[step_name].enabled = step_update["enabled"]
                if "timeout" in step_update:
                    self.config.steps[step_name].timeout = step_update["timeout"]
        
        return self.get_config()
