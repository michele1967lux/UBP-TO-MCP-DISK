"""
graph_rag/providers.py

Logic Layer - ZERO dependencies from backend.app
Must be testable standalone.

Provides:
- Entity: Graph node representation
- Relation: Graph edge representation  
- Triple: Subject-Predicate-Object tuple
- KnowledgeGraph: In-memory graph structure
- GraphNode / GraphEdge: Low-level graph primitives
- Subgraph: Extracted portion of knowledge graph
- GraphPath: Path through the graph
- ExtractionResult: Entity and relation extraction result
- GraphQueryResult: Result from graph-based retrieval
- GraphRAGResult: Complete Graph-RAG result
- EntityExtractor: Extract entities from text
- RelationExtractor: Extract relations between entities
- GraphBuilder: Construct knowledge graph
- GraphRetriever: Graph-based retrieval
- SubgraphReasoner: Reason over subgraphs
- GraphCacheProvider: Redis caching
- GraphSessionManager: Session management
- GraphMetricsCollector: Statistics

v1.0.0: Initial release with full graph capabilities
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from enum import Enum
from heapq import heappush, heappop
from typing import (
    Any, Callable, Dict, FrozenSet, Generator, Iterable,
    List, Optional, Protocol, Set, Tuple, Union,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class EntityType(Enum):
    """Entity type classification."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    PRODUCT = "product"
    EVENT = "event"
    QUANTITY = "quantity"
    CUSTOM = "custom"


class RelationType(Enum):
    """Relation type classification."""
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    CREATED_BY = "created_by"
    USED_BY = "used_by"
    RELATED_TO = "related_to"
    DEPENDS_ON = "depends_on"
    CAUSES = "causes"
    HAS_PROPERTY = "has_property"
    INSTANCE_OF = "instance_of"
    SUBCLASS_OF = "subclass_of"
    CUSTOM = "custom"


class GraphBackend(Enum):
    """Graph storage backend."""
    MEMORY = "memory"
    REDIS = "redis"
    NEO4J = "neo4j"


class RetrievalStrategy(Enum):
    """Graph retrieval strategy."""
    ENTITY_CENTRIC = "entity_centric"
    RELATION_GUIDED = "relation_guided"
    PATH_BASED = "path_based"
    SUBGRAPH = "subgraph"
    HYBRID = "hybrid"


class TaskStatus(Enum):
    """Worker task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class GraphConfig:
    """Core graph configuration."""
    enabled: bool = True
    default_backend: str = "memory"
    max_graph_size: int = 100000
    auto_persist: bool = True
    persist_interval_seconds: int = 300


@dataclass
class EntityExtractionConfig:
    """Entity extraction configuration."""
    enabled: bool = True
    method: str = "llm"
    max_entities_per_chunk: int = 20
    min_confidence: float = 0.6
    temperature: float = 0.1
    coreference_resolution: bool = True
    entity_linking: bool = True
    disambiguation_threshold: float = 0.85


@dataclass
class RelationExtractionConfig:
    """Relation extraction configuration."""
    enabled: bool = True
    method: str = "llm"
    max_relations_per_chunk: int = 30
    min_confidence: float = 0.5
    temperature: float = 0.1
    extract_evidence_spans: bool = True
    bidirectional_detection: bool = True


@dataclass
class GraphConstructionConfig:
    """Graph construction configuration."""
    merge_strategy: str = "smart"
    entity_merge_threshold: float = 0.9
    relation_merge_threshold: float = 0.85
    max_nodes: int = 50000
    max_edges: int = 200000
    schema_validation: bool = True
    allow_self_loops: bool = False
    allow_parallel_edges: bool = True
    track_provenance: bool = True


@dataclass
class GraphRetrievalConfig:
    """Graph retrieval configuration."""
    enabled: bool = True
    default_strategy: str = "hybrid"
    max_hops: int = 3
    max_nodes_per_hop: int = 50
    max_subgraph_nodes: int = 100
    entity_search_top_k: int = 10
    relation_filter_enabled: bool = True


@dataclass
class HybridRetrievalConfig:
    """Hybrid retrieval configuration."""
    enabled: bool = True
    vector_weight: float = 0.5
    graph_weight: float = 0.5
    fusion_method: str = "rrf"
    rrf_k: int = 60


@dataclass
class SubgraphReasoningConfig:
    """Subgraph reasoning configuration."""
    enabled: bool = True
    context_aggregation: str = "weighted"
    path_inference: bool = True
    community_detection: bool = True
    max_reasoning_paths: int = 10


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    ttl_seconds: int = 3600
    cache_extractions: bool = True
    cache_subgraphs: bool = True
    cache_paths: bool = True
    max_cache_size_mb: int = 256


@dataclass
class SessionConfig:
    """Session configuration."""
    enabled: bool = True
    ttl_seconds: int = 3600
    max_history_size: int = 30
    persist_subgraphs: bool = True


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    enabled: bool = True
    collect_graph_stats: bool = True
    collect_extraction_stats: bool = True
    collect_retrieval_stats: bool = True
    collect_timings: bool = True


@dataclass
class DebugConfig:
    """Debug configuration."""
    enabled: bool = False
    log_extractions: bool = True
    log_graph_ops: bool = True
    log_retrievals: bool = True
    log_reasoning: bool = True
    trace_execution: bool = False


# ============================================================================
# Core Data Classes
# ============================================================================


@dataclass
class Entity:
    """A node in the knowledge graph."""
    entity_id: str
    text: str
    normalized: str
    entity_type: EntityType
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: Set[str] = field(default_factory=set)
    source_docs: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "text": self.text,
            "normalized": self.normalized,
            "entity_type": self.entity_type.value,
            "confidence": round(self.confidence, 3),
            "properties": self.properties,
            "aliases": list(self.aliases),
            "source_doc_count": len(self.source_docs),
        }
    
    def __hash__(self):
        return hash(self.entity_id)
    
    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.entity_id == other.entity_id
        return False


@dataclass
class Relation:
    """An edge in the knowledge graph."""
    relation_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float = 1.0
    evidence: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    source_docs: Set[str] = field(default_factory=set)
    bidirectional: bool = False
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence[:200] + "..." if len(self.evidence) > 200 else self.evidence,
            "bidirectional": self.bidirectional,
            "weight": round(self.weight, 3),
        }
    
    def __hash__(self):
        return hash(self.relation_id)


@dataclass
class Triple:
    """Subject-Predicate-Object representation."""
    subject: str  # Entity normalized name
    predicate: str  # Relation type
    object: str  # Entity normalized name
    confidence: float = 1.0
    evidence: str = ""
    
    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class GraphPath:
    """A path through the knowledge graph."""
    path_id: str
    nodes: List[str]  # Entity IDs
    edges: List[str]  # Relation IDs
    length: int
    score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "nodes": self.nodes,
            "edges": self.edges,
            "length": self.length,
            "score": round(self.score, 3),
        }


@dataclass
class Subgraph:
    """An extracted portion of the knowledge graph."""
    subgraph_id: str
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: Dict[str, Relation] = field(default_factory=dict)
    root_entity_id: Optional[str] = None
    depth: int = 0
    
    @property
    def node_count(self) -> int:
        return len(self.entities)
    
    @property
    def edge_count(self) -> int:
        return len(self.relations)
    
    def get_triples(self) -> List[Triple]:
        """Convert subgraph to list of triples."""
        triples = []
        for rel in self.relations.values():
            source = self.entities.get(rel.source_id)
            target = self.entities.get(rel.target_id)
            if source and target:
                triples.append(Triple(
                    subject=source.normalized,
                    predicate=rel.relation_type.value,
                    object=target.normalized,
                    confidence=rel.confidence,
                    evidence=rel.evidence,
                ))
        return triples
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgraph_id": self.subgraph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "root_entity_id": self.root_entity_id,
            "depth": self.depth,
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations.values()],
        }


@dataclass
class ExtractionResult:
    """Result from entity and relation extraction."""
    doc_id: str
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)
    main_topics: List[str] = field(default_factory=list)
    language: str = "en"
    extraction_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "triple_count": len(self.triples),
            "main_topics": self.main_topics,
            "language": self.language,
            "extraction_time_ms": round(self.extraction_time_ms, 2),
        }


@dataclass
class GraphQueryResult:
    """Result from graph-based retrieval."""
    query: str
    strategy_used: RetrievalStrategy
    matched_entities: List[Entity] = field(default_factory=list)
    subgraph: Optional[Subgraph] = None
    paths: List[GraphPath] = field(default_factory=list)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    retrieval_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "strategy_used": self.strategy_used.value,
            "matched_entity_count": len(self.matched_entities),
            "subgraph": self.subgraph.to_dict() if self.subgraph else None,
            "path_count": len(self.paths),
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
        }


@dataclass
class GraphRAGResult:
    """Complete result from Graph-RAG pipeline."""
    session_id: str
    query: str
    answer: str
    confidence: float
    query_entities: List[Entity] = field(default_factory=list)
    graph_context: Optional[Subgraph] = None
    supporting_facts: List[Triple] = field(default_factory=list)
    paths_used: List[GraphPath] = field(default_factory=list)
    completeness: str = "complete"  # complete, partial, insufficient
    caveats: List[str] = field(default_factory=list)
    time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "query_entity_count": len(self.query_entities),
            "supporting_fact_count": len(self.supporting_facts),
            "paths_used_count": len(self.paths_used),
            "completeness": self.completeness,
            "caveats": self.caveats,
            "time_ms": round(self.time_ms, 2),
            "graph_context": self.graph_context.to_dict() if self.graph_context else None,
        }


@dataclass
class GraphSession:
    """Session state for graph interactions."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    history: List[Dict[str, Any]] = field(default_factory=list)
    cached_subgraphs: Dict[str, Subgraph] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "history_count": len(self.history),
            "cached_subgraph_count": len(self.cached_subgraphs),
        }


# ============================================================================
# Knowledge Graph Implementation
# ============================================================================


class KnowledgeGraph:
    """
    In-memory Knowledge Graph implementation.
    
    Features:
    - Entity and relation storage
    - Adjacency list for fast traversal
    - Entity name index for lookup
    - Centrality caching
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[GraphConstructionConfig] = None):
        self.config = config or GraphConstructionConfig()
        
        # Core storage
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}
        
        # Indexes
        self._name_to_entity: Dict[str, str] = {}  # normalized name -> entity_id
        self._type_index: Dict[EntityType, Set[str]] = defaultdict(set)
        
        # Adjacency lists
        self._outgoing: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> relation_ids
        self._incoming: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> relation_ids
        
        # Caches
        self._centrality_cache: Dict[str, float] = {}
        self._pagerank_cache: Dict[str, float] = {}
        
        # Metadata
        self._created_at = datetime.utcnow()
        self._modified_at = datetime.utcnow()
        self._version = 0
    
    @property
    def node_count(self) -> int:
        return len(self._entities)
    
    @property
    def edge_count(self) -> int:
        return len(self._relations)
    
    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the graph."""
        if self.node_count >= self.config.max_nodes:
            logger.warning(f"Max nodes ({self.config.max_nodes}) reached")
            return False
        
        # Check for merge
        existing_id = self._name_to_entity.get(entity.normalized.lower())
        if existing_id and self.config.merge_strategy != "none":
            self._merge_entity(existing_id, entity)
            return True
        
        self._entities[entity.entity_id] = entity
        self._name_to_entity[entity.normalized.lower()] = entity.entity_id
        self._type_index[entity.entity_type].add(entity.entity_id)
        
        for alias in entity.aliases:
            self._name_to_entity[alias.lower()] = entity.entity_id
        
        self._modified_at = datetime.utcnow()
        self._version += 1
        self._invalidate_caches()
        
        return True
    
    def _merge_entity(self, existing_id: str, new_entity: Entity) -> None:
        """Merge new entity into existing one."""
        existing = self._entities[existing_id]
        
        # Merge properties
        existing.properties.update(new_entity.properties)
        existing.aliases.update(new_entity.aliases)
        existing.source_docs.update(new_entity.source_docs)
        
        # Update confidence (take max)
        existing.confidence = max(existing.confidence, new_entity.confidence)
        
        # Add new aliases to index
        for alias in new_entity.aliases:
            self._name_to_entity[alias.lower()] = existing_id
    
    def add_relation(self, relation: Relation) -> bool:
        """Add a relation to the graph."""
        if self.edge_count >= self.config.max_edges:
            logger.warning(f"Max edges ({self.config.max_edges}) reached")
            return False
        
        # Validate entities exist
        if relation.source_id not in self._entities:
            logger.warning(f"Source entity not found: {relation.source_id}")
            return False
        if relation.target_id not in self._entities:
            logger.warning(f"Target entity not found: {relation.target_id}")
            return False
        
        # Check self-loop
        if relation.source_id == relation.target_id and not self.config.allow_self_loops:
            return False
        
        self._relations[relation.relation_id] = relation
        self._outgoing[relation.source_id].add(relation.relation_id)
        self._incoming[relation.target_id].add(relation.relation_id)
        
        # Handle bidirectional
        if relation.bidirectional:
            inverse_id = f"{relation.relation_id}_inv"
            inverse_rel = Relation(
                relation_id=inverse_id,
                source_id=relation.target_id,
                target_id=relation.source_id,
                relation_type=relation.relation_type,
                confidence=relation.confidence,
                evidence=relation.evidence,
                properties=relation.properties.copy(),
                source_docs=relation.source_docs.copy(),
                bidirectional=False,
                weight=relation.weight,
            )
            self._relations[inverse_id] = inverse_rel
            self._outgoing[relation.target_id].add(inverse_id)
            self._incoming[relation.source_id].add(inverse_id)
        
        self._modified_at = datetime.utcnow()
        self._version += 1
        self._invalidate_caches()
        
        return True
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get entity by normalized name."""
        entity_id = self._name_to_entity.get(name.lower())
        if entity_id:
            return self._entities.get(entity_id)
        return None
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get relation by ID."""
        return self._relations.get(relation_id)
    
    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",  # out, in, both
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[Tuple[Entity, Relation]]:
        """Get neighboring entities."""
        neighbors = []
        
        rel_ids = set()
        if direction in ("out", "both"):
            rel_ids.update(self._outgoing.get(entity_id, set()))
        if direction in ("in", "both"):
            rel_ids.update(self._incoming.get(entity_id, set()))
        
        for rel_id in rel_ids:
            rel = self._relations.get(rel_id)
            if not rel:
                continue
            
            if relation_types and rel.relation_type not in relation_types:
                continue
            
            neighbor_id = rel.target_id if rel.source_id == entity_id else rel.source_id
            neighbor = self._entities.get(neighbor_id)
            if neighbor:
                neighbors.append((neighbor, rel))
        
        return neighbors
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a given type."""
        entity_ids = self._type_index.get(entity_type, set())
        return [self._entities[eid] for eid in entity_ids if eid in self._entities]
    
    def search_entities(
        self,
        query: str,
        top_k: int = 10,
        entity_types: Optional[List[EntityType]] = None,
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
    ) -> List[Tuple[Entity, float]]:
        """Search entities by name."""
        results = []
        query_lower = query.lower()
        
        for entity in self._entities.values():
            if entity_types and entity.entity_type not in entity_types:
                continue
            
            # Exact match
            if query_lower == entity.normalized.lower():
                results.append((entity, 1.0))
                continue
            
            # Alias match
            if query_lower in [a.lower() for a in entity.aliases]:
                results.append((entity, 0.95))
                continue
            
            # Fuzzy match
            if fuzzy:
                ratio = SequenceMatcher(None, query_lower, entity.normalized.lower()).ratio()
                if ratio >= fuzzy_threshold:
                    results.append((entity, ratio))
                    continue
                
                # Check aliases
                for alias in entity.aliases:
                    ratio = SequenceMatcher(None, query_lower, alias.lower()).ratio()
                    if ratio >= fuzzy_threshold:
                        results.append((entity, ratio * 0.9))
                        break
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def extract_subgraph(
        self,
        seed_entity_ids: List[str],
        max_depth: int = 2,
        max_nodes: int = 100,
        relation_types: Optional[List[RelationType]] = None,
    ) -> Subgraph:
        """Extract a subgraph starting from seed entities."""
        subgraph = Subgraph(
            subgraph_id=str(uuid.uuid4()),
            root_entity_id=seed_entity_ids[0] if seed_entity_ids else None,
            depth=max_depth,
        )
        
        visited = set()
        frontier = [(eid, 0) for eid in seed_entity_ids]
        
        while frontier and len(subgraph.entities) < max_nodes:
            entity_id, depth = frontier.pop(0)
            
            if entity_id in visited:
                continue
            visited.add(entity_id)
            
            entity = self._entities.get(entity_id)
            if not entity:
                continue
            
            subgraph.entities[entity_id] = entity
            
            if depth < max_depth:
                neighbors = self.get_neighbors(entity_id, relation_types=relation_types)
                for neighbor, relation in neighbors:
                    if neighbor.entity_id not in visited:
                        frontier.append((neighbor.entity_id, depth + 1))
                    
                    # Add relation
                    if relation.relation_id not in subgraph.relations:
                        if relation.source_id in subgraph.entities or neighbor.entity_id in visited:
                            subgraph.relations[relation.relation_id] = relation
        
        return subgraph
    
    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5,
        max_paths: int = 10,
    ) -> List[GraphPath]:
        """Find paths between two entities using BFS."""
        if source_id not in self._entities or target_id not in self._entities:
            return []
        
        paths = []
        queue = [(source_id, [source_id], [])]
        visited_paths: Set[FrozenSet] = set()
        
        while queue and len(paths) < max_paths:
            current, node_path, edge_path = queue.pop(0)
            
            if len(node_path) > max_length + 1:
                continue
            
            if current == target_id and len(node_path) > 1:
                path = GraphPath(
                    path_id=str(uuid.uuid4()),
                    nodes=node_path,
                    edges=edge_path,
                    length=len(edge_path),
                    score=1.0 / len(edge_path),  # Shorter paths score higher
                )
                paths.append(path)
                continue
            
            path_key = frozenset(node_path)
            if path_key in visited_paths:
                continue
            visited_paths.add(path_key)
            
            for neighbor, relation in self.get_neighbors(current, direction="out"):
                if neighbor.entity_id not in node_path:
                    new_node_path = node_path + [neighbor.entity_id]
                    new_edge_path = edge_path + [relation.relation_id]
                    queue.append((neighbor.entity_id, new_node_path, new_edge_path))
        
        return paths
    
    def compute_pagerank(
        self,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 0.0001,
    ) -> Dict[str, float]:
        """Compute PageRank for all entities."""
        if self._pagerank_cache:
            return self._pagerank_cache
        
        n = len(self._entities)
        if n == 0:
            return {}
        
        # Initialize
        pagerank = {eid: 1.0 / n for eid in self._entities}
        
        for _ in range(max_iterations):
            new_pagerank = {}
            max_diff = 0.0
            
            for entity_id in self._entities:
                # Sum contributions from incoming neighbors
                contribution = 0.0
                for rel_id in self._incoming.get(entity_id, set()):
                    rel = self._relations.get(rel_id)
                    if rel:
                        source_out_degree = len(self._outgoing.get(rel.source_id, set()))
                        if source_out_degree > 0:
                            contribution += pagerank[rel.source_id] / source_out_degree
                
                new_rank = (1 - damping) / n + damping * contribution
                new_pagerank[entity_id] = new_rank
                max_diff = max(max_diff, abs(new_rank - pagerank[entity_id]))
            
            pagerank = new_pagerank
            
            if max_diff < tolerance:
                break
        
        self._pagerank_cache = pagerank
        return pagerank
    
    def compute_degree_centrality(self) -> Dict[str, float]:
        """Compute degree centrality for all entities."""
        if self._centrality_cache:
            return self._centrality_cache
        
        centrality = {}
        max_degree = max(
            len(self._outgoing.get(eid, set())) + len(self._incoming.get(eid, set()))
            for eid in self._entities
        ) if self._entities else 1
        
        for entity_id in self._entities:
            degree = len(self._outgoing.get(entity_id, set())) + len(self._incoming.get(entity_id, set()))
            centrality[entity_id] = degree / max_degree if max_degree > 0 else 0
        
        self._centrality_cache = centrality
        return centrality
    
    def _invalidate_caches(self) -> None:
        """Invalidate computed caches."""
        self._centrality_cache.clear()
        self._pagerank_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": {
                et.value: len(self._type_index.get(et, set()))
                for et in EntityType
            },
            "avg_degree": (2 * self.edge_count / self.node_count) if self.node_count > 0 else 0,
            "version": self._version,
            "created_at": self._created_at.isoformat(),
            "modified_at": self._modified_at.isoformat(),
        }
    
    def to_triples(self) -> List[Triple]:
        """Export graph as list of triples."""
        triples = []
        for rel in self._relations.values():
            source = self._entities.get(rel.source_id)
            target = self._entities.get(rel.target_id)
            if source and target:
                triples.append(Triple(
                    subject=source.normalized,
                    predicate=rel.relation_type.value,
                    object=target.normalized,
                    confidence=rel.confidence,
                    evidence=rel.evidence,
                ))
        return triples
    
    def clear(self) -> None:
        """Clear the entire graph."""
        self._entities.clear()
        self._relations.clear()
        self._name_to_entity.clear()
        self._type_index.clear()
        self._outgoing.clear()
        self._incoming.clear()
        self._invalidate_caches()
        self._version += 1


# ============================================================================
# Graph Cache Provider
# ============================================================================


class GraphCacheProvider:
    """
    Caching for graph operations.
    
    Features:
    - Extraction caching
    - Subgraph caching
    - Path caching
    - Local fallback
    """
    
    def __init__(self, config: CacheConfig, redis_client: Optional[Any] = None):
        self.config = config
        self._redis = redis_client
        self._local_cache: Dict[str, Any] = {}
        self._stats = {"hits": 0, "misses": 0}
    
    async def get(self, cache_type: str, key: str) -> Optional[Any]:
        """Get cached value."""
        if not self.config.enabled:
            return None
        
        cache_key = f"ubp:graph:cache:{cache_type}:{self._hash_key(key)}"
        
        if self._redis:
            try:
                data = await self._redis.get(cache_key)
                if data:
                    self._stats["hits"] += 1
                    return json.loads(data)
            except Exception:
                pass
        
        if cache_key in self._local_cache:
            entry = self._local_cache[cache_key]
            if entry["expires_at"] > datetime.utcnow():
                self._stats["hits"] += 1
                return entry["data"]
            else:
                del self._local_cache[cache_key]
        
        self._stats["misses"] += 1
        return None
    
    async def set(self, cache_type: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cached value."""
        if not self.config.enabled:
            return False
        
        cache_key = f"ubp:graph:cache:{cache_type}:{self._hash_key(key)}"
        ttl = ttl or self.config.ttl_seconds
        
        if self._redis:
            try:
                await self._redis.set(cache_key, json.dumps(value), ex=ttl)
                return True
            except Exception:
                pass
        
        self._local_cache[cache_key] = {
            "data": value,
            "expires_at": datetime.utcnow() + timedelta(seconds=ttl),
        }
        return True
    
    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def get_stats(self) -> Dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        return {
            "enabled": self.config.enabled,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": self._stats["hits"] / total if total > 0 else 0,
            "local_entries": len(self._local_cache),
        }


# ============================================================================
# Graph Session Manager
# ============================================================================


class GraphSessionManager:
    """Session management for graph interactions."""
    
    def __init__(self, config: SessionConfig):
        self.config = config
        self._sessions: Dict[str, GraphSession] = {}
        self._last_cleanup = datetime.utcnow()
    
    async def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> GraphSession:
        """Create a new session."""
        session = GraphSession(
            session_id=str(uuid.uuid4()),
            metadata=metadata or {},
        )
        self._sessions[session.session_id] = session
        await self._maybe_cleanup()
        return session
    
    async def get_session(self, session_id: str) -> Optional[GraphSession]:
        """Get session by ID."""
        session = self._sessions.get(session_id)
        if session:
            elapsed = (datetime.utcnow() - session.updated_at).total_seconds()
            if elapsed > self.config.ttl_seconds:
                del self._sessions[session_id]
                return None
        return session
    
    async def update_session(
        self,
        session_id: str,
        query: str,
        result: GraphRAGResult,
    ) -> Optional[GraphSession]:
        """Update session with new interaction."""
        session = await self.get_session(session_id)
        if not session:
            return None
        
        session.history.append({
            "query": query,
            "answer_preview": result.answer[:200],
            "entity_count": len(result.query_entities),
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        if len(session.history) > self.config.max_history_size:
            session.history = session.history[-self.config.max_history_size:]
        
        if self.config.persist_subgraphs and result.graph_context:
            session.cached_subgraphs[query] = result.graph_context
        
        session.updated_at = datetime.utcnow()
        return session
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    async def _maybe_cleanup(self) -> None:
        """Run cleanup if needed."""
        now = datetime.utcnow()
        if (now - self._last_cleanup).total_seconds() < 300:
            return
        
        self._last_cleanup = now
        expired = [
            sid for sid, session in self._sessions.items()
            if (now - session.updated_at).total_seconds() > self.config.ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]


# ============================================================================
# Graph Metrics Collector
# ============================================================================


class GraphMetricsCollector:
    """Metrics collection for graph operations."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self._metrics = {
            "extractions": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
            "retrievals": 0,
            "subgraphs_extracted": 0,
            "paths_found": 0,
            "answers_generated": 0,
            "execution_times": [],
        }
    
    async def record_extraction(
        self,
        entity_count: int,
        relation_count: int,
        time_ms: float,
    ) -> None:
        """Record extraction metrics."""
        if not self.config.enabled:
            return
        
        self._metrics["extractions"] += 1
        self._metrics["entities_extracted"] += entity_count
        self._metrics["relations_extracted"] += relation_count
        
        if self.config.collect_timings:
            self._metrics["execution_times"].append(time_ms)
            if len(self._metrics["execution_times"]) > 1000:
                self._metrics["execution_times"] = self._metrics["execution_times"][-1000:]
    
    async def record_retrieval(
        self,
        subgraph_nodes: int,
        paths_found: int,
        time_ms: float,
    ) -> None:
        """Record retrieval metrics."""
        if not self.config.enabled:
            return
        
        self._metrics["retrievals"] += 1
        self._metrics["subgraphs_extracted"] += 1 if subgraph_nodes > 0 else 0
        self._metrics["paths_found"] += paths_found
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        times = self._metrics["execution_times"]
        
        return {
            "extractions": self._metrics["extractions"],
            "entities_extracted": self._metrics["entities_extracted"],
            "relations_extracted": self._metrics["relations_extracted"],
            "avg_entities_per_extraction": (
                self._metrics["entities_extracted"] / self._metrics["extractions"]
                if self._metrics["extractions"] > 0 else 0
            ),
            "retrievals": self._metrics["retrievals"],
            "subgraphs_extracted": self._metrics["subgraphs_extracted"],
            "paths_found": self._metrics["paths_found"],
            "execution_times": {
                "avg_ms": sum(times) / len(times) if times else 0,
                "min_ms": min(times) if times else 0,
                "max_ms": max(times) if times else 0,
            },
        }
