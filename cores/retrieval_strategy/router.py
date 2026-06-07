"""
retrieval_strategy/router.py

LLM-based query routing for intelligent strategy selection.

Features:
- Query classification
- Strategy selection
- Index routing
- Skip retrieval detection
- Caching of decisions

v1.0.0: Initial release
"""

from __future__ import annotations

# WARN-CV-001 fix: shared LLM response normalizer
try:
    from ubp_enterprise_hybrid.modules.cores._shared.utils import extract_llm_text as _extract_llm_text
except ImportError:
    _extract_llm_text = None  # type: ignore[assignment]

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol

from .providers import (
    RetrievalStrategy,
    QueryClass,
    RouterDecision,
    RouterConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class IModuleRegistry(Protocol):
    """Protocol for module registry."""
    def get_module(self, module_name: str) -> Optional[Any]: ...


# ============================================================================
# Router Prompts
# ============================================================================


QUERY_CLASSIFICATION_PROMPT_EN = """Classify this search query and recommend the best retrieval strategy.

Query: {query}

Available strategies:
- bm25: Best for keyword-heavy queries, exact matches, technical terms
- vector: Best for semantic/conceptual queries, synonyms, paraphrases
- hybrid: Best for most factual queries, combines keyword and semantic
- hierarchical: Best for complex analytical queries requiring document context
- multi_index: Best for comparative queries across different topics

Available indexes: {indexes}

Classify the query and respond in JSON:
{{
    "query_class": "<factual|analytical|comparative|keyword|semantic>",
    "recommended_strategy": "<bm25|vector|hybrid|hierarchical|multi_index>",
    "relevant_indexes": ["<index1>", "<index2>"],
    "skip_retrieval": <true if query doesn't need retrieval (e.g., greetings)>,
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>",
    "suggested_top_k": <recommended number of results>
}}"""


QUERY_CLASSIFICATION_PROMPT_IT = """Classifica questa query di ricerca e raccomanda la migliore strategia di retrieval.

Query: {query}

Strategie disponibili:
- bm25: Migliore per query con molte parole chiave, match esatti, termini tecnici
- vector: Migliore per query semantiche/concettuali, sinonimi, parafrasi
- hybrid: Migliore per la maggior parte delle query fattuali, combina keyword e semantico
- hierarchical: Migliore per query analitiche complesse che richiedono contesto del documento
- multi_index: Migliore per query comparative su argomenti diversi

Indici disponibili: {indexes}

Classifica la query e rispondi in JSON:
{{
    "query_class": "<factual|analytical|comparative|keyword|semantic>",
    "recommended_strategy": "<bm25|vector|hybrid|hierarchical|multi_index>",
    "relevant_indexes": ["<indice1>", "<indice2>"],
    "skip_retrieval": <true se la query non richiede retrieval (es. saluti)>,
    "confidence": <0.0-1.0>,
    "reasoning": "<breve spiegazione>",
    "suggested_top_k": <numero raccomandato di risultati>
}}"""


# ============================================================================
# Query Patterns for Skip Detection
# ============================================================================


SKIP_PATTERNS = {
    "greeting": [
        r"^(hi|hello|hey|ciao|salve|buongiorno|buonasera)[\s!.]*$",
        r"^(good\s*(morning|afternoon|evening))[\s!.]*$",
    ],
    "clarification": [
        r"^(what|cosa)\s*(do\s*you\s*mean|intendi)[\s?]*$",
        r"^(can\s*you|puoi)\s*(explain|clarify|spiegare)[\s?]*$",
        r"^(i\s*don't|non)\s*understand",
    ],
    "follow_up": [
        r"^(and|e)\s*(what\s*about|che\s*ne\s*dici)",
        r"^(tell\s*me\s*more|dimmi\s*di\s*più)[\s!.]*$",
        r"^(continue|continua)[\s!.]*$",
    ],
    "acknowledgment": [
        r"^(thanks|thank\s*you|grazie)[\s!.]*$",
        r"^(ok|okay|va\s*bene|capito)[\s!.]*$",
        r"^(got\s*it|understood|ho\s*capito)[\s!.]*$",
    ],
}


# ============================================================================
# Query Router
# ============================================================================


class QueryRouter:
    """
    LLM-based query router for intelligent retrieval strategy selection.
    
    Features:
    - Query classification
    - Strategy recommendation
    - Index selection
    - Skip detection
    - Decision caching
    """
    
    def __init__(
        self,
        config: RouterConfig,
        module_registry: IModuleRegistry,
        available_indexes: Optional[List[str]] = None,
    ):
        self.config = config
        self._module_registry = module_registry
        self.available_indexes = available_indexes or ["default"]
        
        self._llm_module: Optional[Any] = None
        self._decision_cache: Dict[str, RouterDecision] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
    
    async def route(
        self,
        query: str,
        language: str = "en",
        force_strategy: Optional[RetrievalStrategy] = None,
    ) -> RouterDecision:
        """
        Route a query to the optimal retrieval strategy.
        
        Args:
            query: User's query
            language: Query language
            force_strategy: Force a specific strategy
        
        Returns:
            RouterDecision with strategy and metadata
        """
        # Check for forced strategy
        if force_strategy:
            return RouterDecision(
                query=query,
                query_class=QueryClass.UNKNOWN,
                selected_strategy=force_strategy,
                selected_indexes=self.available_indexes[:1],
                confidence=1.0,
                reasoning="Forced strategy",
            )
        
        # Check cache
        cache_key = self._cache_key(query)
        if self.config.cache_decisions and cache_key in self._decision_cache:
            cached = self._decision_cache[cache_key]
            timestamp = self._cache_timestamps.get(cache_key)
            if timestamp and (datetime.utcnow() - timestamp).total_seconds() < self.config.decision_cache_ttl:
                return cached
        
        # Check skip patterns first (fast path)
        skip_decision = self._check_skip_patterns(query)
        if skip_decision:
            self._cache_decision(cache_key, skip_decision)
            return skip_decision
        
        # Use LLM for classification
        try:
            decision = await self._llm_classify(query, language)
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, using fallback")
            decision = self._fallback_classification(query)
        
        # Cache and return
        self._cache_decision(cache_key, decision)
        return decision
    
    def _check_skip_patterns(self, query: str) -> Optional[RouterDecision]:
        """Check if query matches skip patterns."""
        query_lower = query.lower().strip()
        
        for pattern_type, patterns in SKIP_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, query_lower, re.IGNORECASE):
                    return RouterDecision(
                        query=query,
                        query_class=QueryClass.UNKNOWN,
                        selected_strategy=RetrievalStrategy.HYBRID,
                        selected_indexes=[],
                        skip_retrieval=True,
                        confidence=0.95,
                        reasoning=f"Query matches {pattern_type} pattern",
                        suggested_top_k=0,
                    )
        
        return None
    
    async def _llm_classify(self, query: str, language: str) -> RouterDecision:
        """Use LLM to classify query."""
        module = await self._get_llm_module()
        if not module:
            return self._fallback_classification(query)
        
        # Select prompt
        prompt_template = (
            QUERY_CLASSIFICATION_PROMPT_IT if language == "it"
            else QUERY_CLASSIFICATION_PROMPT_EN
        )
        
        prompt = prompt_template.format(
            query=query,
            indexes=", ".join(self.available_indexes),
        )
        
        # Call LLM
        operation = getattr(module, self.config.llm_operation, None)
        if not operation:
            return self._fallback_classification(query)
        
        try:
            result = await asyncio.wait_for(
                operation(
                    prompt=prompt,
                    temperature=self.config.temperature,
                    max_tokens=500,
                ),
                timeout=self.config.timeout_seconds,
            )
            
            # WARN-CV-001: shared normalizer
            if _extract_llm_text is not None:
                response = _extract_llm_text(result)
            elif isinstance(result, dict):
                response = result.get("text") or result.get("response") or result.get("content", "")
            else:
                response = str(result)

            return self._parse_classification(query, response)
            
        except asyncio.TimeoutError:
            logger.warning("LLM classification timeout")
            return self._fallback_classification(query)
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return self._fallback_classification(query)
    
    def _parse_classification(self, query: str, response: str) -> RouterDecision:
        """Parse LLM classification response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return self._fallback_classification(query)
            
            # Parse query class
            query_class_str = data.get("query_class", "factual").lower()
            try:
                query_class = QueryClass(query_class_str)
            except ValueError:
                query_class = QueryClass.FACTUAL
            
            # Parse strategy
            strategy_str = data.get("recommended_strategy", "hybrid").lower()
            try:
                strategy = RetrievalStrategy(strategy_str)
            except ValueError:
                strategy = RetrievalStrategy.HYBRID
            
            # Parse indexes
            indexes = data.get("relevant_indexes", self.available_indexes[:1])
            if not indexes:
                indexes = self.available_indexes[:1]
            
            # Filter to available indexes
            indexes = [idx for idx in indexes if idx in self.available_indexes]
            if not indexes:
                indexes = self.available_indexes[:1]
            
            return RouterDecision(
                query=query,
                query_class=query_class,
                selected_strategy=strategy,
                selected_indexes=indexes,
                skip_retrieval=data.get("skip_retrieval", False),
                confidence=float(data.get("confidence", 0.8)),
                reasoning=data.get("reasoning", ""),
                suggested_top_k=int(data.get("suggested_top_k", 10)),
            )
            
        except json.JSONDecodeError:
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> RouterDecision:
        """Heuristic-based fallback classification."""
        query_lower = query.lower()
        
        # Detect query characteristics
        has_question_words = any(
            w in query_lower for w in ["what", "who", "when", "where", "why", "how", "quale", "chi", "quando", "dove", "perché", "come"]
        )
        has_comparison = any(
            w in query_lower for w in ["vs", "versus", "compare", "difference", "between", "confronta", "differenza", "tra"]
        )
        has_technical_terms = any(
            w in query_lower for w in ["api", "code", "function", "error", "bug", "config", "setup", "install"]
        )
        word_count = len(query.split())
        
        # Classify
        if has_comparison:
            query_class = QueryClass.COMPARATIVE
            strategy = RetrievalStrategy.MULTI_INDEX
        elif has_technical_terms and word_count <= 5:
            query_class = QueryClass.KEYWORD
            strategy = RetrievalStrategy.BM25
        elif word_count > 15:
            query_class = QueryClass.ANALYTICAL
            strategy = RetrievalStrategy.HIERARCHICAL
        elif has_question_words:
            query_class = QueryClass.FACTUAL
            strategy = RetrievalStrategy.HYBRID
        else:
            query_class = QueryClass.SEMANTIC
            strategy = RetrievalStrategy.VECTOR
        
        return RouterDecision(
            query=query,
            query_class=query_class,
            selected_strategy=strategy,
            selected_indexes=self.available_indexes[:1],
            confidence=0.6,
            reasoning="Heuristic classification (LLM unavailable)",
            suggested_top_k=10,
        )
    
    async def _get_llm_module(self) -> Optional[Any]:
        """Get or resolve LLM module."""
        if self._llm_module:
            return self._llm_module
        
        module = self._module_registry.get_module(self.config.llm_module)
        if module:
            self._llm_module = module
        return module
    
    def _cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _cache_decision(self, cache_key: str, decision: RouterDecision) -> None:
        """Cache a routing decision."""
        if self.config.cache_decisions:
            self._decision_cache[cache_key] = decision
            self._cache_timestamps[cache_key] = datetime.utcnow()
            
            # Cleanup old entries
            if len(self._decision_cache) > 1000:
                self._cleanup_cache()
    
    def _cleanup_cache(self) -> None:
        """Remove expired cache entries."""
        now = datetime.utcnow()
        expired = [
            key for key, ts in self._cache_timestamps.items()
            if (now - ts).total_seconds() > self.config.decision_cache_ttl
        ]
        for key in expired:
            self._decision_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    def clear_cache(self) -> None:
        """Clear decision cache."""
        self._decision_cache.clear()
        self._cache_timestamps.clear()


# ============================================================================
# Strategy Selector
# ============================================================================


class StrategySelector:
    """
    Rule-based strategy selection based on query analysis.
    
    Used as a fast alternative to LLM routing.
    """
    
    def __init__(
        self,
        default_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        strategy_overrides: Optional[Dict[QueryClass, RetrievalStrategy]] = None,
    ):
        self.default_strategy = default_strategy
        self.strategy_overrides = strategy_overrides or {
            QueryClass.FACTUAL: RetrievalStrategy.HYBRID,
            QueryClass.ANALYTICAL: RetrievalStrategy.HIERARCHICAL,
            QueryClass.COMPARATIVE: RetrievalStrategy.MULTI_INDEX,
            QueryClass.KEYWORD: RetrievalStrategy.BM25,
            QueryClass.SEMANTIC: RetrievalStrategy.VECTOR,
        }
    
    def select(self, query_class: QueryClass) -> RetrievalStrategy:
        """Select strategy based on query class."""
        return self.strategy_overrides.get(query_class, self.default_strategy)
    
    def analyze_query(self, query: str) -> QueryClass:
        """Analyze query to determine class."""
        query_lower = query.lower()
        words = query_lower.split()
        
        # Check patterns
        if any(w in query_lower for w in ["vs", "versus", "compare", "difference"]):
            return QueryClass.COMPARATIVE
        
        if any(w in query_lower for w in ["analyze", "explain", "why", "how does"]):
            return QueryClass.ANALYTICAL
        
        if len(words) <= 3 and not any(w in query_lower for w in ["what", "how"]):
            return QueryClass.KEYWORD
        
        if any(w in query_lower for w in ["like", "similar", "related"]):
            return QueryClass.SEMANTIC
        
        return QueryClass.FACTUAL
