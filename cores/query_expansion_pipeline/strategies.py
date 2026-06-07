"""
query_expansion_pipeline/strategies.py

Expansion strategies for query expansion.

Strategies:
- SemanticStrategy: LLM-based semantic variations
- SynonymStrategy: Synonym/hypernym expansion
- DecomposeStrategy: Query decomposition
- ReformulateStrategy: Question reformulation
- KeywordStrategy: Keyword extraction and expansion
- ContextualStrategy: Chat history aware expansion
- HybridStrategy: Multi-strategy combination

Version: 1.0.0
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .providers import (
    ExpandedQuery,
    ExpansionStrategy,
    QueryIntent,
    DetectedIntent,
    ExtractedEntity,
    DecomposedQuery,
    SynonymProvider,
    QualityScorer,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Base Strategy
# ============================================================================


class BaseExpansionStrategy(ABC):
    """Base class for expansion strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Strategy identifier."""
        pass
    
    @abstractmethod
    async def expand(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExpandedQuery]:
        """Expand query using this strategy."""
        pass
    
    def _create_expansion(
        self,
        text: str,
        score: float = 1.0,
        metadata: Dict[str, Any] = None,
    ) -> ExpandedQuery:
        """Create an ExpandedQuery instance."""
        return ExpandedQuery(
            text=text,
            strategy=self.strategy_name,
            score=score,
            metadata=metadata or {},
        )


# ============================================================================
# Semantic Strategy
# ============================================================================


class SemanticStrategy(BaseExpansionStrategy):
    """
    LLM-based semantic expansion.
    
    Generates semantically similar query variations
    using language model.
    """
    
    @property
    def strategy_name(self) -> str:
        return ExpansionStrategy.SEMANTIC.value
    
    async def expand(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExpandedQuery]:
        """Expand using LLM."""
        context = context or {}
        llm_caller = context.get("llm_caller")
        
        if not llm_caller:
            # Fallback: simple variations
            return self._fallback_expand(query)
        
        num_variants = self.config.get("num_variants", 3)
        
        prompt = self._build_prompt(query, num_variants)
        
        try:
            response = await llm_caller(prompt)
            expansions = self._parse_response(response, query)
            return expansions
        except Exception as e:
            logger.warning(f"Semantic expansion failed: {e}")
            return self._fallback_expand(query)
    
    def _build_prompt(self, query: str, num_variants: int) -> str:
        """Build expansion prompt."""
        return f"""Generate {num_variants} semantic variations of this search query.
Each variation should:
- Preserve the original intent and meaning
- Use different phrasing, synonyms, or structure
- Be a valid standalone search query
- Be diverse from each other

Original query: {query}

Return only the variations, one per line, without numbering or bullets:"""
    
    def _parse_response(self, response: str, original: str) -> List[ExpandedQuery]:
        """Parse LLM response."""
        lines = response.strip().split('\n')
        expansions = []
        
        for line in lines:
            text = line.strip()
            
            # Clean up common prefixes
            if text and text[0].isdigit():
                text = re.sub(r'^[\d]+[\.\)\-\s]+', '', text)
            if text.startswith(('-', '•', '*')):
                text = text[1:].strip()
            
            # Skip empty or too similar
            if not text or len(text) < 3:
                continue
            if text.lower() == original.lower():
                continue
            
            expansions.append(self._create_expansion(
                text=text,
                score=0.8,
                metadata={"source": "llm"},
            ))
        
        return expansions[:self.config.get("num_variants", 5)]
    
    def _fallback_expand(self, query: str) -> List[ExpandedQuery]:
        """Fallback expansion without LLM."""
        expansions = []
        
        # Simple reformulations
        if query.lower().startswith("what is"):
            alt = query.replace("What is", "Define").replace("what is", "define")
            expansions.append(self._create_expansion(alt, 0.6))
        
        if query.lower().startswith("how to"):
            alt = query.replace("How to", "Steps to").replace("how to", "steps to")
            expansions.append(self._create_expansion(alt, 0.6))
        
        # Add keyword version
        words = query.split()
        if len(words) > 3:
            # Keep important words
            keywords = [w for w in words if len(w) > 3][:4]
            if keywords:
                expansions.append(self._create_expansion(
                    ' '.join(keywords),
                    0.5,
                    {"source": "keywords"},
                ))
        
        return expansions


# ============================================================================
# Synonym Strategy
# ============================================================================


class SynonymStrategy(BaseExpansionStrategy):
    """
    Synonym-based expansion.
    
    Expands queries by replacing words with synonyms.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._synonym_provider = SynonymProvider()
    
    @property
    def strategy_name(self) -> str:
        return ExpansionStrategy.SYNONYM.value
    
    async def expand(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExpandedQuery]:
        """Expand using synonyms."""
        max_per_term = self.config.get("max_synonyms_per_term", 3)
        
        expansions = self._synonym_provider.expand_query_with_synonyms(
            query,
            max_synonyms_per_term=max_per_term,
        )
        
        return [
            self._create_expansion(text, 0.7, {"source": "synonym"})
            for text in expansions[:self.config.get("max_expansions", 5)]
        ]


# ============================================================================
# Decompose Strategy
# ============================================================================


class DecomposeStrategy(BaseExpansionStrategy):
    """
    Query decomposition strategy.
    
    Breaks complex queries into simpler sub-queries.
    """
    
    @property
    def strategy_name(self) -> str:
        return ExpansionStrategy.DECOMPOSE.value
    
    async def expand(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExpandedQuery]:
        """Decompose complex query."""
        context = context or {}
        llm_caller = context.get("llm_caller")
        
        # Check complexity
        if not self._is_complex(query):
            return []
        
        if llm_caller:
            return await self._decompose_with_llm(query, llm_caller)
        else:
            return self._decompose_rule_based(query)
    
    def _is_complex(self, query: str) -> bool:
        """Check if query is complex enough to decompose."""
        threshold = self.config.get("min_complexity_threshold", 2)
        
        # Count complexity indicators
        complexity = 0
        
        # Multiple clauses
        if " and " in query.lower():
            complexity += 1
        if " or " in query.lower():
            complexity += 1
        if "," in query:
            complexity += 1
        
        # Question combinations
        question_words = ["what", "how", "why", "when", "where", "who"]
        q_count = sum(1 for w in question_words if w in query.lower())
        complexity += max(0, q_count - 1)
        
        # Length
        if len(query.split()) > 10:
            complexity += 1
        
        return complexity >= threshold
    
    async def _decompose_with_llm(
        self,
        query: str,
        llm_caller: Callable,
    ) -> List[ExpandedQuery]:
        """Decompose using LLM."""
        max_subqueries = self.config.get("max_subqueries", 5)
        
        prompt = f"""Break down this complex query into simpler, independent sub-questions.
Each sub-question should be answerable on its own.
Keep the core intent of each part.

Complex query: {query}

Generate up to {max_subqueries} sub-questions, one per line:"""
        
        try:
            response = await llm_caller(prompt)
            lines = response.strip().split('\n')
            
            expansions = []
            for line in lines:
                text = line.strip()
                if text and text[0].isdigit():
                    text = re.sub(r'^[\d]+[\.\)\-\s]+', '', text)
                if text.startswith(('-', '•', '*')):
                    text = text[1:].strip()
                
                if text and len(text) > 3:
                    expansions.append(self._create_expansion(
                        text,
                        0.85,
                        {"source": "decomposition"},
                    ))
            
            return expansions[:max_subqueries]
            
        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}")
            return self._decompose_rule_based(query)
    
    def _decompose_rule_based(self, query: str) -> List[ExpandedQuery]:
        """Rule-based decomposition."""
        expansions = []
        
        # Split on "and"
        if " and " in query.lower():
            parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
            for part in parts:
                part = part.strip()
                if len(part) > 5:
                    expansions.append(self._create_expansion(
                        part, 0.7, {"source": "split_and"}
                    ))
        
        # Split on commas
        if "," in query and not expansions:
            parts = query.split(",")
            for part in parts:
                part = part.strip()
                if len(part) > 5:
                    expansions.append(self._create_expansion(
                        part, 0.6, {"source": "split_comma"}
                    ))
        
        return expansions[:self.config.get("max_subqueries", 5)]


# ============================================================================
# Reformulate Strategy
# ============================================================================


class ReformulateStrategy(BaseExpansionStrategy):
    """
    Question reformulation strategy.
    
    Reformulates queries as different question types.
    """
    
    QUESTION_TEMPLATES = {
        "what": [
            "What is {topic}?",
            "What does {topic} mean?",
            "What are the key aspects of {topic}?",
        ],
        "how": [
            "How does {topic} work?",
            "How to {topic}?",
            "How can I {topic}?",
        ],
        "why": [
            "Why is {topic} important?",
            "Why should I {topic}?",
            "Why does {topic} matter?",
        ],
        "when": [
            "When should I use {topic}?",
            "When is {topic} applicable?",
        ],
        "examples": [
            "Examples of {topic}",
            "{topic} examples and use cases",
        ],
        "comparison": [
            "{topic} pros and cons",
            "{topic} advantages and disadvantages",
        ],
    }
    
    @property
    def strategy_name(self) -> str:
        return ExpansionStrategy.REFORMULATE.value
    
    async def expand(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExpandedQuery]:
        """Reformulate query."""
        context = context or {}
        intent = context.get("intent")
        
        # Extract topic
        topic = self._extract_topic(query)
        
        # Select templates based on intent
        templates = self._select_templates(intent)
        
        expansions = []
        for template in templates:
            try:
                text = template.format(topic=topic)
                expansions.append(self._create_expansion(
                    text, 0.7, {"template": template[:30]}
                ))
            except Exception:
                pass
        
        return expansions[:self.config.get("max_reformulations", 4)]
    
    def _extract_topic(self, query: str) -> str:
        """Extract main topic from query."""
        # Remove question words
        topic = query
        for prefix in ["what is", "how to", "how does", "why is", "when to", "where is", "who is"]:
            if topic.lower().startswith(prefix):
                topic = topic[len(prefix):].strip()
                break
        
        # Remove trailing punctuation
        topic = topic.rstrip("?!.")
        
        return topic
    
    def _select_templates(
        self,
        intent: Optional[DetectedIntent],
    ) -> List[str]:
        """Select templates based on intent."""
        templates = []
        
        if intent:
            intent_type = intent.intent.value
            
            if intent_type == "definition":
                templates.extend(self.QUESTION_TEMPLATES.get("what", []))
            elif intent_type == "procedural":
                templates.extend(self.QUESTION_TEMPLATES.get("how", []))
            elif intent_type == "comparison":
                templates.extend(self.QUESTION_TEMPLATES.get("comparison", []))
        
        # Add defaults
        if not templates:
            templates.extend(self.QUESTION_TEMPLATES.get("what", [])[:2])
            templates.extend(self.QUESTION_TEMPLATES.get("how", [])[:1])
            templates.extend(self.QUESTION_TEMPLATES.get("examples", [])[:1])
        
        return templates


# ============================================================================
# Keyword Strategy
# ============================================================================


class KeywordStrategy(BaseExpansionStrategy):
    """
    Keyword extraction and expansion.
    
    Extracts key terms and creates keyword-based queries.
    """
    
    @property
    def strategy_name(self) -> str:
        return ExpansionStrategy.KEYWORDS.value
    
    async def expand(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExpandedQuery]:
        """Extract and expand keywords."""
        context = context or {}
        entities = context.get("entities", [])
        
        # Extract keywords
        keywords = self._extract_keywords(query)
        
        # Add entity texts
        for entity in entities:
            if hasattr(entity, 'text'):
                keywords.append(entity.text)
        
        # Remove duplicates
        keywords = list(dict.fromkeys(keywords))
        
        expansions = []
        
        # Keywords only
        if keywords:
            kw_query = ' '.join(keywords[:self.config.get("max_keywords", 5)])
            expansions.append(self._create_expansion(
                kw_query, 0.6, {"source": "keywords_only"}
            ))
        
        # Keyword combinations
        if len(keywords) >= 2:
            for i in range(min(3, len(keywords))):
                combo = keywords[:i+2]
                expansions.append(self._create_expansion(
                    ' '.join(combo), 0.5, {"source": "keyword_combo"}
                ))
        
        return expansions[:self.config.get("max_keyword_expansions", 4)]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Simple stopword-based extraction
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "of", "to", "in",
            "for", "on", "with", "at", "by", "from", "as", "what", "how",
            "why", "when", "where", "who", "which", "and", "or", "but",
            "if", "can", "i", "you", "it", "this", "that",
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords


# ============================================================================
# Contextual Strategy
# ============================================================================


class ContextualStrategy(BaseExpansionStrategy):
    """
    Context-aware expansion using chat history.
    
    Uses previous messages to expand and clarify queries.
    """
    
    @property
    def strategy_name(self) -> str:
        return ExpansionStrategy.CONTEXTUAL.value
    
    async def expand(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExpandedQuery]:
        """Expand using conversation context."""
        context = context or {}
        chat_history = context.get("chat_history", [])
        llm_caller = context.get("llm_caller")
        
        if not chat_history:
            return []
        
        # Resolve pronouns
        resolved = self._resolve_pronouns(query, chat_history)
        
        if resolved != query:
            expansions = [self._create_expansion(
                resolved, 0.9, {"source": "pronoun_resolution"}
            )]
        else:
            expansions = []
        
        # Context-aware expansion with LLM
        if llm_caller and chat_history:
            try:
                context_expansion = await self._expand_with_context(
                    query, chat_history, llm_caller
                )
                expansions.extend(context_expansion)
            except Exception as e:
                logger.warning(f"Contextual expansion failed: {e}")
        
        return expansions
    
    def _resolve_pronouns(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
    ) -> str:
        """Resolve pronouns using chat history."""
        pronouns = ["it", "this", "that", "they", "them", "these", "those"]
        
        query_lower = query.lower()
        has_pronoun = any(
            f" {p} " in f" {query_lower} " or
            query_lower.startswith(f"{p} ") or
            query_lower.endswith(f" {p}")
            for p in pronouns
        )
        
        if not has_pronoun:
            return query
        
        # Find likely referent from history
        for msg in reversed(chat_history[-3:]):
            content = msg.get("content", "")
            
            # Extract nouns (simple heuristic)
            words = content.split()
            nouns = [w for w in words if len(w) > 4 and w[0].isupper()]
            
            if nouns:
                # Replace first pronoun with first noun
                for pronoun in pronouns:
                    pattern = rf'\b{pronoun}\b'
                    if re.search(pattern, query, re.IGNORECASE):
                        return re.sub(pattern, nouns[0], query, count=1, flags=re.IGNORECASE)
        
        return query
    
    async def _expand_with_context(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        llm_caller: Callable,
    ) -> List[ExpandedQuery]:
        """Expand using LLM with context."""
        max_history = self.config.get("max_history_messages", 3)
        recent = chat_history[-max_history:]
        
        history_text = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in recent
        ])
        
        prompt = f"""Given this conversation context, generate a more specific version of the current query.

Previous messages:
{history_text}

Current query: {query}

Rewrite the query to be more specific and self-contained, incorporating relevant context:"""
        
        response = await llm_caller(prompt)
        text = response.strip()
        
        if text and text != query:
            return [self._create_expansion(
                text, 0.85, {"source": "context_llm"}
            )]
        
        return []


# ============================================================================
# Hybrid Strategy
# ============================================================================


class HybridStrategy(BaseExpansionStrategy):
    """
    Multi-strategy combination.
    
    Combines results from multiple strategies using
    voting or weighted combination.
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        strategies: Dict[str, BaseExpansionStrategy] = None,
    ):
        super().__init__(config)
        self._strategies = strategies or {}
    
    @property
    def strategy_name(self) -> str:
        return ExpansionStrategy.HYBRID.value
    
    def set_strategies(self, strategies: Dict[str, BaseExpansionStrategy]) -> None:
        """Set available strategies."""
        self._strategies = strategies
    
    async def expand(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExpandedQuery]:
        """Combine multiple strategies."""
        strategy_names = self.config.get("strategies", ["semantic", "synonym", "keywords"])
        weights = self.config.get("weights", {})
        method = self.config.get("combination_method", "weighted")
        
        all_expansions = []
        
        # Run selected strategies
        for name in strategy_names:
            if name in self._strategies:
                strategy = self._strategies[name]
                try:
                    expansions = await strategy.expand(query, context)
                    
                    # Apply weight
                    weight = weights.get(name, 1.0)
                    for exp in expansions:
                        exp.score *= weight
                        exp.metadata["hybrid_source"] = name
                    
                    all_expansions.extend(expansions)
                except Exception as e:
                    logger.warning(f"Strategy {name} failed in hybrid: {e}")
        
        # Combine results
        if method == "voting":
            return self._combine_voting(all_expansions)
        else:  # weighted
            return self._combine_weighted(all_expansions)
    
    def _combine_weighted(
        self,
        expansions: List[ExpandedQuery],
    ) -> List[ExpandedQuery]:
        """Combine by weighted score."""
        # Deduplicate by text, keeping highest score
        by_text: Dict[str, ExpandedQuery] = {}
        
        for exp in expansions:
            text_key = exp.text.lower().strip()
            
            if text_key not in by_text or exp.score > by_text[text_key].score:
                by_text[text_key] = exp
        
        # Sort by score
        sorted_exps = sorted(by_text.values(), key=lambda x: x.score, reverse=True)
        
        return sorted_exps[:self.config.get("max_combined", 8)]
    
    def _combine_voting(
        self,
        expansions: List[ExpandedQuery],
    ) -> List[ExpandedQuery]:
        """Combine by voting (frequency)."""
        votes: Dict[str, Tuple[int, ExpandedQuery]] = {}
        
        for exp in expansions:
            text_key = exp.text.lower().strip()
            
            if text_key in votes:
                count, existing = votes[text_key]
                votes[text_key] = (count + 1, existing)
            else:
                votes[text_key] = (1, exp)
        
        # Filter by minimum agreement
        min_agree = self.config.get("min_agreement", 1)
        
        results = []
        for text_key, (count, exp) in votes.items():
            if count >= min_agree:
                exp.score = count / len(self._strategies)
                exp.metadata["votes"] = count
                results.append(exp)
        
        # Sort by votes
        results.sort(key=lambda x: x.metadata.get("votes", 0), reverse=True)
        
        return results[:self.config.get("max_combined", 8)]


# ============================================================================
# Strategy Factory
# ============================================================================


class StrategyFactory:
    """Factory for creating expansion strategies."""
    
    STRATEGY_MAP = {
        ExpansionStrategy.SEMANTIC: SemanticStrategy,
        ExpansionStrategy.SYNONYM: SynonymStrategy,
        ExpansionStrategy.DECOMPOSE: DecomposeStrategy,
        ExpansionStrategy.REFORMULATE: ReformulateStrategy,
        ExpansionStrategy.KEYWORDS: KeywordStrategy,
        ExpansionStrategy.CONTEXTUAL: ContextualStrategy,
        ExpansionStrategy.HYBRID: HybridStrategy,
    }
    
    @classmethod
    def create(
        cls,
        strategy: Union[str, ExpansionStrategy],
        config: Dict[str, Any] = None,
    ) -> BaseExpansionStrategy:
        """Create a strategy instance."""
        if isinstance(strategy, str):
            strategy = ExpansionStrategy(strategy)
        
        strategy_class = cls.STRATEGY_MAP.get(strategy)
        
        if not strategy_class:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return strategy_class(config)
    
    @classmethod
    def create_all(
        cls,
        configs: Dict[str, Dict[str, Any]] = None,
    ) -> Dict[str, BaseExpansionStrategy]:
        """Create all strategies."""
        configs = configs or {}
        strategies = {}
        
        for strategy in ExpansionStrategy:
            if strategy == ExpansionStrategy.HYBRID:
                continue  # Created separately
            
            config = configs.get(strategy.value, {})
            strategies[strategy.value] = cls.create(strategy, config)
        
        # Create hybrid with reference to other strategies
        hybrid_config = configs.get(ExpansionStrategy.HYBRID.value, {})
        hybrid = HybridStrategy(hybrid_config, strategies)
        strategies[ExpansionStrategy.HYBRID.value] = hybrid
        
        return strategies
