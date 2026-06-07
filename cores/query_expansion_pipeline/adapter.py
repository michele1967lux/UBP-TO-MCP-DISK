"""
query_expansion_pipeline/adapter.py

Bridge Layer - Exposes all module operations.
Orchestrates strategies, providers, and LLM delegation.

This is the main entry point for the module.

Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

from .providers import (
    ExpansionStrategy,
    ExpandedQuery,
    ExpansionResult,
    DetectedIntent,
    ExtractedEntity,
    DecomposedQuery,
    QueryNormalizer,
    IntentClassifier,
    EntityExtractor,
    SynonymProvider,
    QualityScorer,
    CacheProvider,
    LanguageDetector,
)
from .strategies import (
    BaseExpansionStrategy,
    SemanticStrategy,
    SynonymStrategy,
    DecomposeStrategy,
    ReformulateStrategy,
    KeywordStrategy,
    ContextualStrategy,
    HybridStrategy,
    StrategyFactory,
)
from .delegation import (
    LLMDelegator,
    LLMConfig,
    DIContainerModuleRegistry,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Utilities
# ============================================================================


def _coerce_value(value: Any) -> Any:
    """Coerce string values to appropriate types."""
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


def _coerce_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively coerce config values."""
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _coerce_config(value)
        elif isinstance(value, list):
            result[key] = [
                _coerce_config(v) if isinstance(v, dict) else _coerce_value(v)
                for v in value
            ]
        else:
            result[key] = _coerce_value(value)
    return result


def _resolve_env(text: str) -> str:
    """Resolve environment variable placeholders."""
    pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
    
    def replace(match):
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)
    
    return re.sub(pattern, replace, text)


def _load_config(module_path: Path) -> Dict[str, Any]:
    """Load and resolve config.json."""
    config_file = module_path / "config.json"
    
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_file}")
        return {}
    
    with open(config_file, "r", encoding="utf-8") as f:
        raw = f.read()
    
    resolved = _resolve_env(raw)
    parsed = json.loads(resolved)
    
    return _coerce_config(parsed)


# ============================================================================
# Query Expansion Adapter
# ============================================================================


class QueryExpansionAdapter:
    """
    Main adapter for query_expansion_pipeline module.
    
    Implements all operations defined in manifest.json.
    Orchestrates strategies, providers, and LLM delegation.
    """
    
    def __init__(
        self,
        module_path: Path,
        di_container: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        self.module_path = Path(module_path)
        self.di_container = di_container
        self.event_bus = event_bus
        
        # Load configuration
        self.config = _load_config(module_path)
        
        # Environment
        self.env = os.environ.get("UBP_ENV", "dev")
        
        # Components (lazy init)
        self._normalizer: Optional[QueryNormalizer] = None
        self._intent_classifier: Optional[IntentClassifier] = None
        self._entity_extractor: Optional[EntityExtractor] = None
        self._synonym_provider: Optional[SynonymProvider] = None
        self._quality_scorer: Optional[QualityScorer] = None
        self._language_detector: Optional[LanguageDetector] = None
        self._cache: Optional[CacheProvider] = None
        self._llm_delegator: Optional[LLMDelegator] = None
        self._strategies: Dict[str, BaseExpansionStrategy] = {}
        
        # Registry
        self._module_registry: Optional[DIContainerModuleRegistry] = None
        
        # State
        self._initialized = False
    
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
    # Event Publisher
    # ========================================================================
    
    @property
    def publisher(self) -> Optional[Callable]:
        """Get event publisher."""
        if self.event_bus and hasattr(self.event_bus, "publish"):
            return self.event_bus.publish
        return None
    
    async def _publish_event(self, event: str, data: Dict[str, Any]) -> None:
        """Publish event if bus available."""
        if self.publisher:
            try:
                await self.publisher(event, data)
            except Exception as e:
                logger.warning(f"Event publish failed: {e}")
    
    # ========================================================================
    # Lifecycle Operations
    # ========================================================================
    
    async def initialize(
        self,
        preload_llm: bool = False,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Initialize query expansion pipeline."""
        start_time = time.perf_counter()
        
        try:
            # Setup module registry
            redis_client = None
            if self.di_container:
                self._module_registry = DIContainerModuleRegistry(self.di_container)
                
                try:
                    import redis.asyncio as aioredis
                    redis_client = await self.di_container.resolve(aioredis.Redis)
                except Exception:
                    pass
            
            # Initialize providers
            norm_config = self.config.get("normalization", {})
            self._normalizer = QueryNormalizer(
                lowercase=norm_config.get("lowercase", False),
                remove_punctuation=norm_config.get("remove_punctuation", False),
                remove_stopwords=norm_config.get("remove_stopwords", False),
                expand_abbreviations=norm_config.get("expand_abbreviations", True),
            )
            
            self._intent_classifier = IntentClassifier()
            self._entity_extractor = EntityExtractor()
            self._synonym_provider = SynonymProvider()
            self._language_detector = LanguageDetector()
            
            quality_config = self.config.get("quality", {})
            self._quality_scorer = QualityScorer(
                min_length=self.config.get("expansion", {}).get("min_expansion_length", 3),
                max_length=self.config.get("expansion", {}).get("max_expansion_length", 200),
                similarity_threshold=quality_config.get("similarity_threshold", 0.9),
            )
            
            # Initialize cache
            cache_config = self.config.get("cache", {})
            self._cache = CacheProvider(
                redis_client=redis_client,
                prefix=cache_config.get("redis_prefix", "ubp:query_expansion"),
                ttl_seconds=cache_config.get("ttl_seconds", 3600),
                enabled=cache_config.get("enabled", True),
            )
            
            # Initialize LLM delegator
            llm_config = self.config.get("llm", {})
            self._llm_delegator = LLMDelegator(
                config=LLMConfig(
                    module=llm_config.get("module", "inference_ollama_grok"),
                    operation=llm_config.get("operation", "generate"),
                    provider=llm_config.get("provider", "grok"),
                    timeout_seconds=llm_config.get("timeout_seconds", 30),
                    max_retries=llm_config.get("max_retries", 2),
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 300),
                ),
                module_registry=self._module_registry,
            )
            
            if preload_llm:
                await self._llm_delegator.ensure_llm()
            
            # Initialize strategies
            strategy_configs = self.config.get("strategies", {})
            self._strategies = StrategyFactory.create_all(strategy_configs)
            
            self._initialized = True
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            await self._publish_event("query_expansion.initialized", {
                "module": "query_expansion_pipeline",
            })
            
            return {
                "status": "initialized",
                "module": "query_expansion_pipeline",
                "version": "1.0.0",
                "env": self.env,
                "strategies_available": list(self._strategies.keys()),
                "llm_available": self._llm_delegator.is_available(),
                "cache_enabled": cache_config.get("enabled", True),
                "elapsed_ms": round(elapsed_ms, 2),
            }
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return {
                "status": "error",
                "module": "query_expansion_pipeline",
                "error": str(e),
            }
    
    async def shutdown(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Graceful shutdown."""
        self._initialized = False
        
        await self._publish_event("query_expansion.shutdown", {
            "module": "query_expansion_pipeline",
        })
        
        return {
            "status": "shutdown",
            "module": "query_expansion_pipeline",
        }
    
    async def health_check(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Health check."""
        result = {
            "module": "query_expansion_pipeline",
            "version": "1.0.0",
            "status": "healthy",
            "env": self.env,
        }
        
        # Check LLM
        if self._llm_delegator:
            result["llm_delegation"] = await self._llm_delegator.health_check()
        else:
            result["llm_delegation"] = {"status": "not_initialized"}
        
        # Check cache
        if self._cache:
            result["cache"] = self._cache.get_stats()
        
        # Check strategies
        result["strategies"] = list(self._strategies.keys())
        
        return result
    
    # ========================================================================
    # Main Expansion Operations
    # ========================================================================
    
    async def expand(
        self,
        query: str,
        strategy: str = "semantic",
        num_expansions: int = 5,
        chat_history: Optional[List[Dict[str, str]]] = None,
        include_original: bool = True,
        detect_intent: bool = True,
        extract_entities: bool = True,
        filter_quality: bool = True,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Main expansion operation.
        
        Expands query using specified strategy with full pipeline:
        1. Normalize query
        2. Detect language
        3. Detect intent (optional)
        4. Extract entities (optional)
        5. Apply expansion strategy
        6. Quality filter (optional)
        7. Combine results
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        # Check cache
        cache_key = f"{strategy}:{query}"
        if self._cache:
            cached = await self._cache.get(query, strategy)
            if cached:
                return {
                    "original_query": query,
                    "expanded_queries": cached,
                    "combined_query": self._combine_queries([q["text"] for q in cached], include_original, query),
                    "strategy_used": strategy,
                    "from_cache": True,
                    "time_ms": 0,
                }
        
        # Normalize
        normalized = self._normalizer.normalize(query) if self._normalizer else query
        
        # Detect language
        language = "en"
        if self._language_detector:
            language = self._language_detector.detect(query)
        
        # Detect intent
        intent = None
        if detect_intent and self._intent_classifier:
            intent = self._intent_classifier.classify(query)
        
        # Extract entities
        entities = []
        if extract_entities and self._entity_extractor:
            entities = self._entity_extractor.extract(query)
        
        # Build context for strategy
        context = {
            "chat_history": chat_history,
            "intent": intent,
            "entities": entities,
            "language": language,
            "llm_caller": self._llm_delegator.get_caller() if self._llm_delegator else None,
        }
        
        # Get strategy
        strategy_instance = self._strategies.get(strategy)
        if not strategy_instance:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Expand
        expansions = await strategy_instance.expand(normalized, context)
        
        # Quality filter
        if filter_quality and self._quality_scorer:
            expansions = self._quality_scorer.filter_expansions(
                query,
                expansions,
                min_score=self.config.get("quality", {}).get("min_score_threshold", 0.3),
            )
        
        # Limit results
        expansions = expansions[:num_expansions]
        
        # Combine
        combined = self._combine_queries(
            [e.text for e in expansions],
            include_original,
            query,
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Cache results
        if self._cache and expansions:
            await self._cache.set(
                query, strategy,
                [e.to_dict() for e in expansions],
            )
        
        result = ExpansionResult(
            original_query=query,
            expanded_queries=expansions,
            combined_query=combined,
            strategy_used=strategy,
            intent=intent,
            entities=entities,
            language=language,
            time_ms=elapsed_ms,
        )
        
        return result.to_dict()
    
    async def expand_semantic(
        self,
        query: str,
        num_variants: int = 3,
        chat_history: Optional[List[Dict[str, str]]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Semantic expansion using LLM."""
        return await self.expand(
            query=query,
            strategy="semantic",
            num_expansions=num_variants,
            chat_history=chat_history,
            ctx=ctx,
            **kwargs,
        )
    
    async def expand_synonyms(
        self,
        query: str,
        max_synonyms_per_term: int = 3,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Synonym-based expansion."""
        return await self.expand(
            query=query,
            strategy="synonym",
            num_expansions=self.config.get("expansion", {}).get("max_expansions", 10),
            ctx=ctx,
            **kwargs,
        )
    
    async def decompose_query(
        self,
        query: str,
        max_subqueries: int = 5,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Decompose complex query into sub-queries."""
        return await self.expand(
            query=query,
            strategy="decompose",
            num_expansions=max_subqueries,
            ctx=ctx,
            **kwargs,
        )
    
    async def reformulate_query(
        self,
        query: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Reformulate query as different question types."""
        return await self.expand(
            query=query,
            strategy="reformulate",
            num_expansions=4,
            ctx=ctx,
            **kwargs,
        )
    
    async def extract_keywords(
        self,
        query: str,
        max_keywords: int = 5,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Extract and expand keywords."""
        return await self.expand(
            query=query,
            strategy="keywords",
            num_expansions=max_keywords,
            ctx=ctx,
            **kwargs,
        )
    
    async def expand_contextual(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Context-aware expansion using chat history."""
        return await self.expand(
            query=query,
            strategy="contextual",
            chat_history=chat_history,
            ctx=ctx,
            **kwargs,
        )
    
    async def expand_hybrid(
        self,
        query: str,
        strategies: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Multi-strategy hybrid expansion."""
        # Update hybrid config if provided
        if strategies or weights:
            hybrid = self._strategies.get("hybrid")
            if hybrid:
                if strategies:
                    hybrid.config["strategies"] = strategies
                if weights:
                    hybrid.config["weights"] = weights
        
        return await self.expand(
            query=query,
            strategy="hybrid",
            ctx=ctx,
            **kwargs,
        )
    
    # ========================================================================
    # Analysis Operations
    # ========================================================================
    
    async def detect_intent(
        self,
        query: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Detect query intent."""
        if not self._initialized:
            await self.initialize()
        
        if not self._intent_classifier:
            return {"error": "Intent classifier not initialized"}
        
        intent = self._intent_classifier.classify(query)
        return intent.to_dict()
    
    async def extract_entities(
        self,
        query: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Extract named entities from query."""
        if not self._initialized:
            await self.initialize()
        
        if not self._entity_extractor:
            return {"entities": []}
        
        entities = self._entity_extractor.extract(query)
        
        return {
            "query": query,
            "entities": [e.to_dict() for e in entities],
            "entity_count": len(entities),
        }
    
    async def normalize_query(
        self,
        query: str,
        options: Optional[Dict[str, bool]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Normalize and clean query."""
        if not self._initialized:
            await self.initialize()
        
        # Apply options if provided
        if options and self._normalizer:
            self._normalizer.lowercase = options.get("lowercase", self._normalizer.lowercase)
            self._normalizer.remove_punctuation = options.get("remove_punctuation", self._normalizer.remove_punctuation)
            self._normalizer.remove_stopwords = options.get("remove_stopwords", self._normalizer.remove_stopwords)
        
        normalized = self._normalizer.normalize(query) if self._normalizer else query
        
        return {
            "original": query,
            "normalized": normalized,
            "changed": query != normalized,
        }
    
    async def detect_language(
        self,
        query: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Detect query language."""
        if not self._initialized:
            await self.initialize()
        
        language = "en"
        if self._language_detector:
            language = self._language_detector.detect(query)
        
        return {
            "query": query,
            "language": language,
        }
    
    # ========================================================================
    # Configuration Operations
    # ========================================================================
    
    async def get_config(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "expansion": self.config.get("expansion", {}),
            "strategies": list(self._strategies.keys()),
            "default_strategy": self.config.get("expansion", {}).get("default_strategy", "semantic"),
            "llm_module": self.config.get("llm", {}).get("module"),
        }
    
    async def get_stats(
        self,
        period: str = "24h",
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get expansion statistics."""
        return {
            "module": "query_expansion_pipeline",
            "period": period,
            "cache_stats": self._cache.get_stats() if self._cache else {},
        }
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def _combine_queries(
        self,
        expansions: List[str],
        include_original: bool,
        original: str,
    ) -> str:
        """Combine queries for retrieval."""
        method = self.config.get("output", {}).get("combine_method", "union")
        max_length = self.config.get("output", {}).get("max_combined_length", 500)
        
        queries = []
        
        if include_original:
            queries.append(original)
        
        queries.extend(expansions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for q in queries:
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)
        
        if method == "concat":
            combined = " ".join(unique)
        else:  # union
            combined = " | ".join(unique)
        
        # Truncate if needed
        if len(combined) > max_length:
            combined = combined[:max_length] + "..."
        
        return combined
