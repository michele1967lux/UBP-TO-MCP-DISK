"""
Semantic Router for RAG Orchestrator

Intelligent query routing system that classifies user queries into:
- CHAT: Pure conversation (greetings, small talk) -> No vector search
- RAG: Knowledge retrieval (document questions) -> Qdrant search
- WEB: Web search (real-time info, weather, news) -> Web search module

The router uses a multi-layered heuristic system (FEAT-ROUTER-002):
- Layer A: Hard patterns (binary flags)
- Layer B: Keyword scoring (weighted capabilities)
- Layer C: Fuzzy matching (controlled, whitelist-based)
- Layer D: Negation rules (score subtraction)

Falls back to LLM classification for ambiguous queries.

ROADMAP v1.7.x - FEAT-ROUTER-001, FEAT-ROUTER-002
"""

import re
import json
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict

logger = logging.getLogger(__name__)

# Import heuristic engine components
HEURISTIC_ENGINE_AVAILABLE = False

try:
    from .router.heuristic_engine import HeuristicEngine, EngineConfig
    from .router.models import PatternSignals

    HEURISTIC_ENGINE_AVAILABLE = True
except ImportError as e:
    HeuristicEngine = None  # type: ignore
    EngineConfig = None  # type: ignore
    PatternSignals = None  # type: ignore
    logger.warning(f"HeuristicEngine not available: {e}")


class RouteType(str, Enum):
    """Query route classification types."""

    CHAT = "chat"
    RAG = "rag"
    WEB = "web"
    REPORT = "report"  # v2.3: Interactive Analyst - structured report generation


class SmartFallbackStrategy(str, Enum):
    """
    Fallback strategy when primary route returns empty/uncertain results.

    CONSERVATIVE: Only fallback if explicitly configured
    AGGRESSIVE: Always try fallback chain on empty results
    SMART: Context-aware fallback based on query signals (recommended)
    """

    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    SMART = "smart"


@dataclass
class RouterConfig:
    """
    Configuration for Smart Router v2.

    Attributes:
        confidence_threshold: Minimum confidence to trust heuristic result
        fallback_strategy: How to handle empty/uncertain results
        default_route_for_unknown: Route for truly ambiguous queries (WEB recommended)
        enable_auto_retry: Allow automatic retry with fallback route
        max_retry_attempts: Maximum fallback attempts before giving up
        empty_rag_triggers_web: If RAG returns empty, try WEB automatically
        commercial_boost_threshold: Min commercial signal to boost WEB confidence
    """

    confidence_threshold: float = 0.7
    fallback_strategy: SmartFallbackStrategy = SmartFallbackStrategy.SMART
    default_route_for_unknown: RouteType = RouteType.RAG  # v2.1: RAG default (WEB only explicit)
    enable_auto_retry: bool = True
    max_retry_attempts: int = 2
    empty_rag_triggers_web: bool = False  # v2.1: Disabled - WEB only explicit
    commercial_boost_threshold: float = 0.3


@dataclass
class RouterResult:
    """
    Result of query classification.

    Attributes:
        route: The classified route type (CHAT, RAG, WEB)
        confidence: Classification confidence (0.0 - 1.0)
        method: Classification method used ("heuristic", "heuristic_v2", "llm", "fallback")
        reasoning: Optional explanation for the classification
        signals: Full PatternSignals from heuristic engine (if v2)
    """

    route: RouteType
    confidence: float
    method: str
    reasoning: Optional[str] = None
    signals: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "route": self.route.value,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "reasoning": self.reasoning,
        }
        if self.signals:
            result["signals"] = self.signals
        return result


class SemanticRouter:
    """
    Intelligent query router for RAG orchestrator.

    Routes queries to appropriate handler:
    - CHAT: Pure LLM conversation (greetings, small talk)
    - RAG: Knowledge retrieval from vector store
    - WEB: Web search for real-time information

    Now uses multi-layered HeuristicEngine (v2) for improved classification:
    - Layer A: Hard patterns → Binary flags (greeting, temporal, internal)
    - Layer B: Keyword scoring → Capability scores
    - Layer C: Fuzzy matching → Typo tolerance
    - Layer D: Negations → Score adjustments

    Usage:
        router = SemanticRouter(llm_module)
        result = await router.classify("Ciao, come stai?")
        # result.route == RouteType.CHAT
    """

    # LLM classification prompt (used as fallback)
    CLASSIFICATION_PROMPT = """You are a query intent classifier for an enterprise RAG system.
Classify the user query into exactly ONE category:

- CHAT: Greetings, small talk, personal questions, thank you messages, goodbyes, simple yes/no responses.
  Examples: "Ciao!", "How are you?", "Thanks!", "Goodbye"

- RAG: Questions about documents, manuals, internal knowledge, stored information, company policies, procedures.
  Examples: "What does the safety manual say?", "Summarize the HR policy", "Find info about project X"

- WEB: Questions about current events, weather, stock prices, real-time data, news, live information.
  Examples: "What's the weather in Rome?", "Bitcoin price now", "Latest news today"

User Query: {query}

Respond with JSON only (no markdown, no explanation):
{{"route": "CHAT|RAG|WEB", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    def __init__(
        self,
        llm_module: Optional[Any] = None,
        default_route: RouteType = RouteType.RAG,
        confidence_threshold: float = 0.7,
        use_heuristic_v2: bool = True,
        config_dir: Optional[Path] = None,
        router_config: Optional[RouterConfig] = None,
    ):
        """
        Initialize the semantic router.

        Args:
            llm_module: LLM module for ambiguous query classification (optional)
            default_route: Default route when classification fails (default: RAG)
            confidence_threshold: Minimum confidence for LLM classification (default: 0.7)
            use_heuristic_v2: Use new multi-layer heuristic engine (default: True)
            config_dir: Path to router config directory (for heuristic v2)
            router_config: Smart Router v2 configuration (optional, uses defaults)
        """
        self.llm_module = llm_module
        self.default_route = default_route
        self.confidence_threshold = confidence_threshold
        self.use_heuristic_v2 = use_heuristic_v2 and HEURISTIC_ENGINE_AVAILABLE

        # Smart Router v2 Configuration
        self.router_config = router_config or RouterConfig()

        # Initialize heuristic engine (v2)
        self.heuristic_engine: Optional[Any] = None
        if (
            self.use_heuristic_v2
            and HEURISTIC_ENGINE_AVAILABLE
            and HeuristicEngine is not None
        ):
            try:
                if config_dir is None:
                    config_dir = Path(__file__).parent / "router" / "config"

                engine_config = EngineConfig(  # type: ignore
                    config_dir=config_dir,
                    enable_fuzzy=True,
                    enable_negations=True,
                    normalize_scores=True,
                    debug_mode=False,
                )
                self.heuristic_engine = HeuristicEngine(engine_config)  # type: ignore
                logger.info("✅ HeuristicEngine v2 initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize HeuristicEngine v2: {e}")
                self.use_heuristic_v2 = False

        # Statistics
        self.total_classifications = 0
        self.heuristic_v1_hits = 0
        self.heuristic_v2_hits = 0
        self.llm_classifications = 0
        self.fallback_count = 0
        self.auto_retry_count = 0  # New: track auto-retries

        logger.info(
            "SemanticRouter initialized",
            extra={
                "llm_available": llm_module is not None,
                "default_route": default_route.value,
                "confidence_threshold": confidence_threshold,
                "heuristic_v2": self.use_heuristic_v2,
                "fallback_strategy": self.router_config.fallback_strategy.value,
            },
        )

    def set_llm_module(self, llm_module: Any) -> None:
        """
        Set or update the LLM module.

        Args:
            llm_module: LLM module for classification
        """
        self.llm_module = llm_module
        logger.info("SemanticRouter LLM module updated")

    async def classify(
        self, query: str, user_keywords: Optional[set] = None
    ) -> RouterResult:
        """
        Classify query intent using heuristics first, then LLM if needed.

        FEAT-DKI-001 (v1.8.1): Now supports dynamic keyword injection.
        If user_keywords is provided, the router will boost RAG confidence
        when the query contains any of these keywords from the user's accessible KBs.

        Args:
            query: User query to classify
            user_keywords: Optional set of keywords from user's accessible KBs (DKI v1.8.1)

        Returns:
            RouterResult with route, confidence, method, and optional reasoning
        """
        self.total_classifications += 1

        # Normalize query
        query_normalized = query.strip()
        query_lower = query_normalized.lower()

        logger.debug(f"Classifying query: {query_normalized[:100]}")

        # FEAT-DKI-001: Layer K - Dynamic Knowledge Keyword Check
        # Before any other classification, check if query contains user's KB keywords
        dki_boost_applied = False
        if user_keywords:
            for keyword in user_keywords:
                if keyword.lower() in query_lower:
                    dki_boost_applied = True
                    logger.info(
                        f"DKI: Keyword match found - boosting RAG",
                        extra={
                            "keyword": keyword,
                            "query_prefix": query_normalized[:50],
                        },
                    )
                    # Immediate RAG routing with high confidence
                    return RouterResult(
                        route=RouteType.RAG,
                        confidence=0.92,  # High but not 0.95 (reserved for hard flags)
                        method="dki_keyword_match",
                        reasoning=f"Query contains KB keyword: '{keyword}'",
                    )

        # Store heuristic result even if below threshold (for fallback signals)
        heuristic_result: Optional[RouterResult] = None

        # Phase 1: Heuristic v2 classification (multi-layer engine)
        if self.use_heuristic_v2 and self.heuristic_engine:
            heuristic_result = self._classify_heuristic_v2(query_normalized)
            if (
                heuristic_result
                and heuristic_result.confidence >= self.confidence_threshold
            ):
                self.heuristic_v2_hits += 1
                logger.info(
                    f"Query classified via heuristic_v2",
                    extra={
                        "route": heuristic_result.route.value,
                        "confidence": heuristic_result.confidence,
                    },
                )
                return heuristic_result
            elif heuristic_result:
                logger.debug(
                    f"Heuristic v2 below threshold",
                    extra={
                        "route": heuristic_result.route.value,
                        "confidence": heuristic_result.confidence,
                        "threshold": self.confidence_threshold,
                    },
                )

        # Phase 2: LLM classification (for ambiguous queries)
        if self.llm_module:
            try:
                llm_result = await self._classify_with_llm(query_normalized)
                if llm_result and llm_result.confidence >= self.confidence_threshold:
                    self.llm_classifications += 1
                    # Include heuristic signals if available
                    if heuristic_result and heuristic_result.signals:
                        llm_result.signals = heuristic_result.signals
                    logger.info(
                        f"Query classified via LLM",
                        extra={
                            "route": llm_result.route.value,
                            "confidence": llm_result.confidence,
                            "reasoning": llm_result.reasoning,
                        },
                    )
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")

        # Phase 3: Fallback - use heuristic suggestion if available
        self.fallback_count += 1

        # If heuristic gave a result but below threshold, use its route but mark as fallback
        if heuristic_result:
            logger.info(
                f"Query classified via fallback (with heuristic signals)",
                extra={
                    "route": heuristic_result.route.value,
                    "heuristic_confidence": heuristic_result.confidence,
                },
            )
            return RouterResult(
                route=heuristic_result.route,
                confidence=heuristic_result.confidence,
                method="fallback_with_signals",
                reasoning=f"Heuristic below threshold ({heuristic_result.confidence:.2f} < {self.confidence_threshold}), using heuristic suggestion",
                signals=heuristic_result.signals,
            )

        # No heuristic result at all - true fallback
        logger.info(
            f"Query classified via fallback", extra={"route": self.default_route.value}
        )
        return RouterResult(
            route=self.default_route,
            confidence=0.5,
            method="fallback",
            reasoning="Could not classify with high confidence, using default route",
        )

    def _classify_heuristic_v2(self, query: str) -> Optional[RouterResult]:
        """
        Classify using the new multi-layer heuristic engine.

        Args:
            query: Normalized query string

        Returns:
            RouterResult if classification is confident enough, None otherwise
        """
        if not self.heuristic_engine:
            return None

        try:
            # Analyze query through all layers
            signals = self.heuristic_engine.analyze(query)

            # Get suggested route from signals
            suggested_route = signals.get_suggested_route()
            confidence = signals.get_confidence()

            # Map to RouteType
            route_map = {
                "chat": RouteType.CHAT,
                "rag": RouteType.RAG,
                "web": RouteType.WEB,
            }
            route = route_map.get(suggested_route, self.default_route)

            # Build reasoning from trace
            reasoning_parts = []

            if signals.flags.greeting_detected:
                reasoning_parts.append("greeting detected")
            if signals.flags.farewell_detected:
                reasoning_parts.append("farewell detected")
            if signals.flags.temporal_query:
                reasoning_parts.append("temporal marker found")
            if signals.flags.internal_reference:
                reasoning_parts.append("internal document reference")
            if signals.flags.policy_sensitive:
                reasoning_parts.append("policy-sensitive content")

            # Add top capability
            cap_name, cap_score = signals.capabilities.get_dominant_capability()
            if cap_score > 0:
                reasoning_parts.append(
                    f"dominant capability: {cap_name}={cap_score:.2f}"
                )

            # Add negations if any
            if signals.negations_applied:
                reasoning_parts.append(
                    f"negations: {', '.join(signals.negations_applied[:2])}"
                )

            reasoning = (
                "; ".join(reasoning_parts)
                if reasoning_parts
                else "signal-based classification"
            )

            # Include full signals in debug mode or for tracing
            signals_dict = {
                "flags": signals.flags.to_dict(),
                "capabilities": signals.capabilities.to_dict(),
                "ambiguity": signals.ambiguity,
                "processing_time_ms": signals.processing_time_ms,
            }

            return RouterResult(
                route=route,
                confidence=confidence,
                method="heuristic_v2",
                reasoning=reasoning,
                signals=signals_dict,
            )

        except Exception as e:
            logger.error(f"Heuristic v2 classification error: {e}")
            return None

    async def _classify_with_llm(self, query: str) -> Optional[RouterResult]:
        """
        Classify query using LLM.

        Args:
            query: Query string to classify

        Returns:
            RouterResult from LLM classification, or None if failed
        """
        if not self.llm_module:
            return None

        prompt = self.CLASSIFICATION_PROMPT.format(query=query)

        try:
            # Call LLM for classification
            response = await self.llm_module.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.1,  # Low temperature for consistent classification
            )

            # Parse response
            response_text = (
                response.get("text", "")
                if isinstance(response, dict)
                else str(response)
            )

            # Extract JSON from response (handle potential markdown wrapping)
            json_match = re.search(r"\{[^}]+\}", response_text)
            if not json_match:
                logger.warning(
                    f"Could not extract JSON from LLM response: {response_text[:200]}"
                )
                return None

            result_data = json.loads(json_match.group())

            # Validate and convert route
            route_str = result_data.get("route", "").upper()
            if route_str not in ["CHAT", "RAG", "WEB"]:
                logger.warning(f"Invalid route from LLM: {route_str}")
                return None

            route = RouteType(route_str.lower())
            confidence = float(result_data.get("confidence", 0.5))
            reasoning = result_data.get("reasoning", "")

            return RouterResult(
                route=route,
                confidence=min(confidence, 1.0),  # Cap at 1.0
                method="llm",
                reasoning=reasoning,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM classification response: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return None

    # ========================================================================
    # SMART FALLBACK SYSTEM (v2.0)
    # ========================================================================

    def _is_confident_result(
        self, route_result: RouterResult, rag_sources_count: int = 0
    ) -> bool:
        """
        Determine if a routing result should be trusted.

        This method prevents blind trust in RAG when:
        - RAG was selected but returned 0 sources
        - Confidence is below threshold
        - Signals don't strongly support the route

        FIX-BUG-003 v1.8.3: More conservative fallback logic
        - High confidence RAG (>= 0.85) should NOT trigger fallback even if 0 sources
        - This prevents overriding correct routing when Qdrant has issues
        - internal_knowledge signals should be respected

        Args:
            route_result: The classification result to evaluate
            rag_sources_count: Number of sources found (for RAG validation)

        Returns:
            True if result should be trusted, False if fallback needed
        """
        # Always trust high-confidence results with explicit signals
        if route_result.confidence >= 0.9:
            return True

        # For RAG route, check if we actually have sources
        if route_result.route == RouteType.RAG:
            # FIX-BUG-003 v1.8.3: Check for strong RAG signals before triggering fallback
            # If heuristic strongly suggests RAG (internal_knowledge, policy_sensitive),
            # we should NOT override just because sources are empty
            has_strong_rag_signals = False
            if route_result.signals:
                caps = route_result.signals.get("capabilities", {})
                flags = route_result.signals.get("flags", {})
                # Strong RAG indicators
                internal_score = caps.get("internal_knowledge", 0)
                technical_score = caps.get("technical_depth", 0)
                has_internal_ref = flags.get("internal_reference", False)
                has_policy = flags.get("policy_sensitive", False)

                # If any strong signal is present, don't trigger fallback
                if (
                    internal_score >= 0.7
                    or technical_score >= 0.7
                    or has_internal_ref
                    or has_policy
                ):
                    has_strong_rag_signals = True
                    logger.debug(
                        "RAG route with strong signals - keeping despite 0 sources",
                        extra={
                            "internal_knowledge": internal_score,
                            "technical_depth": technical_score,
                            "internal_reference": has_internal_ref,
                            "policy_sensitive": has_policy,
                        },
                    )

            # FIX-BUG-003: Only trigger fallback if:
            # 1. No sources found AND
            # 2. empty_rag_triggers_web is enabled AND
            # 3. Confidence is below 0.85 (not a strong RAG classification) AND
            # 4. No strong RAG signals are present
            if rag_sources_count == 0 and self.router_config.empty_rag_triggers_web:
                if route_result.confidence < 0.85 and not has_strong_rag_signals:
                    logger.debug(
                        "RAG route with 0 sources and low confidence - triggering fallback",
                        extra={
                            "confidence": route_result.confidence,
                            "has_strong_signals": has_strong_rag_signals,
                        },
                    )
                    return False
                else:
                    logger.info(
                        "RAG route with 0 sources but high confidence or strong signals - NOT triggering fallback",
                        extra={
                            "confidence": route_result.confidence,
                            "has_strong_signals": has_strong_rag_signals,
                        },
                    )

        # For WEB route, always trust (worst case: search returns results)
        if route_result.route == RouteType.WEB:
            return True

        # For CHAT route, trust if greeting/farewell detected
        if route_result.route == RouteType.CHAT:
            if route_result.signals:
                flags = route_result.signals.get("flags", {})
                if flags.get("greeting_detected") or flags.get("farewell_detected"):
                    return True
            # Low confidence CHAT without clear signals - might be ambiguous
            if route_result.confidence < self.confidence_threshold:
                return False

        # Default: trust if above threshold
        return route_result.confidence >= self.confidence_threshold

    def _determine_fallback_route(
        self, original_route: RouteType, signals: Optional[Dict[str, Any]] = None
    ) -> RouteType:
        """
        Determine the best fallback route based on context.

        v2.1 EXPLICIT WEB MODE:
        - WEB is NEVER a fallback unless explicitly requested
        - RAG failed → stay RAG (try different collections or return no-results)
        - WEB failed → try RAG
        - CHAT failed → try RAG
        - Unknown → RAG (conservative default)

        Args:
            original_route: The route that failed/was uncertain
            signals: Heuristic signals for context-aware fallback

        Returns:
            Best fallback RouteType
        """
        strategy = self.router_config.fallback_strategy

        # CONSERVATIVE: Just use default (RAG)
        if strategy == SmartFallbackStrategy.CONSERVATIVE:
            return self.router_config.default_route_for_unknown

        # AGGRESSIVE: Try opposite but WEB only if explicit
        if strategy == SmartFallbackStrategy.AGGRESSIVE:
            if original_route == RouteType.RAG:
                # v2.1: Don't fallback to WEB automatically
                # Check if explicit web was requested
                if signals:
                    flags = signals.get("flags", {})
                    if flags.get("explicit_source") and flags.get("explicit_source_type") == "web":
                        return RouteType.WEB
                return RouteType.RAG  # Stay on RAG
            elif original_route == RouteType.WEB:
                return RouteType.RAG
            else:  # CHAT
                return RouteType.RAG  # v2.1: RAG not WEB

        # SMART: Context-aware fallback (v2.1: WEB only if explicit)
        if signals:
            flags = signals.get("flags", {})

            # v2.1: ONLY route to WEB if explicitly requested
            if flags.get("explicit_source") and flags.get("explicit_source_type") == "web":
                return RouteType.WEB

            # If there's internal reference, prefer RAG
            if flags.get("internal_reference"):
                return RouteType.RAG

        # v2.1: Default fallback is always RAG (never automatic WEB)
        return RouteType.RAG

    def _get_fallback_chain(
        self, original_route: RouteType, signals: Optional[Dict[str, Any]] = None
    ) -> List[RouteType]:
        """
        Get ordered list of fallback routes to try.

        Args:
            original_route: The primary route that needs fallback
            signals: Heuristic signals for ordering

        Returns:
            List of RouteType to try in order (excludes original)
        """
        all_routes = [RouteType.RAG, RouteType.WEB, RouteType.CHAT]
        chain = []

        # First fallback: determined by strategy
        first_fallback = self._determine_fallback_route(original_route, signals)
        if first_fallback != original_route:
            chain.append(first_fallback)

        # Add remaining routes (excluding original and first fallback)
        for route in all_routes:
            if route != original_route and route not in chain:
                # Don't add CHAT as fallback (it won't have info)
                if route != RouteType.CHAT:
                    chain.append(route)

        # Limit to max_retry_attempts
        return chain[: self.router_config.max_retry_attempts]

    def get_smart_fallback(
        self, route_result: RouterResult, rag_sources_count: int = 0
    ) -> Optional[RouterResult]:
        """
        Get fallback route if original result is not confident.

        This is the main entry point for the Smart Fallback system.
        Called by adapter.py after receiving empty results.

        Args:
            route_result: Original routing result
            rag_sources_count: Sources found (0 triggers fallback for RAG)

        Returns:
            New RouterResult with fallback route, or None if confident
        """
        if not self.router_config.enable_auto_retry:
            return None

        if self._is_confident_result(route_result, rag_sources_count):
            return None

        # Get fallback route
        fallback_chain = self._get_fallback_chain(
            route_result.route, route_result.signals
        )

        if not fallback_chain:
            return None

        fallback_route = fallback_chain[0]
        self.auto_retry_count += 1

        logger.info(
            f"Smart fallback triggered: {route_result.route.value} → {fallback_route.value}",
            extra={
                "original_route": route_result.route.value,
                "fallback_route": fallback_route.value,
                "original_confidence": route_result.confidence,
                "rag_sources": rag_sources_count,
            },
        )

        return RouterResult(
            route=fallback_route,
            confidence=0.6,  # Lower confidence for fallback
            method="smart_fallback",
            reasoning=f"Fallback from {route_result.route.value} (sources={rag_sources_count})",
            signals=route_result.signals,
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get router statistics.

        Returns:
            Dictionary with classification statistics
        """
        stats = {
            "total_classifications": self.total_classifications,
            "heuristic_v2_hits": self.heuristic_v2_hits,
            "llm_classifications": self.llm_classifications,
            "fallback_count": self.fallback_count,
            "auto_retry_count": self.auto_retry_count,
            "heuristic_v2_rate": (
                self.heuristic_v2_hits / self.total_classifications * 100
                if self.total_classifications > 0
                else 0
            ),
            "llm_rate": (
                self.llm_classifications / self.total_classifications * 100
                if self.total_classifications > 0
                else 0
            ),
            "fallback_rate": (
                self.fallback_count / self.total_classifications * 100
                if self.total_classifications > 0
                else 0
            ),
            "auto_retry_rate": (
                self.auto_retry_count / self.total_classifications * 100
                if self.total_classifications > 0
                else 0
            ),
        }

        # Add heuristic engine stats
        if self.heuristic_engine:
            stats["heuristic_engine"] = self.heuristic_engine.get_stats()

        return stats

    def reset_stats(self) -> None:
        """Reset classification statistics."""
        self.total_classifications = 0
        self.heuristic_v1_hits = 0
        self.heuristic_v2_hits = 0
        self.llm_classifications = 0
        self.fallback_count = 0
        self.auto_retry_count = 0

        if self.heuristic_engine:
            self.heuristic_engine.reset_stats()

        logger.info("SemanticRouter statistics reset")

    def get_engine_trace(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get full trace from heuristic engine for debugging.

        Args:
            query: Query to analyze

        Returns:
            Full signals dict with trace, or None if engine not available
        """
        if not self.heuristic_engine:
            return None

        try:
            signals = self.heuristic_engine.analyze(query)
            return signals.to_dict()
        except Exception as e:
            logger.error(f"Failed to get engine trace: {e}")
            return None


# Export public API
__all__ = [
    "SemanticRouter",
    "RouteType",
    "RouterResult",
    "RouterConfig",
    "SmartFallbackStrategy",
]
