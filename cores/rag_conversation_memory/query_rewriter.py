"""
Query Rewriter - Memory-Aware Query Expansion for RAG Retrieval

Solves the "spiega meglio" problem: when a user sends a vague/continuation
query, the retriever needs an expanded query to find relevant documents.

Architecture:
- Pre-computation: After each eager compression, generate retrieval_hints
  and cache them in Redis (ready BEFORE the next query arrives)
- Query-time: Detect vague queries, fetch cached hints, expand the query

Integration point: Called by RAG Orchestrator BEFORE the retriever.

Flow:
  1. User sends "spiega meglio"
  2. Orchestrator calls query_rewriter.rewrite(session_id, raw_query)
  3. Rewriter detects vague query → fetches cached hints from Redis
  4. Returns expanded query: "UBP Enterprise Hybrid architecture modular
     plug-and-play hot-reload dependency injection 3-File Pattern"
  5. Retriever uses expanded query → finds relevant chunks
  6. LLM generates proper response with real documentation

v1.0.0 - FEAT-MEM-003: Query Rewriting
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import json
import logging
import re

logger = logging.getLogger(__name__)


# =============================================================================
# VAGUE QUERY DETECTION
# =============================================================================

# Continuation patterns (IT + EN)
CONTINUATION_PATTERNS_IT = re.compile(
    r"(?i)^(spiega(mi)?\s*(meglio|di più|ancora|nel dettaglio)?|"
    r"dimmi di più|"
    r"vai avanti|"
    r"continua|"
    r"entra nei dettagli|"
    r"più in dettaglio|"
    r"scendi nel dettaglio|"
    r"e poi\??|"
    r"altro\??|"
    r"cos'altro\??|"
    r"in che senso\??|"
    r"cioè\??|"
    r"fammi capire meglio|"
    r"puoi essere più specifico|"
    r"come funziona|"        # vago solo se corto
    r"perché\??)$"
)

CONTINUATION_PATTERNS_EN = re.compile(
    r"(?i)^(explain\s*(more|better|further|in detail)?|"
    r"tell me more|"
    r"go on|"
    r"continue|"
    r"what else\??|"
    r"and then\??|"
    r"how does (it|that) work\??|"
    r"what do you mean\??|"
    r"can you be more specific\??|"
    r"why\??)$"
)

# Referential patterns (use memory to resolve "it", "that", "this")
REFERENTIAL_PATTERNS = re.compile(
    r"(?i)\b(come funziona (questo|quello|ciò)|"
    r"how does (this|that|it) work|"
    r"tell me about (this|that|it)|"
    r"parlami di (questo|quello)|"
    r"what about (this|that|it))\b"
)

# Max length for a query to be considered potentially vague
VAGUE_MAX_LENGTH = 60


# =============================================================================
# RETRIEVAL HINTS (pre-computed after compression)
# =============================================================================

class RetrievalHints:
    """
    Pre-computed retrieval hints cached in Redis after each compression.

    Contains everything needed to expand a vague query WITHOUT
    accessing the full MemoryState at query time.
    """

    def __init__(
        self,
        primary_query: str,
        focus_keywords: List[str],
        entity_keywords: List[str],
        continuation_query: str,
        deepdive_query: str,
        intent: str,
        current_focus: str,
        hold_focus: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        self.primary_query = primary_query        # Best single query for retrieval
        self.focus_keywords = focus_keywords       # Keywords from current focus
        self.entity_keywords = entity_keywords     # Top entities
        self.continuation_query = continuation_query  # For "spiega meglio" type
        self.deepdive_query = deepdive_query       # For "approfondisci" type
        self.intent = intent
        self.current_focus = current_focus
        self.hold_focus = hold_focus
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def to_json(self) -> str:
        return json.dumps({
            "primary_query": self.primary_query,
            "focus_keywords": self.focus_keywords,
            "entity_keywords": self.entity_keywords,
            "continuation_query": self.continuation_query,
            "deepdive_query": self.deepdive_query,
            "intent": self.intent,
            "current_focus": self.current_focus,
            "hold_focus": self.hold_focus,
            "timestamp": self.timestamp,
        })

    @classmethod
    def from_json(cls, json_str: str) -> "RetrievalHints":
        data = json.loads(json_str)
        return cls(**data)


# =============================================================================
# HINTS BUILDER (called during eager compression)
# =============================================================================

class HintsBuilder:
    """
    Builds RetrievalHints from MemoryState.

    Called AFTER each eager compression in _on_message_added.
    Purely deterministic — no LLM call needed.
    """

    @staticmethod
    def build_from_state(state) -> RetrievalHints:
        """
        Extract retrieval hints from a MemoryState.

        Args:
            state: MemoryState instance (after compression)

        Returns:
            RetrievalHints ready for caching
        """
        current_focus = state.current_focus or ""
        hold_focus = state.hold_focus

        # --- Extract keywords from current focus entry ---
        focus_keywords = []
        current_entry = None
        for turn in state.conversation_thread:
            if turn.focus == current_focus:
                current_entry = turn
                break

        if current_entry:
            # Extract meaningful words from focus label
            focus_keywords = HintsBuilder._extract_keywords(current_entry.focus)

            # Add top keywords from key_facts
            facts_keywords = HintsBuilder._extract_keywords(current_entry.key_facts)
            focus_keywords.extend(facts_keywords[:10])

        # --- Extract entity keywords ---
        entity_keywords = []
        if state.structured_context and state.structured_context.entities:
            for key, value in state.structured_context.entities.items():
                if isinstance(value, list):
                    entity_keywords.extend(str(v) for v in value[:5])
                elif isinstance(value, str):
                    entity_keywords.append(value)

        # Deduplicate preserving order
        seen = set()
        unique_entities = []
        for e in entity_keywords:
            e_lower = e.lower()
            if e_lower not in seen and len(e) > 2:
                seen.add(e_lower)
                unique_entities.append(e)
        entity_keywords = unique_entities[:15]

        # --- Build primary query ---
        # Best single query: focus label + top 3 entities
        primary_parts = [current_focus]
        primary_parts.extend(entity_keywords[:3])
        primary_query = " ".join(primary_parts).strip()

        # --- Build continuation query ---
        # For "spiega meglio": focus + key technical terms
        continuation_parts = [current_focus]
        if current_entry and current_entry.key_facts:
            # Extract technical terms (capitalized words, acronyms)
            tech_terms = HintsBuilder._extract_technical_terms(
                current_entry.key_facts
            )
            continuation_parts.extend(tech_terms[:5])
        continuation_query = " ".join(continuation_parts).strip()

        # --- Build deepdive query ---
        # For "approfondisci": more specific, include last query context
        deepdive_parts = []
        if current_entry and current_entry.query:
            deepdive_parts.append(current_entry.query[:200])
        else:
            deepdive_parts.append(current_focus)
        deepdive_parts.extend(focus_keywords[:5])
        deepdive_query = " ".join(deepdive_parts).strip()

        # --- Intent ---
        intent = "general"
        if state.structured_context:
            intent = state.structured_context.intent or "general"

        return RetrievalHints(
            primary_query=primary_query,
            focus_keywords=focus_keywords,
            entity_keywords=entity_keywords,
            continuation_query=continuation_query,
            deepdive_query=deepdive_query,
            intent=intent,
            current_focus=current_focus,
            hold_focus=hold_focus,
        )

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        if not text:
            return []

        # Remove common stop words (IT + EN minimal set)
        stop_words = {
            "il", "lo", "la", "i", "gli", "le", "un", "una", "di", "del",
            "della", "dei", "degli", "delle", "a", "al", "alla", "in", "nel",
            "nella", "con", "da", "per", "su", "e", "o", "ma", "che", "è",
            "sono", "come", "questo", "quello", "più",
            "the", "a", "an", "of", "in", "on", "at", "to", "for", "and",
            "or", "but", "is", "are", "was", "were", "with", "from", "by",
            "this", "that", "it", "as", "be", "has", "have", "not", "no",
        }

        # Split and filter
        words = re.findall(r'\b[a-zA-Z_\-]{3,}\b', text)
        keywords = [w for w in words if w.lower() not in stop_words]

        # Deduplicate preserving order
        seen = set()
        unique = []
        for w in keywords:
            w_lower = w.lower()
            if w_lower not in seen:
                seen.add(w_lower)
                unique.append(w)

        return unique

    @staticmethod
    def _extract_technical_terms(text: str) -> List[str]:
        """Extract technical/proper terms (capitalized, acronyms, compound)."""
        terms = []

        # Acronyms (2+ uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,}[a-z]*\b', text)
        terms.extend(acronyms)

        # CamelCase / PascalCase
        camel = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
        terms.extend(camel)

        # Hyphenated compounds (e.g. "plug-and-play", "event-driven")
        compounds = re.findall(r'\b\w+-\w+(?:-\w+)*\b', text)
        terms.extend(compounds)

        # Deduplicate
        seen = set()
        unique = []
        for t in terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)

        return unique


# =============================================================================
# QUERY REWRITER (called at query time by orchestrator)
# =============================================================================

class QueryRewriter:
    """
    Rewrites user queries using pre-cached retrieval hints.

    Called by the orchestrator BEFORE sending the query to the retriever.
    Fast path: no LLM calls, only Redis cache lookup + deterministic logic.
    """

    # Redis key for cached hints
    HINTS_CACHE_KEY = "ubp:memory:session:{session_id}:retrieval_hints"

    def __init__(self, redis_client=None):
        self.redis = redis_client

    async def rewrite(
        self,
        session_id: str,
        raw_query: str,
        memory_state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Rewrite a query for optimal retrieval.

        Args:
            session_id: Current session ID
            raw_query: Raw user query
            memory_state: Optional MemoryState (if already loaded)

        Returns:
            Dict with:
                - query: The rewritten query (or original if no rewrite needed)
                - original_query: The raw query
                - rewrite_type: "none" | "continuation" | "deepdive" |
                                "referential" | "enriched"
                - hints_used: bool
                - metadata: Additional rewrite metadata
        """
        query_stripped = raw_query.strip()

        # Step 1: Classify query type
        query_type = self._classify_query(query_stripped)

        if query_type == "specific":
            # Query is specific enough — optionally enrich with entities
            enriched = await self._enrich_specific_query(
                session_id, query_stripped
            )
            return enriched

        # Step 2: Fetch cached hints
        hints = await self._get_cached_hints(session_id)

        if not hints:
            # No hints available — return original with warning
            logger.warning(
                f"[REWRITER] No cached hints for session {session_id}, "
                f"returning raw query"
            )
            return {
                "query": query_stripped,
                "original_query": raw_query,
                "rewrite_type": "none",
                "hints_used": False,
                "metadata": {"reason": "no_hints_cached"},
            }

        # Step 3: Expand based on query type
        if query_type == "continuation":
            expanded = f"{query_stripped} {hints.current_focus}"
            rewrite_type = "continuation"
        elif query_type == "deepdive":
            deepdive_q = getattr(hints, "deepdive_query", None)
            if deepdive_q and deepdive_q.strip():
                expanded = f"{query_stripped} {deepdive_q}"
            else:
                expanded = f"{query_stripped} {hints.current_focus}"
            rewrite_type = "deepdive"
        elif query_type == "referential":
            expanded = self._resolve_referential(query_stripped, hints)
            rewrite_type = "referential"
        else:
            expanded = hints.primary_query
            rewrite_type = "fallback"

        logger.info(
            f"[REWRITER] Rewrote query: '{query_stripped}' -> '{expanded[:100]}...' "
            f"(type={rewrite_type}, focus={hints.current_focus})"
        )

        return {
            "query": expanded,
            "original_query": raw_query,
            "rewrite_type": rewrite_type,
            "hints_used": True,
            "metadata": {
                "current_focus": hints.current_focus,
                "hold_focus": hints.hold_focus,
                "intent": hints.intent,
                "hints_timestamp": hints.timestamp,
            },
        }

    def _classify_query(self, query: str) -> str:
        """
        Classify query into types.

        Returns:
            "continuation" - vague continuation (spiega meglio, tell me more)
            "deepdive" - explicit deepdive request (approfondisci, elaborate)
            "referential" - contains unresolved references (this, that, it)
            "specific" - self-contained, specific query
        """
        # Length check first
        if len(query) > VAGUE_MAX_LENGTH:
            # Long queries are usually specific
            return "specific"

        # Check continuation patterns
        if CONTINUATION_PATTERNS_IT.match(query):
            return "continuation"
        if CONTINUATION_PATTERNS_EN.match(query):
            return "continuation"

        # Check deepdive-specific patterns
        deepdive_it = re.match(
            r"(?i)^(approfondisci|elabora|più dettagli|nel dettaglio)",
            query
        )
        deepdive_en = re.match(
            r"(?i)^(elaborate|more details|in detail|deep dive)",
            query
        )
        if deepdive_it or deepdive_en:
            return "deepdive"

        # Check referential
        if REFERENTIAL_PATTERNS.search(query):
            return "referential"

        # Short queries with few content words are likely vague
        _stop = {
            "come", "cosa", "what", "how", "per", "con", "non", "the",
            "for", "and", "che", "del", "nel", "gli", "una", "uno",
        }
        content_words = [
            w for w in query.split()
            if len(w) >= 3 and w.lower() not in _stop
        ]
        if len(content_words) <= 1 and len(query) < 30:
            return "continuation"

        return "specific"

    async def _get_cached_hints(
        self, session_id: str
    ) -> Optional[RetrievalHints]:
        """Fetch pre-computed hints from Redis cache."""
        if not self.redis:
            return None

        try:
            key = self.HINTS_CACHE_KEY.format(session_id=session_id)
            cached = await self.redis.get(key)
            if cached is None:
                return None
            if isinstance(cached, bytes):
                cached = cached.decode("utf-8")
            return RetrievalHints.from_json(cached)
        except Exception as e:
            logger.warning(f"[REWRITER] Failed to fetch hints: {e}")
            return None

    async def _enrich_specific_query(
        self, session_id: str, query: str
    ) -> Dict[str, Any]:
        """
        Optionally enrich a specific query with entity context.

        For queries like "come funziona l'hot-reload", adds relevant
        entities to improve retrieval without changing the intent.
        """
        hints = await self._get_cached_hints(session_id)

        # Always expose hints metadata for downstream consumers (e.g. web search enrichment)
        _hints_meta = {}
        if hints:
            _hints_meta = {
                "current_focus": hints.current_focus,
                "primary_query": hints.primary_query,
                "focus_keywords": hints.focus_keywords,
            }

        if not hints or not hints.entity_keywords:
            return {
                "query": query,
                "original_query": query,
                "rewrite_type": "none",
                "hints_used": False,
                "metadata": _hints_meta,
            }

        # Only add entities that are related to the query
        query_lower = query.lower()
        relevant_entities = [
            e for e in hints.entity_keywords
            if any(
                word in query_lower
                for word in e.lower().split("_")
                if len(word) > 3
            )
        ]

        if relevant_entities:
            enriched = f"{query} {' '.join(relevant_entities[:3])}"
            return {
                "query": enriched,
                "original_query": query,
                "rewrite_type": "enriched",
                "hints_used": True,
                "metadata": {
                    "added_entities": relevant_entities[:3],
                    "current_focus": hints.current_focus,
                },
            }

        return {
            "query": query,
            "original_query": query,
            "rewrite_type": "none",
            "hints_used": False,
            "metadata": _hints_meta,
        }

    def _resolve_referential(
        self, query: str, hints: RetrievalHints
    ) -> str:
        """
        Resolve referential pronouns using memory context.

        "come funziona questo" → "come funziona UBP Enterprise Hybrid hot-reload"
        """
        # Replace pronouns with current focus
        resolved = query
        replacements = {
            "questo": hints.current_focus,
            "quello": hints.current_focus,
            "ciò": hints.current_focus,
            "this": hints.current_focus,
            "that": hints.current_focus,
            "it": hints.current_focus,
        }

        for pronoun, replacement in replacements.items():
            pattern = rf"\b{pronoun}\b"
            if re.search(pattern, resolved, re.IGNORECASE):
                resolved = re.sub(
                    pattern, replacement, resolved, flags=re.IGNORECASE
                )
                break  # Replace only the first pronoun

        # Add some entity context
        if hints.entity_keywords:
            resolved += " " + " ".join(hints.entity_keywords[:2])

        return resolved.strip()

    # =========================================================================
    # CACHE MANAGEMENT (called from adapter after compression)
    # =========================================================================

    async def cache_hints(
        self,
        session_id: str,
        hints: RetrievalHints,
        ttl_seconds: int = 86400 * 90,
    ) -> None:
        """
        Cache retrieval hints in Redis.

        Called by adapter._on_message_added AFTER eager compression.
        """
        if not self.redis:
            return

        try:
            key = self.HINTS_CACHE_KEY.format(session_id=session_id)
            await self.redis.set(key, hints.to_json())
            await self.redis.expire(key, ttl_seconds)
            logger.debug(
                f"[REWRITER] Cached hints for session {session_id}: "
                f"focus={hints.current_focus}"
            )
        except Exception as e:
            logger.warning(f"[REWRITER] Failed to cache hints: {e}")

    async def invalidate_hints(self, session_id: str) -> None:
        """Invalidate cached hints (e.g., on session clear)."""
        if not self.redis:
            return
        try:
            key = self.HINTS_CACHE_KEY.format(session_id=session_id)
            await self.redis.delete(key)
        except Exception as e:
            logger.warning(f"[REWRITER] Failed to invalidate hints: {e}")
