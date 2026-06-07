"""
Filter RAG Context — Pure Filtering Logic

Zero UBP dependencies. Zero I/O. Deterministic.
Testable with pytest standalone.

Pipeline:
1. Hard-drop (garbage, too short, score < hard threshold)
2. Soft-penalty (boilerplate, truncation, near-dup hash)
3. Keyword bonus (query terms found in chunk)
4. Score: (base + bonuses) * (1 - min(0.9, penalties))
4b. Low-confidence recalculation (halve penalties, double bonuses)
5. Distribution guard (all low relevance flag)
6. Sort by final_score desc (stable)
7. Diversity cap per source
8. Max chunks + max chars
9. G2 guarantee: recover up to min_output_guarantee chunks
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Stopwords for keyword extraction (tokens ≥4 chars that are still noise)
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    "come", "cosa", "sono", "della", "degli", "delle", "dello", "nella",
    "nelle", "negli", "nello", "questo", "questa", "questi", "queste",
    "quale", "quali", "anche", "ancora", "altro", "altri", "altre",
    "tutto", "tutti", "tutte", "ogni", "dopo", "prima", "durante",
    "senza", "sopra", "sotto", "circa", "molto", "poco", "tanto",
    "what", "that", "this", "with", "from", "have", "been", "will",
    "about", "their", "there", "would", "could", "should", "which",
    "these", "those", "some", "more", "than", "them", "then", "into",
    "over", "such", "only", "also", "most", "just", "when", "were",
})

# ---------------------------------------------------------------------------
# Boilerplate detection patterns
# ---------------------------------------------------------------------------
_BOILERPLATE_PATTERNS = [
    re.compile(r"^Table of Contents\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Index\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Chapter\s+\d+\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\.{5,}"),           # lines of dots
    re.compile(r"_{5,}"),            # lines of underscores
    re.compile(r"-{5,}"),            # lines of dashes
    re.compile(r"^\d+\.\s+\w+\s*\.{3,}\s*\d+$", re.MULTILINE),  # ToC entries
]

# ---------------------------------------------------------------------------
# Domain patterns for collection affinity routing
# ---------------------------------------------------------------------------
_DOMAIN_PATTERNS = {
    "medical": {
        "positive": {
            "metformina", "diabete", "paracetamolo", "farmaco",
            "effetti collaterali", "dosaggio", "sintomi", "diagnosi",
            "terapia", "paziente", "glicemia", "insulina", "tachicardia",
            "ipoglicemia", "controindicazioni", "pressione", "colesterolo",
            "antibiotico", "salute",
        },
        "negative": {"art.", "comma", "decreto", "legge", "contratto", "tribunale"},
        "pharma_keywords": {
            "metformina", "paracetamolo", "insulina", "warfarin",
            "ibuprofene", "aspirina", "antibiotico",
        },
        "collections": ["medical"],
        "boost": 3.0,
        "min_matches": 1,
    },
    "legal": {
        "positive": {
            "gdpr", "privacy", "normativa", "decreto", "legge",
            "articolo", "comma", "regolamento", "contratto",
            "responsabilità", "tribunale", "sanzione", "trattamento dati",
        },
        "negative": {"farmaco", "dosaggio", "sintomi", "paziente"},
        "collections": ["legal"],
        "boost": 2.0,
        "min_matches": 1,
    },
    "technical": {
        "positive": {
            "ubp", "pipeline", "modulo", "api", "config", "qdrant",
            "redis", "docker", "deploy", "inference", "embedding", "vllm",
        },
        "negative": set(),
        "collections": ["ubp_system_docs"],
        "boost": 1.8,
        "min_matches": 1,
    },
}

# ---------------------------------------------------------------------------
# Entity synonyms for boost_by_entity()
# ---------------------------------------------------------------------------
ENTITY_SYNONYMS = {
    "paracetamolo": ["acetaminofene", "tachipirina", "paracetamol"],
    "metformina": ["metformin", "glucophage"],
    "insulina": ["insulin"],
    "ibuprofene": ["ibuprofen", "brufen"],
    "aspirina": ["acido acetilsalicilico", "aspirin"],
    "warfarin": ["coumadin"],
}


# =========================================================================
# Data classes
# =========================================================================

@dataclass
class FilterConfig:
    """Tunable thresholds for the filter pipeline."""
    min_chars_hard: int = 30
    min_score_hard: float = 0.06
    min_score_soft: float = 0.13
    min_chars_soft: int = 80
    boilerplate_penalty: float = 0.25
    truncation_penalty: float = 0.15
    keyword_bonus: float = 0.20
    relevance_floor: float = 0.10
    diversity_cap_per_source: int = 3
    max_chunks: int = 10
    max_total_chars: int = 12000
    min_output_guarantee: int = 2
    low_confidence_threshold: float = 0.35
    shadow_mode: bool = True


@dataclass
class ChunkVerdict:
    """Per-chunk filtering decision."""
    chunk_id: str
    action: str          # "KEEP" | "HARD_DROP" | "SOFT_PENALTY"
    final_score: float
    reasons: List[str]
    penalties: float
    bonuses: float
    base_score: float = 0.0


@dataclass
class FilterStats:
    """Aggregate statistics for a filter run."""
    input_count: int = 0
    output_count: int = 0
    dropped_by_reason: Dict[str, int] = field(default_factory=dict)
    avg_score_kept: float = 0.0
    avg_score_dropped: float = 0.0
    all_low_relevance: bool = False
    fallback_triggered: bool = False
    sources_in: int = 0
    sources_out: int = 0


@dataclass
class FilterResult:
    """Complete result of a filter run."""
    kept: List[ChunkVerdict]
    dropped: List[ChunkVerdict]
    stats: FilterStats
    fallback_triggered: bool


# =========================================================================
# Internal helpers
# =========================================================================

def _detect_boilerplate(text: str) -> bool:
    """Return True if text matches any boilerplate pattern."""
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(text):
            return True
    return False


def _detect_truncation(text: str) -> bool:
    """Return True if text appears truncated (starts lowercase or ends without punctuation)."""
    stripped = text.strip()
    if not stripped:
        return False
    starts_lower = stripped[0].islower()
    ends_no_punct = stripped[-1] not in ".!?;:)\"]'"
    return starts_lower or ends_no_punct


def _text_hash(text: str, length: int = 100) -> str:
    """Hash first N chars of text for near-duplicate detection."""
    prefix = text[:length].strip().lower()
    return hashlib.md5(prefix.encode("utf-8")).hexdigest()


def _extract_query_tokens(query: str) -> List[str]:
    """Extract meaningful tokens from query (≥4 chars, not stopwords)."""
    tokens = re.findall(r"[a-zA-ZàèéìòùÀÈÉÌÒÙ]+", query.lower())
    return [t for t in tokens if len(t) >= 4 and t not in _STOPWORDS]


# =========================================================================
# Collection affinity routing
# =========================================================================

def get_collection_affinity(
    query: str,
    available_collections: List[str],
    user_selected: Optional[List[str]] = None,
) -> List[str]:
    """
    Pure function. Returns prioritized collection list.

    Rules:
    1. user_selected non vuoto → return user_selected (bypass)
    2. Score per domain: pos_matches * boost - neg_matches * 0.8
       + pharma_bonus (+1.0 if any pharma_keyword matches)
    3. score > 0 for at least 1 domain → return matched collections
       sorted by score desc + 1 neutral backup
    4. No match → return available_collections
    5. Never return empty list
    """
    # Rule 1: user bypass
    if user_selected:
        return user_selected

    query_lower = query.lower()
    collection_scores: Dict[str, float] = {}

    for _domain_name, domain in _DOMAIN_PATTERNS.items():
        pos_count = sum(1 for kw in domain["positive"] if kw in query_lower)
        if pos_count < domain["min_matches"]:
            continue
        neg_count = sum(1 for kw in domain.get("negative", set()) if kw in query_lower)
        score = pos_count * domain["boost"] - neg_count * 0.8

        # Pharma bonus
        pharma = domain.get("pharma_keywords", set())
        if pharma and any(pk in query_lower for pk in pharma):
            score += 1.0

        if score > 0:
            for coll in domain["collections"]:
                if coll in available_collections:
                    collection_scores[coll] = max(
                        collection_scores.get(coll, 0), score
                    )

    if not collection_scores:
        return available_collections  # Rule 4: neutral

    # Sorted by score desc
    ranked = sorted(
        collection_scores.keys(),
        key=lambda c: collection_scores[c],
        reverse=True,
    )

    # Add 1 neutral backup (first available not already scored)
    for c in available_collections:
        if c not in collection_scores:
            ranked.append(c)
            break

    return ranked if ranked else available_collections  # Rule 5


# =========================================================================
# Entity boost post-retrieval
# =========================================================================

def boost_by_entity(
    chunks: List[Dict],
    query: str,
    boost_factor: float = 1.5,
    miss_penalty: float = 0.7,
) -> List[Dict]:
    """
    Pure function. Modify rerank_score based on entity presence in chunk text.

    Rules:
    1. Extract entity keywords from query (all domain positive + pharma keywords)
    2. Expand with ENTITY_SYNONYMS
    3. For each chunk: boost if entity present, penalize if absent
    4. If no entities found in query → return unchanged
    5. Never set score below 0
    """
    query_lower = query.lower()

    # Step 1: Collect all domain keywords (positive + pharma)
    all_domain_keywords: set = set()
    for domain in _DOMAIN_PATTERNS.values():
        all_domain_keywords.update(domain["positive"])
        all_domain_keywords.update(domain.get("pharma_keywords", set()))

    # Step 2: Find which keywords appear in query
    query_entities: set = set()
    for kw in all_domain_keywords:
        if kw in query_lower:
            query_entities.add(kw)

    if not query_entities:
        return chunks  # Rule 4: noop

    # Step 3: Expand with synonyms → search_terms (entity + all synonyms)
    search_terms: set = set(query_entities)
    for entity in query_entities:
        if entity in ENTITY_SYNONYMS:
            search_terms.update(ENTITY_SYNONYMS[entity])
        # Also check if entity is a synonym value → add the canonical + siblings
        for canonical, synonyms in ENTITY_SYNONYMS.items():
            if entity == canonical or entity in synonyms:
                search_terms.add(canonical)
                search_terms.update(synonyms)

    # Step 4: Apply boost/penalty per chunk
    for chunk in chunks:
        text_lower = chunk.get("text", "").lower()
        score = chunk.get("score", 0.0)

        # Find which entity matched (for reason string)
        matched_entity = None
        for term in search_terms:
            if term in text_lower:
                matched_entity = term
                break

        if matched_entity is not None:
            chunk["score"] = max(0, score * boost_factor)
            reasons = chunk.get("_entity_reasons", [])
            reasons.append(f"ENTITY_BOOST:{matched_entity}")
            chunk["_entity_reasons"] = reasons
        else:
            chunk["score"] = max(0, score * miss_penalty)
            reasons = chunk.get("_entity_reasons", [])
            reasons.append("ENTITY_MISS")
            chunk["_entity_reasons"] = reasons

    return chunks


# =========================================================================
# Main filter function
# =========================================================================

def filter_rag_context(
    chunks: List[Dict],
    query: str,
    config: Optional[FilterConfig] = None,
) -> FilterResult:
    """
    Filter and rank RAG context chunks.

    Args:
        chunks: List of dicts with keys: text, score (rerank), cosine_score,
                source_id, chunk_id (all optional except text).
        query: User query string.
        config: Filtering thresholds. Uses defaults if None.

    Returns:
        FilterResult with kept/dropped verdicts, stats, and fallback flag.
    """
    if config is None:
        config = FilterConfig()

    verdicts: List[ChunkVerdict] = []
    seen_hashes: set = set()
    query_tokens = _extract_query_tokens(query)
    n_tokens = len(query_tokens) if query_tokens else 1  # avoid div-by-zero

    stats = FilterStats(input_count=len(chunks))
    sources_seen_in: set = set()

    # ----- Steps 1-4: Score each chunk -----
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        rerank_score: Optional[float] = chunk.get("score")
        cosine_score: Optional[float] = chunk.get("cosine_score")
        source_id: str = chunk.get("source_id", f"src_{i}")
        chunk_id: str = chunk.get("chunk_id", f"chunk_{i}")

        sources_seen_in.add(source_id)
        reasons: List[str] = []
        penalties: float = 0.0
        bonuses: float = 0.0

        # --- Step 1: Hard-drop ---
        stripped = text.strip()
        if not stripped:
            verdicts.append(ChunkVerdict(
                chunk_id=chunk_id, action="HARD_DROP", final_score=0.0,
                reasons=["EMPTY"], penalties=0.0, bonuses=0.0,
            ))
            stats.dropped_by_reason["EMPTY"] = stats.dropped_by_reason.get("EMPTY", 0) + 1
            continue

        if len(stripped) < config.min_chars_hard:
            verdicts.append(ChunkVerdict(
                chunk_id=chunk_id, action="HARD_DROP", final_score=0.0,
                reasons=["TOO_SHORT"], penalties=0.0, bonuses=0.0,
            ))
            stats.dropped_by_reason["TOO_SHORT"] = stats.dropped_by_reason.get("TOO_SHORT", 0) + 1
            continue

        if rerank_score is not None and rerank_score < config.min_score_hard:
            verdicts.append(ChunkVerdict(
                chunk_id=chunk_id, action="HARD_DROP", final_score=rerank_score,
                reasons=["SCORE_HARD"], penalties=0.0, bonuses=0.0,
            ))
            stats.dropped_by_reason["SCORE_HARD"] = stats.dropped_by_reason.get("SCORE_HARD", 0) + 1
            continue

        # --- Step 2: Soft-penalty ---
        if len(stripped) < config.min_chars_soft:
            penalties += 0.15
            reasons.append("SHORT_TEXT")

        if _detect_boilerplate(stripped):
            penalties += config.boilerplate_penalty
            reasons.append("BOILERPLATE")

        if _detect_truncation(stripped):
            penalties += config.truncation_penalty
            reasons.append("TRUNCATION")

        h = _text_hash(stripped)
        if h in seen_hashes:
            penalties += 0.30
            reasons.append("NEAR_DUP")
        seen_hashes.add(h)

        # --- Step 3: Keyword bonus ---
        text_lower = stripped.lower()
        matched = sum(1 for t in query_tokens if t in text_lower)
        if matched > 0:
            bonuses = min(config.keyword_bonus, config.keyword_bonus * matched / n_tokens)
            reasons.append(f"KW_MATCH({matched}/{n_tokens})")

        # --- Step 4: Score ---
        if rerank_score is not None:
            base = rerank_score
        elif cosine_score is not None:
            base = cosine_score * 0.5
        else:
            base = 0.0

        penalty_factor = 1.0 - min(0.9, penalties)
        final_score = (base + bonuses) * penalty_factor

        action = "SOFT_PENALTY" if penalties > 0 else "KEEP"
        verdicts.append(ChunkVerdict(
            chunk_id=chunk_id, action=action, final_score=final_score,
            reasons=reasons if reasons else ["CLEAN"],
            penalties=penalties, bonuses=bonuses, base_score=base,
        ))

    stats.sources_in = len(sources_seen_in)

    # Separate hard-dropped from candidates
    hard_dropped = [v for v in verdicts if v.action == "HARD_DROP"]
    candidates = [v for v in verdicts if v.action != "HARD_DROP"]

    # ----- Step 4b: Low-confidence recalculation -----
    if candidates:
        max_base = max(v.base_score for v in candidates)
        if max_base < config.low_confidence_threshold:
            for v in candidates:
                new_penalties = v.penalties * 0.5
                new_bonuses = min(1.0, v.bonuses * 2.0)
                penalty_factor = 1.0 - min(0.9, new_penalties)
                v.final_score = (v.base_score + new_bonuses) * penalty_factor
                v.penalties = new_penalties
                v.bonuses = new_bonuses
                v.reasons.append("LOW_CONF_BOOST")

    # ----- Step 5: Distribution guard -----
    if candidates:
        max_score = max(v.final_score for v in candidates)
        if max_score < config.relevance_floor:
            stats.all_low_relevance = True

    # ----- Step 6: Sort desc by final_score (stable) -----
    candidates.sort(key=lambda v: v.final_score, reverse=True)

    # ----- Step 7: Diversity cap per source -----
    source_counts: Dict[str, int] = {}
    diverse_candidates: List[ChunkVerdict] = []
    diversity_dropped: List[ChunkVerdict] = []

    # Build chunk_id → source_id map from original chunks
    cid_to_source: Dict[str, str] = {}
    for i, chunk in enumerate(chunks):
        cid = chunk.get("chunk_id", f"chunk_{i}")
        cid_to_source[cid] = chunk.get("source_id", f"src_{i}")

    for v in candidates:
        src = cid_to_source.get(v.chunk_id, "unknown")
        count = source_counts.get(src, 0)
        if count < config.diversity_cap_per_source:
            diverse_candidates.append(v)
            source_counts[src] = count + 1
        else:
            v.action = "SOFT_PENALTY"
            v.reasons.append("DIVERSITY_CAP")
            diversity_dropped.append(v)

    # ----- Step 8: Max chunks + max chars -----
    kept: List[ChunkVerdict] = []
    budget_dropped: List[ChunkVerdict] = []
    total_chars = 0

    for v in diverse_candidates:
        chunk_text = ""
        for i, chunk in enumerate(chunks):
            cid = chunk.get("chunk_id", f"chunk_{i}")
            if cid == v.chunk_id:
                chunk_text = chunk.get("text", "")
                break

        if len(kept) >= config.max_chunks:
            v.reasons.append("MAX_CHUNKS")
            budget_dropped.append(v)
            continue

        new_total = total_chars + len(chunk_text.strip())
        if total_chars > 0 and new_total > config.max_total_chars:
            v.reasons.append("MAX_CHARS")
            budget_dropped.append(v)
            continue

        v.action = "KEEP"
        kept.append(v)
        total_chars = new_total

    # All dropped
    all_dropped = hard_dropped + diversity_dropped + budget_dropped
    # Mark non-kept candidates that didn't make it through budget
    for v in diverse_candidates:
        if v not in kept and v not in budget_dropped:
            all_dropped.append(v)

    # ----- Step 9: G2 guarantee -----
    fallback_triggered = False
    if len(kept) < config.min_output_guarantee:
        # Try candidates first, then hard-dropped as last resort
        recovery_pool = candidates if candidates else sorted(
            hard_dropped, key=lambda v: v.final_score, reverse=True
        )
        for candidate in recovery_pool:
            if len(kept) >= config.min_output_guarantee:
                break
            if candidate not in kept:
                candidate.action = "KEEP"
                candidate.reasons.append("G2_FALLBACK")
                kept.append(candidate)
                all_dropped = [v for v in all_dropped if v.chunk_id != candidate.chunk_id]
                fallback_triggered = True

    # ----- Compute stats -----
    sources_out: set = set()
    for v in kept:
        sources_out.add(cid_to_source.get(v.chunk_id, "unknown"))

    stats.output_count = len(kept)
    stats.fallback_triggered = fallback_triggered
    stats.sources_out = len(sources_out)

    if kept:
        stats.avg_score_kept = sum(v.final_score for v in kept) / len(kept)
    if all_dropped:
        stats.avg_score_dropped = sum(v.final_score for v in all_dropped) / len(all_dropped)

    return FilterResult(
        kept=kept,
        dropped=all_dropped,
        stats=stats,
        fallback_triggered=fallback_triggered,
    )
