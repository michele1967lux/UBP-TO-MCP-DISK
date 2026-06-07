"""
citations_verifier/providers.py

Pure technical logic — ZERO UBP framework dependencies.
All components testable standalone with mock inputs.

Components:
- ClaimExtractor: Extracts verifiable claims from generated text (LLM + heuristic)
- RAGVerifier: Verifies claims against source RAG chunks (semantic + keyword)
- WebVerifier: Verifies claims via web search using trust lists
- TrustListManager: CRUD + persistence for trust lists (Redis + JSON)
- TrustListDiscovery: Auto-discovers new trusted sources via web analysis
- VerificationOrchestrator: Coordinates full verification pipeline
- SearchFilterBuilder: Builds search query filters from trust lists

v1.0.0 — 2026-02-15
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Set, Tuple,
)
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols (DI interfaces — no concrete imports)
# ============================================================================


class IWebSearch(Protocol):
    """Protocol for web_search module."""

    async def search(
        self,
        query: str,
        max_results: int,
        safe_search: bool,
        region: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> Any: ...

    async def deep_search(
        self,
        query: str,
        max_results: int,
        max_content_urls: int,
        **kwargs,
    ) -> Any: ...


class ILLMModule(Protocol):
    """Protocol for LLM inference."""

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Dict[str, Any]: ...


class IRedis(Protocol):
    """Protocol for Redis client."""

    async def get(self, key: str) -> Optional[str]: ...
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> Any: ...
    async def delete(self, key: str) -> Any: ...
    async def keys(self, pattern: str) -> List[str]: ...
    async def exists(self, key: str) -> bool: ...


# ============================================================================
# Data Models
# ============================================================================


class ClaimStatus(str, Enum):
    """Verification status for a claim."""
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    SKIPPED = "skipped"


class VerificationSource(str, Enum):
    """Where the verification came from."""
    RAG_CHUNK = "rag_chunk"
    TRUSTED_WEB = "trusted_web"
    GENERIC_WEB = "generic_web"
    LLM_REASONING = "llm_reasoning"


@dataclass
class Claim:
    """A verifiable claim extracted from text."""
    text: str
    position_start: int = 0
    position_end: int = 0
    sentence_context: str = ""
    claim_type: str = "factual"  # factual | statistical | citation | definition

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "position": [self.position_start, self.position_end],
            "sentence_context": self.sentence_context,
            "claim_type": self.claim_type,
        }


@dataclass
class Evidence:
    """Evidence for or against a claim."""
    source_type: VerificationSource
    source_url: str = ""
    source_title: str = ""
    relevant_text: str = ""
    similarity_score: float = 0.0
    supports_claim: bool = True
    trust_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "relevant_text": self.relevant_text[:500],
            "similarity_score": round(self.similarity_score, 3),
            "supports_claim": self.supports_claim,
            "trust_score": round(self.trust_score, 2),
        }


@dataclass
class ClaimVerification:
    """Result of verifying a single claim."""
    claim: Claim
    status: ClaimStatus
    confidence: float = 0.0
    evidence: List[Evidence] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    verification_method: str = ""
    time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "status": self.status.value,
            "confidence": round(self.confidence, 3),
            "evidence": [e.to_dict() for e in self.evidence],
            "sources": self.sources,
            "verification_method": self.verification_method,
            "time_ms": round(self.time_ms, 1),
        }


@dataclass
class VerificationReport:
    """Full document verification report."""
    claims_total: int = 0
    claims_verified: int = 0
    claims_partial: int = 0
    claims_unverified: int = 0
    claims_contradicted: int = 0
    claims_skipped: int = 0
    trust_score: float = 0.0
    claims: List[ClaimVerification] = field(default_factory=list)
    verification_time_ms: float = 0.0
    sources_used: List[str] = field(default_factory=list)
    depth: str = "standard"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims_total": self.claims_total,
            "claims_verified": self.claims_verified,
            "claims_partial": self.claims_partial,
            "claims_unverified": self.claims_unverified,
            "claims_contradicted": self.claims_contradicted,
            "claims_skipped": self.claims_skipped,
            "trust_score": round(self.trust_score, 3),
            "claims": [c.to_dict() for c in self.claims],
            "verification_time_ms": round(self.verification_time_ms, 1),
            "sources_used": self.sources_used,
            "depth": self.depth,
        }


@dataclass
class TrustedSource:
    """A trusted source entry in a trust list."""
    url: str
    trust_score: float = 0.7
    notes: str = ""
    added_at: str = ""
    last_verified: str = ""
    auto_discovered: bool = False
    verification_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "trust_score": round(self.trust_score, 2),
            "notes": self.notes,
            "added_at": self.added_at,
            "last_verified": self.last_verified,
            "auto_discovered": self.auto_discovered,
            "verification_count": self.verification_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrustedSource":
        return cls(
            url=d.get("url", ""),
            trust_score=float(d.get("trust_score", d.get("score", 0.7))),
            notes=d.get("notes", ""),
            added_at=d.get("added_at", ""),
            last_verified=d.get("last_verified", ""),
            auto_discovered=d.get("auto_discovered", False),
            verification_count=d.get("verification_count", 0),
        )


@dataclass
class DiscoveredSource:
    """A source found during auto-discovery."""
    url: str
    domain: str
    appearances: int = 0
    trust_score: float = 0.0
    authority_signals: Dict[str, Any] = field(default_factory=dict)
    status: str = "proposed"  # proposed | auto_added | rejected | already_known

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "domain": self.domain,
            "appearances": self.appearances,
            "trust_score": round(self.trust_score, 2),
            "authority_signals": self.authority_signals,
            "status": self.status,
        }


# ============================================================================
# Configuration dataclasses
# ============================================================================


@dataclass
class VerificationConfig:
    enabled: bool = True
    default_depth: str = "standard"
    max_claims_per_document: int = 50
    max_parallel_verifications: int = 5
    claim_min_length: int = 15
    confidence_threshold_supported: float = 0.75
    confidence_threshold_partial: float = 0.45
    web_search_per_claim: int = 2
    timeout_per_claim_s: int = 30


@dataclass
class TrustListConfig:
    storage_backend: str = "redis"
    redis_prefix: str = "ubp:trust_list"
    json_backup_path: str = "data/trust_lists"
    auto_backup_on_change: bool = True
    default_trust_score: float = 0.7
    score_decay_days: int = 90
    max_domains_per_list: int = 200


@dataclass
class DiscoveryConfig:
    enabled: bool = True
    searches_per_domain: int = 5
    min_appearances: int = 3
    authority_signals: List[str] = field(
        default_factory=lambda: ["tld", "https", "domain_age", "academic_refs"]
    )
    auto_add_threshold: float = 0.85
    cooldown_hours: int = 24


@dataclass
class SearchIntegrationConfig:
    filter_mode: str = "prioritize"  # prioritize | exclusive | disabled
    max_sites_in_filter: int = 10
    fallback_to_generic: bool = True
    exclusive_domains: List[str] = field(default_factory=list)


# ============================================================================
# LLM response normalizer — shared single point of extraction (WARN-CV-001)
# ============================================================================

try:
    from ubp_enterprise_hybrid.modules.cores._shared.utils import extract_llm_text as _extract_llm_text
except ImportError:
    # Standalone testing fallback
    def _extract_llm_text(result: Any) -> str:  # type: ignore[misc]
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            msg = result.get("message")
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"] or ""
            return result.get("text") or result.get("response") or result.get("content") or ""
        return str(result)


# ============================================================================
# ClaimExtractor
# ============================================================================


class ClaimExtractor:
    """
    Extracts verifiable claims from generated text.

    Two modes:
    - LLM-based: Sends text to LLM with extraction prompt (higher quality)
    - Heuristic: Regex + sentence splitting (faster, no LLM needed)
    """

    # Patterns that indicate verifiable claims
    CLAIM_INDICATORS = [
        r'\b\d+[%‰]\b',                      # percentages
        r'\b\d{4}\b',                          # years
        r'\b(?:secondo|according to|per)\b',   # attribution
        r'\b(?:causa|provoca|determina)\b',    # causation
        r'\b(?:studio|ricerca|research)\b',    # research refs
        r'\b(?:legge|decreto|normativa|art\.)\b',  # legal refs
        r'\b(?:sempre|mai|tutti|nessuno)\b',   # absolutes
        r'\b\d+(?:\.\d+)?(?:\s*(?:mg|ml|kg|km|EUR|USD|€|\$))\b',  # measurements
    ]

    def __init__(self, config: VerificationConfig):
        self.config = config
        self._indicator_patterns = [re.compile(p, re.IGNORECASE) for p in self.CLAIM_INDICATORS]

    def extract_heuristic(self, text: str) -> List[Claim]:
        """
        Extract claims using heuristics (no LLM needed).

        Strategy:
        1. Split text into sentences
        2. Score each sentence for "verifiability" using indicator patterns
        3. Return sentences above threshold as claims
        """
        sentences = self._split_sentences(text)
        claims = []

        for sent_text, start, end in sentences:
            if len(sent_text) < self.config.claim_min_length:
                continue

            # Score sentence
            score = self._score_verifiability(sent_text)
            if score > 0:
                claim_type = self._classify_claim(sent_text)
                claims.append(Claim(
                    text=sent_text.strip(),
                    position_start=start,
                    position_end=end,
                    sentence_context=sent_text.strip(),
                    claim_type=claim_type,
                ))

            if len(claims) >= self.config.max_claims_per_document:
                break

        return claims

    async def extract_with_llm(self, text: str, llm: ILLMModule) -> List[Claim]:
        """
        Extract claims using LLM for higher quality extraction.
        Falls back to heuristic on failure.
        """
        prompt = (
            "Analizza il seguente testo e identifica TUTTE le affermazioni verificabili "
            "(fatti, statistiche, citazioni, definizioni, riferimenti normativi). "
            "Per ogni affermazione, restituisci una riga con il formato:\n"
            "CLAIM: <testo dell'affermazione>\n"
            "TYPE: <factual|statistical|citation|definition|legal>\n\n"
            f"Testo da analizzare:\n\n{text[:4000]}\n\n"
            "Elenca le affermazioni verificabili:"
        )

        try:
            result = await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.0,
            )

            response_text = _extract_llm_text(result)

            return self._parse_llm_claims(response_text, text)

        except Exception as e:
            logger.warning(f"[CITATIONS] LLM claim extraction failed, using heuristic: {e}")
            return self.extract_heuristic(text)

    def _parse_llm_claims(self, llm_response: str, original_text: str) -> List[Claim]:
        """Parse LLM response into Claim objects."""
        claims = []
        lines = llm_response.strip().split("\n")

        current_claim = None
        current_type = "factual"

        for line in lines:
            line = line.strip()
            if line.upper().startswith("CLAIM:"):
                if current_claim:
                    pos = original_text.find(current_claim[:50])
                    claims.append(Claim(
                        text=current_claim,
                        position_start=max(0, pos),
                        position_end=max(0, pos) + len(current_claim),
                        sentence_context=current_claim,
                        claim_type=current_type,
                    ))
                current_claim = line[6:].strip().strip('"').strip("'")
                current_type = "factual"
            elif line.upper().startswith("TYPE:"):
                current_type = line[5:].strip().lower()
                if current_type not in ("factual", "statistical", "citation", "definition", "legal"):
                    current_type = "factual"

        # Last claim
        if current_claim:
            pos = original_text.find(current_claim[:50])
            claims.append(Claim(
                text=current_claim,
                position_start=max(0, pos),
                position_end=max(0, pos) + len(current_claim),
                sentence_context=current_claim,
                claim_type=current_type,
            ))

        if not claims:
            return self.extract_heuristic(original_text)

        return claims[:self.config.max_claims_per_document]

    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with positions."""
        # Split on sentence boundaries
        pattern = r'(?<=[.!?])\s+(?=[A-ZÀ-Ú])'
        parts = re.split(pattern, text)

        sentences = []
        pos = 0
        for part in parts:
            start = text.find(part, pos)
            if start == -1:
                start = pos
            end = start + len(part)
            if part.strip():
                sentences.append((part.strip(), start, end))
            pos = end

        return sentences

    def _score_verifiability(self, sentence: str) -> int:
        """Score how verifiable a sentence is (0 = not verifiable)."""
        score = 0
        for pattern in self._indicator_patterns:
            if pattern.search(sentence):
                score += 1
        return score

    def _classify_claim(self, text: str) -> str:
        """Classify claim type based on content."""
        if re.search(r'\b\d+[%‰]\b|\bmedia|average|mediana\b', text, re.I):
            return "statistical"
        if re.search(r'\b(?:legge|decreto|art\.|comma|D\.Lgs|DPR)\b', text, re.I):
            return "legal"
        if re.search(r'\b(?:secondo|per|come riportato|afferma)\b', text, re.I):
            return "citation"
        if re.search(r'\b(?:è definit[oa]|si intende|significa)\b', text, re.I):
            return "definition"
        return "factual"


# ============================================================================
# RAGVerifier
# ============================================================================


class RAGVerifier:
    """
    Verifies claims against RAG source chunks.

    Uses keyword matching + semantic similarity (if LLM available) to determine
    if a claim is supported by the chunks used to generate it.
    """

    def __init__(self, config: VerificationConfig):
        self.config = config

    def verify_against_chunks(
        self,
        claim: Claim,
        chunks: List[Dict[str, Any]],
    ) -> ClaimVerification:
        """
        Verify a claim against RAG chunks using keyword overlap.

        Returns ClaimVerification with status and evidence.
        """
        start = time.time()
        best_match = None
        best_score = 0.0

        claim_words = self._extract_keywords(claim.text)

        for chunk in chunks:
            chunk_text = chunk.get("text", chunk.get("content", chunk.get("payload", {}).get("text", "")))
            if not chunk_text:
                continue

            chunk_words = self._extract_keywords(chunk_text)
            overlap = self._keyword_overlap(claim_words, chunk_words)

            if overlap > best_score:
                best_score = overlap
                best_match = chunk

        # Determine status based on score
        status = ClaimStatus.UNSUPPORTED
        if best_score >= self.config.confidence_threshold_supported:
            status = ClaimStatus.SUPPORTED
        elif best_score >= self.config.confidence_threshold_partial:
            status = ClaimStatus.PARTIAL

        evidence = []
        if best_match:
            chunk_text = best_match.get("text", best_match.get("content", ""))
            source_url = best_match.get("metadata", {}).get("source", best_match.get("source", "RAG"))
            evidence.append(Evidence(
                source_type=VerificationSource.RAG_CHUNK,
                source_url=str(source_url),
                source_title=best_match.get("metadata", {}).get("title", ""),
                relevant_text=chunk_text[:500] if chunk_text else "",
                similarity_score=best_score,
                supports_claim=status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIAL),
                trust_score=1.0,  # RAG chunks are trusted (user uploaded)
            ))

        elapsed = (time.time() - start) * 1000

        return ClaimVerification(
            claim=claim,
            status=status,
            confidence=best_score,
            evidence=evidence,
            sources=[e.source_url for e in evidence],
            verification_method="rag_keyword_overlap",
            time_ms=elapsed,
        )

    async def verify_with_llm(
        self,
        claim: Claim,
        chunks: List[Dict[str, Any]],
        llm: ILLMModule,
    ) -> ClaimVerification:
        """
        Verify claim against chunks using LLM for semantic comparison.
        Falls back to keyword overlap on failure.
        """
        start = time.time()

        # Build context from top chunks
        context_parts = []
        for chunk in chunks[:5]:
            text = chunk.get("text", chunk.get("content", ""))
            if text:
                context_parts.append(text[:600])

        if not context_parts:
            return self.verify_against_chunks(claim, chunks)

        context = "\n---\n".join(context_parts)

        prompt = (
            "Verifica se la seguente affermazione è supportata dal contesto fornito.\n\n"
            f"AFFERMAZIONE: {claim.text}\n\n"
            f"CONTESTO (documenti sorgente):\n{context}\n\n"
            "Rispondi con UNA sola parola:\n"
            "- SUPPORTED: l'affermazione è chiaramente supportata dal contesto\n"
            "- PARTIAL: l'affermazione è parzialmente supportata\n"
            "- UNSUPPORTED: il contesto non contiene informazioni su questa affermazione\n"
            "- CONTRADICTED: il contesto contraddice l'affermazione\n\n"
            "Risposta:"
        )

        try:
            result = await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.0,
            )

            response = _extract_llm_text(result).strip().upper()

            status_map = {
                "SUPPORTED": ClaimStatus.SUPPORTED,
                "PARTIAL": ClaimStatus.PARTIAL,
                "UNSUPPORTED": ClaimStatus.UNSUPPORTED,
                "CONTRADICTED": ClaimStatus.CONTRADICTED,
            }

            # Parse first word
            first_word = response.split()[0] if response else ""
            first_word = first_word.strip(".:,;")
            status = status_map.get(first_word, ClaimStatus.UNSUPPORTED)

            confidence = {
                ClaimStatus.SUPPORTED: 0.9,
                ClaimStatus.PARTIAL: 0.6,
                ClaimStatus.UNSUPPORTED: 0.2,
                ClaimStatus.CONTRADICTED: 0.85,
            }.get(status, 0.3)

            elapsed = (time.time() - start) * 1000
            return ClaimVerification(
                claim=claim,
                status=status,
                confidence=confidence,
                evidence=[Evidence(
                    source_type=VerificationSource.RAG_CHUNK,
                    source_url="RAG",
                    relevant_text=context[:300],
                    similarity_score=confidence,
                    supports_claim=status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIAL),
                    trust_score=1.0,
                )],
                sources=["RAG"],
                verification_method="rag_llm_semantic",
                time_ms=elapsed,
            )

        except Exception as e:
            logger.warning(f"[CITATIONS] LLM RAG verification failed: {e}")
            return self.verify_against_chunks(claim, chunks)

    async def verify_batch_with_llm(
        self,
        claims: List[Claim],
        chunks: List[Dict[str, Any]],
        llm: ILLMModule,
    ) -> tuple:
        """
        Verify ALL claims in a single LLM call (batch mode).

        Returns (results, web_queries):
        - results: dict {claim_index: ClaimVerification} for parsed claims
        - web_queries: dict {claim_index: str} suggested search queries
          for UNSUPPORTED/CONTRADICTED claims
        """
        start = time.time()

        # Build shared context (same for all claims)
        context_parts = []
        for chunk in chunks[:5]:
            text = chunk.get("text", chunk.get("content", ""))
            if text:
                context_parts.append(text[:600])

        if not context_parts or not claims:
            return {}, {}

        context = "\n---\n".join(context_parts)

        # Build numbered claims list
        claims_block = "\n".join(
            f"{i+1}. {c.text}" for i, c in enumerate(claims)
        )

        prompt = (
            "Verifica se le seguenti affermazioni sono supportate dal contesto fornito.\n\n"
            f"CONTESTO (documenti sorgente):\n{context}\n\n"
            f"AFFERMAZIONI:\n{claims_block}\n\n"
            "Per OGNI affermazione, rispondi con il numero e UNO dei seguenti verdetti:\n"
            "- SUPPORTED: chiaramente supportata dal contesto\n"
            "- PARTIAL: parzialmente supportata\n"
            "- UNSUPPORTED: il contesto non contiene informazioni\n"
            "- CONTRADICTED: il contesto contraddice l'affermazione\n\n"
            "Se il verdetto è UNSUPPORTED o CONTRADICTED, aggiungi una query di ricerca web "
            "per verificare l'affermazione. La query deve essere breve e contenere solo "
            "i termini chiave (NO nomi di siti, NO parentesi, NO formattazione).\n\n"
            "Formato risposta (una riga per affermazione):\n"
            "1. SUPPORTED\n"
            "2. UNSUPPORTED | QUERY: metformina effetti collaterali diarrea nausea\n"
            "3. CONTRADICTED | QUERY: SGLT2 controindicazioni insufficienza renale\n"
            "4. PARTIAL\n\n"
            "Rispondi SOLO con il formato indicato, niente altro."
        )

        try:
            result = await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max(150, len(claims) * 40),
                temperature=0.0,
            )

            response = _extract_llm_text(result).strip()
            return self._parse_batch_response(response, claims, start)

        except Exception as e:
            logger.warning("[CITATIONS] Batch LLM verification failed: %s", e)
            return {}, {}

    def _parse_batch_response(
        self,
        response: str,
        claims: List[Claim],
        start_time: float,
    ) -> tuple:
        """
        Tolerant parser for batch LLM response.

        Accepts formats: "1. SUPPORTED", "1: SUPPORTED", "1 - SUPPORTED",
        "1 SUPPORTED", "1) SUPPORTED", etc.
        Also extracts web query suggestions after "| QUERY: ...".

        Returns (results, web_queries):
        - results: Dict[int, ClaimVerification]
        - web_queries: Dict[int, str] for UNSUPPORTED/CONTRADICTED claims
        """
        status_map = {
            "SUPPORTED": ClaimStatus.SUPPORTED,
            "PARTIAL": ClaimStatus.PARTIAL,
            "UNSUPPORTED": ClaimStatus.UNSUPPORTED,
            "CONTRADICTED": ClaimStatus.CONTRADICTED,
        }
        confidence_map = {
            ClaimStatus.SUPPORTED: 0.9,
            ClaimStatus.PARTIAL: 0.6,
            ClaimStatus.UNSUPPORTED: 0.2,
            ClaimStatus.CONTRADICTED: 0.85,
        }

        results: Dict[int, ClaimVerification] = {}
        web_queries: Dict[int, str] = {}
        elapsed = (time.time() - start_time) * 1000

        # Tolerant line-by-line parsing
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue

            # Extract web query suggestion before matching verdict
            # Formats: "| QUERY: ..." or "QUERY: ..."
            suggested_query = None
            query_match = re.search(r'\|\s*QUERY:\s*(.+)', line, re.IGNORECASE)
            if query_match:
                suggested_query = query_match.group(1).strip()
                # Remove query part from line for verdict parsing
                line = line[:query_match.start()].strip()

            # Extract leading number: "1. SUPPORTED", "1: PARTIAL", "1 - SUPPORTED", "1) SUPPORTED"
            m = re.match(r'^(\d+)\s*[.:\-)\]]\s*(\S+)', line)
            if not m:
                # Try without separator: "1 SUPPORTED"
                m = re.match(r'^(\d+)\s+(\S+)', line)
            if not m:
                continue

            idx = int(m.group(1)) - 1  # 1-based → 0-based
            verdict = m.group(2).strip(".:,;").upper()

            if idx < 0 or idx >= len(claims):
                continue

            status = status_map.get(verdict)
            if status is None:
                # Fuzzy match: "SUPPORT" → SUPPORTED, "CONTRADICT" → CONTRADICTED
                for key in status_map:
                    if verdict.startswith(key[:5]):
                        status = status_map[key]
                        break
            if status is None:
                continue

            confidence = confidence_map.get(status, 0.3)
            claim = claims[idx]

            results[idx] = ClaimVerification(
                claim=claim,
                status=status,
                confidence=confidence,
                evidence=[Evidence(
                    source_type=VerificationSource.RAG_CHUNK,
                    source_url="RAG",
                    relevant_text="(batch verification)",
                    similarity_score=confidence,
                    supports_claim=status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIAL),
                    trust_score=1.0,
                )],
                sources=["RAG"],
                verification_method="rag_llm_batch",
                time_ms=elapsed / max(len(results), 1),
            )

            # Store web query for UNSUPPORTED/CONTRADICTED claims
            if suggested_query and status in (ClaimStatus.UNSUPPORTED, ClaimStatus.CONTRADICTED):
                web_queries[idx] = suggested_query

        logger.info(
            "[CITATIONS] Batch parsed: %d/%d claims, %d web queries suggested, %.0fms",
            len(results), len(claims), len(web_queries), elapsed,
        )
        return results, web_queries

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        # Remove common stop words (IT + EN minimal set)
        stop_words = {
            "il", "lo", "la", "i", "gli", "le", "di", "a", "da", "in", "con", "su",
            "per", "tra", "fra", "che", "è", "sono", "ha", "hanno", "un", "una", "uno",
            "del", "della", "dei", "delle", "al", "alla", "nel", "nella", "e", "o", "ma",
            "non", "si", "come", "più", "anche", "questo", "quello", "essere", "avere",
            "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or",
            "for", "on", "at", "by", "with", "from", "not", "but", "this", "that",
        }
        words = re.findall(r'\b[a-zà-ú]{3,}\b', text.lower())
        return set(w for w in words if w not in stop_words)

    def _keyword_overlap(self, claim_words: Set[str], chunk_words: Set[str]) -> float:
        """Calculate keyword overlap score (Jaccard-like)."""
        if not claim_words:
            return 0.0
        intersection = claim_words & chunk_words
        # Weighted toward claim coverage (how much of the claim is in the chunk)
        claim_coverage = len(intersection) / len(claim_words) if claim_words else 0
        return claim_coverage


# ============================================================================
# WebVerifier
# ============================================================================


class WebVerifier:
    """
    Verifies claims via web search, prioritizing trusted sources.
    """

    def __init__(self, config: VerificationConfig):
        self.config = config

    async def verify_claim(
        self,
        claim: Claim,
        web_search: IWebSearch,
        trusted_domains: List[str],
        language: str = "it",
        search_query: Optional[str] = None,
    ) -> ClaimVerification:
        """
        Verify a claim using web search.

        Args:
            search_query: LLM-suggested search query. If provided, used instead
                of raw claim.text (which may contain formatting, site names, etc.).

        Strategy:
        1. Search with trusted site filter first
        2. If insufficient results, search generically
        3. Analyze results for support/contradiction
        """
        start = time.time()
        all_evidence: List[Evidence] = []
        all_sources: List[str] = []

        # Use LLM-suggested query or fall back to claim text
        base_query = search_query or claim.text[:150]

        # Phase 1: Search on trusted sites
        if trusted_domains:
            site_filter = " OR ".join(f"site:{d}" for d in trusted_domains[:5])
            trusted_query = f"{base_query[:100]} {site_filter}"

            try:
                results = await web_search.search(
                    query=trusted_query,
                    max_results=self.config.web_search_per_claim,
                    safe_search=True,
                    language=language,
                )

                if isinstance(results, dict):
                    results = results.get("results", [])

                for r in results:
                    if isinstance(r, dict) and not r.get("error"):
                        domain = self._extract_domain(r.get("url", ""))
                        is_trusted = domain in trusted_domains
                        evidence = Evidence(
                            source_type=VerificationSource.TRUSTED_WEB if is_trusted else VerificationSource.GENERIC_WEB,
                            source_url=r.get("url", ""),
                            source_title=r.get("title", ""),
                            relevant_text=r.get("snippet", "")[:500],
                            similarity_score=self._snippet_relevance(claim.text, r.get("snippet", "")),
                            supports_claim=True,  # Will be refined below
                            trust_score=0.9 if is_trusted else 0.5,
                        )
                        all_evidence.append(evidence)
                        if r.get("url"):
                            all_sources.append(r["url"])

            except Exception as e:
                logger.warning(f"[CITATIONS] Trusted web search failed for claim: {e}")

        # Phase 2: Generic search if needed
        if len(all_evidence) < self.config.web_search_per_claim:
            try:
                results = await web_search.search(
                    query=base_query[:150],
                    max_results=self.config.web_search_per_claim,
                    safe_search=True,
                    language=language,
                )

                if isinstance(results, dict):
                    results = results.get("results", [])

                for r in results:
                    if isinstance(r, dict) and not r.get("error"):
                        url = r.get("url", "")
                        if url not in all_sources:
                            all_evidence.append(Evidence(
                                source_type=VerificationSource.GENERIC_WEB,
                                source_url=url,
                                source_title=r.get("title", ""),
                                relevant_text=r.get("snippet", "")[:500],
                                similarity_score=self._snippet_relevance(claim.text, r.get("snippet", "")),
                                supports_claim=True,
                                trust_score=0.4,
                            ))
                            all_sources.append(url)

            except Exception as e:
                logger.warning(f"[CITATIONS] Generic web search failed for claim: {e}")

        # Determine overall status
        status, confidence = self._evaluate_evidence(claim, all_evidence)

        elapsed = (time.time() - start) * 1000
        return ClaimVerification(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=all_evidence,
            sources=all_sources,
            verification_method="web_search",
            time_ms=elapsed,
        )

    def _snippet_relevance(self, claim: str, snippet: str) -> float:
        """Quick relevance score between claim and search snippet."""
        if not claim or not snippet:
            return 0.0
        claim_words = set(re.findall(r'\b\w{4,}\b', claim.lower()))
        snippet_words = set(re.findall(r'\b\w{4,}\b', snippet.lower()))
        if not claim_words:
            return 0.0
        overlap = len(claim_words & snippet_words) / len(claim_words)
        return overlap

    def _evaluate_evidence(
        self,
        claim: Claim,
        evidence: List[Evidence],
    ) -> Tuple[ClaimStatus, float]:
        """Evaluate evidence to determine claim status."""
        if not evidence:
            return ClaimStatus.UNSUPPORTED, 0.1

        # Weight by trust score and relevance
        weighted_scores = []
        for e in evidence:
            weight = e.trust_score * e.similarity_score
            weighted_scores.append(weight)

        avg_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0
        max_score = max(weighted_scores) if weighted_scores else 0

        # Boost if trusted sources confirm
        trusted_evidence = [e for e in evidence if e.source_type == VerificationSource.TRUSTED_WEB]
        if trusted_evidence:
            avg_score *= 1.2
            avg_score = min(avg_score, 1.0)

        if avg_score >= 0.6 or max_score >= 0.75:
            return ClaimStatus.SUPPORTED, min(avg_score + 0.1, 1.0)
        elif avg_score >= 0.3:
            return ClaimStatus.PARTIAL, avg_score
        else:
            return ClaimStatus.UNSUPPORTED, avg_score

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""


# ============================================================================
# TrustListManager
# ============================================================================


class TrustListManager:
    """
    CRUD + persistence for trust lists.

    Storage: Redis (primary) + JSON backup.
    Each domain category has its own list of TrustedSource entries.
    """

    def __init__(self, config: TrustListConfig):
        self.config = config
        self._cache: Dict[str, List[TrustedSource]] = {}
        self._loaded = False

    async def initialize(
        self,
        redis: Optional[IRedis],
        predefined_lists: Dict[str, List[Dict[str, Any]]],
    ) -> int:
        """
        Load trust lists from Redis. If empty, seed from predefined lists.
        Returns total number of trusted domains loaded.
        """
        total = 0

        for domain_name, entries in predefined_lists.items():
            sources = []

            # Try Redis first
            if redis:
                try:
                    key = f"{self.config.redis_prefix}:{domain_name}"
                    data = await redis.get(key)
                    if data:
                        raw_list = json.loads(data)
                        sources = [TrustedSource.from_dict(d) for d in raw_list]
                except Exception as e:
                    logger.warning(f"[TRUST] Redis load failed for {domain_name}: {e}")

            # Seed from predefined if empty
            if not sources and entries:
                from datetime import datetime
                now = datetime.utcnow().isoformat()
                sources = [
                    TrustedSource(
                        url=e.get("url", ""),
                        trust_score=float(e.get("score", e.get("trust_score", 0.7))),
                        notes=e.get("notes", ""),
                        added_at=now,
                        auto_discovered=False,
                    )
                    for e in entries
                    if e.get("url")
                ]

                # Save to Redis
                if redis and sources:
                    await self._save_to_redis(redis, domain_name, sources)

            self._cache[domain_name] = sources
            total += len(sources)

        self._loaded = True
        logger.info(f"[TRUST] Loaded {total} trusted domains across {len(self._cache)} lists")
        return total

    async def get_list(
        self,
        domain: str,
        min_score: float = 0.0,
    ) -> List[TrustedSource]:
        """Get trust list for a domain, filtered by minimum score."""
        sources = self._cache.get(domain, [])
        if min_score > 0:
            sources = [s for s in sources if s.trust_score >= min_score]
        return sorted(sources, key=lambda s: s.trust_score, reverse=True)

    def get_domains(self, domain: str, min_score: float = 0.0) -> List[str]:
        """Get just the domain URLs for a trust list (fast path)."""
        return [s.url for s in self.get_list_sync(domain, min_score)]

    def get_list_sync(self, domain: str, min_score: float = 0.0) -> List[TrustedSource]:
        """Sync version of get_list (uses cache only)."""
        sources = self._cache.get(domain, [])
        if min_score > 0:
            sources = [s for s in sources if s.trust_score >= min_score]
        return sorted(sources, key=lambda s: s.trust_score, reverse=True)

    def get_all_domain_names(self) -> List[str]:
        """List all available trust list domains."""
        return list(self._cache.keys())

    def get_total_domains(self) -> int:
        """Total count of trusted domains across all lists."""
        return sum(len(v) for v in self._cache.values())

    async def add_entries(
        self,
        domain: str,
        entries: List[TrustedSource],
        redis: Optional[IRedis] = None,
    ) -> int:
        """Add entries to a trust list. Returns count added."""
        if domain not in self._cache:
            self._cache[domain] = []

        existing_urls = {s.url for s in self._cache[domain]}
        added = 0

        for entry in entries:
            if entry.url not in existing_urls and len(self._cache[domain]) < self.config.max_domains_per_list:
                if not entry.added_at:
                    from datetime import datetime
                    entry.added_at = datetime.utcnow().isoformat()
                self._cache[domain].append(entry)
                existing_urls.add(entry.url)
                added += 1

        if added > 0 and redis:
            await self._save_to_redis(redis, domain, self._cache[domain])
            if self.config.auto_backup_on_change:
                await self._backup_to_json(domain)

        return added

    async def remove_entry(
        self,
        domain: str,
        url: str,
        redis: Optional[IRedis] = None,
    ) -> bool:
        """Remove an entry from a trust list."""
        if domain not in self._cache:
            return False

        before = len(self._cache[domain])
        self._cache[domain] = [s for s in self._cache[domain] if s.url != url]
        removed = len(self._cache[domain]) < before

        if removed and redis:
            await self._save_to_redis(redis, domain, self._cache[domain])
            if self.config.auto_backup_on_change:
                await self._backup_to_json(domain)

        return removed

    async def update_entry(
        self,
        domain: str,
        url: str,
        trust_score: Optional[float] = None,
        notes: Optional[str] = None,
        redis: Optional[IRedis] = None,
    ) -> bool:
        """Update an existing entry."""
        if domain not in self._cache:
            return False

        for source in self._cache[domain]:
            if source.url == url:
                if trust_score is not None:
                    source.trust_score = trust_score
                if notes is not None:
                    source.notes = notes
                from datetime import datetime
                source.last_verified = datetime.utcnow().isoformat()
                source.verification_count += 1

                if redis:
                    await self._save_to_redis(redis, domain, self._cache[domain])
                return True

        return False

    async def create_list(
        self,
        domain: str,
        entries: Optional[List[TrustedSource]] = None,
        redis: Optional[IRedis] = None,
    ) -> bool:
        """Create a new trust list domain."""
        if domain in self._cache:
            return False
        self._cache[domain] = entries or []
        if redis and self._cache[domain]:
            await self._save_to_redis(redis, domain, self._cache[domain])
        return True

    async def delete_list(
        self,
        domain: str,
        redis: Optional[IRedis] = None,
    ) -> bool:
        """Delete an entire trust list."""
        if domain not in self._cache:
            return False
        del self._cache[domain]
        if redis:
            try:
                key = f"{self.config.redis_prefix}:{domain}"
                await redis.delete(key)
            except Exception as e:
                logger.warning(f"[TRUST] Redis delete failed for {domain}: {e}")
        return True

    def build_search_filter(
        self,
        domain: str,
        min_score: float = 0.7,
        max_sites: int = 10,
    ) -> str:
        """
        Build a search query filter string for web_search.

        Returns: "site:domain1.com OR site:domain2.com" format
        """
        sources = self.get_list_sync(domain, min_score)[:max_sites]
        if not sources:
            return ""
        return " OR ".join(f"site:{s.url}" for s in sources)

    async def _save_to_redis(
        self,
        redis: IRedis,
        domain: str,
        sources: List[TrustedSource],
    ) -> None:
        """Save trust list to Redis."""
        try:
            key = f"{self.config.redis_prefix}:{domain}"
            data = json.dumps([s.to_dict() for s in sources], ensure_ascii=False)
            await redis.set(key, data)
        except Exception as e:
            logger.error(f"[TRUST] Redis save failed for {domain}: {e}")

    async def _backup_to_json(self, domain: str) -> None:
        """Backup trust list to JSON file."""
        try:
            backup_dir = Path(self.config.json_backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            filepath = backup_dir / f"{domain}.json"
            sources = self._cache.get(domain, [])
            data = {
                "domain": domain,
                "count": len(sources),
                "sources": [s.to_dict() for s in sources],
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[TRUST] JSON backup failed for {domain}: {e}")


# ============================================================================
# TrustListDiscovery
# ============================================================================


class TrustListDiscovery:
    """
    Automatically discovers new trusted sources for a domain.

    Strategy:
    1. Generate seed queries for the domain
    2. Search the web with each query
    3. Collect domains that appear frequently
    4. Score them using authority signals
    5. Propose or auto-add above threshold
    """

    # Seed queries per domain
    DOMAIN_SEED_QUERIES = {
        "medical": [
            "clinical guidelines evidence-based medicine",
            "peer reviewed medical journal",
            "linee guida cliniche italiane",
            "systematic review medical database",
            "pharmacological reference database",
        ],
        "legal_it": [
            "legislazione italiana database giuridico",
            "giurisprudenza italiana sentenze",
            "codice civile commentato online",
            "normativa italiana aggiornata",
            "diritto italiano risorse autorevoli",
        ],
        "technical": [
            "official documentation programming language",
            "peer reviewed computer science journal",
            "technical standards organization",
            "open source project documentation",
            "software engineering best practices reference",
        ],
        "financial": [
            "financial regulation authority database",
            "economic statistics official source",
            "borsa mercati finanziari dati ufficiali",
            "central bank publications data",
            "financial reporting standards authority",
        ],
        "academic": [
            "academic journal database search",
            "peer reviewed research repository",
            "university research publications",
            "citation index scientific articles",
            "open access academic papers",
        ],
    }

    # Authority TLD scores
    TLD_AUTHORITY = {
        ".gov": 0.95, ".gov.it": 0.95, ".edu": 0.90,
        ".org": 0.70, ".int": 0.90, ".europa.eu": 0.92,
        ".ac.uk": 0.88, ".edu.au": 0.88,
    }

    def __init__(self, config: DiscoveryConfig):
        self.config = config

    async def discover(
        self,
        domain: str,
        web_search: IWebSearch,
        existing_urls: Set[str],
        seed_queries: Optional[List[str]] = None,
        max_discoveries: int = 20,
    ) -> List[DiscoveredSource]:
        """
        Discover new trusted sources for a domain.

        Returns list of DiscoveredSource with scores and status.
        """
        queries = seed_queries or self.DOMAIN_SEED_QUERIES.get(domain, [])
        if not queries:
            queries = [f"authoritative sources for {domain}"]

        domain_appearances: Dict[str, Dict[str, Any]] = {}

        # Search with each query
        for query in queries[:self.config.searches_per_domain]:
            try:
                results = await web_search.search(
                    query=query,
                    max_results=10,
                    safe_search=True,
                )

                if isinstance(results, dict):
                    results = results.get("results", [])

                for r in results:
                    if isinstance(r, dict) and r.get("url") and not r.get("error"):
                        url_domain = self._extract_root_domain(r["url"])
                        if url_domain and url_domain not in existing_urls:
                            if url_domain not in domain_appearances:
                                domain_appearances[url_domain] = {
                                    "appearances": 0,
                                    "urls": [],
                                    "titles": [],
                                }
                            domain_appearances[url_domain]["appearances"] += 1
                            domain_appearances[url_domain]["urls"].append(r["url"])
                            if r.get("title"):
                                domain_appearances[url_domain]["titles"].append(r["title"])

            except Exception as e:
                logger.warning(f"[DISCOVERY] Search failed for query '{query}': {e}")

        # Score discovered domains
        discoveries = []
        for url_domain, info in domain_appearances.items():
            if info["appearances"] < self.config.min_appearances:
                continue

            # Calculate authority score
            signals = self._evaluate_authority(url_domain, info)
            trust_score = signals.get("final_score", 0.5)

            status = "proposed"
            if url_domain in existing_urls:
                status = "already_known"
            elif trust_score >= self.config.auto_add_threshold:
                status = "auto_added"

            discoveries.append(DiscoveredSource(
                url=url_domain,
                domain=domain,
                appearances=info["appearances"],
                trust_score=trust_score,
                authority_signals=signals,
                status=status,
            ))

        # Sort by score and limit
        discoveries.sort(key=lambda d: d.trust_score, reverse=True)
        return discoveries[:max_discoveries]

    def _evaluate_authority(
        self,
        domain: str,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Score a domain using authority signals."""
        signals: Dict[str, Any] = {}
        score = 0.5  # Base score

        # TLD authority
        for tld, tld_score in self.TLD_AUTHORITY.items():
            if domain.endswith(tld):
                signals["tld"] = tld
                signals["tld_score"] = tld_score
                score = max(score, tld_score * 0.8)
                break

        # HTTPS (all modern sites should have it)
        signals["https"] = True  # We assume search results are HTTPS

        # Appearance frequency (more = more authoritative for this domain)
        appearances = info.get("appearances", 1)
        freq_bonus = min(appearances * 0.05, 0.2)
        score += freq_bonus
        signals["appearances"] = appearances
        signals["freq_bonus"] = round(freq_bonus, 2)

        # Known academic/institutional patterns
        academic_patterns = [
            r'\.edu', r'\.ac\.', r'\.gov', r'university', r'institute',
            r'journal', r'library', r'research', r'national',
        ]
        for pattern in academic_patterns:
            if re.search(pattern, domain, re.I):
                score += 0.1
                signals["academic_pattern"] = pattern
                break

        score = min(score, 1.0)
        signals["final_score"] = round(score, 3)
        return signals

    def _extract_root_domain(self, url: str) -> str:
        """Extract root domain from URL (no path, no www)."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""


# ============================================================================
# SearchFilterBuilder
# ============================================================================


class SearchFilterBuilder:
    """
    Builds search query filters from trust lists for use by web_search module.

    Modes:
    - prioritize: Trusted sources first, then generic
    - exclusive: ONLY trusted sources, no generic
    - disabled: No filtering
    """

    def __init__(self, config: SearchIntegrationConfig):
        self.config = config

    def build_filter(
        self,
        trust_manager: TrustListManager,
        domain: str,
        exclusive: bool = False,
        max_sites: int = 0,
    ) -> Dict[str, Any]:
        """
        Build a search filter for the given domain.

        Returns dict with filter_query, sites list, and exclusive flag.
        """
        max_s = max_sites or self.config.max_sites_in_filter
        is_exclusive = exclusive or (domain in self.config.exclusive_domains)

        filter_query = trust_manager.build_search_filter(domain, min_score=0.7, max_sites=max_s)
        sites = trust_manager.get_domains(domain, min_score=0.7)[:max_s]

        return {
            "filter_query": filter_query,
            "sites": sites,
            "exclusive": is_exclusive,
            "domain": domain,
            "count": len(sites),
            "mode": "exclusive" if is_exclusive else self.config.filter_mode,
        }

    def apply_to_query(
        self,
        query: str,
        filter_result: Dict[str, Any],
    ) -> str:
        """
        Apply the filter to a search query string.

        If exclusive: "query site:a.com OR site:b.com"
        If prioritize: "query" (filter used for post-ranking)
        """
        if not filter_result.get("filter_query"):
            return query

        if filter_result.get("exclusive"):
            return f"{query} ({filter_result['filter_query']})"

        # For prioritize mode, the filter is used after results are returned
        return query


# ============================================================================
# VerificationOrchestrator
# ============================================================================


class VerificationOrchestrator:
    """
    Coordinates the full verification pipeline.

    Flow:
    1. Extract claims from text
    2. For each claim:
       a. If RAG chunks available → RAGVerifier
       b. If claim unsupported or depth >= standard → WebVerifier
    3. Aggregate results into VerificationReport
    """

    def __init__(
        self,
        config: VerificationConfig,
        claim_extractor: ClaimExtractor,
        rag_verifier: RAGVerifier,
        web_verifier: WebVerifier,
        trust_manager: TrustListManager,
    ):
        self.config = config
        self.extractor = claim_extractor
        self.rag = rag_verifier
        self.web = web_verifier
        self.trust = trust_manager

    async def verify_document(
        self,
        text: str,
        rag_chunks: Optional[List[Dict[str, Any]]] = None,
        domain: Optional[str] = None,
        depth: str = "standard",
        language: str = "it",
        web_search: Optional[IWebSearch] = None,
        llm: Optional[ILLMModule] = None,
        max_claims: int = 0,
    ) -> VerificationReport:
        """
        Full document verification pipeline.

        Depths:
        - quick: RAG only, no web search
        - standard: RAG + trusted web search
        - deep: RAG + full web search (trusted + generic)
        """
        start = time.time()
        report = VerificationReport(depth=depth)

        # Step 1: Extract claims
        if llm and depth != "quick":
            claims = await self.extractor.extract_with_llm(text, llm)
        else:
            claims = self.extractor.extract_heuristic(text)

        if max_claims > 0:
            claims = claims[:max_claims]

        report.claims_total = len(claims)

        if not claims:
            report.trust_score = 1.0  # No claims = nothing to verify
            report.verification_time_ms = (time.time() - start) * 1000
            return report

        # Step 2: Get trusted domains for this domain
        trusted_domains: List[str] = []
        if domain:
            trusted_domains = self.trust.get_domains(domain, min_score=0.7)

        # Step 3: Verify claims — batch RAG-LLM + per-claim web
        #
        # Phase A: single LLM call for all claims against RAG chunks
        #          Returns verdicts + suggested web queries for unverified claims
        # Phase B: web verification ONLY for UNSUPPORTED/CONTRADICTED claims
        #          using LLM-suggested queries (not raw claim text)
        #
        batch_results: Dict[int, ClaimVerification] = {}
        web_queries: Dict[int, str] = {}
        if rag_chunks and llm and len(claims) > 1:
            logger.info(
                "[CITATIONS] Batch LLM verification: %d claims, %d chunks, llm=%s",
                len(claims), len(rag_chunks), type(llm).__name__,
            )
            batch_results, web_queries = await self.rag.verify_batch_with_llm(claims, rag_chunks, llm)
            logger.info("[CITATIONS] Batch results: %d/%d parsed, %d web queries", len(batch_results), len(claims), len(web_queries))
        elif not llm:
            logger.info("[CITATIONS] No LLM provided — using keyword overlap for %d claims", len(claims))
        elif not rag_chunks:
            logger.info("[CITATIONS] No RAG chunks — skipping batch verification")

        # Build per-claim results: batch hit → use it, miss → keyword fallback
        rag_results: List[ClaimVerification] = []
        for i, claim in enumerate(claims):
            if i in batch_results:
                rag_results.append(batch_results[i])
            elif rag_chunks:
                rag_results.append(self.rag.verify_against_chunks(claim, rag_chunks))
            else:
                rag_results.append(ClaimVerification(
                    claim=claim, status=ClaimStatus.UNSUPPORTED, confidence=0.1,
                    verification_method="no_chunks",
                ))

        # Phase B: web verification ONLY for UNSUPPORTED/CONTRADICTED claims
        if depth != "quick" and web_search:
            # Identify claims that need web cross-check
            web_candidates = [
                i for i, cv in enumerate(rag_results)
                if cv.status in (ClaimStatus.UNSUPPORTED, ClaimStatus.CONTRADICTED)
            ]
            logger.info(
                "[CITATIONS] Web verification: %d/%d claims need web check",
                len(web_candidates), len(claims),
            )

            semaphore = asyncio.Semaphore(self.config.max_parallel_verifications)

            async def web_verify(idx: int) -> tuple:
                rag_cv = rag_results[idx]
                async with semaphore:
                    web_cv = await self.web.verify_claim(
                        claim=claims[idx], web_search=web_search,
                        trusted_domains=trusted_domains if depth in ("standard", "deep") else [],
                        language=language,
                        search_query=web_queries.get(idx),
                    )
                    # Merge RAG evidence into web result
                    if rag_cv.evidence:
                        web_cv.evidence = rag_cv.evidence + web_cv.evidence
                        web_cv.sources = list(set(rag_cv.sources + web_cv.sources))
                        web_cv.verification_method = "rag+web"
                    return idx, web_cv

            tasks = [web_verify(i) for i in web_candidates]
            web_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge web results back into rag_results
            for wr in web_results:
                if isinstance(wr, Exception):
                    continue
                idx, web_cv = wr
                rag_results[idx] = web_cv

            results = rag_results
        else:
            results = rag_results

        # Step 4: Aggregate
        all_sources: Set[str] = set()

        for result in results:
            if isinstance(result, Exception):
                report.claims_skipped += 1
                continue

            cv: ClaimVerification = result
            report.claims.append(cv)

            if cv.status == ClaimStatus.SUPPORTED:
                report.claims_verified += 1
            elif cv.status == ClaimStatus.PARTIAL:
                report.claims_partial += 1
            elif cv.status == ClaimStatus.CONTRADICTED:
                report.claims_contradicted += 1
            elif cv.status == ClaimStatus.UNSUPPORTED:
                report.claims_unverified += 1
            else:
                report.claims_skipped += 1

            all_sources.update(cv.sources)

        # Calculate trust score
        total_checked = report.claims_verified + report.claims_partial + report.claims_unverified + report.claims_contradicted
        if total_checked > 0:
            score = (
                report.claims_verified * 1.0 +
                report.claims_partial * 0.6 -
                report.claims_contradicted * 0.5
            ) / total_checked
            report.trust_score = max(0.0, min(1.0, score))
        else:
            report.trust_score = 0.5  # Unknown

        report.sources_used = sorted(all_sources)
        report.verification_time_ms = (time.time() - start) * 1000

        return report

    async def _verify_single_claim(
        self,
        claim: Claim,
        rag_chunks: Optional[List[Dict[str, Any]]],
        trusted_domains: List[str],
        depth: str,
        language: str,
        web_search: Optional[IWebSearch],
        llm: Optional[ILLMModule],
    ) -> ClaimVerification:
        """Verify a single claim using the appropriate method."""
        try:
            result = await asyncio.wait_for(
                self._verify_claim_inner(
                    claim, rag_chunks, trusted_domains, depth, language, web_search, llm,
                ),
                timeout=self.config.timeout_per_claim_s,
            )
            return result
        except asyncio.TimeoutError:
            return ClaimVerification(
                claim=claim,
                status=ClaimStatus.SKIPPED,
                confidence=0.0,
                verification_method="timeout",
            )
        except Exception as e:
            logger.warning(f"[CITATIONS] Claim verification error: {e}")
            return ClaimVerification(
                claim=claim,
                status=ClaimStatus.SKIPPED,
                confidence=0.0,
                verification_method=f"error:{e}",
            )

    async def _verify_claim_inner(
        self,
        claim: Claim,
        rag_chunks: Optional[List[Dict[str, Any]]],
        trusted_domains: List[str],
        depth: str,
        language: str,
        web_search: Optional[IWebSearch],
        llm: Optional[ILLMModule],
    ) -> ClaimVerification:
        """Inner verification logic."""

        # Phase 1: RAG verification (always, if chunks available)
        rag_result: Optional[ClaimVerification] = None
        if rag_chunks:
            if llm:
                rag_result = await self.rag.verify_with_llm(claim, rag_chunks, llm)
            else:
                rag_result = self.rag.verify_against_chunks(claim, rag_chunks)

            # If supported by RAG with high confidence, done
            if rag_result.status == ClaimStatus.SUPPORTED and rag_result.confidence >= 0.8:
                return rag_result

        # Phase 2: Web verification (if depth allows)
        if depth == "quick":
            return rag_result or ClaimVerification(
                claim=claim, status=ClaimStatus.UNSUPPORTED, confidence=0.1,
                verification_method="quick_no_chunks",
            )

        if web_search:
            web_result = await self.web.verify_claim(
                claim=claim,
                web_search=web_search,
                trusted_domains=trusted_domains if depth in ("standard", "deep") else [],
                language=language,
            )

            # Merge RAG + Web evidence
            if rag_result and rag_result.evidence:
                web_result.evidence = rag_result.evidence + web_result.evidence
                web_result.sources = list(set(rag_result.sources + web_result.sources))

                # Boost confidence if both agree
                if rag_result.status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIAL):
                    web_result.confidence = min(1.0, web_result.confidence + 0.15)

                web_result.verification_method = "rag+web"

            return web_result

        # No web search available
        return rag_result or ClaimVerification(
            claim=claim, status=ClaimStatus.UNSUPPORTED, confidence=0.1,
            verification_method="no_sources",
        )
