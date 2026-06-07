"""
citation_verification_orchestrator — providers.py

Pure orchestration logic.  ZERO UBP framework dependencies.
Coordinates citations_verifier + reasoning_rag templates via injected callables.

LLM calls are **never** made directly — everything passes through the
``llm_caller`` callback which the adapter wires to ``_call_llm()`` →
``pipeline_orchestrator.execute(simple_chat)``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional
from urllib.parse import urlparse

from .schemas import (
    FilterTrustedResult,
    TrustLevel,
    TrustSummaryEntry,
    TrustUpdateEntry,
    UpdateTrustResult,
    VerificationStatus,
    VerifyResponseResult,
    VerifyWebSourcesResult,
    WebSourceTrust,
    WebSourceVerification,
)

logger = logging.getLogger("ubp.citation_verification_orchestrator.providers")

# Type alias for the LLM caller callback
LLMCaller = Callable[..., Coroutine[Any, Any, str]]


# ============================================================================
# Prompt helpers (claim extraction & grounding via reasoning_rag templates)
# ============================================================================

CLAIM_EXTRACTION_PROMPT = """Analyze the following text and extract all verifiable factual claims.
For each claim, determine if it needs verification.

TEXT:
{text}

SOURCES AVAILABLE:
{sources}

Respond ONLY with valid JSON:
{{
  "claims": [
    {{"claim": "...", "type": "factual|statistical|causal|attribution", "importance": "high|medium|low", "needs_verification": true}}
  ]
}}"""

GROUNDING_CHECK_PROMPT = """Verify whether each statement in the ANSWER is grounded (supported) by the SOURCE chunks.
Rate each statement's grounding level.

ANSWER:
{answer}

SOURCE CHUNKS:
{sources}

Respond ONLY with valid JSON:
{{
  "grounding_analysis": [
    {{"statement": "...", "grounded": true, "source_support": "strong|partial|none", "hallucination_risk": "low|medium|high"}}
  ],
  "overall_grounding_score": 0.85,
  "hallucination_rate": 0.05
}}"""


# ============================================================================
# VerificationOrchestrator — main provider
# ============================================================================

class VerificationOrchestrator:
    """Coordinates the full verification pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        llm_caller: Optional[LLMCaller] = None,
    ):
        self._config = config
        self._llm_caller = llm_caller

        # Verification config
        vc = config.get("verification", {})
        self._auto_tightness = float(vc.get("auto_verify_threshold_tightness", 0.7))
        self._grounding_min = float(vc.get("grounding_min_score", 0.5))
        self._hallucination_max = float(vc.get("hallucination_max_rate", 0.3))
        self._max_claims = int(vc.get("max_claims_to_verify", 20))
        self._temperature = float(vc.get("verification_temperature", 0.0))

        # Trust config
        tc = config.get("trust", {})
        self._min_trust = float(tc.get("min_trust_score", 0.6))
        self._internal_trusted = bool(tc.get("internal_sources_trusted", True))
        self._auto_learn = bool(tc.get("trust_auto_learn", True))
        self._score_increment = float(tc.get("trust_score_increment", 0.02))
        self._score_decrement = float(tc.get("trust_score_decrement", 0.05))
        self._positive_threshold = float(tc.get("trust_grounding_positive_threshold", 0.8))
        self._negative_threshold = float(tc.get("trust_grounding_negative_threshold", 0.3))
        self._decay_rate = float(tc.get("trust_decay_rate_per_day", 0.001))
        self._bootstrap_enabled = bool(tc.get("bootstrap_from_citations_verifier", True))
        self._trust_ttl_days = int(tc.get("trust_database_ttl_days", 30))

        # Disclaimer
        self._disclaimers = config.get("disclaimer_languages", {
            "it": "Alcune affermazioni potrebbero non essere pienamente supportate dalle fonti disponibili.",
            "en": "Some claims may not be fully supported by available sources.",
        })

    # ------------------------------------------------------------------
    # verify_response
    # ------------------------------------------------------------------

    async def verify_response(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        query: str,
        *,
        tightness: float = 0.0,
        web_sources_present: bool = False,
        force_verification: bool = False,
        language: str = "it",
        citations_verifier: Optional[Any] = None,
        min_tightness_trigger: float = 0.7,
        auto_verify_web_sources: bool = True,
        hallucination_threshold: float = 0.3,
        grounding_min_score: float = 0.5,
    ) -> Dict[str, Any]:
        """Full verification pipeline.

        Returns dict compatible with pipeline step output_as.
        """
        start = time.perf_counter()

        # Use param overrides or fall back to config
        tightness_trigger = min_tightness_trigger or self._auto_tightness
        hall_threshold = hallucination_threshold or self._hallucination_max
        grounding_min = grounding_min_score or self._grounding_min

        # --- Trigger check ---
        should_run = force_verification
        if not should_run and tightness >= tightness_trigger:
            should_run = True
        if not should_run and web_sources_present and auto_verify_web_sources:
            should_run = True

        if not should_run:
            result = VerifyResponseResult(
                skipped=True,
                skip_reason="trigger conditions not met",
            )
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result.to_dict()

        # --- Attempt delegation to citations_verifier ---
        if citations_verifier:
            try:
                cv_result = await citations_verifier.verify_document(
                    text=answer,
                    rag_chunks=chunks,
                    domain="general",
                    depth="standard",
                    language=language,
                )
                return self._transform_cv_result(cv_result, language, start, grounding_min, hall_threshold)
            except Exception as e:
                logger.warning(f"[CVO] citations_verifier.verify_document failed: {e}, falling back to LLM")

        # --- Fallback: LLM-based verification ---
        if not self._llm_caller:
            result = VerifyResponseResult(
                skipped=True,
                skip_reason="no verification backend available (no citations_verifier, no LLM)",
            )
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result.to_dict()

        return await self._verify_via_llm(
            answer, chunks, query, language, start, grounding_min, hall_threshold,
        )

    # ------------------------------------------------------------------
    # filter_trusted_sources
    # ------------------------------------------------------------------

    async def filter_trusted_sources(
        self,
        chunks: List[Dict[str, Any]],
        *,
        min_trust_score: float = 0.6,
        internal_sources_trusted: bool = True,
        citations_verifier: Optional[Any] = None,
        redis_client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Pre-generate filter: keep only chunks from trusted sources."""
        filtered = []
        removed = 0
        trust_summary: List[TrustSummaryEntry] = []
        seen_domains: Dict[str, float] = {}

        for chunk in chunks:
            url = chunk.get("url") or chunk.get("source_url") or ""
            collection_id = chunk.get("collection_id") or chunk.get("collection") or ""

            # Internal KB sources → always trusted
            if not url and collection_id and internal_sources_trusted:
                filtered.append(chunk)
                continue

            if not url:
                filtered.append(chunk)
                continue

            domain = self._extract_domain(url)
            if domain in seen_domains:
                score = seen_domains[domain]
            else:
                score = await self._get_domain_trust(domain, citations_verifier, redis_client)
                seen_domains[domain] = score

            kept = score >= min_trust_score
            trust_summary.append(TrustSummaryEntry(domain=domain, trust_score=score, kept=kept))

            if kept:
                filtered.append(chunk)
            else:
                removed += 1
                logger.info(f"[CVO][TRUST-FILTER] Removed chunk from {domain} (trust={score:.2f} < {min_trust_score})")

        if removed:
            logger.info(f"[CVO][TRUST-FILTER] Removed {removed}/{len(chunks)} chunks (trust < {min_trust_score})")

        result = FilterTrustedResult(
            filtered_chunks=filtered,
            removed_count=removed,
            total_count=len(chunks),
            trust_summary=trust_summary,
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # verify_web_sources
    # ------------------------------------------------------------------

    async def verify_web_sources(
        self,
        urls: List[str],
        *,
        domain: str = "general",
        citations_verifier: Optional[Any] = None,
        redis_client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Trust check on a list of URLs."""
        sources: List[WebSourceVerification] = []
        total_score = 0.0

        for url in urls:
            d = self._extract_domain(url)
            score = await self._get_domain_trust(d, citations_verifier, redis_client)
            level = TrustLevel.from_score(score).value
            in_list = score >= 0.8  # high-trust implies in trust list
            sources.append(WebSourceVerification(
                url=url, domain=d, trust_score=score,
                trust_level=level, in_trust_list=in_list,
            ))
            total_score += score

        avg = total_score / len(urls) if urls else 0.0
        result = VerifyWebSourcesResult(sources=sources, average_trust=round(avg, 3))
        return result.to_dict()

    # ------------------------------------------------------------------
    # update_trust_database
    # ------------------------------------------------------------------

    async def update_trust_database(
        self,
        verification_results: Dict[str, Any],
        sources: List[Dict[str, Any]],
        *,
        redis_client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Auto-update trust scores based on verification grounding."""
        if not self._auto_learn or not redis_client:
            return UpdateTrustResult().to_dict()

        updates: List[TrustUpdateEntry] = []
        for src in sources:
            url = src.get("url") or src.get("source_url") or ""
            if not url:
                continue
            domain = self._extract_domain(url)
            grounding = float(src.get("grounding_score", 0.5))

            try:
                key = f"ubp:trust:domain:{domain}"
                raw = await redis_client.get(key)
                current = json.loads(raw) if raw else {"trust_score": 0.5, "verification_count": 0}
                old_score = float(current.get("trust_score", 0.5))

                if grounding >= self._positive_threshold:
                    new_score = min(1.0, old_score + self._score_increment)
                    direction = "up"
                elif grounding < self._negative_threshold:
                    new_score = max(0.1, old_score - self._score_decrement)
                    direction = "down"
                else:
                    new_score = old_score
                    direction = "stable"

                current["trust_score"] = round(new_score, 4)
                current["verification_count"] = current.get("verification_count", 0) + 1
                current["domain"] = domain

                ttl = int(self._config.get("trust", {}).get("trust_database_ttl_days", 30)) * 86400
                await redis_client.set(key, json.dumps(current), ex=ttl)

                updates.append(TrustUpdateEntry(
                    domain=domain, old_score=old_score,
                    new_score=new_score, direction=direction,
                ))
            except Exception as e:
                logger.warning(f"[CVO] Trust update failed for {domain}: {e}")

        return UpdateTrustResult(updated_domains=len(updates), updates=updates).to_dict()

    # ------------------------------------------------------------------
    # bootstrap_trust_database — load predefined domains to Redis
    # ------------------------------------------------------------------

    async def bootstrap_trust_database(
        self,
        *,
        redis_client: Optional[Any] = None,
        citations_verifier: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Populate Redis with predefined trusted domains from citations_verifier config."""
        if not redis_client or not self._bootstrap_enabled:
            return {"bootstrapped": 0, "skipped": True}

        bootstrapped = 0
        ttl = self._trust_ttl_days * 86400

        # Try to get predefined lists from citations_verifier config
        predefined: Dict[str, List[Dict[str, Any]]] = {}
        if citations_verifier:
            try:
                cfg_path = getattr(citations_verifier, "module_path", None)
                if cfg_path:
                    import pathlib
                    cv_config_path = pathlib.Path(cfg_path) / "config.json"
                    if cv_config_path.exists():
                        with open(cv_config_path, "r", encoding="utf-8") as f:
                            cv_config = json.load(f)
                        # v6.4.2: config uses "predefined_lists" (not "predefined_trusted_sources")
                        predefined = cv_config.get("predefined_lists") or cv_config.get("predefined_trusted_sources", {})
            except Exception as e:
                logger.warning(f"[CVO] Could not read citations_verifier config: {e}")

        if not predefined:
            return {"bootstrapped": 0, "skipped": True, "reason": "no predefined lists"}

        for category, domains_list in predefined.items():
            if not isinstance(domains_list, list):
                continue
            for entry in domains_list:
                if not isinstance(entry, dict):
                    continue
                # v6.4.2: config entries use "url"/"score", accept both formats
                domain = entry.get("domain") or entry.get("url", "")
                score = float(entry.get("trust_score") or entry.get("score", 0.8))
                if not domain:
                    continue

                key = f"ubp:trust:domain:{domain}"
                try:
                    existing = await redis_client.get(key)
                    if existing:
                        continue  # Don't overwrite user/auto-learned scores

                    data = {
                        "domain": domain,
                        "trust_score": round(score, 4),
                        "verification_count": 0,
                        "source": "bootstrap",
                        "category": category,
                        "last_verified": time.time(),
                    }
                    await redis_client.set(key, json.dumps(data), ex=ttl)
                    bootstrapped += 1
                except Exception as e:
                    logger.warning(f"[CVO] Bootstrap failed for {domain}: {e}")

        logger.info(f"[CVO] Trust DB bootstrapped: {bootstrapped} domains from {len(predefined)} categories")
        return {"bootstrapped": bootstrapped, "categories": list(predefined.keys())}

    # ------------------------------------------------------------------
    # get_trust_entry — Redis-first trust lookup with decay
    # ------------------------------------------------------------------

    async def get_trust_entry(
        self,
        domain: str,
        *,
        redis_client: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get trust entry from Redis, applying decay if stale."""
        if not redis_client:
            return None
        try:
            key = f"ubp:trust:domain:{domain}"
            raw = await redis_client.get(key)
            if not raw:
                return None

            entry = json.loads(raw)
            last_verified = float(entry.get("last_verified", time.time()))
            days_stale = (time.time() - last_verified) / 86400

            # Apply decay if not verified recently
            if days_stale > 1.0 and self._decay_rate > 0:
                original = float(entry.get("trust_score", 0.5))
                decayed = max(0.1, original - (self._decay_rate * days_stale))
                entry["trust_score"] = round(decayed, 4)
                entry["decayed"] = True
                entry["days_since_verification"] = round(days_stale, 1)

            return entry
        except Exception:
            return None

    # ------------------------------------------------------------------
    # list_trust_entries — admin listing
    # ------------------------------------------------------------------

    async def list_trust_entries(
        self,
        *,
        redis_client: Optional[Any] = None,
        pattern: str = "ubp:trust:domain:*",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List all trust entries from Redis (admin use)."""
        if not redis_client:
            return []
        entries = []
        try:
            cursor = 0
            while len(entries) < limit:
                cursor, keys = await redis_client.scan(cursor, match=pattern, count=50)
                for k in keys:
                    if len(entries) >= limit:
                        break
                    raw = await redis_client.get(k)
                    if raw:
                        entry = json.loads(raw)
                        entries.append(entry)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning(f"[CVO] list_trust_entries failed: {e}")
        return sorted(entries, key=lambda e: e.get("trust_score", 0), reverse=True)

    # ------------------------------------------------------------------
    # set_trust_entry — admin override
    # ------------------------------------------------------------------

    async def set_trust_entry(
        self,
        domain: str,
        trust_score: float,
        *,
        redis_client: Optional[Any] = None,
        category: str = "admin_override",
    ) -> Dict[str, Any]:
        """Force-set a trust score (admin use)."""
        if not redis_client:
            return {"success": False, "reason": "no redis"}
        try:
            key = f"ubp:trust:domain:{domain}"
            data = {
                "domain": domain,
                "trust_score": round(max(0.0, min(1.0, trust_score)), 4),
                "verification_count": 0,
                "source": "admin_override",
                "category": category,
                "last_verified": time.time(),
            }
            ttl = self._trust_ttl_days * 86400
            await redis_client.set(key, json.dumps(data), ex=ttl)
            return {"success": True, "domain": domain, "trust_score": data["trust_score"]}
        except Exception as e:
            return {"success": False, "reason": str(e)}

    # ------------------------------------------------------------------
    # delete_trust_entry — admin delete
    # ------------------------------------------------------------------

    async def delete_trust_entry(
        self,
        domain: str,
        *,
        redis_client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Delete a trust entry (admin use)."""
        if not redis_client:
            return {"success": False, "reason": "no redis"}
        try:
            key = f"ubp:trust:domain:{domain}"
            deleted = await redis_client.delete(key)
            return {"success": bool(deleted), "domain": domain}
        except Exception as e:
            return {"success": False, "reason": str(e)}

    def _transform_cv_result(
        self,
        cv_result: Dict[str, Any],
        language: str,
        start: float,
        grounding_min: float,
        hall_threshold: float,
    ) -> Dict[str, Any]:
        """Transform citations_verifier output to our standard schema."""
        report = cv_result if isinstance(cv_result, dict) else {}

        claims_total = int(report.get("claims_total", 0))
        claims_verified = int(report.get("claims_supported", report.get("claims_verified", 0)))
        claims_unverified = claims_total - claims_verified
        grounding = float(report.get("trust_score", report.get("grounding_score", 0.0)))
        hall_rate = float(report.get("hallucination_rate", 0.0))

        # Determine status
        if grounding >= grounding_min and hall_rate <= hall_threshold:
            status = VerificationStatus.VERIFIED
            verified = True
        elif grounding >= grounding_min * 0.6:
            status = VerificationStatus.PARTIALLY_VERIFIED
            verified = False
        else:
            status = VerificationStatus.UNVERIFIED
            verified = False

        # Disclaimer
        disclaimer = None
        if not verified:
            disclaimer = self._disclaimers.get(language, self._disclaimers.get("en"))

        # Web sources trust
        web_trust: List[WebSourceTrust] = []
        for s in report.get("source_details", report.get("sources", [])):
            if isinstance(s, dict) and s.get("domain"):
                score = float(s.get("trust_score", s.get("confidence", 0.5)))
                web_trust.append(WebSourceTrust(
                    domain=s["domain"],
                    trust_score=score,
                    trust_level=TrustLevel.from_score(score).value,
                ))

        result = VerifyResponseResult(
            verified=verified,
            grounding_score=round(grounding, 3),
            hallucination_rate=round(hall_rate, 3),
            claims_total=claims_total,
            claims_verified=claims_verified,
            claims_unverified=claims_unverified,
            web_sources_trust=web_trust,
            disclaimer=disclaimer,
            status=status.value,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return result.to_dict()

    async def _verify_via_llm(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        query: str,
        language: str,
        start: float,
        grounding_min: float,
        hall_threshold: float,
    ) -> Dict[str, Any]:
        """Fallback verification using LLM prompts (via _call_llm → simple_chat)."""
        sources_text = self._format_chunks_for_prompt(chunks)

        # Step 1: Grounding check
        prompt = GROUNDING_CHECK_PROMPT.format(answer=answer, sources=sources_text)
        try:
            raw = await self._llm_caller(prompt, max_tokens=2000, temperature=self._temperature)
            analysis = self._parse_json_response(raw)
        except Exception as e:
            logger.warning(f"[CVO] LLM grounding check failed: {e}")
            analysis = {}

        grounding = float(analysis.get("overall_grounding_score", 0.5))
        hall_rate = float(analysis.get("hallucination_rate", 0.0))

        statements = analysis.get("grounding_analysis", [])
        claims_total = len(statements)
        claims_verified = sum(1 for s in statements if s.get("grounded", False))
        claims_unverified = claims_total - claims_verified

        # Determine status
        if grounding >= grounding_min and hall_rate <= hall_threshold:
            status = VerificationStatus.VERIFIED
            verified = True
        elif grounding >= grounding_min * 0.6:
            status = VerificationStatus.PARTIALLY_VERIFIED
            verified = False
        else:
            status = VerificationStatus.UNVERIFIED
            verified = False

        disclaimer = None
        if not verified:
            disclaimer = self._disclaimers.get(language, self._disclaimers.get("en"))

        result = VerifyResponseResult(
            verified=verified,
            grounding_score=round(grounding, 3),
            hallucination_rate=round(hall_rate, 3),
            claims_total=claims_total,
            claims_verified=claims_verified,
            claims_unverified=claims_unverified,
            disclaimer=disclaimer,
            status=status.value,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return result.to_dict()

    async def _get_domain_trust(
        self, domain: str, citations_verifier: Optional[Any] = None,
        redis_client: Optional[Any] = None,
    ) -> float:
        """Get trust score: Redis (with decay) → citations_verifier → TLD heuristic."""
        # 1. Redis first (with decay applied)
        if redis_client:
            entry = await self.get_trust_entry(domain, redis_client=redis_client)
            if entry:
                return float(entry.get("trust_score", 0.5))

        # 2. citations_verifier trusted list
        if citations_verifier:
            try:
                result = await citations_verifier.get_trusted_sources(
                    domain="general", format="domains",
                )
                domains = result.get("domains", []) if isinstance(result, dict) else []
                if domain in domains:
                    return 0.9
            except Exception:
                pass

        # 3. Fallback: TLD-based heuristic
        return self._tld_trust_score(domain)

    @staticmethod
    def _tld_trust_score(domain: str) -> float:
        """Simple TLD-based trust heuristic."""
        if domain.endswith(".gov") or domain.endswith(".gov.it"):
            return 0.95
        if domain.endswith(".edu"):
            return 0.90
        if domain.endswith(".org"):
            return 0.70
        if domain.endswith(".ac.uk") or domain.endswith(".ac.jp"):
            return 0.90
        return 0.5

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower().removeprefix("www.") if parsed.netloc else url.lower()
        except Exception:
            return url.lower()

    @staticmethod
    def _format_chunks_for_prompt(chunks: List[Dict[str, Any]], max_chunks: int = 10) -> str:
        parts = []
        for i, c in enumerate(chunks[:max_chunks]):
            text = c.get("text") or c.get("content") or c.get("page_content") or ""
            source = c.get("source") or c.get("collection") or f"chunk_{i}"
            parts.append(f"[Source {i+1}: {source}]\n{text[:800]}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_json_response(raw: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, tolerant of markdown fences."""
        raw = raw.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            raw = match.group(1)
        elif raw.startswith("{"):
            pass
        else:
            brace = raw.find("{")
            if brace >= 0:
                raw = raw[brace:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
