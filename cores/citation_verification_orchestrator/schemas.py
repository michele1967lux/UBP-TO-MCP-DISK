"""
Pydantic-free I/O schemas for citation_verification_orchestrator.

All models use dataclass + to_dict/from_dict for zero-dependency serialization,
consistent with the citations_verifier provider layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TrustLevel(str, Enum):
    HIGH = "high"        # >= 0.8
    MEDIUM = "medium"    # >= 0.5
    LOW = "low"          # < 0.5
    UNKNOWN = "unknown"

    @classmethod
    def from_score(cls, score: float) -> "TrustLevel":
        if score >= 0.8:
            return cls.HIGH
        if score >= 0.5:
            return cls.MEDIUM
        return cls.LOW


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Request / Result: verify_response
# ---------------------------------------------------------------------------

@dataclass
class VerifyResponseRequest:
    answer: str
    chunks: List[Dict[str, Any]]
    query: str
    min_tightness_trigger: float = 0.7
    auto_verify_web_sources: bool = True
    hallucination_threshold: float = 0.3
    grounding_min_score: float = 0.5
    force_verification: bool = False
    language: str = "it"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VerifyResponseRequest":
        return cls(
            answer=d.get("answer", ""),
            chunks=d.get("chunks", []),
            query=d.get("query", ""),
            min_tightness_trigger=float(d.get("min_tightness_trigger", 0.7)),
            auto_verify_web_sources=bool(d.get("auto_verify_web_sources", True)),
            hallucination_threshold=float(d.get("hallucination_threshold", 0.3)),
            grounding_min_score=float(d.get("grounding_min_score", 0.5)),
            force_verification=bool(d.get("force_verification", False)),
            language=d.get("language", "it"),
        )


@dataclass
class WebSourceTrust:
    domain: str
    trust_score: float
    trust_level: str  # TrustLevel value

    def to_dict(self) -> Dict[str, Any]:
        return {"domain": self.domain, "trust_score": self.trust_score, "trust_level": self.trust_level}


@dataclass
class VerifyResponseResult:
    verified: bool = False
    grounding_score: Optional[float] = None
    hallucination_rate: Optional[float] = None
    claims_total: int = 0
    claims_verified: int = 0
    claims_unverified: int = 0
    trust_filtered: bool = False
    web_sources_trust: List[WebSourceTrust] = field(default_factory=list)
    disclaimer: Optional[str] = None
    status: str = VerificationStatus.SKIPPED.value
    skipped: bool = False
    skip_reason: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "grounding_score": self.grounding_score,
            "hallucination_rate": self.hallucination_rate,
            "claims_total": self.claims_total,
            "claims_verified": self.claims_verified,
            "claims_unverified": self.claims_unverified,
            "trust_filtered": self.trust_filtered,
            "web_sources_trust": [s.to_dict() for s in self.web_sources_trust],
            "disclaimer": self.disclaimer,
            "status": self.status,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Request / Result: filter_trusted_sources
# ---------------------------------------------------------------------------

@dataclass
class FilterTrustedRequest:
    chunks: List[Dict[str, Any]]
    min_trust_score: float = 0.6
    internal_sources_trusted: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FilterTrustedRequest":
        return cls(
            chunks=d.get("chunks", []),
            min_trust_score=float(d.get("min_trust_score", 0.6)),
            internal_sources_trusted=bool(d.get("internal_sources_trusted", True)),
        )


@dataclass
class TrustSummaryEntry:
    domain: str
    trust_score: float
    kept: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"domain": self.domain, "trust_score": self.trust_score, "kept": self.kept}


@dataclass
class FilterTrustedResult:
    filtered_chunks: List[Dict[str, Any]] = field(default_factory=list)
    removed_count: int = 0
    total_count: int = 0
    trust_summary: List[TrustSummaryEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filtered_chunks": self.filtered_chunks,
            "removed_count": self.removed_count,
            "total_count": self.total_count,
            "trust_summary": [e.to_dict() for e in self.trust_summary],
        }


# ---------------------------------------------------------------------------
# Request / Result: verify_web_sources
# ---------------------------------------------------------------------------

@dataclass
class VerifyWebSourcesRequest:
    urls: List[str]
    domain: str = "general"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VerifyWebSourcesRequest":
        return cls(urls=d.get("urls", []), domain=d.get("domain", "general"))


@dataclass
class WebSourceVerification:
    url: str
    domain: str
    trust_score: float
    trust_level: str
    in_trust_list: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url, "domain": self.domain, "trust_score": self.trust_score,
            "trust_level": self.trust_level, "in_trust_list": self.in_trust_list,
        }


@dataclass
class VerifyWebSourcesResult:
    sources: List[WebSourceVerification] = field(default_factory=list)
    average_trust: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sources": [s.to_dict() for s in self.sources],
            "average_trust": self.average_trust,
        }


# ---------------------------------------------------------------------------
# Request / Result: update_trust_database
# ---------------------------------------------------------------------------

@dataclass
class TrustUpdateEntry:
    domain: str
    old_score: float
    new_score: float
    direction: str  # "up" | "down" | "stable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain, "old_score": self.old_score,
            "new_score": self.new_score, "direction": self.direction,
        }


@dataclass
class UpdateTrustResult:
    updated_domains: int = 0
    updates: List[TrustUpdateEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "updated_domains": self.updated_domains,
            "updates": [u.to_dict() for u in self.updates],
        }
