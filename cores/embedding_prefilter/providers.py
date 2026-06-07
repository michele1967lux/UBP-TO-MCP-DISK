"""
embedding_prefilter/providers.py — EmbeddingPrefilterEngine (Pure Logic, Zero UBP Imports).

Core logic pura per il meta-routing brain a 4 layer.
Contiene:
- EmbeddingPrefilterEngine: 4-layer decision engine
- RouteStabilityGuard: R3 (preparato, non attivo Phase 1)
- ConfidenceCalibrator: R6 (preparato, non attivo Phase 1)

Layer architecture:
  Layer 1 - Embedding: cosine similarity scoring vs centroids
  Layer 2 - Semantic Scoring: softmax normalization
  Layer 3 - Evidence Enrichment: (disabled Phase 1)
  Layer 4 - Decision Engine: final route selection

Migrato da pipeline_router/embedding_classifier.py (432 LOC) come base.
"""

from __future__ import annotations

import asyncio
import logging
import math
import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np

from .schemas import (
    CLUSTER_TO_ROUTE,
    ROUTE_SEVERITY,
    SEVERITY_BASE_PENALTY,
    VALID_ROUTES,
    InteractionOption,
    InteractionOptions,
    PreRouteDecision,
    RouteScoreBreakdown,
    RoutingProfile,
    StabilityInfo,
)

logger = logging.getLogger("ubp.embedding_prefilter")


# ============================================================================
# Exemplar sets — 30+ per cluster, multilingua, varianti brevi/lunghe
# Migrati integralmente da pipeline_router/embedding_classifier.py
# DEPRECATED: Migrated to Qdrant collection 'routing_prototypes_v2' (ARCH-007).
# Kept as fallback when Qdrant is unavailable. See config.json prototype_fallback.
# ============================================================================

INTENT_EXEMPLARS: Dict[str, List[str]] = {
    "chat": [
        # --- IT brevi ---
        "ciao", "salve", "buongiorno", "buonasera", "buonanotte",
        "come stai", "come va", "chi sei", "come ti chiami", "piacere",
        "grazie mille", "tutto bene",
        # --- IT medie ---
        "raccontami qualcosa di te", "cosa sai fare",
        "parlami un po' di te", "sei un'intelligenza artificiale",
        "come posso chiamarti", "mi fai compagnia",
        "dimmi una curiosita'", "che ore sono",
        "che giorno e' oggi", "raccontami una barzelletta",
        "mi annoio un po'", "facciamo due chiacchiere",
        # --- EN brevi ---
        "hello", "hi", "hey", "good morning", "good evening",
        "how are you", "what's up", "who are you", "what's your name",
        "nice to meet you", "tell me about yourself", "thanks",
    ],
    "rag": [
        # --- IT brevi ---
        "effetti del paracetamolo", "cos'e' il diabete",
        "spiegami la fotosintesi", "trattamento dell'ipertensione",
        # --- IT medie ---
        "quali sono gli effetti collaterali del paracetamolo",
        "spiegami la funzione epatica in dettaglio",
        "cosa dice la documentazione su questo argomento",
        "quali evidenze scientifiche supportano questa tesi",
        "descrivi il processo di diagnosi differenziale",
        "quali sono le linee guida per il trattamento",
        "informazioni sulla prevenzione cardiovascolare",
        "analisi dei fattori di rischio principali",
        "come funziona questo meccanismo biologico",
        "qual e' la differenza tra tipo 1 e tipo 2",
        "spiega il protocollo terapeutico standard",
        "quali controindicazioni ha questo farmaco",
        "come si calcola il dosaggio appropriato",
        "cosa sono gli anticorpi monoclonali",
        "descrivi la cascata infiammatoria",
        # --- IT lunghe ---
        "vorrei capire meglio come funziona il sistema immunitario nella risposta alle infezioni virali",
        "puoi spiegarmi i meccanismi d'azione dei farmaci antinfiammatori non steroidei",
        "quali sono i criteri diagnostici per la sindrome metabolica secondo le linee guida europee",
        # --- EN ---
        "what are the side effects of ibuprofen",
        "explain the cardiovascular risk factors",
        "describe the diagnostic criteria for this condition",
        "how does the immune system respond to viral infections",
        "what is the recommended treatment protocol",
        "summarize the latest research on this topic",
        "explain the mechanism of action of this drug",
        "what are the contraindications for this medication",
        "how does gene therapy work",
        "describe the pathophysiology of heart failure",
    ],
    "report": [
        # --- IT brevi ---
        "genera un report", "crea un documento",
        "scrivi un riassunto", "prepara un'analisi",
        # --- IT medie ---
        "genera un report completo su questo argomento",
        "crea un documento di analisi dettagliata con grafici",
        "scrivi un riassunto esecutivo con raccomandazioni",
        "prepara una presentazione strutturata dei risultati",
        "analisi approfondita con conclusioni e punti chiave",
        "report strutturato con introduzione e conclusioni",
        "produci un documento tecnico formale",
        "elabora un piano d'azione con priorita'",
        "redigi un verbale della situazione attuale",
        "compila una scheda riassuntiva con i dati principali",
        "crea una relazione tecnica su questo tema",
        "genera un documento con indice e sezioni",
        "prepara un executive summary per il management",
        # --- IT lunghe ---
        "vorrei un report dettagliato che analizzi tutti gli aspetti della questione con dati e riferimenti",
        "genera un documento strutturato che includa introduzione metodologia risultati e conclusioni",
        "prepara un'analisi comparativa tra i diversi approcci terapeutici con tabelle e grafici",
        # --- EN ---
        "generate a comprehensive report on this topic",
        "create a detailed analysis document with charts",
        "write an executive summary with recommendations",
        "prepare a structured presentation of the findings",
        "produce a technical document with data and references",
        "compile a summary report with key metrics",
        "draft a formal report covering all aspects",
        "create a comparative analysis document",
        "generate a white paper on this subject",
        "write a detailed assessment with action items",
    ],
    "web_search": [
        # --- IT brevi ---
        "cerca online", "notizie di oggi",
        "ultime novita'", "meteo oggi",
        # --- IT medie ---
        "cerca online le ultime notizie su questo argomento",
        "cosa si dice nel web riguardo a questa questione",
        "confronta con fonti online recenti",
        "aggiornamenti recenti dal web su questo tema",
        "notizie di oggi sulla situazione attuale",
        "cerca nel web informazioni aggiornate",
        "quali sono le ultime novita' su questo argomento",
        "hai notizie recenti riguardo a questo",
        "cerca su internet cosa dicono gli esperti",
        "verifica online se ci sono aggiornamenti",
        "cosa dicono le fonti web su questa questione",
        "controlla le notizie piu' recenti",
        "fai una ricerca web su questo tema",
        # --- IT lunghe ---
        "cerca online le ultime pubblicazioni scientifiche su questo argomento e confrontale con i dati esistenti",
        "vorrei che cercassi nel web le notizie piu' recenti e mi facessi un riassunto delle fonti principali",
        # --- EN ---
        "search the web for the latest news on this topic",
        "find recent online information about this",
        "check the latest updates from the web",
        "look up what experts are saying online",
        "search for recent publications on this subject",
        "find the latest news and developments",
        "what does the internet say about this",
        "search online for current information",
        "look up the latest research papers on this",
        "find me recent web articles about this topic",
    ],
}

# Total: chat=36, rag=34, report=30, web_search=30 = 130 exemplars


# ============================================================================
# ============================================================================
# Math utilities
# ============================================================================

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, safe for zero vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _softmax(scores: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """Softmax normalization of score dict."""
    if not scores:
        return {}
    if temperature <= 0:
        temperature = 1.0
    vals = np.array(list(scores.values()), dtype=np.float64)
    vals = vals / temperature
    # Numerical stability: subtract max
    vals = vals - np.max(vals)
    exp_vals = np.exp(vals)
    total = exp_vals.sum()
    if total < 1e-12:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    probs = exp_vals / total
    return {k: float(p) for k, p in zip(scores.keys(), probs)}


def _entropy(probs: Dict[str, float]) -> float:
    """Shannon entropy of probability distribution."""
    arr = np.array(list(probs.values()), dtype=np.float64)
    arr = np.clip(arr, 1e-12, None)
    return float(-np.sum(arr * np.log2(arr)))


# ============================================================================
# R3: RouteStabilityGuard (preparato, non attivo Phase 1)
# ============================================================================

class RouteStabilityGuard:
    """Anti-flapping guard using per-user routing profile from Redis.

    Phase 1: Completely bypassed when r3_stability_guard.enabled = false.
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config.get("reinforcements", {}).get("r3_stability_guard", {})
        self._window_size = self._config.get("window_size", 5)
        self._flap_threshold = self._config.get("flap_threshold", 3)

    async def check_stability(
        self,
        user_id: str,
        proposed_route: str,
        redis: Any = None,
    ) -> Optional[str]:
        """Check if proposed route causes flapping. Returns stabilized route or None.

        Phase 1: Always returns None (bypassed).
        """
        if not self._config.get("enabled", False):
            return None  # complete bypass, zero computation
        # Future: implement stability logic with Redis
        return None

    async def update_profile(
        self,
        user_id: str,
        final_route: str,
        redis: Any = None,
    ) -> None:
        """Update user's routing profile after final decision.

        Phase 1: No-op (bypassed).
        """
        if not self._config.get("enabled", False):
            return  # complete bypass, zero computation
        # Future: update Redis routing profile
        return


# ============================================================================
# R6: ConfidenceCalibrator (preparato, non attivo Phase 1)
# ============================================================================

class ConfidenceCalibrator:
    """Softmax masking and confidence calibration.

    Phase 1: Completely bypassed when r6_softmax_masking.enabled = false.
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config.get("reinforcements", {}).get("r6_softmax_masking", {})

    def calibrate(
        self,
        raw_scores: Dict[str, float],
        global_bias: float = 0.0,
        severity_map: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Calibrate raw scores with bias and severity adjustments.

        Phase 1: Returns raw_scores unchanged (bypassed).
        """
        if not self._config.get("enabled", False):
            return dict(raw_scores)  # complete bypass
        # Future: apply calibration
        return dict(raw_scores)

    def apply_masking(
        self,
        scores: Dict[str, float],
        baseline_threshold: float = 0.45,
    ) -> Dict[str, float]:
        """Mask routes below baseline threshold.

        Phase 1: Returns scores unchanged (bypassed).
        """
        if not self._config.get("enabled", False):
            return dict(scores)  # complete bypass
        # Future: apply masking
        return dict(scores)


# ============================================================================
# Core Engine: EmbeddingPrefilterEngine
# ============================================================================

class EmbeddingPrefilterEngine:
    """
    4-layer embedding-based meta-routing brain.

    Pure logic, zero UBP imports. Uses injected embed_fn for embeddings.

    Layer 1 - kNN Search: cosine similarity vs individual prototypes in RAM
    Layer 2 - Semantic Scoring: min-max normalization -> per-route scores
    Layer 3 - Evidence Enrichment: midband delta < threshold triggers RAG/Web evidence
    Layer 4 - Decision Engine: final route, confidence, defer logic

    Phase 2C: kNN Ensemble replaces centroid-based scoring.
    """

    # Lane name → route name mapping
    LANE_TO_ROUTE = {
        "chat": "FAST", "rag": "RAG",
        "web_search": "WEB", "report": "REPORT",
    }

    def __init__(
        self,
        embed_fn: Callable[[str], Awaitable[List[float]]],
        config: Dict[str, Any],
        qdrant_loader: Optional[Callable] = None,
        qdrant_search_fn: Optional[Callable] = None,
        vllm_fn: Optional[Callable] = None,
    ):
        self._embed = embed_fn
        self._config = config
        self._initialized = False
        self._qdrant_loader = qdrant_loader  # Injected by adapter for prototype loading
        self._qdrant_search_fn = qdrant_search_fn  # Injected by adapter for L3 RAG Preview
        self._vllm_fn = vllm_fn  # Injected by adapter for R2 vLLM disambiguation
        self._reload_in_progress = False

        # R1 timeout recovery: L2.5 scores saved before L3 (async Qdrant)
        self._last_pre_l3_scores: Optional[Dict[str, float]] = None

        # kNN prototype cache (replaces centroids)
        self._prototype_cache: Optional[Dict[str, Any]] = None
        self._prototype_cache_primary: Optional[Dict[str, Any]] = None
        self._prototype_cache_shadow: Optional[Dict[str, Any]] = None
        # Legacy centroid support (fallback only)
        self._centroids: Dict[str, np.ndarray] = {}
        self._centroid_source = "none"

        # Primary + shadow collection settings
        l1_cfg = config.get("layer1_knn", config.get("layer1_embedding", {}))
        self._primary_collection = l1_cfg.get(
            "collection", l1_cfg.get("prototype_collection", "routing_prototypes_v2")
        )
        shadow_cfg = config.get("shadow_mode", {})
        self._shadow_enabled = bool(shadow_cfg.get("enabled", False))
        self._shadow_collection = str(
            shadow_cfg.get("shadow_collection", "routing_prototypes_v2")
        )
        self._shadow_timeout_ms = int(shadow_cfg.get("timeout_ms", 40))
        self._shadow_sample_rate = float(shadow_cfg.get("sample_rate", 0.20))
        self._shadow_log_tag = str(shadow_cfg.get("log_tag", "PREFILTER-SHADOW"))

        # Layer 1 config (kNN)
        self._top_k = l1_cfg.get("top_k", 15)
        self._anti_lambda = l1_cfg.get("anti_lambda", 0.3)
        self._min_lane_hits = l1_cfg.get("min_lane_hits", 1)
        self._matryoshka_dim = l1_cfg.get("matryoshka_dim",
                                           config.get("layer1_embedding", {}).get("matryoshka_dim", 128))
        self._context_weight = l1_cfg.get("context_weight",
                                           config.get("layer1_embedding", {}).get("context_weight", 0.25))

        # Layer 2 config
        l2_cfg = config.get("layer2_semantic_scoring", {})
        self._softmax_temperature = l2_cfg.get("softmax_temperature", 0.45)

        # Layer 4 config
        l4_cfg = config.get("layer4_decision_engine", {})
        self._high_confidence_threshold = l4_cfg.get("high_confidence_threshold", 0.80)
        self._defer_threshold = l4_cfg.get("defer_to_llm_router_threshold", 0.55)
        self._entropy_threshold = l4_cfg.get("entropy_uncertainty_threshold", 0.85)
        self._min_top2_delta = l4_cfg.get("min_top2_delta", 0.15)

        # R2 Dynamic Interaction config
        r2_cfg = config.get("reinforcements", {}).get("r2_dynamic_interaction", {})
        self._r2_enabled = r2_cfg.get("enabled", False)
        self._r2_confidence_threshold = r2_cfg.get("confidence_threshold", 0.70)
        self._r2_delta_threshold = r2_cfg.get("delta_threshold", 0.15)
        self._r2_excluded_routes = set(r2_cfg.get("excluded_routes", ["FAST"]))
        self._r2_vllm_enabled = r2_cfg.get("vllm_enabled", True)
        self._r2_vllm_timeout = r2_cfg.get("vllm_timeout_s", 3.0)

        # R3, R6 helpers (prepared, bypassed in Phase 1)
        self._stability_guard = RouteStabilityGuard(config)
        self._calibrator = ConfidenceCalibrator(config)

        # L2.5 External Signal config
        l25_cfg = config.get("l2_5_signals", {})
        self._l25_rag_boost_factor = l25_cfg.get("rag_boost_factor", 0.15)
        self._l25_web_boost_factor = l25_cfg.get("web_boost_factor", 0.15)
        self._l25_rag_penalty = l25_cfg.get("rag_penalty", -0.12)
        self._l25_web_penalty = l25_cfg.get("web_penalty", -0.10)
        self._l25_freshness_rag_dampening = l25_cfg.get("freshness_rag_dampening", 0.5)
        self._l25_freshness_web_amplify = l25_cfg.get("freshness_web_amplify", 0.3)
        self._l25_freshness_threshold = l25_cfg.get("freshness_threshold", 0.5)
        self._l25_rag_freshness_exempt_above = l25_cfg.get("rag_freshness_exempt_above", 0.75)
        self._l25_web_boost_cap = l25_cfg.get("web_boost_cap", 0.20)

        # Freshness signal config
        fs_cfg = config.get("freshness_signal", {})
        self._fs_recent_days = fs_cfg.get("recent_days_threshold", 7)
        self._fs_recent_high_count = fs_cfg.get("recent_high_count", 3)
        self._fs_recent_high_boost = fs_cfg.get("recent_high_boost", 0.4)
        self._fs_recent_low_boost = fs_cfg.get("recent_low_boost", 0.2)
        self._fs_marker_multi_boost = fs_cfg.get("marker_multi_boost", 0.4)
        self._fs_marker_single_boost = fs_cfg.get("marker_single_boost", 0.25)
        self._fs_gap_web_min = fs_cfg.get("gap_web_min", 0.5)
        self._fs_gap_rag_max = fs_cfg.get("gap_rag_max", 0.35)
        self._fs_gap_factor = fs_cfg.get("gap_factor", 0.5)
        self._fs_temporal_markers_it = fs_cfg.get("temporal_markers_it", [
            "oggi", "adesso", "attuale", "attuali", "ultimo", "ultimi",
            "recente", "recenti", "domani", "stasera", "stanotte",
            "questa settimana", "questo mese", "quest'anno",
            "in tempo reale", "live", "aggiornato", "aggiornati",
            "ora", "orari", "orario",
        ])
        self._fs_temporal_markers_en = fs_cfg.get("temporal_markers_en", [
            "today", "now", "current", "latest", "recent",
            "tomorrow", "tonight", "this week", "this month",
            "this year", "real-time", "live", "updated", "schedule",
        ])
        self._fs_year_markers = fs_cfg.get("year_markers", ["2026", "2025"])

        # STEP 6: Routing metrics
        self._metrics = {
            "total_queries": 0,
            "direct_routes": 0,
            "deferred_to_llm": 0,
            "dynamic_interaction": 0,
            "ood_floor_rejected": 0,
            "low_range_deferred": 0,
            "route_distribution": {},
            "confidence_buckets": {"0.9+": 0, "0.8-0.9": 0, "0.7-0.8": 0, "0.6-0.7": 0, "<0.6": 0},
            "max_pos_sim_buckets": {"0.8+": 0, "0.6-0.8": 0, "0.4-0.6": 0, "<0.4": 0},
            "shadow": {
                "total": 0,
                "route_changed_count": 0,
                "defer_changed_count": 0,
                "avg_conf_delta": 0.0,
                "conf_delta_sum": 0.0,
                "timeouts": 0,
                "errors": 0,
                "skipped": 0,
                "route_matrix": {},
            },
        }

        # STEP 7: Auto-suggest prototype candidates (circular buffer max 100)
        self._prototype_candidates: List[Dict[str, Any]] = []

    async def _safe_embed(self, text: str) -> List[float]:
        """Safely call embed_fn, handling both sync and async implementations."""
        result = self._embed(text)
        if hasattr(result, '__await__'):
            result = await result
        return result

    def configure_shadow_runtime(
        self,
        *,
        enabled: Optional[bool] = None,
        shadow_collection: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        sample_rate: Optional[float] = None,
        log_tag: Optional[str] = None,
    ) -> None:
        """Update shadow-mode runtime knobs (safe, no Qdrant writes)."""
        if enabled is not None:
            self._shadow_enabled = bool(enabled)
        if shadow_collection:
            new_collection = str(shadow_collection).strip()
            if new_collection and new_collection != self._shadow_collection:
                self._shadow_collection = new_collection
                self._prototype_cache_shadow = None
        if timeout_ms is not None:
            self._shadow_timeout_ms = max(1, int(timeout_ms))
        if sample_rate is not None:
            try:
                sr = float(sample_rate)
            except Exception:
                sr = self._shadow_sample_rate
            self._shadow_sample_rate = max(0.0, min(1.0, sr))
        if log_tag:
            self._shadow_log_tag = str(log_tag).strip() or "PREFILTER-SHADOW"

    def _compute_top2_delta(self, scores: Optional[Dict[str, float]]) -> float:
        ranked = sorted((scores or {}).items(), key=lambda x: x[1], reverse=True)
        if len(ranked) < 2:
            return 1.0
        return float(ranked[0][1] - ranked[1][1])

    def _compute_freshness_signal(
        self,
        query: str,
        web_prefetch_results: list,
        web_score: float,
        rag_score: float,
    ) -> float:
        """Compute freshness signal 0.0-1.0. Higher = query needs fresh data.

        Three signal sources:
        1. Web prefetch results with recent dates
        2. Temporal markers in query
        3. Gap: web_score vs rag_score
        """
        signals = 0.0
        components: list = []

        # --- Signal 1: Recent web results ---
        if web_prefetch_results:
            recent_count = 0
            for r in web_prefetch_results:
                pub_date = (
                    r.get("published_date")
                    or r.get("publishedDate")
                    or r.get("date")
                )
                if pub_date and self._is_recent(pub_date, days=self._fs_recent_days):
                    recent_count += 1
            if recent_count >= self._fs_recent_high_count:
                signals += self._fs_recent_high_boost
                components.append(f"recent_results={recent_count}(+{self._fs_recent_high_boost})")
            elif recent_count >= 1:
                signals += self._fs_recent_low_boost
                components.append(f"recent_results={recent_count}(+{self._fs_recent_low_boost})")

        # --- Signal 2: Temporal markers in query ---
        query_lower = query.lower()
        all_markers = (
            self._fs_temporal_markers_it
            + self._fs_temporal_markers_en
            + self._fs_year_markers
        )
        matched_markers = [m for m in all_markers if m in query_lower]
        if len(matched_markers) >= 2:
            signals += self._fs_marker_multi_boost
            components.append(f"markers={matched_markers}(+{self._fs_marker_multi_boost})")
        elif len(matched_markers) == 1:
            signals += self._fs_marker_single_boost
            components.append(f"markers={matched_markers}(+{self._fs_marker_single_boost})")

        # --- Signal 3: Gap web vs rag ---
        if web_score > self._fs_gap_web_min and rag_score < self._fs_gap_rag_max:
            gap_signal = (web_score - rag_score) * self._fs_gap_factor
            signals += gap_signal
            components.append(
                f"gap={web_score:.2f}-{rag_score:.2f}(+{gap_signal:.2f})"
            )

        result = min(signals, 1.0)

        if result > 0.0:
            logger.info(
                "[FRESHNESS-SIGNAL] query=%s | signal=%.3f | components=%s",
                query[:60], result, ", ".join(components),
            )

        return result

    @staticmethod
    def _is_recent(date_str: str, days: int = 7) -> bool:
        """Check if a date string is within N days from today."""
        from datetime import datetime, timedelta
        try:
            for fmt in (
                "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                "%d/%m/%Y", "%B %d, %Y", "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    parsed = datetime.strptime(date_str.strip(), fmt)
                    return (datetime.now() - parsed) < timedelta(days=days)
                except ValueError:
                    continue
            return False
        except Exception:
            return False
        shadow = self._metrics.setdefault("shadow", {})
        shadow["total"] = int(shadow.get("total", 0)) + 1

        status = payload.get("shadow", {}).get("status", "ok")
        if status == "timeout":
            shadow["timeouts"] = int(shadow.get("timeouts", 0)) + 1
        elif status == "error":
            shadow["errors"] = int(shadow.get("errors", 0)) + 1
        elif payload.get("shadow", {}).get("skipped"):
            shadow["skipped"] = int(shadow.get("skipped", 0)) + 1

        diff = payload.get("diff", {})
        if diff.get("route_changed"):
            shadow["route_changed_count"] = int(shadow.get("route_changed_count", 0)) + 1
        if diff.get("defer_changed"):
            shadow["defer_changed_count"] = int(shadow.get("defer_changed_count", 0)) + 1

        conf_delta = float(diff.get("conf_delta", 0.0))
        shadow["conf_delta_sum"] = float(shadow.get("conf_delta_sum", 0.0)) + conf_delta
        total = max(int(shadow.get("total", 1)), 1)
        shadow["avg_conf_delta"] = float(shadow.get("conf_delta_sum", 0.0)) / total

        primary_route = str(payload.get("primary", {}).get("route", "UNKNOWN"))
        shadow_route = str(payload.get("shadow", {}).get("route", "UNKNOWN"))
        matrix = shadow.setdefault("route_matrix", {})
        matrix_key = f"{primary_route}->{shadow_route}"
        matrix[matrix_key] = int(matrix.get(matrix_key, 0)) + 1

    def _log_shadow_decision(self, payload: Dict[str, Any]) -> None:
        logger.info(f"[{self._shadow_log_tag}] {json.dumps(payload, ensure_ascii=False)}")

    async def _ensure_shadow_cache(self) -> Optional[Dict[str, Any]]:
        if not self._shadow_collection:
            return None

        current_meta = (
            self._prototype_cache_shadow.get("_metadata", {})
            if self._prototype_cache_shadow else {}
        )
        if current_meta.get("collection") == self._shadow_collection:
            return self._prototype_cache_shadow

        cache = await self._load_prototypes_for_knn(collection_name=self._shadow_collection)
        if not cache:
            return None
        self._prototype_cache_shadow = cache
        return cache

    async def _evaluate_shadow_without_l3(
        self,
        *,
        decision_id: str,
        query_vec_routing: np.ndarray,
    ) -> Dict[str, Any]:
        """Run L1+L2+softmax+L4 on shadow cache only (no L3 enrichment)."""
        cache = await self._ensure_shadow_cache()
        if not cache:
            return {
                "status": "error",
                "error": f"shadow_collection_unavailable:{self._shadow_collection}",
                "skipped": True,
            }

        t0 = time.perf_counter()
        shadow_id = f"{decision_id}:shadow"
        layer_trace: List[str] = ["S0:shadow"]

        lane_scores, lane_uncertainty, _lane_evidence, max_pos_similarity, l1_meta = \
            self._layer1_knn_search(query_vec_routing, shadow_id, cache_override=cache)
        layer_trace.append("L1:knn_search_shadow")

        if not lane_scores:
            decision = self._make_defer_decision(shadow_id, "no_lane_scores", t0, layer_trace)
            return {
                "status": "ok",
                "decision": decision,
                "scores": {},
                "max_pos_sim": 0.0,
                "max_pos_sim_posOnly": 0.0,
                "top2_delta": 1.0,
                "top_pos_hit": {"cluster": "", "text": "", "score": 0.0},
                "matches_top1_text": "",
                "anti_proto_triggered": False,
                "anti_hits_by_lane": {},
                "matches_per_lane": {},
                "absolute_floor_triggered": False,
                "low_range_triggered": False,
                "reasoning": decision.reasoning,
                "time_ms": (time.perf_counter() - t0) * 1000,
            }

        l1_cfg = self._config.get("layer1_knn", self._config.get("layer1_embedding", {}))
        abs_floor = float(l1_cfg.get("absolute_floor", 0.25))
        if max_pos_similarity < abs_floor:
            decision = self._make_defer_decision(shadow_id, "ood_floor_check", t0, layer_trace)
            return {
                "status": "ok",
                "decision": decision,
                "scores": lane_scores,
                "max_pos_sim": float(max_pos_similarity),
                "max_pos_sim_posOnly": float(max_pos_similarity),
                "top2_delta": self._compute_top2_delta(lane_scores),
                "top_pos_hit": {
                    "cluster": str(l1_meta.get("top1_cluster", "")),
                    "text": str(l1_meta.get("top1_text", "")),
                    "score": float(l1_meta.get("top1_score", 0.0)),
                },
                "matches_top1_text": str(l1_meta.get("top1_text", "")),
                "anti_proto_triggered": bool(l1_meta.get("anti_proto_triggered", False)),
                "anti_hits_by_lane": dict(l1_meta.get("anti_hits_by_lane", {})),
                "matches_per_lane": dict(l1_meta.get("matches_per_lane", {})),
                "absolute_floor_triggered": True,
                "low_range_triggered": False,
                "reasoning": decision.reasoning,
                "time_ms": (time.perf_counter() - t0) * 1000,
            }

        normalized_scores = self._layer2_semantic_scoring(lane_scores, shadow_id)
        layer_trace.append("L2:semantic_scoring_shadow")
        norm_values = list(normalized_scores.values())
        if norm_values and max(norm_values) < 0.99 and max(norm_values) > 0:
            decision = self._make_defer_decision(shadow_id, "low_discrimination_range", t0, layer_trace)
            return {
                "status": "ok",
                "decision": decision,
                "scores": normalized_scores,
                "max_pos_sim": float(max_pos_similarity),
                "max_pos_sim_posOnly": float(max_pos_similarity),
                "top2_delta": self._compute_top2_delta(normalized_scores),
                "top_pos_hit": {
                    "cluster": str(l1_meta.get("top1_cluster", "")),
                    "text": str(l1_meta.get("top1_text", "")),
                    "score": float(l1_meta.get("top1_score", 0.0)),
                },
                "matches_top1_text": str(l1_meta.get("top1_text", "")),
                "anti_proto_triggered": bool(l1_meta.get("anti_proto_triggered", False)),
                "anti_hits_by_lane": dict(l1_meta.get("anti_hits_by_lane", {})),
                "matches_per_lane": dict(l1_meta.get("matches_per_lane", {})),
                "absolute_floor_triggered": False,
                "low_range_triggered": True,
                "reasoning": decision.reasoning,
                "time_ms": (time.perf_counter() - t0) * 1000,
            }

        cal_scores = _softmax(normalized_scores, self._softmax_temperature)
        cal_scores = self._calibrator.calibrate(cal_scores)
        cal_scores = self._calibrator.apply_masking(cal_scores)
        layer_trace.append("L4:decision_engine_shadow")
        decision = self.final_decision(
            decision_id=shadow_id,
            route_scores=cal_scores,
            raw_scores=lane_scores,
            evidence={},
            layer_trace=layer_trace,
            user_id=None,
            t0=t0,
            lane_uncertainty=lane_uncertainty,
        )

        return {
            "status": "ok",
            "decision": decision,
            "scores": cal_scores,
            "max_pos_sim": float(max_pos_similarity),
            "max_pos_sim_posOnly": float(max_pos_similarity),
            "top2_delta": self._compute_top2_delta(cal_scores),
            "top_pos_hit": {
                "cluster": str(l1_meta.get("top1_cluster", "")),
                "text": str(l1_meta.get("top1_text", "")),
                "score": float(l1_meta.get("top1_score", 0.0)),
            },
            "matches_top1_text": str(l1_meta.get("top1_text", "")),
            "anti_proto_triggered": bool(l1_meta.get("anti_proto_triggered", False)),
            "anti_hits_by_lane": dict(l1_meta.get("anti_hits_by_lane", {})),
            "matches_per_lane": dict(l1_meta.get("matches_per_lane", {})),
            "absolute_floor_triggered": False,
            "low_range_triggered": False,
            "reasoning": decision.reasoning,
            "time_ms": (time.perf_counter() - t0) * 1000,
        }

    async def _maybe_run_shadow_audit(
        self,
        *,
        decision_id: str,
        query_text: str,
        query_vec_routing: np.ndarray,
        primary_decision: PreRouteDecision,
        primary_scores: Dict[str, float],
        primary_top1_text: str,
        primary_max_pos_sim: float,
        primary_total_ms: float,
        primary_l1_meta: Optional[Dict[str, Any]] = None,
        primary_absolute_floor_triggered: bool = False,
        primary_low_range_triggered: bool = False,
        primary_anti_proto_triggered: bool = False,
    ) -> None:
        if not self._shadow_enabled:
            return
        if self._shadow_sample_rate <= 0.0:
            return
        if random.random() > self._shadow_sample_rate:
            return

        primary_l1_meta = primary_l1_meta or {}
        primary_top_pos_hit = {
            "cluster": str(primary_l1_meta.get("top1_cluster", "")),
            "text": primary_top1_text or str(primary_l1_meta.get("top1_text", "")),
            "score": round(float(primary_l1_meta.get("top1_score", 0.0)), 6),
        }
        primary_anti_hits_by_lane = dict(primary_l1_meta.get("anti_hits_by_lane", {}))
        primary_matches_per_lane = dict(primary_l1_meta.get("matches_per_lane", {}))

        shadow_timeout = max(self._shadow_timeout_ms / 1000.0, 0.001)
        shadow_payload: Dict[str, Any]
        shadow_t0 = time.perf_counter()
        try:
            shadow_result = await asyncio.wait_for(
                self._evaluate_shadow_without_l3(
                    decision_id=decision_id,
                    query_vec_routing=query_vec_routing,
                ),
                timeout=shadow_timeout,
            )
            shadow_decision = shadow_result.get("decision")
            if shadow_decision is None:
                shadow_payload = {
                    "status": "error",
                    "skipped": True,
                    "error": shadow_result.get("error", "shadow_decision_missing"),
                    "route": "SKIPPED",
                    "conf": 0.0,
                    "deferred": True,
                    "top2_delta": 0.0,
                    "max_pos_sim": 0.0,
                    "max_pos_sim_posOnly": 0.0,
                    "top_pos_hit": {"cluster": "", "text": "", "score": 0.0},
                    "anti_proto_triggered": False,
                    "anti_hits_by_lane": {},
                    "matches_per_lane": {},
                    "absolute_floor_triggered": False,
                    "low_range_triggered": False,
                    "reasoning": "shadow_decision_missing",
                    "matches_top1": {"text": ""},
                }
            else:
                top_pos_hit = shadow_result.get("top_pos_hit", {}) or {}
                shadow_payload = {
                    "status": shadow_result.get("status", "ok"),
                    "skipped": bool(shadow_result.get("skipped", False)),
                    "error": shadow_result.get("error"),
                    "route": shadow_decision.route,
                    "conf": round(float(shadow_decision.confidence), 6),
                    "deferred": bool(shadow_decision.deferred_to_llm_router),
                    "top2_delta": round(float(shadow_result.get("top2_delta", 0.0)), 6),
                    "max_pos_sim_posOnly": round(
                        float(shadow_result.get("max_pos_sim_posOnly", shadow_result.get("max_pos_sim", 0.0))),
                        6,
                    ),
                    "max_pos_sim": round(
                        float(shadow_result.get("max_pos_sim_posOnly", shadow_result.get("max_pos_sim", 0.0))),
                        6,
                    ),
                    "top_pos_hit": {
                        "cluster": str(top_pos_hit.get("cluster", "")),
                        "text": str(top_pos_hit.get("text", "")),
                        "score": round(float(top_pos_hit.get("score", 0.0)), 6),
                    },
                    "anti_proto_triggered": bool(shadow_result.get("anti_proto_triggered", False)),
                    "anti_hits_by_lane": dict(shadow_result.get("anti_hits_by_lane", {})),
                    "matches_per_lane": dict(shadow_result.get("matches_per_lane", {})),
                    "absolute_floor_triggered": bool(shadow_result.get("absolute_floor_triggered", False)),
                    "low_range_triggered": bool(shadow_result.get("low_range_triggered", False)),
                    "reasoning": str(shadow_result.get("reasoning", shadow_decision.reasoning)),
                    "matches_top1": {"text": shadow_result.get("matches_top1_text", "")},
                }
        except asyncio.TimeoutError:
            shadow_payload = {
                "status": "timeout",
                "skipped": True,
                "error": f"timeout>{self._shadow_timeout_ms}ms",
                "route": "SKIPPED",
                "conf": 0.0,
                "deferred": True,
                "top2_delta": 0.0,
                "max_pos_sim": 0.0,
                "max_pos_sim_posOnly": 0.0,
                "top_pos_hit": {"cluster": "", "text": "", "score": 0.0},
                "anti_proto_triggered": False,
                "anti_hits_by_lane": {},
                "matches_per_lane": {},
                "absolute_floor_triggered": False,
                "low_range_triggered": False,
                "reasoning": "timeout",
                "matches_top1": {"text": ""},
            }
        except Exception as exc:
            shadow_payload = {
                "status": "error",
                "skipped": True,
                "error": f"{type(exc).__name__}:{str(exc)[:120]}",
                "route": "SKIPPED",
                "conf": 0.0,
                "deferred": True,
                "top2_delta": 0.0,
                "max_pos_sim": 0.0,
                "max_pos_sim_posOnly": 0.0,
                "top_pos_hit": {"cluster": "", "text": "", "score": 0.0},
                "anti_proto_triggered": False,
                "anti_hits_by_lane": {},
                "matches_per_lane": {},
                "absolute_floor_triggered": False,
                "low_range_triggered": False,
                "reasoning": f"{type(exc).__name__}",
                "matches_top1": {"text": ""},
            }

        conf_delta = float(shadow_payload.get("conf", 0.0)) - float(primary_decision.confidence)
        route_changed = str(shadow_payload.get("route")) != str(primary_decision.route)
        defer_changed = bool(shadow_payload.get("deferred", True)) != bool(primary_decision.deferred_to_llm_router)
        shadow_top_pos_hit = shadow_payload.get("top_pos_hit", {}) or {}
        primary_top_text = str(primary_top_pos_hit.get("text", ""))
        shadow_top_text = str(shadow_top_pos_hit.get("text", ""))
        primary_top_cluster = str(primary_top_pos_hit.get("cluster", ""))
        shadow_top_cluster = str(shadow_top_pos_hit.get("cluster", ""))
        top_pos_cluster_changed = bool(
            primary_top_cluster and shadow_top_cluster and primary_top_cluster != shadow_top_cluster
        )
        same_text_conflict_detected = bool(
            primary_top_text
            and shadow_top_text
            and primary_top_text == shadow_top_text
            and top_pos_cluster_changed
        )

        payload = {
            "decision_id": decision_id,
            "query": query_text,
            "shadow_enabled": bool(self._shadow_enabled),
            "shadow_collection": self._shadow_collection,
            "primary": {
                "route": primary_decision.route,
                "conf": round(float(primary_decision.confidence), 6),
                "deferred": bool(primary_decision.deferred_to_llm_router),
                "top2_delta": round(self._compute_top2_delta(primary_scores), 6),
                "max_pos_sim_posOnly": round(float(primary_max_pos_sim), 6),
                "max_pos_sim": round(float(primary_max_pos_sim), 6),
                "top_pos_hit": primary_top_pos_hit,
                "anti_proto_triggered": bool(primary_anti_proto_triggered),
                "anti_hits_by_lane": primary_anti_hits_by_lane,
                "matches_per_lane": primary_matches_per_lane,
                "absolute_floor_triggered": bool(primary_absolute_floor_triggered),
                "low_range_triggered": bool(primary_low_range_triggered),
                "reasoning": str(primary_decision.reasoning),
                "matches_top1": {"text": primary_top1_text},
            },
            "shadow": shadow_payload,
            "diff": {
                "route_changed": route_changed,
                "conf_delta": round(conf_delta, 6),
                "defer_changed": defer_changed,
                "top_pos_cluster_changed": top_pos_cluster_changed,
                "same_text_conflict_detected": same_text_conflict_detected,
                "same_text_conflict": same_text_conflict_detected,
            },
            "timing_ms": {
                "primary_total_ms": round(float(primary_total_ms), 3),
                "shadow_extra_ms": round((time.perf_counter() - shadow_t0) * 1000, 3),
                "primary_total": round(float(primary_total_ms), 3),
                "shadow_extra": round((time.perf_counter() - shadow_t0) * 1000, 3),
            },
        }
        self._record_shadow_metrics(payload)
        self._log_shadow_decision(payload)

    async def initialize(self) -> Dict[str, Any]:
        """Load prototypes into RAM cache for kNN search. Falls back to centroids."""
        if self._initialized:
            return {"status": "already_initialized"}

        t0 = time.perf_counter()

        # Try kNN prototype cache first (Phase 2C)
        if self._qdrant_loader:
            try:
                cache = await self._load_prototypes_for_knn(collection_name=self._primary_collection)
                if cache:
                    self._prototype_cache_primary = cache
                    self._prototype_cache = cache
                    self._centroid_source = "knn_cache"
                    self._initialized = True
                    elapsed = (time.perf_counter() - t0) * 1000
                    meta = cache.get("_metadata", {})
                    logger.info(
                        f"[PREFILTER] kNN prototype cache loaded: "
                        f"positive={meta.get('positive_count', 0)}, "
                        f"negative={meta.get('negative_count', 0)}, "
                        f"dim={meta.get('embedding_dim', '?')}, "
                        f"{elapsed:.0f}ms"
                    )
                    return {
                        "status": "initialized", "source": "knn_cache",
                        "positive": meta.get("positive_count", 0),
                        "negative": meta.get("negative_count", 0),
                        "collection": meta.get("collection", self._primary_collection),
                        "elapsed_ms": round(elapsed),
                    }
            except Exception as e:
                logger.warning(f"[PREFILTER] kNN cache load failed, trying centroid fallback: {e}")

        # Fallback: compute centroids from hardcoded INTENT_EXEMPLARS
        await self._init_centroids_from_exemplars()
        elapsed = (time.perf_counter() - t0) * 1000
        dim = next(iter(self._centroids.values())).shape if self._centroids else ()
        logger.info(
            f"[PREFILTER] Centroids loaded from hardcoded INTENT_EXEMPLARS (fallback), "
            f"dim={dim}, {elapsed:.0f}ms"
        )
        return {"status": "initialized", "source": "hardcoded",
                "clusters": {k: 0 for k in self._centroids}, "elapsed_ms": round(elapsed)}

    async def _load_prototypes_for_knn(
        self, collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load all prototypes from a Qdrant collection into structured RAM cache."""
        from collections import Counter
        import hashlib

        target_collection = collection_name or self._primary_collection
        all_points = await self._scroll_all_prototypes(collection_name=target_collection)
        if not all_points:
            return None

        positives = [p for p in all_points
                     if p.payload.get("type", "positive") == "positive"
                     and p.payload.get("active", True)]
        negatives = [p for p in all_points
                     if p.payload.get("type") == "negative"
                     and p.payload.get("active", True)]

        if not positives:
            logger.error("[PREFILTER] No active positive prototypes found")
            return None

        mdim = self._matryoshka_dim

        def _extract_vectors(points):
            vecs = []
            for p in points:
                vec = np.array(p.vector, dtype=np.float32)
                if mdim > 0 and len(vec) > mdim:
                    vec = vec[:mdim]
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
                vecs.append(vec)
            return np.array(vecs) if vecs else np.empty((0, mdim or 128))

        cache = {
            "positive": {
                "vectors": _extract_vectors(positives),
                "lanes": [p.payload["cluster"] for p in positives],
                "weights": np.array([p.payload.get("weight", 1.0) for p in positives], dtype=np.float32),
                "texts": [p.payload.get("text", "") for p in positives],
                "ids": [str(p.id) for p in positives],
            },
            "negative": {
                "vectors": _extract_vectors(negatives),
                "lanes": [p.payload["cluster"] for p in negatives],
                "weights": np.array([p.payload.get("weight", 1.0) for p in negatives], dtype=np.float32) if negatives else np.array([], dtype=np.float32),
                "texts": [p.payload.get("text", "") for p in negatives],
                "ids": [str(p.id) for p in negatives],
            },
        }

        model_name = self._config.get("layer1_knn", {}).get(
            "model", "Snowflake/snowflake-arctic-embed-l-v2.0"
        )
        pos_by_lane = dict(Counter(cache["positive"]["lanes"]))
        neg_by_lane = dict(Counter(cache["negative"]["lanes"]))

        cache["_metadata"] = {
            "collection": target_collection,
            "embedding_model": model_name,
            "embedding_model_hash": hashlib.md5(model_name.encode()).hexdigest()[:8],
            "embedding_dim": mdim or (cache["positive"]["vectors"].shape[1] if len(positives) else 0),
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "positive_by_lane": pos_by_lane,
            "negative_by_lane": neg_by_lane,
            "last_reload": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        logger.info(
            f"[PREFILTER] Prototypes loaded ({target_collection}): "
            f"positive={len(positives)} {pos_by_lane}, "
            f"negative={len(negatives)} {neg_by_lane}, "
            f"dim={cache['positive']['vectors'].shape}"
        )
        return cache

    async def _scroll_all_prototypes(self, collection_name: Optional[str] = None):
        """Scroll all points from a Qdrant collection via read-only client."""
        if not self._qdrant_loader:
            return []

        import os
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            logger.error("[PREFILTER] qdrant_client not available")
            return []

        host = os.getenv("UBP_QDRANT__HOST", "ubp-qdrant")
        port = int(os.getenv("UBP_QDRANT__PORT", "6333"))
        client = QdrantClient(host=host, port=port)

        collection = collection_name or self._primary_collection

        if not client.collection_exists(collection):
            logger.warning(f"[PREFILTER] Collection '{collection}' not found")
            client.close()
            return []

        all_points = []
        offset = None
        while True:
            result = client.scroll(
                collection_name=collection, limit=256, offset=offset,
                with_payload=True, with_vectors=True,
            )
            points, next_offset = result
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        client.close()
        return all_points

    async def _init_centroids_from_exemplars(self) -> None:
        """Compute centroids from hardcoded INTENT_EXEMPLARS (legacy path)."""
        exemplars = self._config.get("exemplars", INTENT_EXEMPLARS)
        mdim = self._matryoshka_dim

        for intent_name, texts in exemplars.items():
            vectors = []
            for text in texts:
                try:
                    vec = await self._safe_embed(text)
                    arr = np.array(vec, dtype=np.float64)
                    if mdim > 0 and len(arr) > mdim:
                        arr = arr[:mdim]
                    vectors.append(arr)
                except Exception as e:
                    logger.warning(
                        f"[PREFILTER] Failed to embed exemplar '{text[:40]}': {e}"
                    )
            if vectors:
                centroid = np.mean(vectors, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-9:
                    centroid = centroid / norm
                self._centroids[intent_name] = centroid
            else:
                logger.error(f"[PREFILTER] No vectors for cluster '{intent_name}'")

        self._centroid_source = "hardcoded"
        self._initialized = True

    async def reload_centroids(self) -> Dict[str, Any]:
        """Reload prototypes from Qdrant into RAM cache (kNN mode).
        
        Backward-compatible name. Performs drift detection by lane count.
        """
        if self._reload_in_progress:
            logger.warning("[PREFILTER] Reload already in progress, skipping")
            return {"status": "skipped", "reason": "reload_in_progress"}

        self._reload_in_progress = True
        t0 = time.perf_counter()
        try:
            old_meta = (
                self._prototype_cache_primary.get("_metadata", {})
                if self._prototype_cache_primary else {}
            )
            old_pos = old_meta.get("positive_by_lane", {})
            old_neg = old_meta.get("negative_by_lane", {})

            new_cache = await self._load_prototypes_for_knn(collection_name=self._primary_collection)
            if not new_cache:
                return {"status": "error", "reason": "qdrant returned empty prototypes"}

            new_meta = new_cache.get("_metadata", {})
            new_pos = new_meta.get("positive_by_lane", {})
            new_neg = new_meta.get("negative_by_lane", {})

            # Drift detection by lane count
            drift_report = {}
            for lane in set(list(old_pos.keys()) + list(new_pos.keys())):
                old_n = old_pos.get(lane, 0)
                new_n = new_pos.get(lane, 0)
                delta_pct = abs(new_n - old_n) / max(old_n, 1) * 100
                drift_report[lane] = {"old": old_n, "new": new_n, "delta_pct": round(delta_pct)}
                if delta_pct > 30:
                    logger.warning(
                        f"[PREFILTER-DRIFT] Lane '{lane}' changed: "
                        f"{old_n} → {new_n} prototypes ({delta_pct:.0f}%)"
                    )

            # Atomic swap
            self._prototype_cache_primary = new_cache
            self._prototype_cache = new_cache
            self._centroid_source = "knn_cache"
            elapsed = (time.perf_counter() - t0) * 1000

            logger.info(
                f"[PREFILTER] Prototypes reloaded: "
                f"pos={new_meta.get('positive_count')}, neg={new_meta.get('negative_count')}, "
                f"{elapsed:.0f}ms"
            )
            return {
                "status": "reloaded", "source": "knn_cache",
                "positive": new_meta.get("positive_count"),
                "negative": new_meta.get("negative_count"),
                "drift": drift_report,
                "elapsed_ms": round(elapsed),
            }
        except Exception as e:
            logger.error(f"[PREFILTER] Prototype reload failed: {e}")
            return {"status": "error", "reason": str(e)}
        finally:
            self._reload_in_progress = False

    async def pre_route(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> PreRouteDecision:
        """
        Main entry point: classify query through 4-layer kNN pipeline.

        Phase 2C: kNN search on individual prototypes replaces centroid scoring.
        """
        decision_id = decision_id or str(uuid.uuid4())[:8]
        t0 = time.perf_counter()
        layer_trace: List[str] = []
        context = context or {}
        lane_uncertainty: Dict[str, float] = {}
        matches_per_lane = {"chat": 0, "web": 0, "rag": 0, "report": 0}
        score_range = 0.0
        max_pos_similarity = 0.0
        primary_top1_text = ""
        l1_meta: Dict[str, Any] = {}
        absolute_floor_triggered = False
        low_range_triggered = False
        anti_proto_triggered = False

        def _finalize_decision(
            decision: PreRouteDecision,
            route_scores_for_log: Optional[Dict[str, float]] = None,
        ) -> PreRouteDecision:
            self._log_structured_decision(
                query_id=decision_id,
                decision=decision,
                lane_uncertainty=lane_uncertainty,
                top_scores=route_scores_for_log or decision.scores,
                score_range=score_range,
                absolute_floor_triggered=absolute_floor_triggered,
                low_range_triggered=low_range_triggered,
                anti_proto_triggered=anti_proto_triggered,
                k_value=self._top_k,
                matches_per_lane=matches_per_lane,
            )
            return decision

        # Update metrics
        self._metrics["total_queries"] += 1

        if not self._initialized:
            return _finalize_decision(
                self._make_defer_decision(
                    decision_id, "engine_not_initialized", t0, layer_trace
                )
            )

        # ── L0: Continuation/deepdive — NO early exit (KB-AWARE) ──
        # All queries (including continuation/deepdive) pass through full
        # L1-L4 routing so the router can consider capabilities, memory
        # failure signals, and kb_relevant for optimal route selection.
        # NOTA: asyncio.to_thread() non è cancellabile da wait_for().
        # Se la funzione sottostante è blocking e lenta, R1 (500ms)
        # scatta prima del L3 timeout. Fix strutturale: usare client
        # async (qdrant_client.AsyncQdrantClient) in futuro.
        rewrite_type = context.get("rewrite_type") if context else None
        if rewrite_type in ("continuation", "deepdive"):
            layer_trace.append(f"L0:continuation_full_routing:{rewrite_type}")
            logger.info(
                f"[PREFILTER][{decision_id}] {rewrite_type} detected -> full L1-L4 routing (no early exit)"
            )

        # ── Embed query ──
        try:
            query_vec = np.array(await self._safe_embed(query), dtype=np.float32)
            mdim = self._matryoshka_dim
            query_vec_full = query_vec.copy()

            # Context fusion (if conversation context available)
            conv_context = context.get("context_text")
            if conv_context:
                context_vec = np.array(
                    await self._safe_embed(conv_context), dtype=np.float32
                )
                w = self._context_weight
                query_vec = (1.0 - w) * query_vec + w * context_vec

            # Matryoshka truncation for routing
            if mdim > 0 and len(query_vec) > mdim:
                query_vec_routing = query_vec[:mdim]
            else:
                query_vec_routing = query_vec

            # Normalize routing vector
            norm = np.linalg.norm(query_vec_routing)
            if norm > 1e-9:
                query_vec_routing = query_vec_routing / norm

        except Exception as e:
            logger.warning(f"[PREFILTER][{decision_id}] Embedding failed: {e}")
            return _finalize_decision(
                self._make_defer_decision(
                    decision_id, f"embed_error:{type(e).__name__}", t0, layer_trace
                )
            )

        # ── Layer 1: kNN Search or Centroid Fallback ──
        primary_cache = self._prototype_cache_primary or self._prototype_cache
        if primary_cache:
            lane_scores, lane_uncertainty, lane_evidence, max_pos_similarity, l1_meta = \
                self._layer1_knn_search(
                    query_vec_routing, decision_id, cache_override=primary_cache
                )
            anti_proto_triggered = bool(l1_meta.get("anti_proto_triggered", False))
            matches_per_lane = dict(l1_meta.get("matches_per_lane", matches_per_lane))
            primary_top1_text = str(l1_meta.get("top1_text", ""))
            layer_trace.append("L1:knn_search")
        else:
            # Legacy centroid fallback
            raw_scores = {}
            for name, centroid in self._centroids.items():
                raw_scores[name] = float(_cosine_sim(query_vec_routing, centroid))
            lane_scores = {self.LANE_TO_ROUTE.get(k, k): v for k, v in raw_scores.items()
                          if k in self.LANE_TO_ROUTE}
            lane_uncertainty = {}
            lane_evidence = {}
            max_pos_similarity = max(lane_scores.values()) if lane_scores else 0.0
            layer_trace.append("L1:centroid_fallback")

        if not lane_scores:
            return _finalize_decision(
                self._make_defer_decision(decision_id, "no_lane_scores", t0, layer_trace)
            )

        # Log RAW scores
        logger.info(f"[PREFILTER-RAW][{decision_id}] raw_scores={lane_scores}")

        # ── Absolute Floor Check (on max POSITIVE similarity, pre-aggregation) ──
        l1_cfg = self._config.get("layer1_knn", self._config.get("layer1_embedding", {}))
        abs_floor = l1_cfg.get("absolute_floor", 0.25)
        if max_pos_similarity < abs_floor:
            absolute_floor_triggered = True
            self._metrics["ood_floor_rejected"] += 1
            self._update_sim_bucket(max_pos_similarity)
            logger.info(
                f"[PREFILTER-L1][{decision_id}] FLOOR CHECK: "
                f"max_pos_sim={max_pos_similarity:.3f} < {abs_floor} — OOD, defer"
            )
            floor_decision = self._make_defer_decision(
                decision_id, "ood_floor_check", t0, layer_trace
            )
            await self._maybe_run_shadow_audit(
                decision_id=decision_id,
                query_text=query,
                query_vec_routing=query_vec_routing,
                primary_decision=floor_decision,
                primary_scores=lane_scores,
                primary_top1_text=primary_top1_text,
                primary_max_pos_sim=max_pos_similarity,
                primary_total_ms=(time.perf_counter() - t0) * 1000,
                primary_l1_meta=l1_meta,
                primary_absolute_floor_triggered=True,
                primary_low_range_triggered=False,
                primary_anti_proto_triggered=anti_proto_triggered,
            )
            return _finalize_decision(
                floor_decision,
                lane_scores,
            )

        self._update_sim_bucket(max_pos_similarity)

        # ── STEP 7: Auto-suggest prototype candidates ──
        if max_pos_similarity < 0.45 and max_pos_similarity >= abs_floor:
            sorted_lanes = sorted(lane_scores.items(), key=lambda x: x[1], reverse=True)
            top_lane = sorted_lanes[0][0] if sorted_lanes else "?"
            self._add_prototype_candidate(query, max_pos_similarity, top_lane, decision_id)

        # ── Feature gate: zero disabled lanes BEFORE L2 normalization ──
        # When rag_enabled=False or web_enabled=False, the lane should not
        # compete in normalization. Zeroing before L2 is cleaner than after.
        if context:
            if not context.get("rag_enabled", True):
                lane_scores["rag"] = 0.0
                logger.debug(f"[PREFILTER][{decision_id}] RAG lane zeroed (rag_enabled=False)")
            if not context.get("web_enabled", True):
                lane_scores["web_search"] = 0.0
                logger.debug(f"[PREFILTER][{decision_id}] WEB lane zeroed (web_enabled=False)")

        # ── Layer 2: Min-max normalization ──
        normalized_scores = self._layer2_semantic_scoring(lane_scores, decision_id)
        lane_uncertainty = dict(lane_uncertainty or {})
        norm_values = list(normalized_scores.values())
        if len(norm_values) >= 2:
            score_range = max(norm_values) - min(norm_values)
        layer_trace.append("L2:semantic_scoring")

        # ── FIX R2: Low-range bypass — force defer BEFORE softmax ──
        if norm_values and max(norm_values) < 0.99 and max(norm_values) > 0:
            low_range_triggered = True
            self._metrics["low_range_deferred"] += 1
            logger.info(
                f"[PREFILTER-L2][{decision_id}] LOW RANGE DETECTED: "
                f"raw_top={max(norm_values):.3f} → bypass softmax → defer"
            )
            low_range_decision = self._make_defer_decision(
                decision_id, "low_discrimination_range", t0, layer_trace
            )
            await self._maybe_run_shadow_audit(
                decision_id=decision_id,
                query_text=query,
                query_vec_routing=query_vec_routing,
                primary_decision=low_range_decision,
                primary_scores=normalized_scores,
                primary_top1_text=primary_top1_text,
                primary_max_pos_sim=max_pos_similarity,
                primary_total_ms=(time.perf_counter() - t0) * 1000,
                primary_l1_meta=l1_meta,
                primary_absolute_floor_triggered=False,
                primary_low_range_triggered=True,
                primary_anti_proto_triggered=anti_proto_triggered,
            )
            return _finalize_decision(
                low_range_decision,
                normalized_scores,
            )

        # ── Layer 2.5: External Signal Injection ──
        # Applies prefetch signals (rag_score, web_prefetch_score) from context
        # directly to normalized scores. NOT gated by L3 midband — these are
        # real evidence from KB/web, zero-cost (already computed in user_router).
        # WEBBIAS-FIX: equalized boost factors + freshness modulation.
        _freshness_signal = 0.0
        if context:
            _ext_rag = context.get("rag_rel_score")  # composite score (preferred)
            if _ext_rag is None:
                _ext_rag = context.get("rag_score")  # fallback to avg score
            _ext_web = context.get("web_prefetch_score")
            _l25_applied = False

            # Compute freshness signal BEFORE boosts
            _freshness_signal = self._compute_freshness_signal(
                query=query,
                web_prefetch_results=context.get("web_prefetch_results", []) or [],
                web_score=_ext_web if _ext_web is not None else 0.0,
                rag_score=_ext_rag if _ext_rag is not None else 0.0,
            )

            if _ext_rag is not None and "RAG" in normalized_scores:
                _rag_before = normalized_scores["RAG"]
                if _ext_rag >= 0.5:
                    _rag_boost = _ext_rag * self._l25_rag_boost_factor
                    # Freshness dampening: reduce RAG boost on time-sensitive queries
                    if (
                        _freshness_signal > self._l25_freshness_threshold
                        and _ext_rag < self._l25_rag_freshness_exempt_above
                    ):
                        _rag_boost *= (1.0 - _freshness_signal * self._l25_freshness_rag_dampening)
                    normalized_scores["RAG"] = min(normalized_scores["RAG"] + _rag_boost, 1.0)
                elif _ext_rag < 0.15:
                    normalized_scores["RAG"] = max(
                        normalized_scores["RAG"] + self._l25_rag_penalty, 0.0
                    )
                if normalized_scores["RAG"] != _rag_before:
                    _l25_applied = True
                    logger.debug(
                        f"[PREFILTER-L2.5][{decision_id}] RAG: "
                        f"{_rag_before:.4f} → {normalized_scores['RAG']:.4f} "
                        f"(rag_score={_ext_rag:.2f})"
                    )

            if _ext_web is not None and "WEB" in normalized_scores:
                _web_before = normalized_scores["WEB"]
                if _ext_web >= 0.5:
                    _web_boost = _ext_web * self._l25_web_boost_factor
                    # Freshness amplification: boost WEB on time-sensitive queries
                    if _freshness_signal > self._l25_freshness_threshold:
                        _web_boost *= (1.0 + _freshness_signal * self._l25_freshness_web_amplify)
                        _web_boost = min(_web_boost, self._l25_web_boost_cap)
                    normalized_scores["WEB"] = min(normalized_scores["WEB"] + _web_boost, 1.0)
                elif _ext_web < 0.15:
                    normalized_scores["WEB"] = max(
                        normalized_scores["WEB"] + self._l25_web_penalty, 0.0
                    )
                if normalized_scores["WEB"] != _web_before:
                    _l25_applied = True
                    logger.debug(
                        f"[PREFILTER-L2.5][{decision_id}] WEB: "
                        f"{_web_before:.4f} → {normalized_scores['WEB']:.4f} "
                        f"(web_score={_ext_web:.2f})"
                    )

            if _l25_applied:
                layer_trace.append("L2.5:external_signals")
                logger.info(
                    "[PREFILTER-L2.5][%s] adjusted_scores=%s "
                    "(rag_score=%s, web_score=%s, freshness=%.3f)",
                    decision_id, normalized_scores, _ext_rag, _ext_web, _freshness_signal,
                )

        # ── Snapshot pre-L3 scores for R1 timeout recovery ──
        # If L3 (async Qdrant) causes timeout, these scores are used directly.
        pre_l3_cal = _softmax(dict(normalized_scores), self._softmax_temperature)
        pre_l3_cal = self._calibrator.calibrate(pre_l3_cal)
        pre_l3_cal = self._calibrator.apply_masking(pre_l3_cal)
        self._last_pre_l3_scores = pre_l3_cal

        # ── L3+L4 with R1 timeout recovery ──
        # If the adapter's asyncio.wait_for raises TimeoutError during L3
        # (Qdrant HTTP), we catch it here and use pre_l3 scores to produce
        # a valid PreRouteDecision instead of propagating the exception.
        try:
            # ── Layer 3: Evidence Enrichment ──
            evidence, adjusted_scores = await self._layer3_evidence_enrichment(
                query=query,
                normalized_scores=normalized_scores,
                query_vec_full=query_vec_full,
                query_vec_routing=query_vec_routing,
                context=context,
                decision_id=decision_id,
                layer_trace=layer_trace,
            )

            # ── Single Softmax: applied ONCE on adjusted scores ──
            cal_scores = _softmax(adjusted_scores, self._softmax_temperature)
            cal_scores = self._calibrator.calibrate(cal_scores)
            cal_scores = self._calibrator.apply_masking(cal_scores)
            logger.info(
                f"[PREFILTER-CAL][{decision_id}] calibrated_scores={cal_scores} "
                f"uncertainty={lane_uncertainty}"
            )

            # ── Layer 4: Decision Engine ──
            decision = await self.final_decision(
                decision_id=decision_id,
                route_scores=cal_scores,
                raw_scores=lane_scores,
                evidence=evidence or {},
                layer_trace=layer_trace,
                user_id=user_id,
                t0=t0,
                lane_uncertainty=lane_uncertainty,
                query_text=query,
            )

            await self._maybe_run_shadow_audit(
                decision_id=decision_id,
                query_text=query,
                query_vec_routing=query_vec_routing,
                primary_decision=decision,
                primary_scores=cal_scores,
                primary_top1_text=primary_top1_text,
                primary_max_pos_sim=max_pos_similarity,
                primary_total_ms=(time.perf_counter() - t0) * 1000,
                primary_l1_meta=l1_meta,
                primary_absolute_floor_triggered=absolute_floor_triggered,
                primary_low_range_triggered=low_range_triggered,
                primary_anti_proto_triggered=anti_proto_triggered,
            )
        except (asyncio.TimeoutError, TimeoutError):
            # R1 timeout recovery: use pre-L3 calibrated scores
            # L1->L2->L2.5->softmax+calibrate completed before L3 (Qdrant HTTP).
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if pre_l3_cal and max(pre_l3_cal.values()) > 0.35:
                best_route = max(pre_l3_cal, key=pre_l3_cal.get)
                best_conf = pre_l3_cal[best_route]
                decision = PreRouteDecision(
                    decision_id=decision_id,
                    route=best_route,
                    confidence=best_conf,
                    raw_confidence=best_conf,
                    reasoning=(
                        f"R1_timeout_recovery:L2.5"
                        f"|{best_route}={best_conf:.3f}"
                    ),
                    layer_trace=layer_trace + ["R1:timeout", "L2.5:recovery"],
                    severity_level="medium",
                    time_ms=elapsed_ms,
                    deferred_to_llm_router=False,
                    scores=pre_l3_cal,
                )
                logger.warning(
                    "[PREFILTER] R1 timeout RECOVERED via L2.5 scores: "
                    "%s=%.3f (skipped L3+L4) | scores=%s",
                    best_route, best_conf, pre_l3_cal,
                )
            else:
                decision = PreRouteDecision(
                    decision_id=decision_id,
                    route="LLM_ROUTER",
                    confidence=0.0,
                    raw_confidence=0.0,
                    reasoning=f"R1_timeout_recovery:low_scores",
                    layer_trace=layer_trace + ["R1:timeout"],
                    severity_level="low",
                    time_ms=elapsed_ms,
                    deferred_to_llm_router=True,
                )
                logger.warning(
                    "[PREFILTER] R1 timeout + low scores -> LLM_ROUTER | "
                    "pre_l3=%s", pre_l3_cal,
                )
        except Exception as _l3l4_exc:
            # Generic L3/L4 failure recovery
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if pre_l3_cal and max(pre_l3_cal.values()) > 0.35:
                best_route = max(pre_l3_cal, key=pre_l3_cal.get)
                best_conf = pre_l3_cal[best_route]
                decision = PreRouteDecision(
                    decision_id=decision_id,
                    route=best_route,
                    confidence=best_conf,
                    raw_confidence=best_conf,
                    reasoning=(
                        f"R1_exception_recovery:L2.5"
                        f"|{type(_l3l4_exc).__name__}"
                    ),
                    layer_trace=layer_trace + ["R1:exception", "L2.5:recovery"],
                    severity_level="medium",
                    time_ms=elapsed_ms,
                    deferred_to_llm_router=False,
                    scores=pre_l3_cal,
                )
                logger.warning(
                    "[PREFILTER] L3/L4 exception RECOVERED via L2.5 scores: "
                    "%s=%.3f | error=%s",
                    best_route, best_conf, _l3l4_exc,
                )
            else:
                decision = self._make_defer_decision(
                    decision_id, f"l3l4_error:{type(_l3l4_exc).__name__}",
                    t0, layer_trace + ["R1:exception"],
                )
                logger.warning(
                    "[PREFILTER] L3/L4 exception -> LLM_ROUTER: %s",
                    _l3l4_exc,
                )

        # Update metrics
        if decision.deferred_to_llm_router:
            self._metrics["deferred_to_llm"] += 1
        elif decision.route == "DYNAMIC_INTERACTION":
            self._metrics["dynamic_interaction"] += 1
        else:
            self._metrics["direct_routes"] += 1
            self._metrics["route_distribution"][decision.route] = \
                self._metrics["route_distribution"].get(decision.route, 0) + 1
        self._update_confidence_bucket(decision.confidence)

        # Log aggregated metrics every 100 queries
        if self._metrics["total_queries"] % 100 == 0:
            self._log_metrics_summary()

        # Log DECISION
        logger.info(
            f"[PREFILTER][{decision_id}] route={decision.route} "
            f"confidence={decision.confidence:.3f} "
            f"raw_confidence={decision.raw_confidence:.3f} "
            f"layer_trace={decision.layer_trace}"
        )

        # Log MIDBAND monitoring (0.55-0.80)
        if self._defer_threshold <= decision.confidence < self._high_confidence_threshold:
            sorted_scores = sorted(
                cal_scores.items(), key=lambda x: x[1], reverse=True
            )
            top2_delta = (
                sorted_scores[0][1] - sorted_scores[1][1]
                if len(sorted_scores) > 1 else 0.0
            )
            logger.info(
                f"[PREFILTER-MIDBAND][{decision_id}] route={decision.route} "
                f"confidence={decision.confidence:.3f} "
                f"top2_delta={top2_delta:.3f} "
                f"accepted_without_llm_router=True"
            )

        # Attach freshness_signal to decision (computed in L2.5)
        decision.freshness_signal = _freshness_signal

        return _finalize_decision(decision, cal_scores)

    # ------------------------------------------------------------------
    # Layer implementations
    # ------------------------------------------------------------------

    def compute_knn_matches(
        self,
        query_vec_128d: np.ndarray,
        prototype_vectors: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarities (dot-product on normalized vectors)."""
        if len(prototype_vectors) == 0:
            return np.array([], dtype=np.float32)
        return prototype_vectors @ query_vec_128d

    def aggregate_by_lane(
        self,
        similarities: np.ndarray,
        lanes: List[str],
        weights: np.ndarray,
        texts: List[str],
        max_per_lane: int,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[Tuple[str, float]]], Dict[str, int]]:
        """Group prototype matches by lane and keep top-N per lane."""
        lane_indices: Dict[str, List[int]] = {}
        for idx in range(len(similarities)):
            lane = lanes[idx]
            if lane not in lane_indices:
                lane_indices[lane] = []
            lane_indices[lane].append(idx)

        lane_scores: Dict[str, List[float]] = {}
        lane_evidence: Dict[str, List[Tuple[str, float]]] = {}
        matches_per_lane = {"chat": 0, "web": 0, "rag": 0, "report": 0}

        for lane, indices in lane_indices.items():
            sorted_indices = sorted(indices, key=lambda i: similarities[i], reverse=True)
            top_for_lane = sorted_indices[:max_per_lane]

            lane_scores[lane] = []
            lane_evidence[lane] = []
            for idx in top_for_lane:
                sim = float(similarities[idx])
                weight = float(weights[idx])
                text = texts[idx]
                lane_scores[lane].append(sim * weight)
                lane_evidence[lane].append((text, sim))

            lane_key = "web" if lane == "web_search" else lane
            if lane_key in matches_per_lane:
                matches_per_lane[lane_key] = len(top_for_lane)

        return lane_scores, lane_evidence, matches_per_lane

    def apply_anti_prototype_penalty(
        self,
        query_vec_128d: np.ndarray,
        cache: Dict[str, Any],
        lane_pos_scores: Dict[str, List[float]],
        max_pos_similarity: float,
        anti_proto_min_signal: float,
        anti_relative_threshold: float,
        decision_id: str,
    ) -> Tuple[Dict[str, List[float]], bool]:
        """Compute anti-prototype lane penalties from negative prototype hits."""
        lane_neg_scores: Dict[str, List[float]] = {}
        neg_vectors = cache["negative"]["vectors"]
        if len(neg_vectors) == 0:
            return lane_neg_scores, False

        global_threshold = max_pos_similarity * 0.70
        if max_pos_similarity < anti_proto_min_signal:
            logger.debug(
                f"[PREFILTER-L1][{decision_id}] anti-protos SKIPPED: "
                f"max_pos_sim={max_pos_similarity:.3f} < {anti_proto_min_signal}"
            )
            return lane_neg_scores, False

        neg_lanes = cache["negative"]["lanes"]
        neg_weights = cache["negative"]["weights"]
        neg_similarities = self.compute_knn_matches(query_vec_128d, neg_vectors)

        for idx in range(len(neg_similarities)):
            sim = float(neg_similarities[idx])
            lane = neg_lanes[idx]
            weight = float(neg_weights[idx])

            if sim < global_threshold:
                continue

            lane_pos = lane_pos_scores.get(lane, [])
            lane_pos_top = max(lane_pos) if lane_pos else 0.0
            if lane_pos_top > 0 and sim < lane_pos_top * anti_relative_threshold:
                continue

            if lane not in lane_neg_scores:
                lane_neg_scores[lane] = []
            lane_neg_scores[lane].append(sim * weight)

        if lane_neg_scores:
            logger.info(
                f"[PREFILTER-L1][{decision_id}] anti-protos applied: "
                f"{{{', '.join(f'{k}: {len(v)} hits' for k, v in lane_neg_scores.items())}}}"
            )
            return lane_neg_scores, True

        return lane_neg_scores, False

    def compute_lane_scores(
        self,
        pos_score: float,
        neg_score: float,
        penalty_min_pos: float,
        route: str,
        decision_id: str,
    ) -> float:
        """Compute lane score with linear and continuous anti-prototype penalty."""
        final_score = max(0.0, pos_score - self._anti_lambda * neg_score)

        if neg_score > 0 and pos_score > penalty_min_pos:
            ratio = neg_score / pos_score
            if ratio > 0.85:
                penalty_factor = 1.0 - ((ratio - 0.85) / 0.15) * 0.5
                penalty_factor = max(0.5, penalty_factor)
                final_score *= penalty_factor
                logger.info(
                    f"[PREFILTER-L1][{decision_id}] CONTINUOUS penalty on {route}: "
                    f"neg/pos ratio={ratio:.3f} → factor={penalty_factor:.3f}"
                )

        return final_score

    def compute_uncertainty(self, pos_scores: List[float]) -> float:
        """Compute lane uncertainty as 1 - max positive match."""
        if pos_scores:
            max_score = max(pos_scores)
            return float(max(0.0, min(1.0, 1.0 - max_score)))
        return 1.0

    def _layer1_knn_search(
        self,
        query_vec_128d: np.ndarray,
        decision_id: str,
        cache_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, list], float, Dict[str, Any]]:
        """Layer 1: kNN search on individual prototypes in RAM cache.

        Replaces centroid-based scoring. Searches all positive prototypes,
        groups by lane, applies anti-prototype penalties.

        Returns: (lane_scores, lane_uncertainty, lane_evidence, max_pos_similarity, l1_meta)
        """
        cache = cache_override or self._prototype_cache_primary or self._prototype_cache
        if cache is None:
            return {}, {}, {}, 0.0, {"anti_proto_triggered": False, "matches_per_lane": {}}

        l1_cfg = self._config.get("layer1_knn", self._config.get("layer1_embedding", {}))
        MAX_PER_LANE = l1_cfg.get("max_per_lane", 3)
        ANTI_PROTO_MIN_SIGNAL = l1_cfg.get("anti_proto_min_signal", 0.60)
        ANTI_REL_THRESHOLD = l1_cfg.get("anti_relative_threshold", 0.85)
        PENALTY_MIN_POS = l1_cfg.get("penalty_min_pos_score", 0.60)

        # ═══ POSITIVE kNN ═══
        pos_vectors = cache["positive"]["vectors"]
        pos_lanes = cache["positive"]["lanes"]
        pos_weights = cache["positive"]["weights"]
        pos_texts = cache["positive"]["texts"]

        # Cosine similarity with ALL positive prototypes
        pos_similarities = self.compute_knn_matches(query_vec_128d, pos_vectors)

        lane_pos_scores, lane_evidence, matches_per_lane = self.aggregate_by_lane(
            similarities=pos_similarities,
            lanes=pos_lanes,
            weights=pos_weights,
            texts=pos_texts,
            max_per_lane=MAX_PER_LANE,
        )

        # Max positive similarity (pre-aggregation, for floor check)
        max_pos_similarity = float(np.max(pos_similarities)) if len(pos_similarities) > 0 else 0.0

        # ═══ NEGATIVE kNN (anti-prototypes) ═══
        lane_neg_scores, anti_proto_triggered = self.apply_anti_prototype_penalty(
            query_vec_128d=query_vec_128d,
            cache=cache,
            lane_pos_scores=lane_pos_scores,
            max_pos_similarity=max_pos_similarity,
            anti_proto_min_signal=ANTI_PROTO_MIN_SIGNAL,
            anti_relative_threshold=ANTI_REL_THRESHOLD,
            decision_id=decision_id,
        )

        # ═══ AGGREGATION: pos_score - λ * neg_score per lane ═══
        all_lanes = set(list(lane_pos_scores.keys()) + list(lane_neg_scores.keys()))
        lane_scores: Dict[str, float] = {}
        lane_uncertainty: Dict[str, float] = {}

        for lane in all_lanes:
            route = self.LANE_TO_ROUTE.get(lane)
            if not route:
                continue

            pos_list = lane_pos_scores.get(lane, [])

            if len(pos_list) < self._min_lane_hits:
                lane_scores[route] = 0.0
                lane_uncertainty[route] = 1.0
                continue

            pos_score = sum(pos_list) / len(pos_list)

            neg_list = lane_neg_scores.get(lane, [])
            neg_score = sum(neg_list) / len(neg_list) if neg_list else 0.0

            final_score = self.compute_lane_scores(
                pos_score=pos_score,
                neg_score=neg_score,
                penalty_min_pos=PENALTY_MIN_POS,
                route=route,
                decision_id=decision_id,
            )
            uncertainty = self.compute_uncertainty(pos_list)

            lane_scores[route] = final_score
            lane_uncertainty[route] = float(uncertainty)

        # ═══ LOGGING ═══
        all_top3 = []
        all_top_full = []
        for lane, ev in lane_evidence.items():
            for text, score in ev[:2]:
                route = self.LANE_TO_ROUTE.get(lane, lane)
                all_top3.append((text[:40], score, route, lane))
                all_top_full.append((text, score, route, lane))
        all_top3.sort(key=lambda x: x[1], reverse=True)
        all_top_full.sort(key=lambda x: x[1], reverse=True)
        top1_text = all_top_full[0][0] if all_top_full else ""
        top1_route = all_top_full[0][2] if all_top_full else ""
        top1_cluster = all_top_full[0][3] if all_top_full else ""
        top1_score = float(all_top_full[0][1]) if all_top_full else 0.0
        anti_hits_by_lane = {lane: len(scores) for lane, scores in lane_neg_scores.items()}

        logger.info(
            f"[PREFILTER-L1][{decision_id}] kNN top-3: "
            f"{[f'{t}({s:.3f},{l})' for t, s, l, _lane in all_top3[:3]]}"
        )
        logger.info(
            f"[PREFILTER-L1][{decision_id}] lane_scores="
            f"{{{', '.join(f'{k}: {v:.3f}' for k, v in sorted(lane_scores.items(), key=lambda x: x[1], reverse=True))}}}"
            f" neg_applied={list(lane_neg_scores.keys())}"
        )

        # Convert lane_evidence keys to route names
        route_evidence = {self.LANE_TO_ROUTE.get(k, k): v for k, v in lane_evidence.items()}

        return lane_scores, lane_uncertainty, route_evidence, max_pos_similarity, {
            "anti_proto_triggered": anti_proto_triggered,
            "anti_hits_by_lane": anti_hits_by_lane,
            "matches_per_lane": matches_per_lane,
            "top1_text": top1_text,
            "top1_route": top1_route,
            "top1_cluster": top1_cluster,
            "top1_score": top1_score,
        }

    def _layer2_semantic_scoring(
        self,
        raw_scores: Dict[str, float],
        decision_id: str,
    ) -> Dict[str, float]:
        """Layer 2: Min-max normalization of route scores -> normalized [0,1].

        Accepts route-keyed scores (from kNN) or cluster-keyed scores (legacy).
        Maps cluster names to route names if needed.
        """
        # Map cluster scores to route scores (if needed)
        route_scores: Dict[str, float] = {}
        for key, score in raw_scores.items():
            route = CLUSTER_TO_ROUTE.get(key, key)  # pass-through if already a route name
            if route in VALID_ROUTES or key in VALID_ROUTES:
                route_scores[route] = score

        if not route_scores:
            return {}

        # Min-max normalization: rescale to [0, 1] range
        values = list(route_scores.values())
        min_score = min(values)
        max_score = max(values)
        score_range = max_score - min_score

        # Minimum discrimination range: if all scores are within this range,
        # the differences are noise — return raw (softmax will produce near-uniform).
        l2_cfg = self._config.get("layer2_semantic_scoring", {})
        min_disc_range = l2_cfg.get("min_discrimination_range", 0.05)
        if score_range < min_disc_range:
            import math
            values_list = list(route_scores.values())
            total = sum(values_list)
            if total > 0:
                probs = [v / total for v in values_list]
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                max_entropy = math.log2(len(probs))
                norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
            else:
                norm_entropy = 1.0

            logger.info(
                f"[PREFILTER-L2][{decision_id}] range={score_range:.4f} "
                f"< {min_disc_range} — scores too close. "
                f"{'LOW_SEPARABILITY ' if norm_entropy > 0.85 else ''}"
                f"entropy={norm_entropy:.3f}"
            )
            # Return raw scores — softmax in pre_route will produce near-uniform → defer
            return route_scores

        normalized = {k: (v - min_score) / score_range for k, v in route_scores.items()}

        logger.info(
            f"[PREFILTER-L2][{decision_id}] range={score_range:.4f} "
            f"normalized={{{', '.join(f'{k}: {v:.3f}' for k, v in normalized.items())}}}"
        )

        return normalized

    async def _layer3_evidence_enrichment(
        self,
        query: str,
        normalized_scores: Dict[str, float],
        query_vec_full: np.ndarray,
        query_vec_routing: np.ndarray,
        context: Dict[str, Any],
        decision_id: str,
        layer_trace: List[str],
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Layer 3: Evidence enrichment for midband decisions.

        Operates on NORMALIZED scores [0,1] from Layer 2 (pre-softmax).
        Only activates when top-2 delta is below threshold (scores too close).
        Softmax is applied ONCE in pre_route() after this method returns.

        Returns (evidence_dict, adjusted_normalized_scores).
        """
        l3_cfg = self._config.get("layer3_evidence_enrichment", {})
        if not l3_cfg.get("enabled", False):
            layer_trace.append("L3:disabled")
            return {}, normalized_scores

        # Midband trigger: check delta between top-1 and top-2 on normalized scores
        ranked = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        top_val = ranked[0][1] if ranked else 0.0
        top2_delta = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 1.0

        # For normalized [0,1] scores: use delta_threshold (top is often 1.0 after min-max)
        delta_thresh = l3_cfg.get("delta_threshold", 0.30)

        if top2_delta >= delta_thresh:
            layer_trace.append("L3:skipped")
            return {}, normalized_scores

        # Collect evidence (with global timeout)
        timeout_ms = l3_cfg.get("timeout_ms", 20)
        sources_cfg = l3_cfg.get("sources", {})
        evidences: Dict[str, Dict[str, float]] = {}

        try:
            import asyncio
            # Source 1: RAG Preview — query real KB collections (1024d)
            # Skip if rag_enabled=False (lane already zeroed, no point querying)
            _rag_enabled = context.get("rag_enabled", True) if context else True
            if _rag_enabled and sources_cfg.get("rag_preview", {}).get("enabled", True):
                try:
                    rag_ev = await asyncio.wait_for(
                        self._rag_preview_evidence(
                            query_vec_full, query_vec_routing,
                            context, sources_cfg.get("rag_preview", {}), decision_id,
                        ),
                        timeout=timeout_ms / 1000.0,
                    )
                    if rag_ev:
                        evidences["rag_preview"] = rag_ev
                except asyncio.TimeoutError:
                    pass

            # Source 2: Web Signal — similarity with WEB prototypes
            # Skip if web_enabled=False (lane already zeroed)
            _web_enabled = context.get("web_enabled", True) if context else True
            if _web_enabled and sources_cfg.get("web_signal", {}).get("enabled", True):
                try:
                    web_ev = await self._web_signal_evidence(query_vec_routing, decision_id)
                    if web_ev:
                        evidences["web_signal"] = web_ev
                except Exception:
                    pass

            # Source 3: Context Memory (sync, ~1ms)
            if sources_cfg.get("context_memory", {}).get("enabled", True):
                ctx_ev = self._context_memory_evidence(context)
                if ctx_ev:
                    evidences["context_memory"] = ctx_ev

        except Exception as e:
            logger.warning(f"[PREFILTER-L3][{decision_id}] Evidence collection failed: {e}")
            layer_trace.append("L3:error")
            return {}, normalized_scores

        if not evidences:
            layer_trace.append("L3:no_evidence")
            return {}, normalized_scores

        # Aggregate evidence into adjusted scores (NO renormalization)
        adjusted = self._aggregate_evidence(normalized_scores, evidences, decision_id)
        layer_trace.append("L3:evidence_enrichment")
        return {"sources": evidences}, adjusted

    async def _rag_preview_evidence(
        self, query_vec_full: np.ndarray, query_vec_routing: np.ndarray,
        context: Dict[str, Any], cfg: Dict, decision_id: str,
    ) -> Dict[str, float]:
        """RAG Preview: query user KB collections for relevance signal.

        Uses 1024d vector against real KB collections (NOT routing_prototypes).
        If relevant documents found → boost RAG. If none → weak WEB boost.
        """
        if not self._qdrant_search_fn:
            return {}

        boost_strong = cfg.get("boost_strong", 0.08)
        boost_weak = cfg.get("boost_weak", 0.04)
        score_threshold = cfg.get("score_threshold", 0.55)

        # Resolve user collections from context
        collections = context.get("collections")
        if not collections:
            collections = ["ubp_system_docs"]
        elif isinstance(collections, str):
            collections = [collections]

        try:
            results = await self._qdrant_search_fn(
                collections=collections,
                vector=query_vec_full.tolist(),
                limit=3,
                score_threshold=score_threshold,
            )

            if results and len(results) >= 2:
                avg_score = sum(r["score"] for r in results) / len(results)
                if avg_score > 0.70:
                    logger.info(
                        f"[PREFILTER-L3][{decision_id}] RAG Preview: "
                        f"{len(results)} hits, avg={avg_score:.3f} → strong RAG boost"
                    )
                    return {"RAG": boost_strong, "WEB": -0.02}
                elif avg_score > score_threshold:
                    logger.info(
                        f"[PREFILTER-L3][{decision_id}] RAG Preview: "
                        f"{len(results)} hits, avg={avg_score:.3f} → weak RAG boost"
                    )
                    return {"RAG": boost_weak}

            if not results:
                logger.info(
                    f"[PREFILTER-L3][{decision_id}] RAG Preview: "
                    f"0 hits → weak WEB boost"
                )
                return {"WEB": 0.03, "RAG": -0.02}

            return {}

        except Exception as e:
            logger.debug(f"[PREFILTER-L3][{decision_id}] RAG Preview error: {e}")
            return {}

    async def _web_signal_evidence(
        self, query_vec_routing: np.ndarray, decision_id: str,
    ) -> Dict[str, float]:
        """Web Signal: similarity with individual WEB prototypes in primary prefilter collection.

        Different from Layer 2 which uses the centroid (mean of 29 prototypes).
        Here we compare against individual prototypes — captures specific patterns
        that the centroid average loses.

        Uses 128d vector for consistency with routing prefilter collection.
        """
        if not self._qdrant_search_fn:
            return {}

        try:
            import asyncio
            results = await asyncio.wait_for(
                self._qdrant_search_fn(
                    collections=[self._primary_collection],
                    vector=query_vec_routing.tolist(),
                    limit=5,
                    score_threshold=0.60,
                    filter_payload={"cluster": "web_search", "active": True},
                ),
                timeout=0.005,  # 5ms — local Qdrant query
            )

            if not results:
                return {}

            avg_sim = sum(r["score"] for r in results) / len(results)
            max_sim = results[0]["score"]

            if max_sim > 0.85:
                logger.info(
                    f"[PREFILTER-L3][{decision_id}] Web Signal: "
                    f"max_sim={max_sim:.3f}, avg={avg_sim:.3f} → strong WEB boost"
                )
                return {"WEB": 0.10, "RAG": -0.03}
            elif avg_sim > 0.70:
                logger.info(
                    f"[PREFILTER-L3][{decision_id}] Web Signal: "
                    f"avg={avg_sim:.3f} → weak WEB boost"
                )
                return {"WEB": 0.05}

            return {}

        except asyncio.TimeoutError:
            return {}
        except Exception as e:
            logger.debug(f"[PREFILTER-L3][{decision_id}] Web Signal error: {e}")
            return {}

    def _context_memory_evidence(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Context Memory: bias from recent session routes."""
        recent_routes = context.get("recent_routes", [])
        if not recent_routes or len(recent_routes) < 2:
            return {}

        from collections import Counter
        route_counts = Counter(recent_routes[-3:])
        dominant, count = route_counts.most_common(1)[0]
        if count >= 2:
            return {dominant: 0.06}

        return {}

    def _aggregate_evidence(
        self,
        base_scores: Dict[str, float],
        evidences: Dict[str, Dict[str, float]],
        decision_id: str,
    ) -> Dict[str, float]:
        """Combine boosts/penalties from evidence sources onto normalized scores.

        Applies deltas directly. Clamps to [0, max] — no score below zero.
        Does NOT renormalize — softmax in pre_route() handles distribution.
        """
        adjusted = dict(base_scores)
        total_adjustments: Dict[str, float] = {}

        for source_name, adjustments in evidences.items():
            for route, delta in adjustments.items():
                if route in adjusted:
                    adjusted[route] = max(0.0, adjusted[route] + delta)
                    total_adjustments[route] = total_adjustments.get(route, 0.0) + delta

        # NO renormalization — softmax in pre_route() manages the distribution

        if total_adjustments:
            logger.info(
                f"[PREFILTER-L3][{decision_id}] "
                f"evidence_deltas={total_adjustments} "
                f"adjusted={{{', '.join(f'{k}: {v:.3f}' for k, v in adjusted.items())}}}"
            )

        return adjusted

    async def final_decision(
        self,
        decision_id: str,
        route_scores: Dict[str, float],
        raw_scores: Dict[str, float],
        evidence: Dict[str, Any],
        layer_trace: List[str],
        user_id: Optional[str],
        t0: float,
        lane_uncertainty: Optional[Dict[str, float]] = None,
        query_text: str = "",
    ) -> PreRouteDecision:
        """Layer 4: Final decision based on scores and policy.

        Phase 2C: receives lane_uncertainty from kNN for uncertainty discount.
        Phase 3: R2 vLLM disambiguation for uncertain queries.
        """
        layer_trace.append("L4:decision_engine")
        elapsed = (time.perf_counter() - t0) * 1000
        lane_uncertainty = lane_uncertainty or {}

        if not route_scores:
            return self._make_defer_decision(
                decision_id, "no_route_scores", t0, layer_trace
            )

        # Sort by softmax probability
        ranked = sorted(route_scores.items(), key=lambda x: x[1], reverse=True)
        top_route, top_confidence = ranked[0]

        # Map raw scores to route-keyed
        raw_route_scores: Dict[str, float] = {}
        for key, score in raw_scores.items():
            route = CLUSTER_TO_ROUTE.get(key, key)
            if route in VALID_ROUTES or key in VALID_ROUTES:
                raw_route_scores[route] = score
        raw_confidence = raw_route_scores.get(top_route, top_confidence)

        # ── Uncertainty discount: ONLY when uncertainty > 0.40 AND delta < 0.20 ──
        uncertainty = lane_uncertainty.get(top_route, 0.0)
        l4_cfg = self._config.get("layer4_decision_engine", {})
        UNCERTAINTY_DISCOUNT = l4_cfg.get("uncertainty_discount", 0.15)
        UNCERTAINTY_DELTA_THRESHOLD = l4_cfg.get("uncertainty_delta_threshold", 0.20)

        top2_delta_l4 = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 1.0

        if uncertainty > 0.40 and top2_delta_l4 < UNCERTAINTY_DELTA_THRESHOLD:
            discount = min(uncertainty * UNCERTAINTY_DISCOUNT, 0.20)
            effective_conf = top_confidence * (1 - discount)
            logger.info(
                f"[PREFILTER-L4][{decision_id}] uncertainty_discount: "
                f"{top_route} conf {top_confidence:.3f} → {effective_conf:.3f} "
                f"(uncertainty={uncertainty:.3f}, delta={top2_delta_l4:.3f})"
            )
            top_confidence = effective_conf

        # R4: Cold-start strategy (bypassed in Phase 1)
        cold_start_override = self._apply_cold_start(user_id)

        # R3: Stability guard (bypassed in Phase 1)
        stability_override = self._apply_stability_guard_sync(
            user_id, top_route
        )

        # R2: Rate limiting (bypassed in Phase 1)
        rate_limit_override = self._apply_rate_limiting()

        # Severity level (informative, not blocking in Phase 1)
        severity = ROUTE_SEVERITY.get(top_route, "low")

        # R7: Severity-adjusted threshold (bypassed in Phase 1)
        effective_threshold = self._get_effective_threshold(top_route)

        # R2: Dynamic Interaction — ask user when uncertain (BEFORE R7/B2 deferral)
        # BOTH conditions must be true: low confidence AND low delta (AND, not OR)
        if self._r2_enabled:
            delta_top2 = top_confidence - ranked[1][1] if len(ranked) > 1 else 1.0

            # FAST exclusion: only exclude when confidence is clearly chat (> 0.60)
            r2_cfg = self._config.get("reinforcements", {}).get("r2_dynamic_interaction", {})
            fast_excl_min = r2_cfg.get("fast_exclusion_min_confidence", 0.60)
            skip_r2 = (top_route in self._r2_excluded_routes and top_confidence > fast_excl_min)

            if not skip_r2:
                below_conf = top_confidence < self._r2_confidence_threshold
                delta_low = delta_top2 < self._r2_delta_threshold
                # AND — both conditions must be true
                if below_conf and delta_low:
                    options = self._build_interaction_options(ranked, query=query_text, decision_id=decision_id)
                    if len(options) >= 2:
                        # R2 vLLM: generate natural disambiguation question
                        vllm_question = None
                        if self._vllm_fn and self._r2_vllm_enabled and query_text:
                            vllm_question = await self._r2_vllm_disambiguation(
                                query=query_text,
                                ranked=ranked,
                                decision_id=decision_id,
                            )

                        logger.info(
                            f"[PREFILTER][{decision_id}] R2: DYNAMIC_INTERACTION triggered "
                            f"(conf={top_confidence:.3f}, delta={delta_top2:.3f}, "
                            f"options={len(options)}, vllm={'yes' if vllm_question else 'no'})"
                        )
                        return PreRouteDecision(
                            decision_id=decision_id,
                            route="DYNAMIC_INTERACTION",
                            confidence=top_confidence,
                            raw_confidence=raw_confidence,
                            reasoning=f"r2_dynamic_interaction:conf={top_confidence:.3f},delta={delta_top2:.3f}",
                            layer_trace=layer_trace,
                            scores=dict(route_scores),
                            raw_scores=raw_route_scores,
                            evidence=evidence,
                            severity_level=severity,
                            time_ms=elapsed,
                            deferred_to_llm_router=False,
                            interaction_options=InteractionOptions(
                                query=vllm_question or query_text,
                                options=options,
                                decision_id=decision_id,
                                fallback_route=top_route,
                            ),
                        )

        # Decision: defer if confidence below effective threshold (R7-adjusted)
        if top_confidence < effective_threshold:
            return PreRouteDecision(
                decision_id=decision_id,
                route="LLM_ROUTER",
                confidence=top_confidence,
                raw_confidence=raw_confidence,
                reasoning=(
                    f"low_confidence:{top_confidence:.3f}"
                    f"<{effective_threshold}(R7:{top_route})"
                ),
                layer_trace=layer_trace,
                scores=dict(route_scores),
                raw_scores=raw_route_scores,
                evidence=evidence,
                severity_level=severity,
                time_ms=elapsed,
                deferred_to_llm_router=True,
            )

        # Top-2 Delta Guard: defer if top-1 and top-2 are too close
        if len(ranked) > 1 and self._min_top2_delta > 0:
            delta_top2 = top_confidence - ranked[1][1]
            if delta_top2 < self._min_top2_delta:
                logger.info(
                    f"[PREFILTER][{decision_id}] Top-2 Delta Guard: "
                    f"delta={delta_top2:.3f}<{self._min_top2_delta} -> defer"
                )
                return PreRouteDecision(
                    decision_id=decision_id,
                    route="LLM_ROUTER",
                    confidence=top_confidence,
                    raw_confidence=raw_confidence,
                    reasoning=(
                        f"top2_delta_guard:{delta_top2:.3f}"
                        f"<{self._min_top2_delta}"
                    ),
                    layer_trace=layer_trace,
                    scores=dict(route_scores),
                    raw_scores=raw_route_scores,
                    evidence=evidence,
                    severity_level=severity,
                    time_ms=elapsed,
                    deferred_to_llm_router=True,
                )

        # Build reasoning string
        second_route = ranked[1][0] if len(ranked) > 1 else "none"
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        delta = top_confidence - second_score

        reasoning = (
            f"ranking:{top_route}={top_confidence:.3f},"
            f"delta={delta:.3f},"
            f"raw={raw_confidence:.3f}"
        )

        return PreRouteDecision(
            decision_id=decision_id,
            route=top_route,
            confidence=top_confidence,
            raw_confidence=raw_confidence,
            reasoning=reasoning,
            layer_trace=layer_trace,
            scores=dict(route_scores),
            raw_scores=raw_route_scores,
            evidence=evidence,
            severity_level=severity,
            time_ms=elapsed,
            deferred_to_llm_router=False,
        )

    async def _layer4_decision_engine(
        self,
        decision_id: str,
        route_scores: Dict[str, float],
        raw_scores: Dict[str, float],
        evidence: Dict[str, Any],
        layer_trace: List[str],
        user_id: Optional[str],
        t0: float,
        lane_uncertainty: Optional[Dict[str, float]] = None,
    ) -> PreRouteDecision:
        """Backward-compatible alias for final_decision()."""
        return await self.final_decision(
            decision_id=decision_id,
            route_scores=route_scores,
            raw_scores=raw_scores,
            evidence=evidence,
            layer_trace=layer_trace,
            user_id=user_id,
            t0=t0,
            lane_uncertainty=lane_uncertainty,
        )

    # ------------------------------------------------------------------
    # Reinforcement methods (all bypassed in Phase 1)
    # ------------------------------------------------------------------

    def _apply_cold_start(self, user_id: Optional[str]) -> Optional[str]:
        """R4: Cold-start user strategy.

        Phase 1: Complete bypass when enabled=false.
        """
        r4_cfg = self._config.get("reinforcements", {}).get("r4_cold_start", {})
        if not r4_cfg.get("enabled", False):
            return None  # complete bypass, zero computation
        # Future: check lifetime query count, apply neutral bias
        return None

    def _apply_stability_guard_sync(
        self, user_id: Optional[str], proposed_route: str
    ) -> Optional[str]:
        """R3: Synchronous stability check wrapper.

        Phase 1: Complete bypass when enabled=false.
        """
        r3_cfg = self._config.get("reinforcements", {}).get("r3_stability_guard", {})
        if not r3_cfg.get("enabled", False):
            return None  # complete bypass, zero computation
        # Future: check flapping via RouteStabilityGuard
        return None

    def _apply_rate_limiting(self) -> Optional[str]:
        """R2: Rate limiting cognitivo.

        Phase 1: Complete bypass when enabled=false.
        """
        r2_cfg = self._config.get("reinforcements", {}).get("r2_rate_limiting", {})
        if not r2_cfg.get("enabled", False):
            return None  # complete bypass, zero computation
        # Future: check dynamic_per_turn, dynamic_per_session
        return None

    def _build_interaction_options(
        self,
        ranked: List[tuple],
        query: str,
        decision_id: str,
    ) -> List[InteractionOption]:
        """Build interaction options from ranked route scores.

        Returns top-3 routes with score > 0.05, excluding FAST (unless only option).
        """
        ROUTE_LABELS = {
            "RAG": {"label": "Cerca nella Knowledge Base", "icon": "📚"},
            "WEB": {"label": "Cerca sul Web", "icon": "🌐"},
            "REPORT": {"label": "Genera un Report", "icon": "📊"},
            "FAST": {"label": "Rispondi direttamente", "icon": "💬"},
        }

        options = []
        for route, confidence in ranked:
            if confidence > 0.05 and route in ROUTE_LABELS:
                meta = ROUTE_LABELS[route]
                options.append(InteractionOption(
                    route=route,
                    label=meta["label"],
                    icon=meta["icon"],
                    confidence=round(confidence, 4),
                ))
        return options[:3]

    # ------------------------------------------------------------------
    # R2 vLLM Disambiguation (Fase 3)
    # ------------------------------------------------------------------

    async def _r2_vllm_disambiguation(
        self,
        query: str,
        ranked: List[tuple],
        decision_id: str,
    ) -> Optional[str]:
        """Generate a natural disambiguation question via vLLM.

        Called when R2 triggers. Returns a question string or None on failure.
        Uses structured JSON output (not tool-calling).
        Timeout: configurable, default 3s.
        """
        if not self._vllm_fn:
            return None

        # Build scores summary for prompt
        scores_str = ", ".join(f"{r}={s:.2f}" for r, s in ranked[:4])

        system_prompt = (
            "Sei un assistente di routing. L'utente ha fatto una domanda e il sistema "
            "non e' sicuro di come gestirla. I percorsi disponibili sono:\n"
            "- RAG: Cerca informazioni nella Knowledge Base aziendale\n"
            "- WEB: Cerca informazioni aggiornate sul web\n"
            "- REPORT: Genera un report strutturato\n"
            "- FAST: Rispondi direttamente\n\n"
            "Rispondi SOLO con un JSON valido: {\"question\": \"domanda breve\"}"
        )
        user_prompt = (
            f"L'utente chiede: \"{query}\"\n"
            f"Punteggi: {scores_str}\n\n"
            f"Formula UNA domanda breve e naturale (max 30 parole) per capire "
            f"cosa vuole l'utente."
        )

        try:
            result = await asyncio.wait_for(
                self._vllm_fn(system_prompt, user_prompt),
                timeout=self._r2_vllm_timeout,
            )
            if not result:
                return None

            # Parse JSON response
            import json as _json
            text = result.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = _json.loads(text)
            question = parsed.get("question", "").strip()
            if question and len(question) > 5:
                logger.info(
                    f"[PREFILTER-R2][{decision_id}] vLLM question: {question[:80]}"
                )
                return question
        except asyncio.TimeoutError:
            logger.warning(f"[PREFILTER-R2][{decision_id}] vLLM timeout ({self._r2_vllm_timeout}s)")
        except Exception as e:
            logger.warning(f"[PREFILTER-R2][{decision_id}] vLLM disambiguation failed: {e}")

        return None

    async def r2_resolve_response(
        self,
        original_query: str,
        user_response: str,
        ranked_scores: Dict[str, float],
    ) -> Optional[Dict[str, str]]:
        """Resolve R2 second turn: interpret user's response to choose route.

        Called from user_router when a pending R2 state is found.
        Returns {"route": "ROUTE_NAME", "reasoning": "..."} or None on failure.
        """
        if not self._vllm_fn:
            return None

        scores_str = ", ".join(f"{r}={s:.2f}" for r, s in sorted(
            ranked_scores.items(), key=lambda x: x[1], reverse=True
        ))

        system_prompt = (
            "Sei un assistente di routing. L'utente aveva fatto una domanda e gli "
            "abbiamo chiesto un chiarimento. Ora ha risposto.\n"
            "I percorsi disponibili sono: RAG, WEB, REPORT, FAST.\n\n"
            "Rispondi SOLO con un JSON valido: "
            "{\"route\": \"ROUTE_NAME\", \"reasoning\": \"breve motivazione\"}"
        )
        user_prompt = (
            f"Domanda originale: \"{original_query}\"\n"
            f"Risposta dell'utente: \"{user_response}\"\n"
            f"Punteggi originali: {scores_str}\n\n"
            f"Quale percorso scegliere?"
        )

        try:
            result = await asyncio.wait_for(
                self._vllm_fn(system_prompt, user_prompt),
                timeout=self._r2_vllm_timeout,
            )
            if not result:
                return None

            import json as _json
            text = result.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = _json.loads(text)
            route = parsed.get("route", "").upper().strip()
            reasoning = parsed.get("reasoning", "")

            if route in VALID_ROUTES and route not in ("LLM_ROUTER", "DYNAMIC_INTERACTION"):
                logger.info(
                    f"[PREFILTER-R2] vLLM resolved: route={route}, reason={reasoning[:60]}"
                )
                return {"route": route, "reasoning": reasoning}
        except asyncio.TimeoutError:
            logger.warning(f"[PREFILTER-R2] vLLM resolve timeout ({self._r2_vllm_timeout}s)")
        except Exception as e:
            logger.warning(f"[PREFILTER-R2] vLLM resolve failed: {e}")

        return None

    def _get_effective_threshold(self, route: str) -> float:
        """R7: Get effective threshold with severity penalty.

        Returns base_threshold + severity_penalty for the route.
        E.g., REPORT: 0.55 + 0.12 = 0.67
        """
        r7_cfg = self._config.get("reinforcements", {}).get("r7_severity", {})
        if not r7_cfg.get("enabled", False):
            return self._defer_threshold  # complete bypass

        base = r7_cfg.get("base_threshold", self._defer_threshold)
        penalties = r7_cfg.get("severity_penalties", {})
        penalty = penalties.get(route, 0.0)
        return base + penalty

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalized_uncertainty_from_scores(self, scores: Dict[str, float]) -> float:
        """Compute normalized entropy uncertainty in [0,1] from score distribution."""
        if not scores or len(scores) <= 1:
            return 0.0
        entropy = _entropy(scores)
        max_entropy = math.log2(len(scores))
        if max_entropy <= 0:
            return 0.0
        return float(max(0.0, min(1.0, entropy / max_entropy)))

    def _log_structured_decision(
        self,
        query_id: str,
        decision: PreRouteDecision,
        lane_uncertainty: Dict[str, float],
        top_scores: Optional[Dict[str, float]],
        score_range: float,
        absolute_floor_triggered: bool,
        low_range_triggered: bool,
        anti_proto_triggered: bool,
        k_value: int,
        matches_per_lane: Dict[str, int],
    ) -> None:
        """Emit one structured JSON log for each prefilter request."""
        ranked = sorted((top_scores or {}).items(), key=lambda x: x[1], reverse=True)
        top2_delta = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 1.0
        uncertainty = lane_uncertainty.get(decision.route, 0.0)
        if uncertainty <= 0.0:
            uncertainty = self._normalized_uncertainty_from_scores(top_scores or {})

        lane_hits = {
            "chat": int(matches_per_lane.get("chat", 0)),
            "web": int(matches_per_lane.get("web", 0)),
            "rag": int(matches_per_lane.get("rag", 0)),
            "report": int(matches_per_lane.get("report", 0)),
        }
        payload = {
            "query_id": query_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selected_lane": decision.route,
            "confidence": round(float(decision.confidence), 6),
            "uncertainty": round(float(uncertainty), 6),
            "top2_delta": round(float(top2_delta), 6),
            "score_range": round(float(score_range), 6),
            "absolute_floor_triggered": bool(absolute_floor_triggered),
            "low_range_triggered": bool(low_range_triggered),
            "anti_proto_triggered": bool(anti_proto_triggered),
            "deferred": bool(decision.deferred_to_llm_router),
            "k_value": int(k_value),
            "matches_per_lane": lane_hits,
        }
        logger.info(f"[PREFILTER-DECISION] {json.dumps(payload, ensure_ascii=False)}")

    def _make_defer_decision(
        self,
        decision_id: str,
        reason: str,
        t0: float,
        layer_trace: List[str],
    ) -> PreRouteDecision:
        """Create a defer-to-LLM_ROUTER decision."""
        elapsed = (time.perf_counter() - t0) * 1000
        return PreRouteDecision(
            decision_id=decision_id,
            route="LLM_ROUTER",
            confidence=0.0,
            raw_confidence=0.0,
            reasoning=f"defer:{reason}",
            layer_trace=layer_trace,
            severity_level="low",
            time_ms=elapsed,
            deferred_to_llm_router=True,
        )

    # ------------------------------------------------------------------
    # STEP 6: Metrics helpers
    # ------------------------------------------------------------------

    def _update_confidence_bucket(self, confidence: float) -> None:
        if confidence >= 0.9:
            self._metrics["confidence_buckets"]["0.9+"] += 1
        elif confidence >= 0.8:
            self._metrics["confidence_buckets"]["0.8-0.9"] += 1
        elif confidence >= 0.7:
            self._metrics["confidence_buckets"]["0.7-0.8"] += 1
        elif confidence >= 0.6:
            self._metrics["confidence_buckets"]["0.6-0.7"] += 1
        else:
            self._metrics["confidence_buckets"]["<0.6"] += 1

    def _update_sim_bucket(self, max_pos_sim: float) -> None:
        if max_pos_sim >= 0.8:
            self._metrics["max_pos_sim_buckets"]["0.8+"] += 1
        elif max_pos_sim >= 0.6:
            self._metrics["max_pos_sim_buckets"]["0.6-0.8"] += 1
        elif max_pos_sim >= 0.4:
            self._metrics["max_pos_sim_buckets"]["0.4-0.6"] += 1
        else:
            self._metrics["max_pos_sim_buckets"]["<0.4"] += 1

    def _log_metrics_summary(self) -> None:
        m = self._metrics
        total = m["total_queries"]
        if total == 0:
            return
        direct_pct = m["direct_routes"] / total * 100
        defer_pct = m["deferred_to_llm"] / total * 100
        r2_pct = m["dynamic_interaction"] / total * 100
        logger.info(
            f"[PREFILTER-METRICS] queries={total} direct={direct_pct:.0f}% "
            f"defer={defer_pct:.0f}% R2={r2_pct:.0f}% "
            f"routes={m['route_distribution']} "
            f"confidence={m['confidence_buckets']} "
            f"max_pos_sim={m['max_pos_sim_buckets']}"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Return current metrics snapshot (for admin endpoint)."""
        snapshot = dict(self._metrics)
        shadow = dict(snapshot.get("shadow", {}))
        shadow.pop("conf_delta_sum", None)
        snapshot["shadow"] = shadow
        return snapshot

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        self._metrics = {
            "total_queries": 0,
            "direct_routes": 0,
            "deferred_to_llm": 0,
            "dynamic_interaction": 0,
            "ood_floor_rejected": 0,
            "low_range_deferred": 0,
            "route_distribution": {},
            "confidence_buckets": {"0.9+": 0, "0.8-0.9": 0, "0.7-0.8": 0, "0.6-0.7": 0, "<0.6": 0},
            "max_pos_sim_buckets": {"0.8+": 0, "0.6-0.8": 0, "0.4-0.6": 0, "<0.4": 0},
            "shadow": {
                "total": 0,
                "route_changed_count": 0,
                "defer_changed_count": 0,
                "avg_conf_delta": 0.0,
                "conf_delta_sum": 0.0,
                "timeouts": 0,
                "errors": 0,
                "skipped": 0,
                "route_matrix": {},
            },
        }

    # ------------------------------------------------------------------
    # STEP 7: Auto-suggest prototype candidates
    # ------------------------------------------------------------------

    def _add_prototype_candidate(self, query: str, max_pos_sim: float,
                                  top_lane: str, decision_id: str) -> None:
        """Buffer a query as candidate for new prototype."""
        self._prototype_candidates.append({
            "query": query[:100],
            "max_pos_sim": round(max_pos_sim, 4),
            "top_lane": top_lane,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": "weak_match",
            "decision_id": decision_id,
        })
        # Circular buffer: keep last 100
        if len(self._prototype_candidates) > 100:
            self._prototype_candidates = self._prototype_candidates[-100:]
        logger.info(
            f"[PREFILTER-AUTOSUGGEST][{decision_id}] Candidate: "
            f"'{query[:50]}' max_sim={max_pos_sim:.3f} → {top_lane}"
        )

    def get_prototype_candidates(self) -> List[Dict[str, Any]]:
        """Return prototype candidates buffer (for admin endpoint)."""
        return list(self._prototype_candidates)
