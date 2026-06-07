"""
Standalone tests for filter_rag_context — zero UBP dependencies.

Run:
    pytest ubp_enterprise_hybrid/modules/cores/filter_rag_context/tests/test_standalone.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Direct import of providers.py — bypasses __init__.py (which imports adapter → UBP deps)
_providers_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _providers_dir)
from providers import (  # noqa: E402
    FilterConfig, FilterResult, filter_rag_context, get_collection_affinity,
    boost_by_entity, ENTITY_SYNONYMS,
)
sys.path.pop(0)

CASES_FILE = Path(__file__).parent / "test_cases.json"


@pytest.fixture(scope="module")
def cases():
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def default_config():
    return FilterConfig()


# -----------------------------------------------------------------------
# A) T12 "ok" — all garbage, G2 fallback
# -----------------------------------------------------------------------
class TestA_AllGarbage:

    def test_all_hard_dropped_g2_recovers(self, cases, default_config):
        data = cases["A_all_garbage"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        assert result.fallback_triggered is True, "G2 fallback must trigger"
        assert len(result.kept) == 2, f"G2 must recover exactly 2 chunks, got {len(result.kept)}"
        assert result.stats.fallback_triggered is True

        # All original chunks had score < 0.06 → hard-dropped
        assert result.stats.dropped_by_reason.get("SCORE_HARD", 0) >= 9

    def test_recovered_chunk_is_highest_score(self, cases, default_config):
        data = cases["A_all_garbage"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        # a3 has highest score (0.0019)
        assert result.kept[0].chunk_id == "a3"
        assert "G2_FALLBACK" in result.kept[0].reasons


# -----------------------------------------------------------------------
# B) Mixed scores — kept 3, dropped 2
# -----------------------------------------------------------------------
class TestB_MixedScores:

    def test_kept_and_dropped_counts(self, cases, default_config):
        data = cases["B_mixed_scores"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        kept_ids = {v.chunk_id for v in result.kept}
        assert "b0" in kept_ids, "0.67 must be kept"
        assert "b1" in kept_ids, "0.54 must be kept"
        assert "b2" in kept_ids, "0.31 must be kept"
        # b3 (0.08) > hard threshold (0.06) — survives hard-drop, may be kept
        assert len(result.kept) >= 3, f"Expected ≥3 kept, got {len(result.kept)}"

    def test_hard_drop_low_score(self, cases, default_config):
        data = cases["B_mixed_scores"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        dropped_ids = {v.chunk_id for v in result.dropped}
        assert "b4" in dropped_ids, "0.02 must be hard-dropped"

    def test_no_fallback(self, cases, default_config):
        data = cases["B_mixed_scores"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)
        assert result.fallback_triggered is False

    def test_keyword_bonus_applied(self, cases, default_config):
        data = cases["B_mixed_scores"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        # b0 contains "effetti", "collaterali", "metformina" → keyword bonus
        b0 = next(v for v in result.kept if v.chunk_id == "b0")
        assert b0.bonuses > 0, "b0 must have keyword bonus"


# -----------------------------------------------------------------------
# C) High uniform — all kept
# -----------------------------------------------------------------------
class TestC_HighUniform:

    def test_all_kept(self, cases, default_config):
        data = cases["C_high_uniform"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        assert len(result.kept) == 5, f"All 5 should be kept, got {len(result.kept)}"
        assert len(result.dropped) == 0
        assert result.fallback_triggered is False

    def test_sorted_by_score_desc(self, cases, default_config):
        data = cases["C_high_uniform"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        scores = [v.final_score for v in result.kept]
        assert scores == sorted(scores, reverse=True), "Must be sorted desc"


# -----------------------------------------------------------------------
# D) Gap analysis — cluster + tail
# -----------------------------------------------------------------------
class TestD_GapAnalysis:

    def test_top_cluster_kept(self, cases, default_config):
        data = cases["D_gap_analysis"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        kept_ids = {v.chunk_id for v in result.kept}
        assert "d0" in kept_ids, "0.71 must be kept"
        assert "d1" in kept_ids, "0.65 must be kept"
        assert "d2" in kept_ids, "0.59 must be kept"

    def test_tail_scores_lower(self, cases, default_config):
        data = cases["D_gap_analysis"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        # d3/d4/d5 scores (0.25/0.22/0.18) are much lower than top cluster (0.71/0.65/0.59)
        all_verdicts = {v.chunk_id: v for v in result.kept + result.dropped}
        d0 = all_verdicts.get("d0")
        d5 = all_verdicts.get("d5")
        assert d0 is not None and d5 is not None
        assert d0.final_score > d5.final_score, "Top cluster must score higher than tail"
        # Top 3 (d0, d1, d2) must be kept
        kept_ids = {v.chunk_id for v in result.kept}
        assert {"d0", "d1", "d2"}.issubset(kept_ids)


# -----------------------------------------------------------------------
# E) Keyword rescue — borderline chunk saved by keyword bonus
# -----------------------------------------------------------------------
class TestE_KeywordRescue:

    def test_keyword_bonus_saves_borderline(self, cases, default_config):
        data = cases["E_keyword_rescue"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        assert len(result.kept) >= 1, "Keyword bonus should save the chunk"
        e0 = result.kept[0]
        assert e0.chunk_id == "e0"
        assert e0.bonuses > 0, "Must have keyword bonus"
        assert e0.final_score > data["chunks"][0]["score"], (
            f"Final score {e0.final_score} must exceed base {data['chunks'][0]['score']}"
        )


# -----------------------------------------------------------------------
# F) Zero output — all hard-dropped, G2 recovers
# -----------------------------------------------------------------------
class TestF_ZeroOutput:

    def test_all_hard_dropped_g2_fallback(self, cases, default_config):
        data = cases["F_zero_output"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        assert result.fallback_triggered is True
        assert len(result.kept) == 2
        assert result.stats.dropped_by_reason.get("SCORE_HARD", 0) >= 4

    def test_recovered_is_highest(self, cases, default_config):
        data = cases["F_zero_output"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        # f4 has highest score (0.05)
        assert result.kept[0].chunk_id == "f4"


# -----------------------------------------------------------------------
# G) Boilerplate — ToC chunk penalized but not dropped
# -----------------------------------------------------------------------
class TestG_Boilerplate:

    def test_boilerplate_detected_and_penalized(self, cases, default_config):
        data = cases["G_boilerplate"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        all_verdicts = result.kept + result.dropped
        g0 = next(v for v in all_verdicts if v.chunk_id == "g0")
        assert "BOILERPLATE" in g0.reasons, "ToC must be flagged as boilerplate"
        assert g0.penalties >= default_config.boilerplate_penalty

    def test_clean_chunk_ranked_higher(self, cases, default_config):
        data = cases["G_boilerplate"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        # g1 (clean, 0.62) should rank above g0 (boilerplate, 0.45 penalized)
        assert len(result.kept) >= 1
        assert result.kept[0].chunk_id == "g1", "Clean chunk must rank first"

    def test_boilerplate_still_kept(self, cases, default_config):
        data = cases["G_boilerplate"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        kept_ids = {v.chunk_id for v in result.kept}
        # g0 has score 0.45 — penalty reduces it but it's still above floor
        assert "g0" in kept_ids, "Boilerplate chunk should still be kept (score 0.45)"


# -----------------------------------------------------------------------
# H) Low confidence — all scores < 0.35, LOW_CONF_BOOST applied
# -----------------------------------------------------------------------
class TestH_LowConfidence:

    def test_low_conf_boost_applied(self, cases, default_config):
        data = cases["H_low_confidence"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        for v in result.kept:
            assert "LOW_CONF_BOOST" in v.reasons, (
                f"{v.chunk_id}: missing LOW_CONF_BOOST reason"
            )

    def test_keyword_bonus_amplified(self, cases, default_config):
        data = cases["H_low_confidence"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        # h0 contains "warfarin", "interazioni", "farmacologiche" → keyword match
        h0 = next(v for v in result.kept if v.chunk_id == "h0")
        # Bonuses should be doubled: original bonus * 2
        assert h0.bonuses > default_config.keyword_bonus, (
            f"h0 bonuses={h0.bonuses} should exceed base keyword_bonus={default_config.keyword_bonus}"
        )

    def test_penalties_reduced(self, cases, default_config):
        data = cases["H_low_confidence"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        # All scores < 0.35 → penalties halved
        for v in result.kept + result.dropped:
            if "LOW_CONF_BOOST" in v.reasons:
                # Halved penalties should be ≤ 0.5 * max possible penalty
                assert v.penalties <= 0.5, f"{v.chunk_id}: penalties={v.penalties} should be halved"

    def test_all_chunks_kept(self, cases, default_config):
        data = cases["H_low_confidence"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        # All 4 chunks are above hard threshold (0.06) and with LOW_CONF_BOOST
        assert len(result.kept) == 4, f"All 4 should be kept, got {len(result.kept)}"
        assert result.fallback_triggered is False


# -----------------------------------------------------------------------
# I) Min kept 2 — all hard-dropped, G2 recovers 2
# -----------------------------------------------------------------------
class TestI_MinKept2:

    def test_g2_recovers_two(self, cases, default_config):
        data = cases["I_min_kept_2"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        assert len(result.kept) == 2, f"G2 must recover 2, got {len(result.kept)}"
        assert result.fallback_triggered is True

    def test_both_recovered_have_g2_reason(self, cases, default_config):
        data = cases["I_min_kept_2"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        for v in result.kept:
            assert "G2_FALLBACK" in v.reasons, (
                f"{v.chunk_id}: missing G2_FALLBACK reason"
            )

    def test_highest_scores_recovered(self, cases, default_config):
        data = cases["I_min_kept_2"]
        result = filter_rag_context(data["chunks"], data["query"], default_config)

        kept_ids = {v.chunk_id for v in result.kept}
        # i4 (0.05) and i2 (0.04) are the two highest scores
        assert "i4" in kept_ids, "i4 (0.05) must be recovered"
        assert "i2" in kept_ids, "i2 (0.04) must be recovered"


# -----------------------------------------------------------------------
# J) Collection affinity — medical query
# -----------------------------------------------------------------------
_AVAILABLE_5 = ["medical", "legal", "ubp_system_docs", "personal", "kbm"]


class TestJ_AffinityMedical:

    def test_medical_first(self):
        result = get_collection_affinity(
            "effetti collaterali metformina", _AVAILABLE_5
        )
        assert result[0] == "medical", f"Expected medical first, got {result}"

    def test_medical_score_high(self):
        # "effetti collaterali" (1 phrase) + "metformina" (1 word) = 2 matches × 3.0 + 1.0 pharma = 7.0
        result = get_collection_affinity(
            "effetti collaterali metformina", _AVAILABLE_5
        )
        assert "medical" in result
        assert "legal" not in result or result.index("medical") < result.index("legal")


# -----------------------------------------------------------------------
# K) Collection affinity — technical query
# -----------------------------------------------------------------------
class TestK_AffinityTechnical:

    def test_ubp_system_docs_first(self):
        result = get_collection_affinity(
            "Come funziona il sistema UBP?", _AVAILABLE_5
        )
        assert result[0] == "ubp_system_docs", f"Expected ubp_system_docs first, got {result}"

    def test_has_backup(self):
        result = get_collection_affinity(
            "Come funziona il sistema UBP?", _AVAILABLE_5
        )
        # Should have ubp_system_docs + 1 neutral backup
        assert len(result) >= 2, "Should include neutral backup"


# -----------------------------------------------------------------------
# L) Collection affinity — ambiguous, both medical and legal
# -----------------------------------------------------------------------
class TestL_AffinityAmbiguous:

    def test_both_domains_present(self):
        result = get_collection_affinity(
            "effetti della normativa sulla salute",
            ["medical", "legal", "ubp_system_docs"],
        )
        assert "medical" in result, "medical should match (salute)"
        assert "legal" in result, "legal should match (normativa)"

    def test_medical_before_legal(self):
        result = get_collection_affinity(
            "effetti della normativa sulla salute",
            ["medical", "legal", "ubp_system_docs"],
        )
        # medical: "salute" → 1×3.0 = 3.0
        # legal: "normativa" → 1×2.0 = 2.0
        assert result.index("medical") < result.index("legal"), (
            f"medical (3.0) should rank before legal (2.0), got {result}"
        )


# -----------------------------------------------------------------------
# M) Collection affinity — user selected bypass
# -----------------------------------------------------------------------
class TestM_AffinityUserBypass:

    def test_user_selected_bypass(self):
        result = get_collection_affinity(
            "effetti collaterali metformina",
            _AVAILABLE_5,
            user_selected=["legal"],
        )
        assert result == ["legal"], f"User bypass must return exactly user_selected, got {result}"


# -----------------------------------------------------------------------
# N) Collection affinity — generic query, no match
# -----------------------------------------------------------------------
class TestN_AffinityNeutral:

    def test_all_returned(self):
        result = get_collection_affinity("ciao come stai?", _AVAILABLE_5)
        assert result == _AVAILABLE_5, f"No match → return all available, got {result}"

    def test_never_empty(self):
        result = get_collection_affinity("xyz abc", ["col1"])
        assert len(result) >= 1, "Must never return empty"


# -----------------------------------------------------------------------
# O) Collection affinity — medical wins over legal
# -----------------------------------------------------------------------
class TestO_AffinityMedicalWins:

    def test_medical_over_legal(self):
        result = get_collection_affinity(
            "metformina contratto assicurativo",
            ["medical", "legal"],
        )
        # medical: "metformina" pos (1×3.0), "contratto" neg (-0.8), pharma +1.0 = 3.2
        # legal: "contratto" pos (1×2.0) = 2.0
        assert result[0] == "medical", f"medical (3.2) must beat legal (2.0), got {result}"
        assert "legal" in result, "legal should also be present (score > 0)"

    def test_medical_score_exceeds_legal(self):
        result = get_collection_affinity(
            "metformina contratto assicurativo",
            ["medical", "legal"],
        )
        assert result.index("medical") < result.index("legal")


# -----------------------------------------------------------------------
# Stats consistency
# -----------------------------------------------------------------------
class TestStats:

    def test_input_output_counts(self, cases, default_config):
        for case_name, data in cases.items():
            result = filter_rag_context(data["chunks"], data["query"], default_config)
            assert result.stats.input_count == len(data["chunks"]), f"{case_name}: input_count mismatch"
            assert result.stats.output_count == len(result.kept), f"{case_name}: output_count mismatch"
            total = result.stats.output_count + len(result.dropped)
            assert total >= result.stats.input_count, f"{case_name}: kept+dropped must cover all chunks"

    def test_deterministic(self, cases, default_config):
        """Same input → same output."""
        for case_name, data in cases.items():
            r1 = filter_rag_context(data["chunks"], data["query"], default_config)
            r2 = filter_rag_context(data["chunks"], data["query"], default_config)
            assert [v.chunk_id for v in r1.kept] == [v.chunk_id for v in r2.kept], (
                f"{case_name}: not deterministic"
            )
            assert [v.final_score for v in r1.kept] == [v.final_score for v in r2.kept]


# =======================================================================
# Entity boost tests (P, Q, R, S)
# =======================================================================

class TestP_EntityBoostPharma:
    """'report paracetamolo' — pharma entity boost + synonym expansion."""

    def _make_chunks(self):
        return [
            {"chunk_id": "c1", "text": "paracetamolo dosaggio 500mg", "score": 0.15, "source_id": "s1"},
            {"chunk_id": "c2", "text": "case report methodology", "score": 0.12, "source_id": "s2"},
            {"chunk_id": "c3", "text": "acetaminofene epatico", "score": 0.10, "source_id": "s3"},
            {"chunk_id": "c4", "text": "reporting bias assessment", "score": 0.11, "source_id": "s4"},
            {"chunk_id": "c5", "text": "tachipirina bambini", "score": 0.08, "source_id": "s5"},
        ]

    def test_boosted_scores(self):
        chunks = self._make_chunks()
        result = boost_by_entity(chunks, "report paracetamolo")
        scores = {c["chunk_id"]: c["score"] for c in result}
        assert abs(scores["c1"] - 0.225) < 0.001, f"c1 expected 0.225, got {scores['c1']}"
        assert abs(scores["c3"] - 0.15) < 0.001, f"c3 expected 0.15, got {scores['c3']}"
        assert abs(scores["c5"] - 0.12) < 0.001, f"c5 expected 0.12, got {scores['c5']}"

    def test_penalized_scores(self):
        chunks = self._make_chunks()
        result = boost_by_entity(chunks, "report paracetamolo")
        scores = {c["chunk_id"]: c["score"] for c in result}
        assert abs(scores["c2"] - 0.084) < 0.001, f"c2 expected 0.084, got {scores['c2']}"
        assert abs(scores["c4"] - 0.077) < 0.001, f"c4 expected 0.077, got {scores['c4']}"

    def test_order_after_boost(self):
        chunks = self._make_chunks()
        result = boost_by_entity(chunks, "report paracetamolo")
        sorted_chunks = sorted(result, key=lambda c: c["score"], reverse=True)
        order = [c["chunk_id"] for c in sorted_chunks]
        assert order == ["c1", "c3", "c5", "c2", "c4"], f"Expected c1>c3>c5>c2>c4, got {order}"

    def test_entity_reasons(self):
        chunks = self._make_chunks()
        boost_by_entity(chunks, "report paracetamolo")
        reasons = {c["chunk_id"]: c.get("_entity_reasons", []) for c in chunks}
        assert any("ENTITY_BOOST" in r for r in reasons["c1"])
        assert any("ENTITY_BOOST" in r for r in reasons["c3"])  # synonym
        assert any("ENTITY_BOOST" in r for r in reasons["c5"])  # synonym
        assert reasons["c2"] == ["ENTITY_MISS"]
        assert reasons["c4"] == ["ENTITY_MISS"]


class TestQ_EntityBoostDomain:
    """'cos'è il GDPR' — domain keyword boost (non-pharma)."""

    def _make_chunks(self):
        return [
            {"chunk_id": "c1", "text": "GDPR regolamento europeo", "score": 0.70, "source_id": "s1"},
            {"chunk_id": "c2", "text": "normativa privacy", "score": 0.50, "source_id": "s2"},
            {"chunk_id": "c3", "text": "GDPR sanzioni", "score": 0.40, "source_id": "s3"},
        ]

    def test_gdpr_boost(self):
        chunks = self._make_chunks()
        result = boost_by_entity(chunks, "cos'è il GDPR")
        scores = {c["chunk_id"]: c["score"] for c in result}
        assert abs(scores["c1"] - 1.05) < 0.001, f"c1 expected 1.05, got {scores['c1']}"
        assert abs(scores["c3"] - 0.60) < 0.001, f"c3 expected 0.60, got {scores['c3']}"

    def test_gdpr_miss(self):
        chunks = self._make_chunks()
        result = boost_by_entity(chunks, "cos'è il GDPR")
        scores = {c["chunk_id"]: c["score"] for c in result}
        assert abs(scores["c2"] - 0.35) < 0.001, f"c2 expected 0.35, got {scores['c2']}"


class TestR_EntityNoopGeneric:
    """'ciao come stai?' — no entity → noop."""

    def test_scores_unchanged(self):
        chunks = [
            {"chunk_id": "c1", "text": "hello world", "score": 0.50, "source_id": "s1"},
            {"chunk_id": "c2", "text": "buongiorno amici", "score": 0.30, "source_id": "s2"},
            {"chunk_id": "c3", "text": "test content here", "score": 0.10, "source_id": "s3"},
        ]
        original_scores = {c["chunk_id"]: c["score"] for c in chunks}
        result = boost_by_entity(chunks, "ciao come stai?")
        for c in result:
            assert c["score"] == original_scores[c["chunk_id"]], (
                f"{c['chunk_id']}: score changed from {original_scores[c['chunk_id']]} to {c['score']}"
            )

    def test_no_entity_reasons(self):
        chunks = [
            {"chunk_id": "c1", "text": "hello world", "score": 0.50, "source_id": "s1"},
        ]
        result = boost_by_entity(chunks, "ciao come stai?")
        assert "_entity_reasons" not in result[0]


class TestS_EntitySynonymExpansion:
    """'metformina interazioni' — synonym 'glucophage' triggers boost."""

    def test_synonym_boost(self):
        chunks = [
            {"chunk_id": "c1", "text": "glucophage side effects", "score": 0.20, "source_id": "s1"},
        ]
        result = boost_by_entity(chunks, "metformina interazioni")
        assert abs(result[0]["score"] - 0.30) < 0.001, f"Expected 0.30, got {result[0]['score']}"

    def test_synonym_reason(self):
        chunks = [
            {"chunk_id": "c1", "text": "glucophage side effects", "score": 0.20, "source_id": "s1"},
        ]
        boost_by_entity(chunks, "metformina interazioni")
        reasons = chunks[0].get("_entity_reasons", [])
        assert any("ENTITY_BOOST" in r for r in reasons)
