#!/usr/bin/env python3
"""
Phase B: Calibration report — run filter_rag_context on real data.

Usage:
    python3 ubp_enterprise_hybrid/modules/cores/filter_rag_context/tests/calibration_report.py
"""

import json
import sys
from pathlib import Path
from statistics import mean, median

# Import providers directly (zero UBP deps)
_providers_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _providers_dir)
from providers import FilterConfig, filter_rag_context  # noqa: E402
sys.path.pop(0)

DATA_FILE = Path(__file__).parent / "real_data_battery.json"


def format_text(text: str, max_len: int = 50) -> str:
    t = text.replace("\n", " ").strip()
    return t[:max_len] + "..." if len(t) > max_len else t


def print_separator(char="─", width=80):
    print(char * width)


def run_calibration():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = FilterConfig()
    all_rerank = []
    all_cosine = []
    zero_output_cases = []
    fallback_cases = []

    print("=" * 80)
    print("CALIBRATION REPORT — filter_rag_context on real battery data")
    print(f"Config: min_score_hard={config.min_score_hard}, "
          f"min_score_soft={config.min_score_soft}, "
          f"keyword_bonus={config.keyword_bonus}")
    print("=" * 80)

    for label, case in data.items():
        query = case["query"]
        route = case.get("route", "?")
        pipeline = case.get("pipeline", "?")
        chunks = case["chunks"]

        # Normalize chunk format for filter (expects "score" key for rerank)
        filter_chunks = []
        for c in chunks:
            fc = {
                "text": c.get("text", ""),
                "chunk_id": c.get("chunk_id", ""),
                "source_id": c.get("source_id", c.get("collection", "")),
            }
            if c.get("rerank_score") is not None:
                fc["score"] = c["rerank_score"]
                all_rerank.append(c["rerank_score"])
            if c.get("cosine_score") is not None:
                fc["cosine_score"] = c["cosine_score"]
                all_cosine.append(c["cosine_score"])
            filter_chunks.append(fc)

        result = filter_rag_context(filter_chunks, query, config)

        print()
        print_separator("━")
        print(f"{label}: \"{query}\"")
        print(f"  route={route} | pipeline={pipeline} | chunks_in={len(chunks)}")
        print_separator()

        # Stats
        s = result.stats
        print(f"  input={s.input_count} → output={s.output_count} "
              f"| dropped_reasons={s.dropped_by_reason}")
        print(f"  avg_kept={s.avg_score_kept:.4f} | avg_dropped={s.avg_score_dropped:.4f} "
              f"| all_low={s.all_low_relevance} | fallback={s.fallback_triggered}")

        if s.output_count == 0:
            zero_output_cases.append(label)
        if s.fallback_triggered:
            fallback_cases.append(label)

        # Kept
        print(f"\n  KEPT ({len(result.kept)}):")
        for v in result.kept:
            txt = ""
            for c in chunks:
                if c.get("chunk_id") == v.chunk_id:
                    txt = format_text(c.get("text", ""))
                    break
            reasons_str = ",".join(v.reasons[:3])
            print(f"    {v.chunk_id:20s} score={v.final_score:.4f} "
                  f"pen={v.penalties:.2f} bon={v.bonuses:.2f} [{reasons_str}] {txt}")

        # Dropped
        if result.dropped:
            print(f"\n  DROPPED ({len(result.dropped)}):")
            for v in result.dropped:
                txt = ""
                for c in chunks:
                    if c.get("chunk_id") == v.chunk_id:
                        txt = format_text(c.get("text", ""), 40)
                        break
                reasons_str = ",".join(v.reasons[:3])
                print(f"    {v.chunk_id:20s} score={v.final_score:.4f} "
                      f"[{v.action}] [{reasons_str}] {txt}")

    # ===== SUMMARY =====
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_rerank:
        sorted_r = sorted(all_rerank)
        n = len(sorted_r)
        p25 = sorted_r[n // 4]
        p75 = sorted_r[3 * n // 4]
        print(f"\n  Rerank score distribution ({n} chunks):")
        print(f"    min={min(sorted_r):.4f}  p25={p25:.4f}  median={median(sorted_r):.4f}  "
              f"p75={p75:.4f}  max={max(sorted_r):.4f}  mean={mean(sorted_r):.4f}")

        # Histogram buckets
        buckets = [0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
        print(f"\n    Distribution:")
        for i in range(len(buckets) - 1):
            lo, hi = buckets[i], buckets[i + 1]
            count = sum(1 for s in sorted_r if lo <= s < hi)
            bar = "█" * count
            label_str = f"[{lo:.2f}-{hi:.2f})"
            print(f"    {label_str:14s} {count:3d} {bar}")

    if all_cosine:
        sorted_c = sorted(all_cosine)
        print(f"\n  Cosine score distribution ({len(sorted_c)} chunks):")
        print(f"    min={min(sorted_c):.4f}  median={median(sorted_c):.4f}  "
              f"max={max(sorted_c):.4f}  mean={mean(sorted_c):.4f}")

    print(f"\n  Zero-output cases: {zero_output_cases or 'none'}")
    print(f"  Fallback cases: {fallback_cases or 'none'}")

    # Threshold recommendations
    if all_rerank:
        sorted_r = sorted(all_rerank)
        n = len(sorted_r)
        print(f"\n  THRESHOLD RECOMMENDATIONS:")
        print(f"    Current min_score_hard = {config.min_score_hard}")
        print(f"    Current min_score_soft = {config.min_score_soft}")

        # Find natural gap
        below_hard = sum(1 for s in sorted_r if s < config.min_score_hard)
        below_soft = sum(1 for s in sorted_r if s < config.min_score_soft)
        print(f"    Chunks below hard ({config.min_score_hard}): {below_hard}/{n} ({100*below_hard/n:.0f}%)")
        print(f"    Chunks below soft ({config.min_score_soft}): {below_soft}/{n} ({100*below_soft/n:.0f}%)")

        # Suggest optimal soft threshold: p25 of kept chunks
        kept_scores = [s for s in sorted_r if s >= config.min_score_hard]
        if kept_scores:
            suggested_soft = sorted(kept_scores)[len(kept_scores) // 4]
            print(f"    Suggested min_score_soft (p25 of non-hard-dropped): {suggested_soft:.4f}")


if __name__ == "__main__":
    run_calibration()
