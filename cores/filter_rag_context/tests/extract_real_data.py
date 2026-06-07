#!/usr/bin/env python3
"""
Phase A: Extract real chunk data from battery test queries.

Runs RAG queries via /api/user/chat and extracts sources (chunks with scores)
from the API response. Saves to real_data_battery.json for calibration.

Usage:
    python3 ubp_enterprise_hybrid/modules/cores/filter_rag_context/tests/extract_real_data.py
"""

import json
import time
from pathlib import Path

import requests

BASE = "http://localhost:8000"
CREDS = {"username": "michele67", "password": "253167Michele"}
OUTPUT = Path(__file__).parent / "real_data_battery.json"

TESTS = [
    ("T03", "Effetti collaterali della metformina", "medical"),
    ("T04", "Cos'è il diabete di tipo 2?", "medical"),
    ("T05", "Cos'è il GDPR?", "legal"),
    ("T06", "Effetti collaterali della metformina", None),
    ("T07", "Come funziona il sistema UBP?", None),
    ("T10", "Genera un report sul paracetamolo", "medical"),
    ("T11", "metformina", "medical"),
    ("T12", "ok", None),
]


def login() -> str:
    r = requests.post(f"{BASE}/api/auth/login", json=CREDS, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def extract_chunks(sources):
    """Extract normalized chunk data from API sources array."""
    chunks = []
    for i, src in enumerate(sources):
        meta = src.get("metadata", {}) if isinstance(src.get("metadata"), dict) else {}
        chunks.append({
            "chunk_id": src.get("chunk_id", src.get("id", f"chunk_{i}")),
            "collection": src.get("kb_id", meta.get("collection", "")),
            "text": (src.get("text", "") or src.get("content", ""))[:200],
            "rerank_score": src.get("rerank_score"),
            "cosine_score": src.get("score"),
            "source_id": meta.get("source", meta.get("doc_id", src.get("kb_id", f"src_{i}"))),
        })
    return chunks


def run_query(token, label, query, collection):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"query": query, "conversation_id": f"calib-{label}"}
    if collection:
        payload["collection"] = collection

    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/api/user/chat", headers=headers, json=payload, timeout=120)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    if r.status_code != 200:
        print(f"  {label}: HTTP {r.status_code} — SKIP")
        return None

    d = r.json()
    sources = d.get("sources", [])
    route_info = d.get("router_classification", {})
    pipeline = d.get("pipeline_name", d.get("metadata", {}).get("pipeline_used", ""))

    chunks = extract_chunks(sources)
    print(f"  {label}: {len(chunks)} chunks | {elapsed_ms}ms | route={route_info.get('intent', '?')} | pipeline={pipeline}")

    return {
        "query": query,
        "collection": collection,
        "route": route_info.get("intent", "unknown"),
        "confidence": route_info.get("confidence", 0),
        "pipeline": pipeline,
        "elapsed_ms": elapsed_ms,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def main():
    print("=== Phase A: Extract real chunk data ===\n")
    token = login()
    print(f"Authenticated. Running {len(TESTS)} queries...\n")

    results = {}
    for label, query, coll in TESTS:
        data = run_query(token, label, query, coll)
        if data:
            results[label] = data

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT}")
    print(f"Total test cases: {len(results)}")
    total_chunks = sum(d["chunk_count"] for d in results.values())
    print(f"Total chunks extracted: {total_chunks}")


if __name__ == "__main__":
    main()
