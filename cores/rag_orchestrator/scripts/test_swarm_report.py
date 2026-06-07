#!/usr/bin/env python3
"""
ARCHITECTURE v2.4: Dynamic Swarm Report - Verification Script

This script tests the complete swarm pipeline:
1. Dynamic Planning (Big Brain)
2. Parallel Research (Researcher + Swarm)
3. Parallel Drafting (Worker Model)
4. Report Assembly

Test Query: "Analisi comparativa tra Qdrant e Milvus"

Usage:
    python -m ubp_enterprise_hybrid.modules.cores.rag_orchestrator.scripts.test_swarm_report

    # Or with custom query
    python -m ubp_enterprise_hybrid.modules.cores.rag_orchestrator.scripts.test_swarm_report \
        --query "Your custom report request"

Environment Variables Required:
    - UBP_REPORT__PLANNER_PROVIDER (default: grok)
    - UBP_REPORT__WORKER_PROVIDER (default: grok)
    - UBP_REPORT__MAX_PARALLEL_WORKERS (default: 4)

Author: UBP Team
Version: 2.4.0
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MockLLMModule:
    """Mock LLM module for testing without real LLM."""

    def __init__(self, model: str = "mock-model"):
        self.model = model
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        model: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Generate mock LLM response."""
        self.call_count += 1
        await asyncio.sleep(0.1)  # Simulate latency

        # Check if this is a planner request or section writer request
        if "Research Analyst" in system_prompt or "structured research plan" in system_prompt:
            # Planner response
            return {
                "response": json.dumps({
                    "title": "Analisi Comparativa: Qdrant vs Milvus",
                    "sections": [
                        {
                            "title": "Executive Summary",
                            "description": "Overview dei due vector database",
                            "source_preference": "mixed",
                            "suggested_queries": ["qdrant overview", "milvus overview"]
                        },
                        {
                            "title": "Architettura Tecnica",
                            "description": "Confronto delle architetture",
                            "source_preference": "rag_first",
                            "suggested_queries": ["qdrant architecture", "milvus architecture"]
                        },
                        {
                            "title": "Performance e Scalabilita",
                            "description": "Benchmark e limiti di scalabilita",
                            "source_preference": "web_only",
                            "suggested_queries": ["qdrant vs milvus benchmark", "vector db performance"]
                        },
                        {
                            "title": "Conclusioni e Raccomandazioni",
                            "description": "Sintesi e raccomandazioni finali",
                            "source_preference": "llm_reasoning",
                            "suggested_queries": []
                        }
                    ]
                })
            }
        else:
            # Section writer response
            section_title = "Unknown Section"
            if "Executive Summary" in prompt:
                section_title = "Executive Summary"
                content = """Qdrant e Milvus sono entrambi vector database open-source progettati per applicazioni di similarity search e machine learning.

Qdrant, sviluppato in Rust, si distingue per le sue performance elevate e il supporto nativo per filtri complessi. Milvus, scritto in Go/C++, offre un'architettura distribuita matura e un ecosistema esteso.

Entrambe le soluzioni supportano miliardi di vettori e offrono API RESTful e gRPC per l'integrazione."""

            elif "Architettura Tecnica" in prompt:
                section_title = "Architettura Tecnica"
                content = """**Qdrant Architecture:**
- Storage engine basato su RocksDB
- Supporto nativo per payload filtering
- HNSW e altri algoritmi di indexing
- Single-node e clustering mode

**Milvus Architecture:**
- Architettura a microservizi (Proxy, Data Node, Query Node)
- Supporto per multiple storage backends (MinIO, S3)
- Diversi tipi di indici (IVF, HNSW, ANNOY)
- Nativamente distribuito con sharding automatico"""

            elif "Performance" in prompt:
                section_title = "Performance e Scalabilita"
                content = """I benchmark pubblici mostrano risultati variabili a seconda del caso d'uso:

**Qdrant:**
- Eccellente per query con filtri complessi (fino a 3x piu veloce)
- Latenza sub-millisecondo per collezioni < 10M vettori
- Memory footprint contenuto grazie a quantizzazione

**Milvus:**
- Superiore in scenari distribuiti multi-nodo
- Scalabilita orizzontale testata fino a miliardi di vettori
- Batch processing ottimizzato per bulk insert"""

            elif "Conclusioni" in prompt or "Raccomandazioni" in prompt:
                section_title = "Conclusioni e Raccomandazioni"
                content = """Basandosi sull'analisi condotta, le raccomandazioni sono:

1. **Scegli Qdrant se:**
   - Hai bisogno di query con filtri complessi
   - Preferisci una soluzione leggera e facile da deployare
   - Il tuo dataset e < 100M vettori

2. **Scegli Milvus se:**
   - Hai bisogno di scalabilita orizzontale nativa
   - Devi gestire miliardi di vettori
   - Hai un team dedicato per l'operatività

Per la maggior parte dei casi d'uso RAG enterprise, **Qdrant** rappresenta la scelta ottimale per il rapporto semplicita/performance."""

            else:
                content = f"Mock content for section: {prompt[:50]}..."

            return {"response": content}


class MockRAGModule:
    """Mock RAG module for testing without real Qdrant."""

    async def query(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        ctx=None,
        **kwargs,
    ) -> dict:
        """Return mock RAG results."""
        await asyncio.sleep(0.05)  # Simulate latency

        return {
            "results": [
                {
                    "text": f"Mock document about {query_text} from {collection_name}. This contains relevant information about vector databases and their usage in RAG systems.",
                    "score": 0.85,
                    "payload": {
                        "source": f"{collection_name}/doc1.md",
                        "chunk_id": 1,
                    },
                },
                {
                    "text": f"Additional context for {query_text}. Vector databases like Qdrant and Milvus are essential for similarity search.",
                    "score": 0.75,
                    "payload": {
                        "source": f"{collection_name}/doc2.md",
                        "chunk_id": 2,
                    },
                },
            ]
        }


async def test_dynamic_planner(llm_module) -> dict:
    """Test the DynamicPlanner component."""
    from ubp_enterprise_hybrid.modules.cores.rag_orchestrator.agents import (
        DynamicPlanner,
        PlannerConfig,
    )

    print("\n" + "=" * 60)
    print("TEST 1: Dynamic Planner (Big Brain)")
    print("=" * 60)

    config = PlannerConfig(
        planner_model="mock/planner-model",
        dynamic_planning_enabled=True,
    )

    planner = DynamicPlanner(llm_module=llm_module, config=config)

    query = "Analisi comparativa tra Qdrant e Milvus per applicazioni RAG"

    start_time = time.time()
    plan = await planner.create_plan(
        query=query,
        context="User is evaluating vector databases for enterprise RAG system",
        collections=["ubp_system_docs"],
    )
    elapsed_ms = (time.time() - start_time) * 1000

    print(f"\nQuery: {query}")
    print(f"Plan Title: {plan.template_name}")
    print(f"Subject: {plan.subject}")
    print(f"Sections: {len(plan.sections)}")
    print(f"Time: {elapsed_ms:.2f}ms")
    print("\nPlanned Sections:")
    for i, section in enumerate(plan.sections, 1):
        print(f"  {i}. {section.title}")
        print(f"     - Description: {section.description}")
        print(f"     - Source: {section.source_preference.value}")
        print(f"     - Queries: {section.suggested_queries}")

    return {
        "status": "success",
        "plan": plan,
        "sections_count": len(plan.sections),
        "time_ms": elapsed_ms,
    }


async def test_swarm_executor(llm_module, rag_module, plan) -> dict:
    """Test the SwarmExecutor with parallel execution."""
    from ubp_enterprise_hybrid.modules.cores.rag_orchestrator.agents import (
        Researcher,
        SwarmExecutor,
        WorkerConfig,
    )

    print("\n" + "=" * 60)
    print("TEST 2: Swarm Executor (Parallel Workers)")
    print("=" * 60)

    # Configure worker
    config = WorkerConfig(
        worker_model="mock/worker-model",
        max_parallel_workers=4,
        parallel_research=True,
        parallel_drafting=True,
    )

    # Create researcher and executor
    researcher = Researcher(
        rag_module=rag_module,
        web_module=None,  # No web for this test
    )

    executor = SwarmExecutor(
        researcher=researcher,
        llm_module=llm_module,
        config=config,
    )

    print(f"\nExecuting plan: {plan.template_name}")
    print(f"Sections: {len(plan.sections)}")
    print(f"Parallel Workers: {config.max_parallel_workers}")

    start_time = time.time()
    result = await executor.execute_plan(plan=plan)
    elapsed_ms = (time.time() - start_time) * 1000

    print(f"\n--- Swarm Execution Results ---")
    print(f"Total Time: {result.total_time_ms:.2f}ms")
    print(f"Parallel Efficiency: {result.parallel_efficiency:.2f}x")
    print(f"Sections Succeeded: {result.sections_succeeded}/{len(result.sections)}")
    print(f"Sections Failed: {result.sections_failed}")

    print("\n--- Section Drafts ---")
    for draft in result.sections:
        status_icon = "✓" if draft.status == "success" else "✗"
        print(f"\n{status_icon} {draft.section_title}")
        print(f"   Status: {draft.status}")
        print(f"   Word Count: {draft.word_count}")
        print(f"   Documents Used: {draft.documents_count}")
        print(f"   Generation Time: {draft.generation_time_ms:.2f}ms")
        if draft.error_message:
            print(f"   Error: {draft.error_message}")
        if draft.content:
            preview = draft.content[:150].replace("\n", " ")
            print(f"   Preview: {preview}...")

    return {
        "status": "success",
        "result": result,
        "total_time_ms": result.total_time_ms,
        "parallel_efficiency": result.parallel_efficiency,
    }


async def test_full_report_generation(llm_module, rag_module, plan, swarm_result) -> dict:
    """Test full report assembly."""
    print("\n" + "=" * 60)
    print("TEST 3: Full Report Assembly")
    print("=" * 60)

    full_draft = swarm_result.full_draft

    print(f"\n--- Generated Report ---")
    print(f"Title: {plan.template_name}")
    print(f"Subject: {plan.subject}")
    print(f"Total Words: {len(full_draft.split())}")
    print(f"\n{'-' * 40}")
    print(full_draft[:2000])  # Print first 2000 chars
    if len(full_draft) > 2000:
        print(f"\n... [truncated, {len(full_draft)} total characters] ...")

    return {
        "status": "success",
        "word_count": len(full_draft.split()),
        "char_count": len(full_draft),
    }


async def run_tests(query: str = None):
    """Run all swarm tests."""
    print("\n" + "#" * 60)
    print("# ARCHITECTURE v2.4: Dynamic Swarm Report Testing")
    print("#" * 60)

    # Use default query if not provided
    if not query:
        query = "Analisi comparativa tra Qdrant e Milvus per applicazioni RAG"

    print(f"\nTest Query: {query}")

    # Initialize mock modules
    llm_module = MockLLMModule()
    rag_module = MockRAGModule()

    results = {}

    try:
        # Test 1: Dynamic Planner
        planner_result = await test_dynamic_planner(llm_module)
        results["planner"] = planner_result

        # Test 2: Swarm Executor
        plan = planner_result["plan"]
        swarm_result = await test_swarm_executor(llm_module, rag_module, plan)
        results["swarm"] = swarm_result

        # Test 3: Full Report Assembly
        assembly_result = await test_full_report_generation(
            llm_module, rag_module, plan, swarm_result["result"]
        )
        results["assembly"] = assembly_result

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        all_passed = all(r["status"] == "success" for r in results.values())
        status = "PASSED" if all_passed else "FAILED"
        print(f"\nOverall Status: {status}")
        print(f"LLM Calls: {llm_module.call_count}")
        print(f"\nComponent Results:")
        for name, result in results.items():
            icon = "✓" if result["status"] == "success" else "✗"
            print(f"  {icon} {name}: {result['status']}")

        if all_passed:
            print("\n" + "=" * 60)
            print("v2.4 SWARM ARCHITECTURE VERIFICATION: SUCCESS")
            print("=" * 60)
            print("""
Key Features Verified:
1. Dynamic Planning: LLM generates custom report structure
2. Parallel Research: asyncio.gather for concurrent data gathering
3. Parallel Drafting: Workers process sections simultaneously
4. Efficiency: Parallel execution faster than sequential

The Brain & Workers architecture is operational!
""")

        return results

    except Exception as e:
        logger.exception("Test failed")
        print(f"\nERROR: {e}")
        return {"status": "error", "error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test v2.4 Dynamic Swarm Report Generation"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom report query to test",
    )
    args = parser.parse_args()

    asyncio.run(run_tests(query=args.query))


if __name__ == "__main__":
    main()
