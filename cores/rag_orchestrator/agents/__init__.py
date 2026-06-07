"""
RAG Orchestrator Agents Package

ARCHITECTURE v2.3/v2.4/v2.5: Interactive Analyst + Dynamic Swarm + Artifact Export

This package contains specialized agents for advanced RAG workflows:
- ReportSessionManager: State machine for interactive report generation
- Researcher: RAG-first data gathering with web fallback
- DynamicPlanner: LLM-powered plan generation (Brain)
- SwarmExecutor: Parallel section processing (Workers)
- ArtifactManager: Report export and download (v2.5)

Redis Keys Used:
- ubp:{env}:report:session:{session_id} - Session state and plan
- ubp:{env}:report:data:{session_id}    - Gathered research data
- ubp:{env}:report:draft:{session_id}   - Generated draft content
- ubp:{env}:artifact:{artifact_id}      - Artifact metadata (v2.5)
- ubp:{env}:artifact:session:{session_id} - Session artifacts list (v2.5)

Environment Configuration (v2.4/v2.5):
- UBP_REPORT__PLANNER_PROVIDER - Provider for planning LLM (e.g. grok, vllm_remote)
- UBP_REPORT__WORKER_PROVIDER - Provider for parallel worker LLM
- UBP_REPORT__MAX_PARALLEL_WORKERS - Swarm parallelism limit
- UBP_ARTIFACT__BASE_PATH - Artifact storage path (default: /app/artifacts)
- UBP_ARTIFACT__TTL_HOURS - Artifact expiration (default: 168 = 7 days)
"""

from .report_session import (
    ReportSessionManager,
    ReportState,
    ReportSession,
    ReportPlan,
    SectionPlan,
)

from .researcher import (
    Researcher,
    SourcePreference,
    ResearchResult,
    ResearchConfig,
    # v2.4: Swarm Execution
    WorkerConfig,
    SectionDraft,
    SwarmResult,
    SwarmExecutor,
)

from .planner import (
    DynamicPlanner,
    PlannerConfig,
    should_use_dynamic_planning,
)

from .artifact_manager import (
    ArtifactManager,
    ArtifactMetadata,
    ArtifactFormat,
)

from .ingestion_job import (
    IngestJob,
    IngestJobState,
    IngestFileEntry,
    FileStatus,
)

__all__ = [
    # Session Management
    "ReportSessionManager",
    "ReportState",
    "ReportSession",
    "ReportPlan",
    "SectionPlan",
    # Research
    "Researcher",
    "SourcePreference",
    "ResearchResult",
    "ResearchConfig",
    # Dynamic Planning (v2.4)
    "DynamicPlanner",
    "PlannerConfig",
    "should_use_dynamic_planning",
    # Swarm Execution (v2.4)
    "WorkerConfig",
    "SectionDraft",
    "SwarmResult",
    "SwarmExecutor",
    # Artifact Export (v2.5)
    "ArtifactManager",
    "ArtifactMetadata",
    "ArtifactFormat",
    # Batch Ingestion
    "IngestJob",
    "IngestJobState",
    "IngestFileEntry",
    "FileStatus",
]
