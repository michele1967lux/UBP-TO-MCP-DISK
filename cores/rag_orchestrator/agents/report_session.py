"""
ARCHITECTURE v2.6: Interactive Analyst - Report Session Manager

State machine for interactive report generation workflow.

States:
    IDLE                      -> No active session
    PLANNING                  -> Proposing template and sections
    AWAITING_APPROVAL         -> Waiting for user confirmation of structure
    AWAITING_ENRICHMENT_CONFIG -> Waiting for user enrichment configuration (v2.6)
    RESEARCHING               -> Executing RAG/Web searches with enrichment
    WRITING                   -> Generating report content
    REVIEW                    -> User reviewing draft
    COMPLETED                 -> Report finalized
    CANCELLED                 -> Session cancelled

v2.6 Changes:
    - Added AWAITING_ENRICHMENT_CONFIG state for per-section enrichment UI
    - Added SectionEnrichmentConfig dataclass for granular control
    - Added debug_mode for real-time worker diagnostics
    - Enrichment applied per-worker during RESEARCHING phase

Redis Storage:
    ubp:{env}:report:session:{session_id}  - Session state + plan (HASH)
    ubp:{env}:report:data:{session_id}     - Research data (HASH)
    ubp:{env}:report:draft:{session_id}    - Draft content (STRING)

Author: UBP Team
Version: 2.3.0
"""

import json
import logging
import uuid
import time
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from . import report_metrics  # v5.0.4: Prometheus metrics
from .report_utils import extract_subject

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================


class ReportState(str, Enum):
    """Report session states."""

    IDLE = "idle"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_ENRICHMENT_CONFIG = "awaiting_enrichment_config"  # v2.6: User configures enrichment
    RESEARCHING = "researching"
    WRITING = "writing"
    REVIEW = "review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class SourcePreference(str, Enum):
    """Data source preferences for sections."""

    RAG_ONLY = "rag_only"
    WEB_ONLY = "web_only"
    RAG_FIRST = "rag_first"
    WEB_FIRST = "web_first"
    MIXED = "mixed"
    LLM_REASONING = "llm_reasoning"


# =============================================================================
# v2.6: ENRICHMENT CONFIGURATION
# =============================================================================


@dataclass
class SectionEnrichmentConfig:
    """
    v2.6: Per-section enrichment configuration.

    Configured by user via UI checkbox panel after structure approval.
    Each worker receives its own enrichment config during RESEARCHING phase.
    """

    # Basic enrichment (fast)
    rerank_enabled: bool = True  # Cross-encoder reranking (default ON)
    query_expansion_enabled: bool = False  # Query variants generation

    # Advanced enrichment (slower, higher quality)
    hyde_enabled: bool = False  # Hypothetical Document Embeddings
    investigative_enabled: bool = False  # Decompose into sub-queries
    metadata_injection_enabled: bool = False  # Enrich context with metadata

    # Debug mode
    debug_enabled: bool = False  # Emit detailed debug events for this section

    # Reranker selection: "primary" (bge-reranker), "medical" (MedCPT), "cascade" (both)
    reranker_type: str = "primary"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rerank_enabled": self.rerank_enabled,
            "query_expansion_enabled": self.query_expansion_enabled,
            "hyde_enabled": self.hyde_enabled,
            "investigative_enabled": self.investigative_enabled,
            "metadata_injection_enabled": self.metadata_injection_enabled,
            "debug_enabled": self.debug_enabled,
            "reranker_type": self.reranker_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SectionEnrichmentConfig":
        """Create from dictionary."""
        return cls(
            rerank_enabled=data.get("rerank_enabled", True),
            query_expansion_enabled=data.get("query_expansion_enabled", False),
            hyde_enabled=data.get("hyde_enabled", False),
            investigative_enabled=data.get("investigative_enabled", False),
            metadata_injection_enabled=data.get("metadata_injection_enabled", False),
            debug_enabled=data.get("debug_enabled", False),
            reranker_type=data.get("reranker_type", "primary"),
        )

    @classmethod
    def from_preset(cls, preset: str, source_preference: SourcePreference = None) -> "SectionEnrichmentConfig":
        """
        Create from preset name with optional source_preference-based defaults.

        Presets:
            - fast: Only rerank (fastest)
            - standard: Rerank + expansion (balanced)
            - quality: Full enrichment (slowest, best quality)
            - custom: All disabled (user configures manually)
        """
        presets = {
            "fast": cls(rerank_enabled=True),
            "standard": cls(rerank_enabled=True, query_expansion_enabled=True, metadata_injection_enabled=True),
            "quality": cls(
                rerank_enabled=True,
                query_expansion_enabled=True,
                hyde_enabled=True,
                investigative_enabled=True,
                metadata_injection_enabled=True,
                reranker_type="cascade",
            ),
            "custom": cls(rerank_enabled=False),
        }

        config = presets.get(preset, presets["standard"])

        # Auto-adjust based on source_preference
        if source_preference == SourcePreference.LLM_REASONING:
            # No retrieval needed - disable all enrichment
            return cls(rerank_enabled=False, debug_enabled=config.debug_enabled)

        if source_preference == SourcePreference.WEB_ONLY:
            # HyDE not useful for web search
            config.hyde_enabled = False

        return config


@dataclass
class SectionPlan:
    """Plan for a single report section."""

    title: str
    description: str
    source_preference: SourcePreference
    required: bool = True
    max_tokens: int = 800
    suggested_queries: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, researching, writing, completed, skipped
    research_data: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    # v2.6: Per-section enrichment configuration (set by user via UI)
    enrichment_config: Optional[SectionEnrichmentConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "source_preference": self.source_preference.value if isinstance(self.source_preference, SourcePreference) else self.source_preference,
            "required": self.required,
            "max_tokens": self.max_tokens,
            "suggested_queries": self.suggested_queries,
            "depends_on": self.depends_on,
            "status": self.status,
            "has_research_data": self.research_data is not None,
            "has_content": self.content is not None,
            # v2.6: Include enrichment config if set
            "enrichment_config": self.enrichment_config.to_dict() if self.enrichment_config else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SectionPlan":
        """Create from dictionary."""
        source_pref = data.get("source_preference", "mixed")
        if isinstance(source_pref, str):
            source_pref = SourcePreference(source_pref)

        # v2.6: Parse enrichment config if present
        enrichment_data = data.get("enrichment_config")
        enrichment_config = None
        if enrichment_data:
            enrichment_config = SectionEnrichmentConfig.from_dict(enrichment_data)

        return cls(
            title=data["title"],
            description=data.get("description", ""),
            source_preference=source_pref,
            required=data.get("required", True),
            max_tokens=data.get("max_tokens", 800),
            suggested_queries=data.get("suggested_queries", []),
            depends_on=data.get("depends_on", []),
            status=data.get("status", "pending"),
            research_data=data.get("research_data"),
            content=data.get("content"),
            enrichment_config=enrichment_config,
        )


@dataclass
class ReportPlan:
    """Complete report generation plan."""

    template_id: str
    template_name: str
    subject: str  # User's query subject
    sections: List[SectionPlan]
    collections: List[str] = field(default_factory=list)
    user_modifications: List[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "template_id": self.template_id,
            "template_name": self.template_name,
            "subject": self.subject,
            "sections": [s.to_dict() for s in self.sections],
            "collections": self.collections,
            "user_modifications": self.user_modifications,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportPlan":
        """Create from dictionary."""
        sections = [SectionPlan.from_dict(s) for s in data.get("sections", [])]
        return cls(
            template_id=data["template_id"],
            template_name=data.get("template_name", ""),
            subject=data["subject"],
            sections=sections,
            collections=data.get("collections", []),
            user_modifications=data.get("user_modifications", []),
            created_at=data.get("created_at", ""),
        )

    def get_proposal_text(self) -> str:
        """Generate human-readable proposal for user confirmation."""
        lines = [
            f"Ho selezionato il template **'{self.template_name}'**.",
            f"\nArgomento: **{self.subject}**",
            f"\nSezioni pianificate:",
        ]

        for i, section in enumerate(self.sections, 1):
            source_label = {
                SourcePreference.RAG_ONLY: "RAG interno",
                SourcePreference.WEB_ONLY: "Ricerca Web",
                SourcePreference.RAG_FIRST: "RAG (fallback Web)",
                SourcePreference.WEB_FIRST: "Web (fallback RAG)",
                SourcePreference.MIXED: "RAG + Web combinati",
                SourcePreference.LLM_REASONING: "Ragionamento LLM",
            }.get(section.source_preference, str(section.source_preference))

            required_label = "" if section.required else " (opzionale)"
            lines.append(f"  {i}. **{section.title}** - Fonte: {source_label}{required_label}")

        lines.append("\n**Vuoi procedere o modificare il piano?**")
        lines.append("_(Rispondi 'Sì/Procedi' per confermare, oppure specifica le modifiche)_")

        return "\n".join(lines)


# =============================================================================
# v2.6: DEBUG EVENT STRUCTURES
# =============================================================================


@dataclass
class WorkerDebugEvent:
    """
    v2.6: Debug event emitted by a worker during execution.

    These events are stored in Redis and streamed to frontend for real-time
    debugging visualization in expandable chat panels.
    """

    event_id: str
    session_id: str
    section_index: int
    section_title: str
    phase: str  # "init", "enrichment", "retrieval", "rerank", "complete", "error"
    timestamp: str
    worker_id: str

    # Settings used
    enrichment_config: Dict[str, Any]
    model_info: Dict[str, Any]  # provider, model_name, temperature, etc.

    # Phase-specific data
    input_data: Optional[Dict[str, Any]] = None  # Query, collections, etc.
    output_data: Optional[Dict[str, Any]] = None  # Results, chunks, etc.
    metrics: Optional[Dict[str, Any]] = None  # Latency, counts, scores, etc.
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "section_index": self.section_index,
            "section_title": self.section_title,
            "phase": self.phase,
            "timestamp": self.timestamp,
            "worker_id": self.worker_id,
            "enrichment_config": self.enrichment_config,
            "model_info": self.model_info,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metrics": self.metrics,
            "error": self.error,
        }

    @classmethod
    def create(
        cls,
        session_id: str,
        section_index: int,
        section_title: str,
        phase: str,
        enrichment_config: "SectionEnrichmentConfig",
        model_info: Dict[str, Any],
        **kwargs,
    ) -> "WorkerDebugEvent":
        """Factory method to create a debug event."""
        return cls(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            section_index=section_index,
            section_title=section_title,
            phase=phase,
            timestamp=datetime.now(timezone.utc).isoformat(),
            worker_id=f"worker_{section_index}",
            enrichment_config=enrichment_config.to_dict() if enrichment_config else {},
            model_info=model_info,
            **kwargs,
        )


@dataclass
class DebugEventBatch:
    """
    v2.6: Collection of debug events for a session.

    Used to batch events for efficient Redis storage and frontend retrieval.
    """

    session_id: str
    events: List[WorkerDebugEvent] = field(default_factory=list)
    global_debug_enabled: bool = False  # Master switch from UI

    def add_event(self, event: WorkerDebugEvent) -> None:
        """Add event to batch."""
        self.events.append(event)

    def get_events_for_section(self, section_index: int) -> List[WorkerDebugEvent]:
        """Get all events for a specific section."""
        return [e for e in self.events if e.section_index == section_index]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "events": [e.to_dict() for e in self.events],
            "events_count": len(self.events),
            "global_debug_enabled": self.global_debug_enabled,
        }


@dataclass
class ReportSession:
    """Active report session with state and data."""

    session_id: str
    user_id: str
    client_id: Optional[str]
    state: ReportState
    plan: Optional[ReportPlan]
    conversation_id: str
    created_at: str
    updated_at: str
    expires_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "client_id": self.client_id,
            "state": self.state.value,
            "plan": self.plan.to_dict() if self.plan else None,
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportSession":
        """Create from dictionary."""
        plan_data = data.get("plan")
        plan = ReportPlan.from_dict(plan_data) if plan_data else None

        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            client_id=data.get("client_id"),
            state=ReportState(data["state"]),
            plan=plan,
            conversation_id=data["conversation_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            expires_at=data["expires_at"],
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# REPORT SESSION MANAGER
# =============================================================================


class ReportSessionManager:
    """
    State machine manager for interactive report generation.

    Manages the lifecycle of report sessions including:
    - Template selection and proposal
    - User approval workflow
    - Research execution
    - Draft generation and review

    All state is persisted in Redis for reliability.

    Usage:
        manager = ReportSessionManager(redis_client, templates_path)
        await manager.initialize()

        # Start new session
        session = await manager.start_session(
            user_id="user123",
            query="Audit sicurezza UBP",
            conversation_id="conv123"
        )

        # Process user feedback
        result = await manager.process_input(
            session_id=session.session_id,
            user_input="Procedi"
        )
    """

    # Redis key patterns
    SESSION_KEY_PATTERN = "ubp:report:session:{session_id}"
    DATA_KEY_PATTERN = "ubp:report:data:{session_id}"
    DRAFT_KEY_PATTERN = "ubp:report:draft:{session_id}"
    USER_SESSIONS_KEY_PATTERN = "ubp:report:user:{user_id}:sessions"

    # Default TTLs (in seconds)
    DEFAULT_SESSION_TTL = 3600  # 1 hour
    PLANNING_TTL = 1800  # 30 minutes
    RESEARCHING_TTL = 1200  # 20 minutes (v5.0.4: increased for large reports with enrichment)
    REVIEW_TTL = 3600  # 1 hour

    def __init__(
        self,
        redis_client,
        templates_path: Optional[Path] = None,
        key_manager=None,
        llm_module=None,
        researcher=None,
        artifact_manager=None,
        enrichment_module=None,  # v2.6: For per-section enrichment
        worker_llm_module=None,  # v5.0.3 RPT-001: Separate worker LLM for drafting
        planner_llm_module=None,  # v6.0.1: Separate planner LLM (e.g. grok for planning)
    ):
        """
        Initialize ReportSessionManager.

        Args:
            redis_client: Async Redis client instance
            templates_path: Path to report_templates.yaml
            key_manager: Optional RedisKeyManager for environment-aware keys
            llm_module: Optional LLM module for dynamic planning (v2.4)
            researcher: Optional Researcher instance for data gathering (v2.4)
            artifact_manager: Optional ArtifactManager for report export (v2.5)
            enrichment_module: Optional enrichment module for per-section enrichment (v2.6)
            worker_llm_module: Optional separate LLM module for SwarmExecutor drafting (v5.0.3)
            planner_llm_module: Optional separate LLM module for DynamicPlanner (v6.0.1)
        """
        self.redis = redis_client
        self._key_manager = key_manager
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._selection_rules: List[Dict[str, Any]] = []
        self._state_machine_config: Dict[str, Any] = {}
        self._initialized = False

        # v2.4: Dynamic Planner (Brain)
        self._llm_module = llm_module
        self._dynamic_planner = None  # Initialized in initialize()

        # v2.4: Swarm Executor (Workers)
        self._researcher = researcher
        self._swarm_executor = None  # Initialized in initialize()
        # v5.0.3 RPT-001: Worker LLM may differ from main LLM (e.g. vllm vs grok)
        self._worker_llm_module = worker_llm_module
        # v6.0.1: Planner LLM may differ from main LLM (e.g. grok for structured planning)
        self._planner_llm_module = planner_llm_module

        # v2.5: Artifact Manager (Export)
        self._artifact_manager = artifact_manager

        # v2.6: Enrichment Module (Per-section enrichment)
        self._enrichment_module = enrichment_module

        # v6.3.3: Track active swarm tasks for error observability
        self._active_swarm_tasks: dict = {}  # session_id → asyncio.Task

        # Resolve templates path
        if templates_path is None:
            self._templates_path = Path(__file__).parent.parent / "config" / "report_templates.yaml"
        else:
            self._templates_path = templates_path

    def _get_key(self, pattern: str, **kwargs) -> str:
        """Get environment-aware Redis key."""
        base_key = pattern.format(**kwargs)
        if self._key_manager:
            return self._key_manager.get_key(base_key)
        return base_key

    async def initialize(self) -> Dict[str, Any]:
        """
        Load templates and initialize manager.

        Returns:
            Status dict with loaded template count
        """
        try:
            if self._templates_path.exists():
                with open(self._templates_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                # Parse templates
                for template in config.get("templates", []):
                    self._templates[template["id"]] = template

                # Parse selection rules
                self._selection_rules = config.get("selection_rules", [])

                # Parse state machine config
                self._state_machine_config = config.get("state_machine", {})

                logger.info(
                    f"ReportSessionManager initialized: {len(self._templates)} templates loaded",
                    extra={"templates": list(self._templates.keys())},
                )
            else:
                logger.warning(f"Templates file not found: {self._templates_path}")

            # v2.4: Initialize Dynamic Planner if LLM available
            # v6.0.1: Use dedicated planner LLM if available (e.g. grok for structured JSON)
            dynamic_planning_enabled = False
            planner_llm = self._planner_llm_module or self._llm_module
            if planner_llm:
                try:
                    from .planner import DynamicPlanner, PlannerConfig

                    planner_config = PlannerConfig.from_env()
                    if planner_config.dynamic_planning_enabled:
                        self._dynamic_planner = DynamicPlanner(
                            llm_module=planner_llm,
                            config=planner_config,
                        )
                        dynamic_planning_enabled = True
                        logger.info(
                            f"DynamicPlanner initialized: provider={planner_config.planner_provider}"
                        )
                except Exception as e:
                    logger.warning(f"Could not initialize DynamicPlanner: {e}")

            # v2.4/v2.6: Initialize Swarm Executor if Researcher and LLM available
            swarm_enabled = False
            if self._researcher and self._llm_module:
                try:
                    from .researcher import SwarmExecutor, WorkerConfig

                    worker_config = WorkerConfig.from_env()
                    # v5.0.3 RPT-001: Use dedicated worker LLM if available,
                    # otherwise fall back to main LLM module
                    swarm_llm = self._worker_llm_module or self._llm_module
                    self._swarm_executor = SwarmExecutor(
                        researcher=self._researcher,
                        llm_module=swarm_llm,
                        config=worker_config,
                        redis_client=self.redis,  # v2.6: For debug events
                        enrichment_module=self._enrichment_module,  # v2.6: For per-section enrichment
                    )
                    swarm_enabled = True
                    logger.info(
                        f"SwarmExecutor initialized: provider={worker_config.worker_provider}, "
                        f"llm_module={type(swarm_llm).__name__}, "
                        f"max_parallel={worker_config.max_parallel_workers}, "
                        f"enrichment={'enabled' if self._enrichment_module else 'disabled'}"
                    )
                except Exception as e:
                    logger.warning(f"Could not initialize SwarmExecutor: {e}")

            self._initialized = True
            return {
                "status": "initialized",
                "templates_loaded": len(self._templates),
                "template_ids": list(self._templates.keys()),
                "dynamic_planning_enabled": dynamic_planning_enabled,
                "swarm_enabled": swarm_enabled,
            }

        except Exception as e:
            logger.error(f"Failed to initialize ReportSessionManager: {e}")
            raise

    # =========================================================================
    # SESSION LIFECYCLE
    # =========================================================================

    async def start_session(
        self,
        user_id: str,
        query: str,
        conversation_id: str,
        client_id: Optional[str] = None,
        collections: Optional[List[str]] = None,
        preferred_template: Optional[str] = None,
        force_dynamic: bool = False,
        context: Optional[str] = None,
        execution_mode: str = "report",
        user_roles: Optional[List[str]] = None,
    ) -> ReportSession:
        """
        Start a new report session.

        Args:
            user_id: User identifier
            query: User's report request query
            conversation_id: Associated conversation ID
            client_id: Optional client identifier
            collections: Optional list of RAG collections to use
            preferred_template: Optional template ID to force (Fast Path)
            force_dynamic: Force dynamic planning even if template matches (v2.4)
            context: Optional conversation context for dynamic planning (v2.4)
            execution_mode: Pipeline mode — report, insight, exploratory (v5.1.0)

        Returns:
            New ReportSession in PLANNING state
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # v5.0.4: Record session start metric
        report_metrics.record_session_started()

        plan = None
        use_dynamic = False
        template = None  # Populated only in static path; used for artifact config

        # =====================================================================
        # v2.4: DYNAMIC vs STATIC PLANNING DECISION
        # =====================================================================
        # Priority:
        # 1. If preferred_template specified by user -> Static (Fast Path)
        # 2. If dynamic planner available -> Always dynamic
        # 3. Fallback -> Static template selection (no dynamic planner)

        if preferred_template:
            # Fast Path: User explicitly selected a template
            logger.info(f"[SESSION] Using preferred template: {preferred_template}")

        elif self._dynamic_planner:
            # Dynamic planning for all non-explicit requests
            use_dynamic = True

        # =====================================================================
        # PLAN CREATION
        # =====================================================================

        planning_start = time.time()

        if use_dynamic and self._dynamic_planner:
            # v2.4: Dynamic Planning (Big Brain)
            logger.info("[SESSION] Using DYNAMIC planning (Big Brain)")
            try:
                plan = await self._dynamic_planner.create_plan(
                    query=query,
                    context=context,
                    collections=collections,
                    conversation_id=conversation_id,
                )
                report_metrics.record_planning(time.time() - planning_start, "dynamic")
                logger.info(
                    f"[SESSION] Dynamic plan created: {len(plan.sections)} sections",
                    extra={"sections": [s.title for s in plan.sections]},
                )
            except Exception as e:
                logger.warning(f"[SESSION] Dynamic planning failed: {e}, falling back to static")
                plan = None

        if plan is None:
            # Static Planning (Template-based)
            template_id = preferred_template or self._select_template(query)
            template = self._templates.get(template_id)

            if not template:
                template_id = list(self._templates.keys())[0] if self._templates else "research_brief"
                template = self._templates.get(template_id, {})

            # Extract subject from query
            subject = extract_subject(query)

            # Build section plans from template
            sections = []
            for section_def in template.get("sections", []):
                source_pref = section_def.get("source_preference", "mixed")
                if isinstance(source_pref, str):
                    source_pref = SourcePreference(source_pref)

                # Substitute {subject} in suggested queries
                suggested = [
                    q.format(subject=subject)
                    for q in section_def.get("suggested_queries", [])
                ]

                sections.append(SectionPlan(
                    title=section_def["title"],
                    description=section_def.get("description", ""),
                    source_preference=source_pref,
                    required=section_def.get("required", True),
                    max_tokens=section_def.get("max_tokens", 800),
                    suggested_queries=suggested,
                    depends_on=section_def.get("depends_on", []),
                ))

            plan = ReportPlan(
                template_id=template_id,
                template_name=template.get("name", template_id),
                subject=subject,
                sections=sections,
                collections=collections or [],
            )

        # v6.4.0: Validate plan has sections
        if not plan.sections:
            logger.error(f"[REPORT] Plan produced 0 sections for template '{plan.template_id}'")
            raise ValueError(f"Report plan has 0 sections (template: {plan.template_id})")

        # v5.0.4: Record static planning metric if dynamic was not used
        if not (use_dynamic and plan.template_id == "dynamic"):
            planning_type = "fallback" if use_dynamic else "static"
            report_metrics.record_planning(time.time() - planning_start, planning_type)

        # Create session - Start in AWAITING_APPROVAL since plan is already ready
        # v2.4.1 FIX: Previously started in PLANNING which caused approval commands to be ignored
        # v2.5.1: Get artifact_type and artifact_formats from template
        artifact_type = template.get("artifact_type", "report") if template else "report"
        artifact_formats = template.get("artifact_formats", ["docx"]) if template else ["docx"]

        session = ReportSession(
            session_id=session_id,
            user_id=user_id,
            client_id=client_id,
            state=ReportState.AWAITING_APPROVAL,  # FIX: Plan is ready, await user approval
            plan=plan,
            conversation_id=conversation_id,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=self.PLANNING_TTL)).isoformat(),
            metadata={
                "original_query": query,
                "template_auto_selected": preferred_template is None,
                "dynamic_planning_used": use_dynamic and plan.template_id == "dynamic",
                "planner_version": "v2.4" if use_dynamic else "v2.3",
                "artifact_type": artifact_type,  # v2.5.1
                "artifact_formats": artifact_formats,  # v2.5.1
                "execution_mode": execution_mode,  # v5.1.0
                "user_roles": user_roles or ["user"],  # v6.3.3: Persist for swarm ctx
            },
        )

        # Persist to Redis
        await self._save_session(session)

        logger.debug(
            "[COLLECTIONS] start_session",
            extra={"collections": collections, "plan_collections": plan.collections},
        )

        logger.info(
            f"Report session started: {session_id}",
            extra={
                "user_id": user_id,
                "template_id": plan.template_id,
                "sections_count": len(plan.sections),
                "dynamic_planning": use_dynamic,
                "collections": plan.collections,  # DEBUG
            },
        )

        return session

    async def get_session(self, session_id: str) -> Optional[ReportSession]:
        """Get session by ID from Redis."""
        key = self._get_key(self.SESSION_KEY_PATTERN, session_id=session_id)
        data = await self.redis.get(key)

        if data:
            parsed_data = json.loads(data)
            plan_data = parsed_data.get("plan", {})
            logger.debug(
                "get_session: loaded session",
                extra={"session_id": session_id, "collections": plan_data.get("collections")},
            )
            session = ReportSession.from_dict(parsed_data)
            logger.debug(
                "get_session: deserialized",
                extra={
                    "session_id": session_id,
                    "collections": session.plan.collections if session.plan else None,
                },
            )
            return session
        return None

    async def get_active_session(
        self,
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> Optional[ReportSession]:
        """
        Get user's active session, optionally filtered by conversation.

        Returns None if no active session or session expired.
        """
        # Check user's session index
        user_key = self._get_key(self.USER_SESSIONS_KEY_PATTERN, user_id=user_id)
        session_ids = await self.redis.smembers(user_key)

        if not session_ids:
            return None

        expired_ids = []
        active_session = None

        for sid in session_ids:
            sid_str = sid.decode("utf-8") if isinstance(sid, bytes) else sid
            session = await self.get_session(sid_str)

            if session is None:
                # Session key expired in Redis — mark for cleanup
                expired_ids.append(sid)
                continue

            if session.state not in [ReportState.COMPLETED, ReportState.CANCELLED]:
                if conversation_id is None or session.conversation_id == conversation_id:
                    if datetime.fromisoformat(session.expires_at) > datetime.now(timezone.utc):
                        active_session = session
                        break  # v6.3.3: Early return — no need to load all sessions

        # v6.3.3: Inline cleanup — remove expired session refs from user set
        if expired_ids:
            try:
                await self.redis.srem(user_key, *expired_ids)
                logger.debug(
                    f"[REPORT] Cleaned {len(expired_ids)} expired session refs for user {user_id[:8]}"
                )
            except Exception as e:
                logger.warning(f"[REPORT] Session cleanup failed: {e}")

        return active_session

    async def _save_session(self, session: ReportSession) -> None:
        """Save session to Redis."""
        session.updated_at = datetime.now(timezone.utc).isoformat()

        key = self._get_key(self.SESSION_KEY_PATTERN, session_id=session.session_id)
        user_key = self._get_key(self.USER_SESSIONS_KEY_PATTERN, user_id=session.user_id)

        # Calculate TTL based on state
        ttl = self._get_ttl_for_state(session.state)
        session.expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()

        logger.debug(
            "_save_session: saving",
            extra={
                "session_id": session.session_id,
                "state": session.state.value,
                "collections": session.plan.collections if session.plan else None,
            },
        )

        # Save session data
        session_dict = session.to_dict()
        await self.redis.setex(key, ttl, json.dumps(session_dict))

        # Add to user's session index
        await self.redis.sadd(user_key, session.session_id)
        await self.redis.expire(user_key, ttl * 2)  # Index lives longer

    def _get_ttl_for_state(self, state: ReportState) -> int:
        """Get appropriate TTL based on session state."""
        state_ttls = {
            ReportState.PLANNING: self.PLANNING_TTL,
            ReportState.AWAITING_APPROVAL: self.PLANNING_TTL,
            ReportState.RESEARCHING: self.RESEARCHING_TTL,
            ReportState.WRITING: self.RESEARCHING_TTL,
            ReportState.REVIEW: self.REVIEW_TTL,
            ReportState.COMPLETED: 300,  # 5 minutes after completion
            ReportState.CANCELLED: 60,   # 1 minute after cancellation
        }
        return state_ttls.get(state, self.DEFAULT_SESSION_TTL)

    # =========================================================================
    # STATE TRANSITIONS
    # =========================================================================

    async def transition_state(
        self,
        session: ReportSession,
        trigger: str,
    ) -> ReportSession:
        """
        Execute state transition based on trigger.

        Args:
            session: Current session
            trigger: Transition trigger (approve, modify, cancel, etc.)

        Returns:
            Updated session with new state

        Raises:
            ValueError: If transition is not valid for current state
        """
        transitions = self._state_machine_config.get("transitions", {})
        valid_transitions = transitions.get(session.state.value.upper(), [])

        # Find matching transition
        target_state = None
        for trans in valid_transitions:
            if trans["trigger"] == trigger:
                target_state = ReportState(trans["target"].lower())
                break

        if target_state is None:
            raise ValueError(
                f"Invalid transition: '{trigger}' from state '{session.state.value}'. "
                f"Valid triggers: {[t['trigger'] for t in valid_transitions]}"
            )

        old_state = session.state
        session.state = target_state

        logger.info(
            f"State transition: {old_state.value} -> {target_state.value}",
            extra={
                "session_id": session.session_id,
                "trigger": trigger,
            },
        )

        await self._save_session(session)
        return session

    async def process_input(
        self,
        session_id: str,
        user_input: str,
    ) -> Dict[str, Any]:
        """
        Process user input based on current session state.

        Args:
            session_id: Session identifier
            user_input: User's message/feedback

        Returns:
            Dict with response message and updated state
        """
        session = await self.get_session(session_id)
        if not session:
            return {
                "error": "session_not_found",
                "message": "Sessione non trovata. Vuoi iniziare un nuovo report?",
            }

        # Handle based on current state
        if session.state == ReportState.PLANNING:
            return await self._handle_planning_input(session, user_input)

        elif session.state == ReportState.AWAITING_APPROVAL:
            return await self._handle_approval_input(session, user_input)

        elif session.state == ReportState.REVIEW:
            return await self._handle_review_input(session, user_input)

        elif session.state == ReportState.AWAITING_ENRICHMENT_CONFIG:
            return {
                "message": (
                    "La sessione è in attesa della configurazione enrichment. "
                    "Usa l'endpoint /api/user/reports/{id}/configure-enrichment "
                    "per configurare le opzioni di ricerca avanzata."
                ),
                "state": session.state.value,
                "session_id": session.session_id,
                "requires_action": "enrichment_config",
            }

        elif session.state.value in ("researching", "writing"):
            return {
                "message": (
                    f"Il report è in fase di {session.state.value}. "
                    "L'elaborazione è in corso, attendi il completamento."
                ),
                "state": session.state.value,
                "session_id": session.session_id,
                "in_progress": True,
            }

        else:
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "message": f"Sessione in stato '{session.state.value}'. Elaborazione in corso...",
            }

    async def _handle_planning_input(
        self,
        session: ReportSession,
        user_input: str,
    ) -> Dict[str, Any]:
        """
        Handle input during PLANNING state.

        v2.4.1 FIX: This now acts as a safety net. If the user sends approval
        commands while in PLANNING state (which shouldn't normally happen),
        handle them correctly instead of just transitioning and showing the plan again.
        """
        # Safety net: Check if this is an approval command
        approval_patterns = [
            "sì", "si", "ok", "procedi", "vai", "conferma", "approva",
            "va bene", "esegui", "inizia", "avvia", "yes", "proceed",
            "go", "confirm", "start", "execute", "approved",
        ]

        if session.plan and self._matches_pattern(user_input, approval_patterns):
            # User is trying to approve - transition directly to approval handler
            session = await self.transition_state(session, "plan_ready")
            # Now call approval handler
            return await self._handle_approval_input(session, user_input)

        # Default: Transition to AWAITING_APPROVAL and show plan
        session = await self.transition_state(session, "plan_ready")

        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "message": session.plan.get_proposal_text() if session.plan else "",
            "action": "awaiting_approval",
        }

    def _matches_pattern(self, text: str, patterns: list) -> bool:
        """
        Check if text matches any pattern using word boundary matching.
        v2.4.2 FIX: Use regex word boundaries for ALL patterns to avoid false positives.

        Examples of false positives prevented:
        - "sicurezza" should NOT match "si"
        - "appropriato" should NOT match "approva"
        - "approvazione" should NOT match "approva"
        """
        import re
        text_lower = text.lower().strip()

        for pattern in patterns:
            # v2.4.2: Use word boundary for ALL patterns (not just short ones)
            # This prevents "appropriato" from matching "approva"
            if re.search(rf'\b{re.escape(pattern)}\b', text_lower):
                return True
        return False

    async def _handle_approval_input(
        self,
        session: ReportSession,
        user_input: str,
    ) -> Dict[str, Any]:
        """Handle input during AWAITING_APPROVAL state."""
        input_lower = user_input.lower().strip()

        # Check for approval patterns (v2.4.1: Expanded for natural language)
        approval_patterns = [
            # Italian
            "sì", "si", "ok", "procedi", "vai", "conferma", "approva",
            "va bene", "esegui", "inizia", "avvia", "parti", "fallo",
            "approve", "approvato", "perfetto", "ottimo",
            # English
            "yes", "proceed", "go", "confirm", "start", "execute",
            "do it", "approved", "perfect", "great", "fine",
        ]

        cancel_patterns = [
            "no", "annulla", "cancel", "stop", "esci", "exit",
            "abort", "termina", "interrompi", "basta",
        ]

        if self._matches_pattern(user_input, approval_patterns):
            session = await self.transition_state(session, "approve")

            # v2.4: Trigger Swarm Execution if available
            if self._swarm_executor and session.plan:
                # Start swarm execution asynchronously
                import asyncio
                _swarm_task = asyncio.create_task(
                    self._execute_swarm(session),
                    name=f"swarm_{session.session_id[:8]}",
                )
                self._active_swarm_tasks[session.session_id] = _swarm_task
                _swarm_task.add_done_callback(
                    lambda t, sid=session.session_id: self._swarm_task_done(t, sid)
                )
                return {
                    "session_id": session.session_id,
                    "state": session.state.value,
                    "message": "Piano approvato! Swarm avviato - ricerca parallela in corso...",
                    "action": "swarm_started",
                    "swarm_enabled": True,
                }
            else:
                return {
                    "session_id": session.session_id,
                    "state": session.state.value,
                    "message": "Piano approvato! Inizio la ricerca...",
                    "action": "start_research",
                    "swarm_enabled": False,
                }

        elif self._matches_pattern(user_input, cancel_patterns):
            session = await self.transition_state(session, "cancel")
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "message": "Report annullato.",
                "action": "cancelled",
            }

        else:
            # User wants modifications - regenerate plan with user feedback
            if session.plan and self._dynamic_planner:
                try:
                    old_titles = {s.title for s in session.plan.sections}
                    old_count = len(session.plan.sections)

                    logger.info(f"[SESSION] Refining plan with user feedback: '{user_input[:50]}...'")
                    refined_plan = await self._dynamic_planner.refine_plan(
                        plan=session.plan,
                        user_feedback=user_input,
                    )
                    session.plan = refined_plan
                    session.updated_at = datetime.now(timezone.utc).isoformat()

                    # Save updated session to Redis
                    await self._save_session(session)

                    # Compute diff for informative message
                    new_titles = {s.title for s in session.plan.sections}
                    new_count = len(session.plan.sections)
                    added = new_titles - old_titles
                    removed = old_titles - new_titles

                    diff_parts = []
                    if added:
                        diff_parts.append(f"+{len(added)} ({', '.join(added)})")
                    if removed:
                        diff_parts.append(f"-{len(removed)} ({', '.join(removed)})")
                    if not added and not removed and new_count == old_count:
                        diff_parts.append("sezioni modificate")

                    diff_summary = "; ".join(diff_parts) if diff_parts else "piano aggiornato"
                    message = f"Piano aggiornato: {diff_summary}. Totale: {new_count} sezioni."

                    # Return updated plan preview
                    return {
                        "session_id": session.session_id,
                        "state": session.state.value,  # Stay in AWAITING_APPROVAL
                        "message": message,
                        "action": "plan_modified",
                        "modification": user_input,
                        "current_plan_preview": session.plan.get_proposal_text() if session.plan else "",
                        "sections": [
                            {
                                "title": s.title,
                                "description": s.description,
                                "source_preference": s.source_preference.value,
                                "required": s.required,
                                "max_tokens": s.max_tokens,
                                "suggested_queries": s.suggested_queries,
                            }
                            for s in session.plan.sections
                        ] if session.plan else [],
                    }
                except Exception as e:
                    logger.warning(f"[SESSION] Plan refinement failed: {e}")
                    # Fall back to recording modification without regeneration

            # Fallback: no dynamic planner or refinement failed
            if session.plan:
                session.plan.user_modifications.append(user_input)
            session = await self.transition_state(session, "modify")
            # v6.3.3: Removed redundant _save_session — transition_state already saves
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "message": f"Ho registrato la modifica: '{user_input}'. Approva per procedere.",
                "action": "modify_recorded",
                "modification": user_input,
            }

    async def _handle_review_input(
        self,
        session: ReportSession,
        user_input: str,
    ) -> Dict[str, Any]:
        """Handle input during REVIEW state."""
        accept_patterns = ["ok", "bene", "perfetto", "accetta", "accept", "good", "fine"]
        revise_patterns = ["rivedi", "revise", "modifica", "cambia", "change"]
        expand_patterns = ["espandi", "expand", "approfondisci", "more", "altro"]

        if self._matches_pattern(user_input, accept_patterns):
            session = await self.transition_state(session, "accept")
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "message": "Report finalizzato!",
                "action": "completed",
            }

        elif self._matches_pattern(user_input, expand_patterns):
            session = await self.transition_state(session, "expand")
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "message": "Espando la ricerca...",
                "action": "expand_research",
            }

        elif self._matches_pattern(user_input, revise_patterns):
            session = await self.transition_state(session, "revise")
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "message": "Rivedo il contenuto...",
                "action": "revise_draft",
            }

        else:
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "message": "Rispondi 'ok' per accettare, 'espandi' per approfondire, o 'rivedi' per modificare.",
            }

    # =========================================================================
    # TEMPLATE SELECTION
    # =========================================================================

    def _select_template_with_score(self, query: str) -> Optional[tuple]:
        """
        Auto-select template and return (template_id, score).
        Returns None if no template scores above 0.
        """
        import re

        query_lower = query.lower()
        scores: Dict[str, float] = {}

        # Check selection rules
        for rule in self._selection_rules:
            pattern = rule.get("pattern", "")
            if re.search(pattern, query_lower, re.IGNORECASE):
                tid = rule["template"]
                boost = rule.get("confidence_boost", 0.2)
                scores[tid] = scores.get(tid, 0) + boost

        # Check template keywords
        for tid, tmpl in self._templates.items():
            for kw in tmpl.get("keywords", []):
                if kw.lower() in query_lower:
                    scores[tid] = scores.get(tid, 0) + 0.15

        if scores:
            best = max(scores.items(), key=lambda x: x[1])
            return best
        return None

    def _select_template(self, query: str) -> str:
        """
        Auto-select template based on query patterns.

        Returns template_id with highest confidence match.
        """
        result = self._select_template_with_score(query)
        if result:
            logger.debug(f"Template auto-selected: {result[0]} (score: {result[1]:.2f})")
            return result[0]
        return "research_brief"

    # =========================================================================
    # v5.1.0 G2: CROSS-SECTION COHERENCE CHECK
    # =========================================================================

    # v5.1.1: Coherence thresholds — calibrated for domain text with shared terminology
    # v6.8.5: raised to 0.70 (was 0.55) — 0.55 was too permissive
    COHERENCE_THRESHOLD = 0.70

    @staticmethod
    def _coherence_level(similarity: float) -> str:
        """Classify similarity into semantic interpretation levels."""
        if similarity >= 0.70:
            return "probable_redundancy"
        elif similarity >= 0.55:
            return "thematic_overlap"
        return "ok"

    @staticmethod
    def _check_coherence(result, threshold: float = 0.70) -> Dict[str, Any]:
        """
        Detect redundancy between report sections using word-frequency cosine similarity.

        Uses pure-Python word vectors (no external dependencies, no GPU).
        v6.8.5: threshold raised to 0.70 (from 0.55), 3-level semantic interpretation.

        Returns a dict with flagged_pairs, all_pairs, and max_similarity.
        """
        from collections import Counter
        import math

        section_texts = []
        section_titles = []
        for section in result.sections:
            if section.status == "success" and section.content:
                section_texts.append(section.content)
                section_titles.append(section.section_title)

        if len(section_texts) < 2:
            return {"flagged_pairs": [], "all_pairs": [], "max_similarity": 0.0, "coherence_ok": True}
        # v6.4.1: Skip coherence check for short sections (false positive prone)
        if any(len(t.split()) < 50 for t in section_texts):
            logger.debug("[REPORT] Skipping coherence check — section too short for reliable cosine")
            return {"flagged_pairs": [], "all_pairs": [], "max_similarity": 0.0, "coherence_ok": True, "skipped": "short_sections"}

        def tokenize(text: str) -> Counter:
            words = [w.lower() for w in text.split() if len(w) >= 4 and w.isalpha()]
            return Counter(words)

        vectors = [tokenize(t) for t in section_texts]

        def cosine_sim(a: Counter, b: Counter) -> float:
            common = set(a.keys()) & set(b.keys())
            if not common:
                return 0.0
            dot = sum(a[k] * b[k] for k in common)
            norm_a = math.sqrt(sum(v * v for v in a.values()))
            norm_b = math.sqrt(sum(v * v for v in b.values()))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        flagged_pairs = []
        all_pairs = []
        max_sim = 0.0

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sim = cosine_sim(vectors[i], vectors[j])
                if sim > max_sim:
                    max_sim = sim
                level = ReportSessionManager._coherence_level(sim)
                pair = {
                    "section_a": section_titles[i],
                    "section_b": section_titles[j],
                    "similarity": round(sim, 3),
                    "level": level,
                }
                all_pairs.append(pair)
                if sim >= threshold:
                    flagged_pairs.append(pair)

        return {
            "flagged_pairs": flagged_pairs,
            "all_pairs": all_pairs,
            "max_similarity": round(max_sim, 3),
        }

    # =========================================================================
    # v5.1.0 G3: OUTPUT CLASSIFICATION
    # =========================================================================

    @staticmethod
    def _classify_output(result, session) -> Dict[str, Any]:
        """
        Classify the generated output based on quality scores and execution mode.

        Returns a dict with label, classification_reason, avg_quality, and grade.
        """
        execution_mode = session.metadata.get("execution_mode", "report")

        # Collect quality scores from successful sections
        scores = []
        for section in result.sections:
            if section.status == "success":
                qs = section.metadata.get("quality_score", 0.0)
                scores.append(qs)

        avg_quality = sum(scores) / len(scores) if scores else 0.0
        succeeded = result.sections_succeeded
        failed = result.sections_failed
        total = succeeded + failed

        # Mode-specific classification
        # v5.1.2: Updated labels for insight/exploratory modes
        if execution_mode == "insight":
            label = "analytical_output"
            reason = (
                f"Modalita' insight: {succeeded}/{total} sezioni completate, "
                f"qualita' media {avg_quality:.2f}"
            )
        elif execution_mode == "exploratory":
            label = "exploratory_memo"
            reason = (
                f"Modalita' esplorativa: {succeeded}/{total} sezioni completate, "
                f"qualita' media {avg_quality:.2f}"
            )
        else:
            # Report mode: tiered classification
            if failed > 0:
                label = "internal_draft"
                reason = (
                    f"{failed}/{total} sezioni fallite — output incompleto, "
                    f"qualita' media {avg_quality:.2f}"
                )
            elif avg_quality >= 0.70:
                label = "reviewed_draft"
                reason = (
                    f"Tutte le sezioni completate, qualita' media {avg_quality:.2f} >= 0.70 — "
                    f"draft di qualita' revisionabile"
                )
            elif avg_quality >= 0.45:
                label = "preliminary_draft"
                reason = (
                    f"Tutte le sezioni completate, qualita' media {avg_quality:.2f} — "
                    f"draft preliminare, revisione consigliata"
                )
            else:
                label = "internal_draft"
                reason = (
                    f"Qualita' media {avg_quality:.2f} < 0.45 — "
                    f"uso interno, richiede revisione significativa"
                )

        return {
            "label": label,
            "classification_reason": reason,
            "avg_quality": round(avg_quality, 3),
            "classification_confidence": round(avg_quality, 3),
            "execution_mode": execution_mode,
        }

    # =========================================================================
    # v5.1.0 G4: SEMANTIC VALIDATION
    # =========================================================================

    _SEMANTIC_VALIDATOR_PROMPT = """\
Sei un revisore editoriale scientifico. Analizza il seguente report e identifica problemi strutturali, di coerenza o di qualita'.

Il report ha {num_sections} sezioni. Di seguito trovi il contenuto completo o un riassunto strutturato.

Per ogni problema, rispondi con una riga nel formato:
TYPE | SEVERITY | SECTION | DETAIL

Dove:
- TYPE: uno tra (fabricated_citation, unsupported_claim, redundancy, contradiction, incomplete, low_quality, structural)
- SEVERITY: uno tra (critical, major, minor)
- SECTION: titolo della sezione interessata (o "general" se trasversale)
- DETAIL: breve descrizione del problema

Se non trovi problemi, rispondi solo: NESSUN_PROBLEMA

Report da analizzare:
---
{draft}
---"""

    @staticmethod
    def _build_reviewer_input(result, max_chars: int = 12000) -> str:
        """
        v5.1.1: Build structured reviewer input using sliding semantic window.

        If full draft fits within max_chars, use it directly.
        Otherwise, build a summary: section title + first 2 sentences + last sentence
        for each section, preserving structural overview without raw truncation.
        """
        full = result.full_draft
        if len(full) <= max_chars:
            return full

        # Build structured summary
        parts = []
        for section in result.sections:
            if section.status != "success" or not section.content:
                parts.append(f"## {section.section_title}\n[Sezione fallita o vuota]")
                continue

            content = section.content.strip()
            # v6.4.1: Better sentence split — respect abbreviations and decimals
        import re as _re
        sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ])', content.replace("\n", " ")) if s.strip()]
        if not sentences:
            sentences = [content.replace("\n", " ")]  # Fallback: entire content

            if len(sentences) <= 5:
                # Short section — include fully
                parts.append(f"## {section.section_title}\n{content}")
            else:
                # Extract opening (first 3 sentences) + closing (last 2 sentences)
                opening = ". ".join(sentences[:3]) + "."
                closing = ". ".join(sentences[-2:]) + "."
                parts.append(
                    f"## {section.section_title}\n"
                    f"{opening}\n[...]\n{closing}"
                )

        return "\n\n".join(parts)

    @staticmethod
    def _build_insight_reviewer_input(result, max_chars: int = 12000) -> str:
        """
        v5.1.2: Build reviewer input for insight mode (structured JSON).

        Formats evidence matrices and reasoning output for semantic validation.
        """
        parts = []

        # Extract evidence and reasoning from metadata
        evidence = result.metadata.get("evidence_matrices", {})
        reasoning = result.metadata.get("reasoning_output", {})

        if evidence:
            parts.append("## Matrici Evidenze")
            for section_title, matrix in evidence.items():
                entries = matrix.get("entries", [])
                parts.append(f"### {section_title} ({len(entries)} entries)")
                for entry in entries[:5]:  # Cap for reviewer
                    parts.append(
                        f"- [{entry.get('source_index', '?')}] "
                        f"{entry.get('condition', '')}: {entry.get('outcomes', '')} "
                        f"(forza: {entry.get('evidence_strength', '?')})"
                    )

        if reasoning:
            parts.append("\n## Analisi Ragionamento")
            for key, label in [
                ("diagnostic_patterns", "Pattern diagnostici"),
                ("emerging_approaches", "Approcci emergenti"),
                ("evidence_gaps", "Lacune evidenze"),
            ]:
                items = reasoning.get(key, [])
                if items:
                    parts.append(f"### {label}")
                    for item in items:
                        parts.append(f"- {item}")

        text = "\n".join(parts)
        return text[:max_chars] if len(text) > max_chars else text

    async def _semantic_validate(
        self, draft: str, session
    ) -> List[Dict[str, Any]]:
        """
        Use the main LLM (Grok) to review the draft and identify issues.

        v5.1.1: Uses sliding semantic window instead of raw truncation.
        Returns a list of structured flags [{type, severity, section, detail}].
        """
        if not self._llm_module:
            return []

        # v5.1.1: Build structured input instead of raw truncation
        # We need the result object — reconstruct from draft sections in session
        # The caller passes full_draft as `draft`, but we need the result object.
        # Since this is called from _execute_swarm which has the result, we accept
        # the pre-built reviewer_input or fall back to truncation.
        reviewer_input = draft  # Caller may pass pre-built input

        # v6.4.1: Count only sections with content for reviewer
        num_sections = len([
            s for s in (session.plan.sections if session.plan else [])
            if getattr(s, 'status', None) == "success"
        ])

        prompt = self._SEMANTIC_VALIDATOR_PROMPT.format(
            draft=reviewer_input, num_sections=num_sections
        )

        result = await self._llm_module.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1000,
        )

        # Extract response text
        if isinstance(result, dict):
            text = result.get("text", result.get("response", ""))
        else:
            text = str(result)

        # Parse structured flags
        flags = []
        if "NESSUN_PROBLEMA" in text:
            return flags

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                flag_type = parts[0].lower().replace(" ", "_")
                severity = parts[1].lower().strip()
                if severity not in ("critical", "major", "minor"):
                    severity = "minor"
                flags.append({
                    "type": flag_type,
                    "severity": severity,
                    "section": parts[2],
                    "detail": parts[3],
                })

        return flags

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available templates for UI."""
        return [
            {
                "id": t["id"],
                "name": t["name"],
                "description": t.get("description", ""),
                "category": t.get("category", "general"),
                "sections_count": len(t.get("sections", [])),
            }
            for t in self._templates.values()
        ]

    async def cancel_session(self, session_id: str) -> bool:
        """Cancel an active session."""
        session = await self.get_session(session_id)
        if session:
            try:
                await self.transition_state(session, "cancel")
                return True
            except ValueError:
                # Already in terminal state
                pass
        return False

    async def cleanup_expired_sessions(self, user_id: str) -> int:
        """Clean up expired sessions for a user."""
        user_key = self._get_key(self.USER_SESSIONS_KEY_PATTERN, user_id=user_id)
        session_ids = await self.redis.smembers(user_key)

        cleaned = 0
        now = datetime.now(timezone.utc)

        for sid in session_ids:
            if isinstance(sid, bytes):
                sid = sid.decode("utf-8")

            session = await self.get_session(sid)
            if session is None:
                # Session key expired, remove from index
                await self.redis.srem(user_key, sid)
                cleaned += 1
            elif datetime.fromisoformat(session.expires_at) < now:
                # Session expired but key still exists
                key = self._get_key(self.SESSION_KEY_PATTERN, session_id=sid)
                await self.redis.delete(key)
                await self.redis.srem(user_key, sid)
                cleaned += 1

        return cleaned

    # =========================================================================
    # v6.3.3: SWARM TASK CALLBACKS
    # =========================================================================

    def _swarm_task_done(self, task: "asyncio.Task", session_id: str) -> None:
        """Callback invoked when a swarm task finishes (success or error)."""
        self._active_swarm_tasks.pop(session_id, None)

        if task.cancelled():
            logger.warning(f"[REPORT] Swarm task cancelled for session {session_id[:8]}")
            return

        exc = task.exception()
        if exc is not None:
            logger.error(
                f"[REPORT] Swarm task FAILED for session {session_id[:8]}: "
                f"{type(exc).__name__}: {exc}",
                exc_info=exc,
            )
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._mark_session_error(session_id, str(exc)))
            except Exception as inner_exc:
                logger.error(f"[REPORT] Failed to mark session error: {inner_exc}")
        else:
            logger.info(f"[REPORT] Swarm task completed for session {session_id[:8]}")

    async def _mark_session_error(self, session_id: str, error_msg: str) -> None:
        """Best-effort: mark session as failed in Redis."""
        try:
            session = await self.get_session(session_id)
            if session and session.state.value not in ("completed", "cancelled"):
                session.state = ReportState.CANCELLED
                session.metadata = session.metadata or {}
                session.metadata["error"] = error_msg[:500]
                session.metadata["error_at"] = datetime.now(timezone.utc).isoformat()
                await self._save_session(session)
                logger.info(
                    f"[REPORT] Session {session_id[:8]} marked as CANCELLED due to swarm error"
                )
        except Exception as e:
            logger.error(f"[REPORT] _mark_session_error failed: {e}")

    # =========================================================================
    # v2.4: SWARM EXECUTION
    # =========================================================================

    async def _execute_swarm(self, session: ReportSession) -> None:
        """
        Execute swarm (parallel research + drafting) for a session.

        This runs asynchronously in the background after user approval.
        Results are stored in Redis and session state is updated.

        Args:
            session: The approved report session
        """
        if not self._swarm_executor or not session.plan:
            logger.warning(f"[SWARM] Cannot execute - missing executor or plan")
            return

        session_id = session.session_id

        # v6.3.3: Extend TTL to prevent session expiry during long swarms
        try:
            _swarm_ttl = max(self.REVIEW_TTL, self.RESEARCHING_TTL * 3)
            _session_key = self._get_key(self.SESSION_KEY_PATTERN, session_id=session_id)
            await self.redis.expire(_session_key, _swarm_ttl)
            logger.info(f"[REPORT] Extended session TTL to {_swarm_ttl}s for swarm {session_id[:8]}")
        except Exception as e:
            logger.warning(f"[REPORT] Failed to extend session TTL: {e}")

        logger.debug(
            "[COLLECTIONS] _execute_swarm",
            extra={"session_id": session_id, "plan_collections": session.plan.collections},
        )

        logger.info(
            f"[SWARM] Starting execution for session {session_id}",
            extra={"sections": len(session.plan.sections)},
        )

        try:
            # Build minimal security context from session user_id
            # Required by rag_qdrant for collection access authorization
            from types import SimpleNamespace
            _stored_roles = (session.metadata or {}).get("user_roles", ["user"])
            _user = SimpleNamespace(
                user_id=session.user_id,
                client_id=session.client_id,
                roles=_stored_roles,
            )
            ctx = SimpleNamespace(user=_user)
            logger.debug(f"[REPORT] Swarm ctx for {session.user_id[:8]}: roles={_stored_roles}")

            # Execute the swarm
            # v2.6: Pass session_id for debug event streaming
            # v5.1.2: Pass execution_mode and planner LLM for M1+M2
            execution_mode = session.metadata.get("execution_mode", "report")
            result = await self._swarm_executor.execute_plan(
                plan=session.plan,
                ctx=ctx,
                conversation_id=session.conversation_id,
                session_id=session_id,  # v2.6: For debug events
                execution_mode=execution_mode,
                planner_llm_module=self._planner_llm_module or self._llm_module,  # v6.0.0: Use dedicated planner LLM if available
            )

            # v5.1.0 G3: Classify output before storing
            classification = self._classify_output(result, session)
            # v5.1.0 G2: Check cross-section coherence (redundancy detection)
            # v5.1.2: Skip coherence check for insight mode (structured JSON, not narrative)
            if execution_mode == "insight":
                coherence = {
                    "flagged_pairs": [], "all_pairs": [],
                    "max_similarity": 0.0, "skipped": True,
                }
            else:
                coherence = self._check_coherence(result)
                # v6.8.5: log warning when probable redundancy detected
                flagged = coherence.get("flagged_pairs", [])
                if flagged:
                    for pair in flagged:
                        level = pair.get("level", "")
                        if level == "probable_redundancy":
                            logger.warning(
                                "[REPORT-COHERENCE] Probable redundancy: '%s' ↔ '%s' (sim=%.3f)",
                                pair.get("section_a", "?"),
                                pair.get("section_b", "?"),
                                pair.get("similarity", 0),
                            )
                        else:
                            logger.info(
                                "[REPORT-COHERENCE] Overlap flagged: '%s' ↔ '%s' (sim=%.3f, level=%s)",
                                pair.get("section_a", "?"),
                                pair.get("section_b", "?"),
                                pair.get("similarity", 0),
                                level,
                            )

            # Store the result in Redis
            draft_key = self._get_key(self.DRAFT_KEY_PATTERN, session_id=session_id)
            draft_data = {
                "session_id": session_id,
                "status": "completed" if result.sections_failed == 0 else "partial",
                "plan_title": result.plan_title,
                "full_draft": result.full_draft,
                "sections": result.to_dict()["sections"],
                "metrics": {
                    "total_time_ms": result.total_time_ms,
                    "parallel_efficiency": result.parallel_efficiency,
                    "sections_succeeded": result.sections_succeeded,
                    "sections_failed": result.sections_failed,
                    "worker_provider": result.worker_provider,
                },
                "output_classification": classification,
                "coherence_check": coherence,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                # v5.1.2: Evidence abstraction + reasoning pass results
                "evidence_matrices": result.metadata.get("evidence_matrices"),
                "reasoning_output": result.metadata.get("reasoning_output"),
                "execution_mode": execution_mode,
            }
            await self.redis.setex(
                draft_key,
                self.REVIEW_TTL,
                json.dumps(draft_data),
            )

            # Update session state to WRITING -> REVIEW
            session = await self.get_session(session_id)
            if session:
                session = await self.transition_state(session, "research_complete")
                session = await self.transition_state(session, "draft_ready")

                # Store completion info in metadata
                session.metadata["swarm_result"] = {
                    "succeeded": result.sections_succeeded,
                    "failed": result.sections_failed,
                    "time_ms": result.total_time_ms,
                }

                # v5.1.0: Store classification and coherence in session metadata
                session.metadata["output_classification"] = classification
                session.metadata["coherence_check"] = coherence

                await self._save_session(session)

                # ============================================================
                # v5.1.0 G4: Semantic validation with Grok (non-blocking)
                # ============================================================
                if self._llm_module and result.sections_failed == 0:
                    try:
                        # v5.1.2: Mode-specific reviewer input
                        if execution_mode == "insight":
                            reviewer_input = self._build_insight_reviewer_input(result)
                        else:
                            # v5.1.1: Build structured reviewer input
                            reviewer_input = self._build_reviewer_input(result)
                        validation_flags = await self._semantic_validate(
                            reviewer_input, session
                        )
                        session.metadata["semantic_validation"] = validation_flags
                        # Also update draft_data in Redis
                        draft_data["semantic_validation"] = validation_flags
                        await self.redis.setex(
                            draft_key, self.REVIEW_TTL, json.dumps(draft_data)
                        )
                        await self._save_session(session)
                        logger.info(
                            f"[G4] Semantic validation: {len(validation_flags)} flags "
                            f"for session {session_id}"
                        )
                    except Exception as val_error:
                        logger.warning(
                            f"[G4] Semantic validation failed for {session_id}: {val_error}"
                        )
                        session.metadata["semantic_validation_error"] = str(val_error)
                        await self._save_session(session)

                # ============================================================
                # v2.5.1: Generate multi-format artifact after successful swarm
                # v5.1.2: Skip artifact generation for insight mode (JSON output)
                # ============================================================
                if self._artifact_manager and result.sections_failed == 0 and execution_mode != "insight":
                    try:
                        artifacts_generated = await self._generate_artifacts(
                            session=session,
                            session_id=session_id,
                            result=result,
                        )

                        if artifacts_generated:
                            logger.info(
                                f"[ARTIFACT] Generated {len(artifacts_generated)} artifact(s) "
                                f"for session {session_id}"
                            )

                    except Exception as artifact_error:
                        logger.warning(
                            f"[ARTIFACT] Failed to generate artifact for {session_id}: {artifact_error}"
                        )
                        # Don't fail the whole operation if artifact generation fails
                        session.metadata["artifact_error"] = str(artifact_error)
                        await self._save_session(session)

            # v5.0.4: Record swarm completion metrics
            status = "completed" if result.sections_failed == 0 else "partial"
            report_metrics.record_swarm_result(
                result.total_time_ms, result.sections_succeeded, result.sections_failed
            )
            report_metrics.record_session_completed(status)

            logger.info(
                f"[SWARM] Execution completed for session {session_id}",
                extra={
                    "succeeded": result.sections_succeeded,
                    "failed": result.sections_failed,
                    "time_ms": result.total_time_ms,
                },
            )

        except Exception as e:
            report_metrics.record_session_completed("error")
            logger.error(
                f"[SWARM] Execution failed for session {session_id}: {e}",
                exc_info=True,
            )

            # Update session with error
            session = await self.get_session(session_id)
            if session:
                session.metadata["swarm_error"] = str(e)
                await self._save_session(session)

    # =========================================================================
    # v2.5.1: MULTI-FORMAT ARTIFACT GENERATION
    # =========================================================================

    async def _generate_artifacts(
        self,
        session: ReportSession,
        session_id: str,
        result: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate artifacts in requested format(s) after swarm completion.

        v2.5.1: Supports multiple output formats based on artifact_type:
        - report (default): DOCX document
        - computo_metrico: XLSX with data extraction
        - capitolato: DOCX with template structure
        - presentazione: PPTX slides
        - data_export: CSV/XLSX tabular data
        - markdown: MD export

        Args:
            session: Current report session
            session_id: Session identifier
            result: Swarm execution result with draft content

        Returns:
            List of artifact metadata dicts
        """
        from ..renderers import (
            DocxRenderer,
            ExcelRenderer,
            CsvRenderer,
            PptxRenderer,
            MarkdownRenderer,
            RenderContext,
            OutputFormat,
        )

        artifacts = []
        artifact_type = session.metadata.get("artifact_type", "report")
        requested_formats = session.metadata.get("artifact_formats", ["docx"])

        logger.info(
            f"[ARTIFACT] Generating artifacts for session {session_id}: "
            f"type={artifact_type}, formats={requested_formats}"
        )

        # Build render context
        render_context = RenderContext(
            title=result.plan_title,
            content=result.full_draft,
            author="UBP Enterprise",
            session_id=session_id,
            user_id=session.user_id,
            blueprint_id=artifact_type,
            metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sections": result.sections_succeeded,
                "template_id": session.plan.template_id if session.plan else None,
            },
        )

        # Get format settings from session metadata
        format_settings = session.metadata.get("format_settings", {})
        render_context.format_settings = format_settings

        # v2.5.1: For data_extraction formats (xlsx, csv), try to extract structured data
        # from markdown tables or generate sample data for computo_metrico
        if artifact_type in ["computo_metrico", "data_export"] and any(f in requested_formats for f in ["xlsx", "csv"]):
            from ..renderers.base import DataExtractor
            import json as json_module
            import re

            extracted = None

            # Attempt 1: Search for JSON code block in content
            json_match = re.search(r'```json\n(.*?)\n```', result.full_draft, re.DOTALL)
            if json_match:
                try:
                    extracted = json_module.loads(json_match.group(1))
                    logger.info("[ARTIFACT] Extracted JSON from code block")
                except json_module.JSONDecodeError:
                    logger.warning("[ARTIFACT] Found JSON block but failed to parse")

            # Attempt 2: Try DataExtractor for inline JSON
            if not extracted:
                extracted = DataExtractor.extract_json(result.full_draft)
                if extracted:
                    logger.info("[ARTIFACT] Extracted JSON via DataExtractor")

            # Attempt 3: Parse markdown tables/bullets
            if not extracted:
                logger.info("[ARTIFACT] Trying markdown table/bullet extraction")
                extracted = self._extract_computo_from_markdown(result.full_draft)

            # v6.4.0: Short-circuit — if method 3 produced substantial data, skip LLM
            is_placeholder = isinstance(extracted, list) and len(extracted) == 1 and "da specificare" in str(extracted[0].get("description", "") if extracted else "")
            _skip_llm = extracted and not is_placeholder and len(str(extracted)) > 10
            if _skip_llm:
                logger.debug("[ARTIFACT] Data extracted via method 3, skipping LLM fallback")

            # Attempt 4: LLM fallback - convert markdown to JSON
            if not _skip_llm and (not extracted or is_placeholder):
                if self._llm_module:
                    logger.info("[ARTIFACT] Using LLM to extract structured data from markdown")
                    try:
                        # Build prompt with system instruction embedded
                        extraction_prompt = f"""[SISTEMA] Sei un estrattore dati. Output SOLO JSON array valido, senza testo aggiuntivo.

[ISTRUZIONI] Analizza il seguente testo e estrai le voci di computo metrico come JSON array.
Ogni voce deve avere: description (string), quantity (number), unit (string, es: m2, m3, kg, nr).

[TESTO DA ANALIZZARE]
{result.full_draft[:3000]}

[OUTPUT]
"""

                        llm_response = await self._llm_module.generate(
                            prompt=extraction_prompt,
                            max_tokens=2000,
                        )
                        logger.debug(f"[ARTIFACT] DEBUG LLM raw response: {llm_response}")

                        # Parse LLM response
                        if llm_response and llm_response.get("text"):
                            llm_text = llm_response["text"].strip()
                            # Clean up response
                            if llm_text.startswith("```"):
                                llm_text = re.sub(r'^```\w*\n?', '', llm_text)
                                llm_text = re.sub(r'\n?```$', '', llm_text)
                            # Try to find JSON array in response
                            json_start = llm_text.find('[')
                            json_end = llm_text.rfind(']') + 1
                            if json_start >= 0 and json_end > json_start:
                                llm_text = llm_text[json_start:json_end]
                            try:
                                extracted = json_module.loads(llm_text)
                                logger.info(f"[ARTIFACT] LLM extracted {len(extracted) if isinstance(extracted, list) else 1} items")
                            except json_module.JSONDecodeError as e:
                                logger.warning(f"[ARTIFACT] LLM response not valid JSON: {e}")
                    except Exception as e:
                        logger.warning(f"[ARTIFACT] LLM extraction failed: {e}")

            if extracted:
                render_context.extracted_data = extracted
                logger.info(f"[ARTIFACT] Final extraction: {len(extracted) if isinstance(extracted, list) else 1} data items")

        # Generate each requested format
        for fmt in requested_formats:
            try:
                renderer = self._get_renderer_for_format(fmt, artifact_type)
                if not renderer:
                    logger.warning(f"[ARTIFACT] No renderer for format: {fmt}")
                    continue

                # Render content
                render_result = renderer.render(render_context)

                if not render_result.success:
                    logger.warning(
                        f"[ARTIFACT] Render failed for {fmt}: {render_result.error}"
                    )
                    continue

                # Save artifact
                artifact_metadata = await self._artifact_manager.save_artifact(
                    content=render_result.content,
                    format=fmt,
                    session_id=session_id,
                    user_id=session.user_id,
                    title=result.plan_title,
                )

                artifact_info = {
                    "artifact_id": artifact_metadata.artifact_id,
                    "filename": artifact_metadata.filename,
                    "format": artifact_metadata.format,
                    "size_bytes": artifact_metadata.size_bytes,
                    "download_url": artifact_metadata.download_url,
                }
                artifacts.append(artifact_info)

                logger.info(
                    f"[ARTIFACT] Generated {fmt}: {artifact_metadata.filename} "
                    f"({artifact_metadata.size_bytes} bytes)"
                )

            except Exception as e:
                logger.error(f"[ARTIFACT] Error generating {fmt}: {e}", exc_info=True)
                continue

        # Store artifacts in session metadata
        if artifacts:
            # Keep backward compatibility: single artifact in "artifact" key
            session.metadata["artifact"] = artifacts[0]
            # v2.5.1: All artifacts in "artifacts" array
            session.metadata["artifacts"] = artifacts
            await self._save_session(session)

        return artifacts

    def _get_renderer_for_format(self, format_name: str, artifact_type: str):
        """
        Get the appropriate renderer for a format and artifact type.

        Args:
            format_name: Output format (docx, xlsx, csv, pptx, md)
            artifact_type: Blueprint type (report, computo_metrico, etc.)

        Returns:
            Renderer instance or None
        """
        from ..renderers import (
            DocxRenderer,
            ExcelRenderer,
            CsvRenderer,
            PptxRenderer,
            MarkdownRenderer,
        )

        format_lower = format_name.lower()

        # Format-specific renderers
        if format_lower == "docx":
            return DocxRenderer()
        elif format_lower == "xlsx":
            return ExcelRenderer()
        elif format_lower == "csv":
            return CsvRenderer()
        elif format_lower == "pptx":
            return PptxRenderer()
        elif format_lower == "md":
            return MarkdownRenderer()

        return None

    def _extract_computo_from_markdown(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract computo metrico items from markdown content.

        Parses markdown tables and bullet points to extract structured data
        for bill of quantities format.

        Args:
            content: Markdown content from report draft

        Returns:
            List of dicts with computo metrico structure
        """
        import re

        items = []

        # Try to parse markdown tables (| col1 | col2 | format)
        table_pattern = r'\|(.+)\|'
        table_rows = re.findall(table_pattern, content)

        # v6.4.0: Validate markdown table structure (header + separator + data)
        if len(table_rows) >= 2:
            separator = table_rows[1]
            if not re.match(r'^[\s|:\-]+$', separator):
                table_rows = []
                logger.debug("[REPORT] Skipped non-table pipe content (no separator row)")

        if len(table_rows) > 2:  # Has header + separator + data
            # Skip header and separator rows
            header_cells = [c.strip() for c in table_rows[0].split('|') if c.strip()]
            for row in table_rows[2:]:  # Skip header and separator
                cells = [c.strip() for c in row.split('|') if c.strip()]
                if len(cells) >= 2:
                    item = {}
                    for i, header in enumerate(header_cells):
                        if i < len(cells):
                            item[header.lower()] = cells[i]
                    if item:
                        items.append(item)

        # If no tables, try to parse bullet points with quantities
        if not items:
            # Pattern: - Description (qty unit) or - Description: qty unit
            bullet_pattern = r'[-*]\s+(.+?)(?:\((\d+(?:\.\d+)?)\s*([a-zA-Z²³]+)\)|:\s*(\d+(?:\.\d+)?)\s*([a-zA-Z²³]+))'
            matches = re.findall(bullet_pattern, content)

            for match in matches:
                description = match[0].strip()
                qty = match[1] or match[3]
                unit = match[2] or match[4]
                if description and qty:
                    items.append({
                        "description": description,
                        "quantity": float(qty),
                        "unit": unit or "nr",
                    })

        # If still no items, generate sample structure based on content analysis
        if not items:
            logger.warning("[ARTIFACT] Could not extract data from markdown, using placeholder")
            # Generate minimal placeholder for Excel
            items = [
                {"description": "Elemento da specificare", "quantity": 1, "unit": "nr"},
            ]

        logger.info(f"[ARTIFACT] Extracted {len(items)} items from markdown")
        return items

    async def get_report_draft(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the generated report draft for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict with draft content and metadata, or None if not ready
        """
        draft_key = self._get_key(self.DRAFT_KEY_PATTERN, session_id=session_id)
        data = await self.redis.get(draft_key)

        if data:
            return json.loads(data)
        return None

    async def get_swarm_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get the current swarm execution status for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict with status, progress, and any available results
        """
        session = await self.get_session(session_id)
        if not session:
            return {"status": "not_found", "message": "Session not found"}

        # Check if draft is ready
        draft = await self.get_report_draft(session_id)

        if draft:
            return {
                "status": "completed",
                "state": session.state.value,
                "draft_available": True,
                "metrics": draft.get("metrics", {}),
            }

        # Check for error
        if session.metadata.get("swarm_error"):
            return {
                "status": "error",
                "state": session.state.value,
                "error": session.metadata["swarm_error"],
            }

        # Still in progress
        return {
            "status": "in_progress",
            "state": session.state.value,
            "message": "Swarm execution in progress...",
        }
