"""
content_planner/adapter.py

Bridge Layer - Content planning operations.

Provides:
- Document structure planning
- Template management
- Microprompt generation
- Plan validation and modification

Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

from .providers import (
    # Enums
    SectionType,
    ContentType,
    SourcePreference,
    DocumentType,
    FormalityLevel,
    # Data classes
    StructuredPlan,
    SectionPlan,
    PlanConstraints,
    PlanMetadata,
    EnrichmentConfig,
    Microprompt,
    DocumentTemplate,
    TemplateMatch,
    ValidationResult,
    TokenEstimate,
    ResourceEstimate,
    # Managers
    TemplateManager,
    MicropromptEngine,
    PlanGenerator,
    PlanValidator,
)

logger = logging.getLogger(__name__)


class ContentPlannerAdapter:
    """
    Main adapter for content planning.
    
    Provides operations for:
    - Planning document structure
    - Managing templates
    - Generating microprompts
    - Validating and modifying plans
    """
    
    def __init__(
        self,
        module_path: Path,
        di_container: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        self.module_path = Path(module_path)
        self.di_container = di_container
        self.event_bus = event_bus
        
        # Components
        self._template_manager: Optional[TemplateManager] = None
        self._microprompt_engine: Optional[MicropromptEngine] = None
        self._plan_generator: Optional[PlanGenerator] = None
        self._validator: Optional[PlanValidator] = None
        
        # LLM module (resolved at runtime)
        self._llm_module: Optional[Any] = None
        
        # State
        self._initialized = False
        
        # Configuration
        self._default_language = "it"
        self._default_formality = FormalityLevel.PROFESSIONAL
        self._planning_model = os.getenv(
            "UBP_CONTENT_PLANNER__MODEL",
            "grok/grok-3-fast",  # HARDCODE-001 remediation: env-overridable default
        )
    
    # MCP-COMPAT: OperationContext helpers (ARCH-008)
    def _build_context_from_di(self) -> OperationContext:
        """Build OperationContext from DI — backward compatibility for REST path."""
        return OperationContext(
            client_id="default",
            user_id=None,
            session_id=None,
            source="rest",
        )

    def _normalize_ctx(self, ctx: Any) -> OperationContext:
        """Normalize any context format to OperationContext."""
        if ctx is None:
            return self._build_context_from_di()
        if isinstance(ctx, OperationContext):
            return ctx
        if hasattr(ctx, "user") and ctx.user:
            user_id = getattr(ctx.user, "user_id", None)
            roles = getattr(ctx.user, "roles", [])
            client_id = getattr(ctx.user, "client_id", "default")
            if not isinstance(roles, (list, tuple)):
                roles = []
            return OperationContext(
                client_id=str(client_id) if client_id else "default",
                user_id=str(user_id) if user_id else None,
                roles=list(roles),
                source="rest",
            )
        return self._build_context_from_di()
    
    # ========================================================================
    # Lifecycle Operations
    # ========================================================================
    
    async def initialize(
        self,
        templates_dir: Optional[str] = None,
        default_language: str = "it",
        default_formality: str = "professional",
        planning_model: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Initialize the content planner.
        
        Args:
            templates_dir: Optional directory with custom templates
            default_language: Default document language
            default_formality: Default formality level
            planning_model: LLM model for planning
        """
        self._default_language = default_language
        self._default_formality = FormalityLevel(default_formality)
        
        if planning_model:
            self._planning_model = planning_model
        
        # Initialize template manager
        templates_path = None
        if templates_dir:
            templates_path = Path(templates_dir)
        elif (self.module_path / "templates").exists():
            templates_path = self.module_path / "templates"
        
        self._template_manager = TemplateManager(templates_path=templates_path)

        # Initialize microprompt engine
        self._microprompt_engine = MicropromptEngine()
        
        # Initialize validator
        self._validator = PlanValidator()
        
        # Resolve LLM module via centralized provider resolver.
        # Role "enrichment": content_planner generates document structures (non-RAG,
        # non-chat, non-routing), same semantic class as advanced_report_generator.
        # Honors FALLBACK_ORDER + ProviderHealthMonitor; no hardcoded provider.
        if self.di_container:
            try:
                from ubp_enterprise_hybrid.backend.app.api.routers.dependencies import (
                    resolve_inference_for_role_from_container,
                )
                adapter, provider_name = resolve_inference_for_role_from_container(
                    self.di_container, role="enrichment"
                )
                if adapter is not None:
                    self._llm_module = adapter
                    logger.info(
                        "content_planner LLM resolved via resolver: role=enrichment provider=%s",
                        provider_name,
                    )
                else:
                    logger.info(
                        "content_planner: no LLM provider available (role=enrichment), "
                        "falling back to template-only planning"
                    )
            except Exception as e:
                logger.warning(f"Could not resolve LLM module via resolver: {e}")
        
        # Initialize plan generator
        self._plan_generator = PlanGenerator(
            template_manager=self._template_manager,
            microprompt_engine=self._microprompt_engine,
        )
        
        self._initialized = True
        
        logger.info("content_planner initialized")
        
        return {
            "status": "initialized",
            "module": "content_planner",
            "version": "1.0.0",
            "templates_loaded": len(self._template_manager.list_templates()),
            "llm_available": self._llm_module is not None,
        }
    
    async def shutdown(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Shutdown the content planner."""
        self._initialized = False
        return {"status": "shutdown"}
    
    async def health_check(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Health check."""
        return {
            "module": "content_planner",
            "version": "1.0.0",
            "status": "healthy" if self._initialized else "not_initialized",
            "templates_count": len(self._template_manager.list_templates()) if self._template_manager else 0,
            "llm_available": self._llm_module is not None,
        }
    
    # ========================================================================
    # Core Planning Operations
    # ========================================================================
    
    async def plan_structure(
        self,
        query: str,
        constraints: Optional[Dict[str, Any]] = None,
        template_id: Optional[str] = None,
        template_hints: Optional[Dict[str, Any]] = None,
        collections: Optional[List[str]] = None,
        auto_validate: bool = True,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a structured plan for a document.
        
        Args:
            query: User's document request
            constraints: Planning constraints (max_sections, max_tokens, etc.)
            template_id: Specific template to use
            template_hints: Hints for template selection
            collections: Collections for research
            auto_validate: Whether to validate plan automatically
        
        Returns:
            Dict with structured plan
        """
        if not self._initialized:
            await self.initialize()
        
        # Build constraints
        plan_constraints = self._build_constraints(constraints)
        
        # Generate plan
        plan = self._plan_generator.generate_plan(
            topic=query,
            document_type=constraints.get("document_type") if isinstance(constraints, dict) else
                          getattr(plan_constraints, "document_type", None),
            template_id=template_id,
            requirements=constraints,
            style_guide=template_hints,
        )
        
        # Validate if requested
        validation = None
        if auto_validate:
            validation = self._validator.validate(plan)
        
        return {
            "success": True,
            "plan": plan.to_dict(),
            "validation": validation.to_dict() if validation else None,
            "estimated_tokens": plan.estimated_tokens,
            "sections_count": len(plan.enabled_sections),
        }
    
    async def plan_presentation(
        self,
        query: str,
        max_slides: int = 20,
        style: str = "professional",
        constraints: Optional[Dict[str, Any]] = None,
        collections: Optional[List[str]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a plan specifically for presentations.
        
        Args:
            query: Presentation topic/request
            max_slides: Maximum number of slides
            style: Presentation style
            constraints: Additional constraints
            collections: Collections for research
        
        Returns:
            Dict with presentation plan
        """
        # Adapt constraints for presentation
        presentation_constraints = constraints or {}
        presentation_constraints.setdefault("max_sections", max_slides)
        presentation_constraints.setdefault("max_tokens_per_section", 200)
        presentation_constraints.setdefault("max_tokens_total", max_slides * 200)
        
        # Use presentation template hint
        template_hints = {
            "type": "presentation",
            "style": style,
        }
        
        return await self.plan_structure(
            query=query,
            constraints=presentation_constraints,
            template_hints=template_hints,
            collections=collections,
        )
    
    # ========================================================================
    # Section Operations
    # ========================================================================
    
    async def design_section(
        self,
        section_type: str,
        title: str,
        description: str = "",
        context: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Design a single section.
        
        Args:
            section_type: Type of section
            title: Section title
            description: Section description
            context: Additional context
        
        Returns:
            Dict with section design
        """
        if not self._initialized:
            await self.initialize()
        
        section_type_enum = SectionType(section_type) if section_type in [e.value for e in SectionType] else SectionType.CUSTOM
        
        section = SectionPlan(
            id=title.lower().replace(" ", "_")[:20],
            title=title,
            description=description,
            order=0,
            section_type=section_type_enum,
            content_type=ContentType.PROSE,
            source_preference=SourcePreference.RAG_FIRST,
            target_tokens=500,
        )
        
        # Generate microprompt
        context = context or {}
        context.setdefault("language", self._default_language)
        context.setdefault("formality", self._default_formality)
        
        microprompt = self._microprompt_engine.generate_microprompt(
            section=section,
            context=context,
        )
        section.microprompt = microprompt
        
        return {
            "success": True,
            "section": section.to_dict(),
        }
    
    async def add_section(
        self,
        plan: Union[Dict, StructuredPlan],
        section: Union[Dict, SectionPlan],
        position: Optional[int] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add a section to an existing plan.
        
        Args:
            plan: Existing plan
            section: Section to add
            position: Optional position (default: end)
        
        Returns:
            Dict with updated plan
        """
        # Convert if needed
        if isinstance(plan, dict):
            plan = self._resolve_plan(plan)
        if isinstance(section, dict):
            section = self._dict_to_section(section)
        
        # Determine position
        if position is None:
            position = len(plan.sections)
        
        # Update order values
        section.order = position
        for s in plan.sections:
            if s.order >= position:
                s.order += 1
        
        # Insert section
        plan.sections.insert(position, section)
        plan.metadata.updated_at = __import__("datetime").datetime.utcnow()
        plan.metadata.version += 1
        
        # Recalculate estimates
        plan.estimated_tokens = sum(s.target_tokens for s in plan.enabled_sections)
        plan.estimated_sections = len(plan.enabled_sections)
        
        return {
            "success": True,
            "plan": plan.to_dict(),
            "added_section_id": section.id,
        }
    
    async def remove_section(
        self,
        plan: Union[Dict, StructuredPlan],
        section_id: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Remove a section from a plan.
        
        Args:
            plan: Existing plan
            section_id: ID of section to remove
        
        Returns:
            Dict with updated plan
        """
        if isinstance(plan, dict):
            plan = self._resolve_plan(plan)
        
        # Find and remove section
        original_count = len(plan.sections)
        plan.sections = [s for s in plan.sections if s.id != section_id]
        
        if len(plan.sections) == original_count:
            return {
                "success": False,
                "error": f"Section '{section_id}' not found",
            }
        
        # Remove from dependencies
        for section in plan.sections:
            section.depends_on = [d for d in section.depends_on if d != section_id]
        
        # Update metadata
        plan.metadata.updated_at = __import__("datetime").datetime.utcnow()
        plan.metadata.version += 1
        
        # Recalculate
        plan.estimated_tokens = sum(s.target_tokens for s in plan.enabled_sections)
        plan.estimated_sections = len(plan.enabled_sections)
        
        return {
            "success": True,
            "plan": plan.to_dict(),
            "removed_section_id": section_id,
        }
    
    async def modify_section(
        self,
        plan: Union[Dict, StructuredPlan],
        section_id: str,
        changes: Dict[str, Any],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Modify a section in a plan.
        
        Args:
            plan: Existing plan
            section_id: ID of section to modify
            changes: Dict of changes to apply
        
        Returns:
            Dict with updated plan
        """
        if isinstance(plan, dict):
            plan = self._resolve_plan(plan)
        
        # Find section
        section = plan.get_section(section_id)
        if not section:
            return {
                "success": False,
                "error": f"Section '{section_id}' not found",
            }
        
        # Apply changes
        allowed_fields = [
            "title", "description", "enabled", "required",
            "min_tokens", "max_tokens", "target_tokens",
            "source_preference", "content_type", "interactive_review",
        ]
        
        for field, value in changes.items():
            if field in allowed_fields:
                if field == "source_preference" and isinstance(value, str):
                    value = SourcePreference(value)
                elif field == "content_type" and isinstance(value, str):
                    value = ContentType(value)
                setattr(section, field, value)
        
        # Regenerate microprompt if content changed significantly
        if any(f in changes for f in ["title", "description", "content_type"]):
            context = {
                "language": plan.language,
                "formality": plan.constraints.formality_level,
                "document_title": plan.title,
            }
            section.microprompt = self._microprompt_engine.generate_microprompt(
                section=section,
                context=context,
            )
        
        # Update metadata
        plan.metadata.updated_at = __import__("datetime").datetime.utcnow()
        plan.metadata.version += 1
        
        return {
            "success": True,
            "plan": plan.to_dict(),
            "modified_section_id": section_id,
        }
    
    async def reorder_sections(
        self,
        plan: Union[Dict, StructuredPlan],
        new_order: List[str],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Reorder sections in a plan.
        
        Args:
            plan: Existing plan
            new_order: List of section IDs in new order
        
        Returns:
            Dict with updated plan
        """
        if isinstance(plan, dict):
            plan = self._resolve_plan(plan)
        
        # Validate all IDs exist
        current_ids = {s.id for s in plan.sections}
        if set(new_order) != current_ids:
            return {
                "success": False,
                "error": "new_order must contain exactly the same section IDs",
            }
        
        # Create mapping and reorder
        section_map = {s.id: s for s in plan.sections}
        plan.sections = [section_map[sid] for sid in new_order]
        
        # Update order values
        for i, section in enumerate(plan.sections):
            section.order = i
        
        # Update metadata
        plan.metadata.updated_at = __import__("datetime").datetime.utcnow()
        plan.metadata.version += 1
        
        return {
            "success": True,
            "plan": plan.to_dict(),
        }
    
    # ========================================================================
    # Microprompt Operations
    # ========================================================================
    
    async def generate_microprompt(
        self,
        section: Union[Dict, SectionPlan],
        context: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate microprompt for a section.
        
        Args:
            section: Section plan
            context: Additional context
        
        Returns:
            Dict with generated microprompt
        """
        if not self._initialized:
            await self.initialize()
        
        if isinstance(section, dict):
            section = self._dict_to_section(section)
        
        context = context or {}
        context.setdefault("language", self._default_language)
        context.setdefault("formality", self._default_formality)
        
        microprompt = self._microprompt_engine.generate_microprompt(
            section=section,
            context=context,
        )
        
        return {
            "success": True,
            "microprompt": microprompt.to_dict(),
            "system_prompt": microprompt.to_system_prompt(),
            "generation_prompt": microprompt.generation_prompt,
        }
    
    async def generate_microprompts_batch(
        self,
        sections: List[Union[Dict, SectionPlan]],
        context: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate microprompts for multiple sections.
        
        Args:
            sections: List of section plans
            context: Shared context
        
        Returns:
            Dict with microprompts by section ID
        """
        if not self._initialized:
            await self.initialize()
        
        sections = [
            self._dict_to_section(s) if isinstance(s, dict) else s
            for s in sections
        ]
        
        context = context or {}
        context.setdefault("language", self._default_language)
        context.setdefault("formality", self._default_formality)
        
        microprompts = self._microprompt_engine.generate_batch(
            sections=sections,
            context=context,
        )
        
        return {
            "success": True,
            "microprompts": {
                sid: mp.to_dict()
                for sid, mp in microprompts.items()
            },
        }
    
    # ========================================================================
    # Template Operations
    # ========================================================================
    
    async def list_templates(
        self,
        category: Optional[str] = None,
        document_type: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        List available templates.
        
        Args:
            category: Filter by category
            document_type: Filter by document type
        
        Returns:
            Dict with templates list
        """
        if not self._initialized:
            await self.initialize()
        
        doc_type = DocumentType(document_type) if document_type else None
        templates = self._template_manager.list_templates(
            category=category,
            document_type=doc_type,
        )
        
        return {
            "success": True,
            "templates": [t.to_dict() for t in templates],
            "count": len(templates),
            "categories": self._template_manager.get_categories(),
        }
    
    async def get_template(
        self,
        template_id: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get a specific template.
        
        Args:
            template_id: Template ID
        
        Returns:
            Dict with template details
        """
        if not self._initialized:
            await self.initialize()
        
        template = self._template_manager.get_template(template_id)
        
        if not template:
            return {
                "success": False,
                "error": f"Template '{template_id}' not found",
            }
        
        return {
            "success": True,
            "template": template.to_dict(),
            "sections": [
                {
                    "id": s.id,
                    "title": s.title_template,
                    "type": s.section_type.value,
                    "required": s.required,
                }
                for s in template.sections
            ],
        }
    
    async def match_template(
        self,
        query: str,
        available_templates: Optional[List[str]] = None,
        min_confidence: float = 0.3,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Find best matching template for a query.
        
        Args:
            query: User query
            available_templates: Restrict to these templates
            min_confidence: Minimum confidence threshold
        
        Returns:
            Dict with match result
        """
        if not self._initialized:
            await self.initialize()
        
        match = self._template_manager.match_template(
            query=query,
            available_templates=available_templates,
            min_confidence=min_confidence,
        )
        
        if not match:
            return {
                "success": True,
                "matched": False,
                "message": "No matching template found",
            }
        
        return {
            "success": True,
            "matched": True,
            "match": match.to_dict(),
        }
    
    # ========================================================================
    # Validation Operations
    # ========================================================================
    
    async def validate_plan(
        self,
        plan: Union[Dict, StructuredPlan],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Validate a plan.
        
        Args:
            plan: Plan to validate
        
        Returns:
            Dict with validation result
        """
        if not self._initialized:
            await self.initialize()
        
        if isinstance(plan, dict):
            plan = self._resolve_plan(plan)
        
        result = self._validator.validate(plan)
        
        return {
            "success": True,
            "validation": result.to_dict(),
            "is_valid": result.is_valid,
        }
    
    # ========================================================================
    # Estimation Operations
    # ========================================================================
    
    async def estimate_tokens(
        self,
        plan: Union[Dict, StructuredPlan],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Estimate token usage for a plan.
        
        Args:
            plan: Plan to estimate
        
        Returns:
            Dict with token estimate
        """
        if not self._initialized:
            await self.initialize()
        
        if isinstance(plan, dict):
            plan = self._resolve_plan(plan)
        
        estimate = self._plan_generator.estimate_tokens(plan)
        
        return {
            "success": True,
            "total_tokens": estimate.total_tokens,
            "tokens_by_section": estimate.tokens_by_section,
            "within_budget": estimate.within_budget,
            "budget_utilization": estimate.budget_utilization,
        }
    
    async def estimate_resources(
        self,
        plan: Union[Dict, StructuredPlan],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Estimate resources needed for plan execution.
        
        Args:
            plan: Plan to estimate
        
        Returns:
            Dict with resource estimate
        """
        if not self._initialized:
            await self.initialize()
        
        if isinstance(plan, dict):
            plan = self._resolve_plan(plan)
        
        estimate = self._plan_generator.estimate_resources(plan)
        
        return {
            "success": True,
            "estimated_time_minutes": estimate.estimated_time_minutes,
            "estimated_api_calls": estimate.estimated_api_calls,
            "parallel_batches": estimate.parallel_batches,
            "sections_requiring_research": estimate.sections_requiring_research,
            "sections_llm_only": estimate.sections_llm_only,
            "token_estimate": {
                "total": estimate.estimated_tokens.total_tokens,
                "within_budget": estimate.estimated_tokens.within_budget,
            },
        }
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _build_constraints(
        self,
        constraints: Optional[Dict[str, Any]],
    ) -> PlanConstraints:
        """Build PlanConstraints from dict."""
        if not constraints:
            return PlanConstraints(
                language=self._default_language,
                formality_level=self._default_formality,
            )
        
        return PlanConstraints(
            max_sections=constraints.get("max_sections", 15),
            max_tokens_total=constraints.get("max_tokens_total", 15000),
            max_tokens_per_section=constraints.get("max_tokens_per_section", 3000),
            min_sections=constraints.get("min_sections", 2),
            target_audience=constraints.get("target_audience", "general"),
            formality_level=FormalityLevel(constraints.get("formality_level", self._default_formality.value)),
            language=constraints.get("language", self._default_language),
            include_citations=constraints.get("include_citations", True),
            include_toc=constraints.get("include_toc", True),
        )
    
    def _resolve_plan(self, plan: Union[Dict, StructuredPlan]) -> StructuredPlan:
        """Resolve a plan argument to a StructuredPlan object.

        Accepts either a StructuredPlan directly, or a dict (from API / previous
        operation result).  When a dict is provided the helper first checks for
        a cached ``plan_object`` key (internal shortcut), then falls back to
        reconstructing the object via ``_dict_to_plan``.
        """
        if isinstance(plan, StructuredPlan):
            return plan
        if isinstance(plan, dict):
            obj = plan.get("plan_object")
            if isinstance(obj, StructuredPlan):
                return obj
            # The dict may be a full result envelope or an inner "plan" dict.
            plan_data = plan.get("plan", plan)
            return self._dict_to_plan(plan_data)
        return plan  # fallback – let caller handle type errors

    def _dict_to_plan(self, data: Dict[str, Any]) -> StructuredPlan:
        """Reconstruct a StructuredPlan from its serialised dict."""
        from .providers import DocumentType, StyleGuide
        sections = [
            self._dict_to_section(s) for s in data.get("sections", [])
        ]
        style_data = data.get("style_guide")
        style_guide = None
        if style_data and isinstance(style_data, dict):
            style_guide = StyleGuide(
                formality=FormalityLevel(style_data.get("formality", "professional")),
                language=style_data.get("language", "it"),
            )
        return StructuredPlan(
            id=data.get("id", ""),
            title=data.get("title", "Untitled"),
            document_type=DocumentType(data.get("document_type", "report")),
            topic=data.get("topic", ""),
            description=data.get("description", ""),
            sections=sections,
            style_guide=style_guide,
            version=data.get("version", "1.0.0"),
            estimated_tokens=data.get("estimated_tokens", 0),
            estimated_sections=data.get("estimated_sections", len(sections)),
            language=data.get("language", "it"),
        )

    def _dict_to_section(self, data: Dict[str, Any]) -> SectionPlan:
        """Convert dict to SectionPlan."""
        return SectionPlan(
            id=data.get("id", "section"),
            title=data.get("title", "Untitled"),
            description=data.get("description", ""),
            order=data.get("order", 0),
            section_type=SectionType(data.get("section_type", "custom")),
            content_type=ContentType(data.get("content_type", "prose")),
            source_preference=SourcePreference(data.get("source_preference", "rag_first")),
            suggested_queries=data.get("suggested_queries", []),
            target_tokens=data.get("target_tokens", 500),
            min_tokens=data.get("min_tokens", 100),
            max_tokens=data.get("max_tokens", 1500),
            required=data.get("required", True),
            enabled=data.get("enabled", True),
            depends_on=data.get("depends_on", []),
        )
